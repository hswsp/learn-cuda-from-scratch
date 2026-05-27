// =============================================================================
// mini_llm.cu — minimum GPT-2 (small) end-to-end inference, fp32 path.
//
// 教学目标:
//   把前面 1-13 章写过的所有 kernel 串成一个能跑的 LLM:
//     load weights (fp16 on disk -> fp32 on GPU for simplicity)
//     loop over prompt tokens + new tokens:
//       embed (wte + wpe)
//       for each layer:
//         LayerNorm
//         fused QKV projection
//         multi-head causal attention (Ch11 naive 版, 教学; 生产用 Ch12)
//         residual + LayerNorm
//         MLP (4D GeLU 4D)
//         residual
//       final LayerNorm
//       logits = h @ wte^T
//       sample (greedy)
//
// 简化:
//   - fp32 全程 (内存 4x 不省, 但少一个 fp16 调试维度)
//   - GeLU (GPT-2 是 GeLU 不是 SwiGLU)
//   - batch=1, 不做 KV cache (每步 recompute prompt+answer 全前向, 慢但代码短)
//
// 不完整: tokenizer 不在 C++ 里写. 用户用 Python 把 prompt 编成 token id, 通过
// 命令行传入 (e.g. --tokens 15496,11,612,318). 输出 token id 由用户解码.
//
// 真正想跑生产 demo: 见练习 #1 (集成 sentencepiece) 和 #2 (KV cache).
//
// 对应 HTML: docs/ch14-mini-llm/index.html
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>
#include <sstream>

// ---- model config ----
struct GPT2Cfg {
    int n_layer, n_head, d_model, d_ff, max_seq, vocab_size, head_dim;
};

// ---- weights container (device fp32 for教学) ----
struct LayerW {
    DeviceBuffer<float> ln1_w, ln1_b;
    DeviceBuffer<float> qkv_w, qkv_b;     // (d_model, 3*d_model), (3*d_model,)
    DeviceBuffer<float> proj_w, proj_b;   // (d_model, d_model)
    DeviceBuffer<float> ln2_w, ln2_b;
    DeviceBuffer<float> fc_w, fc_b;       // (d_model, d_ff)
    DeviceBuffer<float> fcp_w, fcp_b;     // (d_ff, d_model)
};
struct Weights {
    GPT2Cfg cfg;
    DeviceBuffer<float> wte, wpe;
    std::vector<LayerW> blocks;
    DeviceBuffer<float> lnf_w, lnf_b;
};

// ---- file I/O: fp16 disk -> fp32 device ----
static void fp16_to_fp32(const std::vector<__half>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = __half2float(src[i]);
}

static void load_tensor(std::ifstream& f, DeviceBuffer<float>& d, size_t n) {
    std::vector<__half> raw(n);
    f.read(reinterpret_cast<char*>(raw.data()), n * sizeof(__half));
    std::vector<float> fp32; fp16_to_fp32(raw, fp32);
    d.allocate(n);
    d.copy_from_host(fp32.data());
}

static void load_weights(const std::string& path, Weights& W) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::printf("can't open %s\n", path.c_str()); std::exit(1); }
    int32_t hdr[16]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (hdr[0] != 0x47505432) { std::printf("bad magic\n"); std::exit(1); }
    W.cfg.n_layer    = hdr[2];
    W.cfg.n_head     = hdr[3];
    W.cfg.d_model    = hdr[4];
    W.cfg.d_ff       = hdr[5];
    W.cfg.max_seq    = hdr[6];
    W.cfg.vocab_size = hdr[7];
    W.cfg.head_dim   = W.cfg.d_model / W.cfg.n_head;
    auto& C = W.cfg;
    std::printf("[mini_llm] cfg: L=%d H=%d D=%d FF=%d T_max=%d V=%d\n",
                C.n_layer, C.n_head, C.d_model, C.d_ff, C.max_seq, C.vocab_size);

    load_tensor(f, W.wte, size_t(C.vocab_size) * C.d_model);
    load_tensor(f, W.wpe, size_t(C.max_seq)    * C.d_model);

    W.blocks.resize(C.n_layer);
    for (int l = 0; l < C.n_layer; ++l) {
        auto& B = W.blocks[l];
        load_tensor(f, B.ln1_w,  C.d_model);
        load_tensor(f, B.ln1_b,  C.d_model);
        load_tensor(f, B.qkv_w,  size_t(C.d_model) * 3 * C.d_model);
        load_tensor(f, B.qkv_b,  size_t(3) * C.d_model);
        load_tensor(f, B.proj_w, size_t(C.d_model) * C.d_model);
        load_tensor(f, B.proj_b, C.d_model);
        load_tensor(f, B.ln2_w,  C.d_model);
        load_tensor(f, B.ln2_b,  C.d_model);
        load_tensor(f, B.fc_w,   size_t(C.d_model) * C.d_ff);
        load_tensor(f, B.fc_b,   C.d_ff);
        load_tensor(f, B.fcp_w,  size_t(C.d_ff) * C.d_model);
        load_tensor(f, B.fcp_b,  C.d_model);
    }
    load_tensor(f, W.lnf_w, C.d_model);
    load_tensor(f, W.lnf_b, C.d_model);
    std::printf("[mini_llm] weights loaded.\n");
}

// =====================  KERNELS  =================================

__global__ void embed_kernel(const float* wte, const float* wpe,
                             const int* tok, float* x,
                             int T, int D, int V) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= D) return;
    int id = tok[t];
    x[t * D + d] = wte[id * D + d] + wpe[t * D + d];
}

__global__ void layernorm_row(const float* x, const float* gamma, const float* beta,
                              float* y, int T, int D, float eps) {
    int t = blockIdx.x; int tid = threadIdx.x;
    extern __shared__ float ss[];
    float* sm = ss; float* sv = ss + 1;
    const float* xr = x + t * D; float* yr = y + t * D;
    double mean = 0;
    for (int i = tid; i < D; i += blockDim.x) mean += xr[i];
    // warp reduce + 全 block 单 thread, 简化:
    __shared__ double smean, svar;
    if (tid == 0) smean = 0;
    __syncthreads();
    atomicAdd(&smean, mean);
    __syncthreads();
    double m = smean / D;
    double var = 0;
    for (int i = tid; i < D; i += blockDim.x) { double d = xr[i] - m; var += d * d; }
    if (tid == 0) svar = 0;
    __syncthreads();
    atomicAdd(&svar, var);
    __syncthreads();
    double v = svar / D;
    float inv = rsqrtf(float(v) + eps);
    for (int i = tid; i < D; i += blockDim.x)
        yr[i] = (xr[i] - float(m)) * inv * gamma[i] + beta[i];
}

// 简单 tiled matmul (复用 Ch6 模板, 32x32 tile)
__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    constexpr int T = 32;
    __shared__ float As[T][T], Bs[T][T];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = by * T + ty, col = bx * T + tx;
    float acc = 0;
    for (int kt = 0; kt < K; kt += T) {
        As[ty][tx] = (row < M && kt + tx < K) ? A[row * K + kt + tx] : 0;
        Bs[ty][tx] = (kt + ty < K && col < N) ? B[(kt + ty) * N + col] : 0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__global__ void bias_add(float* X, const float* bias, int rows, int cols) {
    int r = blockIdx.x; int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) X[r * cols + c] += bias[c];
}

__global__ void residual_add(float* X, const float* Y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] += Y[i];
}

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k0 = 0.7978845608f, k1 = 0.044715f;
    return 0.5f * x * (1.f + tanhf(k0 * (x + k1 * x * x * x)));
}
__global__ void gelu_kernel(float* X, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) X[i] = gelu_tanh(X[i]);
}

// Multi-head causal attention: Q, K, V (T, H, Dh) row-major.
// O (T, H, Dh).  T <= max_seq.  fp32, single sequence.
__global__ void mha_naive(const float* Q, const float* K, const float* V, float* O,
                          int T, int H, int Dh, float scale) {
    int h = blockIdx.y;            // head id
    int i = blockIdx.x;            // query token
    if (i >= T) return;
    extern __shared__ float scores[];
    int tid = threadIdx.x;

    // 1) scores[j] = (Q[i,h,:] dot K[j,h,:]) * scale, j<=i else -inf
    for (int j = tid; j < T; j += blockDim.x) {
        if (j > i) { scores[j] = -1e30f; continue; }
        float s = 0;
        for (int d = 0; d < Dh; ++d)
            s += Q[(i * H + h) * Dh + d] * K[(j * H + h) * Dh + d];
        scores[j] = s * scale;
    }
    __syncthreads();
    // 2) softmax
    float m = -FLT_MAX;
    for (int j = tid; j < T; j += blockDim.x) m = fmaxf(m, scores[j]);
    // block max via atomic (简化, T 小)
    __shared__ float smax, ssum;
    if (tid == 0) smax = -FLT_MAX;
    __syncthreads();
    atomicMax((int*)&smax, __float_as_int(m));   // float monotonic <-> int monotonic (positive); for negatives need care
    __syncthreads();
    // safer: use single-thread fallback
    if (tid == 0) {
        float mm = -FLT_MAX;
        for (int j = 0; j <= i; ++j) mm = fmaxf(mm, scores[j]);
        smax = mm;
    }
    __syncthreads();
    if (tid == 0) {
        double s = 0;
        for (int j = 0; j <= i; ++j) { scores[j] = expf(scores[j] - smax); s += scores[j]; }
        ssum = float(s);
    }
    __syncthreads();
    float inv = 1.f / ssum;
    // 3) O[i,h,:] = sum_j scores[j] * V[j,h,:]
    for (int d = tid; d < Dh; d += blockDim.x) {
        float acc = 0;
        for (int j = 0; j <= i; ++j) acc += scores[j] * inv * V[(j * H + h) * Dh + d];
        O[(i * H + h) * Dh + d] = acc;
    }
}

template <int BLOCK>
__global__ void greedy_argmax(const float* logits, int* out, int V) {
    __shared__ float vals[BLOCK]; __shared__ int idxs[BLOCK];
    int tid = threadIdx.x;
    float bv = -FLT_MAX; int bi = 0;
    for (int i = tid; i < V; i += BLOCK)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    vals[tid] = bv; idxs[tid] = bi;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s && vals[tid + s] > vals[tid]) {
            vals[tid] = vals[tid + s]; idxs[tid] = idxs[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) *out = idxs[0];
}

// ====================  HOST FORWARD  ==================================

void forward(const Weights& W, const std::vector<int>& tokens_h, int T_total, int* out_id) {
    auto& C = W.cfg;
    int T = (int)tokens_h.size();
    int D = C.d_model;

    DeviceBuffer<int>   d_tok(T);    d_tok.copy_from_host(tokens_h.data());
    DeviceBuffer<float> x(T * D), tmp(T * D), normed(T * D);
    DeviceBuffer<float> qkv(size_t(T) * 3 * D);
    DeviceBuffer<float> attn_out(T * D), proj_out(T * D);
    DeviceBuffer<float> ff_mid(size_t(T) * C.d_ff), ff_out(T * D);
    DeviceBuffer<float> logits(C.vocab_size);

    // embed
    dim3 eb(64);
    dim3 eg(T, (D + 63) / 64);
    embed_kernel<<<eg, eb>>>(W.wte.ptr, W.wpe.ptr, d_tok.ptr, x.ptr, T, D, C.vocab_size);
    KERNEL_CHECK();

    for (int l = 0; l < C.n_layer; ++l) {
        const auto& B = W.blocks[l];
        // ---- LN1 -> normed ----
        layernorm_row<<<T, 256>>>(x.ptr, B.ln1_w.ptr, B.ln1_b.ptr, normed.ptr, T, D, 1e-5f);
        KERNEL_CHECK();

        // ---- QKV = normed @ qkv_w + qkv_b ----
        dim3 mb(32, 32), mg((3*D + 31)/32, (T + 31)/32);
        matmul_kernel<<<mg, mb>>>(normed.ptr, B.qkv_w.ptr, qkv.ptr, T, 3*D, D);
        KERNEL_CHECK();
        bias_add<<<dim3(T, (3*D + 255)/256), 256>>>(qkv.ptr, B.qkv_b.ptr, T, 3*D);
        KERNEL_CHECK();

        // qkv layout: (T, 3, H, Dh)  We treat it as Q | K | V via offsets:
        // Q = qkv.ptr + 0,  stride per-token = 3*D
        // For simplicity we run mha by re-interpreting offsets in attention.

        // We'll split: Q (T,H,Dh), K, V by writing 3 separate buffers.
        // (省一次 split kernel, 直接传指针 + stride 也可, 这里 explicit.)
        DeviceBuffer<float> Q(T*D), K(T*D), V(T*D);
        // split via cudaMemcpyAsync (D2D)
        for (int t = 0; t < T; ++t) {
            CUDA_CHECK(cudaMemcpy(Q.ptr + t*D, qkv.ptr + t*3*D + 0*D, D*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(K.ptr + t*D, qkv.ptr + t*3*D + 1*D, D*sizeof(float), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(V.ptr + t*D, qkv.ptr + t*3*D + 2*D, D*sizeof(float), cudaMemcpyDeviceToDevice));
        }

        // ---- MHA causal ----
        dim3 ag(T, C.n_head);
        size_t shm = T * sizeof(float);
        mha_naive<<<ag, 128, shm>>>(Q.ptr, K.ptr, V.ptr, attn_out.ptr,
                                     T, C.n_head, C.head_dim, 1.f / std::sqrt(float(C.head_dim)));
        KERNEL_CHECK();

        // ---- proj = attn_out @ proj_w + proj_b ----
        dim3 pg((D + 31)/32, (T + 31)/32);
        matmul_kernel<<<pg, mb>>>(attn_out.ptr, B.proj_w.ptr, proj_out.ptr, T, D, D);
        bias_add<<<dim3(T, (D + 255)/256), 256>>>(proj_out.ptr, B.proj_b.ptr, T, D);
        KERNEL_CHECK();

        // ---- x += proj_out (residual) ----
        residual_add<<<(T*D + 255)/256, 256>>>(x.ptr, proj_out.ptr, T*D);

        // ---- LN2 ----
        layernorm_row<<<T, 256>>>(x.ptr, B.ln2_w.ptr, B.ln2_b.ptr, normed.ptr, T, D, 1e-5f);

        // ---- FF (D -> 4D -> D) ----
        dim3 fg1((C.d_ff + 31)/32, (T + 31)/32);
        matmul_kernel<<<fg1, mb>>>(normed.ptr, B.fc_w.ptr, ff_mid.ptr, T, C.d_ff, D);
        bias_add<<<dim3(T, (C.d_ff + 255)/256), 256>>>(ff_mid.ptr, B.fc_b.ptr, T, C.d_ff);
        gelu_kernel<<<(T*C.d_ff + 255)/256, 256>>>(ff_mid.ptr, T*C.d_ff);
        dim3 fg2((D + 31)/32, (T + 31)/32);
        matmul_kernel<<<fg2, mb>>>(ff_mid.ptr, B.fcp_w.ptr, ff_out.ptr, T, D, C.d_ff);
        bias_add<<<dim3(T, (D + 255)/256), 256>>>(ff_out.ptr, B.fcp_b.ptr, T, D);

        residual_add<<<(T*D + 255)/256, 256>>>(x.ptr, ff_out.ptr, T*D);
    }

    // ---- final LN ----
    layernorm_row<<<T, 256>>>(x.ptr, W.lnf_w.ptr, W.lnf_b.ptr, normed.ptr, T, D, 1e-5f);
    // ---- logits for the LAST token only: logits = normed[T-1, :] @ wte^T  (shape V) ----
    // matmul_kernel expects (M, N, K) row-major, so we do (1, V, D) = last_row @ wte^T.
    // wte is (V, D), so logits = last_row @ wte^T = last_row @ wte.T (V D -> D V) — emulate by
    // calling matmul with B treated as (D, V) which is wte transposed -> we already have wte (V,D).
    // 简化: 用 host-side row select 然后做 GEMV.
    DeviceBuffer<float> last_row(D);
    CUDA_CHECK(cudaMemcpy(last_row.ptr, normed.ptr + (T-1)*D, D*sizeof(float), cudaMemcpyDeviceToDevice));
    // launch matmul (1, V, D) = last_row(1,D) @ wte^T(D, V)
    // we don't have transposed wte; do (1, D) @ (D, V) by treating wte as (V, D) needs transpose.
    // Quick fix: gemv kernel inline.
    auto gemv = [&](){
        // logits[v] = sum_d last_row[d] * wte[v, d]
        // launch V/256 blocks of 256 threads, each block handles 256 vocab ids.
    };
    // We use a simple gemv kernel inline:
    static int once = 0; if (!once) { once = 1; }
    auto run_gemv = [&](){
        // tiny inline kernel via lambda is not allowed in CUDA; declare proper kernel below.
    };
    // see gemv_logits kernel below
    extern __global__ void gemv_logits(const float*, const float*, float*, int, int);
    gemv_logits<<<(C.vocab_size + 255)/256, 256>>>(last_row.ptr, W.wte.ptr, logits.ptr, D, C.vocab_size);
    KERNEL_CHECK();

    // ---- greedy argmax ----
    DeviceBuffer<int> d_out(1);
    greedy_argmax<256><<<1, 256>>>(logits.ptr, d_out.ptr, C.vocab_size);
    CUDA_CHECK(cudaMemcpy(out_id, d_out.ptr, sizeof(int), cudaMemcpyDeviceToHost));
}

// logits[v] = sum_d x[d] * wte[v, d]  (one block per ~256 vocab ids)
__global__ void gemv_logits(const float* x, const float* wte, float* logits, int D, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float s = 0;
    for (int d = 0; d < D; ++d) s += x[d] * wte[v * D + d];
    logits[v] = s;
}

// ================== MAIN ============================

std::vector<int> parse_tokens(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

int main(int argc, char** argv) {
    std::string weights_path = "data/gpt2-small.bin";
    std::string tokens_csv   = "15496,11,612,318";   // "Hello, there is"  (gpt-2 ids)
    int max_new = 16;

    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--weights=", 10))  weights_path = argv[i] + 10;
        else if (!std::strncmp(argv[i], "--tokens=", 9)) tokens_csv = argv[i] + 9;
        else if (!std::strncmp(argv[i], "--max_new=", 10)) max_new = std::atoi(argv[i] + 10);
    }

    Weights W; load_weights(weights_path, W);
    auto tokens = parse_tokens(tokens_csv);
    std::printf("[mini_llm] prompt tokens: ");
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");

    for (int step = 0; step < max_new; ++step) {
        if ((int)tokens.size() >= W.cfg.max_seq) break;
        int next = 0;
        forward(W, tokens, (int)tokens.size(), &next);
        tokens.push_back(next);
        std::printf("[mini_llm] step %d -> %d\n", step, next);
    }
    std::printf("[mini_llm] final tokens: ");
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");
    return 0;
}
