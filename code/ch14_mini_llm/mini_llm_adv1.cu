// =============================================================================
// mini_llm_adv1.cu — GPT-2 small 推理进阶版 1
//
// 相比 mini_llm.cu 的改进:
//   1. 支持更长的输入/输出序列 (prompt + max_new 可超过 16, 测试 --max_new=64)
//      - 使用一次性预分配 Workspace, 避免每步重复 cudaMalloc
//      - MHA 直接从 qkv (T, 3*D) 读取, 省掉每 token 3 次 cudaMemcpy
//   2. 解码策略从 greedy 升级为 temperature + top-k
//      - 默认 temperature=0.8, top_k=40
//      - 采样在 CPU 上做 (单 token 拷贝开销远小于 forward, 且容易实验)
//
// 仍保留的教学简化:
//   - fp32 全程
//   - 无 KV cache (每步全序列前向)
//   - batch=1
//   - 无 tokenizer, 输入输出均为 GPT-2 token id
//
// 编译:
//   make mini_llm_adv1
// 运行:
//   ./mini_llm_adv1 --weights=../../data/gpt2-small.bin \
//                  --tokens=15496,11,612,318 \
//                  --max_new=64 --temperature=0.8 --top_k=40 --seed=42
//
// 对应 HTML: docs/ch14-mini-llm/index.html#advanced-1
// =============================================================================
#include "../common/cuda_utils.h"
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
#include <random>
#include <algorithm>
#include <functional>

// ---- model config ----
struct GPT2Cfg {
    int n_layer, n_head, d_model, d_ff, max_seq, vocab_size, head_dim;
};

// ---- weights container (device fp32) ----
struct LayerW {
    DeviceBuffer<float> ln1_w, ln1_b;
    DeviceBuffer<float> qkv_w, qkv_b;
    DeviceBuffer<float> proj_w, proj_b;
    DeviceBuffer<float> ln2_w, ln2_b;
    DeviceBuffer<float> fc_w, fc_b;
    DeviceBuffer<float> fcp_w, fcp_b;
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
    std::printf("[mini_llm_adv1] cfg: L=%d H=%d D=%d FF=%d T_max=%d V=%d\n",
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
    std::printf("[mini_llm_adv1] weights loaded.\n");
}

// =====================  KERNELS  =================================

__global__ void embed_kernel(const float* wte, const float* wpe,
                             const int* tok, float* x,
                             int T, int D, int V) {
    int t = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int id = tok[t];
    x[t * D + d] = wte[id * D + d] + wpe[t * D + d];
}

__global__ void layernorm_row(const float* x, const float* gamma, const float* beta,
                              float* y, int T, int D, float eps) {
    int t = blockIdx.x;
    int tid = threadIdx.x;
    if (t >= T) return;
    const float* xr = x + t * D;
    float* yr = y + t * D;

    double mean = 0;
    for (int i = tid; i < D; i += blockDim.x) mean += xr[i];
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

// Multi-head causal attention, 直接从 qkv (T, 3*D) 读取.
// qkv 布局: token t 的 Q/K/V 分别位于 qkv[t*3D + 0], qkv[t*3D + D], qkv[t*3D + 2D].
// 每个头内是 (H, Dh), O 也是 (T, H, Dh) 行优先.
__global__ void mha_naive(const float* qkv, float* O,
                          int T, int H, int Dh, float scale) {
    int h = blockIdx.y;            // head id
    int i = blockIdx.x;            // query token
    if (i >= T || h >= H) return;
    extern __shared__ float scores[];
    int tid = threadIdx.x;

    // 1) 计算 scores[j] = Q[i,h,:] . K[j,h,:] * scale, j<=i; 否则 -inf
    //    qkv 中 token i 的 Q 起点: (i * 3*H + 0*H + h) * Dh
    const float* Qi = qkv + (i * 3 * H + h) * Dh;
    for (int j = tid; j < T; j += blockDim.x) {
        if (j > i) { scores[j] = -1e30f; continue; }
        // token j 的 K 起点: (j * 3*H + 1*H + h) * Dh
        const float* Kj = qkv + (j * 3 * H + H + h) * Dh;
        float s = 0;
        for (int d = 0; d < Dh; ++d) s += Qi[d] * Kj[d];
        scores[j] = s * scale;
    }
    __syncthreads();

    // 2) softmax (单线程串行, 教学用; T 到几百仍可接受)
    __shared__ float smax, ssum;
    if (tid == 0) {
        float mm = -FLT_MAX;
        for (int j = 0; j <= i; ++j) mm = fmaxf(mm, scores[j]);
        smax = mm;
        double s = 0;
        for (int j = 0; j <= i; ++j) {
            scores[j] = expf(scores[j] - smax);
            s += scores[j];
        }
        ssum = float(s);
    }
    __syncthreads();
    float inv = 1.f / ssum;

    // 3) O[i,h,d] = sum_j scores[j] * V[j,h,d]
    for (int d = tid; d < Dh; d += blockDim.x) {
        float acc = 0;
        for (int j = 0; j <= i; ++j) {
            // token j 的 V 起点: (j * 3*H + 2*H + h) * Dh
            const float* Vj = qkv + (j * 3 * H + 2 * H + h) * Dh;
            acc += scores[j] * inv * Vj[d];
        }
        O[(i * H + h) * Dh + d] = acc;
    }
}

__global__ void gemv_logits(const float* x, const float* wte, float* logits,
                            int D, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float s = 0;
    for (int d = 0; d < D; ++d) s += x[d] * wte[v * D + d];
    logits[v] = s;
}

// ====================  HOST FORWARD  ==================================

// 一次性预分配最大序列长度所需显存, 避免生成过程中反复 cudaMalloc
struct Workspace {
    int maxT = 0;
    DeviceBuffer<int>   d_tok;
    DeviceBuffer<float> x, normed;
    DeviceBuffer<float> qkv;
    DeviceBuffer<float> attn_out, proj_out;
    DeviceBuffer<float> ff_mid, ff_out;
    DeviceBuffer<float> logits;
    DeviceBuffer<float> last_row;

    void allocate(const GPT2Cfg& C) {
        maxT = C.max_seq;
        d_tok.allocate(maxT);
        x.allocate(size_t(maxT) * C.d_model);
        normed.allocate(size_t(maxT) * C.d_model);
        qkv.allocate(size_t(maxT) * 3 * C.d_model);
        attn_out.allocate(size_t(maxT) * C.d_model);
        proj_out.allocate(size_t(maxT) * C.d_model);
        ff_mid.allocate(size_t(maxT) * C.d_ff);
        ff_out.allocate(size_t(maxT) * C.d_model);
        logits.allocate(C.vocab_size);
        last_row.allocate(C.d_model);
    }
};

// 注意: tokens_h.size() 必须 <= ws.maxT
void forward(const Weights& W, Workspace& ws, const std::vector<int>& tokens_h, int T) {
    auto& C = W.cfg;
    int D = C.d_model;
    if (T > ws.maxT) {
        std::printf("[mini_llm_adv1] sequence length %d exceeds workspace maxT %d\n", T, ws.maxT);
        std::exit(1);
    }

    CUDA_CHECK(cudaMemcpy(ws.d_tok.ptr, tokens_h.data(), T * sizeof(int), cudaMemcpyHostToDevice));

    // embed
    dim3 eb(64);
    dim3 eg(T, (D + 63) / 64);
    embed_kernel<<<eg, eb>>>(W.wte.ptr, W.wpe.ptr, ws.d_tok.ptr, ws.x.ptr, T, D, C.vocab_size);
    KERNEL_CHECK();

    for (int l = 0; l < C.n_layer; ++l) {
        const auto& B = W.blocks[l];

        // ---- LN1 ----
        layernorm_row<<<T, 256>>>(ws.x.ptr, B.ln1_w.ptr, B.ln1_b.ptr, ws.normed.ptr, T, D, 1e-5f);
        KERNEL_CHECK();

        // ---- QKV = normed @ qkv_w + qkv_b ----
        dim3 mb(32, 32), mg((3*D + 31)/32, (T + 31)/32);
        matmul_kernel<<<mg, mb>>>(ws.normed.ptr, B.qkv_w.ptr, ws.qkv.ptr, T, 3*D, D);
        KERNEL_CHECK();
        bias_add<<<dim3(T, (3*D + 255)/256), 256>>>(ws.qkv.ptr, B.qkv_b.ptr, T, 3*D);
        KERNEL_CHECK();

        // ---- MHA causal (直接从 qkv 读) ----
        dim3 ag(T, C.n_head);
        size_t shm = T * sizeof(float);
        mha_naive<<<ag, 128, shm>>>(ws.qkv.ptr, ws.attn_out.ptr,
                                     T, C.n_head, C.head_dim, 1.f / std::sqrt(float(C.head_dim)));
        KERNEL_CHECK();

        // ---- proj = attn_out @ proj_w + proj_b ----
        dim3 pg((D + 31)/32, (T + 31)/32);
        matmul_kernel<<<pg, mb>>>(ws.attn_out.ptr, B.proj_w.ptr, ws.proj_out.ptr, T, D, D);
        bias_add<<<dim3(T, (D + 255)/256), 256>>>(ws.proj_out.ptr, B.proj_b.ptr, T, D);
        KERNEL_CHECK();

        // ---- x += proj_out (residual) ----
        residual_add<<<(T*D + 255)/256, 256>>>(ws.x.ptr, ws.proj_out.ptr, T*D);
        KERNEL_CHECK();

        // ---- LN2 ----
        layernorm_row<<<T, 256>>>(ws.x.ptr, B.ln2_w.ptr, B.ln2_b.ptr, ws.normed.ptr, T, D, 1e-5f);
        KERNEL_CHECK();

        // ---- FF (D -> 4D -> D) ----
        dim3 fg1((C.d_ff + 31)/32, (T + 31)/32);
        matmul_kernel<<<fg1, mb>>>(ws.normed.ptr, B.fc_w.ptr, ws.ff_mid.ptr, T, C.d_ff, D);
        bias_add<<<dim3(T, (C.d_ff + 255)/256), 256>>>(ws.ff_mid.ptr, B.fc_b.ptr, T, C.d_ff);
        gelu_kernel<<<(T*C.d_ff + 255)/256, 256>>>(ws.ff_mid.ptr, T*C.d_ff);
        KERNEL_CHECK();
        dim3 fg2((D + 31)/32, (T + 31)/32);
        matmul_kernel<<<fg2, mb>>>(ws.ff_mid.ptr, B.fcp_w.ptr, ws.ff_out.ptr, T, D, C.d_ff);
        bias_add<<<dim3(T, (D + 255)/256), 256>>>(ws.ff_out.ptr, B.fcp_b.ptr, T, D);
        KERNEL_CHECK();

        residual_add<<<(T*D + 255)/256, 256>>>(ws.x.ptr, ws.ff_out.ptr, T*D);
        KERNEL_CHECK();
    }

    // ---- final LN ----
    layernorm_row<<<T, 256>>>(ws.x.ptr, W.lnf_w.ptr, W.lnf_b.ptr, ws.normed.ptr, T, D, 1e-5f);
    KERNEL_CHECK();

    // ---- logits for the LAST token: logits = normed[T-1, :] @ wte^T ----
    CUDA_CHECK(cudaMemcpy(ws.last_row.ptr, ws.normed.ptr + (T-1)*D,
                          D * sizeof(float), cudaMemcpyDeviceToDevice));
    gemv_logits<<<(C.vocab_size + 255)/256, 256>>>(ws.last_row.ptr, W.wte.ptr, ws.logits.ptr, D, C.vocab_size);
    KERNEL_CHECK();
}

// ====================  SAMPLING  ======================================

// 在 CPU 上对 logits 做 temperature + top-k 采样.
// 单 token 拷贝开销很小, 且便于实验不同采样策略.
static int sample_token(const float* d_logits, int V,
                        float temperature, int top_k, std::mt19937& rng) {
    std::vector<float> logits(V);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, V * sizeof(float), cudaMemcpyDeviceToHost));

    // 1) temperature
    for (float& v : logits) v /= temperature;

    // 2) top-k: 保留最大的 k 个, 其余设为 -inf
    //    使用 nth_element 找第 k 大的阈值, 再遍历一次做 mask.
    if (top_k > 0 && top_k < V) {
        std::vector<float> copy = logits;
        std::nth_element(copy.begin(), copy.begin() + (top_k - 1), copy.end(), std::greater<float>());
        float threshold = copy[top_k - 1];
        for (float& v : logits) if (v < threshold) v = -1e30f;
    }

    // 3) softmax
    float mx = -FLT_MAX;
    for (float v : logits) mx = std::max(mx, v);
    double sum = 0;
    for (float& v : logits) {
        v = std::exp(v - mx);
        sum += v;
    }
    float inv = float(1.0 / sum);
    for (float& v : logits) v *= inv;

    // 4) 按概率分布采样
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    double acc = 0;
    for (int i = 0; i < V; ++i) {
        acc += logits[i];
        if (u <= acc) return i;
    }
    return V - 1;
}

// ====================  MAIN  ==========================================

std::vector<int> parse_tokens(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::stoi(tok));
    return out;
}

int main(int argc, char** argv) {
    std::string weights_path = "data/gpt2-small.bin";
    std::string tokens_csv   = "15496,11,612,318";   // "Hello, there is"
    int max_new     = 16;
    float temperature = 0.8f;
    int top_k       = 40;
    unsigned int seed = 42;

    for (int i = 1; i < argc; ++i) {
        if      (!std::strncmp(argv[i], "--weights=",     10)) weights_path = argv[i] + 10;
        else if (!std::strncmp(argv[i], "--tokens=",       9)) tokens_csv   = argv[i] + 9;
        else if (!std::strncmp(argv[i], "--max_new=",     10)) max_new      = std::atoi(argv[i] + 10);
        else if (!std::strncmp(argv[i], "--temperature=", 14)) temperature  = std::atof(argv[i] + 14);
        else if (!std::strncmp(argv[i], "--top_k=",        8)) top_k        = std::atoi(argv[i] + 8);
        else if (!std::strncmp(argv[i], "--seed=",         7)) seed         = (unsigned int)std::atoi(argv[i] + 7);
    }

    if (temperature <= 0.0f) { std::printf("temperature must be > 0\n"); return 1; }
    if (top_k < 0) top_k = 0;

    Weights W; load_weights(weights_path, W);
    Workspace ws; ws.allocate(W.cfg);

    auto tokens = parse_tokens(tokens_csv);
    std::printf("[mini_llm_adv1] prompt tokens (%zu): ", tokens.size());
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");
    std::printf("[mini_llm_adv1] temperature=%.3f top_k=%d seed=%u max_new=%d\n",
                temperature, top_k, seed, max_new);

    std::mt19937 rng(seed);

    for (int step = 0; step < max_new; ++step) {
        if ((int)tokens.size() >= W.cfg.max_seq) {
            std::printf("[mini_llm_adv1] reached max_seq=%d, stop.\n", W.cfg.max_seq);
            break;
        }
        int T = (int)tokens.size();
        forward(W, ws, tokens, T);
        int next = sample_token(ws.logits.ptr, W.cfg.vocab_size, temperature, top_k, rng);
        tokens.push_back(next);
        std::printf("[mini_llm_adv1] step %2d -> %d\n", step, next);
    }

    std::printf("[mini_llm_adv1] final tokens (%zu): ", tokens.size());
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");
    return 0;
}
