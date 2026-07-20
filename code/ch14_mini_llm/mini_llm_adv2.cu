// =============================================================================
// mini_llm_adv2.cu — GPT-2 small 推理进阶版 2（生产向）
//
// 在 v1 基础上再做三个核心升级：
//   1. KV cache：每层维护 K/V cache (n_layer, n_head, T_max, D_head)，
//      decode 阶段每步只 forward 1 个 token， attention 从 O(T²) 降到 O(T)。
//   2. fp16 + WMMA：fc/proj/qkv 的权重存 __half，GEMM 走 Tensor Core。
//      输入实时 fp32→fp16 转换，输出 fp32 accumulator；M/N/K 不对齐时回退到简单 GEMM。
//   3. FlashAttention：prefill 阶段用 Ch12 的 flash_attn_v1 替换 mha_naive，
//      不物化 T×T 中间矩阵；decode 阶段用单 query 的 KV-cache attention。
//
// 预期性能（A100 fp16, prompt=4）：
//   - v1 无 cache / fp32：单步 ~50 ms
//   - v2 有 cache / fp16 / FA：单步 ~5 ms
//
// 仍保留的简化：
//   - batch=1
//   - 无 tokenizer，输入输出仍是 GPT-2 token id
//   - layernorm / bias / residual / logits gemv 保持 fp32（开销小，数值稳）
//
// 编译：
//   make mini_llm_adv2
// 运行：
//   ./mini_llm_adv2 --weights=../../data/gpt2-small.bin \\
//                   --tokens=15496,11,612,318 \\
//                   --max_new=64 --temperature=0.8 --top_k=40 --seed=42
//
// 对应 HTML: docs/ch14-mini-llm/index.html#advanced-2
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/check.h"
#include <cuda_fp16.h>
#include <mma.h>
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

using namespace nvcuda;

// ---- model config ----
struct GPT2Cfg {
    int n_layer, n_head, d_model, d_ff, max_seq, vocab_size, head_dim;
};

// ---- weights：GEMM 权重 fp16，其余 fp32 ----
struct LayerW {
    DeviceBuffer<float> ln1_w, ln1_b;
    DeviceBuffer<float> ln2_w, ln2_b;
    DeviceBuffer<__half> qkv_w, proj_w, fc_w, fcp_w;
    DeviceBuffer<float> qkv_b, proj_b, fc_b, fcp_b;
};
struct Weights {
    GPT2Cfg cfg;
    DeviceBuffer<float> wte;   // embed + logits gemv 用 fp32
    DeviceBuffer<float> wpe;
    std::vector<LayerW> blocks;
    DeviceBuffer<float> lnf_w, lnf_b;
};

// ---- file I/O：fp16 on disk -> 按需 fp16/fp32 device ----
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
    std::printf("[mini_llm_adv2] cfg: L=%d H=%d D=%d FF=%d T_max=%d V=%d\n",
                C.n_layer, C.n_head, C.d_model, C.d_ff, C.max_seq, C.vocab_size);

    auto load_fp32 = [&](DeviceBuffer<float>& d, size_t n) {
        std::vector<__half> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * sizeof(__half));
        std::vector<float> fp32(n);
        for (size_t i = 0; i < n; ++i) fp32[i] = __half2float(raw[i]);
        d.allocate(n); d.copy_from_host(fp32.data());
    };
    auto load_fp16 = [&](DeviceBuffer<__half>& d, size_t n) {
        std::vector<__half> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), n * sizeof(__half));
        d.allocate(n); d.copy_from_host(raw.data());
    };

    load_fp32(W.wte, size_t(C.vocab_size) * C.d_model);
    load_fp32(W.wpe, size_t(C.max_seq)    * C.d_model);

    W.blocks.resize(C.n_layer);
    for (int l = 0; l < C.n_layer; ++l) {
        auto& B = W.blocks[l];
        load_fp32(B.ln1_w,  C.d_model);
        load_fp32(B.ln1_b,  C.d_model);
        load_fp16(B.qkv_w,  size_t(C.d_model) * 3 * C.d_model);
        load_fp32(B.qkv_b,  size_t(3) * C.d_model);
        load_fp16(B.proj_w, size_t(C.d_model) * C.d_model);
        load_fp32(B.proj_b, C.d_model);
        load_fp32(B.ln2_w,  C.d_model);
        load_fp32(B.ln2_b,  C.d_model);
        load_fp16(B.fc_w,   size_t(C.d_model) * C.d_ff);
        load_fp32(B.fc_b,   C.d_ff);
        load_fp16(B.fcp_w,  size_t(C.d_ff) * C.d_model);
        load_fp32(B.fcp_b,  C.d_model);
    }
    load_fp32(W.lnf_w, C.d_model);
    load_fp32(W.lnf_b, C.d_model);
    std::printf("[mini_llm_adv2] weights loaded.\n");
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

__global__ void fp32_to_fp16(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void gemv_logits(const float* x, const float* wte, float* logits,
                            int D, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    float s = 0;
    for (int d = 0; d < D; ++d) s += x[d] * wte[v * D + d];
    logits[v] = s;
}

// -------------------- WMMA GEMM (sm_70+, M/N/K 需 16 整除) --------------------
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;

__global__ void wmma_gemm_kernel(const __half* A, const __half* B, float* C,
                                 int M, int N, int K) {
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warp_m  = blockIdx.y * blockDim.y + warp_id / 2;
    int warp_n  = blockIdx.x * 2 + warp_id % 2;

    int row0 = warp_m * WMMA_M;
    int col0 = warp_n * WMMA_N;
    if (row0 >= M || col0 >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int kt = 0; kt < K; kt += WMMA_K) {
        const __half* a_ptr = A + row0 * K + kt;
        const __half* b_ptr = B + kt * N + col0;
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + row0 * N + col0, c_frag, N, wmma::mem_row_major);
}

// -------------------- 简单 fallback GEMM：A fp32, B fp16, C fp32 ---------------
// 用于 decode (M=1) 或 M/N/K 未 16 对齐时的 prefill。
__global__ void simple_gemm_kernel(const float* A, const __half* B, float* C,
                                   int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int r = idx / N;
    int c = idx % N;
    float acc = 0;
    for (int k = 0; k < K; ++k)
        acc += A[r * K + k] * __half2float(B[k * N + c]);
    C[idx] = acc;
}

// -------------------- FlashAttention v1 (Ch12) 多 head 改造版 -------------------
// 原 kernel 假设单 head、Q/K/V 连续；这里改成：
//   - blockIdx.y = head id
//   - blockIdx.x = Q tile id
//   - Q/K/V/O 的跨行步长为 head_stride = n_head * D_head
// 固定 Dm = 64（与 GPT-2 small head_dim 匹配）。
constexpr int Br = 64, Bc = 64, Dm = 64;

__global__ void flash_attn_v1_kernel(const float* qkv, float* attn_out,
                                     int T, int n_head, float scale, bool causal) {
    int head = blockIdx.y;
    int qi_base = blockIdx.x * Br;
    int tid = threadIdx.x;

    // qkv 布局: (T, 3, n_head, Dm); attn_out 布局: (T, n_head, Dm)
    int qkv_stride = 3 * n_head * Dm;
    int out_stride = n_head * Dm;
    const float* Q = qkv + head * Dm;
    const float* K = qkv + (n_head + head) * Dm;
    const float* V = qkv + (2 * n_head + head) * Dm;
    float* O = attn_out + head * Dm;

    __shared__ float Qs[Br][Dm];
    __shared__ float Ks[Bc][Dm];
    __shared__ float Vs[Bc][Dm];
    __shared__ float Ss[Br][Bc];

    // load Q tile
    for (int i = tid; i < Br * Dm; i += blockDim.x) {
        int r = i / Dm, c = i % Dm;
        int gr = qi_base + r;
        Qs[r][c] = (gr < T) ? Q[gr * qkv_stride + c] : 0.f;
    }
    __syncthreads();

    __shared__ float O_s[Br][Dm];
    __shared__ float m_s[Br];
    __shared__ float l_s[Br];

    if (tid < Br) {
        for (int d = 0; d < Dm; ++d) O_s[tid][d] = 0.f;
        m_s[tid] = -FLT_MAX;
        l_s[tid] = 0.f;
    }
    __syncthreads();

    int n_kv_tiles = (T + Bc - 1) / Bc;
    for (int kt = 0; kt < n_kv_tiles; ++kt) {
        int kj_base = kt * Bc;
        for (int i = tid; i < Bc * Dm; i += blockDim.x) {
            int r = i / Dm, c = i % Dm;
            int gr = kj_base + r;
            Ks[r][c] = (gr < T) ? K[gr * qkv_stride + c] : 0.f;
            Vs[r][c] = (gr < T) ? V[gr * qkv_stride + c] : 0.f;
        }
        __syncthreads();

        // S_ij = Q_i @ K_j^T * scale, 每 thread 处理 S 的一行
        if (tid < Br) {
            int qi = qi_base + tid;
            for (int j = 0; j < Bc; ++j) {
                int kj = kj_base + j;
                float s = 0.f;
                for (int d = 0; d < Dm; ++d) s += Qs[tid][d] * Ks[j][d];
                s *= scale;
                if (causal && kj > qi) s = -1e30f;
                Ss[tid][j] = s;
            }
        }
        __syncthreads();

        // online softmax + O update
        if (tid < Br) {
            float m_old = m_s[tid], l_old = l_s[tid];
            float m_local = -FLT_MAX;
            for (int j = 0; j < Bc; ++j) m_local = fmaxf(m_local, Ss[tid][j]);
            float m_new = fmaxf(m_old, m_local);
            float l_local = 0.f;
            for (int j = 0; j < Bc; ++j) {
                float p = __expf(Ss[tid][j] - m_new);
                Ss[tid][j] = p;
                l_local += p;
            }
            float l_new = __expf(m_old - m_new) * l_old + l_local;
            float rescale = __expf(m_old - m_new);

            for (int d = 0; d < Dm; ++d) {
                float acc = O_s[tid][d] * rescale;
                for (int j = 0; j < Bc; ++j) acc += Ss[tid][j] * Vs[j][d];
                O_s[tid][d] = acc;
            }
            m_s[tid] = m_new;
            l_s[tid] = l_new;
        }
        __syncthreads();
    }

    if (tid < Br) {
        float inv = 1.f / l_s[tid];
        int gr = qi_base + tid;
        if (gr < T) {
            for (int d = 0; d < Dm; ++d) O[gr * out_stride + d] = O_s[tid][d] * inv;
        }
    }
}

// -------------------- decode 阶段单 query KV-cache attention --------------------
// qkv 布局：(T, 3, H, D_head)，当前 token 位于 qkv[pos]。
// cache 布局：(n_layer, n_head, T_max, D_head)。
// 每个 block 处理一个 head，256 threads 并行 D_head。
__global__ void decode_attention_kernel(const float* qkv, int pos, int T, int H, int D_head,
                                        float* K_cache, float* V_cache,
                                        float* attn_out, int layer, int maxT,
                                        float scale) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H) return;

    // 当前 token 的 Q
    const float* Q = qkv + (pos * 3 * H + h) * D_head;

    // 计算 scores[0..T-1]
    extern __shared__ float sdata[];
    float* scores = sdata;                    // T floats
    float* smax   = sdata + T;                // 1
    float* ssum   = sdata + T + 1;            // 1

    for (int j = tid; j < T; j += blockDim.x) {
        float* Kj = K_cache + ((size_t(layer) * H + h) * maxT + j) * D_head;
        float s = 0;
        for (int d = 0; d < D_head; ++d) s += Q[d] * Kj[d];
        scores[j] = s * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float mm = -FLT_MAX;
        for (int j = 0; j < T; ++j) mm = fmaxf(mm, scores[j]);
        *smax = mm;
        double s = 0;
        for (int j = 0; j < T; ++j) { scores[j] = expf(scores[j] - mm); s += scores[j]; }
        *ssum = float(s);
    }
    __syncthreads();

    float inv = 1.f / (*ssum);
    for (int d = tid; d < D_head; d += blockDim.x) {
        float acc = 0;
        for (int j = 0; j < T; ++j) {
            float* Vj = V_cache + ((size_t(layer) * H + h) * maxT + j) * D_head;
            acc += scores[j] * inv * Vj[d];
        }
        attn_out[(pos * H + h) * D_head + d] = acc;
    }
}

// 将 qkv[0:T-1] 的 K/V 写入 cache[layer, head, pos:pos+T-1, :]
__global__ void append_kv_cache(const float* qkv, int pos, int T, int H, int D_head,
                                float* K_cache, float* V_cache,
                                int layer, int maxT) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= H) return;

    for (int t = 0; t < T; ++t) {
        const float* K = qkv + (t * 3 * H + H + h) * D_head;
        const float* V = qkv + (t * 3 * H + 2 * H + h) * D_head;
        float* K_dst = K_cache + ((size_t(layer) * H + h) * maxT + pos + t) * D_head;
        float* V_dst = V_cache + ((size_t(layer) * H + h) * maxT + pos + t) * D_head;
        for (int d = tid; d < D_head; d += blockDim.x) {
            K_dst[d] = K[d];
            V_dst[d] = V[d];
        }
    }
}

// ====================  HOST HELPERS  ==================================

struct Workspace {
    int maxT = 0;
    DeviceBuffer<int>   d_tok;
    DeviceBuffer<float> x, normed;
    DeviceBuffer<float> qkv;
    DeviceBuffer<float> attn_out, proj_out;
    DeviceBuffer<float> ff_mid, ff_out;
    DeviceBuffer<float> logits;
    DeviceBuffer<float> last_row;
    DeviceBuffer<__half> gemm_in_half;
    DeviceBuffer<float> K_cache, V_cache;

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
        gemm_in_half.allocate(size_t(maxT) * C.d_ff);
        size_t cache_elems = size_t(C.n_layer) * C.n_head * maxT * C.head_dim;
        K_cache.allocate(cache_elems);
        V_cache.allocate(cache_elems);
        CUDA_CHECK(cudaMemset(K_cache.ptr, 0, K_cache.bytes()));
        CUDA_CHECK(cudaMemset(V_cache.ptr, 0, V_cache.bytes()));
    }
};

// 封装 GEMM：N/K 16 整除时走 WMMA，并把 M pad 到 16 的倍数；
// 否则回退 simple_gemm。
static void run_gemm(const float* A_fp32, const __half* B_fp16, float* C_fp32,
                     __half* tmp_half, int M, int N, int K) {
    if (N % 16 != 0 || K % 16 != 0) {
        int total = M * N;
        simple_gemm_kernel<<<(total + 255) / 256, 256>>>(A_fp32, B_fp16, C_fp32, M, N, K);
        KERNEL_CHECK();
        return;
    }
    int Mpad = ((M + 15) / 16) * 16;
    // 先把 padding 区域置 0，再转换实际 M 行
    CUDA_CHECK(cudaMemset(tmp_half, 0, size_t(Mpad) * K * sizeof(__half)));
    fp32_to_fp16<<<(M * K + 255) / 256, 256>>>(A_fp32, tmp_half, M * K);
    KERNEL_CHECK();
    dim3 block(32, 4);
    dim3 grid((N + WMMA_N * 2 - 1) / (WMMA_N * 2),
              (Mpad + WMMA_M * 2 - 1) / (WMMA_M * 2));
    wmma_gemm_kernel<<<grid, block>>>(tmp_half, B_fp16, C_fp32, Mpad, N, K);
    KERNEL_CHECK();
}

// ====================  PREFILL & DECODE  =============================

// prefill: 一次性处理 prompt tokens[0:T0-1]，填充 KV cache，并产出 T0-1 位置 logits。
void prefill(const Weights& W, Workspace& ws, const std::vector<int>& tokens, int T0) {
    auto& C = W.cfg;
    int D = C.d_model;
    if (T0 > ws.maxT) { std::printf("prefill T0=%d > maxT\n", T0); std::exit(1); }

    CUDA_CHECK(cudaMemcpy(ws.d_tok.ptr, tokens.data(), T0 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 eb(64);
    dim3 eg(T0, (D + 63) / 64);
    embed_kernel<<<eg, eb>>>(W.wte.ptr, W.wpe.ptr, ws.d_tok.ptr, ws.x.ptr, T0, D, C.vocab_size);
    KERNEL_CHECK();

    for (int l = 0; l < C.n_layer; ++l) {
        const auto& B = W.blocks[l];

        layernorm_row<<<T0, 256>>>(ws.x.ptr, B.ln1_w.ptr, B.ln1_b.ptr, ws.normed.ptr, T0, D, 1e-5f);
        KERNEL_CHECK();

        run_gemm(ws.normed.ptr, B.qkv_w.ptr, ws.qkv.ptr, ws.gemm_in_half.ptr, T0, 3 * D, D);
        bias_add<<<dim3(T0, (3 * D + 255) / 256), 256>>>(ws.qkv.ptr, B.qkv_b.ptr, T0, 3 * D);
        KERNEL_CHECK();

        // FlashAttention prefill (causal)
        dim3 fag((T0 + Br - 1) / Br, C.n_head);
        flash_attn_v1_kernel<<<fag, 256>>>(ws.qkv.ptr, ws.attn_out.ptr, T0, C.n_head,
                                           1.f / std::sqrt(float(C.head_dim)), true);
        KERNEL_CHECK();

        // 写入 KV cache（位置 0 .. T0-1）
        append_kv_cache<<<C.n_head, 256>>>(ws.qkv.ptr, 0, T0, C.n_head, C.head_dim,
                                           ws.K_cache.ptr, ws.V_cache.ptr, l, ws.maxT);
        KERNEL_CHECK();

        run_gemm(ws.attn_out.ptr, B.proj_w.ptr, ws.proj_out.ptr, ws.gemm_in_half.ptr, T0, D, D);
        bias_add<<<dim3(T0, (D + 255) / 256), 256>>>(ws.proj_out.ptr, B.proj_b.ptr, T0, D);
        KERNEL_CHECK();

        residual_add<<<(T0 * D + 255) / 256, 256>>>(ws.x.ptr, ws.proj_out.ptr, T0 * D);
        KERNEL_CHECK();

        layernorm_row<<<T0, 256>>>(ws.x.ptr, B.ln2_w.ptr, B.ln2_b.ptr, ws.normed.ptr, T0, D, 1e-5f);
        KERNEL_CHECK();

        run_gemm(ws.normed.ptr, B.fc_w.ptr, ws.ff_mid.ptr, ws.gemm_in_half.ptr, T0, C.d_ff, D);
        bias_add<<<dim3(T0, (C.d_ff + 255) / 256), 256>>>(ws.ff_mid.ptr, B.fc_b.ptr, T0, C.d_ff);
        gelu_kernel<<<(T0 * C.d_ff + 255) / 256, 256>>>(ws.ff_mid.ptr, T0 * C.d_ff);
        KERNEL_CHECK();

        run_gemm(ws.ff_mid.ptr, B.fcp_w.ptr, ws.ff_out.ptr, ws.gemm_in_half.ptr, T0, D, C.d_ff);
        bias_add<<<dim3(T0, (D + 255) / 256), 256>>>(ws.ff_out.ptr, B.fcp_b.ptr, T0, D);
        KERNEL_CHECK();

        residual_add<<<(T0 * D + 255) / 256, 256>>>(ws.x.ptr, ws.ff_out.ptr, T0 * D);
        KERNEL_CHECK();
    }

    layernorm_row<<<T0, 256>>>(ws.x.ptr, W.lnf_w.ptr, W.lnf_b.ptr, ws.normed.ptr, T0, D, 1e-5f);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(ws.last_row.ptr, ws.normed.ptr + (T0 - 1) * D,
                          D * sizeof(float), cudaMemcpyDeviceToDevice));
    gemv_logits<<<(C.vocab_size + 255) / 256, 256>>>(ws.last_row.ptr, W.wte.ptr, ws.logits.ptr, D, C.vocab_size);
    KERNEL_CHECK();
}

// decode: 处理第 pos 个新 token（d_tok 中已放 1 个 id），使用并追加 KV cache。
void decode(const Weights& W, Workspace& ws, int pos) {
    auto& C = W.cfg;
    int D = C.d_model;
    int T = pos + 1;  // 当前序列长度

    dim3 eb(64);
    dim3 eg(1, (D + 63) / 64);
    embed_kernel<<<eg, eb>>>(W.wte.ptr, W.wpe.ptr, ws.d_tok.ptr, ws.x.ptr, 1, D, C.vocab_size);
    KERNEL_CHECK();

    for (int l = 0; l < C.n_layer; ++l) {
        const auto& B = W.blocks[l];

        layernorm_row<<<1, 256>>>(ws.x.ptr, B.ln1_w.ptr, B.ln1_b.ptr, ws.normed.ptr, 1, D, 1e-5f);
        KERNEL_CHECK();

        run_gemm(ws.normed.ptr, B.qkv_w.ptr, ws.qkv.ptr, ws.gemm_in_half.ptr, 1, 3 * D, D);
        bias_add<<<dim3(1, (3 * D + 255) / 256), 256>>>(ws.qkv.ptr, B.qkv_b.ptr, 1, 3 * D);
        KERNEL_CHECK();

        // 追加当前 token 的 K/V 到 cache
        append_kv_cache<<<C.n_head, 256>>>(ws.qkv.ptr, pos, 1, C.n_head, C.head_dim,
                                           ws.K_cache.ptr, ws.V_cache.ptr, l, ws.maxT);
        KERNEL_CHECK();

        // 单 query attention: Q vs cache[0:T]
        int shm_bytes = T * sizeof(float) + 2 * sizeof(float);
        decode_attention_kernel<<<C.n_head, 256, shm_bytes>>>(
            ws.qkv.ptr, 0, T, C.n_head, C.head_dim,
            ws.K_cache.ptr, ws.V_cache.ptr, ws.attn_out.ptr, l, ws.maxT,
            1.f / std::sqrt(float(C.head_dim)));
        KERNEL_CHECK();

        run_gemm(ws.attn_out.ptr, B.proj_w.ptr, ws.proj_out.ptr, ws.gemm_in_half.ptr, 1, D, D);
        bias_add<<<dim3(1, (D + 255) / 256), 256>>>(ws.proj_out.ptr, B.proj_b.ptr, 1, D);
        KERNEL_CHECK();

        residual_add<<<(D + 255) / 256, 256>>>(ws.x.ptr, ws.proj_out.ptr, D);
        KERNEL_CHECK();

        layernorm_row<<<1, 256>>>(ws.x.ptr, B.ln2_w.ptr, B.ln2_b.ptr, ws.normed.ptr, 1, D, 1e-5f);
        KERNEL_CHECK();

        run_gemm(ws.normed.ptr, B.fc_w.ptr, ws.ff_mid.ptr, ws.gemm_in_half.ptr, 1, C.d_ff, D);
        bias_add<<<dim3(1, (C.d_ff + 255) / 256), 256>>>(ws.ff_mid.ptr, B.fc_b.ptr, 1, C.d_ff);
        gelu_kernel<<<(C.d_ff + 255) / 256, 256>>>(ws.ff_mid.ptr, C.d_ff);
        KERNEL_CHECK();

        run_gemm(ws.ff_mid.ptr, B.fcp_w.ptr, ws.ff_out.ptr, ws.gemm_in_half.ptr, 1, D, C.d_ff);
        bias_add<<<dim3(1, (D + 255) / 256), 256>>>(ws.ff_out.ptr, B.fcp_b.ptr, 1, D);
        KERNEL_CHECK();

        residual_add<<<(D + 255) / 256, 256>>>(ws.x.ptr, ws.ff_out.ptr, D);
        KERNEL_CHECK();
    }

    layernorm_row<<<1, 256>>>(ws.x.ptr, W.lnf_w.ptr, W.lnf_b.ptr, ws.normed.ptr, 1, D, 1e-5f);
    KERNEL_CHECK();

    gemv_logits<<<(C.vocab_size + 255) / 256, 256>>>(ws.normed.ptr, W.wte.ptr, ws.logits.ptr, D, C.vocab_size);
    KERNEL_CHECK();
}

// ====================  SAMPLING  ======================================

static int sample_token(const float* d_logits, int V,
                        float temperature, int top_k, std::mt19937& rng) {
    std::vector<float> logits(V);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits, V * sizeof(float), cudaMemcpyDeviceToHost));

    for (float& v : logits) v /= temperature;

    if (top_k > 0 && top_k < V) {
        std::vector<float> copy = logits;
        std::nth_element(copy.begin(), copy.begin() + (top_k - 1), copy.end(), std::greater<float>());
        float threshold = copy[top_k - 1];
        for (float& v : logits) if (v < threshold) v = -1e30f;
    }

    float mx = -FLT_MAX;
    for (float v : logits) mx = std::max(mx, v);
    double sum = 0;
    for (float& v : logits) {
        v = std::exp(v - mx);
        sum += v;
    }
    float inv = float(1.0 / sum);
    for (float& v : logits) v *= inv;

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
    std::string tokens_csv   = "15496,11,612,318";
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
    std::printf("[mini_llm_adv2] prompt tokens (%zu): ", tokens.size());
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");
    std::printf("[mini_llm_adv2] temperature=%.3f top_k=%d seed=%u max_new=%d\n",
                temperature, top_k, seed, max_new);

    std::mt19937 rng(seed);

    // 1) prefill prompt
    int T0 = (int)tokens.size();
    prefill(W, ws, tokens, T0);
    int next = sample_token(ws.logits.ptr, W.cfg.vocab_size, temperature, top_k, rng);
    tokens.push_back(next);
    std::printf("[mini_llm_adv2] prefill -> %d\n", next);

    // 2) decode loop：处理最后一个已生成 token，产出下一个 token
    for (int step = 1; step < max_new; ++step) {
        if ((int)tokens.size() >= W.cfg.max_seq) {
            std::printf("[mini_llm_adv2] reached max_seq=%d, stop.\n", W.cfg.max_seq);
            break;
        }
        int pos = (int)tokens.size() - 1;  // 最后生成 token 的位置
        int tok = tokens.back();
        CUDA_CHECK(cudaMemcpy(ws.d_tok.ptr, &tok, sizeof(int), cudaMemcpyHostToDevice));
        decode(W, ws, pos);
        int next_tok = sample_token(ws.logits.ptr, W.cfg.vocab_size, temperature, top_k, rng);
        tokens.push_back(next_tok);
        std::printf("[mini_llm_adv2] step %2d -> %d\n", step, next_tok);
    }

    std::printf("[mini_llm_adv2] final tokens (%zu): ", tokens.size());
    for (int t : tokens) std::printf("%d ", t);
    std::printf("\n");
    return 0;
}
