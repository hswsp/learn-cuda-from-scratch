// =============================================================================
// flash_attn_v1.cu — single-head FlashAttention v1 in fp32 (教学版).
//
// 核心算法 (Tri Dao 2022):
//   把 Q 沿 row 切 Br 行 / K, V 沿 col 切 Bc 行.
//   对每个 Q 块 (Br, D):
//     初始化 O_i = 0,  l_i = 0,  m_i = -inf
//     for j in K/V 块:
//       1) load K_j, V_j (Bc, D) 到 shared
//       2) S_ij = Q_i @ K_j^T * scale          (Br, Bc) in reg/shared
//       3) m_new = max(m_i, rowmax(S_ij))
//       4) P_ij  = exp(S_ij - m_new)
//       5) l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
//       6) O_i   = (exp(m_i - m_new) * l_i / l_new) * O_i +
//                  (1 / l_new) * P_ij @ V_j      ← 含一次 rescale
//       7) m_i = m_new,  l_i = l_new
//     end
//     写 O_i 回 HBM
//
// 结果: 全程不物化 T×T 中间矩阵, HBM 流量从 O(T²) 降到 O(T·D).
//       特别适合长 context.
//
// 这里写最简版: 单 head, fp32, 单 block 处理 1 个 Q tile.
// 为简化, 不做 cp.async / Tensor Core (那是 v2/v3 的事).
//
// 对应 HTML: docs/ch12-flashattn/index.html
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cfloat>
#include <cmath>

constexpr int Br = 64;   // Q tile rows
constexpr int Bc = 64;   // K/V tile rows
constexpr int Dm = 64;   // head dim (固定方便教学)

__global__ void flash_attn_v1_kernel(const float* Q, const float* K, const float* V,
                                     float* O, int T, float scale, bool causal) {
    __shared__ float Qs[Br][Dm];
    __shared__ float Ks[Bc][Dm];
    __shared__ float Vs[Bc][Dm];
    __shared__ float Ss[Br][Bc];

    int tid = threadIdx.x;           // 0 .. Br*Bc/?? — we'll use blockDim.x = Br*4 (= 256)
    int row_in_block = tid / Bc;     // 0 .. Br/?? 取决于 blockDim
    int col_in_block = tid % Bc;

    int qi_base = blockIdx.x * Br;   // 这个 block 处理的 Q 块起始行

    // ---- 1) load Q block ----
    // blockDim.x = Br * Dm / per_thread → 用最直接的: 每 thread 负责若干 (r, c)
    // 简化: blockDim.x = 256, Br=64, Dm=64 → 每 thread load Br*Dm/256 = 16 floats
    for (int i = tid; i < Br * Dm; i += blockDim.x) {
        int r = i / Dm, c = i % Dm;
        int gr = qi_base + r;
        Qs[r][c] = (gr < T) ? Q[gr * Dm + c] : 0.f;
    }
    __syncthreads();

    // ---- 2) per-thread O accumulator (Br rows × Dm = 4096 cells > regs) ----
    // 简化教学: 我们让 64 个 thread 各管 1 个 row, 每个 thread 在 reg 里维护 Dm 大小的 O
    // 因此 blockDim.x = Br = 64.  (这样 Br*Bc 的 S 访问不太理想, 但代码清晰)
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
        // -- load K, V tile --
        for (int i = tid; i < Bc * Dm; i += blockDim.x) {
            int r = i / Dm, c = i % Dm;
            int gr = kj_base + r;
            Ks[r][c] = (gr < T) ? K[gr * Dm + c] : 0.f;
            Vs[r][c] = (gr < T) ? V[gr * Dm + c] : 0.f;
        }
        __syncthreads();

        // -- compute S_ij = Q_i @ K_j^T * scale --   each thread does one S row
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

        // -- online softmax + O update --
        if (tid < Br) {
            float m_old = m_s[tid], l_old = l_s[tid];
            // row max of S_ij
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

            // O_i = rescale * O_i + P_ij @ V_j   (这里 norm by l_new 留到最后)
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

    // ---- 3) final O /= l, write back ----
    if (tid < Br) {
        float inv = 1.f / l_s[tid];
        int gr = qi_base + tid;
        if (gr < T) {
            for (int d = 0; d < Dm; ++d) O[gr * Dm + d] = O_s[tid][d] * inv;
        }
    }
}

int main(int argc, char** argv) {
    int T = arg_int(argc, argv, "T", 256);
    if (T % Br) { std::printf("T must be multiple of Br=%d\n", Br); return 1; }
    bool causal = true;

    auto hQ = make_random(T * Dm, 1);
    auto hK = make_random(T * Dm, 2);
    auto hV = make_random(T * Dm, 3);
    std::vector<float> hO(T * Dm), href(T * Dm);

    DeviceBuffer<float> dQ(T*Dm), dK(T*Dm), dV(T*Dm), dO(T*Dm);
    dQ.copy_from_host(hQ.data());
    dK.copy_from_host(hK.data());
    dV.copy_from_host(hV.data());

    float scale = 1.f / std::sqrt(float(Dm));
    int nblocks = T / Br;

    GpuTimer t; t.start();
    flash_attn_v1_kernel<<<nblocks, 256>>>(dQ.ptr, dK.ptr, dV.ptr, dO.ptr, T, scale, causal);
    t.stop(); KERNEL_CHECK();
    dO.copy_to_host(hO.data());

    cpu_ref::attention(hQ.data(), hK.data(), hV.data(), href.data(), T, Dm, causal);
    report("flash_attn_v1", allclose(hO, href, 1e-3f, 1e-3f));
    std::printf("T=%d D=%d  %.3f ms  无 T×T 中间矩阵 (省 %.1f MiB)\n",
                T, Dm, t.ms(), T * T * 4 / (1024.0 * 1024.0));
    return 0;
}
