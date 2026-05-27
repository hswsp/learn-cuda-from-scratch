// =============================================================================
// gemm_register_tile.cu — 2D register-tiled GEMM (BM=128, BN=128, BK=16, TM=TN=8).
//
// 学习目标:
//   1. 在 Ch6 tiled matmul 基础上进一步: 每 thread 算 8x8 个 cell, 寄存器复用 64×
//   2. 看到 GFLOPS 接近 cuBLAS (非 Tensor Core 版) 的水平
//
// 算法 (Cutlass 风格 mainloop 简化):
//   - block 处理 C 的 128x128 tile, 用 16x16 thread (256 thread)
//   - 每 thread 持有 8x8 个 register accumulator
//   - K 维分 BK=16 切片:
//        协作把 A 的 128x16 + B 的 16x128 加载到 shared
//        内层 K 循环: 每 step 把 As 的 8 行 + Bs 的 8 列读到 reg, 做 64 个 fmadd
//
// 对应 HTML: docs/ch09-gemm/index.html#reg-tile
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

constexpr int BM = 128, BN = 128, BK = 16;
constexpr int TM = 8,   TN = 8;          // per-thread reg tile
constexpr int TX = 16,  TY = 16;         // thread block dims (16x16=256)

__global__ void gemm_reg_tile(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * TX + tx;

    int row0 = by * BM + ty * TM;
    int col0 = bx * BN + tx * TN;

    float acc[TM][TN] = {0};

    constexpr int threads = TX * TY;     // 256
    // 每 tile 加载 BM*BK = 2048 floats / 256 threads = 8 per thread for A
    // similarly for B (BK*BN = 2048)

    for (int kt = 0; kt < K; kt += BK) {
        // ---- load A tile ----
        #pragma unroll
        for (int i = 0; i < (BM * BK) / threads; ++i) {
            int idx = i * threads + tid;
            int r = idx / BK, c = idx % BK;
            int gr = by * BM + r, gc = kt + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // ---- load B tile ----
        #pragma unroll
        for (int i = 0; i < (BK * BN) / threads; ++i) {
            int idx = i * threads + tid;
            int r = idx / BN, c = idx % BN;
            int gr = kt + r, gc = bx * BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

        // ---- mainloop: TM*TN fmadd per K step ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a_reg[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b_reg[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = row0 + i, c = col0 + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int M = arg_int(argc, argv, "M", 2048);
    int N = arg_int(argc, argv, "N", 2048);
    int K = arg_int(argc, argv, "K", 2048);
    auto hA = make_random(M * K, 1, 0.01f);
    auto hB = make_random(K * N, 2, 0.01f);
    std::vector<float> hC(M * N), href(M * N);
    DeviceBuffer<float> dA(M*K), dB(K*N), dC(M*N);
    dA.copy_from_host(hA.data()); dB.copy_from_host(hB.data());

    dim3 block(TX, TY);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    GpuTimer t; t.start();
    gemm_reg_tile<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, N, K);
    t.stop(); KERNEL_CHECK();
    float ms = t.ms();
    dC.copy_to_host(hC.data());

    if (M <= 512) {
        cpu_ref::gemm(hA.data(), hB.data(), href.data(), M, N, K);
        report("gemm_reg_tile", allclose(hC, href, 1e-2f, 1e-2f));
    }
    std::printf("reg-tile  MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, ms, gflops(matmul_ops(M, N, K), ms));
    return 0;
}
