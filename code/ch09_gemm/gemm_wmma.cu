// =============================================================================
// gemm_wmma.cu — Tensor Core GEMM using WMMA API (16x16x16 fp16, fp32 accumulator).
//
// 学习目标:
//   1. 第一次摸 Tensor Core: nvcuda::wmma::fragment + load + mma_sync + store
//   2. fp16 输入, fp32 累加 → 既快又稳
//   3. 与 cuBLAS Tensor Core 比, 一般能拿到 ~70-80% 性能 (手写不调优)
//
// 限制:
//   - 仅 sm_70+ 支持 (Volta / Turing / Ampere / Ada / Hopper)
//   - M/N/K 必须能被 16 整除 (这里假设满足)
//
// 对应 HTML: docs/ch09-gemm/index.html#wmma
// =============================================================================
#include "../common/cuda_utils.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
// each warp computes one 16x16 output tile.
// block = 4 warps = 128 threads → 64x32 output tile per block.

__global__ void gemm_wmma_kernel(const __half* A, const __half* B, float* C,
                                 int M, int N, int K) {
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warp_m  = blockIdx.y * (blockDim.y) + warp_id / 2;
    int warp_n  = blockIdx.x * 2 + warp_id % 2;     // 2 warp_n per block

    int row0 = warp_m * WMMA_M;
    int col0 = warp_n * WMMA_N;
    if (row0 >= M || col0 >= N) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int kt = 0; kt < K; kt += WMMA_K) {
        const __half* a_ptr = A + row0 * K + kt;
        const __half* b_ptr = B + kt   * N + col0;
        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + row0 * N + col0, c_frag, N, wmma::mem_row_major);
}

int main(int argc, char** argv) {
    int M = arg_int(argc, argv, "M", 2048);
    int N = arg_int(argc, argv, "N", 2048);
    int K = arg_int(argc, argv, "K", 2048);
    if (M % 16 || N % 16 || K % 16) { std::printf("M/N/K must be multiple of 16\n"); return 1; }

    std::vector<float> hAf(M * K), hBf(K * N), hC(M * N);
    for (auto& v : hAf) v = (rand() & 0xff) * 1e-3f;
    for (auto& v : hBf) v = (rand() & 0xff) * 1e-3f;
    std::vector<__half> hA(M * K), hB(K * N);
    for (int i = 0; i < M * K; ++i) hA[i] = __float2half(hAf[i]);
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half(hBf[i]);

    __half *dA, *dB; float* dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 block(32, 4);                 // 4 warps per block
    dim3 grid(N / (WMMA_N * 2), M / (WMMA_M * 2));

    GpuTimer t; t.start();
    gemm_wmma_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    t.stop(); KERNEL_CHECK();
    float ms = t.ms();
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("wmma fp16  MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, ms, 2.0 * M * N * K / (ms * 1e6));
    std::printf("hC[0,0]=%.4f  hC[7,7]=%.4f\n", hC[0], hC[7 * N + 7]);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
