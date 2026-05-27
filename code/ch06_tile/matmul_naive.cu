// =============================================================================
// matmul_naive.cu — the simplest possible matmul, 1 thread per output cell.
//
// 学习目标:
//   1. 写出 GEMM 的最朴素实现，理解为什么慢
//   2. 计算 arithmetic intensity: 每个输出 cell 读 2K 个 float, 算 2K FLOP
//      → ratio = 2 FLOPs / 8 Bytes = 0.25 FLOP/B → 严重 memory-bound
//   3. 与下一节 tiled 版做对照
//
// 对应 HTML: docs/ch06-tile/index.html#naive
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

int main(int argc, char** argv) {
    int M = arg_int(argc, argv, "M", 1024);
    int N = arg_int(argc, argv, "N", 1024);
    int K = arg_int(argc, argv, "K", 1024);

    auto hA = make_random(M * K, 1, 0.01f);
    auto hB = make_random(K * N, 2, 0.01f);
    std::vector<float> hC(M * N), hC_ref(M * N);

    DeviceBuffer<float> dA(M * K), dB(K * N), dC(M * N);
    dA.copy_from_host(hA.data()); dB.copy_from_host(hB.data());

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    GpuTimer t;
    t.start();
    matmul_naive<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, N, K);
    t.stop(); KERNEL_CHECK();
    float ms = t.ms();
    dC.copy_to_host(hC.data());

    // CPU ref (slow for 1024^3 ~2 sec; smaller for sanity if you want)
    cpu_ref::gemm(hA.data(), hB.data(), hC_ref.data(), M, N, K);
    report("matmul_naive", allclose(hC, hC_ref, 1e-2f, 1e-2f));

    double ops = matmul_ops(M, N, K);
    std::printf("MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, ms, gflops(ops, ms));
    return 0;
}
