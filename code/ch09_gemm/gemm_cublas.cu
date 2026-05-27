// =============================================================================
// gemm_cublas.cu — cuBLAS sgemm baseline for comparison.
//
// 注意: cuBLAS 默认 column-major. 对 row-major C = A @ B 可用恒等:
//   C_row^T = B_row^T @ A_row^T,
//   而 row-major M 转 column-major 就是它的 transpose.
// 我们直接用 C^T = B^T @ A^T = cuBLAS(B, A) (各种 layout 都标 N), 结果也是 N×M col-major,
// 它就等于 row-major (M, N) 的 C —— 巧妙又省一次 transpose.
//
// 对应 HTML: docs/ch09-gemm/index.html#baseline
// =============================================================================
#include "../common/cuda_utils.h"
#include <cublas_v2.h>
#include <cstdio>

int main(int argc, char** argv) {
    int M = arg_int(argc, argv, "M", 2048);
    int N = arg_int(argc, argv, "N", 2048);
    int K = arg_int(argc, argv, "K", 2048);

    auto hA = make_random(M * K, 1, 0.01f);
    auto hB = make_random(K * N, 2, 0.01f);
    DeviceBuffer<float> dA(M*K), dB(K*N), dC(M*N);
    dA.copy_from_host(hA.data()); dB.copy_from_host(hB.data());

    cublasHandle_t h; cublasCreate(&h);

    float alpha = 1.f, beta = 0.f;
    GpuTimer t;
    // C_row(M,N) = A_row(M,K) * B_row(K,N)
    // == C_col(N,M)^T = (B_col^T) * (A_col^T)
    // call: sgemm(N, N, N, M, K, B, N, A, K, &C, N)
    t.start();
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB.ptr, N,
                dA.ptr, K,
                &beta,
                dC.ptr, N);
    t.stop(); KERNEL_CHECK();

    std::printf("cuBLAS sgemm  MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, t.ms(), gflops(matmul_ops(M, N, K), t.ms()));
    cublasDestroy(h);
    return 0;
}
