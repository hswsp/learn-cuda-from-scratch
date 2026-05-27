// =============================================================================
// matmul_tiled.cu — classic tiled matmul, one tile per block (BM=BN=BK=32).
//
// 算法:
//   每个 block 计算 C 的 32x32 子块.
//   K 维度分成 K/32 个 tile, 每个 tile:
//     1) 协作把 A 的 32x32 slice + B 的 32x32 slice 拉进 shared mem
//     2) __syncthreads()
//     3) thread (ty,tx) 在 shared mem 上做 32 个 fmadd 累加到 reg
//     4) __syncthreads(), 进入下一 K-tile
//   最后写回 C.
//
// 提升:  global 读取量从 O(MNK) 降到 O(MNK / BK), 这里降 32 倍.
//
// 对应 HTML: docs/ch06-tile/index.html#tiled
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

constexpr int BM = 32, BN = 32, BK = 32;

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = by * BM + ty;
    int col = bx * BN + tx;

    float acc = 0.f;

    // Walk over K tiles
    for (int kt = 0; kt < K; kt += BK) {
        // load A[row, kt+tx] and B[kt+ty, col] (coalesced since tx varies fastest)
        if (row < M && kt + tx < K) As[ty][tx] = A[row * K + kt + tx];
        else                        As[ty][tx] = 0.f;
        if (kt + ty < K && col < N) Bs[ty][tx] = B[(kt + ty) * N + col];
        else                        Bs[ty][tx] = 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
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

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    GpuTimer t;
    t.start();
    matmul_tiled<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, N, K);
    t.stop(); KERNEL_CHECK();
    float ms = t.ms();
    dC.copy_to_host(hC.data());

    cpu_ref::gemm(hA.data(), hB.data(), hC_ref.data(), M, N, K);
    report("matmul_tiled", allclose(hC, hC_ref, 1e-2f, 1e-2f));

    double ops = matmul_ops(M, N, K);
    std::printf("tiled  MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, ms, gflops(ops, ms));
    return 0;
}
