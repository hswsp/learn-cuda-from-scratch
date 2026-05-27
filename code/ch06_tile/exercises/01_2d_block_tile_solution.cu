// Solution: 4x4 register tile per thread, BM=BN=64 BK=16, block 16x16.
#include "../../common/cuda_utils.h"
#include "../../common/cpu_ref.h"
#include "../../common/check.h"

constexpr int BM = 64, BN = 64, BK = 16;
constexpr int TM = 4,  TN = 4;     // each thread computes 4x4 of C

__global__ void matmul_2d_tile(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

    int row0 = by * BM + ty * TM;
    int col0 = bx * BN + tx * TN;

    float acc[TM][TN] = {0};

    // each tile: load BM*BK + BK*BN floats with the 16x16 thread block
    // 16*16 = 256 threads, BM*BK = 1024 elems → each thread loads 4 of A, 4 of B
    const int threads_per_block = 16 * 16;
    const int tid = ty * 16 + tx;

    for (int kt = 0; kt < K; kt += BK) {
        // load As (BM=64 x BK=16 = 1024 floats), 256 threads × 4 each
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_id = i * threads_per_block + tid;
            int r = load_id / BK, c = load_id % BK;
            int gr = by * BM + r, gc = kt + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        // load Bs (BK=16 x BN=64 = 1024 floats)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_id = i * threads_per_block + tid;
            int r = load_id / BN, c = load_id % BN;
            int gr = kt + r, gc = bx * BN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();

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
    int M = arg_int(argc, argv, "M", 1024);
    int N = arg_int(argc, argv, "N", 1024);
    int K = arg_int(argc, argv, "K", 1024);
    auto hA = make_random(M * K, 1, 0.01f);
    auto hB = make_random(K * N, 2, 0.01f);
    std::vector<float> hC(M * N), href(M * N);
    DeviceBuffer<float> dA(M*K), dB(K*N), dC(M*N);
    dA.copy_from_host(hA.data()); dB.copy_from_host(hB.data());

    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    GpuTimer t; t.start();
    matmul_2d_tile<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, N, K);
    t.stop(); KERNEL_CHECK();
    float ms = t.ms();
    dC.copy_to_host(hC.data());
    cpu_ref::gemm(hA.data(), hB.data(), href.data(), M, N, K);
    report("matmul_2d_tile", allclose(hC, href, 1e-2f, 1e-2f));
    std::printf("MNK=%dx%dx%d  %.3f ms  %.1f GFLOPS\n",
                M, N, K, ms, gflops(matmul_ops(M, N, K), ms));
    return 0;
}
