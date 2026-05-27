// =============================================================================
// matrix_add.cu — 2D grid/block for element-wise matrix add.
//
// 学习目标:
//   1. 2D 索引: row = blockIdx.y * blockDim.y + threadIdx.y
//              col = blockIdx.x * blockDim.x + threadIdx.x
//   2. 行优先 (row-major) 布局: idx = row * cols + col
//   3. 边界判断（M, N 不必是 block 维度的整数倍）
//
// 对应 HTML: docs/ch03-threads/index.html#matrix-add
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

__global__ void mat_add_2d(const float* A, const float* B, float* C, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // 0..N-1
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 0..M-1
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    int M = arg_int(argc, argv, "M", 1023);   // not a multiple of block (on purpose)
    int N = arg_int(argc, argv, "N", 1025);

    auto hA = make_random(M * N, 1);
    auto hB = make_random(M * N, 2);
    std::vector<float> hC(M * N);

    DeviceBuffer<float> dA(M * N), dB(M * N), dC(M * N);
    dA.copy_from_host(hA.data());
    dB.copy_from_host(hB.data());

    dim3 block(16, 16);                                // 256 threads/block
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    mat_add_2d<<<grid, block>>>(dA.ptr, dB.ptr, dC.ptr, M, N);
    KERNEL_CHECK();
    dC.copy_to_host(hC.data());

    std::vector<float> hC_ref(M * N);
    cpu_ref::vec_add(hA.data(), hB.data(), hC_ref.data(), M * N);  // same op
    report("mat_add", allclose(hC, hC_ref));
    std::printf("grid = (%u, %u),  block = (%u, %u)\n",
                grid.x, grid.y, block.x, block.y);
    return 0;
}
