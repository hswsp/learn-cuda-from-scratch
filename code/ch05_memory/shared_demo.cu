// =============================================================================
// shared_demo.cu — first taste of __shared__ memory: tile-based reduction.
//
// 学习目标:
//   1. 用 __shared__ 在 block 内开一块片上 SRAM
//   2. __syncthreads() 的必要性
//   3. 看到 shared mem 比直接走 global mem 快多少
//
// 这里做的是: 把 N 个 float 求和 → 输出每个 block 内的部分和.
// 完整的 device-wide reduction 在 Ch07 讲, 这里只演示 shared mem 用法.
//
// 对应 HTML: docs/ch05-memory/index.html#shared
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

__global__ void block_sum_shared(const float* x, float* partial, int n) {
    __shared__ float sdata[256];          // block 内的临时桶
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 每 thread 从 global 拉一个 float 进 shared
    sdata[tid] = (gid < n) ? x[gid] : 0.0f;
    __syncthreads();                       // 等所有 thread 都填完

    // 2. block 内 tree reduction (会在 Ch07 优化)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 3. 0 号 thread 把本 block 的部分和写回 global
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 1 << 20;                 // 1M
    auto hx = make_random(N, 7);
    DeviceBuffer<float> dx(N);
    dx.copy_from_host(hx.data());

    int block = 256;
    int grid  = (N + block - 1) / block;
    DeviceBuffer<float> dpart(grid);

    block_sum_shared<<<grid, block>>>(dx.ptr, dpart.ptr, N);
    KERNEL_CHECK();

    std::vector<float> hpart(grid);
    dpart.copy_to_host(hpart.data());
    float sum = 0; for (float v : hpart) sum += v;
    float ref = cpu_ref::sum(hx.data(), N);
    std::printf("GPU partial-sum:  %.6f\n", sum);
    std::printf("CPU reference :   %.6f\n", ref);
    std::printf("abs diff      :   %.3e  (浮点累加顺序不同, 1e-3 量级正常)\n",
                std::fabs(sum - ref));
    return 0;
}
