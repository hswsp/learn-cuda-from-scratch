// =============================================================================
// occupancy_probe.cu — ask the runtime: what's my occupancy for kernel X?
//
// 学习目标:
//   1. 使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 查询占用率
//   2. 看到 register / shared mem 用量如何反过来限制 block 数
//
// 对应 HTML: docs/ch04-arch/index.html#occupancy
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

// 轻量 kernel: 用很少寄存器, occupancy 高
__global__ void light_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f + 1.0f;
}

// 重 kernel: 用很大的本地数组 → 寄存器/栈溢出 → occupancy 降低
__global__ void heavy_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc[64];
    #pragma unroll
    for (int k = 0; k < 64; ++k) acc[k] = (k + 1) * x[i % n];
    float s = 0;
    #pragma unroll
    for (int k = 0; k < 64; ++k) s += acc[k] * acc[k];
    if (i < n) x[i] = s;
}

template <typename K>
void report_occupancy(const char* name, K kernel, int block) {
    int active = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, (const void*)kernel, block, /*dynSmem=*/0));
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    int warps_per_sm = active * (block / 32);
    int max_warps = p.maxThreadsPerMultiProcessor / 32;
    std::printf("  %-15s block=%4d  active blocks/SM=%2d  warps/SM=%2d/%d  occ=%5.1f%%\n",
                name, block, active, warps_per_sm, max_warps,
                100.0 * warps_per_sm / max_warps);
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    std::printf("GPU: %s (sm_%d%d, %d SMs)\n\n", p.name, p.major, p.minor, p.multiProcessorCount);

    for (int b : {64, 128, 256, 512, 1024}) {
        std::printf("--- block = %d ---\n", b);
        report_occupancy("light_kernel", light_kernel, b);
        report_occupancy("heavy_kernel", heavy_kernel, b);
    }
    return 0;
}
