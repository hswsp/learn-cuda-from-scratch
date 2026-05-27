// =============================================================================
// mem_modes.cu — pageable / pinned / unified / device memory throughput.
//
// 学习目标:
//   1. cudaMalloc / cudaMallocHost / cudaMallocManaged 三种分配的差别
//   2. 各自典型用途与陷阱
//
// 对应 HTML: docs/ch05-memory/index.html#modes
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>
#include <vector>

__global__ void touch(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] + 1.0f;
}

int main() {
    const int N = 1 << 22;                  // 4M floats = 16 MiB
    size_t bytes = N * sizeof(float);
    int block = 256, grid = (N + block - 1) / block;

    GpuTimer t;
    float ms;

    // ---- 1) device-only ----
    {
        DeviceBuffer<float> d(N);
        std::vector<float> h(N, 1.0f);
        d.copy_from_host(h.data());
        t.start(); touch<<<grid, block>>>(d.ptr, N); t.stop(); KERNEL_CHECK();
        ms = t.ms();
        std::printf("  device-only           kernel %.3f ms (%.1f GB/s mem ops)\n",
                    ms, 2.0 * bytes / (ms * 1e6));
    }

    // ---- 2) unified (managed) ----
    {
        float* u; CUDA_CHECK(cudaMallocManaged(&u, bytes));
        for (int i = 0; i < N; ++i) u[i] = 1.0f;     // host writes — pages on CPU
        t.start(); touch<<<grid, block>>>(u, N); t.stop(); KERNEL_CHECK();
        // 第一次 launch 会触发 page migration（CPU→GPU），通常较慢
        ms = t.ms();
        std::printf("  unified  (first run)  kernel %.3f ms  (含 page migration)\n", ms);

        t.start(); touch<<<grid, block>>>(u, N); t.stop(); KERNEL_CHECK();
        ms = t.ms();
        std::printf("  unified  (warm)       kernel %.3f ms  (页已在 GPU)\n", ms);

        CUDA_CHECK(cudaFree(u));
    }

    // ---- 3) zero-copy (pinned, accessed by GPU over PCIe) ----
    {
        float* h_pinned;
        CUDA_CHECK(cudaHostAlloc(&h_pinned, bytes,
                                 cudaHostAllocMapped | cudaHostAllocPortable));
        for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;
        float* d_view;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_view, h_pinned, 0));
        t.start(); touch<<<grid, block>>>(d_view, N); t.stop(); KERNEL_CHECK();
        ms = t.ms();
        std::printf("  zero-copy (PCIe每读) kernel %.3f ms  (奇慢 — 全程走 PCIe)\n", ms);
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }
    return 0;
}
