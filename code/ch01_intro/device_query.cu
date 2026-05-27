// =============================================================================
// device_query.cu — print hardware capabilities of every visible GPU.
//
// 学习目标:
//   1. 确认 CUDA Toolkit + 驱动正常工作
//   2. 看清自己 GPU 的硬件参数（SM 数、显存、Tensor Core 代际）
//   3. 学会用 cudaGetDeviceProperties 查询硬件信息
//
// 预期输出:
//   ============================================================
//   GPU 0: Tesla T4
//     Compute capability: 7.5  (sm_75, Turing)
//     SM count:           40
//     Global memory:      14.6 GiB
//     ...
//
// 对应 HTML: docs/ch01-intro/index.html#device-query
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

static const char* arch_name(int major, int minor) {
    if (major == 9) return "Hopper";
    if (major == 8 && minor >= 9) return "Ada Lovelace";
    if (major == 8) return "Ampere";
    if (major == 7 && minor >= 5) return "Turing";
    if (major == 7) return "Volta";
    if (major == 6) return "Pascal";
    if (major == 5) return "Maxwell";
    return "Older";
}

int main() {
    int n_dev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_dev));
    if (n_dev == 0) {
        std::printf("No CUDA-capable GPU found.\n");
        return 1;
    }

    int driver_v = 0, runtime_v = 0;
    cudaDriverGetVersion(&driver_v);
    cudaRuntimeGetVersion(&runtime_v);
    std::printf("CUDA driver:  %d.%d\n", driver_v / 1000, (driver_v % 100) / 10);
    std::printf("CUDA runtime: %d.%d\n", runtime_v / 1000, (runtime_v % 100) / 10);
    std::printf("Devices:      %d\n\n", n_dev);

    for (int d = 0; d < n_dev; ++d) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        std::printf("============================================================\n");
        std::printf("GPU %d: %s\n", d, p.name);
        std::printf("  Compute capability : %d.%d  (sm_%d%d, %s)\n",
                    p.major, p.minor, p.major, p.minor, arch_name(p.major, p.minor));
        std::printf("  SM count           : %d\n", p.multiProcessorCount);
        std::printf("  Max threads / block: %d\n", p.maxThreadsPerBlock);
        std::printf("  Max threads / SM   : %d  (= %d warps)\n",
                    p.maxThreadsPerMultiProcessor,
                    p.maxThreadsPerMultiProcessor / p.warpSize);
        std::printf("  Warp size          : %d\n", p.warpSize);
        std::printf("  Registers / SM     : %d (32-bit)\n", p.regsPerMultiprocessor);
        std::printf("  Shared mem / block : %zu B\n", p.sharedMemPerBlock);
        std::printf("  Shared mem / SM    : %zu B\n", p.sharedMemPerMultiprocessor);
        std::printf("  Global memory      : %.2f GiB\n",
                    p.totalGlobalMem / double(1ull << 30));
        std::printf("  L2 cache           : %d KiB\n", p.l2CacheSize / 1024);
        std::printf("  Memory clock       : %.1f GHz\n", p.memoryClockRate / 1.0e6);
        std::printf("  Memory bus width   : %d bit\n", p.memoryBusWidth);
        // Peak memory bandwidth = 2 * clock(Hz) * bus(bytes)  (DDR -> ×2)
        double bw = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8.0) / 1e9;
        std::printf("  Peak bandwidth     : %.1f GB/s\n", bw);
        std::printf("  Clock rate         : %.2f GHz\n", p.clockRate / 1.0e6);
        std::printf("  ECC enabled        : %s\n", p.ECCEnabled ? "yes" : "no");
        std::printf("  Unified addressing : %s\n", p.unifiedAddressing ? "yes" : "no");
        std::printf("  Concurrent kernels : %s\n", p.concurrentKernels ? "yes" : "no");
        std::printf("\n");
    }
    return 0;
}
