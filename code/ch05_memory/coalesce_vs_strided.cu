// =============================================================================
// coalesce_vs_strided.cu — show the cost of non-coalesced global memory access.
//
// 学习目标:
//   1. 看清 "warp 内 32 lane 访问连续 128B" 与 "stride 访问" 的带宽差距
//   2. 用同一个 buffer，只改访问模式，看吞吐量从几百 GB/s 跌到几十
//
// 对应 HTML: docs/ch05-memory/index.html#coalesce
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>
#include <vector>

// Pattern A: lane k of warp w 访问 x[base + k]  → 32 lane 访问 128B 连续 → coalesced.
__global__ void copy_coalesced(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

// Pattern B: 同一 warp 32 lane 访问相距 STRIDE 个 float → 32 次独立 32B 事务.
template <int STRIDE>
__global__ void copy_strided(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i * STRIDE;
    if (idx < n) out[idx] = in[idx];
}

int main(int argc, char** argv) {
    const int N = 1 << 24;                                   // 16M floats = 64 MiB
    DeviceBuffer<float> din(N), dout(N);
    std::vector<float> h(N, 1.0f);
    din.copy_from_host(h.data());

    int block = 256;
    GpuTimer t;

    auto bench = [&](const char* name, auto launch) {
        // warm up
        launch(); KERNEL_CHECK();
        t.start();
        for (int r = 0; r < 5; ++r) launch();
        t.stop(); KERNEL_CHECK();
        float ms = t.ms() / 5.0f;
        // bytes moved: 2 * (effective N) * sizeof(float)
        std::printf("  %-25s  %7.3f ms  %7.1f GB/s\n",
                    name, ms, 2.0 * N * sizeof(float) / (ms * 1e6));
    };

    std::printf("N = %d (64 MiB)\n", N);
    std::printf("Pattern                      ms        BW\n");
    std::printf("------------------------------------------\n");

    bench("coalesced (stride=1)", [&] {
        copy_coalesced<<<(N+block-1)/block, block>>>(din.ptr, dout.ptr, N);
    });
    bench("strided 2", [&] {
        copy_strided<2><<<(N/2+block-1)/block, block>>>(din.ptr, dout.ptr, N);
    });
    bench("strided 4", [&] {
        copy_strided<4><<<(N/4+block-1)/block, block>>>(din.ptr, dout.ptr, N);
    });
    bench("strided 8", [&] {
        copy_strided<8><<<(N/8+block-1)/block, block>>>(din.ptr, dout.ptr, N);
    });
    bench("strided 32", [&] {
        copy_strided<32><<<(N/32+block-1)/block, block>>>(din.ptr, dout.ptr, N);
    });
    return 0;
}
