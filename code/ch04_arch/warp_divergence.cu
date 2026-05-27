// =============================================================================
// warp_divergence.cu — measure cost of intra-warp branching.
//
// 学习目标:
//   1. 看到 if/else 在 warp 内串行化执行的代价
//   2. 区分 "warp-uniform" 分支（每 warp 走同一边）与 "warp-divergent" 分支
//   3. 量化：divergent vs uniform 的时间差
//
// 对应 HTML: docs/ch04-arch/index.html#divergence
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

// uniform: 用 warp id 做分支 → 同一 warp 内 32 个 lane 都走同一支
__global__ void uniform_branch(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = out[i];
    int warp_id = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
    for (int it = 0; it < iters; ++it) {
        if (warp_id & 1) x = x * 1.000001f + 1e-6f;
        else             x = x * 0.999999f - 1e-6f;
    }
    out[i] = x;
}

// divergent: 用 lane id 做分支 → 同一 warp 内 16 lane 走 if、16 lane 走 else
__global__ void divergent_branch(float* out, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = out[i];
    for (int it = 0; it < iters; ++it) {
        if (threadIdx.x & 1) x = x * 1.000001f + 1e-6f;
        else                 x = x * 0.999999f - 1e-6f;
    }
    out[i] = x;
}

int main() {
    const int N = 1 << 20;
    const int iters = 4096;
    DeviceBuffer<float> d(N);
    std::vector<float> h(N, 1.0f);
    d.copy_from_host(h.data());

    int block = 256, grid = (N + block - 1) / block;
    GpuTimer t;

    t.start(); uniform_branch  <<<grid, block>>>(d.ptr, N, iters); t.stop(); KERNEL_CHECK();
    float ms_u = t.ms();
    t.start(); divergent_branch<<<grid, block>>>(d.ptr, N, iters); t.stop(); KERNEL_CHECK();
    float ms_d = t.ms();

    std::printf("uniform   (warp-aligned branch) : %.3f ms\n", ms_u);
    std::printf("divergent (lane-aligned branch) : %.3f ms\n", ms_d);
    std::printf("slowdown                        : %.2fx\n", ms_d / ms_u);
    return 0;
}
