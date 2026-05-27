// =============================================================================
// reduce_v1_to_v5.cu — Mark Harris's classic 5-stage sum reduction.
//
// 学习目标:
//   1. 看到同一个算法被反复优化, 时间从 ~5 ms 降到 ~0.6 ms
//   2. 涉及的优化点:
//        v1: 模 2 分歧, divergent
//        v2: stride 一半折叠, 仍有 bank conflict
//        v3: 启动时减半, 每 thread 加两元素
//        v4: 展开最后一个 warp (no syncthreads)
//        v5: warp shuffle, 完全不走 shared mem 的最后一级
//
// 输出: 5 个版本的 ms 与 GB/s.
//
// 对应 HTML: docs/ch07-reduce/index.html#reduce
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cstdio>

constexpr int BLOCK = 256;

__global__ void reduce_v1(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();
    // bad: divergent within warp
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_v2(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0;
    __syncthreads();
    // sequential addressing: keep warps active until s halves
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_v3(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    float v = 0.f;
    if (i < n)             v += in[i];
    if (i + BLOCK < n)     v += in[i + BLOCK];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__device__ __forceinline__ void warp_reduce(volatile float* s, int tid) {
    s[tid] += s[tid + 32]; s[tid] += s[tid + 16];
    s[tid] += s[tid +  8]; s[tid] += s[tid +  4];
    s[tid] += s[tid +  2]; s[tid] += s[tid +  1];
}

__global__ void reduce_v4(const float* in, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    float v = 0.f;
    if (i < n)         v += in[i];
    if (i + BLOCK < n) v += in[i + BLOCK];
    sdata[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) warp_reduce(sdata, tid);
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_v5(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;
    float v = 0.f;
    if (i < n)         v += in[i];
    if (i + BLOCK < n) v += in[i + BLOCK];
    __shared__ float warp_sums[32];
    // warp-level reduce via shuffle
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (tid < blockDim.x / 32) ? warp_sums[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) out[blockIdx.x] = v;
    }
}

float host_finalize(float* d_part, int g) {
    std::vector<float> h(g);
    CUDA_CHECK(cudaMemcpy(h.data(), d_part, g * sizeof(float), cudaMemcpyDeviceToHost));
    double s = 0; for (float v : h) s += v;
    return float(s);
}

int main() {
    const int N = 1 << 24;          // 16M elements = 64 MiB
    auto h = make_random(N, 42, 1e-3f);
    DeviceBuffer<float> din(N);
    din.copy_from_host(h.data());
    float ref = cpu_ref::sum(h.data(), N);

    int block = BLOCK;
    int g_v1 = (N + block - 1) / block;
    int g_v3 = (N + 2 * block - 1) / (2 * block);
    DeviceBuffer<float> dpart(g_v1);

    GpuTimer t;
    auto run = [&](const char* name, auto launch, int g) {
        launch(); KERNEL_CHECK();
        t.start(); for (int r = 0; r < 5; ++r) launch();
        t.stop(); KERNEL_CHECK();
        float ms = t.ms() / 5;
        float sum = host_finalize(dpart.ptr, g);
        double gb = N * sizeof(float) / 1e9;
        std::printf("  %-10s %.3f ms  %6.1f GB/s   err=%.3e\n",
                    name, ms, gb / (ms * 1e-3), std::fabs(sum - ref));
    };
    run("v1", [&]{ reduce_v1<<<g_v1, block>>>(din.ptr, dpart.ptr, N); }, g_v1);
    run("v2", [&]{ reduce_v2<<<g_v1, block>>>(din.ptr, dpart.ptr, N); }, g_v1);
    run("v3", [&]{ reduce_v3<<<g_v3, block>>>(din.ptr, dpart.ptr, N); }, g_v3);
    run("v4", [&]{ reduce_v4<<<g_v3, block>>>(din.ptr, dpart.ptr, N); }, g_v3);
    run("v5", [&]{ reduce_v5<<<g_v3, block>>>(din.ptr, dpart.ptr, N); }, g_v3);
    return 0;
}
