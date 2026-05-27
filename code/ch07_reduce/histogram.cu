// =============================================================================
// histogram.cu — 256-bin histogram via shared-mem atomics + global atomicAdd.
//
// 学习目标:
//   1. 看到全局 atomicAdd 在高冲突下慢
//   2. 用 shared-mem 私有 histogram 缓冲再合并的两阶段做法
//
// 对应 HTML: docs/ch07-reduce/index.html#histogram
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

constexpr int BINS = 256;

__global__ void hist_global(const unsigned char* in, unsigned int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&hist[in[i]], 1u);   // 256 全局热点
}

__global__ void hist_shared(const unsigned char* in, unsigned int* hist, int n) {
    __shared__ unsigned int local_h[BINS];
    int tid = threadIdx.x;
    for (int i = tid; i < BINS; i += blockDim.x) local_h[i] = 0;
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    for (int i = gid; i < n; i += stride) atomicAdd(&local_h[in[i]], 1u);
    __syncthreads();

    for (int i = tid; i < BINS; i += blockDim.x) atomicAdd(&hist[i], local_h[i]);
}

int main() {
    int N = 1 << 24;
    std::vector<unsigned char> h(N);
    uint32_t s = 1;
    for (int i = 0; i < N; ++i) { s = s * 1664525 + 1013904223; h[i] = s & 0xFF; }

    unsigned char* din; CUDA_CHECK(cudaMalloc(&din, N));
    CUDA_CHECK(cudaMemcpy(din, h.data(), N, cudaMemcpyHostToDevice));

    unsigned int* dhist; CUDA_CHECK(cudaMalloc(&dhist, BINS * sizeof(unsigned int)));

    GpuTimer t;
    auto bench = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(dhist, 0, BINS * sizeof(unsigned int)));
        t.start(); launch(); t.stop(); KERNEL_CHECK();
        float ms = t.ms();
        std::vector<unsigned int> hh(BINS);
        CUDA_CHECK(cudaMemcpy(hh.data(), dhist, BINS * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        long total = 0; for (auto v : hh) total += v;
        std::printf("  %-15s %.3f ms  total=%ld (expect %d)\n",
                    name, ms, total, N);
    };

    int block = 256;
    int grid_g = (N + block - 1) / block;
    int grid_s = 1024;   // grid-stride

    bench("global atomic", [&] { hist_global<<<grid_g, block>>>(din, dhist, N); });
    bench("shared local",  [&] { hist_shared<<<grid_s, block>>>(din, dhist, N); });

    CUDA_CHECK(cudaFree(din));
    CUDA_CHECK(cudaFree(dhist));
    return 0;
}
