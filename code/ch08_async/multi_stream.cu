// =============================================================================
// multi_stream.cu — overlap H2D, kernel, D2H using N streams + pinned mem.
//
// 学习目标:
//   1. 同一份工作切成 nStreams 个 chunk, 分到 nStreams 个 stream
//   2. 用 cudaMemcpyAsync + pinned host memory 实现真正异步
//   3. 看到 wall time 显著缩短（接近 1/3, 因为 H2D/计算/D2H 三阶段重叠）
//
// 对应 HTML: docs/ch08-async/index.html#streams
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__global__ void heavy_kernel(float* x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    for (int it = 0; it < iters; ++it) v = v * 1.000001f + 1e-7f;
    x[i] = v;
}

int main() {
    const int N = 1 << 22;            // 4M floats
    const int chunks = 4;
    const int chunk = N / chunks;
    size_t bytes = N * sizeof(float);

    // pinned host buffer (required for async memcpy)
    float* hbuf;
    CUDA_CHECK(cudaMallocHost(&hbuf, bytes));
    for (int i = 0; i < N; ++i) hbuf[i] = float(i);

    float* dbuf; CUDA_CHECK(cudaMalloc(&dbuf, bytes));

    cudaStream_t streams[chunks];
    for (int i = 0; i < chunks; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int block = 256, grid = (chunk + block - 1) / block;
    int iters = 1000;
    GpuTimer t;

    // --- baseline: single default stream, fully serialized ---
    t.start();
    CUDA_CHECK(cudaMemcpy(dbuf, hbuf, bytes, cudaMemcpyHostToDevice));
    heavy_kernel<<<grid * chunks, block>>>(dbuf, N, iters);
    CUDA_CHECK(cudaMemcpy(hbuf, dbuf, bytes, cudaMemcpyDeviceToHost));
    t.stop(); KERNEL_CHECK();
    float ms_serial = t.ms();

    // --- overlapped: each chunk goes through its own stream ---
    t.start();
    for (int s = 0; s < chunks; ++s) {
        int off = s * chunk;
        CUDA_CHECK(cudaMemcpyAsync(dbuf + off, hbuf + off, chunk * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        heavy_kernel<<<grid, block, 0, streams[s]>>>(dbuf + off, chunk, iters);
        CUDA_CHECK(cudaMemcpyAsync(hbuf + off, dbuf + off, chunk * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    for (int s = 0; s < chunks; ++s) CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    t.stop();
    float ms_async = t.ms();

    std::printf("serial    : %.3f ms\n", ms_serial);
    std::printf("%d streams: %.3f ms  (%.2fx)\n", chunks, ms_async, ms_serial / ms_async);

    for (int i = 0; i < chunks; ++i) cudaStreamDestroy(streams[i]);
    cudaFree(dbuf); cudaFreeHost(hbuf);
    return 0;
}
