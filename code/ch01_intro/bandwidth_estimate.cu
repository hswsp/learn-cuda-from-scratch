// =============================================================================
// bandwidth_estimate.cu — measure host↔device, device→device copy bandwidth.
//
// 学习目标:
//   1. 理解 H2D / D2H / D2D 各自的带宽量级（差几个数量级）
//   2. 看到 pinned memory 对 H2D 的加速效果
//   3. 给后面"为什么要用 stream 重叠 H2D 与计算"埋个伏笔
//
// 对应 HTML: docs/ch01-intro/index.html#bandwidth
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>
#include <vector>

static double bw_gbs(size_t bytes, float ms) {
    return (double)bytes / (ms * 1e-3) / 1e9;
}

int main(int argc, char** argv) {
    int MB    = arg_int(argc, argv, "MB",    64);   // payload per copy
    int iters = arg_int(argc, argv, "iters",  5);

    size_t bytes = size_t(MB) * (1u << 20);
    std::printf("payload = %d MiB, iters = %d\n\n", MB, iters);

    // ---- 1) pageable host buffer ----
    std::vector<char> h_pageable(bytes);
    void* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    GpuTimer t;
    float ms_h2d = 0, ms_d2h = 0, ms_d2d = 0;
    for (int i = 0; i < iters; ++i) {
        t.start(); CUDA_CHECK(cudaMemcpy(d_buf, h_pageable.data(), bytes, cudaMemcpyHostToDevice));
        t.stop();  ms_h2d += t.ms();
        t.start(); CUDA_CHECK(cudaMemcpy(h_pageable.data(), d_buf, bytes, cudaMemcpyDeviceToHost));
        t.stop();  ms_d2h += t.ms();
    }
    ms_h2d /= iters; ms_d2h /= iters;

    // ---- 2) pinned (page-locked) host buffer ----
    char* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));
    float ms_h2d_p = 0, ms_d2h_p = 0;
    for (int i = 0; i < iters; ++i) {
        t.start(); CUDA_CHECK(cudaMemcpy(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice));
        t.stop();  ms_h2d_p += t.ms();
        t.start(); CUDA_CHECK(cudaMemcpy(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost));
        t.stop();  ms_d2h_p += t.ms();
    }
    ms_h2d_p /= iters; ms_d2h_p /= iters;

    // ---- 3) device → device ----
    void* d_buf2 = nullptr; CUDA_CHECK(cudaMalloc(&d_buf2, bytes));
    for (int i = 0; i < iters; ++i) {
        t.start(); CUDA_CHECK(cudaMemcpy(d_buf2, d_buf, bytes, cudaMemcpyDeviceToDevice));
        t.stop();  ms_d2d += t.ms();
    }
    ms_d2d /= iters;

    std::printf("%-30s %10s %12s\n", "transfer", "ms", "GB/s");
    std::printf("------------------------------------------------------\n");
    std::printf("%-30s %10.3f %12.2f\n", "H2D pageable", ms_h2d, bw_gbs(bytes, ms_h2d));
    std::printf("%-30s %10.3f %12.2f\n", "D2H pageable", ms_d2h, bw_gbs(bytes, ms_d2h));
    std::printf("%-30s %10.3f %12.2f\n", "H2D pinned",   ms_h2d_p, bw_gbs(bytes, ms_h2d_p));
    std::printf("%-30s %10.3f %12.2f\n", "D2H pinned",   ms_d2h_p, bw_gbs(bytes, ms_d2h_p));
    std::printf("%-30s %10.3f %12.2f\n", "D2D",          ms_d2d,   bw_gbs(bytes, ms_d2d));

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_buf2));
    CUDA_CHECK(cudaFreeHost(h_pinned));
    return 0;
}
