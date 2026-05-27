// =============================================================================
// constant_demo.cu — using __constant__ memory for read-only broadcast data.
//
// __constant__ 内存特点:
//   - 64 KB 上限（全 GPU 共享）
//   - 走专用 constant cache (8 KB / SM)
//   - 当 warp 内 32 lane 访问同一个地址 → 一次广播, 0 延迟
//
// 学习目标:
//   1. 用 cudaMemcpyToSymbol 上传只读参数
//   2. 看到访问同一 constant 比 global 快很多倍
//
// 对应 HTML: docs/ch05-memory/index.html#constant
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__constant__ float c_coeff[16];

__global__ void use_constant(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    // 所有 lane 访问 c_coeff[0..15] 的同一 lane → 广播命中
    float acc = 0.f;
    for (int k = 0; k < 16; ++k) acc += c_coeff[k] * v;
    y[i] = acc;
}

__global__ void use_global(const float* coeff, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    float acc = 0.f;
    for (int k = 0; k < 16; ++k) acc += coeff[k] * v;  // 走 L2/L1, 命中后也很快
    y[i] = acc;
}

int main() {
    const int N = 1 << 22;
    DeviceBuffer<float> dx(N), dy(N);
    std::vector<float> h(N, 1.0f);
    dx.copy_from_host(h.data());

    float h_coeff[16];
    for (int i = 0; i < 16; ++i) h_coeff[i] = i * 0.5f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeff, h_coeff, sizeof h_coeff));

    DeviceBuffer<float> d_coeff(16);
    d_coeff.copy_from_host(h_coeff);

    int block = 256, grid = (N + block - 1) / block;
    GpuTimer t;

    t.start(); for (int r = 0; r < 10; ++r) use_constant<<<grid, block>>>(dx.ptr, dy.ptr, N);
    t.stop(); KERNEL_CHECK();
    float ms_c = t.ms() / 10;
    t.start(); for (int r = 0; r < 10; ++r) use_global  <<<grid, block>>>(d_coeff.ptr, dx.ptr, dy.ptr, N);
    t.stop(); KERNEL_CHECK();
    float ms_g = t.ms() / 10;

    std::printf("__constant__ : %.3f ms\n", ms_c);
    std::printf("global       : %.3f ms\n", ms_g);
    std::printf("speedup      : %.2fx\n", ms_g / ms_c);
    return 0;
}
