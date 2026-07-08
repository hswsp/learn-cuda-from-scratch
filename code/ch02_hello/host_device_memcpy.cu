// =============================================================================
// host_device_memcpy.cu — minimal H2D / kernel / D2H roundtrip.
//
// 学习目标:
//   1. cudaMalloc / cudaMemcpy / cudaFree 三件套
//   2. 在 kernel 内用 threadIdx.x 算每个线程负责的元素
//   3. 用 RAII DeviceBuffer<T> 替代裸 malloc，少出 bug
//
// 对应 HTML: docs/ch02-hello/index.html#memcpy
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>
#include <vector>

__global__ void scale_kernel(float* x, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= k;
}

__global__ void fma_kernel(const float* x, const float* y, float* z, float bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] * y[i] + bias;
}

int main() {
    const int N = 16;
    std::vector<float> h_scale(N);
    for (int i = 0; i < N; ++i) h_scale[i] = float(i);

    // RAII device buffer: 自动 free，不会泄漏
    DeviceBuffer<float> d(N);
    d.copy_from_host(h_scale.data());
    // 启动：1 个 block，N 个线程 (N <= 1024 的简化情形)
    scale_kernel<<<1, N>>>(d.ptr, 3.14f, N);
    KERNEL_CHECK();

    d.copy_to_host(h_scale.data());

    std::printf("--- scale: x *= k ---\n");
    for (int i = 0; i < N; ++i) std::printf("h[%2d] = %.2f\n", i, h_scale[i]);

    // fma: z = x * y + bias
    std::vector<float> h_x(N), h_y(N), h_z(N);
    for (int i = 0; i < N; ++i) { h_x[i] = float(i); h_y[i] = float(i + 1); }

    DeviceBuffer<float> d_x(N), d_y(N), d_z(N);
    d_x.copy_from_host(h_x.data());
    d_y.copy_from_host(h_y.data());

    fma_kernel<<<1, N>>>(d_x.ptr, d_y.ptr, d_z.ptr, 1.0f, N);
    KERNEL_CHECK();

    d_z.copy_to_host(h_z.data());
    std::printf("\n--- fma: z = x * y + bias ---\n");
    for (int i = 0; i < N; ++i)
        std::printf("z[%2d] = %.0f * %.0f + 1 = %.0f\n", i, h_x[i], h_y[i], h_z[i]);
    return 0;
}
