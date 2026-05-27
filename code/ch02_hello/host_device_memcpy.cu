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

int main() {
    const int N = 16;
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = float(i);

    // RAII device buffer: 自动 free，不会泄漏
    DeviceBuffer<float> d(N);
    d.copy_from_host(h.data());

    // 启动：1 个 block，N 个线程 (N <= 1024 的简化情形)
    scale_kernel<<<1, N>>>(d.ptr, 3.14f, N);
    KERNEL_CHECK();

    d.copy_to_host(h.data());
    for (int i = 0; i < N; ++i) std::printf("h[%2d] = %.2f\n", i, h[i]);
    return 0;
}
