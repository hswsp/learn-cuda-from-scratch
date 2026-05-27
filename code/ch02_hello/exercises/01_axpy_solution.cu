#include "../../common/cuda_utils.h"
#include "../../common/cpu_ref.h"
#include "../../common/check.h"

__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const int N = 1 << 20;
    auto hx = make_random(N, 1), hy = make_random(N, 2), hy_ref(hy);
    float a = 2.0f;

    DeviceBuffer<float> dx(N), dy(N);
    dx.copy_from_host(hx.data()); dy.copy_from_host(hy.data());

    int block = 256;
    int grid  = (N + block - 1) / block;
    saxpy_kernel<<<grid, block>>>(a, dx.ptr, dy.ptr, N);
    KERNEL_CHECK();

    dy.copy_to_host(hy.data());
    cpu_ref::saxpy(a, hx.data(), hy_ref.data(), N);
    report("saxpy", allclose(hy, hy_ref));
    return 0;
}
