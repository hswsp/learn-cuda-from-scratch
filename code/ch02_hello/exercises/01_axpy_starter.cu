// 练习 1: 实现 SAXPY (Single-precision A·X Plus Y)
//   y[i] = a * x[i] + y[i],  i = 0..N-1
//
// 要求:
//   - kernel 处理任意 N（不要假设 N <= blockDim.x）
//   - 用 1D grid + 1D block，block 大小自选（建议 256）
//   - 调用结束后 host 端打印前 8 个元素与最后 1 个元素
//
// 验证: 把你的输出与 cpu_ref::saxpy 比较，allclose 应通过。

#include "../../common/cuda_utils.h"
#include "../../common/cpu_ref.h"
#include "../../common/check.h"

__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
    // TODO: write me
}

int main() {
    const int N = 1 << 20;
    auto hx = make_random(N, 1), hy = make_random(N, 2), hy_ref(hy);
    float a = 2.0f;

    DeviceBuffer<float> dx(N), dy(N);
    dx.copy_from_host(hx.data()); dy.copy_from_host(hy.data());

    // TODO: 启动 saxpy_kernel
    KERNEL_CHECK();

    dy.copy_to_host(hy.data());
    cpu_ref::saxpy(a, hx.data(), hy_ref.data(), N);
    report("saxpy", allclose(hy, hy_ref));
    return 0;
}
