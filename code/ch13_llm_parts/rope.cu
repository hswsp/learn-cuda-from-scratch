// =============================================================================
// rope.cu — Rotary Position Embedding, in-place along the last dim.
//
// 几何意义: 把 D 维向量看成 D/2 个复数, 第 i 个复数乘 exp(j * theta_t,i),
//           theta_t,i = t / base^(2i/D).
// 实际实现: 把 D 维分两半 [x0..x_{D/2-1}], [x_{D/2}..x_{D-1}],
//           pair (x0, x_{D/2}) 旋转角度 theta_t,0.
//
// 形状: x (T, D) — apply per row.  D 必须是偶数.
//
// 对应 HTML: docs/ch13-llm-parts/index.html#rope
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cmath>

__global__ void rope_inplace_kernel(float* x, int T, int D, float base) {
    int t = blockIdx.x;
    int i = threadIdx.x;
    int half = D / 2;
    if (t >= T || i >= half) return;
    float theta = float(t) / powf(base, float(2 * i) / float(D));
    float c = cosf(theta), s = sinf(theta);
    float x0 = x[t * D + i];
    float x1 = x[t * D + i + half];
    x[t * D + i]        = x0 * c - x1 * s;
    x[t * D + i + half] = x0 * s + x1 * c;
}

int main() {
    int T = 512, D = 64;
    auto h = make_random(T * D, 1);
    auto href = h;
    cpu_ref::rope_inplace(href.data(), T, D);

    DeviceBuffer<float> d(T * D);
    d.copy_from_host(h.data());
    rope_inplace_kernel<<<T, D/2>>>(d.ptr, T, D, 10000.0f);
    KERNEL_CHECK();
    d.copy_to_host(h.data());
    report("rope_inplace", allclose(h, href, 1e-4f, 1e-4f));
    return 0;
}
