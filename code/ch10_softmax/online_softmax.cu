// =============================================================================
// online_softmax.cu — single-pass softmax via running (max, sum) merging.
//
// 算法 (FlashAttention v1 的核心):
//   维护当前 (m, l) = (running max, running normalized sum).
//   见到新值 x_new:
//     m_new = max(m, x_new)
//     l_new = l * exp(m - m_new) + exp(x_new - m_new)
//   一次扫描即可同时算出 max 与 sum, 不必两遍.
//
// 这里演示一维, FlashAttention 把它扩展到 tile-wise 二维.
//
// 对应 HTML: docs/ch10-softmax/index.html#online
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cfloat>
#include <cstdio>

__device__ __forceinline__ void merge(float& m, float& l, float xm, float xl) {
    float new_m = fmaxf(m, xm);
    l = l * __expf(m - new_m) + xl * __expf(xm - new_m);
    m = new_m;
}

template <int BLOCK>
__global__ void online_softmax_row(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr       = y + row * cols;

    // 1) 每 thread 局部 (m, l)
    float m = -FLT_MAX, l = 0.f;
    for (int c = tid; c < cols; c += BLOCK) {
        float v = xr[c];
        merge(m, l, v, 1.f);                 // 把 (v, 1) 合并进 (m, l)
    }
    // 2) warp 内 reduce 合并 (m, l)
    for (int o = 16; o > 0; o >>= 1) {
        float om = __shfl_xor_sync(0xffffffff, m, o);
        float ol = __shfl_xor_sync(0xffffffff, l, o);
        merge(m, l, om, ol);
    }
    // 3) 跨 warp 合并
    __shared__ float sm[BLOCK / 32], sl[BLOCK / 32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) { sm[wid] = m; sl[wid] = l; }
    __syncthreads();
    if (wid == 0) {
        m = (tid < BLOCK / 32) ? sm[lane] : -FLT_MAX;
        l = (tid < BLOCK / 32) ? sl[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) {
            float om = __shfl_xor_sync(0xffffffff, m, o);
            float ol = __shfl_xor_sync(0xffffffff, l, o);
            merge(m, l, om, ol);
        }
        if (lane == 0) { sm[0] = m; sl[0] = l; }
    }
    __syncthreads();
    float final_m = sm[0], final_l = sl[0];
    float inv = 1.f / final_l;
    for (int c = tid; c < cols; c += BLOCK)
        yr[c] = __expf(xr[c] - final_m) * inv;
}

int main() {
    int rows = 256, cols = 1024;
    auto hx = make_random(rows * cols, 7, 5.0f);
    std::vector<float> hy(rows * cols), href(rows * cols);
    DeviceBuffer<float> dx(rows * cols), dy(rows * cols);
    dx.copy_from_host(hx.data());

    constexpr int BLOCK = 256;
    GpuTimer t; t.start();
    online_softmax_row<BLOCK><<<rows, BLOCK>>>(dx.ptr, dy.ptr, rows, cols);
    t.stop(); KERNEL_CHECK();
    dy.copy_to_host(hy.data());
    cpu_ref::softmax_lastdim(hx.data(), href.data(), rows, cols);
    report("online_softmax", allclose(hy, href, 1e-5f, 1e-5f));
    std::printf("rows=%d cols=%d  %.3f ms\n", rows, cols, t.ms());
    return 0;
}
