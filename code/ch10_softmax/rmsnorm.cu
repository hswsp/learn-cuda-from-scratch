// =============================================================================
// rmsnorm.cu — RMSNorm (used by Llama / GPT-NeoX) along the last dim.
//
// 公式:  y = x / sqrt(mean(x^2) + eps) * gamma
//
// 实现要点:
//   - 每 row 一个 block
//   - block-wide reduce 算 mean(x^2)
//   - eps 在 sqrt 内防止除 0
//
// 对应 HTML: docs/ch10-softmax/index.html#rmsnorm
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cmath>

__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void rmsnorm_row(const float* x, const float* gamma, float* y,
                            int rows, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* xr = x + row * cols;
    float*       yr = y + row * cols;

    float local_sq = 0.f;
    for (int c = tid; c < cols; c += BLOCK) {
        float v = xr[c];
        local_sq += v * v;
    }
    local_sq = warp_sum(local_sq);
    __shared__ float sm[BLOCK / 32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) sm[wid] = local_sq;
    __syncthreads();
    if (wid == 0) {
        float v = (tid < BLOCK / 32) ? sm[lane] : 0.f;
        v = warp_sum(v);
        if (tid == 0) sm[0] = v;
    }
    __syncthreads();
    float inv = rsqrtf(sm[0] / cols + eps);
    for (int c = tid; c < cols; c += BLOCK)
        yr[c] = xr[c] * inv * gamma[c];
}

int main() {
    int rows = 1024, cols = 1024;
    auto hx = make_random(rows * cols, 1, 1.f);
    auto hg = make_random(cols, 2, 1.f);
    std::vector<float> hy(rows * cols), href(rows * cols);
    DeviceBuffer<float> dx(rows * cols), dg(cols), dy(rows * cols);
    dx.copy_from_host(hx.data()); dg.copy_from_host(hg.data());

    constexpr int BLOCK = 256;
    GpuTimer t; t.start();
    rmsnorm_row<BLOCK><<<rows, BLOCK>>>(dx.ptr, dg.ptr, dy.ptr, rows, cols, 1e-5f);
    t.stop(); KERNEL_CHECK();
    dy.copy_to_host(hy.data());

    cpu_ref::rmsnorm(hx.data(), hg.data(), href.data(), rows, cols, 1e-5f);
    report("rmsnorm", allclose(hy, href, 1e-4f, 1e-4f));
    std::printf("rows=%d cols=%d  %.3f ms  %.1f GB/s\n",
                rows, cols, t.ms(),
                3.0 * rows * cols * sizeof(float) / (t.ms() * 1e6));
    return 0;
}
