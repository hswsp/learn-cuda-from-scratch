// =============================================================================
// softmax.cu — numerically stable softmax along the last dim.
//
// 三阶段:
//   1) max = reduce_max(x)          ← block-wide reduce
//   2) sum = sum(exp(x - max))      ← block-wide reduce
//   3) y = exp(x - max) / sum       ← element-wise
//
// 关键: "max-subtract" 防止 exp 上溢. 经典 LLM bug 第一名.
//
// 假设: 每 row 一个 block, cols <= 1024.
//
// 对应 HTML: docs/ch10-softmax/index.html#stable
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cfloat>

__inline__ __device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void softmax_row(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float*       yr = y + row * cols;

    // 1) max
    float local_max = -FLT_MAX;
    for (int c = tid; c < cols; c += BLOCK) local_max = fmaxf(local_max, xr[c]);
    local_max = warp_max(local_max);
    __shared__ float warp_max_s[BLOCK / 32];
    if ((tid & 31) == 0) warp_max_s[tid >> 5] = local_max;
    __syncthreads();
    if (tid < BLOCK / 32) {
        float v = warp_max_s[tid];
        v = warp_max(v);
        if (tid == 0) warp_max_s[0] = v;
    }
    __syncthreads();
    float row_max = warp_max_s[0];

    // 2) sum
    float local_sum = 0.f;
    for (int c = tid; c < cols; c += BLOCK) local_sum += __expf(xr[c] - row_max);
    local_sum = warp_sum(local_sum);
    __shared__ float warp_sum_s[BLOCK / 32];
    if ((tid & 31) == 0) warp_sum_s[tid >> 5] = local_sum;
    __syncthreads();
    if (tid < BLOCK / 32) {
        float v = warp_sum_s[tid];
        v = warp_sum(v);
        if (tid == 0) warp_sum_s[0] = v;
    }
    __syncthreads();
    float inv = 1.f / warp_sum_s[0];

    // 3) write y
    for (int c = tid; c < cols; c += BLOCK) yr[c] = __expf(xr[c] - row_max) * inv;
}

int main(int argc, char** argv) {
    int rows = arg_int(argc, argv, "rows", 256);
    int cols = arg_int(argc, argv, "cols", 1024);
    auto hx = make_random(rows * cols, 1, 5.0f);  // 大范围, 测稳定性
    std::vector<float> hy(rows * cols), href(rows * cols);

    DeviceBuffer<float> dx(rows * cols), dy(rows * cols);
    dx.copy_from_host(hx.data());

    constexpr int BLOCK = 256;
    GpuTimer t; t.start();
    softmax_row<BLOCK><<<rows, BLOCK>>>(dx.ptr, dy.ptr, rows, cols);
    t.stop(); KERNEL_CHECK();
    dy.copy_to_host(hy.data());

    cpu_ref::softmax_lastdim(hx.data(), href.data(), rows, cols);
    report("softmax", allclose(hy, href, 1e-5f, 1e-5f));
    std::printf("rows=%d cols=%d  %.3f ms  %.1f GB/s\n",
                rows, cols, t.ms(),
                2.0 * rows * cols * sizeof(float) / (t.ms() * 1e6));
    return 0;
}
