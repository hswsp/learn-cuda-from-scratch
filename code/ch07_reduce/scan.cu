// =============================================================================
// scan.cu — block-level inclusive prefix sum using warp shuffle.
//
// 算法:
//   每 warp 内用 __shfl_up_sync 做 5 步 Hillis-Steele.
//   每 warp 末尾值写 shared, 0 号 warp 再扫一次, 把偏移加回各 warp.
//   block 内输出 inclusive scan.  跨 block 拼接见练习.
//
// 用途: 后面 sampling 里 top-p (nucleus) 需要 sorted cumulative prob.
//
// 对应 HTML: docs/ch07-reduce/index.html#scan
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

constexpr int BLOCK = 256;

__device__ __forceinline__ float warp_inclusive_scan(float v) {
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, o);
        if ((threadIdx.x & 31) >= o) v += t;
    }
    return v;
}

__global__ void scan_block(const float* in, float* out, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < n) ? in[gid] : 0;
    v = warp_inclusive_scan(v);

    __shared__ float warp_tails[32];
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 31) warp_tails[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float t = (tid < BLOCK / 32) ? warp_tails[lane] : 0;
        t = warp_inclusive_scan(t);
        if (tid < BLOCK / 32) warp_tails[tid] = t;
    }
    __syncthreads();

    if (wid > 0) v += warp_tails[wid - 1];
    if (gid < n) out[gid] = v;
}

int main() {
    const int N = BLOCK;
    std::vector<float> hin(N), hout(N);
    for (int i = 0; i < N; ++i) hin[i] = 1.0f;
    DeviceBuffer<float> din(N), dout(N);
    din.copy_from_host(hin.data());
    scan_block<<<1, BLOCK>>>(din.ptr, dout.ptr, N);
    KERNEL_CHECK();
    dout.copy_to_host(hout.data());
    // expect inclusive scan of all-ones: 1, 2, 3, ..., N
    std::printf("hout[0..7] = %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                hout[0], hout[1], hout[2], hout[3], hout[4], hout[5], hout[6], hout[7]);
    std::printf("hout[255]  = %.0f  (expect %d)\n", hout[N - 1], N);
    return 0;
}
