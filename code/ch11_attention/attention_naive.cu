// =============================================================================
// attention_naive.cu — straightforward 3-pass single-head attention.
//
// 算法:
//   S = Q @ K^T * (1/sqrt(D))    [T, T]
//   P = softmax_row(S)           [T, T]
//   O = P @ V                    [T, D]
//
// 全程把中间矩阵 S, P 都物化在 HBM (O(T^2) 内存). 教学用; 性能差.
// 单 head, fp32, causal mask 可选.
//
// 对应 HTML: docs/ch11-attention/index.html#naive
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cfloat>

// ---- S = Q @ K^T * scale (+ causal mask) ----
__global__ void qkt_scale(const float* Q, const float* K, float* S,
                          int T, int D, float scale, bool causal) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // row in S, query index
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // col in S, key index
    if (i >= T || j >= T) return;
    if (causal && j > i) { S[i * T + j] = -1e30f; return; }
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += Q[i * D + d] * K[j * D + d];
    S[i * T + j] = s * scale;
}

// ---- row-wise stable softmax (reuse Ch10) ----
__inline__ __device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

template <int BLOCK>
__global__ void softmax_rows(float* X, int rows, int cols) {
    int row = blockIdx.x; int tid = threadIdx.x;
    float* xr = X + row * cols;
    float m = -FLT_MAX;
    for (int c = tid; c < cols; c += BLOCK) m = fmaxf(m, xr[c]);
    m = warp_max(m);
    __shared__ float sm[BLOCK / 32];
    if ((tid & 31) == 0) sm[tid >> 5] = m;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < BLOCK / 32) ? sm[tid] : -FLT_MAX;
        v = warp_max(v);
        if (tid == 0) sm[0] = v;
    }
    __syncthreads();
    float row_max = sm[0];

    float s = 0.f;
    for (int c = tid; c < cols; c += BLOCK) s += __expf(xr[c] - row_max);
    s = warp_sum(s);
    __shared__ float ss[BLOCK / 32];
    if ((tid & 31) == 0) ss[tid >> 5] = s;
    __syncthreads();
    if ((tid >> 5) == 0) {
        float v = (tid < BLOCK / 32) ? ss[tid] : 0;
        v = warp_sum(v);
        if (tid == 0) ss[0] = v;
    }
    __syncthreads();
    float inv = 1.f / ss[0];
    for (int c = tid; c < cols; c += BLOCK) xr[c] = __expf(xr[c] - row_max) * inv;
}

// ---- O = P @ V ----
__global__ void pv_kernel(const float* P, const float* V, float* O, int T, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T || d >= D) return;
    float acc = 0.f;
    for (int j = 0; j < T; ++j) acc += P[i * T + j] * V[j * D + d];
    O[i * D + d] = acc;
}

int main(int argc, char** argv) {
    int T = arg_int(argc, argv, "T", 256);
    int D = arg_int(argc, argv, "D", 64);
    bool causal = true;
    auto hQ = make_random(T * D, 1);
    auto hK = make_random(T * D, 2);
    auto hV = make_random(T * D, 3);
    std::vector<float> hO(T * D), href(T * D);

    DeviceBuffer<float> dQ(T*D), dK(T*D), dV(T*D), dO(T*D), dS(T*T);
    dQ.copy_from_host(hQ.data()); dK.copy_from_host(hK.data()); dV.copy_from_host(hV.data());

    float scale = 1.f / std::sqrt(float(D));

    dim3 block_s(16, 16), grid_s((T+15)/16, (T+15)/16);
    dim3 block_o(16, 16), grid_o((D+15)/16, (T+15)/16);

    GpuTimer t; t.start();
    qkt_scale<<<grid_s, block_s>>>(dQ.ptr, dK.ptr, dS.ptr, T, D, scale, causal);
    softmax_rows<256><<<T, 256>>>(dS.ptr, T, T);
    pv_kernel<<<grid_o, block_o>>>(dS.ptr, dV.ptr, dO.ptr, T, D);
    t.stop(); KERNEL_CHECK();

    dO.copy_to_host(hO.data());
    cpu_ref::attention(hQ.data(), hK.data(), hV.data(), href.data(), T, D, causal);
    report("attention_naive", allclose(hO, href, 1e-4f, 1e-4f));
    std::printf("T=%d D=%d  %.3f ms  S占显存=%.1f MiB\n",
                T, D, t.ms(), T * T * 4 / (1024.0 * 1024.0));
    return 0;
}
