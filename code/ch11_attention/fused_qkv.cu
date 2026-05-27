// =============================================================================
// fused_qkv.cu — single GEMM produces Q, K, V from input X.
//
// 思想:
//   单层 Transformer 有 3 个 projection: W_q, W_k, W_v, 各 (D, D).
//   horizontally concat 成 W_qkv: (D, 3*D), 一次 GEMM 算出 (T, 3*D).
//   随后把它视为 (T, 3, D) 拆出 Q, K, V.
//
// 工业实现 (TRT-LLM, vLLM) 都这么做, 启动 1 次 GEMM 比 3 次省 launch + 更高 GPU 利用率.
//
// 这里用 Ch6 tiled matmul 模板复用, 重点是 layout reshape.
//
// 对应 HTML: docs/ch11-attention/index.html#fused-qkv
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

constexpr int BM = 32, BN = 32, BK = 32;
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    __shared__ float As[BM][BK], Bs[BK][BN];
    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = by * BM + ty, col = bx * BN + tx;
    float acc = 0.f;
    for (int kt = 0; kt < K; kt += BK) {
        As[ty][tx] = (row < M && kt + tx < K) ? A[row * K + kt + tx] : 0.f;
        Bs[ty][tx] = (kt + ty < K && col < N) ? B[(kt + ty) * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// split (T, 3*D) → 3 个 (T, D)
__global__ void split_qkv(const float* qkv, float* q, float* k, float* v, int T, int D) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || d >= D) return;
    int base = t * (3 * D);
    q[t * D + d] = qkv[base + 0 * D + d];
    k[t * D + d] = qkv[base + 1 * D + d];
    v[t * D + d] = qkv[base + 2 * D + d];
}

int main() {
    int T = 128, D = 256;
    auto hX  = make_random(T * D, 1);
    auto hW  = make_random(D * 3 * D, 2);  // [D, 3*D]
    std::vector<float> hQ(T*D), hK(T*D), hV(T*D);
    std::vector<float> hRef_qkv(T * 3 * D);

    DeviceBuffer<float> dX(T*D), dW(D*3*D), dQKV(T*3*D), dQ(T*D), dK(T*D), dV(T*D);
    dX.copy_from_host(hX.data()); dW.copy_from_host(hW.data());

    dim3 block(BN, BM);
    dim3 grid((3*D + BN - 1)/BN, (T + BM - 1)/BM);
    gemm_tiled<<<grid, block>>>(dX.ptr, dW.ptr, dQKV.ptr, T, 3*D, D);
    KERNEL_CHECK();

    dim3 sb(16, 16), sg((D+15)/16, (T+15)/16);
    split_qkv<<<sg, sb>>>(dQKV.ptr, dQ.ptr, dK.ptr, dV.ptr, T, D);
    KERNEL_CHECK();

    dQ.copy_to_host(hQ.data());
    cpu_ref::gemm(hX.data(), hW.data(), hRef_qkv.data(), T, 3*D, D);
    std::vector<float> hQ_ref(T*D);
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d) hQ_ref[t*D + d] = hRef_qkv[t*3*D + 0*D + d];
    report("fused_qkv (Q slice)", allclose(hQ, hQ_ref, 1e-3f, 1e-3f));

    std::printf("Fused QKV done. 1 GEMM 取代 3 GEMM, 同样的总 FLOPs.\n");
    return 0;
}
