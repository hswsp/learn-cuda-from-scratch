// =============================================================================
// swiglu.cu — SwiGLU activation (Llama-style FFN).
//
// FFN(x) = (silu(x @ W_gate)) ⊙ (x @ W_up)  @ W_down
// 本 kernel: 给定 G = x @ W_gate, U = x @ W_up (两者 (T, hidden)),
//            算 out = silu(G) ⊙ U.  典型 hidden = 4 * D 或 2.66 * D (Llama).
//
// 对应 HTML: docs/ch13-llm-parts/index.html#swiglu
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

__device__ __forceinline__ float silu(float x) {
    return x * (1.f / (1.f + __expf(-x)));
}

__global__ void swiglu_kernel(const float* G, const float* U, float* O, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) O[i] = silu(G[i]) * U[i];
}

int main() {
    int T = 1024, H = 4096;
    int n = T * H;
    auto hG = make_random(n, 1, 0.5f);
    auto hU = make_random(n, 2, 0.5f);
    std::vector<float> hO(n), href(n);
    for (int i = 0; i < n; ++i) href[i] = cpu_ref::silu(hG[i]) * hU[i];

    DeviceBuffer<float> dG(n), dU(n), dO(n);
    dG.copy_from_host(hG.data()); dU.copy_from_host(hU.data());

    int block = 256, grid = (n + block - 1) / block;
    GpuTimer t; t.start();
    swiglu_kernel<<<grid, block>>>(dG.ptr, dU.ptr, dO.ptr, n);
    t.stop(); KERNEL_CHECK();
    dO.copy_to_host(hO.data());
    report("swiglu", allclose(hO, href, 1e-4f, 1e-4f));
    std::printf("T=%d H=%d  %.3f ms\n", T, H, t.ms());
    return 0;
}
