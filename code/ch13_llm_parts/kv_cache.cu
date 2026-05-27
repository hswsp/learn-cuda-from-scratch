// =============================================================================
// kv_cache.cu — append new (K, V) row into a pre-allocated cache, decode style.
//
// 推理 decode 阶段每步只生成 1 个 token. 新算出的 K_new, V_new 形状 (1, D)
// 要写入 cache 的 cache[layer, head, t, d] 位置. 这里假设 single layer / head 简化.
//
// kernel 实际只是一个 indexed copy + 拼接.
//
// 对应 HTML: docs/ch13-llm-parts/index.html#kv-cache
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__global__ void append_kv(const float* K_new, const float* V_new,
                          float* K_cache, float* V_cache,
                          int t_pos, int D) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    K_cache[t_pos * D + d] = K_new[d];
    V_cache[t_pos * D + d] = V_new[d];
}

int main() {
    int T_max = 1024, D = 128;
    DeviceBuffer<float> K_cache(T_max * D), V_cache(T_max * D);
    DeviceBuffer<float> K_new(D), V_new(D);
    std::vector<float> hk(D, 0), hv(D, 0);

    for (int step = 0; step < 8; ++step) {
        for (int i = 0; i < D; ++i) { hk[i] = step + 0.1f * i; hv[i] = -step - 0.1f * i; }
        K_new.copy_from_host(hk.data()); V_new.copy_from_host(hv.data());
        append_kv<<<(D + 31) / 32, 32>>>(K_new.ptr, V_new.ptr,
                                          K_cache.ptr, V_cache.ptr, step, D);
        KERNEL_CHECK();
    }
    // verify slot 7
    std::vector<float> probe(D);
    CUDA_CHECK(cudaMemcpy(probe.data(), K_cache.ptr + 7 * D,
                          D * sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("K_cache[7, 0..3] = %.2f %.2f %.2f %.2f (expect 7.00 7.10 7.20 7.30)\n",
                probe[0], probe[1], probe[2], probe[3]);
    return 0;
}
