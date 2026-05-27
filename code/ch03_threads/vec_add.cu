// =============================================================================
// vec_add.cu — canonical 1D vector addition with proper bounds check.
//
// 学习目标:
//   1. 标准 1D 索引: i = blockIdx.x * blockDim.x + threadIdx.x
//   2. 总线程数 < N 时怎么办：grid-stride loop
//   3. 与 cpu_ref 对拍验证正确性
//
// 对应 HTML: docs/ch03-threads/index.html#vec-add
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

__global__ void vec_add_v1(const float* a, const float* b, float* c, int n) {
    // 一线程一元素（要保证 grid * block >= N）
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vec_add_grid_stride(const float* a, const float* b, float* c, int n) {
    // grid-stride loop: 每个线程处理多个元素，
    // 即使 grid * block < N 也能跑完，且对大数组更友好。
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride)
        c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    int N     = arg_int(argc, argv, "N", 1 << 22);   // 4M elements
    int block = arg_int(argc, argv, "block", 256);

    auto ha = make_random(N, 1);
    auto hb = make_random(N, 2);
    std::vector<float> hc(N);

    DeviceBuffer<float> da(N), db(N), dc(N);
    da.copy_from_host(ha.data());
    db.copy_from_host(hb.data());

    GpuTimer t;

    // --- v1: 一线程一元素 ---
    int grid = (N + block - 1) / block;
    t.start();
    vec_add_v1<<<grid, block>>>(da.ptr, db.ptr, dc.ptr, N);
    t.stop(); KERNEL_CHECK();
    float ms_v1 = t.ms();
    dc.copy_to_host(hc.data());

    std::vector<float> hc_ref(N);
    cpu_ref::vec_add(ha.data(), hb.data(), hc_ref.data(), N);
    report("vec_add_v1", allclose(hc, hc_ref));

    // --- v2: grid-stride loop, only 1024 blocks ---
    int grid2 = 1024;
    t.start();
    vec_add_grid_stride<<<grid2, block>>>(da.ptr, db.ptr, dc.ptr, N);
    t.stop(); KERNEL_CHECK();
    float ms_v2 = t.ms();
    dc.copy_to_host(hc.data());
    report("vec_add_grid_stride", allclose(hc, hc_ref));

    // Bandwidth: 3 arrays * 4 bytes per element = 12N bytes moved
    double gb = 3.0 * N * sizeof(float) / 1e9;
    std::printf("\nN = %d (%.2f MiB each array), block = %d\n",
                N, N * sizeof(float) / (1024.0 * 1024.0), block);
    std::printf("v1 one-per-thread    : %.3f ms, %.1f GB/s\n", ms_v1, gb / (ms_v1 * 1e-3));
    std::printf("v2 grid-stride loop  : %.3f ms, %.1f GB/s\n", ms_v2, gb / (ms_v2 * 1e-3));
    return 0;
}
