// =============================================================================
// error_handling.cu — demonstrate CUDA_CHECK / KERNEL_CHECK catching bugs.
//
// 学习目标:
//   1. 看清"没装错误检查"会发生什么：程序"成功"退出但结果错
//   2. 学会三种错误来源:
//        - cudaMalloc 失败（CUDA_CHECK 抓）
//        - 配置错误（grid/block 超限，cudaPeekAtLastError 抓）
//        - in-kernel 非法访问（cudaDeviceSynchronize 抓）
//
// 运行: 默认演示三种错误；加 --safe 跑无错版本
//
// 对应 HTML: docs/ch02-hello/index.html#errors
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__global__ void bad_kernel(int* p) {
    // 故意越界写: p 只分配了 4 字节 (1 int)，但访问 1000000 号位置
    p[1000000] = 42;
}

int main(int argc, char** argv) {
    bool safe = arg_flag(argc, argv, "safe");

    if (safe) {
        std::printf("[safe mode] simply allocate, copy, free.\n");
        int* d; CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
        int v = 7;
        CUDA_CHECK(cudaMemcpy(d, &v, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&v, d, sizeof(int), cudaMemcpyDeviceToHost));
        std::printf("read back: %d\n", v);
        CUDA_CHECK(cudaFree(d));
        return 0;
    }

    // -------- 错误 1: malloc 巨大显存触发 OOM --------
    std::printf("[bug 1] try to allocate 1 PiB:\n");
    void* huge = nullptr;
    cudaError_t e = cudaMalloc(&huge, size_t(1) << 50);  // 1 PiB
    if (e != cudaSuccess) {
        std::printf("  ✓ caught: %s\n\n", cudaGetErrorString(e));
        // 必须 reset 否则 sticky error 会污染后续调用
        cudaGetLastError();
    }

    // -------- 错误 2: bad launch config --------
    std::printf("[bug 2] block dim too large (10000 threads):\n");
    int* p; CUDA_CHECK(cudaMalloc(&p, sizeof(int)));
    bad_kernel<<<1, 10000>>>(p);                  // > 1024 max threads/block
    e = cudaPeekAtLastError();
    if (e != cudaSuccess)
        std::printf("  ✓ caught at peek: %s\n\n", cudaGetErrorString(e));

    // -------- 错误 3: 越界写，sync 时才知道 --------
    std::printf("[bug 3] OOB write inside kernel:\n");
    bad_kernel<<<1, 1>>>(p);
    e = cudaPeekAtLastError();
    std::printf("  peek: %s\n", cudaGetErrorString(e));   // 可能仍然 success
    e = cudaDeviceSynchronize();
    std::printf("  sync: %s\n", cudaGetErrorString(e));   // <-- 这里才报错

    return 0;
}
