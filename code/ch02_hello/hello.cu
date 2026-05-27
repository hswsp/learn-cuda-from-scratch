// =============================================================================
// hello.cu — the canonical first CUDA program.
//
// 学习目标:
//   1. 写出第一个 __global__ kernel
//   2. 理解 <<<grid, block>>> 启动语法
//   3. 明白 host 与 device 是两个独立的执行空间
//   4. 用 KERNEL_CHECK() 抓住静默错误
//
// 对应 HTML: docs/ch02-hello/index.html#hello
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__global__ void hello_kernel() {
    // 每个线程独立打印自己的 (block, thread) 坐标
    printf("hello from thread (%d, %d) of block (%d, %d)\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x,  blockIdx.y);
}

int main() {
    std::printf("--- launching 2 blocks x 4 threads ---\n");

    // 启动语法: kernel<<<grid_dim, block_dim>>>(args)
    // 这里 grid = 2 (一维), block = 4 (一维)，总共 2*4 = 8 个线程
    hello_kernel<<<2, 4>>>();
    KERNEL_CHECK();   // 等 kernel 跑完 + 检查错误

    std::printf("--- launching 1 block x dim3(2,2) threads ---\n");
    hello_kernel<<<1, dim3(2, 2)>>>();
    KERNEL_CHECK();

    std::printf("done.\n");
    return 0;
}
