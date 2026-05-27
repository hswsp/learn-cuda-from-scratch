// 练习: 进阶 tiled matmul, 每个 thread 算 4x4 = 16 个 output cell.
// 思路: BM=BN=64, BK=16; block size 16x16 (每 thread 算 4x4 of C).
// 这样每 thread 寄存器复用更多, GFLOPS 应能再翻 2-3 倍.
//
// 提示:
//   - 每 thread 维护 4x4 个 reg accumulator
//   - 每个 K-tile 中, 拉 A 的 64x16 + B 的 16x64 到 shared mem
//   - 内层循环: 每 thread 读 4 个 As 列 + 4 个 Bs 行, 做 16 次 fmadd
//
// 自检: 与 cpu_ref::gemm allclose.

#include "../../common/cuda_utils.h"
// TODO
int main() { return 0; }
