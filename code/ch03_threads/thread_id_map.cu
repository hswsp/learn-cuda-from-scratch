// =============================================================================
// thread_id_map.cu — print (block, thread) → global id mapping for tiny grid.
//
// 学习目标:
//   - 直观看到 blockIdx / threadIdx / blockDim 如何拼出全局 id
//   - 看到 warp 边界（thread id 每 32 个一组同步执行）
//
// 对应 HTML: docs/ch03-threads/index.html#mapping
// =============================================================================
#include "../common/cuda_utils.h"

__global__ void map_kernel(int n) {
    int tid_in_block = threadIdx.x;
    int gid          = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id      = tid_in_block / 32;
    int lane_id      = tid_in_block & 31;
    if (gid < n)
        printf("blk=%d  tid=%2d  warp=%d  lane=%2d  gid=%2d\n",
               blockIdx.x, tid_in_block, warp_id, lane_id, gid);
}

int main() {
    // 2 blocks × 40 threads = 80 lines.  block 0 lane 0..31 是 warp 0, lane 32..39 是 warp 1 头.
    map_kernel<<<2, 40>>>(80);
    KERNEL_CHECK();
    return 0;
}
