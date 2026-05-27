// 练习: 实现灰度图像反相 (y = 255 - x) 的 2D kernel.
//   - 图像是 H × W uint8 行优先存储
//   - 使用 dim3(16, 16) block
// 自检: 像素和应该满足 sum(invert) = H*W*255 - sum(orig)
#include "../../common/cuda_utils.h"
#include <vector>
#include <cstdio>

__global__ void invert_kernel(const unsigned char* in, unsigned char* out, int H, int W) {
    // TODO
}

int main() {
    int H = 480, W = 640;
    std::vector<unsigned char> hin(H * W), hout(H * W);
    for (int i = 0; i < H * W; ++i) hin[i] = (unsigned char)(i & 0xff);

    unsigned char *din, *dout;
    CUDA_CHECK(cudaMalloc(&din,  H * W));
    CUDA_CHECK(cudaMalloc(&dout, H * W));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), H * W, cudaMemcpyHostToDevice));

    // TODO: launch invert_kernel
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(hout.data(), dout, H * W, cudaMemcpyDeviceToHost));
    long s_in = 0, s_out = 0;
    for (int i = 0; i < H * W; ++i) { s_in += hin[i]; s_out += hout[i]; }
    std::printf("sum_in + sum_out = %ld (expect %ld)\n", s_in + s_out, long(H) * W * 255);
    return 0;
}
