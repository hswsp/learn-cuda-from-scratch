#include "../../common/cuda_utils.h"
#include <vector>
#include <cstdio>

__global__ void invert_kernel(const unsigned char* in, unsigned char* out, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) {
        int i = row * W + col;
        out[i] = 255 - in[i];
    }
}

int main() {
    int H = 480, W = 640;
    std::vector<unsigned char> hin(H * W), hout(H * W);
    for (int i = 0; i < H * W; ++i) hin[i] = (unsigned char)(i & 0xff);

    unsigned char *din, *dout;
    CUDA_CHECK(cudaMalloc(&din,  H * W));
    CUDA_CHECK(cudaMalloc(&dout, H * W));
    CUDA_CHECK(cudaMemcpy(din, hin.data(), H * W, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    invert_kernel<<<grid, block>>>(din, dout, H, W);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(hout.data(), dout, H * W, cudaMemcpyDeviceToHost));
    long s_in = 0, s_out = 0;
    for (int i = 0; i < H * W; ++i) { s_in += hin[i]; s_out += hout[i]; }
    std::printf("sum_in + sum_out = %ld (expect %ld)\n", s_in + s_out, long(H) * W * 255);
    return 0;
}
