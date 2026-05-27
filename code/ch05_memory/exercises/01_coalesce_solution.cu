#include "../../common/cuda_utils.h"
#include <cstdio>

__global__ void copy_bad(const float* in, float* out, int H, int W) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;     // strided
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) out[row * W + col] = in[row * W + col];
}

__global__ void copy_good(const float* in, float* out, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // x → col, coalesced
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < H && col < W) out[row * W + col] = in[row * W + col];
}

int main() {
    int H = 4096, W = 4096;
    size_t bytes = size_t(H) * W * sizeof(float);
    DeviceBuffer<float> di(H * W), dout(H * W);
    std::vector<float> h(H * W, 1.0f);
    di.copy_from_host(h.data());

    dim3 block(32, 8);
    dim3 grid_bad ((H + block.x - 1) / block.x, (W + block.y - 1) / block.y);
    dim3 grid_good((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    GpuTimer t;
    t.start(); copy_bad <<<grid_bad,  block>>>(di.ptr, dout.ptr, H, W); t.stop(); KERNEL_CHECK();
    float ms_bad = t.ms();
    t.start(); copy_good<<<grid_good, block>>>(di.ptr, dout.ptr, H, W); t.stop(); KERNEL_CHECK();
    float ms_good = t.ms();
    std::printf("bad : %.3f ms (%.1f GB/s)\n", ms_bad,  2.0 * bytes / (ms_bad  * 1e6));
    std::printf("good: %.3f ms (%.1f GB/s)\n", ms_good, 2.0 * bytes / (ms_good * 1e6));
    return 0;
}
