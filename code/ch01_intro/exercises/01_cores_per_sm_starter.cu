// 练习 1: 扩展 device_query，打印每个 SM 上的 CUDA core 数。
//
// 提示: cudaDeviceProp 没有直接给 cores/SM。需要根据 (major, minor) 查表:
//   Pascal sm_60: 64,  sm_61: 128,  sm_62: 128
//   Volta  sm_70: 64
//   Turing sm_75: 64
//   Ampere sm_80: 64,  sm_86: 128,  sm_87: 128
//   Ada    sm_89: 128
//   Hopper sm_90: 128
//
// 把你的实现填到 cores_per_sm 函数里，然后:
//   total_cuda_cores = cores_per_sm * p.multiProcessorCount

#include "../../common/cuda_utils.h"
#include <cstdio>

int cores_per_sm(int major, int minor) {
    // TODO: fill in
    return 0;
}

int main() {
    int n; CUDA_CHECK(cudaGetDeviceCount(&n));
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        int cps = cores_per_sm(p.major, p.minor);
        std::printf("GPU %d %s: %d SMs × %d cores = %d CUDA cores\n",
                    d, p.name, p.multiProcessorCount, cps,
                    p.multiProcessorCount * cps);
    }
    return 0;
}
