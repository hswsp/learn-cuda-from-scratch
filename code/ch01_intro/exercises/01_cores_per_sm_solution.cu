// Solution: cores_per_sm lookup table (synced with NVIDIA's helper_cuda.h).
#include "../../common/cuda_utils.h"
#include <cstdio>

int cores_per_sm(int major, int minor) {
    int v = (major << 4) | minor;
    switch (v) {
        case 0x30: case 0x32: case 0x35: case 0x37: return 192;  // Kepler
        case 0x50: case 0x52: case 0x53:            return 128;  // Maxwell
        case 0x60:                                  return  64;  // P100
        case 0x61: case 0x62:                       return 128;  // GP10x
        case 0x70: case 0x72:                       return  64;  // V100, Xavier
        case 0x75:                                  return  64;  // Turing
        case 0x80:                                  return  64;  // A100
        case 0x86: case 0x87:                       return 128;  // GA10x / Orin
        case 0x89:                                  return 128;  // Ada (4090)
        case 0x90:                                  return 128;  // H100
        default:                                    return  64;  // fallback
    }
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
