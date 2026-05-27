// =============================================================================
// sampling.cu — greedy / top-k argmax sampling for next-token prediction.
//
// 这里只演示 greedy (argmax). top-k / top-p 在练习里.
//
// 对应 HTML: docs/ch13-llm-parts/index.html#sampling
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"
#include <cfloat>

// block-wide argmax over V elements (V <= 50272 GPT-2 vocab)
template <int BLOCK>
__global__ void greedy_argmax(const float* logits, int* out_idx, int V) {
    __shared__ float vals[BLOCK];
    __shared__ int   idxs[BLOCK];
    int tid = threadIdx.x;
    float bv = -FLT_MAX; int bi = -1;
    for (int i = tid; i < V; i += BLOCK) {
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    }
    vals[tid] = bv; idxs[tid] = bi;
    __syncthreads();
    // tree reduce
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (vals[tid + s] > vals[tid]) {
                vals[tid] = vals[tid + s];
                idxs[tid] = idxs[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = idxs[0];
}

int main() {
    int V = 50272;
    auto h = make_random(V, 1, 5.0f);
    int ref = cpu_ref::argmax(h.data(), V);

    DeviceBuffer<float> dl(V); DeviceBuffer<int> di(1);
    dl.copy_from_host(h.data());
    greedy_argmax<256><<<1, 256>>>(dl.ptr, di.ptr, V);
    KERNEL_CHECK();
    int got; CUDA_CHECK(cudaMemcpy(&got, di.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("greedy argmax: gpu=%d  cpu=%d  match=%s\n",
                got, ref, got == ref ? "YES" : "NO");
    return 0;
}
