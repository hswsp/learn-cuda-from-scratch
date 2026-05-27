// =============================================================================
// cuda_graph_demo.cu — capture a sequence of kernel launches into a CUDA Graph.
//
// 学习目标:
//   1. 看到 "launch overhead" 在重复短 kernel 时占多大
//   2. 用 cudaStreamBeginCapture / cudaGraphInstantiate / cudaGraphLaunch
//      把序列固化, 重放时只剩 1 次 launch overhead
//
// 现代 LLM 推理: vLLM 与 TensorRT-LLM 用 CUDA Graph 包裹整轮 decoding 步骤.
//
// 对应 HTML: docs/ch08-async/index.html#graphs
// =============================================================================
#include "../common/cuda_utils.h"
#include <cstdio>

__global__ void tiny_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 1.000001f + 1e-7f;
}

int main() {
    const int N = 4096;
    const int repeats = 1000;
    DeviceBuffer<float> d(N);
    std::vector<float> h(N, 1.0f);
    d.copy_from_host(h.data());

    int block = 128, grid = (N + block - 1) / block;
    cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));

    GpuTimer t;

    // --- baseline: 1000 separate kernel launches ---
    t.start();
    for (int i = 0; i < repeats; ++i)
        tiny_kernel<<<grid, block, 0, s>>>(d.ptr, N);
    CUDA_CHECK(cudaStreamSynchronize(s));
    t.stop();
    float ms_loop = t.ms();

    // --- captured into a graph, then launched once ---
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < repeats; ++i)
        tiny_kernel<<<grid, block, 0, s>>>(d.ptr, N);
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    t.start();
    CUDA_CHECK(cudaGraphLaunch(exec, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    t.stop();
    float ms_graph = t.ms();

    std::printf("%d separate launches : %.3f ms (%.2f us/launch)\n",
                repeats, ms_loop, ms_loop * 1000.0 / repeats);
    std::printf("cuda graph (1 launch): %.3f ms (%.2fx faster)\n",
                ms_graph, ms_loop / ms_graph);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    return 0;
}
