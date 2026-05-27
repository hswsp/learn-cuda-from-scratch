// =============================================================================
// transpose.cu — naive transpose vs shared-mem (no-bank-conflict) transpose.
//
// 三个 kernel:
//   1. naive_transpose  — 写 strided, 内存带宽爆炸式下降
//   2. tiled_transpose  — 用 shared 把 tile 转下来再写出, 读写都 coalesced
//   3. padded_transpose — shared 数组 +1 padding 消除 bank conflict
//
// 对应 HTML: docs/ch06-tile/index.html#transpose
// =============================================================================
#include "../common/cuda_utils.h"
#include "../common/cpu_ref.h"
#include "../common/check.h"

constexpr int TILE = 32;

__global__ void transpose_naive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // col in in
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row in in
    if (x < N && y < M) out[x * M + y] = in[y * N + x];     // <-- 写是 strided
}

__global__ void transpose_shared(const float* in, float* out, int M, int N) {
    __shared__ float tile[TILE][TILE];     // 没有 padding, 有 bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;       // 注意:换 block 坐标
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < M && y2 < N) out[y2 * M + x2] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_padded(const float* in, float* out, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding 消除 bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * TILE + threadIdx.x;
    int y2 = blockIdx.x * TILE + threadIdx.y;
    if (x2 < M && y2 < N) out[y2 * M + x2] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int M = 4096, N = 4096;
    auto hin = make_random(M * N);
    std::vector<float> hout(M * N), href(M * N);
    cpu_ref::transpose(hin.data(), href.data(), M, N);

    DeviceBuffer<float> din(M*N), dout(M*N);
    din.copy_from_host(hin.data());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    GpuTimer t;
    auto bench = [&](const char* name, auto launch) {
        launch(); KERNEL_CHECK();
        t.start(); for (int r = 0; r < 5; ++r) launch();
        t.stop(); KERNEL_CHECK();
        float ms = t.ms() / 5;
        dout.copy_to_host(hout.data());
        auto r = allclose(hout, href);
        double gb = 2.0 * size_t(M) * N * sizeof(float) / 1e9;
        std::printf("  %-20s %s  %.3f ms  %.1f GB/s\n",
                    name, r.pass ? "PASS" : "FAIL", ms, gb / (ms * 1e-3));
    };
    bench("naive",  [&]{ transpose_naive  <<<grid, block>>>(din.ptr, dout.ptr, M, N); });
    bench("shared", [&]{ transpose_shared <<<grid, block>>>(din.ptr, dout.ptr, M, N); });
    bench("padded", [&]{ transpose_padded <<<grid, block>>>(din.ptr, dout.ptr, M, N); });
    return 0;
}
