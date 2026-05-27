// =============================================================================
// cuda_utils.h — minimal CUDA helpers shared by every chapter.
//
// Goals:
//   1. Make every CUDA API call check its return code (CUDA_CHECK).
//   2. Make every kernel launch check both launch error and runtime error
//      (KERNEL_CHECK).
//   3. Provide a one-liner GPU timer (GpuTimer) so examples don't drown in
//      cudaEventCreate boilerplate.
//   4. Provide an RAII device buffer (DeviceBuffer<T>) to avoid manual
//      cudaMalloc/cudaFree pairs in tutorial code.
//
// This file is header-only and intentionally tiny. Read it once, you'll see
// it referenced by every .cu in this repo.
// =============================================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// CPU_ONLY mode: stub out CUDA so cpu_ref demos can compile on macOS.
#ifdef CPU_ONLY
// Minimal stubs — enough to satisfy header inclusion when no CUDA is present.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA stubbed (CPU_ONLY)"; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaMalloc(void**, size_t) { return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return cudaSuccess; }
constexpr int cudaMemcpyHostToDevice = 1;
constexpr int cudaMemcpyDeviceToHost = 2;
constexpr int cudaMemcpyDeviceToDevice = 3;
#else
#include <cuda_runtime.h>
#endif

// ---- Error checking macros --------------------------------------------------
//
// Use CUDA_CHECK around every CUDA runtime API call:
//     CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
//
// Use KERNEL_CHECK() *immediately after* every kernel launch:
//     mykernel<<<g, b>>>(...);
//     KERNEL_CHECK();
//
// KERNEL_CHECK does two things:
//   1. cudaPeekAtLastError    — catches launch config errors (bad grid, etc.)
//   2. cudaDeviceSynchronize  — surfaces asynchronous in-kernel errors
//                               (illegal address, misaligned load, ...)
// On a hot loop you may want to skip the sync; for tutorial clarity we always
// sync so failures point at the right line.

#define CUDA_CHECK(stmt) do {                                                  \
    cudaError_t _e = (stmt);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "[CUDA_CHECK] %s:%d: %s -> %s\n",                 \
                     __FILE__, __LINE__, #stmt, cudaGetErrorString(_e));       \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)

#define KERNEL_CHECK() do {                                                    \
    CUDA_CHECK(cudaPeekAtLastError());                                         \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
} while (0)

// ---- GpuTimer ---------------------------------------------------------------
//
// Usage:
//     GpuTimer t;
//     t.start();
//     mykernel<<<g, b>>>(...);
//     t.stop();
//     printf("kernel = %.3f ms\n", t.ms());
//
// Backed by cudaEvent_t so it measures GPU wall time, not CPU launch overhead.

#ifndef CPU_ONLY
struct GpuTimer {
    cudaEvent_t s_, e_;
    GpuTimer()  { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~GpuTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }

    void start(cudaStream_t stream = 0) { cudaEventRecord(s_, stream); }
    void stop (cudaStream_t stream = 0) {
        cudaEventRecord(e_, stream);
        cudaEventSynchronize(e_);
    }
    float ms() const {
        float v = 0.f;
        cudaEventElapsedTime(&v, s_, e_);
        return v;
    }
};
#else
struct GpuTimer {
    void start(int = 0) {}
    void stop (int = 0) {}
    float ms() const { return 0.f; }
};
#endif

// ---- DeviceBuffer<T> --------------------------------------------------------
//
// RAII wrapper around cudaMalloc/cudaFree. Use it whenever a host helper
// allocates GPU memory:
//
//     DeviceBuffer<float> d(1024);     // cudaMalloc(1024 * sizeof(float))
//     d.copy_from_host(host_ptr);
//     mykernel<<<g, b>>>(d.ptr, ...);
//     d.copy_to_host(host_ptr);        // dtor frees

template <typename T>
struct DeviceBuffer {
    T*     ptr  = nullptr;
    size_t size = 0;  // element count, not bytes

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { allocate(n); }
    ~DeviceBuffer() { free_(); }

    // Movable, non-copyable.
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr; o.size = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { free_(); ptr = o.ptr; size = o.size; o.ptr = nullptr; o.size = 0; }
        return *this;
    }

    void allocate(size_t n) {
        free_();
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
        size = n;
    }
    void copy_from_host(const T* host) {
        CUDA_CHECK(cudaMemcpy(ptr, host, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* host) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    size_t bytes() const { return size * sizeof(T); }

  private:
    void free_() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; }
    }
};

// ---- Simple host RNG --------------------------------------------------------
//
// Deterministic fill so allclose() checks are reproducible across runs.

inline void fill_random(std::vector<float>& v, uint32_t seed = 42, float scale = 1.f) {
    uint32_t s = seed;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;       // LCG
        float u = (s >> 8) * (1.f / 16777216.f); // [0,1)
        x = (u * 2.f - 1.f) * scale;             // [-scale, scale)
    }
}

inline std::vector<float> make_random(size_t n, uint32_t seed = 42, float scale = 1.f) {
    std::vector<float> v(n);
    fill_random(v, seed, scale);
    return v;
}

// ---- Tiny CLI arg parser ----------------------------------------------------
//
// Supports --M=1024 --N=512 --verbose style. Used by chapter demos.

inline int arg_int(int argc, char** argv, const char* name, int dflt) {
    std::string key = std::string("--") + name + "=";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0)
            return std::atoi(argv[i] + key.size());
    }
    return dflt;
}
inline bool arg_flag(int argc, char** argv, const char* name) {
    std::string key = std::string("--") + name;
    for (int i = 1; i < argc; ++i)
        if (key == argv[i]) return true;
    return false;
}
