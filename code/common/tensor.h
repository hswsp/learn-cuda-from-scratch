// =============================================================================
// tensor.h — minimal row-major Tensor wrapper.
//
// We avoid pulling in PyTorch / Eigen / xtensor: this struct exists only so
// chapter examples can talk about shapes without endless raw pointer + size
// arguments. Storage is owned by DeviceBuffer<T> (see cuda_utils.h).
//
// Layout: always row-major. shape[0] is the slowest-varying dim.
// =============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <numeric>
#include <functional>
#include <initializer_list>

// View struct only — does NOT own the pointer. Pair with DeviceBuffer<T>
// or a host std::vector<T>.
template <typename T, int N>
struct TensorView {
    T* data = nullptr;
    std::array<int, N> shape{};

    TensorView() = default;
    TensorView(T* p, std::initializer_list<int> s) : data(p) {
        int i = 0;
        for (int v : s) shape[i++] = v;
    }

    size_t numel() const {
        size_t n = 1;
        for (int v : shape) n *= size_t(v);
        return n;
    }

    // stride[i] = product of shape[i+1..N-1]. Row-major.
    size_t stride(int dim) const {
        size_t s = 1;
        for (int i = dim + 1; i < N; ++i) s *= size_t(shape[i]);
        return s;
    }
};

// Convenience aliases used heavily in attention / GEMM chapters.
template <typename T> using Tensor1D = TensorView<T, 1>;
template <typename T> using Tensor2D = TensorView<T, 2>;
template <typename T> using Tensor3D = TensorView<T, 3>;
template <typename T> using Tensor4D = TensorView<T, 4>;
