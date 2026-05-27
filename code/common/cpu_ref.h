// =============================================================================
// cpu_ref.h — CPU reference implementations of every kernel in the tutorial.
//
// Purpose:
//   1. Ground truth for `allclose` correctness checks.
//   2. Lets you author / test the tutorial on a machine without an NVIDIA GPU
//      (e.g. macOS arm64).
//
// Style: prefers clarity over performance. NO OpenMP, NO SIMD intrinsics.
// If you want a fast CPU baseline, link against OpenBLAS / Accelerate.
// =============================================================================
#pragma once

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cstddef>

namespace cpu_ref {

// ---- vector add: C[i] = A[i] + B[i] ----------------------------------------
inline void vec_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ---- SAXPY: y = a*x + y ----------------------------------------------------
inline void saxpy(float a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// ---- GEMM (row-major): C = A @ B,  A:(M,K)  B:(K,N)  C:(M,N) ---------------
inline void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
            C[m * N + n] = s;
        }
}

// ---- Sum reduction ---------------------------------------------------------
inline float sum(const float* x, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return float(s);
}

// ---- 2D transpose: B = A^T  (A: M×N → B: N×M) ------------------------------
inline void transpose(const float* A, float* B, int M, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            B[j * M + i] = A[i * N + j];
}

// ---- Stable softmax along the last dim ------------------------------------
//   x: (rows, cols)  y: (rows, cols)
inline void softmax_lastdim(const float* x, float* y, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols;
        float*       yr = y + r * cols;
        float m = xr[0];
        for (int c = 1; c < cols; ++c) m = std::max(m, xr[c]);
        double s = 0.0;
        for (int c = 0; c < cols; ++c) { yr[c] = std::exp(xr[c] - m); s += yr[c]; }
        float inv = 1.f / float(s);
        for (int c = 0; c < cols; ++c) yr[c] *= inv;
    }
}

// ---- LayerNorm along the last dim ------------------------------------------
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
inline void layernorm(const float* x, const float* gamma, const float* beta,
                      float* y, int rows, int cols, float eps = 1e-5f) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols; float* yr = y + r * cols;
        double mean = 0.0;
        for (int c = 0; c < cols; ++c) mean += xr[c];
        mean /= cols;
        double var = 0.0;
        for (int c = 0; c < cols; ++c) { double d = xr[c] - mean; var += d * d; }
        var /= cols;
        float inv = 1.f / std::sqrt(float(var) + eps);
        for (int c = 0; c < cols; ++c)
            yr[c] = (xr[c] - float(mean)) * inv * gamma[c] + beta[c];
    }
}

// ---- RMSNorm along the last dim (used by Llama/GPT-NeoX) -------------------
//   y = x / sqrt(mean(x^2) + eps) * gamma
inline void rmsnorm(const float* x, const float* gamma, float* y,
                    int rows, int cols, float eps = 1e-6f) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + r * cols; float* yr = y + r * cols;
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += double(xr[c]) * double(xr[c]);
        float inv = 1.f / std::sqrt(float(s / cols) + eps);
        for (int c = 0; c < cols; ++c) yr[c] = xr[c] * inv * gamma[c];
    }
}

// ---- Activations -----------------------------------------------------------
inline float gelu_exact(float x) {
    const float kSqrt2 = 1.41421356237f;
    return 0.5f * x * (1.f + std::erf(x / kSqrt2));
}
inline float gelu_tanh(float x) {
    const float k0 = 0.7978845608f;        // sqrt(2/pi)
    const float k1 = 0.044715f;
    return 0.5f * x * (1.f + std::tanh(k0 * (x + k1 * x * x * x)));
}
inline float silu(float x) {
    return x / (1.f + std::exp(-x));
}

// ---- Scaled Dot-Product Attention (single head, fp32) ----------------------
//   Q, K, V: (T, D), out: (T, D), causal optional.
inline void attention(const float* Q, const float* K, const float* V,
                      float* out, int T, int D, bool causal = true) {
    std::vector<float> scores(T * T);
    float scale = 1.f / std::sqrt(float(D));
    // scores = Q @ K^T * scale
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j) {
            float s = 0.f;
            for (int d = 0; d < D; ++d) s += Q[i * D + d] * K[j * D + d];
            scores[i * T + j] = s * scale;
            if (causal && j > i) scores[i * T + j] = -1e30f;
        }
    std::vector<float> P(T * T);
    softmax_lastdim(scores.data(), P.data(), T, T);
    // out = P @ V
    for (int i = 0; i < T; ++i)
        for (int d = 0; d < D; ++d) {
            float s = 0.f;
            for (int j = 0; j < T; ++j) s += P[i * T + j] * V[j * D + d];
            out[i * D + d] = s;
        }
}

// ---- Rotary Position Embedding (RoPE) --------------------------------------
//   x: (T, D) — D must be even; in-place rotation by position-dependent angles.
inline void rope_inplace(float* x, int T, int D, float base = 10000.f) {
    int half = D / 2;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < half; ++i) {
            float theta = float(t) / std::pow(base, float(2 * i) / float(D));
            float c = std::cos(theta), s = std::sin(theta);
            float x0 = x[t * D + i];
            float x1 = x[t * D + i + half];
            x[t * D + i]        = x0 * c - x1 * s;
            x[t * D + i + half] = x0 * s + x1 * c;
        }
}

// ---- Greedy argmax sampling ------------------------------------------------
inline int argmax(const float* logits, int V) {
    int best = 0; float bv = logits[0];
    for (int i = 1; i < V; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

} // namespace cpu_ref
