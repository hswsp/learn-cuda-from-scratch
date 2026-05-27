// =============================================================================
// check.h — numerical comparison helpers.
//
// Every CUDA kernel in this repo gets verified against a CPU reference. The
// functions below return a struct describing max abs / max rel / L2 error
// and print a one-line summary so tutorial output stays readable.
// =============================================================================
#pragma once

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <string>

struct CheckResult {
    float max_abs = 0.f;
    float max_rel = 0.f;
    float l2_err  = 0.f;
    size_t bad_idx = 0;
    bool   pass    = true;
};

// Compare two flat float arrays element-wise.
// atol: absolute tolerance, rtol: relative tolerance.
// Default thresholds suit fp32 matmul up to K=1024.
inline CheckResult allclose(const float* a, const float* b, size_t n,
                            float atol = 1e-3f, float rtol = 1e-3f) {
    CheckResult r;
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float da = std::fabs(a[i] - b[i]);
        float dr = da / (std::fabs(b[i]) + 1e-9f);
        if (da > r.max_abs) { r.max_abs = da; r.bad_idx = i; }
        if (dr > r.max_rel) r.max_rel = dr;
        if (da > atol && dr > rtol) r.pass = false;
        acc += double(da) * double(da);
    }
    r.l2_err = std::sqrt(float(acc / std::max<size_t>(n, 1)));
    return r;
}

inline CheckResult allclose(const std::vector<float>& a, const std::vector<float>& b,
                            float atol = 1e-3f, float rtol = 1e-3f) {
    return allclose(a.data(), b.data(), std::min(a.size(), b.size()), atol, rtol);
}

// Print "[name] PASS  max_abs=... l2=..." style line.
inline void report(const std::string& name, const CheckResult& r) {
    std::printf("[%s] %s  max_abs=%.3e  max_rel=%.3e  l2=%.3e",
                name.c_str(), r.pass ? "PASS" : "FAIL",
                r.max_abs, r.max_rel, r.l2_err);
    if (!r.pass) std::printf("  (first bad idx=%zu)", r.bad_idx);
    std::printf("\n");
}

// GFLOPS helper for matmul / attention reporting.
inline double gflops(double ops, float ms) {
    return ops / (ms * 1e6);  // ops / (ms * 1e-3) / 1e9
}

inline double matmul_ops(int M, int N, int K) {
    return 2.0 * double(M) * double(N) * double(K);  // each cell: K muls + K adds
}
