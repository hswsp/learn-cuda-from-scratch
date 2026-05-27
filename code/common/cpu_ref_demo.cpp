// =============================================================================
// cpu_ref_demo.cpp — sanity check that you can build & run the CPU reference
// implementations on a machine *without* CUDA. Useful for macOS users.
//
// Build & run:
//     clang++ -O2 -std=c++17 -DCPU_ONLY cpu_ref_demo.cpp -o /tmp/ref && /tmp/ref
// =============================================================================
#include "cuda_utils.h"   // pulls in make_random / fill_random (CPU_ONLY stubs CUDA)
#include "cpu_ref.h"
#include "check.h"
#include <cstdio>
#include <vector>

int main() {
    // 1) vec_add
    {
        std::vector<float> a{1,2,3,4}, b{10,20,30,40}, c(4);
        cpu_ref::vec_add(a.data(), b.data(), c.data(), 4);
        std::printf("vec_add  : %.0f %.0f %.0f %.0f\n", c[0], c[1], c[2], c[3]);
    }
    // 2) gemm 2x3 @ 3x2
    {
        std::vector<float> A{1,2,3, 4,5,6};       // 2x3
        std::vector<float> B{7,8, 9,10, 11,12};   // 3x2
        std::vector<float> C(4);
        cpu_ref::gemm(A.data(), B.data(), C.data(), 2, 2, 3);
        // expected: [58, 64, 139, 154]
        std::printf("gemm     : %.0f %.0f %.0f %.0f\n", C[0], C[1], C[2], C[3]);
    }
    // 3) softmax
    {
        std::vector<float> x{1, 2, 3, 4}, y(4);
        cpu_ref::softmax_lastdim(x.data(), y.data(), 1, 4);
        float s = y[0] + y[1] + y[2] + y[3];
        std::printf("softmax  : sum=%.4f (want 1.0), y[3]=%.4f\n", s, y[3]);
    }
    // 4) attention smoke test (T=4, D=8)
    {
        int T = 4, D = 8;
        auto Q = make_random(T * D, 1), K = make_random(T * D, 2), V = make_random(T * D, 3);
        std::vector<float> out(T * D);
        cpu_ref::attention(Q.data(), K.data(), V.data(), out.data(), T, D, /*causal=*/true);
        std::printf("attn[0]  : %.4f %.4f %.4f ...\n", out[0], out[1], out[2]);
    }
    std::printf("\nALL OK — cpu_ref builds and runs on this host.\n");
    return 0;
}
