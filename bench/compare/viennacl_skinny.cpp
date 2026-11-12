// Skinny GEMM in ViennaCL 1.7.1 (OpenCL through NEO): C (M×N) = A (M×K) · Wᵀ, W (N×K) row-major -- the same shapes as
// bench/skinny_gemm.py (libgen9blas). FP32 only (ViennaCL has no FP16). For M = 1 also GEMV y = W·x.
// Method: warm-up 3×, then repeat up to ≥ 0.5 s, median of single-call times (with viennacl::backend::finish()).
// Build: g++ -O2 -DVIENNACL_WITH_OPENCL -I <ViennaCL 1.7.1 source dir> viennacl_skinny.cpp -lOpenCL
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/prod.hpp"

template <class F> static double measure(F f) {
    for (int i = 0; i < 3; i++) f();
    viennacl::backend::finish();
    std::vector<double> t; double sum = 0;
    while (sum < 0.5 || t.size() < 5) {
        auto a = std::chrono::steady_clock::now();
        f(); viennacl::backend::finish();
        double d = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        t.push_back(d); sum += d;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2] * 1e3;
}

int main() {
    std::mt19937 g(1); std::uniform_real_distribution<float> u(-1, 1);
    const int ksz[][2] = { { 4096, 4096 }, { 11008, 4096 }, { 4096, 11008 }, { 2048, 2048 } };
    for (auto &nk : ksz) {
        const int N = nk[0], K = nk[1];
        std::vector<float> h((size_t)N * K); for (auto &x : h) x = u(g);
        viennacl::matrix<float, viennacl::row_major> W(N, K); viennacl::fast_copy(h.data(), h.data() + h.size(), W);
        {
            std::vector<float> hx(K); for (auto &x : hx) x = u(g);
            viennacl::vector<float> x(K), y(N); viennacl::copy(hx, x);
            double t = measure([&] { y = viennacl::linalg::prod(W, x); });
            std::printf("gemv %d %d 1 %.3f\n", N, K, t);
        }
        for (int M : { 1, 2, 4, 8, 16 }) {
            std::vector<float> ha((size_t)M * K); for (auto &x : ha) x = u(g);
            viennacl::matrix<float, viennacl::row_major> A(M, K), C(M, N);
            viennacl::fast_copy(ha.data(), ha.data() + ha.size(), A);
            double t = measure([&] { C = viennacl::linalg::prod(A, viennacl::trans(W)); });
            std::printf("gemm %d %d %d %.3f\n", N, K, M, t);
            std::fflush(stdout);
        }
    }
}
