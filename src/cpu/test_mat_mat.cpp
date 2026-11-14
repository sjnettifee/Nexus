// Oracle test: prove mat_mat(W, X, ..., n_tokens) is BIT-IDENTICAL to calling
// mat_vec(W, X+t*in_cols, Y+t*out_rows, ...) for each token t.
//
// If this passes, mat_mat is a drop-in batched replacement for the per-token
// prefill loop with zero numerical drift — which makes it the validation
// oracle for the GPU GEMM shader later.
//
// Build:
//   g++ -O3 -march=native -std=gnu++17 -I/mnt/user-data/uploads \
//       test_mat_mat.cpp -o test_mat_mat -lpthread
//   ./test_mat_mat

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>

#include "matmul.h"     // our patched copy (passed via -I or copied alongside)

using namespace nexus::cpu;
using nexus::quant::QuantType;

static std::mt19937 rng(12345);

// Fill a raw byte buffer deterministically. For the oracle test the actual
// decoded weights don't matter — only that mat_mat and mat_vec decode the
// SAME bytes the SAME way. Random bytes fully exercise the quant decoders.
static std::vector<uint8_t> random_bytes(size_t n) {
    std::vector<uint8_t> v(n);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& b : v) b = (uint8_t)d(rng);
    return v;
}

static std::vector<float> random_floats(size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    std::uniform_real_distribution<float> d(-scale, scale);
    for (auto& f : v) f = d(rng);
    return v;
}

struct Case { const char* name; QuantType qt; size_t bytes_per_row_blocks; };

// returns max abs diff and whether bit-identical
static bool run_case(const char* name, QuantType qt,
                     size_t out_rows, size_t in_cols, size_t n_tokens,
                     size_t weight_bytes, int n_threads) {
    auto W = random_bytes(weight_bytes);
    auto X = random_floats(in_cols * n_tokens, 1.0f);  // in_cols × n_tokens

    // Reference: N separate mat_vec calls.
    std::vector<float> Yref(out_rows * n_tokens, 0.0f);
    for (size_t t = 0; t < n_tokens; ++t) {
        mat_vec(W.data(), X.data() + t * in_cols,
                Yref.data() + t * out_rows,
                out_rows, in_cols, qt, n_threads);
    }

    // Batched.
    std::vector<float> Ybatch(out_rows * n_tokens, 0.0f);
    mat_mat(W.data(), X.data(), Ybatch.data(),
            out_rows, in_cols, n_tokens, qt, n_threads);

    // Compare bit-for-bit.
    size_t mismatches = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_rows * n_tokens; ++i) {
        uint32_t a, b;
        memcpy(&a, &Yref[i], 4);
        memcpy(&b, &Ybatch[i], 4);
        if (a != b) {
            ++mismatches;
            float d = std::fabs(Yref[i] - Ybatch[i]);
            if (d > max_abs) max_abs = d;
        }
    }

    bool ok = (mismatches == 0);
    printf("  %-10s rows=%-5zu cols=%-5zu N=%-4zu thr=%d : %s",
           name, out_rows, in_cols, n_tokens, n_threads,
           ok ? "BIT-IDENTICAL ✓" : "MISMATCH ✗");
    if (!ok) printf("  (%zu mismatches, max|Δ|=%.3e)", mismatches, max_abs);
    printf("\n");
    return ok;
}

int main() {
    printf("mat_mat oracle test — must be bit-identical to N×mat_vec\n");
    printf("--------------------------------------------------------\n");

    bool all = true;

    // F32: fully controlled, exact. in_cols sized to realistic hidden dim.
    {
        size_t rows = 512, cols = 5120, N = 37;
        size_t wbytes = rows * cols * sizeof(float);
        for (int thr : {1, 4}) all &= run_case("F32", QuantType::F32, rows, cols, N, wbytes, thr);
    }

    // Q4_K: 144 bytes / 256-elem block. cols multiple of 256 → q8k path active.
    {
        size_t rows = 512, cols = 5120, N = 37;     // 5120 = 20 blocks/row
        size_t blocks_per_row = cols / 256;
        size_t wbytes = rows * blocks_per_row * 144;
        for (int thr : {1, 4}) all &= run_case("Q4_K", QuantType::Q4_K, rows, cols, N, wbytes, thr);
    }

    // Q6_K: 210 bytes / 256-elem block.
    {
        size_t rows = 256, cols = 5120, N = 37;
        size_t blocks_per_row = cols / 256;
        size_t wbytes = rows * blocks_per_row * 210;
        for (int thr : {1, 4}) all &= run_case("Q6_K", QuantType::Q6_K, rows, cols, N, wbytes, thr);
    }

    // Edge: N=1 must hit the mat_vec fast path and still match.
    {
        size_t rows = 128, cols = 512, N = 1;
        size_t wbytes = rows * cols * sizeof(float);
        all &= run_case("F32 N=1", QuantType::F32, rows, cols, N, wbytes, 4);
    }

    // Edge: large N (full prompt) to exercise the batched sweep.
    {
        size_t rows = 256, cols = 5120, N = 1742;
        size_t blocks_per_row = cols / 256;
        size_t wbytes = rows * blocks_per_row * 144;
        all &= run_case("Q4_K big", QuantType::Q4_K, rows, cols, N, wbytes, 4);
    }

    printf("--------------------------------------------------------\n");
    printf("%s\n", all ? "ALL PASS ✓ — mat_mat is a safe batched drop-in"
                       : "FAILURES ✗ — do NOT proceed to shader work");
    return all ? 0 : 1;
}
