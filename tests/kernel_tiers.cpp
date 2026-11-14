// kernel_tiers.cpp -- correctness of the CPU matmul kernels for the tier this file is
// compiled with.  Build it once per tier (-march=x86-64, x86-64-v2, x86-64-v3, ...) and run:
//   kernel_tiers <out.bin>
// For every quantization type it
//   * builds valid random quantized weights,
//   * runs cpu::mat_vec and cpu::mat_mat (multi-thread and single-thread),
//   * compares against a double-precision reference built from quant::dequantize_row,
//   * checks mat_mat == mat_vec column by column,
// and dumps all outputs to <out.bin> so tiers can be compared with each other
// (tests/compare_tiers.py).  Exit code 0 = all within tolerance.
#include "cpu/matmul.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>

using namespace nexus;
using quant::QuantType;

static uint16_t f32_to_f16(float f) {           // simple RNE-free converter, fine for small positives
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000, e = ((x >> 23) & 0xFF), m = x & 0x7FFFFF;
    int ne = (int)e - 127 + 15;
    if (ne <= 0) return (uint16_t)sign;
    if (ne >= 31) return (uint16_t)(sign | 0x7BFF);
    return (uint16_t)(sign | (ne << 10) | (m >> 13));
}

struct T { const char* name; QuantType t; size_t blk_elems; size_t blk_bytes; double tol; };

static size_t row_bytes(const T& t, size_t cols) { return (cols / t.blk_elems) * t.blk_bytes; }

int main(int argc, char** argv) {
    const char* outp = argc > 1 ? argv[1] : nullptr;
    const T types[] = {
        {"F32",  QuantType::F32,  1,   4,   1e-4},
        {"F16",  QuantType::F16,  1,   2,   1e-4},
        {"Q4_0", QuantType::Q4_0, 32,  18,  1e-4},
        {"Q5_0", QuantType::Q5_0, 32,  22,  1e-4},
        {"Q8_0", QuantType::Q8_0, 32,  34,  1e-4},
        {"Q4_K", QuantType::Q4_K, 256, 144, 4e-2},   // activations are quantized to Q8_K
        {"Q5_K", QuantType::Q5_K, 256, 176, 4e-2},
        {"Q6_K", QuantType::Q6_K, 256, 210, 4e-2},
        {"NQ4",  QuantType::NQ4_G64, 64, 34, 1e-4},
        {"NQ5",  QuantType::NQ5_G64, 64, 42, 1e-4},
        {"NQ6",  QuantType::NQ6_G64, 64, 50, 1e-4},
        {"NQ8",  QuantType::NQ8_G64, 64, 66, 1e-4},
    };
    const size_t ROWS = 37, COLS = 512, NTOK = 5;     // COLS multiple of 256 / 64 / 32; odd ROWS on purpose
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> U(-1.f, 1.f);

    std::vector<float> x(COLS), X(COLS * NTOK);
    for (auto& v : x) v = U(rng);
    for (auto& v : X) v = U(rng);

    FILE* out = outp ? fopen(outp, "wb") : nullptr;
    int fails = 0;
    printf("%-5s %-12s %-12s %-12s %-10s\n", "type", "mat_vec err", "mat_vec 1thr", "mat_mat err", "mat_mat==vec");
    for (const T& t : types) {
        const size_t rb = row_bytes(t, COLS);
        std::vector<uint8_t> W(rb * ROWS);
        for (auto& b : W) b = (uint8_t)(rng() & 0xFF);
        // make every block's scale fields sane (random bytes would give inf/NaN in f16 fields)
        auto setd = [&](size_t off) { uint16_t h = f32_to_f16(0.01f + 0.02f * (float)(rng() % 100) / 100.f);
                                      memcpy(&W[off], &h, 2); };
        for (size_t r = 0; r < ROWS; ++r)
            for (size_t b = 0; b < COLS / t.blk_elems; ++b) {
                size_t o = r * rb + b * t.blk_bytes;
                switch (t.t) {
                    case QuantType::F32: { float f = U(rng); memcpy(&W[o], &f, 4); break; }
                    case QuantType::F16: { uint16_t h = f32_to_f16(U(rng) + 2.f); memcpy(&W[o], &h, 2); break; }
                    case QuantType::Q4_K: case QuantType::Q5_K: setd(o); setd(o + 2); break;       // d, dmin
                    case QuantType::Q6_K: setd(o + 208); break;                                      // d is last
                    default: setd(o); break;                                                         // d first
                }
            }

        // double-precision reference
        std::vector<double> ref(ROWS * NTOK);
        std::vector<float> deq(COLS);
        for (size_t r = 0; r < ROWS; ++r) {
            quant::dequantize_row(W.data(), deq.data(), r, COLS, t.t);
            for (size_t k = 0; k < NTOK; ++k) {
                double s = 0; for (size_t c = 0; c < COLS; ++c) s += (double)deq[c] * X[k * COLS + c];
                ref[k * ROWS + r] = s;
            }
        }
        double rmax = 0; for (double v : ref) rmax = std::max(rmax, std::fabs(v));

        auto relerr = [&](const std::vector<float>& y) {
            double e = 0; for (size_t i = 0; i < y.size(); ++i) e = std::max(e, std::fabs((double)y[i] - ref[i]));
            return e / (rmax + 1e-9);
        };
        std::vector<float> yv(ROWS * NTOK), yv1(ROWS * NTOK), ym(ROWS * NTOK);
        for (size_t k = 0; k < NTOK; ++k) cpu::mat_vec(W.data(), &X[k * COLS], &yv[k * ROWS], ROWS, COLS, t.t, 4);
        for (size_t k = 0; k < NTOK; ++k) cpu::mat_vec(W.data(), &X[k * COLS], &yv1[k * ROWS], ROWS, COLS, t.t, 1);
        cpu::mat_mat(W.data(), X.data(), ym.data(), ROWS, COLS, NTOK, t.t, 4);

        double e_v = relerr(yv), e_v1 = relerr(yv1), e_m = relerr(ym);
        double d_mv = 0; for (size_t i = 0; i < ym.size(); ++i) d_mv = std::max(d_mv, std::fabs((double)ym[i] - yv[i]));
        d_mv /= (rmax + 1e-9);
        bool ok = e_v < t.tol && e_v1 < t.tol && e_m < t.tol && d_mv < 1e-5 + t.tol * 0.0;
        // mat_mat is documented bit-identical to mat_vec; allow only float-rounding noise
        printf("%-5s %-12.3e %-12.3e %-12.3e %-10.1e %s\n", t.name, e_v, e_v1, e_m, d_mv, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
        if (out) { fwrite(yv.data(), 4, yv.size(), out); }
    }
    // elementwise ops vs scalar reference
    {
        std::vector<float> a(1003), b(1003);
        for (auto& v : a) v = U(rng); for (auto& v : b) v = U(rng);
        double ds = 0; for (size_t i = 0; i < a.size(); ++i) ds += (double)a[i] * b[i];
        double e_dot = std::fabs(cpu::dot(a.data(), b.data(), a.size()) - ds) / (std::fabs(ds) + 1);
        std::vector<float> s = a; cpu::softmax(s.data(), s.size());
        double mx = -1e30; for (float v : a) mx = std::max(mx, (double)v);
        double den = 0; for (float v : a) den += std::exp((double)v - mx);
        double e_sm = 0; for (size_t i = 0; i < a.size(); ++i) e_sm = std::max(e_sm, std::fabs(s[i] - std::exp((double)a[i] - mx) / den));
        std::vector<float> si = a; cpu::silu(si.data(), si.size());
        double e_si = 0; for (size_t i = 0; i < a.size(); ++i) { double v = a[i]; e_si = std::max(e_si, std::fabs(si[i] - v / (1.0 + std::exp(-v)))); }
        std::vector<float> w(1003, 1.f), rn(1003); cpu::rms_norm(a.data(), w.data(), rn.data(), a.size(), 1e-5f);
        double ss = 0; for (float v : a) ss += (double)v * v; double inv = 1.0 / std::sqrt(ss / a.size() + 1e-5);
        double e_rn = 0; for (size_t i = 0; i < a.size(); ++i) e_rn = std::max(e_rn, std::fabs(rn[i] - a[i] * inv));
        bool ok = e_dot < 1e-5 && e_sm < 1e-5 && e_si < 2e-4 && e_rn < 1e-4;
        printf("elementwise: dot %.1e softmax %.1e silu %.1e rms_norm %.1e %s\n", e_dot, e_sm, e_si, e_rn, ok ? "ok" : "FAIL");
        if (!ok) ++fails;
        if (out) { fwrite(s.data(), 4, 16, out); fwrite(si.data(), 4, 16, out); fwrite(rn.data(), 4, 16, out); }
    }
    if (out) fclose(out);
    printf(fails ? "RESULT: %d FAILURE(S)\n" : "RESULT: all ok\n", fails);
    return fails ? 1 : 0;
}
