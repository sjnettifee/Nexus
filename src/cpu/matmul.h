/*
 * nexus/src/cpu/matmul.h + matmul.cpp
 *
 * CPU matrix multiply kernels. This is the hottest code path in inference.
 * Every token generated calls mat_vec hundreds of times.
 *
 * Operations:
 *   mat_vec_f32     — dense float32, basic, always correct.
 *   mat_vec_q8_0    — quantized Q8_0 × float32 vector (no dequant intermediate)
 *   mat_vec_q4_k    — quantized Q4_K × float32 vector
 *   mat_vec_dispatch — routes to best kernel for the weight type
 *
 * AVX2 strategy:
 *   Process 8 floats at a time using 256-bit YMM registers.
 *   For quantized types: dequantize one block, dot product, accumulate.
 *   This avoids materializing the full dequantized matrix in RAM.
 *
 * Inference hot path: mat_vec
 *   Input:  weight matrix W [out_dim × in_dim], vector x [in_dim]
 *   Output: y [out_dim] = W × x
 *   This computes one output neuron per row. Parallelized across rows.
 */

#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "../gguf/quant.h"

// SIMD tier detection
// Tier 1 (best):  AVX2  — modern CPUs, 256-bit registers, 8 floats at once
// Tier 2 (good):  SSE4.1 — Sandy Bridge+ (2011+), 128-bit, 4 floats at once
//                  Enabled by -march=x86-64-v2 (the portable build target)
// Tier 3 (scalar): fallback for very old CPUs
#if defined(__AVX2__)
#  include <immintrin.h>
#  define NEXUS_AVX2 1
#  define NEXUS_SSE4 0
#elif defined(__SSE4_1__)
#  include <smmintrin.h>   // SSE4.1 — also pulls in SSSE3 and SSE3
#  define NEXUS_AVX2 0
#  define NEXUS_SSE4 1
#else
#  define NEXUS_AVX2 0
#  define NEXUS_SSE4 0
#endif

namespace nexus {
namespace cpu {

using quant::QuantType;

// ─── Core mat_vec: y = W × x ─────────────────────────────────────────────────
// W: [out_rows × in_cols], quantized
// x: [in_cols], float32
// y: [out_rows], float32 output
// n_threads: parallelize row computation across threads

void mat_vec(
    const void*  W,          // quantized weight matrix
    const float* x,          // input vector
    float*       y,          // output vector
    size_t       out_rows,   // number of output neurons
    size_t       in_cols,    // input dimension
    QuantType    qtype,      // weight quantization type
    int          n_threads = 1
);

// ─── Fused: mat_vec with bias add ─────────────────────────────────────────────
void mat_vec_bias(
    const void*  W,
    const float* x,
    const float* bias,
    float*       y,
    size_t       out_rows,
    size_t       in_cols,
    QuantType    qtype,
    int          n_threads = 1
);

// ─── Batched matmul (GEMM): Y = W · X for n_tokens columns at once ─────────────
// X is in_cols × n_tokens, Y is out_rows × n_tokens (column-major token blocks:
// token t's input at X + t*in_cols, output at Y + t*out_rows). Reads W once and
// reuses it across all tokens. Bit-identical to n_tokens separate mat_vec calls.
void mat_mat(
    const void*  W,
    const float* X,
    float*       Y,
    size_t       out_rows,
    size_t       in_cols,
    size_t       n_tokens,
    QuantType    qtype,
    int          n_threads = 1
);

void mat_mat_bias(
    const void*  W,
    const float* X,
    const float* bias,
    float*       Y,
    size_t       out_rows,
    size_t       in_cols,
    size_t       n_tokens,
    QuantType    qtype,
    int          n_threads = 1
);

// ─── Elementwise ops needed by transformer ────────────────────────────────────

// RMS normalization: y[i] = x[i] / rms(x) * weight[i]
void rms_norm(const float* x, const float* weight, float* y,
              size_t n, float eps = 1e-5f);

// SiLU activation: y[i] = x[i] * sigmoid(x[i])
void silu(float* x, size_t n);

// Elementwise multiply: z[i] = x[i] * y[i]
void mul(const float* x, const float* y, float* z, size_t n);

// Elementwise add (in-place): x[i] += y[i]
void add_inplace(float* x, const float* y, size_t n);

// Softmax over n elements
void softmax(float* x, size_t n);

// Scale: x[i] *= scale
void scale(float* x, float s, size_t n);

// Dot product: returns sum(x[i] * y[i])
float dot(const float* x, const float* y, size_t n);

} // namespace cpu
} // namespace nexus


// ─────────────────────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>
#include <cstdlib>   // getenv
#include <algorithm>

// ─── Q8K: quantized activation format for fast Q4_K dot products ─────────────
//
// The key insight: instead of dequanting Q4_K weights to float (256 floats → stack
// → dot), we quantize the input vector x to int8 ONCE before the row loop, then
// compute the dot product entirely in integer registers using maddubs + madd_epi16.
//
// Layout: one Q8KBlock covers 256 consecutive elements of x.
//   d      — scale: real_value ≈ d * q
//   qs     — int8 values in [-127, 127]
//   bsums  — sum of qs[k*32..k*32+31] for k=0..7 (used for min-term correction)
struct Q8KBlock {
    float   d;
    int8_t  qs[256];
    int32_t bsums[8];
};

// NQ4 inputs rounded to 8 bits, one per 64 weights, split into the even and odd positions so they line up with the low and
// high nibbles of the weight bytes. Used only when "8-bit output math" is on (NEXUS_CPU_INT8=1).
struct NQ8X { float d; int32_t isum; int8_t ev[32]; int8_t od[32]; };

struct MatVecJob {
    const NQ8X*      nq8_x = nullptr;  // non-null: NQ4 rows with 8-bit inputs
    const void*      W;
    const float*     x;
    float*           y;
    size_t           out_rows;
    size_t           in_cols;
    int              qtype;        // QuantType stored as int
    const Q8KBlock*  q8k_x;       // non-null for Q4_K: pre-quantized x (one block per 256 elems)
};

// Forward declarations needed by get_pool's dot function
namespace nexus { namespace cpu {
    using quant::QuantType;
    static float dot_row(const void* W, size_t row, size_t n_cols,
                          const float* x, QuantType qtype);
    static float dot_q4_k_q8k_row(const quant::BlockQ4_K* blocks,
                                    const Q8KBlock* xblocks, size_t n_blocks);
    static float dot_q5_k_q8k_row(const quant::BlockQ5_K* blocks,
                                    const Q8KBlock* xblocks, size_t n_blocks);
    static float dot_q6_k_q8k_row(const quant::BlockQ6_K* blocks,
                                    const Q8KBlock* xblocks, size_t n_blocks);
    static float dot_nq4_q8_row(const quant::BlockNQ_G64<4>* blocks, const NQ8X* xq, size_t n_blocks);
} }

// ─── Thread pool with generation counter ─────────────────────────────────────
//
// Root cause of the previous hang: a worker finishing job N-1 could wake up
// late, see ready_=true for job N, and steal rows from it. pending_ was set
// to n_workers_ but n_workers_+1 threads decremented it — pending_ went to
// -1, cv_done_ was never notified, main thread waited forever.
//
// Fix: monotonically increasing generation counter. Each worker records which
// generation it last served and only participates in NEW generations. pending_
// is atomic — the last worker to finish notifies the main thread. No worker
// can accidentally double-decrement for a job it wasn't assigned.

class ThreadPool {
public:
    // Chunk size is now computed adaptively in steal_rows()

    explicit ThreadPool(int n_workers) : n_workers_(n_workers) {
        workers_.reserve(n_workers);
        for (int i = 0; i < n_workers; ++i)
            workers_.emplace_back([this]{ worker_loop(); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(mu_);
            stop_ = true;
            cur_gen_++;
        }
        cv_work_.notify_all();
        for (auto& t : workers_) t.join();
    }

    void set_dot_fn(std::function<void(const MatVecJob&, size_t, size_t)> fn) {
        dot_fn_ = std::move(fn);
    }

    // ── Mode 0: mat_vec row dispatch ─────────────────────────────────────────
    void run(const MatVecJob& job) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            mode_         = 0;
            job_          = job;
            ++cur_gen_;
            pending_.store(n_workers_, std::memory_order_release);
            row_cursor_.store(0, std::memory_order_relaxed);
        }
        cv_work_.notify_all();
        steal_rows(job);
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait(lk, [&]{ return pending_.load(std::memory_order_acquire) == 0 || stop_; });
    }

    // ── Mode 1: generic indexed dispatch ─────────────────────────────────────
    // Calls fn(i) for i in [0, n_items). Each call is independent.
    // Main thread participates as a worker so all n_workers+1 threads share load.
    void run_indexed(int n_items, std::function<void(int)> fn) {
        if (n_items <= 0) return;
        if (n_workers_ <= 0 || n_items == 1) { fn(0); return; }
        {
            std::unique_lock<std::mutex> lk(mu_);
            mode_          = 1;
            item_fn_       = std::move(fn);
            item_count_    = n_items;
            ++cur_gen_;
            pending_.store(n_workers_, std::memory_order_release);
            item_cursor_.store(0, std::memory_order_relaxed);
        }
        cv_work_.notify_all();
        steal_items();
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [&]{ return pending_.load(std::memory_order_acquire) == 0 || stop_; });
        }
        item_fn_ = nullptr;  // release captured state
    }

    int size() const { return n_workers_; }

private:
    void steal_rows(const MatVecJob& job) {
        // Adaptive chunk size:
        //   - Large matrix (e.g. 14336-row FFN): use ~64 rows so threads stay
        //     busy without one thread hogging a huge block.
        //   - Small matrix (< 4*threads rows): use 1 row so all threads share.
        //   This eliminates the "one core 100%, rest idle" pattern caused by
        //   the old fixed ROW_CHUNK=8 leaving large unfinished tails.
        size_t n_active = (size_t)(n_workers_ + 1);
        size_t chunk = job.out_rows / (n_active * 8);
        if (chunk < 1)  chunk = 1;
        if (chunk > 32) chunk = 32;
        while (true) {
            size_t s = row_cursor_.fetch_add(chunk, std::memory_order_relaxed);
            if (s >= job.out_rows) break;
            size_t e = std::min(s + chunk, job.out_rows);
            dot_fn_(job, s, e);
        }
    }

    void steal_items() {
        while (true) {
            int i = item_cursor_.fetch_add(1, std::memory_order_relaxed);
            if (i >= item_count_) break;
            try { item_fn_(i); } catch (...) {}
        }
    }

    void worker_loop() {
        uint64_t my_gen = 0;
        while (true) {
            int mode;
            MatVecJob job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_work_.wait(lk, [&]{ return stop_ || cur_gen_ > my_gen; });
                if (stop_) return;
                my_gen = cur_gen_;
                mode   = mode_;
                job    = job_;
            }
            try {
                if (mode == 0) steal_rows(job);
                else           steal_items();
            } catch (...) {}
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_done_.notify_one();
            }
        }
    }

    const int                n_workers_;
    int                      mode_      = 0;  // 0=mat_vec rows, 1=indexed items
    MatVecJob                job_{};
    std::atomic<size_t>      row_cursor_{0};
    std::atomic<int>         pending_{0};
    uint64_t                 cur_gen_{0};
    bool                     stop_{false};
    std::mutex               mu_;
    std::condition_variable  cv_work_, cv_done_;
    std::function<void(const MatVecJob&, size_t, size_t)> dot_fn_;
    // Mode 1 state
    std::function<void(int)> item_fn_;
    std::atomic<int>         item_cursor_{0};
    int                      item_count_ = 0;
    std::vector<std::thread> workers_;
};



static ThreadPool* g_pool      = nullptr;
static int         g_pool_size = 0;
static const int   g_pool_ver  = 3;  // bump this to force pool recreation on kernel changes
static int         g_pool_built_ver = 0;

static ThreadPool* get_pool(int n_threads) {
    if (g_pool && g_pool_size == n_threads && g_pool_built_ver == g_pool_ver)
        return g_pool;
    delete g_pool;
    g_pool = new ThreadPool(n_threads);
    g_pool_size = n_threads;
    g_pool_built_ver = g_pool_ver;
    g_pool->set_dot_fn([](const MatVecJob& j, size_t s, size_t e) {
        const nexus::cpu::QuantType qt = (nexus::cpu::QuantType)j.qtype;
        const size_t bpr = j.in_cols / 256;
        if (j.nq8_x) {
            const size_t nb = (j.in_cols + 63) / 64;
            for (size_t row = s; row < e; ++row)
                j.y[row] = nexus::cpu::dot_nq4_q8_row((const nexus::quant::BlockNQ_G64<4>*)j.W + row * nb, j.nq8_x, nb);
            return;
        }
        for (size_t row = s; row < e; ++row) {
            if (j.q8k_x) {
                if (qt == nexus::cpu::QuantType::Q4_K) {
                    j.y[row] = nexus::cpu::dot_q4_k_q8k_row(
                        (const nexus::quant::BlockQ4_K*)j.W + row*bpr, j.q8k_x, bpr);
                } else if (qt == nexus::cpu::QuantType::Q5_K) {
                    j.y[row] = nexus::cpu::dot_q5_k_q8k_row(
                        (const nexus::quant::BlockQ5_K*)j.W + row*bpr, j.q8k_x, bpr);
                } else if (qt == nexus::cpu::QuantType::Q6_K) {
                    j.y[row] = nexus::cpu::dot_q6_k_q8k_row(
                        (const nexus::quant::BlockQ6_K*)j.W + row*bpr, j.q8k_x, bpr);
                } else {
                    j.y[row] = nexus::cpu::dot_row(j.W, row, j.in_cols, j.x, qt);
                }
            } else {
                j.y[row] = nexus::cpu::dot_row(j.W, row, j.in_cols, j.x, qt);
            }
        }
    });
    return g_pool;
}

// Public accessor — attention.h and other callers outside this TU can use the shared pool
static inline ThreadPool* get_shared_pool(int n_threads) { return get_pool(n_threads); }

namespace nexus {
namespace cpu {

// ─── float16 helper ───────────────────────────────────────────────────────────
static inline float f16(uint16_t h) { return quant::f16_to_f32(h); }


// NEXUS_MATMUL_NATIVE_NQ_G64 2026-05-25
// Direct dot kernels for Nexus native .nxm NQ*_G64 rows.
// This makes NQ a first-class CPU mat_vec type instead of forcing transformer.h
// to dequantize every row into a temporary float vector.
template<int BITS>
static inline uint32_t nx_nq_get_bits_local(const uint8_t* p, int idx) {
    const int bit = idx * BITS;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t v = ((uint32_t)p[byte]) >> shift;
    if (shift + BITS > 8) v |= ((uint32_t)p[byte + 1]) << (8 - shift);
    return v & ((1u << BITS) - 1u);
}


#if NEXUS_AVX2
static inline float dot_nq4_g64_row_avx2(const quant::BlockNQ_G64<4>* blocks,
                                         const float* x,
                                         size_t n_cols) {
    float acc = 0.0f;
    size_t c = 0;
    const size_t n_blocks = (n_cols + 63) / 64;
    alignas(32) int8_t qv[64];
    for (size_t b = 0; b < n_blocks; ++b) {
        const float d = quant::f16_to_f32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        const size_t n = std::min<size_t>(64, n_cols - c);
        for (int j = 0; j < 32; ++j) {
            const uint8_t byte = qs[j];
            qv[2*j + 0] = (int8_t)((int)(byte & 0x0F) - 8);
            qv[2*j + 1] = (int8_t)((int)(byte >> 4)   - 8);
        }
        __m256 vacc = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qv + i));
            __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            __m256 xf = _mm256_loadu_ps(x + c + i);
            vacc = _mm256_fmadd_ps(qf, xf, vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float block_acc = _mm_cvtss_f32(lo);
        for (; i < n; ++i) block_acc += (float)qv[i] * x[c + i];
        acc += d * block_acc;
        c += n;
    }
    return acc;
}
#endif

template<int BITS>
static inline float dot_nq_g64_row(const quant::BlockNQ_G64<BITS>* blocks,
                                   const float* x,
                                   size_t n_cols) {
    const int off = 1 << (BITS - 1);
    float acc = 0.0f;
    size_t c = 0;
    const size_t n_blocks = (n_cols + 63) / 64;
    for (size_t b = 0; b < n_blocks; ++b) {
        const float d = quant::f16_to_f32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        const size_t n = std::min<size_t>(64, n_cols - c);
#if NEXUS_AVX2
        // Keep this scalar per 64-pack for now. It avoids the huge row-dequant
        // allocation and preserves exact native packing; SIMD can be added later.
#endif
        for (size_t i = 0; i < n; ++i) {
            const int q = (int)nx_nq_get_bits_local<BITS>(qs, (int)i) - off;
            acc += (d * (float)q) * x[c + i];
        }
        c += n;
    }
    return acc;
}

// ─── Scalar dot: one row of W dotted with x ───────────────────────────────────

// Float32 row × float32 vector — reference, always correct
static float dot_f32(const float* row, const float* x, size_t n) {
    float acc = 0.0f;
#if NEXUS_AVX2
    __m256 vacc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vr = _mm256_loadu_ps(row + i);
        __m256 vx = _mm256_loadu_ps(x + i);
        vacc = _mm256_fmadd_ps(vr, vx, vacc);
    }
    // Horizontal sum of vacc
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
    // Remainder
    for (; i < n; ++i) acc += row[i] * x[i];
#elif NEXUS_SSE4
    __m128 vacc = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        vacc = _mm_add_ps(vacc, _mm_mul_ps(_mm_loadu_ps(row+i), _mm_loadu_ps(x+i)));
    vacc = _mm_hadd_ps(vacc, vacc);
    vacc = _mm_hadd_ps(vacc, vacc);
    acc = _mm_cvtss_f32(vacc);
    for (; i < n; ++i) acc += row[i] * x[i];
#else
    for (size_t i = 0; i < n; ++i) acc += row[i] * x[i];
#endif
    return acc;
}

// Q8_0 row × float32 vector: dot without materializing full float row
// Processes 32 elements at a time (one block) — much better cache behavior
static float dot_q8_0_row(const quant::BlockQ8_0* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
#if NEXUS_AVX2
    __m256 vacc = _mm256_setzero_ps();
    for (size_t b = 0; b < n_blocks; ++b) {
        const float d = f16(blocks[b].d);
        const __m256 vd = _mm256_set1_ps(d);
        const float* xb = x + b * 32;

        for (int i = 0; i < 4; ++i) {
            // Load 8 int8 values, sign-extend to int32, convert to float
            const int8_t* qs = blocks[b].qs + i * 8;
            __m128i vi8 = _mm_loadl_epi64((const __m128i*)qs);
            __m256i vi32 = _mm256_cvtepi8_epi32(vi8);
            __m256 vf = _mm256_cvtepi32_ps(vi32);
            __m256 vx = _mm256_loadu_ps(xb + i * 8);
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(vd, vf), vx, vacc);
        }
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
#elif NEXUS_SSE4
    // SSE4.1: sign-extend 4 int8 → int32 → float, process 4 at a time
    __m128 vacc = _mm_setzero_ps();
    for (size_t b = 0; b < n_blocks; ++b) {
        const __m128 vd = _mm_set1_ps(f16(blocks[b].d));
        const float* xb = x + b * 32;
        for (int i = 0; i < 32; i += 4) {
            __m128i vi = _mm_cvtsi32_si128(*(const int32_t*)(blocks[b].qs + i));
            __m128 vf  = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(vi));
            vacc = _mm_add_ps(vacc, _mm_mul_ps(_mm_mul_ps(vd, vf), _mm_loadu_ps(xb + i)));
        }
    }
    vacc = _mm_hadd_ps(vacc, vacc);
    vacc = _mm_hadd_ps(vacc, vacc);
    acc = _mm_cvtss_f32(vacc);
#else
    // Scalar fallback
    float tmp[32];
    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q8_0(blocks + b, tmp);
        for (int i = 0; i < 32; ++i) acc += tmp[i] * x[b*32+i];
    }
#endif
    return acc;
}

// Q4_0 row × float32 vector
static float dot_q4_0_row(const quant::BlockQ4_0* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
    float tmp[32];
    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q4_0(blocks + b, tmp);
        acc += dot_f32(tmp, x + b*32, 32);
    }
    return acc;
}

// Q5_0 row × float32 vector
static float dot_q5_0_row(const quant::BlockQ5_0* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
    float tmp[32];

    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q5_0(blocks + b, tmp);
        acc += dot_f32(tmp, x + b * 32, 32);
    }

    return acc;
}

// ─── Q8K quantizer ────────────────────────────────────────────────────────────
// Quantize one 256-element block of float32 activations to int8.
// Computes per-sub-block sums (bsums) for the Q4_K min-term correction.
static void quantize_block_q8k(const float* x, Q8KBlock* out) {
    // Find max absolute value across all 256 elements
    float amax = 0.0f;
#if NEXUS_AVX2
    __m256 vmax = _mm256_setzero_ps();
    const __m256 vsign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // abs mask
    for (int i = 0; i < 256; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, vsign));
    }
    // Horizontal max of vmax
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    amax = _mm_cvtss_f32(lo);
#elif NEXUS_SSE4
    __m128 vmax = _mm_setzero_ps();
    const __m128i vmask_abs = _mm_set1_epi32(0x7FFFFFFF);
    for (int i = 0; i < 256; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        vmax = _mm_max_ps(vmax, _mm_and_ps(v, _mm_castsi128_ps(vmask_abs)));
    }
    vmax = _mm_max_ps(vmax, _mm_movehl_ps(vmax, vmax));
    vmax = _mm_max_ss(vmax, _mm_shuffle_ps(vmax, vmax, 1));
    amax = _mm_cvtss_f32(vmax);
#else
    for (int i = 0; i < 256; ++i) {
        float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
#endif

    if (amax < 1e-30f) {
        out->d = 0.0f;
        memset(out->qs,    0, 256);
        memset(out->bsums, 0, sizeof(out->bsums));
        return;
    }

    out->d = amax / 127.0f;
    const float scale = 127.0f / amax;

    // Quantize all 256 elements and compute per-32-group sums.
    // CLAMP to [-127, 127] — floating point rounding can push the max element
    // to 128.0f which is UB when cast to int8_t and silently wraps to -128,
    // corrupting the bsums min-correction term and producing garbage logits.
    for (int k = 0; k < 8; ++k) {
        const float* xk = x + k * 32;
        int8_t*      qk = out->qs + k * 32;
        int32_t      bs = 0;
        for (int i = 0; i < 32; ++i) {
            int v = (int)roundf(xk[i] * scale);
            // Clamp to safe int8 range (avoids UB and preserves bsums integrity)
            if (v >  127) v =  127;
            if (v < -127) v = -127;
            qk[i] = (int8_t)v;
            bs   += v;
        }
        out->bsums[k] = bs;
    }
}

// ─── Horizontal int32 sum of __m256i ─────────────────────────────────────────
#if NEXUS_AVX2
static inline int32_t hsum_i32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif

// ─── SSE4.1 helpers ───────────────────────────────────────────────────────────
#if NEXUS_SSE4
// Horizontal sum of 4 int32 in a __m128i
static inline int32_t hsum_i32_128(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}
// Sum of 16 signed int8 values stored in a __m128i
static inline int32_t sum_i8_16(__m128i v) {
    __m128i lo8 = _mm_cvtepi8_epi16(v);                          // lower 8 → 8 int16
    __m128i hi8 = _mm_cvtepi8_epi16(_mm_srli_si128(v, 8));       // upper 8 → 8 int16
    __m128i s16 = _mm_add_epi16(lo8, hi8);                       // 8 int16
    __m128i s32 = _mm_madd_epi16(s16, _mm_set1_epi16(1));        // 4 int32
    return hsum_i32_128(s32);
}
// Integer dot: 16 uint8 weights × 16 int8 activations → int32 sum (SSSE3)
static inline int32_t dot16_u8s8(__m128i vq, __m128i vx) {
    return hsum_i32_128(_mm_madd_epi16(_mm_maddubs_epi16(vq, vx), _mm_set1_epi16(1)));
}
// Reconstruct 16 raw-6-bit values (lo nibble path) for Q6_K
static inline __m128i q6_mk_lo(__m128i vql, int shift, __m128i vqh) {
    __m128i lo4 = _mm_and_si128(vql, _mm_set1_epi8(0x0F));
    __m128i hb  = _mm_and_si128(_mm_srli_epi16(vqh, shift), _mm_set1_epi8(3));
    hb = _mm_and_si128(_mm_slli_epi16(hb, 4), _mm_set1_epi8(0x30));
    return _mm_add_epi8(lo4, hb);
}
// Reconstruct 16 raw-6-bit values (hi nibble path) for Q6_K
static inline __m128i q6_mk_hi(__m128i vql, int shift, __m128i vqh) {
    __m128i hi4 = _mm_and_si128(_mm_srli_epi16(vql, 4), _mm_set1_epi8(0x0F));
    __m128i hb  = _mm_and_si128(_mm_srli_epi16(vqh, shift), _mm_set1_epi8(3));
    hb = _mm_and_si128(_mm_slli_epi16(hb, 4), _mm_set1_epi8(0x30));
    return _mm_add_epi8(hi4, hb);
}
#endif

// ─── Q4_K × Q8K integer dot product ─────────────────────────────────────────
//
// Core idea: instead of dequanting 256 nibbles → 256 floats → dot,
// we do the entire dot in integer arithmetic using AVX2:
//
//   For each group j (0–3), 32 packed nibble pairs → split lo/hi:
//     dot_lo = maddubs(lo_nibbles[uint8], q8_lo[int8]) → 16 int16 → sum → int32
//     dot_hi = maddubs(hi_nibbles[uint8], q8_hi[int8]) → 16 int16 → sum → int32
//
//   Scale: real_dot += d * sc[j*2] * dot_lo - dmin * mn[j*2] * bsum_lo
//                    + d * sc[j*2+1] * dot_hi - dmin * mn[j*2+1] * bsum_hi
//
//   maddubs: unsigned_a[i]*signed_b[i] + unsigned_a[i+1]*signed_b[i+1] → int16
//   No intermediate 256-float buffer. ~2× faster than dequant path.
static float dot_q4_k_q8k_row(const quant::BlockQ4_K* blocks,
                                const Q8KBlock* xblocks, size_t n_blocks) {
    float acc = 0.0f;

    for (size_t b = 0; b < n_blocks; ++b) {
        const quant::BlockQ4_K& wb = blocks[b];
        const Q8KBlock&         xb = xblocks[b];

        const float d    = quant::f16_to_f32(wb.d);
        const float dmin = quant::f16_to_f32(wb.dmin);

        // Unpack 8 sub-block scales and mins (same bit-packing as dequant_q4_k)
        float sc[8], mn[8];
        const uint8_t* s = wb.scales;
        for (int i = 0; i < 4; ++i) {
            sc[i]   = (float)( s[i]   & 0x3F);
            mn[i]   = (float)( s[i+4] & 0x3F);
            sc[i+4] = (float)((s[i+8] & 0x0F) | ((s[i]   >> 6) << 4));
            mn[i+4] = (float)((s[i+8] >> 4)   | ((s[i+4] >> 6) << 4));
        }

#if NEXUS_AVX2
        const __m256i vmask  = _mm256_set1_epi8(0x0F);
        const __m256i ones16 = _mm256_set1_epi16(1);

        float sum_q = 0.0f;   // accumulates sc[s] * integer_dot[s]
        float sum_m = 0.0f;   // accumulates mn[s] * bsums[s]

        for (int j = 0; j < 4; ++j) {
            // Load 32 packed bytes — each byte holds two 4-bit nibbles
            const __m256i vqs = _mm256_loadu_si256(
                (const __m256i*)(wb.qs + j * 32));

            // Extract lo nibbles (bits 3:0 of each byte) — 32 values in [0,15]
            const __m256i vlo = _mm256_and_si256(vqs, vmask);

            // Extract hi nibbles (bits 7:4) via logical right-shift then mask.
            // _mm256_srli_epi16 shifts 16-bit lanes: for bytes [lo|hi] packed
            // as word [hi<<8|lo], shift-by-4 yields [hi>>4 | hi<<4|lo>>4].
            // Low byte after mask = lo>>4 = original hi nibble. ✓
            const __m256i vhi = _mm256_and_si256(
                _mm256_srli_epi16(vqs, 4), vmask);

            // Q8 activations: lo sub-block uses x[j*64..j*64+31],
            //                 hi sub-block uses x[j*64+32..j*64+63]
            const __m256i vx_lo = _mm256_loadu_si256(
                (const __m256i*)(xb.qs + j * 64));
            const __m256i vx_hi = _mm256_loadu_si256(
                (const __m256i*)(xb.qs + j * 64 + 32));

            // Integer dot: uint8 nibble × int8 activation
            // maddubs: pairs of (uint8 × int8) → int16, summed pairwise
            // madd with ones16: pairs of int16 → int32, summed pairwise
            const __m256i dl = _mm256_madd_epi16(
                _mm256_maddubs_epi16(vlo, vx_lo), ones16);
            const __m256i dh = _mm256_madd_epi16(
                _mm256_maddubs_epi16(vhi, vx_hi), ones16);

            sum_q += sc[j*2]   * (float)hsum_i32(dl);
            sum_q += sc[j*2+1] * (float)hsum_i32(dh);

            // Min correction: dmin * mn[s] * sum(x_values for sub-block s)
            // bsums[j*2] = sum of xb.qs[j*64 .. j*64+31]
            // bsums[j*2+1] = sum of xb.qs[j*64+32 .. j*64+63]
            sum_m += mn[j*2]   * (float)xb.bsums[j*2];
            sum_m += mn[j*2+1] * (float)xb.bsums[j*2+1];
        }

        // xb.d converts int8 → float; d/dmin are the Q4_K super-block scales
        acc += xb.d * (d * sum_q - dmin * sum_m);

#elif NEXUS_SSE4
        // SSE4.1/SSSE3: process 16 elements at a time (half of AVX2's 32)
        const __m128i vmask4  = _mm_set1_epi8(0x0F);
        float sum_q = 0.0f, sum_m = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const uint8_t* qs_ptr = wb.qs + j * 32;
            for (int h = 0; h < 2; ++h) {  // two halves of 16 elements
                __m128i vqs = _mm_loadu_si128((const __m128i*)(qs_ptr + h*16));
                __m128i vlo = _mm_and_si128(vqs, vmask4);
                __m128i vhi = _mm_and_si128(_mm_srli_epi16(vqs, 4), vmask4);
                // activations: lo sub-block at j*64+h*16, hi sub-block at j*64+32+h*16
                __m128i vx_lo = _mm_loadu_si128((const __m128i*)(xb.qs + j*64 + h*16));
                __m128i vx_hi = _mm_loadu_si128((const __m128i*)(xb.qs + j*64 + 32 + h*16));
                sum_q += sc[j*2]   * (float)dot16_u8s8(vlo, vx_lo);
                sum_q += sc[j*2+1] * (float)dot16_u8s8(vhi, vx_hi);
            }
            sum_m += mn[j*2]   * (float)xb.bsums[j*2];
            sum_m += mn[j*2+1] * (float)xb.bsums[j*2+1];
        }
        acc += xb.d * (d * sum_q - dmin * sum_m);
#else
        // Scalar fallback: dequant + dot (same as old path)
        float tmp[256];
        quant::dequant_q4_k(&wb, tmp);
        for (int i = 0; i < 256; ++i) acc += tmp[i] * xb.d * xb.qs[i];
#endif
    }
    return acc;
}

// ─── Old dequant-to-float path kept as fallback for non-Q4_K types ───────────
// Q4_K: dequant to stack buffer then AVX2 dot — REPLACED by Q8K path above.
// Kept only because dot_row still needs it for the single-threaded n_threads=1
// case when no Q8K buffer has been allocated (shouldn't happen in practice,
// but keeps the scalar fallback path alive).
static float dot_q4_k_row(const quant::BlockQ4_K* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
    float tmp[256];
    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q4_k(blocks + b, tmp);
        acc += dot_f32(tmp, x + b*256, 256);
    }
    return acc;
}

// Q5_K: dequant to stack buffer then AVX2 dot
static float dot_q5_k_row(const quant::BlockQ5_K* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
    float tmp[256];
    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q5_k(blocks + b, tmp);
        acc += dot_f32(tmp, x + b*256, 256);
    }
    return acc;
}

// Q6_K: dequant to stack buffer then AVX2 dot
static float dot_q6_k_row(const quant::BlockQ6_K* blocks, const float* x,
                            size_t n_blocks) {
    float acc = 0.0f;
    float tmp[256];
    for (size_t b = 0; b < n_blocks; ++b) {
        quant::dequant_q6_k(blocks + b, tmp);
        acc += dot_f32(tmp, x + b*256, 256);
    }
    return acc;
}
// ─── Q5_K × Q8K integer dot product ─────────────────────────────────────────
//
// Q5_K is identical to Q4_K in group/scale/min structure. The only difference:
// each nibble gets a 5th high bit from qh[l], stored per-element-index across
// all 4 groups. Values are 0-31, still fit in uint8 → maddubs works fine.
//
// For group j:
//   lo elements: 5th bit = (qh[l] >> (2*j)) & 1  → add 16 if set
//   hi elements: 5th bit = (qh[l] >> (2*j+1)) & 1 → add 16 if set
static float dot_q5_k_q8k_row(const quant::BlockQ5_K* blocks,
                                const Q8KBlock* xblocks, size_t n_blocks) {
    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const quant::BlockQ5_K& wb = blocks[b];
        const Q8KBlock&         xb = xblocks[b];

        const float d    = quant::f16_to_f32(wb.d);
        const float dmin = quant::f16_to_f32(wb.dmin);

        // Same scale/min unpacking as Q4_K
        float sc[8], mn[8];
        const uint8_t* s = wb.scales;
        for (int i = 0; i < 4; ++i) {
            sc[i]   = (float)( s[i]   & 0x3F);
            mn[i]   = (float)( s[i+4] & 0x3F);
            sc[i+4] = (float)((s[i+8] & 0x0F) | ((s[i]   >> 6) << 4));
            mn[i+4] = (float)((s[i+8] >> 4)   | ((s[i+4] >> 6) << 4));
        }

#if NEXUS_AVX2
        const __m256i vmask4  = _mm256_set1_epi8(0x0F);
        const __m256i v16     = _mm256_set1_epi8(16);
        const __m256i ones16  = _mm256_set1_epi16(1);

        float sum_q = 0.0f;
        float sum_m = 0.0f;

        for (int j = 0; j < 4; ++j) {
            // Load 32 packed nibble bytes for this group
            const __m256i vqs = _mm256_loadu_si256(
                (const __m256i*)(wb.qs + j * 32));

            // Extract lo nibbles (bits 3:0) and hi nibbles (bits 7:4)
            const __m256i vlo4 = _mm256_and_si256(vqs, vmask4);
            const __m256i vhi4 = _mm256_and_si256(_mm256_srli_epi16(vqs, 4), vmask4);

            // Load 32 qh bytes — each byte has bits for all 4 groups
            // For group j: lo 5th bit = bit (2*j), hi 5th bit = bit (2*j+1)
            const __m256i vqh = _mm256_loadu_si256((const __m256i*)wb.qh);

            // Isolate the relevant bits for this group, broadcast to 0/1 bytes,
            // then multiply by 16 to get the 5th-bit addend.
            // _mm256_srli_epi16 shifts 16-bit lanes — for byte pairs, shifting by
            // (2*j) moves the target bit into bit-0 of the low byte.
            const int shift_lo = 2 * j;
            const int shift_hi = 2 * j + 1;

            // Shift so target bit lands at bit 0, mask to byte, multiply by 16
            __m256i vlo_hb = _mm256_and_si256(
                _mm256_srli_epi16(vqh, shift_lo), _mm256_set1_epi8(1));
            __m256i vhi_hb = _mm256_and_si256(
                _mm256_srli_epi16(vqh, shift_hi), _mm256_set1_epi8(1));
            vlo_hb = _mm256_mullo_epi16(vlo_hb, _mm256_set1_epi16(16));
            vhi_hb = _mm256_mullo_epi16(vhi_hb, _mm256_set1_epi16(16));
            // The above sets byte-pairs; we only want the low byte of each pair.
            // Re-mask to get clean byte values 0 or 16.
            vlo_hb = _mm256_and_si256(vlo_hb, _mm256_set1_epi8(16));
            vhi_hb = _mm256_and_si256(vhi_hb, _mm256_set1_epi8(16));

            // 5-bit values: nibble | high_bit_addend, range [0,31] → safe uint8
            const __m256i vlo = _mm256_add_epi8(vlo4, vlo_hb);
            const __m256i vhi = _mm256_add_epi8(vhi4, vhi_hb);

            // Load Q8 activations
            const __m256i vx_lo = _mm256_loadu_si256(
                (const __m256i*)(xb.qs + j * 64));
            const __m256i vx_hi = _mm256_loadu_si256(
                (const __m256i*)(xb.qs + j * 64 + 32));

            // Integer dot: uint8 weights × int8 activations → int32 sums
            const __m256i dl = _mm256_madd_epi16(
                _mm256_maddubs_epi16(vlo, vx_lo), ones16);
            const __m256i dh = _mm256_madd_epi16(
                _mm256_maddubs_epi16(vhi, vx_hi), ones16);

            sum_q += sc[j*2]   * (float)hsum_i32(dl);
            sum_q += sc[j*2+1] * (float)hsum_i32(dh);
            sum_m += mn[j*2]   * (float)xb.bsums[j*2];
            sum_m += mn[j*2+1] * (float)xb.bsums[j*2+1];
        }
        acc += xb.d * (d * sum_q - dmin * sum_m);

#elif NEXUS_SSE4
        // SSE4.1/SSSE3: 16 elements at a time with 5th-bit reconstruction
        const __m128i vmask4 = _mm_set1_epi8(0x0F);
        float sum_q = 0.0f, sum_m = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const int shift_lo = 2 * j;
            const int shift_hi = 2 * j + 1;
            const uint8_t* qs_ptr = wb.qs + j * 32;
            for (int h = 0; h < 2; ++h) {
                __m128i vqs  = _mm_loadu_si128((const __m128i*)(qs_ptr + h*16));
                __m128i vqh  = _mm_loadu_si128((const __m128i*)(wb.qh + h*16));
                // Extract 4-bit nibbles
                __m128i vlo4 = _mm_and_si128(vqs, vmask4);
                __m128i vhi4 = _mm_and_si128(_mm_srli_epi16(vqs, 4), vmask4);
                // Extract 5th bit addend (0 or 16) for lo and hi sub-groups
                __m128i vlo_hb = _mm_and_si128(_mm_srli_epi16(vqh, shift_lo), _mm_set1_epi8(1));
                vlo_hb = _mm_and_si128(_mm_slli_epi16(vlo_hb, 4), _mm_set1_epi8(0x10));
                __m128i vhi_hb = _mm_and_si128(_mm_srli_epi16(vqh, shift_hi), _mm_set1_epi8(1));
                vhi_hb = _mm_and_si128(_mm_slli_epi16(vhi_hb, 4), _mm_set1_epi8(0x10));
                // 5-bit values [0,31]
                __m128i vlo = _mm_add_epi8(vlo4, vlo_hb);
                __m128i vhi = _mm_add_epi8(vhi4, vhi_hb);
                __m128i vx_lo = _mm_loadu_si128((const __m128i*)(xb.qs + j*64 + h*16));
                __m128i vx_hi = _mm_loadu_si128((const __m128i*)(xb.qs + j*64 + 32 + h*16));
                sum_q += sc[j*2]   * (float)dot16_u8s8(vlo, vx_lo);
                sum_q += sc[j*2+1] * (float)dot16_u8s8(vhi, vx_hi);
            }
            sum_m += mn[j*2]   * (float)xb.bsums[j*2];
            sum_m += mn[j*2+1] * (float)xb.bsums[j*2+1];
        }
        acc += xb.d * (d * sum_q - dmin * sum_m);
#else
        // Scalar fallback
        float tmp[256];
        quant::dequant_q5_k(&wb, tmp);
        for (int i = 0; i < 256; ++i) acc += tmp[i] * xb.d * xb.qs[i];
#endif
    }
    return acc;
}

// ─── Q6_K × Q8K integer dot product ─────────────────────────────────────────
//
// Q6_K: single super-block scale d, 16 int8 sub-scales (16-element groups).
// Raw 6-bit values in [0,63]; signed_q = raw_q - 32, range [-32,31].
//
// Unsigned trick — per 32-element sub-group g, with scales sc[g*2]/sc[g*2+1]:
//   dot  = sc[g*2]  * Σ_{l< 16} signed_q[l]*x[l]  + sc[g*2+1] * Σ_{l>=16} signed_q[l]*x[l]
//        = sc[g*2]  * (Σ raw_q[l]*x[l]  - 32*Σ x[l])_{l<16}
//        + sc[g*2+1]* (Σ raw_q[l]*x[l]  - 32*Σ x[l])_{l>=16}
//
// We compute dot_lo, dot_hi (integer dot of 16 elements each) via maddubs,
// and xsum_lo, xsum_hi (exact 16-element x-sums) via cvtepi8_epi16 + madd.
// The -32 correction is exact — no approximation.
static float dot_q6_k_q8k_row(const quant::BlockQ6_K* blocks,
                                const Q8KBlock* xblocks, size_t n_blocks) {
    float acc = 0.0f;
    for (size_t b = 0; b < n_blocks; ++b) {
        const quant::BlockQ6_K& wb = blocks[b];
        const Q8KBlock&         xb = xblocks[b];
        const float d = quant::f16_to_f32(wb.d);

#if NEXUS_AVX2
        const __m256i vmask4  = _mm256_set1_epi8(0x0F);
        const __m256i vmask2  = _mm256_set1_epi8(3);
        const __m256i ones16  = _mm256_set1_epi16(1);
        const __m256i ones16v = _mm256_set1_epi16(1);  // alias for xsum use

        // Helper: exact sum of 16 int8 values stored in a __m128i
        auto sum16_i8 = [&](__m128i v) -> int32_t {
            __m256i v16 = _mm256_cvtepi8_epi16(v);        // 16 int16
            __m256i v32 = _mm256_madd_epi16(v16, ones16v); // 8 int32 (sum adjacent pairs)
            __m128i lo  = _mm256_castsi256_si128(v32);
            __m128i hi  = _mm256_extracti128_si256(v32, 1);
            __m128i s   = _mm_add_epi32(lo, hi);           // 4 int32
            s = _mm_hadd_epi32(s, s);                      // 2 int32
            s = _mm_hadd_epi32(s, s);                      // 1 int32
            return _mm_cvtsi128_si32(s);
        };

        // Helper: split int32 dot of 32 (uint8 q) × (int8 x) into two 16-element halves
        auto dot32_split = [&](__m256i vq, __m256i vx,
                                int32_t& dot_lo, int32_t& dot_hi) {
            __m256i prod = _mm256_madd_epi16(_mm256_maddubs_epi16(vq, vx), ones16);
            // Low 128-bit half = elements 0-15, high = elements 16-31
            __m128i lo128 = _mm256_castsi256_si128(prod);
            __m128i hi128 = _mm256_extracti128_si256(prod, 1);
            lo128 = _mm_hadd_epi32(lo128, lo128); lo128 = _mm_hadd_epi32(lo128, lo128);
            hi128 = _mm_hadd_epi32(hi128, hi128); hi128 = _mm_hadd_epi32(hi128, hi128);
            dot_lo = _mm_cvtsi128_si32(lo128);
            dot_hi = _mm_cvtsi128_si32(hi128);
        };

        float sum = 0.0f;

        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql   = wb.ql     + n * 64;
            const uint8_t* qh   = wb.qh     + n * 32;
            const int8_t*  sc   = wb.scales  + n * 8;
            const int      xbase = n * 128;

            const __m256i vql_lo = _mm256_loadu_si256((const __m256i*)(ql));
            const __m256i vql_hi = _mm256_loadu_si256((const __m256i*)(ql + 32));
            const __m256i vqh_v  = _mm256_loadu_si256((const __m256i*)(qh));

            // Extract 6-bit raw [0,63] for each stride-32 sub-group:
            //   q1 = (ql_lo & 0xF) | ((qh>>0)&3)<<4   positions 0..31
            //   q2 = (ql_hi & 0xF) | ((qh>>2)&3)<<4   positions 32..63
            //   q3 = (ql_lo >> 4)  | ((qh>>4)&3)<<4   positions 64..95
            //   q4 = (ql_hi >> 4)  | ((qh>>6)&3)<<4   positions 96..127
            auto mk6lo = [&](__m256i vql, int shift) -> __m256i {
                __m256i lo4 = _mm256_and_si256(vql, vmask4);
                __m256i hb  = _mm256_and_si256(_mm256_srli_epi16(vqh_v, shift), vmask2);
                hb = _mm256_and_si256(_mm256_slli_epi16(hb, 4), _mm256_set1_epi8(0x30));
                return _mm256_add_epi8(lo4, hb);
            };
            auto mk6hi = [&](__m256i vql, int shift) -> __m256i {
                __m256i hi4 = _mm256_and_si256(_mm256_srli_epi16(vql, 4), vmask4);
                __m256i hb  = _mm256_and_si256(_mm256_srli_epi16(vqh_v, shift), vmask2);
                hb = _mm256_and_si256(_mm256_slli_epi16(hb, 4), _mm256_set1_epi8(0x30));
                return _mm256_add_epi8(hi4, hb);
            };

            const __m256i vq[4] = {
                mk6lo(vql_lo, 0), mk6lo(vql_hi, 2),
                mk6hi(vql_lo, 4), mk6hi(vql_hi, 6),
            };
            const __m256i vx[4] = {
                _mm256_loadu_si256((const __m256i*)(xb.qs + xbase +   0)),
                _mm256_loadu_si256((const __m256i*)(xb.qs + xbase +  32)),
                _mm256_loadu_si256((const __m256i*)(xb.qs + xbase +  64)),
                _mm256_loadu_si256((const __m256i*)(xb.qs + xbase +  96)),
            };

            for (int g = 0; g < 4; ++g) {
                int32_t dot_lo, dot_hi;
                dot32_split(vq[g], vx[g], dot_lo, dot_hi);

                // Exact x-sums per 16-element half
                int32_t xsum_lo = sum16_i8(_mm256_castsi256_si128(   vx[g]));
                int32_t xsum_hi = sum16_i8(_mm256_extracti128_si256(vx[g], 1));

                float sca = (float)sc[g*2];
                float scb = (float)sc[g*2 + 1];
                sum += sca * (float)(dot_lo - 32 * xsum_lo)
                     + scb * (float)(dot_hi - 32 * xsum_hi);
            }
        }

        acc += xb.d * d * sum;

#elif NEXUS_SSE4
        // SSE4.1/SSSE3: 16 elements at a time, -32 correction for signed values
        float sum = 0.0f;
        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql   = wb.ql + n * 64;
            const uint8_t* qh   = wb.qh + n * 32;
            const int8_t*  sc   = wb.scales + n * 8;
            const int      xbase = n * 128;
            // Each n-half has 4 groups of 32 elements = 128 elements
            // Groups 0,1 use lo nibbles; groups 2,3 use hi nibbles
            // qh provides 2 high bits per position for all 4 groups
            for (int g = 0; g < 4; ++g) {
                int32_t dot_lo = 0, dot_hi = 0;
                int32_t xsum_lo = 0, xsum_hi = 0;
                const bool use_hi_nib = (g >= 2);
                const int  qh_shift   = g * 2;  // bits 2*g and 2*g+1 in qh bytes
                // ql source: groups 0,2 use ql[0..63] lo/hi; groups 1,3 use ql[32..63]
                const uint8_t* ql_src = (g == 1 || g == 3) ? ql + 32 : ql;
                for (int h = 0; h < 2; ++h) {  // two 16-element halves per group
                    __m128i vql = _mm_loadu_si128((const __m128i*)(ql_src + h*16));
                    __m128i vqh = _mm_loadu_si128((const __m128i*)(qh + h*16));
                    __m128i vq  = use_hi_nib ? q6_mk_hi(vql, qh_shift, vqh)
                                             : q6_mk_lo(vql, qh_shift, vqh);
                    // activations for this half: g*32 + h*16 within the n-half
                    __m128i vx = _mm_loadu_si128((const __m128i*)(xb.qs + xbase + g*32 + h*16));
                    if (h == 0) { dot_lo  += dot16_u8s8(vq, vx); xsum_lo  += sum_i8_16(vx); }
                    else        { dot_hi  += dot16_u8s8(vq, vx); xsum_hi  += sum_i8_16(vx); }
                }
                sum += (float)sc[g*2]   * (float)(dot_lo - 32 * xsum_lo)
                     + (float)sc[g*2+1] * (float)(dot_hi - 32 * xsum_hi);
            }
        }
        acc += xb.d * d * sum;
#else
        float tmp[256];
        quant::dequant_q6_k(&wb, tmp);
        for (int i = 0; i < 256; ++i) acc += tmp[i] * xb.d * xb.qs[i];
#endif
    }
    return acc;
}

static float dot_k_tail_safe_row(const void* W,
                                 size_t row,
                                 size_t n_cols,
                                 const float* x,
                                 QuantType qtype) {
    const size_t padded_cols = ((n_cols + 255) / 256) * 256;
    const size_t bpr = padded_cols / 256;

    std::vector<float> tmp(padded_cols, 0.0f);

    switch (qtype) {
    case QuantType::Q4_K: {
        const auto* blocks =
            (const quant::BlockQ4_K*)W + row * bpr;

        for (size_t b = 0; b < bpr; ++b) {
            quant::dequant_q4_k(blocks + b, tmp.data() + b * 256);
        }
        break;
    }

    case QuantType::Q5_K: {
        const auto* blocks =
            (const quant::BlockQ5_K*)W + row * bpr;

        for (size_t b = 0; b < bpr; ++b) {
            quant::dequant_q5_k(blocks + b, tmp.data() + b * 256);
        }
        break;
    }

    case QuantType::Q6_K: {
        const auto* blocks =
            (const quant::BlockQ6_K*)W + row * bpr;

        for (size_t b = 0; b < bpr; ++b) {
            quant::dequant_q6_k(blocks + b, tmp.data() + b * 256);
        }
        break;
    }

    case QuantType::Q2_K: {
        const auto* blocks = (const quant::BlockQ2_K*)W + row * bpr;
        for (size_t b = 0; b < bpr; ++b) quant::dequant_q2_k(blocks + b, tmp.data() + b * 256);
        break;
    }

    case QuantType::Q3_K: {
        const auto* blocks = (const quant::BlockQ3_K*)W + row * bpr;
        for (size_t b = 0; b < bpr; ++b) quant::dequant_q3_k(blocks + b, tmp.data() + b * 256);
        break;
    }

    default:
        return 0.0f;
    }

    return dot_f32(tmp.data(), x, n_cols);
}

// ─── Row dispatch ─────────────────────────────────────────────────────────────

static float dot_row(const void* W, size_t row, size_t n_cols,
                      const float* x, QuantType qtype) {
    switch (qtype) {
    case QuantType::F32: {
        return dot_f32((const float*)W + row * n_cols, x, n_cols);
    }
    case QuantType::F16: {
        const uint16_t* Wh = (const uint16_t*)W + row * n_cols;
#if NEXUS_AVX2
        __m256 vacc = _mm256_setzero_ps();
        size_t i = 0; float buf[8];
        for (; i + 8 <= n_cols; i += 8) {
            for (int k = 0; k < 8; ++k) buf[k] = quant::f16_to_f32(Wh[i+k]);
            vacc = _mm256_fmadd_ps(_mm256_loadu_ps(buf), _mm256_loadu_ps(x+i), vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc,1);
        lo = _mm_hadd_ps(_mm_add_ps(lo,hi), lo); lo = _mm_hadd_ps(lo,lo);
        float acc = _mm_cvtss_f32(lo);
        for (; i < n_cols; ++i) acc += quant::f16_to_f32(Wh[i]) * x[i];
        return acc;
#elif NEXUS_SSE4
        __m128 vacc = _mm_setzero_ps();
        size_t i = 0; float buf[4];
        for (; i + 4 <= n_cols; i += 4) {
            for (int k = 0; k < 4; ++k) buf[k] = quant::f16_to_f32(Wh[i+k]);
            vacc = _mm_add_ps(vacc, _mm_mul_ps(_mm_loadu_ps(buf), _mm_loadu_ps(x+i)));
        }
        vacc = _mm_hadd_ps(vacc, vacc); vacc = _mm_hadd_ps(vacc, vacc);
        float acc = _mm_cvtss_f32(vacc);
        for (; i < n_cols; ++i) acc += quant::f16_to_f32(Wh[i]) * x[i];
        return acc;
#else
        float acc = 0.0f;
        for (size_t i = 0; i < n_cols; ++i) acc += quant::f16_to_f32(Wh[i]) * x[i];
        return acc;
#endif
    }
    case QuantType::BF16: {
        const uint16_t* Wb = (const uint16_t*)W + row * n_cols;
#if NEXUS_AVX2
        __m256 vacc = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 8 <= n_cols; i += 8) {
            float buf[8];
            for (int k = 0; k < 8; ++k) buf[k] = quant::bf16_to_f32(Wb[i+k]);
            vacc = _mm256_fmadd_ps(_mm256_loadu_ps(buf), _mm256_loadu_ps(x+i), vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc), hi = _mm256_extractf128_ps(vacc,1);
        lo = _mm_hadd_ps(_mm_add_ps(lo,hi), lo); lo = _mm_hadd_ps(lo,lo);
        float acc = _mm_cvtss_f32(lo);
        for (; i < n_cols; ++i) acc += quant::bf16_to_f32(Wb[i]) * x[i];
        return acc;
#elif NEXUS_SSE4
        __m128 vacc = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 4 <= n_cols; i += 4) {
            float buf[4];
            for (int k = 0; k < 4; ++k) buf[k] = quant::bf16_to_f32(Wb[i+k]);
            vacc = _mm_add_ps(vacc, _mm_mul_ps(_mm_loadu_ps(buf), _mm_loadu_ps(x+i)));
        }
        vacc = _mm_hadd_ps(vacc, vacc); vacc = _mm_hadd_ps(vacc, vacc);
        float acc = _mm_cvtss_f32(vacc);
        for (; i < n_cols; ++i) acc += quant::bf16_to_f32(Wb[i]) * x[i];
        return acc;
#else
        float acc = 0.0f;
        for (size_t i = 0; i < n_cols; ++i) acc += quant::bf16_to_f32(Wb[i]) * x[i];
        return acc;
#endif
    }
    case QuantType::Q8_0: {
        if ((n_cols % 32) != 0) {
            const size_t padded_cols = ((n_cols + 31) / 32) * 32;
            const size_t bpr = padded_cols / 32;

            std::vector<float> tmp(padded_cols, 0.0f);
            const auto* blocks = (const quant::BlockQ8_0*)W + row * bpr;

            for (size_t b = 0; b < bpr; ++b) {
                quant::dequant_q8_0(blocks + b, tmp.data() + b * 32);
            }

            return dot_f32(tmp.data(), x, n_cols);
        }

        size_t bpr = n_cols / 32;
        return dot_q8_0_row((const quant::BlockQ8_0*)W + row * bpr, x, bpr);
    }
    case QuantType::Q4_0: {
        if ((n_cols % 32) != 0) {
            const size_t padded_cols = ((n_cols + 31) / 32) * 32;
            const size_t bpr = padded_cols / 32;

            std::vector<float> tmp(padded_cols, 0.0f);
            const auto* blocks = (const quant::BlockQ4_0*)W + row * bpr;

            for (size_t b = 0; b < bpr; ++b) {
                quant::dequant_q4_0(blocks + b, tmp.data() + b * 32);
            }

            return dot_f32(tmp.data(), x, n_cols);
        }

        size_t bpr = n_cols / 32;
        return dot_q4_0_row((const quant::BlockQ4_0*)W + row * bpr, x, bpr);
    }

    case QuantType::Q5_0: {
        if ((n_cols % 32) != 0) {
            const size_t padded_cols = ((n_cols + 31) / 32) * 32;
            const size_t bpr = padded_cols / 32;

            std::vector<float> tmp(padded_cols, 0.0f);
            const auto* blocks = (const quant::BlockQ5_0*)W + row * bpr;

            for (size_t b = 0; b < bpr; ++b) {
                quant::dequant_q5_0(blocks + b, tmp.data() + b * 32);
            }

            return dot_f32(tmp.data(), x, n_cols);
        }

        size_t bpr = n_cols / 32;
        return dot_q5_0_row((const quant::BlockQ5_0*)W + row * bpr, x, bpr);
    }

    case QuantType::Q4_K: {
        if ((n_cols % 256) != 0) {
            return dot_k_tail_safe_row(W, row, n_cols, x, qtype);
        }

        size_t bpr = n_cols / 256;
        return dot_q4_k_row((const quant::BlockQ4_K*)W + row * bpr, x, bpr);
    }

    case QuantType::Q5_K: {
        if ((n_cols % 256) != 0) {
            return dot_k_tail_safe_row(W, row, n_cols, x, qtype);
        }

        size_t bpr = n_cols / 256;
        return dot_q5_k_row((const quant::BlockQ5_K*)W + row * bpr, x, bpr);
    }

    case QuantType::Q6_K: {
        if ((n_cols % 256) != 0) {
            return dot_k_tail_safe_row(W, row, n_cols, x, qtype);
        }

        size_t bpr = n_cols / 256;
        return dot_q6_k_row((const quant::BlockQ6_K*)W + row * bpr, x, bpr);
    }

    case QuantType::Q2_K:
    case QuantType::Q3_K: {
        // Q2_K / Q3_K: dequantise the row to float, then a float dot (the GPU kernels are the fast path for these formats)
        const size_t bpr = (n_cols + 255) / 256;
        static thread_local std::vector<float> tmp; tmp.resize(bpr * 256);
        if (qtype == QuantType::Q2_K) { const auto* blocks = (const quant::BlockQ2_K*)W + row * bpr; for (size_t b = 0; b < bpr; ++b) quant::dequant_q2_k(blocks + b, tmp.data() + b * 256); }
        else                          { const auto* blocks = (const quant::BlockQ3_K*)W + row * bpr; for (size_t b = 0; b < bpr; ++b) quant::dequant_q3_k(blocks + b, tmp.data() + b * 256); }
        return dot_f32(tmp.data(), x, n_cols);
    }

    case QuantType::NQ4_G64: {
        const size_t bpr = (n_cols + 63) / 64;
#if NEXUS_AVX2
        return dot_nq4_g64_row_avx2((const quant::BlockNQ_G64<4>*)W + row * bpr, x, n_cols);
#else
        return dot_nq_g64_row<4>((const quant::BlockNQ_G64<4>*)W + row * bpr, x, n_cols);
#endif
    }
    case QuantType::NQ5_G64: {
        const size_t bpr = (n_cols + 63) / 64;
        return dot_nq_g64_row<5>((const quant::BlockNQ_G64<5>*)W + row * bpr, x, n_cols);
    }
    case QuantType::NQ6_G64: {
        const size_t bpr = (n_cols + 63) / 64;
        return dot_nq_g64_row<6>((const quant::BlockNQ_G64<6>*)W + row * bpr, x, n_cols);
    }
    case QuantType::NQ8_G64: {
        const size_t bpr = (n_cols + 63) / 64;
        return dot_nq_g64_row<8>((const quant::BlockNQ_G64<8>*)W + row * bpr, x, n_cols);
    }

    default:
        return 0.0f;
    }
}

// ─── mat_vec ──────────────────────────────────────────────────────────────────

// Set by the model loader for heavily quantised files (Q2_K / Q3_K): int8 activations add ~1-6% error per matmul, which such models cannot absorb.
inline bool& q8k_force_float() { static bool v = false; return v; }
static bool q8k_disabled() {
    static int v = -1;
    if (v < 0) v = (getenv("NEXUS_DISABLE_Q8K") != nullptr) ? 1 : 0;
    return v == 1 || q8k_force_float();
}

// "8-bit output math" for NQ4 models on the CPU (off by default; launcher toggle, NEXUS_CPU_INT8=1). The inputs to each
// matrix are rounded to 8 bits per 64 values -- like the Q8K path K-quant GGUF models always use -- so the multiply-adds
// run on the CPU's byte instructions instead of unpacking every weight to a float. Faster; a very slight precision loss.
// NEXUS_CPU_INT8: "1" on, "0"/unset off, "auto" = measured on this CPU at the first real matrix (see nq_cpu_int8_autotune).
inline int nq_cpu_int8_mode() {
    static const int m = [] {
        const char* e = getenv("NEXUS_CPU_INT8");
        if (!e || !*e) return 0;
        if (e[0] == '1') return 1;
        if (e[0] == 'a' || e[0] == 'A') return 2;
        return 0;
    }();
    return m;
}
inline std::atomic<int>& nq_cpu_int8_decided() { static std::atomic<int> d{-1}; return d; }   // auto: -1 not yet, 0 float, 1 8-bit
inline bool nq_cpu_int8() {
#if NEXUS_AVX2
    const int m = nq_cpu_int8_mode();
    if (m != 2) return m == 1;
    return nq_cpu_int8_decided().load(std::memory_order_relaxed) == 1;
#else
    return false;
#endif
}

static inline void quantize_nq8(const float* x, size_t in_cols, NQ8X* out) {
    const size_t nb = (in_cols + 63) / 64;
    for (size_t b = 0; b < nb; ++b) {
        float v[64];
        const size_t c = b * 64, n = std::min<size_t>(64, in_cols - c);
        for (size_t i = 0; i < 64; ++i) v[i] = i < n ? x[c + i] : 0.0f;
        float amax = 0.0f;
        for (int i = 0; i < 64; ++i) amax = std::max(amax, std::fabs(v[i]));
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        int32_t sum = 0;
        for (int j = 0; j < 32; ++j) {
            const int a = (int)std::lrintf(v[2*j] * id), bq = (int)std::lrintf(v[2*j + 1] * id);
            out[b].ev[j] = (int8_t)std::max(-127, std::min(127, a));
            out[b].od[j] = (int8_t)std::max(-127, std::min(127, bq));
            sum += out[b].ev[j] + out[b].od[j];
        }
        out[b].d = d; out[b].isum = sum;
    }
}

#if NEXUS_AVX2
// sum over a row of  d_w * d_x * sum_i (q_i - 8) * x8_i ,  with the -8 folded out as  8 * sum(x8)
static float dot_nq4_q8_row(const quant::BlockNQ_G64<4>* blocks, const NQ8X* xq, size_t nb) {
    const __m256i m = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float corr = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blocks[b].qs));
        const __m256i lo = _mm256_and_si256(raw, m), hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m);
        const __m256i p = _mm256_add_epi16(
            _mm256_maddubs_epi16(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].ev))),
            _mm256_maddubs_epi16(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq[b].od))));
        const float sc = quant::f16_to_f32(blocks[b].d) * xq[b].d;
        acc = _mm256_fmadd_ps(_mm256_set1_ps(sc), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc);
        corr += sc * 8.0f * (float)xq[b].isum;
    }
    __m128 lo4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo4 = _mm_hadd_ps(lo4, lo4); lo4 = _mm_hadd_ps(lo4, lo4);
    return _mm_cvtss_f32(lo4) - corr;
}
#else
static float dot_nq4_q8_row(const quant::BlockNQ_G64<4>*, const NQ8X*, size_t) { return 0.0f; }
#endif

inline void mat_vec(const void* W, const float* x, float* y,
             size_t out_rows, size_t in_cols, QuantType qtype, int n_threads);

// Auto: time both ways on THIS CPU with the real thread count (a 8192 x 5120 NQ4 matrix, ~22 MB: bigger than the caches,
// so memory speed counts like it does for the model), and keep 8-bit only if it is at least 10% faster -- it costs a
// little precision, so it has to earn it. Whether it wins depends on the machine: where memory is the bottleneck,
// the faster arithmetic buys nothing.
inline void nq_cpu_int8_autotune(int n_threads) {
#if NEXUS_AVX2
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    if (nq_cpu_int8_decided().load() >= 0) return;
    const size_t rows = 8192, cols = 5120, nb = cols / 64;
    std::vector<quant::BlockNQ_G64<4>> Wt(rows * nb);
    uint32_t r = 2463534242u;
    for (auto& b : Wt) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; b.d = (uint16_t)(0x2000 + (r & 63)); for (auto& q : b.qs) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; q = (uint8_t)r; } }
    std::vector<float> xt(cols), yt(rows);
    for (auto& v : xt) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; v = ((int)(r >> 20) - 2048) / 2048.0f; }
    auto timed = [&](int mode) {
        nq_cpu_int8_decided().store(mode);
        mat_vec(Wt.data(), xt.data(), yt.data(), rows, cols, QuantType::NQ4_G64, n_threads);   // warm-up
        double best = 1e30;
        for (int k = 0; k < 5; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            mat_vec(Wt.data(), xt.data(), yt.data(), rows, cols, QuantType::NQ4_G64, n_threads);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    const double tf = timed(0), t8 = timed(1);
    const int pick = t8 < tf * 0.9 ? 1 : 0;
    nq_cpu_int8_decided().store(pick);
    fprintf(stderr, "[cpu] 8-bit output math (Auto): measured on this CPU with %d threads: full precision %.2f ms, 8-bit %.2f ms -> %s\n",
            n_threads, tf, t8, pick ? "8-bit" : "full precision (8-bit isn't at least 10% faster here)");
    fflush(stderr);
#else
    (void)n_threads;
#endif
}

inline void mat_vec(const void* W, const float* x, float* y,
             size_t out_rows, size_t in_cols, QuantType qtype, int n_threads) {
    if (qtype == QuantType::NQ4_G64 && nq_cpu_int8_mode() == 2 && nq_cpu_int8_decided().load(std::memory_order_relaxed) < 0)
        nq_cpu_int8_autotune(n_threads);
    if (qtype == QuantType::NQ4_G64 && nq_cpu_int8()) {
        static thread_local std::vector<NQ8X> xq;
        const size_t nb = (in_cols + 63) / 64;
        xq.resize(nb);
        quantize_nq8(x, in_cols, xq.data());
        if (n_threads <= 1) {
            for (size_t row = 0; row < out_rows; ++row)
                y[row] = dot_nq4_q8_row((const quant::BlockNQ_G64<4>*)W + row * nb, xq.data(), nb);
            return;
        }
        MatVecJob job;
        job.W = W; job.x = x; job.y = y; job.out_rows = out_rows; job.in_cols = in_cols;
        job.qtype = (int)qtype; job.q8k_x = nullptr; job.nq8_x = xq.data();
        get_pool(n_threads - 1)->run(job);
        return;
    }
    static thread_local Q8KBlock q8k_tls[128];
    const Q8KBlock* q8k_x = nullptr;

    const bool q8k_eligible = (qtype == QuantType::Q4_K ||
                               qtype == QuantType::Q5_K ||
                               qtype == QuantType::Q6_K);
    
    if (!q8k_disabled() && q8k_eligible && in_cols % 256 == 0) {
        const size_t n_blocks = in_cols / 256;
        if (n_blocks <= 128) {
            for (size_t b = 0; b < n_blocks; ++b)
                quantize_block_q8k(x + b * 256, &q8k_tls[b]);
            q8k_x = q8k_tls;
        }
    }

    if (n_threads <= 1) {
        if (q8k_x) {
            const size_t bpr = in_cols / 256;
            for (size_t row = 0; row < out_rows; ++row) {
                if      (qtype == QuantType::Q4_K)
                    y[row] = dot_q4_k_q8k_row((const quant::BlockQ4_K*)W + row*bpr, q8k_x, bpr);
                else if (qtype == QuantType::Q5_K)
                    y[row] = dot_q5_k_q8k_row((const quant::BlockQ5_K*)W + row*bpr, q8k_x, bpr);
                else if (qtype == QuantType::Q6_K)
                    y[row] = dot_q6_k_q8k_row((const quant::BlockQ6_K*)W + row*bpr, q8k_x, bpr);
                else
                    y[row] = dot_row(W, row, in_cols, x, qtype);
            }
        } else {
            for (size_t row = 0; row < out_rows; ++row)
                y[row] = dot_row(W, row, in_cols, x, qtype);
        }
        return;
    }

    ThreadPool* pool = get_pool(n_threads - 1);
    MatVecJob job;
    job.W = W; job.x = x; job.y = y;
    job.out_rows = out_rows; job.in_cols = in_cols;
    job.qtype = (int)qtype; job.q8k_x = q8k_x;
    pool->run(job);
}

inline void mat_vec_bias(const void* W, const float* x, const float* bias,
                  float* y, size_t out_rows, size_t in_cols,
                  QuantType qtype, int n_threads) {
    mat_vec(W, x, y, out_rows, in_cols, qtype, n_threads);
    if (bias) for (size_t i = 0; i < out_rows; ++i) y[i] += bias[i];
}

// ─── Batched matmul (GEMM): Y = W · X  ────────────────────────────────────────
//
// This is the prefill primitive. Instead of reading the entire weight matrix
// W once per token (what N separate mat_vec calls do), we read each weight row
// ONCE and dot it against all N token columns before moving on. For an
// N-token prefill that turns N full passes over W into a single pass — the
// whole reason batched prefill is fast.
//
// Layout (column-major token blocks, so each token's input is contiguous and
// can be Q8K-prequantized exactly like mat_vec does):
//   X : in_cols × N      X[token*in_cols + col]   (N contiguous input vectors)
//   Y : out_rows × N     Y[token*out_rows + row]  (N contiguous output vectors)
//
// Numerics are bit-identical to calling mat_vec(W, X+t*in_cols, Y+t*out_rows,
// ...) for each t, because we reuse the exact same dot_row / dot_q4_k_q8k_row
// kernels per (row, token) pair. This is the validation oracle for the GPU
// GEMM shader: the shader is correct iff its output matches this.
//
// Parallelized over output rows via run_indexed: each worker owns a slice of
// rows, dequantizes those weight rows once, and sweeps all N tokens. Weight
// traffic = one pass over W regardless of N.
// NQ*_G64 rows against several tokens at once (the MTP check, CPU prompt batches). The single-token kernels unpack every
// 64-weight block before using it, and that unpacking -- not reading memory -- is most of their time. mat_mat used to call
// them once PER TOKEN, so a batch of 4 cost 4.2 single tokens and MTP could never win. Here (NQ4, AVX2) each block is
// unpacked once with SIMD, scaled once, kept in registers while up to 4 tokens stream past, and the sum is reduced once per
// row. Same math; float rounding differs from the single-token kernel only in the last bits (summation order).
#if NEXUS_AVX2
static inline void nq4_g64_row_multi_avx2(const quant::BlockNQ_G64<4>* blocks, const float* X, size_t in_cols,
                                          size_t n_tokens, float* Y, size_t out_rows, size_t row) {
    const size_t full = in_cols / 64, n_blocks = (in_cols + 63) / 64;
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m256i e8  = _mm256_set1_epi32(8);
    for (size_t t0 = 0; t0 < n_tokens; t0 += 4) {
        const int nt = (int)std::min<size_t>(4, n_tokens - t0);
        // two accumulators per token, tokens interleaved: up to 8 independent multiply-add chains instead of one long one
        __m256 tot[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        __m256 tot2[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        float tail[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        const float* xs[4];
        for (int tt = 0; tt < 4; ++tt) xs[tt] = X + (t0 + (size_t)std::min(tt, nt - 1)) * in_cols;
        for (size_t b = 0; b < full; ++b) {
            const __m256 vd = _mm256_set1_ps(quant::f16_to_f32(blocks[b].d));
            const uint8_t* qs = blocks[b].qs;
            __m256 q[8];
            for (int h = 0; h < 2; ++h) {                 // 16 bytes -> 32 weights, in the kernel's order (low nibble first)
                const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs + 16 * h));
                const __m128i lo = _mm_and_si128(raw, m0f);
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m0f);
                const __m128i i0 = _mm_unpacklo_epi8(lo, hi), i1 = _mm_unpackhi_epi8(lo, hi);
                q[4*h + 0] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(i0), e8)));
                q[4*h + 1] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(i0, 8)), e8)));
                q[4*h + 2] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(i1), e8)));
                q[4*h + 3] = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(i1, 8)), e8)));
            }
            const size_t c = b * 64;
            if (nt == 4) {
                const float* x0 = xs[0] + c; const float* x1 = xs[1] + c; const float* x2 = xs[2] + c; const float* x3 = xs[3] + c;
                for (int k = 0; k < 8; k += 2) {
                    tot[0]  = _mm256_fmadd_ps(q[k],     _mm256_loadu_ps(x0 + 8 * k),       tot[0]);
                    tot[1]  = _mm256_fmadd_ps(q[k],     _mm256_loadu_ps(x1 + 8 * k),       tot[1]);
                    tot[2]  = _mm256_fmadd_ps(q[k],     _mm256_loadu_ps(x2 + 8 * k),       tot[2]);
                    tot[3]  = _mm256_fmadd_ps(q[k],     _mm256_loadu_ps(x3 + 8 * k),       tot[3]);
                    tot2[0] = _mm256_fmadd_ps(q[k + 1], _mm256_loadu_ps(x0 + 8 * (k + 1)), tot2[0]);
                    tot2[1] = _mm256_fmadd_ps(q[k + 1], _mm256_loadu_ps(x1 + 8 * (k + 1)), tot2[1]);
                    tot2[2] = _mm256_fmadd_ps(q[k + 1], _mm256_loadu_ps(x2 + 8 * (k + 1)), tot2[2]);
                    tot2[3] = _mm256_fmadd_ps(q[k + 1], _mm256_loadu_ps(x3 + 8 * (k + 1)), tot2[3]);
                }
            } else {
                for (int tt = 0; tt < nt; ++tt) {
                    const float* x = xs[tt] + c;
                    for (int k = 0; k < 8; k += 2) {
                        tot[tt]  = _mm256_fmadd_ps(q[k],     _mm256_loadu_ps(x + 8 * k),       tot[tt]);
                        tot2[tt] = _mm256_fmadd_ps(q[k + 1], _mm256_loadu_ps(x + 8 * (k + 1)), tot2[tt]);
                    }
                }
            }
        }
        for (size_t b = full; b < n_blocks; ++b) {        // a short last block
            const float d = quant::f16_to_f32(blocks[b].d);
            const uint8_t* qs = blocks[b].qs;
            const size_t c = b * 64, n = in_cols - c;
            for (size_t i2 = 0; i2 < n; ++i2) {
                const uint8_t byte = qs[i2 >> 1];
                const float w = d * (float)((int)((i2 & 1) ? (byte >> 4) : (byte & 0x0F)) - 8);
                for (int tt = 0; tt < nt; ++tt) tail[tt] += w * xs[tt][c + i2];
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            const __m256 t2 = _mm256_add_ps(tot[tt], tot2[tt]);
            __m128 lo = _mm256_castps256_ps128(t2);
            lo = _mm_add_ps(lo, _mm256_extractf128_ps(t2, 1));
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            Y[(t0 + (size_t)tt) * out_rows + row] = _mm_cvtss_f32(lo) + tail[tt];
        }
    }
}
#endif

// Other NQ widths: unpack each block once, then the single-token kernel's exact sum for every token (bit-identical).
template<int BITS>
static inline void nq_g64_row_multi(const quant::BlockNQ_G64<BITS>* blocks, const float* X, size_t in_cols,
                                    size_t n_tokens, float* Y, size_t out_rows, size_t row) {
#if NEXUS_AVX2
    if (BITS == 4) { nq4_g64_row_multi_avx2((const quant::BlockNQ_G64<4>*)(const void*)blocks, X, in_cols, n_tokens, Y, out_rows, row); return; }
#endif
    static thread_local std::vector<float> acc;
    acc.assign(n_tokens, 0.0f);
    const size_t n_blocks = (in_cols + 63) / 64;
    float qf[64];
    size_t c = 0;
    const int off = 1 << (BITS - 1);
    for (size_t b = 0; b < n_blocks; ++b) {
        const float d = quant::f16_to_f32(blocks[b].d);
        const uint8_t* qs = blocks[b].qs;
        const size_t n = std::min<size_t>(64, in_cols - c);
        for (size_t i = 0; i < n; ++i) qf[i] = d * (float)((int)nx_nq_get_bits_local<BITS>(qs, (int)i) - off);
        for (size_t t = 0; t < n_tokens; ++t) {
            const float* x = X + t * in_cols + c;
            float a = acc[t];
            for (size_t i = 0; i < n; ++i) a += qf[i] * x[i];
            acc[t] = a;
        }
        c += n;
    }
    for (size_t t = 0; t < n_tokens; ++t) Y[t * out_rows + row] = acc[t];
}

inline void mat_mat(const void* W, const float* X, float* Y,
                    size_t out_rows, size_t in_cols, size_t n_tokens,
                    QuantType qtype, int n_threads) {
    if (!W || !X || !Y || out_rows == 0 || in_cols == 0 || n_tokens == 0) {
        if (Y && out_rows && n_tokens) memset(Y, 0, out_rows * n_tokens * sizeof(float));
        return;
    }
    if (n_tokens == 1) {                       // degenerate → exact mat_vec path
        mat_vec(W, X, Y, out_rows, in_cols, qtype, n_threads);
        return;
    }

    // Plain float-ish weights (BF16/F16/F32), e.g. the tiny Qwen3.5 in_proj_a/b matrices.
    // The generic path below re-converts every weight once PER TOKEN (354x for a 354-token prompt) and
    // is single-row-at-a-time. Convert the whole matrix to float ONCE, then parallelise over tokens.
    if ((qtype == QuantType::BF16 || qtype == QuantType::F16 || qtype == QuantType::F32) &&
        out_rows * in_cols <= (size_t)32 * 1024 * 1024 && n_tokens >= 4) {
        std::vector<float> Wf(out_rows * in_cols);
        for (size_t r = 0; r < out_rows; ++r) {
            float* d = Wf.data() + r * in_cols;
            if (qtype == QuantType::F32) {
                memcpy(d, (const float*)W + r * in_cols, in_cols * sizeof(float));
            } else if (qtype == QuantType::F16) {
                const uint16_t* s = (const uint16_t*)W + r * in_cols;
                for (size_t i = 0; i < in_cols; ++i) d[i] = quant::f16_to_f32(s[i]);
            } else {
                const uint16_t* s = (const uint16_t*)W + r * in_cols;
                for (size_t i = 0; i < in_cols; ++i) d[i] = quant::bf16_to_f32(s[i]);
            }
        }
        auto do_tok = [&](size_t t) {
            const float* x = X + t * in_cols;
            for (size_t r = 0; r < out_rows; ++r) {
                const float* w = Wf.data() + r * in_cols;
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
                size_t i = 0;
                for (; i + 8 <= in_cols; i += 8) {
                    a0 += w[i]   * x[i];   a1 += w[i+1] * x[i+1];
                    a2 += w[i+2] * x[i+2]; a3 += w[i+3] * x[i+3];
                    a4 += w[i+4] * x[i+4]; a5 += w[i+5] * x[i+5];
                    a6 += w[i+6] * x[i+6]; a7 += w[i+7] * x[i+7];
                }
                float acc = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));
                for (; i < in_cols; ++i) acc += w[i] * x[i];
                Y[t * out_rows + r] = acc;
            }
        };
        if (n_threads <= 1) { for (size_t t = 0; t < n_tokens; ++t) do_tok(t); }
        else get_pool(n_threads - 1)->run_indexed((int)n_tokens, [&](int t){ do_tok((size_t)t); });
        return;
    }

    const bool q8k_eligible = (qtype == QuantType::Q4_K ||
                               qtype == QuantType::Q5_K ||
                               qtype == QuantType::Q6_K)
                              && (in_cols % 256 == 0) && !q8k_disabled();
    const size_t bpr = q8k_eligible ? (in_cols / 256) : 0;

    // Pre-quantize every token column to Q8K once (shared across all rows).
    // For non-Q8K-eligible types this stays empty and we use dot_row directly.
    std::vector<Q8KBlock> q8k_all;
    if (q8k_eligible) {
        q8k_all.resize(bpr * n_tokens);
        for (size_t t = 0; t < n_tokens; ++t) {
            const float* xt = X + t * in_cols;
            Q8KBlock* dst = q8k_all.data() + t * bpr;
            for (size_t b = 0; b < bpr; ++b)
                quantize_block_q8k(xt + b * 256, &dst[b]);
        }
    }

    const bool nq = (qtype == QuantType::NQ4_G64 || qtype == QuantType::NQ5_G64 ||
                     qtype == QuantType::NQ6_G64 || qtype == QuantType::NQ8_G64);
    const size_t nq_bpr = (in_cols + 63) / 64;
    std::vector<NQ8X> xq_all;
    const bool nq_i8 = qtype == QuantType::NQ4_G64 && nq_cpu_int8();
    if (nq_i8) {
        xq_all.resize(n_tokens * nq_bpr);
        for (size_t t = 0; t < n_tokens; ++t) quantize_nq8(X + t * in_cols, in_cols, xq_all.data() + t * nq_bpr);
    }
    auto do_row = [&](size_t row) {
        if (nq_i8) {
            const auto* blk = (const quant::BlockNQ_G64<4>*)W + row * nq_bpr;
            for (size_t t = 0; t < n_tokens; ++t)
                Y[t * out_rows + row] = dot_nq4_q8_row(blk, xq_all.data() + t * nq_bpr, nq_bpr);
            return;
        }
        if (nq) {
            switch (qtype) {
            case QuantType::NQ4_G64: nq_g64_row_multi<4>((const quant::BlockNQ_G64<4>*)W + row * nq_bpr, X, in_cols, n_tokens, Y, out_rows, row); return;
            case QuantType::NQ5_G64: nq_g64_row_multi<5>((const quant::BlockNQ_G64<5>*)W + row * nq_bpr, X, in_cols, n_tokens, Y, out_rows, row); return;
            case QuantType::NQ6_G64: nq_g64_row_multi<6>((const quant::BlockNQ_G64<6>*)W + row * nq_bpr, X, in_cols, n_tokens, Y, out_rows, row); return;
            default:                 nq_g64_row_multi<8>((const quant::BlockNQ_G64<8>*)W + row * nq_bpr, X, in_cols, n_tokens, Y, out_rows, row); return;
            }
        }
        if (qtype == QuantType::Q2_K || qtype == QuantType::Q3_K) {      // dequantise the row ONCE, then dot it with every token
            const size_t rbpr = (in_cols + 255) / 256;
            static thread_local std::vector<float> rtmp; rtmp.resize(rbpr * 256);
            if (qtype == QuantType::Q2_K) { const auto* bl = (const quant::BlockQ2_K*)W + row * rbpr; for (size_t b = 0; b < rbpr; ++b) quant::dequant_q2_k(bl + b, rtmp.data() + b * 256); }
            else                          { const auto* bl = (const quant::BlockQ3_K*)W + row * rbpr; for (size_t b = 0; b < rbpr; ++b) quant::dequant_q3_k(bl + b, rtmp.data() + b * 256); }
            for (size_t t = 0; t < n_tokens; ++t) Y[t * out_rows + row] = dot_f32(rtmp.data(), X + t * in_cols, in_cols);
            return;
        }
        for (size_t t = 0; t < n_tokens; ++t) {
            float v;
            if (q8k_eligible) {
                const Q8KBlock* q8k_x = q8k_all.data() + t * bpr;
                if      (qtype == QuantType::Q4_K)
                    v = dot_q4_k_q8k_row((const quant::BlockQ4_K*)W + row*bpr, q8k_x, bpr);
                else if (qtype == QuantType::Q5_K)
                    v = dot_q5_k_q8k_row((const quant::BlockQ5_K*)W + row*bpr, q8k_x, bpr);
                else
                    v = dot_q6_k_q8k_row((const quant::BlockQ6_K*)W + row*bpr, q8k_x, bpr);
            } else {
                v = dot_row(W, row, in_cols, X + t * in_cols, qtype);
            }
            Y[t * out_rows + row] = v;
        }
    };

    if (n_threads <= 1) {
        for (size_t row = 0; row < out_rows; ++row) do_row(row);
        return;
    }
    ThreadPool* pool = get_pool(n_threads - 1);
    pool->run_indexed((int)out_rows, [&](int row){ do_row((size_t)row); });
}

inline void mat_mat_bias(const void* W, const float* X, const float* bias,
                         float* Y, size_t out_rows, size_t in_cols,
                         size_t n_tokens, QuantType qtype, int n_threads) {
    mat_mat(W, X, Y, out_rows, in_cols, n_tokens, qtype, n_threads);
    if (bias)
        for (size_t t = 0; t < n_tokens; ++t)
            for (size_t r = 0; r < out_rows; ++r)
                Y[t * out_rows + r] += bias[r];
}

// ─── Elementwise ops ──────────────────────────────────────────────────────────

inline void rms_norm(const float* x, const float* weight, float* y, size_t n, float eps) {
    float sum_sq = 0.0f;
#if NEXUS_AVX2
    __m256 vacc = _mm256_setzero_ps(); size_t i = 0;
    for (; i + 8 <= n; i += 8) { __m256 v = _mm256_loadu_ps(x+i); vacc = _mm256_fmadd_ps(v,v,vacc); }
    { __m128 lo=_mm256_castps256_ps128(vacc), hi=_mm256_extractf128_ps(vacc,1);
      lo=_mm_add_ps(lo,hi); lo=_mm_hadd_ps(lo,lo); lo=_mm_hadd_ps(lo,lo); sum_sq=_mm_cvtss_f32(lo); }
    for (; i < n; ++i) sum_sq += x[i]*x[i];
#elif NEXUS_SSE4
    { __m128 vacc = _mm_setzero_ps(); size_t i = 0;
      for (; i + 4 <= n; i += 4) { __m128 v = _mm_loadu_ps(x+i); vacc = _mm_add_ps(vacc, _mm_mul_ps(v,v)); }
      vacc = _mm_hadd_ps(vacc, vacc); vacc = _mm_hadd_ps(vacc, vacc); sum_sq = _mm_cvtss_f32(vacc);
      for (; i < n; ++i) sum_sq += x[i]*x[i]; }
#else
    for (size_t i = 0; i < n; ++i) sum_sq += x[i]*x[i];
#endif
    float rms_inv = 1.0f / sqrtf(sum_sq / (float)n + eps);
#if NEXUS_AVX2
    __m256 vrms = _mm256_set1_ps(rms_inv); size_t j = 0;
    for (; j + 8 <= n; j += 8)
        _mm256_storeu_ps(y+j, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),_mm256_loadu_ps(weight+j)),vrms));
    for (; j < n; ++j) y[j] = x[j]*weight[j]*rms_inv;
#elif NEXUS_SSE4
    { __m128 vrms = _mm_set1_ps(rms_inv); size_t j = 0;
      for (; j + 4 <= n; j += 4)
          _mm_storeu_ps(y+j, _mm_mul_ps(_mm_mul_ps(_mm_loadu_ps(x+j),_mm_loadu_ps(weight+j)),vrms));
      for (; j < n; ++j) y[j] = x[j]*weight[j]*rms_inv; }
#else
    for (size_t i = 0; i < n; ++i) y[i] = x[i]*weight[i]*rms_inv;
#endif
}

inline void add_inplace(float* x, const float* y, size_t n) {
#if NEXUS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x+i, _mm256_add_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(y+i)));
    for (; i < n; ++i) x[i] += y[i];
#elif NEXUS_SSE4
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(x+i, _mm_add_ps(_mm_loadu_ps(x+i), _mm_loadu_ps(y+i)));
    for (; i < n; ++i) x[i] += y[i];
#else
    for (size_t i = 0; i < n; ++i) x[i] += y[i];
#endif
}

inline void softmax(float* x, size_t n) {
    float max_val = x[0];
    for (size_t i = 1; i < n; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) x[i] *= inv;
}

inline void scale(float* x, float s, size_t n) {
#if NEXUS_AVX2
    __m256 vs = _mm256_set1_ps(s); size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x+i, _mm256_mul_ps(_mm256_loadu_ps(x+i), vs));
    for (; i < n; ++i) x[i] *= s;
#elif NEXUS_SSE4
    { __m128 vs = _mm_set1_ps(s); size_t i = 0;
      for (; i + 4 <= n; i += 4)
          _mm_storeu_ps(x+i, _mm_mul_ps(_mm_loadu_ps(x+i), vs));
      for (; i < n; ++i) x[i] *= s; }
#else
    for (size_t i = 0; i < n; ++i) x[i] *= s;
#endif
}

inline float dot(const float* x, const float* y, size_t n) { return dot_f32(x, y, n); }

// SiLU activation: x[i] = x[i] * sigmoid(x[i])
inline void silu(float* x, size_t n) {
#if NEXUS_AVX2
    // sigmoid(x) = 1 / (1 + exp(-x))  — use fast exp approximation
    const __m256 one  = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v   = _mm256_loadu_ps(x + i);
        // Fast sigmoid via exp: clamp input to avoid overflow, compute exp(-v)
        __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), v);
        // Clamp to [-88, 88] to prevent expf overflow
        neg = _mm256_max_ps(neg, _mm256_set1_ps(-88.0f));
        neg = _mm256_min_ps(neg, _mm256_set1_ps( 88.0f));
        // Scalar exp per lane (AVX2 has no vectorized exp — use libm)
        float tmp[8]; _mm256_storeu_ps(tmp, neg);
        for (int k = 0; k < 8; ++k) tmp[k] = 1.0f / (1.0f + expf(tmp[k]));
        __m256 sig = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, sig));
    }
    for (; i < n; ++i) x[i] = x[i] / (1.0f + expf(-x[i]));
#elif NEXUS_SSE4
    // SSE4.1: process 4 floats at a time, scalar expf per lane (same as AVX2 approach)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(x + i);
        float tmp[4]; _mm_storeu_ps(tmp, v);
        for (int k = 0; k < 4; ++k) tmp[k] = 1.0f / (1.0f + expf(-tmp[k]));
        __m128 sig = _mm_loadu_ps(tmp);
        _mm_storeu_ps(x + i, _mm_mul_ps(v, sig));
    }
    for (; i < n; ++i) x[i] = x[i] / (1.0f + expf(-x[i]));
#else
    for (size_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + expf(-x[i]));
#endif
}

// Element-wise multiply: z[i] = x[i] * y[i]
inline void mul(const float* x, const float* y, float* z, size_t n) {
#if NEXUS_AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(z+i, _mm256_mul_ps(_mm256_loadu_ps(x+i), _mm256_loadu_ps(y+i)));
    for (; i < n; ++i) z[i] = x[i] * y[i];
#elif NEXUS_SSE4
    size_t i = 0;
    for (; i + 4 <= n; i += 4)
        _mm_storeu_ps(z+i, _mm_mul_ps(_mm_loadu_ps(x+i), _mm_loadu_ps(y+i)));
    for (; i < n; ++i) z[i] = x[i] * y[i];
#else
    for (size_t i = 0; i < n; ++i) z[i] = x[i] * y[i];
#endif
}

} // namespace cpu
} // namespace nexus
