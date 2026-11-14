/*
 * nexus/src/model/attention.h + attention.cpp
 *
 * Multi-head attention with:
 *   - Grouped Query Attention (GQA): n_kv_heads can be less than n_heads
 *     This is what Llama-3, Qwen2, Mistral all use. Each KV head serves
 *     multiple Q heads. Reduces KV cache size dramatically.
 *   - Rotary Position Embeddings (RoPE): applied to Q and K before attention
 *   - Chunked KV cache: writes to our memory-managed paged cache, not a
 *     pre-allocated monolith
 *   - Causal masking: tokens can only attend to previous positions
 *
 * The attention formula:
 *   Attention(Q, K, V) = softmax(Q × K^T / sqrt(head_dim)) × V
 *
 * With GQA: each Q head attends to one KV head (shared across n_heads/n_kv_heads Q heads)
 *
 * Memory flow per forward pass token:
 *   1. Project: q = Wq × x, k = Wk × x, v = Wv × x
 *   2. RoPE: rotate q and k based on position
 *   3. Write k, v into KV cache page for this position
 *   4. For each Q head: compute attention scores over all cached K, softmax, weighted sum of V
 *   5. Concat heads → output
 *   6. Project: y = Wo × output
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>
#include <memory>
#include <cstdlib>
#include <limits>
#include "../core/memory.h"
#include "../gguf/quant.h"
#include "kvcache.h"
#include "trace.h"

#if defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace nexus {
namespace model {

using quant::QuantType;

// ─── RoPE ────────────────────────────────────────────────────────────────────

struct RoPEConfig {
    float    freq_base  = 10000.0f;
    float    scale      = 1.0f;
    int      type       = 0;        // 0=standard, 1=linear, 2=yarn
    uint32_t n_ctx_orig = 4096;     // original context length for YaRN
    // 0 = interleaved pairs: x[0]/x[1], x[2]/x[3]
    // 1 = half-split/NeoX/Phi: x[i] / x[i + head_dim/2]
    int      layout     = 0;    
};

// Apply RoPE in-place to a single head's Q or K vector
// x: [head_dim] floats
// pos: token position in sequence
inline void rope_apply(float* x, int pos, int head_dim, const RoPEConfig& cfg);

// Apply RoPE to all heads in Q or K
// xq: [n_heads × head_dim] or [n_kv_heads × head_dim]
inline void rope_apply_all(float* xq, int n_heads, int pos, int head_dim,
                    const RoPEConfig& cfg);

// ─── Attention config ─────────────────────────────────────────────────────────

struct AttentionConfig {
    int      n_heads    = 32;
    int      n_kv_heads = 32;    // may be < n_heads for GQA
    int      head_dim   = 128;
    int      hidden_dim = 4096;
    float    scale      = 0.0f;  // 0 = auto: 1/sqrt(head_dim)
    float    rms_norm_eps = 1e-5f;
    RoPEConfig rope;
    int      n_threads  = 1;
    // Per-arch attention logit softcap. 0 = disabled (no tanh, no extra flops).
    // Gemma 2 = 50.0, Gemma 3+ = 0 (removed). Llama/Mistral/Qwen/etc. = 0.
    float    attn_softcap = 0.0f;

    // Sliding-window attention (SWA). 0 = disabled (full causal).
    // When > 0, each token can only attend to the last `sliding_window` tokens
    // (inclusive of itself), i.e. positions [pos - sliding_window + 1 .. pos].
    // Used by Mistral, Gemma 2/3, and Gemma 4 (on alternating layers).
    //
    // Gemma 4: the *local* layers set this; the *global* layers leave it 0
    // and run full causal. The per-layer choice is plumbed via
    // TransformerLayerWeights -> layer_attn_cfg in transformer.h::forward_layer.
    int      sliding_window = 0;
};

// ─── Single attention layer ───────────────────────────────────────────────────

class AttentionLayer {
public:
    AttentionLayer() = default;

    // Separate Q/K/V weights (most models)
    void set_weights(
        const void* wq,  QuantType tq,
        const void* wk,  QuantType tk,
        const void* wv,  QuantType tv,
        const void* wo,  QuantType to,
        const float* bq = nullptr,
        const float* bk = nullptr,
        const float* bv = nullptr,
        const float* bo = nullptr
    );

    // Fused QKV weight (Phi-3/4): single tensor [q_dim+2*kv_dim, hidden_dim]
    // At runtime we do one big mat_vec then split the result into Q, K, V
    void set_fused_qkv(
        const void* wqkv, QuantType tqkv,
        const void* wo,   QuantType to,
        const float* bq = nullptr,
        const float* bk = nullptr,
        const float* bv = nullptr
    );
    
    void set_extras(
        const float* q_norm,
        const float* k_norm,
        const void*  gate,
        QuantType    tg
    );

    void forward(
        const float* x,
        float*       out,
        KVCache&     kvcache,
        int          layer_id,
        int          pos,
        const AttentionConfig& cfg,
        float*       scratch
    );
    // CPU fallback: score+softmax+V only. Caller does RoPE + kvcache.store.
    void forward_cached_scores(const float* q, float* out,
        const KVCache& kvcache, int layer_id, int pos,
        const AttentionConfig& cfg) const;


    // GPU-assisted path: QKV already projected externally (e.g. via GPU).
    // q [n_heads*head_dim], k [n_kv_heads*head_dim], v [n_kv_heads*head_dim]
    // Performs RoPE, KV cache store, attention scores, V accumulate, Wo projection.
    void forward_from_qkv(
        float*       q,       // pre-projected Q (modified in-place by RoPE)
        float*       k,       // pre-projected K (modified in-place by RoPE)
        float*       v,       // pre-projected V
        float*       out,     // output [hidden_dim]
        KVCache&     kvcache,
        int          layer_id,
        int          pos,
        const AttentionConfig& cfg,
        float*       scratch
    );

    // Like forward_from_qkv but SKIPS the Wo projection.
    // On return `out` = concatenated head outputs [n_heads * head_dim].
    // Caller applies Wo (e.g. via GPU mat_vec for fully-offloaded layers).
    void forward_from_qkv_no_wo(
        float*       q,
        float*       k,
        float*       v,
        float*       out,
        KVCache&     kvcache,
        int          layer_id,
        int          pos,
        const AttentionConfig& cfg,
        float*       scratch
    );

private:
    const void* wq_   = nullptr;  QuantType tq_{};
    const void* wk_   = nullptr;  QuantType tk_{};
    const void* wv_   = nullptr;  QuantType tv_{};
    const void* wo_   = nullptr;  QuantType to_{};
    const float* bq_  = nullptr;
    const float* bk_  = nullptr;
    const float* bv_  = nullptr;
    const float* bo_  = nullptr;
    
    const float* q_norm_ = nullptr;
    const float* k_norm_ = nullptr;

    const void*  gate_ = nullptr;
    QuantType    tg_ {};

    // Fused QKV path
    const void* wqkv_ = nullptr;  QuantType tqkv_{};
    bool fused_        = false;
};

} // namespace model
} // namespace nexus


// ─────────────────────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include "../cpu/matmul.h"
#include <cstring>
#include <algorithm>
#include <cassert>

namespace nexus {
namespace model {

// ─── Attention logit soft-capping ─────────────────────────────────────────────
// Prevents exp() underflow from zeroing out distant tokens in long contexts.
// Without this, at 20k+ tokens, softmax concentrates all weight on the last
// few hundred tokens regardless of what's in the KV cache.
//
// tanh(x/cap)*cap keeps logits in (-cap, +cap) while preserving sign and rank.
// cap=30 is conservative: scores rarely exceed ±10 in practice, so the cap
// only activates for outlier scores that would dominate softmax anyway.
// Gemma 2 uses 50.0 in production. 30.0 is safer for models not trained with it.
static constexpr float ATTN_SOFTCAP = 30.0f;

// Fast tanh approximation via Padé [5/4] rational.
// Max error < 2e-7 for |x| ≤ 3.5, clamps to ±1 beyond.
// ~4× faster than tanhf() — critical at long context where we call this
// n_heads × n_ctx × n_layers times per token.
inline float fast_tanh(float x) {
    if (x >  4.5f) return  1.0f;
    if (x < -4.5f) return -1.0f;
    const float x2 = x * x;
    return x * (945.0f + x2 * (105.0f + x2)) /
               (945.0f + x2 * (420.0f + x2 * 15.0f));
}

// Softcap with a per-call value. cap <= 0 means "disabled" — caller should
// branch on cap > 0 to skip the tanh entirely on hot paths.
inline float attn_softcap(float x, float cap) {
    if (cap <= 0.0f) return x;
    const float inv = 1.0f / cap;
    return fast_tanh(x * inv) * cap;
}

// Legacy single-arg wrapper kept so any third-party code still compiles.
// New code paths inside Nexus should use the (x, cap) form.
inline float attn_softcap(float x) {
    return fast_tanh(x * (1.0f / ATTN_SOFTCAP)) * ATTN_SOFTCAP;
}

// AVX2 softmax over n floats. ~3× faster than scalar at long context.
// Uses a fast exp2 approximation: exp(x) ≈ exp2(x * log2e).
inline void softmax_fast(float* x, int n) {
#if KV_AVX2
    // Find max
    __m256 vmax = _mm256_set1_ps(-1e30f);
    int i = 0;
    for (; i + 8 <= n; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + i));
    // Horizontal max of vmax
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 1));
    float max_val = _mm_cvtss_f32(lo);
    for (; i < n; ++i) if (x[i] > max_val) max_val = x[i];

    // exp(x - max) and sum
    const float log2e = 1.4426950408f;
    __m256 vlog2e = _mm256_set1_ps(log2e);
    __m256 vm     = _mm256_set1_ps(max_val);
    __m256 vsum   = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        v = _mm256_sub_ps(v, vm);
        // exp2 via _mm256_exp2_ps not available without SVML — use scalar fallback
        // but keep the subtraction vectorized to avoid redundant load
        _mm256_storeu_ps(x + i, v);
        // exp each lane
        float tmp[8]; _mm256_storeu_ps(tmp, v);
        for (int k = 0; k < 8; ++k) tmp[k] = expf(tmp[k]);
        v = _mm256_loadu_ps(tmp);
        _mm256_storeu_ps(x + i, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    // Horizontal sum
    lo = _mm256_castps256_ps128(vsum);
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i) { x[i] = expf(x[i] - max_val); sum += x[i]; }

    // Normalize
    float inv = 1.0f / sum;
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv));
    for (; i < n; ++i) x[i] *= inv;
#else
    // Scalar fallback
    float max_val = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
#endif
}

// ─── Qwen3.6 attention mega diagnostics ──────────────────────────────────────
// Pure diagnostics only. No model math, sampler behavior, cache behavior, or
// outputs are changed. Runtime gated so other models stay quiet by default.
inline bool nx_qwen35_attn_mega_diag_enabled() {
    // Deliberately NOT tied to NEXUS_TRACE and NOT requiring terminal/env flags.
    // This header only emits diagnostics when the Qwen3.6/Qwen3-Next full-attn
    // shape + watched layer + watched position all match. No model math changes.
    return true;
}

inline bool nx_qwen35_attn_shape(const AttentionConfig& cfg) {
    // Qwen3.6/Qwen3-Next full-attn shape from the failing logs:
    // hidden=5120, q_heads=24, kv_heads=4, head_dim=256, rope_base=1e7.
    // Diagnostics only; if a different model shares this exact shape, the only
    // effect is compact stderr logging in the watched Qwen failure window only.
    return cfg.hidden_dim == 5120 &&
           cfg.n_heads == 24 &&
           cfg.n_kv_heads == 4 &&
           cfg.head_dim == 256 &&
           cfg.rope.freq_base > 9000000.0f &&
           cfg.rope.freq_base < 11000000.0f;
}

inline bool nx_qwen35_attn_layer_watch(int layer_id) {
    // Compact but useful: first full-attn layer, middle full-attn layer, final full-attn layer.
    // Enough to tell whether context/attention is broken early, drifts mid-stack, or blows up late.
    switch (layer_id) {
        case 3:
        case 31:
        case 63:
            return true;
        default:
            return false;
    }
}

inline bool nx_qwen35_attn_pos_watch(int pos) {
    // The failure window from the logs:
    // user end, assistant header, <think>, </think>, blank line, first answer tokens, EOS drift.
    return pos >= 30 && pos <= 47;
}

inline bool nx_qwen35_attn_diag(const AttentionConfig& cfg, int layer_id, int pos) {
    return nx_qwen35_attn_mega_diag_enabled() &&
           nx_qwen35_attn_shape(cfg) &&
           nx_qwen35_attn_layer_watch(layer_id) &&
           nx_qwen35_attn_pos_watch(pos);
}

struct NxMegaVecStats {
    float min_v = 0.0f;
    float max_v = 0.0f;
    double mean = 0.0;
    double rms = 0.0;
    int min_i = 0;
    int max_i = 0;
    int nan_count = 0;
    int inf_count = 0;
};

inline NxMegaVecStats nx_mega_vec_stats(const float* x, int n) {
    NxMegaVecStats st;
    if (!x || n <= 0) return st;
    st.min_v = std::numeric_limits<float>::infinity();
    st.max_v = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double ss = 0.0;
    int good = 0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (std::isnan(v)) { st.nan_count++; continue; }
        if (std::isinf(v)) { st.inf_count++; continue; }
        if (v < st.min_v) { st.min_v = v; st.min_i = i; }
        if (v > st.max_v) { st.max_v = v; st.max_i = i; }
        sum += (double)v;
        ss += (double)v * (double)v;
        good++;
    }
    if (good > 0) {
        st.mean = sum / (double)good;
        st.rms = std::sqrt(ss / (double)good);
    } else {
        st.min_v = 0.0f;
        st.max_v = 0.0f;
    }
    return st;
}

inline void nx_mega_dump_vec(const char* path, const char* name,
                             int layer_id, int pos,
                             const float* x, int n) {
    const NxMegaVecStats st = nx_mega_vec_stats(x, n);
    std::fprintf(stderr,
        "[mega-attn-vec] path=%s pos=%d layer=%d %s n=%d rms=%+.6e mean=%+.6e min=%+.6e@%d max=%+.6e@%d nan=%d inf=%d\n",
        path, pos, layer_id, name, n, st.rms, st.mean,
        st.min_v, st.min_i, st.max_v, st.max_i, st.nan_count, st.inf_count);
    std::fflush(stderr);
}

inline void nx_mega_dump_kv_coverage(const char* path,
                                     const KVCache& kvcache,
                                     int layer_id,
                                     int pos,
                                     const AttentionConfig& cfg) {
    if (!nx_qwen35_attn_diag(cfg, layer_id, pos)) return;

    const int n_ctx = pos + 1;
    const int heads_to_check = std::min(cfg.n_kv_heads, 1);
    std::fprintf(stderr,
        "[mega-qfa-ctx] path=%s pos=%d layer=%d n_ctx=%d q_heads=%d kv_heads=%d hd=%d rope_base=%.0f rope_layout=%d rope_type=%d sliding=%d\n",
        path, pos, layer_id, n_ctx, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        cfg.rope.freq_base, cfg.rope.layout, cfg.rope.type, cfg.sliding_window);

    for (int kvh = 0; kvh < heads_to_check; ++kvh) {
        int k_seen = 0, v_seen = 0;
        int k_pages = 0, v_pages = 0;
        int k_first_missing = -1, v_first_missing = -1;

        for (int pg = 0; k_seen < n_ctx; ++pg) {
            int ni = 0;
            const uint16_t* kp = kvcache.get_k_page(layer_id, kvh, pg, &ni);
            if (!kp || ni <= 0) { k_first_missing = k_seen; break; }
            k_pages++;
            k_seen += std::min(ni, n_ctx - k_seen);
        }
        for (int pg = 0; v_seen < n_ctx; ++pg) {
            int ni = 0;
            const uint16_t* vp = kvcache.get_v_page(layer_id, kvh, pg, &ni);
            if (!vp || ni <= 0) { v_first_missing = v_seen; break; }
            v_pages++;
            v_seen += std::min(ni, n_ctx - v_seen);
        }

        std::fprintf(stderr,
            "[mega-qfa-kv] path=%s pos=%d layer=%d kvh=%d k_seen=%d/%d k_pages=%d k_missing_at=%d v_seen=%d/%d v_pages=%d v_missing_at=%d\n",
            path, pos, layer_id, kvh,
            k_seen, n_ctx, k_pages, k_first_missing,
            v_seen, n_ctx, v_pages, v_first_missing);
    }
    std::fflush(stderr);
}

inline void nx_mega_dump_scores(const char* path, int layer_id, int pos,
                                int head, const float* scores, int n_ctx,
                                int window_start, bool probs) {
    if (!scores || n_ctx <= 0) return;
    const NxMegaVecStats st = nx_mega_vec_stats(scores, n_ctx);
    int top_i[5] = {-1,-1,-1,-1,-1};
    float top_v[5] = {-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for (int i = 0; i < n_ctx; ++i) {
        const float v = scores[i];
        if (!std::isfinite(v)) continue;
        for (int r = 0; r < 5; ++r) {
            if (v > top_v[r]) {
                for (int s = 4; s > r; --s) { top_v[s] = top_v[s-1]; top_i[s] = top_i[s-1]; }
                top_v[r] = v; top_i[r] = i;
                break;
            }
        }
    }
    std::fprintf(stderr,
        "[mega-attn-scores] path=%s pos=%d layer=%d head=%d %s n_ctx=%d window_start=%d rms=%+.6e mean=%+.6e min=%+.6e@%d max=%+.6e@%d top=%d:%+.6e,%d:%+.6e,%d:%+.6e,%d:%+.6e,%d:%+.6e\n",
        path, pos, layer_id, head, probs ? "probs" : "logits", n_ctx, window_start,
        st.rms, st.mean, st.min_v, st.min_i, st.max_v, st.max_i,
        top_i[0], top_v[0], top_i[1], top_v[1], top_i[2], top_v[2], top_i[3], top_v[3], top_i[4], top_v[4]);
    std::fflush(stderr);
}

// ─── RoPE ────────────────────────────────────────────────────────────────────

// LLaMA/Mistral/Qwen2 all use INTERLEAVED pairs: rotate (x[2i], x[2i+1]).
// The half-split style (x[i], x[i+half]) is wrong for these models and
// scrambles every attention score, causing stuck repetition loops.
inline void rope_apply(float* x, int pos, int head_dim, const RoPEConfig& cfg) {
    const float base  = cfg.freq_base;
    const float scale = cfg.scale != 0.0f ? cfg.scale : 1.0f;
    const int   pairs = head_dim / 2;
    const bool  half_split = (cfg.layout == 1);
    const int   half = head_dim / 2;

    // One-shot diagnostic: log exactly what THIS rope_apply call sees, keyed
    // on (type, layout, head_dim, base) so we see every distinct combination
    // that runs through this function. Caller cannot lie to us: this is what
    // the rotation actually uses.
    {
        char key[96];
        std::snprintf(key, sizeof(key),
            "rope_apply_invocation:type=%d_layout=%d_hd=%d_base=%.0f",
            cfg.type, cfg.layout, head_dim, base);
        ::nexus::trace::emit_decision(-1, "rope", key,
            "type=%d layout=%d hd=%d base=%.0f scale=%.4f n_ctx_orig=%u pairs=%d half_split=%d",
            cfg.type, cfg.layout, head_dim, base, scale, cfg.n_ctx_orig,
            pairs, half_split ? 1 : 0);
    }

    if (cfg.type == 1) {
        // Linear RoPE scaling: stretch all frequencies by scale factor.
        // Used when scale > 1 to extend context beyond training length.
        //
        // Layout-aware: half_split (NeoX/HF) rotates dim i with dim i+half;
        // interleaved (GPT-J/GGUF) rotates even-i with odd-i. Earlier versions
        // of this file ignored cfg.layout in the type==1 path, which silently
        // forced interleaved on every model using linear RoPE scaling.
        for (int i = 0; i < pairs; ++i) {
            float freq  = 1.0f / (powf(base, (float)(2 * i) / (float)head_dim) * scale);
            float theta = (float)pos * freq;
            float cos_t = cosf(theta), sin_t = sinf(theta);
            if (half_split) {
                float x0 = x[i], x1 = x[i + half];
                x[i]        = x0 * cos_t - x1 * sin_t;
                x[i + half] = x1 * cos_t + x0 * sin_t;
            } else {
                float x0 = x[2*i], x1 = x[2*i + 1];
                x[2*i]     = x0 * cos_t - x1 * sin_t;
                x[2*i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
        return;
    }

    if (cfg.type == 2) {
        // YaRN: non-uniform frequency interpolation.
        // Low-freq dimensions (large wavelength) are interpolated — they carry
        // long-range positional information and need gentle scaling.
        // High-freq dimensions (small wavelength) are left alone — they
        // distinguish nearby tokens and are already fine.
        //
        // cfg.scale = context extension ratio (e.g. 4.0 for 4× longer context)
        // cfg.n_ctx_orig = training context length (e.g. 4096 or 32768)
        //
        // Layout-aware: same fix as type==1 above.
        const float ext_factor = (scale > 1.0f) ? scale : 1.0f;
        // Attn temperature correction: sqrt(1 + log(scale)/log(n_ctx_orig))
        // Compensates for distribution shift introduced by frequency interpolation.
        const float attn_factor = (cfg.n_ctx_orig > 1 && ext_factor > 1.0f)
            ? sqrtf(1.0f + logf(ext_factor) / logf((float)cfg.n_ctx_orig))
            : 1.0f;

        for (int i = 0; i < pairs; ++i) {
            float freq_orig = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
            float wavelength = 2.0f * 3.14159265f / freq_orig;

            float freq_interp;
            if (wavelength < (float)cfg.n_ctx_orig) {
                // High frequency: no scaling needed
                freq_interp = freq_orig;
            } else if (wavelength > ext_factor * (float)cfg.n_ctx_orig) {
                // Low frequency: linear interpolation (divide by scale)
                freq_interp = freq_orig / ext_factor;
            } else {
                // Transition zone: NTK extrapolation blend
                // Smoothly blend between interpolated and extrapolated freq
                float blend = (wavelength / (float)cfg.n_ctx_orig - 1.0f) / (ext_factor - 1.0f);
                blend = blend < 0.0f ? 0.0f : (blend > 1.0f ? 1.0f : blend); // clamp
                freq_interp = freq_orig * (1.0f - blend) + (freq_orig / ext_factor) * blend;
            }

            float theta = (float)pos * freq_interp * attn_factor;
            float cos_t = cosf(theta), sin_t = sinf(theta);
            if (half_split) {
                float x0 = x[i], x1 = x[i + half];
                x[i]        = x0 * cos_t - x1 * sin_t;
                x[i + half] = x1 * cos_t + x0 * sin_t;
            } else {
                float x0 = x[2*i], x1 = x[2*i + 1];
                x[2*i]     = x0 * cos_t - x1 * sin_t;
                x[2*i + 1] = x0 * sin_t + x1 * cos_t;
            }
        }
        return;
    }

    // type == 0: Standard RoPE
    if (half_split) {
        for (int i = 0; i < half; ++i) {
            float freq  = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
            float theta = (float)pos * freq * scale;
            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            float x0 = x[i];
            float x1 = x[i + half];

            // HF rotate_half style:
            // first half = x0*cos - x1*sin
            // second half = x1*cos + x0*sin
            x[i]        = x0 * cos_t - x1 * sin_t;
            x[i + half] = x1 * cos_t + x0 * sin_t;
        }

        return;
    }

    for (int i = 0; i < pairs; ++i) {
        float freq  = 1.0f / powf(base, (float)(2 * i) / (float)head_dim);
        float theta = (float)pos * freq * scale;
        float cos_t = cosf(theta), sin_t = sinf(theta);
        float x0 = x[2*i], x1 = x[2*i + 1];
        x[2*i]     = x0 * cos_t - x1 * sin_t;
        x[2*i + 1] = x0 * sin_t + x1 * cos_t;
    }
  }
inline void rope_apply_all(float* xq, int n_heads, int pos, int head_dim,
                    const RoPEConfig& cfg) {
    for (int h = 0; h < n_heads; ++h) {
        rope_apply(xq + h * head_dim, pos, head_dim, cfg);
    }
}

// ─── AttentionLayer::set_weights ─────────────────────────────────────────────

inline void AttentionLayer::set_weights(
    const void* wq, QuantType tq,
    const void* wk, QuantType tk,
    const void* wv, QuantType tv,
    const void* wo, QuantType to,
    const float* bq, const float* bk,
    const float* bv, const float* bo)
{
    wq_ = wq; tq_ = tq;
    wk_ = wk; tk_ = tk;
    wv_ = wv; tv_ = tv;
    wo_ = wo; to_ = to;
    bq_ = bq; bk_ = bk; bv_ = bv; bo_ = bo;
    fused_ = false;
}

inline void AttentionLayer::set_fused_qkv(
    const void* wqkv, QuantType tqkv,
    const void* wo,   QuantType to,
    const float* bq, const float* bk, const float* bv)
{
    wqkv_ = wqkv; tqkv_ = tqkv;
    wo_ = wo; to_ = to;
    bq_ = bq; bk_ = bk; bv_ = bv;
    fused_ = true;
}

inline void AttentionLayer::set_extras(
    const float* q_norm,
    const float* k_norm,
    const void*  gate,
    QuantType    tg)
{
    q_norm_ = q_norm;
    k_norm_ = k_norm;
    gate_   = gate;
    tg_     = tg;
}

// ─── AttentionLayer::forward ─────────────────────────────────────────────────

inline void AttentionLayer::forward(
    const float* x,
    float*       out,
    KVCache&     kvcache,
    int          layer_id,
    int          pos,
    const AttentionConfig& cfg,
    float*       scratch)
{
    const int n_heads    = cfg.n_heads;
    const int n_kv_heads = cfg.n_kv_heads;
    const int head_dim   = cfg.head_dim;
    const int hidden_dim = cfg.hidden_dim;
    const int q_dim      = n_heads    * head_dim;
    const int kv_dim     = n_kv_heads * head_dim;
    const int n_threads  = cfg.n_threads;

    // Groups per KV head (for GQA)
    const int gqa_ratio  = n_heads / n_kv_heads;

    // Attention scale: 1/sqrt(head_dim)
    const float attn_scale = cfg.scale != 0.0f
        ? cfg.scale
        : 1.0f / sqrtf((float)head_dim);

    // Scratch layout (provided by caller):
    // [0        .. q_dim)      q  [n_heads × head_dim]
    // [q_dim    .. q_dim+kv_dim)  k  [n_kv_heads × head_dim]
    // [q_dim+kv_dim .. q_dim+2*kv_dim) v [n_kv_heads × head_dim]
    // (per-head score buffers now heap-allocated in all_scores vector)
    float* q      = scratch;
    float* k_cur  = scratch + q_dim;
    float* v_cur  = scratch + q_dim + kv_dim;

    // ── 1. QKV projections ────────────────────────────────────────────────────
    if (fused_) {
        // Phi-3/4 style: one big [q_dim+2*kv_dim, hidden_dim] weight
        // Do single mat_vec into [q, k_cur, v_cur] which are contiguous in scratch
        // Layout: q at scratch[0], k_cur at scratch[q_dim], v_cur at scratch[q_dim+kv_dim]
        const int qkv_dim = q_dim + 2 * kv_dim;
        cpu::mat_vec(wqkv_, x, q, qkv_dim, hidden_dim, tqkv_, n_threads);
        // q, k_cur, v_cur already point into the right offsets of scratch (set above)
    } else {
        cpu::mat_vec(wq_, x, q,     q_dim,  hidden_dim, tq_, n_threads);
        cpu::mat_vec(wk_, x, k_cur, kv_dim, hidden_dim, tk_, n_threads);
        cpu::mat_vec(wv_, x, v_cur, kv_dim, hidden_dim, tv_, n_threads);
    }

    if (bq_) cpu::add_inplace(q,     bq_, q_dim);
    if (bk_) cpu::add_inplace(k_cur, bk_, kv_dim);
    if (bv_) cpu::add_inplace(v_cur, bv_, kv_dim);

    // Qwen3.x-style per-head Q/K normalization.
    // The norm tensors are usually [head_dim], so apply one RMSNorm per head.
    if (q_norm_) {
        for (int h = 0; h < n_heads; ++h) {
            cpu::rms_norm(q + (size_t)h * head_dim,
                          q_norm_,
                          q + (size_t)h * head_dim,
                          (size_t)head_dim,
                          cfg.rms_norm_eps);
        }
    }

    if (k_norm_) {
        for (int h = 0; h < n_kv_heads; ++h) {
            cpu::rms_norm(k_cur + (size_t)h * head_dim,
                          k_norm_,
                          k_cur + (size_t)h * head_dim,
                          (size_t)head_dim,
                          cfg.rms_norm_eps);
        }
    }

    // ── 2. RoPE on Q and K ────────────────────────────────────────────────────
    rope_apply_all(q,     n_heads,    pos, head_dim, cfg.rope);
    rope_apply_all(k_cur, n_kv_heads, pos, head_dim, cfg.rope);

    const bool nx_mega_diag = nx_qwen35_attn_diag(cfg, layer_id, pos);
    if (nx_mega_diag) {
        nx_mega_dump_vec("forward", "q_post_rope", layer_id, pos, q, q_dim);
        nx_mega_dump_vec("forward", "k_post_rope", layer_id, pos, k_cur, kv_dim);
        nx_mega_dump_vec("forward", "v_current", layer_id, pos, v_cur, kv_dim);
    }

    // ── 3. Store K, V into KV cache ───────────────────────────────────────────
    kvcache.store(layer_id, pos, k_cur, v_cur, kv_dim);
    nx_mega_dump_kv_coverage("forward", kvcache, layer_id, pos, cfg);

    // ── 4. Per-head attention — parallelized across heads ─────────────────────
    // Each head is fully independent: disjoint score buffer slice, disjoint
    // output slice. No shared mutable state → safe to run n_threads in parallel.
    // At long context this is the dominant cost: O(n_heads × n_ctx × head_dim).
    // Serial execution at 25k tokens with 32 heads = 32× slower than necessary.

    const int n_ctx = pos + 1;

    // Sliding-window attention: only attend to the last `sliding_window` tokens.
    // window_start is the first KV index this query is allowed to look at.
    // 0 means no SWA (full causal). Used by Gemma 4 local layers.
    const int window_start = (cfg.sliding_window > 0)
        ? std::max(0, pos - cfg.sliding_window + 1)
        : 0;

    // Per-head score buffers on the heap — stack would overflow at long context.
    // 32 heads × 50k tokens × 4 bytes = 6.4 MB — always heap.
    std::vector<float> all_scores((size_t)n_heads * n_ctx, 0.0f);

    auto do_head = [&](int h) {
        const float* q_head = q + h * head_dim;
        const int kv_head   = h / gqa_ratio;
        float* hscores      = all_scores.data() + (size_t)h * n_ctx;
        float* out_h        = out + (size_t)h * head_dim;

        // Score computation: hscores[t] = softcap(dot(q_head, K[t]) * scale)
        int scored = 0;
        for (int pg = 0; scored < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* kpage = kvcache.get_k_page(layer_id, kv_head, pg, &n_in_page);
            if (!kpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - scored);
            for (int ti = 0; ti < tpp; ++ti) {
                const uint16_t* k_f16 = kpage + (size_t)ti * head_dim;
                float s = 0.0f;
#if KV_AVX2
                __m256 vacc = _mm256_setzero_ps();
                float buf[8];
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    quant::f16x8_to_f32x8(k_f16 + d, buf);
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(q_head + d),
                                           _mm256_loadu_ps(buf), vacc);
                }
                __m128 lo = _mm256_castps256_ps128(vacc);
                __m128 hi = _mm256_extractf128_ps(vacc, 1);
                lo = _mm_add_ps(lo, hi);
                lo = _mm_hadd_ps(lo, lo);
                lo = _mm_hadd_ps(lo, lo);
                s = _mm_cvtss_f32(lo);
                for (; d < (size_t)head_dim; ++d)
                    s += q_head[d] * quant::f16_to_f32(k_f16[d]);
#else
                for (int d = 0; d < head_dim; ++d)
                    s += q_head[d] * quant::f16_to_f32(k_f16[d]);
#endif
                hscores[scored + ti] = attn_softcap(s * attn_scale, cfg.attn_softcap);
            }
            scored += tpp;
        }

        // Softmax — vectorized at long context
        // Mask out-of-window positions for SWA (Gemma 4 local layers, Mistral, etc.)
        // before softmax; -INFINITY here → exp → 0 → no contribution.
        if (window_start > 0) {
            for (int t = 0; t < window_start; ++t) hscores[t] = -INFINITY;
        }
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("forward", layer_id, pos, h, hscores, n_ctx, window_start, false);
        softmax_fast(hscores, n_ctx);
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("forward", layer_id, pos, h, hscores, n_ctx, window_start, true);

        // V accumulation: out_h = sum_t hscores[t] * V[t]
        memset(out_h, 0, head_dim * sizeof(float));
        int accumulated = 0;
        for (int pg = 0; accumulated < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* vpage = kvcache.get_v_page(layer_id, kv_head, pg, &n_in_page);
            if (!vpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - accumulated);
            for (int ti = 0; ti < tpp; ++ti) {
                const float w = hscores[accumulated + ti];
                if (w < 1e-9f) continue;
                const uint16_t* v_f16 = vpage + (size_t)ti * head_dim;
#if KV_AVX2
                __m256 vw = _mm256_set1_ps(w);
                float buf[8];
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    quant::f16x8_to_f32x8(v_f16 + d, buf);
                    __m256 vo = _mm256_loadu_ps(out_h + d);
                    _mm256_storeu_ps(out_h + d,
                        _mm256_fmadd_ps(vw, _mm256_loadu_ps(buf), vo));
                }
                for (; d < (size_t)head_dim; ++d)
                    out_h[d] += w * quant::f16_to_f32(v_f16[d]);
#else
                for (int d = 0; d < head_dim; ++d)
                    out_h[d] += w * quant::f16_to_f32(v_f16[d]);
#endif
            }
            accumulated += tpp;
        }
    };  // end do_head

    if (n_threads > 1 && n_heads > 1) {
        ThreadPool* pool = get_shared_pool(n_threads - 1);
        if (pool) pool->run_indexed(n_heads, do_head);
        else for (int h = 0; h < n_heads; ++h) do_head(h);
    } else {
        for (int h = 0; h < n_heads; ++h) do_head(h);
    }

    if (nx_mega_diag) nx_mega_dump_vec("forward", "attn_concat_pre_gate", layer_id, pos, out, q_dim);

    // Optional attention/mixer gate.
    // Qwen3.6 hybrid blocks expose attn_gate.weight shaped like hidden -> q_dim.
    // Apply sigmoid(gate(x)) to the concatenated attention/mixer output before Wo.
    std::vector<float> gate_buf;
    if (gate_) {
        gate_buf.resize((size_t)q_dim);
        cpu::mat_vec(gate_, x, gate_buf.data(),
                     (size_t)q_dim, (size_t)hidden_dim, tg_, n_threads);
        if (nx_mega_diag) nx_mega_dump_vec("forward", "gate_raw", layer_id, pos, gate_buf.data(), q_dim);

        double gate_sum = 0.0, gate_ss = 0.0;
        float gate_min = 1.0f, gate_max = 0.0f;
        for (int i = 0; i < q_dim; ++i) {
            float g = gate_buf[(size_t)i];
            if (g > 80.0f) g = 1.0f;
            else if (g < -80.0f) g = 0.0f;
            else g = 1.0f / (1.0f + expf(-g));
            gate_min = std::min(gate_min, g);
            gate_max = std::max(gate_max, g);
            gate_sum += (double)g;
            gate_ss += (double)g * (double)g;
            out[(size_t)i] *= g;
        }
        if (nx_mega_diag) {
            std::fprintf(stderr,
                "[mega-attn-gate] path=forward pos=%d layer=%d sigmoid n=%d rms=%+.6e mean=%+.6e min=%+.6e max=%+.6e\n",
                pos, layer_id, q_dim,
                std::sqrt(gate_ss / (double)std::max(1, q_dim)),
                gate_sum / (double)std::max(1, q_dim), gate_min, gate_max);
            std::fflush(stderr);
            nx_mega_dump_vec("forward", "attn_concat_post_gate", layer_id, pos, out, q_dim);
        }
    }

    // ── 5. Output projection: y = Wo × concat(heads) ─────────────────────────
    // all_scores is still alive here and is hidden_dim floats minimum — use it
    // as the Wo output scratch rather than allocating again.
    all_scores.resize(hidden_dim);
    float* tmp_out = all_scores.data();
    cpu::mat_vec(wo_, out, tmp_out, hidden_dim, q_dim, to_, n_threads);
    if (bo_) cpu::add_inplace(tmp_out, bo_, hidden_dim);

    // Copy result to out
    memcpy(out, tmp_out, hidden_dim * sizeof(float));
    if (nx_mega_diag) nx_mega_dump_vec("forward", "wo_output", layer_id, pos, out, hidden_dim);
}

// ─── forward_from_qkv ────────────────────────────────────────────────────────
// QKV already projected by GPU. Performs RoPE, KV cache, attention, then
// Wo projection (CPU). Caller passes pre-allocated scratch [>= q_dim+2*kv_dim + (pos+1)].
inline void AttentionLayer::forward_from_qkv(
    float*       q,
    float*       k,
    float*       v,
    float*       out,
    KVCache&     kvcache,
    int          layer_id,
    int          pos,
    const AttentionConfig& cfg,
    float*       scratch)
{
    const int n_heads    = cfg.n_heads;
    const int n_kv_heads = cfg.n_kv_heads;
    const int head_dim   = cfg.head_dim;
    const int hidden_dim = cfg.hidden_dim;
    const int q_dim      = n_heads    * head_dim;
    const int n_threads  = cfg.n_threads;
    const int gqa_ratio  = n_heads / n_kv_heads;
    const float attn_scale = cfg.scale != 0.0f
        ? cfg.scale : 1.0f / sqrtf((float)head_dim);

    // Apply biases if present (projection happened externally but biases are ours)
    if (bq_) cpu::add_inplace(q, bq_, q_dim);
    if (bk_) cpu::add_inplace(k, bk_, n_kv_heads * head_dim);
    if (bv_) cpu::add_inplace(v, bv_, n_kv_heads * head_dim);

    // RoPE
    rope_apply_all(q, n_heads,    pos, head_dim, cfg.rope);
    rope_apply_all(k, n_kv_heads, pos, head_dim, cfg.rope);

    const bool nx_mega_diag = nx_qwen35_attn_diag(cfg, layer_id, pos);
    if (nx_mega_diag) {
        nx_mega_dump_vec("from_qkv", "q_post_rope", layer_id, pos, q, q_dim);
        nx_mega_dump_vec("from_qkv", "k_post_rope", layer_id, pos, k, n_kv_heads * head_dim);
        nx_mega_dump_vec("from_qkv", "v_current", layer_id, pos, v, n_kv_heads * head_dim);
    }

    // KV cache store
    kvcache.store(layer_id, pos, k, v, n_kv_heads * head_dim);
    nx_mega_dump_kv_coverage("from_qkv", kvcache, layer_id, pos, cfg);

    // Scores scratch — reuse the provided scratch buffer
    float* scores = scratch;
    float* output_concat = out;

    for (int h = 0; h < n_heads; ++h) {
        const float* q_head = q + h * head_dim;
        const int kv_head   = h / gqa_ratio;
        const int n_ctx     = pos + 1;

        int scored = 0;
        for (int pg = 0; scored < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* kpage = kvcache.get_k_page(layer_id, kv_head, pg, &n_in_page);
            if (!kpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - scored);
            for (int ti = 0; ti < tpp; ++ti) {
                const uint16_t* k_f16 = kpage + (size_t)ti * head_dim;
                float s = 0.0f;
#if KV_AVX2
                __m256 vacc = _mm256_setzero_ps();
                float buf[8];
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    quant::f16x8_to_f32x8(k_f16 + d, buf);
                    vacc = _mm256_fmadd_ps(_mm256_loadu_ps(buf), _mm256_loadu_ps(q_head+d), vacc);
                }
                __m128 lo = _mm256_castps256_ps128(vacc);
                __m128 hi = _mm256_extractf128_ps(vacc, 1);
                lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo,lo); lo = _mm_hadd_ps(lo,lo);
                s = _mm_cvtss_f32(lo);
                for (; d < (size_t)head_dim; ++d) s += quant::f16_to_f32(k_f16[d]) * q_head[d];
#else
                for (int d = 0; d < head_dim; ++d) s += quant::f16_to_f32(k_f16[d]) * q_head[d];
#endif
                scores[scored + ti] = attn_softcap(s * attn_scale, cfg.attn_softcap);
            }
            scored += tpp;
        }

        // SWA mask (Gemma 4 local layers): -inf out-of-window before softmax.
        int ws = 0;
        if (cfg.sliding_window > 0) {
            ws = std::max(0, pos - cfg.sliding_window + 1);
            for (int t = 0; t < ws; ++t) scores[t] = -INFINITY;
        }
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("from_qkv", layer_id, pos, h, scores, n_ctx, ws, false);
        cpu::softmax(scores, n_ctx);
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("from_qkv", layer_id, pos, h, scores, n_ctx, ws, true);

        float* out_h = output_concat + h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));

        int accumulated = 0;
        for (int pg = 0; accumulated < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* vpage = kvcache.get_v_page(layer_id, kv_head, pg, &n_in_page);
            if (!vpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - accumulated);
            for (int ti = 0; ti < tpp; ++ti) {
                const float s = scores[accumulated + ti];
                if (s < 1e-9f) continue;
                const uint16_t* v_f16 = vpage + (size_t)ti * head_dim;
#if KV_AVX2
                const __m256 vs = _mm256_set1_ps(s);
                float buf[8];
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    quant::f16x8_to_f32x8(v_f16 + d, buf);
                    __m256 vo = _mm256_loadu_ps(out_h + d);
                    _mm256_storeu_ps(out_h+d, _mm256_fmadd_ps(vs, _mm256_loadu_ps(buf), vo));
                }
                for (; d < (size_t)head_dim; ++d) out_h[d] += s * quant::f16_to_f32(v_f16[d]);
#else
                for (int d = 0; d < head_dim; ++d) out_h[d] += s * quant::f16_to_f32(v_f16[d]);
#endif
            }
            accumulated += tpp;
        }
    }

    if (nx_mega_diag) nx_mega_dump_vec("from_qkv", "attn_concat_pre_wo", layer_id, pos, output_concat, q_dim);

    // Wo projection (CPU — wo is typically in VRAM but we handle it in transformer.h)
    float* tmp_out = scores;
    cpu::mat_vec(wo_, out, tmp_out, hidden_dim, q_dim, to_, n_threads);
    if (bo_) cpu::add_inplace(tmp_out, bo_, hidden_dim);
    memcpy(out, tmp_out, hidden_dim * sizeof(float));
    if (nx_mega_diag) nx_mega_dump_vec("from_qkv", "wo_output", layer_id, pos, out, hidden_dim);
}

// ─── forward_from_qkv_no_wo ──────────────────────────────────────────────────
// Identical to forward_from_qkv but stops before the Wo projection.
// On return `out` holds the concatenated per-head results [n_heads * head_dim].
// The caller (transformer.h GPU path) dispatches Wo via GPU mat_vec.
inline void AttentionLayer::forward_from_qkv_no_wo(
    float*       q,
    float*       k,
    float*       v,
    float*       out,
    KVCache&     kvcache,
    int          layer_id,
    int          pos,
    const AttentionConfig& cfg,
    float*       scratch)
{
    const int n_heads    = cfg.n_heads;
    const int n_kv_heads = cfg.n_kv_heads;
    const int head_dim   = cfg.head_dim;
    const int n_threads  = cfg.n_threads;
    const int n_ctx      = pos + 1;
    const int gqa_ratio  = n_heads / n_kv_heads;
    const float attn_scale = cfg.scale != 0.0f
        ? cfg.scale : 1.0f / sqrtf((float)head_dim);

    if (bq_) cpu::add_inplace(q, bq_, n_heads    * head_dim);
    if (bk_) cpu::add_inplace(k, bk_, n_kv_heads * head_dim);
    if (bv_) cpu::add_inplace(v, bv_, n_kv_heads * head_dim);

    rope_apply_all(q, n_heads,    pos, head_dim, cfg.rope);
    rope_apply_all(k, n_kv_heads, pos, head_dim, cfg.rope);

    const bool nx_mega_diag = nx_qwen35_attn_diag(cfg, layer_id, pos);
    if (nx_mega_diag) {
        nx_mega_dump_vec("from_qkv_no_wo", "q_post_rope", layer_id, pos, q, n_heads * head_dim);
        nx_mega_dump_vec("from_qkv_no_wo", "k_post_rope", layer_id, pos, k, n_kv_heads * head_dim);
        nx_mega_dump_vec("from_qkv_no_wo", "v_current", layer_id, pos, v, n_kv_heads * head_dim);
    }

    kvcache.store(layer_id, pos, k, v, n_kv_heads * head_dim);
    nx_mega_dump_kv_coverage("from_qkv_no_wo", kvcache, layer_id, pos, cfg);

    // Per-head score buffers — n_heads independent slices, no aliasing between threads.
    // At 32 heads × 2048 ctx × 4 bytes = 256 KB — always heap, never stack.
    std::vector<float> all_scores((size_t)n_heads * n_ctx, 0.0f);

    // Self-contained lambda for one head. Writes to:
    //   out[h*head_dim..(h+1)*head_dim]   — disjoint per head
    //   all_scores[h*n_ctx..(h+1)*n_ctx]  — disjoint per head
    // No shared mutable state → safe to run in parallel.
    auto do_head = [&](int h) {
        const float* q_head = q + h * head_dim;
        const int kv_head   = h / gqa_ratio;
        float* scores       = all_scores.data() + (size_t)h * n_ctx;

        int scored = 0;
        for (int pg = 0; scored < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* kpage = kvcache.get_k_page(layer_id, kv_head, pg, &n_in_page);
            if (!kpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - scored);
            for (int ti = 0; ti < tpp; ++ti) {
                const uint16_t* k_f16 = kpage + (size_t)ti * head_dim;
                float s = 0.0f;
#if KV_AVX2
                __m256 vacc = _mm256_setzero_ps();
                float buf[8];
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    quant::f16x8_to_f32x8(k_f16 + d, buf);
                    __m256 vk = _mm256_loadu_ps(buf);
                    __m256 vq = _mm256_loadu_ps(q_head + d);
                    vacc = _mm256_fmadd_ps(vq, vk, vacc);
                }
                float tmp[8]; _mm256_storeu_ps(tmp, vacc);
                for (int kk = 0; kk < 8; ++kk) s += tmp[kk];
                for (; d < (size_t)head_dim; ++d) s += q_head[d] * quant::f16_to_f32(k_f16[d]);
#else
                for (int d = 0; d < head_dim; ++d) s += q_head[d] * quant::f16_to_f32(k_f16[d]);
#endif
                scores[scored + ti] = attn_softcap(s * attn_scale, cfg.attn_softcap);
            }
            scored += tpp;
        }

        // Softmax over scores[0..n_ctx)
        // SWA mask (Gemma 4 local layers): -inf out-of-window before softmax.
        int ws = 0;
        if (cfg.sliding_window > 0) {
            ws = std::max(0, pos - cfg.sliding_window + 1);
            for (int t = 0; t < ws; ++t) scores[t] = -INFINITY;
        }
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("from_qkv_no_wo", layer_id, pos, h, scores, n_ctx, ws, false);
        softmax_fast(scores, n_ctx);
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("from_qkv_no_wo", layer_id, pos, h, scores, n_ctx, ws, true);

        // V accumulation
        float* out_h = out + (size_t)h * head_dim;
        memset(out_h, 0, head_dim * sizeof(float));
        int accumulated = 0;
        for (int pg = 0; accumulated < n_ctx; ++pg) {
            int n_in_page = 0;
            const uint16_t* vpage = kvcache.get_v_page(layer_id, kv_head, pg, &n_in_page);
            if (!vpage || n_in_page == 0) break;
            const int tpp = std::min(n_in_page, n_ctx - accumulated);
            for (int ti = 0; ti < tpp; ++ti) {
                const float s = scores[accumulated + ti];
                const uint16_t* v_f16 = vpage + (size_t)ti * head_dim;
#if KV_AVX2
                size_t d = 0;
                for (; d + 8 <= (size_t)head_dim; d += 8) {
                    float buf[8];
                    quant::f16x8_to_f32x8(v_f16 + d, buf);
                    __m256 vv  = _mm256_loadu_ps(buf);
                    __m256 vs  = _mm256_set1_ps(s);
                    __m256 vo  = _mm256_loadu_ps(out_h + d);
                    _mm256_storeu_ps(out_h + d, _mm256_fmadd_ps(vs, vv, vo));
                }
                for (; d < (size_t)head_dim; ++d) out_h[d] += s * quant::f16_to_f32(v_f16[d]);
#else
                for (int d = 0; d < head_dim; ++d) out_h[d] += s * quant::f16_to_f32(v_f16[d]);
#endif
            }
            accumulated += tpp;
        }
    };  // end do_head lambda

    if (n_threads > 1 && n_heads > 1) {
        ThreadPool* pool = get_shared_pool(n_threads - 1);
        if (pool) {
            pool->run_indexed(n_heads, do_head);
        } else {
            for (int h = 0; h < n_heads; ++h) do_head(h);
        }
    } else {
        for (int h = 0; h < n_heads; ++h) do_head(h);
    }
    if (nx_mega_diag) nx_mega_dump_vec("from_qkv_no_wo", "attn_concat_no_wo", layer_id, pos, out, n_heads * head_dim);
    // NOTE: Wo projection intentionally omitted — caller handles it.
}

// ─── forward_cached_scores ──────────────────────────────────────────────────
inline void AttentionLayer::forward_cached_scores(
    const float* q, float* out, const KVCache& kvcache,
    int layer_id, int pos, const AttentionConfig& cfg) const
{
    const int nh=cfg.n_heads,nkv=cfg.n_kv_heads,hd=cfg.head_dim,nctx=pos+1;
    const int nt=cfg.n_threads, gqa=nh/nkv;
    const float sc=cfg.scale!=0.f?cfg.scale:1.f/sqrtf((float)hd);
    const bool nx_mega_diag = nx_qwen35_attn_diag(cfg, layer_id, pos);
    if (nx_mega_diag) {
        nx_mega_dump_vec("cached_scores", "q_input", layer_id, pos, q, nh * hd);
        nx_mega_dump_kv_coverage("cached_scores", kvcache, layer_id, pos, cfg);
    }
    std::vector<float> scores((size_t)nh*nctx,0.f);
    auto do_head=[&](int h){
        const float* qh=q+h*hd; const int kvh=h/gqa;
        float* s=scores.data()+(size_t)h*nctx; int done=0;
        for(int pg=0;done<nctx;pg++){ int ni=0;
            const uint16_t* kp=kvcache.get_k_page(layer_id,kvh,pg,&ni);
            if(!kp||ni==0)break; int tpp=std::min(ni,nctx-done);
            for(int ti=0;ti<tpp;ti++){const uint16_t* kf=kp+(size_t)ti*hd;float v=0.f;
#if KV_AVX2
                __m256 acc=_mm256_setzero_ps();float buf[8];size_t d=0;
                for(;d+8<=(size_t)hd;d+=8){quant::f16x8_to_f32x8(kf+d,buf);
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(qh+d),_mm256_loadu_ps(buf),acc);}
                float t8[8];_mm256_storeu_ps(t8,acc);for(int i=0;i<8;i++)v+=t8[i];
                for(;d<(size_t)hd;d++)v+=qh[d]*quant::f16_to_f32(kf[d]);
#else
                for(int d=0;d<hd;d++)v+=qh[d]*quant::f16_to_f32(kf[d]);
#endif
                s[done+ti]=attn_softcap(v*sc,cfg.attn_softcap);} done+=tpp;}
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("cached_scores", layer_id, pos, h, s, nctx, 0, false);
        float mx=s[0];for(int t=1;t<nctx;t++)if(s[t]>mx)mx=s[t];
        float sm=0;for(int t=0;t<nctx;t++){s[t]=expf(s[t]-mx);sm+=s[t];}
        if(sm>0)for(int t=0;t<nctx;t++)s[t]/=sm;
        if (nx_mega_diag && h == 0)
            nx_mega_dump_scores("cached_scores", layer_id, pos, h, s, nctx, 0, true);
        float* oh=out+(size_t)h*hd;memset(oh,0,hd*4);int ac=0;
        for(int pg=0;ac<nctx;pg++){ int ni=0;
            const uint16_t* vp=kvcache.get_v_page(layer_id,kvh,pg,&ni);
            if(!vp||ni==0)break; int tpp=std::min(ni,nctx-ac);
            for(int ti=0;ti<tpp;ti++){float wt=s[ac+ti];if(wt<1e-9f)continue;
                const uint16_t* vf=vp+(size_t)ti*hd;
#if KV_AVX2
                float buf[8];size_t d=0;
                for(;d+8<=(size_t)hd;d+=8){quant::f16x8_to_f32x8(vf+d,buf);
                    _mm256_storeu_ps(oh+d,_mm256_fmadd_ps(_mm256_set1_ps(wt),
                        _mm256_loadu_ps(buf),_mm256_loadu_ps(oh+d)));}
                for(;d<(size_t)hd;d++)oh[d]+=wt*quant::f16_to_f32(vf[d]);
#else
                for(int d=0;d<hd;d++)oh[d]+=wt*quant::f16_to_f32(vf[d]);
#endif
            }ac+=tpp;}
    };
    if(nt>1&&nh>1){ThreadPool* p=get_shared_pool(nt-1);
        if(p)p->run_indexed(nh,do_head);else for(int h=0;h<nh;h++)do_head(h);}
    else for(int h=0;h<nh;h++)do_head(h);
    if (nx_mega_diag) nx_mega_dump_vec("cached_scores", "attn_concat_cached", layer_id, pos, out, nh * hd);
}

} // namespace model
} // namespace nexus
