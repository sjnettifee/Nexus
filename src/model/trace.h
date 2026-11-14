/*
 * nexus/src/model/trace.h
 *
 * Universal forward-pass diagnostic system.
 *
 * Philosophy:
 *   - Every routing decision the engine makes during a forward pass should be
 *     visible in a log. Hidden decisions cause "why doesn't this model work"
 *     bug hunts that waste days.
 *   - Logging must be CHEAP when disabled (default) and STRUCTURED when on.
 *   - Once-per-layer logging, not every-token spam. Trace fires on the first
 *     token of the first prefill of each loaded model.
 *
 * Controlled by environment variable NEXUS_TRACE:
 *   unset or 0    : silent (only pre-existing diag stays)
 *   1             : log every routing decision once per layer on token 0
 *   2             : level 1 + tensor sample values (Q/K/V/norm I/O) for layer 0
 *   3             : level 2 + every layer (verbose, slow, big logs)
 *
 * Also accepts NEXUS_TRACE_LAYER=N to focus tensor samples on layer N (default 0).
 * Also accepts NEXUS_TRACE_FILE=path to redirect trace to a file (default stderr).
 *
 * Usage in the forward-pass code:
 *
 *   NX_TRACE_DECISION(l, "rope_layout", "%d (%s)", cfg.rope.layout,
 *                     cfg.rope.layout == 1 ? "half_split" : "interleaved");
 *   NX_TRACE_DECISION(l, "rope_freq_base", "%.0f", cfg.rope.freq_base);
 *   NX_TRACE_SAMPLE(l, "q_pre_rope", q, head_dim);
 *   NX_TRACE_SAMPLE(l, "q_post_rope", q, head_dim);
 *
 * The macros expand to nothing in hot paths when trace level is below the
 * threshold AND we've already logged this (layer, key) combination.
 *
 * No new dependencies. No threading magic. The "logged once" set is a flat
 * unordered_set guarded by a mutex, touched only on the first prefill of
 * each model load.
 */

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <mutex>
#include <atomic>

namespace nexus {
namespace trace {

// ─── Configuration ──────────────────────────────────────────────────────────

class TraceConfig {
public:
    static TraceConfig& instance() {
        static TraceConfig c;
        return c;
    }

    int  level()         const { return level_; }
    int  focus_layer()   const { return focus_layer_; }
    FILE* out()          const { return out_; }

    // True if we should emit DECISION events (any non-zero level).
    bool decisions_on()  const { return level_ >= 1; }

    // True if we should emit SAMPLE events for this layer.
    //   level 2: only focus_layer_  (default 0)
    //   level 3: every layer
    bool samples_on(int layer) const {
        if (level_ >= 3) return true;
        if (level_ >= 2 && layer == focus_layer_) return true;
        return false;
    }

    // Mark (layer, key) as logged. Returns true if this is the FIRST time —
    // caller should emit the log. Returns false if already logged (skip).
    // When level >= 3 or key starts with "!" we always return true (no dedup).
    bool first_time(int layer, const char* key) {
        if (level_ >= 3) return true;
        if (key && key[0] == '!') return true;
        std::lock_guard<std::mutex> lk(mu_);
        std::string k = std::to_string(layer) + ":" + (key ? key : "");
        return logged_.insert(std::move(k)).second;
    }

    // Reset the "logged once" set. Called automatically on model reload.
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        logged_.clear();
    }

private:
    TraceConfig() {
        const char* e = std::getenv("NEXUS_TRACE");
        level_ = e ? std::atoi(e) : 0;

        const char* l = std::getenv("NEXUS_TRACE_LAYER");
        focus_layer_ = l ? std::atoi(l) : 0;

        const char* f = std::getenv("NEXUS_TRACE_FILE");
        if (f && *f) {
            FILE* fp = std::fopen(f, "w");
            out_ = fp ? fp : stderr;
        } else {
            out_ = stderr;
        }

        if (level_ > 0) {
            std::fprintf(out_,
                "[trace] nexus forward-pass tracer ACTIVE: level=%d focus_layer=%d\n"
                "[trace] decisions log once per (layer,key); samples on focus_layer "
                "(use NEXUS_TRACE_LAYER=N to change; level=3 for all layers)\n",
                level_, focus_layer_);
            std::fflush(out_);
        }
    }

    int   level_       = 0;
    int   focus_layer_ = 0;
    FILE* out_         = stderr;
    std::mutex mu_;
    std::unordered_set<std::string> logged_;
};

// ─── Emit functions ────────────────────────────────────────────────────────

// One key routing decision. layer<0 means "model-global" (e.g. arch detection).
#if defined(__GNUC__) || defined(__clang__)
#  define NX_TRACE_PRINTF_ATTR(fmt_index, first_arg) \
      __attribute__((format(printf, fmt_index, first_arg)))
#else
#  define NX_TRACE_PRINTF_ATTR(fmt_index, first_arg)
#endif

inline void emit_decision(int layer, const char* category, const char* key,
                          const char* fmt, ...) NX_TRACE_PRINTF_ATTR(4, 5);

inline void emit_decision(int layer, const char* category, const char* key,
                          const char* fmt, ...) {
    auto& cfg = TraceConfig::instance();
    if (!cfg.decisions_on()) return;
    if (!cfg.first_time(layer, key)) return;

    FILE* out = cfg.out();
    if (layer < 0) {
        std::fprintf(out, "[trace.%s] %-30s = ", category, key);
    } else {
        std::fprintf(out, "[trace.%s] L=%-2d %-30s = ", category, layer, key);
    }
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(out, fmt, ap);
    va_end(ap);
    std::fputc('\n', out);
    std::fflush(out);
}

// Tensor sample: print first 4 + last 2 values + stats.
inline void emit_sample(int layer, const char* category, const char* key,
                        const float* x, size_t n) {
    auto& cfg = TraceConfig::instance();
    if (!cfg.samples_on(layer)) return;
    if (!cfg.first_time(layer, key)) return;
    if (!x || n == 0) return;

    double sumsq = 0.0, mn = x[0], mx = x[0], sum = 0.0;
    size_t nan = 0, inf = 0;
    int    mx_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        if (std::isnan(v)) { ++nan; continue; }
        if (std::isinf(v)) { ++inf; continue; }
        sum   += v;
        sumsq += (double)v * v;
        if (v < mn) mn = v;
        if (v > mx) { mx = v; mx_idx = (int)i; }
    }
    double mean = sum / (double)n;
    double rms  = std::sqrt(sumsq / (double)n);

    FILE* out = cfg.out();
    std::fprintf(out,
        "[trace.%s] L=%-2d %-30s n=%-6zu rms=%+.3e mean=%+.3e min=%+.3e "
        "max=%+.3e@%d nan=%zu inf=%zu  head=[",
        category, layer, key, n, rms, mean, mn, mx, mx_idx, nan, inf);
    const size_t head = std::min<size_t>(4, n);
    for (size_t i = 0; i < head; ++i) {
        std::fprintf(out, "%s%+.3e", i ? "," : "", x[i]);
    }
    std::fprintf(out, "]  tail=[");
    const size_t tail_start = n > 2 ? n - 2 : 0;
    for (size_t i = tail_start; i < n; ++i) {
        std::fprintf(out, "%s%+.3e", i == tail_start ? "" : ",", x[i]);
    }
    std::fprintf(out, "]\n");
    std::fflush(out);
}

// Sample of f16 (uint16_t) tensor — for KV cache inspection.
inline float f16_to_f32_trace(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) f = sign << 31;
        else {
            int e = -1;
            do { mant <<= 1; ++e; } while ((mant & 0x400) == 0);
            mant &= 0x3FF;
            f = (sign << 31) | (((uint32_t)(127 - 15 - e)) << 23) | (mant << 13);
        }
    } else if (expo == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | (((uint32_t)(expo - 15 + 127)) << 23) | (mant << 13);
    }
    float r; std::memcpy(&r, &f, 4); return r;
}

inline void emit_sample_f16(int layer, const char* category, const char* key,
                            const uint16_t* x, size_t n) {
    auto& cfg = TraceConfig::instance();
    if (!cfg.samples_on(layer)) return;
    if (!cfg.first_time(layer, key)) return;
    if (!x || n == 0) return;

    // Convert just enough to compute stats / show head.
    const size_t head = std::min<size_t>(4, n);
    float h0[8];
    for (size_t i = 0; i < head; ++i) h0[i] = f16_to_f32_trace(x[i]);

    double sumsq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float v = f16_to_f32_trace(x[i]);
        sumsq += (double)v * v;
    }
    double rms = std::sqrt(sumsq / (double)n);

    FILE* out = cfg.out();
    std::fprintf(out,
        "[trace.%s] L=%-2d %-30s n=%-6zu rms=%+.3e (f16)  head=[",
        category, layer, key, n, rms);
    for (size_t i = 0; i < head; ++i) {
        std::fprintf(out, "%s%+.3e", i ? "," : "", h0[i]);
    }
    std::fprintf(out, "]\n");
    std::fflush(out);
}

// Free-form note. Always fires when level >= 1, dedup'd by key.
inline void emit_note(int layer, const char* category, const char* key,
                      const char* msg) {
    auto& cfg = TraceConfig::instance();
    if (!cfg.decisions_on()) return;
    if (!cfg.first_time(layer, key)) return;
    FILE* out = cfg.out();
    if (layer < 0) {
        std::fprintf(out, "[trace.%s] %s: %s\n", category, key, msg);
    } else {
        std::fprintf(out, "[trace.%s] L=%-2d %s: %s\n", category, layer, key, msg);
    }
    std::fflush(out);
}

inline void section(const char* title) {
    auto& cfg = TraceConfig::instance();
    if (!cfg.decisions_on()) return;
    std::fprintf(cfg.out(),
        "[trace] ──── %s ────────────────────────────────────\n", title);
    std::fflush(cfg.out());
}

inline void reset() { TraceConfig::instance().reset(); }

} // namespace trace
} // namespace nexus

// ─── Convenience macros ────────────────────────────────────────────────────

// LAYER-scoped decision (e.g. "this layer used SWA with window=1024").
#define NX_TRACE_DECISION(layer, key, fmt, ...) \
    ::nexus::trace::emit_decision((layer), "decision", (key), (fmt), ##__VA_ARGS__)

// MODEL-global decision (e.g. arch detected, rope_layout chosen).
#define NX_TRACE_MODEL(key, fmt, ...) \
    ::nexus::trace::emit_decision(-1, "model", (key), (fmt), ##__VA_ARGS__)

// Tensor sample (only on focus layer at level >=2).
#define NX_TRACE_SAMPLE(layer, key, ptr, n) \
    ::nexus::trace::emit_sample((layer), "sample", (key), (ptr), (n))

#define NX_TRACE_SAMPLE_F16(layer, key, ptr, n) \
    ::nexus::trace::emit_sample_f16((layer), "sample", (key), (ptr), (n))

// One-line note keyed for dedup.
#define NX_TRACE_NOTE(layer, key, msg) \
    ::nexus::trace::emit_note((layer), "note", (key), (msg))

// Section banner.
#define NX_TRACE_SECTION(title) \
    ::nexus::trace::section((title))
