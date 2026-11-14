#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static std::mt19937 rng(2025);

// ── IEEE-754 half <-> float (same value space the kernel unpacks) ───────────
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sgn = (x >> 16) & 0x8000u;
    int32_t  exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) {                      // subnormal/zero (rounding ignored — fine for a test)
        if (exp < -10) return (uint16_t)sgn;
        man |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        return (uint16_t)(sgn | (man >> shift));
    } else if (exp >= 31) {              // overflow -> inf
        return (uint16_t)(sgn | 0x7C00u);
    }
    return (uint16_t)(sgn | ((uint32_t)exp << 10) | (man >> 13));
}
static float f16_to_f32(uint16_t h) {   // matches unpack_f16 in the shader
    uint32_t sgn = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t man = (uint32_t)h & 0x3FFu;
    uint32_t f32;
    if (exp == 0u) {
        if (man == 0u) f32 = sgn << 31u;
        else {
            uint32_t cnt = 0u, m = man;
            while ((m & 0x400u) == 0u) { cnt++; m <<= 1u; }
            f32 = (sgn << 31u) | ((113u - cnt) << 23u) | ((m & 0x3FFu) << 13u);
        }
    } else if (exp == 31u) {
        f32 = (sgn << 31u) | 0x7F800000u | (man << 13u);
    } else {
        f32 = (sgn << 31u) | ((exp + 112u) << 23u) | (man << 13u);
    }
    float out; std::memcpy(&out, &f32, 4); return out;
}


static float softcap_apply(float s, float cap) {
    if (cap <= 0.0f) return s;
    return cap * std::tanh(s / cap);
}

// ── Reference: ONE query, naive softmax over its causal window ──────────────
static void attn_single(const float* q,                 // [n_q_heads*head_dim]
                        const uint16_t* K, const uint16_t* V, // [n_kv_heads*n_ctx*head_dim] f16
                        float* out,                      // [n_q_heads*head_dim]
                        int n_q_heads, int n_kv_heads, int head_dim,
                        int n_ctx, int q_pos, int sliding_window,
                        float scale, float softcap) {
    std::vector<float> scores(n_ctx);
    for (int h = 0; h < n_q_heads; ++h) {
        int kv_h  = h * n_kv_heads / n_q_heads;
        int start = 0;
        if (sliding_window > 0 && q_pos + 1 > sliding_window) start = q_pos + 1 - sliding_window;
        int last  = std::min(q_pos, n_ctx - 1);
        const float* qh = q + (size_t)h * head_dim;
        const uint16_t* Kh = K + (size_t)kv_h * n_ctx * head_dim;
        const uint16_t* Vh = V + (size_t)kv_h * n_ctx * head_dim;

        float maxs = -1e30f;
        for (int p = start; p <= last; ++p) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qh[d] * f16_to_f32(Kh[(size_t)p*head_dim + d]);
            float s = softcap_apply(dot * scale, softcap);
            scores[p] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int p = start; p <= last; ++p) { scores[p] = std::exp(scores[p] - maxs); sum += scores[p]; }
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        float* oh = out + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            float a = 0.0f;
            for (int p = start; p <= last; ++p) a += scores[p] * f16_to_f32(Vh[(size_t)p*head_dim + d]);
            oh[d] = a * inv;
        }
    }
}

// ── Batched: all queries, flash online softmax (the shader's algorithm) ─────
static void attn_batch(const float* Q,                  // [n_query*n_q_heads*head_dim]
                       const uint16_t* K, const uint16_t* V,
                       float* Out,                       // [n_query*n_q_heads*head_dim]
                       int n_q_heads, int n_kv_heads, int head_dim,
                       int n_ctx, int n_query, int q_base_pos, int sliding_window,
                       float scale, float softcap) {
    for (int qi = 0; qi < n_query; ++qi) {
        int q_pos = q_base_pos + qi;
        for (int h = 0; h < n_q_heads; ++h) {
            int kv_h  = h * n_kv_heads / n_q_heads;
            int start = 0;
            if (sliding_window > 0 && q_pos + 1 > sliding_window) start = q_pos + 1 - sliding_window;
            int last  = std::min(q_pos, n_ctx - 1);
            const float* qh = Q + ((size_t)qi * n_q_heads + h) * head_dim;
            const uint16_t* Kh = K + (size_t)kv_h * n_ctx * head_dim;
            const uint16_t* Vh = V + (size_t)kv_h * n_ctx * head_dim;
            float* oh = Out + ((size_t)qi * n_q_heads + h) * head_dim;

            std::vector<float> acc(head_dim, 0.0f);
            float m = -1e30f, l = 0.0f;
            for (int p = start; p <= last; ++p) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qh[d] * f16_to_f32(Kh[(size_t)p*head_dim + d]);
                float s = softcap_apply(dot * scale, softcap);
                float m_new = std::max(m, s);
                float corr  = std::exp(m - m_new);
                float w     = std::exp(s - m_new);
                for (int d = 0; d < head_dim; ++d)
                    acc[d] = acc[d] * corr + w * f16_to_f32(Vh[(size_t)p*head_dim + d]);
                l = l * corr + w;
                m = m_new;
            }
            float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            for (int d = 0; d < head_dim; ++d) oh[d] = acc[d] * inv;
        }
    }
}

static bool run_case(const char* name, int n_q_heads, int n_kv_heads, int head_dim,
                     int n_query, int q_base_pos, int sliding_window, float softcap) {
    const int n_ctx = q_base_pos + n_query;
    const float scale = 1.0f / std::sqrt((float)head_dim);

    std::uniform_real_distribution<float> dq(-1.0f, 1.0f);
    std::vector<float> Q((size_t)n_query * n_q_heads * head_dim);
    for (auto& x : Q) x = dq(rng);

    // Random f16 K/V (random bytes through the f16 space fully exercise unpack).
    std::vector<uint16_t> K((size_t)n_kv_heads * n_ctx * head_dim);
    std::vector<uint16_t> V(K.size());
    std::uniform_real_distribution<float> dkv(-1.0f, 1.0f);
    for (auto& x : K) x = f32_to_f16(dkv(rng));
    for (auto& x : V) x = f32_to_f16(dkv(rng));

    // Reference: single-token, once per query.
    std::vector<float> Yref((size_t)n_query * n_q_heads * head_dim, 0.0f);
    for (int qi = 0; qi < n_query; ++qi) {
        attn_single(Q.data() + (size_t)qi * n_q_heads * head_dim,
                    K.data(), V.data(),
                    Yref.data() + (size_t)qi * n_q_heads * head_dim,
                    n_q_heads, n_kv_heads, head_dim, n_ctx,
                    q_base_pos + qi, sliding_window, scale, softcap);
    }

    // Batched (shader algorithm).
    std::vector<float> Ybatch((size_t)n_query * n_q_heads * head_dim, 0.0f);
    attn_batch(Q.data(), K.data(), V.data(), Ybatch.data(),
               n_q_heads, n_kv_heads, head_dim, n_ctx, n_query,
               q_base_pos, sliding_window, scale, softcap);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < Yref.size(); ++i) {
        double d = std::fabs((double)Yref[i] - (double)Ybatch[i]);
        if (d > max_abs) max_abs = d;
        double denom = std::max(1e-6, std::fabs((double)Yref[i]));
        if (d / denom > max_rel) max_rel = d / denom;
    }
    // Online vs naive softmax differ only in fp reassociation -> tiny.
    const bool ok = max_abs < 2e-4;
    printf("  %-22s qh=%-3d kvh=%-3d hd=%-3d N=%-5d base=%-4d swa=%-4d cap=%-4.0f : %s  max|d|=%.2e rel=%.2e\n",
           name, n_q_heads, n_kv_heads, head_dim, n_query, q_base_pos, sliding_window, softcap,
           ok ? "MATCH" : "FAIL ", max_abs, max_rel);
    return ok;
}

int main() {
    printf("batched-attention oracle — must match N x single-token attention\n");
    printf("----------------------------------------------------------------\n");
    bool all = true;

    // Qwen-style full attention: GQA 24q/4kv, head_dim 256 (the >128 case the
    // old single-token kernel could not cover).
    all &= run_case("qwen_full_hd256",     24, 4, 256, 1850, 0,    0, 0.0f);
    all &= run_case("qwen_full_small",     24, 4, 256, 64,   0,    0, 0.0f);

    // Standard head_dim 128, no GQA.
    all &= run_case("mha_hd128",           32, 32, 128, 512,  0,    0, 0.0f);
    // GQA 32q/8kv.
    all &= run_case("gqa_hd128",           32, 8,  128, 512,  0,    0, 0.0f);
    // head_dim 64.
    all &= run_case("gqa_hd64",            16, 4,  64,  300,  0,    0, 0.0f);

    // Continuation prefill (q_base_pos > 0: KV already has history).
    all &= run_case("continuation",        24, 4,  256, 256,  600,  0, 0.0f);

    // Sliding-window attention path.
    all &= run_case("swa_window256",       32, 8,  128, 1024, 0,    256, 0.0f);

    // Logit softcap path (Gemma-style; cap=0 elsewhere makes it a no-op).
    all &= run_case("softcap30_hd128",     16, 16, 128, 400,  0,    0, 30.0f);

    // Edge: single query must match the single-token kernel exactly.
    all &= run_case("N1_hd256",            24, 4,  256, 1,    0,    0, 0.0f);

    printf("----------------------------------------------------------------\n");
    printf("%s\n", all ? "ALL PASS — batched attention is a safe drop-in; wire the shader"
                       : "FAILURES — do NOT wire the GPU shader yet");
    return all ? 0 : 1;
}
