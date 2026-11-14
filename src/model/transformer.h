/*
 * nexus/src/model/transformer.h + transformer.cpp
 *
 * Full transformer forward pass.
 * This is what produces tokens. Everything else feeds into here.
 *
 * Architecture (LLaMA-style, covers Llama2/3, Mistral, Qwen2, Phi-3):
 *
 *   For each layer:
 *     1. RMSNorm(x)
 *     2. Self-attention --- attn_out
 *     3. x = x + attn_out           (residual)
 *     4. RMSNorm(x)
 *     5. FFN (SwiGLU):
 *        gate = silu(Wgate -- x)
 *        up   = Wup -- x
 *        ffn_out = Wdown -- (gate * up)
 *     6. x = x + ffn_out            (residual)
 *
 *   Final:
 *     7. RMSNorm(x)
 *     8. logits = Wlm_head -- x      (vocab projection)
 *     9. Sample next token
 *
 * Memory management:
 *   - Layer weights accessed via our pool (pinned before each layer, unpinned after)
 *   - Activations (x, attn_out, ffn_gate, etc.) from our ring buffer pool
 *   - KV cache from our paged allocator
 *   - Zero pre-allocated context: KV cache grows token by token
 */

#pragma once

// NEXUS_QWEN3NEXT_NATIVE_AUDIT_FIX 2026-05-25: native Qwen3Next path audited against HF layout; no GGUF-flattened defaults.
// NEXUS_QWEN3NEXT_MAIN_MATMUL_SPEED 2026-05-25: native NQ*_G64/BF16 matvec handled by cpu::mat_vec main route.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <unordered_map>
#include "../core/memguard.h"
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <future>
#include "../core/memory.h"
#include "../gguf/parser.h"
#include "../gguf/quant.h"
#include "../cpu/matmul.h"
#include "../gpu/vulkan_backend.h"
#include "attention.h"
#include "kvcache.h"
#if defined(__GLIBC__)
#  include <malloc.h>
#endif
#if defined(__linux__)
#  include <unistd.h>
#endif
#include "trace.h"
#include <cmath>
#include <climits>
#include <cassert>
#include <chrono>
#include <mutex>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <new>
#include <utility>

namespace nexus {
namespace model {

using quant::QuantType;
using gguf::TensorDesc;
using gguf::GGMLType;


// Runtime switch for the very heavy Qwen3Next bring-up diagnostics.
// Default OFF so normal generation is not buried in printf/log overhead.
static inline bool nx_qwen3next_diag_enabled() {
    const char* v = std::getenv("NEXUS_QWEN3NEXT_DIAG");
    if (!v || !*v) return false;
    if (v[0] == '0') return false;
    if ((v[0] == 'f' || v[0] == 'F') &&
        (v[1] == 'a' || v[1] == 'A') &&
        (v[2] == 'l' || v[2] == 'L') &&
        (v[3] == 's' || v[3] == 'S') &&
        (v[4] == 'e' || v[4] == 'E')) return false;
    if ((v[0] == 'n' || v[0] == 'N') &&
        (v[1] == 'o' || v[1] == 'O') && v[2] == '\0') return false;
    return true;
}

// --------- Convert our GGUF type enum to our quant type enum ---------------------------------------------------------------------
inline QuantType ggml_to_qtype(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return QuantType::F32;
        case GGMLType::F16:  return QuantType::F16;
        case GGMLType::BF16: return QuantType::BF16;

        case GGMLType::Q4_0: return QuantType::Q4_0;
        case GGMLType::Q5_0: return QuantType::Q5_0;
        case GGMLType::Q8_0: return QuantType::Q8_0;

        case GGMLType::Q4_K: return QuantType::Q4_K;
        case GGMLType::Q5_K: return QuantType::Q5_K;
        case GGMLType::Q6_K: return QuantType::Q6_K;
        case GGMLType::Q2_K: return QuantType::Q2_K;
        case GGMLType::Q3_K: return QuantType::Q3_K;

        case GGMLType::NQ4_G64: return QuantType::NQ4_G64;
        case GGMLType::NQ5_G64: return QuantType::NQ5_G64;
        case GGMLType::NQ6_G64: return QuantType::NQ6_G64;
        case GGMLType::NQ8_G64: return QuantType::NQ8_G64;

        default:
            fprintf(stderr,
                "[nexus] FATAL: unsupported GGML tensor type %d. Refusing unsafe F32 fallback.\n",
                (int)t);
            std::abort();
    }
}

// Checks every tensor of an opened GGUF BEFORE any weight is loaded.  Returns false (after printing a plain-English message) when the file uses a
// weight format the engine cannot run, instead of aborting half way through loading.
inline bool nexus_gguf_formats_supported(const gguf::Parser& parser) {
    std::map<std::string, int> bad; std::string first_tensor;
    bool low_bit = false;
    for (const auto& kv : parser.tensors()) {
        if (kv.second.type == GGMLType::Q2_K || kv.second.type == GGMLType::Q3_K) low_bit = true;
        switch (kv.second.type) {
            case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16: case GGMLType::Q4_0: case GGMLType::Q5_0: case GGMLType::Q8_0:
            case GGMLType::Q4_K: case GGMLType::Q5_K: case GGMLType::Q6_K: case GGMLType::Q2_K: case GGMLType::Q3_K:
            case GGMLType::NQ4_G64: case GGMLType::NQ5_G64: case GGMLType::NQ6_G64: case GGMLType::NQ8_G64:
                break;
            case GGMLType::I8: case GGMLType::I16: case GGMLType::I32: case GGMLType::I64: case GGMLType::F64:
                break;   // never used as weight matrices
            default:
                if (bad.empty()) first_tensor = kv.second.name;
                ++bad[gguf::ggml_type_name((int)kv.second.type)];
        }
    }
    cpu::q8k_force_float() = low_bit;   // Q2_K/Q3_K files: keep CPU activations in float (matches the GPU); other files keep the faster int8 path
    if (low_bit) fprintf(stderr, "[nexus] low-bit (Q2_K/Q3_K) model: CPU matmuls use float activations for accuracy\n");
    if (bad.empty()) return true;
    std::string list; for (auto& b : bad) { if (!list.empty()) list += ", "; list += b.first + " (" + std::to_string(b.second) + " tensors)"; }
    fprintf(stderr,
        "[nexus] UNSUPPORTED MODEL FORMAT: this file stores weights as %s, which Nexus cannot run yet (first seen in tensor \"%s\").\n"
        "[nexus]   Supported weight formats: F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Nexus NQ4/5/6/8.\n"
        "[nexus]   Please pick a different quantization of this model (for example Q4_K_M, Q5_K_M, Q6_K or Q8_0).\n",
        list.c_str(), first_tensor.c_str());
    fflush(stderr);
    return false;
}


// Qwen3.5 decode: the large projections of hybrid layers used to run on the CPU even with every layer on the GPU.  Try the GPU first (the backend returns
// false for unsupported shapes/types or when the weight cannot be made resident, and the caller then runs the CPU kernel exactly as before).
// NEXUS_Q35_GPU_PROJ=0 turns this off for A/B testing.
inline bool nx_q35_gpu_proj_enabled() {
    static const bool on = [] { const char* e = std::getenv("NEXUS_Q35_GPU_PROJ"); return !(e && e[0] == '0'); }();
    return on;
}
// `card` = the card the layer lives on (nullptr = the CPU). Before, this always used the first card, so layers given to a
// second, third or fourth card were computed on the CPU while writing and that card was never used.
inline bool nx_q35_try_gpu_matvec(gpu::VulkanBackend* card, const void* W, const float* x, float* y, size_t rows, size_t cols, QuantType wt) {
#ifdef NEXUS_VULKAN
    if (!card || !nx_q35_gpu_proj_enabled() || !W || rows * cols < 65536) return false;
    if (!card->ready()) return false;
    return card->mat_vec(W, x, y, rows, cols, wt);
#else
    (void)card; (void)W; (void)x; (void)y; (void)rows; (void)cols; (void)wt; return false;
#endif
}

// --------- Transformer layer (one decoder block) ------------------------------------------------------------------------------------------------------------

struct TransformerLayerWeights {
    // Attention --- separate Q/K/V (most models)
    const void* attn_norm  = nullptr;  QuantType attn_norm_type{};
    const float* attn_norm_bias = nullptr;
    const void* wq         = nullptr;  QuantType wq_type{};
    const void* wk         = nullptr;  QuantType wk_type{};
    const void* wv         = nullptr;  QuantType wv_type{};
    const void* wo         = nullptr;  QuantType wo_type{};
    // Gemma4: some layers do not expose a separate attn_v.weight.
    // For those layers, run V from the K projection so the layer can index/load.
    // This is launch-first support; generation quality may still need Gemma4's
    // layer scales/post norms after this.
    bool gemma4_tied_kv = false;    

    // ─── Gemma 4: Shared KV Cache donor layer ──────────────────────────────
    //
    // Gemma 4 introduces "Shared KV Cache" — the last num_kv_shared_layers
    // layers of the model DO NOT compute their own K/V projections at all.
    // Their Wk and Wv tensors are intentionally absent from the GGUF.
    // Instead, when attention runs on these layers, K and V are taken from
    // the cached K/V of an earlier "donor" layer.
    //
    // Sharing respects attention type:
    //   - A LOCAL (sliding-window) layer donates only to other LOCAL layers
    //   - A GLOBAL (full-attention) layer donates only to other GLOBAL layers
    //
    // We detect this at load: any gemma4 layer missing Wk/Wv is a shared
    // layer; we scan backwards for the nearest layer of the SAME attention
    // type that still has its own Wk and Wv, and record its index here.
    //
    // -1 = this layer computes its own K/V (the default for all non-gemma4
    // archs and for gemma4's earlier non-shared layers).
    int kv_donor_layer = -1;

    // ─── Per-tensor RMSNorm convention flags ──────────────────────────────
    //
    // Different GGUF converters bake different conventions into different
    // tensors of the SAME model. Gemma 4 31B Q4_K_M observed in the wild:
    //   attn_norm        mean=4.84   → pre-baked (use x*w)
    //   ffn_norm         mean=2.82   → pre-baked (use x*w)
    //   attn_q_norm      mean=1.02   → pre-baked +1 (use x*w directly)
    //   attn_k_norm      mean=0.12   → raw HF       (use x*(1+w))
    //
    // Applying a single global flag is therefore always wrong on at least one
    // tensor. We detect convention per-tensor at load time using its own
    // weight statistics:
    //   |mean(w)| < 0.5   → raw HF, needs (1+w)   → zero_centered = true
    //   |mean(w)| >= 0.5  → pre-baked or scaled   → use w directly
    //
    // If the tensor isn't present (nullptr in TransformerLayerWeights), the
    // flag is irrelevant and stays false.
    bool attn_norm_zc       = false;
    bool ffn_norm_zc        = false;
    bool attn_post_norm_zc  = false;
    bool ffn_post_norm_zc   = false;
    bool attn_q_norm_zc     = false;
    bool attn_k_norm_zc     = false;
    const float* bq        = nullptr;
    const float* bk        = nullptr;
    const float* bv        = nullptr;
    const float* bo        = nullptr;
    // Qwen3.x / some modern hybrid blocks use per-head Q/K RMSNorm and an
    // attention/mixer gate. If ignored, the model can run but output nonsense.
    const float* attn_q_norm = nullptr;
    const float* attn_k_norm = nullptr;
    size_t attn_q_norm_len = 0;
    size_t attn_k_norm_len = 0;
    
    // Gemma4 can mix attention projection widths by layer.
    // Do not force every layer through the single global metadata head_dim.
    // If these are non-zero, forward_layer() uses them for this layer only.
    int attn_q_dim_layer      = 0;
    int attn_kv_dim_layer     = 0;
    int attn_head_dim_layer   = 0;
    int attn_heads_layer      = 0;
    int attn_kv_heads_layer   = 0;    

    const void*  attn_gate      = nullptr;
    QuantType    attn_gate_type {};

    // Qwen3-Next full-attention layers can store query+gate in attn_q.weight.
    // Only used when !qwen35_linear; normal models leave this false.
    bool     qwen35_full_attn_q_has_gate = false;
    uint32_t qwen35_full_attn_q_rows     = 0;
    
    // Qwen3.5 / Qwen3.6 linear-attention / Gated DeltaNet path.
    // These are not normal softmax attention layers.
    bool qwen35_linear = false;

    const void* q35_qkv = nullptr;        QuantType q35_qkv_type {};
    const void* q35_z   = nullptr;        QuantType q35_z_type {};
    const void* q35_a   = nullptr;        QuantType q35_a_type {};
    const void* q35_b   = nullptr;        QuantType q35_b_type {};
    const void* q35_ba  = nullptr;        QuantType q35_ba_type {}; // ssm_beta_alpha.weight fused [hidden, 2*num_v_heads]
    const float* q35_conv = nullptr;      // ssm_conv1d.weight, F32, [kernel, conv_dim]
    const float* q35_dt   = nullptr;      // ssm_dt.bias or ssm_dt, [num_v_heads]
    const float* q35_A    = nullptr;      // ssm_a, [num_v_heads]
    const float* q35_norm = nullptr;      // ssm_norm.weight, [head_v_dim]
    const void* q35_out   = nullptr;      QuantType q35_out_type {};

    int q35_conv_dim    = 0;
    int q35_ba_dim      = 0;
    int q35_key_dim     = 0;
    int q35_value_dim   = 0;
    int q35_num_k_heads = 0;
    int q35_num_v_heads = 0;
    int q35_head_dim    = 0;
    int q35_kernel      = 0;
    
    int q35_qkv_rows = 0;

    // Fused QKV (Phi-3/4 style: one tensor [q_dim+2*kv_dim, hidden_dim])
    const void* wqkv       = nullptr;  QuantType wqkv_type{};
    bool fused_qkv         = false;    // true = use wqkv instead of wq/wk/wv

    // FFN
    const void* ffn_norm   = nullptr;  QuantType ffn_norm_type{};
    const float* ffn_norm_bias = nullptr;
    // Post-attention and post-FFN RMSNorm (Gemma 2/3/4 "sandwich norm" pattern).
    // Applied to the sub-layer output BEFORE the residual add.
    // nullptr for non-Gemma archs -> forward path skips them entirely.
    const float* attn_post_norm = nullptr;
    const float* ffn_post_norm  = nullptr;
    // Gemma4 layer scalar / output scale.
    //
    // Per ollama reference (model_text.go):
    //   LayerScalar ml.Tensor `gguf:"layer_scalar,alt:layer_output_scale.weight"`
    //   if l.LayerScalar != nil { hiddenState = hiddenState.Mul(ctx, l.LayerScalar) }
    //
    // `Mul` is ELEMENT-WISE multiplication. The tensor is a per-channel
    // vector of length hidden_dim (5376 for 31B), NOT a scalar of length 1.
    // Earlier code read element [0] as a single scalar and multiplied the
    // entire residual by it. For L=59 that scalar=0.036, which crushed the
    // residual stream by 27× — even though most channels actually have
    // larger weights and only some are 0.036.
    const float* layer_output_scale = nullptr;
    size_t       layer_output_scale_len = 0;
    const void* wgate      = nullptr;  QuantType wgate_type{};
    const void* wup        = nullptr;  QuantType wup_type{};
    const void* wdown      = nullptr;  QuantType wdown_type{};
    // Phi-3/4 fuses gate+up into a single tensor (first half=gate, second half=up)
    bool fused_gate_up     = false;
    
    // Routed MoE FFN: router + per-expert gate/up/down + optional shared expert.
    // Expert tensors are expected as merged GGUF tensors shaped like:
    //   ffn_gate_exps: [expert_ffn, hidden, n_experts]
    //   ffn_up_exps:   [expert_ffn, hidden, n_experts]
    //   ffn_down_exps: [hidden, expert_ffn, n_experts]
    bool moe              = false;
    bool moe_fused_gate_up = false;

    const void* moe_router = nullptr; QuantType moe_router_type{};
    const float* moe_router_bias = nullptr; // exp_probs_b.bias, usually F32

    const void* wgate_exps = nullptr; QuantType wgate_exps_type{}; size_t wgate_exps_stride = 0;
    const void* wup_exps   = nullptr; QuantType wup_exps_type{};   size_t wup_exps_stride   = 0;
    const void* wdown_exps = nullptr; QuantType wdown_exps_type{}; size_t wdown_exps_stride = 0;

    // Mixtral/OpenGPT-style GGUFs store each expert as separate tensors:
    //   blk.N.ffn_gate.0.weight / blk.N.ffn_up.0.weight / blk.N.ffn_down.0.weight
    std::vector<const void*> wgate_exp_ptrs;
    std::vector<const void*> wup_exp_ptrs;
    std::vector<const void*> wdown_exp_ptrs;

    std::vector<QuantType> wgate_exp_types;
    std::vector<QuantType> wup_exp_types;
    std::vector<QuantType> wdown_exp_types;

    bool moe_split_experts = false;

    uint32_t moe_expert_count = 0;
    uint32_t moe_top_k        = 0;
    uint32_t moe_ffn_dim      = 0;
    uint32_t moe_gate_rows = 0;
    uint32_t moe_gate_cols = 0;
    uint32_t moe_up_rows   = 0;
    uint32_t moe_up_cols   = 0;
    uint32_t moe_down_rows = 0;
    uint32_t moe_down_cols = 0;

    uint32_t moe_fused_rows = 0;
    uint32_t moe_fused_cols = 0;

    // True when GGUF stores the matrix as [output, input] instead of [input, output].
    // K-quants especially care about this because ne0 is the quantized row width.
    bool moe_gate_transposed  = false;
    bool moe_up_transposed    = false;
    bool moe_down_transposed  = false;
    bool moe_fused_transposed = false;
    
    // MoE router behavior:
    // 0 = standard softmax top-k routing
    // 1 = sparse top-2 mixer routing
    uint32_t moe_router_kind = 0;
    float    moe_router_jitter_eps = 0.0f;

    // Optional shared expert path.
    bool moe_has_shared = false;

    const void* wgate_shexp = nullptr; QuantType wgate_shexp_type{};
    const void* wup_shexp   = nullptr; QuantType wup_shexp_type{};
    const void* wdown_shexp = nullptr; QuantType wdown_shexp_type{};

    // Optional scalar/vector gate for the shared expert.
    const float* wgate_inp_shexp = nullptr;

    uint32_t moe_shared_ffn_dim = 0;

    uint32_t moe_shared_gate_rows = 0;
    uint32_t moe_shared_gate_cols = 0;
    uint32_t moe_shared_up_rows   = 0;
    uint32_t moe_shared_up_cols   = 0;
    uint32_t moe_shared_down_rows = 0;
    uint32_t moe_shared_down_cols = 0;

    bool moe_shared_gate_transposed = false;
    bool moe_shared_up_transposed   = false;
    bool moe_shared_down_transposed = false;

    // Per-layer routing behavior. Do not read global meta inside runtime.
    bool  moe_normalize_topk = false;
    float moe_route_scale    = 1.0f;

    // Set to true by Transformer::set_gpu_layers() for layers that should use the GPU
    bool gpu_layer         = false;
    bool gpu2_layer        = false;  // secondary GPU (iGPU)
    int  gpux              = -1;     // >=0: index into gpux_ (3rd, 4th, ... graphics card); -1 = the card set by set_gpu2_layers

    // ─── Gemma 4: per-layer attention kind ─────────────────────────────────
    //
    // Gemma 4 alternates local sliding-window attention (SWA) layers with
    // global full-context layers. Typical pattern: 5 local + 1 global, repeat.
    //   - Local layers:  cap attention to last `sw_window_size` tokens.
    //                    Uses standard RoPE freq_base (e.g. 10000).
    //   - Global layers: full causal, no window.
    //                    Uses "pruned" RoPE freq_base (e.g. 1000000).
    //
    // For non-Gemma archs, sw_window_size stays 0 and rope_freq_base_layer
    // stays 0.0f; forward_layer's per-layer cfg sees no overrides.
    int   sw_window_size       = 0;     // 0 = no SWA (global layer or non-Gemma)
    float rope_freq_base_layer = 0.0f;  // 0 = use AttentionConfig.rope.freq_base

    // ─── Gemma 4: Per-Layer Embeddings (PLE) slice index ──────────────────
    //
    // PLE table holds [n_layers × ple_dim] per token. Each layer reads its
    // slice [L * ple_dim .. (L+1) * ple_dim] and uses it as an auxiliary
    // residual after attention + FFN. The PLE buffer itself lives on the
    // Transformer (see ple_buffer_, ple_dim_), not in layer weights — this
    // field just tells forward_layer which slice to read.
    //
    // -1 = layer doesn't use PLE (non-Gemma 4 or PLE disabled).
    int   ple_layer_index = -1;
};

// --------- Full model ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct SamplerConfig {
    // ------ Core sampling ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    float temperature       = 0.7f;   // 0 = greedy argmax
    float top_p             = 0.95f;  // nucleus --- cumulative prob cutoff
    int   top_k             = 40;     // keep only top-k logits (0 = disabled)
    float min_p             = 0.05f;  // min-p: drop tokens < min_p * p_max (0 = disabled)
    float tfs_z             = 1.0f;   // tail-free sampling z (1.0 = disabled)

    // ------ Repetition control ---------------------------------------------------------------------------------------------------------------------------------------------------
    float repeat_penalty    = 1.1f;   // multiplicative penalty for recent tokens
    int   repeat_last_n     = 64;     // how far back to look for repeats
    float frequency_penalty = 0.0f;   // additive penalty scaled by token frequency
    float presence_penalty  = 0.0f;   // flat additive penalty if token appeared at all

    // ------ Misc ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int   seed              = -1;
    const std::vector<int32_t>* recent_tokens = nullptr;
};

// ---- Attention extras for modern MoE / hybrid blocks ------------------------
// These keep correctness without forcing the entire layer through the slow
// full-CPU AttentionLayer::forward path.

inline void nx_add_bias_inplace(float* y, const float* b, int n) {
    if (!y || !b || n <= 0) return;
    for (int i = 0; i < n; ++i) y[i] += b[i];
}


inline void nx_layer_norm_affine(const float* x,
                                 const float* weight,
                                 const float* bias,
                                 float* y,
                                 size_t n,
                                 float eps) {
    if (!x || !weight || !y || n == 0) return;

    float mean = 0.0f;
    for (size_t i = 0; i < n; ++i) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    const float inv = 1.0f / sqrtf(var + eps);
    if (bias) {
        for (size_t i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    } else {
        for (size_t i = 0; i < n; ++i) y[i] = (x[i] - mean) * inv * weight[i];
    }
}

inline void nx_rms_norm_add_one(const float* x,
                                const float* weight,
                                float* y,
                                size_t n,
                                float eps) {
    if (!x || !weight || !y || n == 0) return;

    float ss = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        ss += x[i] * x[i];
    }

    const float inv = 1.0f / sqrtf(ss / (float)n + eps);

    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] * inv * (1.0f + weight[i]);
    }
}

// ---- prefill CPU glue helpers: uninitialised reusable scratch + token-parallel loops ----
struct nx_view {
    float* p = nullptr;
    explicit nx_view(float* q) : p(q) {}
    float* data() const { return p; }
    float& operator[](size_t i) const { return p[i]; }
};
// Grow-only scratch: contents are NOT cleared (callers overwrite every element before reading); avoids re-faulting ~2 GB of fresh zero pages per layer.
static inline float* nx_scratch(std::vector<float>& v, size_t n) {
    if (v.size() < n) { nexus::memguard::require((n - v.size()) * sizeof(float), "reading the prompt"); std::vector<float>().swap(v); v.resize(n); }
    return v.data();
}
// Run fn(begin, end) over [0, n) split across the CPU worker pool (n_threads <= 1 or small n: inline).
template <class F>
static inline void nx_par_range(int n, int n_threads, F&& fn) {
    if (n <= 0) return;
    if (n_threads <= 1 || n < 128) { fn(0, n); return; }
    const int chunks = std::min(n, n_threads * 4);
    get_shared_pool(n_threads - 1)->run_indexed(chunks, [&](int c) {
        const int a = (int)((long long)n * c / chunks), b = (int)((long long)n * (c + 1) / chunks);
        fn(a, b);
    });
}

inline void nx_apply_model_norm(const float* x,
                                const void* weight,
                                const float* bias,
                                float* y,
                                size_t n,
                                float eps,
                                bool zero_centered_weight = false) {
    // PhiMoE / SlimMoE GGUF has norm.bias tensors and matches affine LayerNorm.
    if (bias) {
        nx_layer_norm_affine(x, (const float*)weight, bias, y, n, eps);
        return;
    }

    // Qwen3-Next / Qwen3.5 uses zero-centered RMSNorm:
    // output = rms_norm(x) * (1 + weight)
    if (zero_centered_weight) {
        nx_rms_norm_add_one(x, (const float*)weight, y, n, eps);
        return;
    }

    // Llama/Mistral/OLMoE/etc.
    cpu::rms_norm(x, (const float*)weight, y, n, eps);
}

inline void nx_apply_head_rms_inplace(float* x, const float* w,
                                      int n_heads, int head_dim, float eps) {
    if (!x || !w || n_heads <= 0 || head_dim <= 0) return;
    for (int h = 0; h < n_heads; ++h) {
        float* row = x + (size_t)h * head_dim;
        cpu::rms_norm(row, w, row, (size_t)head_dim, eps);
    }
}

inline void nx_apply_projection_norm_inplace(float* x,
                                             const float* w,
                                             size_t w_len,
                                             int total_dim,
                                             int n_heads,
                                             int head_dim,
                                             float eps,
                                             const char* label,
                                             bool zero_centered_weight = false) {
    if (!x || !w || total_dim <= 0 || n_heads <= 0 || head_dim <= 0) return;

    if (w_len == (size_t)total_dim) {
        // OLMoE-style: norm the full projected Q/K vector before reshape.
        // Qwen3-Next/Gemma-style norms use (1 + weight); normal models use weight.
        if (zero_centered_weight) {
            nx_rms_norm_add_one(x, w, x, (size_t)total_dim, eps);
        } else {
            cpu::rms_norm(x, w, x, (size_t)total_dim, eps);
        }
        return;
    }

    if (w_len == (size_t)head_dim) {
        // Some architectures use one norm vector per head.
        for (int h = 0; h < n_heads; ++h) {
            float* row = x + (size_t)h * head_dim;
            if (zero_centered_weight) {
                nx_rms_norm_add_one(row, w, row, (size_t)head_dim, eps);
            } else {
                cpu::rms_norm(row, w, row, (size_t)head_dim, eps);
            }
        }
        return;
    }

    fprintf(stderr,
        "[nexus] WARNING: %s norm length mismatch: w_len=%zu total_dim=%d head_dim=%d heads=%d zero_centered=%d. Skipping norm.\n",
        label, w_len, total_dim, head_dim, n_heads, zero_centered_weight ? 1 : 0);
}

inline void nx_apply_projection_rms_norm_unweighted_inplace(float* x,
                                                            int total_dim,
                                                            int n_heads,
                                                            int head_dim,
                                                            float eps,
                                                            const char* label) {
    if (!x || total_dim <= 0 || n_heads <= 0 || head_dim <= 0) return;

    const int expected = n_heads * head_dim;
    if (expected != total_dim) {
        fprintf(stderr,
            "[nexus] WARNING: %s unweighted RMSNorm dim mismatch: total_dim=%d heads=%d head_dim=%d expected=%d. Skipping.\n",
            label ? label : "v_norm",
            total_dim,
            n_heads,
            head_dim,
            expected);
        fflush(stderr);
        return;
    }

    for (int h = 0; h < n_heads; ++h) {
        float* row = x + (size_t)h * (size_t)head_dim;

        double ss = 0.0;
        for (int i = 0; i < head_dim; ++i) {
            ss += (double)row[i] * (double)row[i];
        }

        const float inv = 1.0f / std::sqrt((float)(ss / (double)head_dim) + eps);

        for (int i = 0; i < head_dim; ++i) {
            row[i] *= inv;
        }
    }
}

inline void nx_apply_qkv_extras(const TransformerLayerWeights& lw,
                                const AttentionConfig& cfg,
                                float* q, float* k, float* v,
                                int q_dim, int kv_dim,
                                float eps,
                                bool gemma4_v_unweighted_norm = false) {
    // Gemma 4 Shared KV Cache: on shared layers the k/v buffers contain
    // garbage (we skipped Wk/Wv). Apply bias+norm only to Q. K/V will be
    // sourced from the donor's KV cache later, and that data is already
    // normalised+RoPE'd from when the donor stored it.
    const bool shared_kv = (lw.kv_donor_layer >= 0);

    nx_add_bias_inplace(q, lw.bq, q_dim);
    if (!shared_kv) {
        nx_add_bias_inplace(k, lw.bk, kv_dim);
        nx_add_bias_inplace(v, lw.bv, kv_dim);
    }

    // Per-tensor convention. lw.attn_q_norm_zc and lw.attn_k_norm_zc were
    // decided at load time based on each tensor's own weight statistics.
    // The legacy bool argument is ignored — it predates per-tensor detection.
    nx_apply_projection_norm_inplace(q,
                                     lw.attn_q_norm,
                                     lw.attn_q_norm_len,
                                     q_dim,
                                     cfg.n_heads,
                                     cfg.head_dim,
                                     eps,
                                     "q_norm",
                                     lw.attn_q_norm_zc);

    if (!shared_kv) {
        nx_apply_projection_norm_inplace(k,
                                         lw.attn_k_norm,
                                         lw.attn_k_norm_len,
                                         kv_dim,
                                         cfg.n_kv_heads,
                                         cfg.head_dim,
                                         eps,
                                         "k_norm",
                                         lw.attn_k_norm_zc);

        // Gemma4 requires V RMSNorm with NO learned weight.
        // Reference path:
        //   if Value exists: v = Value(hidden)
        //   else:            v = raw K projection
        //   k = KeyNorm(k)
        //   v = RMSNorm(v, nil)
        if (gemma4_v_unweighted_norm && v) {
            nx_apply_projection_rms_norm_unweighted_inplace(v,
                                                            kv_dim,
                                                            cfg.n_kv_heads,
                                                            cfg.head_dim,
                                                            eps,
                                                            "gemma4_v_norm");
        }
    }
}

inline void nx_apply_rope_gemma4_prope_or_full(float* x,
                                               int n_heads,
                                               int pos,
                                               int head_dim,
                                               const RoPEConfig& base_cfg,
                                               bool gemma4_global_prope,
                                               const float* rope_factors,
                                               size_t rope_factors_len) {
    if (!x || n_heads <= 0 || head_dim <= 0) return;

    // Non-Gemma4/global path: leave normal models exactly alone.
    if (!gemma4_global_prope) {
        model::rope_apply_all(x, n_heads, pos, head_dim, base_cfg);
        return;
    }

    // Gemma4 global attention:
    // IMPORTANT: use the FULL global head dim here.
    //
    // GGUF rope_freqs.weight is the p-RoPE mask/factor table:
    //   factor ~= 1     -> rotate this RoPE pair
    //   factor ~= 1e30  -> effectively disable rotation for this pair
    //
    // So do NOT hard-cut to head_dim / 4 here. The factor tensor already
    // decides which pairs rotate and which pairs stay unchanged.
    const int full_head_dim = head_dim;      // expected 512 on Gemma4 global
    int rotary_dim = head_dim;               // use full global RoPE dim
    rotary_dim &= ~1;

    const int pairs = rotary_dim / 2;
    const int half  = rotary_dim / 2;

    if (rotary_dim <= 0 || rotary_dim > head_dim || pairs <= 0) {
        fprintf(stderr,
            "[nexus] WARNING: bad Gemma4 p-RoPE rotary_dim=%d head_dim=%d; falling back to full RoPE\n",
            rotary_dim,
            head_dim);
        fflush(stderr);
        model::rope_apply_all(x, n_heads, pos, head_dim, base_cfg);
        return;
    }

    const float base  = base_cfg.freq_base;
    const float scale = base_cfg.scale != 0.0f ? base_cfg.scale : 1.0f;

    size_t active_factor_pairs = 0;
    if (rope_factors) {
        const size_t n = std::min((size_t)pairs, rope_factors_len);
        for (size_t i = 0; i < n; ++i) {
            const float f = rope_factors[i];
            if (std::isfinite(f) && f != 0.0f && f <= 1.0e20f) {
                ++active_factor_pairs;
            }
        }
    }

    static bool printed = false;
    if (!printed) {
        printed = true;
        fprintf(stderr,
            "[nexus] Gemma4 p-RoPE factors: rotary_dim=%d full_head_dim=%d pairs=%d active_pairs=%zu theta=%.0f factors=%s n=%zu layout=half_split\n",
            rotary_dim,
            full_head_dim,
            pairs,
            active_factor_pairs,
            base,
            rope_factors ? "yes" : "no",
            rope_factors_len);
        fflush(stderr);
    }

    NX_TRACE_DECISION(-1, "rope_gemma4_prope.rotary_dim", "%d", rotary_dim);
    NX_TRACE_DECISION(-1, "rope_gemma4_prope.full_head_dim", "%d", full_head_dim);
    NX_TRACE_DECISION(-1, "rope_gemma4_prope.factors", "%s", rope_factors ? "yes" : "no");
    NX_TRACE_DECISION(-1, "rope_gemma4_prope.factors_len", "%zu", rope_factors_len);
    NX_TRACE_DECISION(-1, "rope_gemma4_prope.layout", "%s", "half_split");

    for (int h = 0; h < n_heads; ++h) {
        float* row = x + (size_t)h * (size_t)head_dim;

        // Half-split / rotate_half on the rotary slice only:
        // pair row[i] with row[i + rotary_dim/2].
        for (int i = 0; i < pairs; ++i) {
            float factor = 1.0f;

            // Most GGUF Gemma4 converters store rope_freqs.weight per rotary pair.
            if (rope_factors && (size_t)i < rope_factors_len) {
                factor = rope_factors[i];

                // Guard bad values; keep sane fallback instead of poisoning RoPE.
                if (!std::isfinite(factor) || factor == 0.0f) {
                    factor = 1.0f;
                }
            }

            // Very huge factor means effectively no rotation for that frequency.
            if (factor > 1.0e20f) {
                continue;
            }

            const float freq =
                1.0f / powf(base, (float)(2 * i) / (float)full_head_dim);

            const float theta = (float)pos * freq * scale / factor;
            const float c = cosf(theta);
            const float s = sinf(theta);

            const float x0 = row[i];
            const float x1 = row[i + half];

            row[i]        = x0 * c - x1 * s;
            row[i + half] = x1 * c + x0 * s;
        }
    }
}

// lm_head on the GPU for native NQ weights. Measured on a GTX 1060: the NQ5 decode shader is still slow (moving lm_head to the GPU
// made a TinyLlama token 38 ms -> 74 ms), so only NQ4 (the fast v3 kernel) goes to the GPU by default. NEXUS_LMHEAD_GPU=1 forces
// every NQ type onto the GPU, NEXUS_LMHEAD_CPU=1 forces the CPU. This default flips once the other NQ kernels are rewritten.
// float -> IEEE half, round to nearest even, with subnormals (the KV-cache helper truncates and flushes them, which would zero tiny Q8_0 block scales)
inline uint16_t nx_f32_to_f16_rne(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u; x &= 0x7FFFFFFFu;
    if (x >= 0x7F800000u) return (uint16_t)(sign | 0x7C00u | (x > 0x7F800000u ? 0x200u : 0u));   // inf / nan
    if (x >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);                                       // overflow -> inf
    if (x < 0x33000001u) return (uint16_t)sign;                                                    // underflow -> 0 (below half the smallest subnormal)
    if (x < 0x38800000u) {                                                                         // subnormal half
        const int e = (int)(x >> 23);
        const uint32_t m = (x & 0x7FFFFFu) | 0x800000u;
        const int sh = 126 - e;                                                                    // 14..24
        uint32_t r = m >> sh; const uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t)(sign | r);
    }
    uint32_t r = ((x - 0x38000000u) >> 13); const uint32_t rem = x & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;                                        // carries into the exponent correctly
    return (uint16_t)(sign | r);
}

// Re-encode a K-quant matrix as Q8_0 (host side, multi-threaded).  The extra rounding error (<= half a step of amax/127 per 32-block) is far below the
// quantisation error the source already carries; the point is speed: the Q8_0 decode kernel streams at ~100 GB/s vs ~45-50 GB/s for the K-quant ones.
inline bool nx_requant_to_q8_0(const void* src, QuantType t, size_t rows, size_t cols, int n_threads, std::vector<uint8_t>& out) {
    if (!src || cols == 0 || cols % 32 != 0 || rows == 0) return false;
    const size_t bpr = cols / 32, row_bytes = bpr * sizeof(quant::BlockQ8_0);
    try { out.assign(rows * row_bytes, 0); } catch (...) { out.clear(); return false; }
    const int nt = (int)std::max<size_t>(1, std::min<size_t>((size_t)std::max(1, n_threads), 64));
    std::atomic<size_t> next{0};
    auto work = [&]() {
        std::vector<float> tmp(cols);
        for (;;) {
            const size_t r0 = next.fetch_add(256); if (r0 >= rows) break;
            const size_t r1 = std::min(rows, r0 + 256);
            for (size_t r = r0; r < r1; ++r) {
                quant::dequantize_row(src, tmp.data(), r, cols, t);
                quant::BlockQ8_0* blk = (quant::BlockQ8_0*)(out.data() + r * row_bytes);
                for (size_t b = 0; b < bpr; ++b) {
                    const float* x = tmp.data() + b * 32;
                    float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
                    const uint16_t d16 = nx_f32_to_f16_rne(amax / 127.f);
                    const float d = quant::f16_to_f32(d16), inv = d > 0.f ? 1.f / d : 0.f;
                    blk[b].d = d16;
                    for (int i = 0; i < 32; ++i) { long q = std::lrintf(x[i] * inv); blk[b].qs[i] = (int8_t)std::max(-127L, std::min(127L, q)); }
                }
            }
        }
    };
    std::vector<std::thread> th; for (int i = 1; i < nt; ++i) th.emplace_back(work);
    work(); for (auto& t2 : th) t2.join();
    return true;
}

inline bool nx_lm_head_gpu_ok(QuantType t) {
    const bool is_nq = (t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
                        t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64);
    if (!is_nq) return true;
    static const bool force_cpu = (std::getenv("NEXUS_LMHEAD_CPU") != nullptr);
    static const bool force_gpu = (std::getenv("NEXUS_LMHEAD_GPU") != nullptr);
    if (force_cpu) return false;
    if (force_gpu) return true;
    return t == QuantType::NQ4_G64;
}

inline void nx_apply_rope_qwen35_partial_or_full(float* x,
                                                 int n_heads,
                                                 int pos,
                                                 int head_dim,
                                                 const RoPEConfig& base_cfg,
                                                 bool qwen35_partial) {
    if (!x || n_heads <= 0 || head_dim <= 0) return;

    NX_TRACE_DECISION(-1, "rope_qwen35_partial.partial_flag",
                      "%d", qwen35_partial ? 1 : 0);
    NX_TRACE_DECISION(-1, "rope_qwen35_partial.base_layout",
                      "%d", base_cfg.layout);

    if (!qwen35_partial) {
        model::rope_apply_all(x, n_heads, pos, head_dim, base_cfg);
        return;
    }

    // Qwen3-Next / Qwen3.5 full-attention layers use partial RoPE.
    // IMPORTANT: preserve the engine's existing RoPE layout. Do not force
    // half-split here. The previous forced-layout version made output worse.
    int rotary_dim = head_dim / 4;
    if (head_dim == 256) rotary_dim = 64;

    if (rotary_dim <= 0 || rotary_dim > head_dim || (rotary_dim & 1)) {
        fprintf(stderr,
                "[nexus] WARNING: bad Qwen35 partial RoPE dim rotary=%d head_dim=%d; falling back to full RoPE\n",
                rotary_dim,
                head_dim);
        model::rope_apply_all(x, n_heads, pos, head_dim, base_cfg);
        return;
    }

    RoPEConfig rope_cfg = base_cfg;
    // HF Qwen3-Next uses rotate_half (NeoX-style) RoPE on the rotary slice,
    // not interleaved-pairs. Force layout=1 here regardless of engine default.
    rope_cfg.layout = 1;

    static bool printed = false;
    if (!printed) {
        printed = true;
        fprintf(stderr,
                "[nexus] Qwen35 partial RoPE enabled: rotary_dim=%d head_dim=%d layout=%d (forced rotate_half)\n",
                rotary_dim,
                head_dim,
                rope_cfg.layout);
    }

    for (int h = 0; h < n_heads; ++h) {
        model::rope_apply(x + (size_t)h * head_dim, pos, rotary_dim, rope_cfg);
    }
}

inline void nx_apply_attn_gate_fast(const TransformerLayerWeights& lw,
                                    const float* gate_input,
                                    float* attn_out,
                                    float* gate_tmp,
                                    int q_dim,
                                    int hidden,
                                    int n_threads,
                                    gpu::VulkanBackend* g) {
    if (!lw.attn_gate || !gate_input || !attn_out || !gate_tmp) return;

    bool ok = false;
    if (g && g->ready()) {
        ok = g->mat_vec(lw.attn_gate, gate_input, gate_tmp,
                        (size_t)q_dim, (size_t)hidden, lw.attn_gate_type);
    }
    if (!ok) {
        cpu::mat_vec(lw.attn_gate, gate_input, gate_tmp,
                     (size_t)q_dim, (size_t)hidden,
                     lw.attn_gate_type, n_threads);
    }

    for (int i = 0; i < q_dim; ++i) {
        float z = gate_tmp[i];
        float s;
        if (z > 80.0f) s = 1.0f;
        else if (z < -80.0f) s = 0.0f;
        else s = 1.0f / (1.0f + expf(-z));
        attn_out[i] *= s;
    }
}

inline float nx_sigmoid_f32(float x) {
    if (x > 80.0f) return 1.0f;
    if (x < -80.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

inline float nx_softplus_f32(float x) {
    if (x > 40.0f) return x;
    if (x < -40.0f) return expf(x);
    return log1pf(expf(x));
}

// GELU with tanh approximation. Used by Gemma 1/2/3/4 and PaLM-family models.
//   gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
//
// HuggingFace calls this `gelu_pytorch_tanh`. It is NOT the same as SiLU
// (x * sigmoid(x)) used by Llama/Mistral/Qwen, and NOT the same as exact
// GELU (uses erf instead of tanh approximation). Wrong activation
// here compounds across 60+ layers and produces collapse-to-single-token output.
inline void nx_gelu_tanh_inplace(float* x, size_t n) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kFactor      = 0.044715f;
    for (size_t i = 0; i < n; ++i) {
        const float xi   = x[i];
        const float xi3  = xi * xi * xi;
        const float in   = kSqrt2OverPi * (xi + kFactor * xi3);
        x[i] = 0.5f * xi * (1.0f + tanhf(in));
    }
}

inline void nx_l2norm_inplace(float* x, int n, float eps = 1e-6f) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

inline void nx_qwen35_gated_rmsnorm(float* x,
                                    const float* z,
                                    const float* w,
                                    int n,
                                    float eps,
                                    bool zero_centered_weight = false) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];

    float inv = 1.0f / sqrtf(ss / (float)n + eps);

    for (int i = 0; i < n; ++i) {
        float zi = z[i];
        float silu = zi / (1.0f + expf(-zi));
        // GatedDeltaNet RMSNormGated uses normal RMSNorm weight semantics.
        // Do not apply Qwen/Gemma zero-centered (1 + w) here.
        float wi = w[i];
        x[i] = x[i] * inv * wi * silu;
    }
}

class Transformer {
public:
    Transformer() = default;
    ~Transformer() { q35_drop_dev(-1); q35_kv_drop_all(); }

    // Load model from GGUF file.
    // Returns false on failure.
    bool load(const std::string& model_path, int n_threads = 1,
              std::function<void(const std::string&, float)> progress_cb = nullptr);

    // Load from an in-memory GGUF blob. Zero disk I/O on all platforms.
    // Uses Parser::open_from_memory() which borrows the buffer without copying.
    bool load_from_memory(const void* data, size_t size, int n_threads = 1,
                          std::function<void(const std::string&, float)> progress_cb = nullptr);

    // Load only source layers [layer_start, layer_end) directly from the original GGUF.
    // No mini-GGUF, no memfd/temp file, no full shard staging copy.
    bool load_range_direct(const std::string& model_path, int layer_start, int layer_end,
                           int n_threads = 1,
                           std::function<void(const std::string&, float)> progress_cb = nullptr);

    // True single-layer streaming pair, for hybrid prefill's "ghost" donor Engine.
    // Call open_for_layer_streaming() exactly once (opens the file, loads globals +
    // tokenizer, sizes KV cache + scratch once), then swap_layer_direct() as many
    // times as needed -- each call frees the previous layer's weights and loads
    // just the new one, never touching the file header, tokenizer, or global
    // tensors again. See the long comment above each implementation for why this
    // is safe and what it deliberately does NOT redo on every swap.
    // probe_layer_start/probe_layer_end: the layer range this ghost engine will actually
    // stream (e.g. the donor's split_layer_..all_layers). Before the KV cache is sized,
    // every layer in this range is swapped in once (cheap: one memcpy, no file/tokenizer
    // reload) so its REAL per-layer attention geometry (linear/recurrent vs full, and its
    // real kv_heads/head_dim) is known -- without this, an architecture with per-layer-
    // varying attention (Qwen3Next's interleaved linear-attention layers) gets a WRONG,
    // uniform KV cache shape, and every layer whose real shape differs from that guess
    // fails to apply on the receiving side. -1 (the default) probes the whole model.
    bool open_for_layer_streaming(const std::string& model_path, int n_threads = 1,
                                  int probe_layer_start = 0, int probe_layer_end = -1);
    bool swap_layer_direct(int new_layer, int prev_layer);

    // *** v29: for an engine that owns a PERSISTENT range of layers (the primary,
    // running its own decode) rather than a layer-streaming ghost -- temporarily frees
    // ONE already-loaded layer's weight tensors (parser_/mm_ already open; KV cache and
    // every other layer's residency untouched) so the freed memory can make room for
    // something else, then brings it back later with reload_resident_layer(). Unlike
    // swap_layer_direct(), this never touches active_layer_start_/active_layer_end_ --
    // the caller still owns its whole normal range the entire time, just with this one
    // layer's weights absent in between the two calls. The caller MUST reload every
    // layer it frees before running forward()/prefill on it again -- a freed layer's
    // weight pointers are all null, same as a layer that was never loaded.
    bool free_resident_layer(int l);
    bool reload_resident_layer(int l);

    // ---- Layer parking: "Maximize GPU" and oversized-prompt protection ---------------------------------------
    // A PARKED layer has no copy in system RAM. Its home for writing (decode) is the graphics card. Before a prompt
    // is read, parked layers are moved off the card only if the card lacks the room the prompt speed-up needs; those are read from the model file like any other streamed layer, and go back on the card before
    // writing starts (a pause while they are read from the drive). Needs the prompt speed-up (turbo streaming) on a
    // discrete card and the big layer weights mapped in place from the model file (the server arranges both).
    //   park_layers     : -1 = as many as fit on the card, 0 = none at startup, N = that many (from layer 0 up)
    //   below_mb        : > 0 = while a prompt is read, whenever free RAM is under this, drop more layers' RAM copies
    //                     (layers already finished for this prompt first); they go onto the card after the prompt
    //   keep_mb         : VRAM always left free when putting layers on the card (KV mirror, activations)
    //   prefill_free_mb : card memory to have free while a prompt is read; 0 = auto (exactly what the prompt
    //                     speed-up needs). Layers move off only when that much is not free already.
    //   min_tokens      : 0 (default) = no token rule; N = prompts shorter than N never move layers
    bool park_setup(int park_layers, int below_mb, int keep_mb, int prefill_free_mb, int min_tokens);
    bool park_active() const { return park_on_; }
    // Called by the engine after a prompt has been read (and its buffers freed): parked layers back onto the card.
    void park_finish_pending();
    // Called every 128 tokens while writing: if the card is about to run out of real memory, one parked layer moves
    // off it (into RAM if there's room, else read from the file); if free RAM is under the protection threshold, one
    // more layer leaves RAM and goes onto the card if it fits. Cheap when nothing needs doing.
    void park_watch();
    // While layers are being moved onto the card: how many of how many (false when nothing is moving).
    bool park_moving(int& done, int& total) const {
        total = park_mv_total_.load(); done = park_mv_done_.load(); return total > 0;
    }

    // Load a streamed shard without ever assembling a full fake GGUF in RAM/disk.
    // gguf_header is metadata + selected tensor descriptors only. read_fn is called
    // once per selected tensor to fill final owned weight storage directly.
    bool load_streamed_shard(const void* gguf_header, size_t gguf_header_size,
                             int n_threads,
                             std::function<bool(void*, size_t)> read_fn,
                             std::function<void(const std::string&, float)> progress_cb = nullptr);

    // Generate one token given the current context.
    // Returns the token ID of the next token.
    int32_t forward(const std::vector<int32_t>& tokens, int pos);

    // Sample from logits (call after forward)
    int32_t sample(const SamplerConfig& cfg);
    // Sample from an EXTERNAL logits vector using the exact same sampler path
    // (temporarily swaps logits_). Used by MTP verify to sample the draft-check
    // and bonus-token distributions without disturbing the main logits_.
    int32_t sample_from(std::vector<float>& ext_logits, const SamplerConfig& cfg) {
        std::vector<float> saved;
        saved.swap(logits_);
        logits_.swap(ext_logits);
        int32_t t = sample(cfg);
        logits_.swap(ext_logits);
        saved.swap(logits_);
        return t;
    }

    // Encode text to token IDs
    std::vector<int32_t> tokenize(const std::string& text, bool add_bos = true) const;

    // Decode token ID to string
    std::string detokenize(int32_t token_id) const;
    std::string detokenize(const std::vector<int32_t>& tokens) const;

    // Reset KV cache (new conversation)
    void reset_context();
    
    // Layer-major prefill: feeds tokens[start..start+n) through the model
    // by inverting the loops — for each layer, iterate all tokens, instead
    // of for each token, iterate all layers.
    //
    // Identical semantics to:
    //     for (int i = 0; i < n; ++i) forward(tokens, start + i);
    // The KV cache ends up with the same contents and logits_ holds the
    // logits at the last position (start+n-1) so a follow-up sample()
    // returns the correct first generated token.
    //
    // Why this exists:
    //   (a) Each layer's weights are touched once per batch instead of
    //       once per token (better mlock/madvise/page-fault behavior).
    //   (b) Foundation for GPU weight streaming (later: upload layer L+1
    //       while compute runs on layer L).
    //   (c) lm_head/final-norm runs only at the last position — saves
    //       (N-1) lm_head matmuls vs the per-token path.
    //
    // Falls back to per-token forward() on Gemma 4 (per-token PLE prevents
    // straightforward batching). Qwen35/Qwen3Next Gated DeltaNet works
    // natively — its recurrent state evolves correctly because tokens are
    // still presented in order within each layer.
    void prefill_batch(const std::vector<int32_t>& tokens, int start_pos, int n,
                       bool force_no_stream = false, bool keep_prefill_scratch = false);

    // Normal CPU prompt prefill: one full-prompt batched prefill call, but
    // WITHOUT turbo GPU streaming, resident-window layer loading, or eviction.
    // This is the default CPU route when Turbo Prefill is OFF.
    void prefill_cpu_prompt_batch(const std::vector<int32_t>& tokens,
                                  int start_pos, int n, bool keep_prefill_scratch = false);

    // Distributed prefill: run ONLY layers [layer_start, layer_end) over n prompt tokens in one
    // batched pass (same kernels as prefill_batch / prefill_cpu_prompt_batch).
    //   layer_start == 0 : tokens are embedded; x_in must be null
    //   layer_start  > 0 : x_in = n*hidden floats from the previous machine
    //   x_out (optional) : receives n*hidden floats after layer_end-1
    // No final norm / lm_head is run. Returns false if unsupported (Gemma PLE) or the range is not owned.
    bool prefill_layer_range(const std::vector<int32_t>& tokens, int start_pos, int n,
                             int layer_start, int layer_end,
                             const float* x_in, float* x_out, bool use_turbo,
                             bool keep_prefill_scratch = false);
    bool range_prefill_ok() const { return !ple_enabled_; }
    bool pr_quiet_ = true;                    // no per-layer "[progress]" lines from a range prefill (the distributed code prints its own)
    std::atomic<int> pr_layers_done_{0};      // layers finished so far in the current range prefill (for progress reports)
    // Prompt-reading progress for ANY prefill path (all of them run batchable layers through layer_batch_tiled):
    // which layer of how many, and how far through that layer's tokens. Read by the server's progress reporter.
    std::atomic<int> pf_layer_{-1}, pf_layers_{0}, pf_tok_done_{0}, pf_tok_total_{0};
    std::atomic<bool> pf_external_{false};    // the server reports progress itself: no "[progress]" lines from here

    // Enable GPU layer offloading.
    // Call after load(). n=0 disables GPU. n=-1 offloads all layers.
    // Returns the number of layers actually offloaded (may be less if GPU init fails).
    int set_gpu_layers(int n);
    
    // Enable/disable turbo GPU streaming for prefill_batch. Only takes effect
    // when prefill_batch is also active (Engine::set_turbo_prefill(true)).
    // Set the backend's weight cache budget separately via the engine.
    void set_turbo_gpu_stream(bool e) { turbo_gpu_stream_ = e; }
    bool turbo_gpu_stream() const     { return turbo_gpu_stream_; }
    // Prompt-reading matrix multiplies run on the card for any layer whose weights live there -- turbo prefill is only
    // needed to SWAP weights in for layers that do not live there.
    // While a prompt is streamed every layer is flagged for the prompt card (gpu_layer = true), so this is that card. Otherwise --
    // the MTP check of a few drafted tokens, a short batch -- it is the card the layer lives on (card 2/3/4 included), or none.
    // Before, any batch with the prompt speed-up switched on sent EVERY layer to the prompt card: the MTP check then uploaded
    // the whole model through it, evicting as it went (~5.7 s per check of 4 tokens).
    gpu::VulkanBackend* prefill_card_for_layer(int layer_idx) const {
        if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size()) return nullptr;
        gpu::VulkanBackend* c = layer_card(layer_weights_[layer_idx]);
        return (c && c->ready()) ? c : nullptr;
    }
    bool prefill_on_gpu_for_layer(int layer_idx) const { return prefill_card_for_layer(layer_idx) != nullptr; }

    // Secondary GPU (iGPU / second device).
    // Assigns layers [first_layer, first_layer+n) to a second Vulkan device.
    int set_gpu2_layers(int n, int device_index, int first_layer);
    // 3rd, 4th, 5th ... graphics card: layers [first_layer, first_layer+n). Each call adds one more card.
    int add_gpu_layers(int n, int device_index, int first_layer);
    // n < 0 for set_gpu2_layers / add_gpu_layers = "Maximize": as many layers as fit on that card, measured from what
    // the card really has free (when the driver reports it), keeping this much free (default 512 MB).
    void set_card_fill_keep_mb(int mb) { card_fill_keep_bytes_ = (size_t)std::max(64, mb) << 20; }
    // MTP drafting on the graphics card runs on the prompt card and puts its weights there the first time it drafts --
    // after Maximize GPU has filled the card. With this on, Maximize leaves that much room for it.
    void set_mtp_card_reserve(bool on) { mtp_card_reserve_ = on; }

    // GPU info (empty if no GPU)
    gpu::GpuInfo gpu_info()      const;
    std::string  gpu2_device_name() const { return gpu2_device_name_; }
    std::string  gpux_device_name(int i) const { return (i >= 0 && (size_t)i < gpux_names_.size()) ? gpux_names_[(size_t)i] : std::string(); }

    // Memory report
    void print_memory_report() const;
    // One-line KV vs process-memory report; also hands freed heap (batch scratch
    // from big prefills) back to the OS. Safe to call any time between forwards.
    void kv_report_and_trim(const char* tag);

    // Model properties
    const gguf::ModelMeta& meta() const { return meta_; }
    int vocab_size()    const { return (int)meta_.vocab_size; }
    int n_layers()      const { return (int)meta_.n_layers; }
    int hidden_dim()    const { return (int)meta_.hidden_dim; }
    int context_length() const { return (int)meta_.context_length; }
    bool is_loaded()    const { return loaded_; }

    // RoPE position stabilizer: non-destructive KV position surgery.
    // It keeps every token in KV, but compresses the RoPE coordinate space when
    // long absolute positions start hurting GPU/long-context stability.
    void set_rope_stabilizer(bool enabled, int trigger_effective_tokens, int target_effective_tokens);
    bool rope_stabilizer_enabled() const { return rope_stabilizer_enabled_; }
    int  rope_stabilizer_trigger() const { return rope_stabilizer_trigger_; }
    int  rope_stabilizer_target()  const { return rope_stabilizer_target_; }
    float rope_position_scale() const { return rope_pos_scale_; }
    void set_rope_position_scale_state(float scale);
    int  rope_effective_position(int actual_pos) const;
    bool maybe_rope_stabilize(int actual_pos, const char* reason);
    bool rope_rescale(float new_scale, const char* reason);

    // ── Context editing: remove / export / import a range of the KV cache ─────────
    // kv_remove_range: drop tokens [start, end) from every layer's KV and shift the keys of
    //   everything after the cut back into place (RoPE), so the kept text sits exactly where the
    //   model would have stored it. Exact for attention layers; see has_recurrent_state().
    // kv_export_range / kv_import_append: copy [start, end) out as a self-describing blob, and
    //   append such a blob at the end of the current context (keys re-rotated to the new spot).
    bool   kv_remove_range(int start, int end);
    bool   kv_export_range(int start, int end, std::vector<uint8_t>& out) const;
    int    kv_import_append(const uint8_t* data, size_t size, std::string* err);
    // True when some layers keep a running (recurrent) state -- Qwen3-Next / Qwen3.5 Gated
    // DeltaNet. That state can't have individual tokens taken out, so for these models removing
    // part of the conversation is exact in the attention layers but only approximate overall.
    bool   has_recurrent_state() const;
    size_t kv_bytes_per_token() const { return kvcache_.bytes_per_token(); }
    size_t kv_capacity_tokens() const { return kvcache_.min_capacity_tokens(); }
    // Rotate the cached keys of positions [from, to) by `delta` positions in every layer.
    bool   kv_rotate_positions(int from, int to, int delta);
    // Apply layer `layer_idx`'s own RoPE (same config the forward pass uses for keys) to one head.
    void   rope_apply_layer_one_head(int layer_idx, float* x, int pos) const;
    // After any KV edit: GPU copies are stale, MTP's short history no longer lines up.
    void   kv_after_edit();

    // Raw logits from last forward pass (for debug/inspection)
    const std::vector<float>& logits() const { return logits_; }

    // Last hidden state (before lm_head projection) --- for text embeddings/RAG
    // Returns reference to internal buffer: copy if needed across forward() calls
    const std::vector<float>& last_hidden() const { return x_scratch_; }

    // Forward pass with embed override --- injects precomputed embedding vector
    // at the given position instead of looking up token_embedding_table.
    // Used to inject vision/audio tokens into the model's token stream.
    // embed must be [hidden_dim] floats.
    bool token_embedding(int32_t token, float* out) const;
    int32_t forward_embed(const float* embed, int pos);
    int32_t forward_embed_prefill(const float* embed, int pos);
    void prefill_embeddings_stream(const float* embeddings, int hidden_dim,
                                   int start_pos, int n,
                                   bool force_no_stream = false);
    void prefill_embeddings_cpu_prompt_batch(const float* embeddings, int hidden_dim,
                                             int start_pos, int n);

    // ------ LoRA adapter support ---------------------------------------------------------------------------------------------------------------------------------------------
    // Load a LoRA adapter from a GGUF file. Multiple adapters can be stacked.
    // scale: 0.0 = disabled, 1.0 = full strength
    bool   lora_load(const std::string& path, float scale = 1.0f);
    void   lora_unload_all();
    void   lora_set_scale(int adapter_idx, float scale);
    int    lora_count() const;

    // ------ Speculative decoding ---------------------------------------------------------------------------------------------------------------------------------------------
    // Run n_draft tokens through a smaller draft model, verify against this
    // (target) model's distribution. Accepted tokens are free --- only rejected
    // tokens need a target model pass. Typical speedup: 2-4x for aligned models.
    // draft_xfmr: pointer to a loaded Transformer (the small fast model)
    // Returns tokens generated (same as generate_stream but faster)
    using TokenCB = std::function<bool(const std::string&, int32_t)>;
    void generate_speculative(
        const std::vector<int32_t>& prompt_tokens,
        Transformer*                draft_model,
        const SamplerConfig&        cfg,
        int                         max_tokens,
        int                         n_draft,      // tokens to speculate per step (4-8)
        TokenCB                     on_token);

private:
    bool loaded_ = false;
    int  n_threads_ = 1;
    // reusable prefill scratch (grow-only, freed by release_prefill_scratch() once the prompt is done)
    std::vector<float> pf_xb_, pf_attn_, pf_ffn_, q35s_mixed_, q35s_z_, q35s_a_, q35s_b_, q35s_ba_, q35s_conv_, q35s_core_;
    void release_prefill_scratch() {
        for (auto* v : {&pf_xb_, &pf_attn_, &pf_ffn_, &q35s_mixed_, &q35s_z_, &q35s_a_, &q35s_b_, &q35s_ba_, &q35s_conv_, &q35s_core_}) std::vector<float>().swap(*v);
    }
    int  n_gpu_layers_  = 0;  // layers on primary GPU
    // Turbo GPU streaming: during prefill_batch only, temporarily flips
    // every layer to GPU and lets the bounded weight cache stream them
    // in/out of VRAM. Decode falls back to whatever set_gpu_layers() chose.
    bool turbo_gpu_stream_ = false;
    size_t card_fill_keep_bytes_ = 768ull << 20;
    int layers_that_fit(gpu::VulkanBackend& b, int first_layer, const char* card_name);
    // Layer parking (see park_setup)
    struct ParkRange { uint8_t* p; size_t n; };
    bool park_on_ = false;
    bool park_pending_ = false;                          // a long prompt moved layers off the card; put them back
    std::vector<std::vector<ParkRange>> park_ranges_;    // per layer: weight bytes mapped in place from the model file
    std::vector<uint8_t> park_state_;                    // 0 in RAM, 1 parked (no RAM copy, read from the file), 2 parked and on the card
    std::vector<uint8_t> park_home_card_;                // 1 = Maximize GPU chose this layer for the card; 0 = its home is RAM
    std::vector<std::vector<const void*>> park_gpu_w_;   // per layer: weights pinned on the card
    long long park_below_bytes_ = 0;
    size_t park_keep_bytes_ = 768ull << 20;
    bool   mtp_card_reserve_ = false;
    size_t mtp_reserve_bytes_ = 0;     // room kept on the prompt card for the MTP weights until they are on it
    size_t park_prefill_free_bytes_ = 0;                 // 0 = auto: what the prompt stream needs
    int  park_min_tokens_ = 0;
    size_t park_stream_needs(int ctx_tokens = 0) const;
    size_t park_card_room() const;                      // card memory free for more layers (real reading when the driver gives one)
    long long park_ram_floor() const;                   // free RAM to keep: threshold (or 512 MB) + room for the reply's KV
    bool park_relieve_exhausted_ = false;
    std::atomic<int> park_mv_done_{0}, park_mv_total_{0};
    bool park_on_card(int l) const { return park_on_ && l >= 0 && l < (int)park_state_.size() && park_state_[l] == 2; }
    // Layers placed on the prompt speed-up card ("GPU layers" on the first card) live there for writing AND reading:
    // the prompt is read through them in place and they are never pushed out by the layers streaming past.
    std::vector<uint8_t> turbo_keep_;
    bool turbo_keep_ready_ = false;
    bool turbo_keep_on(int l) const { return turbo_keep_ready_ && l >= 0 && l < (int)turbo_keep_.size() && turbo_keep_[l]; }
    void turbo_keep_prepare();
    bool park_any_on_card() const { for (auto v : park_state_) if (v == 2) return true; return false; }
    size_t park_layer_bytes(int l) const;
    void park_touch_layer(int l);
    void park_drop_layer(int l);
    int  park_to_gpu(const char* why);
    void park_off_gpu(int n_tokens, int ctx_tokens);
    void park_relieve_ram(int done_upto, int range_start, int range_end);
    bool         pr_active_ = false;       // prefill_layer_range in progress
    int          pr_start_ = 0, pr_end_ = -1;
    const float* pr_x_in_ = nullptr;
    float*       pr_x_out_ = nullptr;
    int          turbo_range_end_ = -1;    // clamp for the GPU weight window while a range prefill runs
    // Turbo resident-window futures. The tail future uploads weights needed after
    // the CPU-only Qwen3Next conv/recurrent scan. The window future keeps a rolling
    // resident GPU layer window loaded ahead of the current layer.
    std::future<void> turbo_prefetch_tail_future_;
    std::future<void> turbo_prefetch_next_future_;
    int  n_gpu2_layers_ = 0;  // layers on secondary GPU
    bool gpu_lm_head_   = false; // true when lm_head is fully offloaded to primary GPU
    // Q6_K decode matrices (qkv, gate, up, ...) re-encoded as Q8_0 on first use: the Q8_0 kernel streams ~2x faster than the Q6_K one on 8192x1024-class matrices.
    // Keyed by source pointer; an empty vector = "not converted" (cached so we do not retry).  NEXUS_Q6_Q8=1 enables (off by default).
    std::unordered_map<const void*, std::vector<uint8_t>> q8swap_;
    size_t q8swap_bytes_ = 0;
    inline void q8swap(const void*& w, QuantType& t, size_t rows, size_t cols);
    std::vector<uint8_t> lm_head_q8_;   // lm_head re-encoded as Q8_0 for decode speed (lm_head_weight/type point here when active)
    bool zero_centered_norm_ = false; // Qwen3-Next style RMSNorm: weight means (1 + weight)

    // RoPE stabilizer state. rope_pos_scale_ maps actual token index -> effective
    // RoPE position. 1.0 means native/exact. Values below 1.0 compress the whole
    // cached conversation into a lower absolute RoPE range without deleting KV.
    bool  rope_stabilizer_enabled_ = false;
    int   rope_stabilizer_trigger_ = 24000;
    int   rope_stabilizer_target_  = 16000;
    float rope_pos_scale_          = 1.0f;
    void  rope_rotate_cached_k_delta(int layer_idx, int pos, int delta_pos);

    // Model-wide max |mean| over all per-layer norm tensors. Populated during
    // build_layer_weights. Kept for the legacy log line; not used to decide.
    double norm_stats_max_abs_ = 0.0;

    // ─── Per-class RMSNorm statistics (model-wide per class) ──────────────
    //
    // The convention used for each norm-tensor class is set by the GGUF
    // converter once for that class — it does NOT vary per layer. Some
    // classes can be raw HF (need x*(1+w)) while OTHERS in the same model
    // are pre-baked (use x*w). Gemma 4 31B Q4_K_M observed in the wild has
    // q_norm pre-baked (avg ~1.0) while k_norm raw HF (avg ~0.12), in the
    // same file. One global flag can't satisfy both; per-layer auto-detect
    // flips noisily within the same tensor class. Per-class is correct.
    struct NormClassStats {
        double sum_mean_abs = 0.0;
        int    n_samples    = 0;
        bool   decided_zc   = false;  // decided after all layers seen
    };
    NormClassStats nc_attn_norm_;
    NormClassStats nc_ffn_norm_;
    NormClassStats nc_attn_post_norm_;
    NormClassStats nc_ffn_post_norm_;
    NormClassStats nc_attn_q_norm_;
    NormClassStats nc_attn_k_norm_;
    // Per-arch scaling/capping. Detected from arch string at load time.
    // 0 = disabled / identity. Set positive to enable.
    float embed_scale_        = 0.0f; // Gemma: sqrt(d_model). 0 = no embed rescale.
    float final_logit_softcap_ = 0.0f; // Gemma 2: 30.0. Gemma 3+, others: 0.
    // attn_softcap is plumbed through AttentionConfig (attn_cfg_.attn_softcap).
    // Dense FFN activation. False = SiLU (Llama/Mistral/Qwen/Phi/most models).
    // True = GELU-tanh (Gemma 1/2/3/4, PaLM family).
    bool ffn_activation_gelu_ = false;
    int  gpu2_first_layer_ = 0;  // first layer index on secondary GPU
    std::string gpu2_device_name_;
    std::unique_ptr<gpu::VulkanBackend> gpu2_;  // secondary GPU backend (iGPU)
    std::vector<std::unique_ptr<gpu::VulkanBackend>> gpux_;  // further graphics cards (3rd, 4th, ...)
    std::vector<std::string> gpux_names_;
    // the backend / readiness for a layer that is not on the primary card
    gpu::VulkanBackend* backend2_for(const TransformerLayerWeights& lw) const {
        if (lw.gpux >= 0) return (size_t)lw.gpux < gpux_.size() ? gpux_[(size_t)lw.gpux].get() : nullptr;
        return gpu2_.get();
    }
    bool gpu2_ready_for(const TransformerLayerWeights& lw) const {
        if (!lw.gpu2_layer) return false;
        gpu::VulkanBackend* b = backend2_for(lw); return b && b->ready();
    }
    // The card a layer is computed on while writing (nullptr = the CPU).
    gpu::VulkanBackend* layer_card(const TransformerLayerWeights& lw) const {
        if (lw.gpu_layer) return &gpu::gpu();
        if (lw.gpu2_layer) return backend2_for(lw);
        return nullptr;
    }
    // Card 2, 3, 4: put layers [first, first+n) on card b now, pinned, every weight the card can take (Qwen3.5 linear-
    // attention matrices included). Stops at the first layer that doesn't fit; returns how many went on.
    int fill_other_card(gpu::VulkanBackend& b, int first, int n, int card_no);
    std::vector<std::vector<const void*>> other_card_w_;   // weights fill_other_card pinned, per layer
    void other_cards_watch();                                // while writing: a card 2/3/4 that is nearly full gives a layer back
    
    void forward_qwen35_linear(const TransformerLayerWeights& lw, float* x, float* out, int layer_idx, int pos);
    bool q35_gpu_linear_step(const TransformerLayerWeights& lw, int l, float* x, bool async = false);   // whole linear-attn + FFN layer on the GPU, one submit
    bool q35_gpu_full_step(const TransformerLayerWeights& lw, int l, float* x, int pos, bool async = false);   // also plain attention layers (see plain_attn_gpu_ok)
    bool plain_attn_gpu_ok() const;
    // layer chaining: consecutive GPU layers are submitted without waiting (the CPU records layer N+1 while the card runs layer N); q35_chain_flush() joins
    struct Q35ChainKv { int layer; int pos; int nkv; int hd; uint32_t mirror_valid; int slot; };
    std::vector<Q35ChainKv> q35_chain_kv_;
    void q35_chain_flush(float* x);   // whole full-attn + FFN layer on the GPU (KV mirrored on the card), one submit

    // Turbo prefill helpers: batch the expensive Qwen3Next projection GEMMs
    // through Vulkan mat_mat when possible, with CPU mat_mat fallback.
    static bool turbo_is_native_nq_type(QuantType t);
    static bool turbo_is_standard_gguf_mm_type(QuantType t);
    static const char* turbo_mm_family_name(QuantType t);
    bool turbo_mat_mat_or_cpu(const void* W, QuantType wt,
                              const float* X, float* Y,
                              size_t rows, size_t cols, size_t n_tokens,
                              const char* tag, int layer_idx);
    bool turbo_mat_mat_family_or_cpu(const void* W, QuantType wt,
                                     const float* X, float* Y,
                                     size_t rows, size_t cols, size_t n_tokens,
                                     const char* tag, int layer_idx,
                                     const char* forced_family);
    void turbo_prefetch_wait(std::future<void>& fut);
    int  turbo_layer_window_size() const;
    // Sum of one layer's NQ4 weight bytes as the turbo loader would upload them.
    size_t turbo_layer_weight_bytes(int layer_idx) const;
    // Compute the exact VRAM the resident window needs (worst-case adjacent-window
    // sum + KV/attention reserve) and set the backend budget. Replaces the
    // launcher's guessed budget when streaming. Returns the budget in bytes.
    size_t compute_and_set_turbo_vram_budget(size_t kv_attn_reserve_bytes, int ctx_tokens = 0);
    void turbo_prefetch_mm_weight(const void* W, QuantType wt,
                                  size_t rows, size_t cols,
                                  const char* tag, int layer_idx,
                                  bool pin_after_load = true);
    void turbo_unpin_evict_mm_weight(const void* W);
    void turbo_load_layer_weights_resident(int layer_idx, const char* prefix, bool pin_after_load);
    void turbo_release_layer_weights_resident(int layer_idx);
    void turbo_prefetch_current_tail_async(int layer_idx);
    void turbo_prefetch_next_layer_async(int layer_idx);
    // Strictly sequential streaming: make layer l the ONLY resident layer, fully
    // uploaded and verified, with the GPU idle, before any compute touches it.
    void turbo_stream_begin_layer(int layer_idx, const char* tag);
    void turbo_stream_end_layer(int layer_idx);
    std::chrono::steady_clock::time_point turbo_stream_t_loaded_{};
    double turbo_stream_load_ms_ = 0.0;
    // GPU layer window: how many layers are resident on the GPU at once (NEXUS_TURBO_GPU_LAYERS).
    // 1 = strictly sequential (default). 2..4 = a background thread uploads the next layer(s) on the
    // transfer queue while the current layer computes.
    static int turbo_gpu_layers();
    struct TurboLoader {
        std::thread t;
        TurboLoader() = default;
        TurboLoader(const TurboLoader&) {}                       // copies start empty
        TurboLoader& operator=(const TurboLoader&) { return *this; }
        void join() { if (t.joinable()) t.join(); }
        ~TurboLoader() { join(); }
    };
    TurboLoader turbo_loader_;
    int  turbo_last_begin_layer_ = -1;
    void forward_qwen35_linear_batch(const TransformerLayerWeights& lw,
                                     const float* xb_batch,
                                     float* out_batch,
                                     int layer_idx,
                                     int start_pos,
                                     int n_tokens);
    void forward_dense_ffn_batch(const TransformerLayerWeights& lw,
                                 const float* xb_batch,
                                 float* out_batch,
                                 int layer_idx,
                                 int n_tokens);
    bool forward_qwen35_prefill_layer_batch(const TransformerLayerWeights& lw,
                                            float* x_batch,
                                            int layer_idx,
                                            int start_pos,
                                            int n_tokens);

    bool forward_full_attn_prefill_layer_batch(const TransformerLayerWeights& lw,
                                               float* x_batch,
                                               int layer_idx,
                                               int start_pos,
                                               int n_tokens);
    bool forward_full_attn_prefill_layer_batch_native_legacy(const TransformerLayerWeights& lw,
                                                             float* x_batch,
                                                             int layer_idx,
                                                             int start_pos,
                                                             int n_tokens);
    // Run one layer's batched prefill over all n tokens in token tiles so that
    // every per-layer temporary (projections, conv/scan buffers, FFN gate/up,
    // attention state) is sized by the tile, not by the whole prompt. KV and the
    // DeltaNet recurrent/conv state carry from one tile to the next exactly as
    // they do between turns. NEXUS_PREFILL_TILE=N sets the tile (default 1024,
    // 0 = whole prompt at once). Returns false only if the layer can't batch at
    // all (caller then does the per-token fallback for the whole layer).
    bool layer_batch_tiled(float* x_buf, int n, int start_pos, int hidden, int l,
                           const std::function<bool(float*, int, int)>& fn);
    void attn_batch_cpu(const float* Q, const uint16_t* K, const uint16_t* V, float* Out,
                        int n_q_heads, int n_kv_heads, int head_dim,
                        int n_ctx, int n_query, int q_base_pos, int sliding_window,
                        float scale, float softcap);

    // Memory manager (owns all weight memory)
    std::unique_ptr<MemoryManager> mm_;

    // GGUF parser (holds tensor descriptors)
    std::unique_ptr<gguf::Parser> parser_;

    // For streamed distributed shards: owns tensor bytes received over the socket.
    // Each buffer is final model weight storage, not a fake GGUF staging blob.
    std::vector<std::unique_ptr<uint8_t[]>> streamed_weight_storage_;

    // Owns converted F32 copies of scalar/vector tensors originally stored as F16/BF16.
    // Pointers handed to layer weights must stay valid for the model lifetime.
    std::vector<std::vector<float>> f32_tensor_storage_;

    // Gemma4 global p-RoPE factors.
    // GGUF loads this as rope_freqs.weight. Current code was loading the tensor
    // but never indexing/using it, so global Gemma4 layers used fallback p-RoPE.
    const float* gemma4_rope_factors_ = nullptr;
    size_t gemma4_rope_factors_len_ = 0;

    const float* tensor_as_f32(TensorDesc* t, const char* label);

    // Model metadata
    gguf::ModelMeta meta_;

    // Per-layer weights
    std::vector<TransformerLayerWeights> layer_weights_;

    // Distributed shard load: when >= 0, only this original source-layer
    // interval is required to have weights. Keep original layer IDs instead
    // of remapping worker layers to zero so KV, lm_head detection, and any
    // layer-index-specific logic use the same numbering as the full model.
    int active_layer_start_ = 0;
    int active_layer_end_   = -1;

    // Global weights
    const void*  token_embedding_table = nullptr; QuantType emb_type{};
    const float* rms_final_weight      = nullptr;
    const float* rms_final_bias        = nullptr;
    const void*  lm_head_weight        = nullptr; QuantType lm_head_type{};
    uint64_t     lm_head_rows_ = 0;        // rows actually present in the output tensor (0 = unknown)
    const float* lm_head_bias          = nullptr;

    // ─── Gemma 4: Per-Layer Embeddings (PLE) ────────────────────────────────
    //
    // PLE adds an auxiliary residual to each decoder layer. For each token T
    // and layer L we produce a small vector ple_dim_-wide that gets added back
    // into the hidden stream after attention + FFN.
    //
    // Two sources are summed and scaled by 1/sqrt(2):
    //   (a) token-identity:   embed_tokens_per_layer[T] * sqrt(ple_dim_)
    //   (b) context-aware:    per_layer_model_projection(x) * 1/sqrt(hidden)
    //                         then RMSNorm with per_layer_projection_norm
    //
    // GGUF tensor names (HF → GGUF convention; aliases tolerated):
    //   per_layer_token_embd.weight              [vocab, n_layers * ple_dim]
    //   per_layer_model_projection.weight        [n_layers * ple_dim, hidden]
    //   per_layer_projection_norm.weight         [ple_dim]
    //
    // For non-Gemma-4 models these stay null and PLE is skipped at runtime.
    const void*  ple_token_embed_    = nullptr;  QuantType ple_token_embed_type_ {};
    const void*  ple_model_proj_     = nullptr;  QuantType ple_model_proj_type_ {};
    const float* ple_proj_norm_      = nullptr;
    int          ple_dim_            = 0;        // hidden_size_per_layer_input (typ. 256)
    bool         ple_enabled_        = false;

    // Per-forward scratch holding [n_layers * ple_dim_] for the current token.
    std::vector<float> ple_buffer_;

    // Tokenizer data
    std::vector<std::string> vocab_tokens_;
    std::vector<float>       vocab_scores_;
    std::vector<uint8_t>     token_type_; // 0=normal, 1=bos, 2=eos, 3=unknown

    // Attention config (filled from meta)
    AttentionConfig attn_cfg_;

    // KV cache
    KVCache kvcache_;

public:
    // Direct KV cache access for persistence
    // ── KV cache allocation + persistence ────────────────────────────────────
    // Hybrid models (Qwen3-Next / 3.5 / 3.6) only have K/V on every 4th
    // (full-attention) layer; the GatedDeltaNet layers keep a fixed-size
    // recurrent state instead. KV geometry is set per layer so those layers
    // get zero heads: nothing is allocated for them and nothing is written to
    // saved sessions / checkpoints for them (4x smaller on a 64-layer 27B).
    void init_kv_cache_for_model() {
        const int L = (int)meta_.n_layers;
        std::vector<int> nh((size_t)L, (int)meta_.n_kv_heads);
        std::vector<int> hd((size_t)L, (int)meta_.head_dim);
        int no_kv = 0, differs = 0;
        for (int l = 0; l < L && l < (int)layer_weights_.size(); ++l) {
            const TransformerLayerWeights& lw = layer_weights_[(size_t)l];
            if (lw.qwen35_linear) { nh[(size_t)l] = 0; ++no_kv; continue; }
            // Use the geometry attention really runs with (from the weight
            // shapes), not the global metadata, so the cache can never be
            // sized differently from what is stored into it.
            if (lw.attn_kv_heads_layer > 0 && lw.attn_kv_heads_layer != nh[(size_t)l]) { nh[(size_t)l] = lw.attn_kv_heads_layer; ++differs; }
            if (lw.attn_head_dim_layer > 0 && lw.attn_head_dim_layer != hd[(size_t)l]) { hd[(size_t)l] = lw.attn_head_dim_layer; ++differs; }
        }
        size_t per_tok = 0;
        for (int l = 0; l < L; ++l) per_tok += (size_t)nh[(size_t)l] * (size_t)hd[(size_t)l] * 2 /*K+V*/ * 2 /*f16*/;
        if (no_kv > 0 || differs > 0) {
            kvcache_.init_per_layer(nh, hd);
        } else {
            kvcache_.init(L, (int)meta_.n_kv_heads, (int)meta_.head_dim);
        }
        int first_kv = -1;
        for (int l = 0; l < L; ++l) if (nh[(size_t)l] > 0) { first_kv = l; break; }
        fprintf(stderr,
            "[kv] cache geometry: %d layers, %d with KV (%d recurrent, none), first KV layer %d = %d kv-heads x %d dim | metadata says %u x %u%s | expected %.1f KB/token = %.2f GB at 41k tokens, %.2f GB at 100k\n",
            L, L - no_kv, no_kv, first_kv,
            first_kv >= 0 ? nh[(size_t)first_kv] : 0, first_kv >= 0 ? hd[(size_t)first_kv] : 0,
            (unsigned)meta_.n_kv_heads, (unsigned)meta_.head_dim,
            differs ? "  <-- DIFFERS, using the real per-layer values" : "",
            per_tok / 1024.0, per_tok * 41000.0 / 1073741824.0, per_tok * 100000.0 / 1073741824.0);
        fflush(stderr);
    }

    // Recurrent (GatedDeltaNet) state is part of the conversation's memory.
    // Saved as a trailer after the KV data: [payload][u64 size]["NXQ35ST1"].
    // Per-layer sibling of q35_state_trailer()/q35_state_apply_trailer() below, for the same
    // one-layer-at-a-time streaming save_layer_to_buffer()/load_layer_from_buffer() on the plain
    // KV cache. An empty return means this layer has no recurrent state at all (an ordinary
    // full-attention layer) -- nothing to append for it.
    std::vector<uint8_t> q35_state_layer_trailer(int l) const {
        const_cast<Transformer*>(this)->q35_sync_host(l);
        std::vector<uint8_t> out;
        if (l < 0 || (size_t)l >= q35_recurrent_state_.size()) return out;
        const std::vector<float>& cv = q35_conv_state_[l];
        const std::vector<float>& rc = q35_recurrent_state_[l];
        if (cv.empty() && rc.empty()) return out;
        auto put = [&](const void* p, size_t n) { const uint8_t* b = (const uint8_t*)p; out.insert(out.end(), b, b + n); };
        uint64_t cn = cv.size(), rn = rc.size();
        put(&cn, 8); put(cv.data(), cn * 4);
        put(&rn, 8); put(rc.data(), rn * 4);
        const uint64_t payload = out.size();
        put(&payload, 8);
        put("NXQ35LY1", 8);
        return out;
    }
    // data/size: a buffer that MAY end with a q35 layer trailer (from q35_state_layer_trailer()
    // above) for layer l specifically. Returns *kv_size = the size with that trailer stripped off
    // either way, so the caller can pass the rest on to load_layer_from_buffer() unchanged.
    bool q35_state_apply_layer_trailer(int l, const uint8_t* data, size_t size, size_t* kv_size) {
        if (kv_size) *kv_size = size;
        if (size < 16 || memcmp(data + size - 8, "NXQ35LY1", 8) != 0) return false;
        uint64_t payload = 0; memcpy(&payload, data + size - 16, 8);
        if (payload + 16 > size) return false;
        const uint8_t* p = data + size - 16 - payload;
        const uint8_t* end = data + size - 16;
        if (l < 0) return false;
        // *** v25 fix: a computer RECEIVING a hybrid-prefill layer's recurrent/conv state has, by
        // design, never run that layer's forward pass itself -- that's the whole point of hybrid
        // prefill, this computer only receives the already-finished state. So these vectors are
        // still at their default empty size the first time a layer's state arrives here (they're
        // only grown lazily inside the actual compute path, which never runs on this side for a
        // hybrid-prefilled layer). The old check below bailed out BEFORE *kv_size was ever updated,
        // so the caller fed the WHOLE buffer -- trailer bytes included -- to
        // kvcache_.load_layer_from_buffer() as if it were plain K/V data, which always failed with
        // the generic "size/layer-geometry mismatch" message. Grow instead of refusing.
        if ((size_t)l >= q35_recurrent_state_.size()) {
            q35_conv_state_.resize((size_t)l + 1);
            q35_recurrent_state_.resize((size_t)l + 1);
        }
        if (kv_size) *kv_size = size - 16 - payload;
        q35_host_acquire(l);
        uint64_t cn, rn;
        if (p + 8 > end) return false; memcpy(&cn, p, 8); p += 8;
        // *** v25 fix: same reasoning -- this layer's state vector may still be empty or the
        // wrong size because this computer never computed it locally. The SENDER just finished
        // running this exact layer, so its size is authoritative; adopt it instead of treating a
        // differing size as a hard failure.
        if (q35_conv_state_[l].size() != cn) q35_conv_state_[l].assign((size_t)cn, 0.0f);
        if (p + cn * 4 > end) return false;
        memcpy(q35_conv_state_[l].data(), p, cn * 4); p += cn * 4;
        if (p + 8 > end) return false; memcpy(&rn, p, 8); p += 8;
        if (q35_recurrent_state_[l].size() != rn) q35_recurrent_state_[l].assign((size_t)rn, 0.0f);
        if (p + rn * 4 > end) return false;
        memcpy(q35_recurrent_state_[l].data(), p, rn * 4); p += rn * 4;
        return true;
    }
    // One layer's full KV + (if it has one) recurrent state, ready to hand to another machine's
    // load_kv_layer_buffer() for the same layer index. Empty = nothing to send for this layer.
    std::vector<uint8_t> save_kv_layer_buffer(int l) const {
        std::vector<uint8_t> b = kvcache_.save_layer_to_buffer(l);
        std::vector<uint8_t> t = q35_state_layer_trailer(l);
        b.insert(b.end(), t.begin(), t.end());
        return b;
    }
    bool load_kv_layer_buffer(int l, const uint8_t* data, size_t size) {
        if (size == 0) return true;   // this layer had nothing to send (e.g. linear-attention layer with no state either) -- fine
        size_t kv_size = size;
        q35_state_apply_layer_trailer(l, data, size, &kv_size);   // absent trailer is not an error: ordinary full-attention layers don't have one
        if (kv_size == 0) return true;
        return kvcache_.load_layer_from_buffer(data, kv_size);
    }
    // Used on the SENDING side right after save_kv_layer_buffer(l) has been handed off: this
    // layer's KV is no longer needed here, so free it now instead of waiting for the whole ghost
    // engine to be destroyed. (The q35 recurrent/conv state, if any, is tiny -- left alone; freeing
    // the growing KV cache is what actually matters for a long prompt.)
    void free_kv_layer(int l) { kvcache_.free_layer(l); }

    std::vector<uint8_t> q35_state_trailer() const {
        const_cast<Transformer*>(this)->q35_sync_host(-1);
        std::vector<uint8_t> out;
        if (q35_recurrent_state_.empty() && q35_conv_state_.empty()) return out;
        auto put = [&](const void* p, size_t n) {
            const uint8_t* b = (const uint8_t*)p; out.insert(out.end(), b, b + n); };
        const uint32_t n = (uint32_t)q35_recurrent_state_.size();
        put(&n, 4);
        for (uint32_t l = 0; l < n; ++l) {
            const std::vector<float>& cv = q35_conv_state_[l];
            const std::vector<float>& rc = q35_recurrent_state_[l];
            uint64_t cn = cv.size(), rn = rc.size();
            put(&cn, 8); put(cv.data(), cn * 4);
            put(&rn, 8); put(rc.data(), rn * 4);
        }
        const uint64_t payload = out.size();
        put(&payload, 8);
        put("NXQ35ST1", 8);
        return out;
    }
    // data/size: whole blob. Returns payload-stripped size via *kv_size (size
    // of the KV part) and true if a trailer was found and applied.
    bool q35_state_apply_trailer(const uint8_t* data, size_t size, size_t* kv_size) {
        if (kv_size) *kv_size = size;
        if (size < 16 || memcmp(data + size - 8, "NXQ35ST1", 8) != 0) return false;
        uint64_t payload = 0; memcpy(&payload, data + size - 16, 8);
        if (payload + 16 > size) return false;
        const uint8_t* p = data + size - 16 - payload;
        const uint8_t* end = data + size - 16;
        q35_host_acquire(-1);
        if (kv_size) *kv_size = size - 16 - payload;
        uint32_t n = 0; if (p + 4 > end) return false; memcpy(&n, p, 4); p += 4;
        if (n != q35_recurrent_state_.size()) return false;
        for (uint32_t l = 0; l < n; ++l) {
            uint64_t cn, rn;
            if (p + 8 > end) return false; memcpy(&cn, p, 8); p += 8;
            if (cn != q35_conv_state_[l].size() || p + cn * 4 > end) return false;
            memcpy(q35_conv_state_[l].data(), p, cn * 4); p += cn * 4;
            if (p + 8 > end) return false; memcpy(&rn, p, 8); p += 8;
            if (rn != q35_recurrent_state_[l].size() || p + rn * 4 > end) return false;
            memcpy(q35_recurrent_state_[l].data(), p, rn * 4); p += rn * 4;
        }
        return true;
    }
    void q35_state_zero() {
        q35_drop_dev(-1);
        for (auto& v : q35_conv_state_)      std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : q35_recurrent_state_) std::fill(v.begin(), v.end(), 0.0f);
    }
    // Diagnostic fingerprint of the model's running memory (KV length + DeltaNet state).
    // Equal fingerprints at save time and after a load mean the state came back intact.
    std::string state_fingerprint() const {
        const_cast<Transformer*>(this)->q35_sync_host(-1);
        double rs = 0.0, cs = 0.0; size_t layers = 0;
        for (const auto& v : q35_recurrent_state_) {
            if (v.empty()) continue;
            ++layers;
            for (float x : v) rs += std::fabs((double)x);
        }
        for (const auto& v : q35_conv_state_)
            for (float x : v) cs += std::fabs((double)x);
        char b[200];
        snprintf(b, sizeof b, "kv_tokens=%d deltanet_layers=%zu recurrent|sum|=%.4f conv|sum|=%.4f",
                 kvcache_.n_tokens(), layers, rs, cs);
        return b;
    }

    std::vector<uint8_t> save_kv_buffer() const {
        std::vector<uint8_t> b = kvcache_.save_to_buffer();
        std::vector<uint8_t> t = q35_state_trailer();
        b.insert(b.end(), t.begin(), t.end());
        return b;
    }
    bool load_kv_buffer(const uint8_t* data, size_t size) {
        size_t kv_size = size;
        const bool had = q35_state_apply_trailer(data, size, &kv_size);
        if (!kvcache_.load_from_buffer(data, kv_size)) return false;
        if (!had && !q35_recurrent_state_.empty()) {
            q35_state_zero();
            fprintf(stderr, "[kv] snapshot has no recurrent state (old format): DeltaNet layers start empty\n");
        }
        return true;
    }
    bool save_kv_file(const std::string& path) const {
        if (!kvcache_.save(path)) return false;
        std::vector<uint8_t> t = q35_state_trailer();
        if (t.empty()) return true;
        FILE* f = fopen(path.c_str(), "ab");
        if (!f) return false;
        const bool ok = fwrite(t.data(), 1, t.size(), f) == t.size();
        fclose(f);
        return ok;
    }
    bool load_kv_file(const std::string& path) {
        std::vector<uint8_t> trailer_blob;
        bool have_trailer = false;
        if (FILE* f = fopen(path.c_str(), "rb")) {
            uint8_t foot[16];
            if (fseek(f, -16, SEEK_END) == 0 && fread(foot, 1, 16, f) == 16 &&
                memcmp(foot + 8, "NXQ35ST1", 8) == 0) {
                uint64_t payload; memcpy(&payload, foot, 8);
                if (payload < (1ull << 32) && fseek(f, -(long)(16 + payload), SEEK_END) == 0) {
                    trailer_blob.resize((size_t)payload + 16);
                    have_trailer = fread(trailer_blob.data(), 1, (size_t)payload + 16, f) == (size_t)payload + 16;
                }
            }
            fclose(f);
        }
        if (!kvcache_.load(path)) return false;
        bool applied = false;
        if (have_trailer) { size_t ks; applied = q35_state_apply_trailer(trailer_blob.data(), trailer_blob.size(), &ks); }
        if (!applied && !q35_recurrent_state_.empty()) {
            q35_state_zero();
            fprintf(stderr, "[kv] session file has no recurrent state (old format): DeltaNet layers start empty\n");
        }
        return true;
    }

          KVCache& kvcache_ref()       { return kvcache_; }
    const KVCache& kvcache_ref() const { return kvcache_; }

    // Distributed inference: run a subset of layers, access hidden state
    void forward_layer_range(const std::vector<int32_t>& tokens, int pos,
                             int layer_start, int layer_end);
    const float* hidden_data() const;
    size_t hidden_dim_bytes() const;
    void set_hidden_data(const float* src, size_t n_floats);

    // Multimodal/native helper: expose the already-loaded tensor table so native
    // embedded vision can borrow TensorDesc::data pointers. This avoids opening
    // or mmap'ing the same .nxm a second time.
    const std::unordered_map<std::string, TensorDesc>* loaded_tensor_map() const {
        return parser_ ? &parser_->tensors() : nullptr;
    }

private:

    // Last computed logits
    std::vector<float> logits_;

    // Scratch buffers (sized on load, reused per token)
    std::vector<float> x_scratch_;     // hidden state [hidden_dim]

    // ── Multi-Token Prediction (MTP) ─────────────────────────────────────────
    // Qwen3-Next native MTP: one extra decoder block (mtp.layers.0.*) that, given
    // the main model's last hidden state + the embedding of the just-emitted
    // token, predicts the NEXT token. Loaded lazily by name (does not touch the
    // main load path). Universal: keyed on tensor presence, not this model.
    struct MtpBlock {
        bool   loaded = false;
        bool   available = false;        // tensors found in this model?
        // fc fuses [hidden | next-token-embed] -> hidden  (rows=hidden, cols=2*hidden)
        const void* fc = nullptr;        QuantType fc_type{};
        const float* pre_fc_norm_embed = nullptr;
        const float* pre_fc_norm_hidden = nullptr;
        const float* final_norm = nullptr;
        // one decoder layer (same shape as a full-attn layer, with gated Q)
        const void* wq = nullptr;        QuantType wq_type{};
        const void* wk = nullptr;        QuantType wk_type{};
        const void* wv = nullptr;        QuantType wv_type{};
        const void* wo = nullptr;        QuantType wo_type{};
        const float* q_norm = nullptr;
        const float* k_norm = nullptr;
        const float* input_norm = nullptr;
        const float* post_attn_norm = nullptr;
        const void* w_gate = nullptr;    QuantType w_gate_type{};
        const void* w_up = nullptr;      QuantType w_up_type{};
        const void* w_down = nullptr;    QuantType w_down_type{};
        int q_rows = 0;                  // 2*q_dim if gated, else q_dim
        bool q_has_gate = false;
    };
    MtpBlock mtp_;
    bool mtp_enabled_ = false;           // runtime toggle (set from engine/server)
    int  mtp_n_predict_ = 1;            // tokens MTP proposes per step (>=1)
    bool mtp_use_gpu_ = false;          // run MTP projections on GPU when available
    // MTP's own KV history (f16-as-float), built as the main model decodes, so
    // the MTP block attends over the SAME context the head was trained on.
    // Layout per position: [n_kv_heads * head_dim]. Index by kv-head then dim.
    std::vector<float> mtp_k_hist_;     // [n_positions * n_kv_heads * head_dim]
    std::vector<float> mtp_v_hist_;
    int                mtp_hist_len_ = 0;
    // When non-null during prefill_batch, also capture logits at position n-2
    // into this buffer (used by forward_verify2 to get the draft-check dist).
    std::vector<float>* mtp_verify_second_logits_ = nullptr;
    // When non-null during prefill_batch, capture logits at EVERY position
    // [0..n-1] into this vector (used by forward_verifyN for multi-token verify).
    std::vector<std::vector<float>>* mtp_verify_all_logits_ = nullptr;
    // MTP without a second pass: during a check, every linear-attention layer's per-token conv input and gates are kept
    // (small: a few MB), plus the final hidden state of every checked position. On a miss the recurrent state is rewound
    // and only the cheap state update is replayed for the accepted tokens -- no weights, no pass through the layers.
    struct Q35VerifyRec { std::vector<float> mixed, a, b; int n = 0; };
    std::vector<Q35VerifyRec> q35_vrec_;
    std::vector<float> mtp_verify_hidden_;
    int mtp_verify_n_ = 0;
    void q35_replay_state(int layer_idx, int keep);

public:
    // Enable/disable MTP, tokens-per-step, and whether to run MTP projections
    // on the GPU when available.
    void set_mtp(bool on, int n_predict, bool use_gpu = false) {
        mtp_enabled_ = on;
        mtp_n_predict_ = (n_predict < 1) ? 1 : n_predict;
        mtp_use_gpu_ = use_gpu;
        fprintf(stderr, "[mtp] set_mtp(on=%d, n_predict=%d, gpu=%d)\n",
                on?1:0, mtp_n_predict_, use_gpu?1:0);
        if (on) mtp_load();
    }
    bool mtp_enabled() const { return mtp_enabled_ && mtp_.available; }
    int  mtp_n_predict() const { return mtp_n_predict_; }

    // Load the MTP block from the parser by tensor name. Safe to call repeatedly;
    // sets mtp_.available=false if the model has no MTP tensors. Returns available.
    bool mtp_load();

    // Propose up to mtp_n_predict_ tokens after `last_token` was emitted at
    // `pos`, using the main model's last hidden state. Fills `out` with proposed
    // token ids. Caller verifies each against the main model before accepting.
    // Returns number proposed.
    int mtp_propose(int32_t last_token, int pos, const model::SamplerConfig& scfg,
                    std::vector<int32_t>& out);
private:
    std::vector<float> xb_scratch_;    // second hidden buffer
    std::vector<float> attn_scratch_;  // attention computation scratch
    std::vector<float> ffn_scratch_;   // FFN gate+up scratch [ffn_dim * 2]
    std::vector<float>    qkv_scratch_;  // GPU QKV output: [q_dim + 2*kv_dim]
    std::vector<uint16_t> kv_flat_k_;    // packed f16 K staging for GPU attention
    std::vector<uint16_t> kv_flat_v_;    // packed f16 V staging
    
    // Qwen35/Qwen3Next-style Gated DeltaNet recurrent state.
    // Per layer:
    //   conv state:      [conv_dim * (kernel - 1)]
    //   recurrent state: [num_v_heads * head_dim * head_dim]
    std::vector<std::vector<float>> q35_conv_state_;
    std::vector<std::vector<float>> q35_recurrent_state_;
    // GPU-resident decode: while q35_dev_ahead_[l] is set, the card holds the NEWER recurrent/conv state of layer l and the two host vectors above are stale.
    // Anything that reads the host copy calls q35_sync_host(); anything that writes it calls q35_drop_dev() (the card copy is re-uploaded on the next GPU step).
    std::vector<uint8_t> q35_dev_ahead_;
    std::vector<float> q35_kv_out_;   // K,V of the token just produced by the GPU full-attention step (stored into kvcache_)
    void q35_sync_host(int l) {
        for (int i = (l < 0 ? 0 : l); i < (l < 0 ? (int)q35_dev_ahead_.size() : std::min<int>(l + 1, (int)q35_dev_ahead_.size())); ++i) {
            if (!q35_dev_ahead_[i]) continue;
            q35_dev_ahead_[i] = 0;
            if (i < (int)q35_conv_state_.size() && i < (int)q35_recurrent_state_.size() &&
                gpu::gpu().q35_state_download(this, i, q35_conv_state_[i].data(), q35_recurrent_state_[i].data())) continue;
            fprintf(stderr, "[nexus] WARNING: could not read the Qwen3.5 recurrent state of layer %d back from the GPU\n", i); fflush(stderr);
        }
    }
    void q35_drop_dev(int l) {
        gpu::gpu().q35_state_drop(this, l);
        if (l < 0) std::fill(q35_dev_ahead_.begin(), q35_dev_ahead_.end(), (uint8_t)0);
        else if (l < (int)q35_dev_ahead_.size()) q35_dev_ahead_[l] = 0;
    }
    void q35_kv_drop_all() { gpu::gpu().q35_kv_drop(this, -1); }   // free the card's KV mirrors (reload / new context / destructor)
    void q35_host_acquire(int l) { q35_sync_host(l); q35_drop_dev(l); }   // host copy is about to be read AND modified

    // Sampler RNG
    std::mt19937 rng_;
    int rng_seed_active_ = -2; // -2 = never seeded, -1 = random seeded, >=0 = fixed seed

    // Internal
    bool build_layer_weights();
    bool load_tokenizer();
    TensorDesc* find(const std::string& suffix);
    TensorDesc* find_layer(int layer, const std::string& suffix);

    // Forward through one layer
    void forward_layer(int layer_idx, float* x, int pos);

    // ── MTP speculative verify support ───────────────────────────────────────
    // Snapshot/restore the recurrent GatedDeltaNet state so a rejected draft can
    // be rolled back (full-attention KV is pos-indexed, so not advancing pos is
    // its own rollback; only the recurrent state needs explicit save/restore).
public:
    struct Q35StateSnapshot {
        std::vector<std::vector<float>> conv;
        std::vector<std::vector<float>> rec;
        bool valid = false;
    };
    Q35StateSnapshot mtp_snapshot_q35_state() const {
        Q35StateSnapshot s;
        const_cast<Transformer*>(this)->q35_sync_host(-1);
        s.conv = q35_conv_state_;
        s.rec  = q35_recurrent_state_;
        s.valid = true;
        return s;
    }
    void mtp_restore_q35_state(const Q35StateSnapshot& s) {
        if (!s.valid) return;
        q35_drop_dev(-1);
        q35_conv_state_ = s.conv;
        q35_recurrent_state_ = s.rec;
    }
    // After mtp_restore_q35_state(snapshot taken before the check): keep the first `keep` checked tokens (the confirmed
    // token + the accepted drafts) WITHOUT running them through the model again. Replays the recurrent state update from
    // what the check recorded, trims the conversation memory to them, and hands MTP the hidden state of the last kept one.
    // Returns false (nothing changed) if the check didn't record every linear layer -- the caller then re-runs them.
    bool mtp_keep_checked(int keep, int pos);

    // Verify forward: process confirmed_tok at `pos` and draft_tok at `pos+1`
    // through ALL layers in one sweep, returning logits at BOTH positions.
    // out_logits_confirmed = distribution for the token AFTER confirmed_tok
    // (i.e. the model's own next token — used to verify the draft).
    // out_logits_draft     = distribution for the token AFTER draft_tok
    // (the bonus token, valid only if the draft is accepted).
    // The recurrent state ends advanced through BOTH tokens; caller restores
    // from a snapshot if the draft is rejected. KV is written at pos and pos+1.
    void forward_verify2(int32_t confirmed_tok, int32_t draft_tok, int pos,
                         std::vector<float>& out_logits_confirmed,
                         std::vector<float>& out_logits_draft);

    // Multi-token verify: feed [confirmed, draft[0], ..., draft[k-1]] through ONE
    // batched sweep at positions [pos .. pos+k], returning the model's own
    // distribution at EVERY position. out_logits[j] is the distribution AFTER the
    // j-th input token, i.e. out_logits[0] verifies draft[0], out_logits[1]
    // verifies draft[1] (only valid if draft[0] accepted), etc. The recurrent
    // state advances through ALL tokens; caller restores from a snapshot for the
    // rejected tail. Returns number of positions filled (k+1).
    int forward_verifyN(int32_t confirmed_tok, const std::vector<int32_t>& drafts,
                        int pos, std::vector<std::vector<float>>& out_logits);
private:

    // Eagerly push all quantized weight matrices for a GPU layer into VRAM.
    // Called from set_gpu_layers() so the first token isn't stalled by PCIe uploads.
    bool eager_upload_layer_weights(int l, gpu::VulkanBackend& g, std::vector<const void*>* uploaded = nullptr);

    // FFN: SwiGLU
    void forward_ffn(const TransformerLayerWeights& w, float* x);

    // Sampling helpers
    int32_t sample_argmax() const;
    int32_t sample_top_p(float top_p, float temp) const;
};

} // namespace model
} // namespace nexus


// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//  Implementation
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace nexus {
namespace model {

// NEXUS_PATCH_QWEN35_NATIVE_QGATE_DETECTOR_V2

// --------- find tensor helpers ------------------------------------------------------------------------------------------------------------------------------------------------------------------

TensorDesc* Transformer::find(const std::string& suffix) {
    return parser_->find_tensor(suffix);
}

TensorDesc* Transformer::find_layer(int layer, const std::string& suffix) {
    auto& tensors = const_cast<std::unordered_map<std::string, TensorDesc>&>(parser_->tensors());

    // GGUF path.
    {
        std::string name = "blk." + std::to_string(layer) + "." + suffix;
        auto it = tensors.find(name);
        if (it != tensors.end()) return &it->second;
    }

    // Native .nxm path: tensors keep original HF names. Try the suffix as-is
    // first, then map the canonical GGUF names used internally by Nexus onto
    // the original HF names written by the native builder.  The layer prefix differs by model family:
    //   model.language_model.layers.N.   Qwen3.5 / Qwen3Next / multimodal wrappers
    //   model.layers.N.                  plain HF text models (Llama, Mistral, Qwen2/3, Phi, Gemma, ...)
    //   language_model.model.layers.N.   some multimodal exports
    static const char* const kNativeLayerPrefixes[] = {
        "model.language_model.layers.", "model.layers.", "language_model.model.layers."
    };
    std::string base = std::string(kNativeLayerPrefixes[0]) + std::to_string(layer) + ".";
    for (const char* pre : kNativeLayerPrefixes) {
        const std::string b = std::string(pre) + std::to_string(layer) + ".";
        auto it = tensors.find(b + suffix);
        if (it != tensors.end()) return &it->second;
        // remember which family this model uses so the alias lookups below search the right prefix
        if (tensors.find(b + "input_layernorm.weight") != tensors.end() || tensors.find(b + "self_attn.q_proj.weight") != tensors.end()
            || tensors.find(b + "linear_attn.in_proj_qkv.weight") != tensors.end()) base = b;
    }

    auto try_native = [&](std::initializer_list<const char*> alts) -> TensorDesc* {
        for (const char* a : alts) {
            auto it = tensors.find(base + a);
            if (it != tensors.end()) return &it->second;
        }
        return nullptr;
    };

    // Qwen3Next/Qwen3.6 Gated DeltaNet original HF names.
    if (suffix == "attn_qkv.weight")      return try_native({"linear_attn.in_proj_qkv.weight"});
    if (suffix == "attn_gate.weight")     return try_native({"linear_attn.in_proj_z.weight"});
    if (suffix == "ssm_alpha.weight")     return try_native({"linear_attn.in_proj_a.weight"});
    if (suffix == "ssm_beta.weight")      return try_native({"linear_attn.in_proj_b.weight"});
    if (suffix == "ssm_conv1d.weight")    return try_native({"linear_attn.conv1d.weight"});
    if (suffix == "ssm_dt.bias" || suffix == "ssm_dt")
                                             return try_native({"linear_attn.dt_bias"});
    if (suffix == "ssm_a")                return try_native({"linear_attn.A_log"});
    if (suffix == "ssm_norm.weight")      return try_native({"linear_attn.norm.weight"});
    if (suffix == "ssm_out.weight")       return try_native({"linear_attn.out_proj.weight"});

    // Qwen3Next full-attention norm aliases in native HF form.
    if (suffix == "attn_q_norm.weight" || suffix == "q_norm.weight")
                                             return try_native({"self_attn.q_norm.weight"});
    if (suffix == "attn_k_norm.weight" || suffix == "k_norm.weight")
                                             return try_native({"self_attn.k_norm.weight"});

    return nullptr;
}

inline const float* Transformer::tensor_as_f32(TensorDesc* t, const char* label) {
    if (!t || !t->data) return nullptr;

    if (t->type == GGMLType::F32) {
        return (const float*)t->data;
    }

    if (t->type == GGMLType::F16 || t->type == GGMLType::BF16) {
        f32_tensor_storage_.emplace_back((size_t)t->n_elements);
        float* out = f32_tensor_storage_.back().data();

        quant::dequantize(
            t->data,
            out,
            (size_t)t->n_elements,
            ggml_to_qtype(t->type));

        return out;
    }

    fprintf(stderr,
        "[nexus] FATAL: tensor %s must be F32/F16/BF16 but has GGML type %d\n",
        label,
        (int)t->type);
    std::abort();
}

// --------- Transformer::load ----------------------------------------------

inline bool Transformer::load(const std::string& model_path, int n_threads,
                        std::function<void(const std::string&, float)> progress_cb)
{
    q35_drop_dev(-1);   // a reload must never see GPU state left over from a previous model
    q35_kv_drop_all();
    q35_dev_ahead_.clear();
    n_threads_ = n_threads;
    active_layer_start_ = 0;
    active_layer_end_   = -1;

    // Trace: fresh model load → clear the per-(layer,key) dedup set so we
    // re-emit decisions for the new model.
    ::nexus::trace::reset();
    NX_TRACE_SECTION("model load");
    NX_TRACE_MODEL("model_path", "%s", model_path.c_str());
    NX_TRACE_MODEL("n_threads",  "%d", n_threads);

    if (progress_cb) progress_cb("Opening model file", 0.0f);

    // Create memory manager
    MemoryManager::Config mm_cfg;
    mm_cfg.prefer_huge_pages  = true;
    mm_cfg.lock_active_layers = true;
    mm_cfg.max_locked_layers  = std::min(n_threads * 2, 8);
    mm_ = std::make_unique<MemoryManager>(mm_cfg);
    fprintf(stderr,"[load] step 1: MemoryManager created\n");fflush(stderr);

    // Register pressure handler
    mm_->on_pressure([this](size_t avail) {
        fprintf(stderr, "[nexus] Memory pressure! %s available. Evicting KV cache pages.\n",
                MemoryManager::fmt_bytes(avail).c_str());
        kvcache_.evict_oldest(128);
    });

    // Parse GGUF header
    parser_ = std::make_unique<gguf::Parser>();
    fprintf(stderr,"[load] step 2: opening parser...\n");fflush(stderr);
    if (!parser_->open(model_path)) {
        fprintf(stderr, "[nexus] Failed to open: %s\n", model_path.c_str());
        return false;
    }

    if (!nexus_gguf_formats_supported(*parser_)) return false;
    meta_ = parser_->meta();
    fprintf(stderr,"[load] step 3: meta read\n");fflush(stderr);

    if (progress_cb) progress_cb("Reading model metadata", 0.05f);
    parser_->print_meta();

    // Load tokenizer
    fprintf(stderr,"[load] step 4: loading tokenizer...\n");fflush(stderr);
    if (!load_tokenizer()) {
        fprintf(stderr, "[nexus] Warning: tokenizer load failed\n");
    }

    if (!vocab_tokens_.empty() && vocab_tokens_.size() != meta_.vocab_size) {
        fprintf(stderr,
                "[nexus] Vocab metadata corrected from %u to %zu based on tokenizer tokens\n",
                meta_.vocab_size, vocab_tokens_.size());
        meta_.vocab_size = (uint32_t)vocab_tokens_.size();
    }

    // Load weights into our memory pool
    if (progress_cb) progress_cb("Loading weights into memory", 0.1f);

    size_t loaded = 0;

    fprintf(stderr,"[load] step 5: loading weights...\n");fflush(stderr);
    bool ok = parser_->load_weights(*mm_, [&](const gguf::LoadProgress& p) {
        ++loaded;
        float frac = 0.1f + 0.8f * ((float)p.bytes_loaded / std::max((float)p.bytes_total, 1.0f));
        if (progress_cb) {
            progress_cb("Loading: " + p.current_tensor, frac);
        }
        if (loaded % 50 == 0) {
            fprintf(stderr, "  [%3zu%%] %s\n",
                    (size_t)(frac * 100), p.current_tensor.c_str());
        }
    });

    if (!ok) {
        fprintf(stderr, "[nexus] Weight loading failed\n");
        return false;
    }

    if (progress_cb) progress_cb("Building layer index", 0.92f);

    // Build layer weight index
    fprintf(stderr,"[load] step 6: building layer index...\n");fflush(stderr);
    if (!build_layer_weights()) {
        fprintf(stderr, "[nexus] Failed to index layer weights\n");
        return false;
    }

    // Validate critical dimensions before anything that could crash on zero
    fprintf(stderr,"[load] step 7: validating dims...\n");fflush(stderr);
    if (meta_.head_dim == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[nexus] FATAL: head_dim=%u hidden_dim=%u --- cannot continue.\n"
                        "  This usually means an unknown tensor naming convention.\n"
                        "  Known tensors in this model:\n",
                meta_.head_dim, meta_.hidden_dim);
        int shown = 0;
        for (auto& [name, t] : parser_->tensors()) {
            if (name.find("blk.0.") != std::string::npos && shown++ < 12)
                fprintf(stderr, "    %s  shape=[", name.c_str());
            // just show the names so we can fix
            if (shown > 0 && shown <= 12) {
                for (size_t si = 0; si < t.shape.size(); ++si)
                    fprintf(stderr, "%s%llu", si?",":"", (unsigned long long)t.shape[si]);
                fprintf(stderr, "]\n");
            }
        }
        return false;
    }

    // Native Qwen3.6/Qwen3.5 official config correction.
    // Some early .nxm manifests were written with GGUF/default RoPE theta=10000
    // and eps=1e-5. The official Qwen3.6 config is rope_theta=10000000 and
    // rms_norm_eps=1e-6 for the text model; vision/text multimodal placement is
    // especially sensitive to this because image tokens use the same decoder.
    {
        std::string arch_l = meta_.arch;
        std::string name_l = meta_.name;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);
        const bool is_qwen35_native =
            arch_l.find("qwen3next") != std::string::npos ||
            arch_l.find("qwen3_5")   != std::string::npos ||
            arch_l.find("qwen3_6")   != std::string::npos ||
            name_l.find("qwen 3.6")  != std::string::npos ||
            name_l.find("qwen3.6")   != std::string::npos ||
            name_l.find("qwen 27")   != std::string::npos;
        if (is_qwen35_native) {
            const float old_rope = meta_.rope_freq_base;
            const float old_eps  = meta_.rms_norm_eps;
            if (meta_.rope_freq_base < 9999999.0f) meta_.rope_freq_base = 10000000.0f;
            meta_.rms_norm_eps = 1.0e-6f;
            fprintf(stderr,
                "[nexus] Native Qwen3.6 config correction: rope_theta %.0f -> %.0f, rms_eps %.2e -> %.2e\n",
                old_rope, meta_.rope_freq_base, old_eps, meta_.rms_norm_eps);
            fflush(stderr);
        }
    }

    // Setup attention config
    attn_cfg_.n_heads    = (int)meta_.n_heads;
    attn_cfg_.n_kv_heads = (int)meta_.n_kv_heads;
    attn_cfg_.head_dim   = (int)meta_.head_dim;
    attn_cfg_.hidden_dim = (int)meta_.hidden_dim;
    attn_cfg_.rms_norm_eps = meta_.rms_norm_eps;
    attn_cfg_.rope.freq_base = meta_.rope_freq_base;
    attn_cfg_.n_threads  = n_threads;

    // ------ Per-arch scaling / softcapping --------------------------------
    // Detect once at load. These are zero (= no-op) for any arch that
    // doesn't explicitly need them, so unrelated models aren't affected.
    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        std::string name_l = meta_.name;
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);

        const bool is_gemma  = (arch_l.find("gemma")  != std::string::npos) ||
                               (name_l.find("gemma")  != std::string::npos);
        // Gemma 1 has no softcap. Gemma 2 has both attn (50) and final (30).
        // Gemma 3 / Gemma 4 dropped softcaps. All Gemma versions use embed_scale.
        const bool is_gemma2 = (arch_l.find("gemma2") != std::string::npos) ||
                               (name_l.find("gemma-2") != std::string::npos) ||
                               (name_l.find("gemma2") != std::string::npos);

if (is_gemma && meta_.hidden_dim > 0) {
            embed_scale_ = sqrtf((float)meta_.hidden_dim);
        }

        const bool is_gemma4 = (arch_l.find("gemma4") != std::string::npos) ||
                               (name_l.find("gemma-4") != std::string::npos) ||
                               (name_l.find("gemma4")  != std::string::npos);

        if (is_gemma2) {
            attn_cfg_.attn_softcap = 50.0f;
            final_logit_softcap_   = 30.0f;
        } else if (is_gemma4) {
            // Gemma 4 does not use Gemma 2 attention softcap,
            // but Gemma 4 31B does use final logit softcapping.
            attn_cfg_.attn_softcap = 0.0f;

            float fc = 0.0f;
            if (parser_) {
                fc = parser_->get_f32("gemma4.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("gemma4_text.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("final_logit_softcapping", 0.0f);
            }

            final_logit_softcap_ = (fc > 0.0f) ? fc : 30.0f;
        } else {
            attn_cfg_.attn_softcap = 0.0f;
            final_logit_softcap_   = 0.0f;
        }

        // Gemma family uses GeGLU (GELU-tanh-gated) FFN, not SwiGLU.
        // Same gated structure (gate * up -> down) but with GELU instead of SiLU.
        // Without this, the FFN nonlinearity is wrong every layer and the model
        // collapses to repetitive output even with norms/scaling correct.
        ffn_activation_gelu_ = is_gemma;

        fprintf(stderr,
                "[nexus] Arch scaling: arch=%s gemma=%d gemma2=%d "
                "embed_scale=%.4f attn_softcap=%.1f final_softcap=%.1f ffn_act=%s\n",
                meta_.arch.c_str(),
                is_gemma  ? 1 : 0,
                is_gemma2 ? 1 : 0,
                embed_scale_,
                attn_cfg_.attn_softcap,
                final_logit_softcap_,
                ffn_activation_gelu_ ? "gelu_tanh" : "silu");
        fflush(stderr);

        NX_TRACE_SECTION("arch scaling decisions");
        NX_TRACE_MODEL("arch_raw",          "%s",     meta_.arch.c_str());
        NX_TRACE_MODEL("name_raw",          "%s",     meta_.name.c_str());
        NX_TRACE_MODEL("is_gemma",          "%d",     is_gemma ? 1 : 0);
        NX_TRACE_MODEL("is_gemma2",         "%d",     is_gemma2 ? 1 : 0);
        NX_TRACE_MODEL("embed_scale",       "%.6f",   embed_scale_);
        NX_TRACE_MODEL("attn_softcap",      "%.1f",   attn_cfg_.attn_softcap);
        NX_TRACE_MODEL("final_logit_softcap","%.1f",  final_logit_softcap_);
        NX_TRACE_MODEL("ffn_activation",    "%s",     ffn_activation_gelu_ ? "gelu_tanh" : "silu");
        NX_TRACE_MODEL("hidden_dim",        "%u",     meta_.hidden_dim);
        NX_TRACE_MODEL("ffn_dim",           "%u",     meta_.ffn_dim);
        NX_TRACE_MODEL("n_layers",          "%u",     meta_.n_layers);
        NX_TRACE_MODEL("n_heads",           "%u",     meta_.n_heads);
        NX_TRACE_MODEL("n_kv_heads",        "%u",     meta_.n_kv_heads);
        NX_TRACE_MODEL("head_dim",          "%u",     meta_.head_dim);
        NX_TRACE_MODEL("vocab_size",        "%u",     meta_.vocab_size);
        NX_TRACE_MODEL("context_length",    "%u",     meta_.context_length);
        NX_TRACE_MODEL("rope_freq_base_meta","%.0f",  meta_.rope_freq_base);
        NX_TRACE_MODEL("rms_norm_eps",      "%.2e",   meta_.rms_norm_eps);
        NX_TRACE_MODEL("n_experts",         "%u",     meta_.n_experts);
        NX_TRACE_MODEL("n_experts_used",    "%u",     meta_.n_experts_used);
    }

    {
        std::string arch = meta_.arch;
        std::string name = meta_.name;
        for (char& c : arch) c = (char)std::tolower((unsigned char)c);
        for (char& c : name) c = (char)std::tolower((unsigned char)c);

        const bool match_phimoe = arch.find("phimoe") != std::string::npos;
        const bool match_phi    = arch.find("phi")    != std::string::npos;
        const bool match_olmoe  = arch.find("olmoe")  != std::string::npos;
        const bool match_gemma  = arch.find("gemma")  != std::string::npos;
        // Matched on the ARCHITECTURE only -- never the model's name: a name says nothing about how the
        // q/k weights were laid out, and substrings misfire ("Dolphin" contains "phi", a Llama
        // fine-tune with "gemma" in its name is still a Llama).
        // llama.cpp's rope-type table (llama_model_rope_type): these architectures use NEOX
        // (half-split) rotation; everything else listed there (llama, mistral, mixtral, ...) is
        // NORM (interleaved).  Matched on the exact GGUF architecture string, not the model name.
        static const char* const kHalfSplitArchs[] = {
            "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe",
            "qwen3next", "qwen35", "qwen35moe", "phi2", "phi3", "phimoe", "starcoder2", "olmo2",
            "olmoe", "stablelm", "internlm2", "gptneox", "falcon", "dbrx", "exaone", "openelm",
            "gemma", "gemma2", "gemma3", "gemma3n", "gemma4"};
        bool match_arch_table = false;
        for (const char* a : kHalfSplitArchs) if (arch == a) { match_arch_table = true; break; }
        const bool half_split   = match_phimoe || match_phi || match_olmoe || match_gemma || match_arch_table;

        fprintf(stderr,
            "[nexus] RoPE layout auto-detect: arch_lower=\"%s\" name_lower=\"%s\" "
            "match_phimoe=%d match_phi=%d match_olmoe=%d match_gemma=%d -> half_split=%d\n",
            arch.c_str(), name.c_str(),
            match_phimoe ? 1 : 0, match_phi ? 1 : 0,
            match_olmoe  ? 1 : 0, match_gemma ? 1 : 0,
            half_split ? 1 : 0);
        fflush(stderr);

        if (half_split) {
            attn_cfg_.rope.layout = 1;
            fprintf(stderr, "[nexus] RoPE layout: half-split/HF-style arch=%s\n", meta_.arch.c_str());
        } else {
            attn_cfg_.rope.layout = 0;
            fprintf(stderr, "[nexus] RoPE layout: interleaved arch=%s\n", meta_.arch.c_str());
        }

        // Native .nxm packages hold the ORIGINAL HuggingFace weights (never permuted), so the GGUF arch table above -- which assumes
        // llama.cpp's converter already reordered Llama/Mistral q/k weights for interleaved rotation -- does not apply to them.
        // The builder labels the layout from the HF model_type ("nexus.rope_layout"); older packages without the label are handled
        // by the HF rule: everything except GPT-J style models rotates half-split.
        if (parser_->get_str("nexus.native_hf_weights", "") == "1") {
            const std::string lab = parser_->get_str("nexus.rope_layout", "");
            int want = -1; const char* why = "";
            if (lab == "half_split")        { want = 1; why = "manifest label"; }
            else if (lab == "interleaved")  { want = 0; why = "manifest label"; }
            else if (arch == "llama" || arch == "mistral" || arch == "mixtral") { want = 1; why = "HF rule for unpermuted llama-family weights"; }
            if (want >= 0 && want != attn_cfg_.rope.layout) {
                fprintf(stderr, "[nexus] RoPE layout (native HF weights, %s): %s -> %s\n", why,
                        attn_cfg_.rope.layout == 1 ? "half_split" : "interleaved", want == 1 ? "half_split" : "interleaved");
                attn_cfg_.rope.layout = want;
            }
        }

        // Runtime override for testing convention without recompiling.
        // NEXUS_ROPE_LAYOUT={interleaved|half_split} forces the layout.
        // Useful when the auto-detect is wrong for a specific GGUF.
        if (const char* env = std::getenv("NEXUS_ROPE_LAYOUT")) {
            int was = attn_cfg_.rope.layout;
            if (std::string(env) == "interleaved") attn_cfg_.rope.layout = 0;
            else if (std::string(env) == "half_split") attn_cfg_.rope.layout = 1;
            else if (std::string(env) == "0") attn_cfg_.rope.layout = 0;
            else if (std::string(env) == "1") attn_cfg_.rope.layout = 1;
            fprintf(stderr,
                "[nexus] RoPE layout OVERRIDE by NEXUS_ROPE_LAYOUT=%s: was=%d now=%d\n",
                env, was, attn_cfg_.rope.layout);
            fflush(stderr);
        }

        NX_TRACE_MODEL("rope_layout", "%d (%s) %s",
            attn_cfg_.rope.layout,
            attn_cfg_.rope.layout == 1 ? "half_split/NeoX" : "interleaved/GPT-J",
            std::getenv("NEXUS_ROPE_LAYOUT") ? "[ENV-OVERRIDE]" : "[auto]");
        NX_TRACE_MODEL("rope_half_split_detected", "%d", half_split ? 1 : 0);
    }

    // ------ RoPE long-context setup ---------------------------------------------------------------------------------------------------------------------------------------------
    // n_ctx_orig: the context length the model was originally trained at.
    // Used by YaRN to know which frequencies need interpolation vs extrapolation.
    // We use context_length from GGUF as the best proxy --- fine-tuned models
    // encode their extended context here. Fall back to 4096 for legacy models.
    attn_cfg_.rope.n_ctx_orig = (meta_.context_length > 0) ? meta_.context_length : 4096;

    // Static NTK-aware RoPE scaling for models with a short trained context
    // that we are running at longer context.
    //
    // Standard RoPE with base=10000 starts producing aliased (non-unique)
    // position encodings past ~8k tokens --- Q--K dot products for distant tokens
    // become near-zero, softmax weight collapses to zero, model loses the past.
    //
    // NTK scaling multiplies the base so ALL frequency dimensions scale together,
    // preserving the ratio between dimensions that the model learned during
    // training. It is KV-cache safe because it applies uniformly --- Q and K
    // use the same scaled frequencies so their dot products still cancel correctly.
    //
    // We only apply this when:
    //   1. The model has a short native theta (---10001 = not already extended)
    //   2. The model declares a context longer than 8k (it needs extension)
    //   3. head_dim is valid (guards against divide-by-zero in exponent)
    if (meta_.rope_freq_base <= 10001.0f &&
        meta_.context_length > 8192 &&
        meta_.head_dim > 2)
    {
        float ctx_ratio = (float)meta_.context_length / (float)attn_cfg_.rope.n_ctx_orig;
        if (ctx_ratio > 1.0f) {
            // NTK formula: new_base = old_base -- ratio^(d / (d-2))
            // For head_dim=128, exponent = 128/126 --- 1.016. Small but meaningful.
            float exponent = (float)meta_.head_dim / (float)(meta_.head_dim - 2);
            attn_cfg_.rope.freq_base *= powf(ctx_ratio, exponent);
        }
    }

    fprintf(stderr, "[nexus] RoPE: base=%.0f  type=%d  n_ctx_orig=%u  context=%u\n",
            attn_cfg_.rope.freq_base,
            attn_cfg_.rope.type,
            attn_cfg_.rope.n_ctx_orig,
            meta_.context_length); fflush(stderr);

    NX_TRACE_MODEL("rope_base_global",    "%.0f", attn_cfg_.rope.freq_base);
    NX_TRACE_MODEL("rope_type",           "%d",   attn_cfg_.rope.type);
    NX_TRACE_MODEL("rope_n_ctx_orig",     "%u",   attn_cfg_.rope.n_ctx_orig);
    NX_TRACE_MODEL("rope_scale",          "%.4f", attn_cfg_.rope.scale);

    // Initialize KV cache
    init_kv_cache_for_model();

    // Allocate scratch buffers
    const int hidden = (int)meta_.hidden_dim;
    const int ffn    = (int)meta_.ffn_dim;
    const int q_dim  = (int)(meta_.n_heads * meta_.head_dim);
    const int kv_dim = (int)(meta_.n_kv_heads * meta_.head_dim);

    int scratch_q_dim  = q_dim;
    int scratch_kv_dim = kv_dim;

    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        // Gemma4 mixes attention widths by layer. Layer 0 can be smaller than
        // later global-attention layers. Scratch must fit the largest real layer,
        // or layer 5/11/17/etc can write past qkv_scratch_ and corrupt output.
        if (arch_l.find("gemma4") != std::string::npos) {
            for (const TransformerLayerWeights& lw : layer_weights_) {
                if (lw.attn_q_dim_layer  > scratch_q_dim)  scratch_q_dim  = lw.attn_q_dim_layer;
                if (lw.attn_kv_dim_layer > scratch_kv_dim) scratch_kv_dim = lw.attn_kv_dim_layer;
            }

            fprintf(stderr,
                "[nexus] Gemma4 scratch dims: base q=%d kv=%d -> scratch q=%d kv=%d total=%d\n",
                q_dim,
                kv_dim,
                scratch_q_dim,
                scratch_kv_dim,
                scratch_q_dim + 2 * scratch_kv_dim);
            fflush(stderr);
        }
    }

    // attention scratch: q + k + v + scores(max_ctx) + output
    const int max_ctx = (int)meta_.context_length;
    const size_t attn_scratch_sz =
        (size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim +
        (size_t)max_ctx + (size_t)scratch_q_dim + 1024;

    x_scratch_.resize(hidden, 0.0f);
    xb_scratch_.resize(hidden, 0.0f);
    attn_scratch_.resize(attn_scratch_sz, 0.0f);
    ffn_scratch_.resize(std::max((size_t)hidden, (size_t)ffn * 2), 0.0f);   // gate + up
    qkv_scratch_.resize((size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim, 0.0f);
    logits_.resize(meta_.vocab_size, 0.0f);

    if (progress_cb) progress_cb("Ready", 1.0f);

    fprintf(stderr, "\n[nexus] Model loaded successfully.\n"); fflush(stderr);
    mm_->print_report();

    loaded_ = true;
    return true;
}

// --------- Build layer weight index ---------------------------------------------------------------------------------------------------------------------------------------------------

inline bool Transformer::build_layer_weights() {
    auto& tensors = const_cast<std::unordered_map<std::string, TensorDesc>&>(parser_->tensors());

    f32_tensor_storage_.clear();

    auto try_find = [&](std::initializer_list<std::string> names) -> TensorDesc* {
        for (const auto& n : names) {
            auto it = tensors.find(n);
            if (it != tensors.end()) return &it->second;
        }
        return nullptr;
    };

    TensorDesc* emb = try_find({
        "token_embd.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "tok_embeddings.weight"
    });
    if (!emb) {
        fprintf(stderr, "[nexus] FATAL: could not find token embedding tensor\n");
        return false;
    }
    token_embedding_table = emb->data;
    emb_type = ggml_to_qtype(emb->type);

    TensorDesc* fnorm = try_find({
        "output_norm.weight",
        "model.norm.weight",
        "model.language_model.norm.weight",
        "norm.weight"
    });
    if (fnorm) rms_final_weight = tensor_as_f32(fnorm, "output_norm.weight");

    TensorDesc* fnorm_bias = try_find({
        "output_norm.bias",
        "model.norm.bias",
        "model.language_model.norm.bias",
        "norm.bias"
    });
    if (fnorm_bias && fnorm_bias->data) {
        rms_final_bias = tensor_as_f32(fnorm_bias, "output_norm.bias");
        fprintf(stderr, "[nexus] final norm bias indexed; using affine LayerNorm for final norm\n");
    } else {
        rms_final_bias = nullptr;
    }

    TensorDesc* lmh = try_find({
        "output.weight",
        "lm_head.weight",
        "model.lm_head.weight",
        "model.language_model.lm_head.weight"
    });
    if (!lmh) lmh = emb;
    lm_head_weight = lmh->data;
    lm_head_type   = ggml_to_qtype(lmh->type);
    lm_head_rows_  = meta_.hidden_dim ? (uint64_t)(lmh->n_elements / meta_.hidden_dim) : 0;
    fprintf(stderr, "[nexus] output (lm_head) weights: %s%s, %llu rows x %u, type id %d, %s\n", lmh->name.c_str(), (lmh == emb) ? " (shared with the token embedding)" : "",
            (unsigned long long)lm_head_rows_, (unsigned)meta_.hidden_dim, (int)lm_head_type, lm_head_weight ? "loaded" : "NOT LOADED on this computer");
    fflush(stderr);
    TensorDesc* lmh_bias = try_find({
        "output.bias",
        "lm_head.bias",
        "model.lm_head.bias",
        "model.language_model.lm_head.bias"
    });

    if (lmh_bias && lmh_bias->data) {
        lm_head_bias = tensor_as_f32(lmh_bias, "output.bias");
        fprintf(stderr, "[nexus] lm_head bias indexed\n");
    } else {
        lm_head_bias = nullptr;
    }

    // ─── Gemma 4: Per-Layer Embeddings (PLE) tensors ────────────────────────
    {
        std::string arch_l_ple = meta_.arch;
        for (char& c : arch_l_ple) c = (char)std::tolower((unsigned char)c);
        const bool is_gemma4 = arch_l_ple.find("gemma4") != std::string::npos;

        if (is_gemma4) {
            // Read ple_dim_ (hidden_size_per_layer_input). Default 256 per HF.
            ple_dim_ = (int)parser_->get_u32("gemma4.embedding_length_per_layer_input", 0);
            if (ple_dim_ == 0)
                ple_dim_ = (int)parser_->get_u32("gemma4.hidden_size_per_layer_input", 256);

            // Find tensors — tolerate a few naming variants since this is new.
            TensorDesc* td_ple_tok = try_find({
                "per_layer_token_embd.weight",
                "embed_tokens_per_layer.weight",
                "gemma4.per_layer_token_embd.weight",
            });

            TensorDesc* td_ple_proj = try_find({
                "per_layer_model_proj.weight",
                "per_layer_model_projection.weight",
                "per_layer_proj.weight",
                "gemma4.per_layer_model_proj.weight",
                "gemma4.per_layer_model_projection.weight",
                "gemma4.per_layer_proj.weight",
            });

            TensorDesc* td_ple_norm = try_find({
                "per_layer_proj_norm.weight",
                "per_layer_projection_norm.weight",
                "per_layer_norm.weight",
                "gemma4.per_layer_proj_norm.weight",
                "gemma4.per_layer_projection_norm.weight",
                "gemma4.per_layer_norm.weight",
            });

            fprintf(stderr,
                "[nexus] Gemma4 PLE tensor names: tok=%s proj=%s norm=%s ple_dim=%d\n",
                td_ple_tok  ? td_ple_tok->name.c_str()  : "(nil)",
                td_ple_proj ? td_ple_proj->name.c_str() : "(nil)",
                td_ple_norm ? td_ple_norm->name.c_str() : "(nil)",
                ple_dim_);
            fflush(stderr);

            if (td_ple_tok && td_ple_proj && td_ple_norm && ple_dim_ > 0) {
                ple_token_embed_      = td_ple_tok->data;
                ple_token_embed_type_ = ggml_to_qtype(td_ple_tok->type);

                ple_model_proj_       = td_ple_proj->data;
                ple_model_proj_type_  = ggml_to_qtype(td_ple_proj->type);

                ple_proj_norm_        = tensor_as_f32(td_ple_norm, "per_layer_projection_norm.weight");

                ple_enabled_          = true;
                ple_buffer_.assign((size_t)meta_.n_layers * (size_t)ple_dim_, 0.0f);

                fprintf(stderr,
                    "[nexus] Gemma4 PLE enabled: ple_dim=%d  n_layers=%u  buffer=%zu floats\n",
                    ple_dim_, meta_.n_layers, ple_buffer_.size());
            } else {
                fprintf(stderr,
                    "[nexus] Gemma4 PLE not present in GGUF (tok=%p proj=%p norm=%p ple_dim=%d) — "
                    "running without PLE residual. Quality may be reduced.\n",
                    (void*)td_ple_tok, (void*)td_ple_proj, (void*)td_ple_norm, ple_dim_);
            }
            fflush(stderr);
        }
    }
    
    // ─── Gemma4 global p-RoPE factor table ────────────────────────────────
    // GGUF tensor: rope_freqs.weight
    // This is required for Gemma4 global attention layers. Without this, the
    // global p-RoPE path only uses theta/base math and ignores the converter's
    // frequency factors.
    {
        std::string arch_l_rope_factors = meta_.arch;
        for (char& c : arch_l_rope_factors) {
            c = (char)std::tolower((unsigned char)c);
        }

        const bool is_gemma4 =
            arch_l_rope_factors.find("gemma4") != std::string::npos;

        if (is_gemma4) {
            TensorDesc* rf = try_find({
                "rope_freqs.weight",
                "gemma4.rope_freqs.weight",
            });

            if (rf && rf->data && rf->n_elements > 0) {
                gemma4_rope_factors_ = tensor_as_f32(rf, "rope_freqs.weight");
                gemma4_rope_factors_len_ = (size_t)rf->n_elements;

                fprintf(stderr,
                    "[nexus] Gemma4 rope_freqs.weight indexed: n=%zu first=%+.9g last=%+.9g\n",
                    gemma4_rope_factors_len_,
                    gemma4_rope_factors_ ? gemma4_rope_factors_[0] : 0.0f,
                    gemma4_rope_factors_ ? gemma4_rope_factors_[gemma4_rope_factors_len_ - 1] : 0.0f);
                fflush(stderr);
            } else {
                gemma4_rope_factors_ = nullptr;
                gemma4_rope_factors_len_ = 0;

                fprintf(stderr,
                    "[nexus] Gemma4 rope_freqs.weight NOT found; global p-RoPE will use fallback math\n");
                fflush(stderr);
            }
        }
    }

    layer_weights_.clear();
    layer_weights_.resize(meta_.n_layers);
    
    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        if (arch_l.find("gemma4") != std::string::npos) {
            const uint32_t old_heads    = meta_.n_heads;
            const uint32_t old_kv_heads = meta_.n_kv_heads;
            const uint32_t old_head_dim = meta_.head_dim;

            bool fixed = false;

            // Gemma4 GGUF metadata can report KV heads as if KV dim == Q dim.
            // Trust the actual tensor rows instead:
            //
            //   attn_q.weight shape [hidden, q_rows]
            //   attn_k.weight shape [hidden, kv_rows]
            //
            // For your crash layer:
            //   q_rows  = 16384
            //   k_rows  = 2048
            //   headdim = 512
            //   heads   = 32
            //   kvheads = 4
            for (int li = 0; li < (int)meta_.n_layers; ++li) {
                const std::string p = "blk." + std::to_string(li) + ".";

                auto qit = tensors.find(p + "attn_q.weight");
                auto kit = tensors.find(p + "attn_k.weight");

                if (qit == tensors.end() || kit == tensors.end()) continue;

                TensorDesc& q = qit->second;
                TensorDesc& k = kit->second;

                if (q.shape.size() < 2 || k.shape.size() < 2) continue;

                const uint32_t q_rows = (uint32_t)q.shape[1];
                const uint32_t k_rows = (uint32_t)k.shape[1];

                // Gemma4 in this GGUF is mixed-width by layer.
                // Prefer the actual q/k norm vector length when available.
                // The old code trusted metadata head_dim first, so it picked
                // layer 5's 512-wide path and then ran 256-wide layers wrong.
                auto qnit = tensors.find(p + "attn_q_norm.weight");
                auto knit = tensors.find(p + "attn_k_norm.weight");

                uint32_t hd = 0;

                if (qnit != tensors.end() &&
                    knit != tensors.end() &&
                    qnit->second.n_elements > 0 &&
                    qnit->second.n_elements == knit->second.n_elements &&
                    q_rows % (uint32_t)qnit->second.n_elements == 0 &&
                    k_rows % (uint32_t)knit->second.n_elements == 0)
                {
                    hd = (uint32_t)qnit->second.n_elements;
                }
                else if (knit != tensors.end() &&
                         knit->second.n_elements > 0 &&
                         q_rows % (uint32_t)knit->second.n_elements == 0 &&
                         k_rows % (uint32_t)knit->second.n_elements == 0)
                {
                    hd = (uint32_t)knit->second.n_elements;
                }
                else if (qnit != tensors.end() &&
                         qnit->second.n_elements > 0 &&
                         q_rows % (uint32_t)qnit->second.n_elements == 0 &&
                         k_rows % (uint32_t)qnit->second.n_elements == 0)
                {
                    hd = (uint32_t)qnit->second.n_elements;
                }
                else {
                    hd = meta_.head_dim;
                }

                if (hd == 0) continue;
                if (q_rows % hd != 0 || k_rows % hd != 0) continue;

                const uint32_t inferred_heads    = q_rows / hd;
                const uint32_t inferred_kv_heads = k_rows / hd;

                // Prefer the tensor that agrees with the declared Q head count.
                if (old_heads != 0 && inferred_heads != old_heads) {
                    continue;
                }

                if (inferred_heads > 0 && inferred_kv_heads > 0) {
                    meta_.head_dim   = hd;
                    meta_.n_heads    = inferred_heads;
                    meta_.n_kv_heads = inferred_kv_heads;
                    fixed = true;

                    fprintf(stderr,
                        "[nexus] Gemma4 metadata corrected from tensors: "
                        "heads %u->%u kv_heads %u->%u head_dim %u->%u "
                        "using layer %d q_rows=%u k_rows=%u\n",
                        old_heads,
                        meta_.n_heads,
                        old_kv_heads,
                        meta_.n_kv_heads,
                        old_head_dim,
                        meta_.head_dim,
                        li,
                        q_rows,
                        k_rows);
                    fflush(stderr);
                    break;
                }
            }

            if (!fixed) {
                fprintf(stderr,
                    "[nexus] Gemma4 metadata correction skipped: no usable attn_q/attn_k tensor pair found; "
                    "keeping heads=%u kv_heads=%u head_dim=%u\n",
                    meta_.n_heads,
                    meta_.n_kv_heads,
                    meta_.head_dim);
                fflush(stderr);
            }
        }
    }    

    {
        // GGUF Qwen3Next/Qwen3.5/Qwen3.6 converters usually bake zero-centered
        // RMSNorm weights into normal RMSNorm weights by adding +1 during
        // conversion. Runtime must NOT blindly apply (1 + weight) again.
        //
        // Detect from the loaded norm tensor itself:
        //   mean around 0 => raw zero-centered weight, use (1 + w)
        //   mean around 1 => GGUF-baked RMSNorm weight, use w directly
zero_centered_norm_ = false;

        // Architecture-aware override:
        // Qwen3-Next / Qwen3.5+ / Qwen3.6 explicitly use zero-centered RMSNorm
        // for ALL decoder norms (attn_norm, ffn_norm) per the official spec
        // ("zero-centered and weight-decayed layernorm"). The auto-probe below
        // can give the wrong answer because output_norm doesn't follow the
        // same convention as the per-layer norms in this family.
        std::string arch_l_norm = meta_.arch;
        for (char& c : arch_l_norm) c = (char)std::tolower((unsigned char)c);

        const bool is_qwen3_next_family =
            (arch_l_norm.find("qwen35") != std::string::npos) ||
            (arch_l_norm.find("qwen3_5") != std::string::npos) ||
            (arch_l_norm.find("qwen3.5") != std::string::npos) ||
            (arch_l_norm.find("qwen3_6") != std::string::npos) ||
            (arch_l_norm.find("qwen3.6") != std::string::npos) ||
            (arch_l_norm.find("qwen3next") != std::string::npos) ||
            (arch_l_norm.find("qwen3_next") != std::string::npos);

        bool native_raw_hf_tensor_layout = false;
        for (const auto& kv_tensor_native_probe : tensors) {
            const std::string& tn = kv_tensor_native_probe.first;
            const GGMLType tt = kv_tensor_native_probe.second.type;
            if (tn.rfind("model.language_model.", 0) == 0 ||
                tn.rfind("language_model.", 0) == 0 ||
                tt == GGMLType::NQ4_G64 ||
                tt == GGMLType::NQ5_G64 ||
                tt == GGMLType::NQ6_G64 ||
                tt == GGMLType::NQ8_G64) {
                native_raw_hf_tensor_layout = true;
                break;
            }
        }

        const bool qwen3next_native_raw_norms = is_qwen3_next_family && native_raw_hf_tensor_layout;

        if (qwen3next_native_raw_norms) {
            // Native .nxm is built from raw HF safetensors, not GGUF-converted
            // tensors. Qwen3NextRMSNorm inherits Gemma3RMSNorm semantics:
            //     output = rms_norm(x) * (1 + weight)
            // The GGUF path may bake the +1 into weights, but our native file
            // deliberately preserves the original HF tensors. Therefore qwen3next
            // native must force zero-centered RMSNorm for decoder + final norms.
            // GatedDeltaNet's per-head gated norm is handled separately and stays
            // direct weight semantics.
            zero_centered_norm_ = true;
            fprintf(stderr,
                "[nexus] Norm mode: native Qwen3Next raw HF tensors detected "
                "(arch=%s) -> FORCE decoder/final RMSNorm zero-centered (1+w); "
                "gated-deltanet norm remains direct\n",
                meta_.arch.c_str());
            fflush(stderr);
} else {
            // Auto-detect for everything else. Probe ORDER matters:
            //
            //   output_norm FIRST (matches original behavior — works for Mistral,
            //   Llama, Mixtral, OLMoE, etc. where output_norm has GGUF-baked
            //   weights with mean ~1).
            //
            //   Per-layer norm SECOND (only as a fallback for models that
            //   don't have output_norm.weight in the GGUF).
            //
            // Critical: do NOT probe per-layer norm first as primary signal.
            // Some models (Mistral, Ministral) have small-magnitude per-layer
            // norm weights (mean ~0.07) which are NOT zero-centered — they
            // are small because the residual stream is large. Probing those
            // first incorrectly classifies them as zero-centered and forces
            // (1+w) semantics, which destroys the model output.
            TensorDesc* norm_probe = nullptr;

            auto find_global_norm = [&]() -> TensorDesc* {
                auto it = tensors.find("output_norm.weight");
                if (it != tensors.end()) return &it->second;
                it = tensors.find("model.norm.weight");
                if (it != tensors.end()) return &it->second;
                return nullptr;
            };

            auto find_layer_norm = [&]() -> TensorDesc* {
                const int probe_start = std::max(0, active_layer_start_);
                const int probe_end   = active_layer_end_ >= 0 ? active_layer_end_ : (int)meta_.n_layers;

                for (int li = probe_start; li < probe_end; ++li) {
                    const std::string p = "blk." + std::to_string(li) + ".";

                    auto it = tensors.find(p + "attn_norm.weight");
                    if (it != tensors.end()) return &it->second;

                    it = tensors.find(p + "input_layernorm.weight");
                    if (it != tensors.end()) return &it->second;

                    it = tensors.find(p + "post_attention_norm.weight");
                    if (it != tensors.end()) return &it->second;
                }

                return nullptr;
            };

            // ORIGINAL ORDER: global first, layer fallback second.
            norm_probe = find_global_norm();
            if (!norm_probe) norm_probe = find_layer_norm();

            if (norm_probe && norm_probe->data && norm_probe->n_elements > 0) {
                const float* w = tensor_as_f32(norm_probe, "norm_probe");
                const size_t n = std::min<size_t>((size_t)norm_probe->n_elements, 256);

                double mean = 0.0;
                double mean_abs = 0.0;

                for (size_t i = 0; i < n; ++i) {
                    mean += (double)w[i];
                    mean_abs += std::fabs((double)w[i]);
                }

                mean /= (double)n;
                mean_abs /= (double)n;

                zero_centered_norm_ = (std::fabs(mean) < 0.25 && mean_abs < 0.50);

                fprintf(stderr,
                    "[nexus] Norm mode auto: tensor=%s mean=%.6f mean_abs=%.6f -> %s RMSNorm\n",
                    norm_probe->name.c_str(),
                    mean,
                    mean_abs,
                    zero_centered_norm_ ? "zero-centered (1+w)" : "direct GGUF weight");
                fflush(stderr);
            } else {
                fprintf(stderr,
                    "[nexus] Norm mode auto: no norm probe found -> direct GGUF RMSNorm weight\n");
                fflush(stderr);
            }

            // ─── Gemma family: ARCHITECTURAL rule, no probing ───────────────
            //
            // Gemma 2 / Gemma 3 / Gemma 4 ALL use zero-centered (1+w) RMSNorm
            // for every norm tensor. This is from the HF reference:
            //
            //   class Gemma2RMSNorm(nn.Module):
            //       def __init__(self, dim, eps=1e-6):
            //           self.weight = nn.Parameter(torch.zeros(dim))
            //       def forward(self, x):
            //           return _norm(x) * (1.0 + self.weight)
            //
            // The +1 is ALWAYS present at inference time. The stored weight
            // mean drifting away from 0 (e.g. 4.8 for attn_norm on 31B) is
            // simply the trained weight value — the math is still (1+4.8)
            // = 5.8, not 4.8 directly.
            //
            // Previously we PROBED the weight statistics and guessed wrong:
            // mean=4.8 looked "too far from zero to be raw HF" so we used
            // direct(w). That was the bug. Every public Gemma 4 GGUF on
            // HuggingFace (unsloth, lmstudio, ggml-org) uses the standard
            // converter which keeps weights raw HF style — meaning (1+w) is
            // ALWAYS the right choice. No probing needed.
            const bool is_gemma_family =
                arch_l_norm.find("gemma") != std::string::npos;

            if (is_gemma_family) {
                const bool was = zero_centered_norm_;
                zero_centered_norm_ = true;
                fprintf(stderr,
                    "[nexus] Gemma norm: arch=%s -> FORCED zero-centered (1+w) "
                    "per HF Gemma2/3/4 RMSNorm spec (was: %s)\n",
                    meta_.arch.c_str(),
                    was ? "zero-centered (1+w)" : "direct (w)");
                fflush(stderr);
            }
        }

        // Runtime override for testing convention without recompiling.
        // NEXUS_NORM_CONVENTION={direct|zero_centered} forces the choice.
        if (const char* env = std::getenv("NEXUS_NORM_CONVENTION")) {
            bool was = zero_centered_norm_;
            if (std::string(env) == "zero_centered" || std::string(env) == "1")
                zero_centered_norm_ = true;
            else if (std::string(env) == "direct" || std::string(env) == "0")
                zero_centered_norm_ = false;
            fprintf(stderr,
                "[nexus] Norm convention OVERRIDE by NEXUS_NORM_CONVENTION=%s: "
                "was=%s now=%s\n",
                env,
                was ? "zero_centered (1+w)" : "direct (w)",
                zero_centered_norm_ ? "zero_centered (1+w)" : "direct (w)");
            fflush(stderr);
        }

        NX_TRACE_MODEL("zero_centered_norm", "%d (%s) %s",
            zero_centered_norm_ ? 1 : 0,
            zero_centered_norm_ ? "x*(1+w)" : "x*w",
            std::getenv("NEXUS_NORM_CONVENTION") ? "[ENV-OVERRIDE]" : "[auto]");
    }

    const int active_start = active_layer_start_;
    const int active_end   = (active_layer_end_ >= 0) ? active_layer_end_ : (int)meta_.n_layers;

    fprintf(stderr,
        "[nexus] Building layer index: arch=%s layers=%u hidden=%u head_dim=%u ffn=%u experts=%u used=%u active=%d-%d\n",
        meta_.arch.c_str(), meta_.n_layers, meta_.hidden_dim, meta_.head_dim,
        meta_.ffn_dim, meta_.n_experts, meta_.n_experts_used,
        active_start, active_end - 1);

    for (int l = 0; l < (int)meta_.n_layers; ++l) {
        if (active_layer_end_ >= 0 && (l < active_start || l >= active_end)) {
            continue;
        }

        TransformerLayerWeights& lw = layer_weights_[l];
        std::string prefix = "blk." + std::to_string(l) + ".";

        auto get = [&](const std::string& suffix) -> TensorDesc* {
            // Centralized lookup handles both GGUF blk.N.* and native .nxm
            // model.language_model.layers.N.* names plus Qwen3Next aliases.
            return find_layer(l, suffix);
        };

        auto set = [&](const void*& ptr, QuantType& qt, std::initializer_list<std::string> names) {
            for (const auto& n : names) {
                TensorDesc* t = get(n);
                if (t) {
                    ptr = t->data;
                    qt  = ggml_to_qtype(t->type);
                    return;
                }
            }
        };

        auto has_any = [&](std::initializer_list<std::string> names) -> bool {
            for (const auto& n : names) {
                if (get(n)) return true;
            }
            return false;
        };

        auto dump_layer_tensors = [&](const char* reason) {
            fprintf(stderr, "\n[nexus] Layer %d index failure: %s\n", l, reason);
            const std::string native_prefix = "model.language_model.layers." + std::to_string(l) + ".";
            fprintf(stderr, "[nexus] Prefix: %s OR %s OR model.layers.%d. OR language_model.model.layers.%d.\n", prefix.c_str(), native_prefix.c_str(), l, l);
            fprintf(stderr, "[nexus] Layer %d tensors present:\n", l);

            int shown = 0;
            for (const auto& kv : tensors) {
                const std::string& name = kv.first;
                const TensorDesc&  t    = kv.second;
                if (name.rfind(prefix, 0) != 0 && name.rfind(native_prefix, 0) != 0) continue;

                fprintf(stderr, "  %-64s shape=[", name.c_str());
                for (size_t i = 0; i < t.shape.size(); ++i) {
                    fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)t.shape[i]);
                }
                fprintf(stderr, "] type=%d bytes=%llu loaded=%d\n",
                        (int)t.type, (unsigned long long)t.data_size, t.loaded ? 1 : 0);

                if (++shown >= 160) {
                    fprintf(stderr, "  ... truncated after 160 tensors ...\n");
                    break;
                }
            }

            if (shown == 0) {
                fprintf(stderr, "  <none with blk.%d or native HF layer prefix>\n", l);
            }
            fprintf(stderr, "\n");
        };

        const bool has_moe =
            has_any({
                "ffn_gate_inp.weight",
                "ffn_gate_inp.bias",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
                "ffn_gate_shexp.weight",
                "ffn_up_shexp.weight",
                "ffn_down_shexp.weight",
                "ffn_gate_up_exps.weight",
                "exp_probs_b.bias"
            });

        const bool has_hybrid_or_alt =
            has_any({
                "mixer.q_proj.weight",
                "mixer.k_proj.weight",
                "mixer.v_proj.weight",
                "mixer.o_proj.weight",
                "mixer.gate.weight",
                "linear_attn.in_proj_qkv.weight",
                "ssm_in.weight",
                "ssm_x.weight",
                "ssm_dt.weight",
                "ssm_a.weight",
                "ssm_d.weight"
            });

        // Norm aliases
        TensorDesc* an = get("attn_norm.weight");
        if (!an) an = get("input_layernorm.weight");
        if (!an) an = get("pre_attention_layernorm.weight");
        if (!an) an = get("pre_feedforward_layernorm.weight");
        if (!an) an = get("pre_mlp_layernorm.weight");
        if (!an) an = get("mixer_norm.weight");
        if (an) {
            lw.attn_norm      = tensor_as_f32(an, "attn_norm.weight");
            lw.attn_norm_type = QuantType::F32;
        }
        TensorDesc* anb = get("attn_norm.bias");
        if (!anb) anb = get("input_layernorm.bias");
        if (!anb) anb = get("pre_attention_layernorm.bias");
        if (!anb) anb = get("pre_feedforward_layernorm.bias");
        if (!anb) anb = get("pre_mlp_layernorm.bias");
        if (!anb) anb = get("mixer_norm.bias");
        if (anb) lw.attn_norm_bias = tensor_as_f32(anb, "attn_norm.bias");

        TensorDesc* fn = get("ffn_norm.weight");

        // Qwen3.6 / hybrid MoE uses this exact name.
        if (!fn) fn = get("post_attention_norm.weight");

        // Other common aliases.
        if (!fn) fn = get("post_attention_layernorm.weight");
        if (!fn) fn = get("post_feedforward_norm.weight");
        if (!fn) fn = get("post_feedforward_layernorm.weight");
        if (!fn) fn = get("post_mlp_norm.weight");
        if (!fn) fn = get("post_mlp_layernorm.weight");
        if (!fn) fn = get("post_moe_norm.weight");
        if (!fn) fn = get("pre_ff_norm.weight");
        if (!fn) fn = get("pre_ff_layernorm.weight");
        if (fn) {
            lw.ffn_norm      = tensor_as_f32(fn, "ffn_norm.weight");
            lw.ffn_norm_type = QuantType::F32;
        }
        TensorDesc* fnb = get("ffn_norm.bias");
        if (!fnb) fnb = get("post_attention_norm.bias");
        if (!fnb) fnb = get("post_attention_layernorm.bias");
        if (!fnb) fnb = get("post_feedforward_norm.bias");
        if (!fnb) fnb = get("post_feedforward_layernorm.bias");
        if (!fnb) fnb = get("post_mlp_norm.bias");
        if (!fnb) fnb = get("post_mlp_layernorm.bias");
        if (!fnb) fnb = get("post_moe_norm.bias");
        if (!fnb) fnb = get("pre_ff_norm.bias");
        if (!fnb) fnb = get("pre_ff_layernorm.bias");
        if (fnb) lw.ffn_norm_bias = tensor_as_f32(fnb, "ffn_norm.bias");

        // ---- Sandwich norms (Gemma 2/3/4) -------------------------------------
        // Gemma uses FOUR norms per layer instead of the usual two:
        //   pre-attn (attn_norm) -> attn -> post-attn -> + residual ->
        //   pre-ffn  (ffn_norm)  -> ffn  -> post-ffn  -> + residual
        //
        // Without the post-norms, the residual stream grows unbounded across
        // 60+ layers and the model collapses to repetitive output ("de de de").
        //
        // Gated by arch detection: non-Gemma models skip this entirely so we
        // don't accidentally double-load tensors that some other arch uses
        // under one of these names (e.g. Qwen3.6's post_attention_norm fallback
        // already handled by the chain above).
        {
            std::string arch_l = meta_.arch;
            for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

            const bool is_gemma = (arch_l.find("gemma") != std::string::npos);

            if (is_gemma) {
                TensorDesc* pan = get("post_attention_norm.weight");
                if (!pan) pan = get("post_attention_layernorm.weight");
                if (pan) {
                    lw.attn_post_norm = tensor_as_f32(pan, "post_attention_norm.weight");
                }

                TensorDesc* pfn = get("post_ffw_norm.weight");
                if (!pfn) pfn = get("post_feedforward_layernorm.weight");
                if (!pfn) pfn = get("post_feedforward_norm.weight");
                if (!pfn) pfn = get("post_mlp_norm.weight");
                if (!pfn) pfn = get("post_mlp_layernorm.weight");
                if (pfn) {
                    lw.ffn_post_norm = tensor_as_f32(pfn, "post_ffw_norm.weight");
                }

                // Print on first owned layer only.
                if (l == active_start) {
                    fprintf(stderr,
                        "[nexus] Gemma sandwich norms: layer %d attn_post=%s ffn_post=%s\n",
                        l,
                        lw.attn_post_norm ? "yes" : "no",
                        lw.ffn_post_norm  ? "yes" : "no");
                    fflush(stderr);
                }
            }
        }
        
        // ---- Gemma4 layer output scale -----------------------------------------
        // GGUF: blk.N.layer_output_scale.weight, shape [1]
        // Must be indexed for EVERY layer, not only active_start.
        {
            std::string arch_l = meta_.arch;
            for (char& c : arch_l) {
                c = (char)std::tolower((unsigned char)c);
            }

            const bool is_gemma4 = (arch_l.find("gemma4") != std::string::npos);

            if (is_gemma4) {
                TensorDesc* los = get("layer_output_scale.weight");
                if (!los) los = get("layer_scalar");
                if (!los) los = get("layer_scalar.weight");

                if (los && los->data && los->n_elements > 0) {
                    lw.layer_output_scale     = tensor_as_f32(los, "layer_output_scale.weight");
                    lw.layer_output_scale_len = (size_t)los->n_elements;
                }

                if (l < active_start + 8 || l == active_end - 1) {
                    fprintf(stderr,
                        "[nexus] Gemma4 layer_output_scale layer %d: %s len=%zu first=%+.6g last=%+.6g\n",
                        l,
                        lw.layer_output_scale ? "yes" : "no",
                        lw.layer_output_scale_len,
                        lw.layer_output_scale ? lw.layer_output_scale[0] : 1.0f,
                        (lw.layer_output_scale && lw.layer_output_scale_len > 1)
                            ? lw.layer_output_scale[lw.layer_output_scale_len - 1] : 1.0f);
                    fflush(stderr);
                }
            }
        }

        // Attention aliases. These keep dense/self-attention models working and
        // catch some newer GGUF canonical names.
        set(lw.wq, lw.wq_type, {
            "attn_q.weight",
            "self_attn.q_proj.weight",
            "attention.wq.weight",
            "mixer.q_proj.weight",
            "q_proj.weight"
        });
        set(lw.wk, lw.wk_type, {
            "attn_k.weight",
            "self_attn.k_proj.weight",
            "attention.wk.weight",
            "mixer.k_proj.weight",
            "k_proj.weight"
        });
        set(lw.wv, lw.wv_type, {
            "attn_v.weight",
            "self_attn.v_proj.weight",
            "attention.wv.weight",
            "mixer.v_proj.weight",
            "v_proj.weight"
        });
        {
            std::string arch_l = meta_.arch;
            for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

            // ─── Gemma 4: KV-tensor handling ────────────────────────────────
            //
            // Gemma 4 GGUFs have THREE distinct shapes for K/V at each layer:
            //
            //   (a) Both Wk and Wv present.            Normal layer.
            //   (b) Wk present, Wv missing.            Tied-K/V layer — V is
            //                                          aliased to K's weights.
            //                                          This is the GGUF
            //                                          converter's choice for
            //                                          this Gemma 4 build; the
            //                                          reference model uses
            //                                          k_proj for both.
            //   (c) Both Wk and Wv missing.            "Shared KV cache" layer
            //                                          per Gemma 4 docs — reuse
            //                                          the K/V cache from an
            //                                          earlier same-attention-
            //                                          type layer (the donor).
            //
            // Earlier versions of this code conflated (b) and (c). Crashed at
            // L5 of gemma-4-31B-it because L5 is case (b) — Wk present, Wv
            // missing — and we wrongly took the donor path, then the validator
            // below rejected the still-null Wv.
            const bool is_gemma4 = arch_l.find("gemma4") != std::string::npos;
            const bool full_shared_kv = is_gemma4 && !lw.wv && !lw.wk;
            const bool tied_v_to_k    = is_gemma4 && !lw.wv &&  lw.wk;

            if (full_shared_kv) {
                // Case (c). Flag for second-pass donor resolution.
                lw.kv_donor_layer = -2;
                fprintf(stderr,
                    "[nexus] Layer %d: Gemma4 full shared-KV (both Wk and Wv "
                    "absent) — donor to be resolved in second pass\n", l);
                fflush(stderr);
            } else if (tied_v_to_k) {
                // Case (b). Alias V = K. Same as the pre-fix-#5 behaviour
                // because this is what the GGUF format actually uses for
                // these layers — the converter expects loaders to compute
                // V = Wk @ x. (Distinct from full shared-KV where the model
                // is meant to reuse cached K/V from an earlier layer.)
                lw.wv = lw.wk;
                lw.wv_type = lw.wk_type;
                lw.gemma4_tied_kv = true;
                fprintf(stderr,
                    "[nexus] Layer %d: Gemma4 V=K alias (Wv absent, Wk "
                    "present)\n", l);
                fflush(stderr);
            }

            // ─── Gemma 4: per-layer SWA + dual RoPE setup ─────────────────
            //
            // Pattern: every `sliding_window_pattern` layers, exactly one is
            // GLOBAL (full causal, large RoPE base); the rest are LOCAL
            // (windowed, standard RoPE base). The conventional pattern is 6 → 5 local + 1 global.
            //
            // GGUF keys (with sensible fallbacks):
            //   gemma4.attention.sliding_window         (default 1024 for 31B / 26B, 512 for E2B/E4B)
            //   gemma4.attention.sliding_window_pattern (default 6)
            //   gemma4.rope.local_base_freq             (default 10000)
            //   gemma4.rope.global_base_freq            (default 1000000)
            //
            // Older / alternate GGUFs may use `gemma3.*` or generic
            // `<arch>.attention.sliding_window`; we accept both.
            if (arch_l.find("gemma4") != std::string::npos ||
                arch_l.find("gemma3") != std::string::npos) {

                // Read once (cheap, parser is in-memory)
                uint32_t sw = parser_->get_u32("gemma4.attention.sliding_window", 0);
                if (sw == 0) sw = parser_->get_u32("gemma3.attention.sliding_window", 0);
                if (sw == 0) sw = parser_->get_u32(meta_.arch + ".attention.sliding_window", 0);
                if (sw == 0) {
                    // Reasonable defaults by model size (head count is a decent proxy)
                    sw = (meta_.n_heads >= 16) ? 1024 : 512;
                }

                uint32_t pat = parser_->get_u32("gemma4.attention.sliding_window_pattern", 0);
                if (pat == 0) pat = parser_->get_u32("gemma3.attention.sliding_window_pattern", 0);
                if (pat == 0) pat = 6;  // Gemma 3/4 default: 5 local + 1 global

                float rope_local  = parser_->get_f32("gemma4.rope.local_base_freq", 0.0f);
                if (rope_local  == 0.0f) rope_local  = parser_->get_f32("gemma3.rope.local_base_freq", 0.0f);
                if (rope_local  == 0.0f) rope_local  = 10000.0f;

                float rope_global = parser_->get_f32("gemma4.rope.global_base_freq", 0.0f);
                if (rope_global == 0.0f) rope_global = parser_->get_f32("gemma3.rope.global_base_freq", 0.0f);
                if (rope_global == 0.0f) rope_global = meta_.rope_freq_base;          // GGUF-stored value
                if (rope_global == 0.0f || rope_global == 10000.0f) rope_global = 1000000.0f;

                // The last layer in each pattern block is GLOBAL.
                // i.e. layer L is global iff ((L + 1) % pat) == 0, equivalently (L % pat) == pat-1.
                const bool is_global = ((l % (int)pat) == (int)pat - 1);

                if (is_global) {
                    lw.sw_window_size       = 0;             // full causal
                    lw.rope_freq_base_layer = rope_global;
                } else {
                    lw.sw_window_size       = (int)sw;       // local SWA
                    lw.rope_freq_base_layer = rope_local;
                }

                // PLE: this layer reads ple_buffer_[L*ple_dim_..(L+1)*ple_dim_]
                if (ple_enabled_) {
                    lw.ple_layer_index = l;
                }

                // Log only the first few layers so we don't spam at load
                static std::atomic<int> gemma4_layer_logs{0};
                if (gemma4_layer_logs.fetch_add(1) < 8) {
                    fprintf(stderr,
                        "[nexus] Gemma4 layer %d: %s  window=%d  rope_base=%.0f  (pat=%u)\n",
                        l,
                        is_global ? "GLOBAL" : "LOCAL ",
                        lw.sw_window_size,
                        lw.rope_freq_base_layer,
                        pat);
                    fflush(stderr);
                }
            }
        }        
        set(lw.wo, lw.wo_type, {
            "attn_output.weight",
            "attn_out.weight",
            "self_attn.o_proj.weight",
            "self_attn.out_proj.weight",
            "attention.wo.weight",
            "mixer.o_proj.weight",
            "out_proj.weight",
            "o_proj.weight",

            // Qwen3.6 / hybrid-MoE GGUF path.
            // These layers have attn_qkv.weight but no attn_output.weight.
            // blk.N.ssm_out.weight has shape [q_dim, hidden] and is the available
            // projection back to hidden for the attention/mixer output.
            "ssm_out.weight"
        });

        if (!lw.wq) {
            TensorDesc* qkv = get("attn_qkv.weight");
            if (!qkv) qkv = get("self_attn.qkv_proj.weight");
            if (!qkv) qkv = get("mixer.in_proj.weight");
            if (!qkv) qkv = get("linear_attn.in_proj_qkv.weight");

            if (qkv) {
                lw.wqkv      = qkv->data;
                lw.wqkv_type = ggml_to_qtype(qkv->type);
                lw.fused_qkv = true;
                fprintf(stderr, "[nexus] Layer %d: using fused QKV tensor %s\n",
                        l, qkv->name.c_str());
            }
        }

        // Dense FFN aliases.
        set(lw.wgate, lw.wgate_type, {
            "ffn_gate.weight",
            "mlp.gate_proj.weight",
            "feed_forward.gate_proj.weight",
            "feed_forward.w1.weight",
            "mlp.w1.weight"
        });
        set(lw.wup, lw.wup_type, {
            "ffn_up.weight",
            "mlp.up_proj.weight",
            "feed_forward.up_proj.weight",
            "feed_forward.w3.weight",
            "mlp.w3.weight",
            "mlp.gate_up_proj.weight"
        });
        set(lw.wdown, lw.wdown_type, {
            "ffn_down.weight",
            "mlp.down_proj.weight",
            "feed_forward.down_proj.weight",
            "feed_forward.w2.weight",
            "mlp.w2.weight"
        });
        
        // ---- MoE routed FFN tensors -------------------------------------------------
        // These are used by newer GGUF MoE models. Dense models leave all of this empty.
        auto set_moe_tensor = [&](TensorDesc* t, const void*& ptr, QuantType& qt, size_t& stride) {
            if (!t) return;
            ptr = t->data;
            qt  = ggml_to_qtype(t->type);

            // For merged expert tensors, the last dim is normally n_experts and each
            // expert's matrix is contiguous. This keeps quantized expert slices aligned.
            if (t->shape.size() >= 3 && t->shape[2] > 0) {
                stride = (size_t)(t->data_size / t->shape[2]);
            } else {
                stride = (size_t)t->data_size;
            }
        };
        
        auto infer_expert_linear = [&](TensorDesc* t,
                                       uint32_t expected_in,
                                       uint32_t expected_out,
                                       uint32_t& rows,
                                       uint32_t& cols,
                                       bool& transposed,
                                       const char* label) -> bool {
            rows = 0;
            cols = 0;
            transposed = false;

            if (!t || t->shape.size() < 2) {
                fprintf(stderr, "[nexus] MoE shape error: %s tensor missing or rank < 2\n", label);
                return false;
            }

            const uint32_t a = (uint32_t)t->shape[0];
            const uint32_t b = (uint32_t)t->shape[1];

            // Normal GGUF linear layout:
            //   shape[0] = input columns
            //   shape[1] = output rows
            //
            // Important: trust the tensor's output row count. Do NOT force
            // expected_out here. Qwen A3B routed experts use hidden->512 even
            // when the model-level FFN metadata is much larger.
            if (a == expected_in) {
                cols = a;
                rows = b;
                transposed = false;
                return true;
            }

            // Transposed/storage-flipped layout:
            //   shape[0] = output rows
            //   shape[1] = input columns
            if (b == expected_in) {
                rows = a;
                cols = b;
                transposed = true;
                return true;
            }

            // Last-chance fallback when expected_out is known.
            if (expected_out != 0 && a == expected_out) {
                rows = a;
                cols = b;
                transposed = true;
                return true;
            }

            fprintf(stderr,
                "[nexus] MoE shape error: %s cannot infer rows/cols from shape=[",
                label);
            for (size_t i = 0; i < t->shape.size(); ++i) {
                fprintf(stderr, "%s%llu", i ? "," : "", (unsigned long long)t->shape[i]);
            }
            fprintf(stderr, "] expected_in=%u expected_out=%u\n", expected_in, expected_out);
            return false;
        };

        TensorDesc* router = get("ffn_gate_inp.weight");
        if (!router) router = get("router.weight");
        if (!router) router = get("mlp.gate.weight");
        if (!router) router = get("mlp.router.weight");
        if (!router) router = get("block_sparse_moe.gate.weight");
        if (!router) router = get("feed_forward.gate.weight");

        TensorDesc* router_bias = get("exp_probs_b.bias");
        if (!router_bias) router_bias = get("router.bias");
        if (!router_bias) router_bias = get("mlp.gate.bias");
        if (!router_bias) router_bias = get("mlp.router.bias");

        TensorDesc* egate = get("ffn_gate_exps.weight");
        if (!egate) egate = get("mlp.experts.gate_proj.weight");
        if (!egate) egate = get("block_sparse_moe.experts.gate_proj.weight");

        TensorDesc* eup = get("ffn_up_exps.weight");
        if (!eup) eup = get("mlp.experts.up_proj.weight");
        if (!eup) eup = get("block_sparse_moe.experts.up_proj.weight");

        TensorDesc* edown = get("ffn_down_exps.weight");
        if (!edown) edown = get("mlp.experts.down_proj.weight");
        if (!edown) edown = get("block_sparse_moe.experts.down_proj.weight");

        // Some models fuse gate+up experts into one tensor.
        TensorDesc* egateup = get("ffn_gate_up_exps.weight");
        if (!egateup) egateup = get("mlp.experts.gate_up_proj.weight");
        if (!egateup) egateup = get("block_sparse_moe.experts.gate_up_proj.weight");

        // Mixtral/OpenGPT-style split experts:
        //   blk.N.ffn_gate.0.weight / blk.N.ffn_up.0.weight / blk.N.ffn_down.0.weight
        std::vector<TensorDesc*> split_gate;
        std::vector<TensorDesc*> split_up;
        std::vector<TensorDesc*> split_down;

        auto collect_split_experts = [&]() -> bool {
            split_gate.clear();
            split_up.clear();
            split_down.clear();

            const uint32_t declared = meta_.n_experts;
            const uint32_t scan_limit = declared ? declared : 1024;

            for (uint32_t e = 0; e < scan_limit; ++e) {
                char name[128];

                snprintf(name, sizeof(name), "ffn_gate.%u.weight", e);
                TensorDesc* g = get(name);

                snprintf(name, sizeof(name), "ffn_up.%u.weight", e);
                TensorDesc* u = get(name);

                snprintf(name, sizeof(name), "ffn_down.%u.weight", e);
                TensorDesc* d = get(name);

                if (!g || !u || !d) {
                    if (declared) {
                        split_gate.clear();
                        split_up.clear();
                        split_down.clear();
                        return false;
                    }
                    break;
                }

                split_gate.push_back(g);
                split_up.push_back(u);
                split_down.push_back(d);
            }

            return !split_gate.empty();
        };

        const bool split_moe = collect_split_experts();

        if (egate || eup || edown || egateup || split_moe) {
            lw.moe = true;

            if (router) {
                lw.moe_router      = router->data;
                lw.moe_router_type = ggml_to_qtype(router->type);
            }
            if (router_bias) {
                lw.moe_router_bias = tensor_as_f32(router_bias, "exp_probs_b.bias");
            }

            if (split_moe) {
                lw.moe_split_experts = true;

                lw.wgate_exp_ptrs.clear();
                lw.wup_exp_ptrs.clear();
                lw.wdown_exp_ptrs.clear();

                lw.wgate_exp_types.clear();
                lw.wup_exp_types.clear();
                lw.wdown_exp_types.clear();

                for (size_t e = 0; e < split_gate.size(); ++e) {
                    lw.wgate_exp_ptrs.push_back(split_gate[e]->data);
                    lw.wup_exp_ptrs.push_back(split_up[e]->data);
                    lw.wdown_exp_ptrs.push_back(split_down[e]->data);

                    lw.wgate_exp_types.push_back(ggml_to_qtype(split_gate[e]->type));
                    lw.wup_exp_types.push_back(ggml_to_qtype(split_up[e]->type));
                    lw.wdown_exp_types.push_back(ggml_to_qtype(split_down[e]->type));
                }

                if (!infer_expert_linear(split_gate[0],
                                         meta_.hidden_dim,
                                         0,
                                         lw.moe_gate_rows,
                                         lw.moe_gate_cols,
                                         lw.moe_gate_transposed,
                                         "split_gate")) {
                    dump_layer_tensors("bad split MoE gate expert shape");
                    return false;
                }

                if (!infer_expert_linear(split_up[0],
                                         meta_.hidden_dim,
                                         lw.moe_gate_rows,
                                         lw.moe_up_rows,
                                         lw.moe_up_cols,
                                         lw.moe_up_transposed,
                                         "split_up")) {
                    dump_layer_tensors("bad split MoE up expert shape");
                    return false;
                }

                if (lw.moe_gate_rows != lw.moe_up_rows) {
                    fprintf(stderr,
                        "[nexus] MoE shape error: split gate_rows=%u up_rows=%u mismatch\n",
                        lw.moe_gate_rows, lw.moe_up_rows);
                    dump_layer_tensors("bad split MoE gate/up FFN mismatch");
                    return false;
                }

                lw.moe_ffn_dim = lw.moe_gate_rows;

                if (!infer_expert_linear(split_down[0],
                                         lw.moe_ffn_dim,
                                         meta_.hidden_dim,
                                         lw.moe_down_rows,
                                         lw.moe_down_cols,
                                         lw.moe_down_transposed,
                                         "split_down")) {
                    dump_layer_tensors("bad split MoE down expert shape");
                    return false;
                }
            } else {
                if (egateup && !egate && !eup) {
                    lw.moe_fused_gate_up = true;
                    set_moe_tensor(egateup, lw.wup_exps, lw.wup_exps_type, lw.wup_exps_stride);
                } else {
                    set_moe_tensor(egate, lw.wgate_exps, lw.wgate_exps_type, lw.wgate_exps_stride);
                    set_moe_tensor(eup,   lw.wup_exps,   lw.wup_exps_type,   lw.wup_exps_stride);
                }
                set_moe_tensor(edown, lw.wdown_exps, lw.wdown_exps_type, lw.wdown_exps_stride);

                if (lw.moe_fused_gate_up) {
                    if (!infer_expert_linear(egateup, meta_.hidden_dim, meta_.ffn_dim ? meta_.ffn_dim * 2 : 0,
                                             lw.moe_fused_rows, lw.moe_fused_cols,
                                             lw.moe_fused_transposed,
                                             "fused_gate_up_exps")) {
                        dump_layer_tensors("bad MoE fused gate/up expert shape");
                        return false;
                    }

                    if ((lw.moe_fused_rows % 2) != 0) {
                        fprintf(stderr,
                            "[nexus] MoE shape error: fused gate/up rows=%u is not even\n",
                            lw.moe_fused_rows);
                        dump_layer_tensors("bad MoE fused gate/up row count");
                        return false;
                    }

                    lw.moe_ffn_dim = lw.moe_fused_rows / 2;
                } else {
                    if (!infer_expert_linear(egate, meta_.hidden_dim, meta_.ffn_dim,
                                             lw.moe_gate_rows, lw.moe_gate_cols,
                                             lw.moe_gate_transposed,
                                             "gate_exps")) {
                        dump_layer_tensors("bad MoE gate expert shape");
                        return false;
                    }

                    if (!infer_expert_linear(eup, meta_.hidden_dim, lw.moe_gate_rows,
                                             lw.moe_up_rows, lw.moe_up_cols,
                                             lw.moe_up_transposed,
                                             "up_exps")) {
                        dump_layer_tensors("bad MoE up expert shape");
                        return false;
                    }

                    if (lw.moe_gate_rows != lw.moe_up_rows) {
                        fprintf(stderr,
                            "[nexus] MoE shape error: gate_rows=%u up_rows=%u mismatch\n",
                            lw.moe_gate_rows, lw.moe_up_rows);
                        dump_layer_tensors("bad MoE gate/up FFN mismatch");
                        return false;
                    }

                    lw.moe_ffn_dim = lw.moe_gate_rows;
                }

                if (!infer_expert_linear(edown, lw.moe_ffn_dim, meta_.hidden_dim,
                                         lw.moe_down_rows, lw.moe_down_cols,
                                         lw.moe_down_transposed,
                                         "down_exps")) {
                    dump_layer_tensors("bad MoE down expert shape");
                    return false;
                }
            }

            if (lw.moe_down_rows != meta_.hidden_dim || lw.moe_down_cols != lw.moe_ffn_dim) {
                fprintf(stderr,
                    "[nexus] MoE shape error: down rows/cols are %u/%u, expected hidden/ffn %u/%u\n",
                    lw.moe_down_rows, lw.moe_down_cols,
                    meta_.hidden_dim, lw.moe_ffn_dim);
                dump_layer_tensors("bad MoE down projection dimensions");
                return false;
            }

            fprintf(stderr,
                "[nexus] Layer %d: MoE dims layout=%s gate=%ux%u%s up=%ux%u%s fused=%ux%u%s down=%ux%u%s\n",
                l,
                lw.moe_split_experts ? "split" : "merged",
                lw.moe_gate_rows, lw.moe_gate_cols, lw.moe_gate_transposed ? " T" : " N",
                lw.moe_up_rows, lw.moe_up_cols, lw.moe_up_transposed ? " T" : " N",
                lw.moe_fused_rows, lw.moe_fused_cols, lw.moe_fused_transposed ? " T" : " N",
                lw.moe_down_rows, lw.moe_down_cols, lw.moe_down_transposed ? " T" : " N");

            uint32_t experts_from_shape = 0;
            if (split_moe) experts_from_shape = (uint32_t)split_gate.size();
            else if (edown && edown->shape.size() >= 3) experts_from_shape = (uint32_t)edown->shape[2];
            else if (eup && eup->shape.size() >= 3) experts_from_shape = (uint32_t)eup->shape[2];
            else if (egate && egate->shape.size() >= 3) experts_from_shape = (uint32_t)egate->shape[2];
            else if (egateup && egateup->shape.size() >= 3) experts_from_shape = (uint32_t)egateup->shape[2];

            lw.moe_expert_count = meta_.n_experts ? meta_.n_experts : experts_from_shape;
            lw.moe_top_k = meta_.n_experts_used ? meta_.n_experts_used : 2;
            
            // Router kinds:
            //   0 = softmax top-k
            //   1 = sparse top-2 mixer, currently experimental
            //   2 = sigmoid top-k
            //
            // Do not guess Phi/SlimMoE sparse routing here. A wrong sparse router
            // produces fluent garbage. Keep the universal path safe first.
            {
                std::string arch = meta_.arch;
                std::string model_name = meta_.name;
                std::string gate_func = meta_.expert_gating_func;

                for (char& c : arch)       c = (char)std::tolower((unsigned char)c);
                for (char& c : model_name) c = (char)std::tolower((unsigned char)c);
                for (char& c : gate_func)  c = (char)std::tolower((unsigned char)c);

                lw.moe_router_kind = 0;
                lw.moe_router_jitter_eps = 0.0f;

                if (gate_func.find("sigmoid") != std::string::npos) {
                    lw.moe_router_kind = 2;
                }

                lw.moe_normalize_topk = meta_.expert_weights_norm;
                lw.moe_route_scale = meta_.expert_weights_scale;

                if (lw.moe_route_scale == 0.0f) {
                    lw.moe_route_scale = 1.0f;
                }

// Qwen MoE family (Qwen2-MoE, Qwen3-MoE, Qwen3.5/3.6-A3B) all
                // use norm_topk_prob = True in their HF reference implementation.
                // Many GGUF converters do not populate expert_weights_norm metadata,
                // so leaving moe_normalize_topk = false under-scales every routed
                // FFN output by ~1/sum(top_k_probs) ~= 5x. Compounded across all
                // layers this produces semi-coherent jibberish, not coherent text.
                //
                // Since this branch only executes when lw.moe == true, any Qwen
                // model that hits this code path is by definition a Qwen MoE,
                // so forcing the flag is safe and correct.
                //
                // The original "qwen35" substring detection is preserved as a
                // belt-and-suspenders fallback for unusual model_name strings
                // where arch isn't set to a qwen* value.
                {
                    const bool is_qwen_moe_arch =
                        arch.find("qwen") != std::string::npos;

                    const bool is_qwen_moe_name =
                        model_name.find("qwen3.5") != std::string::npos ||
                        model_name.find("qwen3.6") != std::string::npos ||
                        model_name.find("qwen3-30b") != std::string::npos ||
                        model_name.find("qwen3-235b") != std::string::npos ||
                        model_name.find("qwen3-moe") != std::string::npos;

                    if (is_qwen_moe_arch || is_qwen_moe_name) {
                        lw.moe_normalize_topk = true;

                        // Diagnostic on first owned MoE layer only (l == active_start
                        // is the first layer we own; checking l_first prevents the
                        // log from firing 60 times).
                        static std::atomic<bool> qwen_moe_logged{false};
                        bool expected = false;
                        if (qwen_moe_logged.compare_exchange_strong(expected, true)) {
                            fprintf(stderr,
                                "[nexus] Qwen MoE family detected (arch=%s name=%s) -> "
                                "forcing moe_normalize_topk=true (HF norm_topk_prob default)\n",
                                meta_.arch.c_str(),
                                meta_.name.c_str());
                            fflush(stderr);
                        }
                    }
                }

                fprintf(stderr,
                    "[nexus] Layer %d: MoE router=%s top_k=%u norm_topk=%s scale=%.4f gate_func=%s\n",
                    l,
                    lw.moe_router_kind == 2 ? "sigmoid-topk" :
                    lw.moe_router_kind == 1 ? "sparse-top2-mixer" :
                                              "softmax-topk",
                    lw.moe_top_k,
                    lw.moe_normalize_topk ? "yes" : "no",
                    lw.moe_route_scale,
                    gate_func.empty() ? "none" : gate_func.c_str());
            }

            TensorDesc* sg = get("ffn_gate_shexp.weight");
            if (!sg) sg = get("shared_expert_gate.weight");
            if (!sg) sg = get("shared_expert.gate_proj.weight");
            if (!sg) sg = get("mlp.shared_expert.gate_proj.weight");
            if (!sg) sg = get("block_sparse_moe.shared_expert.gate_proj.weight");

            TensorDesc* su = get("ffn_up_shexp.weight");
            if (!su) su = get("shared_expert_up.weight");
            if (!su) su = get("shared_expert.up_proj.weight");
            if (!su) su = get("mlp.shared_expert.up_proj.weight");
            if (!su) su = get("block_sparse_moe.shared_expert.up_proj.weight");

            TensorDesc* sd = get("ffn_down_shexp.weight");
            if (!sd) sd = get("shared_expert_down.weight");
            if (!sd) sd = get("shared_expert.down_proj.weight");
            if (!sd) sd = get("mlp.shared_expert.down_proj.weight");
            if (!sd) sd = get("block_sparse_moe.shared_expert.down_proj.weight");

            TensorDesc* sg_inp = get("ffn_gate_inp_shexp.weight");
            if (!sg_inp) sg_inp = get("shared_expert_gate_inp.weight");
            if (!sg_inp) sg_inp = get("shared_expert_gate.weight");
            if (!sg_inp) sg_inp = get("mlp.shared_expert_gate.weight");
            if (!sg_inp) sg_inp = get("block_sparse_moe.shared_expert_gate.weight");

            if (sg || su || sd) {
                if (!(sg && su && sd)) {
                    fprintf(stderr,
                        "[nexus] FATAL: layer %d has incomplete shared expert tensors "
                        "(gate=%p up=%p down=%p)\n",
                        l,
                        sg ? sg->data : nullptr,
                        su ? su->data : nullptr,
                        sd ? sd->data : nullptr);
                    dump_layer_tensors("incomplete shared expert");
                    return false;
                }

                lw.moe_has_shared = true;

                lw.wgate_shexp = sg->data;
                lw.wgate_shexp_type = ggml_to_qtype(sg->type);

                lw.wup_shexp = su->data;
                lw.wup_shexp_type = ggml_to_qtype(su->type);

                lw.wdown_shexp = sd->data;
                lw.wdown_shexp_type = ggml_to_qtype(sd->type);

                if (sg_inp) {
                    lw.wgate_inp_shexp = tensor_as_f32(sg_inp, "shared_expert_gate");
                }

                const uint32_t expected_shared =
                    meta_.expert_shared_ffn_dim ? meta_.expert_shared_ffn_dim : 0;

                if (!infer_expert_linear(sg,
                                         meta_.hidden_dim,
                                         expected_shared,
                                         lw.moe_shared_gate_rows,
                                         lw.moe_shared_gate_cols,
                                         lw.moe_shared_gate_transposed,
                                         "shared_gate")) {
                    dump_layer_tensors("bad shared expert gate shape");
                    return false;
                }

                if (!infer_expert_linear(su,
                                         meta_.hidden_dim,
                                         lw.moe_shared_gate_rows,
                                         lw.moe_shared_up_rows,
                                         lw.moe_shared_up_cols,
                                         lw.moe_shared_up_transposed,
                                         "shared_up")) {
                    dump_layer_tensors("bad shared expert up shape");
                    return false;
                }

                if (!infer_expert_linear(sd,
                                         lw.moe_shared_gate_rows,
                                         meta_.hidden_dim,
                                         lw.moe_shared_down_rows,
                                         lw.moe_shared_down_cols,
                                         lw.moe_shared_down_transposed,
                                         "shared_down")) {
                    dump_layer_tensors("bad shared expert down shape");
                    return false;
                }

                if (lw.moe_shared_gate_rows != lw.moe_shared_up_rows ||
                    lw.moe_shared_gate_cols != meta_.hidden_dim ||
                    lw.moe_shared_up_cols   != meta_.hidden_dim ||
                    lw.moe_shared_down_rows != meta_.hidden_dim ||
                    lw.moe_shared_down_cols != lw.moe_shared_gate_rows) {
                    fprintf(stderr,
                        "[nexus] FATAL: layer %d bad shared expert dims "
                        "gate=%ux%u%s up=%ux%u%s down=%ux%u%s hidden=%u\n",
                        l,
                        lw.moe_shared_gate_rows,
                        lw.moe_shared_gate_cols,
                        lw.moe_shared_gate_transposed ? " T" : " N",
                        lw.moe_shared_up_rows,
                        lw.moe_shared_up_cols,
                        lw.moe_shared_up_transposed ? " T" : " N",
                        lw.moe_shared_down_rows,
                        lw.moe_shared_down_cols,
                        lw.moe_shared_down_transposed ? " T" : " N",
                        meta_.hidden_dim);
                    dump_layer_tensors("bad shared expert dimensions");
                    return false;
                }

                lw.moe_shared_ffn_dim = lw.moe_shared_gate_rows;

                fprintf(stderr,
                    "[nexus] Layer %d: shared expert indexed gate=%ux%u%s up=%ux%u%s down=%ux%u%s gate_inp=%s\n",
                    l,
                    lw.moe_shared_gate_rows,
                    lw.moe_shared_gate_cols,
                    lw.moe_shared_gate_transposed ? " T" : " N",
                    lw.moe_shared_up_rows,
                    lw.moe_shared_up_cols,
                    lw.moe_shared_up_transposed ? " T" : " N",
                    lw.moe_shared_down_rows,
                    lw.moe_shared_down_cols,
                    lw.moe_shared_down_transposed ? " T" : " N",
                    lw.wgate_inp_shexp ? "yes" : "no");
            }

            fprintf(stderr,
                "[nexus] Layer %d: MoE FFN indexed: layout=%s experts=%u top_k=%u ffn=%u "
                "shared=%s shared_ffn=%u router=%s bias=%s route_kind=%u norm_topk=%s scale=%.4f\n",
                l,
                lw.moe_split_experts ? "split" : "merged",
                lw.moe_expert_count,
                lw.moe_top_k,
                lw.moe_ffn_dim,
                lw.moe_has_shared ? "yes" : "no",
                lw.moe_shared_ffn_dim,
                lw.moe_router ? "yes" : "NO",
                lw.moe_router_bias ? "yes" : "no",
                lw.moe_router_kind,
                lw.moe_normalize_topk ? "yes" : "no",
                lw.moe_route_scale);

            const bool has_split =
                lw.moe_split_experts &&
                lw.wgate_exp_ptrs.size() == lw.moe_expert_count &&
                lw.wup_exp_ptrs.size()   == lw.moe_expert_count &&
                lw.wdown_exp_ptrs.size() == lw.moe_expert_count;

            const bool has_merged =
                lw.wdown_exps &&
                (
                    (!lw.moe_fused_gate_up && lw.wgate_exps && lw.wup_exps) ||
                    ( lw.moe_fused_gate_up && lw.wup_exps)
                );

            if (!lw.moe_router ||
                (!has_split && !has_merged) ||
                lw.moe_expert_count == 0 ||
                lw.moe_top_k == 0 ||
                lw.moe_ffn_dim == 0) {
                fprintf(stderr,
                    "[nexus] FATAL: layer %d has incomplete MoE tensors "
                    "(router=%p split=%d merged=%d experts=%u top_k=%u ffn=%u)\n",
                    l,
                    lw.moe_router,
                    has_split ? 1 : 0,
                    has_merged ? 1 : 0,
                    lw.moe_expert_count,
                    lw.moe_top_k,
                    lw.moe_ffn_dim);
                return false;
            }
        }

        if (!lw.moe && !lw.wgate && lw.wup) {
            // Phi-style fused gate/up. This is only valid if the tensor really is
            // [2*ffn, hidden]. forward_ffn still guards before dereferencing.
            lw.fused_gate_up = true;
        }

        TensorDesc* bq = get("attn_q.bias");
        TensorDesc* bk = get("attn_k.bias");
        TensorDesc* bv = get("attn_v.bias");
        TensorDesc* bo = get("attn_output.bias");

        if (bq) lw.bq = tensor_as_f32(bq, "attn_q.bias");
        if (bk) lw.bk = tensor_as_f32(bk, "attn_k.bias");
        if (bv) lw.bv = tensor_as_f32(bv, "attn_v.bias");
        if (bo) lw.bo = tensor_as_f32(bo, "attn_output.bias");
        
        TensorDesc* qn = get("attn_q_norm.weight");
        TensorDesc* kn = get("attn_k_norm.weight");
        if (!qn) qn = get("q_norm.weight");
        if (!kn) kn = get("k_norm.weight");

        if (qn) {
            lw.attn_q_norm = tensor_as_f32(qn, "attn_q_norm.weight");
            lw.attn_q_norm_len = (size_t)qn->n_elements;
        }
        
        if (kn) {
            lw.attn_k_norm = tensor_as_f32(kn, "attn_k_norm.weight");
            lw.attn_k_norm_len = (size_t)kn->n_elements;
        }

        // Gemma4 mixed attention width:
        // infer this layer's Q/KV/head dims from actual tensors, not global meta.
        {
            std::string arch_l = meta_.arch;
            for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

            if (arch_l.find("gemma4") != std::string::npos) {
                TensorDesc* qd = get("attn_q.weight");
                TensorDesc* kd = get("attn_k.weight");

                auto proj_rows = [&](TensorDesc* t) -> int {
                    if (!t || t->shape.size() < 2) return 0;

                    const int a = (int)t->shape[0];
                    const int b = (int)t->shape[1];
                    const int hidden = (int)meta_.hidden_dim;

                    // GGUF in this code path has been treated as [hidden, rows].
                    if (a == hidden && b > 0) return b;

                    // Safety for converters that store [rows, hidden].
                    if (b == hidden && a > 0) return a;

                    // Last-resort fallback: use the larger side as projection rows.
                    return std::max(a, b);
                };

                const int q_rows = proj_rows(qd);
                const int k_rows = proj_rows(kd);

                int hd = 0;

                if (lw.attn_q_norm_len > 0 &&
                    lw.attn_q_norm_len == lw.attn_k_norm_len &&
                    q_rows > 0 &&
                    k_rows > 0 &&
                    q_rows % (int)lw.attn_q_norm_len == 0 &&
                    k_rows % (int)lw.attn_k_norm_len == 0)
                {
                    hd = (int)lw.attn_q_norm_len;
                }
                else if (lw.attn_k_norm_len > 0 &&
                         q_rows > 0 &&
                         k_rows > 0 &&
                         q_rows % (int)lw.attn_k_norm_len == 0 &&
                         k_rows % (int)lw.attn_k_norm_len == 0)
                {
                    hd = (int)lw.attn_k_norm_len;
                }
                else if ((int)meta_.head_dim > 0 &&
                         q_rows > 0 &&
                         k_rows > 0 &&
                         q_rows % (int)meta_.head_dim == 0 &&
                         k_rows % (int)meta_.head_dim == 0)
                {
                    hd = (int)meta_.head_dim;
                }

                if (hd > 0 && q_rows > 0 && k_rows > 0) {
                    lw.attn_q_dim_layer    = q_rows;
                    lw.attn_kv_dim_layer   = k_rows;
                    lw.attn_head_dim_layer = hd;
                    lw.attn_heads_layer    = q_rows / hd;
                    lw.attn_kv_heads_layer = k_rows / hd;

                    fprintf(stderr,
                        "[nexus] Layer %d Gemma4 dims: q_dim=%d kv_dim=%d heads=%d kv_heads=%d head_dim=%d qnorm=%zu knorm=%zu\n",
                        l,
                        lw.attn_q_dim_layer,
                        lw.attn_kv_dim_layer,
                        lw.attn_heads_layer,
                        lw.attn_kv_heads_layer,
                        lw.attn_head_dim_layer,
                        lw.attn_q_norm_len,
                        lw.attn_k_norm_len);
                    fflush(stderr);
                }
            }
        }

        TensorDesc* ag = get("attn_gate.weight");
        if (!ag) ag = get("mixer.gate.weight");
        if (ag) {
            lw.attn_gate      = ag->data;
            lw.attn_gate_type = ggml_to_qtype(ag->type);
        }

        if (lw.attn_q_norm || lw.attn_k_norm || lw.attn_gate) {
            fprintf(stderr,
                "[nexus] Layer %d: attention extras q_norm=%s len=%zu k_norm=%s len=%zu gate=%s\n",
                l,
                lw.attn_q_norm ? "yes" : "no",
                lw.attn_q_norm_len,
                lw.attn_k_norm ? "yes" : "no",
                lw.attn_k_norm_len,
                lw.attn_gate   ? "yes" : "no");
        }

        // ─── Per-layer trace: every flag that changes forward-pass routing ──
        // Logged once per layer at load time. Search the trace output for
        // "L=N" to see all routing decisions for a specific layer.
        NX_TRACE_DECISION(l, "has_attn_norm",      "%d",   lw.attn_norm      ? 1 : 0);
        NX_TRACE_DECISION(l, "has_attn_norm_bias", "%d",   lw.attn_norm_bias ? 1 : 0);
        NX_TRACE_DECISION(l, "has_ffn_norm",       "%d",   lw.ffn_norm       ? 1 : 0);
        NX_TRACE_DECISION(l, "has_ffn_norm_bias",  "%d",   lw.ffn_norm_bias  ? 1 : 0);
        NX_TRACE_DECISION(l, "has_attn_post_norm", "%d",   lw.attn_post_norm ? 1 : 0);
        NX_TRACE_DECISION(l, "has_ffn_post_norm",  "%d",   lw.ffn_post_norm  ? 1 : 0);
        NX_TRACE_DECISION(l, "has_q_norm",         "%d len=%zu",  lw.attn_q_norm ? 1 : 0, lw.attn_q_norm_len);
        NX_TRACE_DECISION(l, "has_k_norm",         "%d len=%zu",  lw.attn_k_norm ? 1 : 0, lw.attn_k_norm_len);
        NX_TRACE_DECISION(l, "has_attn_gate",      "%d",   lw.attn_gate      ? 1 : 0);
        NX_TRACE_DECISION(l, "has_q_bias",         "%d",   lw.bq             ? 1 : 0);
        NX_TRACE_DECISION(l, "has_k_bias",         "%d",   lw.bk             ? 1 : 0);
        NX_TRACE_DECISION(l, "has_v_bias",         "%d",   lw.bv             ? 1 : 0);
        NX_TRACE_DECISION(l, "has_o_bias",         "%d",   lw.bo             ? 1 : 0);
        NX_TRACE_DECISION(l, "gemma4_tied_kv",     "%d",   lw.gemma4_tied_kv ? 1 : 0);
        NX_TRACE_DECISION(l, "sw_window_size",     "%d",   lw.sw_window_size);
        NX_TRACE_DECISION(l, "rope_freq_base_layer","%.0f",lw.rope_freq_base_layer);
        NX_TRACE_DECISION(l, "ple_layer_index",    "%d",   lw.ple_layer_index);
        NX_TRACE_DECISION(l, "attn_q_dim_layer",   "%d",   lw.attn_q_dim_layer);
        NX_TRACE_DECISION(l, "attn_kv_dim_layer",  "%d",   lw.attn_kv_dim_layer);
        NX_TRACE_DECISION(l, "attn_head_dim_layer","%d",   lw.attn_head_dim_layer);
        NX_TRACE_DECISION(l, "attn_heads_layer",   "%d",   lw.attn_heads_layer);
        NX_TRACE_DECISION(l, "attn_kv_heads_layer","%d",   lw.attn_kv_heads_layer);
        NX_TRACE_DECISION(l, "fused_gate_up",      "%d",   lw.fused_gate_up  ? 1 : 0);
        NX_TRACE_DECISION(l, "moe",                "%d",   lw.moe            ? 1 : 0);
        NX_TRACE_DECISION(l, "moe_fused_gate_up",  "%d",   lw.moe_fused_gate_up ? 1 : 0);
        NX_TRACE_DECISION(l, "moe_has_shared",     "%d",   lw.moe_has_shared ? 1 : 0);
        NX_TRACE_DECISION(l, "qwen35_linear",      "%d",   lw.qwen35_linear  ? 1 : 0);
        NX_TRACE_DECISION(l, "qwen35_q_has_gate",  "%d",   lw.qwen35_full_attn_q_has_gate ? 1 : 0);

        // Norm-weight statistics + PER-TENSOR convention decision.
        //
        // Different converters produce different conventions on different
        // tensors of the same model. Probing each tensor independently is
        // the only correct way to handle this — a single global flag will
        // always be wrong on at least one tensor for some models.
        //
        // Convention rule (universal, not arch-specific):
        //   |mean(w)| < 0.5     → raw HF, needs (1+w) → zc=true
        //   |mean(w)| >= 0.5    → pre-baked or scaled → use w directly → zc=false
        //
        // ─── Norm-weight statistics collection (no decision yet) ─────────
        //
        // Auto-detection of RMSNorm convention used to fire here per-tensor,
        // which produced absurd results: in this Gemma 4 GGUF, attn_norm at
        // L0 has mean=4.84 but at L2 has mean=27.86 — same tensor type, same
        // model, drastically different stats. Per-tensor decision would flip
        // convention across layers, which makes no sense (the convention is
        // a model-wide property of the GGUF converter, not a per-layer one).
        //
        // New approach: collect stats here, then decide ONCE at the end of
        // build_layer_weights using the model-wide max |mean|. Applies the
        // same flag to every per-layer norm tensor.
        std::string arch_l_tensor_norm = meta_.arch;
        for (char& c : arch_l_tensor_norm) {
            c = (char)std::tolower((unsigned char)c);
        }

        const bool gemma4_tensor_norm_auto =
            arch_l_tensor_norm.find("gemma4") != std::string::npos;

        auto probe_norm_stats = [&](const float* w,
                                    size_t len,
                                    const char* name,
                                    NormClassStats* cls,
                                    bool* out_tensor_zc) {
            if (!w || len == 0) return;

            double s = 0.0;
            double sa = 0.0;
            const int N = (int)std::min<size_t>(256, len);

            for (int i = 0; i < N; ++i) {
                s  += (double)w[i];
                sa += std::fabs((double)w[i]);
            }

            const double mean     = s  / (double)N;
            const double mean_abs = sa / (double)N;

            if (cls) {
                cls->sum_mean_abs += mean_abs;
                cls->n_samples    += 1;
            }

            if (mean_abs > norm_stats_max_abs_) {
                norm_stats_max_abs_ = mean_abs;
            }

            // Gemma4 needs this per actual tensor, not per class average.
            // Raw HF-ish norm weights sit near zero and need x*(1+w).
            // Pre-baked/scaled weights sit away from zero and need x*w.
            const bool tensor_zc =
                (std::fabs(mean) < 0.50 && mean_abs < 0.50);

            if (out_tensor_zc) {
                *out_tensor_zc = tensor_zc;
            }

            NX_TRACE_DECISION(l, name,
                "mean=%.4f mean_abs=%.4f n=%d tensor_auto=%s",
                mean,
                mean_abs,
                N,
                tensor_zc ? "zero_centered" : "direct");
        };

        probe_norm_stats(static_cast<const float*>(lw.attn_norm),
                         meta_.hidden_dim,
                         "attn_norm_stats",
                         &nc_attn_norm_,
                         gemma4_tensor_norm_auto ? &lw.attn_norm_zc : nullptr);

        probe_norm_stats(static_cast<const float*>(lw.ffn_norm),
                         meta_.hidden_dim,
                         "ffn_norm_stats",
                         &nc_ffn_norm_,
                         gemma4_tensor_norm_auto ? &lw.ffn_norm_zc : nullptr);

        probe_norm_stats(static_cast<const float*>(lw.attn_post_norm),
                         meta_.hidden_dim,
                         "attn_post_norm_stats",
                         &nc_attn_post_norm_,
                         gemma4_tensor_norm_auto ? &lw.attn_post_norm_zc : nullptr);

        probe_norm_stats(static_cast<const float*>(lw.ffn_post_norm),
                         meta_.hidden_dim,
                         "ffn_post_norm_stats",
                         &nc_ffn_post_norm_,
                         gemma4_tensor_norm_auto ? &lw.ffn_post_norm_zc : nullptr);

        probe_norm_stats(lw.attn_q_norm,
                         lw.attn_q_norm_len,
                         "q_norm_stats",
                         &nc_attn_q_norm_,
                         gemma4_tensor_norm_auto ? &lw.attn_q_norm_zc : nullptr);

        probe_norm_stats(lw.attn_k_norm,
                         lw.attn_k_norm_len,
                         "k_norm_stats",
                         &nc_attn_k_norm_,
                         gemma4_tensor_norm_auto ? &lw.attn_k_norm_zc : nullptr);


        if (!lw.attn_norm) {
            dump_layer_tensors("missing attention/input norm");
            return false;
        }
        
        // ---- Qwen3.5 / Qwen3.6 Gated DeltaNet linear-attention path -----------------
        // If these tensors are present, this layer is NOT normal softmax attention.
        // Do not run it through the regular AttentionLayer path.
        TensorDesc* q35_qkv  = get("attn_qkv.weight");
        TensorDesc* q35_z    = get("attn_gate.weight");
        TensorDesc* q35_a    = get("ssm_alpha.weight");
        TensorDesc* q35_b    = get("ssm_beta.weight");
        TensorDesc* q35_ba   = get("ssm_beta_alpha.weight");
        TensorDesc* q35_conv = get("ssm_conv1d.weight");
        TensorDesc* q35_dt   = get("ssm_dt.bias");
        if (!q35_dt) q35_dt = get("ssm_dt");
        TensorDesc* q35_A    = get("ssm_a");
        TensorDesc* q35_norm = get("ssm_norm.weight");
        TensorDesc* q35_out  = get("ssm_out.weight");

        const bool q35_has_split_ba = q35_a && q35_b;
        const bool q35_has_fused_ba = q35_ba != nullptr;

        if (q35_qkv && q35_z && (q35_has_fused_ba || q35_has_split_ba) && q35_conv && q35_dt && q35_A && q35_norm && q35_out) {
            lw.qwen35_linear = true;

            lw.q35_qkv = q35_qkv->data; lw.q35_qkv_type = ggml_to_qtype(q35_qkv->type);
            lw.q35_z   = q35_z->data;   lw.q35_z_type   = ggml_to_qtype(q35_z->type);
            if (q35_has_fused_ba) {
                lw.q35_ba = q35_ba->data;
                lw.q35_ba_type = ggml_to_qtype(q35_ba->type);
            }
            if (q35_has_split_ba) {
                lw.q35_a   = q35_a->data;   lw.q35_a_type   = ggml_to_qtype(q35_a->type);
                lw.q35_b   = q35_b->data;   lw.q35_b_type   = ggml_to_qtype(q35_b->type);
            }

            lw.q35_conv = tensor_as_f32(q35_conv, "ssm_conv1d.weight");
            lw.q35_dt   = tensor_as_f32(q35_dt,   "ssm_dt");
            lw.q35_A    = tensor_as_f32(q35_A,    "ssm_a");
            lw.q35_norm = tensor_as_f32(q35_norm, "ssm_norm.weight");

            // Native .nxm preserves the HF tensor linear_attn.A_log. The
            // existing Qwen35 runtime expects the already-converted no-scan
            // coefficient used by GGUF/llama.cpp: ssm_a = -exp(A_log). Convert
            // only when the actual tensor name proves this came from A_log.
            const bool q35_A_is_native_A_log =
                q35_A && (q35_A->name.find("linear_attn.A_log") != std::string::npos ||
                          (q35_A->name.size() >= 5 && q35_A->name.rfind("A_log") == q35_A->name.size() - 5));
            if (q35_A_is_native_A_log && lw.q35_A && q35_A->n_elements > 0) {
                f32_tensor_storage_.emplace_back((size_t)q35_A->n_elements);
                float* converted = f32_tensor_storage_.back().data();
                for (size_t ai = 0; ai < (size_t)q35_A->n_elements; ++ai) {
                    converted[ai] = -std::exp(lw.q35_A[ai]);
                }
                lw.q35_A = converted;
                if (l == active_start) {
                    fprintf(stderr, "[nexus] Native Qwen3Next: converted linear_attn.A_log -> ssm_a=-exp(A_log)\n");
                    fflush(stderr);
                }
            }

            lw.q35_out = q35_out->data; lw.q35_out_type = ggml_to_qtype(q35_out->type);

            const bool q35_native_layout =
                q35_qkv && q35_qkv->name.rfind("model.language_model.layers.", 0) == 0;

            if (q35_native_layout) {
                // Native .nxm preserves HF row-major matrix shapes [out, in].
                // linear_attn.in_proj_qkv.weight: [conv_dim, hidden]
                // linear_attn.in_proj_z.weight:   [value_dim, hidden]
                // linear_attn.out_proj.weight:    [hidden, value_dim]
                // linear_attn.conv1d.weight:      [conv_dim, 1, kernel]
                lw.q35_conv_dim  = (q35_qkv->shape.size()  >= 2) ? (int)q35_qkv->shape[0]  : 0;
                lw.q35_ba_dim    = (q35_ba  && q35_ba->shape.size() >= 2) ? (int)q35_ba->shape[0] : 0;
                lw.q35_value_dim = (q35_z   && q35_z->shape.size()   >= 2) ? (int)q35_z->shape[0]
                                   : ((q35_out->shape.size() >= 2) ? (int)q35_out->shape[1] : 0);
                lw.q35_head_dim  = (q35_norm->shape.size() >= 1) ? (int)q35_norm->shape[0] : 128;
                lw.q35_kernel    = (q35_conv->shape.size() >= 3) ? (int)q35_conv->shape[2]
                                   : ((q35_conv->shape.size() >= 1) ? (int)q35_conv->shape[0] : 4);
            } else {
                // GGUF shapes here are [input_cols, output_rows].
                // attn_qkv: [hidden, conv_dim]
                // attn_gate/z: [hidden, value_dim]
                // ssm_out: [value_dim, hidden]
                // ssm_norm: [head_v_dim]
                lw.q35_conv_dim  = (q35_qkv->shape.size()  >= 2) ? (int)q35_qkv->shape[1]  : 0;
                lw.q35_ba_dim    = (q35_ba  && q35_ba->shape.size() >= 2) ? (int)q35_ba->shape[1] : 0;
                lw.q35_value_dim = (q35_out->shape.size()  >= 1) ? (int)q35_out->shape[0]  : 0;
                lw.q35_head_dim  = (q35_norm->shape.size() >= 1) ? (int)q35_norm->shape[0] : 128;
                lw.q35_kernel    = (q35_conv->shape.size() >= 1) ? (int)q35_conv->shape[0] : 4;
            }

            lw.q35_key_dim = (lw.q35_conv_dim - lw.q35_value_dim) / 2;

            if (lw.q35_head_dim <= 0) lw.q35_head_dim = 128;
            lw.q35_num_v_heads = lw.q35_value_dim / lw.q35_head_dim;
            lw.q35_num_k_heads = lw.q35_key_dim   / lw.q35_head_dim;

            fprintf(stderr,
                "[nexus] Layer %d: Qwen35 linear-attn indexed conv_dim=%d key_dim=%d value_dim=%d k_heads=%d v_heads=%d head=%d kernel=%d ba=%s\n",
                l,
                lw.q35_conv_dim,
                lw.q35_key_dim,
                lw.q35_value_dim,
                lw.q35_num_k_heads,
                lw.q35_num_v_heads,
                lw.q35_head_dim,
                lw.q35_kernel,
                q35_has_fused_ba ? "fused:ssm_beta_alpha" : "split:ssm_beta+ssm_alpha");

            if (lw.q35_ba && lw.q35_ba_dim != lw.q35_num_v_heads * 2) {
                fprintf(stderr,
                    "[nexus] WARNING Layer %d: ignoring ssm_beta_alpha.weight with unexpected rows=%d expected=%d; falling back to split alpha/beta if present.\n",
                    l, lw.q35_ba_dim, lw.q35_num_v_heads * 2);
                lw.q35_ba = nullptr;
            }

            if (lw.q35_conv_dim <= 0 || lw.q35_key_dim <= 0 || lw.q35_value_dim <= 0 ||
                lw.q35_num_k_heads <= 0 || lw.q35_num_v_heads <= 0 ||
                lw.q35_num_v_heads % lw.q35_num_k_heads != 0 ||
                (!lw.q35_ba && (!lw.q35_a || !lw.q35_b)))
            {
                dump_layer_tensors("bad Qwen35 linear-attention dimensions or missing alpha/beta projection");
                return false;
            }
            {
                float amin =  1e30f;
                float amax = -1e30f;

                for (int i = 0; i < lw.q35_num_v_heads; ++i) {
                    const float v = lw.q35_A ? lw.q35_A[i] : 0.0f;
                    if (v < amin) amin = v;
                    if (v > amax) amax = v;
                }

                fprintf(stderr,
                    "[nexus] Layer %d: Qwen35 ssm_a/noscan range=[%.6g, %.6g] proj_order=flat-attn_qkv\n",
                    l, amin, amax);
                fflush(stderr);

                if (amax > 0.0f) {
                    fprintf(stderr,
                        "[nexus] WARNING Layer %d: q35 ssm_a has positive values. Expected no-scan negative coefficients. "
                        "Decay clamp will prevent state explosion, but converter mapping may need review.\n",
                        l);
                    fflush(stderr);
                }
            }            
        }
        
        // Qwen3-Next full-attention layers: attn_q.weight may contain
        // [query rows][gate rows].
        //
        // IMPORTANT:
        // Gemma4 also has layers where attn_q.weight is exactly 2x the small
        // layer width, but that is a real Gemma4 wide-Q attention layer, NOT
        // a packed Qwen query+gate tensor. Do not run this detector on Gemma4.
        {
            std::string arch_l = meta_.arch;
            for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

            const bool is_qwen_arch =
                arch_l.find("qwen") != std::string::npos;

            if (is_qwen_arch && !lw.qwen35_linear) {
                // Native .nxm keeps HF names, so the full-attention packed
                // query+gate tensor is usually:
                //   model.language_model.layers.N.self_attn.q_proj.weight
                // not blk.N.attn_q.weight.
                //
                // The previous detector only asked for attn_q.weight, so it
                // missed native .nxm Qwen3.5/3.6 full-attention layers. Forward
                // then projected only q_dim rows from a [2*q_dim, hidden] tensor,
                // throwing away the gate half and leaving Qwen3Next full attention
                // mathematically wrong.
                TensorDesc* q_gate_q = get("attn_q.weight");
                if (!q_gate_q) q_gate_q = get("self_attn.q_proj.weight");
                if (!q_gate_q) q_gate_q = get("q_proj.weight");
                if (!q_gate_q) q_gate_q = get("mixer.q_proj.weight");

                const int q_dim_model = (int)(meta_.n_heads * meta_.head_dim);

                if (q_gate_q && q_gate_q->shape.size() >= 2 && q_dim_model > 0) {
                    const int s0 = (int)q_gate_q->shape[0];
                    const int s1 = (int)q_gate_q->shape[1];

                    int packed_rows = 0;
                    const char* layout = "unknown";

                    if (s0 == q_dim_model * 2) {
                        packed_rows = s0;       // HF/native [out,in]
                        layout = "native[out,in]";
                    } else if (s1 == q_dim_model * 2) {
                        packed_rows = s1;       // GGUF-style [in,out]
                        layout = "gguf[in,out]";
                    }

                    if (packed_rows == q_dim_model * 2) {
                        lw.qwen35_full_attn_q_has_gate = true;
                        lw.qwen35_full_attn_q_rows = (uint32_t)packed_rows;

                        fprintf(stderr,
                            "[nexus] Layer %d: Qwen3Next full-attn q_proj contains query+gate rows=%u shape=[%d,%d] layout=%s; using gated attention split\n",
                            l, lw.qwen35_full_attn_q_rows, s0, s1, layout);
                        fflush(stderr);
                    } else if (s0 == q_dim_model || s1 == q_dim_model) {
                        // Normal Qwen/Qwen2/Qwen3 attention without packed gate.
                    } else {
                        fprintf(stderr,
                            "[nexus] Layer %d: Qwen full-attn q_proj shape=[%d,%d] does not match q_dim=%d or packed q_dim*2=%d; leaving ungated\n",
                            l, s0, s1, q_dim_model, q_dim_model * 2);
                        fflush(stderr);
                    }
                }
            }
        }        

        if (!lw.qwen35_linear) {
            if (lw.fused_qkv && !lw.wo) {
                dump_layer_tensors("fused QKV found but no output projection found");
                return false;
            }

            if (!lw.fused_qkv && (!lw.wq || !lw.wk || !lw.wv || !lw.wo)) {
                dump_layer_tensors(has_hybrid_or_alt
                    ? "missing standard attention weights; hybrid/alternate mixer tensors detected"
                    : "missing standard attention weights");
                return false;
            }
        }

        if (!lw.ffn_norm) {
            dump_layer_tensors("missing FFN/post-attention norm");
            return false;
        }

        if (!lw.moe && !lw.fused_gate_up && (!lw.wgate || !lw.wup || !lw.wdown)) {
            dump_layer_tensors("missing dense FFN gate/up/down weights");
            return false;
        }

        if (!lw.moe && lw.fused_gate_up && (!lw.wup || !lw.wdown)) {
            dump_layer_tensors("fused gate/up detected but missing fused-up or down projection");
            return false;
        }
    }

    fprintf(stderr, "[nexus] Layer index built successfully for active range %d-%d\n",
            active_start, active_end - 1);

    // ─── Gemma 4: re-init KV cache with per-layer geometry ──────────────────
    //
    // Gemma 4 has variable (n_kv_heads, head_dim) per layer (local layers
    // have 16 heads × 256 dim, global layers have 4 heads × 512 dim). The
    // initial kvcache_.init() call ran on uniform `meta_` values, which is
    // wrong for any layer that doesn't match. Detect per-layer dims in
    // layer_weights_ and re-init with the correct prefix-sum slot layout.
    //
    // If no layer has per-layer overrides, this is a no-op.
    {
        bool any_per_layer = false;
        for (const auto& lw : layer_weights_) {
            if (lw.attn_kv_heads_layer > 0 && lw.attn_head_dim_layer > 0) {
                any_per_layer = true;
                break;
            }
        }
        if (any_per_layer) {
            std::vector<int> kv_heads_v((size_t)meta_.n_layers);
            std::vector<int> head_dim_v((size_t)meta_.n_layers);
            for (int L = 0; L < (int)meta_.n_layers; ++L) {
                const auto& lw = layer_weights_[L];
                kv_heads_v[L] = (lw.attn_kv_heads_layer > 0)
                              ? lw.attn_kv_heads_layer
                              : (int)meta_.n_kv_heads;
                head_dim_v[L] = (lw.attn_head_dim_layer > 0)
                              ? lw.attn_head_dim_layer
                              : (int)meta_.head_dim;
            }
            kvcache_.init_per_layer(kv_heads_v, head_dim_v);

            int local_h  = kv_heads_v[0], local_d  = head_dim_v[0];
            int global_h = 0, global_d = 0;
            for (int L = 0; L < (int)meta_.n_layers; ++L) {
                if (kv_heads_v[L] != local_h || head_dim_v[L] != local_d) {
                    global_h = kv_heads_v[L]; global_d = head_dim_v[L]; break;
                }
            }
            fprintf(stderr,
                "[nexus] KV cache re-init with per-layer geometry: "
                "local(kv_heads=%d,head_dim=%d) global(kv_heads=%d,head_dim=%d) "
                "n_layers=%u\n",
                local_h, local_d, global_h, global_d, meta_.n_layers);
            fflush(stderr);
        }
    }

    // ─── Gemma 4: Shared KV Cache donor resolution (second pass) ────────────
    //
    // Per the Gemma 4 architecture: the LAST N layers of the model reuse
    // K/V from earlier layers, where N comes from the GGUF metadata key
    // `gemma4.attention.shared_kv_layers`. The reference implementation
    // (ollama model_text.go) reads this exact key.
    //
    // IMPORTANT: not every layer that's missing Wv is a shared-KV layer.
    // Gemma 4 also has K=V alias layers (Wv missing but Wk present, V is
    // computed as raw K). These are SEPARATE from shared-KV.
    //
    // The classification:
    //   shared_kv_layers = N from GGUF
    //   firstShared = num_layers - N
    //   Layer i >= firstShared AND missing Wv  -> shared-KV (read from donor)
    //   Layer i <  firstShared AND missing Wv  -> K=V alias (compute V = raw K)
    //
    // For Gemma 4 31B: per Unsloth's documentation, shared_kv_layers = 0.
    // Every "Wv missing" layer is K=V alias, no shared-KV at all.
    // For Gemma 4 E2B / E4B: shared_kv_layers > 0, the last N layers ARE
    // shared. We must handle both.
    {
        uint32_t shared_kv_layers = 0;
        if (parser_) {
            shared_kv_layers = parser_->get_u32("gemma4.attention.shared_kv_layers", 0);
            if (shared_kv_layers == 0) {
                shared_kv_layers = parser_->get_u32("gemma4_text.attention.shared_kv_layers", 0);
            }
            if (shared_kv_layers == 0) {
                shared_kv_layers = parser_->get_u32("attention.shared_kv_layers", 0);
            }
        }
        const int firstShared = (int)meta_.n_layers - (int)shared_kv_layers;
        fprintf(stderr,
            "[nexus] Gemma 4 shared_kv_layers metadata = %u  (firstShared = layer %d of %d)\n",
            shared_kv_layers, firstShared, (int)meta_.n_layers);
        fflush(stderr);
        NX_TRACE_MODEL("gemma4.shared_kv_layers", "%u", shared_kv_layers);
        NX_TRACE_MODEL("gemma4.first_shared_layer", "%d", firstShared);

        int n_resolved = 0;
        int n_unresolved = 0;
        int n_kv_alias = 0;
        for (int L = 0; L < (int)meta_.n_layers; ++L) {
            auto& lw = layer_weights_[L];
            if (lw.kv_donor_layer != -2) continue;   // not flagged

            // Decide: shared-KV (reads donor cache) vs K=V alias (computes V=K).
            // Only layers in the [firstShared, n_layers) range are shared.
            const bool is_in_shared_range = (L >= firstShared) && (shared_kv_layers > 0);

            if (!is_in_shared_range) {
                // K=V alias. V is computed from Wk projection.
                lw.kv_donor_layer = -1;
                if (!lw.wv && lw.wk) {
                    lw.wv = lw.wk;
                    lw.wv_type = lw.wk_type;
                    lw.gemma4_tied_kv = true;
                }
                ++n_kv_alias;
                fprintf(stderr,
                    "[nexus] Layer %d: Gemma4 K=V alias (Wv absent, Wk present) "
                    "[L < firstShared=%d, so NOT shared-KV]\n",
                    L, firstShared);
                fflush(stderr);
                continue;
            }

            // True shared-KV. Find donor: nearest earlier layer of same
            // attention type (local/global) that owns its K/V.
            const bool i_am_local = (lw.sw_window_size > 0);
            int donor = -1;
            for (int D = L - 1; D >= 0; --D) {
                const auto& dw = layer_weights_[D];
                if (dw.kv_donor_layer != -1) continue;     // donor is also shared, skip
                if (!dw.wk || !dw.wv) continue;            // donor has no K/V
                const bool d_local = (dw.sw_window_size > 0);
                if (d_local != i_am_local) continue;       // type mismatch
                if (lw.attn_head_dim_layer != dw.attn_head_dim_layer) continue;
                if (lw.attn_kv_heads_layer != dw.attn_kv_heads_layer) continue;
                donor = D;
                break;
            }

            if (donor >= 0) {
                lw.kv_donor_layer = donor;
                ++n_resolved;
                fprintf(stderr,
                    "[nexus] Layer %d: TRUE shared-KV → donor=L%d (%s, head_dim=%d, kv_heads=%d)\n",
                    L, donor, i_am_local ? "LOCAL" : "GLOBAL",
                    lw.attn_head_dim_layer, lw.attn_kv_heads_layer);
            } else {
                // No donor available. Fall back to K=V alias.
                lw.kv_donor_layer = -1;
                if (!lw.wv && lw.wk) {
                    lw.wv = lw.wk;
                    lw.wv_type = lw.wk_type;
                    lw.gemma4_tied_kv = true;
                }
                ++n_unresolved;
                fprintf(stderr,
                    "[nexus] Layer %d: shared-KV with NO donor (%s) — falling back to K=V alias\n",
                    L, i_am_local ? "LOCAL" : "GLOBAL");
            }
            fflush(stderr);
        }
        if (n_resolved + n_unresolved + n_kv_alias > 0) {
            fprintf(stderr,
                "[nexus] Gemma 4 KV summary: %d true-shared, %d unresolved-shared (→alias), %d K=V-alias\n",
                n_resolved, n_unresolved, n_kv_alias);
            fflush(stderr);
        }
    }

    // ─── RMSNorm convention decision: PER CLASS, model-wide per class ───────
    //
    // Each of the six norm-tensor classes (attn_norm, ffn_norm,
    // attn_post_norm, ffn_post_norm, q_norm, k_norm) decides its own
    // convention based on the AVERAGE mean(|w|) across every layer that
    // owns one of those tensors. Each class compared independently against
    // a single threshold (0.5).
    //
    //   avg|mean| < 0.5  → raw HF, needs (1+w) → zc=true
    //   avg|mean| >= 0.5 → pre-baked or scaled → use w directly → zc=false
    //
    // Previously this was a SINGLE global-max decision driven by whichever
    // class had the biggest stats. In Gemma 4 31B, attn_norm averages ~20
    // → that pushed the global decision to "direct" → applied incorrectly
    // to k_norm which has avg ≈ 0.12 (needs zero_centered). Per-class fixes
    // that: each class picks the right thing.
    //
    // Manual override: NEXUS_NORM_CONVENTION=direct|zero_centered forces ALL
    // six classes to the same value (overrides auto). =auto or unset =
    // per-class auto-detect runs.
    {
        // Architecture-level rule first: ALL Gemma models (2, 3, 4) use
        // zero-centered RMSNorm on every norm tensor unconditionally. The
        // reference HF implementation (Gemma2RMSNorm / Gemma3RMSNorm) has
        //   weight = Parameter(torch.zeros(dim))
        //   output = norm(x) * (1.0 + weight)
        // The weight is *initialized to zero* and grows during training; at
        // inference the offset of +1 is always present. The actual stored
        // mean drifting away from zero (e.g. 20 for attn_norm) does NOT
        // change the math — it is still (1+w), not w directly.
        //
        // Statistics-based auto-detect can never see this from the weights
        // alone, because the architecture decision is independent of the
        // weight magnitudes. We use stats only for non-Gemma archs.
        std::string arch_l_norm_decide = meta_.arch;
        for (char& c : arch_l_norm_decide) c = (char)std::tolower((unsigned char)c);

        const bool is_any_gemma =
            arch_l_norm_decide.find("gemma") != std::string::npos;

        const bool is_qwen35_family =
            arch_l_norm_decide.find("qwen35")     != std::string::npos ||
            arch_l_norm_decide.find("qwen3_5")    != std::string::npos ||
            arch_l_norm_decide.find("qwen3.5")    != std::string::npos ||
            arch_l_norm_decide.find("qwen3_6")    != std::string::npos ||
            arch_l_norm_decide.find("qwen3.6")    != std::string::npos ||
            arch_l_norm_decide.find("qwen3next")  != std::string::npos ||
            arch_l_norm_decide.find("qwen3_next") != std::string::npos;

        bool native_raw_hf_tensor_layout_for_norm_classes = false;
        for (const auto& kv_tensor_native_probe : tensors) {
            const std::string& tn = kv_tensor_native_probe.first;
            const GGMLType tt = kv_tensor_native_probe.second.type;
            if (tn.rfind("model.language_model.", 0) == 0 ||
                tn.rfind("language_model.", 0) == 0 ||
                tt == GGMLType::NQ4_G64 ||
                tt == GGMLType::NQ5_G64 ||
                tt == GGMLType::NQ6_G64 ||
                tt == GGMLType::NQ8_G64) {
                native_raw_hf_tensor_layout_for_norm_classes = true;
                break;
            }
        }

        const bool qwen3next_native_raw_norm_classes =
            is_qwen35_family && native_raw_hf_tensor_layout_for_norm_classes;

        const bool arch_forces_zero_centered =
            is_any_gemma || qwen3next_native_raw_norm_classes;

        auto decide_stats = [](const NormClassStats& s) -> bool {
            if (s.n_samples == 0) return false;
            double avg = s.sum_mean_abs / s.n_samples;
            return avg < 0.9;
        };

        if (arch_forces_zero_centered) {
            nc_attn_norm_.decided_zc      = true;
            nc_ffn_norm_.decided_zc       = true;
            nc_attn_post_norm_.decided_zc = true;
            nc_ffn_post_norm_.decided_zc  = true;
            nc_attn_q_norm_.decided_zc    = true;
            nc_attn_k_norm_.decided_zc    = true;
        } else {
            // Only the Gemma family and Qwen3-Next/3.5/3.6 use (1+w) -- handled above.  Every other
            // architecture (Llama, Mistral, Qwen2, Qwen3, Phi, ...) is plain x*w, both in llama.cpp
            // GGUFs (which bake any offset in) and in raw HF weights.  Weight statistics must NOT
            // decide this: Llama 3.1's attn_norm averages 0.42 and was misread as zero-centered.
            // The stats are still computed and logged below for diagnosis.
            (void)decide_stats;
            nc_attn_norm_.decided_zc      = false;
            nc_ffn_norm_.decided_zc       = false;
            nc_attn_post_norm_.decided_zc = false;
            nc_ffn_post_norm_.decided_zc  = false;
            nc_attn_q_norm_.decided_zc    = false;
            nc_attn_k_norm_.decided_zc    = false;
        }

        const char* source = arch_forces_zero_centered
            ? (is_any_gemma
                ? "auto (gemma arch: all decoder RMSNorm zero_centered)"
                : "auto (qwen35/qwen3.5/qwen3.6 arch: decoder RMSNorm zero_centered)")
            : "auto (architecture rule: direct x*w; stats shown for reference only)";

        bool force = false, force_val = false;
        if (const char* env = std::getenv("NEXUS_NORM_CONVENTION")) {
            std::string e = env;
            if (e == "zero_centered" || e == "1") {
                force = true; force_val = true;
                source = "env NEXUS_NORM_CONVENTION=zero_centered (forced)";
            } else if (e == "direct" || e == "0") {
                force = true; force_val = false;
                source = "env NEXUS_NORM_CONVENTION=direct (forced)";
            } else if (e == "auto") {
                source = arch_forces_zero_centered
                    ? (is_any_gemma
                        ? "env NEXUS_NORM_CONVENTION=auto (gemma all decoder RMSNorm zero_centered)"
                        : "env NEXUS_NORM_CONVENTION=auto (qwen35/qwen3.5/qwen3.6 decoder RMSNorm zero_centered)")
                    : "env NEXUS_NORM_CONVENTION=auto (per-class)";
            }
        }

        if (force) {
            nc_attn_norm_.decided_zc      = force_val;
            nc_ffn_norm_.decided_zc       = force_val;
            nc_attn_post_norm_.decided_zc = force_val;
            nc_ffn_post_norm_.decided_zc  = force_val;
            nc_attn_q_norm_.decided_zc    = force_val;
            nc_attn_k_norm_.decided_zc    = force_val;
        }

        // ALSO force the global zero_centered_norm_ flag for architectures
        // whose final output_norm uses raw HF zero-centered RMSNorm semantics.
        // For native Qwen3Next .nxm this is critical: the model.norm/output_norm
        // tensor was NOT GGUF-baked, so final norm must be x*(1+w). The per-head
        // GatedDeltaNet norm stays direct in forward_qwen35_linear_attention().
        if (arch_forces_zero_centered && !force) {
            zero_centered_norm_ = true;
        } else if (force) {
            zero_centered_norm_ = force_val;
        } else {
            // Same architecture rule as the per-layer norms above: everything that isn't Gemma or
            // native-HF Qwen3-Next/3.5/3.6 is plain x*w. Without this the final output_norm kept the
            // weight-statistics guess made earlier in build_layer_weights() -- exactly the guess the
            // comment above says misreads small-magnitude norm weights -- so a model whose output_norm
            // happened to average small would get (1+w) on its very last norm and produce garbage.
            if (zero_centered_norm_)
                fprintf(stderr, "[nexus] Norm: final output_norm stats suggested (1+w), but arch=%s uses x*w -> direct\n",
                        meta_.arch.c_str());
            zero_centered_norm_ = false;
        }

        // Apply per-class decisions to every layer of every architecture.
        for (auto& lw : layer_weights_) {
            lw.attn_norm_zc      = nc_attn_norm_.decided_zc;
            lw.ffn_norm_zc       = nc_ffn_norm_.decided_zc;
            lw.attn_post_norm_zc = nc_attn_post_norm_.decided_zc;
            lw.ffn_post_norm_zc  = nc_ffn_post_norm_.decided_zc;
            lw.attn_q_norm_zc    = nc_attn_q_norm_.decided_zc;
            lw.attn_k_norm_zc    = nc_attn_k_norm_.decided_zc;
        }

        auto avg = [](const NormClassStats& s) -> double {
            return s.n_samples ? (s.sum_mean_abs / s.n_samples) : 0.0;
        };
        const char* dz = "zero_centered (x*(1+w))";
        const char* dr = "direct (x*w)";
        fprintf(stderr,
            "[nexus] Norm convention: %s\n"
            "[nexus]   attn_norm:      avg|mean|=%.4f  -> %s\n"
            "[nexus]   ffn_norm:       avg|mean|=%.4f  -> %s\n"
            "[nexus]   attn_post_norm: avg|mean|=%.4f  -> %s\n"
            "[nexus]   ffn_post_norm:  avg|mean|=%.4f  -> %s\n"
            "[nexus]   q_norm:         avg|mean|=%.4f  -> %s\n"
            "[nexus]   k_norm:         avg|mean|=%.4f  -> %s\n",
            source,
            avg(nc_attn_norm_),      nc_attn_norm_.decided_zc      ? dz : dr,
            avg(nc_ffn_norm_),       nc_ffn_norm_.decided_zc       ? dz : dr,
            avg(nc_attn_post_norm_), nc_attn_post_norm_.decided_zc ? dz : dr,
            avg(nc_ffn_post_norm_),  nc_ffn_post_norm_.decided_zc  ? dz : dr,
            avg(nc_attn_q_norm_),    nc_attn_q_norm_.decided_zc    ? dz : dr,
            avg(nc_attn_k_norm_),    nc_attn_k_norm_.decided_zc    ? dz : dr);
        fflush(stderr);

        NX_TRACE_MODEL("norm_convention_source", "%s", source);
        NX_TRACE_MODEL("norm_conv.attn_norm",      "avg=%.4f -> %s",
                       avg(nc_attn_norm_),      nc_attn_norm_.decided_zc      ? "zc" : "direct");
        NX_TRACE_MODEL("norm_conv.ffn_norm",       "avg=%.4f -> %s",
                       avg(nc_ffn_norm_),       nc_ffn_norm_.decided_zc       ? "zc" : "direct");
        NX_TRACE_MODEL("norm_conv.attn_post_norm", "avg=%.4f -> %s",
                       avg(nc_attn_post_norm_), nc_attn_post_norm_.decided_zc ? "zc" : "direct");
        NX_TRACE_MODEL("norm_conv.ffn_post_norm",  "avg=%.4f -> %s",
                       avg(nc_ffn_post_norm_),  nc_ffn_post_norm_.decided_zc  ? "zc" : "direct");
        NX_TRACE_MODEL("norm_conv.q_norm",         "avg=%.4f -> %s",
                       avg(nc_attn_q_norm_),    nc_attn_q_norm_.decided_zc    ? "zc" : "direct");
        NX_TRACE_MODEL("norm_conv.k_norm",         "avg=%.4f -> %s",
                       avg(nc_attn_k_norm_),    nc_attn_k_norm_.decided_zc    ? "zc" : "direct");
    }

    return true;
}

// --------- Load tokenizer ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

inline bool Transformer::load_tokenizer() {
    const auto& kv = parser_->kv();

    // Get token list from GGUF metadata
    auto it = kv.find("tokenizer.ggml.tokens");
    if (it == kv.end()) return false;

    const auto* arr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&it->second);
    if (!arr || !*arr) return false;

    vocab_tokens_.reserve((*arr)->values.size());
    for (const auto& v : (*arr)->values) {
        if (const auto* s = std::get_if<std::string>(&v)) {
            vocab_tokens_.push_back(*s);
        } else {
            vocab_tokens_.push_back("");
        }
    }

    // Scores (for BPE merge priority)
    auto sit = kv.find("tokenizer.ggml.scores");
    if (sit != kv.end()) {
        if (const auto* sarr = std::get_if<std::shared_ptr<gguf::GGUFArray>>(&sit->second)) {
            if (*sarr) {
                vocab_scores_.reserve((*sarr)->values.size());
                for (const auto& v : (*sarr)->values) {
                    if (const auto* f = std::get_if<float>(&v)) vocab_scores_.push_back(*f);
                    else vocab_scores_.push_back(0.0f);
                }
            }
        }
    }

    auto nx_find_tok = [&](const std::string& needle)->int {
        for (size_t i = 0; i < vocab_tokens_.size(); ++i) {
            if (vocab_tokens_[i] == needle) return (int)i;
        }
        return -1;
    };
    const int nx_im_start = nx_find_tok("<|im_start|>");
    const int nx_im_end   = nx_find_tok("<|im_end|>");
    const int nx_think    = nx_find_tok("<think>");
    const int nx_nothink  = nx_find_tok("</think>");

    // New model starts in native RoPE space. Stabilizer settings remain whatever
    // the launcher/server requested, but the active scale must reset per load.
    rope_pos_scale_ = 1.0f;

    fprintf(stderr, "[nexus] Tokenizer: %zu tokens loaded\n", vocab_tokens_.size());
    if (nx_im_start >= 0 || nx_im_end >= 0 || nx_think >= 0 || nx_nothink >= 0) {
        fprintf(stderr,
                "[nexus] Tokenizer specials: im_start=%d im_end=%d think=%d /think=%d\n",
                nx_im_start, nx_im_end, nx_think, nx_nothink);
    }
    fflush(stderr);
    return true;
}


// --------- RoPE position stabilizer -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

inline void Transformer::set_rope_stabilizer(bool enabled,
                                             int trigger_effective_tokens,
                                             int target_effective_tokens) {
    rope_stabilizer_enabled_ = enabled;
    rope_stabilizer_trigger_ = std::max(1024, trigger_effective_tokens);
    rope_stabilizer_target_  = std::max(1024, target_effective_tokens);
    if (rope_stabilizer_target_ > rope_stabilizer_trigger_)
        rope_stabilizer_target_ = rope_stabilizer_trigger_;
    fprintf(stderr,
        "[rope-stab] %s trigger_eff=%d target_eff=%d current_scale=%.6f\n",
        rope_stabilizer_enabled_ ? "ENABLED" : "disabled",
        rope_stabilizer_trigger_, rope_stabilizer_target_, rope_pos_scale_);
    fflush(stderr);
}

inline void Transformer::set_rope_position_scale_state(float scale) {
    if (!(scale > 0.0f)) scale = 1.0f;
    if (scale > 1.0f) scale = 1.0f;
    if (scale < 0.05f) scale = 0.05f;
    rope_pos_scale_ = scale;
    fprintf(stderr, "[rope-stab] restored active scale=%.6f\n", rope_pos_scale_);
    fflush(stderr);
}

inline int Transformer::rope_effective_position(int actual_pos) const {
    if (actual_pos <= 0) return 0;
    double v = (double)actual_pos * (double)rope_pos_scale_;
    if (v < 0.0) v = 0.0;
    if (v > (double)INT_MAX) v = (double)INT_MAX;
    return (int)std::llround(v);
}

inline void Transformer::rope_rotate_cached_k_delta(int layer_idx, int pos, int delta_pos) {
    if (delta_pos == 0 || pos < 0 || pos >= kvcache_.n_tokens()) return;
    if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size()) return;

    const TransformerLayerWeights& lw = layer_weights_[(size_t)layer_idx];

    AttentionConfig cfg = attn_cfg_;
    if (lw.attn_heads_layer > 0)    cfg.n_heads    = lw.attn_heads_layer;
    if (lw.attn_kv_heads_layer > 0) cfg.n_kv_heads = lw.attn_kv_heads_layer;
    if (lw.attn_head_dim_layer > 0) cfg.head_dim   = lw.attn_head_dim_layer;
    if (lw.sw_window_size > 0)      cfg.sliding_window = lw.sw_window_size;
    if (lw.rope_freq_base_layer > 0) cfg.rope.freq_base = lw.rope_freq_base_layer;

    const bool qwen35_partial_rope = lw.qwen35_full_attn_q_has_gate && !lw.qwen35_linear;

    std::string arch_l_rope = meta_.arch;
    for (char& c : arch_l_rope) c = (char)std::tolower((unsigned char)c);

    const bool gemma4_global_prope =
        arch_l_rope.find("gemma4") != std::string::npos &&
        cfg.sliding_window == 0 &&
        cfg.head_dim == 512;

    const int nh = kvcache_.kv_heads_for_layer(layer_idx);
    const int hd = kvcache_.head_dim_for_layer(layer_idx);
    if (nh <= 0 || hd <= 0) return;

    std::vector<float> tmp((size_t)hd);
    for (int h = 0; h < nh; ++h) {
        if (!kvcache_.k_mut_ptr(layer_idx, h)) continue;
        kvcache_.get_k_f32(layer_idx, pos, h, tmp.data());
        if (gemma4_global_prope) {
            nx_apply_rope_gemma4_prope_or_full(tmp.data(), 1, delta_pos, hd,
                                               cfg.rope, true,
                                               gemma4_rope_factors_,
                                               gemma4_rope_factors_len_);
        } else {
            nx_apply_rope_qwen35_partial_or_full(tmp.data(), 1, delta_pos, hd,
                                                 cfg.rope, qwen35_partial_rope);
        }
        kvcache_.set_k_f32(layer_idx, pos, h, tmp.data());
    }
}

inline bool Transformer::rope_rescale(float new_scale, const char* reason) {
    if (!loaded_) return false;
    if (!(new_scale > 0.0f)) return false;
    if (new_scale > 1.0f) new_scale = 1.0f;
    if (new_scale < 0.05f) new_scale = 0.05f;

    const float old_scale = rope_pos_scale_;
    if (std::fabs((double)new_scale - (double)old_scale) < 0.0005) return false;

    const int n_tok = kvcache_.n_tokens();
    if (n_tok <= 0) { rope_pos_scale_ = new_scale; return true; }

    fprintf(stderr,
        "[rope-stab] rescale reason=%s tokens=%d scale %.6f -> %.6f eff_end %d -> %d\n",
        reason ? reason : "?", n_tok, old_scale, new_scale,
        (int)std::llround((double)(n_tok - 1) * (double)old_scale),
        (int)std::llround((double)(n_tok - 1) * (double)new_scale));
    fflush(stderr);

    for (int p = 0; p < n_tok; ++p) {
        const int old_eff = (int)std::llround((double)p * (double)old_scale);
        const int new_eff = (int)std::llround((double)p * (double)new_scale);
        const int delta   = new_eff - old_eff;
        if (delta == 0) continue;
        for (int l = 0; l < (int)layer_weights_.size(); ++l) {
            rope_rotate_cached_k_delta(l, p, delta);
        }
    }

    rope_pos_scale_ = new_scale;
    kvcache_.mark_all_gpu_dirty();
    return true;
}

// ─── Context editing ──────────────────────────────────────────────────────────────────────
// Keys are stored already rotated by RoPE for their position. Removing tokens [start, end)
// moves everything after the cut `count` places earlier, so those keys are rotated back by
// `count` positions. The rotation for each layer is built by running that layer's own RoPE
// function on unit vectors (one column of the rotation matrix each), so every layout the engine
// handles -- interleaved, half-split, Qwen3.5 partial rotary, Gemma 4 p-RoPE/frequency factors --
// is covered without a per-model special case. RoPE rotates each dimension pair by an angle
// proportional to position, so "rotate by d" is the same small matrix for every position: it is
// built once per layer and applied as a sparse multiply (at most two non-zeros per output).
inline void Transformer::rope_apply_layer_one_head(int layer_idx, float* x, int pos) const {
    if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size() || !x) return;
    const TransformerLayerWeights& lw = layer_weights_[(size_t)layer_idx];
    AttentionConfig cfg = attn_cfg_;
    if (lw.attn_heads_layer > 0)     cfg.n_heads    = lw.attn_heads_layer;
    if (lw.attn_kv_heads_layer > 0)  cfg.n_kv_heads = lw.attn_kv_heads_layer;
    if (lw.attn_head_dim_layer > 0)  cfg.head_dim   = lw.attn_head_dim_layer;
    if (lw.sw_window_size > 0)       cfg.sliding_window = lw.sw_window_size;
    if (lw.rope_freq_base_layer > 0) cfg.rope.freq_base = lw.rope_freq_base_layer;
    const bool qwen35_partial_rope = lw.qwen35_full_attn_q_has_gate && !lw.qwen35_linear;
    std::string arch_l = meta_.arch;
    for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);
    const bool gemma4_global_prope = arch_l.find("gemma4") != std::string::npos &&
                                     cfg.sliding_window == 0 && cfg.head_dim == 512;
    const int hd = kvcache_.head_dim_for_layer(layer_idx);
    if (gemma4_global_prope)
        nx_apply_rope_gemma4_prope_or_full(x, 1, pos, hd, cfg.rope, true, gemma4_rope_factors_, gemma4_rope_factors_len_);
    else
        nx_apply_rope_qwen35_partial_or_full(x, 1, pos, hd, cfg.rope, qwen35_partial_rope);
}

inline bool Transformer::kv_rotate_positions(int from, int to, int delta) {
    if (delta == 0 || from >= to) return true;
    struct Ent { int i; float w; };
    const int ad = delta < 0 ? -delta : delta;          // only ever ask RoPE for a forward rotation
    const int L  = (int)layer_weights_.size();
    for (int l = 0; l < L; ++l) {
        const int nh = kvcache_.kv_heads_for_layer(l);
        const int hd = kvcache_.head_dim_for_layer(l);
        if (nh <= 0 || hd <= 0) continue;               // no KV here (linear-attention / shared-KV layer)
        std::vector<std::vector<Ent>> col((size_t)hd), use((size_t)hd);
        std::vector<float> e((size_t)hd);
        for (int c = 0; c < hd; ++c) {                  // column c of R(ad) = RoPE(unit vector c)
            std::fill(e.begin(), e.end(), 0.0f); e[(size_t)c] = 1.0f;
            rope_apply_layer_one_head(l, e.data(), ad);
            // Pure rotation only: YaRN also scales magnitude (attention temperature), and the stored
            // keys already carry that once -- so normalise each column back to length 1.
            double nrm = 0.0;
            for (int r = 0; r < hd; ++r) nrm += (double)e[(size_t)r] * e[(size_t)r];
            const float inv = nrm > 0.0 ? (float)(1.0 / std::sqrt(nrm)) : 1.0f;
            for (int r = 0; r < hd; ++r) if (e[(size_t)r] != 0.0f) col[(size_t)c].push_back({r, e[(size_t)r] * inv});
        }
        if (delta < 0) use = col;                        // back by ad:    out[c] = sum_r R[r][c] * in[r]  (R transposed)
        else for (int c = 0; c < hd; ++c)                // forward by ad: out[r] = sum_c R[r][c] * in[c]
            for (const Ent& en : col[(size_t)c]) use[(size_t)en.i].push_back({c, en.w});
        for (int h = 0; h < nh; ++h) {
            uint16_t* kp = kvcache_.k_mut_ptr(l, h);
            if (!kp) continue;
            auto work = [&, kp, hd](int p0, int p1) {
                std::vector<float> in((size_t)hd), out((size_t)hd);
                for (int p = p0; p < p1; ++p) {
                    uint16_t* row = kp + (size_t)p * (size_t)hd;
                    for (int d = 0; d < hd; ++d) in[(size_t)d] = quant::f16_to_f32(row[d]);
                    for (int d = 0; d < hd; ++d) {
                        float acc = 0.0f;
                        for (const Ent& en : use[(size_t)d]) acc += en.w * in[(size_t)en.i];
                        out[(size_t)d] = acc;
                    }
                    for (int d = 0; d < hd; ++d) row[d] = KVCache::f32_to_f16_rne(out[(size_t)d]);
                }
            };
            const int n  = to - from;
            const int nt = std::max(1, std::min(std::max(1, n_threads_), n / 512 + 1));
            if (nt <= 1) work(from, to);
            else {
                std::vector<std::thread> th; const int per = (n + nt - 1) / nt;
                for (int t = 0; t < nt; ++t) {
                    const int a = from + t * per, b = std::min(to, a + per);
                    if (a < b) th.emplace_back(work, a, b);
                }
                for (auto& t : th) t.join();
            }
            kvcache_.mark_slot_gpu_dirty(l, h);
        }
    }
    return true;
}

inline void Transformer::kv_after_edit() {
    kvcache_.mark_all_gpu_dirty();
    q35_kv_drop_all();                                   // the card's Qwen3.5 KV mirror is rebuilt from the edited host copy
    mtp_k_hist_.clear(); mtp_v_hist_.clear(); mtp_hist_len_ = 0;   // MTP's own short history: simply start it over
}

inline bool Transformer::has_recurrent_state() const {
    for (const auto& lw : layer_weights_) if (lw.qwen35_linear) return true;
    return false;
}

inline bool Transformer::kv_remove_range(int start, int end) {
    const int n = kvcache_.n_tokens();
    if (start < 0) start = 0;
    if (end > n) end = n;
    if (start >= end) return true;
    const int count = end - start;
    kvcache_.evict_range(start, end);
    const int new_n = n - count;
    if (start < new_n) {
        if (std::fabs((double)rope_pos_scale_ - 1.0) <= 1e-9) {
            kv_rotate_positions(start, new_n, -count);
        } else {                                         // RoPE stabilizer active: the shift differs per position
            for (int p = start; p < new_n; ++p) {
                const int d = rope_effective_position(p) - rope_effective_position(p + count);
                if (d) for (int l = 0; l < (int)layer_weights_.size(); ++l) rope_rotate_cached_k_delta(l, p, d);
            }
        }
    }
    kv_after_edit();
    fprintf(stderr, "[ctx] removed tokens [%d,%d) from the KV cache (%d kept, %d shifted back)%s\n",
            start, end, new_n, std::max(0, new_n - start),
            has_recurrent_state() ? " -- recurrent layers keep their running state" : "");
    fflush(stderr);
    return true;
}

// Archive blob: "NXKVSEG1" | u32 version | u32 n_layers | u32 start (RoPE position) | u32 n_tokens
//   | u32 hidden | u32 vocab | u32 arch_len | arch (padded to 4) | per layer: u32 kv_heads, u32 head_dim
//   | per layer, per kv head: K [n_tokens*head_dim f16] then V [same]
inline bool Transformer::kv_export_range(int start, int end, std::vector<uint8_t>& out) const {
    const int all = kvcache_.n_tokens();
    if (start < 0 || end > all || start >= end) return false;
    const int n = end - start;
    out.clear();
    auto put32 = [&](uint32_t v) { const uint8_t* b = (const uint8_t*)&v; out.insert(out.end(), b, b + 4); };
    const char magic[8] = {'N','X','K','V','S','E','G','1'};
    out.insert(out.end(), magic, magic + 8);
    const int L = kvcache_.total_layers();
    put32(1); put32((uint32_t)L); put32((uint32_t)rope_effective_position(start)); put32((uint32_t)n);
    put32((uint32_t)meta_.hidden_dim); put32((uint32_t)meta_.vocab_size);
    put32((uint32_t)meta_.arch.size());
    out.insert(out.end(), meta_.arch.begin(), meta_.arch.end());
    while (out.size() % 4) out.push_back(0);
    for (int l = 0; l < L; ++l) { put32((uint32_t)kvcache_.kv_heads_for_layer(l)); put32((uint32_t)kvcache_.head_dim_for_layer(l)); }
    std::vector<uint16_t> kb, vb;
    for (int l = 0; l < L; ++l) {
        const int nh = kvcache_.kv_heads_for_layer(l), hd = kvcache_.head_dim_for_layer(l);
        for (int h = 0; h < nh; ++h) {
            kb.resize((size_t)n * hd); vb.resize((size_t)n * hd);
            if (!kvcache_.read_raw(l, h, start, n, kb.data(), vb.data())) return false;
            const uint8_t* kbb = (const uint8_t*)kb.data(); const uint8_t* vbb = (const uint8_t*)vb.data();
            out.insert(out.end(), kbb, kbb + kb.size() * sizeof(uint16_t));
            out.insert(out.end(), vbb, vbb + vb.size() * sizeof(uint16_t));
        }
    }
    return true;
}

inline int Transformer::kv_import_append(const uint8_t* data, size_t size, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = m; return -1; };
    if (!data || size < 36 || std::memcmp(data, "NXKVSEG1", 8) != 0) return fail("not a Nexus memory archive");
    size_t off = 8;
    auto get32 = [&](uint32_t& v) { if (off + 4 > size) return false; std::memcpy(&v, data + off, 4); off += 4; return true; };
    uint32_t ver = 0, L = 0, start_eff = 0, n = 0, hidden = 0, vocab = 0, alen = 0;
    if (!get32(ver) || !get32(L) || !get32(start_eff) || !get32(n) || !get32(hidden) || !get32(vocab) || !get32(alen))
        return fail("the archive file is cut short");
    if (ver != 1) return fail("this archive was written by a newer version of Nexus");
    if (off + alen > size) return fail("the archive file is cut short");
    const std::string arch((const char*)data + off, alen);
    off += (alen + 3u) & ~3u;
    if ((int)L != kvcache_.total_layers() || hidden != (uint32_t)meta_.hidden_dim ||
        vocab != (uint32_t)meta_.vocab_size || arch != meta_.arch)
        return fail("this memory was saved from a different model");
    size_t need = 0;
    for (uint32_t l = 0; l < L; ++l) {
        uint32_t nh = 0, hd = 0;
        if (!get32(nh) || !get32(hd)) return fail("the archive file is cut short");
        if ((int)nh != kvcache_.kv_heads_for_layer((int)l) || (int)hd != kvcache_.head_dim_for_layer((int)l))
            return fail("this memory was saved from a different model (layer shapes differ)");
        need += (size_t)nh * n * hd * 2 * sizeof(uint16_t);
    }
    if (n == 0) return 0;
    if (off + need > size) return fail("the archive file is cut short");
    if (std::fabs((double)rope_pos_scale_ - 1.0) > 1e-9) return fail("can't restore while the RoPE stabilizer is active");
    const int pos0 = kvcache_.n_tokens();
    const uint8_t* p = data + off;
    for (int l = 0; l < (int)L; ++l) {
        const int nh = kvcache_.kv_heads_for_layer(l), hd = kvcache_.head_dim_for_layer(l);
        const size_t blk = (size_t)n * hd * sizeof(uint16_t);
        for (int h = 0; h < nh; ++h) {
            const uint16_t* kp = (const uint16_t*)p; p += blk;
            const uint16_t* vp = (const uint16_t*)p; p += blk;
            if (!kvcache_.write_raw(l, h, pos0, (int)n, kp, vp)) return fail("not enough memory to restore it");
        }
    }
    kvcache_.set_n_tokens(pos0 + (int)n);
    const int delta = rope_effective_position(pos0) - (int)start_eff;
    if (delta != 0) kv_rotate_positions(pos0, pos0 + (int)n, delta);
    kv_after_edit();
    fprintf(stderr, "[ctx] restored %u archived token(s) at positions [%d,%d) (moved %+d positions)\n",
            n, pos0, pos0 + (int)n, delta);
    fflush(stderr);
    return (int)n;
}

inline bool Transformer::maybe_rope_stabilize(int actual_pos, const char* reason) {
    if (!rope_stabilizer_enabled_ || actual_pos <= 0) return false;
    const int eff = rope_effective_position(actual_pos);
    if (eff <= rope_stabilizer_trigger_) return false;

    float new_scale = (float)rope_stabilizer_target_ / (float)std::max(1, actual_pos);
    if (new_scale > rope_pos_scale_) new_scale = rope_pos_scale_; // never expand during a session
    return rope_rescale(new_scale, reason);
}

// --------- Forward pass ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// NEXUS_DECODE_PROF: time spent outside the decoder layers (embedding, final norm + lm_head, sampling), added to the per-token line.
struct NxProfExtra { double embed = 0, loop = 0, head = 0, samp = 0; };
inline NxProfExtra& nx_prof_extra() { static NxProfExtra e; return e; }
struct NxProfScope {
    double* acc; bool on; std::chrono::steady_clock::time_point t0;
    explicit NxProfScope(double* a) : acc(a), on([] { static const bool v = std::getenv("NEXUS_DECODE_PROF") != nullptr; return v; }()) { if (on) t0 = std::chrono::steady_clock::now(); }
    ~NxProfScope() { if (on) *acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
};

inline int32_t Transformer::forward(const std::vector<int32_t>& tokens, int pos) {
    if (!loaded_) return -1;

    const int hidden = (int)meta_.hidden_dim;
    const int vocab  = (int)meta_.vocab_size;

    float* x = x_scratch_.data();

    // 1. Token embedding lookup
    {
        NxProfScope prof_embed(&nx_prof_extra().embed);
        const int32_t token = tokens[pos];
        quant::dequantize_row(token_embedding_table, x,
                               (size_t)token, (size_t)hidden, emb_type);

        NX_TRACE_DECISION(-1, "forward.token0_id", "%d", token);
        NX_TRACE_DECISION(-1, "forward.embed_type", "%d", (int)emb_type);
        NX_TRACE_DECISION(-1, "forward.embed_scale_active", "%.4f", embed_scale_);
        NX_TRACE_SAMPLE(0, "embed_pre_scale", x, (size_t)hidden);

        // Gemma scales embeddings by sqrt(d_model) after lookup. For all
        // other archs embed_scale_ stays 0 and we skip this entirely.
        if (embed_scale_ > 0.0f) {
            for (int i = 0; i < hidden; ++i) x[i] *= embed_scale_;
        }

        NX_TRACE_SAMPLE(0, "embed_post_scale", x, (size_t)hidden);
    }

    // 1b. Gemma 4: Per-Layer Embeddings (PLE) compute for this token.
    //
    //   token-identity[L]  = embed_tokens_per_layer[token, L*ple_dim..]
    //                        * sqrt(ple_dim_)
    //   context[L]         = per_layer_model_projection(x) [L*ple_dim..]
    //                        * 1/sqrt(hidden)
    //                        then RMSNorm with per_layer_projection_norm
    //   ple_buffer_[L]     = (token-identity[L] + context[L]) * (1/sqrt(2))
    //
    // Non-Gemma 4 / PLE-disabled: skipped entirely.
    if (ple_enabled_ && ple_dim_ > 0) {
        const int32_t token = tokens[pos];
        const int     L     = (int)meta_.n_layers;
        const size_t  total = (size_t)L * (size_t)ple_dim_;

        if (ple_buffer_.size() < total) ple_buffer_.assign(total, 0.0f);

        // (a) Token-identity row: dequantize one row of [vocab, L*ple_dim].
        std::vector<float> tok_id(total);
        quant::dequantize_row(ple_token_embed_, tok_id.data(),
                              (size_t)token, total, ple_token_embed_type_);
        const float tok_scale = std::sqrt((float)ple_dim_);
        for (size_t i = 0; i < total; ++i) tok_id[i] *= tok_scale;

        // (b) Context projection: y = W_proj @ x, shape [L*ple_dim].
        std::vector<float> ctx(total);
        cpu::mat_vec(ple_model_proj_, x, ctx.data(),
                     total, (size_t)hidden, ple_model_proj_type_, n_threads_);
        const float ctx_scale = 1.0f / std::sqrt((float)hidden);
        for (size_t i = 0; i < total; ++i) ctx[i] *= ctx_scale;

        // RMSNorm per ple_dim slice using per_layer_projection_norm (shared across layers).
        for (int li = 0; li < L; ++li) {
            float* slice = ctx.data() + (size_t)li * (size_t)ple_dim_;
            cpu::rms_norm(slice, ple_proj_norm_, slice,
                          (size_t)ple_dim_, attn_cfg_.rms_norm_eps);
        }

        // Combine: (ctx + tok_id) * (1/sqrt(2))
        const float comb = 1.0f / std::sqrt(2.0f);
        for (size_t i = 0; i < total; ++i) {
            ple_buffer_[i] = (ctx[i] + tok_id[i]) * comb;
        }
    }

    // 2. Run through each transformer layer
    {
        NxProfScope prof_loop(&nx_prof_extra().loop);
        // Qwen3.5 on the GPU: consecutive GPU layers are submitted without waiting (see q35_chain_flush).  NEXUS_GPU_CHAIN=0 turns it off.
#ifdef NEXUS_VULKAN
        static const bool chain_env_on = [] { const char* e = std::getenv("NEXUS_GPU_CHAIN"); return !(e && e[0] == '0'); }();
        const bool chain_try = chain_env_on && pos > 0 && gpu_lm_head_;
#else
        const bool chain_try = false;
#endif
        static unsigned long long n_fa_ok = 0, n_fa_no = 0, n_lin_no = 0;   // how often each layer kind stayed in / dropped out of the chain
        for (int l = 0; l < (int)meta_.n_layers; ++l) {
            LayerPinGuard pin_guard(*mm_, l);
            if (chain_try) {
                const TransformerLayerWeights& clw = layer_weights_[l];
                bool done = false;
                if (clw.qwen35_linear) { done = q35_gpu_linear_step(clw, l, x, true); if (!done) ++n_lin_no; }
                else if (!clw.moe) { done = q35_gpu_full_step(clw, l, x, pos, true); if (done) ++n_fa_ok; else ++n_fa_no; }   // gated (Qwen3.5) or plain attention layer
                if (done) continue;
                q35_chain_flush(x);                      // x is on the card: bring it back before the synchronous path touches it
            }
            forward_layer(l, x, pos);
        }
        {   // the join: waits for the last submitted layers (and copies the hidden state + new K/V rows back)
            const bool was_open = chain_try && gpu::gpu().chain_is_open();
            const auto tj = std::chrono::steady_clock::now();
            q35_chain_flush(x);
            if (was_open) {
                static const bool prof = std::getenv("NEXUS_DECODE_PROF") != nullptr;
                static const int interval = [] { const char* e = std::getenv("NEXUS_DECODE_PROF"); int n = e ? std::atoi(e) : 0; return n > 1 ? n : 8; }();
                static int ntok = 0; static double join_ms = 0;
                if (prof) {
                    join_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tj).count();
                    if (++ntok >= interval) {
                        NxProfExtra& e = nx_prof_extra();
                        fprintf(stderr, "[decode-prof] per token over %d tokens (layers chained):  record+run %.2f ms + final wait %.2f ms = layers %.2f ms  | embed %.2f  lm_head+norm %.2f  sampling %.2f ms  | layer cache %llu hits / %llu records | in chain: attention %llu ok / %llu dropped out, linear %llu dropped out\n",
                                ntok, e.loop / ntok - join_ms / ntok, join_ms / ntok, e.loop / ntok, e.embed / ntok, e.head / ntok, e.samp / ntok,
                                (unsigned long long)gpu::gpu().lin_cache_hits(), (unsigned long long)gpu::gpu().lin_cache_misses(),
                                n_fa_ok, n_fa_no, n_lin_no);
                        fflush(stderr); e = NxProfExtra(); ntok = 0; join_ms = 0;
                    }
                }
            }
        }
    }
    NxProfScope prof_head(&nx_prof_extra().head);   // final norm + lm_head + logit post-processing, until forward() returns

    // ------ DIAG: post-decoder residual stream (first token only) ----------
    // After all layers complete, before output_norm. This is the residual
    // stream as it leaves the decoder stack. If this looks healthy but the
    // final logits are bad, the bug is in output_norm or lm_head. If this
    // looks bad, the bug is in the decoder layers.
    const bool diag_final = (pos == 0);
    if (diag_final) {
        double sumsq = 0.0;
        float mn = x[0], mx = x[0];
        int mx_idx = 0;
        int n_nan = 0, n_inf = 0;
        for (int i = 0; i < hidden; ++i) {
            const float xi = x[i];
            if (std::isnan(xi)) { ++n_nan; continue; }
            if (std::isinf(xi)) { ++n_inf; continue; }
            sumsq += (double)xi * (double)xi;
            if (xi < mn) mn = xi;
            if (xi > mx) { mx = xi; mx_idx = i; }
        }
        const float rms = std::sqrt((float)(sumsq / (double)hidden));
        fprintf(stderr,
            "[diag-final] post-decoder residual: rms=%+.4e min=%+.4e max=%+.4e at_idx=%d nan=%d inf=%d\n",
            rms, mn, mx, mx_idx, n_nan, n_inf);
        fflush(stderr);
    }
    // ------ END DIAG ------

    NX_TRACE_SAMPLE(0, "post_decoder_residual", x, (size_t)hidden);

    // 3. Final norm + lm_head projection.
    //    PhiMoE has affine LayerNorm here (output_norm.bias/model.norm.bias exists).
    //    Llama-style models have no bias here and stay on RMSNorm.
    NX_TRACE_DECISION(-1, "final.has_rms_final_weight", "%d", rms_final_weight ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.has_rms_final_bias",   "%d", rms_final_bias   ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.zero_centered_norm",   "%d", zero_centered_norm_ ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.has_lm_head_weight",   "%d", lm_head_weight   ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.has_lm_head_bias",     "%d", lm_head_bias     ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.lm_head_tied_to_embed","%d",
                      (lm_head_weight == token_embedding_table) ? 1 : 0);
    NX_TRACE_DECISION(-1, "final.logit_softcap",        "%.1f", final_logit_softcap_);

    const float* head_in = x;
    if (rms_final_weight) {
        float* xb = xb_scratch_.data();
        nx_apply_model_norm(x, rms_final_weight, rms_final_bias, xb,
                    (size_t)hidden, meta_.rms_norm_eps,
                    zero_centered_norm_);
        head_in = xb;
    }

    // ------ DIAG: post-output-norm residual --------------------------------
    if (diag_final) {
        double sumsq = 0.0;
        float mn = head_in[0], mx = head_in[0];
        int mx_idx = 0, mn_idx = 0;
        int n_nan = 0, n_inf = 0;
        for (int i = 0; i < hidden; ++i) {
            const float xi = head_in[i];
            if (std::isnan(xi)) { ++n_nan; continue; }
            if (std::isinf(xi)) { ++n_inf; continue; }
            sumsq += (double)xi * (double)xi;
            if (xi < mn) { mn = xi; mn_idx = i; }
            if (xi > mx) { mx = xi; mx_idx = i; }
        }
        const float rms = std::sqrt((float)(sumsq / (double)hidden));
        fprintf(stderr,
            "[diag-final] post-output-norm:      rms=%+.4e min=%+.4e at_idx=%d max=%+.4e at_idx=%d nan=%d inf=%d "
            "norm_in_place=%s\n",
            rms, mn, mn_idx, mx, mx_idx, n_nan, n_inf,
            (head_in == x) ? "skipped" : "applied");
        fflush(stderr);
    }
    // ------ END DIAG ------

    bool lm_done = false;

    auto final_is_native_nq_type = [](QuantType t) -> bool {
        return t == QuantType::NQ4_G64 ||
               t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 ||
               t == QuantType::NQ8_G64;
    };

    auto final_native_nq_matvec = [&](const void* W, const float* vin, float* vout,
                                      size_t rows, size_t cols, QuantType wt) -> bool {
        if (!W || !vin || !vout || rows == 0 || cols == 0) return false;
        if (!final_is_native_nq_type(wt)) return false;

        static std::atomic<int> lm_nq_logs{0};
        const int log_id = lm_nq_logs.fetch_add(1, std::memory_order_relaxed);

        const unsigned hw = std::thread::hardware_concurrency();
        int nt = n_threads_ > 1 ? n_threads_ : (hw ? (int)hw : 4);
        nt = std::max(1, std::min<int>(nt, (int)rows));

        if (log_id < 16) {
            fprintf(stderr,
                "[nexus] native NQ threaded lm_head matvec: rows=%zu cols=%zu type=%d threads=%d\n",
                rows, cols, (int)wt, nt);
            fflush(stderr);
        }

        std::atomic<size_t> next_row{0};
        std::vector<std::thread> workers;
        workers.reserve((size_t)nt);
        for (int ti = 0; ti < nt; ++ti) {
            workers.emplace_back([&]() {
                std::vector<float> row(cols);
                for (;;) {
                    const size_t r = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (r >= rows) break;
                    quant::dequantize_row(W, row.data(), r, cols, wt);
                    double acc = 0.0;
                    for (size_t c = 0; c < cols; ++c) {
                        acc += (double)row[c] * (double)vin[c];
                    }
                    vout[r] = (float)acc;
                }
            });
        }
        for (auto& th : workers) th.join();

        if (log_id < 16) {
            float mn = vout[0], mx = vout[0];
            int mn_id = 0, mx_id = 0;
            double ss = 0.0;
            for (size_t i = 0; i < rows; ++i) {
                const float v = vout[i];
                ss += (double)v * (double)v;
                if (v < mn) { mn = v; mn_id = (int)i; }
                if (v > mx) { mx = v; mx_id = (int)i; }
            }
            fprintf(stderr,
                "[nexus] native NQ lm_head result: rms=%+.4e min=%+.4e at_id=%d max=%+.4e at_id=%d\n",
                std::sqrt((float)(ss / (double)rows)), mn, mn_id, mx, mx_id);
            fflush(stderr);
        }
        return true;
    };

    // Native .nxm NQ lm_head is now handled by the normal CPU matmul route.
    // Do not send private NQ*_G64 rows to GPU kernels that only know GGUF layouts.
    if (gpu_lm_head_ && nx_lm_head_gpu_ok(lm_head_type)) {
        lm_done = gpu::gpu().mat_vec(lm_head_weight, head_in, logits_.data(),
                                     (size_t)vocab, (size_t)hidden, lm_head_type);
    }
    if (!lm_done) {
        static std::atomic<int> lm_mainroute_logs{0};
        const int log_id = lm_mainroute_logs.fetch_add(1, std::memory_order_relaxed);
        if (final_is_native_nq_type(lm_head_type) && log_id < 16) {
            fprintf(stderr,
                "[nexus] native NQ lm_head using main cpu::mat_vec route: rows=%zu cols=%zu type=%d threads=%d\n",
                (size_t)vocab, (size_t)hidden, (int)lm_head_type, n_threads_);
            fflush(stderr);
        }
        cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                     (size_t)vocab, (size_t)hidden, lm_head_type, n_threads_);
    }
    if (lm_head_bias) {
        for (uint32_t i = 0; i < meta_.vocab_size; ++i) {
            logits_[i] += lm_head_bias[i];
        }
    }

    // Gemma 2 final-logit softcap: tanh(logits/cap) * cap.
    // Bounds the lm_head output so softmax doesn't tunnel-vision on a single
    // token. For all other archs final_logit_softcap_ stays 0 and we skip.
    if (final_logit_softcap_ > 0.0f) {
        const float cap = final_logit_softcap_;
        const float inv = 1.0f / cap;
        for (uint32_t i = 0; i < meta_.vocab_size; ++i) {
            logits_[i] = fast_tanh(logits_[i] * inv) * cap;
        }
    }

    // ------ DIAG: post-lm_head logits + top-15 -----------------------------
    // The actual values we're sampling from. If the top tokens here are
    // garbage IDs (high token numbers, special tokens unrelated to context),
    // we know inference is producing wrong probability mass. If the top
    // tokens look reasonable but generation degrades, the issue is in the
    // sampling/sequence path.
    if (diag_final) {
        const int vocab_n = (int)meta_.vocab_size;
        double sumsq = 0.0;
        float mn = logits_[0], mx = logits_[0];
        int mx_idx = 0, mn_idx = 0;
        int n_nan = 0, n_inf = 0;
        for (int i = 0; i < vocab_n; ++i) {
            const float v = logits_[i];
            if (std::isnan(v)) { ++n_nan; continue; }
            if (std::isinf(v)) { ++n_inf; continue; }
            sumsq += (double)v * (double)v;
            if (v < mn) { mn = v; mn_idx = i; }
            if (v > mx) { mx = v; mx_idx = i; }
        }
        const float rms = std::sqrt((float)(sumsq / (double)vocab_n));

        fprintf(stderr,
            "[diag-final] logits (vocab=%d):       rms=%+.4e min=%+.4e at_id=%d max=%+.4e at_id=%d nan=%d inf=%d\n",
            vocab_n, rms, mn, mn_idx, mx, mx_idx, n_nan, n_inf);

        // Top-15 token IDs by logit value. Use a small partial-sort.
        constexpr int K = 15;
        struct LogitRank { float v; int id; };
        LogitRank top[K];
        for (int k = 0; k < K; ++k) { top[k].v = -1e30f; top[k].id = -1; }
        for (int i = 0; i < vocab_n; ++i) {
            const float v = logits_[i];
            if (std::isnan(v) || std::isinf(v)) continue;
            // Insert into top[] if larger than smallest current top
            int worst = 0;
            for (int k = 1; k < K; ++k) {
                if (top[k].v < top[worst].v) worst = k;
            }
            if (v > top[worst].v) {
                top[worst].v = v;
                top[worst].id = i;
            }
        }
        // Sort top[] descending by value (small K, simple selection sort).
        for (int a = 0; a < K - 1; ++a) {
            int best = a;
            for (int b = a + 1; b < K; ++b) {
                if (top[b].v > top[best].v) best = b;
            }
            if (best != a) std::swap(top[a], top[best]);
        }

        fprintf(stderr, "[diag-final] top-%d tokens (rank: id=value):\n", K);
        for (int k = 0; k < K; ++k) {
            fprintf(stderr, "[diag-final]   #%-2d id=%-7d logit=%+.4e\n",
                    k + 1, top[k].id, top[k].v);
        }
        fflush(stderr);
    }
    // ------ END DIAG ------

    return 0;
}
// --------- GPU layer offloading ------------------------------------------------------------------------------------------------------------------------------------------------------------

inline int Transformer::set_gpu_layers(int n) {
    if (n <= 0) {
        n_gpu_layers_ = 0;
        gpu::gpu().shutdown();
        fprintf(stderr, "[nexus] GPU offload disabled\n"); fflush(stderr);
        return 0;
    }

    // Clamp to the owned layer count. For distributed shards, meta_.n_layers
    // stays as the full model count, but this process only owns
    // [active_layer_start_, active_layer_end_). GPU layers must apply to that
    // owned slice, not blindly to blk.0 first.
    int max_layers = (int)meta_.n_layers;
    const int own_start = (active_layer_end_ >= 0) ? std::max(0, active_layer_start_) : 0;
    const int own_end   = (active_layer_end_ >= 0) ? std::min(max_layers, active_layer_end_) : max_layers;
    const int own_count = std::max(0, own_end - own_start);
    n = (n == -1) ? own_count : std::min(n, own_count);

    // Initialize Vulkan
    auto& g = gpu::gpu();
    if (!g.ready()) {
        if (!g.init(/*prefer_apu_zero_copy=*/true)) {
            fprintf(stderr, "[nexus] GPU init failed --- running CPU-only\n"); fflush(stderr);
            n_gpu_layers_ = 0;
            gguf::nx_pin_direct_layers_from(0);
            return 0;
        }
    }

    g.set_gpu_layers(n);
    n_gpu_layers_ = n;

    // Mark which owned layers will use the primary GPU.
    for (int l = 0; l < (int)layer_weights_.size(); ++l)
        layer_weights_[l].gpu_layer = false;
    for (int i = 0; i < n_gpu_layers_; ++i) {
        int l = own_start + i;
        if (l >= 0 && l < (int)layer_weights_.size())
            layer_weights_[l].gpu_layer = true;
    }

    // Eagerly upload all GPU layer weight matrices to VRAM now.
    // Without this, each weight uploads lazily on first inference, stalling the
    // first token and pinning the CPU during PCIe DMA.
    fprintf(stderr, "[nexus] Uploading %d GPU layer weights to VRAM from owned layer range %d-%d---\n",
            n_gpu_layers_, own_start, own_start + n_gpu_layers_ - 1);
    fflush(stderr);
    {
        // Discrete card: only as many layers go to VRAM as actually fit (minus room for the
        // KV/activation buffers).  The rest stay on the CPU, which already holds their weights --
        // so nothing is copied twice and nothing is left half-uploaded.
        const gpu::GpuInfo gi_up = g.info();
        const bool discrete = !gi_up.unified_memory && gi_up.vram_bytes > 0;
        const size_t keep_free = 640ull * 1024ull * 1024ull;
        size_t prev_used = g.weight_cache_used();
        size_t biggest_layer = 0;
        int fitted = n_gpu_layers_;
        int vram_layers = -1;     // first layer that is read from system RAM instead of VRAM (-1 = none)
        bool host_mode = false;
        auto enter_host_mode = [&](int i) -> bool {
            if (!discrete || host_mode || !g.host_import_available()) return false;
            host_mode = true; vram_layers = i; g.set_host_only_weights(true); return true;
        };
        for (int i = 0; i < n_gpu_layers_; ++i) {
            int l = own_start + i;
            if (l < 0 || l >= (int)layer_weights_.size()) continue;
            if (discrete && !host_mode && biggest_layer > 0 &&
                prev_used + biggest_layer + keep_free > gi_up.vram_bytes) {
                if (!enter_host_mode(i)) { fitted = i; break; }
            }
            bool ok = eager_upload_layer_weights(l, g);
            if (!ok && !host_mode && enter_host_mode(i)) ok = eager_upload_layer_weights(l, g);   // VRAM ran out unexpectedly: retry this layer from RAM
            size_t now_used = g.weight_cache_used();
            if (!host_mode && now_used > prev_used) biggest_layer = std::max(biggest_layer, now_used - prev_used);
            prev_used = now_used;
            if (!ok) { fitted = i; break; }
        }
        if (host_mode) g.set_host_only_weights(false);
        if (vram_layers >= 0 && fitted > vram_layers)
            fprintf(stderr, "[nexus] Placement: layers %d-%d in VRAM (%zu of %zu MB); layers %d-%d computed on the GPU but read from system RAM "
                            "(no extra copy, runs at RAM speed)\n",
                    own_start, own_start + vram_layers - 1, prev_used / (1024*1024), (size_t)(gi_up.vram_bytes / (1024*1024)),
                    own_start + vram_layers, own_start + fitted - 1);
        if (fitted < n_gpu_layers_) {
            fprintf(stderr, "[nexus] %d of %d requested layers could be put on the GPU -- the other %d layers run on the CPU\n",
                    fitted, n_gpu_layers_, n_gpu_layers_ - fitted);
            for (int i = fitted; i < n_gpu_layers_; ++i) {
                int l = own_start + i;
                if (l >= 0 && l < (int)layer_weights_.size()) layer_weights_[l].gpu_layer = false;
            }
            n_gpu_layers_ = fitted;
            g.set_gpu_layers(fitted);
        }
        // Layers that stay on the CPU need their weights in RAM (those read in place from the model file are copied now);
        // layers that live on the card keep no RAM copy.
        gguf::nx_pin_direct_layers_from(own_start + n_gpu_layers_);
    }
    fprintf(stderr, "[nexus] Layer weight upload complete\n");
    fflush(stderr);

    // Init persistent GPU activation buffers for batched layer recording
    {
        const size_t q_dim  = (size_t)(meta_.n_heads    * meta_.head_dim);
        const size_t kv_dim = (size_t)(meta_.n_kv_heads * meta_.head_dim);
        g.init_activation_bufs(
            meta_.hidden_dim, q_dim, kv_dim, meta_.ffn_dim);
    }

    // Offload lm_head to GPU only when ALL transformer layers are on the primary GPU.
    // With a split config the CPU is already busy with its own layers, so leaving
    // lm_head there keeps the pipeline tighter.  When fully offloaded the CPU has
    // nothing to do between token callbacks except sampling --- give it that freedom.
    gpu_lm_head_ = (active_layer_end_ < 0 && n_gpu_layers_ == max_layers);
    if (gpu_lm_head_) {
        // Eager upload: push lm_head into the GPU weight cache at set_gpu_layers()
        // time so the first token doesn't pay the PCIe upload cost.
        const size_t hidden = (size_t)meta_.hidden_dim;
        const size_t vocab  = (size_t)meta_.vocab_size;
        size_t W_size = 0; uint32_t dummy = 0;
        // K-quant lm_head (a 248k-row matrix, ~4-5 ms per token): re-encode as Q8_0 once, which the GPU streams ~2x faster.  NEXUS_LMHEAD_Q8=0 disables.
        const void* orig_w = lm_head_weight; const QuantType orig_t = lm_head_type;
        bool lm_q8 = false;
        {
            static const bool off = [] { const char* e = std::getenv("NEXUS_LMHEAD_Q8"); return e && e[0] == '0'; }();
            const bool kq = lm_head_type == QuantType::Q2_K || lm_head_type == QuantType::Q3_K || lm_head_type == QuantType::Q4_K ||
                            lm_head_type == QuantType::Q5_K || lm_head_type == QuantType::Q6_K ||
                            lm_head_type == QuantType::F16  || lm_head_type == QuantType::BF16 || lm_head_type == QuantType::F32;   // all slower to stream than Q8_0
            const char* tn = "other";
            switch (lm_head_type) { case QuantType::F32: tn = "F32"; break; case QuantType::F16: tn = "F16"; break; case QuantType::BF16: tn = "BF16"; break;
                case QuantType::Q8_0: tn = "Q8_0"; break; case QuantType::Q2_K: tn = "Q2_K"; break; case QuantType::Q3_K: tn = "Q3_K"; break;
                case QuantType::Q4_K: tn = "Q4_K"; break; case QuantType::Q5_K: tn = "Q5_K"; break; case QuantType::Q6_K: tn = "Q6_K"; break;
                case QuantType::Q4_0: tn = "Q4_0"; break; case QuantType::Q5_0: tn = "Q5_0"; break; default: break; }
            fprintf(stderr, "[nexus] lm_head: type=%s (id %d) %zu x %zu%s%s\n", tn, (int)lm_head_type, vocab, hidden,
                    off ? "  | Q8_0 re-encode disabled (NEXUS_LMHEAD_Q8=0)" : (kq ? "  | will re-encode to Q8_0" : "  | kept as is (already Q8_0 or a type with no faster form)"),
                    hidden % 32 ? "  | cols not a multiple of 32, kept" : "");
            fflush(stderr);
            if (!off && kq && lm_head_weight && hidden % 32 == 0) {
                const auto t0 = std::chrono::steady_clock::now();
                if (nx_requant_to_q8_0(lm_head_weight, lm_head_type, vocab, hidden, n_threads_, lm_head_q8_)) {
                    lm_head_weight = lm_head_q8_.data(); lm_head_type = QuantType::Q8_0; lm_q8 = true;
                    fprintf(stderr, "[nexus] lm_head re-encoded to Q8_0 for faster decode (%.0f MB, %.1f s)\n",
                            lm_head_q8_.size() / 1048576.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                    fflush(stderr);
                }
            }
        }
        auto upload_lm = [&]() -> bool {
            if (!g.size_for_public(lm_head_type, vocab, hidden, W_size, dummy)) return false;
            g.ensure_activation_buffers_public(hidden * sizeof(float), vocab * sizeof(float));
            return g.get_weight_buffer_public(lm_head_weight, W_size, lm_head_type);
        };
        bool lm_ok = upload_lm();
        if (!lm_ok && lm_q8) {                                   // the larger Q8_0 copy did not fit: go back to the original encoding
            lm_head_weight = orig_w; lm_head_type = orig_t; lm_head_q8_.clear(); lm_head_q8_.shrink_to_fit(); lm_q8 = false;
            fprintf(stderr, "[nexus] Q8_0 lm_head did not fit in VRAM - using the original encoding\n"); fflush(stderr);
            lm_ok = upload_lm();
        }
        if (!lm_ok) gpu_lm_head_ = false;
        if (gpu_lm_head_) { fprintf(stderr, "[nexus] lm_head offloaded to GPU (full offload mode)\n"); fflush(stderr); }
        else fprintf(stderr, "[nexus] lm_head did not fit in VRAM -- stays on CPU\n");
    } else {
        fprintf(stderr, "[nexus] lm_head on CPU (split config: %d/%d layers on GPU)\n",
                n_gpu_layers_, max_layers);
    }

    g.print_info();

    fprintf(stderr, "[nexus] GPU offload: %d / %d layers\n", n_gpu_layers_, max_layers); fflush(stderr);
    return n_gpu_layers_;
}

inline int Transformer::set_gpu2_layers(int n, int device_index, int first_layer) {
    if (!loaded_ || n == 0) return 0;
    const int max_layers = (int)meta_.n_layers;
    if (first_layer >= max_layers) return 0;
    if (n > 0) n = std::min(n, max_layers - first_layer);

    // Create a second VulkanBackend for the iGPU device
    gpu2_ = std::make_unique<gpu::VulkanBackend>();
    gpu2_->set_device_index(device_index);
    if (!gpu2_->init(/*prefer_apu_zero_copy=*/true)) {
        fprintf(stderr, "[nexus] Secondary GPU init failed --- device %d\n", device_index);
        gpu2_.reset();
        return 0;
    }
    if (n < 0) {                                    // Maximize: as many as fit on this card
        n = layers_that_fit(*gpu2_, first_layer, gpu2_->info().device_name.c_str());
        if (n <= 0) { gpu2_.reset(); return 0; }
    }
    gpu2_->set_gpu_layers(n);
    n = fill_other_card(*gpu2_, first_layer, n, 2);
    if (n <= 0) { gpu2_.reset(); return 0; }

    n_gpu2_layers_    = n;
    gpu2_first_layer_ = first_layer;

    // Mark those layers with a separate flag (reuse gpu_layer=false, add gpu2_layer flag)
    for (int l = first_layer; l < first_layer + n && l < max_layers; ++l) {
        layer_weights_[l].gpu2_layer = true;
        layer_weights_[l].gpu_layer  = false;  // not primary GPU
    }

    // Store device name for banner display
    auto info2 = gpu2_->info();
    gpu2_device_name_ = info2.device_name;

    fprintf(stderr, "[nexus] Secondary GPU (%s): layers %d---%d\n",
            gpu2_device_name_.c_str(), first_layer, first_layer+n-1);
    return n;
}

// "Maximize" on a card that isn't the prompt card: count how many layers, from first_layer on, fit in what the card
// really has free (VK_EXT_memory_budget; otherwise its size), keeping card_fill_keep_bytes_ free for its working
// buffers and anything else on the card. Stops at a layer another card already holds.
inline int Transformer::layers_that_fit(gpu::VulkanBackend& b, int first_layer, const char* card_name) {
#if defined(NEXUS_VULKAN)
    size_t fr = 0; const bool real = b.vram_free_now(fr);
    if (!real) fr = (size_t)b.info().vram_bytes;
    const size_t room = fr > card_fill_keep_bytes_ ? fr - card_fill_keep_bytes_ : 0;
    int cnt = 0; size_t sum = 0;
    for (int l = first_layer; l < (int)meta_.n_layers; ++l) {
        if (layer_weights_[l].gpu_layer || layer_weights_[l].gpu2_layer) break;
        const size_t bl = turbo_layer_weight_bytes(l);
        if (bl == 0 || sum + bl > room) break;
        sum += bl; ++cnt;
    }
    fprintf(stderr, "[nexus] %s, Maximize: %d layers fit (%zu MB of %zu MB free%s, keeping %zu MB free)\n",
            card_name ? card_name : "card", cnt, sum >> 20, fr >> 20, real ? "" : " -- estimated from its size, the driver doesn't report free memory",
            card_fill_keep_bytes_ >> 20);
    fflush(stderr);
    return cnt;
#else
    (void)b; (void)first_layer; (void)card_name; return 0;
#endif
}

inline int Transformer::fill_other_card(gpu::VulkanBackend& g, int first, int n, int card_no) {
#if defined(NEXUS_VULKAN)
    const int nl = (int)layer_weights_.size();
    {   // its writing buffers first, so the room measured below is what is left after them
        const size_t q_dim  = (size_t)(meta_.n_heads    * meta_.head_dim);
        const size_t kv_dim = (size_t)(meta_.n_kv_heads * meta_.head_dim);
        g.init_activation_bufs(meta_.hidden_dim, q_dim, kv_dim, meta_.ffn_dim);
    }
    size_t fr = 0; const bool real = g.vram_free_now(fr);
    if (!real) {     // no driver report: estimate from the card's size, and keep extra back for what it can't see (the desktop)
        const size_t v = (size_t)g.info().vram_bytes, u = g.weight_cache_used() + (256ull << 20);
        fr = v > u ? v - u : 0;
        fprintf(stderr, "[nexus] graphics card %d: the driver doesn't report free memory -- estimating from its size (256 MB extra kept back)\n", card_no);
    }
    size_t room = fr > card_fill_keep_bytes_ ? fr - card_fill_keep_bytes_ : 0;
    g.set_weight_cache_budget(g.weight_cache_used() + room);
    if ((int)other_card_w_.size() < nl) other_card_w_.resize((size_t)nl);
    const auto t0 = std::chrono::steady_clock::now();
    int done = 0; size_t done_b = 0;
    for (int l = first; l < first + n && l < nl; ++l) {
        const size_t need = turbo_layer_weight_bytes(l);
        if (need > room) {
            fprintf(stderr, "[nexus] graphics card %d (%s): layer %d (%zu MB) doesn't fit (%zu MB left, keeping %zu MB free) -- "
                            "it and the layers after it that were meant for this card run on the CPU\n",
                    card_no, g.info().device_name.c_str(), l, need >> 20, room >> 20, card_fill_keep_bytes_ >> 20);
            break;
        }
        std::vector<const void*> mine;
        if (!eager_upload_layer_weights(l, g, &mine)) break;
        const TransformerLayerWeights& lw = layer_weights_[l];
        bool ok = true;
        auto up = [&](const void* W, QuantType qt, size_t rows, size_t cols) {
            if (!ok || !W || rows == 0 || cols == 0) return;
            size_t sz = 0; uint32_t dummy = 0;
            if (!g.size_for_public(qt, rows, cols, sz, dummy) || sz == 0) return;
            if (!g.get_weight_buffer_public(W, sz, qt) || g.weight_size_public(W) < sz) { ok = false; return; }
            mine.push_back(W);
        };
        if (lw.qwen35_linear) {
            const size_t hidden = (size_t)meta_.hidden_dim;
            const size_t conv  = (size_t)std::max(0, lw.q35_conv_dim);
            const size_t value = (size_t)std::max(0, lw.q35_value_dim);
            const size_t vh    = (size_t)std::max(0, lw.q35_num_v_heads);
            up(lw.q35_qkv, lw.q35_qkv_type, conv, hidden);
            up(lw.q35_z,   lw.q35_z_type,   value, hidden);
            up(lw.q35_out, lw.q35_out_type, hidden, value);
            if (lw.q35_ba) up(lw.q35_ba, lw.q35_ba_type, vh * 2u, hidden);
            else { up(lw.q35_a, lw.q35_a_type, vh, hidden); up(lw.q35_b, lw.q35_b_type, vh, hidden); }
        }
        if (!ok) { for (const void* w : mine) g.evict_weight_public(w); break; }
        for (const void* w : mine) g.pin_weight_public(w);
        other_card_w_[(size_t)l] = mine;
        room = room > need ? room - need : 0;
        ++done; done_b += need;
        fprintf(stderr, "[load] %d%%  Moving layers onto graphics card %d (%d of %d)\n", (int)(100.0 * done / std::max(1, n)), card_no, done, n);
        fflush(stderr);
    }
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t fr2 = 0; const bool real2 = g.vram_free_now(fr2);
    fprintf(stderr, "[nexus] graphics card %d (%s): %d layers (%zu MB) on the card for writing in %.1f s%s\n",
            card_no, g.info().device_name.c_str(), done, done_b >> 20, el,
            real2 ? (", card memory free now: " + std::to_string(fr2 >> 20) + " MB").c_str() : "");
    fflush(stderr);
    return done;
#else
    (void)g; (void)first; (void)n; (void)card_no; return 0;
#endif
}

inline int Transformer::add_gpu_layers(int n, int device_index, int first_layer) {
    if (!loaded_ || n == 0) return 0;
    const int max_layers = (int)meta_.n_layers;
    if (first_layer >= max_layers) return 0;
    if (n > 0) n = std::min(n, max_layers - first_layer);
    auto b = std::make_unique<gpu::VulkanBackend>();
    b->set_device_index(device_index);
    if (!b->init(/*prefer_apu_zero_copy=*/true)) {
        fprintf(stderr, "[nexus] Extra GPU init failed --- device %d\n", device_index);
        return 0;
    }
    if (n < 0) {                                    // Maximize: as many as fit on this card
        n = layers_that_fit(*b, first_layer, b->info().device_name.c_str());
        if (n <= 0) return 0;
    }
    b->set_gpu_layers(n);
    n = fill_other_card(*b, first_layer, n, (int)gpux_.size() + 3);
    if (n <= 0) return 0;
    const int idx = (int)gpux_.size();
    for (int l = first_layer; l < first_layer + n && l < max_layers; ++l) {
        layer_weights_[l].gpu2_layer = true;     // "not the primary card" - same handling as the second card
        layer_weights_[l].gpu_layer  = false;
        layer_weights_[l].gpux       = idx;
    }
    gpux_names_.push_back(b->info().device_name);
    fprintf(stderr, "[nexus] GPU %d (%s): layers %d---%d\n", idx + 3, gpux_names_.back().c_str(), first_layer, first_layer + n - 1);
    gpux_.push_back(std::move(b));
    return n;
}

inline gpu::GpuInfo Transformer::gpu_info() const {
    return gpu::gpu().info();
}


inline void Transformer::turbo_prefetch_wait(std::future<void>& fut)
{
    if (fut.valid()) fut.wait();
}

inline size_t Transformer::turbo_layer_weight_bytes(int layer_idx) const
{
#ifdef NEXUS_VULKAN
    if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size()) return 0;
    const TransformerLayerWeights& lw = layer_weights_[layer_idx];
    const size_t hidden = (size_t)meta_.hidden_dim;
    const size_t ffn    = (size_t)meta_.ffn_dim;
    size_t total = 0;

    auto gpu_weight_compatible = [](QuantType wt) -> bool {
        return wt == QuantType::Q8_0 ||
               wt == QuantType::Q4_K ||
               wt == QuantType::Q5_K ||
               wt == QuantType::Q6_K ||
               wt == QuantType::Q2_K ||
               wt == QuantType::Q3_K ||
               wt == QuantType::NQ4_G64 ||
               wt == QuantType::NQ5_G64 ||
               wt == QuantType::NQ6_G64 ||
               wt == QuantType::NQ8_G64;
    };

    auto add = [&](const void* W, QuantType wt, size_t rows, size_t cols) {
        if (!W || rows == 0 || cols == 0) return;
        if (!gpu_weight_compatible(wt)) return;
        size_t bytes = 0; uint32_t nb = 0;
        if (gpu::gpu().size_for_public(wt, rows, cols, bytes, nb)) total += bytes;
    };

    if (lw.qwen35_linear) {
        const size_t conv  = (size_t)std::max(0, lw.q35_conv_dim);
        const size_t value = (size_t)std::max(0, lw.q35_value_dim);
        const size_t vh    = (size_t)std::max(0, lw.q35_num_v_heads);
        if (conv  > 0) add(lw.q35_qkv, lw.q35_qkv_type, conv,  hidden);
        if (value > 0) { add(lw.q35_z, lw.q35_z_type, value, hidden);
                         add(lw.q35_out, lw.q35_out_type, hidden, value); }
        if (lw.q35_ba && vh > 0)      add(lw.q35_ba, lw.q35_ba_type, vh*2u, hidden);
        else if (vh > 0)            { add(lw.q35_a, lw.q35_a_type, vh, hidden);
                                      add(lw.q35_b, lw.q35_b_type, vh, hidden); }
    } else {
        const size_t q_dim  = lw.attn_q_dim_layer  > 0 ? (size_t)lw.attn_q_dim_layer  : (size_t)(meta_.n_heads * meta_.head_dim);
        const size_t kv_dim = lw.attn_kv_dim_layer > 0 ? (size_t)lw.attn_kv_dim_layer : (size_t)(meta_.n_kv_heads * meta_.head_dim);
        if (lw.fused_qkv && lw.wqkv) {
            add(lw.wqkv, lw.wqkv_type, q_dim + 2u*kv_dim, hidden);
        } else {
            // Match the resident loader: gated wq is the packed q_dim*2 tensor.
            const size_t wq_rows = (lw.qwen35_full_attn_q_has_gate && lw.qwen35_full_attn_q_rows > 0)
                                       ? (size_t)lw.qwen35_full_attn_q_rows : q_dim;
            add(lw.wq, lw.wq_type, wq_rows, hidden);
            add(lw.wk, lw.wk_type, kv_dim, hidden);
            add(lw.wv, lw.wv_type, kv_dim, hidden);
        }
        add(lw.wo, lw.wo_type, hidden, q_dim);
    }

    if (!lw.moe && hidden > 0 && ffn > 0) {
        if (lw.fused_gate_up) add(lw.wup, lw.wup_type, ffn*2u, hidden);
        else { add(lw.wgate, lw.wgate_type, ffn, hidden);
               add(lw.wup,   lw.wup_type,   ffn, hidden); }
        add(lw.wdown, lw.wdown_type, hidden, ffn);
    }
    return total;
#else
    (void)layer_idx; return 0;
#endif
}

inline size_t Transformer::compute_and_set_turbo_vram_budget(size_t kv_attn_reserve_bytes, int ctx_tokens)
{
    // Attention reads one layer's whole conversation memory on the card (K+V, f16): ~400 MB at 100k tokens. A fixed 320 MB
    // reserve stopped covering it long before that, so the reserve now grows with the conversation.
    if (ctx_tokens > 0) {
        const size_t kv_layer = (size_t)ctx_tokens * (size_t)meta_.n_kv_heads * (size_t)meta_.head_dim * 2u * 2u + (64ull << 20);
        kv_attn_reserve_bytes = std::max(kv_attn_reserve_bytes, kv_layer);
    }
#ifdef NEXUS_VULKAN
    const int n = (int)meta_.n_layers;
    const int win = turbo_layer_window_size();
    if (n <= 0) return gpu::gpu().weight_cache_budget();

    // Estimate the rolling resident-window footprint for logging and for
    // unified-memory / unknown-VRAM fallback. This estimate is NOT the hard
    // budget on a discrete GPU; the actual card size is the budget source.
    size_t max_window = 0, cur = 0;
    std::vector<size_t> bytes(n);
    for (int l = 0; l < n; ++l) bytes[l] = turbo_layer_weight_bytes(l);
    for (int l = 0; l < n; ++l) {
        cur += bytes[l];
        if (l >= win) cur -= bytes[l - win];
        if (l >= win - 1 && cur > max_window) max_window = cur;
    }

    size_t largest_layer = 0;
    for (int l = 0; l < n; ++l) largest_layer = std::max(largest_layer, bytes[l]);
    const size_t required_window = max_window + largest_layer;

    const size_t old_budget = gpu::gpu().weight_cache_budget();
    const gpu::GpuInfo gi = gpu::gpu().info();

    auto env_mb = [](const char* name, int defv) -> int {
        const char* v = std::getenv(name);
        if (!v || !*v) return defv;
        int x = std::atoi(v);
        return x < 0 ? defv : x;
    };

    // Hard override for diagnosis. This is a WEIGHT-CACHE budget, not total VRAM.
    const int forced_mb = env_mb("NEXUS_TURBO_WEIGHT_CACHE_MB", -1);
    size_t final_budget = 0;
    const char* source = "unknown";

    if (forced_mb >= 0) {
        final_budget = (size_t)forced_mb * 1024ull * 1024ull;
        source = "forced-env";
    } else if (gi.vram_bytes > 0 && !gi.unified_memory) {
        // Correct semantics for a discrete card:
        //   weight_cache_budget = total device VRAM - scratch/KV reserve - safety.
        // The launcher field was never supposed to cap this to only the old
        // layer-residency guess. The backend budget tracks cached WEIGHTS only;
        // scratch buffers are separate allocations, so we subtract their reserve
        // here instead of adding it to the weight budget.
        const size_t safety = (size_t)env_mb("NEXUS_TURBO_VRAM_SAFETY_MB", 384)
                            * 1024ull * 1024ull;
        if (gi.vram_bytes > kv_attn_reserve_bytes + safety + 64ull * 1024ull * 1024ull) {
            final_budget = gi.vram_bytes - kv_attn_reserve_bytes - safety;
            source = "device-vram";
        } else {
            final_budget = old_budget ? old_budget : required_window;
            source = "device-vram-too-small-keep";
        }
    } else {
        // Unified memory / unknown VRAM: use the resident-window estimate and do
        // not collapse a user/launcher budget if the estimator cannot see weights.
        final_budget = required_window ? required_window : old_budget;
        source = required_window ? "window-estimate" : "keep-old-estimator-zero";
    }

    // If we have a real required-window estimate, never set a budget below it.
    // If the estimate is zero (common for paths we do not pre-scan, like routed
    // MoE experts), keep the card-derived budget instead of shrinking to reserve.
    if (required_window > 0 && final_budget < required_window) {
        final_budget = required_window;
        source = "raised-to-window";
    }

#ifdef NEXUS_VULKAN
    // The budget above comes from the card's SIZE: it doesn't see the desktop, Nexus's own buffers, or layers parked on
    // the card. Filling to it packed a 3 GB card to ~2.9 GB and ran out at the edges (MTP, long chats). When the driver
    // reports what is really free, the cache may only grow into that, minus the attention reserve and the safety margin.
    // Launcher "Prompt working space on the card": NEXUS_TURBO_BUDGET_REAL=0 = by the card's size (the older behaviour).
    static const bool budget_real = [] { const char* e = std::getenv("NEXUS_TURBO_BUDGET_REAL"); return !(e && e[0] == '0'); }();
    if (budget_real && gi.vram_bytes > 0 && !gi.unified_memory) {
        size_t fr = 0;
        if (gpu::gpu().vram_free_now(fr)) {
            const size_t safety = (size_t)env_mb("NEXUS_TURBO_VRAM_SAFETY_MB", 384) * 1024ull * 1024ull;
            const size_t hold   = kv_attn_reserve_bytes + safety;
            const size_t real_cap = gpu::gpu().weight_cache_used() + (fr > hold ? fr - hold : 0);
            if (final_budget > real_cap) {
                if (real_cap < required_window)
                    fprintf(stderr, "[turbo:auto-budget] the card has %zu MB really free: less than the prompt speed-up's window "
                                    "(%zu MB) after its reserve -- it will swap layers more often\n", fr >> 20, required_window >> 20);
                final_budget = real_cap; source = "real-free";
            }
        }
    }
#endif
    gpu::gpu().set_weight_cache_budget(final_budget);
    fprintf(stderr,
        "[turbo:auto-budget] source=%s vram=%zuMB reserve=%zuMB window=%d "
        "max_window=%zuMB spare=%zuMB old=%zuMB => weight_cache_budget=%zuMB\n",
        source,
        gi.vram_bytes / (1024*1024),
        kv_attn_reserve_bytes / (1024*1024),
        win,
        max_window / (1024*1024),
        largest_layer / (1024*1024),
        old_budget / (1024*1024),
        final_budget / (1024*1024));
    fflush(stderr);
    return final_budget;
#else
    (void)kv_attn_reserve_bytes; return 0;
#endif
}

inline int Transformer::turbo_layer_window_size() const
{
    const char* env = std::getenv("NEXUS_TURBO_LAYER_WINDOW");
    int w = env && *env ? std::atoi(env) : 3;
    if (w < 1) w = 1;
    if (w > 8) w = 8;
    return w;
}

// True only on the background loader thread (see turbo_stream_begin_layer).
inline bool& Transformer_turbo_loader_flag() { static thread_local bool f = false; return f; }
inline bool turbo_loader_active_tl() { return Transformer_turbo_loader_flag(); }

inline void Transformer::turbo_prefetch_mm_weight(const void* W, QuantType wt,
                                                   size_t rows, size_t cols,
                                                   const char* tag, int layer_idx,
                                                   bool pin_after_load)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready() || !W || rows == 0 || cols == 0)
        return;
    const bool gpu_weight_compatible =
        wt == QuantType::Q8_0  ||
        wt == QuantType::Q4_K  ||
        wt == QuantType::Q5_K  ||
        wt == QuantType::Q6_K  ||
        wt == QuantType::Q2_K  ||
        wt == QuantType::Q3_K  ||
        wt == QuantType::NQ4_G64 ||
        wt == QuantType::NQ5_G64 ||
        wt == QuantType::NQ6_G64 ||
        wt == QuantType::NQ8_G64;
    if (!gpu_weight_compatible)
        return;

    size_t W_size = 0;
    uint32_t n_blocks = 0;
    if (!gpu::gpu().size_for_public(wt, rows, cols, W_size, n_blocks))
        return;

    // Called from the loader thread when the GPU layer window is > 1: same upload, but without
    // holding the cache lock during the copy and without ever evicting.
    const bool ok = turbo_loader_active_tl()
        ? gpu::gpu().load_weight_async_public(W, W_size, wt, pin_after_load)
        : gpu::gpu().load_weight_force_public(W, W_size, wt, pin_after_load);
    static std::atomic<int> pre_logs{0};
    const int log_id = pre_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id < 160) {
        fprintf(stderr,
            "[turbo:resident] L=%d %-28s %s rows=%zu cols=%zu type=%d size=%zuMB pin=%d\n",
            layer_idx,
            tag ? tag : "?",
            ok ? "OK" : "skip",
            rows,
            cols,
            (int)wt,
            W_size / (1024*1024),
            pin_after_load ? 1 : 0);
        fflush(stderr);
    }
#else
    (void)W; (void)wt; (void)rows; (void)cols; (void)tag; (void)layer_idx;
#endif
}


inline void Transformer::turbo_unpin_evict_mm_weight(const void* W)
{
#ifdef NEXUS_VULKAN
    if (!W || !gpu::gpu().ready()) return;
    gpu::gpu().unpin_weight_public(W);
    gpu::gpu().evict_weight_public(W);
#else
    (void)W;
#endif
}

inline void Transformer::turbo_load_layer_weights_resident(int layer_idx, const char* prefix, bool pin_after_load)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size()) return;
    if (park_on_card(layer_idx)) return;      // a parked layer that stayed on the card: already there, pinned by parking
    if (turbo_keep_on(layer_idx)) return;     // lives on this card: loaded and pinned once by turbo_keep_prepare()

    const TransformerLayerWeights& lw = layer_weights_[layer_idx];
    const size_t hidden = (size_t)meta_.hidden_dim;
    const size_t ffn    = (size_t)meta_.ffn_dim;

    auto tagbuf = [&](const char* name) -> std::string {
        std::string s = prefix ? prefix : "win";
        s += ".";
        s += name ? name : "?";
        return s;
    };

    if (lw.qwen35_linear) {
        const size_t conv  = (size_t)std::max(0, lw.q35_conv_dim);
        const size_t value = (size_t)std::max(0, lw.q35_value_dim);
        const size_t vh    = (size_t)std::max(0, lw.q35_num_v_heads);
        if (conv > 0) {
            auto t = tagbuf("linear_attn.in_proj_qkv");
            turbo_prefetch_mm_weight(lw.q35_qkv, lw.q35_qkv_type,
                                     conv, hidden, t.c_str(), layer_idx, pin_after_load);
        }
        if (value > 0) {
            auto t1 = tagbuf("linear_attn.in_proj_z");
            turbo_prefetch_mm_weight(lw.q35_z, lw.q35_z_type,
                                     value, hidden, t1.c_str(), layer_idx, pin_after_load);
            auto t2 = tagbuf("linear_attn.out_proj");
            turbo_prefetch_mm_weight(lw.q35_out, lw.q35_out_type,
                                     hidden, value, t2.c_str(), layer_idx, pin_after_load);
        }
        if (lw.q35_ba && vh > 0) {
            auto t = tagbuf("linear_attn.beta_alpha");
            turbo_prefetch_mm_weight(lw.q35_ba, lw.q35_ba_type,
                                     vh * 2u, hidden, t.c_str(), layer_idx, pin_after_load);
        } else if (vh > 0) {
            auto t1 = tagbuf("linear_attn.in_proj_a");
            turbo_prefetch_mm_weight(lw.q35_a, lw.q35_a_type,
                                     vh, hidden, t1.c_str(), layer_idx, pin_after_load);
            auto t2 = tagbuf("linear_attn.in_proj_b");
            turbo_prefetch_mm_weight(lw.q35_b, lw.q35_b_type,
                                     vh, hidden, t2.c_str(), layer_idx, pin_after_load);
        }
    } else {
        // Best-effort for normal/full-attention layers. The Qwen3Next test path is
        // mostly qwen35_linear, but this keeps the window from being empty if a
        // full-attention layer appears.
        const size_t q_dim  = lw.attn_q_dim_layer  > 0 ? (size_t)lw.attn_q_dim_layer  : (size_t)(meta_.n_heads * meta_.head_dim);
        const size_t kv_dim = lw.attn_kv_dim_layer > 0 ? (size_t)lw.attn_kv_dim_layer : (size_t)(meta_.n_kv_heads * meta_.head_dim);
        if (lw.fused_qkv && lw.wqkv) {
            auto t = tagbuf("attn.qkv");
            turbo_prefetch_mm_weight(lw.wqkv, lw.wqkv_type,
                                     q_dim + 2u * kv_dim, hidden, t.c_str(), layer_idx, pin_after_load);
        } else {
            // Gated full-attn (Qwen3-Next) packs query+gate into wq as q_dim*2
            // rows. Pin at the PACKED size so the batched GPU mat_mat can read all
            // q_dim*2 rows in-bounds. Verified safe: for gated layers no other GPU
            // site reads wq from the weight cache (the per-token GPU mat_vec is
            // explicitly disabled when qwen35_full_attn_q_has_gate). Plain layers
            // pin at q_dim as before.
            const size_t wq_rows = (lw.qwen35_full_attn_q_has_gate && lw.qwen35_full_attn_q_rows > 0)
                                       ? (size_t)lw.qwen35_full_attn_q_rows
                                       : q_dim;
            auto tq = tagbuf("attn.q");
            turbo_prefetch_mm_weight(lw.wq, lw.wq_type, wq_rows, hidden, tq.c_str(), layer_idx, pin_after_load);
            auto tk = tagbuf("attn.k");
            turbo_prefetch_mm_weight(lw.wk, lw.wk_type, kv_dim, hidden, tk.c_str(), layer_idx, pin_after_load);
            auto tv = tagbuf("attn.v");
            turbo_prefetch_mm_weight(lw.wv, lw.wv_type, kv_dim, hidden, tv.c_str(), layer_idx, pin_after_load);
        }
        auto to = tagbuf("attn.o");
        turbo_prefetch_mm_weight(lw.wo, lw.wo_type, hidden, q_dim, to.c_str(), layer_idx, pin_after_load);
    }

    if (!lw.moe && hidden > 0 && ffn > 0) {
        if (lw.fused_gate_up) {
            auto t = tagbuf("ffn.fused_gate_up");
            turbo_prefetch_mm_weight(lw.wup, lw.wup_type,
                                     ffn * 2u, hidden, t.c_str(), layer_idx, pin_after_load);
        } else {
            auto tg = tagbuf("ffn.gate");
            turbo_prefetch_mm_weight(lw.wgate, lw.wgate_type,
                                     ffn, hidden, tg.c_str(), layer_idx, pin_after_load);
            auto tu = tagbuf("ffn.up");
            turbo_prefetch_mm_weight(lw.wup, lw.wup_type,
                                     ffn, hidden, tu.c_str(), layer_idx, pin_after_load);
        }
        auto td = tagbuf("ffn.down");
        turbo_prefetch_mm_weight(lw.wdown, lw.wdown_type,
                                 hidden, ffn, td.c_str(), layer_idx, pin_after_load);
    }
#else
    (void)layer_idx; (void)prefix; (void)pin_after_load;
#endif
}

inline void Transformer::turbo_release_layer_weights_resident(int layer_idx)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    if (layer_idx < 0 || layer_idx >= (int)layer_weights_.size()) return;
    if (park_on_card(layer_idx)) return;      // stays on the card for writing: never evicted by the prompt stream
    if (turbo_keep_on(layer_idx)) return;     // lives on this card for writing too: never evicted by the prompt stream
    const TransformerLayerWeights& lw = layer_weights_[layer_idx];

    turbo_unpin_evict_mm_weight(lw.q35_qkv);
    turbo_unpin_evict_mm_weight(lw.q35_z);
    turbo_unpin_evict_mm_weight(lw.q35_a);
    turbo_unpin_evict_mm_weight(lw.q35_b);
    turbo_unpin_evict_mm_weight(lw.q35_ba);
    turbo_unpin_evict_mm_weight(lw.q35_out);

    turbo_unpin_evict_mm_weight(lw.wqkv);
    turbo_unpin_evict_mm_weight(lw.wq);
    turbo_unpin_evict_mm_weight(lw.wk);
    turbo_unpin_evict_mm_weight(lw.wv);
    turbo_unpin_evict_mm_weight(lw.wo);

    turbo_unpin_evict_mm_weight(lw.wgate);
    turbo_unpin_evict_mm_weight(lw.wup);
    turbo_unpin_evict_mm_weight(lw.wdown);
#else
    (void)layer_idx;
#endif
}

inline void Transformer::turbo_prefetch_current_tail_async(int /*layer_idx*/)
{
    // Tail weights (out_proj, ffn gate/up/down) are already resident+pinned by
    // the window loader. No thread needed; avoids concurrent queue submits.
}

inline void Transformer::turbo_prefetch_next_layer_async(int layer_idx)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    const int n_layers = turbo_range_end_ >= 0 ? turbo_range_end_ : (int)layer_weights_.size();
    if (layer_idx < 0 || layer_idx >= n_layers) return;

    const int win  = turbo_layer_window_size();
    const int edge = layer_idx + win - 2;           // same edge math as before
    if (edge >= 0 && edge < n_layers)
        turbo_load_layer_weights_resident(edge, "win", true);   // on main thread
#else
    (void)layer_idx;
#endif
}

inline int Transformer::turbo_gpu_layers()
{
    const char* e = std::getenv("NEXUS_TURBO_GPU_LAYERS");
    int n = (e && *e) ? std::atoi(e) : 1;
    if (n < 1) n = 1;
    if (n > 4) n = 4;
    return n;
}

inline void Transformer::turbo_stream_begin_layer(int layer_idx, const char* tag)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    const int W = turbo_gpu_layers();
    const int n_layers = turbo_range_end_ >= 0 ? turbo_range_end_ : (int)layer_weights_.size();
    const auto t_load0 = std::chrono::steady_clock::now();
    if (layer_idx <= turbo_last_begin_layer_) turbo_loader_.join();   // a new prefill started
    turbo_last_begin_layer_ = layer_idx;

    // If the background loader is still copying this layer's weights, wait for it.
    // (Never touches the device while a transfer is in flight.)
    turbo_loader_.join();

    // Clean slate for everything that is not part of the window (unpinned leftovers).
    gpu::gpu().wait_idle();
    gpu::gpu().evict_unpinned_weights();
    turbo_load_layer_weights_resident(layer_idx, tag, true);
    // Second pass is a no-op for weights already cached and retries any weight
    // whose first upload failed or was skipped (failed uploads are never cached).
    turbo_load_layer_weights_resident(layer_idx, tag, true);
    gpu::gpu().wait_idle();

    // Window > 1: start uploading the next layer(s) now, while this layer computes.
    if (W > 1 && layer_idx + 1 < n_layers) {
        const int first = layer_idx + 1;
        const int last  = std::min(n_layers - 1, layer_idx + W - 1);
        turbo_loader_.t = std::thread([this, first, last]() {
            Transformer_turbo_loader_flag() = true;
            for (int k = first; k <= last; ++k)
                turbo_load_layer_weights_resident(k, "win", true);
            Transformer_turbo_loader_flag() = false;
        });
    }
    // Timing: how long this layer had to wait for its weights (with a window > 1 this
    // should drop towards 0 because the weights were uploaded during the previous layer).
    turbo_stream_t_loaded_ = std::chrono::steady_clock::now();
    if (gpu::gpu().prof_enabled()) gpu::gpu().prof_reset();
    turbo_stream_load_ms_  = std::chrono::duration<double, std::milli>(turbo_stream_t_loaded_ - t_load0).count();
#else
    (void)layer_idx; (void)tag;
#endif
}

inline void Transformer::turbo_stream_end_layer(int layer_idx)
{
#ifdef NEXUS_VULKAN
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    const auto t_c1 = std::chrono::steady_clock::now();
    const double comp_ms = std::chrono::duration<double, std::milli>(t_c1 - turbo_stream_t_loaded_).count();
    // The next layer's upload must be finished before anything frees or waits on the device.
    turbo_loader_.join();
    const double tail_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_c1).count();
    // Everything queued for this layer must be finished before its VRAM is freed.
    gpu::gpu().wait_idle();
    fprintf(stderr,
        "[turbo:layer] L=%d load-wait=%.0f ms compute=%.0f ms next-layer-wait=%.0f ms "
        "(waiting on weights: %.1f%% of layer time) gpu-layers=%d\n",
        layer_idx, turbo_stream_load_ms_, comp_ms, tail_ms,
        100.0 * (turbo_stream_load_ms_ + tail_ms) / std::max(1.0, turbo_stream_load_ms_ + comp_ms + tail_ms),
        turbo_gpu_layers());
    fflush(stderr);
    if (gpu::gpu().prof_enabled()) {
        const auto sn = gpu::gpu().prof_snapshot();
        static const char* nm[] = {"mat_mat", "conv", "scan", "attn"};
        double ops = 0.0;
        for (int i = 0; i < gpu::VulkanBackend::P_N; ++i) ops += sn.total_ms[i];
        fprintf(stderr, "[turbo:profile] L=%d compute=%.0f ms |", layer_idx, comp_ms);
        for (int i = 0; i < gpu::VulkanBackend::P_N; ++i)
            fprintf(stderr, " %s %.0f ms (GPU-wait %.0f, host %.0f, %llu calls) |", nm[i],
                    sn.total_ms[i], sn.wait_ms[i], sn.total_ms[i] - sn.wait_ms[i], sn.calls[i]);
        fprintf(stderr, " outside GPU ops (CPU work: norms, residuals, gating, conv state...) %.0f ms\n",
                std::max(0.0, comp_ms - ops));
        fflush(stderr);
        gpu::gpu().prof_dump_mm(layer_idx);
    }
    turbo_release_layer_weights_resident(layer_idx);
    gpu::gpu().wait_idle();
#else
    (void)layer_idx;
#endif
}

inline bool Transformer::turbo_is_native_nq_type(QuantType t)
{
    return t == QuantType::NQ4_G64 ||
           t == QuantType::NQ5_G64 ||
           t == QuantType::NQ6_G64 ||
           t == QuantType::NQ8_G64;
}

inline bool Transformer::turbo_is_standard_gguf_mm_type(QuantType t)
{
    return t == QuantType::Q8_0 ||
           t == QuantType::Q4_K ||
           t == QuantType::Q5_K ||
           t == QuantType::Q6_K ||
           t == QuantType::Q2_K ||
           t == QuantType::Q3_K;
}

inline const char* Transformer::turbo_mm_family_name(QuantType t)
{
    if (turbo_is_native_nq_type(t)) return "native_nq";
    if (turbo_is_standard_gguf_mm_type(t)) return "standard_gguf";
    return "unsupported";
}

inline bool Transformer::turbo_mat_mat_family_or_cpu(const void* W, QuantType wt,
                                                     const float* X, float* Y,
                                                     size_t rows, size_t cols, size_t n_tokens,
                                                     const char* tag, int layer_idx,
                                                     const char* forced_family)
{
    if (!W || !X || !Y || rows == 0 || cols == 0 || n_tokens == 0) {
        if (Y && rows && n_tokens) std::memset(Y, 0, rows * n_tokens * sizeof(float));
        return false;
    }

    const bool is_native = turbo_is_native_nq_type(wt);
    const bool is_std    = turbo_is_standard_gguf_mm_type(wt);
    const bool force_native = forced_family && std::strcmp(forced_family, "native_nq") == 0;
    const bool force_std    = forced_family && std::strcmp(forced_family, "standard_gguf") == 0;

    // NATIVE NQ LEGACY FENCE:
    // The .nxm Qwen3Next path was already correct before standard GGUF Q4/Q5/Q6/Q8
    // mat_mat support was added. Keep native NQ on the old proven route: only the
    // original NQ4_G64 batched Vulkan mat_mat is allowed; everything else falls
    // back to the old CPU oracle. This preserves today's standard-GGUF speed work
    // without letting it change the custom native Qwen path.
    if (is_native || force_native) {
        bool gpu_ok = false;
#ifdef NEXUS_VULKAN
        if (gpu::VulkanBackend* card = prefill_card_for_layer(layer_idx); card && is_native && wt == QuantType::NQ4_G64) {
            gpu_ok = card->mat_mat(W, X, Y, rows, cols, n_tokens, wt);
        }
#endif
        static std::atomic<int> native_mm_logs{0};
        const int log_id = native_mm_logs.fetch_add(1, std::memory_order_relaxed);
        if (log_id < 240) {
            fprintf(stderr,
                "[prefill:mm] L=%d family=native_nq_legacy %-28s %s rows=%zu cols=%zu ntok=%zu type=%d\n",
                layer_idx,
                tag ? tag : "?",
                gpu_ok ? "GPU" : "CPU",
                rows,
                cols,
                n_tokens,
                (int)wt);
            fflush(stderr);
        }
        if (!gpu_ok) {
            cpu::mat_mat(W, X, Y, rows, cols, n_tokens, wt, n_threads_);
        }
        return gpu_ok;
    }

    const char* family = forced_family ? forced_family : turbo_mm_family_name(wt);

    // Standard GGUF route: this is the new universal downloaded-model path.
    // It is deliberately separated from native_nq above.
    const bool family_matches = (!forced_family) || (force_std && is_std);

    bool gpu_mm_compatible = false;
    bool gpu_ok = false;
#ifdef NEXUS_VULKAN
    gpu::VulkanBackend* mm_card = prefill_card_for_layer(layer_idx);
    gpu_mm_compatible = family_matches && mm_card && mm_card->mat_mat_supported(wt);
    if (gpu_mm_compatible) {
        gpu_ok = mm_card->mat_mat(W, X, Y, rows, cols, n_tokens, wt);
    }
#else
    (void)is_std;
#endif

    static std::atomic<int> mm_logs{0};
    const int log_id = mm_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id < 240) {
        fprintf(stderr,
            "[prefill:mm] L=%d family=%s %-28s %s rows=%zu cols=%zu ntok=%zu type=%d\n",
            layer_idx,
            family,
            tag ? tag : "?",
            gpu_ok ? "GPU" : (gpu_mm_compatible ? "CPU-GPUFAIL" : "CPU-NOPIPE"),
            rows,
            cols,
            n_tokens,
            (int)wt);
        fflush(stderr);
    }

    if (!gpu_ok) {
        cpu::mat_mat(W, X, Y, rows, cols, n_tokens, wt, n_threads_);
    }
    return gpu_ok;
}

inline bool Transformer::turbo_mat_mat_or_cpu(const void* W, QuantType wt,
                                               const float* X, float* Y,
                                               size_t rows, size_t cols, size_t n_tokens,
                                               const char* tag, int layer_idx)
{
    return turbo_mat_mat_family_or_cpu(W, wt, X, Y, rows, cols, n_tokens,
                                       tag, layer_idx, nullptr);
}

inline void Transformer::forward_qwen35_linear_batch(const TransformerLayerWeights& lw,
                                                     const float* xb_batch,
                                                     float* out_batch,
                                                     int layer_idx,
                                                     int start_pos,
                                                     int n_tokens)
{
    const int hidden    = (int)meta_.hidden_dim;
    const int conv_dim  = lw.q35_conv_dim;
    const int key_dim   = lw.q35_key_dim;
    const int value_dim = lw.q35_value_dim;
    const int kh        = lw.q35_num_k_heads;
    const int vh        = lw.q35_num_v_heads;
    const int hd        = lw.q35_head_dim;
    const int kernel    = lw.q35_kernel;
    const int repeat    = (kh > 0) ? (vh / kh) : 0;

    if (!xb_batch || !out_batch || n_tokens <= 0 || hidden <= 0 ||
        conv_dim <= 0 || key_dim <= 0 || value_dim <= 0 ||
        kh <= 0 || vh <= 0 || hd <= 0 || repeat <= 0)
    {
        if (out_batch && hidden > 0 && n_tokens > 0)
            std::memset(out_batch, 0, (size_t)n_tokens * (size_t)hidden * sizeof(float));
        fprintf(stderr, "[turbo:q35] bad dims L=%d hidden=%d conv=%d key=%d value=%d kh=%d vh=%d hd=%d\n",
                layer_idx, hidden, conv_dim, key_dim, value_dim, kh, vh, hd);
        fflush(stderr);
        return;
    }

    if ((int)q35_conv_state_.size() < (int)meta_.n_layers) q35_conv_state_.resize(meta_.n_layers);
    if ((int)q35_recurrent_state_.size() < (int)meta_.n_layers) q35_recurrent_state_.resize(meta_.n_layers);

    q35_host_acquire(layer_idx);   // the card may hold newer state; the CPU path below reads and rewrites the host copy
    std::vector<float>& conv_state = q35_conv_state_[layer_idx];
    std::vector<float>& rec_state  = q35_recurrent_state_[layer_idx];

    const int conv_state_len = conv_dim * std::max(0, kernel - 1);
    const int rec_state_len  = vh * hd * hd;
    if ((int)conv_state.size() != conv_state_len) conv_state.assign((size_t)conv_state_len, 0.0f);
    if ((int)rec_state.size()  != rec_state_len)  rec_state.assign((size_t)rec_state_len,  0.0f);

    nx_view mixed(nx_scratch(q35s_mixed_, (size_t)n_tokens * (size_t)conv_dim));
    nx_view z    (nx_scratch(q35s_z_,     (size_t)n_tokens * (size_t)value_dim));
    nx_view a    (nx_scratch(q35s_a_,     (size_t)n_tokens * (size_t)vh));
    nx_view b    (nx_scratch(q35s_b_,     (size_t)n_tokens * (size_t)vh));

    turbo_mat_mat_or_cpu(lw.q35_qkv, lw.q35_qkv_type,
                         xb_batch, mixed.data(),
                         (size_t)conv_dim, (size_t)hidden, (size_t)n_tokens,
                         "linear_attn.in_proj_qkv", layer_idx);

    turbo_mat_mat_or_cpu(lw.q35_z, lw.q35_z_type,
                         xb_batch, z.data(),
                         (size_t)value_dim, (size_t)hidden, (size_t)n_tokens,
                         "linear_attn.in_proj_z", layer_idx);

    if (lw.q35_ba) {
        nx_view ba(nx_scratch(q35s_ba_, (size_t)n_tokens * (size_t)vh * 2u));
        turbo_mat_mat_or_cpu(lw.q35_ba, lw.q35_ba_type,
                             xb_batch, ba.data(),
                             (size_t)vh * 2u, (size_t)hidden, (size_t)n_tokens,
                             "linear_attn.ssm_beta_alpha", layer_idx);

        nx_par_range(n_tokens, n_threads_, [&](int t0, int t1) {
          for (int t = t0; t < t1; ++t) {
            const float* bat = ba.data() + (size_t)t * (size_t)vh * 2u;
            float* at = a.data() + (size_t)t * (size_t)vh;
            float* bt = b.data() + (size_t)t * (size_t)vh;
            for (int hk_i = 0; hk_i < kh; ++hk_i) {
                const size_t src = (size_t)hk_i * (size_t)(2 * repeat);
                const size_t dst = (size_t)hk_i * (size_t)repeat;
                for (int r = 0; r < repeat; ++r) {
                    bt[dst + (size_t)r] = bat[src + (size_t)r];
                    at[dst + (size_t)r] = bat[src + (size_t)repeat + (size_t)r];
                }
            }
          }
        });
    } else {
        turbo_mat_mat_or_cpu(lw.q35_a, lw.q35_a_type,
                             xb_batch, a.data(),
                             (size_t)vh, (size_t)hidden, (size_t)n_tokens,
                             "linear_attn.in_proj_a", layer_idx);
        turbo_mat_mat_or_cpu(lw.q35_b, lw.q35_b_type,
                             xb_batch, b.data(),
                             (size_t)vh, (size_t)hidden, (size_t)n_tokens,
                             "linear_attn.in_proj_b", layer_idx);
    }

    if (mtp_verify_all_logits_ && layer_idx >= 0 && layer_idx < (int)q35_vrec_.size()) {
        Q35VerifyRec& r = q35_vrec_[(size_t)layer_idx];
        r.mixed.insert(r.mixed.end(), mixed.data(), mixed.data() + (size_t)n_tokens * (size_t)conv_dim);
        r.a.insert(r.a.end(), a.data(), a.data() + (size_t)n_tokens * (size_t)vh);
        r.b.insert(r.b.end(), b.data(), b.data() + (size_t)n_tokens * (size_t)vh);
        r.n += n_tokens;
    }

    // Second overlap window: the conv + recurrent scan below is still CPU-side,
    // so use that time to make sure this layer's tail weights are resident.
    //
    // Next-layer *window advancement* is deliberately NOT triggered here.
    // That now happens exactly once per layer, in the outer loop, only after
    // layer l is fully processed and its own weights released -- the two-step
    // safety margin the staggered-eviction design depends on. Calling it here
    // too (as this function used to) fired it mid-layer, before l was even
    // done, which for GDN layers (most of this model) was the actual dominant
    // trigger regardless of the outer loop's timing. Layer l's own tail
    // weights are a separate concern and are still prefetched below.
    turbo_prefetch_current_tail_async(layer_idx);

    nx_view conv_out(nx_scratch(q35s_conv_, (size_t)n_tokens * (size_t)conv_dim));

    // Full-prompt depthwise causal conv. This used to be one of the obvious
    // CPU spikes in turbo prefill. Try the Qwen3Next Vulkan shader first;
    // fallback is the exact previous CPU loop.
    bool q35_conv_gpu_ok = false;
#ifdef NEXUS_VULKAN
    // A handful of tokens (an MTP check) is cheaper on the CPU than copying this layer's state to the card and back.
    if (turbo_gpu_stream_ && gpu::gpu().ready() && n_tokens >= 16) {
        q35_conv_gpu_ok = gpu::gpu().qwen35_conv1d(mixed.data(),
                                                   conv_state.data(),
                                                   lw.q35_conv,
                                                   conv_out.data(),
                                                   (uint32_t)n_tokens,
                                                   (uint32_t)conv_dim,
                                                   (uint32_t)kernel);
    }
#endif
    static std::atomic<int> q35_conv_logs{0};
    int q35_conv_log_id = q35_conv_logs.fetch_add(1, std::memory_order_relaxed);
    if (q35_conv_log_id < 80) {
        fprintf(stderr, "[turbo:q35-conv] L=%d %s ntok=%d conv_dim=%d kernel=%d\n",
                layer_idx, q35_conv_gpu_ok ? "GPU" : "CPU", n_tokens, conv_dim, kernel);
        fflush(stderr);
    }

    if (q35_conv_gpu_ok) {
        // Update recurrent conv state to match the old token-order loop:
        // state after the batch is the last (kernel-1) mixed vectors.
        if (kernel > 1) {
            const std::vector<float> old_state = conv_state;
            for (int c = 0; c < conv_dim; ++c) {
                float* st = conv_state.data() + (size_t)c * (size_t)(kernel - 1);
                for (int k = 0; k < kernel - 1; ++k) {
                    const int src_t = n_tokens + k - (kernel - 1);
                    if (src_t < 0) {
                        st[k] = old_state[(size_t)c * (size_t)(kernel - 1) + (size_t)(src_t + kernel - 1)];
                    } else {
                        st[k] = mixed[(size_t)src_t * (size_t)conv_dim + (size_t)c];
                    }
                }
            }
        }
    } else {
        // Exact old CPU path.
        for (int t = 0; t < n_tokens; ++t) {
            const float* mixt = mixed.data() + (size_t)t * (size_t)conv_dim;
            float* convt = conv_out.data() + (size_t)t * (size_t)conv_dim;

            for (int c = 0; c < conv_dim; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < kernel; ++k) {
                    float xv;
                    if (k < kernel - 1) xv = conv_state[(size_t)c * (size_t)(kernel - 1) + (size_t)k];
                    else                xv = mixt[c];
                    sum += xv * lw.q35_conv[(size_t)c * (size_t)kernel + (size_t)k];
                }
                convt[c] = sum / (1.0f + expf(-sum));
            }

            if (kernel > 1) {
                for (int c = 0; c < conv_dim; ++c) {
                    float* st = conv_state.data() + (size_t)c * (size_t)(kernel - 1);
                    for (int k = 0; k < kernel - 2; ++k) st[k] = st[k + 1];
                    st[kernel - 2] = mixt[c];
                }
            }
        }
    }

    const char* qkv_layout_env = std::getenv("NEXUS_QWEN35_LINEAR_QKV_LAYOUT");
    const bool qkv_grouped_layout = (qkv_layout_env && std::strcmp(qkv_layout_env, "grouped") == 0);
    const float scale = 1.0f / sqrtf((float)hd);

    nx_view core(nx_scratch(q35s_core_, (size_t)n_tokens * (size_t)value_dim));

    bool q35_scan_gpu_ok = false;
#ifdef NEXUS_VULKAN
    if (turbo_gpu_stream_ && gpu::gpu().ready() && hd <= 256 && n_tokens >= 16) {
        q35_scan_gpu_ok = gpu::gpu().qwen35_scan(conv_out.data(),
                                                 a.data(),
                                                 b.data(),
                                                 z.data(),
                                                 lw.q35_A,
                                                 lw.q35_dt,
                                                 lw.q35_norm,
                                                 rec_state.data(),
                                                 core.data(),
                                                 (uint32_t)n_tokens,
                                                 (uint32_t)conv_dim,
                                                 (uint32_t)key_dim,
                                                 (uint32_t)value_dim,
                                                 (uint32_t)kh,
                                                 (uint32_t)vh,
                                                 (uint32_t)hd,
                                                 (uint32_t)repeat,
                                                 qkv_grouped_layout,
                                                 meta_.rms_norm_eps);
    }
#endif
    static std::atomic<int> q35_scan_logs{0};
    int q35_scan_log_id = q35_scan_logs.fetch_add(1, std::memory_order_relaxed);
    if (q35_scan_log_id < 80) {
        fprintf(stderr, "[turbo:q35-scan] L=%d %s ntok=%d vh=%d hd=%d grouped=%d\n",
                layer_idx, q35_scan_gpu_ok ? "GPU" : "CPU", n_tokens, vh, hd, qkv_grouped_layout ? 1 : 0);
        fflush(stderr);
    }

    if (!q35_scan_gpu_ok) {
    std::vector<float> q_flat((size_t)key_dim, 0.0f);
    std::vector<float> k_flat((size_t)key_dim, 0.0f);
    std::vector<float> v_flat((size_t)value_dim, 0.0f);
    std::vector<float> q_normed((size_t)key_dim, 0.0f);
    std::vector<float> k_normed((size_t)key_dim, 0.0f);
    std::vector<float> q_scaled((size_t)hd, 0.0f);

    if (hd > 256) {
        fprintf(stderr, "[turbo:q35] hd=%d > 256 scratch not supported\n", hd);
        std::memset(out_batch, 0, (size_t)n_tokens * (size_t)hidden * sizeof(float));
        return;
    }

    for (int t = 0; t < n_tokens; ++t) {
        const float* convt = conv_out.data() + (size_t)t * (size_t)conv_dim;
        float* coret = core.data() + (size_t)t * (size_t)value_dim;

        const float* q_base = nullptr;
        const float* k_base = nullptr;
        const float* v_base = nullptr;

        if (qkv_grouped_layout) {
            const int group_span = 2 * hd + repeat * hd;
            for (int hk_i = 0; hk_i < kh; ++hk_i) {
                const size_t src = (size_t)hk_i * (size_t)group_span;
                std::memcpy(q_flat.data() + (size_t)hk_i * (size_t)hd,
                            convt + src,
                            (size_t)hd * sizeof(float));
                std::memcpy(k_flat.data() + (size_t)hk_i * (size_t)hd,
                            convt + src + (size_t)hd,
                            (size_t)hd * sizeof(float));
                for (int r = 0; r < repeat; ++r) {
                    const size_t dst_v = ((size_t)hk_i * (size_t)repeat + (size_t)r) * (size_t)hd;
                    const size_t src_v = src + (size_t)(2 * hd) + (size_t)r * (size_t)hd;
                    std::memcpy(v_flat.data() + dst_v, convt + src_v, (size_t)hd * sizeof(float));
                }
            }
            q_base = q_flat.data();
            k_base = k_flat.data();
            v_base = v_flat.data();
        } else {
            q_base = convt;
            k_base = q_base + key_dim;
            v_base = k_base + key_dim;
        }

        std::memcpy(q_normed.data(), q_base, (size_t)key_dim * sizeof(float));
        std::memcpy(k_normed.data(), k_base, (size_t)key_dim * sizeof(float));
        for (int hk_i = 0; hk_i < kh; ++hk_i) {
            nx_l2norm_inplace(q_normed.data() + (size_t)hk_i * (size_t)hd, hd, 1e-6f);
            nx_l2norm_inplace(k_normed.data() + (size_t)hk_i * (size_t)hd, hd, 1e-6f);
        }

        const float* at = a.data() + (size_t)t * (size_t)vh;
        const float* bt = b.data() + (size_t)t * (size_t)vh;
        const float* zt = z.data() + (size_t)t * (size_t)value_dim;

        for (int hv_i = 0; hv_i < vh; ++hv_i) {
            const int hk_i = hv_i / repeat;
            const float* q = q_normed.data() + (size_t)hk_i * (size_t)hd;
            const float* k = k_normed.data() + (size_t)hk_i * (size_t)hd;
            const float* v = v_base + (size_t)hv_i * (size_t)hd;

            for (int i = 0; i < hd; ++i) q_scaled[(size_t)i] = q[i] * scale;

            const float beta = nx_sigmoid_f32(bt[hv_i]);
            float g = lw.q35_A[hv_i] * nx_softplus_f32(at[hv_i] + lw.q35_dt[hv_i]);
            if (!std::isfinite(g)) g = -60.0f;
            if (g > 0.0f) g = 0.0f;
            if (g < -60.0f) g = -60.0f;
            const float decay = expf(g);

            float* S = rec_state.data() + (size_t)hv_i * (size_t)hd * (size_t)hd;
            for (int i = 0; i < hd * hd; ++i) S[i] *= decay;

            float kv_mem[256];
            for (int j = 0; j < hd; ++j) {
                float s = 0.0f;
                for (int i = 0; i < hd; ++i) s += k[i] * S[(size_t)i * (size_t)hd + (size_t)j];
                kv_mem[j] = s;
            }

            float delta[256];
            for (int j = 0; j < hd; ++j) delta[j] = (v[j] - kv_mem[j]) * beta;

            for (int i = 0; i < hd; ++i) {
                const float ki = k[i];
                float* Srow = S + (size_t)i * (size_t)hd;
                for (int j = 0; j < hd; ++j) Srow[j] += ki * delta[j];
            }

            float* dst = coret + (size_t)hv_i * (size_t)hd;
            for (int j = 0; j < hd; ++j) {
                float s = 0.0f;
                for (int i = 0; i < hd; ++i) s += q_scaled[(size_t)i] * S[(size_t)i * (size_t)hd + (size_t)j];
                dst[j] = s;
            }

            nx_qwen35_gated_rmsnorm(dst,
                                    zt + (size_t)hv_i * (size_t)hd,
                                    lw.q35_norm,
                                    hd,
                                    meta_.rms_norm_eps,
                                    false);
        }
    }

    }

    // q35_out/FFN weights were prefetched during the CPU conv+scan window.
    // If the upload is still finishing, wait here instead of letting mat_mat()
    // race the cache or duplicate the upload.
    turbo_prefetch_wait(turbo_prefetch_tail_future_);

    turbo_mat_mat_or_cpu(lw.q35_out, lw.q35_out_type,
                         core.data(), out_batch,
                         (size_t)hidden, (size_t)value_dim, (size_t)n_tokens,
                         "linear_attn.out_proj", layer_idx);
}

inline void Transformer::forward_dense_ffn_batch(const TransformerLayerWeights& lw,
                                                 const float* xb_batch,
                                                 float* out_batch,
                                                 int layer_idx,
                                                 int n_tokens)
{
    const int hidden = (int)meta_.hidden_dim;
    const int ffn = (int)meta_.ffn_dim;
    if (!xb_batch || !out_batch || n_tokens <= 0 || hidden <= 0 || ffn <= 0 || lw.moe) {
        if (out_batch && n_tokens > 0 && hidden > 0)
            std::memset(out_batch, 0, (size_t)n_tokens * (size_t)hidden * sizeof(float));
        return;
    }

    // Token-tiled: FFN is independent per token, so long prompts are processed
    // in slices. This caps the transient gate/up/gateup buffers (n * ffn * 4 bytes
    // each -- ~350 MB apiece at 5k tokens) that caused the RAM spikes/crashes.
    {
        constexpr int FFN_TOK_TILE = 512;
        if (n_tokens > FFN_TOK_TILE) {
            for (int t0 = 0; t0 < n_tokens; t0 += FFN_TOK_TILE) {
                const int cnt = std::min(FFN_TOK_TILE, n_tokens - t0);
                forward_dense_ffn_batch(lw,
                                        xb_batch  + (size_t)t0 * (size_t)hidden,
                                        out_batch + (size_t)t0 * (size_t)hidden,
                                        layer_idx, cnt);
            }
            return;
        }
    }

#ifdef NEXUS_VULKAN
    // Fused GPU FFN (gate, up, SiLU*up, down in one submit; intermediates stay in VRAM). NEXUS_FFN_NOFUSE=1 disables it.
    {
        static const bool no_fuse = std::getenv("NEXUS_FFN_NOFUSE") != nullptr;
        if (!no_fuse && !lw.fused_gate_up && !ffn_activation_gelu_ &&
            lw.wgate_type == QuantType::NQ4_G64 && lw.wup_type == QuantType::NQ4_G64 && lw.wdown_type == QuantType::NQ4_G64 &&
            lw.wgate && lw.wup && lw.wdown &&
            prefill_on_gpu_for_layer(layer_idx)) {
            if (prefill_card_for_layer(layer_idx)->ffn_prefill(lw.wgate, lw.wup, lw.wdown, xb_batch, out_batch,
                                       (size_t)ffn, (size_t)hidden, (size_t)n_tokens, QuantType::NQ4_G64))
                return;
        }
    }
#endif

    std::vector<float> gate((size_t)n_tokens * (size_t)ffn, 0.0f);
    std::vector<float> up  ((size_t)n_tokens * (size_t)ffn, 0.0f);

    if (lw.fused_gate_up) {
        std::vector<float> gateup((size_t)n_tokens * (size_t)ffn * 2u, 0.0f);
        turbo_mat_mat_or_cpu(lw.wup, lw.wup_type,
                             xb_batch, gateup.data(),
                             (size_t)ffn * 2u, (size_t)hidden, (size_t)n_tokens,
                             "ffn.fused_gate_up", layer_idx);
        for (int t = 0; t < n_tokens; ++t) {
            const float* src = gateup.data() + (size_t)t * (size_t)ffn * 2u;
            std::memcpy(gate.data() + (size_t)t * (size_t)ffn, src, (size_t)ffn * sizeof(float));
            std::memcpy(up.data()   + (size_t)t * (size_t)ffn, src + ffn, (size_t)ffn * sizeof(float));
        }
    } else {
        turbo_mat_mat_or_cpu(lw.wgate, lw.wgate_type,
                             xb_batch, gate.data(),
                             (size_t)ffn, (size_t)hidden, (size_t)n_tokens,
                             "ffn.gate", layer_idx);
        turbo_mat_mat_or_cpu(lw.wup, lw.wup_type,
                             xb_batch, up.data(),
                             (size_t)ffn, (size_t)hidden, (size_t)n_tokens,
                             "ffn.up", layer_idx);
    }

    for (int t = 0; t < n_tokens; ++t) {
        float* gt = gate.data() + (size_t)t * (size_t)ffn;
        float* ut = up.data()   + (size_t)t * (size_t)ffn;
        if (ffn_activation_gelu_) nx_gelu_tanh_inplace(gt, (size_t)ffn);
        else                     cpu::silu(gt, (size_t)ffn);
        cpu::mul(gt, ut, gt, (size_t)ffn);
    }

    turbo_mat_mat_or_cpu(lw.wdown, lw.wdown_type,
                         gate.data(), out_batch,
                         (size_t)hidden, (size_t)ffn, (size_t)n_tokens,
                         "ffn.down", layer_idx);
}

inline bool Transformer::forward_qwen35_prefill_layer_batch(const TransformerLayerWeights& lw,
                                                            float* x_batch,
                                                            int layer_idx,
                                                            int start_pos,
                                                            int n_tokens)
{
    const int hidden = (int)meta_.hidden_dim;
    if (!lw.qwen35_linear || lw.moe || !x_batch || n_tokens <= 1 || hidden <= 0)
        return false;

    // IMPORTANT: turbo prefill is layer-major over the FULL prompt.
    // Do not split the Transformer-level layer pass into prompt chunks here.
    // The whole point is: load/keep layer L on GPU once, process every prompt
    // token through that layer, then move to layer L+1.
    //
    // The Vulkan mat_mat backend may internally dispatch TOK_TILE-sized shader
    // chunks, but that does not change this layer-major contract: the same
    // weight buffer stays resident/reused while all n_tokens are processed.

    nx_view xb_batch  (nx_scratch(pf_xb_,   (size_t)n_tokens * (size_t)hidden));
    nx_view attn_batch(nx_scratch(pf_attn_, (size_t)n_tokens * (size_t)hidden));
    nx_view ffn_batch (nx_scratch(pf_ffn_,  (size_t)n_tokens * (size_t)hidden));
    const bool glue_prof = gpu::gpu().prof_enabled();
    double glue_ms = 0;
    auto glue_t0 = std::chrono::steady_clock::now();
    auto glue_lap = [&]() { if (!glue_prof) return; const auto n = std::chrono::steady_clock::now(); glue_ms += std::chrono::duration<double, std::milli>(n - glue_t0).count(); glue_t0 = n; };
    auto glue_skip = [&]() { if (glue_prof) glue_t0 = std::chrono::steady_clock::now(); };

    nx_par_range(n_tokens, n_threads_, [&](int a, int b) {
        for (int t = a; t < b; ++t) {
            float* xt = x_batch + (size_t)t * (size_t)hidden;
            float* xb = xb_batch.data() + (size_t)t * (size_t)hidden;
            nx_apply_model_norm(xt, lw.attn_norm, lw.attn_norm_bias, xb,
                                (size_t)hidden, meta_.rms_norm_eps,
                                lw.attn_norm_zc);
        }
    });
    glue_lap();

    glue_skip();
    forward_qwen35_linear_batch(lw, xb_batch.data(), attn_batch.data(),
                                layer_idx, start_pos, n_tokens);
    glue_skip();

    nx_par_range(n_tokens, n_threads_, [&](int a, int b) {
        for (int t = a; t < b; ++t) {
            float* xt = x_batch + (size_t)t * (size_t)hidden;
            const float* at = attn_batch.data() + (size_t)t * (size_t)hidden;
            cpu::add_inplace(xt, at, (size_t)hidden);
            float* xb = xb_batch.data() + (size_t)t * (size_t)hidden;
            nx_apply_model_norm(xt, lw.ffn_norm, lw.ffn_norm_bias, xb,
                                (size_t)hidden, meta_.rms_norm_eps,
                                lw.ffn_norm_zc);
        }
    });
    glue_lap();

    forward_dense_ffn_batch(lw, xb_batch.data(), ffn_batch.data(),
                            layer_idx, n_tokens);
    glue_skip();

    nx_par_range(n_tokens, n_threads_, [&](int a, int b) {
        for (int t = a; t < b; ++t) {
            float* xt = x_batch + (size_t)t * (size_t)hidden;
            const float* ft = ffn_batch.data() + (size_t)t * (size_t)hidden;
            cpu::add_inplace(xt, ft, (size_t)hidden);
        }
    });
    glue_lap();
    if (glue_prof) { fprintf(stderr, "[turbo:glue] L=%d layer-level CPU (norms + residual adds) %.0f ms\n", layer_idx, glue_ms); fflush(stderr); }

    static std::atomic<int> layer_logs{0};
    const int log_id = layer_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id < 96) {
        fprintf(stderr,
            "[prefill:q35-layer] L=%d family=native_nq batched ntok=%d pos=%d..%d\n",
            layer_idx,
            n_tokens,
            start_pos,
            start_pos + n_tokens - 1);
        fflush(stderr);
    }
    return true;
}

// ---- NEXUS_GPU_VERIFY=1 : check every GPU result of the full-attention prefill against the CPU and print the difference ----
// Prints one line per operation, e.g.  [gpu-verify] L=3 wq(q+gate) rows=8192 maxerr=1.2e-06 OK   or   ... maxerr=3.4e+00 WRONG (first bad at #5120)
// Only for diagnosis (it repeats the work on the CPU).  Does nothing unless the environment variable is set.
static inline bool nx_gpu_verify_on() {
    static const bool on = [] {
        if (std::getenv("NEXUS_GPU_VERIFY")) return true;
#ifdef __linux__
        // also accepted as a plain command-line word (e.g. typed into the launcher's extra-arguments box)
        if (FILE* f = fopen("/proc/self/cmdline", "rb")) {
            std::string c; char b[4096]; size_t n;
            while ((n = fread(b, 1, sizeof b, f)) > 0) c.append(b, n);
            fclose(f);
            if (c.find("NEXUS_GPU_VERIFY") != std::string::npos) return true;
        }
#endif
        return false;
    }();
    return on;
}
static inline void nx_gpu_verify_report(const char* what, int layer, const float* gpu, const float* cpu_ref, size_t n, int rows) {
    double mx = 1e-6; for (size_t i = 0; i < n; ++i) mx = std::max(mx, (double)std::fabs(cpu_ref[i]));
    double worst = 0; size_t first = (size_t)-1, nbad = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::isfinite(gpu[i]) ? std::fabs((double)gpu[i] - (double)cpu_ref[i]) / mx : 1e30;
        if (e > worst) worst = e;
        if (e > 2e-3) { ++nbad; if (first == (size_t)-1) first = i; }
    }
    if (nbad) fprintf(stderr, "[gpu-verify] L=%d %-14s rows=%d  maxerr=%.2e  WRONG: %zu of %zu values differ (first at #%zu = row %zu)\n", layer, what, rows, worst, nbad, n, first, rows > 0 ? (first % (size_t)rows) : first);
    else       fprintf(stderr, "[gpu-verify] L=%d %-14s rows=%d  maxerr=%.2e  OK\n", layer, what, rows, worst);
    fflush(stderr);
}

inline bool Transformer::forward_full_attn_prefill_layer_batch_native_legacy(
    const TransformerLayerWeights& lw,
    float* x_batch, int layer_idx, int start_pos, int n_tokens)
{
    const int hidden = (int)meta_.hidden_dim;
    if (!x_batch || n_tokens <= 1 || hidden <= 0) return false;

    // Plain + Qwen3.5 gated softmax attention only. Exotic paths -> per-token.
    if (lw.qwen35_linear || lw.moe) return false;
    if (lw.kv_donor_layer >= 0)     return false;   // Gemma4 shared KV
    if (lw.gemma4_tied_kv)          return false;
    // Gate form we handle: query+gate packed into wq (qwen35_full_attn_q_has_gate).
    // A separate attn_gate weight WITHOUT the packed form is a different layout
    // we don't batch yet -> per-token.
    if (lw.attn_gate && !lw.qwen35_full_attn_q_has_gate) return false;

    // GPU mat_mat implements only NQ4_G64 (type 1004) — what this model uses.
    auto mat_mat_ok = [](QuantType t){ return t == QuantType::NQ4_G64; };
    bool proj_ok;
    if (lw.fused_qkv) {
        proj_ok = mat_mat_ok(lw.wqkv_type);
    } else {
        // wq (packed q+gate when gated), wk, wv all go through GPU mat_mat now.
        proj_ok = mat_mat_ok(lw.wq_type) && mat_mat_ok(lw.wk_type) && mat_mat_ok(lw.wv_type);
    }
    if (!proj_ok)            return false;
    if (!mat_mat_ok(lw.wo_type)) return false;

    // No hard GPU requirement: this batched path runs on CPU too. Every GPU
    // mat_mat below falls back to cpu::mat_mat (one weight pass for all n tokens),
    // and the attention has attn_batch_cpu fallback. This lets the MTP verify
    // batch the full-attention layers on CPU instead of n per-token passes.
    // The card this layer lives on (every layer while a prompt is streamed); none for a layer in RAM. Before, this always
    // used the prompt card: an MTP check uploaded every full-attention layer's weights AND its whole conversation memory
    // to it, on every check -- gigabytes over the bus, on a card that is already full.
    gpu::VulkanBackend* gcard = prefill_card_for_layer(layer_idx);
    gpu::VulkanBackend& g = gcard ? *gcard : gpu::gpu();
    auto gmm = [&](auto&&... args) -> bool { return gcard && g.mat_mat(std::forward<decltype(args)>(args)...); };
    // Attention for a handful of tokens (an MTP check) reads the conversation memory where it already is, on the CPU,
    // instead of copying all of it to the card for 4 queries. A prompt (a big batch) still uses the card.
    const bool attn_on_card = gcard && n_tokens >= 16;

    // Per-layer attention config (mirror forward_layer).
    AttentionConfig cfg = attn_cfg_;
    if (lw.attn_heads_layer > 0)    cfg.n_heads    = lw.attn_heads_layer;
    if (lw.attn_kv_heads_layer > 0) cfg.n_kv_heads = lw.attn_kv_heads_layer;
    if (lw.attn_head_dim_layer > 0) cfg.head_dim   = lw.attn_head_dim_layer;
    if (lw.sw_window_size > 0)       cfg.sliding_window = lw.sw_window_size;
    if (lw.rope_freq_base_layer > 0) cfg.rope.freq_base = lw.rope_freq_base_layer;

    const int n_heads  = cfg.n_heads;
    const int n_kv     = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;
    const int q_dim    = n_heads * head_dim;
    const int kv_dim   = n_kv    * head_dim;
    const float eps    = meta_.rms_norm_eps;
    const float scale  = (cfg.scale != 0.0f) ? cfg.scale : 1.0f/std::sqrt((float)head_dim);
    const float softcap = cfg.attn_softcap;
    const int n_ctx    = start_pos + n_tokens;

    // Gate values come from the wq projection's second half (perhead split),
    // NOT from a separate attn_gate weight. Matches the per-token gate path.
    const bool has_gate          = lw.qwen35_full_attn_q_has_gate;
    const bool qwen35_partial    = lw.qwen35_full_attn_q_has_gate && !lw.qwen35_linear;

    std::vector<float> xb ((size_t)n_tokens * hidden);
    std::vector<float> Qb ((size_t)n_tokens * q_dim);
    std::vector<float> Kb ((size_t)n_tokens * kv_dim);
    std::vector<float> Vb ((size_t)n_tokens * kv_dim);
    std::vector<float> Gate;                              // [n_tokens × q_dim] if gated
    std::vector<float> Ao ((size_t)n_tokens * q_dim);
    std::vector<float> Wp ((size_t)n_tokens * hidden);
    std::vector<float> Fb ((size_t)n_tokens * hidden);
    if (has_gate) Gate.assign((size_t)n_tokens * q_dim, 0.0f);

    const bool dg_prof = gpu::gpu().prof_enabled();
    double dg_ms[8] = {0,0,0,0,0,0,0,0};
    auto dg_t0 = std::chrono::steady_clock::now();
    auto dg_lap = [&](int i) { if (!dg_prof) return; const auto n = std::chrono::steady_clock::now(); dg_ms[i] += std::chrono::duration<double, std::milli>(n - dg_t0).count(); dg_t0 = n; };
    auto dg_skip = [&]() { if (dg_prof) dg_t0 = std::chrono::steady_clock::now(); };
    // 1) RMS norm (attn) for all tokens. Keep normed input for the gate proj too.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            nx_apply_model_norm(x_batch + (size_t)t*hidden, lw.attn_norm, lw.attn_norm_bias,
                                xb.data() + (size_t)t*hidden, (size_t)hidden, eps, lw.attn_norm_zc);
    });

    dg_lap(0);
    // 2) Projections (batched GPU mat_mat). All before any KV store.
    if (lw.fused_qkv) {
        const int qkv = q_dim + 2*kv_dim;
        std::vector<float> QKV((size_t)n_tokens * qkv);
        if (!gmm(lw.wqkv, xb.data(), QKV.data(), qkv, hidden, n_tokens, lw.wqkv_type))
            cpu::mat_mat(lw.wqkv, xb.data(), QKV.data(), qkv, hidden, n_tokens, lw.wqkv_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(qkv) * n_tokens);
            cpu::mat_mat(lw.wqkv, xb.data(), ref_.data(), qkv, hidden, n_tokens, lw.wqkv_type, n_threads_);
            nx_gpu_verify_report("wqkv", layer_idx, QKV.data(), ref_.data(), ref_.size(), (int)(qkv));
        }
        for (int t = 0; t < n_tokens; ++t) {
            const float* s = QKV.data() + (size_t)t*qkv;
            memcpy(Qb.data()+(size_t)t*q_dim,  s,              q_dim*sizeof(float));
            memcpy(Kb.data()+(size_t)t*kv_dim, s+q_dim,        kv_dim*sizeof(float));
            memcpy(Vb.data()+(size_t)t*kv_dim, s+q_dim+kv_dim, kv_dim*sizeof(float));
        }
    } else if (has_gate) {
        // wq is the packed [q_dim*2, hidden] query+gate tensor, now pinned at the
        // packed size by the resident loader, so the batched GPU mat_mat reads it
        // in-bounds. This moves the big Q+gate GEMM off the CPU (it was the ~7s
        // per-layer spike). On failure, fall back to CPU for just this projection.
        const int twoq = q_dim * 2;
        std::vector<float> QG((size_t)n_tokens * twoq);
        if (!gmm(lw.wq, xb.data(), QG.data(), twoq, hidden, n_tokens, lw.wq_type))
            cpu::mat_mat(lw.wq, xb.data(), QG.data(), twoq, hidden, n_tokens, lw.wq_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(twoq) * n_tokens);
            cpu::mat_mat(lw.wq, xb.data(), ref_.data(), twoq, hidden, n_tokens, lw.wq_type, n_threads_);
            nx_gpu_verify_report("wq", layer_idx, QG.data(), ref_.data(), ref_.size(), (int)(twoq));
        }
        nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
          for (int t = a_; t < b_; ++t) {
            const float* src = QG.data() + (size_t)t*twoq;
            float* qd = Qb.data()   + (size_t)t*q_dim;
            float* gd = Gate.data() + (size_t)t*q_dim;
            for (int h = 0; h < n_heads; ++h) {
                memcpy(qd + (size_t)h*head_dim, src + (size_t)h*2*head_dim,            head_dim*sizeof(float));
                memcpy(gd + (size_t)h*head_dim, src + (size_t)h*2*head_dim + head_dim, head_dim*sizeof(float));
            }
          }
        });
        if (!gmm(lw.wk, xb.data(), Kb.data(), kv_dim, hidden, n_tokens, lw.wk_type))
            cpu::mat_mat(lw.wk, xb.data(), Kb.data(), kv_dim, hidden, n_tokens, lw.wk_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(kv_dim) * n_tokens);
            cpu::mat_mat(lw.wk, xb.data(), ref_.data(), kv_dim, hidden, n_tokens, lw.wk_type, n_threads_);
            nx_gpu_verify_report("wk", layer_idx, Kb.data(), ref_.data(), ref_.size(), (int)(kv_dim));
        }
        if (!gmm(lw.wv, xb.data(), Vb.data(), kv_dim, hidden, n_tokens, lw.wv_type))
            cpu::mat_mat(lw.wv, xb.data(), Vb.data(), kv_dim, hidden, n_tokens, lw.wv_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(kv_dim) * n_tokens);
            cpu::mat_mat(lw.wv, xb.data(), ref_.data(), kv_dim, hidden, n_tokens, lw.wv_type, n_threads_);
            nx_gpu_verify_report("wv", layer_idx, Vb.data(), ref_.data(), ref_.size(), (int)(kv_dim));
        }
    } else {
        if (!gmm(lw.wq, xb.data(), Qb.data(), q_dim,  hidden, n_tokens, lw.wq_type))
            cpu::mat_mat(lw.wq, xb.data(), Qb.data(), q_dim,  hidden, n_tokens, lw.wq_type, n_threads_);
        if (!gmm(lw.wk, xb.data(), Kb.data(), kv_dim, hidden, n_tokens, lw.wk_type))
            cpu::mat_mat(lw.wk, xb.data(), Kb.data(), kv_dim, hidden, n_tokens, lw.wk_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(kv_dim) * n_tokens);
            cpu::mat_mat(lw.wk, xb.data(), ref_.data(), kv_dim, hidden, n_tokens, lw.wk_type, n_threads_);
            nx_gpu_verify_report("wk", layer_idx, Kb.data(), ref_.data(), ref_.size(), (int)(kv_dim));
        }
        if (!gmm(lw.wv, xb.data(), Vb.data(), kv_dim, hidden, n_tokens, lw.wv_type))
            cpu::mat_mat(lw.wv, xb.data(), Vb.data(), kv_dim, hidden, n_tokens, lw.wv_type, n_threads_);
        else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
            std::vector<float> ref_((size_t)(kv_dim) * n_tokens);
            cpu::mat_mat(lw.wv, xb.data(), ref_.data(), kv_dim, hidden, n_tokens, lw.wv_type, n_threads_);
            nx_gpu_verify_report("wv", layer_idx, Vb.data(), ref_.data(), ref_.size(), (int)(kv_dim));
        }
    }

    dg_skip();   // (projection time is accounted by the mat_mat profile; the Q/gate split above is counted as 'split')
    // 3) Per token: bias + q/k norm, partial/full RoPE on Q and K, store KV.
    //    Identical helper calls to the per-token path -> identical numbers.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
      for (int t = a_; t < b_; ++t) {
        const int pos = start_pos + t;
        const int rope_pos = rope_effective_position(pos);
        float* q = Qb.data() + (size_t)t*q_dim;
        float* k = Kb.data() + (size_t)t*kv_dim;
        float* v = Vb.data() + (size_t)t*kv_dim;

        nx_apply_qkv_extras(lw, cfg, q, k, v, q_dim, kv_dim, eps);
        nx_apply_rope_qwen35_partial_or_full(q, n_heads, rope_pos, head_dim, cfg.rope, qwen35_partial);
        nx_apply_rope_qwen35_partial_or_full(k, n_kv,    rope_pos, head_dim, cfg.rope, qwen35_partial);
      }
    });
    dg_lap(1);
    for (int t = 0; t < n_tokens; ++t)      // KV store stays serial
        kvcache_.store(layer_idx, start_pos + t, Kb.data() + (size_t)t*kv_dim, Vb.data() + (size_t)t*kv_dim, kv_dim);

    dg_lap(2);
    // 4) Pack the full KV for this layer ONCE.
    const size_t kv_count = (size_t)n_kv * n_ctx * head_dim;
    kv_flat_k_.resize(kv_count);
    kv_flat_v_.resize(kv_count);
    kvcache_.pack_k(layer_idx, n_ctx, kv_flat_k_.data());
    kvcache_.pack_v(layer_idx, n_ctx, kv_flat_v_.data());

    dg_lap(3);
    // 5) Attention over all N query tokens. Two GPU paths, both full-causal
    //    (matching forward_cached_scores): the new streaming path (KV in chunks,
    //    GPU never holds full KV) and the previous single-shot batched path.
    //    NEXUS_ATTN_PATH=batch forces the proven batched path for A/B diagnosis;
    //    default is streaming.
    bool ok = false;
    const char* attn_path = std::getenv("NEXUS_ATTN_PATH");

    // Short-prompt turbo bugfix:
    // Keep prefill on GPU, but do NOT use the chunked streaming attention kernel
    // for tiny prompts by default. The short 48-token case that produced skewed
    // input runs through the same high-level turbo path as long prompts, but the
    // streaming attention shader is a different GPU kernel from the older batch
    // attention path. For small n_ctx/n_query, the batch GPU path is cheaper,
    // simpler, and avoids the chunk-state/finalize path entirely.
    //
    // Overrides:
    //   NEXUS_ATTN_PATH=stream  -> force streaming GPU attention
    //   NEXUS_ATTN_PATH=batch   -> force batch GPU attention
    //   NEXUS_SHORT_ATTN_BATCH_TOKENS=N -> auto-batch when n_ctx or n_query <= N
    int short_attn_threshold = 128;
    if (const char* e = std::getenv("NEXUS_SHORT_ATTN_BATCH_TOKENS")) {
        long v = std::atol(e);
        if (v >= 0 && v <= 8192) short_attn_threshold = (int)v;
    }
    const bool force_batch  = (attn_path && (attn_path[0]=='b' || attn_path[0]=='B'));
    const bool force_stream = (attn_path && (attn_path[0]=='s' || attn_path[0]=='S'));
    const bool auto_short_batch = (!force_stream && short_attn_threshold > 0 &&
                                  (n_ctx <= short_attn_threshold || n_tokens <= short_attn_threshold));
    const bool use_batched = force_batch || auto_short_batch;

    const char* attn_used = use_batched ? "batch" : "stream";
    bool flash_ok = false;
    {   // the tiled kernel wins over NEXUS_ATTN_PATH (launchers set it to 'stream'); NEXUS_ATTN_NOFLASH=1 disables it
        flash_ok = attn_on_card && g.attention_flash_gpu(
            Qb.data(), Ao.data(),
            n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos, /*sliding_window=*/0,
            scale, softcap,
            kv_flat_k_.data(), kv_flat_v_.data());
        if (flash_ok) { ok = true; attn_used = "flash"; }
    }
    if (flash_ok) {
        // done: tiled flash attention (K/V uploaded once per layer)
    } else if (use_batched) {
        ok = attn_on_card && g.attention_batch_gpu(
            Qb.data(), Ao.data(),
            n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos, /*sliding_window=*/0,
            scale, softcap,
            kv_flat_k_.data(), kv_flat_v_.data(), kv_count * sizeof(uint16_t));
    } else {
        ok = attn_on_card && g.attention_stream_gpu(
            Qb.data(), Ao.data(),
            n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos,
            scale, softcap,
            kv_flat_k_.data(), kv_flat_v_.data(), /*chunk_tokens=*/0);
    }
    if (ok && nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: same attention on the CPU, compare
        std::vector<float> ref_((size_t)n_tokens * q_dim);
        attn_batch_cpu(Qb.data(), kv_flat_k_.data(), kv_flat_v_.data(), ref_.data(),
                       n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos,
                       /*sliding_window=*/0, scale, softcap);
        nx_gpu_verify_report(use_batched ? "attn(batch)" : "attn(stream)", layer_idx, Ao.data(), ref_.data(), ref_.size(), q_dim);
    }
    if (!ok) {
        attn_used = "cpu-fallback";
        attn_batch_cpu(Qb.data(), kv_flat_k_.data(), kv_flat_v_.data(), Ao.data(),
                       n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos,
                       /*sliding_window=*/0, scale, softcap);
    }

    dg_skip();
    // 5b) POST-gate: attn_out *= sigmoid(gate), per token. Matches per-token path.
    if (has_gate) {
        nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
          for (int t = a_; t < b_; ++t) {
            float* ao = Ao.data()   + (size_t)t*q_dim;
            const float* gt = Gate.data() + (size_t)t*q_dim;
            for (int i = 0; i < q_dim; ++i) {
                float z = gt[i];
                float s = (z > 80.0f) ? 1.0f : (z < -80.0f) ? 0.0f : 1.0f/(1.0f+expf(-z));
                ao[i] *= s;
            }
          }
        });
    }

    dg_lap(4);
    // 6) Wo projection + bias + residual.
    if (!gmm(lw.wo, Ao.data(), Wp.data(), hidden, q_dim, n_tokens, lw.wo_type))
        cpu::mat_mat(lw.wo, Ao.data(), Wp.data(), hidden, q_dim, n_tokens, lw.wo_type, n_threads_);
    else if (nx_gpu_verify_on()) {      // NEXUS_GPU_VERIFY: recompute on the CPU and compare
        std::vector<float> ref_((size_t)(hidden) * n_tokens);
        cpu::mat_mat(lw.wo, Ao.data(), ref_.data(), hidden, q_dim, n_tokens, lw.wo_type, n_threads_);
        nx_gpu_verify_report("wo", layer_idx, Wp.data(), ref_.data(), ref_.size(), (int)(hidden));
    }
    dg_skip();
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t) {
            if (lw.bo) nx_add_bias_inplace(Wp.data()+(size_t)t*hidden, lw.bo, hidden);
            cpu::add_inplace(x_batch+(size_t)t*hidden, Wp.data()+(size_t)t*hidden, hidden);
        }
    });

    dg_lap(7);
    // 7) FFN norm + batched FFN + residual.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            nx_apply_model_norm(x_batch+(size_t)t*hidden, lw.ffn_norm, lw.ffn_norm_bias,
                                xb.data()+(size_t)t*hidden, (size_t)hidden, eps, lw.ffn_norm_zc);
    });
    dg_lap(5);
    forward_dense_ffn_batch(lw, xb.data(), Fb.data(), layer_idx, n_tokens);
    dg_skip();
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            cpu::add_inplace(x_batch+(size_t)t*hidden, Fb.data()+(size_t)t*hidden, hidden);
    });

    dg_lap(6);
    if (dg_prof) { fprintf(stderr, "[turbo:glue-dense] L=%d norm %.0f | rope/extras %.0f | kv-store %.0f | kv-pack %.0f | gate %.0f | wo-bias+add %.0f | ffn-norm %.0f | ffn-add %.0f ms\n", layer_idx, dg_ms[0], dg_ms[1], dg_ms[2], dg_ms[3], dg_ms[4], dg_ms[7], dg_ms[5], dg_ms[6]); fflush(stderr); }
    static std::atomic<int> fa_logs{0};
    if (fa_logs.fetch_add(1, std::memory_order_relaxed) < 96) {
        fprintf(stderr, "[turbo:full-attn-native-legacy] L=%d batched ntok=%d pos=%d..%d hd=%d gate=%d prope=%d gpu_attn=%d attn_path=%s\n",
                layer_idx, n_tokens, start_pos, start_pos+n_tokens-1, head_dim,
                has_gate?1:0, qwen35_partial?1:0, ok?1:0, attn_used);
        fflush(stderr);
    }
    return true;
}


inline bool Transformer::forward_full_attn_prefill_layer_batch(
    const TransformerLayerWeights& lw,
    float* x_batch, int layer_idx, int start_pos, int n_tokens)
{
    const int hidden = (int)meta_.hidden_dim;
    if (!x_batch || n_tokens <= 1 || hidden <= 0) return false;

    // Plain + Qwen3.5 gated softmax attention only. Exotic paths -> per-token.
    if (lw.qwen35_linear || lw.moe) return false;
    if (lw.kv_donor_layer >= 0)     return false;   // Gemma4 shared KV
    if (lw.gemma4_tied_kv)          return false;
    // Gate form we handle: query+gate packed into wq (qwen35_full_attn_q_has_gate).
    // A separate attn_gate weight WITHOUT the packed form is a different layout
    // we don't batch yet -> per-token.
    if (lw.attn_gate && !lw.qwen35_full_attn_q_has_gate) return false;

    // Hard family isolation. Native Nexus NQ models (your .nxm Qwen path) and
    // downloaded GGUF K-quants are both allowed to use batched prefill, but they
    // must not share a mixed/generic projection route. A layer is either native_nq
    // or standard_gguf. Mixed/unknown projection types fall back to the old
    // per-token path instead of risking silent corruption.
    auto all_native = [](std::initializer_list<QuantType> ts) {
        for (QuantType t : ts) if (!Transformer::turbo_is_native_nq_type(t)) return false;
        return true;
    };
    auto all_standard = [](std::initializer_list<QuantType> ts) {
        for (QuantType t : ts) if (!Transformer::turbo_is_standard_gguf_mm_type(t)) return false;
        return true;
    };

    bool proj_native = false;
    bool proj_standard = false;
    if (lw.fused_qkv) {
        proj_native   = all_native({lw.wqkv_type, lw.wo_type});
        proj_standard = all_standard({lw.wqkv_type, lw.wo_type});
    } else {
        proj_native   = all_native({lw.wq_type, lw.wk_type, lw.wv_type, lw.wo_type});
        proj_standard = all_standard({lw.wq_type, lw.wk_type, lw.wv_type, lw.wo_type});
    }
    if (proj_native == proj_standard) return false;
    const char* mm_family = proj_native ? "native_nq" : "standard_gguf";

    // No hard GPU requirement: this batched path runs on CPU too. Every GPU
    // mat_mat below falls back to cpu::mat_mat (one weight pass for all n tokens),
    // and the attention has attn_batch_cpu fallback. This lets the MTP verify
    // batch the full-attention layers on CPU instead of n per-token passes.
    gpu::VulkanBackend* gcard = prefill_card_for_layer(layer_idx);   // see the legacy path above
    gpu::VulkanBackend& g = gcard ? *gcard : gpu::gpu();
    const bool attn_on_card = gcard && n_tokens >= 16;

    // Per-layer attention config (mirror forward_layer).
    AttentionConfig cfg = attn_cfg_;
    if (lw.attn_heads_layer > 0)    cfg.n_heads    = lw.attn_heads_layer;
    if (lw.attn_kv_heads_layer > 0) cfg.n_kv_heads = lw.attn_kv_heads_layer;
    if (lw.attn_head_dim_layer > 0) cfg.head_dim   = lw.attn_head_dim_layer;
    if (lw.sw_window_size > 0)       cfg.sliding_window = lw.sw_window_size;
    if (lw.rope_freq_base_layer > 0) cfg.rope.freq_base = lw.rope_freq_base_layer;

    const int n_heads  = cfg.n_heads;
    const int n_kv     = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;
    const int q_dim    = n_heads * head_dim;
    const int kv_dim   = n_kv    * head_dim;
    const float eps    = meta_.rms_norm_eps;
    const float scale  = (cfg.scale != 0.0f) ? cfg.scale : 1.0f/std::sqrt((float)head_dim);
    const float softcap = cfg.attn_softcap;
    const int n_ctx    = start_pos + n_tokens;

    // Gate values come from the wq projection's second half (perhead split),
    // NOT from a separate attn_gate weight. Matches the per-token gate path.
    const bool has_gate          = lw.qwen35_full_attn_q_has_gate;
    const bool qwen35_partial    = lw.qwen35_full_attn_q_has_gate && !lw.qwen35_linear;

    std::vector<float> xb ((size_t)n_tokens * hidden);
    std::vector<float> Qb ((size_t)n_tokens * q_dim);
    std::vector<float> Kb ((size_t)n_tokens * kv_dim);
    std::vector<float> Vb ((size_t)n_tokens * kv_dim);
    std::vector<float> Gate;                              // [n_tokens × q_dim] if gated
    std::vector<float> Ao ((size_t)n_tokens * q_dim);
    std::vector<float> Wp ((size_t)n_tokens * hidden);
    std::vector<float> Fb ((size_t)n_tokens * hidden);
    if (has_gate) Gate.assign((size_t)n_tokens * q_dim, 0.0f);

    // 1) RMS norm (attn) for all tokens. Keep normed input for the gate proj too.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            nx_apply_model_norm(x_batch + (size_t)t*hidden, lw.attn_norm, lw.attn_norm_bias,
                                xb.data() + (size_t)t*hidden, (size_t)hidden, eps, lw.attn_norm_zc);
    });

    // 2) Projections (batched GPU mat_mat). All before any KV store.
    if (lw.fused_qkv) {
        const int qkv = q_dim + 2*kv_dim;
        std::vector<float> QKV((size_t)n_tokens * qkv);
        turbo_mat_mat_family_or_cpu(lw.wqkv, lw.wqkv_type,
                                     xb.data(), QKV.data(),
                                     (size_t)qkv, (size_t)hidden, (size_t)n_tokens,
                                     "attn.qkv", layer_idx, mm_family);
        for (int t = 0; t < n_tokens; ++t) {
            const float* s = QKV.data() + (size_t)t*qkv;
            memcpy(Qb.data()+(size_t)t*q_dim,  s,              q_dim*sizeof(float));
            memcpy(Kb.data()+(size_t)t*kv_dim, s+q_dim,        kv_dim*sizeof(float));
            memcpy(Vb.data()+(size_t)t*kv_dim, s+q_dim+kv_dim, kv_dim*sizeof(float));
        }
    } else if (has_gate) {
        // wq is the packed [q_dim*2, hidden] query+gate tensor, now pinned at the
        // packed size by the resident loader, so the batched GPU mat_mat reads it
        // in-bounds. This moves the big Q+gate GEMM off the CPU (it was the ~7s
        // per-layer spike). On failure, fall back to CPU for just this projection.
        const int twoq = q_dim * 2;
        std::vector<float> QG((size_t)n_tokens * twoq);
        turbo_mat_mat_family_or_cpu(lw.wq, lw.wq_type,
                                     xb.data(), QG.data(),
                                     (size_t)twoq, (size_t)hidden, (size_t)n_tokens,
                                     "attn.q_gate", layer_idx, mm_family);
        nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
          for (int t = a_; t < b_; ++t) {
            const float* src = QG.data() + (size_t)t*twoq;
            float* qd = Qb.data()   + (size_t)t*q_dim;
            float* gd = Gate.data() + (size_t)t*q_dim;
            for (int h = 0; h < n_heads; ++h) {
                memcpy(qd + (size_t)h*head_dim, src + (size_t)h*2*head_dim,            head_dim*sizeof(float));
                memcpy(gd + (size_t)h*head_dim, src + (size_t)h*2*head_dim + head_dim, head_dim*sizeof(float));
            }
          }
        });
        turbo_mat_mat_family_or_cpu(lw.wk, lw.wk_type,
                                     xb.data(), Kb.data(),
                                     (size_t)kv_dim, (size_t)hidden, (size_t)n_tokens,
                                     "attn.k", layer_idx, mm_family);
        turbo_mat_mat_family_or_cpu(lw.wv, lw.wv_type,
                                     xb.data(), Vb.data(),
                                     (size_t)kv_dim, (size_t)hidden, (size_t)n_tokens,
                                     "attn.v", layer_idx, mm_family);
    } else {
        turbo_mat_mat_family_or_cpu(lw.wq, lw.wq_type,
                                     xb.data(), Qb.data(),
                                     (size_t)q_dim, (size_t)hidden, (size_t)n_tokens,
                                     "attn.q", layer_idx, mm_family);
        turbo_mat_mat_family_or_cpu(lw.wk, lw.wk_type,
                                     xb.data(), Kb.data(),
                                     (size_t)kv_dim, (size_t)hidden, (size_t)n_tokens,
                                     "attn.k", layer_idx, mm_family);
        turbo_mat_mat_family_or_cpu(lw.wv, lw.wv_type,
                                     xb.data(), Vb.data(),
                                     (size_t)kv_dim, (size_t)hidden, (size_t)n_tokens,
                                     "attn.v", layer_idx, mm_family);
    }

    // 3) Per token: bias + q/k norm, partial/full RoPE on Q and K, store KV.
    //    Identical helper calls to the per-token path -> identical numbers.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
      for (int t = a_; t < b_; ++t) {
        const int pos = start_pos + t;
        const int rope_pos = proj_native ? pos : rope_effective_position(pos);
        float* q = Qb.data() + (size_t)t*q_dim;
        float* k = Kb.data() + (size_t)t*kv_dim;
        float* v = Vb.data() + (size_t)t*kv_dim;

        nx_apply_qkv_extras(lw, cfg, q, k, v, q_dim, kv_dim, eps);
        nx_apply_rope_qwen35_partial_or_full(q, n_heads, rope_pos, head_dim, cfg.rope, qwen35_partial);
        nx_apply_rope_qwen35_partial_or_full(k, n_kv,    rope_pos, head_dim, cfg.rope, qwen35_partial);
      }
    });
    for (int t = 0; t < n_tokens; ++t)      // KV store stays serial
        kvcache_.store(layer_idx, start_pos + t, Kb.data() + (size_t)t*kv_dim, Vb.data() + (size_t)t*kv_dim, kv_dim);

    // 4) Pack the full KV for this layer ONCE.
    const size_t kv_count = (size_t)n_kv * n_ctx * head_dim;
    kv_flat_k_.resize(kv_count);
    kv_flat_v_.resize(kv_count);
    kvcache_.pack_k(layer_idx, n_ctx, kv_flat_k_.data());
    kvcache_.pack_v(layer_idx, n_ctx, kv_flat_v_.data());

    // 5) Attention over all N query tokens. Two GPU paths, both full-causal
    //    (matching forward_cached_scores): the new streaming path (KV in chunks,
    //    GPU never holds full KV) and the previous single-shot batched path.
    //    NEXUS_ATTN_PATH=batch forces the proven batched path for A/B diagnosis;
    //    default is streaming.
    bool ok = false;
    const char* attn_path = std::getenv("NEXUS_ATTN_PATH");
    const bool use_batched = (attn_path && (attn_path[0]=='b' || attn_path[0]=='B'));
    const bool use_stream_forced = (attn_path && (attn_path[0]=='s' || attn_path[0]=='S'));
    const bool flash_ok2 =
        attn_on_card && g.attention_flash_gpu(Qb.data(), Ao.data(),
                              n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos, /*sliding_window=*/0,
                              scale, softcap, kv_flat_k_.data(), kv_flat_v_.data());
    if (flash_ok2) {
        ok = true;
    } else if (use_batched) {
        ok = attn_on_card && g.attention_batch_gpu(
            Qb.data(), Ao.data(),
            n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos, /*sliding_window=*/0,
            scale, softcap,
            kv_flat_k_.data(), kv_flat_v_.data(), kv_count * sizeof(uint16_t));
    } else {
        ok = attn_on_card && g.attention_stream_gpu(
            Qb.data(), Ao.data(),
            n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos,
            scale, softcap,
            kv_flat_k_.data(), kv_flat_v_.data(), /*chunk_tokens=*/0);
    }
    if (!ok) {
        attn_batch_cpu(Qb.data(), kv_flat_k_.data(), kv_flat_v_.data(), Ao.data(),
                       n_heads, n_kv, head_dim, n_ctx, n_tokens, start_pos,
                       /*sliding_window=*/0, scale, softcap);
    }

    // 5b) POST-gate: attn_out *= sigmoid(gate), per token. Matches per-token path.
    if (has_gate) {
        nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
          for (int t = a_; t < b_; ++t) {
            float* ao = Ao.data()   + (size_t)t*q_dim;
            const float* gt = Gate.data() + (size_t)t*q_dim;
            for (int i = 0; i < q_dim; ++i) {
                float z = gt[i];
                float s = (z > 80.0f) ? 1.0f : (z < -80.0f) ? 0.0f : 1.0f/(1.0f+expf(-z));
                ao[i] *= s;
            }
          }
        });
    }

    // 6) Wo projection + bias + residual.
    turbo_mat_mat_family_or_cpu(lw.wo, lw.wo_type,
                                 Ao.data(), Wp.data(),
                                 (size_t)hidden, (size_t)q_dim, (size_t)n_tokens,
                                 "attn.o", layer_idx, mm_family);
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t) {
            if (lw.bo) nx_add_bias_inplace(Wp.data()+(size_t)t*hidden, lw.bo, hidden);
            cpu::add_inplace(x_batch+(size_t)t*hidden, Wp.data()+(size_t)t*hidden, hidden);
        }
    });

    // 7) FFN norm + batched FFN + residual.
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            nx_apply_model_norm(x_batch+(size_t)t*hidden, lw.ffn_norm, lw.ffn_norm_bias,
                                xb.data()+(size_t)t*hidden, (size_t)hidden, eps, lw.ffn_norm_zc);
    });
    forward_dense_ffn_batch(lw, xb.data(), Fb.data(), layer_idx, n_tokens);
    nx_par_range(n_tokens, n_threads_, [&](int a_, int b_) {
        for (int t = a_; t < b_; ++t)
            cpu::add_inplace(x_batch+(size_t)t*hidden, Fb.data()+(size_t)t*hidden, hidden);
    });

    static std::atomic<int> fa_logs{0};
    if (fa_logs.fetch_add(1, std::memory_order_relaxed) < 96) {
        fprintf(stderr, "[turbo:full-attn-layer] L=%d family=%s batched ntok=%d pos=%d..%d hd=%d gate=%d prope=%d gpu_attn=%d\n",
                layer_idx, mm_family, n_tokens, start_pos, start_pos+n_tokens-1, head_dim,
                has_gate?1:0, qwen35_partial?1:0, ok?1:0);
        fflush(stderr);
    }
    return true;
}

inline bool Transformer::layer_batch_tiled(float* x_buf, int n, int start_pos, int hidden, int l,
                                           const std::function<bool(float*, int, int)>& fn)
{
    static const int T = []() {
        const char* e = std::getenv("NEXUS_PREFILL_TILE");
        if (!e || !*e) return 1024;
        const int v = std::atoi(e);
        return v < 0 ? 1024 : v;
    }();
    // Progress line for big prompts: layers go one after another over the whole prompt, so progress = (layer + share of tiles done) / layers.
    // Printed at most every 2 s (and when a new prompt starts), so the log shows steady life without flooding.
    struct ProgState { int last_l = 1 << 30; std::chrono::steady_clock::time_point t_start, t_last; };
    static ProgState ps; static std::mutex ps_m;
    const int nl = (int)meta_.n_layers;
    auto progress = [&](int done_tokens) {
        pf_layer_.store(l - (pr_active_ ? pr_start_ : 0)); pf_layers_.store(pr_active_ ? pr_end_ - pr_start_ : nl);
        pf_tok_done_.store(done_tokens); pf_tok_total_.store(n);
        if (n < 256 || (pr_active_ && pr_quiet_) || pf_external_.load()) return;
        std::lock_guard<std::mutex> lk(ps_m);
        const auto now = std::chrono::steady_clock::now();
        if (l < ps.last_l) { ps.t_start = now; ps.t_last = now - std::chrono::seconds(10); }
        ps.last_l = l;
        if (std::chrono::duration<double>(now - ps.t_last).count() < 2.0) return;
        ps.t_last = now;
        const double frac = ((double)l + (double)done_tokens / (double)n) / (double)std::max(nl, 1);
        const double el = std::chrono::duration<double>(now - ps.t_start).count();
        char eta[48] = "";
        if (frac > 0.02 && el > 4.0) {
            long long rem = (long long)(el / frac * (1.0 - frac));
            if (rem >= 3600) std::snprintf(eta, sizeof eta, ", about %lldh %lldm left", rem / 3600, (rem % 3600) / 60);
            else if (rem >= 60) std::snprintf(eta, sizeof eta, ", about %lldm %llds left", rem / 60, rem % 60);
            else std::snprintf(eta, sizeof eta, ", about %llds left", rem);
        }
        // *** v32: tok/s, using the same formula as the single-computer turbo-path gauge
        // (server.cpp's v28 reporter thread: prompt_tokens * layer_fraction_done / elapsed) --
        // n and frac here play the same role as that thread's pr_prompt_tok_est and frac, so
        // both gauges report speed the same way.
        const double tps = el > 0.05 ? (double)n * frac / el : 0.0;
        std::fprintf(stderr, "[progress] reading prompt: %.0f%%  (layer %d of %d, %d of %d tokens in this layer, %.1f tok/s, %.0fs elapsed%s)\n",
                     frac * 100.0, l + 1, nl, done_tokens, n, tps, el, eta);
        std::fflush(stderr);
        nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
    };

    if (T <= 0 || n <= T) { progress(0); return fn(x_buf, start_pos, n); }

    // Even split so no tile is a lone token (the batched layers refuse n<=1).
    const int n_tiles = (n + T - 1) / T;
    int tile = (n + n_tiles - 1) / n_tiles;
    // A full-attention slice gets slower the further into the prompt it is (it reads every token before it). If one
    // slice takes longer than NEXUS_PREFILL_TILE_MAX_S (default 3 s), the rest of this layer uses half-size slices,
    // so no single GPU job grows long enough for a graphics driver's watchdog to reset the card (AMD on Linux allows
    // 10 s on the graphics queue). 0 turns this off.
    // On by default for AMD cards only (their Linux driver resets a card after 10 s on one job). On other cards it only cost
    // speed (smaller slices, and every slice rebuilds the layer's conversation-memory buffer), so it is off there.
    // Launcher: "Split long prompt steps" (Auto / On = 3 s / Off) -> NEXUS_PREFILL_TILE_MAX_S.
    static const double max_s_env = []() {
        const char* e = std::getenv("NEXUS_PREFILL_TILE_MAX_S");
        return (e && *e) ? std::atof(e) : -1.0;
    }();
    double max_s = max_s_env;
#ifdef NEXUS_VULKAN
    if (max_s < 0) {
        // AMD card? Told by its name (RADV / AMDVLK / amdgpu-pro all say AMD or Radeon), so this needs nothing new in the
        // Vulkan header.
        bool amd = false;
        if (gpu::gpu().ready()) {
            std::string nm = gpu::gpu().info().device_name;
            for (char& ch : nm) ch = (char)std::tolower((unsigned char)ch);
            amd = nm.find("amd") != std::string::npos || nm.find("radeon") != std::string::npos || nm.find("radv") != std::string::npos;
        }
        max_s = amd ? 3.0 : 0.0;
    }
#else
    if (max_s < 0) max_s = 0.0;
#endif
    // Never below 256 tokens: under 192 the card's fastest prompt kernels (the big tiles, 8-bit math) switch off.
    static const int min_tile = []() { const char* e = std::getenv("NEXUS_PREFILL_TILE_MIN"); const int v = e ? std::atoi(e) : 0; return v > 0 ? v : 256; }();
    for (int t0 = 0; t0 < n; ) {
        progress(t0);
        int cnt = std::min(tile, n - t0);
        if (n - t0 - cnt == 1) ++cnt;           // never leave a lone last token
        const auto ts = std::chrono::steady_clock::now();
        if (!fn(x_buf + (size_t)t0 * (size_t)hidden, start_pos + t0, cnt)) {
            if (t0 == 0) return false;          // layer can't batch: caller falls back
            for (int t = t0; t < n; ++t)        // refused mid-way: finish per token, in order
                forward_layer(l, x_buf + (size_t)t * (size_t)hidden, start_pos + t);
            return true;
        }
        t0 += cnt;
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
        if (max_s > 0 && took > max_s && tile > min_tile) {
            tile = std::max(min_tile, tile / 2);
            std::fprintf(stderr, "[prefill] layer %d: a %d-token slice took %.1f s at position %d; smaller slices (%d tokens) for the rest "
                                 "of this layer, so no GPU job runs long enough to trip the graphics driver's time limit\n",
                         l, cnt, took, start_pos + t0, tile);
            std::fflush(stderr);
        }
    }
    return true;
}

inline void Transformer::attn_batch_cpu(
    const float* Q, const uint16_t* K, const uint16_t* V, float* Out,
    int n_q_heads, int n_kv_heads, int head_dim,
    int n_ctx, int n_query, int q_base_pos, int sliding_window,
    float scale, float softcap)
{
    auto f16 = [](uint16_t h)->float {
        uint32_t s=(h>>15)&1u,e=(h>>10)&0x1Fu,m=h&0x3FFu,o;
        if(e==0u){ if(m==0u)o=s<<31u; else { uint32_t c=0,x=m; while(!(x&0x400u)){c++;x<<=1u;} o=(s<<31u)|((113u-c)<<23u)|((x&0x3FFu)<<13u);} }
        else if(e==31u) o=(s<<31u)|0x7F800000u|(m<<13u);
        else o=(s<<31u)|((e+112u)<<23u)|(m<<13u);
        float f; memcpy(&f,&o,4); return f;
    };
    for (int qi=0; qi<n_query; ++qi) for (int h=0; h<n_q_heads; ++h) {
        int kv=h*n_kv_heads/n_q_heads, qpos=q_base_pos+qi;
        int start = (sliding_window>0 && qpos+1>sliding_window) ? qpos+1-sliding_window : 0;
        int last  = std::min(qpos, n_ctx-1);
        const float* qh = Q + ((size_t)qi*n_q_heads+h)*head_dim;
        const uint16_t* Kh = K + (size_t)kv*n_ctx*head_dim;
        const uint16_t* Vh = V + (size_t)kv*n_ctx*head_dim;
        float* oh = Out + ((size_t)qi*n_q_heads+h)*head_dim;
        std::vector<float> acc(head_dim,0.f); float m=-1e30f,l=0.f;
        for (int p=start;p<=last;++p){
            float d=0.f; for(int e=0;e<head_dim;++e) d+=qh[e]*f16(Kh[(size_t)p*head_dim+e]);
            float s=d*scale; if(softcap>0.f) s=softcap*std::tanh(s/softcap);
            float mn=std::max(m,s),cr=std::exp(m-mn),w=std::exp(s-mn);
            for(int e=0;e<head_dim;++e) acc[e]=acc[e]*cr+w*f16(Vh[(size_t)p*head_dim+e]);
            l=l*cr+w; m=mn;
        }
        float inv=(l>0.f)?1.f/l:0.f;
        for(int e=0;e<head_dim;++e) oh[e]=acc[e]*inv;
    }
}

inline void Transformer::forward_qwen35_linear(
    const TransformerLayerWeights& lw,
    float* x,
    float* out,
    int layer_idx,
    int pos)
{
    const int hidden    = (int)meta_.hidden_dim;
    const int conv_dim  = lw.q35_conv_dim;
    const int key_dim   = lw.q35_key_dim;
    const int value_dim = lw.q35_value_dim;
    const int kh        = lw.q35_num_k_heads;
    const int vh        = lw.q35_num_v_heads;
    const int hd        = lw.q35_head_dim;
    const int kernel    = lw.q35_kernel;
    const int repeat    = vh / kh;

    if (conv_dim <= 0 || key_dim <= 0 || value_dim <= 0 || kh <= 0 || vh <= 0 || hd <= 0) {
        fprintf(stderr, "[nexus] FATAL-RUNTIME: bad qwen35 linear dims\n");
        memset(out, 0, (size_t)hidden * sizeof(float));
        return;
    }

    if ((int)q35_conv_state_.size() < (int)meta_.n_layers) q35_conv_state_.resize(meta_.n_layers);
    if ((int)q35_recurrent_state_.size() < (int)meta_.n_layers) q35_recurrent_state_.resize(meta_.n_layers);

    q35_host_acquire(layer_idx);   // the card may hold newer state; the CPU path below reads and rewrites the host copy
    std::vector<float>& conv_state = q35_conv_state_[layer_idx];
    std::vector<float>& rec_state  = q35_recurrent_state_[layer_idx];

    const int conv_state_len = conv_dim * std::max(0, kernel - 1);
    const int rec_state_len  = vh * hd * hd;

    if ((int)conv_state.size() != conv_state_len) conv_state.assign((size_t)conv_state_len, 0.0f);
    if ((int)rec_state.size()  != rec_state_len)  rec_state.assign((size_t)rec_state_len,  0.0f);

    std::vector<float> mixed((size_t)conv_dim, 0.0f);
    std::vector<float> z((size_t)value_dim, 0.0f);
    std::vector<float> a((size_t)vh, 0.0f);
    std::vector<float> b((size_t)vh, 0.0f);

    // ------ DIAG: dump linear-attn internals on first call to layer 0 -----
    // Fires exactly once. Compares numerical values against the HF reference
    // to identify which step of the Gated DeltaNet recurrence is wrong.
    static std::atomic<int> q35_diag_count{0};
    const bool diag_this_call = nx_qwen3next_diag_enabled() &&
                                 (layer_idx == 0) &&
                                 (q35_diag_count.fetch_add(1) == 0);

    auto diag_stats = [](const char* tag, const float* v, int n) {
        if (n <= 0 || !v) {
            fprintf(stderr, "[diag-q35] %s: n=%d (skipped)\n", tag, n);
            return;
        }
        float mn = v[0], mx = v[0], sum = 0.0f, sumsq = 0.0f;
        int n_nan = 0, n_inf = 0;
        for (int i = 0; i < n; ++i) {
            const float x = v[i];
            if (std::isnan(x))      ++n_nan;
            else if (std::isinf(x)) ++n_inf;
            else {
                if (x < mn) mn = x;
                if (x > mx) mx = x;
                sum   += x;
                sumsq += x * x;
            }
        }
        const int n_finite = n - n_nan - n_inf;
        const float mean = (n_finite > 0) ? sum / n_finite : 0.0f;
        const float rms  = (n_finite > 0) ? sqrtf(sumsq / n_finite) : 0.0f;
        fprintf(stderr,
            "[diag-q35] %-22s n=%-6d min=%+.4e max=%+.4e mean=%+.4e rms=%+.4e nan=%d inf=%d "
            "first4=[%+.4e %+.4e %+.4e %+.4e]\n",
            tag, n, mn, mx, mean, rms, n_nan, n_inf,
            n >= 1 ? v[0] : 0.0f,
            n >= 2 ? v[1] : 0.0f,
            n >= 3 ? v[2] : 0.0f,
            n >= 4 ? v[3] : 0.0f);
    };
    // ------ END DIAG setup -----------------------------------------------

    auto is_nq_g64 = [](QuantType t) -> bool {
        return t == QuantType::NQ4_G64 ||
               t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 ||
               t == QuantType::NQ8_G64;
    };

    // Native .nxm tensors use our NQ*_G64 block layout. The existing CPU
    // mat_vec path can silently return zeros for these new private types on
    // some builds, which is exactly what the first native-model test showed:
    // non-zero normalized input, but some projection rows came back zero. Route rows
    // through quant::dequantize_row directly until the shared CPU kernel is
    // updated for NQ. This is deliberately correctness-first.
    auto mat_vec_native_safe = [&](const void* W, QuantType wt,
                                   const float* inp, float* dst,
                                   size_t rows, size_t cols,
                                   const char* tag) {
        if (!W || !inp || !dst || rows == 0 || cols == 0) {
            if (dst && rows) memset(dst, 0, rows * sizeof(float));
            return;
        }

        if (nx_q35_try_gpu_matvec(layer_card(lw), W, inp, dst, rows, cols, wt)) return;

        // NQ*_G64 and BF16 are now first-class in cpu::mat_vec. This keeps the
        // correctness baseline but stops dequantizing every row into temp floats.
        cpu::mat_vec(W, inp, dst, rows, cols, wt, n_threads_);
        if (diag_this_call) {
            fprintf(stderr,
                "[diag-q35] main-route matvec %s rows=%zu cols=%zu type=%d\n",
                tag ? tag : "?", rows, cols, (int)wt);
            fflush(stderr);
        }
        return;

        // Do not trust the generic cpu::mat_vec for Qwen3Next native-file
        // projection tensors during bring-up. The small a/b projections are
        // usually kept as source BF16/F16 (not NQ), and the previous build
        // still produced all-zero a/b because they fell back to the shared
        // matvec path. Row-dequant + dot is slow but handles F32/F16/BF16 and
        // all GGUF/NXM quant types consistently.
        const bool row_dequant_supported =
            wt == QuantType::F32  ||
            wt == QuantType::F16  ||
            wt == QuantType::BF16 ||
            wt == QuantType::Q4_0 ||
            wt == QuantType::Q5_0 ||
            wt == QuantType::Q8_0 ||
            wt == QuantType::Q4_K ||
            wt == QuantType::Q5_K ||
            wt == QuantType::Q6_K ||
            wt == QuantType::Q2_K ||
            wt == QuantType::Q3_K ||
            is_nq_g64(wt);

        if (!row_dequant_supported) {
            cpu::mat_vec(W, inp, dst, rows, cols, wt, n_threads_);
            if (diag_this_call) {
                fprintf(stderr,
                    "[diag-q35] generic matvec fallback %s rows=%zu cols=%zu type=%d\n",
                    tag ? tag : "?", rows, cols, (int)wt);
                fflush(stderr);
            }
            return;
        }

        // Parallel row-dequant matvec. This keeps the correctness-safe native
        // path without turning Qwen3Next bring-up into a single-core slog.
        const unsigned hw = std::thread::hardware_concurrency();
        int nt = n_threads_ > 1 ? n_threads_ : (hw ? (int)hw : 4);
        nt = std::max(1, std::min<int>(nt, (int)rows));
        if (diag_this_call) {
            fprintf(stderr,
                "[diag-q35] native-safe threaded rowdeq %s rows=%zu cols=%zu type=%d threads=%d\n",
                tag ? tag : "?", rows, cols, (int)wt, nt);
            fflush(stderr);
        }

        std::atomic<size_t> next_row{0};
        std::vector<std::thread> workers;
        workers.reserve((size_t)nt);
        for (int ti = 0; ti < nt; ++ti) {
            workers.emplace_back([&]() {
                std::vector<float> row(cols);
                for (;;) {
                    const size_t r = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (r >= rows) break;
                    quant::dequantize_row(W, row.data(), r, cols, wt);
                    double acc = 0.0;
                    for (size_t c = 0; c < cols; ++c) {
                        acc += (double)row[c] * (double)inp[c];
                    }
                    dst[r] = (float)acc;
                }
            });
        }
        for (auto& th : workers) th.join();

        if (diag_this_call) {
            float mn = dst[0], mx = dst[0], ss = 0.0f;
            for (size_t i = 0; i < rows; ++i) {
                if (dst[i] < mn) mn = dst[i];
                if (dst[i] > mx) mx = dst[i];
                ss += dst[i] * dst[i];
            }
            fprintf(stderr,
                "[diag-q35] native-safe rowdeq matvec %s rows=%zu cols=%zu type=%d rms=%+.4e min=%+.4e max=%+.4e\n",
                tag ? tag : "?", rows, cols, (int)wt, sqrtf(ss / (float)rows), mn, mx);
            fflush(stderr);
        }
    };
    
    mat_vec_native_safe(lw.q35_qkv, lw.q35_qkv_type, x, mixed.data(),
                        (size_t)conv_dim, (size_t)hidden,
                        "linear_attn.in_proj_qkv");

    mat_vec_native_safe(lw.q35_z, lw.q35_z_type, x, z.data(),
                        (size_t)value_dim, (size_t)hidden,
                        "linear_attn.in_proj_z");

    if (lw.q35_ba) {
        // Latest llama.cpp / GGUF stores beta+alpha as one projection:
        //   ssm_beta_alpha.weight -> reshape [num_k_heads][2 * repeat]
        //   per key-head layout: [beta_0..beta_{repeat-1}][alpha_0..alpha_{repeat-1}]
        // This is NOT the same as flat [all_beta][all_alpha].
        std::vector<float> ba((size_t)vh * 2, 0.0f);
        mat_vec_native_safe(lw.q35_ba, lw.q35_ba_type, x, ba.data(),
                            (size_t)vh * 2, (size_t)hidden,
                            "linear_attn.ssm_beta_alpha");

        for (int hk_i = 0; hk_i < kh; ++hk_i) {
            const size_t src = (size_t)hk_i * (size_t)(2 * repeat);
            const size_t dst = (size_t)hk_i * (size_t)repeat;
            for (int r = 0; r < repeat; ++r) {
                b[dst + (size_t)r] = ba[src + (size_t)r];
                a[dst + (size_t)r] = ba[src + (size_t)repeat + (size_t)r];
            }
        }
    } else {
        mat_vec_native_safe(lw.q35_a, lw.q35_a_type, x, a.data(),
                            (size_t)vh, (size_t)hidden,
                            "linear_attn.in_proj_a");

        mat_vec_native_safe(lw.q35_b, lw.q35_b_type, x, b.data(),
                            (size_t)vh, (size_t)hidden,
                            "linear_attn.in_proj_b");
    }

    if (diag_this_call) {
        fprintf(stderr,
            "[nexus] Qwen3Next native audit fix active: split-QKV defaults flat; raw HF decoder/final RMSNorm=(1+w); gated-deltanet norm direct; full-attn q+gate per-head; NQ/BF16 uses main cpu::mat_vec.\n");
        fprintf(stderr,
            "[diag-q35] === Linear-attn dump layer=%d pos=first kh=%d vh=%d hd=%d "
            "key_dim=%d value_dim=%d conv_dim=%d kernel=%d ===\n",
            layer_idx, kh, vh, hd, key_dim, value_dim, conv_dim, kernel);
        diag_stats("input x",         x,            hidden);
        diag_stats("mixed (pre-conv)", mixed.data(), conv_dim);
        diag_stats("z (gate)",        z.data(),     value_dim);
        diag_stats("a (raw)",         a.data(),     vh);
        diag_stats("b (raw)",         b.data(),     vh);
        diag_stats("q35_A (per-head)", lw.q35_A,    vh);
        diag_stats("q35_dt (per-head)", lw.q35_dt,  vh);
    }
                 
    // Qwen3Next/Qwen35 optimized GGUF path:
    //
    // If the model has blk.N.attn_qkv.weight + blk.N.attn_gate.weight, then
    // attn_qkv.weight is already the optimized flat conv input:
    //
    //   [all Q][all K][all V]
    //
    // The grouped [q][k][v][z] deinterleave is only for the legacy ssm_in/qkvz
    // tensor path. This loader is not using that legacy tensor here.
    //
    // Reordering attn_qkv here scrambles per-channel depthwise conv kernels and
    // produces exactly the kind of multilingual/random-token garbage we are seeing.
    static bool qwen35_preconv_order_printed = false;
    if (!qwen35_preconv_order_printed) {
        qwen35_preconv_order_printed = true;
        fprintf(stderr,
            "[nexus] Qwen35 linear pre-conv QKV order: native channel order preserved for depthwise conv "
            "kh=%d vh=%d repeat=%d hd=%d conv_dim=%d key_dim=%d value_dim=%d\n",
            kh, vh, repeat, hd, conv_dim, key_dim, value_dim);
        fflush(stderr);
    }               
    
    // Depthwise causal conv, single-token recurrent mode:
    // input window = old conv_state + current mixed.
    std::vector<float> conv_out((size_t)conv_dim, 0.0f);
    for (int c = 0; c < conv_dim; ++c) {
        float sum = 0.0f;

        for (int k = 0; k < kernel; ++k) {
            float xv;
            if (k < kernel - 1) {
                xv = conv_state[(size_t)c * (kernel - 1) + k];
            } else {
                xv = mixed[c];
            }

            // GGUF reports this tensor as [kernel, conv_dim], where the
            // kernel dimension is the fastest/inner dimension. That means
            // each channel owns a contiguous kernel: [c][k]. Do NOT index
            // it as [k][c], or every channel reads another channel's kernel.
            sum += xv * lw.q35_conv[(size_t)c * kernel + k];
        }

        conv_out[c] = sum / (1.0f + expf(-sum)); // silu
    }

    // Shift conv state left and append current pre-conv mixed.
    // IMPORTANT: for native Qwen3Next .nxm, the depthwise conv channels are in
    // the original HF grouped order, not flat [all Q][all K][all V].  Keep the
    // conv state and conv kernel in that native order, then deinterleave only
    // the conv output before feeding the delta rule.
    if (kernel > 1) {
        for (int c = 0; c < conv_dim; ++c) {
            float* st = conv_state.data() + (size_t)c * (kernel - 1);
            for (int k = 0; k < kernel - 2; ++k) st[k] = st[k + 1];
            st[kernel - 2] = mixed[c];
        }
    }

    const char* qkv_layout_env = std::getenv("NEXUS_QWEN35_LINEAR_QKV_LAYOUT");
    const bool qkv_grouped_layout = (qkv_layout_env && std::strcmp(qkv_layout_env, "grouped") == 0);

    std::vector<float> q_flat((size_t)key_dim, 0.0f);
    std::vector<float> k_flat((size_t)key_dim, 0.0f);
    std::vector<float> v_flat((size_t)value_dim, 0.0f);

    float* q_base = nullptr;
    float* k_base = nullptr;
    float* v_base = nullptr;

    if (qkv_grouped_layout) {
        // Legacy/debug path only.
        // This is useful if a future native file stores the unsplit HF
        // in_proj_qkvz tensor after dropping z. The 27B .nxm we built does NOT:
        // it contains separate in_proj_qkv + in_proj_z tensors. HF's reference
        // first splits qkvz by key-head group, then concatenates query/key/value
        // into flat [all Q][all K][all V] BEFORE the depthwise conv. Therefore
        // split in_proj_qkv tensors must default to flat order, not grouped.
        const int group_span = 2 * hd + repeat * hd;
        for (int hk_i = 0; hk_i < kh; ++hk_i) {
            const size_t src = (size_t)hk_i * (size_t)group_span;
            std::memcpy(q_flat.data() + (size_t)hk_i * hd,
                        conv_out.data() + src,
                        (size_t)hd * sizeof(float));
            std::memcpy(k_flat.data() + (size_t)hk_i * hd,
                        conv_out.data() + src + (size_t)hd,
                        (size_t)hd * sizeof(float));
            for (int r = 0; r < repeat; ++r) {
                const size_t dst_v = ((size_t)hk_i * (size_t)repeat + (size_t)r) * (size_t)hd;
                const size_t src_v = src + (size_t)(2 * hd) + (size_t)r * (size_t)hd;
                std::memcpy(v_flat.data() + dst_v,
                            conv_out.data() + src_v,
                            (size_t)hd * sizeof(float));
            }
        }
        q_base = q_flat.data();
        k_base = k_flat.data();
        v_base = v_flat.data();
    } else {
        // Correct default for native .nxm split tensors:
        //   linear_attn.in_proj_qkv.weight [key_dim + key_dim + value_dim, hidden]
        //   linear_attn.in_proj_z.weight   [value_dim, hidden]
        // The runtime projection output is already flat [all Q][all K][all V].
        q_base = conv_out.data();
        k_base = q_base + key_dim;
        v_base = k_base + key_dim;
    }

    if (diag_this_call) {
        diag_stats("conv_out (post-silu raw)", conv_out.data(), conv_dim);
        diag_stats(qkv_grouped_layout ? "conv_out Q-slice degrouped" : "conv_out Q-slice native-flat",
                   q_base, key_dim);
        diag_stats(qkv_grouped_layout ? "conv_out K-slice degrouped" : "conv_out K-slice native-flat",
                   k_base, key_dim);
        diag_stats(qkv_grouped_layout ? "conv_out V-slice degrouped" : "conv_out V-slice native-flat",
                   v_base, value_dim);
    }

    static bool qwen35_postconv_split_printed = false;
    if (!qwen35_postconv_split_printed) {
        qwen35_postconv_split_printed = true;
        fprintf(stderr,
            "[nexus] Qwen35 linear post-conv split: %s "
            "key_dim=%d value_dim=%d conv_dim=%d kh=%d repeat=%d hd=%d "
            "(NEXUS_QWEN35_LINEAR_QKV_LAYOUT=%s)\n",
            qkv_grouped_layout ? "legacy grouped [per-kh Q,K,Vrepeat] -> flat [Q][K][V]" : "native split tensor flat [Q][K][V]",
            key_dim, value_dim, conv_dim, kh, repeat, hd,
            qkv_layout_env ? qkv_layout_env : "(unset, default=flat-native)");
        fflush(stderr);
    }

    std::vector<float> core((size_t)value_dim, 0.0f);

    // Q/K heads are shared across repeated V heads. Normalize each Q/K head once,
    // into separate buffers. Do not mutate conv_out in the hv loop, or the same
    // key head gets normalized repeatedly when repeat > 1.
    std::vector<float> q_normed((size_t)key_dim, 0.0f);
    std::vector<float> k_normed((size_t)key_dim, 0.0f);

    memcpy(q_normed.data(), q_base, (size_t)key_dim * sizeof(float));
    memcpy(k_normed.data(), k_base, (size_t)key_dim * sizeof(float));

    for (int hk = 0; hk < kh; ++hk) {
        nx_l2norm_inplace(q_normed.data() + (size_t)hk * hd, hd, 1e-6f);
        nx_l2norm_inplace(k_normed.data() + (size_t)hk * hd, hd, 1e-6f);
    }

    const float scale = 1.0f / sqrtf((float)hd);

    for (int hv_i = 0; hv_i < vh; ++hv_i) {
        const int hk = hv_i / repeat;

        const float* q = q_normed.data() + (size_t)hk * hd;
        const float* k = k_normed.data() + (size_t)hk * hd;
        const float* v = v_base + (size_t)hv_i * hd;

        // HF recurrent rule uses q scaled by 1/sqrt(head_dim).
        std::vector<float> q_scaled((size_t)hd);
        for (int i = 0; i < hd; ++i) {
            q_scaled[i] = q[i] * scale;
        }

        const float beta = nx_sigmoid_f32(b[hv_i]);

        // llama.cpp GGUF uses ssm_a as the no-scan coefficient. That means this
        // is already the negative coefficient equivalent to -exp(A_log).
        // Clamp g defensively: mathematically it should be <= 0. If it goes
        // positive because of a bad converter or corrupted tensor, exp(g) will
        // blow the recurrent state up immediately.
        float g = lw.q35_A[hv_i] * nx_softplus_f32(a[hv_i] + lw.q35_dt[hv_i]);

        if (!std::isfinite(g)) g = -60.0f;
        if (g > 0.0f) g = 0.0f;
        if (g < -60.0f) g = -60.0f;

        const float decay = expf(g);

        float* S = rec_state.data() + (size_t)hv_i * hd * hd;

        // Qwen3Next / Qwen35 Gated DeltaNet recurrent state is [K][V].
        for (int i = 0; i < hd * hd; ++i) {
            S[i] *= decay;
        }

        float kv_mem[256];
        if (hd > 256) {
            fprintf(stderr, "[nexus] FATAL-RUNTIME: qwen35 hd=%d > 256 scratch\n", hd);
            memset(out, 0, (size_t)hidden * sizeof(float));
            return;
        }

        // kv_mem = k @ S, output shape [V]
        for (int j = 0; j < hd; ++j) {
            float s = 0.0f;
            for (int i = 0; i < hd; ++i) {
                s += k[i] * S[(size_t)i * hd + j];
            }
            kv_mem[j] = s;
        }

        // delta = (v - kv_mem) * beta
        float delta[256];
        for (int j = 0; j < hd; ++j) {
            delta[j] = (v[j] - kv_mem[j]) * beta;
        }

        // S += outer(k, delta), still [K][V]
        for (int i = 0; i < hd; ++i) {
            const float ki = k[i];
            float* Srow = S + (size_t)i * hd;
            for (int j = 0; j < hd; ++j) {
                Srow[j] += ki * delta[j];
            }
        }

// core = q_scaled @ S, output shape [V]
        float* dst = core.data() + (size_t)hv_i * hd;
        for (int j = 0; j < hd; ++j) {
            float s = 0.0f;
            for (int i = 0; i < hd; ++i) {
                s += q_scaled[i] * S[(size_t)i * hd + j];
            }
            dst[j] = s;
        }

        static std::atomic<int> q35_recur_diag_count{0};
        const bool diag_recur = nx_qwen3next_diag_enabled() &&
            (layer_idx == 0 && hv_i == 0 && pos < 12 &&
             q35_recur_diag_count.fetch_add(1) < 12);

        if (diag_recur) {
            fprintf(stderr,
                "[diag-q35-recur] pos=%d layer=%d hv=%d g=%+.6e decay=%+.6e beta=%+.6e A=%+.6e a=%+.6e dt=%+.6e b=%+.6e\n",
                pos, layer_idx, hv_i, g, decay, beta, lw.q35_A[0], a[0], lw.q35_dt[0], b[0]);
            fprintf(stderr,
                "[diag-q35] head 0: g=%+.4e decay=%+.4e beta=%+.4e q35_A=%+.4e a=%+.4e b=%+.4e\n",
                g, decay, beta, lw.q35_A[0], a[0], b[0]);
            diag_stats("head0 q (post-norm)", q,            hd);
            diag_stats("head0 q_scaled",      q_scaled.data(), hd);
            diag_stats("head0 k (post-norm)", k,            hd);
            diag_stats("head0 v",             v,            hd);
            diag_stats("head0 kv_mem",        kv_mem,       hd);
            diag_stats("head0 delta",         delta,        hd);
            diag_stats("head0 S after upd",   S,            hd * hd);
            diag_stats("head0 core (pre-norm)", dst,        hd);
        }

        // Qwen3NextRMSNormGated is NOT Gemma3/Qwen3Next zero-centered RMSNorm.
        // HF applies: norm(core_head) * norm.weight * silu(z_head).
        // The decoder input/post-attn norms can be zero-centered, and full-attn
        // q/k norms can be zero-centered, but this GatedDeltaNet per-head norm
        // must use direct weight semantics. Passing zero_centered_norm_ here
        // doubles a ~1.0 norm weight and amplifies the recurrent stream.
        nx_qwen35_gated_rmsnorm(
            dst,
            z.data() + (size_t)hv_i * hd,
            lw.q35_norm,
            hd,
            meta_.rms_norm_eps,
            false);
    }

if (diag_this_call) {
        diag_stats("core (post-gated-rmsnorm)", core.data(), value_dim);
    }

    mat_vec_native_safe(lw.q35_out, lw.q35_out_type, core.data(), out,
                        (size_t)hidden, (size_t)value_dim,
                        "linear_attn.out_proj");

    if (diag_this_call) {
        diag_stats("output (post-out_proj)", out, hidden);
        fprintf(stderr, "[diag-q35] === END dump ===\n");
        fflush(stderr);
    }
}

inline bool Transformer::q35_gpu_linear_step(const TransformerLayerWeights& lw, int l, float* x, bool async)
{
#ifdef NEXUS_VULKAN
    static const bool off = [] { const char* e = std::getenv("NEXUS_Q35_GPU_LAYER"); return e && e[0] == '0'; }();
    static const bool allow_nq = [] { const char* e = std::getenv("NEXUS_Q35_GPU_LAYER_NQ"); return e && e[0] == '1'; }();
    if (off || !lw.qwen35_linear || !lw.gpu_layer || lw.moe || lw.fused_gate_up || lw.q35_ba) return false;
    if (lw.attn_norm_zc || lw.ffn_norm_zc || lw.attn_norm_bias || lw.ffn_norm_bias || ffn_activation_gelu_) return false;
    if (!lw.attn_norm || !lw.ffn_norm || !lw.q35_qkv || !lw.q35_z || !lw.q35_a || !lw.q35_b || !lw.q35_out ||
        !lw.wgate || !lw.wup || !lw.wdown || !lw.q35_conv || !lw.q35_A || !lw.q35_dt || !lw.q35_norm) return false;
    gpu::VulkanBackend& g = gpu::gpu();
    if (!g.ready()) return false;
    if (!allow_nq) {
        for (QuantType t : { lw.q35_qkv_type, lw.q35_z_type, lw.q35_a_type, lw.q35_b_type, lw.q35_out_type, lw.wgate_type, lw.wup_type, lw.wdown_type })
            if (t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 || t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64) return false;
    }
    const int n_layers = (int)meta_.n_layers;
    if ((int)q35_conv_state_.size() < n_layers) q35_conv_state_.resize(n_layers);
    if ((int)q35_recurrent_state_.size() < n_layers) q35_recurrent_state_.resize(n_layers);
    if ((int)q35_dev_ahead_.size() < n_layers) q35_dev_ahead_.resize(n_layers, 0);
    const size_t conv_len = (size_t)lw.q35_conv_dim * (size_t)std::max(0, lw.q35_kernel - 1);
    const size_t rec_len  = (size_t)lw.q35_num_v_heads * lw.q35_head_dim * lw.q35_head_dim;
    if (!g.q35_state_resident(this, l)) {                       // host copy is authoritative: make sure it has the right shape
        if (q35_conv_state_[l].size() != conv_len) q35_conv_state_[l].assign(conv_len, 0.0f);
        if (q35_recurrent_state_[l].size() != rec_len) q35_recurrent_state_[l].assign(rec_len, 0.0f);
    }
    gpu::VulkanBackend::Q35LinearDesc d;
    d.owner = this; d.layer = l;
    d.hidden = (uint32_t)meta_.hidden_dim; d.conv_dim = (uint32_t)lw.q35_conv_dim; d.key_dim = (uint32_t)lw.q35_key_dim; d.value_dim = (uint32_t)lw.q35_value_dim;
    d.kh = (uint32_t)lw.q35_num_k_heads; d.vh = (uint32_t)lw.q35_num_v_heads; d.hd = (uint32_t)lw.q35_head_dim; d.kernel = (uint32_t)lw.q35_kernel; d.ffn = (uint32_t)meta_.ffn_dim;
    d.norm_eps = meta_.rms_norm_eps;
    d.attn_norm = lw.attn_norm; d.ffn_norm = lw.ffn_norm;
    d.qkv = lw.q35_qkv; d.qkv_t = lw.q35_qkv_type; d.z = lw.q35_z; d.z_t = lw.q35_z_type; d.a = lw.q35_a; d.a_t = lw.q35_a_type; d.b = lw.q35_b; d.b_t = lw.q35_b_type;
    d.conv_w = lw.q35_conv; d.A = lw.q35_A; d.dt = lw.q35_dt; d.gnorm = lw.q35_norm;
    d.out = lw.q35_out; d.out_t = lw.q35_out_type;
    d.gate = lw.wgate; d.gate_t = lw.wgate_type; d.up = lw.wup; d.up_t = lw.wup_type; d.down = lw.wdown; d.down_t = lw.wdown_type;
    q8swap(d.qkv, d.qkv_t, d.conv_dim, d.hidden); q8swap(d.z, d.z_t, d.value_dim, d.hidden);
    q8swap(d.gate, d.gate_t, d.ffn, d.hidden); q8swap(d.up, d.up_t, d.ffn, d.hidden);
    if (!g.q35_linear_layer(d, x, q35_conv_state_[l].data(), q35_recurrent_state_[l].data(), async)) return false;
    q35_dev_ahead_[l] = 1;
    return true;
#else
    (void)lw; (void)l; (void)x; (void)async; return false;
#endif
}

inline void Transformer::q8swap(const void*& w, QuantType& t, size_t rows, size_t cols)
{
#ifdef NEXUS_VULKAN
    static const bool off = [] { const char* e = std::getenv("NEXUS_Q6_Q8"); return !(e && e[0] == '1'); }();   // OPT-IN: accuracy first, off by default
    if (off || t != QuantType::Q6_K || !w || cols % 32 != 0 || rows * cols < 4000000u) return;
    auto it = q8swap_.find(w);
    if (it == q8swap_.end()) {
        std::vector<uint8_t> out;
        const size_t extra = rows * (cols / 32) * sizeof(quant::BlockQ8_0);
        gpu::VulkanBackend& g = gpu::gpu();
        const size_t bud = g.weight_cache_budget();
        bool ok = bud == 0 || (g.weight_cache_used() + q8swap_bytes_ + extra) <= bud / 100 * 85;      // never push a tight card into evicting weights
        const auto t0 = std::chrono::steady_clock::now();
        if (ok) ok = nx_requant_to_q8_0(w, t, rows, cols, n_threads_, out);
        if (ok) {
            q8swap_bytes_ += out.size();
            static int n = 0; if (++n <= 3 || n % 20 == 0)
                fprintf(stderr, "[nexus] Q6_K %zux%zu -> Q8_0 for decode (%.1f MB, %.2f s; %.0f MB converted so far)\n", rows, cols, out.size() / 1048576.0,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), q8swap_bytes_ / 1048576.0);
        } else out.clear();
        it = q8swap_.emplace(w, std::move(out)).first;
    }
    if (!it->second.empty()) { w = it->second.data(); t = QuantType::Q8_0; }
#else
    (void)w; (void)t; (void)rows; (void)cols;
#endif
}

// Architectures whose ordinary attention layer is exactly what the whole-layer GPU step computes (RMSNorm, optional q/k/v/o
// biases, optional per-head q/k norms, full RoPE, softmax attention, SwiGLU FFN) with nothing arch-specific on the CPU path
// in between. Others (Gemma's extra norms, Granite's multipliers, Llama 4's NoPE layers, ...) keep the CPU-orchestrated path;
// NEXUS_GPU_LAYER_ANY=1 lets any architecture try (the per-layer checks still apply), NEXUS_GPU_LAYER_ANY=0 limits it to Qwen3.5.
inline bool Transformer::plain_attn_gpu_ok() const {
    static const int any = [] { const char* e = std::getenv("NEXUS_GPU_LAYER_ANY"); return e && *e ? (e[0] == '0' ? 0 : 1) : -1; }();
    if (any == 0) return false;
    if (any == 1) return true;
    std::string a = meta_.arch;
    for (char& c : a) c = (char)std::tolower((unsigned char)c);
    static const char* const kPlain[] = { "llama", "mistral", "mistral3", "qwen2", "qwen2vl", "qwen3", "qwen3vl",
                                          "internlm2", "exaone", "deepseek" };
    for (const char* k : kPlain) if (a == k) return true;
    return false;
}

inline bool Transformer::q35_gpu_full_step(const TransformerLayerWeights& lw, int l, float* x, int pos, bool async)
{
#ifdef NEXUS_VULKAN
    static const bool off = [] { const char* e = std::getenv("NEXUS_Q35_GPU_ATTN"); return e && e[0] == '0'; }();
    static const bool allow_nq = [] { const char* e = std::getenv("NEXUS_Q35_GPU_LAYER_NQ"); return e && e[0] == '1'; }();
    static const bool qgate_ok = [] { const char* e = std::getenv("NEXUS_QWEN35_QGATE_LAYOUT"); return !e || !*e || std::strcmp(e, "perhead") == 0; }();
    // Qwen3.5 gated layers, and ordinary attention layers of architectures whose layer is exactly
    // norm -> q/k/v (+bias) -> [q/k head norm] -> RoPE -> softmax attention -> o (+bias) -> residual -> norm -> SwiGLU -> residual.
    const bool gated = lw.qwen35_full_attn_q_has_gate;
    if (off || lw.qwen35_linear || !lw.gpu_layer || lw.moe || lw.fused_gate_up || lw.fused_qkv) return gpu::nx_decline("linear-attention / MoE / fused-weight layer, not on the GPU, or NEXUS_Q35_GPU_ATTN=0");
    if (gated && !qgate_ok) return gpu::nx_decline("NEXUS_QWEN35_QGATE_LAYOUT is not perhead");
    if (!gated && !plain_attn_gpu_ok()) return gpu::nx_decline("architecture not on the plain-attention GPU list (NEXUS_GPU_LAYER_ANY=1 to try it anyway)");
    if (lw.kv_donor_layer >= 0 || lw.gemma4_tied_kv || lw.attn_gate) return gpu::nx_decline("lw.kv_donor_layer >= 0 || lw.gemma4_tied_kv || lw.attn_gate");
    if (lw.attn_norm_zc || lw.ffn_norm_zc || lw.attn_norm_bias || lw.ffn_norm_bias || ffn_activation_gelu_) return gpu::nx_decline("lw.attn_norm_zc || lw.ffn_norm_zc || lw.attn_norm_bias || lw.ffn_norm_bias || ffn_activation_gelu_");
    if (lw.attn_post_norm || lw.ffn_post_norm || lw.layer_output_scale || lw.ple_layer_index >= 0) return gpu::nx_decline("lw.attn_post_norm || lw.ffn_post_norm || lw.layer_output_scale || lw.ple_layer_index >= 0");
    if (!lw.attn_norm || !lw.ffn_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo || !lw.wgate || !lw.wup || !lw.wdown) return gpu::nx_decline("!lw.attn_norm || !lw.ffn_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo || !lw.wgate || !lw.wup || !lw.wdown");
    if (gated && (!lw.attn_q_norm || !lw.attn_k_norm)) return gpu::nx_decline("gated layer without q/k norm");
    AttentionConfig cfg = attn_cfg_;
    if (lw.attn_heads_layer > 0)     cfg.n_heads    = lw.attn_heads_layer;
    if (lw.attn_kv_heads_layer > 0)  cfg.n_kv_heads = lw.attn_kv_heads_layer;
    if (lw.attn_head_dim_layer > 0)  cfg.head_dim   = lw.attn_head_dim_layer;
    if (lw.sw_window_size > 0)       cfg.sliding_window = lw.sw_window_size;
    if (lw.rope_freq_base_layer > 0) cfg.rope.freq_base = lw.rope_freq_base_layer;
    if (cfg.sliding_window != 0 || cfg.attn_softcap != 0.0f) return gpu::nx_decline("cfg.sliding_window != 0 || cfg.attn_softcap != 0.0f");
    const int hd = cfg.head_dim, nh = cfg.n_heads, nkv = cfg.n_kv_heads;
    if (hd <= 0 || nh <= 0 || nkv <= 0) return gpu::nx_decline("hd <= 0 || nh <= 0 || nkv <= 0");
    // head norms must be per head (some models -- OLMo 2 -- normalise the whole q vector instead)
    if ((lw.attn_q_norm && (int)lw.attn_q_norm_len != hd) || (lw.attn_k_norm && (int)lw.attn_k_norm_len != hd)) return gpu::nx_decline("q/k norm is not per head");
    if (kvcache_.kv_heads_for_layer(l) != nkv || kvcache_.head_dim_for_layer(l) != hd) return gpu::nx_decline("kvcache_.kv_heads_for_layer(l) != nkv || kvcache_.head_dim_for_layer(l) != hd");
    if (pos < 0 || kvcache_.n_tokens() < pos) return gpu::nx_decline("pos < 0 || kvcache_.n_tokens() < pos");
    int rot = hd;                                                     // ordinary layers rotate the whole head
    if (gated) { rot = hd / 4; if (hd == 256) rot = 64; }            // same partial-RoPE width as nx_apply_rope_qwen35_partial_or_full
    if (rot <= 0 || rot > hd || (rot & 1)) return gpu::nx_decline("rot <= 0 || rot > hd || (rot & 1)");
    gpu::VulkanBackend& g = gpu::gpu();
    if (!g.ready()) return gpu::nx_decline("!g.ready()");
    if (!allow_nq) {
        for (QuantType t : { lw.wq_type, lw.wk_type, lw.wv_type, lw.wo_type, lw.wgate_type, lw.wup_type, lw.wdown_type })
            if (t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 || t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64) return gpu::nx_decline("t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 || t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64");
    }
    // cos/sin of this position, produced by the engine's own rope code (rotate_half layout on the first `rot` dims) so every RoPE variant matches the CPU path
    const int rope_pos = rope_effective_position(pos);
    // The table is built in rotate_half form (cos[i], sin[i] per pair i -- identical frequencies for both layouts); the shader
    // applies it as rotate_half (Qwen3.5 always, NeoX/HF models) or to interleaved pairs (layout 0, Llama-style GGUF), like the CPU path.
    const bool interleaved = !gated && cfg.rope.layout == 0;
    RoPEConfig rc = cfg.rope; rc.layout = 1;
    struct TabKey { int pos; int rot; float base, scale; int type; uint32_t n_ctx_orig; };
    static thread_local TabKey last_key{-1, 0, 0.f, 0.f, 0, 0};
    static thread_local std::vector<float> tab;
    const TabKey key{rope_pos, rot, rc.freq_base, rc.scale, rc.type, rc.n_ctx_orig};
    if (tab.size() != (size_t)rot || std::memcmp(&key, &last_key, sizeof(key)) != 0) {
        std::vector<float> probe((size_t)rot, 0.0f);
        for (int i = 0; i < rot / 2; ++i) probe[(size_t)i] = 1.0f;
        model::rope_apply(probe.data(), rope_pos, rot, rc);
        tab = probe; last_key = key;
    }
    gpu::VulkanBackend::Q35FullDesc d;
    d.owner = this; d.layer = l;
    d.hidden = (uint32_t)meta_.hidden_dim; d.n_heads = (uint32_t)nh; d.n_kv = (uint32_t)nkv; d.hd = (uint32_t)hd; d.rot = (uint32_t)rot; d.ffn = (uint32_t)meta_.ffn_dim; d.pos = (uint32_t)pos;
    d.norm_eps = meta_.rms_norm_eps; d.scale = cfg.scale;
    d.attn_norm = lw.attn_norm; d.ffn_norm = lw.ffn_norm;
    d.wq = lw.wq; d.wq_t = lw.wq_type; d.wk = lw.wk; d.wk_t = lw.wk_type; d.wv = lw.wv; d.wv_t = lw.wv_type; d.wo = lw.wo; d.wo_t = lw.wo_type;
    d.q_norm = lw.attn_q_norm; d.k_norm = lw.attn_k_norm; d.q_zc = lw.attn_q_norm_zc; d.k_zc = lw.attn_k_norm_zc;
    d.has_gate = gated; d.rope_interleaved = interleaved;
    d.bq = lw.bq; d.bk = lw.bk; d.bv = lw.bv; d.bo = lw.bo;
    d.rope_tab = tab.data();
    d.gate = lw.wgate; d.gate_t = lw.wgate_type; d.up = lw.wup; d.up_t = lw.wup_type; d.down = lw.wdown; d.down_t = lw.wdown_type;
    q8swap(d.wq, d.wq_t, (size_t)nh * hd * (gated ? 2 : 1), d.hidden); q8swap(d.gate, d.gate_t, d.ffn, d.hidden); q8swap(d.up, d.up_t, d.ffn, d.hidden);
    const uint16_t* kp[16]; const uint16_t* vp[16];
    if (nkv > 16) return gpu::nx_decline("nkv > 16");
    for (int h = 0; h < nkv; ++h) { kp[h] = kvcache_.k_ptr(l, h); vp[h] = kvcache_.v_ptr(l, h); }
    d.k_host = kp; d.v_host = vp;
    d.host_tokens = (uint32_t)kvcache_.n_tokens();
    d.host_synced = (uint32_t)std::min<size_t>(kvcache_.mirror_valid(l), (size_t)pos);
    std::vector<float>& kvo = q35_kv_out_;
    kvo.resize((size_t)2 * nkv * hd);
    uint32_t mirror_valid = 0;
    if (async) {
        if (q35_chain_kv_.size() >= (size_t)gpu::VulkanBackend::kChainKvSlots) return gpu::nx_decline("q35_chain_kv_.size() >= (size_t)gpu::VulkanBackend::kChainKvSlots");          // ring full: caller flushes, then this layer runs synchronously
        const int slot = (int)q35_chain_kv_.size();
        if (!g.q35_full_layer(d, x, nullptr, &mirror_valid, slot)) return gpu::nx_decline("!g.q35_full_layer(d, x, nullptr, &mirror_valid, slot)");
        q35_chain_kv_.push_back({l, pos, nkv, hd, mirror_valid, slot});                                // host store happens in q35_chain_flush()
        return true;
    }
    if (!g.q35_full_layer(d, x, kvo.data(), &mirror_valid)) return gpu::nx_decline("!g.q35_full_layer(d, x, kvo.data(), &mirror_valid)");
    kvcache_.store(l, pos, kvo.data(), kvo.data() + (size_t)nkv * hd, nkv * hd);     // host cache stays the source of truth (sessions, CPU fallback, MTP)
    kvcache_.set_mirror_valid(l, mirror_valid);
    return true;
#else
    (void)lw; (void)l; (void)x; (void)pos; (void)async; return false;
#endif
}

inline void Transformer::q35_chain_flush(float* x)
{
#ifdef NEXUS_VULKAN
    gpu::VulkanBackend& g = gpu::gpu();
    if (g.chain_is_open()) g.chain_end(x, (size_t)meta_.hidden_dim);
    for (const Q35ChainKv& c : q35_chain_kv_) {
        const float* kv = g.chain_kv_ptr(c.slot);
        if (!kv) continue;
        kvcache_.store(c.layer, c.pos, kv, kv + (size_t)c.nkv * c.hd, c.nkv * c.hd);
        kvcache_.set_mirror_valid(c.layer, c.mirror_valid);
    }
    q35_chain_kv_.clear();
#else
    (void)x;
#endif
}

inline void Transformer::forward_layer(int l, float* x, int pos) {
    const TransformerLayerWeights& lw = layer_weights_[l];
    const int hidden = (int)meta_.hidden_dim;

    // NEXUS_DECODE_PROF=1: wall time per layer kind, printed every 8 layer-0 calls (= tokens); NEXUS_DECODE_PROF=N changes the interval.
    struct LayerProf {
        bool on; int kind; std::chrono::steady_clock::time_point t0;
        static double* acc() { static double a[4] = {0, 0, 0, 0}; return a; }
        static int interval() { static int v = [] { const char* e = std::getenv("NEXUS_DECODE_PROF"); int n = e ? std::atoi(e) : 0; return n > 1 ? n : 8; }(); return v; }
        static int* cnt() { static int c[5] = {0, 0, 0, 0, 0}; return c; }       // [0..3] layers per kind, [4] tokens
        LayerProf(bool o, int k, int layer) : on(o), kind(k) {
            if (!on) return;
            if (layer == 0 && cnt()[4] >= interval()) {
                const int n = cnt()[4]; const char* nm[4] = {"linear", "fullattn", "dense", "moe"};
                fprintf(stderr, "[decode-prof] per token over %d tokens:", n);
                double tot = 0; for (int i = 0; i < 4; ++i) tot += acc()[i];
                for (int i = 0; i < 4; ++i) if (cnt()[i]) fprintf(stderr, "  %s %.2f ms (%d layers, %.2f ms each)", nm[i], acc()[i] / n, cnt()[i] / n, acc()[i] / cnt()[i]);
                { NxProfExtra& e = nx_prof_extra(); fprintf(stderr, "  | layers total %.2f ms (loop %.2f)  embed %.2f  lm_head+norm %.2f  sampling %.2f ms\n", tot / n, e.loop / n, e.embed / n, e.head / n, e.samp / n); e = NxProfExtra(); }
                fflush(stderr);
                for (int i = 0; i < 4; ++i) { acc()[i] = 0; cnt()[i] = 0; } cnt()[4] = 0;
            }
            if (layer == 0) cnt()[4]++;
            t0 = std::chrono::steady_clock::now();
        }
        ~LayerProf() { if (!on) return; acc()[kind] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); cnt()[kind]++; }
    };
    static const bool nx_dec_prof = std::getenv("NEXUS_DECODE_PROF") != nullptr;
    LayerProf nx_layer_prof(nx_dec_prof, lw.qwen35_linear ? 0 : (lw.qwen35_full_attn_q_has_gate ? 1 : (lw.moe ? 3 : 2)), l);

    // ------ DIAG: per-layer residual stream trace -------------------------
    // Fires for every layer on the FIRST token only (pos == 0), then never
    // again. Prints rms / max / outlier-position of the residual stream
    // BEFORE and AFTER this layer's contribution. This pinpoints exactly
    // which layer first injects an outlier into the residual stream.
    //
    // Output format:
    //   [diag-layer]   L=NN  IN  rms=X max=Y at_idx=Z  type=linear|fullattn|dense
    //   [diag-layer]   L=NN  OUT rms=X max=Y at_idx=Z  delta_rms=±Δ delta_max=±Δ
    //
    // To find the bad layer:
    //   1. Find the first layer where OUT max jumps far above IN max
    //   2. That layer's sub-layer (linear, fullattn, or dense) is broken
    //
    // If at_idx changes between layers, the model is producing different
    // outliers; if it stays the same, an outlier from one specific embedding
    // dim is being amplified through unrelated layers.
    auto layer_diag_stats = [](const float* v, int n,
                               float& rms_out, float& abs_max_out, int& abs_max_idx_out,
                               int& nan_count, int& inf_count) {
        rms_out = 0.0f;
        abs_max_out = 0.0f;
        abs_max_idx_out = 0;
        nan_count = 0;
        inf_count = 0;
        if (n <= 0 || !v) return;
        double sumsq = 0.0;
        int n_finite = 0;
        for (int i = 0; i < n; ++i) {
            const float xi = v[i];
            if (std::isnan(xi))      { ++nan_count; continue; }
            if (std::isinf(xi))      { ++inf_count; continue; }
            ++n_finite;
            sumsq += (double)xi * (double)xi;
            const float a = std::fabs(xi);
            if (a > abs_max_out) {
                abs_max_out = a;
                abs_max_idx_out = i;
            }
        }
        if (n_finite > 0) {
            rms_out = (float)std::sqrt(sumsq / (double)n_finite);
        }
    };

    const bool diag_this_layer = nx_qwen3next_diag_enabled() && (pos == 0);

    float in_rms = 0.0f, in_max = 0.0f;
    int   in_max_idx = 0, in_nan = 0, in_inf = 0;
    if (diag_this_layer) {
        layer_diag_stats(x, hidden, in_rms, in_max, in_max_idx, in_nan, in_inf);

        // Identify the sub-layer kind for this layer.
        const char* kind;
        if (lw.qwen35_linear) kind = "linear ";
        else if (lw.qwen35_full_attn_q_has_gate) kind = "fullatt";
        else if (lw.moe) kind = "moeattn";
        else kind = "dense  ";

        fprintf(stderr,
            "[diag-layer]   L=%2d IN  rms=%+.3e max=%+.3e at_idx=%-5d nan=%d inf=%d kind=%s\n",
            l, in_rms, in_max, in_max_idx, in_nan, in_inf, kind);
        fflush(stderr);
    }
    // ------ END DIAG (head) ----------------------------------------------

    AttentionConfig layer_attn_cfg = attn_cfg_;

    if (lw.attn_q_dim_layer > 0 &&
        lw.attn_kv_dim_layer > 0 &&
        lw.attn_head_dim_layer > 0 &&
        lw.attn_heads_layer > 0 &&
        lw.attn_kv_heads_layer > 0)
    {
        layer_attn_cfg.n_heads    = lw.attn_heads_layer;
        layer_attn_cfg.n_kv_heads = lw.attn_kv_heads_layer;
        layer_attn_cfg.head_dim   = lw.attn_head_dim_layer;
    }

    // ─── Gemma 4: per-layer SWA + RoPE overrides ───────────────────────────
    // For Gemma 4 local/global alternation. Non-Gemma archs leave these zero
    // and the original AttentionConfig values pass through unchanged.
    if (lw.sw_window_size > 0) {
        layer_attn_cfg.sliding_window = lw.sw_window_size;
    }
    if (lw.rope_freq_base_layer > 0.0f) {
        layer_attn_cfg.rope.freq_base = lw.rope_freq_base_layer;
    }

    std::string arch_l_attn_scale = meta_.arch;
    for (char& c : arch_l_attn_scale) {
        c = (char)std::tolower((unsigned char)c);
    }

    const bool is_gemma4_attn_scale =
        arch_l_attn_scale.find("gemma4") != std::string::npos;

    // Gemma 4 uses a non-standard attention scale based on query_pre_attn_scalar,
    // NOT 1/sqrt(head_dim). For 31B this scalar is 256, giving scale=1/sqrt(256)
    // = 0.0625. For local layers head_dim=256 this coincidentally equals
    // 1/sqrt(head_dim), but for global layers head_dim=512, scale would be
    // 1/sqrt(512)=0.0442 with the default — too small by a factor of ~1.4×.
    //
    // Previous code set scale=1.0, which is wrong: pre-softmax dot products
    // are 16× too large, softmax collapses to one-hot per token, attention
    // output ≈ V[argmax], Wo amplifies → final logits all saturate at
    // softcap=30 (every token rounding to the same Wo-projected vector).
    //
    // Gemma 4 attention scale = 1.0 (NOT 1/sqrt(head_dim), NOT 1/sqrt(qpas)).
    //
    // Reference: ollama gemma4/model_text.go SelfAttention.Forward:
    //   attention := nn.Attention(ctx, q, k, v, 1.0, cache)
    //
    // Q and K are RMS-normalized to unit magnitude by q_norm/k_norm BEFORE
    // the dot product, which acts as the scaling. Adding another factor of
    // 1/sqrt(head_dim) or 1/sqrt(qpas) on top of that over-scales.
    if (is_gemma4_attn_scale) {
        layer_attn_cfg.scale = 1.0f;
    }

    NX_TRACE_DECISION(l, "attn.scale_effective", "%.8g",
        layer_attn_cfg.scale != 0.0f
            ? layer_attn_cfg.scale
            : 1.0f / std::sqrt((float)layer_attn_cfg.head_dim));

    NX_TRACE_DECISION(l, "attn.scale_source", "%s",
        is_gemma4_attn_scale ? "gemma4_qknorm_unit_scale" : "default");

    const int q_dim  = layer_attn_cfg.n_heads    * layer_attn_cfg.head_dim;
    const int kv_dim = layer_attn_cfg.n_kv_heads * layer_attn_cfg.head_dim;

    float* xb = xb_scratch_.data();
    
    #ifndef NEXUS_TRACE_LAYER
    constexpr bool trace_layer = false;
    #else
    static std::atomic<int> layer_trace_count{0};
    const bool trace_layer = (layer_trace_count.fetch_add(1) < 64);
    #endif

    auto trace = [&](const char* phase) {
        if (!trace_layer) return;
        fprintf(stderr,
            "[nexus] TRACE layer=%d pos=%d phase=%s moe=%d fused_qkv=%d q35=%d "
            "attn_norm=%p ffn_norm=%p qkv=%p q=%p k=%p v=%p o=%p router=%p gate_exps=%p up_exps=%p down_exps=%p\n",
            l, pos, phase,
            lw.moe ? 1 : 0,
            lw.fused_qkv ? 1 : 0,
            lw.qwen35_linear ? 1 : 0,
            lw.attn_norm,
            lw.ffn_norm,
            lw.wqkv,
            lw.wq,
            lw.wk,
            lw.wv,
            lw.wo,
            lw.moe_router,
            lw.wgate_exps,
            lw.wup_exps,
            lw.wdown_exps);
        fflush(stderr);
    };

    auto is_native_nq_type = [](QuantType t) -> bool {
        return t == QuantType::NQ4_G64 ||
               t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 ||
               t == QuantType::NQ8_G64;
    };

    auto mat_vec_runtime_safe = [&](const void* W,
                                    const float* vin,
                                    float* vout,
                                    size_t rows,
                                    size_t cols,
                                    QuantType wt,
                                    const char* tag)
    {
        if (!W || !vin || !vout || rows == 0 || cols == 0) {
            if (vout && rows) memset(vout, 0, rows * sizeof(float));
            return;
        }

        if (nx_q35_try_gpu_matvec(layer_card(lw), W, vin, vout, rows, cols, wt)) return;

        // NQ*_G64 is now supported by cpu::mat_vec itself. Keep GPU disabled
        // for native NQ above, but use the normal CPU matmul route here.
        cpu::mat_vec(W, vin, vout, rows, cols, wt, n_threads_);
        if (nx_qwen3next_diag_enabled() && is_native_nq_type(wt)) {
            static std::atomic<int> nq_dense_mainroute_logs{0};
            const int log_id = nq_dense_mainroute_logs.fetch_add(1, std::memory_order_relaxed);
            if (log_id < 96) {
                fprintf(stderr,
                    "[nexus] native NQ main-route matvec: layer=%d tag=%s rows=%zu cols=%zu type=%d threads=%d\n",
                    l, tag ? tag : "?", rows, cols, (int)wt, n_threads_);
                fflush(stderr);
            }
        }
        return;

        if (!is_native_nq_type(wt)) {
            cpu::mat_vec(W, vin, vout, rows, cols, wt, n_threads_);
            return;
        }

        static std::atomic<int> nq_dense_logs{0};
        const int log_id = nq_dense_logs.fetch_add(1);
        if (log_id < 96) {
            fprintf(stderr,
                "[nexus] native NQ safe matvec: layer=%d tag=%s rows=%zu cols=%zu type=%d\n",
                l, tag ? tag : "?", rows, cols, (int)wt);
            fflush(stderr);
        }

        const unsigned hw = std::thread::hardware_concurrency();
        int nt = n_threads_ > 1 ? n_threads_ : (hw ? (int)hw : 4);
        nt = std::max(1, std::min<int>(nt, (int)rows));
        if (log_id < 96) {
            fprintf(stderr,
                "[nexus] native NQ threaded matvec: layer=%d tag=%s rows=%zu cols=%zu type=%d threads=%d\n",
                l, tag ? tag : "?", rows, cols, (int)wt, nt);
            fflush(stderr);
        }

        std::atomic<size_t> next_row{0};
        std::vector<std::thread> workers;
        workers.reserve((size_t)nt);
        for (int ti = 0; ti < nt; ++ti) {
            workers.emplace_back([&]() {
                std::vector<float> row(cols);
                for (;;) {
                    const size_t r = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (r >= rows) break;
                    quant::dequantize_row(W, row.data(), r, cols, wt);
                    double acc = 0.0;
                    for (size_t c = 0; c < cols; ++c) {
                        acc += (double)row[c] * (double)vin[c];
                    }
                    vout[r] = (float)acc;
                }
            });
        }
        for (auto& th : workers) th.join();
    };

    const bool native_nq_attention_weights =
        (lw.fused_qkv && is_native_nq_type(lw.wqkv_type)) ||
        (!lw.fused_qkv && (
            is_native_nq_type(lw.wq_type) ||
            is_native_nq_type(lw.wk_type) ||
            is_native_nq_type(lw.wv_type))) ||
        is_native_nq_type(lw.wo_type);

    trace("enter");
    
    static std::atomic<int> gemma4_tied_kv_logs{0};
    if (lw.gemma4_tied_kv && gemma4_tied_kv_logs.fetch_add(1) < 16) {
        fprintf(stderr,
            "[nexus] Gemma4 runtime: layer %d using tied K/V projection\n",
            l);
        fflush(stderr);
    }    

    // Guard: should never happen after the stricter load checks, but be safe
    if (!lw.attn_norm ||
    (!lw.qwen35_linear &&
     (
        (lw.fused_qkv && (!lw.wqkv || !lw.wo)) ||
        (!lw.fused_qkv && (!lw.wq || !lw.wk || (!lw.wv && !lw.gemma4_tied_kv) || !lw.wo))
     )))
    {
        fprintf(stderr,
            "[nexus] BUG: layer %d has null attention weights "
            "(fused=%d qkv=%p q=%p k=%p v=%p o=%p norm=%p), skipping\n",
            l,
            lw.fused_qkv ? 1 : 0,
            lw.wqkv,
            lw.wq,
            lw.wk,
            lw.wv,
            lw.wo,
            lw.attn_norm);
        fflush(stderr);
        return;
    }

    // Qwen3.5 linear-attention layers: the whole layer (incl. its FFN) as ONE GPU submit, recurrent state kept on the card.
    if (lw.qwen35_linear && q35_gpu_linear_step(lw, l, x)) return;
    if (!lw.qwen35_linear && !lw.moe && q35_gpu_full_step(lw, l, x, pos)) return;   // full-attention layer (Qwen3.5 gated or plain): whole layer on the GPU with a KV mirror

    // ------ Batched GPU layer path ------------------------------------------------------------------------------------------------------------------------------------------------
    // When GPU activation buffers are ready, run all GPU ops as 2 submits per
    // layer instead of 5-6. This eliminates ~80% of fence-wait overhead.
    const bool try_gpu  = (lw.gpu_layer  && gpu::gpu().ready());
    const bool try_gpu2 = gpu2_ready_for(lw);
    gpu::VulkanBackend& gbatch = (try_gpu2 && !try_gpu) ? *backend2_for(lw) : gpu::gpu();
    // MoE FFN is CPU-first for now. Attention can still use the non-batched GPU path,
    // but the fused/batched GPU FFN recorder assumes dense wgate/wup/wdown.
    // Qwen3.5 hybrid layers (gated-deltanet linear attention, and full-attention layers with the q-gate / partial rope) are NOT
    // handled by the generic batched recipe below.  NQ models never reach it (native_nq_attention_weights), but plain GGUF Qwen3.5 did:
    // it ran the full-attention recipe on linear layers (-> "tried to store K/V for layer 0") and decoded garbage on the GPU.
    const bool q35_hybrid_layer = lw.qwen35_linear || lw.qwen35_full_attn_q_has_gate;
    const bool can_batch = !native_nq_attention_weights && !q35_hybrid_layer && !lw.moe && (try_gpu || try_gpu2) && gbatch.act_bufs_ok()
        && gbatch.pipe_rms_ok() && gbatch.pipe_rope_ok()
        && gbatch.pipe_silu_ok() && gbatch.pipe_add_ok();


    if (can_batch) {
        const size_t w_attn_bytes = (size_t)meta_.hidden_dim * sizeof(float);

        // ------ Submit 1: RMS norm + QKV + RoPE ------------------------------------------------------------------------------------------------------
        gbatch.begin_layer_record(x, hidden);

        // RMS norm (attn)
        gbatch.record_rms_norm_g(
            gbatch.act_hidden_buf(), lw.attn_norm, w_attn_bytes,
            gbatch.act_xb_buf(), (uint32_t)hidden, meta_.rms_norm_eps);
        gbatch.layer_barrier();

        // QKV projections
        trace("before_qkv_projection");
        if (lw.fused_qkv) {
            const int qkv_total = q_dim + 2 * kv_dim;
            // Fused: project to a combined q+k+v buffer, then split
            // We use act_q_ as the full qkv_total output (must be large enough)
            gbatch.record_mat_vec_g(lw.wqkv, lw.wqkv_type,
                gbatch.act_xb_buf(), gbatch.act_q_buf(), qkv_total, hidden);
        } else {
            gbatch.record_mat_vec_g(lw.wq, lw.wq_type,
                gbatch.act_xb_buf(), gbatch.act_q_buf(), q_dim, hidden);
            gbatch.layer_barrier();
            // Gemma 4 Shared KV Cache: skip Wk/Wv on shared-KV layers
            // (these tensors are null in the GGUF; we'll read from donor).
            if (lw.kv_donor_layer < 0) {
                gbatch.record_mat_vec_g(lw.wk, lw.wk_type,
                    gbatch.act_xb_buf(), gbatch.act_k_buf(), kv_dim, hidden);
                gbatch.layer_barrier();
                gbatch.record_mat_vec_g(lw.wv, lw.wv_type,
                    gbatch.act_xb_buf(), gbatch.act_v_buf(), kv_dim, hidden);
            }
        }
        gbatch.layer_barrier();

        // RoPE on Q and K
        gbatch.record_rope_g(
            gbatch.act_q_buf(), lw.fused_qkv ? gbatch.act_q_buf() : gbatch.act_k_buf(),
            (uint32_t)layer_attn_cfg.n_heads, (uint32_t)layer_attn_cfg.n_kv_heads,
            (uint32_t)layer_attn_cfg.head_dim, (uint32_t)pos,
            layer_attn_cfg.rope.freq_base,
            layer_attn_cfg.rope.scale != 0.f ? layer_attn_cfg.rope.scale : 1.f,
            (uint32_t)(layer_attn_cfg.rope.layout == 1 ? 1 : 0));   // half-split (NeoX) archs rotate (i, i+hd/2), not (2i, 2i+1)

        // Submit batch 1, wait --- CPU needs results for KV store
        gbatch.partial_submit();

        // CPU: store K+V into KV cache from GPU activation buffers
        // Gemma 4 Shared KV Cache: skip store on shared-KV layers and read
        // from the donor instead.
        const int kv_layer = (lw.kv_donor_layer >= 0)
                                ? lw.kv_donor_layer
                                : l;
        if (lw.kv_donor_layer < 0) {
            const float* k_data = lw.fused_qkv
                ? (const float*)gbatch.act_q_buf().mapped_ptr + q_dim
                : (const float*)gbatch.act_k_buf().mapped_ptr;
            const float* v_data = lw.fused_qkv
                ? (const float*)gbatch.act_q_buf().mapped_ptr + q_dim + kv_dim
                : (const float*)gbatch.act_v_buf().mapped_ptr;
            kvcache_.store(l, pos, k_data, v_data, kv_dim);
        } else {
            NX_TRACE_DECISION(l, "shared_kv.skip_store_gpu", "donor=L%d", lw.kv_donor_layer);
        }

        // Pack KV for attention shader
        const int n_ctx = pos + 1;
        const size_t kv_b = (size_t)layer_attn_cfg.n_kv_heads * n_ctx * layer_attn_cfg.head_dim;
        kv_flat_k_.resize(kv_b); kv_flat_v_.resize(kv_b);
        kvcache_.pack_k(kv_layer, n_ctx, kv_flat_k_.data());
        kvcache_.pack_v(kv_layer, n_ctx, kv_flat_v_.data());

        // ------ Submit 2: attention + Wo + residual + FFN ------------------------------------------------------------------------
        // layer_recording_ is still true after partial_submit --- continue recording

        // Attention: GPU shader if available, otherwise CPU into mapped output buffer
        bool attn_recorded = gbatch.record_attention_g(
            gbatch.act_q_buf(), gbatch.act_attn_out_buf(),
            layer_attn_cfg.n_heads, layer_attn_cfg.n_kv_heads, layer_attn_cfg.head_dim, n_ctx,
            kv_flat_k_.data(), kv_flat_v_.data(),
            kv_b * sizeof(uint16_t));
        if (!attn_recorded) {
            // GPU attention shader not compiled --- submit what's recorded so far,
            // run CPU attention directly into the mapped output buffer, then
            // restart recording for Wo + FFN.
            gbatch.partial_submit();
            const float* q_cpu = (const float*)gbatch.act_q_buf().mapped_ptr;
            float*       o_cpu = (float*)gbatch.act_attn_out_buf().mapped_ptr;
            if (q_cpu && o_cpu) {
                AttentionLayer attn_tmp;
                if (lw.fused_qkv)
                    attn_tmp.set_fused_qkv(lw.wqkv, lw.wqkv_type, lw.wo, lw.wo_type,
                                           lw.bq, lw.bk, lw.bv);
                else
                    attn_tmp.set_weights(lw.wq, lw.wq_type, lw.wk, lw.wk_type,
                                        lw.wv, lw.wv_type, lw.wo, lw.wo_type,
                                        lw.bq, lw.bk, lw.bv);
                attn_tmp.forward_cached_scores(q_cpu, o_cpu, kvcache_, l, pos, layer_attn_cfg);
            }
        } else {
            gbatch.layer_barrier();
        }

        // Wo projection: attn_out --- ffn_out_buf (reuse as temp)
        const int wo_in = q_dim;
        gbatch.record_mat_vec_g(lw.wo, lw.wo_type,
            gbatch.act_attn_out_buf(), gbatch.act_ffn_out_buf(), hidden, wo_in);
        gbatch.layer_barrier();

        // Residual add: hidden += wo_out
        gbatch.record_add_g(gbatch.act_hidden_buf(), gbatch.act_ffn_out_buf(), hidden);
        gbatch.layer_barrier();

        // RMS norm (FFN)
        const size_t w_ffn_bytes = (size_t)meta_.hidden_dim * sizeof(float);
        gbatch.record_rms_norm_g(
            gbatch.act_hidden_buf(), lw.ffn_norm, w_ffn_bytes,
            gbatch.act_xb_buf(), (uint32_t)hidden, meta_.rms_norm_eps);
        gbatch.layer_barrier();

        // FFN gate + up projections
        const int ffn = (int)meta_.ffn_dim;
        if (lw.fused_gate_up) {
            gbatch.record_mat_vec_g(lw.wup, lw.wup_type,
                gbatch.act_xb_buf(), gbatch.act_gate_buf(), ffn * 2, hidden);
        } else {
            gbatch.record_mat_vec_g(lw.wgate, lw.wgate_type,
                gbatch.act_xb_buf(), gbatch.act_gate_buf(), ffn, hidden);
            gbatch.layer_barrier();
            gbatch.record_mat_vec_g(lw.wup, lw.wup_type,
                gbatch.act_xb_buf(), gbatch.act_up_buf(), ffn, hidden);
        }
        gbatch.layer_barrier();

        // SiLU activation: gate = silu(gate) -- up
        gbatch.record_silu_mul_g(gbatch.act_gate_buf(), gbatch.act_up_buf(), ffn);
        gbatch.layer_barrier();

        // FFN down projection
        gbatch.record_mat_vec_g(lw.wdown, lw.wdown_type,
            gbatch.act_gate_buf(), gbatch.act_ffn_out_buf(), hidden, ffn);
        gbatch.layer_barrier();

        // Residual add: hidden += ffn_out
        gbatch.record_add_g(gbatch.act_hidden_buf(), gbatch.act_ffn_out_buf(), hidden);

        // Submit batch 2, wait, download x
        gbatch.end_layer_record(x, hidden);
        return;  // done --- skip old path below
    }

    // ------ Attention sub-layer ---------------------------------------------------------------------------------------------------------------------------------------------------------

    NX_TRACE_SAMPLE(l, "pre_attn_norm_in", x, (size_t)hidden);

    // a. Pre-attention norm: PhiMoE/SlimMoE uses affine LayerNorm; Llama-style uses RMSNorm.
    nx_apply_model_norm(x, lw.attn_norm, lw.attn_norm_bias, xb,
                    (size_t)hidden, meta_.rms_norm_eps,
                    lw.attn_norm_zc);
    
    trace("after_attn_norm");

    NX_TRACE_SAMPLE(l, "post_attn_norm_out", xb, (size_t)hidden);
    
    if (lw.qwen35_linear) {
        forward_qwen35_linear(lw, xb, ffn_scratch_.data(), l, pos);
        memcpy(xb, ffn_scratch_.data(), (size_t)hidden * sizeof(float));

        cpu::add_inplace(x, xb, (size_t)hidden);

        nx_apply_model_norm(x, lw.ffn_norm, lw.ffn_norm_bias, xb,
                    (size_t)hidden, meta_.rms_norm_eps,
                    lw.ffn_norm_zc);
        forward_ffn(lw, xb);
        cpu::add_inplace(x, xb, (size_t)hidden);
        return;
    }

    // b. Self-attention / mixer attention.
    //
    // Important:
    //   attn_out is q_dim, NOT hidden. Some models have q_dim != hidden
    //   (Qwen3.6: q_dim=4096, hidden=2048). Never write attention output into xb.
    gpu::VulkanBackend& active_gpu = (try_gpu2 && !try_gpu) ? *backend2_for(lw) : gpu::gpu();
    const bool use_any_gpu = try_gpu || try_gpu2;

    float* q        = qkv_scratch_.data();
    float* k        = q + q_dim;
    float* v        = k + kv_dim;
    float* attn_out = attn_scratch_.data();          // q_dim floats
    float* gate_in  = attn_out + q_dim;              // hidden floats
    float* gate_tmp = qkv_scratch_.data();           // reuse after attention

    std::vector<float> qwen35_full_attn_gate;
    const float* qwen35_full_attn_gate_ptr = nullptr;

    if (lw.attn_gate) {
        memcpy(gate_in, xb, (size_t)hidden * sizeof(float));
    }

    bool projected = false;

    if (lw.fused_qkv) {
        const int qkv_dim = q_dim + 2 * kv_dim;

        if (use_any_gpu && !is_native_nq_type(lw.wqkv_type)) {
            projected = active_gpu.mat_vec(lw.wqkv, xb, q,
                                           (size_t)qkv_dim,
                                           (size_t)hidden,
                                           lw.wqkv_type);
        }

        if (!projected) {
            mat_vec_runtime_safe(lw.wqkv, xb, q,
                                 (size_t)qkv_dim,
                                 (size_t)hidden,
                                 lw.wqkv_type,
                                 "fused_qkv");
            projected = true;
        }
    } else {
        bool qok = false, kok = false, vok = false;

        if (use_any_gpu) {
            // If Qwen3-Next packed query+gate into attn_q.weight, do Q on CPU
            // so we can request q_dim*2 rows and keep the second half.
            if (!lw.qwen35_full_attn_q_has_gate && !is_native_nq_type(lw.wq_type)) {
                qok = active_gpu.mat_vec(lw.wq, xb, q,
                                         (size_t)q_dim, (size_t)hidden, lw.wq_type);
            }

            if (!is_native_nq_type(lw.wk_type)) {
                kok = active_gpu.mat_vec(lw.wk, xb, k,
                                         (size_t)kv_dim, (size_t)hidden, lw.wk_type);
            }
            if (!is_native_nq_type(lw.wv_type)) {
                vok = active_gpu.mat_vec(lw.wv, xb, v,
                                         (size_t)kv_dim, (size_t)hidden, lw.wv_type);
            }
        }

        if (!qok) {
            if (lw.qwen35_full_attn_q_has_gate) {
                std::vector<float> q_and_gate((size_t)q_dim * 2, 0.0f);

                mat_vec_runtime_safe(lw.wq, xb, q_and_gate.data(),
                                     (size_t)q_dim * 2, (size_t)hidden,
                                     lw.wq_type,
                                     "q_gate");

                qwen35_full_attn_gate.assign((size_t)q_dim, 0.0f);

                const int q_heads = layer_attn_cfg.n_heads;
                const int q_hd    = layer_attn_cfg.head_dim;

                if (q_heads <= 0 || q_hd <= 0 || q_heads * q_hd != q_dim) {
                    fprintf(stderr,
                        "[nexus] FATAL-RUNTIME: bad Qwen35 full-attn q/gate split dims heads=%d head_dim=%d q_dim=%d\n",
                        q_heads, q_hd, q_dim);
                    std::fill(q, q + q_dim, 0.0f);
                    std::fill(qwen35_full_attn_gate.begin(), qwen35_full_attn_gate.end(), 0.0f);
                } else {
                    // Q+gate split layout for Qwen3-Next gated full-attention.
                    //
                    // HF reference does:
                    //   q_proj(x).view(..., n_heads, head_dim * 2).chunk(2, dim=-1)
                    //
                    // That `.view(..., n_heads, head_dim * 2)` reshape interleaves
                    // q and gate PER HEAD, so the natural row order is:
                    //   PERHEAD: [h0_q][h0_gate][h1_q][h1_gate]...
                    //
                    // FLAT [all_q][all_gate] is kept ONLY as an experimental
                    // fallback for debugging. It is NOT the HF layout.
                    //
                    // Switch via NEXUS_QWEN35_QGATE_LAYOUT env var:
                    //   "flat"    -> use FLAT (debugging only, NOT correct for HF GGUFs)
                    //   "perhead" -> use PERHEAD (default, matches HF)
                    //   unset     -> default = PERHEAD
// Q+gate split layouts. Three candidates:
                    //   0 = flat        : [all_q (q_dim)][all_gate (q_dim)]
                    //   1 = perhead     : [h0_q][h0_gate][h1_q][h1_gate]...
                    //   2 = perhead_swap: [h0_gate][h0_q][h1_gate][h1_q]...   (DEFAULT)
                    //
                    // perhead had Q stats centered but gate stats heavily
                    // negative (mean ~-3) — sigmoid was suppressing nearly all
                    // attention. perhead_swap tests whether q and gate halves
                    // are simply assigned backward inside each head.
                    static int qgate_layout_cached = -1;
                    if (qgate_layout_cached < 0) {
                        const char* env = std::getenv("NEXUS_QWEN35_QGATE_LAYOUT");
                        if      (env && std::strcmp(env, "flat") == 0)         qgate_layout_cached = 0;
                        else if (env && std::strcmp(env, "perhead_swap") == 0) qgate_layout_cached = 2;
                        else                                                    qgate_layout_cached = 1; // perhead default
                        const char* layout_name =
                            qgate_layout_cached == 0 ? "flat[all_q|all_gate]" :
                            qgate_layout_cached == 1 ? "perhead[h0q|h0g|h1q|h1g]" :
                                                       "perhead_swap[h0g|h0q|h1g|h1q]";
                        fprintf(stderr,
                            "[nexus] Qwen35 full-attn Q+gate layout: %s "
                            "(NEXUS_QWEN35_QGATE_LAYOUT=%s)\n",
                            layout_name,
                            env ? env : "(unset, default=perhead)");
                        fflush(stderr);
                    }

if (qgate_layout_cached == 0) {
                        // FLAT: q_and_gate = [Q_0..Q_{q_dim-1}][G_0..G_{q_dim-1}]
                        memcpy(q,
                               q_and_gate.data(),
                               (size_t)q_dim * sizeof(float));
                        memcpy(qwen35_full_attn_gate.data(),
                               q_and_gate.data() + (size_t)q_dim,
                               (size_t)q_dim * sizeof(float));
                    } else if (qgate_layout_cached == 1) {
                        // PERHEAD: within each head, q first then gate.
                        for (int h = 0; h < q_heads; ++h) {
                            const size_t src_head = (size_t)h * 2 * q_hd;
                            const size_t dst_head = (size_t)h * q_hd;
                            memcpy(q + dst_head,
                                   q_and_gate.data() + src_head,
                                   (size_t)q_hd * sizeof(float));
                            memcpy(qwen35_full_attn_gate.data() + dst_head,
                                   q_and_gate.data() + src_head + q_hd,
                                   (size_t)q_hd * sizeof(float));
                        }
                    } else {
                        // PERHEAD_SWAP: within each head, gate first then q.
                        for (int h = 0; h < q_heads; ++h) {
                            const size_t src_head = (size_t)h * 2 * q_hd;
                            const size_t dst_head = (size_t)h * q_hd;
                            memcpy(qwen35_full_attn_gate.data() + dst_head,
                                   q_and_gate.data() + src_head,
                                   (size_t)q_hd * sizeof(float));
                            memcpy(q + dst_head,
                                   q_and_gate.data() + src_head + q_hd,
                                   (size_t)q_hd * sizeof(float));
                        }
                    }

                    // ------ DIAG: dump first full-attn layer Q+gate intermediates ------
                    static std::atomic<int> qfa_diag_count{0};
                    if (qfa_diag_count.fetch_add(1) == 0) {
                        auto qfa_stats = [](const char* tag, const float* v, int n) {
                            if (n <= 0 || !v) {
                                fprintf(stderr, "[diag-qfa] %s: n=%d (skipped)\n", tag, n);
                                return;
                            }
                            float mn = v[0], mx = v[0], sum = 0.0f, sumsq = 0.0f;
                            int n_nan = 0, n_inf = 0;
                            for (int i = 0; i < n; ++i) {
                                const float x = v[i];
                                if (std::isnan(x))      ++n_nan;
                                else if (std::isinf(x)) ++n_inf;
                                else {
                                    if (x < mn) mn = x;
                                    if (x > mx) mx = x;
                                    sum += x; sumsq += x * x;
                                }
                            }
                            const int n_finite = n - n_nan - n_inf;
                            const float mean = (n_finite > 0) ? sum / n_finite : 0.0f;
                            const float rms  = (n_finite > 0) ? sqrtf(sumsq / n_finite) : 0.0f;
                            fprintf(stderr,
                                "[diag-qfa] %-22s n=%-6d min=%+.4e max=%+.4e mean=%+.4e rms=%+.4e nan=%d inf=%d "
                                "first4=[%+.4e %+.4e %+.4e %+.4e]\n",
                                tag, n, mn, mx, mean, rms, n_nan, n_inf,
                                n >= 1 ? v[0] : 0.0f,
                                n >= 2 ? v[1] : 0.0f,
                                n >= 3 ? v[2] : 0.0f,
                                n >= 4 ? v[3] : 0.0f);
                        };

                        fprintf(stderr,
                            "[diag-qfa] === Full-attn Q+gate dump layer=%d q_heads=%d kv_heads=%d q_hd=%d q_dim=%d kv_dim=%d ===\n",
                            l, q_heads, layer_attn_cfg.n_kv_heads, q_hd, q_dim, kv_dim);
                        qfa_stats("input xb (post-attn-norm)", xb,                 hidden);
                        qfa_stats("q_and_gate (raw output)",   q_and_gate.data(),  q_dim * 2);
                        qfa_stats("q_and_gate first half",     q_and_gate.data(),                 q_dim);
                        qfa_stats("q_and_gate second half",    q_and_gate.data() + q_dim,         q_dim);
                        qfa_stats("q (after split)",           q,                  q_dim);
                        qfa_stats("gate (after split)",        qwen35_full_attn_gate.data(), q_dim);
                        qfa_stats("q head 0",                  q,                  q_hd);
                        qfa_stats("q head 1",                  q + q_hd,           q_hd);
                        qfa_stats("gate head 0",               qwen35_full_attn_gate.data(), q_hd);
                        qfa_stats("gate head 1",               qwen35_full_attn_gate.data() + q_hd, q_hd);
                        fflush(stderr);
                    }
                    // ------ END DIAG ------
                }

                qwen35_full_attn_gate_ptr = qwen35_full_attn_gate.data();
            } else {
                mat_vec_runtime_safe(lw.wq, xb, q,
                                     (size_t)q_dim, (size_t)hidden, lw.wq_type,
                                     "q_proj");
            }

            qok = true;
        }

        if (!kok) {
            // Gemma 4 Shared KV Cache: if this layer reuses K from a donor,
            // skip Wk projection entirely. We'll read K from the donor's slot
            // in the cache below. k buffer's contents are irrelevant — they
            // get overwritten before use.
            if (lw.kv_donor_layer >= 0) {
                NX_TRACE_DECISION(l, "shared_kv.skip_wk", "donor=L%d", lw.kv_donor_layer);
                kok = true;  // pretend it's done
            } else {
                mat_vec_runtime_safe(lw.wk, xb, k,
                                     (size_t)kv_dim, (size_t)hidden, lw.wk_type,
                                     "k_proj");
                kok = true;
            }
        }
        if (!vok) {
            // Same as K above.
            if (lw.kv_donor_layer >= 0) {
                NX_TRACE_DECISION(l, "shared_kv.skip_wv", "donor=L%d", lw.kv_donor_layer);
                vok = true;
            } else {
                mat_vec_runtime_safe(lw.wv, xb, v,
                                     (size_t)kv_dim, (size_t)hidden, lw.wv_type,
                                     "v_proj");
                vok = true;
            }
        }

        projected = qok && kok && vok;
    }

    if (!projected) {
        fprintf(stderr, "[nexus] FATAL-RUNTIME: layer %d failed QKV projection\n", l);
        memset(xb, 0, (size_t)hidden * sizeof(float));
    } else {
        trace("after_qkv_projection");
        // Qwen3-Next gated full-attn layers (the ones with Q+gate packed
        // into attn_q.weight) use ZERO-CENTERED QK-Norm — i.e. apply
        // (1 + weight) to the per-head RMS-normed Q/K, not plain weight.
        // Reference: "Gated Attention" paper (Qwen team, NeurIPS 2025);
        // Qwen3-Next architecture spec describes these layers as having
        // "a zero-centered QK-Norm variant instead of standard RMSNorm
        // for q and k".
        //
        // Plain Qwen3-MoE (no Q+gate packing) uses regular weight semantics.
        // The qwen35_full_attn_q_has_gate flag distinguishes the two cases:
        //   - true  -> Qwen3-Next gated full-attn -> zero-centered QK-Norm
        //   - false -> regular Qwen3-MoE / Qwen2 / etc -> plain weight norm

        // Gemma 4 unweighted V-norm.
        //
        // Reference: ollama gemma4/model_text.go SelfAttention.Forward:
        //   v = v.RMSNorm(ctx, nil, opts.eps)
        //
        // V passes through an unweighted RMSNorm before attention. Without
        // it, raw Wv output (rms ~22 at L=0) blows the attention output.
        std::string arch_l_v = meta_.arch;
        for (char& c : arch_l_v) c = (char)std::tolower((unsigned char)c);
        const bool is_gemma4_v_norm =
            arch_l_v.find("gemma4") != std::string::npos;
        const bool gemma4_v_unweighted_norm =
            is_gemma4_v_norm && lw.kv_donor_layer < 0;

        const bool qk_norm_zero_centered = lw.qwen35_full_attn_q_has_gate;

        NX_TRACE_DECISION(l, "gemma4_v_unweighted_norm", "%d", gemma4_v_unweighted_norm ? 1 : 0);

        NX_TRACE_DECISION(l, "qk_norm_zero_centered_passed", "%d", qk_norm_zero_centered ? 1 : 0);
        NX_TRACE_DECISION(l, "qkv_extras.q_dim",  "%d", q_dim);
        NX_TRACE_DECISION(l, "qkv_extras.kv_dim", "%d", kv_dim);
        NX_TRACE_DECISION(l, "qkv_extras.eps",    "%.2e", meta_.rms_norm_eps);
        NX_TRACE_SAMPLE(l, "q_before_qkv_extras", q, (size_t)q_dim);
        NX_TRACE_SAMPLE(l, "k_before_qkv_extras", k, (size_t)kv_dim);
        NX_TRACE_SAMPLE(l, "v_before_qkv_extras", v, (size_t)kv_dim);

        nx_apply_qkv_extras(lw, layer_attn_cfg, q, k, v,
                            q_dim, kv_dim, meta_.rms_norm_eps,
                            gemma4_v_unweighted_norm);

        NX_TRACE_SAMPLE(l, "q_after_qkv_extras", q, (size_t)q_dim);
        NX_TRACE_SAMPLE(l, "k_after_qkv_extras", k, (size_t)kv_dim);
        NX_TRACE_SAMPLE(l, "v_after_qkv_extras", v, (size_t)kv_dim);

        const bool qwen35_partial_rope =
            lw.qwen35_full_attn_q_has_gate &&
            !lw.qwen35_linear;

        std::string arch_l_rope = meta_.arch;
        for (char& c : arch_l_rope) {
            c = (char)std::tolower((unsigned char)c);
        }

        const bool gemma4_global_prope =
            arch_l_rope.find("gemma4") != std::string::npos &&
            layer_attn_cfg.sliding_window == 0 &&
            layer_attn_cfg.head_dim == 512;

        NX_TRACE_DECISION(l, "rope.gemma4_global_prope", "%d", gemma4_global_prope ? 1 : 0);

        NX_TRACE_DECISION(l, "rope.layout_at_layer", "%d (%s)",
            layer_attn_cfg.rope.layout,
            layer_attn_cfg.rope.layout == 1 ? "half_split" : "interleaved");
        NX_TRACE_DECISION(l, "rope.freq_base_at_layer", "%.0f", layer_attn_cfg.rope.freq_base);
        NX_TRACE_DECISION(l, "rope.n_heads_q",  "%d", layer_attn_cfg.n_heads);
        NX_TRACE_DECISION(l, "rope.n_heads_kv", "%d", layer_attn_cfg.n_kv_heads);
        NX_TRACE_DECISION(l, "rope.head_dim",   "%d", layer_attn_cfg.head_dim);
        const int rope_pos = rope_effective_position(pos);
        NX_TRACE_DECISION(l, "rope.pos",        "%d", pos);
        NX_TRACE_DECISION(l, "rope.effective_pos", "%d", rope_pos);
        NX_TRACE_DECISION(l, "rope.scale",      "%.6f", rope_pos_scale_);
        NX_TRACE_DECISION(l, "rope.qwen35_partial", "%d", qwen35_partial_rope ? 1 : 0);
        NX_TRACE_DECISION(l, "rope.sliding_window", "%d", layer_attn_cfg.sliding_window);

        // Sample first head of Q and K so we can see the rotation effect.
        NX_TRACE_SAMPLE(l, "q_head0_pre_rope", q, (size_t)layer_attn_cfg.head_dim);
        NX_TRACE_SAMPLE(l, "k_head0_pre_rope", k, (size_t)layer_attn_cfg.head_dim);


        // Gemma 4 Shared KV Cache: don't RoPE K on shared layers. K is read
        // from the donor's cache, which was already RoPE'd when stored.
        if (gemma4_global_prope) {
            nx_apply_rope_gemma4_prope_or_full(q,
                                               layer_attn_cfg.n_heads,
                                               rope_pos,
                                               layer_attn_cfg.head_dim,
                                               layer_attn_cfg.rope,
                                               true,
                                               gemma4_rope_factors_,
                                               gemma4_rope_factors_len_);
        } else {
            nx_apply_rope_qwen35_partial_or_full(q,
                                                 layer_attn_cfg.n_heads,
                                                 rope_pos,
                                                 layer_attn_cfg.head_dim,
                                                 layer_attn_cfg.rope,
                                                 qwen35_partial_rope);
        }

        if (lw.kv_donor_layer < 0) {
            if (gemma4_global_prope) {
                nx_apply_rope_gemma4_prope_or_full(k,
                                                   layer_attn_cfg.n_kv_heads,
                                                   rope_pos,
                                                   layer_attn_cfg.head_dim,
                                                   layer_attn_cfg.rope,
                                                   true,
                                                   gemma4_rope_factors_,
                                                   gemma4_rope_factors_len_);
            } else {
                nx_apply_rope_qwen35_partial_or_full(k,
                                                     layer_attn_cfg.n_kv_heads,
                                                     rope_pos,
                                                     layer_attn_cfg.head_dim,
                                                     layer_attn_cfg.rope,
                                                     qwen35_partial_rope);
            }
        } else {
            NX_TRACE_DECISION(l, "shared_kv.skip_rope_k", "donor=L%d", lw.kv_donor_layer);
        }

        NX_TRACE_DECISION(l, "rope.gemma4_global_prope", "%d", gemma4_global_prope ? 1 : 0);

        NX_TRACE_SAMPLE(l, "q_head0_post_rope", q, (size_t)layer_attn_cfg.head_dim);
        NX_TRACE_SAMPLE(l, "k_head0_post_rope", k, (size_t)layer_attn_cfg.head_dim);
                              
        trace("before_kv_store");

        // Gemma 4 Shared KV Cache: do NOT store K/V on shared layers
        // (we didn't project them). Read from donor's slot instead.
        const int kv_layer = (lw.kv_donor_layer >= 0)
                                ? lw.kv_donor_layer
                                : l;
        if (lw.kv_donor_layer < 0) {
            kvcache_.store(l, pos, k, v, kv_dim);
        } else {
            NX_TRACE_DECISION(l, "shared_kv.skip_store", "donor=L%d", lw.kv_donor_layer);
        }

        const int n_ctx = pos + 1;
        const size_t kv_b =
            (size_t)layer_attn_cfg.n_kv_heads * n_ctx * layer_attn_cfg.head_dim;

        kv_flat_k_.resize(kv_b);
        kv_flat_v_.resize(kv_b);
        kvcache_.pack_k(kv_layer, n_ctx, kv_flat_k_.data());
        kvcache_.pack_v(kv_layer, n_ctx, kv_flat_v_.data());
        
        NX_TRACE_DECISION(l, "kv_flat.n_ctx", "%d", n_ctx);
        NX_TRACE_DECISION(l, "kv_flat.bytes", "%zu", kv_b * sizeof(uint16_t));
        NX_TRACE_DECISION(l, "kv_flat.source_layer", "%d %s",
                          kv_layer,
                          lw.kv_donor_layer >= 0 ? "(donor)" : "(own)");

        NX_TRACE_SAMPLE_F16(l, "k_flat_first_head", kv_flat_k_.data(),
                            (size_t)std::min(layer_attn_cfg.head_dim * 4, (int)kv_b));
        NX_TRACE_SAMPLE_F16(l, "v_flat_first_head", kv_flat_v_.data(),
                            (size_t)std::min(layer_attn_cfg.head_dim * 4, (int)kv_b));

        trace("before_attention");

        bool attn_ok = false;
        if (use_any_gpu) {
            attn_ok = active_gpu.attention_gpu(
                q, attn_out,
                layer_attn_cfg.n_heads,
                layer_attn_cfg.n_kv_heads,
                layer_attn_cfg.head_dim,
                l,
                n_ctx,
                kv_flat_k_.data(),
                kv_flat_v_.data(),
                kv_b * sizeof(uint16_t));
        }

        if (!attn_ok) {
            AttentionLayer attn_scores;
            attn_scores.forward_cached_scores(
                q,
                attn_out,
                kvcache_,
                l,
                pos,
                layer_attn_cfg);
        }
        
        trace("after_attention");

if (qwen35_full_attn_gate_ptr) {
            // ------ DIAG: dump attn output before/after gate on first call ----
            static std::atomic<int> qfa_gate_diag{0};
            const bool dump_gate = (qfa_gate_diag.fetch_add(1) == 0);
            if (dump_gate) {
                auto qfa_stats2 = [](const char* tag, const float* v, int n) {
                    float mn = v[0], mx = v[0], sumsq = 0.0f;
                    int n_nan = 0, n_inf = 0;
                    for (int i = 0; i < n; ++i) {
                        const float x = v[i];
                        if (std::isnan(x))      ++n_nan;
                        else if (std::isinf(x)) ++n_inf;
                        else { if (x < mn) mn = x; if (x > mx) mx = x; sumsq += x * x; }
                    }
                    const int nf = n - n_nan - n_inf;
                    fprintf(stderr,
                        "[diag-qfa] %-22s n=%-6d min=%+.4e max=%+.4e rms=%+.4e nan=%d inf=%d\n",
                        tag, n, mn, mx, nf > 0 ? sqrtf(sumsq / nf) : 0.0f, n_nan, n_inf);
                };
                qfa_stats2("attn_out PRE-gate", attn_out, q_dim);
            }

            for (int i = 0; i < q_dim; ++i) {
                const float z = qwen35_full_attn_gate_ptr[i];
                float s;
                if (z > 80.0f) s = 1.0f;
                else if (z < -80.0f) s = 0.0f;
                else s = 1.0f / (1.0f + expf(-z));
                attn_out[i] *= s;
            }

            if (dump_gate) {
                auto qfa_stats3 = [](const char* tag, const float* v, int n) {
                    float mn = v[0], mx = v[0], sumsq = 0.0f;
                    for (int i = 0; i < n; ++i) {
                        if (v[i] < mn) mn = v[i];
                        if (v[i] > mx) mx = v[i];
                        sumsq += v[i] * v[i];
                    }
                    fprintf(stderr,
                        "[diag-qfa] %-22s n=%-6d min=%+.4e max=%+.4e rms=%+.4e\n",
                        tag, n, mn, mx, sqrtf(sumsq / n));
                };
                qfa_stats3("attn_out POST-gate", attn_out, q_dim);
                fflush(stderr);
            }
            // ------ END DIAG ------
        } else {
            nx_apply_attn_gate_fast(
                lw,
                lw.attn_gate ? gate_in : nullptr,
                attn_out,
                gate_tmp,
                q_dim,
                hidden,
                n_threads_,
                use_any_gpu ? &active_gpu : nullptr);
        }
            
        trace("before_wo");

        bool wo_ok = false;
        if (use_any_gpu && !is_native_nq_type(lw.wo_type)) {
            wo_ok = active_gpu.mat_vec(lw.wo, attn_out, ffn_scratch_.data(),
                                       (size_t)hidden,
                                       (size_t)q_dim,
                                       lw.wo_type);
        }

        if (!wo_ok) {
            mat_vec_runtime_safe(lw.wo, attn_out, ffn_scratch_.data(),
                                 (size_t)hidden,
                                 (size_t)q_dim,
                                 lw.wo_type,
                                 "o_proj");
        }

        nx_add_bias_inplace(ffn_scratch_.data(), lw.bo, hidden);

        memcpy(xb, ffn_scratch_.data(), (size_t)hidden * sizeof(float));
        trace("after_wo");
    }

    // Post-attention norm (Gemma 2/3/4 sandwich norm). Applied to the
    // attention output AFTER Wo and BEFORE the residual add. nullptr for
    // every non-Gemma arch -> this branch is skipped entirely.
    NX_TRACE_SAMPLE(l, "attn_out_pre_post_norm", xb, (size_t)hidden);
    if (lw.attn_post_norm) {
        // Stage through ffn_scratch_ since nx_apply_model_norm needs distinct
        // input/output buffers. ffn_scratch_ is sized max(hidden, 2*ffn) so
        // it has room. We then memcpy the result back into xb.
        nx_apply_model_norm(xb, lw.attn_post_norm, nullptr,
                            ffn_scratch_.data(),
                            (size_t)hidden, meta_.rms_norm_eps,
                            lw.attn_post_norm_zc);
        memcpy(xb, ffn_scratch_.data(), (size_t)hidden * sizeof(float));
        trace("after_post_attn_norm");
        NX_TRACE_SAMPLE(l, "attn_out_post_post_norm", xb, (size_t)hidden);
    }

    // c. Residual
    NX_TRACE_SAMPLE(l, "x_before_attn_residual", x, (size_t)hidden);
    cpu::add_inplace(x, xb, (size_t)hidden);
    NX_TRACE_SAMPLE(l, "x_after_attn_residual", x, (size_t)hidden);

    // ------ FFN sub-layer -----------------------------------------------------

    // a. Pre-FFN norm: PhiMoE/SlimMoE uses affine LayerNorm; Llama-style uses RMSNorm.
    nx_apply_model_norm(x, lw.ffn_norm, lw.ffn_norm_bias, xb,
                    (size_t)hidden, meta_.rms_norm_eps,
                    lw.ffn_norm_zc);
    NX_TRACE_SAMPLE(l, "ffn_norm_out", xb, (size_t)hidden);

    // b. SwiGLU FFN
    trace("before_ffn");
    forward_ffn(lw, xb);  // writes result back into xb
    trace("after_ffn");
    NX_TRACE_SAMPLE(l, "ffn_out", xb, (size_t)hidden);

    // Post-FFN norm (Gemma 2/3/4 sandwich norm). Same pattern as post-attn
    // norm above. nullptr for non-Gemma -> skipped.
    //
    // Note: forward_ffn() already used ffn_scratch_ internally, but it's
    // free again by the time we get here (the FFN result was written to xb).
    if (lw.ffn_post_norm) {
        nx_apply_model_norm(xb, lw.ffn_post_norm, nullptr,
                            ffn_scratch_.data(),
                            (size_t)hidden, meta_.rms_norm_eps,
                            lw.ffn_post_norm_zc);
        memcpy(xb, ffn_scratch_.data(), (size_t)hidden * sizeof(float));
        trace("after_post_ffn_norm");
        NX_TRACE_SAMPLE(l, "ffn_out_post_post_norm", xb, (size_t)hidden);
    }

// c. Residual
    cpu::add_inplace(x, xb, (size_t)hidden);
    NX_TRACE_SAMPLE(l, "x_after_ffn_residual", x, (size_t)hidden);

    // ─── Gemma 4: PLE per-layer auxiliary residual ────────────────────────
    //
    // Each Gemma 4 layer reads its own [ple_dim_] slice and projects it back
    // to [hidden_dim] as an extra residual. We use the LAST ple_dim_ entries
    // of the hidden stream as the canonical channel (matches HF's "lightweight
    // residual block" — the PLE vector is added into a designated slice of x
    // rather than passed through a separate projection, since the projection
    // already happened on the way INTO ple_buffer_).
    //
    // For non-Gemma 4 / PLE-disabled, ple_layer_index < 0 → skipped.
    if (lw.ple_layer_index >= 0 && ple_enabled_ && ple_dim_ > 0 &&
        (int)ple_buffer_.size() >= (lw.ple_layer_index + 1) * ple_dim_)
    {
        const float* slice = ple_buffer_.data()
                           + (size_t)lw.ple_layer_index * (size_t)ple_dim_;
        // Add into the trailing ple_dim_ channels of the hidden stream.
        // If hidden < ple_dim_ (very small models) we cap at hidden.
        const int n_add = std::min(hidden, ple_dim_);
        float* tail = x + (size_t)(hidden - n_add);
        cpu::add_inplace(tail, slice, (size_t)n_add);
    }
    
    // Gemma 4 layer scalar / output scale.
    //
    // Reference (ollama) does element-wise multiply with a per-channel
    // tensor of length hidden_dim. We support both:
    //   - len == hidden : per-channel multiply (correct)
    //   - len == 1      : scalar broadcast (legacy / Q-only quirk)
    //   - otherwise     : log + skip
    //
    // The 31B Q4_K_M GGUF stores this as a vector of 5376 weights per
    // layer. Some channels are large (~0.9), some near zero (e.g. 0.036)
    // — multiplying the whole residual by min(weight) crushed the stream.
    if (lw.layer_output_scale && lw.layer_output_scale_len > 0) {
        const float* s = lw.layer_output_scale;
        const size_t n = lw.layer_output_scale_len;

        if (n == (size_t)hidden) {
            // Per-channel multiply (reference behaviour).
            for (int i = 0; i < hidden; ++i) {
                x[i] *= s[i];
            }
            NX_TRACE_DECISION(l, "has_layer_output_scale", "%d (per-channel)", 1);
        } else if (n == 1) {
            // Single scalar (some converters write it this way).
            const float sc = s[0];
            if (sc != 1.0f) {
                for (int i = 0; i < hidden; ++i) x[i] *= sc;
            }
            NX_TRACE_DECISION(l, "has_layer_output_scale", "%d (scalar=%.6g)", 1, sc);
        } else {
            static std::atomic<int> warn_count{0};
            if (warn_count.fetch_add(1) < 4) {
                fprintf(stderr,
                    "[nexus] WARNING layer %d: layer_output_scale len=%zu "
                    "doesn't match hidden=%d or 1 — skipping\n",
                    l, n, hidden);
                fflush(stderr);
            }
            NX_TRACE_DECISION(l, "has_layer_output_scale", "%d (bad len=%zu)", 1, n);
        }

        NX_TRACE_SAMPLE(l, "x_after_layer_output_scale", x, (size_t)hidden);
    } else {
        NX_TRACE_DECISION(l, "has_layer_output_scale", "%d", 0);
    }

    // ------ DIAG: per-layer residual stream trace (tail) -----------------
    if (diag_this_layer) {
        float out_rms = 0.0f, out_max = 0.0f;
        int   out_max_idx = 0, out_nan = 0, out_inf = 0;
        layer_diag_stats(x, hidden, out_rms, out_max, out_max_idx, out_nan, out_inf);

        fprintf(stderr,
            "[diag-layer]   L=%2d OUT rms=%+.3e max=%+.3e at_idx=%-5d nan=%d inf=%d  delta_rms=%+.3e delta_max=%+.3e\n",
            l, out_rms, out_max, out_max_idx, out_nan, out_inf,
            out_rms - in_rms,
            out_max - in_max);
        fflush(stderr);
    }
    // ------ END DIAG (tail) ----------------------------------------------
}

inline void Transformer::forward_ffn(const TransformerLayerWeights& lw, float* x) {
    const int hidden = (int)meta_.hidden_dim;
    const int dense_ffn = (int)meta_.ffn_dim;

    auto is_native_nq_type = [](QuantType t) -> bool {
        return t == QuantType::NQ4_G64 ||
               t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 ||
               t == QuantType::NQ8_G64;
    };

    auto mat_vec_ffn_safe = [&](const void* W,
                                const float* vin,
                                float* vout,
                                size_t rows,
                                size_t cols,
                                QuantType wt,
                                const char* tag)
    {
        if (!W || !vin || !vout || rows == 0 || cols == 0) {
            if (vout && rows) memset(vout, 0, rows * sizeof(float));
            return;
        }

        // A layer that lives on the graphics card computes its FFN there too. Before, this always ran on the CPU --
        // for a layer on the card (which keeps no copy in RAM) that meant reading these, the biggest matrices of the
        // layer, straight from the model file on every token while writing. Same call the attention / linear-
        // attention projections use; anything the card can't do falls through to the CPU as before.
        if (nx_q35_try_gpu_matvec(layer_card(lw), W, vin, vout, rows, cols, wt)) return;
        // Native NQ/BF16 FFN projections now use the same main CPU matmul route
        // as every other model instead of a one-off row-dequant fallback.
        cpu::mat_vec(W, vin, vout, rows, cols, wt, n_threads_);
        if (nx_qwen3next_diag_enabled() && is_native_nq_type(wt)) {
            static std::atomic<int> nq_ffn_mainroute_logs{0};
            const int log_id = nq_ffn_mainroute_logs.fetch_add(1, std::memory_order_relaxed);
            if (log_id < 96) {
                fprintf(stderr,
                    "[nexus] native NQ main-route FFN matvec: tag=%s rows=%zu cols=%zu type=%d threads=%d\n",
                    tag ? tag : "?", rows, cols, (int)wt, n_threads_);
                fflush(stderr);
            }
        }
        return;

        if (!is_native_nq_type(wt)) {
            cpu::mat_vec(W, vin, vout, rows, cols, wt, n_threads_);
            return;
        }

        static std::atomic<int> nq_ffn_logs{0};
        const int log_id = nq_ffn_logs.fetch_add(1);
        if (log_id < 96) {
            fprintf(stderr,
                "[nexus] native NQ safe FFN matvec: tag=%s rows=%zu cols=%zu type=%d\n",
                tag ? tag : "?", rows, cols, (int)wt);
            fflush(stderr);
        }

        const unsigned hw = std::thread::hardware_concurrency();
        int nt = n_threads_ > 1 ? n_threads_ : (hw ? (int)hw : 4);
        nt = std::max(1, std::min<int>(nt, (int)rows));
        if (log_id < 96) {
            fprintf(stderr,
                "[nexus] native NQ threaded FFN matvec: tag=%s rows=%zu cols=%zu type=%d threads=%d\n",
                tag ? tag : "?", rows, cols, (int)wt, nt);
            fflush(stderr);
        }

        std::atomic<size_t> next_row{0};
        std::vector<std::thread> workers;
        workers.reserve((size_t)nt);
        for (int ti = 0; ti < nt; ++ti) {
            workers.emplace_back([&]() {
                std::vector<float> row(cols);
                for (;;) {
                    const size_t r = next_row.fetch_add(1, std::memory_order_relaxed);
                    if (r >= rows) break;
                    quant::dequantize_row(W, row.data(), r, cols, wt);
                    double acc = 0.0;
                    for (size_t c = 0; c < cols; ++c) {
                        acc += (double)row[c] * (double)vin[c];
                    }
                    vout[r] = (float)acc;
                }
            });
        }
        for (auto& th : workers) th.join();
    };

    if (!x || hidden <= 0) {
        fprintf(stderr, "[nexus] FATAL-RUNTIME: forward_ffn invalid hidden=%d\n", hidden);
        return;
    }

    // -------------------------------------------------------------------------
    // Routed MoE FFN path.
    // CPU-first on purpose: portable, Windows/Ubuntu-safe, works with your
    // distributed split, and avoids touching Vulkan until the math is proven.
    // -------------------------------------------------------------------------
    if (lw.moe) {
        
        const int n_exp = (int)lw.moe_expert_count;
        const int top_k = std::max(1, std::min((int)lw.moe_top_k, n_exp));
        const int ffn = (int)lw.moe_ffn_dim;
        
        static std::atomic<int> moe_runtime_logs{0};
        int log_id = moe_runtime_logs.fetch_add(1);
        if (log_id < 8) {
            fprintf(stderr,
                "[nexus] MoE runtime dims hidden=%d experts=%d top_k=%d ffn=%d "
                "gate=%ux%u up=%ux%u fused=%ux%u down=%ux%u "
                "types router=%d gate=%d up=%d down=%d "
                "strides gate=%zu up=%zu down=%zu\n",
                hidden, n_exp, top_k, ffn,
                lw.moe_gate_rows, lw.moe_gate_cols,
                lw.moe_up_rows, lw.moe_up_cols,
                lw.moe_fused_rows, lw.moe_fused_cols,
                lw.moe_down_rows, lw.moe_down_cols,
                (int)lw.moe_router_type,
                (int)lw.wgate_exps_type,
                (int)lw.wup_exps_type,
                (int)lw.wdown_exps_type,
                lw.wgate_exps_stride,
                lw.wup_exps_stride,
                lw.wdown_exps_stride);
            fflush(stderr);
        }
        
        const int gate_rows  = lw.moe_fused_gate_up ? (int)lw.moe_fused_rows : (int)lw.moe_gate_rows;
        const int gate_cols  = lw.moe_fused_gate_up ? (int)lw.moe_fused_cols : (int)lw.moe_gate_cols;
        const int up_rows    = (int)lw.moe_up_rows;
        const int up_cols    = (int)lw.moe_up_cols;
        const int down_rows  = (int)lw.moe_down_rows;
        const int down_cols  = (int)lw.moe_down_cols;

        const bool has_split =
            lw.moe_split_experts &&
            lw.wgate_exp_ptrs.size() >= (size_t)n_exp &&
            lw.wup_exp_ptrs.size()   >= (size_t)n_exp &&
            lw.wdown_exp_ptrs.size() >= (size_t)n_exp;

        const bool has_merged =
            lw.wdown_exps &&
            (
                (!lw.moe_fused_gate_up && lw.wgate_exps && lw.wup_exps) ||
                ( lw.moe_fused_gate_up && lw.wup_exps)
            );

        if (!lw.moe_router || (!has_split && !has_merged) ||
            ffn <= 0 || n_exp <= 0 ||
            down_rows != hidden || down_cols != ffn) {
            fprintf(stderr,
                "[nexus] FATAL-RUNTIME: incomplete MoE FFN "
                "(router=%p split=%d merged=%d ffn=%d experts=%d down=%ux%u)\n",
                lw.moe_router,
                has_split ? 1 : 0,
                has_merged ? 1 : 0,
                ffn,
                n_exp,
                lw.moe_down_rows,
                lw.moe_down_cols);
            std::fill(x, x + hidden, 0.0f);
            return;
        }

        auto expert_ptr = [](const void* base, size_t stride, int e) -> const void* {
            return (const void*)((const uint8_t*)base + (size_t)e * stride);
        };
        
        auto moe_mat_vec = [&](const void* W,
                               const float* vin,
                               float* vout,
                               int out_rows,
                               int in_cols,
                               QuantType qt,
                               bool transposed) {
            if (!W || !vin || !vout || out_rows <= 0 || in_cols <= 0) {
                if (vout && out_rows > 0) {
                    std::fill(vout, vout + out_rows, 0.0f);
                }
                return;
            }

            std::fill(vout, vout + out_rows, 0.0f);

#ifdef NEXUS_VULKAN
            auto moe_gpu_matvec_supported = [](QuantType q, int cols) -> bool {
                switch (q) {
                    case QuantType::Q8_0:    return (cols % 32)  == 0;
                    case QuantType::Q4_K:
                    case QuantType::Q5_K:
                    case QuantType::Q2_K:
                    case QuantType::Q3_K:
                    case QuantType::Q6_K:    return (cols % 256) == 0;
                    case QuantType::NQ4_G64:
                    case QuantType::NQ5_G64:
                    case QuantType::NQ6_G64:
                    case QuantType::NQ8_G64: return (cols % 64)  == 0;
                    default: return false;
                }
            };

            const char* moe_gpu_env = std::getenv("NEXUS_MOE_GPU_EXPERTS");
            const bool moe_gpu_enabled = !(moe_gpu_env && moe_gpu_env[0] == '0');
            if (!transposed && moe_gpu_enabled && lw.gpu_layer && gpu::gpu().ready() &&
                moe_gpu_matvec_supported(qt, in_cols))
            {
                if (gpu::gpu().mat_vec(W, vin, vout,
                                       (size_t)out_rows, (size_t)in_cols, qt))
                {
                    static std::atomic<int> moe_gpu_logs{0};
                    const int gid = moe_gpu_logs.fetch_add(1, std::memory_order_relaxed);
                    if (gid < 96) {
                        fprintf(stderr,
                            "[turbo:moe-mm] GPU expert mat_vec rows=%d cols=%d type=%d\n",
                            out_rows, in_cols, (int)qt);
                        fflush(stderr);
                    }
                    return;
                }
            }
#endif

            // Worker count is capped by the row count along the parallel axis
            // so we never spawn idle threads.
            // n_threads_ comes from the enclosing Transformer instance and is
            // the same value passed to cpu::mat_vec elsewhere.
            const int nt_req = std::max(1, n_threads_);

            if (!transposed) {
                // Safe reference path for MoE experts.
                //
                // Do NOT use cpu::mat_vec here yet.
                // PhiMoE expert-down has cols=960, which is not divisible by 256.
                // This forces every row through quant::dequantize_row(), including
                // the partial final Q4_K block.
                //
                // Each output row is independent (dequant W[r,:], dot with vin),
                // so we shard rows across threads. Per-row math is identical to
                // the single-threaded path -> bit-identical outputs.

                const int nt = std::min(nt_req, out_rows);

                auto worker = [&](int r0, int r1) {
                    std::vector<float> row((size_t)in_cols, 0.0f);
                    for (int r = r0; r < r1; ++r) {
                        nexus::quant::dequantize_row(W,
                                                     row.data(),
                                                     (size_t)r,
                                                     (size_t)in_cols,
                                                     qt);

                        float sum = 0.0f;
                        for (int c = 0; c < in_cols; ++c) {
                            sum += row[c] * vin[c];
                        }

                        vout[r] = sum;
                    }
                };

                if (nt <= 1) {
                    worker(0, out_rows);
                    return;
                }

                std::vector<std::thread> threads;
                threads.reserve((size_t)nt - 1);
                const int per = (out_rows + nt - 1) / nt;

                // Spawn workers 1..nt-1; main thread handles chunk 0.
                for (int t = 1; t < nt; ++t) {
                    const int r0 = t * per;
                    if (r0 >= out_rows) break;
                    const int r1 = std::min(r0 + per, out_rows);
                    threads.emplace_back(worker, r0, r1);
                }
                worker(0, std::min(per, out_rows));

                for (auto& th : threads) th.join();
                return;
            }

            // Transposed/storage-flipped layout:
            // memory rows are input columns, each row has output columns.
            //
            // The natural loop accumulates s*row[c] into vout[c] across all r.
            // Different r's writing to the same vout[c] cannot be parallelized
            // directly, so each thread accumulates into its own partial buffer
            // and we reduce at the end. Reduction changes float-sum order, so
            // outputs may differ from the single-threaded path by a few ULPs.

            const int nt = std::min(nt_req, in_cols);

            if (nt <= 1) {
                std::vector<float> row((size_t)out_rows, 0.0f);
                for (int r = 0; r < in_cols; ++r) {
                    nexus::quant::dequantize_row(W,
                                                 row.data(),
                                                 (size_t)r,
                                                 (size_t)out_rows,
                                                 qt);
                    const float s = vin[r];
                    if (s == 0.0f) continue;
                    for (int c = 0; c < out_rows; ++c) {
                        vout[c] += s * row[c];
                    }
                }
                return;
            }

            std::vector<std::vector<float>> partials(
                (size_t)nt,
                std::vector<float>((size_t)out_rows, 0.0f));

            auto worker = [&](int t, int r0, int r1) {
                std::vector<float> row((size_t)out_rows, 0.0f);
                float* acc = partials[(size_t)t].data();
                for (int r = r0; r < r1; ++r) {
                    nexus::quant::dequantize_row(W,
                                                 row.data(),
                                                 (size_t)r,
                                                 (size_t)out_rows,
                                                 qt);
                    const float s = vin[r];
                    if (s == 0.0f) continue;
                    for (int c = 0; c < out_rows; ++c) {
                        acc[c] += s * row[c];
                    }
                }
            };

            std::vector<std::thread> threads;
            threads.reserve((size_t)nt - 1);
            const int per = (in_cols + nt - 1) / nt;

            for (int t = 1; t < nt; ++t) {
                const int r0 = t * per;
                if (r0 >= in_cols) break;
                const int r1 = std::min(r0 + per, in_cols);
                threads.emplace_back(worker, t, r0, r1);
            }
            worker(0, 0, std::min(per, in_cols));

            for (auto& th : threads) th.join();

            // Reduce per-thread partials into vout.
            for (int t = 0; t < nt; ++t) {
                const float* acc = partials[(size_t)t].data();
                for (int c = 0; c < out_rows; ++c) {
                    vout[c] += acc[c];
                }
            }
        };

        // Router projection: hidden -> n_experts.
        std::vector<float> router_logits((size_t)n_exp, 0.0f);
        
        cpu::mat_vec(lw.moe_router, x, router_logits.data(),
                     (size_t)n_exp, (size_t)hidden, lw.moe_router_type, n_threads_);

        std::vector<float> route_weight((size_t)n_exp, 0.0f);
        std::vector<int> idx((size_t)top_k, 0);

        auto sigmoid = [](float v) -> float {
            if (v >  80.0f) return 1.0f;
            if (v < -80.0f) return 0.0f;
            return 1.0f / (1.0f + expf(-v));
        };

        auto masked_softmax = [&](const std::vector<float>& scores) -> std::vector<float> {
            std::vector<float> out(scores.size(), 0.0f);

            float mx = -1.0e30f;
            for (float v : scores) {
                if (v > mx) mx = v;
            }

            if (mx <= -0.9e30f) return out;

            float sum = 0.0f;
            for (size_t i = 0; i < scores.size(); ++i) {
                if (scores[i] <= -0.9e30f) {
                    out[i] = 0.0f;
                } else {
                    out[i] = expf(scores[i] - mx);
                    sum += out[i];
                }
            }

            if (sum > 0.0f) {
                for (float& v : out) v /= sum;
            }
            return out;
        };

        float selected_sum = 0.0f;

        // Sparse top-2 mixer router.
        // Used by architectures that select one expert, mask it out, then select the second.
        if (lw.moe_router_kind == 1 && top_k == 2 && n_exp >= 2) {
            const float jitter_eps =
                (lw.moe_router_jitter_eps > 0.0f) ? lw.moe_router_jitter_eps : 0.01f;
            const float threshold_scale = 2.0f * jitter_eps;
            const float NEG = -1.0e30f;

            // First expert: argmax raw router logits.
            int e0 = 0;
            for (int i = 1; i < n_exp; ++i) {
                if (router_logits[i] > router_logits[e0]) e0 = i;
            }

            const float max0 = router_logits[e0];

            std::vector<float> masked0((size_t)n_exp, NEG);
            for (int i = 0; i < n_exp; ++i) {
                float factor = std::max(std::fabs(router_logits[i]), max0);
                if (factor < 1.0e-12f) factor = 1.0e-12f;

                const bool mask = ((max0 - router_logits[i]) / factor) > threshold_scale;
                masked0[i] = mask ? NEG : router_logits[i];
            }

            std::vector<float> probs0 = masked_softmax(masked0);
            const float w0 = probs0[e0];

            // Second expert: remove first expert, then repeat sparse selection.
            std::vector<float> scores1 = router_logits;
            scores1[e0] = NEG;

            int e1 = (e0 == 0) ? 1 : 0;
            for (int i = 0; i < n_exp; ++i) {
                if (scores1[i] > scores1[e1]) e1 = i;
            }

            const float max1 = scores1[e1];

            std::vector<float> masked1((size_t)n_exp, NEG);
            for (int i = 0; i < n_exp; ++i) {
                float factor = std::max(std::fabs(router_logits[i]), max1);
                if (factor < 1.0e-12f) factor = 1.0e-12f;

                const bool mask = ((max1 - router_logits[i]) / factor) > threshold_scale;
                masked1[i] = mask ? NEG : scores1[i];
            }

            std::vector<float> probs1 = masked_softmax(masked1);
            const float w1 = probs1[e1];

            idx[0] = e0;
            idx[1] = e1;

            route_weight[e0] += w0;
            route_weight[e1] += w1;

            selected_sum = w0 + w1;
        } else {
            std::vector<float> select_score((size_t)n_exp, 0.0f);

            if (lw.moe_router_kind == 2) {
                // Sigmoid top-k router. Qwen/DeepSeek-style GGUFs may declare
                // this through expert_gating_func=sigmoid. Select by sigmoid
                // score plus optional correction bias, then optionally normalize
                // the selected routed weights below.
                for (int i = 0; i < n_exp; ++i) {
                    float p = sigmoid(router_logits[i]);
                    route_weight[i] = p;
                    select_score[i] = p + (lw.moe_router_bias ? lw.moe_router_bias[i] : 0.0f);
                }
            } else if (lw.moe_router_bias) {
                // Bias-corrected router path used by some MoE GGUFs.
                for (int i = 0; i < n_exp; ++i) {
                    float p = sigmoid(router_logits[i]);
                    route_weight[i] = p;
                    select_score[i] = p + lw.moe_router_bias[i];
                }
            } else {
                // Standard Mixtral / OLMoE style: softmax router -> top-k.
                route_weight = router_logits;
                cpu::softmax(route_weight.data(), (size_t)n_exp);
                select_score = route_weight;
            }

            std::vector<int> order((size_t)n_exp);
            std::iota(order.begin(), order.end(), 0);

            std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                [&](int a, int b) { return select_score[a] > select_score[b]; });

            for (int j = 0; j < top_k; ++j) {
                idx[j] = order[j];
                selected_sum += route_weight[idx[j]];
            }
        }

        const bool normalize_selected = lw.moe_normalize_topk;
        const float scale = lw.moe_route_scale;
        std::vector<float> out((size_t)hidden, 0.0f);
        std::vector<float> gate((size_t)ffn, 0.0f);
        std::vector<float> up((size_t)ffn, 0.0f);
        std::vector<float> gateup((size_t)ffn * 2, 0.0f);
        std::vector<float> down((size_t)hidden, 0.0f);

        for (int j = 0; j < top_k; ++j) {
            const int e = idx[j];

            float w = route_weight[e];
            if (normalize_selected && selected_sum > 1e-12f) w /= selected_sum;
            w *= scale;
            
            static std::atomic<int> moe_route_logs{0};
            int route_log_id = moe_route_logs.fetch_add(1);
            if (route_log_id < 16) {
                fprintf(stderr,
                    "[nexus] MoE route check: selected_sum=%.6f normalize=%s scale=%.4f "
                    "top=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
                    selected_sum,
                    normalize_selected ? "yes" : "no",
                    scale,
                    top_k > 0 ? idx[0] : -1,
                    top_k > 1 ? idx[1] : -1,
                    top_k > 2 ? idx[2] : -1,
                    top_k > 3 ? idx[3] : -1,
                    top_k > 4 ? idx[4] : -1,
                    top_k > 5 ? idx[5] : -1,
                    top_k > 6 ? idx[6] : -1,
                    top_k > 7 ? idx[7] : -1);
                fflush(stderr);
            }

            if (w == 0.0f) continue;
            

            if (lw.moe_fused_gate_up) {
                const void* Wgu = expert_ptr(lw.wup_exps, lw.wup_exps_stride, e);
                moe_mat_vec(Wgu, x, gateup.data(),
                            gate_rows, gate_cols,
                            lw.wup_exps_type,
                            lw.moe_fused_transposed);

                if (gate_rows != ffn * 2 || gate_cols != hidden) {
                    fprintf(stderr,
                        "[nexus] FATAL-RUNTIME: bad fused MoE dims rows=%d cols=%d ffn=%d hidden=%d\n",
                        gate_rows, gate_cols, ffn, hidden);
                    std::fill(x, x + hidden, 0.0f);
                    return;
                }

                memcpy(gate.data(), gateup.data(), (size_t)ffn * sizeof(float));
                memcpy(up.data(), gateup.data() + ffn, (size_t)ffn * sizeof(float));
            } else {
                const void* Wg = lw.moe_split_experts
                    ? lw.wgate_exp_ptrs[(size_t)e]
                    : expert_ptr(lw.wgate_exps, lw.wgate_exps_stride, e);

                const void* Wu = lw.moe_split_experts
                    ? lw.wup_exp_ptrs[(size_t)e]
                    : expert_ptr(lw.wup_exps, lw.wup_exps_stride, e);

                const QuantType Wg_type = lw.moe_split_experts
                    ? lw.wgate_exp_types[(size_t)e]
                    : lw.wgate_exps_type;

                const QuantType Wu_type = lw.moe_split_experts
                    ? lw.wup_exp_types[(size_t)e]
                    : lw.wup_exps_type;

                if (lw.moe_gate_rows != lw.moe_up_rows ||
                    lw.moe_gate_cols != (uint32_t)hidden ||
                    lw.moe_up_cols   != (uint32_t)hidden ||
                    lw.moe_gate_rows != (uint32_t)ffn) {
                    fprintf(stderr,
                        "[nexus] FATAL-RUNTIME: bad separate MoE dims gate=%ux%u up=%ux%u ffn=%d hidden=%d\n",
                        lw.moe_gate_rows, lw.moe_gate_cols,
                        lw.moe_up_rows, lw.moe_up_cols,
                        ffn, hidden);
                    std::fill(x, x + hidden, 0.0f);
                    return;
                }

                moe_mat_vec(Wg, x, gate.data(),
                            (int)lw.moe_gate_rows, (int)lw.moe_gate_cols,
                            Wg_type,
                            lw.moe_gate_transposed);

                moe_mat_vec(Wu, x, up.data(),
                            (int)lw.moe_up_rows, (int)lw.moe_up_cols,
                            Wu_type,
                            lw.moe_up_transposed);
            }

            cpu::silu(gate.data(), (size_t)ffn);
            cpu::mul(gate.data(), up.data(), gate.data(), (size_t)ffn);

            const void* Wd = lw.moe_split_experts
                ? lw.wdown_exp_ptrs[(size_t)e]
                : expert_ptr(lw.wdown_exps, lw.wdown_exps_stride, e);

            const QuantType Wd_type = lw.moe_split_experts
                ? lw.wdown_exp_types[(size_t)e]
                : lw.wdown_exps_type;

            moe_mat_vec(Wd, gate.data(), down.data(),
                        (int)lw.moe_down_rows, (int)lw.moe_down_cols,
                        Wd_type,
                        lw.moe_down_transposed);

            for (int i = 0; i < hidden; ++i) out[i] += w * down[i];
        }

        // Optional shared expert path:
        // shexp = down_shexp(silu(gate_shexp(x)) * up_shexp(x))
        //
        // Use the same safe row-dequant matvec as routed experts. Do not use
        // cpu::mat_vec here; shared expert dimensions and K-quants can still hit
        // partial-block cases.
        if (lw.moe_has_shared && lw.wgate_shexp && lw.wup_shexp && lw.wdown_shexp) {
            const int sffn = (int)lw.moe_shared_ffn_dim;

            if (sffn <= 0 ||
                lw.moe_shared_gate_rows != (uint32_t)sffn ||
                lw.moe_shared_up_rows   != (uint32_t)sffn ||
                lw.moe_shared_down_rows != (uint32_t)hidden ||
                lw.moe_shared_gate_cols != (uint32_t)hidden ||
                lw.moe_shared_up_cols   != (uint32_t)hidden ||
                lw.moe_shared_down_cols != (uint32_t)sffn) {
                fprintf(stderr,
                    "[nexus] FATAL-RUNTIME: bad shared expert runtime dims "
                    "sffn=%d hidden=%d gate=%ux%u up=%ux%u down=%ux%u\n",
                    sffn,
                    hidden,
                    lw.moe_shared_gate_rows,
                    lw.moe_shared_gate_cols,
                    lw.moe_shared_up_rows,
                    lw.moe_shared_up_cols,
                    lw.moe_shared_down_rows,
                    lw.moe_shared_down_cols);
                std::fill(x, x + hidden, 0.0f);
                return;
            }

            std::vector<float> sg((size_t)sffn, 0.0f);
            std::vector<float> su((size_t)sffn, 0.0f);
            std::vector<float> sd((size_t)hidden, 0.0f);

            moe_mat_vec(lw.wgate_shexp,
                        x,
                        sg.data(),
                        (int)lw.moe_shared_gate_rows,
                        (int)lw.moe_shared_gate_cols,
                        lw.wgate_shexp_type,
                        lw.moe_shared_gate_transposed);

            moe_mat_vec(lw.wup_shexp,
                        x,
                        su.data(),
                        (int)lw.moe_shared_up_rows,
                        (int)lw.moe_shared_up_cols,
                        lw.wup_shexp_type,
                        lw.moe_shared_up_transposed);

            cpu::silu(sg.data(), (size_t)sffn);
            cpu::mul(sg.data(), su.data(), sg.data(), (size_t)sffn);

            moe_mat_vec(lw.wdown_shexp,
                        sg.data(),
                        sd.data(),
                        (int)lw.moe_shared_down_rows,
                        (int)lw.moe_shared_down_cols,
                        lw.wdown_shexp_type,
                        lw.moe_shared_down_transposed);

            float shexp_scale = 1.0f;
            if (lw.wgate_inp_shexp) {
                float dot = 0.0f;
                for (int i = 0; i < hidden; ++i) {
                    dot += x[i] * lw.wgate_inp_shexp[i];
                }
                shexp_scale = sigmoid(dot);
            }

            for (int i = 0; i < hidden; ++i) {
                out[i] += shexp_scale * sd[i];
            }
        }

        memcpy(x, out.data(), (size_t)hidden * sizeof(float));
        return;
    }

    // -------------------------------------------------------------------------
    // Existing dense SwiGLU path.
    // -------------------------------------------------------------------------
    const int ffn = dense_ffn;

    if (!x || hidden <= 0 || ffn <= 0) {
        fprintf(stderr, "[nexus] FATAL-RUNTIME: forward_ffn invalid dims hidden=%d ffn=%d\n",
                hidden, ffn);
        if (x && hidden > 0) std::fill(x, x + hidden, 0.0f);
        return;
    }

    if (!lw.fused_gate_up && (!lw.wgate || !lw.wup || !lw.wdown)) {
        fprintf(stderr,
            "[nexus] FATAL-RUNTIME: dense FFN pointer missing at forward time "
            "(gate=%p up=%p down=%p). Zeroing FFN output to avoid SIGSEGV.\n",
            lw.wgate, lw.wup, lw.wdown);
        std::fill(x, x + hidden, 0.0f);
        return;
    }

    if (lw.fused_gate_up && (!lw.wup || !lw.wdown)) {
        fprintf(stderr,
            "[nexus] FATAL-RUNTIME: fused FFN pointer missing at forward time "
            "(up/fused=%p down=%p). Zeroing FFN output to avoid SIGSEGV.\n",
            lw.wup, lw.wdown);
        std::fill(x, x + hidden, 0.0f);
        return;
    }

    const bool try_gpu  = (lw.gpu_layer  && gpu::gpu().ready());
    const bool try_gpu2 = gpu2_ready_for(lw);
    gpu::VulkanBackend& active_gpu = (try_gpu2 && !try_gpu) ? *backend2_for(lw) : gpu::gpu();
    const bool use_any_gpu = try_gpu || try_gpu2;

    float* gate = ffn_scratch_.data();
    float* up   = ffn_scratch_.data() + ffn;

    if (lw.fused_gate_up) {
        bool ok = use_any_gpu && !is_native_nq_type(lw.wup_type) &&
            active_gpu.mat_vec(lw.wup, x, gate,
                               (size_t)ffn * 2, (size_t)hidden, lw.wup_type);
        if (!ok) {
            mat_vec_ffn_safe(lw.wup, x, gate,
                             (size_t)ffn * 2, (size_t)hidden, lw.wup_type,
                             "fused_gate_up");
        }
    } else {
        bool pair_ok = use_any_gpu &&
            !is_native_nq_type(lw.wgate_type) &&
            !is_native_nq_type(lw.wup_type) &&
            active_gpu.mat_vec_pair(
                lw.wgate, gate, (size_t)ffn, lw.wgate_type,
                lw.wup,   up,   (size_t)ffn, lw.wup_type,
                x, (size_t)hidden);

        if (!pair_ok) {
            mat_vec_ffn_safe(lw.wgate, x, gate,
                             (size_t)ffn, (size_t)hidden, lw.wgate_type,
                             "ffn_gate");
            mat_vec_ffn_safe(lw.wup, x, up,
                             (size_t)ffn, (size_t)hidden, lw.wup_type,
                             "ffn_up");
        }
    }

// Activation: SiLU for Llama/Mistral/Qwen/Phi/most models; GELU-tanh for Gemma.
    if (ffn_activation_gelu_) {
        nx_gelu_tanh_inplace(gate, (size_t)ffn);
    } else {
        cpu::silu(gate, (size_t)ffn);
    }
    cpu::mul(gate, up, gate, (size_t)ffn);

    bool dok = use_any_gpu && !is_native_nq_type(lw.wdown_type) &&
        active_gpu.mat_vec(lw.wdown, gate, x,
                           (size_t)hidden, (size_t)ffn, lw.wdown_type);
    if (!dok) {
        mat_vec_ffn_safe(lw.wdown, gate, x,
                         (size_t)hidden, (size_t)ffn, lw.wdown_type,
                         "ffn_down");
    }
}
// --------- Eager weight upload ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Upload all quantized GPU-layer weight matrices to VRAM at set_gpu_layers() time,
// so inference never pays the PCIe transfer cost mid-token.
// RMS norm weights (float) stay on CPU --- they're used in cpu::rms_norm(), not dispatched.

inline bool Transformer::eager_upload_layer_weights(int l, gpu::VulkanBackend& g, std::vector<const void*>* uploaded) {
    const TransformerLayerWeights& lw = layer_weights_[l];
    const size_t hidden = (size_t)meta_.hidden_dim;
    const size_t ffn    = (size_t)meta_.ffn_dim;
    const size_t q_dim  = (size_t)(meta_.n_heads    * meta_.head_dim);
    const size_t kv_dim = (size_t)(meta_.n_kv_heads * meta_.head_dim);

    bool all_ok = true;
    std::vector<std::pair<const void*, size_t>> mine;      // every weight of this layer that was uploaded, with the size it must have
    // Helper: compute byte size for a weight matrix and trigger upload
    auto upload_w = [&](const void* W, QuantType qt, size_t rows, size_t cols) {
        if (!W || rows == 0 || cols == 0) return;
        size_t W_size = 0; uint32_t dummy = 0;
        if (g.size_for_public(qt, rows, cols, W_size, dummy) && W_size > 0) {
            mine.emplace_back(W, W_size);
            if (!g.get_weight_buffer_public(W, W_size, qt)) all_ok = false;
        }
    };

    if (lw.fused_qkv) {
        // Phi-3 style: single [q_dim + 2*kv_dim, hidden] fused tensor
        // Qwen3.5 linear-attention layers: in_proj_qkv has conv_dim rows (key*2 + value), NOT q_dim + 2*kv_dim.  Uploading the
        // smaller size left the card with a truncated matrix and the batched projection read past its end.
        upload_w(lw.wqkv, lw.wqkv_type, (lw.qwen35_linear && lw.q35_conv_dim > 0) ? (size_t)lw.q35_conv_dim : q_dim + 2 * kv_dim, hidden);
    } else {
        // Qwen3.5/3-Next full-attention q_proj holds query AND gate rows (2 x q_dim).  Uploading only q_dim rows left the
        // card with half of the matrix; the batched projection then read past the end (gate half came out wrong).
        upload_w(lw.wq,  lw.wq_type,  (lw.qwen35_full_attn_q_has_gate ? ((lw.qwen35_full_attn_q_rows > 0) ? (size_t)lw.qwen35_full_attn_q_rows : q_dim * 2) : q_dim),  hidden);
        upload_w(lw.wk,  lw.wk_type,  kv_dim, hidden);
        upload_w(lw.wv,  lw.wv_type,  kv_dim, hidden);
    }
    upload_w(lw.wo, lw.wo_type, hidden, q_dim);

    if (lw.fused_gate_up) {
        // Phi-3 style: single [2*ffn, hidden] fused tensor
        upload_w(lw.wup, lw.wup_type, ffn * 2, hidden);
    } else {
        upload_w(lw.wgate, lw.wgate_type, ffn, hidden);
        upload_w(lw.wup,   lw.wup_type,   ffn, hidden);
    }
    upload_w(lw.wdown, lw.wdown_type, hidden, ffn);

    // Norm weights are f32 scalars [hidden] --- also need to be in GPU cache
    // so record_rms_norm_g doesn't call upload() mid-recording (deadlock).
    if (lw.attn_norm) { mine.emplace_back(lw.attn_norm, hidden * sizeof(float)); if (!g.get_weight_buffer_public(lw.attn_norm, hidden * sizeof(float), QuantType::F32)) all_ok = false; }
    if (lw.ffn_norm)  { mine.emplace_back(lw.ffn_norm,  hidden * sizeof(float)); if (!g.get_weight_buffer_public(lw.ffn_norm,  hidden * sizeof(float), QuantType::F32)) all_ok = false; }
    // Check pass: every weight of the layer must now be on the card AND at least as large as it needs to be.  If anything is
    // missing or short the layer is not left half-uploaded: its weights are freed and the caller puts the layer on the CPU.
    if (all_ok) for (auto& m : mine) if (g.weight_size_public(m.first) < m.second) { all_ok = false; break; }
    if (!all_ok) {
        for (auto& m : mine) g.evict_weight_public(m.first);
        fprintf(stderr, "[nexus] layer %d did not upload completely -- freed it again; it runs on the CPU\n", l);
    }
    if (all_ok && uploaded) for (auto& m : mine) uploaded->push_back(m.first);
    return all_ok;
}

// --------- Sampling ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

inline int32_t Transformer::sample(const SamplerConfig& cfg) {
    NxProfScope prof_samp(&nx_prof_extra().samp);
    if (logits_.empty()) return 0;

    if (cfg.temperature <= 0.0f || cfg.temperature < 1e-6f) {
        return sample_argmax();
    }

    std::vector<float> logits = logits_;
    const int vsz = (int)logits.size();

    // ------ Repetition penalties (applied to logits before softmax) ------------------------------------------
    if (cfg.recent_tokens && !cfg.recent_tokens->empty()) {
        const auto& rt = *cfg.recent_tokens;
        int n = std::min((int)rt.size(), cfg.repeat_last_n);

        // Count frequency of each token in recent window
        std::unordered_map<int32_t, int> freq;
        for (int i = (int)rt.size() - n; i < (int)rt.size(); ++i)
            if (rt[i] >= 0 && rt[i] < vsz) freq[rt[i]]++;

        for (auto& [tok, cnt] : freq) {
            // Multiplicative repeat penalty (llama.cpp style)
            if (cfg.repeat_penalty != 1.0f) {
                if (logits[tok] > 0.0f) logits[tok] /= cfg.repeat_penalty;
                else                    logits[tok] *= cfg.repeat_penalty;
            }
            // Frequency penalty: scale by how often token appeared
            if (cfg.frequency_penalty != 0.0f)
                logits[tok] -= cfg.frequency_penalty * cnt;
            // Presence penalty: flat penalty if token appeared at all
            if (cfg.presence_penalty != 0.0f)
                logits[tok] -= cfg.presence_penalty;
        }
    }

    // ------ Temperature ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cpu::scale(logits.data(), 1.0f / cfg.temperature, vsz);

    // ------ Top-K (on logits, before softmax --- keeps numerical stability) ------------------------
    if (cfg.top_k > 0 && cfg.top_k < vsz) {
        std::vector<int> idx(vsz);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
            [&](int a, int b) { return logits[a] > logits[b]; });
        float thresh = logits[idx[cfg.top_k - 1]];
        for (auto& v : logits) if (v < thresh) v = -1e9f;
    }

    // ------ Softmax ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    cpu::softmax(logits.data(), vsz);
    std::vector<float>& probs = logits;

    // ------ Min-P (applied to probs --- more natural than top-k alone) ---------------------------------------
    // Keeps tokens whose prob >= min_p * max_prob. Cleaner vocab at low temp.
    if (cfg.min_p > 0.0f) {
        float max_p = *std::max_element(probs.begin(), probs.end());
        float cutoff = cfg.min_p * max_p;
        for (auto& p : probs) if (p < cutoff) p = 0.0f;
    }

    // ------ Tail-Free Sampling (TFS) ------------------------------------------------------------------------------------------------------------------------------------------
    // Removes the "tail" of the distribution using second derivatives.
    // tfs_z=1.0 disables it; 0.95-0.99 gives good results.
    if (cfg.tfs_z < 1.0f) {
        // Sort by prob descending
        std::vector<std::pair<float,int>> sorted;
        sorted.reserve(vsz);
        for (int i = 0; i < vsz; ++i) if (probs[i] > 0.0f) sorted.push_back({probs[i], i});
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.first > b.first; });

        if (sorted.size() > 2) {
            // First derivative of sorted probs
            std::vector<float> d1(sorted.size() - 1);
            for (size_t i = 0; i < d1.size(); ++i) d1[i] = sorted[i].first - sorted[i+1].first;
            // Second derivative
            std::vector<float> d2(d1.size() - 1);
            for (size_t i = 0; i < d2.size(); ++i) d2[i] = std::abs(d1[i] - d1[i+1]);
            // Normalize d2
            float sum_d2 = 0.0f; for (auto v : d2) sum_d2 += v;
            if (sum_d2 > 0.0f) {
                float cum = 0.0f;
                size_t cutoff = d2.size();
                for (size_t i = 0; i < d2.size(); ++i) {
                    cum += d2[i] / sum_d2;
                    if (cum >= cfg.tfs_z) { cutoff = i + 1; break; }
                }
                // Zero out tokens beyond the cutoff
                for (size_t i = cutoff + 1; i < sorted.size(); ++i)
                    probs[sorted[i].second] = 0.0f;
            }
        }
    }

    // ------ Top-P nucleus sampling ------------------------------------------------------------------------------------------------------------------------------------------------
    if (cfg.top_p < 1.0f) {
        std::vector<std::pair<float,int>> sorted;
        sorted.reserve(vsz);
        for (int i = 0; i < vsz; ++i) if (probs[i] > 0.0f) sorted.push_back({probs[i], i});
        std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.first > b.first; });

        float cumsum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i].first;
            if (cumsum >= cfg.top_p) { cutoff = i + 1; break; }
        }
        for (size_t i = 0; i < probs.size(); ++i) probs[i] = 0.0f;
        for (size_t i = 0; i < cutoff; ++i) probs[sorted[i].second] = sorted[i].first;
    }

    // ------ Renormalize after filters -------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Top-k / min-p / TFS / top-p zero out probability mass. If we do not
    // renormalize here, random r in [0,1] can land above the remaining mass
    // and repeatedly fall through to the last surviving token.
    float psum = 0.0f;
    int32_t best_tok = 0;
    float best_p = -1.0f;

    for (int32_t i = 0; i < vsz; ++i) {
        if (!std::isfinite(probs[i]) || probs[i] < 0.0f) {
            probs[i] = 0.0f;
            continue;
        }

        if (probs[i] > best_p) {
            best_p = probs[i];
            best_tok = i;
        }

        psum += probs[i];
    }

    if (!(psum > 0.0f) || !std::isfinite(psum)) {
        return sample_argmax();
    }

    const float inv_sum = 1.0f / psum;
    for (float& p : probs) {
        p *= inv_sum;
    }

    // ------ Weighted sample ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (cfg.seed >= 0) {
        // Do NOT reseed every token. Reseeding every sample repeats the same random draw.
        if (rng_seed_active_ != cfg.seed) {
            rng_.seed((unsigned)cfg.seed);
            rng_seed_active_ = cfg.seed;
        }
    } else if (rng_seed_active_ != -1) {
        std::random_device rd;
        rng_.seed(rd());
        rng_seed_active_ = -1;
    }

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float r = uniform(rng_);

    float cumsum = 0.0f;
    for (int32_t i = 0; i < vsz; ++i) {
        if (probs[i] <= 0.0f) continue;
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }

    // Floating point fallback. Return highest surviving probability, not last token.
    return best_tok;
    }

inline int32_t Transformer::sample_argmax() const {
    return (int32_t)(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
}

// ------ forward_embed: inject precomputed embedding, skip token lookup ---------------------------------------
inline bool Transformer::token_embedding(int32_t token, float* out) const {
    if (!loaded_ || !out || !token_embedding_table) return false;
    if (token < 0 || token >= (int32_t)meta_.vocab_size) return false;
    const int hidden = (int)meta_.hidden_dim;
    quant::dequantize_row(token_embedding_table, out,
                          (size_t)token, (size_t)hidden, emb_type);
    if (embed_scale_ > 0.0f) {
        for (int i = 0; i < hidden; ++i) out[i] *= embed_scale_;
    }
    return true;
}

inline int32_t Transformer::forward_embed_prefill(const float* embed, int pos) {
    const int hidden = (int)meta_.hidden_dim;
    float* x = x_scratch_.data();
    memcpy(x, embed, hidden * sizeof(float));
    for (int l = 0; l < (int)meta_.n_layers; ++l) {
        LayerPinGuard pin_guard(*mm_, l);
        forward_layer(l, x, pos);
    }
    return 0;
}

inline void Transformer::prefill_embeddings_stream(const float* embeddings, int hidden_dim,
                                                 int start_pos, int n,
                                                 bool force_no_stream)
{
    if (!loaded_ || !embeddings || n <= 0) return;

    const int hidden = (int)meta_.hidden_dim;
    if (hidden_dim != hidden) {
        fprintf(stderr,
            "[vision:turboprefill] rejected: embedding hidden=%d model hidden=%d n=%d\n",
            hidden_dim, hidden, n);
        return;
    }

    // Single token: exact old behavior.
    if (n == 1) {
        forward_embed_prefill(embeddings, start_pos);
        return;
    }

    // Gemma 4 PLE needs per-position state; preserve exact old behavior.
    if (ple_enabled_) {
        for (int i = 0; i < n; ++i) {
            forward_embed_prefill(embeddings + (size_t)i * (size_t)hidden, start_pos + i);
        }
        return;
    }

    // The vision/audio encoder must complete before this point. Once we have the
    // projected embeddings, stream them through the language model layer-major:
    // layer 0 over every image token, then layer 1 over every image token, etc.
    std::vector<float> x_buf((size_t)n * (size_t)hidden);
    std::memcpy(x_buf.data(), embeddings, (size_t)n * (size_t)hidden * sizeof(float));

    std::vector<uint8_t> saved_gpu_flag;
    const bool stream = turbo_gpu_stream_ && gpu::gpu().ready() && !force_no_stream;
    if (stream) {
        turbo_keep_prepare();                // layers that live on this card stay there; the rest stream through it
        saved_gpu_flag.resize(meta_.n_layers);
        for (int l = 0; l < (int)meta_.n_layers; ++l) {
            saved_gpu_flag[l] = layer_weights_[l].gpu_layer ? 1 : 0;
            layer_weights_[l].gpu_layer = true;
        }

        const char* ab = std::getenv("NEXUS_AUTO_BUDGET");
        const bool auto_budget = !(ab && ab[0]=='0');
        if (auto_budget) {
            const size_t kv_attn_reserve = (size_t)320 * 1024 * 1024;
            compute_and_set_turbo_vram_budget(kv_attn_reserve, start_pos + n);
        }

        fprintf(stderr,
            "[vision:turboprefill] GPU streaming ON (%d embedding tokens, %d layers, cache budget %zu MB, resident window %d layers)\n",
            n, (int)meta_.n_layers,
            gpu::gpu().weight_cache_budget() / (1024*1024),
            turbo_layer_window_size());

        // Sequential streaming: layers are loaded one at a time in the loop below.
    } else {
        fprintf(stderr,
            "[vision:turboprefill] CPU/layer-major path (%d embedding tokens, %d layers)\n",
            n, (int)meta_.n_layers);
    }

    for (int l = 0; l < (int)meta_.n_layers; ++l) {
        LayerPinGuard pin_guard(*mm_, l);
        const TransformerLayerWeights& lw = layer_weights_[l];
        bool handled_by_batched_turbo = false;

        if (stream) {
            turbo_stream_begin_layer(l, "V");
        }

        if (stream && lw.qwen35_linear && !lw.moe &&
            layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_qwen35_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        else if (stream && !lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_full_attn_prefill_layer_batch_native_legacy(lw, xp, l, sp, cnt) || forward_full_attn_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        else if (!stream && n > 1 && lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_qwen35_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        else if (!stream && n > 1 && !lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_full_attn_prefill_layer_batch_native_legacy(lw, xp, l, sp, cnt) || forward_full_attn_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }

        if (!handled_by_batched_turbo) {
            for (int t = 0; t < n; ++t) {
                forward_layer(l,
                              x_buf.data() + (size_t)t * (size_t)hidden,
                              start_pos + t);
            }
        }

        if (stream) turbo_stream_end_layer(l);
    }

    if (stream) {
        turbo_prefetch_wait(turbo_prefetch_tail_future_);
        turbo_prefetch_wait(turbo_prefetch_next_future_);
        for (int l = 0; l < (int)meta_.n_layers; ++l) {
            layer_weights_[l].gpu_layer = saved_gpu_flag[l] != 0;
        }
        gpu::gpu().evict_unpinned_weights();
    }

    // Match forward_embed_prefill side effects as closely as useful: leave
    // x_scratch_ containing the final hidden state of the last embedding token.
    float* x_last = x_buf.data() + (size_t)(n - 1) * (size_t)hidden;
    std::memcpy(x_scratch_.data(), x_last, (size_t)hidden * sizeof(float));

    // IMPORTANT: sample() reads logits_. The old per-token forward()/prefill_batch()
    // path computes final norm + lm_head for the last position. The layer-major
    // embedding path must do the same, otherwise a mixed text+vision prefill leaves
    // logits_ stale from whatever happened before the image turn and the model
    // starts generation from the wrong distribution.
    const int vocab = (int)meta_.vocab_size;
    const float* head_in = x_last;
    if (rms_final_weight) {
        float* xb = xb_scratch_.data();
        nx_apply_model_norm(x_last, rms_final_weight, rms_final_bias, xb,
                            (size_t)hidden, meta_.rms_norm_eps,
                            zero_centered_norm_);
        head_in = xb;
    }

    auto is_native_nq = [](QuantType t) {
        return t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64;
    };

    bool lm_done = false;
    if (gpu_lm_head_ && nx_lm_head_gpu_ok(lm_head_type)) {
        lm_done = gpu::gpu().mat_vec(lm_head_weight, head_in, logits_.data(),
                                     (size_t)vocab, (size_t)hidden, lm_head_type);
    }
    if (!lm_done) {
        cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                     (size_t)vocab, (size_t)hidden, lm_head_type, n_threads_);
    }
    if (lm_head_bias) {
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] += lm_head_bias[i];
    }
    if (final_logit_softcap_ > 0.0f) {
        const float cap = final_logit_softcap_;
        const float inv = 1.0f / cap;
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] = fast_tanh(logits_[i] * inv) * cap;
    }

    fprintf(stderr,
        "[vision:turboprefill] done: %d embedding tokens pos=%d..%d logits=updated\n",
        n, start_pos, start_pos + n - 1);
}


inline int32_t Transformer::forward_embed(const float* embed, int pos) {
    const int hidden = (int)meta_.hidden_dim;
    forward_embed_prefill(embed, pos);
    float* x = x_scratch_.data();
    const float* head_in = x;
    std::vector<float> xb;
    if (rms_final_weight) {
        xb.resize((size_t)hidden);
        nx_apply_model_norm(x, rms_final_weight, rms_final_bias, xb.data(),
                    (size_t)hidden, meta_.rms_norm_eps,
                    zero_centered_norm_);
        head_in = xb.data();
    }
    logits_.resize(meta_.vocab_size);
    cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                 meta_.vocab_size, hidden, lm_head_type);
    if (lm_head_bias) {
        for (uint32_t i = 0; i < meta_.vocab_size; ++i) {
            logits_[i] += lm_head_bias[i];
        }
    }
    return 0;
}

// ------ LoRA adapter ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// LoRA applies low-rank updates: W' = W + scale * (B @ A)
// A: [rank, in_dim], B: [out_dim, rank]
// We store per-layer LoRA deltas and add them during forward

struct LoRAAdapter {
    float  scale = 1.0f;
    std::string path;
    // Per-layer, per-weight deltas stored as flat float arrays
    // Key: tensor name (e.g. "blk.0.attn_q.weight"), value: [out * in] delta
    std::unordered_map<std::string, std::vector<float>> deltas;
};

// LoRA state --- stored as static per transformer instance
// (proper impl would be a member, but we add it here as external storage)
namespace lora_state {
    static std::unordered_map<const Transformer*, std::vector<LoRAAdapter>> adapters;
    static std::mutex mu;
}

inline bool Transformer::lora_load(const std::string& path, float scale) {
    gguf::Parser parser;
    if (!parser.open(path)) {
        fprintf(stderr, "[lora] Failed to open %s\n", path.c_str());
        return false;
    }

    LoRAAdapter adapter;
    adapter.scale = scale;
    adapter.path  = path;

    // Read file for tensor extraction
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long fsz = ftell(f); rewind(f);
    std::vector<uint8_t> buf(fsz);
    (void)fread(buf.data(), 1, fsz, f);
    fclose(f);

    size_t total_bytes = 0;
    for (auto& [n, t] : parser.tensors()) total_bytes += t.data_size;
    size_t data_start = (buf.size() - total_bytes) & ~(size_t)31;

    const int hidden = (int)meta_.hidden_dim;

    // LoRA tensors come in pairs: "blk.N.attn_q.weight.lora_a" and ".lora_b"
    // We compute the delta = lora_b @ lora_a and store it
    for (auto& [name, td_a] : parser.tensors()) {
        if (name.find(".lora_a") == std::string::npos) continue;
        std::string base_name = name.substr(0, name.size() - 7); // strip ".lora_a"
        std::string b_name    = base_name + ".lora_b";

        gguf::TensorDesc* td_b = parser.find_tensor(b_name);
        if (!td_b) continue;

        int rank    = (int)td_a.shape[0]; // [rank, in_dim]
        int in_dim  = (int)td_a.shape.size() > 1 ? (int)td_a.shape[1] : hidden;
        int out_dim = (int)td_b->shape.size() > 1 ? (int)td_b->shape[0] : hidden;

        // Dequantize A and B
        std::vector<float> A(rank * in_dim), B(out_dim * rank);
        quant::QuantType qt_a = (td_a.type == gguf::GGMLType::F16) ?
            quant::QuantType::F16 : quant::QuantType::F32;
        quant::QuantType qt_b = (td_b->type == gguf::GGMLType::F16) ?
            quant::QuantType::F16 : quant::QuantType::F32;

        quant::dequantize(buf.data() + data_start + td_a.offset,
                          A.data(), A.size(), qt_a);
        quant::dequantize(buf.data() + data_start + td_b->offset,
                          B.data(), B.size(), qt_b);

        // delta = scale * B @ A: [out_dim, in_dim]
        std::vector<float> delta(out_dim * in_dim, 0.0f);
        for (int o = 0; o < out_dim; ++o)
            for (int r = 0; r < rank; ++r) {
                float b_val = B[o * rank + r] * scale;
                for (int i = 0; i < in_dim; ++i)
                    delta[o * in_dim + i] += b_val * A[r * in_dim + i];
            }
        adapter.deltas[base_name] = std::move(delta);
    }

    fprintf(stderr, "[lora] Loaded %s: %zu weight deltas, scale=%.2f\n",
            path.c_str(), adapter.deltas.size(), scale);

    std::lock_guard<std::mutex> lg(lora_state::mu);
    lora_state::adapters[this].push_back(std::move(adapter));
    return true;
}

inline void Transformer::lora_unload_all() {
    std::lock_guard<std::mutex> lg(lora_state::mu);
    lora_state::adapters.erase(this);
}

inline void Transformer::lora_set_scale(int idx, float scale) {
    std::lock_guard<std::mutex> lg(lora_state::mu);
    auto it = lora_state::adapters.find(this);
    if (it != lora_state::adapters.end() && idx < (int)it->second.size())
        it->second[idx].scale = scale;
}

inline int Transformer::lora_count() const {
    std::lock_guard<std::mutex> lg(lora_state::mu);
    auto it = lora_state::adapters.find(this);
    return it == lora_state::adapters.end() ? 0 : (int)it->second.size();
}

inline void Transformer::q35_replay_state(int layer_idx, int keep) {
    // The recurrent part of forward_qwen35_linear_batch for tokens [0, keep) of the last check, state only (no output):
    // causal conv with its state, then the gated delta-rule update. Same math as that function's CPU path.
    const TransformerLayerWeights& lw = layer_weights_[(size_t)layer_idx];
    const Q35VerifyRec& r = q35_vrec_[(size_t)layer_idx];
    const int conv_dim = lw.q35_conv_dim, key_dim = lw.q35_key_dim;
    const int kh = lw.q35_num_k_heads, vh = lw.q35_num_v_heads, hd = lw.q35_head_dim, kernel = lw.q35_kernel;
    const int repeat = kh > 0 ? vh / kh : 0;
    if (repeat <= 0 || hd <= 0 || hd > 256 || conv_dim <= 0 || key_dim <= 0) return;
    std::vector<float>& conv_state = q35_conv_state_[(size_t)layer_idx];
    std::vector<float>& rec_state  = q35_recurrent_state_[(size_t)layer_idx];
    const size_t conv_len = (size_t)conv_dim * (size_t)std::max(0, kernel - 1), rec_len = (size_t)vh * hd * hd;
    if (conv_state.size() != conv_len) conv_state.assign(conv_len, 0.0f);
    if (rec_state.size()  != rec_len)  rec_state.assign(rec_len, 0.0f);
    const char* qkv_layout_env = std::getenv("NEXUS_QWEN35_LINEAR_QKV_LAYOUT");
    const bool grouped = (qkv_layout_env && std::strcmp(qkv_layout_env, "grouped") == 0);
    std::vector<float> convt((size_t)conv_dim), k_flat((size_t)key_dim), v_flat((size_t)vh * hd);
    for (int t = 0; t < keep; ++t) {
        const float* mixt = r.mixed.data() + (size_t)t * (size_t)conv_dim;
        for (int c = 0; c < conv_dim; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < kernel; ++k) {
                const float xv = (k < kernel - 1) ? conv_state[(size_t)c * (size_t)(kernel - 1) + (size_t)k] : mixt[c];
                sum += xv * lw.q35_conv[(size_t)c * (size_t)kernel + (size_t)k];
            }
            convt[(size_t)c] = sum / (1.0f + expf(-sum));
        }
        if (kernel > 1)
            for (int c = 0; c < conv_dim; ++c) {
                float* st = conv_state.data() + (size_t)c * (size_t)(kernel - 1);
                for (int k = 0; k < kernel - 2; ++k) st[k] = st[k + 1];
                st[kernel - 2] = mixt[c];
            }
        const float* v_base;
        if (grouped) {
            const int span = 2 * hd + repeat * hd;
            for (int hk = 0; hk < kh; ++hk) {
                const size_t src = (size_t)hk * (size_t)span;
                std::memcpy(k_flat.data() + (size_t)hk * hd, convt.data() + src + (size_t)hd, (size_t)hd * sizeof(float));
                for (int rr = 0; rr < repeat; ++rr)
                    std::memcpy(v_flat.data() + ((size_t)hk * repeat + rr) * hd,
                                convt.data() + src + (size_t)(2 * hd) + (size_t)rr * hd, (size_t)hd * sizeof(float));
            }
            v_base = v_flat.data();
        } else {
            std::memcpy(k_flat.data(), convt.data() + key_dim, (size_t)key_dim * sizeof(float));
            v_base = convt.data() + 2 * (size_t)key_dim;
        }
        for (int hk = 0; hk < kh; ++hk) nx_l2norm_inplace(k_flat.data() + (size_t)hk * hd, hd, 1e-6f);
        const float* at = r.a.data() + (size_t)t * (size_t)vh;
        const float* bt = r.b.data() + (size_t)t * (size_t)vh;
        for (int hv = 0; hv < vh; ++hv) {
            const float* k = k_flat.data() + (size_t)(hv / repeat) * hd;
            const float* v = v_base + (size_t)hv * hd;
            const float beta = nx_sigmoid_f32(bt[hv]);
            float g = lw.q35_A[hv] * nx_softplus_f32(at[hv] + lw.q35_dt[hv]);
            if (!std::isfinite(g)) g = -60.0f;
            if (g > 0.0f) g = 0.0f;
            if (g < -60.0f) g = -60.0f;
            const float decay = expf(g);
            float* S = rec_state.data() + (size_t)hv * hd * hd;
            for (int i = 0; i < hd * hd; ++i) S[i] *= decay;
            float kv_mem[256], delta[256];
            for (int j = 0; j < hd; ++j) { float s2 = 0.0f; for (int i = 0; i < hd; ++i) s2 += k[i] * S[(size_t)i * hd + j]; kv_mem[j] = s2; }
            for (int j = 0; j < hd; ++j) delta[j] = (v[j] - kv_mem[j]) * beta;
            for (int i = 0; i < hd; ++i) { const float ki = k[i]; float* Sr = S + (size_t)i * hd; for (int j = 0; j < hd; ++j) Sr[j] += ki * delta[j]; }
        }
    }
}

inline bool Transformer::mtp_keep_checked(int keep, int pos) {
    const int hidden = (int)meta_.hidden_dim;
    if (keep < 1 || keep > mtp_verify_n_ || (int)mtp_verify_hidden_.size() < mtp_verify_n_ * hidden) return false;
    if ((int)q35_vrec_.size() < (int)meta_.n_layers) return false;
    std::vector<int> lin;
    for (int l = 0; l < (int)meta_.n_layers; ++l) {
        if (!layer_weights_[(size_t)l].qwen35_linear) continue;
        const Q35VerifyRec& r = q35_vrec_[(size_t)l];
        if (r.n != mtp_verify_n_ || layer_weights_[(size_t)l].moe) return false;   // this layer wasn't recorded: re-run instead
        lin.push_back(l);
    }
    if ((int)q35_conv_state_.size() < (int)meta_.n_layers) q35_conv_state_.resize(meta_.n_layers);
    if ((int)q35_recurrent_state_.size() < (int)meta_.n_layers) q35_recurrent_state_.resize(meta_.n_layers);
    nx_par_range((int)lin.size(), n_threads_, [&](int a, int b) { for (int i = a; i < b; ++i) q35_replay_state(lin[(size_t)i], keep); });
    kvcache_.set_n_tokens(pos + keep);             // the rejected drafts' memory is past the end; it is overwritten next
    if (x_scratch_.size() >= (size_t)hidden)
        std::memcpy(x_scratch_.data(), mtp_verify_hidden_.data() + (size_t)(keep - 1) * hidden, (size_t)hidden * sizeof(float));
    return true;
}

inline int Transformer::forward_verifyN(
    int32_t confirmed_tok, const std::vector<int32_t>& drafts, int pos,
    std::vector<std::vector<float>>& out_logits)
{
    out_logits.clear();
    if (!loaded_) return 0;
    const int k = (int)drafts.size();
    if (k <= 0) return 0;
    const int total = k + 1;  // confirmed + k drafts

    // Build the token window [confirmed, draft0, .., draft(k-1)] at [pos..pos+k].
    std::vector<int32_t> seq((size_t)pos + (size_t)total);
    seq[pos] = confirmed_tok;
    for (int j = 0; j < k; ++j) seq[pos + 1 + j] = drafts[(size_t)j];

    // One batched sweep, capture logits at every position. plain decode cost
    // (no per-token prefill streaming machinery).
    q35_vrec_.assign((size_t)meta_.n_layers, Q35VerifyRec{});
    mtp_verify_hidden_.clear();
    mtp_verify_n_ = total;
    mtp_verify_all_logits_ = &out_logits;
    prefill_batch(seq, pos, total, /*force_no_stream=*/true);
    mtp_verify_all_logits_ = nullptr;
    return (int)out_logits.size();
}

inline void Transformer::forward_verify2(
    int32_t confirmed_tok, int32_t draft_tok, int pos,
    std::vector<float>& out_logits_confirmed,
    std::vector<float>& out_logits_draft)
{
    const int vocab = (int)meta_.vocab_size;
    out_logits_confirmed.assign(vocab, 0.0f);
    out_logits_draft.assign(vocab, 0.0f);
    if (!loaded_) return;

    // Run BOTH tokens through the layers in ONE batched sweep so layer weights
    // stream across the bus only once for the pair — this is what makes MTP
    // faster than two single-token forwards on a streaming-bound engine. The
    // batched prefill path already does correct 2-token DeltaNet/attention math
    // and per-position residuals; we ask it to ALSO emit logits at the first of
    // the two positions (the draft-verification distribution) via
    // mtp_verify_second_logits_, in addition to the last (the bonus-token dist).
    std::vector<int32_t> two((size_t)pos + 2);
    two[pos]     = confirmed_tok;
    two[pos + 1] = draft_tok;

    mtp_verify_second_logits_ = &out_logits_confirmed;  // capture position pos (n-2)
    prefill_batch(two, pos, 2, /*force_no_stream=*/true); // plain decode-cost path
    mtp_verify_second_logits_ = nullptr;
    out_logits_draft = logits_;                          // distribution after draft
}

// ── MTP implementation ───────────────────────────────────────────────────────
inline bool Transformer::mtp_load() {
    if (mtp_.loaded) {
#ifdef NEXUS_VULKAN
        // Already loaded — but if GPU was just turned on, pin now (idempotent).
        if (mtp_.available && mtp_use_gpu_ && gpu::gpu().ready()) {
            auto pin = [&](const void* W){ if (W) gpu::gpu().pin_weight(W); };
            pin(mtp_.fc); pin(mtp_.wq); pin(mtp_.wk); pin(mtp_.wv); pin(mtp_.wo);
            pin(mtp_.w_gate); pin(mtp_.w_up); pin(mtp_.w_down);
        }
#endif
        return mtp_.available;
    }
    mtp_.loaded = true;
    mtp_.available = false;
    if (!parser_) return false;

    auto qd = [&](const char* n)->const void* {
        TensorDesc* t = find(n); return t ? t->data : nullptr;
    };
    auto qt = [&](const char* n)->QuantType {
        TensorDesc* t = find(n); return t ? ggml_to_qtype(t->type) : QuantType{};
    };
    auto f32 = [&](const char* n)->const float* {
        TensorDesc* t = find(n); return t ? tensor_as_f32(t, n) : nullptr;
    };

    mtp_.fc   = qd("mtp.fc.weight");                 mtp_.fc_type = qt("mtp.fc.weight");
    mtp_.wq   = qd("mtp.layers.0.self_attn.q_proj.weight"); mtp_.wq_type = qt("mtp.layers.0.self_attn.q_proj.weight");
    mtp_.wk   = qd("mtp.layers.0.self_attn.k_proj.weight"); mtp_.wk_type = qt("mtp.layers.0.self_attn.k_proj.weight");
    mtp_.wv   = qd("mtp.layers.0.self_attn.v_proj.weight"); mtp_.wv_type = qt("mtp.layers.0.self_attn.v_proj.weight");
    mtp_.wo   = qd("mtp.layers.0.self_attn.o_proj.weight"); mtp_.wo_type = qt("mtp.layers.0.self_attn.o_proj.weight");
    mtp_.w_gate = qd("mtp.layers.0.mlp.gate_proj.weight"); mtp_.w_gate_type = qt("mtp.layers.0.mlp.gate_proj.weight");
    mtp_.w_up   = qd("mtp.layers.0.mlp.up_proj.weight");   mtp_.w_up_type   = qt("mtp.layers.0.mlp.up_proj.weight");
    mtp_.w_down = qd("mtp.layers.0.mlp.down_proj.weight"); mtp_.w_down_type = qt("mtp.layers.0.mlp.down_proj.weight");
    mtp_.pre_fc_norm_embed  = f32("mtp.pre_fc_norm_embedding.weight");
    mtp_.pre_fc_norm_hidden = f32("mtp.pre_fc_norm_hidden.weight");
    mtp_.final_norm         = f32("mtp.norm.weight");
    mtp_.q_norm             = f32("mtp.layers.0.self_attn.q_norm.weight");
    mtp_.k_norm             = f32("mtp.layers.0.self_attn.k_norm.weight");
    mtp_.input_norm         = f32("mtp.layers.0.input_layernorm.weight");
    mtp_.post_attn_norm     = f32("mtp.layers.0.post_attention_layernorm.weight");

    const int hidden = (int)meta_.hidden_dim;
    const int q_dim  = (int)meta_.n_heads * (int)meta_.head_dim;
    if (TensorDesc* tq = find("mtp.layers.0.self_attn.q_proj.weight")) {
        // q_proj rows: 2*q_dim => gated (q + gate interleaved), else q_dim.
        long rows = (tq->shape.size() >= 1) ? (long)tq->shape[0] : (long)q_dim;
        // CSV shows [12288,5120] = [rows,cols]; shape[0] is rows.
        mtp_.q_rows = (int)rows;
        mtp_.q_has_gate = (rows >= 2 * (long)q_dim);
    } else {
        mtp_.q_rows = q_dim;
    }

    const bool core = mtp_.fc && mtp_.wq && mtp_.wk && mtp_.wv && mtp_.wo &&
                      mtp_.w_gate && mtp_.w_up && mtp_.w_down &&
                      mtp_.final_norm && mtp_.input_norm && mtp_.post_attn_norm;
    mtp_.available = core && hidden > 0;
    fprintf(stderr, "[mtp] load: available=%d gated_q=%d q_rows=%d (fc=%p wq=%p gate=%p)\n",
            mtp_.available ? 1 : 0, mtp_.q_has_gate ? 1 : 0, mtp_.q_rows,
            mtp_.fc, mtp_.wq, mtp_.w_gate);

#ifdef NEXUS_VULKAN
    // If MTP is to run on the GPU, pin its weights resident NOW so they upload to
    // VRAM exactly once and stay there. Without pinning, each per-token mat_mat
    // would re-evaluate residency and could re-upload — the MTP block is tiny
    // (one decoder layer + fc), so it fits permanently even on a small card.
    // Pinning makes the GPU path actually fast instead of re-streaming each step.
    if (mtp_.available && mtp_use_gpu_ && gpu::gpu().ready()) {
        auto pin = [&](const void* W){ if (W) gpu::gpu().pin_weight(W); };
        pin(mtp_.fc);   pin(mtp_.wq); pin(mtp_.wk); pin(mtp_.wv); pin(mtp_.wo);
        pin(mtp_.w_gate); pin(mtp_.w_up); pin(mtp_.w_down);
        mtp_reserve_bytes_ = 0;           // on the card now: the real free reading counts them from here on
        fprintf(stderr, "[mtp] GPU: pinned MTP weights resident in VRAM\n");
    }
#endif
    return mtp_.available;
}

inline int Transformer::mtp_propose(int32_t last_token, int pos,
                                    const model::SamplerConfig& scfg,
                                    std::vector<int32_t>& out) {
    out.clear();
    if (!mtp_enabled()) return 0;
    if (!mtp_.available && !mtp_load()) return 0;

    const int hidden = (int)meta_.hidden_dim;
    const int n_heads = (int)meta_.n_heads;
    const int n_kv    = (int)meta_.n_kv_heads;
    const int head_dim = (int)meta_.head_dim;
    const int q_dim   = n_heads * head_dim;
    const int kv_dim  = n_kv * head_dim;
    const float eps   = meta_.rms_norm_eps;
    if ((int)x_scratch_.size() < hidden) return 0;

    // Running hidden state seed = main model's last hidden (for token `pos`).
    std::vector<float> h(x_scratch_.begin(), x_scratch_.begin() + hidden);
    int32_t prev_tok = last_token;

    AttentionConfig cfg = attn_cfg_;
    cfg.n_heads = n_heads; cfg.n_kv_heads = n_kv; cfg.head_dim = head_dim;
    const float scale = (cfg.scale != 0.0f) ? cfg.scale : 1.0f/std::sqrt((float)head_dim);

    const int n = mtp_n_predict_;
    out.reserve(n);

    // Sync MTP's KV history to the accepted sequence length. `pos` is the
    // position whose hidden state we were handed (the last ACCEPTED token).
    // Any speculative k/v appended in a prior call beyond this are unaccepted —
    // truncate them so the context matches the real, committed sequence.
    {
        const int kvstride = n_kv * head_dim;
        int want = pos;                     // committed positions before this one
        if (want < 0) want = 0;
        if (mtp_hist_len_ > want) {
            mtp_hist_len_ = want;
            mtp_k_hist_.resize((size_t)want * kvstride);
            mtp_v_hist_.resize((size_t)want * kvstride);
        }
    }

    // Projection helper: GPU mat_mat (single column = mat_vec equivalent) when
    // the GPU toggle is on and the type is GPU-capable; else CPU mat_vec. Output
    // is numerically equivalent either way (single-token = one-column GEMM).
    auto project = [&](const void* W, QuantType wt, const float* X, float* Y,
                       size_t rows, size_t cols) {
#ifdef NEXUS_VULKAN
        if (mtp_use_gpu_ && gpu::gpu().ready() && wt == QuantType::NQ4_G64) {
            if (gpu::gpu().mat_mat(W, X, Y, rows, cols, 1, wt)) return;
        }
#endif
        cpu::mat_vec(W, X, Y, rows, cols, wt, n_threads_);
    };

    for (int step = 0; step < n; ++step) {
        // 1. Embed the previous token.
        std::vector<float> emb(hidden);
        quant::dequantize_row(token_embedding_table, emb.data(),
                              (size_t)prev_tok, (size_t)hidden, emb_type);
        if (embed_scale_ > 0.0f) for (int i=0;i<hidden;++i) emb[i] *= embed_scale_;

        // 2. Norm hidden + embed, concat, fc -> hidden.
        // Reference (DeepSeek-V3 / Qwen MTP) concatenates EMBEDDING first, then
        // hidden: fc * concat(norm_emb(embed(t+1)), norm_hidden(h_t)).
        std::vector<float> hn(hidden), en(hidden);
        nx_apply_model_norm(h.data(),   mtp_.pre_fc_norm_hidden, nullptr, hn.data(), hidden, eps, zero_centered_norm_);
        nx_apply_model_norm(emb.data(), mtp_.pre_fc_norm_embed,  nullptr, en.data(), hidden, eps, zero_centered_norm_);
        std::vector<float> cat(2*hidden);
        std::copy(en.begin(), en.end(), cat.begin());            // embedding first
        std::copy(hn.begin(), hn.end(), cat.begin()+hidden);     // hidden second
        std::vector<float> x(hidden);
        project(mtp_.fc, mtp_.fc_type, cat.data(), x.data(), (size_t)hidden, (size_t)(2*hidden));

        // 3. One decoder layer (attention is over a single token: trivial self-attn).
        std::vector<float> xb(hidden);
        nx_apply_model_norm(x.data(), mtp_.input_norm, nullptr, xb.data(), hidden, eps, zero_centered_norm_);

        std::vector<float> q(mtp_.q_has_gate ? q_dim*2 : q_dim), k(kv_dim), v(kv_dim);
        project(mtp_.wq, mtp_.wq_type, xb.data(), q.data(), (size_t)(mtp_.q_has_gate?q_dim*2:q_dim), (size_t)hidden);
        project(mtp_.wk, mtp_.wk_type, xb.data(), k.data(), (size_t)kv_dim, (size_t)hidden);
        project(mtp_.wv, mtp_.wv_type, xb.data(), v.data(), (size_t)kv_dim, (size_t)hidden);

        // Split gate if present (perhead [q|gate]).
        std::vector<float> qpure(q_dim), gate;
        if (mtp_.q_has_gate) {
            gate.resize(q_dim);
            for (int hh=0; hh<n_heads; ++hh) {
                std::copy(q.begin()+hh*2*head_dim,            q.begin()+hh*2*head_dim+head_dim,   qpure.begin()+hh*head_dim);
                std::copy(q.begin()+hh*2*head_dim+head_dim,   q.begin()+hh*2*head_dim+2*head_dim, gate.begin()+hh*head_dim);
            }
        } else {
            std::copy(q.begin(), q.begin()+q_dim, qpure.begin());
        }

        // q/k norm (zero-centered, gated style) — reuse the helper via cfg.
        // Apply q_norm/k_norm per head if present.
        auto per_head_norm = [&](float* vec, int heads, const float* w) {
            if (!w) return;
            for (int hh=0; hh<heads; ++hh) {
                float* hv = vec + hh*head_dim; double ss=0;
                for (int d=0; d<head_dim; ++d) ss += (double)hv[d]*hv[d];
                float inv = 1.0f/std::sqrt((float)(ss/head_dim)+eps);
                for (int d=0; d<head_dim; ++d) hv[d] = hv[d]*inv*(zero_centered_norm_ ? (1.0f+w[d]) : w[d]);
            }
        };
        per_head_norm(qpure.data(), n_heads, mtp_.q_norm);
        per_head_norm(k.data(),     n_kv,    mtp_.k_norm);

        // RoPE at this effective position (partial, Qwen3.5 style).
        const int mtp_rope_pos = rope_effective_position(pos + 1 + step);
        nx_apply_rope_qwen35_partial_or_full(qpure.data(), n_heads, mtp_rope_pos, head_dim, cfg.rope, true);
        nx_apply_rope_qwen35_partial_or_full(k.data(),     n_kv,    mtp_rope_pos, head_dim, cfg.rope, true);

        // Append this position's k/v to MTP's own history, then attend the query
        // over the FULL history [0..now] (causal). This is the context the MTP
        // head was trained with — without it, proposals never match (0% accept).
        const int kvstride = n_kv * head_dim;
        mtp_k_hist_.insert(mtp_k_hist_.end(), k.begin(), k.begin() + kvstride);
        mtp_v_hist_.insert(mtp_v_hist_.end(), v.begin(), v.begin() + kvstride);
        mtp_hist_len_ += 1;
        const int nctx = mtp_hist_len_;

        std::vector<float> attn(q_dim, 0.0f);
        for (int hh = 0; hh < n_heads; ++hh) {
            const int kvh = hh * n_kv / n_heads;
            const float* qh = qpure.data() + hh * head_dim;
            // online softmax over all stored positions for this head
            std::vector<float> acc(head_dim, 0.0f);
            float m = -1e30f, l = 0.0f;
            for (int p = 0; p < nctx; ++p) {
                const float* kp = mtp_k_hist_.data() + ((size_t)p * kvstride) + (size_t)kvh * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += qh[d] * kp[d];
                float s = dot * scale;
                if (cfg.attn_softcap > 0.0f) s = cfg.attn_softcap * std::tanh(s / cfg.attn_softcap);
                float mn = std::max(m, s), cr = std::exp(m - mn), w = std::exp(s - mn);
                const float* vp = mtp_v_hist_.data() + ((size_t)p * kvstride) + (size_t)kvh * head_dim;
                for (int d = 0; d < head_dim; ++d) acc[d] = acc[d] * cr + w * vp[d];
                l = l * cr + w; m = mn;
            }
            float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
            float* ah = attn.data() + hh * head_dim;
            for (int d = 0; d < head_dim; ++d) ah[d] = acc[d] * inv;
        }
        // gate
        if (mtp_.q_has_gate) {
            for (int i=0;i<q_dim;++i) {
                float z=gate[i]; float s=(z>80.f)?1.f:(z<-80.f)?0.f:1.f/(1.f+std::exp(-z));
                attn[i]*=s;
            }
        }
        std::vector<float> ao(hidden);
        project(mtp_.wo, mtp_.wo_type, attn.data(), ao.data(), (size_t)hidden, (size_t)q_dim);
        for (int i=0;i<hidden;++i) x[i] += ao[i];   // residual

        // FFN (SwiGLU).
        nx_apply_model_norm(x.data(), mtp_.post_attn_norm, nullptr, xb.data(), hidden, eps, zero_centered_norm_);
        const int ffn = (int)meta_.ffn_dim;
        std::vector<float> g(ffn), u(ffn);
        project(mtp_.w_gate, mtp_.w_gate_type, xb.data(), g.data(), (size_t)ffn, (size_t)hidden);
        project(mtp_.w_up, mtp_.w_up_type, xb.data(), u.data(), (size_t)ffn, (size_t)hidden);
        for (int i=0;i<ffn;++i){ float gv=g[i]; g[i]= (gv/(1.f+std::exp(-gv)))*u[i]; }
        std::vector<float> fo(hidden);
        project(mtp_.w_down, mtp_.w_down_type, g.data(), fo.data(), (size_t)hidden, (size_t)ffn);
        for (int i=0;i<hidden;++i) x[i] += fo[i];

        // 4. Final norm + reuse main lm_head -> logits -> sample.
        std::vector<float> xn(hidden);
        nx_apply_model_norm(x.data(), mtp_.final_norm, nullptr, xn.data(), hidden, eps, zero_centered_norm_);
        // Reuse the main lm_head and the EXACT main sampler by writing into
        // logits_ and calling sample(). Save/restore logits_ so we don't disturb
        // the main model's state.
        std::vector<float> saved = logits_;
        if ((int)logits_.size() < (int)meta_.vocab_size) logits_.assign(meta_.vocab_size, 0.0f);
        cpu::mat_vec(lm_head_weight, xn.data(), logits_.data(), (size_t)meta_.vocab_size, (size_t)hidden, lm_head_type, n_threads_);
        if (lm_head_bias) for (uint32_t i=0;i<meta_.vocab_size;++i) logits_[i]+=lm_head_bias[i];
        int32_t tok = sample(scfg);
        logits_ = saved;
        out.push_back(tok);
        // feed forward for the next MTP step
        prev_tok = tok;
        h = x;   // chain hidden state
    }
    return (int)out.size();
}


// Target model verifies all n_draft at once (one batched forward, amortized cost).
// Accepted tokens are emitted; first rejected token triggers resample from target.

inline void Transformer::generate_speculative(
    const std::vector<int32_t>& prompt_tokens,
    Transformer*                draft,
    const SamplerConfig&        cfg,
    int                         max_tokens,
    int                         n_draft,
    TokenCB                     on_token)
{
    // Prefill both models with the prompt
    std::vector<int32_t> context = prompt_tokens;
    for (int i = 0; i < (int)prompt_tokens.size(); ++i) {
        this->forward(context, i);
        draft->forward(context, i);
    }
    int pos = (int)prompt_tokens.size();

    SamplerConfig draft_cfg = cfg;  // draft uses same sampler settings
    draft_cfg.seed = cfg.seed;

    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    while ((int)generated.size() < max_tokens) {
        // ------ Draft: generate n_draft tokens ------------------------------------------------------------------------------------------------
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(n_draft);
        for (int d = 0; d < n_draft; ++d) {
            int32_t last = context.empty() ? 1 : context.back();
            context.push_back(last);
            draft->forward(context, pos + d);
            draft_cfg.recent_tokens = &context;
            int32_t dt = draft->sample(draft_cfg);
            draft_tokens.push_back(dt);
            context.back() = dt;  // update with draft token for next step
        }
        context.resize(context.size() - n_draft); // remove draft tokens from context

        // ------ Target: verify all draft tokens in one pass ------------------------------------------------------------
        // Run target model forward at each draft position
        int n_accepted = 0;
        for (int d = 0; d < n_draft; ++d) {
            int verify_pos = pos + d;
            context.push_back(draft_tokens[d]);
            this->forward(context, verify_pos);

            // Target distribution at this position
            SamplerConfig tcfg = cfg;
            tcfg.recent_tokens = &context;
            tcfg.seed = -1; // random for target

            // Simple rejection sampling:
            // Accept draft token if it's in top-k of target distribution
            const auto& target_logits = this->logits_;
            float draft_logit = target_logits[draft_tokens[d]];
            float max_logit   = *std::max_element(target_logits.begin(), target_logits.end());
            bool accept = (draft_logit >= max_logit - 2.0f); // within top-e^2 of max

            if (accept) {
                ++n_accepted;
                generated.push_back(draft_tokens[d]);
                std::string tok_str = this->detokenize(draft_tokens[d]);
                if (!on_token(tok_str, draft_tokens[d])) goto done;
                if ((int)generated.size() >= max_tokens) goto done;
            } else {
                // Rejection: resample from target at this position
                int32_t target_tok = this->sample(tcfg);
                generated.push_back(target_tok);
                context.back() = target_tok;
                std::string tok_str = this->detokenize(target_tok);
                if (!on_token(tok_str, target_tok)) goto done;
                if ((int)generated.size() >= max_tokens) goto done;
                // Reset draft model to match target's accepted context
                for (int i = 0; i <= d; ++i)
                    draft->forward(context, pos + i);
                n_accepted = d; // stop accepting drafts after rejection
                break;
            }
        }

        // Generate one bonus token from target (free after verification pass)
        if (n_accepted == n_draft && (int)generated.size() < max_tokens) {
            SamplerConfig tcfg = cfg; tcfg.recent_tokens = &context;
            int32_t bonus = this->sample(tcfg);
            context.push_back(bonus);
            generated.push_back(bonus);
            std::string tok_str = this->detokenize(bonus);
            if (!on_token(tok_str, bonus)) goto done;
        }

        pos += n_accepted + 1;
    }
    done:;
    fprintf(stderr, "[speculative] Generated %d tokens\n", (int)generated.size());
}

// --------- Tokenization (simple BPE for now) ---------------------------------------------------------------------------------------------------------------------

inline std::vector<int32_t> Transformer::tokenize(const std::string& text, bool add_bos) const {
    std::vector<int32_t> tokens;
    if (add_bos) tokens.push_back(1);  // BOS token (token 1 for most models)

    if (vocab_tokens_.empty()) {
        fprintf(stderr, "[nexus] Warning: tokenizer not loaded, returning BOS only\n");
        return tokens;
    }

    // HARD special-token prepass for ChatML/Qwen-style templates.
    //
    // This fixes the exact failure seen in the prompt dump:
    //   ".<" "|i" "m" "_end" "|" ">"
    // instead of:
    //   "." "<|im_end|>"
    //
    // The critical rule is: normal vocab tokens are never allowed to cross
    // into the byte position where a known special marker begins.
    struct NxSpecialTok { std::string text; int id; };
    std::vector<NxSpecialTok> specials;
    specials.reserve(64);

    auto find_token_id = [&](const std::string& needle)->int {
        for (size_t i = 0; i < vocab_tokens_.size(); ++i) {
            if (vocab_tokens_[i] == needle) return (int)i;
        }
        return -1;
    };

    auto add_special = [&](const std::string& s, int fallback_id = -1) {
        int id = find_token_id(s);
        if (id < 0 && fallback_id >= 0 && (size_t)fallback_id < vocab_tokens_.size()) {
            // Qwen tokenizer IDs are stable for these markers in this model.
            // This fallback is only used if parser loaded the token but the
            // string table lookup missed it for any reason.
            id = fallback_id;
        }
        if (id < 0) return;
        for (const auto& sp : specials) {
            if (sp.text == s || sp.id == id) return;
        }
        specials.push_back({s, id});
    };

    // Qwen/Qwen3 core markers. Keep these explicit so they work even if the
    // generic added-token scan misses one.
    add_special("<|im_start|>", 248045);
    add_special("<|im_end|>",   248046);
    add_special("<think>",      248068);
    add_special("</think>",     248069);
    add_special("<|endoftext|>", 151643);

    // Generic special scan for any other ChatML/fullwidth markers present.
    for (size_t i = 0; i < vocab_tokens_.size(); ++i) {
        const std::string& tok = vocab_tokens_[i];
        if (tok.empty()) continue;
        const bool chatml = (tok.rfind("<|", 0) == 0 && tok.size() >= 4 &&
                             tok.compare(tok.size() - 2, 2, "|>") == 0);
        const bool think = (tok == "<think>" || tok == "</think>");
        const bool fullwidth = (tok.rfind("<｜", 0) == 0 && tok.size() >= 6);
        if (chatml || think || fullwidth) add_special(tok, (int)i);
    }

    std::sort(specials.begin(), specials.end(), [](const NxSpecialTok& a, const NxSpecialTok& b) {
        return a.text.size() > b.text.size();
    });

    static bool nx_printed_special_guard = false;
    if (!nx_printed_special_guard) {
        int im_start_id = find_token_id("<|im_start|>");
        int im_end_id   = find_token_id("<|im_end|>");
        fprintf(stderr,
                "[tokenizer] special-boundary guard active: specials=%zu im_start=%d im_end=%d\n",
                specials.size(), im_start_id, im_end_id);
        fflush(stderr);
        nx_printed_special_guard = true;
    }

    auto special_at = [&](size_t pos, int& out_id, size_t& out_len)->bool {
        for (const auto& sp : specials) {
            if (pos + sp.text.size() <= text.size() &&
                text.compare(pos, sp.text.size(), sp.text) == 0) {
                out_id = sp.id;
                out_len = sp.text.size();
                return true;
            }
        }
        return false;
    };

    auto next_special_pos = [&](size_t pos)->size_t {
        size_t best = std::string::npos;
        for (const auto& sp : specials) {
            size_t p = text.find(sp.text, pos);
            if (p != std::string::npos && p < best) best = p;
        }
        return best;
    };

    size_t pos = 0;
    while (pos < text.size()) {
        int sid = -1;
        size_t slen = 0;
        if (special_at(pos, sid, slen)) {
            tokens.push_back((int32_t)sid);
            pos += slen;
            continue;
        }

        const size_t next_sp = next_special_pos(pos);
        const size_t hard_limit = (next_sp == std::string::npos) ? text.size() : next_sp;
        const size_t max_normal_len = (hard_limit > pos) ? (hard_limit - pos) : 1;

        int best_id = -1;
        size_t best_len = 0;

        for (size_t i = 0; i < vocab_tokens_.size(); ++i) {
            const auto& tok = vocab_tokens_[i];
            if (tok.empty()) continue;

            // Never let an ordinary vocab token eat bytes that belong to a
            // following special marker. This is the actual .<|im_end|> fix.
            if (tok.size() > max_normal_len) continue;

            if (pos + tok.size() <= text.size() &&
                text.compare(pos, tok.size(), tok) == 0) {
                if (tok.size() > best_len) {
                    best_len = tok.size();
                    best_id  = (int)i;
                }
            }
        }

        if (best_id >= 0 && best_len > 0) {
            tokens.push_back((int32_t)best_id);
            pos += best_len;
        } else {
            // Unknown byte --- encode as individual byte token.
            const unsigned char c = (unsigned char)text[pos];
            char hex_tok[8];
            snprintf(hex_tok, sizeof(hex_tok), "<0x%02X>", c);
            int byte_id = find_token_id(hex_tok);
            tokens.push_back(byte_id >= 0 ? (int32_t)byte_id : 0);
            pos += 1;
        }
    }

    return tokens;
}

inline std::string Transformer::detokenize(int32_t token_id) const {
    if (token_id < 0 || token_id >= (int)vocab_tokens_.size()) return "";
    std::string tok = vocab_tokens_[token_id];

    // Handle sentencepiece --- prefix (represents space)
    // UTF-8 for --- is E2 96 81
    const std::string sp_space = "\xe2\x96\x81";
    size_t pos = tok.find(sp_space);
    while (pos != std::string::npos) {
        tok.replace(pos, sp_space.size(), " ");
        pos = tok.find(sp_space, pos + 1);
    }

    return tok;
}

inline std::string Transformer::detokenize(const std::vector<int32_t>& toks) const {
    std::string result;
    for (auto t : toks) result += detokenize(t);
    return result;
}



inline void Transformer::prefill_embeddings_cpu_prompt_batch(const float* embeddings,
                                                            int hidden_dim,
                                                            int start_pos,
                                                            int n)
{
    if (!loaded_ || !embeddings || n <= 0) return;

    const int hidden = (int)meta_.hidden_dim;
    if (hidden_dim != hidden) {
        fprintf(stderr,
            "[cpu-prefill:emb] rejected: embedding hidden=%d model hidden=%d n=%d\n",
            hidden_dim, hidden, n);
        return;
    }

    // IMPORTANT:
    // Image/mixed embeddings must NOT use the layer-major prefill skeleton on CPU.
    // The layer-major route is the turbo/layer-sweep family and it makes image
    // turns look like the CPU is doing the same layer-swapping behavior we are
    // trying to keep out of normal CPU mode.
    //
    // CPU embedding prefill therefore uses the normal token-major transformer
    // route: one embedded position goes through all layers, then the next
    // embedded position. Only the final embedded position computes logits.
    //
    // Text-only CPU prompt batch remains separate in prefill_cpu_prompt_batch().
    static std::atomic<int> cpu_emb_prefill_logs{0};
    const int log_id = cpu_emb_prefill_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id < 64) {
        fprintf(stderr,
            "[cpu-prefill:emb] token-major CPU image/mixed prefill ntok=%d pos=%d..%d no_layer_major no_gpu_stream no_resident_window no_evict\n",
            n, start_pos, start_pos + n - 1);
        fflush(stderr);
    }

    for (int i = 0; i < n; ++i) {
        const float* e = embeddings + (size_t)i * (size_t)hidden;
        if (i + 1 == n)
            forward_embed(e, start_pos + i);          // updates logits for generation
        else
            forward_embed_prefill(e, start_pos + i);  // fills KV only
    }
}

inline void Transformer::prefill_cpu_prompt_batch(const std::vector<int32_t>& tokens,
                                                  int start_pos, int n, bool keep_prefill_scratch)
{
    // Report KV vs process memory and return freed batch scratch to the OS
    // when this prefill finishes (on every exit path) -- UNLESS the caller says more
    // calls on this same prompt are coming right after (keep_prefill_scratch=true),
    // e.g. hybrid prefill streaming one layer at a time: tearing these down and
    // regrowing them on every single-layer call is what made that take minutes per
    // layer instead of seconds. See the v21 patch note at the top of this file.
    struct KvTrimGuard {
        Transformer* t; const char* tag; bool on; int n_tok;
        ~KvTrimGuard() {
            if (!on) return;
            if (n_tok >= 256) {
                // Prompt-sized GPU staging buffers aren't needed by decode: give the RAM back.
                t->release_prefill_scratch();
                const size_t freed = gpu::gpu().release_prefill_buffers();
                if (freed) fprintf(stderr, "[kv] released %.0f MB of prompt-sized GPU staging buffers\n",
                                   freed / 1048576.0);
            }
            t->kv_report_and_trim(tag);
        }
    } kv_trim_guard_{this, "after cpu prefill", (n > 1) && !keep_prefill_scratch, n};
    if (!loaded_ || n <= 0) return;

    // Single-token degenerates to the normal decode step.
    if (n == 1 && !pr_active_) { forward(tokens, start_pos); return; }

    // Gemma 4 PLE requires per-token PLE state. Keep the proven route here.
    if (ple_enabled_) {
        for (int i = 0; i < n; ++i) forward(tokens, start_pos + i);
        return;
    }

    const int hidden = (int)meta_.hidden_dim;
    const int vocab  = (int)meta_.vocab_size;

    // Full prompt activation batch. CPU owns all weights already; there is no
    // GPU stream flag flip, no resident window, no prefetch/evict loop here.
    // In a distributed range prefill the caller's buffer IS the activation buffer (no second copy of n*hidden floats).
    std::vector<float> x_store;
    float* const x_ext = (pr_active_ && pr_x_out_) ? pr_x_out_ : nullptr;
    if (!x_ext) x_store.assign((size_t)n * (size_t)hidden, 0.0f);
    struct XView { float* p; float* data() const { return p; } } x_buf{ x_ext ? x_ext : x_store.data() };

    if (pr_active_ && pr_x_in_ && pr_x_in_ != x_buf.data()) std::memcpy(x_buf.data(), pr_x_in_, (size_t)n * (size_t)hidden * sizeof(float));
    for (int t = 0; t < n && !(pr_active_ && pr_x_in_); ++t) {
        const int32_t token = tokens[start_pos + t];
        float* xt = x_buf.data() + (size_t)t * (size_t)hidden;
        quant::dequantize_row(token_embedding_table, xt,
                              (size_t)token, (size_t)hidden, emb_type);
        if (embed_scale_ > 0.0f) {
            for (int i = 0; i < hidden; ++i) xt[i] *= embed_scale_;
        }
    }

    static std::atomic<int> cpu_prefill_logs{0};
    const int log_id = cpu_prefill_logs.fetch_add(1, std::memory_order_relaxed);
    if (log_id < 32) {
        int on_gpu = 0, total = (int)meta_.n_layers;
        for (int l2 = 0; l2 < total; ++l2) if (prefill_on_gpu_for_layer(l2)) ++on_gpu;
        fprintf(stderr,
            "[prefill] reading %d prompt tokens (pos %d..%d): %d layers on GPU, %d layers on CPU "
            "(see [prefill:mm] lines for the exact device of each step)\n",
            n, start_pos, start_pos + n - 1, on_gpu, total - on_gpu);
        fflush(stderr);
    }

    bool qwen_native_batch_disabled = false;

    for (int l = (pr_active_ ? pr_start_ : 0); l < (pr_active_ ? pr_end_ : (int)meta_.n_layers); ++l) {
        if (pr_active_) pr_layers_done_.store(l - pr_start_);
        LayerPinGuard pin_guard(*mm_, l);
        const TransformerLayerWeights& lw = layer_weights_[l];

        bool handled_by_cpu_batch = false;

        if (!qwen_native_batch_disabled && n > 1 && lw.qwen35_linear && !lw.moe &&
            layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_qwen35_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_cpu_batch = true;
        }
        else if (!qwen_native_batch_disabled && n > 1 && !lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_full_attn_prefill_layer_batch_native_legacy(lw, xp, l, sp, cnt) || forward_full_attn_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_cpu_batch = true;
        }

        if (!handled_by_cpu_batch) {
            // Correct fallback for exotic layers. This is inside the one prompt
            // prefill call, and still skips lm_head until the final prompt token.
            for (int t = 0; t < n; ++t) {
                forward_layer(l,
                              x_buf.data() + (size_t)t * (size_t)hidden,
                              start_pos + t);
            }
        }
    }

    if (pr_active_) { if (pr_x_out_ && pr_x_out_ != x_buf.data()) std::memcpy(pr_x_out_, x_buf.data(), (size_t)n * (size_t)hidden * sizeof(float)); return; }

    // Final norm + lm_head only for the last prompt position. Output generation
    // continues with normal forward()/sample() token-by-token after this.
    float* x_last = x_buf.data() + (size_t)(n - 1) * (size_t)hidden;

    const float* head_in = x_last;
    if (rms_final_weight) {
        float* xb = xb_scratch_.data();
        nx_apply_model_norm(x_last, rms_final_weight, rms_final_bias, xb,
                            (size_t)hidden, meta_.rms_norm_eps,
                            zero_centered_norm_);
        head_in = xb;
    }

    auto is_native_nq = [](QuantType t) {
        return t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64;
    };

    bool lm_done = false;
    if (gpu_lm_head_ && nx_lm_head_gpu_ok(lm_head_type)) {
        lm_done = gpu::gpu().mat_vec(lm_head_weight, head_in, logits_.data(),
                                     (size_t)vocab, (size_t)hidden, lm_head_type);
    }
    if (!lm_done) {
        cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                     (size_t)vocab, (size_t)hidden, lm_head_type, n_threads_);
    }
    if (lm_head_bias) {
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] += lm_head_bias[i];
    }
    if (final_logit_softcap_ > 0.0f) {
        const float cap = final_logit_softcap_;
        const float inv = 1.0f / cap;
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] = fast_tanh(logits_[i] * inv) * cap;
    }
}

inline void Transformer::prefill_batch(const std::vector<int32_t>& tokens,
                                        int start_pos, int n,
                                        bool force_no_stream, bool keep_prefill_scratch)
{
    // Report KV vs process memory and return freed batch scratch to the OS
    // when this prefill finishes (on every exit path) -- UNLESS the caller says more
    // calls on this same prompt are coming right after (keep_prefill_scratch=true).
    // See the v21 patch note at the top of this file: calling this once per layer
    // (hybrid prefill's single-layer streaming) used to pay this teardown-and-regrow
    // cost on EVERY layer instead of once for the whole range, which is almost
    // certainly why a 32-layer donor range was taking minutes per layer.
    struct KvTrimGuard {
        Transformer* t; const char* tag; bool on; int n_tok;
        ~KvTrimGuard() {
            if (!on) return;
            if (n_tok >= 256) {
                // Prompt-sized GPU staging buffers aren't needed by decode: give the RAM back.
                t->release_prefill_scratch();
                const size_t freed = gpu::gpu().release_prefill_buffers();
                if (freed) fprintf(stderr, "[kv] released %.0f MB of prompt-sized GPU staging buffers\n",
                                   freed / 1048576.0);
            }
            t->kv_report_and_trim(tag);
        }
    } kv_trim_guard_{this, "after prefill", (n > 1) && !keep_prefill_scratch, n};
    if (!loaded_ || n <= 0) return;

    // Single-token degenerates to forward() — no point inverting one iteration.
    if (n == 1 && !pr_active_) { forward(tokens, start_pos); return; }

    // Fallback path: Gemma 4 PLE recomputes per-token state that each layer
    // reads. The clean way to support it is a per-position ple_buffer; not
    // worth the complication for v1. Per-token loop is still correct.
    if (ple_enabled_) {
        for (int i = 0; i < n; ++i) forward(tokens, start_pos + i);
        return;
    }

    const int hidden = (int)meta_.hidden_dim;
    const int vocab  = (int)meta_.vocab_size;

    // Per-position residual stream: hidden floats per token, contiguous.
    // ~32 MB for hidden=4096, n=2048. Negligible vs the model itself.
    // In a distributed range prefill the caller's buffer IS the activation buffer (no second copy of n*hidden floats).
    std::vector<float> x_store;
    float* const x_ext = (pr_active_ && pr_x_out_) ? pr_x_out_ : nullptr;
    if (!x_ext) x_store.assign((size_t)n * (size_t)hidden, 0.0f);
    struct XView { float* p; float* data() const { return p; } } x_buf{ x_ext ? x_ext : x_store.data() };

    // ── 1. Token embedding for all N positions ────────────────────────────
    if (pr_active_ && pr_x_in_ && pr_x_in_ != x_buf.data()) std::memcpy(x_buf.data(), pr_x_in_, (size_t)n * (size_t)hidden * sizeof(float));
    for (int t = 0; t < n && !(pr_active_ && pr_x_in_); ++t) {
        const int32_t token = tokens[start_pos + t];
        float* xt = x_buf.data() + (size_t)t * (size_t)hidden;
        quant::dequantize_row(token_embedding_table, xt,
                              (size_t)token, (size_t)hidden, emb_type);
        if (embed_scale_ > 0.0f) {
            for (int i = 0; i < hidden; ++i) xt[i] *= embed_scale_;
        }
    }

    // ── 2. Inverted layer loop ─────────────────────────────────────────────
    // For each layer, sweep all tokens in order. KV cache writes for layer L
    // at positions start_pos..start_pos+n-1 complete before layer L+1 starts,
    // so attention at layer L+1 sees the full populated KV at layer L. Any
    // recurrent state (Qwen35) evolves in token order within each layer,
    // matching the per-token semantics.
    // Optionally enable GPU streaming for this prefill. Temporarily flip
    // every layer's gpu_layer flag to true; the bounded weight cache in
    // the backend will stream weights in (and evict LRU) as needed.
    // Snapshot current flags so decode runs at the user's chosen split.
    std::vector<uint8_t> saved_gpu_flag;
    // Parking: before the stream, room is cleared on the card only if the prompt speed-up lacks it; free RAM is watched
    // while the prompt is read. (park_min_tokens, 0 by default, can make short prompts skip the stream instead.)
    const bool park_short = park_on_ && !pr_active_ && n < park_min_tokens_ && park_any_on_card();
    const bool stream = turbo_gpu_stream_ && gpu::gpu().ready() && !force_no_stream && !park_short;
    const bool park_long = park_on_ && stream && !pr_active_;
    if (park_long) park_off_gpu(n, start_pos + n);
    if (stream) {
        if (!pr_active_) turbo_keep_prepare();   // layers that live on this card stay there; the rest stream through it
        saved_gpu_flag.resize(meta_.n_layers);
        turbo_range_end_ = pr_active_ ? pr_end_ : -1;
        for (int l = (pr_active_ ? pr_start_ : 0); l < (pr_active_ ? pr_end_ : (int)meta_.n_layers); ++l) {
            saved_gpu_flag[l] = layer_weights_[l].gpu_layer ? 1 : 0;
            layer_weights_[l].gpu_layer = true;
        }

        // Auto-compute the VRAM budget from the actual resident-window weight
        // footprint + a KV/attention reserve, instead of trusting the launcher
        // guess. Reserve = streaming KV chunks (K+V) + Q/out/state scratch.
        // NEXUS_AUTO_BUDGET=0 keeps the launcher-set budget (for A/B diagnosis).
        const char* ab = std::getenv("NEXUS_AUTO_BUDGET");
        const bool auto_budget = !(ab && ab[0]=='0');
        if (auto_budget) {
            const size_t kv_attn_reserve = (size_t)320 * 1024 * 1024;
            compute_and_set_turbo_vram_budget(kv_attn_reserve, start_pos + n);
        }

        fprintf(stderr,
            "[prefill_batch] GPU streaming ON (%d layers, cache budget %zu MB, resident window %d layers)\n",
            (int)meta_.n_layers,
            gpu::gpu().weight_cache_budget() / (1024*1024),
            turbo_layer_window_size());
        fprintf(stderr,
            "[turbo:cap] arch=%s ntok=%d std_mm: Q8_0=%s Q4_K=%s Q5_K=%s Q6_K=%s native_mm: NQ4=%s NQ5=%s NQ6=%s NQ8=%s attn_batch=%s attn_stream=%s q35_conv=%s q35_scan=%s\n",
            meta_.arch.c_str(), n,
            gpu::gpu().mat_mat_supported(QuantType::Q8_0)    ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::Q4_K)    ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::Q5_K)    ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::Q6_K)    ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::NQ4_G64) ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::NQ5_G64) ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::NQ6_G64) ? "GPU" : "--",
            gpu::gpu().mat_mat_supported(QuantType::NQ8_G64) ? "GPU" : "--",
            "GPU", "GPU", "GPU", "GPU");
        fflush(stderr);

        // Safety fence: native Qwen3Next was correct before the universal GGUF
        // standard-Q4 mat_mat work touched the shared batched turbo path. Keep
        // the new universal Q4/Q5/Q6/Q8 path for downloaded GGUF models, but do
        // not let Qwen3Next native .nxm use the currently-suspect batched layer
        // path unless explicitly requested.
        // Override for controlled testing only:
        //   NEXUS_QWEN_NATIVE_BATCH=1

        // Sequential streaming: no rolling window, no priming, no prefetch.
        // Each layer is loaded, verified, computed and released in strict order
        // inside the layer loop below.
    }

    // Qwen native text prefill must stay on the fast native streaming path.
    // The earlier blunt guard forced fallback-per-token and destroyed the real
    // prefill win. Correctness is protected by routing native full-attention
    // layers through the pre-universal native legacy implementation below, while
    // standard GGUF Q4/Q5/Q6/Q8 keeps the new universal path.
    bool qwen_native_batch_disabled = false;

    if (park_long) park_relieve_ram(-1, 0, (int)meta_.n_layers);

    for (int l = (pr_active_ ? pr_start_ : 0); l < (pr_active_ ? pr_end_ : (int)meta_.n_layers); ++l) {
        if (pr_active_) pr_layers_done_.store(l - pr_start_);
        LayerPinGuard pin_guard(*mm_, l);
        const TransformerLayerWeights& lw = layer_weights_[l];

        const auto turbo_layer_t0 = std::chrono::steady_clock::now();
        if (stream) {
            const char* kind = lw.qwen35_linear ? "q35" : (lw.moe ? "moe" : "dense");
            fprintf(stderr, "[turbo:layer] begin L=%d/%u kind=%s ntok=%d pos=%d..%d\n",
                    l, meta_.n_layers, kind, n, start_pos, start_pos + n - 1);
            fflush(stderr);
        }

        bool handled_by_batched_turbo = false;

        // Start the next resident-window upload before this layer launches its
        // heavy full-prompt GPU GEMMs. With the transfer-queue backend, layer
        // L+window can copy into device-local VRAM while compute_queue_ is busy
        // on layer L. This is the first real overlap point; the older call inside
        // forward_qwen35_prefill_layer_batch only overlapped with CPU conv/scan.
        // Close the prefetch race WITHOUT disturbing the pin/evict accounting.
        // Simply wait for any in-flight prefetch to finish before computing l, so
        // the main thread never touches a weight that is still mid-upload. We do
        // NOT re-load or re-pin l here — the rolling window already loaded+pinned
        // it; an extra pin/load was unbalancing the accounting and corrupting the
        // final layers (output collapsed to an immediate EOS token). The wait is
        // the whole fix; the prefetch for l+1 starts right after to keep overlap.
        if (stream) {
            turbo_stream_begin_layer(l, "L");
        }

        // Qwen3Next/Gated-DeltaNet turbo path. The old loop below still works,
        // but it calls the single-token CPU linear-attention projection route.
        // This batched path keeps the recurrent/conv math in correct token order
        // while pushing the large projection GEMMs through Vulkan mat_mat when
        // possible. Non-Qwen/full-attention layers fall through unchanged.
        if (!qwen_native_batch_disabled && stream && lw.qwen35_linear && !lw.moe &&
            layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_qwen35_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        else if (!qwen_native_batch_disabled && stream && !lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_full_attn_prefill_layer_batch_native_legacy(lw, xp, l, sp, cnt) || forward_full_attn_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        // CPU batched path for GatedDeltaNet (qwen35_linear) layers: their
        // projections run through turbo_mat_mat_or_cpu, which falls back to
        // cpu::mat_mat — ONE weight pass for all n tokens instead of n. This is
        // the real CPU MTP speedup: the 48 linear layers are most of the compute.
        // (Full-attn layers' batched path is GPU-only, so on CPU they use the
        // per-token loop below — only 16 of 64 layers.)
        else if (!qwen_native_batch_disabled && !stream && n > 1 && lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_qwen35_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }
        // CPU batched path for full-attention layers (the other 16). Their batched
        // function now falls back to cpu::mat_mat for every projection, so it runs
        // on CPU too — all 64 layers batch the n verify tokens in one weight pass.
        else if (!qwen_native_batch_disabled && !stream && n > 1 && !lw.qwen35_linear && !lw.moe &&
                 layer_batch_tiled(x_buf.data(), n, start_pos, hidden, l, [&](float* xp, int sp, int cnt) { return forward_full_attn_prefill_layer_batch_native_legacy(lw, xp, l, sp, cnt) || forward_full_attn_prefill_layer_batch(lw, xp, l, sp, cnt); }))
        {
            handled_by_batched_turbo = true;
        }

        if (!handled_by_batched_turbo) {
            for (int t = 0; t < n; ++t) {
                forward_layer(l,
                              x_buf.data() + (size_t)t * (size_t)hidden,
                              start_pos + t);
            }
        }

        // The layer is complete and never needed again for this prefill pass.
        // Unpin + evict it so the rolling resident window can reuse that budget.
        if (stream) {
            const auto turbo_layer_t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(turbo_layer_t1 - turbo_layer_t0).count();
            fprintf(stderr, "[turbo:layer] done  L=%d path=%s ms=%.1f\n",
                    l, handled_by_batched_turbo ? "batched" : "fallback-per-token", ms);
            fflush(stderr);
            turbo_stream_end_layer(l);
        }
        if (park_long) park_relieve_ram(l, 0, (int)meta_.n_layers);
    }

    // Restore gpu_layer flags and free streamed VRAM so decode starts clean.
    if (stream) {
        turbo_prefetch_wait(turbo_prefetch_tail_future_);
        turbo_prefetch_wait(turbo_prefetch_next_future_);
        for (int l = (pr_active_ ? pr_start_ : 0); l < (pr_active_ ? pr_end_ : (int)meta_.n_layers); ++l) {
            layer_weights_[l].gpu_layer = saved_gpu_flag[l] != 0;
        }
        turbo_range_end_ = -1;
        gpu::gpu().evict_unpinned_weights();
    }
    if (park_long) park_pending_ = true;     // the engine puts parked layers back on the card once this prompt's buffers are freed

    if (pr_active_) { if (pr_x_out_ && pr_x_out_ != x_buf.data()) std::memcpy(pr_x_out_, x_buf.data(), (size_t)n * (size_t)hidden * sizeof(float)); return; }

    // ── 3. Final norm + lm_head — ONLY for the last position ──────────────
    // sample() reads logits_, and prefill discards intermediate logits, so we
    // skip lm_head for positions 0..n-2 entirely. This is a free win on top
    // of the layer-major reorganization: saves (n-1) lm_head matmuls.
    float* x_last = x_buf.data() + (size_t)(n - 1) * (size_t)hidden;
    // MTP's draft head starts from the main model's last hidden state. Before, a prompt or an MTP check read in one batch
    // never handed it over, so the next drafts started from an older token's state.
    if (x_scratch_.size() >= (size_t)hidden) std::memcpy(x_scratch_.data(), x_last, (size_t)hidden * sizeof(float));
    if (mtp_verify_all_logits_) mtp_verify_hidden_.assign(x_buf.data(), x_buf.data() + (size_t)n * (size_t)hidden);

    const float* head_in = x_last;
    if (rms_final_weight) {
        float* xb = xb_scratch_.data();
        nx_apply_model_norm(x_last, rms_final_weight, rms_final_bias, xb,
                            (size_t)hidden, meta_.rms_norm_eps,
                            zero_centered_norm_);
        head_in = xb;
    }

    // lm_head: mirror forward()'s native-NQ + GPU + CPU routing exactly so
    // numerics match bit-for-bit.
    auto is_native_nq = [](QuantType t) {
        return t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
               t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64;
    };

    // An MTP check gets every position's logits from ONE batched output pass below; the last position's single pass here
    // would read the whole output matrix (the biggest weight in the model) a second time for nothing.
    if (!mtp_verify_all_logits_) {
    bool lm_done = false;
    if (gpu_lm_head_ && nx_lm_head_gpu_ok(lm_head_type)) {
        lm_done = gpu::gpu().mat_vec(lm_head_weight, head_in, logits_.data(),
                                     (size_t)vocab, (size_t)hidden, lm_head_type);
    }
    if (!lm_done) {
        cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                     (size_t)vocab, (size_t)hidden, lm_head_type, n_threads_);
    }
    if (lm_head_bias) {
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] += lm_head_bias[i];
    }
    if (final_logit_softcap_ > 0.0f) {
        const float cap = final_logit_softcap_;
        const float inv = 1.0f / cap;
        for (uint32_t i = 0; i < meta_.vocab_size; ++i)
            logits_[i] = fast_tanh(logits_[i] * inv) * cap;
    }
    }

    // MTP verify: also produce logits at position n-2 (the confirmed token's
    // distribution, used to check the draft). Only when forward_verify2 set the
    // capture pointer, and only meaningful for n>=2. Reuses the exact lm_head
    // routing so numerics match the last-position path.
    if (mtp_verify_second_logits_ && n >= 2) {
        std::vector<float>& dst = *mtp_verify_second_logits_;
        dst.assign(vocab, 0.0f);
        float* x_prev = x_buf.data() + (size_t)(n - 2) * (size_t)hidden;
        const float* head_in2 = x_prev;
        if (rms_final_weight) {
            float* xb2 = xb_scratch_.data();
            nx_apply_model_norm(x_prev, rms_final_weight, rms_final_bias, xb2,
                                (size_t)hidden, meta_.rms_norm_eps, zero_centered_norm_);
            head_in2 = xb2;
        }
        auto is_native_nq2 = [](QuantType t){
            return t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
                   t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64; };
        bool done2 = false;
        if (gpu_lm_head_ && nx_lm_head_gpu_ok(lm_head_type)) {
            done2 = gpu::gpu().mat_vec(lm_head_weight, head_in2, dst.data(),
                                       (size_t)vocab, (size_t)hidden, lm_head_type);
        }
        if (!done2) {
            cpu::mat_vec(lm_head_weight, head_in2, dst.data(),
                         (size_t)vocab, (size_t)hidden, lm_head_type, n_threads_);
        }
        if (lm_head_bias)
            for (uint32_t i = 0; i < meta_.vocab_size; ++i) dst[i] += lm_head_bias[i];
        if (final_logit_softcap_ > 0.0f) {
            const float cap = final_logit_softcap_, inv = 1.0f / cap;
            for (uint32_t i = 0; i < meta_.vocab_size; ++i)
                dst[i] = fast_tanh(dst[i] * inv) * cap;
        }
    }

    // MTP multi-token verify: capture logits at EVERY position [0..n-1] so the
    // caller can accept the longest matching draft prefix from one batched sweep.
    // Reuses the exact lm_head routing per position so numerics match.
    if (mtp_verify_all_logits_) {
        std::vector<std::vector<float>>& all = *mtp_verify_all_logits_;
        all.assign((size_t)n, std::vector<float>());

        // Batch the lm_head across ALL positions in ONE matmul. lm_head is the
        // single biggest weight in the model (vocab*hidden ~1.27B), so reading it
        // once for all n verify positions instead of n times is the dominant
        // speed win for MTP. Norm each position into a contiguous [n*hidden]
        // buffer, run one cpu::mat_mat (or GPU), then split per position.
        std::vector<float> hin_all((size_t)n * (size_t)hidden);
        for (int p = 0; p < n; ++p) {
            float* xp = x_buf.data() + (size_t)p * (size_t)hidden;
            float* dstn = hin_all.data() + (size_t)p * (size_t)hidden;
            if (rms_final_weight) {
                nx_apply_model_norm(xp, rms_final_weight, rms_final_bias, dstn,
                                    (size_t)hidden, meta_.rms_norm_eps, zero_centered_norm_);
            } else {
                std::memcpy(dstn, xp, (size_t)hidden * sizeof(float));
            }
        }

        // One batched projection: Y is [n][vocab] (cpu::mat_mat lays out
        // Y[t*out_rows + row], i.e. token-major, row=vocab index).
        std::vector<float> Yall((size_t)n * (size_t)vocab);
        bool doneA = false;
#ifdef NEXUS_VULKAN
        auto is_native_nqA = [](QuantType t){
            return t == QuantType::NQ4_G64 || t == QuantType::NQ5_G64 ||
                   t == QuantType::NQ6_G64 || t == QuantType::NQ8_G64; };
        if (mtp_use_gpu_ && gpu::gpu().ready() && is_native_nqA(lm_head_type)) {
            doneA = gpu::gpu().mat_mat(lm_head_weight, hin_all.data(), Yall.data(),
                                       (size_t)vocab, (size_t)hidden, (size_t)n, lm_head_type);
        }
#endif
        if (!doneA) {
            cpu::mat_mat(lm_head_weight, hin_all.data(), Yall.data(),
                         (size_t)vocab, (size_t)hidden, (size_t)n, lm_head_type, n_threads_);
        }
        // Split into per-position logit vectors (+ bias + softcap).
        for (int p = 0; p < n; ++p) {
            std::vector<float>& dst = all[(size_t)p];
            dst.assign(vocab, 0.0f);
            const float* ysrc = Yall.data() + (size_t)p * (size_t)vocab;
            std::memcpy(dst.data(), ysrc, (size_t)vocab * sizeof(float));
            if (lm_head_bias)
                for (uint32_t i = 0; i < meta_.vocab_size; ++i) dst[i] += lm_head_bias[i];
            if (final_logit_softcap_ > 0.0f) {
                const float cap = final_logit_softcap_, inv = 1.0f / cap;
                for (uint32_t i = 0; i < meta_.vocab_size; ++i)
                    dst[i] = fast_tanh(dst[i] * inv) * cap;
            }
        }
        if (!all.empty() && all.back().size() >= (size_t)vocab && logits_.size() >= (size_t)vocab)
            std::memcpy(logits_.data(), all.back().data(), (size_t)vocab * sizeof(float));   // what the skipped single pass gave
    }
}


inline bool Transformer::prefill_layer_range(const std::vector<int32_t>& tokens, int start_pos, int n,
                                             int layer_start, int layer_end,
                                             const float* x_in, float* x_out, bool use_turbo,
                                             bool keep_prefill_scratch)
{
    if (!loaded_ || n <= 0 || layer_end <= layer_start || layer_end > (int)meta_.n_layers) return false;
    if (ple_enabled_) return false;
    if (layer_start > 0 && !x_in) return false;
    if (layer_start == 0 && (start_pos < 0 || (size_t)(start_pos + n) > tokens.size())) return false;
    if (active_layer_end_ >= 0 && (layer_start < active_layer_start_ || layer_end > active_layer_end_)) {
        fprintf(stderr, "[nexus] prefill_layer_range: asked for layers %d-%d but this node owns %d-%d\n",
                layer_start, layer_end - 1, active_layer_start_, active_layer_end_ - 1);
        return false;
    }
    struct Guard { Transformer* t; ~Guard() { t->pr_active_ = false; t->pr_x_in_ = nullptr; t->pr_x_out_ = nullptr; t->turbo_range_end_ = -1; } } g{this};
    pr_active_ = true; pr_start_ = layer_start; pr_end_ = layer_end; pr_x_in_ = x_in; pr_x_out_ = x_out;
    if (use_turbo) prefill_batch(tokens, start_pos, n, false, keep_prefill_scratch);
    else           prefill_cpu_prompt_batch(tokens, start_pos, n, keep_prefill_scratch);
    return true;
}

inline void Transformer::reset_context() {
    kvcache_.clear();
    rope_pos_scale_ = 1.0f;
    mtp_k_hist_.clear(); mtp_v_hist_.clear(); mtp_hist_len_ = 0;   // a new conversation must not see the old one's MTP history

    q35_kv_drop_all();
    q35_drop_dev(-1);
    for (auto& v : q35_conv_state_) {
        std::fill(v.begin(), v.end(), 0.0f);
    }
    for (auto& v : q35_recurrent_state_) {
        std::fill(v.begin(), v.end(), 0.0f);
    }
}

inline void Transformer::kv_report_and_trim(const char* tag) {
    // RssAnon = ordinary heap/mmap memory (what an out-of-memory kill counts);
    // RssFile = the memory-mapped model file, which the kernel can drop under pressure.
    auto status_mb = [](const char* key) -> long {
#if defined(__linux__)
        if (FILE* f = fopen("/proc/self/status", "r")) {
            char line[256]; long kb = -1;
            const size_t kl = strlen(key);
            while (fgets(line, sizeof line, f)) {
                if (strncmp(line, key, kl) == 0) { kb = atol(line + kl); break; }
            }
            fclose(f);
            return kb < 0 ? -1 : kb / 1024;
        }
#endif
        (void)key;
        return -1;
    };
    const long anon_before = status_mb("RssAnon:");
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
    fprintf(stderr,
        "[kv] %s tokens=%d | KV: %d/%d layers hold KV, live=%.0f MB, alloc=%.0f MB | process: heap(anon)=%ld MB (was %ld before trim), model-file=%ld MB, swap=%ld MB\n",
        tag ? tag : "", kvcache_.n_tokens(), kvcache_.layers_with_kv(), kvcache_.total_layers(),
        kvcache_.live_bytes() / 1048576.0, kvcache_.memory_bytes() / 1048576.0,
        status_mb("RssAnon:"), anon_before, status_mb("RssFile:"), status_mb("VmSwap:"));
    fflush(stderr);
}

inline void Transformer::print_memory_report() const {
    if (mm_) mm_->print_report();
    fprintf(stderr, "  KV Cache: %zu bytes (%zu tokens)\n",
            kvcache_.memory_bytes(), (size_t)kvcache_.n_tokens());
}

inline void Transformer::forward_layer_range(const std::vector<int32_t>& tokens,
                                              int pos, int layer_start, int layer_end) {
    const int hidden  = (int)meta_.hidden_dim;
    const int n_layrs = (int)meta_.n_layers;
    // Distributed shards preserve full-model source layer IDs.
    // If older dist code asks a worker shard to run local range 0..owned_count,
    // remap it to the actual source-layer range, e.g. 0..12 -> 28..40.
    if (active_layer_end_ >= 0) {
        const int own_start = active_layer_start_;
        const int own_end   = active_layer_end_;
        const int own_count = own_end - own_start;

        const int req_start = layer_start;
        const int req_end   = layer_end;

        if (own_start != 0 &&
            req_start == 0 &&
            req_end > 0 &&
            req_end <= own_count &&
            tokens.empty())
        {
            layer_start = own_start;
            layer_end   = own_start + req_end;

            fprintf(stderr,
                "[nexus] forward_layer_range: remapped worker-local range %d-%d -> source range %d-%d\n",
                req_start,
                req_end - 1,
                layer_start,
                layer_end - 1);
            fflush(stderr);
        }

        if (layer_start < own_start || layer_end > own_end) {
            fprintf(stderr,
                "[nexus] FATAL-RUNTIME: shard asked to run layers %d-%d, "
                "but this shard owns source layers %d-%d. Refusing invalid range.\n",
                layer_start,
                layer_end - 1,
                own_start,
                own_end - 1);
            fflush(stderr);
            return;
        }
    }
    // Primary-side entry: tokens are provided, so layer 0 embeds the current token.
    // Worker-side entry: tokens is intentionally empty because x_scratch_ already
    // contains the received activation from the primary. Do NOT overwrite it.
    if (layer_start == 0 && !tokens.empty()) {
        float* x = x_scratch_.data();
        int idx = pos;
        if (idx < 0) idx = 0;
        if (idx >= (int)tokens.size()) idx = (int)tokens.size() - 1;
        const int tok = tokens[idx];
        quant::dequantize_row(token_embedding_table, x,
                              (size_t)tok, (size_t)hidden, emb_type);
    }
    for (int l = layer_start; l < layer_end && l < n_layrs; ++l) {
        forward_layer(l, x_scratch_.data(), pos);
    }
    if (layer_end >= n_layrs) {
        float* x  = x_scratch_.data();
        const float* head_in = x;
        if (rms_final_weight) {
            float* xb = xb_scratch_.data();
            nx_apply_model_norm(x, rms_final_weight, rms_final_bias, xb,
                    (size_t)hidden, meta_.rms_norm_eps,
                    zero_centered_norm_);
            head_in = xb;
        }
        int head_rows = (int)meta_.vocab_size;
        if (!lm_head_weight) {                       // never read through a missing pointer: say what is wrong instead of crashing
            static bool warned = false;
            if (!warned) { warned = true; fprintf(stderr, "[nexus] ERROR: this computer holds the last layers but not the output (lm_head) weights - nothing to compute the next word with. Use a model file/piece that contains them.\n"); fflush(stderr); }
            std::fill(logits_.begin(), logits_.end(), 0.0f);
            return;
        }
        if (lm_head_rows_ > 0 && (uint64_t)head_rows > lm_head_rows_) {   // tokenizer lists more words than the output tensor has rows: never read past the tensor
            static bool warned2 = false;
            if (!warned2) { warned2 = true; fprintf(stderr, "[nexus] WARNING: vocabulary %d is larger than the output tensor (%llu rows); extra entries are ignored\n", head_rows, (unsigned long long)lm_head_rows_); fflush(stderr); }
            head_rows = (int)lm_head_rows_;
            for (size_t i = (size_t)head_rows; i < logits_.size(); ++i) logits_[i] = -1e30f;
        }
        cpu::mat_vec(lm_head_weight, head_in, logits_.data(),
                     head_rows, hidden, lm_head_type, n_threads_);
        if (lm_head_bias) {
            for (int i = 0; i < head_rows; ++i) {
                logits_[i] += lm_head_bias[i];
            }
        }
    }
}

inline const float* Transformer::hidden_data() const { return x_scratch_.data(); }
inline size_t Transformer::hidden_dim_bytes() const { return x_scratch_.size() * sizeof(float); }
inline void Transformer::set_hidden_data(const float* s, size_t n) {
    if (n > x_scratch_.size()) n = x_scratch_.size();
    std::copy(s, s + n, x_scratch_.data());
}

// load_from_memory: identical to load() but calls open_from_memory() instead of open().
// Zero disk I/O. Cross-platform. The buffer is borrowed -- caller must keep it alive.
inline bool Transformer::load_from_memory(const void* data, size_t size, int n_threads,
                                           std::function<void(const std::string&, float)> cb)
{
    n_threads_ = n_threads;
    active_layer_start_ = 0;
    active_layer_end_   = -1;
    if (cb) cb("Opening model file", 0.0f);
    MemoryManager::Config mm_cfg;
    mm_cfg.prefer_huge_pages  = true;
    mm_cfg.lock_active_layers = true;
    mm_cfg.max_locked_layers  = std::min(n_threads * 2, 8);
    mm_ = std::make_unique<MemoryManager>(mm_cfg);
    mm_->on_pressure([this](size_t avail) {
        fprintf(stderr, "[nexus] Memory pressure! Evicting KV cache pages.\n");
        kvcache_.evict_oldest(128);
    });
    parser_ = std::make_unique<gguf::Parser>();
    fprintf(stderr, "[load] Opening in-memory GGUF (%zu MB)...\n", size/(1024*1024));
    fflush(stderr);
    if (!parser_->open_from_memory(data, size)) {
        fprintf(stderr, "[nexus] Failed to parse in-memory GGUF\n");
        return false;
    }
    meta_ = parser_->meta();
    if (cb) cb("Reading model metadata", 0.05f);
    parser_->print_meta();
    fprintf(stderr,"[load] Loading tokenizer...\n"); fflush(stderr);
    if (!load_tokenizer())
        fprintf(stderr, "[nexus] Warning: tokenizer load failed\n");

    if (!vocab_tokens_.empty() && vocab_tokens_.size() != meta_.vocab_size) {
        fprintf(stderr,
                "[nexus] Vocab metadata corrected from %u to %zu based on tokenizer tokens\n",
                meta_.vocab_size, vocab_tokens_.size());
        meta_.vocab_size = (uint32_t)vocab_tokens_.size();
    }
    if (cb) cb("Loading weights into memory", 0.1f);
    size_t loaded_cnt = 0;
    bool ok = parser_->load_weights(*mm_, [&](const gguf::LoadProgress& p) {
        ++loaded_cnt;
        float frac = 0.1f + 0.8f * ((float)p.bytes_loaded / std::max((float)p.bytes_total, 1.0f));
        if (cb) cb("Loading: " + p.current_tensor, frac);
        if (loaded_cnt % 50 == 0)
            fprintf(stderr, "  [%3zu%%] %s\n", (size_t)(frac*100), p.current_tensor.c_str());
    });
    if (!ok) { fprintf(stderr, "[nexus] Weight loading failed\n"); return false; }
    if (cb) cb("Building layer index", 0.92f);
    if (!build_layer_weights()) { fprintf(stderr, "[nexus] Failed to index\n"); return false; }
    if (meta_.head_dim == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[nexus] FATAL: head_dim=%u hidden_dim=%u\n",
                meta_.head_dim, meta_.hidden_dim);
        return false;
    }
    attn_cfg_.n_heads    = (int)meta_.n_heads;
    attn_cfg_.n_kv_heads = (int)meta_.n_kv_heads;
    attn_cfg_.head_dim   = (int)meta_.head_dim;
    attn_cfg_.hidden_dim = (int)meta_.hidden_dim;
    attn_cfg_.rms_norm_eps = meta_.rms_norm_eps;
    attn_cfg_.rope.freq_base    = meta_.rope_freq_base;
    attn_cfg_.rope.n_ctx_orig   = (meta_.context_length > 0) ? meta_.context_length : 4096;
    attn_cfg_.n_threads  = n_threads;
    if (meta_.rope_freq_base <= 10001.0f && meta_.context_length > 8192 && meta_.head_dim > 2) {
        float ctx_ratio = (float)meta_.context_length / (float)attn_cfg_.rope.n_ctx_orig;
        if (ctx_ratio > 1.0f) {
            float exponent = (float)meta_.head_dim / (float)(meta_.head_dim - 2);
            attn_cfg_.rope.freq_base *= powf(ctx_ratio, exponent);
        }
    }
    init_kv_cache_for_model();
    const int hidden = (int)meta_.hidden_dim;
    const int ffn    = (int)meta_.ffn_dim;
    const int q_dim  = (int)(meta_.n_heads * meta_.head_dim);
    const int kv_dim = (int)(meta_.n_kv_heads * meta_.head_dim);

    int scratch_q_dim  = q_dim;
    int scratch_kv_dim = kv_dim;

    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        if (arch_l.find("gemma4") != std::string::npos) {
            for (const TransformerLayerWeights& lw : layer_weights_) {
                if (lw.attn_q_dim_layer  > scratch_q_dim)  scratch_q_dim  = lw.attn_q_dim_layer;
                if (lw.attn_kv_dim_layer > scratch_kv_dim) scratch_kv_dim = lw.attn_kv_dim_layer;
            }

            fprintf(stderr,
                "[nexus] Gemma4 scratch dims: base q=%d kv=%d -> scratch q=%d kv=%d total=%d\n",
                q_dim,
                kv_dim,
                scratch_q_dim,
                scratch_kv_dim,
                scratch_q_dim + 2 * scratch_kv_dim);
            fflush(stderr);
        }
    }

    const int max_ctx = (int)meta_.context_length;
    const size_t attn_scratch_sz =
        (size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim +
        (size_t)max_ctx + (size_t)scratch_q_dim + 1024;

    x_scratch_.resize(hidden, 0.0f);
    xb_scratch_.resize(hidden, 0.0f);
    attn_scratch_.resize(attn_scratch_sz, 0.0f);
    ffn_scratch_.resize(std::max((size_t)hidden, (size_t)ffn * 2), 0.0f);
    qkv_scratch_.resize((size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim, 0.0f);
    logits_.resize(meta_.vocab_size, 0.0f);
    
    if (cb) cb("Ready", 1.0f);
    fprintf(stderr, "\n[nexus] Model loaded (from memory).\n"); fflush(stderr);
    mm_->print_report();
    loaded_ = true;
    return true;
}


// load_range_direct: load only selected source layers from the original GGUF.
// No mini-GGUF. No memfd. No temp file. No second full-shard staging copy.
inline bool Transformer::load_range_direct(const std::string& model_path,
                                           int layer_start, int layer_end,
                                           int n_threads,
                                           std::function<void(const std::string&, float)> cb)
{
    n_threads_ = n_threads;
    loaded_ = false;

    if (layer_start < 0 || layer_end <= layer_start) {
        fprintf(stderr, "[load_range_direct] invalid layer range %d-%d\n",
                layer_start, layer_end - 1);
        return false;
    }

    if (cb) cb("Opening source model", 0.0f);

    MemoryManager::Config mm_cfg;
    mm_cfg.prefer_huge_pages  = true;
    mm_cfg.lock_active_layers = true;
    mm_cfg.max_locked_layers  = std::min(n_threads * 2, 8);
    mm_ = std::make_unique<MemoryManager>(mm_cfg);
    mm_->on_pressure([this](size_t avail) {
        fprintf(stderr, "[nexus] Memory pressure! %s available. Evicting KV cache pages.\n",
                MemoryManager::fmt_bytes(avail).c_str());
        kvcache_.evict_oldest(128);
    });

    parser_ = std::make_unique<gguf::Parser>();
    fprintf(stderr, "[load_range_direct] Opening source model: %s\n", model_path.c_str());
    fflush(stderr);
    if (!parser_->open(model_path)) {
        fprintf(stderr, "[load_range_direct] Failed to open source model\n");
        return false;
    }

    gguf::ModelMeta full_meta = parser_->meta();
    const int full_layers = (int)full_meta.n_layers;
    if (full_layers <= 0 || layer_start >= full_layers || layer_end > full_layers) {
        fprintf(stderr, "[load_range_direct] bad range %d-%d for full_layers=%d\n",
                layer_start, layer_end - 1, full_layers);
        return false;
    }

    meta_ = full_meta;
    active_layer_start_ = layer_start;
    active_layer_end_   = layer_end;

    // Filter the parser's tensor table in-place. Tensor byte offsets still point
    // into the original GGUF mmap, so Parser::load_weights() can copy each selected
    // tensor once straight from the source mapping into MemoryManager-owned RAM.
    auto& tensors = const_cast<std::unordered_map<std::string, TensorDesc>&>(parser_->tensors());
    std::unordered_map<std::string, TensorDesc> filtered;
    filtered.reserve(tensors.size());

    size_t selected_layers = 0;
    size_t selected_global = 0;
    auto keep_global = [](const std::string& n) -> bool {
        return n == "token_embd.weight" || n == "model.embed_tokens.weight" ||
               n == "model.language_model.embed_tokens.weight" ||
               n == "output_norm.weight" || n == "model.norm.weight" ||
               n == "model.language_model.norm.weight" ||
               n == "output.weight" || n == "lm_head.weight" ||
               n == "model.lm_head.weight" ||
               n == "model.language_model.lm_head.weight";
    };

    for (const auto& kv : tensors) {
        const std::string& name = kv.first;
        const TensorDesc&  td   = kv.second;

        std::string out_name;
        if (keep_global(name)) {
            out_name = name;
            ++selected_global;
        } else if (name.size() > 4 && name.compare(0, 4, "blk.") == 0) {
            size_t dot2 = name.find('.', 4);
            if (dot2 == std::string::npos) continue;
            int src_layer = std::atoi(name.c_str() + 4);
            if (src_layer < layer_start || src_layer >= layer_end) continue;
            out_name = name;
            ++selected_layers;
        } else {
            const int src_layer = ::nexus::gguf::nx_layer_of_tensor(name);
            if (src_layer >= 0) {
                if (src_layer < layer_start || src_layer >= layer_end) continue;
                out_name = name;
                ++selected_layers;
            } else {
                continue;
            }
        }

        TensorDesc copy = td;
        copy.name = out_name;
        filtered.emplace(out_name, std::move(copy));
    }

    if (selected_layers == 0) {
        fprintf(stderr, "[load_range_direct] no blk tensors matched source range %d-%d\n",
                layer_start, layer_end - 1);
        return false;
    }

    fprintf(stderr,
            "[load_range_direct] selected %zu tensors (%zu layer tensors, %zu global) from source layers %d-%d; full layer count=%u\n",
            filtered.size(), selected_layers, selected_global,
            layer_start, layer_end - 1, (unsigned)meta_.n_layers);
    fflush(stderr);

    tensors.swap(filtered);

    if (cb) cb("Reading model metadata", 0.05f);
    parser_->print_meta();

    fprintf(stderr, "[load_range_direct] Loading tokenizer from source model...\n");
    fflush(stderr);
    if (!load_tokenizer()) {
        fprintf(stderr, "[nexus] Warning: tokenizer load failed\n");
    }

    if (!vocab_tokens_.empty() && vocab_tokens_.size() != meta_.vocab_size) {
        fprintf(stderr,
                "[nexus] Vocab metadata corrected from %u to %zu based on tokenizer tokens\n",
                meta_.vocab_size, vocab_tokens_.size());
        meta_.vocab_size = (uint32_t)vocab_tokens_.size();
    }

    if (cb) cb("Loading selected weights into memory", 0.1f);
    size_t loaded_cnt = 0;
    bool ok = parser_->load_weights(*mm_, [&](const gguf::LoadProgress& p) {
        ++loaded_cnt;
        float frac = 0.1f + 0.8f * ((float)p.bytes_loaded / std::max((float)p.bytes_total, 1.0f));
        if (cb) cb("Loading: " + p.current_tensor, frac);
        if (loaded_cnt % 50 == 0) {
            fprintf(stderr, "  [%3zu%%] %s\n",
                    (size_t)(frac * 100), p.current_tensor.c_str());
        }
    });
    if (!ok) {
        fprintf(stderr, "[load_range_direct] selected weight loading failed\n");
        return false;
    }

    if (cb) cb("Building layer index", 0.92f);
    if (!build_layer_weights()) {
        fprintf(stderr, "[load_range_direct] failed to index selected layer weights\n");
        return false;
    }

    if (meta_.head_dim == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[load_range_direct] FATAL: head_dim=%u hidden_dim=%u\n",
                meta_.head_dim, meta_.hidden_dim);
        return false;
    }

    // Same architecture setup as a normal full load (Qwen3.6 corrections, RoPE layout, embed scale, softcaps, activation,
    // long-context RoPE, scratch sizes). Without it a donor ran with default settings and computed wrong values.
    {
        std::string arch_l = meta_.arch;
        std::string name_l = meta_.name;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);
        const bool is_qwen35_native =
            arch_l.find("qwen3next") != std::string::npos ||
            arch_l.find("qwen3_5")   != std::string::npos ||
            arch_l.find("qwen3_6")   != std::string::npos ||
            name_l.find("qwen 3.6")  != std::string::npos ||
            name_l.find("qwen3.6")   != std::string::npos ||
            name_l.find("qwen 27")   != std::string::npos;
        if (is_qwen35_native) {
            const float old_rope = meta_.rope_freq_base;
            const float old_eps  = meta_.rms_norm_eps;
            if (meta_.rope_freq_base < 9999999.0f) meta_.rope_freq_base = 10000000.0f;
            meta_.rms_norm_eps = 1.0e-6f;
            fprintf(stderr,
                "[nexus] Native Qwen3.6 config correction: rope_theta %.0f -> %.0f, rms_eps %.2e -> %.2e\n",
                old_rope, meta_.rope_freq_base, old_eps, meta_.rms_norm_eps);
            fflush(stderr);
        }
    }

    // Setup attention config
    attn_cfg_.n_heads    = (int)meta_.n_heads;
    attn_cfg_.n_kv_heads = (int)meta_.n_kv_heads;
    attn_cfg_.head_dim   = (int)meta_.head_dim;
    attn_cfg_.hidden_dim = (int)meta_.hidden_dim;
    attn_cfg_.rms_norm_eps = meta_.rms_norm_eps;
    attn_cfg_.rope.freq_base = meta_.rope_freq_base;
    attn_cfg_.n_threads  = n_threads;

    // ------ Per-arch scaling / softcapping --------------------------------
    // Detect once at load. These are zero (= no-op) for any arch that
    // doesn't explicitly need them, so unrelated models aren't affected.
    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        std::string name_l = meta_.name;
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);

        const bool is_gemma  = (arch_l.find("gemma")  != std::string::npos) ||
                               (name_l.find("gemma")  != std::string::npos);
        // Gemma 1 has no softcap. Gemma 2 has both attn (50) and final (30).
        // Gemma 3 / Gemma 4 dropped softcaps. All Gemma versions use embed_scale.
        const bool is_gemma2 = (arch_l.find("gemma2") != std::string::npos) ||
                               (name_l.find("gemma-2") != std::string::npos) ||
                               (name_l.find("gemma2") != std::string::npos);

if (is_gemma && meta_.hidden_dim > 0) {
            embed_scale_ = sqrtf((float)meta_.hidden_dim);
        }

        const bool is_gemma4 = (arch_l.find("gemma4") != std::string::npos) ||
                               (name_l.find("gemma-4") != std::string::npos) ||
                               (name_l.find("gemma4")  != std::string::npos);

        if (is_gemma2) {
            attn_cfg_.attn_softcap = 50.0f;
            final_logit_softcap_   = 30.0f;
        } else if (is_gemma4) {
            // Gemma 4 does not use Gemma 2 attention softcap,
            // but Gemma 4 31B does use final logit softcapping.
            attn_cfg_.attn_softcap = 0.0f;

            float fc = 0.0f;
            if (parser_) {
                fc = parser_->get_f32("gemma4.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("gemma4_text.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("final_logit_softcapping", 0.0f);
            }

            final_logit_softcap_ = (fc > 0.0f) ? fc : 30.0f;
        } else {
            attn_cfg_.attn_softcap = 0.0f;
            final_logit_softcap_   = 0.0f;
        }

        // Gemma family uses GeGLU (GELU-tanh-gated) FFN, not SwiGLU.
        // Same gated structure (gate * up -> down) but with GELU instead of SiLU.
        // Without this, the FFN nonlinearity is wrong every layer and the model
        // collapses to repetitive output even with norms/scaling correct.
        ffn_activation_gelu_ = is_gemma;

        fprintf(stderr,
                "[nexus] Arch scaling: arch=%s gemma=%d gemma2=%d "
                "embed_scale=%.4f attn_softcap=%.1f final_softcap=%.1f ffn_act=%s\n",
                meta_.arch.c_str(),
                is_gemma  ? 1 : 0,
                is_gemma2 ? 1 : 0,
                embed_scale_,
                attn_cfg_.attn_softcap,
                final_logit_softcap_,
                ffn_activation_gelu_ ? "gelu_tanh" : "silu");
        fflush(stderr);

        NX_TRACE_SECTION("arch scaling decisions");
        NX_TRACE_MODEL("arch_raw",          "%s",     meta_.arch.c_str());
        NX_TRACE_MODEL("name_raw",          "%s",     meta_.name.c_str());
        NX_TRACE_MODEL("is_gemma",          "%d",     is_gemma ? 1 : 0);
        NX_TRACE_MODEL("is_gemma2",         "%d",     is_gemma2 ? 1 : 0);
        NX_TRACE_MODEL("embed_scale",       "%.6f",   embed_scale_);
        NX_TRACE_MODEL("attn_softcap",      "%.1f",   attn_cfg_.attn_softcap);
        NX_TRACE_MODEL("final_logit_softcap","%.1f",  final_logit_softcap_);
        NX_TRACE_MODEL("ffn_activation",    "%s",     ffn_activation_gelu_ ? "gelu_tanh" : "silu");
        NX_TRACE_MODEL("hidden_dim",        "%u",     meta_.hidden_dim);
        NX_TRACE_MODEL("ffn_dim",           "%u",     meta_.ffn_dim);
        NX_TRACE_MODEL("n_layers",          "%u",     meta_.n_layers);
        NX_TRACE_MODEL("n_heads",           "%u",     meta_.n_heads);
        NX_TRACE_MODEL("n_kv_heads",        "%u",     meta_.n_kv_heads);
        NX_TRACE_MODEL("head_dim",          "%u",     meta_.head_dim);
        NX_TRACE_MODEL("vocab_size",        "%u",     meta_.vocab_size);
        NX_TRACE_MODEL("context_length",    "%u",     meta_.context_length);
        NX_TRACE_MODEL("rope_freq_base_meta","%.0f",  meta_.rope_freq_base);
        NX_TRACE_MODEL("rms_norm_eps",      "%.2e",   meta_.rms_norm_eps);
        NX_TRACE_MODEL("n_experts",         "%u",     meta_.n_experts);
        NX_TRACE_MODEL("n_experts_used",    "%u",     meta_.n_experts_used);
    }

    {
        std::string arch = meta_.arch;
        std::string name = meta_.name;
        for (char& c : arch) c = (char)std::tolower((unsigned char)c);
        for (char& c : name) c = (char)std::tolower((unsigned char)c);

        const bool match_phimoe = arch.find("phimoe") != std::string::npos;
        const bool match_phi    = arch.find("phi")    != std::string::npos;
        const bool match_olmoe  = arch.find("olmoe")  != std::string::npos;
        const bool match_gemma  = arch.find("gemma")  != std::string::npos;
        // Matched on the ARCHITECTURE only -- never the model's name: a name says nothing about how the
        // q/k weights were laid out, and substrings misfire ("Dolphin" contains "phi", a Llama
        // fine-tune with "gemma" in its name is still a Llama).
        // llama.cpp's rope-type table (llama_model_rope_type): these architectures use NEOX
        // (half-split) rotation; everything else listed there (llama, mistral, mixtral, ...) is
        // NORM (interleaved).  Matched on the exact GGUF architecture string, not the model name.
        static const char* const kHalfSplitArchs[] = {
            "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe",
            "qwen3next", "qwen35", "qwen35moe", "phi2", "phi3", "phimoe", "starcoder2", "olmo2",
            "olmoe", "stablelm", "internlm2", "gptneox", "falcon", "dbrx", "exaone", "openelm",
            "gemma", "gemma2", "gemma3", "gemma3n", "gemma4"};
        bool match_arch_table = false;
        for (const char* a : kHalfSplitArchs) if (arch == a) { match_arch_table = true; break; }
        const bool half_split   = match_phimoe || match_phi || match_olmoe || match_gemma || match_arch_table;

        fprintf(stderr,
            "[nexus] RoPE layout auto-detect: arch_lower=\"%s\" name_lower=\"%s\" "
            "match_phimoe=%d match_phi=%d match_olmoe=%d match_gemma=%d -> half_split=%d\n",
            arch.c_str(), name.c_str(),
            match_phimoe ? 1 : 0, match_phi ? 1 : 0,
            match_olmoe  ? 1 : 0, match_gemma ? 1 : 0,
            half_split ? 1 : 0);
        fflush(stderr);

        if (half_split) {
            attn_cfg_.rope.layout = 1;
            fprintf(stderr, "[nexus] RoPE layout: half-split/HF-style arch=%s\n", meta_.arch.c_str());
        } else {
            attn_cfg_.rope.layout = 0;
            fprintf(stderr, "[nexus] RoPE layout: interleaved arch=%s\n", meta_.arch.c_str());
        }

        // Native .nxm packages hold the ORIGINAL HuggingFace weights (never permuted), so the GGUF arch table above -- which assumes
        // llama.cpp's converter already reordered Llama/Mistral q/k weights for interleaved rotation -- does not apply to them.
        // The builder labels the layout from the HF model_type ("nexus.rope_layout"); older packages without the label are handled
        // by the HF rule: everything except GPT-J style models rotates half-split.
        if (parser_->get_str("nexus.native_hf_weights", "") == "1") {
            const std::string lab = parser_->get_str("nexus.rope_layout", "");
            int want = -1; const char* why = "";
            if (lab == "half_split")        { want = 1; why = "manifest label"; }
            else if (lab == "interleaved")  { want = 0; why = "manifest label"; }
            else if (arch == "llama" || arch == "mistral" || arch == "mixtral") { want = 1; why = "HF rule for unpermuted llama-family weights"; }
            if (want >= 0 && want != attn_cfg_.rope.layout) {
                fprintf(stderr, "[nexus] RoPE layout (native HF weights, %s): %s -> %s\n", why,
                        attn_cfg_.rope.layout == 1 ? "half_split" : "interleaved", want == 1 ? "half_split" : "interleaved");
                attn_cfg_.rope.layout = want;
            }
        }

        // Runtime override for testing convention without recompiling.
        // NEXUS_ROPE_LAYOUT={interleaved|half_split} forces the layout.
        // Useful when the auto-detect is wrong for a specific GGUF.
        if (const char* env = std::getenv("NEXUS_ROPE_LAYOUT")) {
            int was = attn_cfg_.rope.layout;
            if (std::string(env) == "interleaved") attn_cfg_.rope.layout = 0;
            else if (std::string(env) == "half_split") attn_cfg_.rope.layout = 1;
            else if (std::string(env) == "0") attn_cfg_.rope.layout = 0;
            else if (std::string(env) == "1") attn_cfg_.rope.layout = 1;
            fprintf(stderr,
                "[nexus] RoPE layout OVERRIDE by NEXUS_ROPE_LAYOUT=%s: was=%d now=%d\n",
                env, was, attn_cfg_.rope.layout);
            fflush(stderr);
        }

        NX_TRACE_MODEL("rope_layout", "%d (%s) %s",
            attn_cfg_.rope.layout,
            attn_cfg_.rope.layout == 1 ? "half_split/NeoX" : "interleaved/GPT-J",
            std::getenv("NEXUS_ROPE_LAYOUT") ? "[ENV-OVERRIDE]" : "[auto]");
        NX_TRACE_MODEL("rope_half_split_detected", "%d", half_split ? 1 : 0);
    }

    // ------ RoPE long-context setup ---------------------------------------------------------------------------------------------------------------------------------------------
    // n_ctx_orig: the context length the model was originally trained at.
    // Used by YaRN to know which frequencies need interpolation vs extrapolation.
    // We use context_length from GGUF as the best proxy --- fine-tuned models
    // encode their extended context here. Fall back to 4096 for legacy models.
    attn_cfg_.rope.n_ctx_orig = (meta_.context_length > 0) ? meta_.context_length : 4096;

    // Static NTK-aware RoPE scaling for models with a short trained context
    // that we are running at longer context.
    //
    // Standard RoPE with base=10000 starts producing aliased (non-unique)
    // position encodings past ~8k tokens --- Q--K dot products for distant tokens
    // become near-zero, softmax weight collapses to zero, model loses the past.
    //
    // NTK scaling multiplies the base so ALL frequency dimensions scale together,
    // preserving the ratio between dimensions that the model learned during
    // training. It is KV-cache safe because it applies uniformly --- Q and K
    // use the same scaled frequencies so their dot products still cancel correctly.
    //
    // We only apply this when:
    //   1. The model has a short native theta (---10001 = not already extended)
    //   2. The model declares a context longer than 8k (it needs extension)
    //   3. head_dim is valid (guards against divide-by-zero in exponent)
    if (meta_.rope_freq_base <= 10001.0f &&
        meta_.context_length > 8192 &&
        meta_.head_dim > 2)
    {
        float ctx_ratio = (float)meta_.context_length / (float)attn_cfg_.rope.n_ctx_orig;
        if (ctx_ratio > 1.0f) {
            // NTK formula: new_base = old_base -- ratio^(d / (d-2))
            // For head_dim=128, exponent = 128/126 --- 1.016. Small but meaningful.
            float exponent = (float)meta_.head_dim / (float)(meta_.head_dim - 2);
            attn_cfg_.rope.freq_base *= powf(ctx_ratio, exponent);
        }
    }

    fprintf(stderr, "[nexus] RoPE: base=%.0f  type=%d  n_ctx_orig=%u  context=%u\n",
            attn_cfg_.rope.freq_base,
            attn_cfg_.rope.type,
            attn_cfg_.rope.n_ctx_orig,
            meta_.context_length); fflush(stderr);

    NX_TRACE_MODEL("rope_base_global",    "%.0f", attn_cfg_.rope.freq_base);
    NX_TRACE_MODEL("rope_type",           "%d",   attn_cfg_.rope.type);
    NX_TRACE_MODEL("rope_n_ctx_orig",     "%u",   attn_cfg_.rope.n_ctx_orig);
    NX_TRACE_MODEL("rope_scale",          "%.4f", attn_cfg_.rope.scale);

    // Initialize KV cache
    init_kv_cache_for_model();

    // Allocate scratch buffers
    const int hidden = (int)meta_.hidden_dim;
    const int ffn    = (int)meta_.ffn_dim;
    const int q_dim  = (int)(meta_.n_heads * meta_.head_dim);
    const int kv_dim = (int)(meta_.n_kv_heads * meta_.head_dim);

    int scratch_q_dim  = q_dim;
    int scratch_kv_dim = kv_dim;

    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);

        // Gemma4 mixes attention widths by layer. Layer 0 can be smaller than
        // later global-attention layers. Scratch must fit the largest real layer,
        // or layer 5/11/17/etc can write past qkv_scratch_ and corrupt output.
        if (arch_l.find("gemma4") != std::string::npos) {
            for (const TransformerLayerWeights& lw : layer_weights_) {
                if (lw.attn_q_dim_layer  > scratch_q_dim)  scratch_q_dim  = lw.attn_q_dim_layer;
                if (lw.attn_kv_dim_layer > scratch_kv_dim) scratch_kv_dim = lw.attn_kv_dim_layer;
            }

            fprintf(stderr,
                "[nexus] Gemma4 scratch dims: base q=%d kv=%d -> scratch q=%d kv=%d total=%d\n",
                q_dim,
                kv_dim,
                scratch_q_dim,
                scratch_kv_dim,
                scratch_q_dim + 2 * scratch_kv_dim);
            fflush(stderr);
        }
    }

    // attention scratch: q + k + v + scores(max_ctx) + output
    const int max_ctx = (int)meta_.context_length;
    const size_t attn_scratch_sz =
        (size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim +
        (size_t)max_ctx + (size_t)scratch_q_dim + 1024;

    x_scratch_.resize(hidden, 0.0f);
    xb_scratch_.resize(hidden, 0.0f);
    attn_scratch_.resize(attn_scratch_sz, 0.0f);
    ffn_scratch_.resize(std::max((size_t)hidden, (size_t)ffn * 2), 0.0f);   // gate + up
    qkv_scratch_.resize((size_t)scratch_q_dim + 2 * (size_t)scratch_kv_dim, 0.0f);
    logits_.resize(meta_.vocab_size, 0.0f);

    if (cb) cb("Ready", 1.0f);
    fprintf(stderr, "\n[nexus] Model shard loaded directly: preserving source layer IDs %d-%d of full %u.\n",
            layer_start, layer_end - 1, (unsigned)meta_.n_layers);
    fflush(stderr);
    mm_->print_report();
    loaded_ = true;
    return true;
}

// open_for_layer_streaming(): the one-time half of true single-layer load/unload.
// Opens the GGUF, loads ONLY the always-resident global tensors (embeddings, final
// norm, lm_head, Gemma4 PLE/rope-factor tensors if present) plus the tokenizer and
// chat template, and runs every piece of per-architecture setup that depends only
// on model metadata (Qwen3.6 rope/eps correction, Gemma embed-scale/softcap/FFN
// activation, RoPE layout auto-detect, long-context NTK scaling), then inits the KV
// cache and sizes scratch buffers -- all exactly once, never repeated by
// swap_layer_direct() below. Deliberately does NOT filter or swap() the parser's
// tensor table the way load_range_direct() does: every layer's TensorDesc stays
// present (just unloaded, data==nullptr) so swap_layer_direct() can find and load
// any layer on demand, and free_layer() can find and free it again later.
inline bool Transformer::open_for_layer_streaming(const std::string& model_path, int n_threads,
                                                   int probe_layer_start, int probe_layer_end) {
    n_threads_ = n_threads;
    loaded_ = false;
    active_layer_start_ = 0;
    active_layer_end_   = 0;   // owns zero layers until the first swap_layer_direct() call

    MemoryManager::Config mm_cfg;
    mm_cfg.prefer_huge_pages  = true;
    mm_cfg.lock_active_layers = true;
    mm_cfg.max_locked_layers  = std::min(n_threads * 2, 8);
    mm_ = std::make_unique<MemoryManager>(mm_cfg);
    mm_->on_pressure([this](size_t avail) {
        fprintf(stderr, "[nexus] Memory pressure! %s available. Evicting KV cache pages.\n",
                MemoryManager::fmt_bytes(avail).c_str());
        kvcache_.evict_oldest(128);
    });

    parser_ = std::make_unique<gguf::Parser>();
    fprintf(stderr, "[layer_stream] Opening source model for layer streaming: %s\n", model_path.c_str());
    fflush(stderr);
    if (!parser_->open(model_path)) {
        fprintf(stderr, "[layer_stream] Failed to open source model\n");
        return false;
    }

    meta_ = parser_->meta();
    if ((int)meta_.n_layers <= 0) {
        fprintf(stderr, "[layer_stream] bad model: n_layers=%u\n", (unsigned)meta_.n_layers);
        return false;
    }

    // Load ONLY the global tensors. Unlike load_range_direct()'s filter+swap(), the
    // full tensor table (every layer's descriptor, unloaded) is left in place.
    {
        auto& tensors = const_cast<std::unordered_map<std::string, TensorDesc>&>(parser_->tensors());
        auto is_global = [](const std::string& n) -> bool {
            static const char* const names[] = {
                "token_embd.weight", "model.embed_tokens.weight",
                "model.language_model.embed_tokens.weight", "tok_embeddings.weight",
                "output_norm.weight", "model.norm.weight", "model.language_model.norm.weight", "norm.weight",
                "output_norm.bias", "model.norm.bias", "model.language_model.norm.bias", "norm.bias",
                "output.weight", "lm_head.weight", "model.lm_head.weight", "model.language_model.lm_head.weight",
                "output.bias", "lm_head.bias", "model.lm_head.bias", "model.language_model.lm_head.bias",
                "per_layer_token_embd.weight", "embed_tokens_per_layer.weight", "gemma4.per_layer_token_embd.weight",
                "per_layer_model_proj.weight", "per_layer_model_projection.weight", "per_layer_proj.weight",
                "gemma4.per_layer_model_proj.weight", "gemma4.per_layer_model_projection.weight", "gemma4.per_layer_proj.weight",
                "per_layer_proj_norm.weight", "per_layer_projection_norm.weight", "per_layer_norm.weight",
                "gemma4.per_layer_proj_norm.weight", "gemma4.per_layer_projection_norm.weight", "gemma4.per_layer_norm.weight",
                "rope_freqs.weight",
            };
            for (const char* g : names) if (n == g) return true;
            return false;
        };
        size_t n_global = 0;
        for (auto& [name, t] : tensors) {
            if (!is_global(name)) continue;
            if (t.loaded) continue;
            t.data = mm_->alloc_weights(t.data_size, name, -1);
            if (!t.data) {
                fprintf(stderr, "[layer_stream] failed to allocate global tensor %s (%zu bytes)\n", name.c_str(), t.data_size);
                return false;
            }
            memcpy(t.data, parser_->data_ptr() + t.offset, t.data_size);
            t.loaded = true;
            ++n_global;
        }
        fprintf(stderr, "[layer_stream] loaded %zu global tensor(s); %u layers available to stream on demand, none resident yet\n",
                n_global, (unsigned)meta_.n_layers);
        fflush(stderr);
    }

    fprintf(stderr, "[layer_stream] Loading tokenizer from source model...\n");
    fflush(stderr);
    if (!load_tokenizer()) {
        fprintf(stderr, "[nexus] Warning: tokenizer load failed\n");
    }
    if (!vocab_tokens_.empty() && vocab_tokens_.size() != meta_.vocab_size) {
        fprintf(stderr, "[nexus] Vocab metadata corrected from %u to %zu based on tokenizer tokens\n",
                meta_.vocab_size, vocab_tokens_.size());
        meta_.vocab_size = (uint32_t)vocab_tokens_.size();
    }

    // Same architecture setup as load_range_direct() -- all of it is metadata-only
    // (model-wide, not per-layer), so it is correct to run it exactly once here and
    // never again on a swap.
    {
        std::string arch_l = meta_.arch;
        std::string name_l = meta_.name;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);
        const bool is_qwen35_native =
            arch_l.find("qwen3next") != std::string::npos ||
            arch_l.find("qwen3_5")   != std::string::npos ||
            arch_l.find("qwen3_6")   != std::string::npos ||
            name_l.find("qwen 3.6")  != std::string::npos ||
            name_l.find("qwen3.6")   != std::string::npos ||
            name_l.find("qwen 27")   != std::string::npos;
        if (is_qwen35_native) {
            const float old_rope = meta_.rope_freq_base;
            const float old_eps  = meta_.rms_norm_eps;
            if (meta_.rope_freq_base < 9999999.0f) meta_.rope_freq_base = 10000000.0f;
            meta_.rms_norm_eps = 1.0e-6f;
            fprintf(stderr, "[nexus] Native Qwen3.6 config correction: rope_theta %.0f -> %.0f, rms_eps %.2e -> %.2e\n",
                    old_rope, meta_.rope_freq_base, old_eps, meta_.rms_norm_eps);
            fflush(stderr);
        }
    }

    attn_cfg_.n_heads    = (int)meta_.n_heads;
    attn_cfg_.n_kv_heads = (int)meta_.n_kv_heads;
    attn_cfg_.head_dim   = (int)meta_.head_dim;
    attn_cfg_.hidden_dim = (int)meta_.hidden_dim;
    attn_cfg_.rms_norm_eps = meta_.rms_norm_eps;
    attn_cfg_.rope.freq_base = meta_.rope_freq_base;
    attn_cfg_.n_threads  = n_threads;

    {
        std::string arch_l = meta_.arch;
        for (char& c : arch_l) c = (char)std::tolower((unsigned char)c);
        std::string name_l = meta_.name;
        for (char& c : name_l) c = (char)std::tolower((unsigned char)c);

        const bool is_gemma  = (arch_l.find("gemma")  != std::string::npos) || (name_l.find("gemma")  != std::string::npos);
        const bool is_gemma2 = (arch_l.find("gemma2") != std::string::npos) || (name_l.find("gemma-2") != std::string::npos) || (name_l.find("gemma2") != std::string::npos);
        if (is_gemma && meta_.hidden_dim > 0) embed_scale_ = sqrtf((float)meta_.hidden_dim);
        const bool is_gemma4 = (arch_l.find("gemma4") != std::string::npos) || (name_l.find("gemma-4") != std::string::npos) || (name_l.find("gemma4") != std::string::npos);

        if (is_gemma2) {
            attn_cfg_.attn_softcap = 50.0f;
            final_logit_softcap_   = 30.0f;
        } else if (is_gemma4) {
            attn_cfg_.attn_softcap = 0.0f;
            float fc = 0.0f;
            if (parser_) {
                fc = parser_->get_f32("gemma4.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("gemma4_text.final_logit_softcapping", 0.0f);
                if (fc == 0.0f) fc = parser_->get_f32("final_logit_softcapping", 0.0f);
            }
            final_logit_softcap_ = (fc > 0.0f) ? fc : 30.0f;
        } else {
            attn_cfg_.attn_softcap = 0.0f;
            final_logit_softcap_   = 0.0f;
        }
        ffn_activation_gelu_ = is_gemma;

        fprintf(stderr, "[nexus] (layer streaming) Arch scaling: arch=%s gemma=%d gemma2=%d embed_scale=%.4f attn_softcap=%.1f final_softcap=%.1f ffn_act=%s\n",
                meta_.arch.c_str(), is_gemma ? 1 : 0, is_gemma2 ? 1 : 0, embed_scale_,
                attn_cfg_.attn_softcap, final_logit_softcap_, ffn_activation_gelu_ ? "gelu_tanh" : "silu");
        fflush(stderr);
    }

    {
        std::string arch = meta_.arch;
        std::string name = meta_.name;
        for (char& c : arch) c = (char)std::tolower((unsigned char)c);
        for (char& c : name) c = (char)std::tolower((unsigned char)c);
        const bool match_phimoe = arch.find("phimoe") != std::string::npos;
        const bool match_phi    = arch.find("phi")    != std::string::npos;
        const bool match_olmoe  = arch.find("olmoe")  != std::string::npos;
        const bool match_gemma  = arch.find("gemma")  != std::string::npos;
        static const char* const kHalfSplitArchs[] = {
            "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe", "qwen3vl", "qwen3vlmoe",
            "qwen3next", "qwen35", "qwen35moe", "phi2", "phi3", "phimoe", "starcoder2", "olmo2",
            "olmoe", "stablelm", "internlm2", "gptneox", "falcon", "dbrx", "exaone", "openelm",
            "gemma", "gemma2", "gemma3", "gemma3n", "gemma4"};
        bool match_arch_table = false;
        for (const char* a : kHalfSplitArchs) if (arch == a) { match_arch_table = true; break; }
        const bool half_split = match_phimoe || match_phi || match_olmoe || match_gemma || match_arch_table;
        attn_cfg_.rope.layout = half_split ? 1 : 0;

        if (parser_->get_str("nexus.native_hf_weights", "") == "1") {
            const std::string lab = parser_->get_str("nexus.rope_layout", "");
            int want = -1;
            if (lab == "half_split")        want = 1;
            else if (lab == "interleaved")  want = 0;
            else if (arch == "llama" || arch == "mistral" || arch == "mixtral") want = 1;
            if (want >= 0) attn_cfg_.rope.layout = want;
        }
        if (const char* env = std::getenv("NEXUS_ROPE_LAYOUT")) {
            if (std::string(env) == "interleaved" || std::string(env) == "0") attn_cfg_.rope.layout = 0;
            else if (std::string(env) == "half_split" || std::string(env) == "1") attn_cfg_.rope.layout = 1;
        }
    }

    attn_cfg_.rope.n_ctx_orig = (meta_.context_length > 0) ? meta_.context_length : 4096;
    if (meta_.rope_freq_base <= 10001.0f && meta_.context_length > 8192 && meta_.head_dim > 2) {
        float ctx_ratio = (float)meta_.context_length / (float)attn_cfg_.rope.n_ctx_orig;
        if (ctx_ratio > 1.0f) {
            float exponent = (float)meta_.head_dim / (float)(meta_.head_dim - 2);
            attn_cfg_.rope.freq_base *= powf(ctx_ratio, exponent);
        }
    }

    layer_weights_.resize(meta_.n_layers);
    if (!build_layer_weights()) {
        fprintf(stderr, "[layer_stream] failed to index global tensors\n");
        return false;
    }

    if (meta_.head_dim == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[layer_stream] FATAL: head_dim=%u hidden_dim=%u\n", meta_.head_dim, meta_.hidden_dim);
        return false;
    }
    
    loaded_ = true;

    // *** v26: used to run a full separate "probe pass" here -- swap into every donor
    // layer once just to read its real KV geometry, THEN leave nothing resident so the
    // real streaming loop could swap into every one of them again to actually compute
    // it. That touched every donor layer's file data twice before any of them were run
    // once. Folded into a single pass instead: this call does no file I/O at all, just
    // sizes the bookkeeping arrays so indexing by a real (not-yet-reached) layer number
    // is always valid. swap_layer_direct() below now discovers and records each
    // layer's real geometry itself, the one time it's actually streamed in for real.
    (void)probe_layer_start; (void)probe_layer_end;  // kept for call-site compatibility; no longer used to bound a separate probe
    kvcache_.init_per_layer_streaming((int)meta_.n_layers);

    const int hidden = (int)meta_.hidden_dim;
    const int ffn    = (int)meta_.ffn_dim;
    const int q_dim  = (int)(meta_.n_heads * meta_.head_dim);
    const int kv_dim = (int)(meta_.n_kv_heads * meta_.head_dim);
    const int max_ctx = (int)meta_.context_length;
    const size_t attn_scratch_sz = (size_t)q_dim + 2 * (size_t)kv_dim + (size_t)max_ctx + (size_t)q_dim + 1024;
    x_scratch_.resize(hidden, 0.0f);
    xb_scratch_.resize(hidden, 0.0f);
    attn_scratch_.resize(attn_scratch_sz, 0.0f);
    ffn_scratch_.resize(std::max((size_t)hidden, (size_t)ffn * 2), 0.0f);
    qkv_scratch_.resize((size_t)q_dim + 2 * (size_t)kv_dim, 0.0f);
    logits_.resize(meta_.vocab_size, 0.0f);
    // NOTE: Gemma4's per-layer variable attention width (load_range_direct() widens
    // scratch using every loaded layer's attn_q_dim_layer/attn_kv_dim_layer) cannot
    // be done here, since layers are never all resident at once in streaming mode.
    // Hybrid prefill on a Gemma4 donor uses the base q_dim/kv_dim size; if a later
    // Gemma4 GGUF needs a wider scratch for some layers, widen attn_scratch_sz here
    // with a fixed safety margin rather than scanning layer_weights_.

    mm_->print_report();
    loaded_ = true;
    fprintf(stderr, "[layer_stream] READY -- model opened once (globals + tokenizer only); call swap_layer_direct() to stream in each layer\n");
    fflush(stderr);
    return true;
}

// swap_layer_direct(): the repeated half. Frees prev_layer's weights (no-op if
// prev_layer < 0, e.g. the first call), loads new_layer's weights via the existing
// Parser::load_layer(), and re-points layer_weights_[new_layer] at them by narrowing
// active_layer_start_/active_layer_end_ to just that one layer and calling the
// existing build_layer_weights() -- which already skips every layer outside that
// range, so no other layer's (there are none resident anyway) or global tensor's
// pointers are disturbed. Never reopens the file, never recreates parser_/mm_,
// never reloads the tokenizer, never touches the KV cache -- all genuinely new work
// here is one layer's worth of memcpy from the still-open mmap.
inline bool Transformer::swap_layer_direct(int new_layer, int prev_layer) {
    if (!parser_ || !mm_ || !loaded_) {
        fprintf(stderr, "[layer_stream] swap_layer_direct: not open -- call open_for_layer_streaming() first\n");
        return false;
    }
    if (new_layer < 0 || new_layer >= (int)meta_.n_layers) {
        fprintf(stderr, "[layer_stream] swap_layer_direct: layer %d out of range (0-%d)\n", new_layer, (int)meta_.n_layers - 1);
        return false;
    }

    if (prev_layer >= 0 && prev_layer != new_layer) {
        parser_->free_layer(prev_layer, *mm_);
    }

    if (!parser_->load_layer(new_layer, *mm_)) {
        fprintf(stderr, "[layer_stream] swap_layer_direct: failed to load layer %d\n", new_layer);
        return false;
    }

    active_layer_start_ = new_layer;
    active_layer_end_   = new_layer + 1;

    if (!build_layer_weights()) {
        fprintf(stderr, "[layer_stream] swap_layer_direct: failed to index layer %d\n", new_layer);
        return false;
    }

    // *** v26: this used to be learned in a separate probe pass (see
    // open_for_layer_streaming()) that swapped into every donor layer once just to read
    // this, before the real loop swapped into every one of them again. Recorded here
    // instead, the one time this layer is genuinely streamed in -- add_layer_geometry()
    // is idempotent, so this is harmless even if a layer is ever swapped in more than once.
    {
        const TransformerLayerWeights& lw = layer_weights_[(size_t)new_layer];
        int nh = (int)meta_.n_kv_heads, hd = (int)meta_.head_dim;
        if (lw.qwen35_linear) {
            nh = 0;
        } else {
            if (lw.attn_kv_heads_layer > 0) nh = lw.attn_kv_heads_layer;
            if (lw.attn_head_dim_layer > 0) hd = lw.attn_head_dim_layer;
        }
        kvcache_.add_layer_geometry(new_layer, nh, hd);
    }
    return true;
}

// *** v29: free_resident_layer()/reload_resident_layer() -- see the declaration
// comment above. Both reuse the SAME Parser::free_layer()/load_layer() primitives
// swap_layer_direct() already relies on for the layer-streaming ghost, and the
// same build_layer_weights() every load path already calls to (re)point
// layer_weights_ at whatever is currently loaded/unloaded in the tensor table --
// nothing new is added to the parser or the memory manager. The difference from
// swap_layer_direct() is deliberate: active_layer_start_/active_layer_end_ are
// left completely alone, because the caller (the primary, decoding with its own
// normal layer range) is not a one-layer-at-a-time ghost -- it still owns every
// layer in that range, just with this one temporarily unloaded.
//
// Guarded to only operate within [active_layer_start_, active_layer_end_): a
// layer this engine does not itself own was never loaded by it in the first
// place, so freeing/reloading it here would be meaningless at best.
inline bool Transformer::free_resident_layer(int l) {
    if (!parser_ || !mm_ || l < active_layer_start_ || l >= active_layer_end_) return false;
    parser_->free_layer(l, *mm_);
    return build_layer_weights();
}
inline bool Transformer::reload_resident_layer(int l) {
    if (!parser_ || !mm_ || l < active_layer_start_ || l >= active_layer_end_) return false;
    if (!parser_->load_layer(l, *mm_)) return false;
    return build_layer_weights();
}

// =====================================================================================================================
// Layer parking: "Maximize GPU" and oversized-prompt protection (see the declaration of park_setup).
//
// How a layer leaves RAM without being unloaded: the server has the loader map every big layer weight in place from
// the model file (a private, copy-on-write mapping -- the file itself is never written). Touching each page once makes
// the ordinary RAM copy a normal layer has (park_touch_layer). madvise(MADV_DONTNEED) throws that copy away again
// (park_drop_layer): the weight pointers stay valid, and the next read comes from the model file. So a parked layer can
// still be read at any time -- by the prompt stream, by the upload to the card, by the CPU -- it just costs a read from
// the drive (or the OS file cache) instead of RAM. Nothing ever writes to these weights after loading, so the file's
// bytes are the weights.
// =====================================================================================================================
inline size_t Transformer::park_layer_bytes(int l) const {
    size_t b = 0;
    if (l >= 0 && l < (int)park_ranges_.size()) for (auto& r : park_ranges_[l]) b += r.n;
    return b;
}
inline void Transformer::park_touch_layer(int l) {
#if defined(__linux__)
    if (l < 0 || l >= (int)park_ranges_.size()) return;
    for (auto& r : park_ranges_[l]) {
        const uintptr_t b = (uintptr_t)r.p, e = b + r.n;
        for (uintptr_t a = b & ~(uintptr_t)4095; a < e; a += 4096) {
            volatile uint8_t* q = (volatile uint8_t*)(a < b ? b : a);
            *q = *q;                              // copy-on-write: this page now lives in RAM
        }
    }
#else
    (void)l;
#endif
}
inline void Transformer::park_drop_layer(int l) {
#if defined(__linux__)
    if (l < 0 || l >= (int)park_ranges_.size()) return;
    static const uintptr_t ps = (uintptr_t)std::max(4096L, sysconf(_SC_PAGESIZE));
    for (auto& r : park_ranges_[l]) {
        // Whole pages only. A page shared with a neighbouring tensor holds the same bytes as the file, so letting it
        // fall back to the file is harmless (that neighbour just reads it from the file cache next time).
        const uintptr_t b = (uintptr_t)r.p & ~(ps - 1), e = (uintptr_t)r.p + r.n;
        if (e > b) madvise((void*)b, (size_t)(e - b), MADV_DONTNEED);
    }
#else
    (void)l;
#endif
}

inline bool Transformer::park_setup(int park_layers, int below_mb, int keep_mb, int prefill_free_mb, int min_tokens) {
    const int nl = (int)layer_weights_.size();
    park_on_ = false; park_pending_ = false;
    park_ranges_.assign(nl, {}); park_state_.assign(nl, 0); park_gpu_w_.assign(nl, {}); park_home_card_.assign(nl, 0);
    size_t mapped = 0;
    for (auto& r : gguf::nx_direct_ranges())
        if (r.layer >= 0 && r.layer < nl) { park_ranges_[r.layer].push_back({(uint8_t*)r.p, r.n}); mapped += r.n; r.layer = -2; }
    auto off = [&](const char* why) -> bool {
        fprintf(stderr, "[park] Maximize GPU / oversized-prompt protection is OFF: %s\n", why);
        size_t b = 0; for (int l = 0; l < nl; ++l) { park_touch_layer(l); b += park_layer_bytes(l); }   // everything into RAM, as without it
        if (b) fprintf(stderr, "[park] %zu MB of layer weights copied into RAM (normal loading)\n", b >> 20);
        fflush(stderr);
        park_on_ = false;
        return false;
    };
#if !defined(__linux__) || !defined(NEXUS_VULKAN)
    (void)park_layers; (void)below_mb; (void)keep_mb; (void)prefill_free_mb; (void)min_tokens;
    return off("this build or system has no support for it (Linux with graphics-card support only)");
#else
    if (mapped == 0)          return off("the model's weights could not be mapped in place from its file on this system, so layers can't leave RAM");
    if (active_layer_end_ >= 0) return off("this computer runs only part of the model (shared with other computers)");
    if (!turbo_gpu_stream_ || !gpu::gpu().ready()) return off("it needs the prompt speed-up (prefill acceleration) on a graphics card");
    const gpu::GpuInfo gi = gpu::gpu().info();
    if (gi.unified_memory || gi.vram_bytes == 0) return off("the card shares system RAM (integrated graphics), so moving layers onto it frees nothing");
    park_below_bytes_        = below_mb > 0 ? (long long)below_mb << 20 : 0;
    mtp_reserve_bytes_ = 0;
    if (mtp_card_reserve_) {
        static const char* mtp_names[] = {"mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight", "mtp.layers.0.self_attn.o_proj.weight", "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight", "mtp.layers.0.mlp.down_proj.weight"};
        for (const char* n : mtp_names) if (TensorDesc* t = find(n)) mtp_reserve_bytes_ += (size_t)t->data_size;
        if (mtp_reserve_bytes_) {
            mtp_reserve_bytes_ += 64ull << 20;   // its activations and buffers
            fprintf(stderr, "[park] keeping %zu MB free on the card for MTP drafting on the card\n", mtp_reserve_bytes_ >> 20);
        }
    }
    park_keep_bytes_         = (size_t)std::max(64, keep_mb) << 20;
    park_prefill_free_bytes_ = (size_t)std::max(0, prefill_free_mb) << 20;
    park_min_tokens_         = std::max(0, min_tokens);

    int want = park_layers;
    if (want < 0) {                                    // as many as fit on the card (what is really free on it, when the driver says)
        const size_t room = park_card_room();
        size_t sum = 0; want = 0;
        for (int l = 0; l < nl; ++l) {
            if (layer_weights_[l].gpu2_layer) continue;        // already lives on another card
            if (park_ranges_[l].empty()) break;
            const size_t b = std::max(park_layer_bytes(l), turbo_layer_weight_bytes(l));
            if (sum + b > room) break;
            sum += b; ++want;
        }
    }
    want = std::max(0, std::min(want, nl));
    size_t in_ram = 0, parked = 0; int n_parked = 0;
    fprintf(stderr, "[park] copying the layers that live in RAM into RAM...\n"); fflush(stderr);
    for (int l = 0; l < nl; ++l) {
        if (n_parked < want && !park_ranges_[l].empty() && !layer_weights_[l].gpu2_layer) {   // layers on other cards are theirs
            park_state_[l] = 1; park_home_card_[l] = 1; parked += park_layer_bytes(l); ++n_parked;
        }
        else { park_touch_layer(l); in_ram += park_layer_bytes(l); }
        if ((l & 3) == 3 || l == nl - 1) {                       // the loading bar, not the prompt bar
            fprintf(stderr, "[load] %d%%  Reading layers into RAM (%d of %d)\n", (int)(100.0 * (l + 1) / nl), l + 1, nl);
            fflush(stderr);
        }
    }
    {   // writing on the card needs its activation buffers (normally set up with "layers on the card")
        const size_t q_dim  = (size_t)(meta_.n_heads    * meta_.head_dim);
        const size_t kv_dim = (size_t)(meta_.n_kv_heads * meta_.head_dim);
        gpu::gpu().init_activation_bufs(meta_.hidden_dim, q_dim, kv_dim, meta_.ffn_dim);
    }
    park_on_ = true;
    fprintf(stderr, "[park] ON: %zu MB of layer weights in RAM; %d layers (%zu MB) parked for the card, with no copy in RAM. "
                    "Card %zu MB, %zu MB always kept free; before a prompt is read, layers move off the card only if %s is not free. "
                    "Oversized-prompt protection: %s.\n",
            in_ram >> 20, n_parked, parked >> 20, (size_t)(gi.vram_bytes >> 20), park_keep_bytes_ >> 20,
            park_prefill_free_bytes_ ? (std::to_string(park_prefill_free_bytes_ >> 20) + " MB").c_str()
                                     : ("what the prompt speed-up needs (auto, " + std::to_string(park_stream_needs() >> 20) + " MB)").c_str(),
            park_below_bytes_ ? ("drop layers from RAM when free RAM is under " + std::to_string(park_below_bytes_ >> 20) + " MB").c_str() : "off");
    fflush(stderr);
    if (n_parked) { park_to_gpu("model loaded"); park_finish_pending(); }     // any the card can't take: RAM, not the drive
    return true;
#endif
}

// Parked layers that are not on the card yet go onto it, as far as the card allows. Returns how many moved.
inline int Transformer::park_to_gpu(const char* why) {
#if defined(__linux__) && defined(NEXUS_VULKAN)
    auto& g = gpu::gpu();
    if (!park_on_ || !g.ready()) return 0;
    std::vector<int> todo; size_t todo_b = 0;
    for (int pass = 0; pass < 2; ++pass)              // Maximize GPU's own layers first, then layers protection dropped
        for (int l = 0; l < (int)park_state_.size(); ++l)
            if (park_state_[l] == 1 && park_home_card_[l] == (pass == 0 ? 1 : 0)) { todo.push_back(l); todo_b += park_layer_bytes(l); }
    if (todo.empty()) return 0;
    g.wait_idle();
    g.evict_unpinned_weights();                       // what the prompt stream left behind goes first
    // Room is checked before every layer against what the card really has free (desktop and Nexus's own buffers
    // included, when the driver reports it), so the card is never filled past its real memory -- a card filled past
    // it either fails or quietly spills into system RAM, and then everything on it crawls.
    g.set_weight_cache_budget(g.weight_cache_used() + park_card_room());
    fprintf(stderr, "[park] %s: moving %zu layers (%zu MB) onto the card for writing -- reading them from the model file, "
                    "this pauses before the reply starts...\n", why, todo.size(), todo_b >> 20);
    fflush(stderr);
    const auto t0 = std::chrono::steady_clock::now();
    park_mv_total_.store((int)todo.size()); park_mv_done_.store(0);
    int moved = 0; size_t moved_b = 0;
    for (int l : todo) {
        const size_t need = std::max(park_layer_bytes(l), turbo_layer_weight_bytes(l));
        if (need > park_card_room()) break;
        std::vector<const void*> mine;
        if (!eager_upload_layer_weights(l, g, &mine)) break;
        {   // Qwen3.5 linear-attention matrices are not in the normal upload; without them here they would be fetched
            // lazily from the model file while writing and could be pushed out again -- a drive read on every token.
            const TransformerLayerWeights& lw = layer_weights_[l];
            bool ok = true;
            auto up = [&](const void* W, QuantType qt, size_t rows, size_t cols) {
                if (!ok || !W || rows == 0 || cols == 0) return;
                size_t sz = 0; uint32_t dummy = 0;
                if (!g.size_for_public(qt, rows, cols, sz, dummy) || sz == 0) return;     // a type the card can't use: stays on the CPU
                if (!g.get_weight_buffer_public(W, sz, qt) || g.weight_size_public(W) < sz) { ok = false; return; }
                mine.push_back(W);
            };
            if (lw.qwen35_linear) {
                const size_t hidden = (size_t)meta_.hidden_dim;
                const size_t conv  = (size_t)std::max(0, lw.q35_conv_dim);
                const size_t value = (size_t)std::max(0, lw.q35_value_dim);
                const size_t vh    = (size_t)std::max(0, lw.q35_num_v_heads);
                up(lw.q35_qkv, lw.q35_qkv_type, conv, hidden);
                up(lw.q35_z,   lw.q35_z_type,   value, hidden);
                up(lw.q35_out, lw.q35_out_type, hidden, value);
                if (lw.q35_ba) up(lw.q35_ba, lw.q35_ba_type, vh * 2u, hidden);
                else { up(lw.q35_a, lw.q35_a_type, vh, hidden); up(lw.q35_b, lw.q35_b_type, vh, hidden); }
            }
            if (!ok) { for (const void* w : mine) g.evict_weight_public(w); break; }
        }
        for (const void* w : mine) g.pin_weight_public(w);
        {   // Any weight of this layer that did NOT go onto the card (a type or size the card doesn't take) is still
            // computed on the CPU while writing, so it keeps its RAM copy -- the CPU never reads it from the drive.
            size_t kept = 0;
            for (auto& r : park_ranges_[l]) {
                bool on_card = false;
                for (const void* w : mine) if (w == (const void*)r.p) { on_card = true; break; }
                if (on_card) continue;
                const uintptr_t b = (uintptr_t)r.p, e = b + r.n;
                for (uintptr_t a = b & ~(uintptr_t)4095; a < e; a += 4096) { volatile uint8_t* q = (volatile uint8_t*)(a < b ? b : a); *q = *q; }
                kept += r.n;
            }
            if (kept) fprintf(stderr, "[park] layer %d: %zu MB of it can't go on the card and stays in RAM\n", l, kept >> 20);
        }
        park_gpu_w_[l] = std::move(mine);
        park_state_[l] = 2;
        layer_weights_[l].gpu_layer = true;
        ++moved; moved_b += park_layer_bytes(l);
        park_mv_done_.store(moved);
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // While the model loads this belongs on the loading bar ("[load]"); after a prompt it is its own step on the
        // prompt bar, labelled as what it is -- never "reading prompt".
        if (strcmp(why, "model loaded") == 0)
            fprintf(stderr, "[load] %.0f%%  Moving layers onto the graphics card (%d of %zu, %zu MB)\n",
                    100.0 * moved / (double)todo.size(), moved, todo.size(), moved_b >> 20);
        else
            fprintf(stderr, "[progress] moving layers onto the card: %.0f%%  (prompt read; layer %d, %d of %zu, %zu MB, %.0fs elapsed)\n",
                    100.0 * moved / (double)todo.size(), l, moved, todo.size(), moved_b >> 20, el);
        fflush(stderr);
    }
    g.wait_idle();
    park_mv_total_.store(0); park_mv_done_.store(0);
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    size_t fr = 0; const bool real = g.vram_free_now(fr);
    fprintf(stderr, "[park] %d of %zu layers (%zu MB) on the card in %.1f s%s; card memory free now: %zu MB%s\n", moved, todo.size(),
            moved_b >> 20, el, moved < (int)todo.size() ? " -- the card is full; the rest stay off it" : "",
            (real ? fr : park_card_room() + park_keep_bytes_) >> 20, real ? "" : " (estimated: this driver doesn't report it)");
    fflush(stderr);
    return moved;
#else
    (void)why; return 0;
#endif
}

inline void Transformer::turbo_keep_prepare() {
#if defined(NEXUS_VULKAN)
    if (turbo_keep_ready_ || !turbo_gpu_stream_ || !gpu::gpu().ready()) return;
    const int n = (int)layer_weights_.size();
    turbo_keep_.assign(n, 0);
    int cnt = 0;
    for (int l = 0; l < n; ++l)
        if (layer_weights_[l].gpu_layer && !layer_weights_[l].gpu2_layer && !park_on_card(l)) { turbo_keep_[l] = 1; ++cnt; }
    if (cnt) {
        // Every weight of these layers onto the card once and pinned (the normal upload leaves some out), so reading a
        // prompt never pushes them off and writing finds them all there.
        for (int l = 0; l < n; ++l) if (turbo_keep_[l]) turbo_load_layer_weights_resident(l, "keep", true);
        auto& g = gpu::gpu();
        const size_t vram = (size_t)g.info().vram_bytes, used = g.weight_cache_used();
        size_t fr = 0; const bool real = g.vram_free_now(fr);
        if (!real) fr = vram > used ? vram - used : 0;
        fprintf(stderr, "[turbo] %d layers live on the prompt speed-up card (%zu MB): prompts are read through them in place and "
                        "they stay there for writing; the other layers stream through the rest of the card (%zu MB free%s)\n",
                cnt, used >> 20, fr >> 20, real ? "" : ", estimated");
        fflush(stderr);
    }
    turbo_keep_ready_ = true;
#endif
}

// What the prompt stream needs on the card: its rolling window of layers plus the largest layer, and the
// attention/KV reserve and safety margin compute_and_set_turbo_vram_budget() uses.
inline size_t Transformer::park_stream_needs(int ctx_tokens) const {
#if defined(NEXUS_VULKAN)
    const int n = (int)layer_weights_.size(), win = std::max(1, turbo_layer_window_size());
    size_t max_window = 0, cur = 0, largest = 0;
    std::vector<size_t> bytes(n);
    for (int l = 0; l < n; ++l) { bytes[l] = std::max(turbo_layer_weight_bytes(l), park_layer_bytes(l)); largest = std::max(largest, bytes[l]); }
    for (int l = 0; l < n; ++l) { cur += bytes[l]; if (l >= win) cur -= bytes[l - win]; if (l >= win - 1) max_window = std::max(max_window, cur); }
    if (n < win) max_window = cur;
    // Attention needs one layer's keys and values for the WHOLE conversation on the card (f16 K and V): at 86k tokens
    // on a 27B that's about 344 MB and it grows with the prompt, so it is sized from the context, not a fixed amount.
    const size_t kv_layer = (size_t)std::max(0, ctx_tokens) * (size_t)meta_.n_kv_heads * (size_t)meta_.head_dim * 2u * sizeof(uint16_t);
    const size_t attn = std::max<size_t>((size_t)320 << 20, kv_layer + ((size_t)64 << 20));
    return max_window + largest + attn + ((size_t)384 << 20);
#else
    return 0;
#endif
}

// Cards 2, 3, 4 while writing: if one really has less than 256 MB free (the driver reports it), its last layer goes back to
// the CPU (those layers keep their RAM copy). Nothing watched them before.
inline void Transformer::other_cards_watch() {
#if defined(NEXUS_VULKAN)
    std::vector<gpu::VulkanBackend*> cards;
    if (gpu2_ && gpu2_->ready()) cards.push_back(gpu2_.get());
    for (auto& b : gpux_) if (b && b->ready()) cards.push_back(b.get());
    for (gpu::VulkanBackend* b : cards) {
        size_t fr = 0;
        if (!b->vram_free_now(fr) || fr >= (256ull << 20)) continue;
        for (int l = (int)layer_weights_.size() - 1; l >= 0; --l) {
            TransformerLayerWeights& lw = layer_weights_[(size_t)l];
            if (!lw.gpu2_layer || backend2_for(lw) != b) continue;
            b->wait_idle();
            if ((size_t)l < other_card_w_.size()) {
                for (const void* w : other_card_w_[(size_t)l]) { b->unpin_weight_public(w); b->evict_weight_public(w); }
                other_card_w_[(size_t)l].clear();
            }
            lw.gpu2_layer = false; lw.gpux = -1;
            if (park_on_ && (size_t)l < park_state_.size() && park_state_[(size_t)l] == 0) park_touch_layer(l);   // its RAM copy
            size_t fr2 = 0; b->vram_free_now(fr2);
            fprintf(stderr, "[nexus] graphics card %s is nearly full while writing (%zu MB free): layer %d moved off it, to the CPU "
                            "(card free now %zu MB)\n", b->info().device_name.c_str(), fr >> 20, l, fr2 >> 20);
            fflush(stderr);
            break;
        }
    }
#endif
}

inline void Transformer::park_watch() {
#if defined(__linux__) && defined(NEXUS_VULKAN)
    other_cards_watch();
    if (!park_on_) return;
    auto& g = gpu::gpu();
    if (!g.ready()) return;
    // 1. The card: only acted on when the driver reports real free memory (an estimate can't see a spill coming).
    //    256 MB (was 64): working buffers grow in steps of tens of MB, so 64 MB was too late to act on.
    size_t fr = 0;
    if (park_any_on_card() && g.vram_free_now(fr) && fr < (256ull << 20)) {
        for (int l = (int)park_state_.size() - 1; l >= 0; --l) {
            if (park_state_[l] != 2) continue;
            g.wait_idle();
            for (const void* w : park_gpu_w_[l]) { g.unpin_weight_public(w); g.evict_weight_public(w); }
            park_gpu_w_[l].clear();
            layer_weights_[l].gpu_layer = false;
            park_touch_layer(l);                      // into RAM -- never left on the drive while writing
            park_state_[l] = 0;
            size_t fr2 = 0; g.vram_free_now(fr2);
            fprintf(stderr, "[park] the card is nearly full while writing (%zu MB free): layer %d (%zu MB) moved off it into RAM (card free now %zu MB)\n",
                    fr >> 20, l, park_layer_bytes(l) >> 20, fr2 >> 20);
            fflush(stderr);
            break;
        }
    }
    // 2. RAM: the reply's conversation memory grows. With protection on and RAM under its number, one layer moves
    //    from RAM onto the card -- only if the card has room for it. A layer is dropped to be read from the drive only
    //    in a real emergency (under the hard minimum), because that makes every token re-read it.
    if (park_below_bytes_ > 0) {
        const long long av = nexus::memguard::available_bytes();
        if (av >= 0 && av < park_below_bytes_) {
            int pick = -1;
            for (int l = (int)park_state_.size() - 1; l >= 0; --l)
                if (park_state_[l] == 0 && !park_ranges_[l].empty()) { pick = l; break; }
            if (pick >= 0) {
                const size_t need = std::max(park_layer_bytes(pick), turbo_layer_weight_bytes(pick));
                if (need <= park_card_room()) {          // only when the card can take it; never onto the drive
                    park_drop_layer(pick); park_state_[pick] = 1;
                    fprintf(stderr, "[park] free RAM %lld MB is under %lld MB while writing: layer %d (%zu MB) moves from RAM onto the card\n",
                            av >> 20, park_below_bytes_ >> 20, pick, park_layer_bytes(pick) >> 20);
                    fflush(stderr);
                    park_to_gpu("RAM short while writing");
                    park_finish_pending();               // if the upload failed after all, back into RAM
                }
            }
        }
    }
#endif
}

// Card memory free for more layers. With VK_EXT_memory_budget this is what the driver says is really free (the
// desktop and every Nexus buffer included) minus the kept-free amount; otherwise an estimate from the card's size.
inline size_t Transformer::park_card_room() const {
#if defined(NEXUS_VULKAN)
    auto& g = gpu::gpu();
    if (!g.ready()) return 0;
    size_t fr = 0;
    if (!g.vram_free_now(fr)) {
        const size_t vram = (size_t)g.info().vram_bytes, used = g.weight_cache_used() + g.kv_mirror_used();
        fr = vram > used ? vram - used : 0;
    }
    const size_t keep = park_keep_bytes_ + mtp_reserve_bytes_;
    return fr > keep ? fr - keep : 0;
#else
    return 0;
#endif
}
// Free RAM to keep when putting layers back into RAM: the protection threshold (at least 512 MB) plus room for the
// reply's own conversation memory (4096 tokens of KV), so the reply doesn't push RAM straight back under the line.
inline long long Transformer::park_ram_floor() const {
    const long long base = std::max<long long>(park_below_bytes_, 512LL << 20);
    return base + (long long)kv_bytes_per_token() * 4096;
}

// Before a prompt is read: if the card does not have the room the prompt speed-up needs (`prefill_free_mb`, or auto),
// take parked layers off it until it does. Those are read from the model file while the prompt is read, and go back
// on the card afterwards. When the room is already there, nothing moves and there is no pause.
inline void Transformer::park_off_gpu(int n_tokens, int ctx_tokens) {
#if defined(__linux__) && defined(NEXUS_VULKAN)
    auto& g = gpu::gpu();
    if (!park_on_ || !g.ready() || !park_any_on_card()) return;
    const size_t vram = (size_t)g.info().vram_bytes;
    auto free_now = [&]() -> size_t {           // what the card really has free, when the driver reports it
        size_t fr = 0;
        if (g.vram_free_now(fr)) return fr;
        const size_t used = g.weight_cache_used() + g.kv_mirror_used();
        return vram > used ? vram - used : 0;
    };
    const size_t target = park_prefill_free_bytes_ ? park_prefill_free_bytes_ : park_stream_needs(ctx_tokens);
    g.wait_idle();
    g.evict_unpinned_weights();
    if (free_now() >= target) return;                 // the prompt speed-up already has its room: nothing moves
    int n = 0; size_t b = 0;
    for (int l = (int)park_state_.size() - 1; l >= 0; --l) {
        if (park_state_[l] != 2) continue;
        if (free_now() >= target) break;
        for (const void* w : park_gpu_w_[l]) { g.unpin_weight_public(w); g.evict_weight_public(w); }
        park_gpu_w_[l].clear();
        park_state_[l] = 1;
        layer_weights_[l].gpu_layer = false;
        ++n; b += park_layer_bytes(l);
    }
    g.wait_idle();
    int stay = 0; for (auto v : park_state_) if (v == 2) ++stay;
    fprintf(stderr, "[park] prompt of %d tokens: the prompt speed-up needs %zu MB of the card%s; %d parked layers (%zu MB) moved "
                    "off it, %zu MB is free now; %d parked layers stay on the card and are read there. The moved ones are read "
                    "from the model file while the prompt is read, then go back onto the card.\n",
            n_tokens, target >> 20, park_prefill_free_bytes_ ? " (your setting)" : " (auto)", n, b >> 20, free_now() >> 20, stay);
    fflush(stderr);
#else
    (void)n_tokens;
#endif
}

// While a long prompt is read: if free RAM is under the threshold, drop layers' RAM copies -- layers this prompt is done
// with first (it never needs them again), then layers it has not reached yet (read from the model file when their turn
// comes). Dropped layers go onto the card after the prompt.
inline void Transformer::park_relieve_ram(int done_upto, int range_start, int range_end) {
#if defined(__linux__)
    if (!park_on_ || park_below_bytes_ <= 0) return;
    long long av = nexus::memguard::available_bytes();
    if (av < 0 || av >= park_below_bytes_) return;
    while (av < park_below_bytes_) {
        int pick = -1;
        for (int l = range_start; l <= done_upto && l < range_end; ++l)
            if (park_state_[l] == 0 && !park_ranges_[l].empty()) { pick = l; break; }
        if (pick < 0)
            for (int l = range_end - 1; l > done_upto && l >= range_start; --l)
                if (park_state_[l] == 0 && !park_ranges_[l].empty()) { pick = l; break; }
        if (pick < 0) {
            if (!park_relieve_exhausted_)
                fprintf(stderr, "[park] free RAM is %lld MB, under %lld MB, but every layer that can leave RAM already has\n",
                        av >> 20, park_below_bytes_ >> 20);
            park_relieve_exhausted_ = true;
            break;
        }
        park_drop_layer(pick);
        park_state_[pick] = 1;
        const long long av2 = nexus::memguard::available_bytes();
        fprintf(stderr, "[park] free RAM %lld MB is under %lld MB: layer %d (%zu MB) dropped from RAM -- %s; it goes onto the "
                        "card after the prompt (free RAM now %lld MB)\n",
                av >> 20, park_below_bytes_ >> 20, pick, park_layer_bytes(pick) >> 20,
                pick <= done_upto ? "this prompt is done with it" : "it is read from the model file when its turn comes",
                av2 >> 20);
        fflush(stderr);
        if (av2 < 0) break;
        av = av2;
    }
#else
    (void)done_upto; (void)range_start; (void)range_end;
#endif
}

// After a prompt, once its buffers are freed: dropped layers back into RAM (as RAM allows), parked layers onto the card,
// and whatever the card can't take into RAM. Nothing stays on the drive while writing.
// THE RULE: the model file (the drive) is a home for a layer only WHILE A PROMPT IS READ. Before writing starts,
// every layer is on the card or in RAM -- never read from the drive while writing. Called after every prompt and,
// as a safety net, before the first token of every reply (so a stopped or failed prompt read is covered too).
inline void Transformer::park_finish_pending() {
    if (!park_on_) return;
    bool any_off = false;
    for (auto v : park_state_) if (v == 1) { any_off = true; break; }
    if (!any_off) { park_pending_ = false; return; }
    park_pending_ = false;
    park_relieve_exhausted_ = false;
#if defined(__linux__)
    // The prompt's buffers are gone now, so RAM is looked at again: layers protection dropped while reading go back
    // into RAM first, as far as RAM allows with room left for the reply; only the rest go onto the card.
    {
        const long long floor = park_ram_floor();
        int back = 0; size_t back_b = 0;
        for (int l = 0; l < (int)park_state_.size(); ++l) {
            if (park_state_[l] != 1 || park_home_card_[l]) continue;
            const long long av = nexus::memguard::available_bytes();
            const size_t b = park_layer_bytes(l);
            if (av < 0 || av - (long long)b <= floor) break;
            park_touch_layer(l); park_state_[l] = 0; ++back; back_b += b;
        }
        if (back)
            fprintf(stderr, "[park] prompt read; RAM has room again: %d dropped layers (%zu MB) back into RAM (free RAM now %lld MB, keeping %lld MB for the reply)\n",
                    back, back_b >> 20, nexus::memguard::available_bytes() >> 20, floor >> 20);
    }
#endif
    park_to_gpu("prompt read");
#if defined(__linux__)
    // Everything the card couldn't take goes into RAM. No floor and no drive fallback: with the memory safety
    // buffer off, running out of RAM is the user's call to make (the safety buffer, when on, stops a reply first).
    int back = 0; size_t back_b = 0;
    for (int l = 0; l < (int)park_state_.size(); ++l) {
        if (park_state_[l] != 1) continue;
        park_touch_layer(l); park_state_[l] = 0; ++back; back_b += park_layer_bytes(l);
    }
    if (back) fprintf(stderr, "[park] %d layers (%zu MB) did not fit on the card and are back in RAM (free RAM now %lld MB)\n",
                      back, back_b >> 20, nexus::memguard::available_bytes() >> 20);
    fflush(stderr);
#endif
}




// load_streamed_shard: receiver-side distributed load without fake GGUF staging.
// The primary sends a header-only GGUF (metadata + tensor descriptors) followed by
// raw tensor bytes in deterministic name order. We keep only the small header plus
// one final allocation per tensor. No memfd, no temp file, no full shard blob.
inline bool Transformer::load_streamed_shard(const void* gguf_header,
                                             size_t gguf_header_size,
                                             int n_threads,
                                             std::function<bool(void*, size_t)> read_fn,
                                             std::function<void(const std::string&, float)> cb)
{
    n_threads_ = n_threads;
    loaded_ = false;
    streamed_weight_storage_.clear();

    if (!gguf_header || gguf_header_size < 32 || !read_fn) {
        fprintf(stderr, "[stream_load] invalid streamed shard header/read_fn\n");
        return false;
    }

    if (cb) cb("Opening streamed shard header", 0.0f);

    MemoryManager::Config mm_cfg;
    mm_cfg.prefer_huge_pages  = true;
    mm_cfg.lock_active_layers = true;
    mm_cfg.max_locked_layers  = std::min(n_threads * 2, 8);
    mm_ = std::make_unique<MemoryManager>(mm_cfg);
    mm_->on_pressure([this](size_t avail) {
        fprintf(stderr, "[nexus] Memory pressure! %s available. Evicting KV cache pages.\n",
                MemoryManager::fmt_bytes(avail).c_str());
        kvcache_.evict_oldest(128);
    });

    parser_ = std::make_unique<gguf::Parser>();
    if (!parser_->open_from_memory(gguf_header, gguf_header_size)) {
        fprintf(stderr, "[stream_load] Failed to parse streamed shard header (%zu bytes)\n",
                gguf_header_size);
        return false;
    }

    meta_ = parser_->meta();

    // *** v27 fix: this used to only recognize the old "blk.N." naming when inferring
    // which layer IDs are present. Qwen3Next/Qwen3.6's native GGUF export never uses
    // "blk." at all ("model.language_model.layers.N.xxx" instead), so this loop found
    // nothing for that family, left active_layer_end_ at its -1 sentinel, and every
    // layer-range-dependent step downstream (build_layer_weights(),
    // init_kv_cache_for_model()) worked from a wrong, open-ended range. Same mismatch
    // patch_engine_v27.py fixes on the sending side; nx_layer_of_tensor() already
    // understands every naming scheme this codebase supports (load_range_direct()
    // elsewhere in this file already uses it this same way).
    // Infer which original layer IDs are present in this streamed shard.
    active_layer_start_ = 0;
    active_layer_end_   = -1;
    {
        int min_l = INT_MAX, max_l = -1;
        for (const auto& kv : parser_->tensors()) {
            const std::string& name = kv.first;
            const int L = ::nexus::gguf::nx_layer_of_tensor(name);
            if (L >= 0) {
                min_l = std::min(min_l, L);
                max_l = std::max(max_l, L);
            }
        }
        if (max_l >= min_l && min_l != INT_MAX) {
            active_layer_start_ = min_l;
            active_layer_end_   = max_l + 1;
        }
    }

    if (meta_.n_layers == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[stream_load] bad streamed metadata: n_layers=%u hidden=%u\n",
                meta_.n_layers, meta_.hidden_dim);
        return false;
    }

    if (cb) cb("Loading tokenizer", 0.05f);
    if (!load_tokenizer()) {
        fprintf(stderr, "[nexus] Warning: tokenizer load failed for streamed shard\n");
    }

    if (!vocab_tokens_.empty() && vocab_tokens_.size() != meta_.vocab_size) {
        fprintf(stderr,
                "[nexus] Vocab metadata corrected from %u to %zu based on tokenizer tokens\n",
                meta_.vocab_size, vocab_tokens_.size());
        meta_.vocab_size = (uint32_t)vocab_tokens_.size();
    }

    auto& tensors = const_cast<std::unordered_map<std::string, TensorDesc>&>(parser_->tensors());
    std::vector<TensorDesc*> ordered;
    ordered.reserve(tensors.size());
    uint64_t total_bytes = 0;
    for (auto& kv : tensors) {
        ordered.push_back(&kv.second);
        total_bytes += (uint64_t)kv.second.data_size;
    }
    std::sort(ordered.begin(), ordered.end(), [](const TensorDesc* a, const TensorDesc* b) {
        return a->name < b->name;
    });

    fprintf(stderr,
            "[stream_load] Receiving %zu tensors, %llu MB total, directly into final owned weight buffers...\n",
            ordered.size(), (unsigned long long)(total_bytes / (1024ULL * 1024ULL)));
    fflush(stderr);

    streamed_weight_storage_.reserve(ordered.size());
    uint64_t done = 0;
    size_t idx = 0;
    for (TensorDesc* td : ordered) {
        const size_t sz = td->data_size;
        std::unique_ptr<uint8_t[]> mem;
        if (sz > 0) {
            mem.reset(new (std::nothrow) uint8_t[sz]);
            if (!mem) {
                fprintf(stderr, "[stream_load] allocation failed for %s (%zu bytes)\n",
                        td->name.c_str(), sz);
                return false;
            }
            if (!read_fn(mem.get(), sz)) {
                fprintf(stderr, "[stream_load] socket read failed for %s (%zu bytes)\n",
                        td->name.c_str(), sz);
                return false;
            }
            td->data = mem.get();
        } else {
            td->data = nullptr;
        }
        streamed_weight_storage_.push_back(std::move(mem));
        done += (uint64_t)sz;
        ++idx;

        float frac = total_bytes ? (0.1f + 0.8f * ((float)done / (float)total_bytes)) : 0.9f;
        if (cb) cb("Receiving: " + td->name, frac);
        fprintf(stderr, "[stream_recv] [%3zu/%3zu] %.0f%%  %s (%zu MB)\n",
                idx, ordered.size(), total_bytes ? 100.0 * (double)done / (double)total_bytes : 100.0,
                td->name.c_str(), sz / (1024ULL * 1024ULL));
        fflush(stderr);
    }

    if (cb) cb("Building layer index", 0.92f);
    if (!build_layer_weights()) {
        fprintf(stderr, "[stream_load] failed to index streamed shard weights\n");
        return false;
    }

    if (meta_.head_dim == 0 || meta_.hidden_dim == 0) {
        fprintf(stderr, "[stream_load] FATAL: head_dim=%u hidden_dim=%u\n",
                meta_.head_dim, meta_.hidden_dim);
        return false;
    }

    attn_cfg_.n_heads    = (int)meta_.n_heads;
    attn_cfg_.n_kv_heads = (int)meta_.n_kv_heads;
    attn_cfg_.head_dim   = (int)meta_.head_dim;
    attn_cfg_.hidden_dim = (int)meta_.hidden_dim;
    attn_cfg_.rms_norm_eps = meta_.rms_norm_eps;
    attn_cfg_.rope.freq_base = meta_.rope_freq_base;
    attn_cfg_.n_threads  = n_threads;
    attn_cfg_.rope.n_ctx_orig = (meta_.context_length > 0) ? meta_.context_length : 4096;

    if (meta_.rope_freq_base <= 10001.0f &&
        meta_.context_length > 8192 &&
        meta_.head_dim > 2)
    {
        float ctx_ratio = (float)meta_.context_length / (float)attn_cfg_.rope.n_ctx_orig;
        if (ctx_ratio > 1.0f) {
            float exponent = (float)meta_.head_dim / (float)(meta_.head_dim - 2);
            attn_cfg_.rope.freq_base *= powf(ctx_ratio, exponent);
        }
    }

    fprintf(stderr, "[nexus] RoPE: base=%.0f  type=%d  n_ctx_orig=%u  context=%u\n",
            attn_cfg_.rope.freq_base,
            attn_cfg_.rope.type,
            attn_cfg_.rope.n_ctx_orig,
            meta_.context_length);
    fflush(stderr);

    init_kv_cache_for_model();

    const int hidden = (int)meta_.hidden_dim;
    const int ffn    = (int)meta_.ffn_dim;
    const int q_dim  = (int)(meta_.n_heads * meta_.head_dim);
    const int kv_dim = (int)(meta_.n_kv_heads * meta_.head_dim);
    const int max_ctx = (int)meta_.context_length;
    const size_t attn_scratch_sz = q_dim + 2*kv_dim + max_ctx + q_dim + 1024;

    x_scratch_.resize(hidden, 0.0f);
    xb_scratch_.resize(hidden, 0.0f);
    attn_scratch_.resize(attn_scratch_sz, 0.0f);
    ffn_scratch_.resize(std::max((size_t)hidden, (size_t)ffn * 2), 0.0f);
    qkv_scratch_.resize(q_dim + 2 * kv_dim, 0.0f);
    logits_.resize(meta_.vocab_size, 0.0f);

    if (cb) cb("Ready", 1.0f);
    fprintf(stderr, "\n[nexus] Streamed shard loaded directly: preserving source layer IDs %d-%d of full %u, no fake GGUF blob.\n",
            active_layer_start_, active_layer_end_ - 1, (unsigned)meta_.n_layers);
    fflush(stderr);
    if (mm_) mm_->print_report();
    loaded_ = true;
    return true;
}

} // namespace model
} // namespace nexus
