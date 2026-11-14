/*
 * nexus/src/model/vision_encoder.h
 *
 * Vision encoder for multimodal (image+text) models.
 * Handles CLIP/SigLIP encoders used by LLaVA, BakLLaVA, Moondream, etc.
 *
 * Architecture:
 *   1. Patch embedding: image → fixed-size patches → linear projection
 *   2. Vision transformer: N layers of self-attention over patch tokens
 *   3. Projection: vision token dim → LLM hidden dim (MLP or linear)
 *   4. Integration: vision tokens injected into LLM token stream at <image> position
 *
 * Supported model families:
 *   LLaVA-1.5/1.6 (CLIP ViT-L/14 + MLP projector)
 *   BakLLaVA      (CLIP ViT-L/14 + Mistral backbone)
 *   Moondream2    (SigLIP encoder + Phi-2 backbone)
 *   LLaVA-Next    (AnyRes tiling + CLIP)
 *
 * GGUF tensor names follow llama.cpp convention so existing GGUF files load.
 *
 * Usage:
 *   VisionEncoder enc;
 *   enc.load("model-mmproj.gguf");
 *   std::vector<float> img_tokens = enc.encode(rgb_pixels, width, height);
 *   // img_tokens is now a flat [n_patches * hidden_dim] float array
 *   // Pass to engine.inject_vision_tokens(img_tokens) before generating
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <cstring>

#include "../gguf/parser.h"
#include "../gguf/quant.h"
#include "../cpu/matmul.h"
#include "../core/memory.h"
#ifdef NEXUS_VULKAN
#include "../gpu/vulkan_backend.h"
#endif

namespace nexus {
namespace vision {


inline int nx_vision_env_int(const char* name, int defv, int lo, int hi) {
    const char* v = std::getenv(name);
    if (!v || !*v) return defv;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return defv;
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return (int)x;
}



inline uint16_t nx_vision_f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007fffffu;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x00001000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x00001000u) >> 13));
}
template <typename Fn>
inline void nx_vision_parallel_for(int64_t n, int n_threads, const Fn& fn) {
    if (n <= 0) return;
    if (n_threads <= 1 || n < 2) {
        fn((int64_t)0, n);
        return;
    }
    const int nt = std::max(1, std::min((int)n, n_threads));
    std::vector<std::thread> workers;
    workers.reserve((size_t)nt);
    for (int t = 0; t < nt; ++t) {
        const int64_t b = (n * (int64_t)t) / nt;
        const int64_t e = (n * (int64_t)(t + 1)) / nt;
        workers.emplace_back([b, e, &fn]() { fn(b, e); });
    }
    for (auto& th : workers) th.join();
}

// ── Image preprocessing ───────────────────────────────────────────────────────

struct ImagePreprocessConfig {
    int    image_size   = 336;   // ViT input size (224 for standard CLIP, 336 for LLaVA-1.5)
    int    patch_size   = 14;    // patch size in pixels
    float  mean[3]      = {0.48145466f, 0.4578275f,  0.40821073f};  // CLIP normalization
    float  std_dev[3]   = {0.26862954f, 0.26130258f, 0.27577711f};
    bool   any_res      = false; // LLaVA-Next: tile into multiple crops + thumbnail
    int    max_tiles    = 6;
};

// Resize + center-crop + normalize RGB image into flat float tensor.
// Input: rgb_bytes = raw RGB pixels [height*width*3]
// Output: flat NCHW tensor [3 * image_size * image_size] normalized floats
inline std::vector<float> preprocess_image(
    const uint8_t* rgb_bytes, int width, int height,
    const ImagePreprocessConfig& cfg)
{
    const int sz = cfg.image_size;
    const int out_pixels = sz * sz;

    // ── Resize with bilinear interpolation ───────────────────────────────────
    // Scale so shorter side = image_size, then center crop
    float scale = std::max((float)sz / width, (float)sz / height);
    int   rw = (int)std::round(width  * scale);
    int   rh = (int)std::round(height * scale);

    std::vector<float> resized(rw * rh * 3);
    for (int y = 0; y < rh; ++y) {
        for (int x = 0; x < rw; ++x) {
            float src_x = (x + 0.5f) / scale - 0.5f;
            float src_y = (y + 0.5f) / scale - 0.5f;
            int x0 = std::max(0, (int)src_x);
            int y0 = std::max(0, (int)src_y);
            int x1 = std::min(width  - 1, x0 + 1);
            int y1 = std::min(height - 1, y0 + 1);
            float dx = src_x - x0;
            float dy = src_y - y0;
            for (int c = 0; c < 3; ++c) {
                float tl = rgb_bytes[(y0 * width + x0) * 3 + c];
                float tr = rgb_bytes[(y0 * width + x1) * 3 + c];
                float bl = rgb_bytes[(y1 * width + x0) * 3 + c];
                float br = rgb_bytes[(y1 * width + x1) * 3 + c];
                resized[(y * rw + x) * 3 + c] =
                    tl * (1-dx) * (1-dy) + tr * dx * (1-dy) +
                    bl * (1-dx) * dy     + br * dx * dy;
            }
        }
    }

    // ── Center crop to sz×sz ─────────────────────────────────────────────────
    int cx = (rw - sz) / 2;
    int cy = (rh - sz) / 2;

    // ── Normalize to NCHW, apply CLIP mean/std ───────────────────────────────
    std::vector<float> tensor(3 * out_pixels);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < sz; ++y) {
            for (int x = 0; x < sz; ++x) {
                float v = resized[((cy + y) * rw + (cx + x)) * 3 + c];
                tensor[c * out_pixels + y * sz + x] =
                    (v / 255.0f - cfg.mean[c]) / cfg.std_dev[c];
            }
        }
    }
    return tensor;
}

// AnyRes: tile image into up to max_tiles crops + 1 thumbnail
// Returns [n_tiles] preprocessed tensors
inline std::vector<std::vector<float>> preprocess_anyres(
    const uint8_t* rgb_bytes, int width, int height,
    const ImagePreprocessConfig& cfg)
{
    std::vector<std::vector<float>> tiles;

    // Thumbnail: always include full image resized to image_size
    tiles.push_back(preprocess_image(rgb_bytes, width, height, cfg));

    // Determine tile grid: find best fit within max_tiles
    const int sz = cfg.image_size;
    int best_tiles = 1;
    int best_gw = 1, best_gh = 1;
    for (int gh = 1; gh <= cfg.max_tiles; ++gh) {
        for (int gw = 1; gw <= cfg.max_tiles; ++gw) {
            if (gw * gh > cfg.max_tiles) continue;
            float ratio = std::min((float)(gw * sz) / width,
                                   (float)(gh * sz) / height);
            if (ratio < 1.0f) continue; // tile is smaller than image — skip
            if (gw * gh > best_tiles) {
                best_tiles = gw * gh; best_gw = gw; best_gh = gh;
            }
        }
    }

    // Extract each tile
    for (int ty = 0; ty < best_gh; ++ty) {
        for (int tx = 0; tx < best_gw; ++tx) {
            // Crop region
            int x0 = tx * width  / best_gw;
            int y0 = ty * height / best_gh;
            int x1 = (tx + 1) * width  / best_gw;
            int y1 = (ty + 1) * height / best_gh;
            int tw = x1 - x0, th = y1 - y0;

            // Extract tile pixels
            std::vector<uint8_t> tile(tw * th * 3);
            for (int y = 0; y < th; ++y)
                for (int x = 0; x < tw; ++x)
                    for (int c = 0; c < 3; ++c)
                        tile[(y * tw + x) * 3 + c] =
                            rgb_bytes[((y0 + y) * width + (x0 + x)) * 3 + c];

            tiles.push_back(preprocess_image(tile.data(), tw, th, cfg));
        }
    }
    return tiles;
}

// ── Vision Transformer weights ────────────────────────────────────────────────

struct VisionLayerWeights {
    // Layer norm
    const float* ln1_w = nullptr;  // [d_model]
    const float* ln1_b = nullptr;
    const float* ln2_w = nullptr;
    const float* ln2_b = nullptr;
    // Self-attention
    const float* wq = nullptr;    // [d_model, d_model]
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;
    const float* bq = nullptr;    // [d_model]
    const float* bk = nullptr;
    const float* bv = nullptr;
    const float* bo = nullptr;
    // FFN
    const float* fc1_w = nullptr; // [d_ffn, d_model]
    const float* fc1_b = nullptr;
    const float* fc2_w = nullptr; // [d_model, d_ffn]
    const float* fc2_b = nullptr;
};

struct VisionConfig {
    int image_size       = 336;
    int patch_size       = 14;
    int d_model          = 1024;   // ViT-L hidden dim
    int n_heads          = 16;
    int n_layers         = 24;     // ViT-L depth
    int d_ffn            = 4096;
    int d_proj_in        = 1024;   // projector input (= d_model)
    int d_proj_out       = 4096;   // projector output (= LLM hidden dim)
    bool has_cls_token   = true;
    bool use_cls_only    = false;  // if true, use only CLS token; else all patches
    std::string proj_type = "mlp"; // "mlp", "linear", "resampler"
    ImagePreprocessConfig preproc;
};

// ── Vision Encoder ────────────────────────────────────────────────────────────

class VisionEncoder {
public:
    VisionEncoder() = default;
    ~VisionEncoder() = default;

    // Load vision encoder weights. Supports legacy external CLIP/SigLIP mmproj GGUF
    // and native Qwen3Next/Qwen3.6 visual tensors embedded inside a .nxm model.
    // Returns false when this file has no usable visual encoder.
    bool load(const std::string& mmproj_path,
              std::function<void(const std::string&, float)> progress_cb = nullptr);

    // Bind native Qwen3Next/Qwen3.6 visual tensors that are already loaded by
    // the main Transformer. This does NOT open/mmap the .nxm again and does NOT
    // duplicate/dequantize the visual tower. It only stores pointers to the
    // existing TensorDesc::data buffers.
    bool bind_native_qwen3next_from_loaded_tensors(
              const std::unordered_map<std::string, gguf::TensorDesc>& tensors,
              std::function<void(const std::string&, float)> progress_cb = nullptr);

    // Returns true if loaded successfully
    bool loaded() const { return loaded_; }

    // Get config (available after load)
    const VisionConfig& config() const { return cfg_; }

    // Encode raw RGB image into vision tokens [n_tokens * d_proj_out].
    // n_tokens = n_patches for non-CLS, 1 for CLS-only.
    // With AnyRes: n_tokens = n_tiles * n_patches (tiles concatenated).
    std::vector<float> encode(const uint8_t* rgb_bytes, int width, int height);

    // Convenience: encode from float NCHW tensor (already preprocessed)
    std::vector<float> encode_tensor(const std::vector<float>& nchw_tensor);

    // Number of vision tokens produced per image (for context budgeting)
    int tokens_per_image() const;

private:
    bool loaded_ = false;
    bool native_qwen3next_ = false;
    int  native_grid_ = 0;
    VisionConfig cfg_;

    struct NativeTensorRef {
        const uint8_t* data = nullptr;
        size_t n_elements = 0;
        std::vector<int64_t> shape;
        gguf::GGMLType ggml_type = gguf::GGMLType::F32;
        quant::QuantType qtype = quant::QuantType::F32;
        bool valid() const { return data != nullptr && n_elements > 0; }
        size_t rows() const { return shape.empty() ? 0 : (size_t)shape[0]; }
        size_t cols() const { size_t r = rows(); return r ? (n_elements / r) : 0; }
    };
    std::unique_ptr<gguf::Parser> native_parser_;   // keeps the .nxm mmap alive; no tensor copy
    std::unordered_map<std::string, NativeTensorRef> native_refs_;

    // Per-encode vision GPU accounting. Reset at the start of each native encode.
    mutable uint64_t vision_mm_calls_ = 0;
    mutable uint64_t vision_mm_gpu_ = 0;
    mutable uint64_t vision_mm_cpu_ = 0;
    mutable uint64_t vision_mm_no_gpu_ready_ = 0;
    mutable uint64_t vision_mm_non_tight_ = 0;
    mutable uint64_t vision_mm_gpu_fail_ = 0;
    mutable uint64_t vision_mm_disabled_ = 0;
    mutable uint64_t vision_mm_float_pipe_missing_ = 0;

    bool load_native_qwen3next(const std::string& model_path,
              std::function<void(const std::string&, float)> progress_cb = nullptr);
    const NativeTensorRef* native_tensor(const std::string& name) const;
    std::vector<float> native_tensor_f32(const std::string& name) const;
    bool native_matvec(const std::string& name, const float* x, float* y, int n_threads = 1) const;
    bool native_matmul_many(const std::string& name, const float* x, size_t n_vecs, size_t in_stride, float* y, size_t out_stride, int n_threads) const;
    std::vector<float> forward_native_qwen3next(const std::vector<float>& nchw);
    std::vector<float> project_native_qwen3next(const std::vector<float>& vision_features);

    // Legacy external encoder weights are still stored as float arrays. Native .nxm visual
    // tensors stay quantized/mmap-backed in native_refs_ and are never duplicated here.
    std::vector<float> native_merger_norm_w_, native_merger_norm_b_;
    std::vector<float> patch_embed_w_;  // [d_model, 3 * patch_size * patch_size]
    std::vector<float> patch_embed_b_;  // [d_model]
    std::vector<float> cls_token_;      // [d_model] (if has_cls_token)
    std::vector<float> pos_embed_;      // [n_positions, d_model]
    std::vector<float> pre_ln_w_, pre_ln_b_;   // pre-norm (some models)
    std::vector<float> post_ln_w_, post_ln_b_; // post-norm (CLIP standard)
    // Projector weights
    std::vector<float> proj_w1_, proj_b1_;  // MLP layer 1
    std::vector<float> proj_w2_, proj_b2_;  // MLP layer 2 (if mlp)
    // Per-layer weights
    std::vector<VisionLayerWeights> layer_weights_;
    std::vector<std::vector<float>> layer_storage_; // backing storage

    // Forward pass
    std::vector<float> forward(const std::vector<float>& nchw);
    std::vector<float> project(const std::vector<float>& vision_features);

    static void layer_norm(float* x, const float* w, const float* b, int n, float eps = 1e-6f);
    static void gelu(float* x, int n);
    static void softmax_rows(float* x, int rows, int cols);
};

// ── Implementations ───────────────────────────────────────────────────────────


// ── Native Qwen3Next/Qwen3.6 visual tensor helpers ───────────────────────────

static inline quant::QuantType nx_vision_qtype_from_ggml(gguf::GGMLType t) {
    switch (t) {
        case gguf::GGMLType::F32:  return quant::QuantType::F32;
        case gguf::GGMLType::F16:  return quant::QuantType::F16;
        case gguf::GGMLType::BF16: return quant::QuantType::BF16;
        case gguf::GGMLType::Q4_0: return quant::QuantType::Q4_0;
        case gguf::GGMLType::Q5_0: return quant::QuantType::Q5_0;
        case gguf::GGMLType::Q8_0: return quant::QuantType::Q8_0;
        case gguf::GGMLType::Q4_K: return quant::QuantType::Q4_K;
        case gguf::GGMLType::Q5_K: return quant::QuantType::Q5_K;
        case gguf::GGMLType::Q6_K: return quant::QuantType::Q6_K;
        case gguf::GGMLType::NQ4_G64: return quant::QuantType::NQ4_G64;
        case gguf::GGMLType::NQ5_G64: return quant::QuantType::NQ5_G64;
        case gguf::GGMLType::NQ6_G64: return quant::QuantType::NQ6_G64;
        case gguf::GGMLType::NQ8_G64: return quant::QuantType::NQ8_G64;
        default: return quant::QuantType::F32;
    }
}

static inline bool nx_env_truthy_vision(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && v[0] != '0' && v[0] != 'n' && v[0] != 'N' && v[0] != 'f' && v[0] != 'F';
}

static inline const char* nx_vision_qtype_name(quant::QuantType qt) {
    switch (qt) {
    case quant::QuantType::F32: return "F32";
    case quant::QuantType::F16: return "F16";
    case quant::QuantType::BF16: return "BF16";
    case quant::QuantType::Q8_0: return "Q8_0";
    case quant::QuantType::Q4_K: return "Q4_K";
    case quant::QuantType::Q5_K: return "Q5_K";
    case quant::QuantType::Q6_K: return "Q6_K";
    case quant::QuantType::NQ4_G64: return "NQ4_G64";
    case quant::QuantType::NQ5_G64: return "NQ5_G64";
    case quant::QuantType::NQ6_G64: return "NQ6_G64";
    case quant::QuantType::NQ8_G64: return "NQ8_G64";
    default: return "unknown";
    }
}

static inline const char* nx_vision_gpu_fallback_hint(quant::QuantType qt, bool tight) {
    if (!tight) return "non_tight_strides";
    switch (qt) {
    case quant::QuantType::F32:
    case quant::QuantType::F16:
    case quant::QuantType::BF16:
        return "no_float_or_bf16_batched_mm_pipeline_yet";
    default:
        return "gpu_mat_mat_returned_false";
    }
}


inline const VisionEncoder::NativeTensorRef* VisionEncoder::native_tensor(const std::string& name) const {
    auto it = native_refs_.find(name);
    return it == native_refs_.end() ? nullptr : &it->second;
}

inline std::vector<float> VisionEncoder::native_tensor_f32(const std::string& name) const {
    std::vector<float> out;
    const NativeTensorRef* t = native_tensor(name);
    if (!t || !t->valid()) return out;
    out.resize(t->n_elements);
    try { quant::dequantize(t->data, out.data(), t->n_elements, t->qtype); }
    catch (...) { out.clear(); }
    return out;
}

inline bool VisionEncoder::native_matvec(const std::string& name, const float* x, float* y, int n_threads) const {
    const NativeTensorRef* t = native_tensor(name);
    if (!t || !t->valid() || t->rows() == 0 || t->cols() == 0) return false;
    cpu::mat_vec(t->data, x, y, t->rows(), t->cols(), t->qtype, n_threads > 0 ? n_threads : 1);
    return true;
}

inline bool VisionEncoder::native_matmul_many(const std::string& name, const float* x, size_t n_vecs, size_t in_stride, float* y, size_t out_stride, int n_threads) const {
    const NativeTensorRef* t = native_tensor(name);
    if (!t || !t->valid() || t->rows() == 0 || t->cols() == 0 || !x || !y) return false;
    const size_t rows = t->rows();
    const size_t cols = t->cols();
    if (in_stride < cols || out_stride < rows) return false;

    const bool want_gpu = nx_env_truthy_vision("NEXUS_VISION_GPU");
    const bool trace_gpu = nx_env_truthy_vision("NEXUS_VISION_GPU_TRACE");
    const bool tight = (in_stride == cols && out_stride == rows);
    vision_mm_calls_++;

#ifdef NEXUS_VULKAN
    if (want_gpu) {
        if (!::nexus::gpu::gpu().ready()) {
            vision_mm_no_gpu_ready_++;
            if (trace_gpu) {
                fprintf(stderr, "[vision-gpu:mm] %s CPU-NOREADY rows=%zu cols=%zu ntok=%zu type=%s(%d)\n",
                        name.c_str(), rows, cols, n_vecs, nx_vision_qtype_name(t->qtype), (int)t->qtype);
            }
        } else if (!tight) {
            vision_mm_non_tight_++;
            if (trace_gpu) {
                fprintf(stderr, "[vision-gpu:mm] %s CPU-NONTIGHT rows=%zu cols=%zu ntok=%zu in_stride=%zu out_stride=%zu type=%s(%d)\n",
                        name.c_str(), rows, cols, n_vecs, in_stride, out_stride, nx_vision_qtype_name(t->qtype), (int)t->qtype);
            }
        } else {
            if (::nexus::gpu::gpu().mat_mat(t->data, x, y, rows, cols, n_vecs, t->qtype)) {
                vision_mm_gpu_++;
                if (trace_gpu) {
                    fprintf(stderr, "[vision-gpu:mm] %s GPU rows=%zu cols=%zu ntok=%zu type=%s(%d)\n",
                            name.c_str(), rows, cols, n_vecs, nx_vision_qtype_name(t->qtype), (int)t->qtype);
                }
                return true;
            }
            vision_mm_gpu_fail_++;
            if (t->qtype == quant::QuantType::F32 || t->qtype == quant::QuantType::F16 || t->qtype == quant::QuantType::BF16) {
                vision_mm_float_pipe_missing_++;
            }
            if (trace_gpu) {
                fprintf(stderr, "[vision-gpu:mm] %s CPU-GPUFAIL reason=%s rows=%zu cols=%zu ntok=%zu type=%s(%d)\n",
                        name.c_str(), nx_vision_gpu_fallback_hint(t->qtype, tight), rows, cols, n_vecs,
                        nx_vision_qtype_name(t->qtype), (int)t->qtype);
            }
        }
    } else {
        vision_mm_disabled_++;
    }
#else
    if (want_gpu) {
        vision_mm_no_gpu_ready_++;
        if (trace_gpu) {
            fprintf(stderr, "[vision-gpu:mm] %s CPU-NOVULKAN rows=%zu cols=%zu ntok=%zu type=%s(%d)\n",
                    name.c_str(), rows, cols, n_vecs, nx_vision_qtype_name(t->qtype), (int)t->qtype);
        }
    } else {
        vision_mm_disabled_++;
    }
#endif

    vision_mm_cpu_++;
    const int64_t total = (int64_t)n_vecs * (int64_t)rows;
    nx_vision_parallel_for(total, n_threads > 0 ? n_threads : 1, [&](int64_t b, int64_t e) {
        for (int64_t idx = b; idx < e; ++idx) {
            const size_t vi = (size_t)(idx / (int64_t)rows);
            const size_t row = (size_t)(idx - (int64_t)vi * (int64_t)rows);
            y[vi * out_stride + row] = cpu::dot_row(t->data, row, cols, x + vi * in_stride, t->qtype);
        }
    });
    return true;
}


inline bool VisionEncoder::bind_native_qwen3next_from_loaded_tensors(
    const std::unordered_map<std::string, gguf::TensorDesc>& tensors,
    std::function<void(const std::string&, float)> progress_cb)
{
    // Never open or mmap the .nxm here. The language model has already loaded
    // every tensor once; vision only borrows those TensorDesc::data pointers.
    auto find_exact = [&](const std::string& name) -> const gguf::TensorDesc* {
        auto it = tensors.find(name);
        return it == tensors.end() ? nullptr : &it->second;
    };

    if (!find_exact("model.visual.patch_embed.proj.weight")) {
        return false;
    }

    native_parser_.reset();   // absolutely no second mmap/parser ownership
    native_refs_.clear();

    auto bind_tensor = [&](const std::string& name, bool required=false) -> bool {
        const gguf::TensorDesc* td = find_exact(name);
        if (!td) {
            if (required) fprintf(stderr, "[vision-native] missing loaded tensor: %s\n", name.c_str());
            return false;
        }
        if (!td->data) {
            if (required) fprintf(stderr, "[vision-native] loaded tensor has null data: %s\n", name.c_str());
            return false;
        }
        NativeTensorRef r;
        r.data = static_cast<const uint8_t*>(td->data);
        r.n_elements = (size_t)td->n_elements;
        if (!r.n_elements) {
            r.n_elements = 1;
            for (auto d : td->shape) r.n_elements *= (size_t)d;
        }
        r.shape.clear();
        r.shape.reserve(td->shape.size());
        for (auto d : td->shape) r.shape.push_back((int64_t)d);
        r.ggml_type = td->type;
        r.qtype = nx_vision_qtype_from_ggml(td->type);
        native_refs_[name] = std::move(r);
        return true;
    };

    int n_layers = 0;
    for (const auto& kv : tensors) {
        const std::string& n = kv.first;
        const std::string pfx = "model.visual.blocks.";
        if (n.rfind(pfx, 0) != 0) continue;
        size_t ss = pfx.size();
        size_t e = n.find('.', ss);
        if (e == std::string::npos) continue;
        int id = std::atoi(n.substr(ss, e - ss).c_str());
        if (id + 1 > n_layers) n_layers = id + 1;
    }
    if (n_layers <= 0) n_layers = 27;

    int image_size = 768; // native Qwen visual default; override with NEXUS_QWEN_NATIVE_IMAGE_SIZE
    if (const char* env = std::getenv("NEXUS_QWEN_NATIVE_IMAGE_SIZE")) {
        int v = std::atoi(env);
        if (v >= 64 && v <= 896) image_size = v;
    }
    image_size = (image_size / 32) * 32;
    if (image_size < 64) image_size = 448;

    native_qwen3next_ = true;
    cfg_ = VisionConfig{};
    cfg_.image_size = image_size;
    cfg_.patch_size = 16;
    cfg_.d_model    = 1152;
    cfg_.n_heads    = 16;
    cfg_.n_layers   = n_layers;
    cfg_.d_ffn      = 4304;
    cfg_.d_proj_in  = 1152;
    cfg_.d_proj_out = 5120;
    cfg_.has_cls_token = false;
    cfg_.use_cls_only  = false;
    cfg_.proj_type     = "qwen3next_native_merger";
    cfg_.preproc.image_size = image_size;
    cfg_.preproc.patch_size = 16;
    // Qwen3.6/Qwen3-VL image processor uses mean/std = 0.5, not CLIP mean/std.
    // Wrong CLIP normalization makes real images look like garbage/static to the visual tower.
    cfg_.preproc.mean[0] = cfg_.preproc.mean[1] = cfg_.preproc.mean[2] = 0.5f;
    cfg_.preproc.std_dev[0] = cfg_.preproc.std_dev[1] = cfg_.preproc.std_dev[2] = 0.5f;
    native_grid_ = image_size / 16;

    bool ok = true;
    ok &= bind_tensor("model.visual.patch_embed.proj.weight", true);
    bind_tensor("model.visual.patch_embed.proj.bias");
    bind_tensor("model.visual.pos_embed.weight");
    bind_tensor("model.visual.merger.norm.weight");
    bind_tensor("model.visual.merger.norm.bias");
    ok &= bind_tensor("model.visual.merger.linear_fc1.weight", true);
    bind_tensor("model.visual.merger.linear_fc1.bias");
    ok &= bind_tensor("model.visual.merger.linear_fc2.weight", true);
    bind_tensor("model.visual.merger.linear_fc2.bias");

    for (int i = 0; i < cfg_.n_layers; ++i) {
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";
        bind_tensor(p + "norm1.weight"); bind_tensor(p + "norm1.bias");
        bind_tensor(p + "norm2.weight"); bind_tensor(p + "norm2.bias");
        ok &= bind_tensor(p + "attn.qkv.weight", true);
        bind_tensor(p + "attn.qkv.bias");
        ok &= bind_tensor(p + "attn.proj.weight", true);
        bind_tensor(p + "attn.proj.bias");
        ok &= bind_tensor(p + "mlp.linear_fc1.weight", true);
        bind_tensor(p + "mlp.linear_fc1.bias");
        ok &= bind_tensor(p + "mlp.linear_fc2.weight", true);
        bind_tensor(p + "mlp.linear_fc2.bias");
        if (progress_cb) progress_cb("Binding already-loaded native Qwen visual tensor refs", (float)(i + 1) / (float)cfg_.n_layers);
    }

    if (!ok) {
        fprintf(stderr, "[vision-native] loaded visual tensor binding failed; not enabling vision\n");
        native_refs_.clear(); native_qwen3next_ = false; loaded_ = false;
        return false;
    }

    // No duplicate visual weights. These legacy vectors are external-encoder-only.
    patch_embed_w_.clear(); patch_embed_b_.clear(); pos_embed_.clear();
    native_merger_norm_w_.clear(); native_merger_norm_b_.clear();
    proj_w1_.clear(); proj_b1_.clear(); proj_w2_.clear(); proj_b2_.clear();
    layer_weights_.clear(); layer_storage_.clear();

    loaded_ = true;
    fprintf(stderr,
            "[vision-native] Bound Qwen3Next visual tensors from already-loaded model weights: image=%d patch=16 grid=%dx%d layers=%d d=%d -> hidden=%d tokens=%d refs=%zu (no second mmap, no dequant copy)\n",
            cfg_.image_size, native_grid_, native_grid_, cfg_.n_layers,
            cfg_.d_model, cfg_.d_proj_out, tokens_per_image(), native_refs_.size());
    return true;
}

inline bool VisionEncoder::load_native_qwen3next(
    const std::string& model_path,
    std::function<void(const std::string&, float)> progress_cb)
{
    native_parser_ = std::make_unique<gguf::Parser>();
    if (!native_parser_->open(model_path)) return false;

    auto find_exact = [&](const std::string& name) -> const gguf::TensorDesc* {
        const auto& map = native_parser_->tensors();
        auto it = map.find(name);
        return it == map.end() ? nullptr : &it->second;
    };

    if (!find_exact("model.visual.patch_embed.proj.weight")) {
        native_parser_.reset();
        return false;
    }

    const uint8_t* base = native_parser_->data_ptr();
    if (!base) {
        fprintf(stderr, "[vision-native] parser has visual tensors but no mmap data pointer\n");
        native_parser_.reset();
        return false;
    }

    native_refs_.clear();
    auto bind_tensor = [&](const std::string& name, bool required=false) -> bool {
        const gguf::TensorDesc* td = find_exact(name);
        if (!td) {
            if (required) fprintf(stderr, "[vision-native] missing tensor: %s\n", name.c_str());
            return false;
        }
        NativeTensorRef r;
        r.data = base + (size_t)td->offset;
        r.n_elements = (size_t)td->n_elements;
        if (!r.n_elements) { r.n_elements = 1; for (auto d : td->shape) r.n_elements *= (size_t)d; }
        r.shape.clear();
        r.shape.reserve(td->shape.size());
        for (auto d : td->shape) r.shape.push_back((int64_t)d);
        r.ggml_type = td->type;
        r.qtype = nx_vision_qtype_from_ggml(td->type);
        native_refs_[name] = std::move(r);
        return true;
    };

    int n_layers = 0;
    for (const auto& kv : native_parser_->tensors()) {
        const std::string& n = kv.first;
        const std::string pfx = "model.visual.blocks.";
        if (n.rfind(pfx, 0) != 0) continue;
        size_t s = pfx.size();
        size_t e = n.find('.', s);
        if (e == std::string::npos) continue;
        int id = std::atoi(n.substr(s, e - s).c_str());
        if (id + 1 > n_layers) n_layers = id + 1;
    }
    if (n_layers <= 0) n_layers = 27;

    int image_size = 768; // native Qwen visual default; override with NEXUS_QWEN_NATIVE_IMAGE_SIZE
    if (const char* env = std::getenv("NEXUS_QWEN_NATIVE_IMAGE_SIZE")) {
        int v = std::atoi(env);
        if (v >= 64 && v <= 896) image_size = v;
    }
    image_size = (image_size / 32) * 32;
    if (image_size < 64) image_size = 448;

    native_qwen3next_ = true;
    cfg_ = VisionConfig{};
    cfg_.image_size = image_size;
    cfg_.patch_size = 16;
    cfg_.d_model    = 1152;
    cfg_.n_heads    = 16;
    cfg_.n_layers   = n_layers;
    cfg_.d_ffn      = 4304;
    cfg_.d_proj_in  = 1152;
    cfg_.d_proj_out = 5120;
    cfg_.has_cls_token = false;
    cfg_.use_cls_only  = false;
    cfg_.proj_type     = "qwen3next_native_merger";
    cfg_.preproc.image_size = image_size;
    cfg_.preproc.patch_size = 16;
    // Qwen3.6/Qwen3-VL image processor uses mean/std = 0.5, not CLIP mean/std.
    // Wrong CLIP normalization makes real images look like garbage/static to the visual tower.
    cfg_.preproc.mean[0] = cfg_.preproc.mean[1] = cfg_.preproc.mean[2] = 0.5f;
    cfg_.preproc.std_dev[0] = cfg_.preproc.std_dev[1] = cfg_.preproc.std_dev[2] = 0.5f;
    native_grid_ = image_size / 16;

    bool ok = true;
    ok &= bind_tensor("model.visual.patch_embed.proj.weight", true);
    bind_tensor("model.visual.patch_embed.proj.bias");
    bind_tensor("model.visual.pos_embed.weight");
    bind_tensor("model.visual.merger.norm.weight");
    bind_tensor("model.visual.merger.norm.bias");
    ok &= bind_tensor("model.visual.merger.linear_fc1.weight", true);
    bind_tensor("model.visual.merger.linear_fc1.bias");
    ok &= bind_tensor("model.visual.merger.linear_fc2.weight", true);
    bind_tensor("model.visual.merger.linear_fc2.bias");

    for (int i = 0; i < cfg_.n_layers; ++i) {
        const std::string p = "model.visual.blocks." + std::to_string(i) + ".";
        bind_tensor(p + "norm1.weight"); bind_tensor(p + "norm1.bias");
        bind_tensor(p + "norm2.weight"); bind_tensor(p + "norm2.bias");
        ok &= bind_tensor(p + "attn.qkv.weight", true);
        bind_tensor(p + "attn.qkv.bias");
        ok &= bind_tensor(p + "attn.proj.weight", true);
        bind_tensor(p + "attn.proj.bias");
        ok &= bind_tensor(p + "mlp.linear_fc1.weight", true);
        bind_tensor(p + "mlp.linear_fc1.bias");
        ok &= bind_tensor(p + "mlp.linear_fc2.weight", true);
        bind_tensor(p + "mlp.linear_fc2.bias");
        if (progress_cb) progress_cb("Binding native Qwen visual tensor refs", (float)(i + 1) / (float)cfg_.n_layers);
    }

    if (!ok) {
        fprintf(stderr, "[vision-native] visual tensor binding failed; not enabling vision\n");
        native_refs_.clear(); native_parser_.reset(); native_qwen3next_ = false; loaded_ = false;
        return false;
    }

    // Important: no full visual tensor is dequantized or copied here. The .nxm mmap
    // is kept alive and rows are decoded only while encoding an actual image.
    patch_embed_w_.clear(); patch_embed_b_.clear(); pos_embed_.clear();
    native_merger_norm_w_.clear(); native_merger_norm_b_.clear();
    proj_w1_.clear(); proj_b1_.clear(); proj_w2_.clear(); proj_b2_.clear();
    layer_weights_.clear(); layer_storage_.clear();

    loaded_ = true;
    fprintf(stderr,
            "[vision-native] Bound Qwen3Next visual tensors from %s without duplicate dequant/copy: image=%d patch=16 grid=%dx%d layers=%d d=%d -> hidden=%d tokens=%d refs=%zu\n",
            model_path.c_str(), cfg_.image_size, native_grid_, native_grid_, cfg_.n_layers,
            cfg_.d_model, cfg_.d_proj_out, tokens_per_image(), native_refs_.size());
    return true;
}

inline bool VisionEncoder::load(
    const std::string& path,
    std::function<void(const std::string&, float)> progress_cb)
{
    gguf::Parser parser;
    if (!parser.open(path)) {
        fprintf(stderr, "[vision] Failed to open %s\n", path.c_str());
        return false;
    }

    // Read config from GGUF KV metadata using actual parser API
    cfg_.image_size  = (int)parser.get_u32("clip.vision.image_size",             336);
    cfg_.patch_size  = (int)parser.get_u32("clip.vision.patch_size",             14);
    cfg_.d_model     = (int)parser.get_u32("clip.vision.hidden_size",            1024);
    cfg_.n_heads     = (int)parser.get_u32("clip.vision.num_attention_heads",    16);
    cfg_.n_layers    = (int)parser.get_u32("clip.vision.num_hidden_layers",      24);
    cfg_.d_ffn       = (int)parser.get_u32("clip.vision.intermediate_size",      4096);
    cfg_.d_proj_out  = (int)parser.get_u32("clip.projection_dim",               4096);
    cfg_.d_proj_in   = cfg_.d_model;

    std::string proj_type = parser.get_str("clip.projector_type", "mlp_gelu");
    cfg_.proj_type = (proj_type.find("linear") != std::string::npos) ? "linear" : "mlp";

    cfg_.preproc.image_size = cfg_.image_size;
    cfg_.preproc.patch_size = cfg_.patch_size;

    int n_patches = (cfg_.image_size / cfg_.patch_size);
    n_patches = n_patches * n_patches;

    // Helper: find tensor by exact name, dequantize to float32
    // Vision GGUF files are mmap'd by open() — data ptr is set in tensors()
    // Helper: find tensor by name, read and dequantize to float32
    // Read raw file bytes for tensor extraction
    FILE* ff = fopen(path.c_str(), "rb");
    std::vector<uint8_t> filebuf;
    if (ff) {
        fseek(ff, 0, SEEK_END); long fsz = ftell(ff); rewind(ff);
        filebuf.resize(fsz);
        (void)fread(filebuf.data(), 1, fsz, ff);
        fclose(ff);
    }
    size_t total_tbytes = 0;
    for (auto& [tn, t] : parser.tensors()) total_tbytes += t.data_size;
    size_t raw_data_start = filebuf.empty() ? 0 :
        ((filebuf.size() - total_tbytes) & ~(size_t)31);

    auto load_tensor = [&](const std::string& name, std::vector<float>& out) -> bool {
        gguf::TensorDesc* td = parser.find_tensor(name);
        if (!td || filebuf.empty()) return false;
        size_t n_elems = td->n_elements;
        if (!n_elems) { n_elems = 1; for (auto d : td->shape) n_elems *= d; }
        out.resize(n_elems);
        quant::QuantType qt;
        switch (td->type) {
            case gguf::GGMLType::F32: qt = quant::QuantType::F32; break;
            case gguf::GGMLType::F16: qt = quant::QuantType::F16; break;
            default: qt = quant::QuantType::F32; break;
        }
        const uint8_t* base = filebuf.data() + raw_data_start + td->offset;
        quant::dequantize(base, out.data(), n_elems, qt);
        return true;
    };

    // Patch embedding
    load_tensor("v.patch_embd.weight", patch_embed_w_);
    load_tensor("v.patch_embd.bias",   patch_embed_b_);

    // CLS token + positional embedding
    load_tensor("v.class_embedding",       cls_token_);
    load_tensor("v.position_embedding.weight", pos_embed_);

    // Pre/post layer norms
    load_tensor("v.pre_ln.weight",  pre_ln_w_);
    load_tensor("v.pre_ln.bias",    pre_ln_b_);
    load_tensor("v.post_ln.weight", post_ln_w_);
    load_tensor("v.post_ln.bias",   post_ln_b_);

    // Projector
    load_tensor("mm.0.weight", proj_w1_);
    load_tensor("mm.0.bias",   proj_b1_);
    if (cfg_.proj_type == "mlp") {
        load_tensor("mm.2.weight", proj_w2_);
        load_tensor("mm.2.bias",   proj_b2_);
    }

    // Per-layer weights
    layer_weights_.resize(cfg_.n_layers);
    layer_storage_.resize(cfg_.n_layers * 16); // 16 tensors per layer max
    int store_idx = 0;

    for (int i = 0; i < cfg_.n_layers; ++i) {
        auto& lw = layer_weights_[i];
        auto L = [&](const std::string& suffix, const float*& dst) {
            std::string name = "v.blk." + std::to_string(i) + "." + suffix;
            load_tensor(name, layer_storage_[store_idx]);
            dst = layer_storage_[store_idx].empty() ? nullptr
                                                     : layer_storage_[store_idx].data();
            ++store_idx;
        };
        L("ln1.weight",  lw.ln1_w); L("ln1.bias",   lw.ln1_b);
        L("ln2.weight",  lw.ln2_w); L("ln2.bias",   lw.ln2_b);
        L("attn_q.weight", lw.wq); L("attn_q.bias", lw.bq);
        L("attn_k.weight", lw.wk); L("attn_k.bias", lw.bk);
        L("attn_v.weight", lw.wv); L("attn_v.bias", lw.bv);
        L("attn_out.weight", lw.wo); L("attn_out.bias", lw.bo);
        L("ffn_up.weight",  lw.fc1_w); L("ffn_up.bias",  lw.fc1_b);
        L("ffn_down.weight", lw.fc2_w); L("ffn_down.bias", lw.fc2_b);

        if (progress_cb)
            progress_cb("Loading vision encoder", (float)(i + 1) / cfg_.n_layers);
    }

    loaded_ = true;
    fprintf(stderr, "[vision] Loaded %s: %dx%d image → %d patches, d=%d\n",
            path.c_str(), cfg_.image_size, cfg_.image_size,
            n_patches, cfg_.d_model);
    return true;
}

inline int VisionEncoder::tokens_per_image() const {
    int n = cfg_.image_size / cfg_.patch_size;
    if (native_qwen3next_) {
        n = (n / 2) * (n / 2);
        return n;
    }
    n = n * n;
    return cfg_.use_cls_only ? 1 : n;
}

inline void VisionEncoder::layer_norm(float* x, const float* w, const float* b, int n, float eps) {
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var = sqrtf(var / n + eps);
    for (int i = 0; i < n; ++i)
        x[i] = (x[i] - mean) / var * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

inline void VisionEncoder::gelu(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

inline std::vector<float> VisionEncoder::encode(
    const uint8_t* rgb_bytes, int width, int height)
{
    auto nchw = preprocess_image(rgb_bytes, width, height, cfg_.preproc);
    return encode_tensor(nchw);
}

inline std::vector<float> VisionEncoder::encode_tensor(const std::vector<float>& nchw) {
    if (native_qwen3next_) {
        auto feats = forward_native_qwen3next(nchw);
        return project_native_qwen3next(feats);
    }
    auto feats = forward(nchw);
    return project(feats);
}



// Qwen3.5 / Qwen3.6 native vision attention uses 2-D rotary embeddings
// on Q/K inside every visual self-attention block.  The learned pos_embed is
// not a replacement for this; HF applies both pos_embed and visual RoPE.
static inline void nx_qwen35_apply_vision_rotary_qk(std::vector<float>& Q,
                                                    std::vector<float>& K,
                                                    int n_tokens,
                                                    int n_heads,
                                                    int head_dim,
                                                    int grid) {
    if (n_tokens <= 0 || n_heads <= 0 || head_dim <= 0 || grid <= 0) return;
    if ((int)Q.size() < n_tokens * n_heads * head_dim) return;
    if ((int)K.size() < n_tokens * n_heads * head_dim) return;
    if (nx_env_truthy_vision("NEXUS_QWEN_NATIVE_DISABLE_VISION_ROPE")) return;

    const int half = head_dim / 2;
    if (half <= 0 || (head_dim & 1)) return;
    const int rope_axis_dim = half;          // per H/W axis before cat(freqs,freqs)
    const int inv_count = rope_axis_dim / 2;
    if (inv_count <= 0) return;

    static bool logged = false;
    if (!logged) {
        fprintf(stderr,
            "[vision-native] visual RoPE active: grid=%d tokens=%d heads=%d head_dim=%d axis_dim=%d theta=10000\n",
            grid, n_tokens, n_heads, head_dim, rope_axis_dim);
        logged = true;
    }

    std::vector<float> inv((size_t)inv_count);
    for (int i = 0; i < inv_count; ++i) {
        const float num = (float)(2 * i);
        inv[(size_t)i] = 1.0f / powf(10000.0f, num / (float)rope_axis_dim);
    }

    std::vector<float> cosv((size_t)head_dim), sinv((size_t)head_dim);
    const int d_model = n_heads * head_dim;
    for (int tok = 0; tok < n_tokens; ++tok) {
        const int y = tok / grid;
        const int x = tok % grid;

        // Qwen3_5VisionRotaryEmbedding(position_ids=[y,x]) gives
        // freqs=[y*inv..., x*inv...], then emb=cat(freqs,freqs), so cos/sin
        // are full head_dim and rotate_half() splits the 72-d head in half.
        for (int i = 0; i < inv_count; ++i) {
            const float fy = (float)y * inv[(size_t)i];
            const float fx = (float)x * inv[(size_t)i];
            cosv[(size_t)i] = cosf(fy);
            sinv[(size_t)i] = sinf(fy);
            cosv[(size_t)inv_count + i] = cosf(fx);
            sinv[(size_t)inv_count + i] = sinf(fx);
            cosv[(size_t)half + i] = cosf(fy);
            sinv[(size_t)half + i] = sinf(fy);
            cosv[(size_t)half + inv_count + i] = cosf(fx);
            sinv[(size_t)half + inv_count + i] = sinf(fx);
        }

        for (int h = 0; h < n_heads; ++h) {
            float* q = Q.data() + (size_t)tok * d_model + (size_t)h * head_dim;
            float* k = K.data() + (size_t)tok * d_model + (size_t)h * head_dim;
            for (int i = 0; i < half; ++i) {
                const int j = i + half;
                const float q0 = q[i], q1 = q[j];
                const float k0 = k[i], k1 = k[j];
                q[i] = q0 * cosv[(size_t)i] - q1 * sinv[(size_t)i];
                q[j] = q1 * cosv[(size_t)j] + q0 * sinv[(size_t)j];
                k[i] = k0 * cosv[(size_t)i] - k1 * sinv[(size_t)i];
                k[j] = k1 * cosv[(size_t)j] + k0 * sinv[(size_t)j];
            }
        }
    }
}

inline std::vector<float> VisionEncoder::forward_native_qwen3next(const std::vector<float>& nchw) {
    const int sz = cfg_.image_size;
    const int ps = cfg_.patch_size;
    const int grid = sz / ps;
    const int np = grid * grid;
    const int d = cfg_.d_model;
    const int patch_dim = 3 * 2 * ps * ps;
    const int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());
    const int worker_threads = nx_vision_env_int("NEXUS_VISION_THREADS", hw_threads, 1, 256);
    vision_mm_calls_ = 0;
    vision_mm_gpu_ = 0;
    vision_mm_cpu_ = 0;
    vision_mm_no_gpu_ready_ = 0;
    vision_mm_non_tight_ = 0;
    vision_mm_gpu_fail_ = 0;
    vision_mm_disabled_ = 0;
    vision_mm_float_pipe_missing_ = 0;

    if (std::getenv("NEXUS_VISION_TRACE") || nx_env_truthy_vision("NEXUS_VISION_GPU")) {
        bool gpu_ready = false;
#ifdef NEXUS_VULKAN
        gpu_ready = ::nexus::gpu::gpu().ready();
#endif
        fprintf(stderr, "[vision-native] schedule: worker_threads=%d native_matmul=%s gpu_ready=%d trace=%d safe_batch_matmul=1\n",
                worker_threads, nx_env_truthy_vision("NEXUS_VISION_GPU") ? "gpu-first" : "cpu",
                gpu_ready ? 1 : 0, nx_env_truthy_vision("NEXUS_VISION_GPU_TRACE") ? 1 : 0);
    }

    std::vector<float> tokens((size_t)np * d, 0.f);
    std::vector<float> patches((size_t)np * patch_dim);
    std::vector<float> patch_bias = native_tensor_f32("model.visual.patch_embed.proj.bias");

    nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
        for (int64_t pii = b; pii < e; ++pii) {
            const int pi = (int)pii;
            int py = pi / grid;
            int px = pi % grid;
            float* patch = patches.data() + (size_t)pi * patch_dim;
            for (int c = 0; c < 3; ++c)
                for (int t = 0; t < 2; ++t)
                    for (int y = 0; y < ps; ++y)
                        for (int x = 0; x < ps; ++x)
                            patch[((c * 2 + t) * ps + y) * ps + x] =
                                nchw[c * sz * sz + (py * ps + y) * sz + (px * ps + x)];
        }
    });
    native_matmul_many("model.visual.patch_embed.proj.weight", patches.data(), (size_t)np, (size_t)patch_dim, tokens.data(), (size_t)d, worker_threads);
    if (patch_bias.size() >= (size_t)d) {
        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                float* tok = tokens.data() + (size_t)ii * d;
                for (int j = 0; j < d; ++j) tok[j] += patch_bias[j];
            }
        });
    }

    std::vector<float> pos = native_tensor_f32("model.visual.pos_embed.weight");
    if (pos.size() >= (size_t)d) {
        const int pos_rows = (int)(pos.size() / (size_t)d);
        int pos_grid = (int)std::lround(std::sqrt((float)pos_rows));
        static bool pos_log_once = false;
        if (pos_grid * pos_grid == pos_rows && pos_grid > 0) {
            if (!pos_log_once) {
                fprintf(stderr,
                    "[vision-native] visual pos_embed: source_grid=%dx%d target_grid=%dx%d mode=%s\n",
                    pos_grid, pos_grid, grid, grid, (pos_grid == grid ? "direct" : "bilinear-resample"));
                pos_log_once = true;
            }
            if (pos_grid == grid) {
                for (int i = 0; i < np; ++i)
                    for (int j = 0; j < d; ++j)
                        tokens[(size_t)i * d + j] += pos[(size_t)i * d + j];
            } else {
                // Learned visual position table is 48x48 in Qwen3.6-27B.  When the
                // launcher uses a smaller/larger Image px value, taking the first N
                // rows means "top-left crop of the position table" and destroys the
                // visual layout.  Resample the 2-D table to the actual patch grid.
                for (int ty = 0; ty < grid; ++ty) {
                    float sy = ((float)ty + 0.5f) * (float)pos_grid / (float)grid - 0.5f;
                    int y0 = std::max(0, std::min(pos_grid - 1, (int)std::floor(sy)));
                    int y1 = std::max(0, std::min(pos_grid - 1, y0 + 1));
                    float wy = sy - (float)y0;
                    if (wy < 0.f) wy = 0.f;
                    for (int tx = 0; tx < grid; ++tx) {
                        float sx = ((float)tx + 0.5f) * (float)pos_grid / (float)grid - 0.5f;
                        int x0 = std::max(0, std::min(pos_grid - 1, (int)std::floor(sx)));
                        int x1 = std::max(0, std::min(pos_grid - 1, x0 + 1));
                        float wx = sx - (float)x0;
                        if (wx < 0.f) wx = 0.f;
                        const float* p00 = pos.data() + ((size_t)y0 * pos_grid + x0) * d;
                        const float* p01 = pos.data() + ((size_t)y0 * pos_grid + x1) * d;
                        const float* p10 = pos.data() + ((size_t)y1 * pos_grid + x0) * d;
                        const float* p11 = pos.data() + ((size_t)y1 * pos_grid + x1) * d;
                        float* dst = tokens.data() + ((size_t)ty * grid + tx) * d;
                        const float w00 = (1.f - wx) * (1.f - wy);
                        const float w01 = wx * (1.f - wy);
                        const float w10 = (1.f - wx) * wy;
                        const float w11 = wx * wy;
                        for (int j = 0; j < d; ++j)
                            dst[j] += p00[j] * w00 + p01[j] * w01 + p10[j] * w10 + p11[j] * w11;
                    }
                }
            }
        } else if (pos_rows >= np) {
            if (!pos_log_once) {
                fprintf(stderr,
                    "[vision-native] visual pos_embed: rows=%d target=%d mode=direct-flat\n",
                    pos_rows, np);
                pos_log_once = true;
            }
            for (int i = 0; i < np; ++i)
                for (int j = 0; j < d; ++j)
                    tokens[(size_t)i * d + j] += pos[(size_t)i * d + j];
        } else {
            fprintf(stderr,
                "[vision-native] visual pos_embed too small: rows=%d target=%d; continuing without pos add\n",
                pos_rows, np);
        }
    }

    const int nh = cfg_.n_heads;
    const int hd = d / nh;
    std::vector<float> buf((size_t)np * d);

    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        const std::string p = "model.visual.blocks." + std::to_string(layer) + ".";
        std::vector<float> ln1w = native_tensor_f32(p + "norm1.weight");
        std::vector<float> ln1b = native_tensor_f32(p + "norm1.bias");
        std::vector<float> ln2w = native_tensor_f32(p + "norm2.weight");
        std::vector<float> ln2b = native_tensor_f32(p + "norm2.bias");
        std::vector<float> qkvb = native_tensor_f32(p + "attn.qkv.bias");
        std::vector<float> ob   = native_tensor_f32(p + "attn.proj.bias");
        std::vector<float> f1b  = native_tensor_f32(p + "mlp.linear_fc1.bias");
        std::vector<float> f2b  = native_tensor_f32(p + "mlp.linear_fc2.bias");

        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                const int i = (int)ii;
                memcpy(buf.data() + (size_t)i * d, tokens.data() + (size_t)i * d, (size_t)d * sizeof(float));
                layer_norm(buf.data() + (size_t)i * d,
                           ln1w.empty() ? nullptr : ln1w.data(),
                           ln1b.empty() ? nullptr : ln1b.data(), d);
            }
        });

        std::vector<float> Q((size_t)np * d), K((size_t)np * d), V((size_t)np * d);
        std::vector<float> qkv_all((size_t)np * (size_t)3 * d);
        native_matmul_many(p + "attn.qkv.weight", buf.data(), (size_t)np, (size_t)d, qkv_all.data(), (size_t)3 * d, worker_threads);
        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                const int i = (int)ii;
                float* qkv = qkv_all.data() + (size_t)i * (size_t)3 * d;
                if (qkvb.size() >= (size_t)3 * d) for (int j = 0; j < 3*d; ++j) qkv[j] += qkvb[j];
                memcpy(Q.data() + (size_t)i * d, qkv, (size_t)d * sizeof(float));
                memcpy(K.data() + (size_t)i * d, qkv + d, (size_t)d * sizeof(float));
                memcpy(V.data() + (size_t)i * d, qkv + 2*d, (size_t)d * sizeof(float));
            }
        });

        nx_qwen35_apply_vision_rotary_qk(Q, K, np, nh, hd, grid);

        float scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out((size_t)np * d, 0.f);

        bool vision_attn_on_gpu = false;
#ifdef NEXUS_VULKAN
        const bool want_vision_gpu = nx_env_truthy_vision("NEXUS_VISION_GPU");
        const bool trace_vision_gpu = nx_env_truthy_vision("NEXUS_VISION_GPU_TRACE");
        if (want_vision_gpu && ::nexus::gpu::gpu().ready()) {
            const auto attn_gpu_t0 = std::chrono::steady_clock::now();
            // Dedicated visual full-attention path. Q/K/V stay token-major float32,
            // exactly like the CPU visual path. This avoids the text/KV causal kernel
            // and its q_base/cache-position assumptions.
            vision_attn_on_gpu = ::nexus::gpu::gpu().vision_attention_full_gpu(
                Q.data(), K.data(), V.data(), attn_out.data(),
                nh, hd, np, scale);
            const auto attn_gpu_t1 = std::chrono::steady_clock::now();
            if (trace_vision_gpu || !vision_attn_on_gpu) {
                const size_t qkv_bytes = (size_t)np * (size_t)nh * (size_t)hd * sizeof(float);
                fprintf(stderr,
                    "[vision-gpu:attn] L=%d %s tokens=%d heads=%d hd=%d qkv=%.1fMB ms=%.2f kernel=vision_attn_full_f32\n",
                    layer, vision_attn_on_gpu ? "GPU-FULL" : "CPU-GPUFAIL", np, nh, hd,
                    (double)(qkv_bytes * 3u) / (1024.0 * 1024.0),
                    std::chrono::duration<double, std::milli>(attn_gpu_t1 - attn_gpu_t0).count());
                fflush(stderr);
            }
        } else if (want_vision_gpu) {
            fprintf(stderr, "[vision-gpu:attn] L=%d CPU-NOREADY tokens=%d heads=%d hd=%d\n", layer, np, nh, hd);
            fflush(stderr);
        }
#endif
        if (!vision_attn_on_gpu) {
            const auto attn_cpu_t0 = std::chrono::steady_clock::now();
            nx_vision_parallel_for((int64_t)nh * (int64_t)np, worker_threads, [&](int64_t b, int64_t e) {
                std::vector<float> row((size_t)np);
                for (int64_t idx = b; idx < e; ++idx) {
                    const int h = (int)(idx / np);
                    const int i = (int)(idx % np);
                    const int off = h * hd;
                    const float* qi = Q.data() + (size_t)i * d + off;
                    float mx = -INFINITY;
                    for (int j = 0; j < np; ++j) {
                        const float* kj = K.data() + (size_t)j * d + off;
                        float score = 0.f;
                        for (int k = 0; k < hd; ++k) score += qi[k] * kj[k];
                        score *= scale;
                        row[(size_t)j] = score;
                        if (score > mx) mx = score;
                    }
                    float sm = 0.f;
                    for (int j = 0; j < np; ++j) {
                        float v = expf(row[(size_t)j] - mx);
                        row[(size_t)j] = v;
                        sm += v;
                    }
                    const float inv = sm > 0.f ? 1.0f / sm : 0.f;
                    float* out = attn_out.data() + (size_t)i * d + off;
                    for (int k = 0; k < hd; ++k) {
                        float sum = 0.f;
                        for (int j = 0; j < np; ++j)
                            sum += (row[(size_t)j] * inv) * V[(size_t)j * d + off + k];
                        out[k] = sum;
                    }
                }
            });
            const auto attn_cpu_t1 = std::chrono::steady_clock::now();
            if (nx_env_truthy_vision("NEXUS_VISION_GPU_TRACE")) {
                fprintf(stderr, "[vision-gpu:attn] L=%d CPU tokens=%d heads=%d hd=%d ms=%.2f\n",
                    layer, np, nh, hd,
                    std::chrono::duration<double, std::milli>(attn_cpu_t1 - attn_cpu_t0).count());
                fflush(stderr);
            }
        }

        std::vector<float> proj_all((size_t)np * d);
        native_matmul_many(p + "attn.proj.weight", attn_out.data(), (size_t)np, (size_t)d, proj_all.data(), (size_t)d, worker_threads);
        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                const int i = (int)ii;
                float* proj = proj_all.data() + (size_t)i * d;
                if (ob.size() >= (size_t)d) for (int j = 0; j < d; ++j) proj[j] += ob[j];
                for (int j = 0; j < d; ++j) tokens[(size_t)i * d + j] += proj[j];
            }
        });

        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                const int i = (int)ii;
                memcpy(buf.data() + (size_t)i * d, tokens.data() + (size_t)i * d, (size_t)d * sizeof(float));
                layer_norm(buf.data() + (size_t)i * d,
                           ln2w.empty() ? nullptr : ln2w.data(),
                           ln2b.empty() ? nullptr : ln2b.data(), d);
            }
        });

        std::vector<float> h1_all((size_t)np * (size_t)cfg_.d_ffn);
        native_matmul_many(p + "mlp.linear_fc1.weight", buf.data(), (size_t)np, (size_t)d, h1_all.data(), (size_t)cfg_.d_ffn, worker_threads);
        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                float* h1 = h1_all.data() + (size_t)ii * (size_t)cfg_.d_ffn;
                if (f1b.size() >= (size_t)cfg_.d_ffn) for (int j = 0; j < cfg_.d_ffn; ++j) h1[j] += f1b[j];
                gelu(h1, cfg_.d_ffn);
            }
        });
        std::vector<float> h2_all((size_t)np * d);
        native_matmul_many(p + "mlp.linear_fc2.weight", h1_all.data(), (size_t)np, (size_t)cfg_.d_ffn, h2_all.data(), (size_t)d, worker_threads);
        nx_vision_parallel_for(np, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t ii = b; ii < e; ++ii) {
                const int i = (int)ii;
                float* h2 = h2_all.data() + (size_t)i * d;
                if (f2b.size() >= (size_t)d) for (int j = 0; j < d; ++j) h2[j] += f2b[j];
                for (int j = 0; j < d; ++j) tokens[(size_t)i * d + j] += h2[j];
            }
        });
    }

    return tokens;
}

inline std::vector<float> VisionEncoder::project_native_qwen3next(const std::vector<float>& feats) {
    const int n = (int)feats.size() / cfg_.d_proj_in;
    const int in_d = cfg_.d_proj_in;
    const int out_d = cfg_.d_proj_out;
    const int grid = native_grid_ > 0 ? native_grid_ : (int)std::sqrt((float)n);
    const int groups = (grid / 2) * (grid / 2);
    const int merge_d = in_d * 4;
    const int hw_threads = std::max(1, (int)std::thread::hardware_concurrency());
    const int worker_threads = nx_vision_env_int("NEXUS_VISION_THREADS", hw_threads, 1, 256);

    std::vector<float> norm_w = native_tensor_f32("model.visual.merger.norm.weight");
    std::vector<float> norm_b = native_tensor_f32("model.visual.merger.norm.bias");
    std::vector<float> fc1_b  = native_tensor_f32("model.visual.merger.linear_fc1.bias");
    std::vector<float> fc2_b  = native_tensor_f32("model.visual.merger.linear_fc2.bias");

    std::vector<float> out((size_t)groups * (size_t)out_d, 0.f);
    std::vector<float> concat_all((size_t)groups * (size_t)merge_d);
    const int groups_x = grid / 2;
    nx_vision_parallel_for(groups, worker_threads, [&](int64_t b, int64_t e) {
        std::vector<float> tmp((size_t)in_d);
        for (int64_t gii = b; gii < e; ++gii) {
            const int gi = (int)gii;
            const int gy = (gi / groups_x) * 2;
            const int gx = (gi % groups_x) * 2;
            int ids[4] = { gy * grid + gx, gy * grid + gx + 1,
                           (gy + 1) * grid + gx, (gy + 1) * grid + gx + 1 };
            float* concat = concat_all.data() + (size_t)gi * merge_d;
            for (int q = 0; q < 4; ++q) {
                const float* src = feats.data() + (size_t)ids[q] * in_d;
                memcpy(tmp.data(), src, (size_t)in_d * sizeof(float));
                layer_norm(tmp.data(), norm_w.empty() ? nullptr : norm_w.data(),
                           norm_b.empty() ? nullptr : norm_b.data(), in_d);
                memcpy(concat + (size_t)q * in_d, tmp.data(), (size_t)in_d * sizeof(float));
            }
        }
    });
    std::vector<float> h1_all((size_t)groups * (size_t)merge_d);
    native_matmul_many("model.visual.merger.linear_fc1.weight", concat_all.data(), (size_t)groups, (size_t)merge_d, h1_all.data(), (size_t)merge_d, worker_threads);
    nx_vision_parallel_for(groups, worker_threads, [&](int64_t b, int64_t e) {
        for (int64_t gii = b; gii < e; ++gii) {
            float* h1 = h1_all.data() + (size_t)gii * merge_d;
            if (fc1_b.size() >= (size_t)merge_d) for (int j = 0; j < merge_d; ++j) h1[j] += fc1_b[j];
            gelu(h1, merge_d);
        }
    });
    native_matmul_many("model.visual.merger.linear_fc2.weight", h1_all.data(), (size_t)groups, (size_t)merge_d, out.data(), (size_t)out_d, worker_threads);
    if (fc2_b.size() >= (size_t)out_d) {
        nx_vision_parallel_for(groups, worker_threads, [&](int64_t b, int64_t e) {
            for (int64_t gii = b; gii < e; ++gii) {
                float* y = out.data() + (size_t)gii * out_d;
                for (int j = 0; j < out_d; ++j) y[j] += fc2_b[j];
            }
        });
    }
    double ss = 0.0, mm = 0.0;
    size_t nn = out.size();
    for (float v : out) { ss += (double)v * (double)v; mm += (double)v; }
    double rms = nn ? std::sqrt(ss / (double)nn) : 0.0;
    double mean = nn ? mm / (double)nn : 0.0;
    fprintf(stderr, "[vision-native] projected image features without duplicate visual weight copy: patches=%d grid=%d groups=%d hidden=%d rms=%.4g mean=%.4g\n",
            n, grid, groups, out_d, rms, mean);
    if (nx_env_truthy_vision("NEXUS_VISION_GPU")) {
        fprintf(stderr, "[vision-gpu:summary] calls=%llu gpu=%llu cpu=%llu disabled=%llu no_ready=%llu non_tight=%llu gpu_fail=%llu float_pipe_missing=%llu\n",
                (unsigned long long)vision_mm_calls_,
                (unsigned long long)vision_mm_gpu_,
                (unsigned long long)vision_mm_cpu_,
                (unsigned long long)vision_mm_disabled_,
                (unsigned long long)vision_mm_no_gpu_ready_,
                (unsigned long long)vision_mm_non_tight_,
                (unsigned long long)vision_mm_gpu_fail_,
                (unsigned long long)vision_mm_float_pipe_missing_);
        if (vision_mm_gpu_ == 0 && vision_mm_float_pipe_missing_ > 0) {
            fprintf(stderr, "[vision-gpu:summary] RESULT=CPU reason=visual weights are float/F16/BF16 but Vulkan has no batched float/BF16 vision mat_mat pipeline yet\n");
        } else if (vision_mm_gpu_ == 0 && vision_mm_gpu_fail_ > 0) {
            fprintf(stderr, "[vision-gpu:summary] RESULT=CPU reason=all requested Vulkan mat_mat calls returned false; enable Vision GPU Trace for per-tensor type/shape\n");
        } else if (vision_mm_gpu_ > 0 && vision_mm_cpu_ > 0) {
            fprintf(stderr, "[vision-gpu:summary] RESULT=MIXED gpu=%llu cpu=%llu; enable Vision GPU Trace for exact tensors\n",
                    (unsigned long long)vision_mm_gpu_, (unsigned long long)vision_mm_cpu_);
        } else if (vision_mm_gpu_ > 0) {
            fprintf(stderr, "[vision-gpu:summary] RESULT=GPU gpu=%llu cpu=%llu\n",
                    (unsigned long long)vision_mm_gpu_, (unsigned long long)vision_mm_cpu_);
        }
    }
    return out;
}

inline std::vector<float> VisionEncoder::forward(const std::vector<float>& nchw) {
    const int sz = cfg_.image_size;
    const int ps = cfg_.patch_size;
    const int np = (sz / ps) * (sz / ps);  // num patches
    const int d  = cfg_.d_model;
    const int patch_dim = native_qwen3next_ ? (3 * 2 * ps * ps) : (3 * ps * ps);

    // ── Patch embedding: extract patches, linear project ─────────────────────
    int n_tokens = np + (cfg_.has_cls_token ? 1 : 0);
    std::vector<float> tokens(n_tokens * d, 0.f);

    // CLS token
    int offset = 0;
    if (cfg_.has_cls_token) {
        if (!cls_token_.empty())
            memcpy(tokens.data(), cls_token_.data(), d * sizeof(float));
        offset = 1;
    }

    // Extract each patch and project: tokens[i+offset] = W_patch * patch_i + b
    for (int pi = 0; pi < np; ++pi) {
        int py = pi / (sz / ps);
        int px = pi % (sz / ps);
        std::vector<float> patch(patch_dim);
        if (native_qwen3next_) {
            // Native Qwen visual patch_embed weight is [out, 3, temporal=2, 16, 16].
            // For still images, duplicate the same frame into both temporal slots.
            for (int c = 0; c < 3; ++c)
                for (int t = 0; t < 2; ++t)
                    for (int y = 0; y < ps; ++y)
                        for (int x = 0; x < ps; ++x)
                            patch[((c * 2 + t) * ps + y) * ps + x] =
                                nchw[c * sz * sz + (py * ps + y) * sz + (px * ps + x)];
        } else {
            // Extract patch in CHW order
            for (int c = 0; c < 3; ++c)
                for (int y = 0; y < ps; ++y)
                    for (int x = 0; x < ps; ++x)
                        patch[c * ps * ps + y * ps + x] =
                            nchw[c * sz * sz + (py * ps + y) * sz + (px * ps + x)];
        }

        float* tok = tokens.data() + (pi + offset) * d;
        // tok = W_patch * patch + b_patch
        for (int j = 0; j < d; ++j) {
            float v = patch_embed_b_.empty() ? 0.f : patch_embed_b_[j];
            for (int k = 0; k < patch_dim; ++k)
                v += patch_embed_w_[j * patch_dim + k] * patch[k];
            tok[j] = v;
        }
    }

    // Add positional embeddings
    if (!pos_embed_.empty()) {
        for (int i = 0; i < n_tokens; ++i)
            for (int j = 0; j < d; ++j)
                tokens[i * d + j] += pos_embed_[i * d + j];
    }

    // Pre-norm (some models)
    if (!pre_ln_w_.empty())
        for (int i = 0; i < n_tokens; ++i)
            layer_norm(tokens.data() + i * d,
                       pre_ln_w_.data(), pre_ln_b_.data(), d);

    // ── Vision transformer layers ─────────────────────────────────────────────
    const int nh = cfg_.n_heads;
    const int hd = d / nh;
    std::vector<float> buf(n_tokens * d);
    std::vector<float> qkv(n_tokens * d);
    std::vector<float> attn(n_tokens * n_tokens);

    for (int layer = 0; layer < cfg_.n_layers; ++layer) {
        const auto& lw = layer_weights_[layer];
        if (!lw.ln1_w) continue;

        // ── Self-attention ────────────────────────────────────────────────────
        // Pre-norm
        for (int i = 0; i < n_tokens; ++i) {
            memcpy(buf.data() + i * d, tokens.data() + i * d, d * sizeof(float));
            layer_norm(buf.data() + i * d, lw.ln1_w, lw.ln1_b, d);
        }

        // Q, K, V projections (GEMM: [n_tokens, d] × [d, d])
        std::vector<float> Q(n_tokens * d), K(n_tokens * d), V(n_tokens * d);
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < d; ++j) {
                float vq = lw.bq ? lw.bq[j] : 0.f;
                float vk = lw.bk ? lw.bk[j] : 0.f;
                float vv = lw.bv ? lw.bv[j] : 0.f;
                for (int k = 0; k < d; ++k) {
                    float b = buf[i * d + k];
                    vq += lw.wq[j * d + k] * b;
                    vk += lw.wk[j * d + k] * b;
                    vv += lw.wv[j * d + k] * b;
                }
                Q[i * d + j] = vq;
                K[i * d + j] = vk;
                V[i * d + j] = vv;
            }
        }

        // Multi-head attention
        float scale = 1.0f / sqrtf((float)hd);
        std::vector<float> attn_out(n_tokens * d, 0.f);
        for (int h = 0; h < nh; ++h) {
            int off = h * hd;
            // Attention scores [n_tokens, n_tokens]
            for (int i = 0; i < n_tokens; ++i)
                for (int j = 0; j < n_tokens; ++j) {
                    float s = 0.f;
                    for (int k = 0; k < hd; ++k)
                        s += Q[i * d + off + k] * K[j * d + off + k];
                    attn[i * n_tokens + j] = s * scale;
                }
            // Softmax per row
            for (int i = 0; i < n_tokens; ++i) {
                float mx = *std::max_element(attn.data() + i * n_tokens,
                                              attn.data() + (i + 1) * n_tokens);
                float sm = 0.f;
                for (int j = 0; j < n_tokens; ++j)
                    sm += (attn[i * n_tokens + j] = expf(attn[i * n_tokens + j] - mx));
                for (int j = 0; j < n_tokens; ++j) attn[i * n_tokens + j] /= sm;
            }
            // Weighted sum of V
            for (int i = 0; i < n_tokens; ++i)
                for (int k = 0; k < hd; ++k) {
                    float s = 0.f;
                    for (int j = 0; j < n_tokens; ++j)
                        s += attn[i * n_tokens + j] * V[j * d + off + k];
                    attn_out[i * d + off + k] = s;
                }
        }

        // Output projection
        for (int i = 0; i < n_tokens; ++i) {
            std::vector<float> proj(d);
            for (int j = 0; j < d; ++j) {
                float v = lw.bo ? lw.bo[j] : 0.f;
                for (int k = 0; k < d; ++k)
                    v += lw.wo[j * d + k] * attn_out[i * d + k];
                proj[j] = v;
            }
            // Residual
            for (int j = 0; j < d; ++j)
                tokens[i * d + j] += proj[j];
        }

        // ── FFN ───────────────────────────────────────────────────────────────
        for (int i = 0; i < n_tokens; ++i) {
            memcpy(buf.data() + i * d, tokens.data() + i * d, d * sizeof(float));
            layer_norm(buf.data() + i * d, lw.ln2_w, lw.ln2_b, d);
        }

        for (int i = 0; i < n_tokens; ++i) {
            float* tok = tokens.data() + i * d;
            float* b   = buf.data()    + i * d;
            std::vector<float> h1(cfg_.d_ffn);
            for (int j = 0; j < cfg_.d_ffn; ++j) {
                float v = lw.fc1_b ? lw.fc1_b[j] : 0.f;
                for (int k = 0; k < d; ++k) v += lw.fc1_w[j * d + k] * b[k];
                h1[j] = v;
            }
            gelu(h1.data(), cfg_.d_ffn);
            // fc2
            std::vector<float> h2(d);
            for (int j = 0; j < d; ++j) {
                float v = lw.fc2_b ? lw.fc2_b[j] : 0.f;
                for (int k = 0; k < cfg_.d_ffn; ++k)
                    v += lw.fc2_w[j * cfg_.d_ffn + k] * h1[k];
                h2[j] = v;
            }
            // Residual
            for (int j = 0; j < d; ++j) tok[j] += h2[j];
        }
    }

    // Post layer-norm
    if (!post_ln_w_.empty())
        for (int i = 0; i < n_tokens; ++i)
            layer_norm(tokens.data() + i * d,
                       post_ln_w_.data(), post_ln_b_.data(), d);

    // Return: CLS-only or all patch tokens
    if (cfg_.use_cls_only) {
        return std::vector<float>(tokens.begin(), tokens.begin() + d);
    }
    // Skip CLS, return patch tokens only
    int start = cfg_.has_cls_token ? d : 0;
    return std::vector<float>(tokens.begin() + start, tokens.end());
}

inline std::vector<float> VisionEncoder::project(const std::vector<float>& feats) {
    // feats: [n_patches, d_proj_in]
    int n = (int)feats.size() / cfg_.d_proj_in;
    int in_d = cfg_.d_proj_in;
    int out_d = cfg_.d_proj_out;

    if (native_qwen3next_) {
        const int grid = native_grid_ > 0 ? native_grid_ : (int)std::sqrt((float)n);
        const int groups = (grid / 2) * (grid / 2);
        const int merge_d = in_d * 4;
        std::vector<float> out((size_t)groups * (size_t)out_d, 0.f);
        std::vector<float> concat(merge_d), h1(merge_d);
        int oi = 0;
        for (int gy = 0; gy + 1 < grid; gy += 2) {
            for (int gx = 0; gx + 1 < grid; gx += 2) {
                int ids[4] = { gy * grid + gx, gy * grid + gx + 1,
                               (gy + 1) * grid + gx, (gy + 1) * grid + gx + 1 };
                for (int q = 0; q < 4; ++q) {
                    const float* src = feats.data() + (size_t)ids[q] * in_d;
                    std::vector<float> tmp(src, src + in_d);
                    if (!native_merger_norm_w_.empty())
                        layer_norm(tmp.data(), native_merger_norm_w_.data(),
                                   native_merger_norm_b_.empty() ? nullptr : native_merger_norm_b_.data(), in_d);
                    memcpy(concat.data() + (size_t)q * in_d, tmp.data(), (size_t)in_d * sizeof(float));
                }
                for (int j = 0; j < merge_d; ++j) {
                    float v = proj_b1_.empty() ? 0.f : proj_b1_[j];
                    for (int k = 0; k < merge_d; ++k) v += proj_w1_[(size_t)j * merge_d + k] * concat[k];
                    h1[j] = v;
                }
                gelu(h1.data(), merge_d);
                float* y = out.data() + (size_t)oi * out_d;
                for (int j = 0; j < out_d; ++j) {
                    float v = proj_b2_.empty() ? 0.f : proj_b2_[j];
                    for (int k = 0; k < merge_d; ++k) v += proj_w2_[(size_t)j * merge_d + k] * h1[k];
                    y[j] = v;
                }
                ++oi;
            }
        }
        fprintf(stderr, "[vision-native] projected image features: patches=%d grid=%d groups=%d hidden=%d\n",
                n, grid, groups, out_d);
        return out;
    }

    std::vector<float> out(n * out_d);

    if (cfg_.proj_type == "linear" || proj_w2_.empty()) {
        // Single linear: out = W1 * feats + b1
        for (int i = 0; i < n; ++i) {
            const float* x = feats.data() + i * in_d;
            float* y = out.data() + i * out_d;
            for (int j = 0; j < out_d; ++j) {
                float v = proj_b1_.empty() ? 0.f : proj_b1_[j];
                for (int k = 0; k < in_d; ++k) v += proj_w1_[j * in_d + k] * x[k];
                y[j] = v;
            }
        }
    } else {
        // MLP: GELU(W1*x + b1), then W2*h + b2
        int mid_d = (int)proj_b1_.size(); // intermediate dim
        std::vector<float> h(mid_d);
        for (int i = 0; i < n; ++i) {
            const float* x = feats.data() + i * in_d;
            float* y = out.data() + i * out_d;
            for (int j = 0; j < mid_d; ++j) {
                float v = proj_b1_.empty() ? 0.f : proj_b1_[j];
                for (int k = 0; k < in_d; ++k) v += proj_w1_[j * in_d + k] * x[k];
                h[j] = v;
            }
            gelu(h.data(), mid_d);
            for (int j = 0; j < out_d; ++j) {
                float v = proj_b2_.empty() ? 0.f : proj_b2_[j];
                for (int k = 0; k < mid_d; ++k) v += proj_w2_[j * mid_d + k] * h[k];
                y[j] = v;
            }
        }
    }
    return out;
}

} // namespace vision
} // namespace nexus
