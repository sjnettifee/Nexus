/*
 * nexus/src/gpu/vulkan_backend.h
 *
 * Vulkan compute backend for GPU-accelerated inference.
 *
 * Design for APU (AMD Ryzen integrated Vega / RDNA):
 *   The CPU and GPU share the same physical DDR4/LPDDR4 RAM.
 *   We use VK_EXT_external_memory_host to import our mmap'd weight data
 *   directly as GPU-visible memory — ZERO COPY, ZERO VRAM duplication.
 *   The GPU reads weights from the same pages the CPU already has mapped.
 *
 *   This is fundamentally different from discrete GPU offloading where you
 *   must transfer weights over PCIe. On an APU it is free.
 *
 * Layer offloading strategy:
 *   - First n_gpu_layers transformer blocks run on GPU
 *   - Remaining blocks run on CPU (existing path)
 *   - Embedding lookup and final lm_head always on CPU (small, cache-hot)
 *   - Only the activation vector (hidden_dim floats) crosses CPU↔GPU boundary
 *     per layer — typically 3072–4096 floats = 12–16 KB
 *
 * Supported quantization types (in this order of priority):
 *   F32   — reference path, always works
 *   F16   — dequant in shader, ~2x memory savings
 *   Q8_0  — 8-bit quant, fastest GPU path
 *   Q4_K  — 4-bit K-quant, best quality/speed tradeoff at 4bpw
 *   Q5_K  — 5-bit K-quant
 *   Q6_K  — 6-bit K-quant
 *
 * Usage:
 *   VulkanBackend gpu;
 *   if (!gpu.init()) { return; }  // fall back to CPU
 *   gpu.set_gpu_layers(n);
 *   // In transformer forward pass:
 *   if (layer_idx < gpu.n_gpu_layers()) {
 *       gpu.mat_vec(W_ptr, x, y, out_rows, in_cols, qtype);
 *   } else {
 *       cpu::mat_vec(W_ptr, x, y, out_rows, in_cols, qtype, n_threads);
 *   }
 *
 * Requirements:
 *   Vulkan 1.1+, VK_EXT_external_memory_host (AMD RADV supports this).
 *   Install: sudo apt install libvulkan-dev vulkan-tools glslang-tools
 *   Build:   cmake -DNEXUS_VULKAN=ON ..
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <map>
#include <tuple>
#include <string>
#include <algorithm>
#include <functional>
#include <mutex>
#include <set>
#include <chrono>
#include <atomic>
#include <thread>
#include <cstdlib>

#ifdef NEXUS_VULKAN
#include "vk_loader.h"   // Vulkan is loaded at runtime (dlopen/LoadLibrary), never linked
#endif
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>   // Sleep() for device enumeration retry
#endif

#include "../gguf/quant.h"

namespace nexus {
namespace gpu {

using quant::QuantType;

// ─── GPU backend status ───────────────────────────────────────────────────────

enum class GpuBackend {
    NONE,       // No GPU — all inference on CPU
    VULKAN,     // Vulkan compute (AMD RADV, Intel ANV, etc.)
};

struct GpuInfo {
    GpuBackend backend     = GpuBackend::NONE;
    std::string device_name;
    size_t      vram_bytes = 0;   // 0 = unified memory (APU)
    bool        unified_memory = false;  // CPU and GPU share RAM
    bool        host_import    = false;  // VK_EXT_external_memory_host supported
    int         compute_units  = 0;
    float       tflops_fp32    = 0.0f;
    uint32_t    vendor_id      = 0;      // PCI vendor: 0x1002 AMD, 0x10DE NVIDIA, 0x8086 Intel
};


#ifdef NEXUS_VULKAN

// ─────────────────────────────────────────────────────────────────────────────
//  SPIRV shader bytecode (embedded at build time by compile_shaders.sh)
//  shader_spirv.h is generated into src/gpu/ and contains the actual
//  uint32_t array definitions. Including it here (inside the NEXUS_VULKAN
//  guard) puts the definitions into every translation unit that includes
//  vulkan_backend.h with Vulkan enabled.
// ─────────────────────────────────────────────────────────────────────────────

#ifdef NEXUS_SHADERS_EMBEDDED
#  ifdef __has_include
#    if __has_include("nexus_shader_spirv.h")
#      include "nexus_shader_spirv.h"   // generated into <build>/gen by tools/gen_shaders.py
#    else
#      error "NEXUS_SHADERS_EMBEDDED is set but nexus_shader_spirv.h was not generated (see tools/gen_shaders.py)"
#    endif
#  else
#    include "nexus_shader_spirv.h"
#  endif
#else
// NEXUS_SHADERS_EMBEDDED not defined — emit stubs (already inside nexus::gpu)
static const uint32_t kSpvMatVecF32[]  = {0}; static const size_t kSpvMatVecF32Size  = 0;
static const uint32_t kSpvMatVecF16[]  = {0}; static const size_t kSpvMatVecF16Size  = 0;
static const uint32_t kSpvMatVecQ8_0[] = {0}; static const size_t kSpvMatVecQ8_0Size = 0;
       static const uint32_t kSpvMatMatQ8_0[] = {0}; static const size_t kSpvMatMatQ8_0Size = 0;
static const uint32_t kSpvMatVecQ4K[]  = {0}; static const size_t kSpvMatVecQ4KSize  = 0;
       static const uint32_t kSpvMatMatQ4K[]  = {0}; static const size_t kSpvMatMatQ4KSize  = 0;
static const uint32_t kSpvMatVecQ5K[]  = {0}; static const size_t kSpvMatVecQ5KSize  = 0;
       static const uint32_t kSpvMatMatQ5K[]  = {0}; static const size_t kSpvMatMatQ5KSize  = 0;
       static const uint32_t kSpvMatVecQ6K[]  = {0}; static const size_t kSpvMatVecQ6KSize  = 0;
       static const uint32_t kSpvMatMatQ6K[]  = {0}; static const size_t kSpvMatMatQ6KSize  = 0;
static const uint32_t kSpvMatVecNQ4G64[] = {0}; static const size_t kSpvMatVecNQ4G64Size = 0;
static const uint32_t kSpvMatMatNQ4G64[] = {0}; static const size_t kSpvMatMatNQ4G64Size = 0;
static const uint32_t kSpvMatVecNQ5G64[] = {0}; static const size_t kSpvMatVecNQ5G64Size = 0;
static const uint32_t kSpvMatMatNQ5G64[] = {0}; static const size_t kSpvMatMatNQ5G64Size = 0;
static const uint32_t kSpvMatVecNQ6G64[] = {0}; static const size_t kSpvMatVecNQ6G64Size = 0;
static const uint32_t kSpvMatMatNQ6G64[] = {0}; static const size_t kSpvMatMatNQ6G64Size = 0;
static const uint32_t kSpvMatVecNQ8G64[] = {0}; static const size_t kSpvMatVecNQ8G64Size = 0;
static const uint32_t kSpvMatMatNQ8G64[] = {0}; static const size_t kSpvMatMatNQ8G64Size = 0;
#endif
#include "spirv_attn3.h"  // low-register decode attention, needs subgroup shuffle (Vulkan 1.1)
#include "spirv_attn2.h"  // register-q decode attention (prebuilt from shd/qwen35_attn_part2.comp)
#include "spirv_dp4a.h"   // int8 dot-product prompt kernels (prebuilt; not generated by gen_shaders.py)

// ─────────────────────────────────────────────────────────────────────────────
//  VulkanBackend — manages Vulkan device, pipelines, and dispatch
// ─────────────────────────────────────────────────────────────────────────────

struct PushConstants {
    uint32_t out_rows;
    uint32_t in_cols;
    uint32_t n_blocks_per_row;  // for quantized types
    float    pad;               // alignment
};

struct GpuBuffer {
    VkBuffer       buffer     = VK_NULL_HANDLE;
    VkDeviceMemory memory     = VK_NULL_HANDLE;
    size_t         size       = 0;
    bool           imported   = false;
    const void*    host_ptr   = nullptr;
    void*          mapped_ptr = nullptr;  // non-null = persistently mapped (HOST_VISIBLE bufs)
};

struct Pipeline {
    VkPipeline       pipeline   = VK_NULL_HANDLE;
    VkPipelineLayout layout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    // Token-generation (matrix x vector) kernels: how many output rows one workgroup produces.  Old kernels: 64 (one thread per row).
    // Rewritten kernels share a row between a team of threads, so they take fewer rows per workgroup -- this MUST equal the ROWS
    // constant at the top of that shader, or rows get skipped.
    uint32_t         rows_per_group = 64;
    // Prompt-reading (matrix x many tokens) kernels: the output tile one workgroup covers.  Old kernels: 16 rows x 8 tokens (one thread per
    // output).  The tiled NQ4 kernel covers 64 rows x 64 tokens -- MUST equal BM / BN at the top of that shader.
    uint32_t         mm_tile_rows = 16;
    uint32_t         mm_tile_toks = 8;
};

// Reports (once per distinct reason) why a GPU fast path stepped aside, so a silent CPU fallback never goes unnoticed.
inline bool nx_decline(const char* why) {
    static std::mutex m; static std::set<std::string> seen;
    std::lock_guard<std::mutex> l(m);
    if (seen.size() < 24 && seen.insert(why).second) { fprintf(stderr, "[nexus] GPU full-attention layer declined: %s\n", why); fflush(stderr); }
    return false;
}

class VulkanBackend {
public:
    VulkanBackend() = default;
    ~VulkanBackend() { shutdown(); }

    // Initialize Vulkan — returns false if GPU not available
    bool init(bool verbose = false);
    void shutdown();

    // Query
    bool       ready()        const { return ready_; }
    int        n_gpu_layers() const { return n_gpu_layers_; }
    GpuInfo    info()         const { return info_; }

    // Set how many transformer layers run on GPU
    void set_gpu_layers(int n) { n_gpu_layers_ = n; }
    
    // Bounded weight cache: set a VRAM budget for cached layer weights.
    // 0 = unlimited (default, backward compatible). When the budget would be
    // exceeded by a new upload, the LRU unpinned weight is evicted first.
    // Minimum useful budget = ~2× the largest single layer's weight bytes so
    // streaming prefill can hold "current + next" at any moment.
    void set_weight_cache_budget(size_t bytes) {
        weight_cache_budget_ = bytes;
        fprintf(stderr, "[vulkan] weight cache budget = %s\n",
                bytes ? (std::to_string(bytes / (1024*1024)) + " MB").c_str()
                      : "unlimited");
    }
    size_t weight_cache_budget() const { return weight_cache_budget_; }
    size_t weight_cache_used()   const { return weight_cache_used_; }

    // Mark a weight as un-evictable. Call for weights you always want resident
    // (lm_head on a full-residency model, persistent norm weights, etc.).
    // If the weight isn't in the cache yet, this is a no-op.
    void pin_weight(const void* W) { pin_weight_public(W); }

    // Free all unpinned cached weights. Call between prefill (streaming) and
    // decode (resident) so decode starts with a clean VRAM slate.
    void evict_unpinned_weights();
    // Block until every queue is idle (streaming prefill uses this between layers).
    void wait_idle() { if (device_) device_idle_locked(); }

    // ---- Optional prefill profiler (NEXUS_PREFILL_PROFILE=1). Off by default; costs nothing then. ----
    // Per GPU op: wall time inside the call, and the part of it spent blocked waiting for the GPU.
    // (wall - wait) is host-side work: copying activations to/from the staging buffers, setup.
    enum ProfOp { P_MAT_MAT = 0, P_CONV, P_SCAN, P_ATTN, P_N };
    struct ProfSnap { double total_ms[P_N]; double wait_ms[P_N]; unsigned long long calls[P_N]; };
    bool prof_enabled() const { return prof_on_; }
    // Per-shape matrix timing (only collected when profiling is on): GPU time of each (rows x cols x type) and the speed it implies.
    void prof_dump_mm(int layer) {
        std::lock_guard<std::mutex> lk(mm_shape_mu_);
        for (auto& kv : mm_shape_) {
            const auto& k = kv.first; const auto& v = kv.second;
            fprintf(stderr, "[turbo:mm-shape] L=%d type=%d rows=%zu cols=%zu  %llu calls  %llu tokens  GPU %.0f ms  = %.0f GFLOP/s  (X up+Y down incl.)\n",
                    layer, std::get<2>(k), std::get<0>(k), std::get<1>(k), v.calls, v.tokens, v.ms,
                    v.ms > 0 ? 2.0 * std::get<0>(k) * std::get<1>(k) * (double)v.tokens / (v.ms * 1e-3) / 1e9 : 0.0);
        }
        fflush(stderr);
        mm_shape_.clear();
    }
    void prof_reset() {
        for (int i = 0; i < P_N; ++i) { prof_total_ns_[i] = 0; prof_wait_op_ns_[i] = 0; prof_calls_[i] = 0; }
    }
    ProfSnap prof_snapshot() const {
        ProfSnap p{};
        for (int i = 0; i < P_N; ++i) {
            p.total_ms[i] = prof_total_ns_[i].load() / 1e6;
            p.wait_ms[i]  = prof_wait_op_ns_[i].load() / 1e6;
            p.calls[i]    = prof_calls_[i].load();
        }
        return p;
    }
    // Free the prompt-sized host staging + scratch buffers (GEMM in/out, DeltaNet
    // conv/scan, attention). They are sized by the largest prompt seen and would
    // otherwise stay resident forever; everything is re-created on demand.
    // Returns bytes released. Weight cache is untouched.
    size_t release_prefill_buffers();

    // Override which physical device to use (index from startup log, -1 = auto)
    void set_device_index(int idx) { gpu_device_index_ = idx; }

    // List available Vulkan devices without fully initializing — for UI/CLI enumeration
    static std::vector<std::string> list_devices() {
        std::vector<std::string> result;
#ifdef NEXUS_VULKAN
        if (!vkl::load()) return result;      // no Vulkan loader on this machine -> CPU only
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &ai;
        {   // portability enumeration (MoltenVK) -- see create_instance()
            uint32_t ne = 0; vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
            std::vector<VkExtensionProperties> ex(ne);
            vkEnumerateInstanceExtensionProperties(nullptr, &ne, ex.data());
            static const char* pe = "VK_KHR_portability_enumeration";
            for (auto& e : ex) if (strcmp(e.extensionName, pe) == 0) {
                ci.enabledExtensionCount = 1; ci.ppEnabledExtensionNames = &pe;
                ci.flags |= 0x00000001; break;
            }
        }
        VkInstance inst = VK_NULL_HANDLE;
        if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return result;
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(inst, &n, devs.data());
        for (uint32_t i = 0; i < n; ++i) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(devs[i], &p);
            const char* dtype =
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? "discrete" :
                p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU  ? "integrated" : "other";
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
            size_t vram = 0;
            for (uint32_t j = 0; j < mp.memoryHeapCount; ++j)
                if (mp.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    vram = std::max(vram, (size_t)mp.memoryHeaps[j].size);
            char buf[256];
            snprintf(buf, sizeof(buf), "[%u] %s (%s, %zu MB)",
                i, p.deviceName, dtype, vram/1048576);
            result.push_back(buf);
        }
        vkDestroyInstance(inst, nullptr);
#endif
        return result;
    }

    // Core operation: y = W × x
    // Returns true if GPU handled it, false if caller should use CPU instead.
    bool mat_vec(
        const void* W,
        const float* x,
        float*       y,
        size_t       out_rows,
        size_t       in_cols,
        QuantType    qtype
    );

    // Batched GEMM: Y = W * X for n_tokens columns at once.
    //   X : in_cols  * n_tokens   (token t input  at X + t*in_cols)
    //   Y : out_rows * n_tokens   (token t output at Y + t*out_rows)
    // Reads W once and reuses across all tokens — the prefill speed primitive.
    // Chunks n_tokens internally to the shader's TOK_TILE. Returns false if the
    // quant type has no batched pipeline (caller falls back to CPU mat_mat).
    bool mat_mat(
        const void*  W,
        const float* X,
        float*       Y,
        size_t       out_rows,
        size_t       in_cols,
        size_t       n_tokens,
        QuantType    qtype
    );

    bool ffn_prefill(const void* Wgate, const void* Wup, const void* Wdown, const float* X, float* Y, size_t ffn, size_t hidden, size_t n_tokens, QuantType qtype);
    bool use_int8_mm(QuantType q, size_t ntok, size_t cols) const;
    bool int8_ensure(size_t ntok, size_t cols);
    bool int8_alloc_ds(const Pipeline& p, VkBuffer* bufs, uint32_t n, VkDescriptorSet& out);
    bool int8_rec_quant(VkCommandBuffer cmd, VkBuffer xbuf, size_t ntok, size_t cols, VkDescriptorSet& ds);
    bool int8_rec_mm(VkCommandBuffer cmd, VkBuffer wbuf, VkBuffer ybuf, size_t rows, size_t cols, uint32_t nb, size_t ntok, VkDescriptorSet& ds);
    struct Int8DsGuard {
        VulkanBackend* b; VkDescriptorSet s[6]{};
        explicit Int8DsGuard(VulkanBackend* bb) : b(bb) {}
        ~Int8DsGuard() { for (auto d : s) if (d) vkFreeDescriptorSets(b->device_, b->ds_pool_, 1, &d); }
    };
    bool mat_mat_multi(const void* W, const float* X, float* Y, size_t out_rows, size_t in_cols, size_t n_tokens, QuantType qtype);

    // True only when the backend has a compiled batched GEMM pipeline for qtype.
    // This is intentionally separate from mat_vec support so transformer logs can
    // tell the truth: "GPU streaming ON" is only useful when the heavy
    // projections are actually backed by mat_mat, not just resident weight cache.
    bool mat_mat_supported(QuantType qtype) const;

    // Qwen3Next/GatedDeltaNet full-prompt GPU helpers.
    bool qwen35_conv1d(const float* mixed,
                       const float* conv_state,
                       const float* conv_weight,
                       float* conv_out,
                       uint32_t n_tokens,
                       uint32_t conv_dim,
                       uint32_t kernel);

    bool qwen35_scan(const float* conv_out,
                     const float* a,
                     const float* b,
                     const float* z,
                     const float* A_log,
                     const float* dt,
                     const float* norm,
                     float* rec_state_inout,
                     float* core_out,
                     uint32_t n_tokens,
                     uint32_t conv_dim,
                     uint32_t key_dim,
                     uint32_t value_dim,
                     uint32_t kh,
                     uint32_t vh,
                     uint32_t hd,
                     uint32_t repeat,
                     bool grouped_layout,
                     float eps);

    // Batched: dispatch two independent mat_vecs that share the same input x.
    // y1 = W1*x,  y2 = W2*x — one GPU submit instead of two.
    // Returns true if BOTH were handled by GPU, false if caller should use CPU for both.
    // ── Init GPU activation buffers ──────────────────────────────────────
    void init_activation_bufs(size_t hidden, size_t q_dim, size_t kv_dim, size_t ffn_dim);

    // Accessors for persistent GPU activation buffers
    GpuBuffer& act_hidden_buf()    { return act_hidden_; }
    GpuBuffer& act_xb_buf()        { return act_xb_; }
    GpuBuffer& act_q_buf()         { return act_q_; }
    GpuBuffer& act_k_buf()         { return act_k_; }
    GpuBuffer& act_v_buf()         { return act_v_; }
    GpuBuffer& act_attn_out_buf()  { return act_attn_out_2_; }
    GpuBuffer& act_gate_buf()      { return act_gate_; }
    GpuBuffer& act_up_buf()        { return act_up_; }
    GpuBuffer& act_ffn_out_buf()   { return act_ffn_out_2_; }
    bool       act_bufs_ok()  const { return act_bufs_ready_; }
    bool       pipe_rms_ok()  const { return pipe_rms_.pipeline  != VK_NULL_HANDLE; }
    bool       pipe_rope_ok() const { return pipe_rope_.pipeline != VK_NULL_HANDLE; }
    bool       pipe_silu_ok() const { return pipe_silu_.pipeline != VK_NULL_HANDLE; }
    bool       pipe_add_ok()  const { return pipe_add_.pipeline  != VK_NULL_HANDLE; }

    // ── Layer-level command recording (one fence wait per layer) ─────────
    void begin_layer_record(const float* x_in, size_t hidden);
    bool chain_begin(const float* x, size_t hidden);   // opens a chain: uploads x to act_hidden_
    void chain_wait_all();                             // wait for every in-flight chained layer (no copy-back)
    bool chain_submit_cmd(VkCommandBuffer cmd, bool end);
    bool chain_submit_layer();                         // end + submit the layer being recorded, no wait
    void partial_submit();       // submit recorded cmds, wait, continue
    void end_layer_record(float* x_out, size_t hidden);
    void layer_barrier(const char* what = nullptr);   // barrier between consecutive dispatches; `what` names the phase that just ran (per-phase timing only)
    void ops_prof_init();
    void ops_prof_collect();
    // int8 dot-product decode path (K-quants): quantise the input vector once, then run the _dp4 matvec kernels.  NEXUS_DECODE_INT8=0 disables.
    bool record_quant_act_g(GpuBuffer& x, size_t cols);   // records quantise + barrier; later record_mat_vec_g calls reading x use the dp4 kernels
    bool dq_mat_vec(const void* W, const float* x, float* y, size_t out_rows, size_t in_cols, QuantType qt);   // standalone (own submit)
    bool decode_int8_active() const { return dec_int8_; }
    // Recorded op functions — only valid between begin/end_layer_record
    bool record_rms_norm_g(GpuBuffer& x_in, const void* w_ptr, size_t w_bytes,
                           GpuBuffer& x_out, uint32_t n, float eps);
    bool record_mat_vec_g(const void* W, QuantType qt, GpuBuffer& x_in,
                          GpuBuffer& y_out, size_t rows, size_t cols);
    bool record_mat_vec_pair_g(const void* W1, QuantType qt1, GpuBuffer& y1,
                               const void* W2, QuantType qt2, GpuBuffer& y2,
                               size_t rows, GpuBuffer& x_in, size_t cols);
    // rope_layout: 0 = interleaved pairs (Llama, Mistral), 1 = half-split pairs (Qwen, Phi, Gemma, EXAONE, OLMo ...)
    bool record_rope_g(GpuBuffer& q, GpuBuffer& k,
                       uint32_t n_q, uint32_t n_kv, uint32_t hd,
                       uint32_t pos, float theta, float scale, uint32_t rope_layout = 0);
    bool record_attention_g(GpuBuffer& q, GpuBuffer& out,
                            int n_heads, int n_kv_heads, int hd, int n_ctx,
                            const uint16_t* k_flat, const uint16_t* v_flat,
                            size_t kv_bytes);
    bool record_silu_mul_g(GpuBuffer& gate, GpuBuffer& up, uint32_t n);
    bool record_add_g(GpuBuffer& x_inout, GpuBuffer& y, uint32_t n);

    // ── Qwen3.5 gated-DeltaNet (linear attention) layer, ONE token, whole layer in ONE submit ──────────────────────────────────────────
    // rms norm -> qkv/z/a/b projections -> conv step -> delta-rule step + gated norm -> out projection -> residual -> FFN norm -> gate/up -> silu*up -> down -> residual.
    // The rolling conv state and the 128x128-per-head recurrent state live on the card between tokens (keyed by owner + layer); the host copies are only read
    // when the card has no valid copy yet, and the host is told to refresh with q35_state_download() before it touches them again.
    struct Q35LinearDesc {
        const void* owner = nullptr; int layer = 0;
        uint32_t hidden = 0, conv_dim = 0, key_dim = 0, value_dim = 0, kh = 0, vh = 0, hd = 0, kernel = 0, ffn = 0;
        float norm_eps = 1e-6f;
        const void* attn_norm = nullptr;
        const void* qkv = nullptr; QuantType qkv_t{};
        const void* z   = nullptr; QuantType z_t{};
        const void* a   = nullptr; QuantType a_t{};
        const void* b   = nullptr; QuantType b_t{};
        const float* conv_w = nullptr; const float* A = nullptr; const float* dt = nullptr; const float* gnorm = nullptr;
        const void* out = nullptr; QuantType out_t{};
        const void* ffn_norm = nullptr;
        const void* gate = nullptr; QuantType gate_t{};
        const void* up   = nullptr; QuantType up_t{};
        const void* down = nullptr; QuantType down_t{};
    };
    // x is read and replaced by the layer output.  conv_state_host / rec_state_host are only used (copied to the card) when the card holds no valid
    // state for (owner, layer).  Returns false BEFORE anything is submitted when the layer cannot run this way (the caller then uses the CPU path).
    bool q35_linear_layer(const Q35LinearDesc& d, float* x, const float* conv_state_host, const float* rec_state_host, bool async = false);
    bool q35_state_resident(const void* owner, int layer) const;
    bool q35_state_download(const void* owner, int layer, float* conv_out, float* rec_out);   // card -> host
    void q35_state_drop(const void* owner, int layer);                                       // layer < 0: all layers of owner.  Next use re-uploads from host.

    // ── Full-attention layer, ONE token, whole layer in ONE submit ─────────────────────────────────────────────────────────────────────
    // Written for Qwen3.5's gated layers and used for ordinary attention layers too (Llama, Mistral, Qwen2/2.5/3, ...): the output gate,
    // the q/k head norms and the q/k/v/o biases are optional (has_gate / q_norm / k_norm / bq.. below), RoPE can be rotate_half or interleaved.
    // rms norm -> wq(q+gate packed)/wk/wv -> q/k head norm + partial RoPE + store K/V in the card's cache -> split-K attention over the whole
    // cache -> output gate -> wo -> residual -> FFN (same tail as the linear layer).  The host KV cache stays the source of truth:
    // the card keeps a MIRROR of it (f16, [kv_head][cap][hd]) that is filled from the host when stale and appended to on the card; the new
    // token's K/V are also returned in kv_out_host so the host cache can store the identical values.  Returns false BEFORE submitting anything
    // when the layer can't run this way (no mirror budget, missing pipeline, ...): the caller then uses the CPU path.
    struct Q35FullDesc {
        const void* owner = nullptr; int layer = 0;
        uint32_t hidden = 0, n_heads = 0, n_kv = 0, hd = 0, rot = 0, ffn = 0, pos = 0;
        float norm_eps = 1e-6f, scale = 0.f;
        const void* attn_norm = nullptr;
        const void* wq = nullptr; QuantType wq_t{};     // rows = 2*n_heads*hd (per-head [q|gate]) when has_gate, else n_heads*hd
        bool has_gate = true;                            // Qwen3.5: q and an output gate packed per head; false = plain attention
        bool rope_interleaved = false;                   // RoPE pairs (2i, 2i+1) instead of rotate_half (i, i+rot/2)
        const float* bq = nullptr; const float* bk = nullptr; const float* bv = nullptr; const float* bo = nullptr;   // optional biases
        const void* wk = nullptr; QuantType wk_t{};
        const void* wv = nullptr; QuantType wv_t{};
        const float* q_norm = nullptr; const float* k_norm = nullptr; bool q_zc = false, k_zc = false;   // per-head norms, optional (nullptr = none)
        const float* rope_tab = nullptr;                 // [rot]: cos[0..rot/2) then sin[0..rot/2) for this position
        const void* wo = nullptr; QuantType wo_t{};
        const void* ffn_norm = nullptr;
        const void* gate = nullptr; QuantType gate_t{};
        const void* up   = nullptr; QuantType up_t{};
        const void* down = nullptr; QuantType down_t{};
        // host KV cache (source of truth)
        const uint16_t* const* k_host = nullptr;         // [n_kv] -> f16 [tokens][hd]
        const uint16_t* const* v_host = nullptr;
        uint32_t host_tokens = 0;                        // tokens present on the host (>= pos)
        uint32_t host_synced = 0;                        // host promises tokens [0, host_synced) are unchanged since the mirror was last filled
    };
    bool q35_full_layer(const Q35FullDesc& d, float* x, float* kv_out_host /* [2*n_kv*hd] */, uint32_t* mirror_valid_out, int async_kv_slot = -1);
    // ---- layer chaining (async submit) ------------------------------------------------------------------------------------------------
    // With async=true the layer is recorded and submitted WITHOUT waiting; the hidden state stays in act_hidden_ between layers, so the CPU records layer N+1
    // while the GPU runs layer N.  The first async layer uploads x (chain opens); chain_end() waits for everything and copies the hidden state back.
    // Async full-attention layers write their new K/V row into a ring slot (chain_kv_ptr) instead of kv_out_host; the caller stores them after chain_end().
    bool chain_is_open() const { return chain_open_; }
    void invalidate_layer_cache() { ++buf_gen_; }   // force every cached recorded layer to be re-recorded on next use
    uint64_t lin_cache_hits() const { return lin_cache_hits_; }
    uint64_t lin_cache_misses() const { return lin_cache_miss_; }
    bool chain_end(float* x_out, size_t hidden);
    const float* chain_kv_ptr(int slot) const { return (slot >= 0 && slot < kChainKvSlots && q35_a_kvring_[slot].mapped_ptr) ? (const float*)q35_a_kvring_[slot].mapped_ptr : nullptr; }
    static constexpr int kChainKvSlots = 16;
    void q35_kv_drop(const void* owner, int layer);                   // layer < 0: all of owner's layers
    // GPU KV mirror budget: -1 = auto (free VRAM after weights, with a safety margin; off on unified-memory GPUs), 0 = off, N = at most N MB in total.
    void   set_gpu_kv_mb(int mb) { gpu_kv_mb_ = mb; }
    int    gpu_kv_mb() const { return gpu_kv_mb_; }
    size_t kv_mirror_used() const { return kv_mirror_used_; }
    // What the card really has free right now, as the driver sees it -- including memory used by other programs
    // (the desktop) and by Nexus's own buffers. False when the driver can't tell (then callers estimate).
    bool vram_free_now(size_t& free_bytes) const {
        free_bytes = 0;
        if (!has_mem_budget_ || !instance_ || !phys_device_ || inst_api_ < VK_API_VERSION_1_1) return false;
        auto fn = (PFN_vkGetPhysicalDeviceMemoryProperties2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceMemoryProperties2");
        if (!fn) return false;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
        bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp{};
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        mp.pNext = &bud;
        fn(phys_device_, &mp);
        // the largest device-local heap is the card's own memory
        int best = -1; VkDeviceSize best_size = 0;
        for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; ++i)
            if ((mp.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && mp.memoryProperties.memoryHeaps[i].size > best_size) {
                best = (int)i; best_size = mp.memoryProperties.memoryHeaps[i].size;
            }
        if (best < 0 || bud.heapBudget[best] == 0) return false;
        free_bytes = bud.heapBudget[best] > bud.heapUsage[best] ? (size_t)(bud.heapBudget[best] - bud.heapUsage[best]) : 0;
        return true;
    }
    size_t kv_mirror_cap_bytes() const;

    // GPU attention: Q (f32), K/V from contiguous KVCache (f16), output (f32).
    // k_ptrs[h], v_ptrs[h]: pointer to KVCache::k_ptr(layer, h) for kv_head h.
    // On APU (unified memory): these are imported zero-copy. No staging buffer.
    // On discrete GPU: K/V copied into a flat staging buffer (skip if ctx too large).
    bool attention_gpu(const float* q, float* out,
                       int n_heads, int n_kv_heads, int head_dim,
                       int layer, int n_ctx,
                       const void* k_flat, const void* v_flat,
                       size_t kv_bytes);

    // Batched prefill attention: all n_query tokens in one dispatch, arbitrary
    // head_dim, per-query causal cutoff, optional sliding window + softcap.
    bool attention_batch_gpu(const float* q_batch, float* out_batch,
                             int n_q_heads, int n_kv_heads, int head_dim,
                             int n_ctx, int n_query, int q_base_pos, int sliding_window,
                             float scale, float softcap,
                             const void* k_flat, const void* v_flat, size_t kv_bytes);

    // Streaming chunked-KV attention. KV stays in host RAM (k_flat/v_flat,
    // [n_kv_heads × n_ctx × head_dim] f16); this uploads it one position-chunk at
    // a time so the GPU never holds the whole KV. chunk_tokens = positions per
    // chunk (0 = auto-size to a VRAM target). Bit-equivalent to attention_batch_gpu.
    bool attention_stream_gpu(const float* q_batch, float* out_batch,
                              int n_q_heads, int n_kv_heads, int head_dim,
                              int n_ctx, int n_query, int q_base_pos,
                              float scale, float softcap,
                              const void* k_flat, const void* v_flat,
                              int chunk_tokens);

    // Tiled (flash-style) causal prompt attention.  K/V are uploaded to VRAM once per call and each workgroup handles 64 queries x 64 keys
    // (see attn_flash.comp).  Returns false (caller falls back to the older kernels) when head_dim is not a multiple of 64 or > 256,
    // the pipeline is missing, or the K/V copy does not fit in VRAM.  k_flat/v_flat: f16 [n_kv_heads][n_ctx][head_dim].
    bool attention_flash_gpu(const float* q_batch, float* out_batch,
                             int n_q_heads, int n_kv_heads, int head_dim,
                             int n_ctx, int n_query, int q_base_pos, int sliding_window,
                             float scale, float softcap,
                             const void* k_flat, const void* v_flat);

    // Dedicated visual-tower full attention. Non-causal, token-major f32 Q/K/V:
    // [n_tokens][n_heads][head_dim]. Do not reuse the text/KV attention kernel.
    bool vision_attention_full_gpu(const float* q, const float* k, const float* v, float* out,
                                   int n_heads, int head_dim, int n_tokens, float scale);

    bool mat_vec_pair(
        const void*  W1, float* y1, size_t rows1, QuantType qt1,
        const void*  W2, float* y2, size_t rows2, QuantType qt2,
        const float* x,  size_t in_cols
    );

    // Batched triple: Q, K, V projections all sharing the same input x.
    // yq = Wq*x,  yk = Wk*x,  yv = Wv*x — single GPU submit.
    bool mat_vec_qkv(
        const void*  wq, float* yq, size_t rows_q, QuantType tq,
        const void*  wk, float* yk, size_t rows_k, QuantType tk,
        const void*  wv, float* yv, size_t rows_v, QuantType tv,
        const float* x,  size_t in_cols
    );

    // Print GPU info to stderr
    void print_info() const;

    // ── Helpers used by Transformer for eager lm_head upload ─────────────────
    // Compute weight buffer byte size without constructing a full dispatch.
    bool size_for_public(QuantType qt, size_t rows, size_t cols,
                         size_t& W_bytes, uint32_t& n_blocks) {
        return size_for(qt, rows, cols, W_bytes, n_blocks);
    }
    // Ensure activation buffers are big enough for a given x/y size.
    void ensure_activation_buffers_public(size_t x_size, size_t y_size) {
        ensure_activation_buffers(x_size, y_size, 0);
    }
    // Trigger eager upload of a weight matrix into the GPU weight cache.
    // Subsequent mat_vec calls with the same pointer hit the cache instantly.
    bool get_weight_buffer_public(const void* W, size_t W_size, QuantType qt) {
        if (!ready_ || !W || W_size == 0) return false;
        GpuBuffer& b = get_weight_buffer(W, W_size, qt);
        return b.buffer != VK_NULL_HANDLE;
    }

    // Asynchronously usable prefetch entry point. Unlike the normal get_weight_buffer()
    // path, this refuses to evict existing cached weights to make room. That keeps
    // in-flight compute safe while a background thread opportunistically uploads
    // weights that will be needed later in the layer/prefill stream.
    bool prefetch_weight_public(const void* W, size_t W_size, QuantType qt);

    // Resident rolling-window path: force weights into the GPU cache using the
    // normal eviction policy, then optionally pin them so the window cannot
    // evict itself while the current layer is running.
    bool load_weight_force_public(const void* W, size_t W_size, QuantType qt, bool pin_after_load = false);
    // Same result as load_weight_force_public, but safe to call from a background loader
    // thread while the main thread keeps computing: the cache lock is NOT held during the
    // upload, it never evicts (a weight that does not fit the budget is simply skipped and
    // loaded synchronously later), and the copy runs on the transfer queue.
    bool load_weight_async_public(const void* W, size_t W_size, QuantType qt, bool pin_after_load = true);
    void pin_weight_public(const void* W);
    void unpin_weight_public(const void* W);
    bool evict_weight_public(const void* W);
    // bytes the cache holds for this weight (0 = not resident)
    size_t weight_size_public(const void* W) {
        std::lock_guard<std::mutex> clk(weight_cache_mu_);
        auto it = weight_cache_.find(W);
        return (it == weight_cache_.end() || !it->second.buf.buffer) ? 0 : it->second.buf.size;
    }
    // "Read from system RAM" mode (discrete cards): weights that are asked for while this is on are NOT copied into VRAM.  The
    // card reads them straight out of the host memory the CPU already holds (zero-copy import), so there is still only one copy.
    // If the memory cannot be imported (not page-aligned, no driver support) the request simply fails and the caller keeps the layer on the CPU.
    void set_host_only_weights(bool on) { host_only_weights_ = on; }
    bool host_import_available() const { return has_host_import_; }

private:
    bool prof_on_ = []{ const char* e = std::getenv("NEXUS_PREFILL_PROFILE"); return e && e[0] == '1'; }();
    struct MMShapeStat { double ms = 0; unsigned long long calls = 0, tokens = 0; };
    std::map<std::tuple<size_t, size_t, int>, MMShapeStat> mm_shape_;
    std::mutex mm_shape_mu_;
    std::atomic<unsigned long long> prof_total_ns_[P_N]{}, prof_wait_op_ns_[P_N]{}, prof_calls_[P_N]{};
    std::atomic<unsigned long long> prof_wait_ns_{0};
    VkResult wait_reuse_fence() {
        VkResult r;
        if (!prof_on_) r = vkWaitForFences(device_, 1, &reuse_fence_, VK_TRUE, UINT64_MAX);
        else {
            const auto t0 = std::chrono::steady_clock::now();
            r = vkWaitForFences(device_, 1, &reuse_fence_, VK_TRUE, UINT64_MAX);
            prof_wait_ns_ += (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - t0).count();
        }
        if (r == VK_ERROR_DEVICE_LOST) note_device_lost();
        return r;
    }
    // The graphics driver reset the card (usually: one GPU job ran past the driver's time limit). From here on this
    // card can't be used until Nexus restarts, so the work falls back to the CPU -- say so once, plainly.
    std::atomic<bool> device_lost_noted_{false};
    void note_device_lost() {
        if (device_lost_noted_.exchange(true)) return;
        fprintf(stderr, "[vulkan] THE GRAPHICS DRIVER RESET THE CARD: a GPU job ran longer than the driver allows (AMD on Linux: "
                        "10 s on the graphics queue, 60 s on a compute queue). Everything from here runs on the CPU until Nexus "
                        "restarts. Smaller prompt slices help: NEXUS_PREFILL_TILE=256 (default 1024).\n");
        fflush(stderr);
    }
    struct ProfScope {
        VulkanBackend* b; int op; std::chrono::steady_clock::time_point t0; unsigned long long w0 = 0;
        ProfScope(VulkanBackend* bb, int o) : b(bb), op(o) {
            if (b->prof_on_) { t0 = std::chrono::steady_clock::now(); w0 = b->prof_wait_ns_.load(); }
        }
        ~ProfScope() {
            if (!b->prof_on_) return;
            b->prof_total_ns_[op] += (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0).count();
            b->prof_wait_op_ns_[op] += b->prof_wait_ns_.load() - w0;
            ++b->prof_calls_[op];
        }
    };
    // vkDeviceWaitIdle needs every queue externally synchronised. Take both queue locks so a
    // background transfer submit cannot race with it.
    void device_idle_locked() {
        std::scoped_lock lk(queue_mu_, transfer_queue_mu_);
        vkDeviceWaitIdle(device_);
    }
    // ── Vulkan objects ──────────────────────────────────────────────────────
    VkInstance       instance_      = VK_NULL_HANDLE;
    VkPhysicalDevice phys_device_   = VK_NULL_HANDLE;
    size_t           max_weight_bytes_ = (size_t)-1;  // largest single weight buffer this device can bind
    VkDevice         device_        = VK_NULL_HANDLE;
    VkQueue          compute_queue_ = VK_NULL_HANDLE;
    VkQueue          transfer_queue_ = VK_NULL_HANDLE;  // dedicated transfer queue when available
    VkCommandPool    cmd_pool_      = VK_NULL_HANDLE;
    VkCommandPool    transfer_cmd_pool_ = VK_NULL_HANDLE;
    VkDescriptorPool ds_pool_       = VK_NULL_HANDLE;

    uint32_t compute_family_ = 0;
    uint32_t transfer_family_ = 0;
    bool     dedicated_transfer_queue_ = false;
    uint32_t min_host_align_ = 4096;  // VK_EXT_external_memory_host requirement

    // ── Pipelines ───────────────────────────────────────────────────────────
    Pipeline pipe_f32_{};
    Pipeline pipe_f16_{};
    Pipeline pipe_q8_0_{};
    Pipeline pipe_mm_q8_0_{};
    Pipeline pipe_q4k_{};
    Pipeline pipe_mm_q4k_{};
    Pipeline pipe_q5k_{};
    Pipeline pipe_mm_q5k_{};
    Pipeline pipe_q6k_{};
    Pipeline pipe_mm_q6k_{};
    Pipeline pipe_q2k_{};
    Pipeline pipe_mm_q2k_{};
    Pipeline pipe_q3k_{};
    Pipeline pipe_mm_q3k_{};
    Pipeline pipe_nq4g64_{};   // Nexus native NQ4_G64 (group-64 signed int4, f16 scale)
    Pipeline pipe_mm_nq4g64_{}; // Batched GEMM variant (N tokens per dispatch)
    Pipeline pipe_mm_nq4g64_dp4_{};  // int8 dot-product (dp4a) 128x128 kernel; needs VK_KHR_shader_integer_dot_product
    Pipeline pipe_quant_q8_{};       // activation quantiser feeding it
    bool     has_int8_dot_ = false;
    bool     int8_mm_on_ = true;          // int8 prompt kernels in use (decided by int8_autotune when the driver doesn't vouch for hardware)
    bool     int8_hw_reported_ = false;   // driver says the 4x8 dot product is hardware-accelerated
    void     int8_autotune();
    // Per-card choice from the launcher: NEXUS_MM_INT8_CARDS="<card name>=1;<card name>=0" (1 on, 0 off; a card not
    // listed follows the all-cards setting). Matched on the card's name, case-insensitive. -1 = not listed.
    static int int8_card_override(const char* device_name) {
        const char* e = std::getenv("NEXUS_MM_INT8_CARDS");
        if (!e || !*e || !device_name) return -1;
        auto low = [](std::string t) { for (char& c : t) c = (char)std::tolower((unsigned char)c); return t; };
        const std::string me = low(device_name), all = e;
        size_t p = 0;
        while (p <= all.size()) {
            size_t q = all.find(';', p); if (q == std::string::npos) q = all.size();
            const std::string item = all.substr(p, q - p); p = q + 1;
            const size_t eq = item.rfind('=');
            if (eq == std::string::npos || eq == 0) continue;
            std::string nm = low(item.substr(0, eq));
            while (!nm.empty() && nm.back() == ' ') nm.pop_back();
            while (!nm.empty() && nm.front() == ' ') nm.erase(0, 1);
            if (!nm.empty() && (me == nm || me.find(nm) != std::string::npos)) return item[eq + 1] == '1' ? 1 : 0;
        }
        return -1;
    }
    bool     has_mem_budget_ = false;      // VK_EXT_memory_budget: the driver reports what is really free on the card
    uint32_t inst_api_ = 0;
    // decode (token generation) int8 dot-product kernels for K-quants
    Pipeline pipe_q4k_dp4_{}, pipe_q5k_dp4_{}, pipe_q6k_dp4_{}, pipe_quant_q8n_{};
    bool     dec_int8_ = false;
    GpuBuffer dq_xq_{}, dq_xs_{};                 // quantised input vector (int8 x4 words) + scale per 32
    static constexpr size_t kDqMaxCols = 65536;
    VkBuffer rec_q8_src_ = VK_NULL_HANDLE;        // buffer whose quantised copy is in dq_xq_/dq_xs_ (valid inside the current recorded layer)
    size_t   rec_q8_cols_ = 0;
    VkDescriptorPool dq_pool_ = VK_NULL_HANDLE;   // 4 persistent sets for the standalone path: [0] quantiser, [1..3] q4k/q5k/q6k
    VkDescriptorSet  dq_ds_[4] = {};
    Pipeline* pipeline_dp4_for(QuantType qt) {
        if (!dec_int8_) return nullptr;
        switch (qt) { case QuantType::Q4_K: return &pipe_q4k_dp4_; case QuantType::Q5_K: return &pipe_q5k_dp4_; case QuantType::Q6_K: return &pipe_q6k_dp4_; default: return nullptr; }
    }
    GpuBuffer mm_xq_{}, mm_xs_{};    // quantised activations (int8 packed) and their scales, VRAM only
    Pipeline pipe_mm_nq4g64_big_{}; // 128x128-tile variant for long prompts (>= 192 tokens)
    Pipeline pipe_nq5g64_{};
    Pipeline pipe_mm_nq5g64_{};
    Pipeline pipe_nq6g64_{};
    Pipeline pipe_mm_nq6g64_{};
    Pipeline pipe_nq8g64_{};
    Pipeline pipe_mm_nq8g64_{};
    Pipeline pipe_q35_conv_{};    // Qwen3Next depthwise causal conv full-prompt
    Pipeline pipe_q35_scan_{};    // Qwen3Next GatedDeltaNet recurrent scan full-prompt
    Pipeline pipe_q35_scan_r_{};  // same, register-resident state (head_dim == 128 only)
    Pipeline pipe_q35_conv_step_{};   // one-token conv with in-place state shift (decode)
    struct Q35DevState { GpuBuffer conv{}, rec{}; size_t conv_n = 0, rec_n = 0; };
    std::map<std::pair<const void*, int>, Q35DevState> q35_dev_state_;
    GpuBuffer q35_t_mixed_{}, q35_t_cout_{}, q35_t_z_{}, q35_t_a_{}, q35_t_b_{}, q35_t_core_{}, q35_t_stage_conv_{}, q35_t_stage_rec_{};
    Pipeline pipe_q35_aprep_{}, pipe_q35_apart_{}, pipe_q35_apart2_{}, pipe_q35_apart3_{}, pipe_q35_acomb_{};   // full-attention decode: prep (norm/rope/cache store), split-K partials, merge+gate
    struct Q35KvMirror { GpuBuffer k{}, v{}; size_t cap = 0, valid = 0, bytes = 0, refused_need = 0; uint32_t n_kv = 0, hd = 0; };
    std::map<std::pair<const void*, int>, Q35KvMirror> q35_kv_mirror_;
    size_t kv_mirror_used_ = 0;
    int    gpu_kv_mb_ = -2;                   // -2 = not decided yet (reads NEXUS_GPU_KV on first use)
    GpuBuffer q35_a_qg_{}, q35_a_k_{}, q35_a_v_{}, q35_a_q_{}, q35_a_gate_{}, q35_a_out_{}, q35_a_part_{}, q35_a_kvout_{}, q35_a_tab_{}, q35_kv_stage_{};
    Pipeline pipe_attn_{};  // GPU attention (score+softmax+V-accum), single-token
    Pipeline pipe_attn_batch_{};  // Batched prefill attention (all N tokens, arbitrary head_dim)
    Pipeline pipe_attn_stream_{};  // Streaming chunked-KV attention (GPU never holds full KV)
    Pipeline pipe_attn_flash_{};   // Tiled flash-style prompt attention (K/V resident in VRAM)
    VkDescriptorSet attn_flash_ds_ = VK_NULL_HANDLE;
    Pipeline pipe_vision_attn_full_{}; // Visual encoder full self-attention, non-causal f32
    GpuBuffer attn_state_buf_{};   // persistent online-softmax state [n_query*n_q_heads*(hd+2)]
    GpuBuffer attn_kchunk_buf_{};  // one KV chunk (K)
    GpuBuffer attn_vchunk_buf_{};  // one KV chunk (V)
    VkDescriptorSet attn_stream_ds_ = VK_NULL_HANDLE;
    Pipeline pipe_rope_{};  // RoPE in-place Q+K
    Pipeline pipe_rms_{};   // RMS normalization
    Pipeline pipe_silu_{};  // SiLU×gate
    Pipeline pipe_add_{};   // residual add

    // ── Buffers ─────────────────────────────────────────────────────────────
    // Activation staging: reused every layer.
    // act_x_  — input x (uploaded once, shared across paired dispatches)
    // act_y_  — output y for single op or first of a pair (gate / q)
    // act_y2_ — output y for second of a pair (up / k)
    // act_y3_ — output y for third op in triple (v)
    GpuBuffer act_x_;
    GpuBuffer act_y_;
    
    GpuBuffer mm_x_{};        // batched mat_mat input GPU buffer (DEVICE_LOCAL)
    GpuBuffer mm_y_{};        // batched mat_mat output GPU buffer (DEVICE_LOCAL)
    GpuBuffer mm_x_stage_{};  // host-visible upload staging for full prompt X
    GpuBuffer mm_y_stage_{};  // host-visible download staging for full prompt Y
    GpuBuffer mm_x_stage_b_{}; // second staging pair: lets the host copy chunk i+1 / chunk i-1 while chunk i runs
    GpuBuffer mm_y_stage_b_{};
    GpuBuffer ffn_u_{};       // fused-FFN prefill: up-projection output (VRAM only)
    GpuBuffer ffn_o_{};       // fused-FFN prefill: down-projection output (VRAM only)

    // Qwen3Next full-prompt conv/scan buffers. Device-local compute buffers + host staging.
    GpuBuffer q35_conv_mixed_{};
    GpuBuffer q35_conv_state_{};
    GpuBuffer q35_conv_out_{};
    GpuBuffer q35_conv_mixed_stage_{};
    GpuBuffer q35_conv_state_stage_{};
    GpuBuffer q35_conv_out_stage_{};

    GpuBuffer q35_scan_conv_{};
    GpuBuffer q35_scan_a_{};
    GpuBuffer q35_scan_b_{};
    GpuBuffer q35_scan_z_{};
    GpuBuffer q35_scan_A_{};
    GpuBuffer q35_scan_dt_{};
    GpuBuffer q35_scan_norm_{};
    GpuBuffer q35_scan_state_{};
    GpuBuffer q35_scan_core_{};
    GpuBuffer q35_scan_conv_stage_{};
    GpuBuffer q35_scan_a_stage_{};
    GpuBuffer q35_scan_b_stage_{};
    GpuBuffer q35_scan_z_stage_{};
    GpuBuffer q35_scan_A_stage_{};
    GpuBuffer q35_scan_dt_stage_{};
    GpuBuffer q35_scan_norm_stage_{};
    GpuBuffer q35_scan_state_stage_{};
    GpuBuffer q35_scan_core_stage_{};
    
    // Attention buffers
    GpuBuffer attn_q_buf_{};   // [n_heads × head_dim] f32
    GpuBuffer attn_out_buf_{}; // [n_heads × head_dim] f32
    GpuBuffer attn_flash_host_{}; // flash attention result readback (host-visible)
    // Per-kv-head K and V GPU handles (imported or copied)
    // Indexed [kv_head]: valid until slot_capacity grows (gpu_dirty flag)
    std::vector<GpuBuffer> attn_k_slots_{};
    std::vector<GpuBuffer> attn_v_slots_{};
    // Flat staging for discrete GPU (when import_host_buffer unavailable)
    GpuBuffer attn_k_stage_{};
    GpuBuffer attn_v_stage_{};
    size_t    attn_stage_cap_ = 0;
    VkDescriptorSet attn_ds_  = VK_NULL_HANDLE;
    VkDescriptorSet attn_batch_ds_ = VK_NULL_HANDLE;  // batched prefill attention
    VkDescriptorSet attn_visual_full_ds_ = VK_NULL_HANDLE; // visual full attention
    GpuBuffer act_y2_;
    GpuBuffer act_y3_;

    // ── Persistent GPU activation buffers (sized by init_activation_bufs) ──
    GpuBuffer act_hidden_{};    // hidden state [hidden_dim] f32
    GpuBuffer act_xb_{};        // normed hidden [hidden_dim] f32
    GpuBuffer act_q_{};         // Q [q_dim] f32
    GpuBuffer act_k_{};         // K [kv_dim] f32
    GpuBuffer act_v_{};         // V [kv_dim] f32
    GpuBuffer act_attn_out_2_{}; // attention output [q_dim] f32
    GpuBuffer act_gate_{};      // FFN gate [ffn_dim] f32
    GpuBuffer act_up_{};        // FFN up [ffn_dim] f32
    GpuBuffer act_ffn_out_2_{}; // FFN down output [hidden_dim] f32
    bool      act_bufs_ready_ = false;

    // ── Layer-level command buffer for batching ───────────────────────────
    VkCommandBuffer layer_cmd_ = VK_NULL_HANDLE;  // separate from reuse_cmd_
    VkDescriptorPool rec_pool_ = VK_NULL_HANDLE;  // reset each layer
    bool layer_recording_      = false;
    // layer chaining: two command-buffer / descriptor-pool slots used alternately, one fence each
    bool            chain_open_   = false;       // act_hidden_ holds the live hidden state, layers may be in flight
    bool            chain_async_  = false;       // the layer being recorded is submitted without waiting
    bool            chain_ready_  = false;
    VkCommandBuffer cslot_cmd_[2]  = {};
    VkDescriptorPool cslot_pool_[2] = {};
    VkFence         cslot_fence_[2] = {};
    bool            cslot_busy_[2]  = {};
    int             cslot_cur_      = 0;
    int64_t         chain_tab_pos_  = -1;        // rope table currently in q35_a_tab_ (identical for all full layers of a token)
    // Linear layers are identical every token (same buffers, descriptors, push constants), so each is recorded once into its own command buffer +
    // descriptor pool and just resubmitted.  Valid while buf_gen_ (bumped by every free_buffer) and the layer signature are unchanged.
    struct Q35LinCache { VkCommandBuffer cmd = VK_NULL_HANDLE; VkDescriptorPool pool = VK_NULL_HANDLE; uint64_t gen = 0, sig = 0; };
    std::map<std::pair<const void*, int>, Q35LinCache> q35_lin_cache_;
    uint64_t        buf_gen_ = 1;
    VkCommandBuffer chain_rec_cmd_  = VK_NULL_HANDLE;  // when set, the next async begin_layer_record records into this (cached) buffer/pool
    VkDescriptorPool chain_rec_pool_ = VK_NULL_HANDLE;
    uint64_t        lin_cache_hits_ = 0, lin_cache_miss_ = 0;
    void lin_cache_drop(Q35LinCache& c);
    GpuBuffer       q35_a_kvring_[16]{};         // per-layer K/V output rows of async full-attention layers
    // ---- per-phase GPU timing of recorded layers (NEXUS_OPS_PROF=N: print averages every N layers of each kind) ----
    int           ops_prof_n_   = -1;                // -1 = not initialised, 0 = off
    VkQueryPool   ops_qp_       = VK_NULL_HANDLE;
    double        ops_period_ns_ = 0.0;
    uint32_t      ops_ts_       = 0;
    int           ops_kind_     = 0;                 // 0 = linear layer, 1 = full-attention layer (set by the caller before begin_layer_record)
    double        ops_acc_[2][32] = {};
    const char*   ops_lab_cur_[32] = {};
    const char*   ops_lab_[2][32] = {};
    int           ops_calls_[2] = {0, 0};
    GpuBuffer rec_dummy_{};    // 16-byte dummy for unused bindings

    // Upload/prefetch synchronization. Queue submissions require external
    // synchronization; the weight cache also must not be mutated concurrently.
    // This lets a transformer-side background prefetch thread stage/upload future
    // weights while the main thread is doing CPU-only Qwen3Next scan/conv work.
    std::mutex queue_mu_;
    std::mutex transfer_queue_mu_;
    std::mutex upload_mu_;
    // Pipelined weight upload (see upload()): two persistent host-visible staging chunks, each with its own command buffer and fence, so
    // copying chunk N+1 from system RAM overlaps the DMA of chunk N and no multi-hundred-MB staging buffer is allocated per tensor.
    struct UpSlot { GpuBuffer buf{}; VkCommandBuffer cmd = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE; };
    UpSlot up_slot_[2];
    bool   up_slots_ready_ = false;
    std::mutex weight_cache_mu_;

    // Pre-allocated reusable command buffer (reset between uses instead of alloc/free)
    VkCommandBuffer reuse_cmd_   = VK_NULL_HANDLE;
    VkFence         reuse_fence_ = VK_NULL_HANDLE;

    // Pre-allocated descriptor sets: one per pipeline slot, updated in-place per call
    // Index matches pipeline order: 0=f32, 1=f16, 2=q8_0, 3=q4k, 4=q5k, 5=q6k
    static constexpr int N_PIPE = 12;  // +NQ5(7) NQ6(8) NQ8(9) Q2_K(10) Q3_K(11)
    VkDescriptorSet  reuse_ds_[N_PIPE]  = {};   // for single mat_vec / first op
    VkDescriptorSet  reuse_ds2_[N_PIPE] = {};   // for second op in pair/triple
    VkDescriptorSet  reuse_ds3_[N_PIPE] = {};   // for third op in triple (QKV)
    bool             ds_allocated_ = false;

    // Weight import cache: maps host pointer → CachedWeight.
    // On APU, imported buffers reference the same DDR4 pages as the CPU mmap
    // (zero-copy; size_bytes==0 so they don't count against the budget).
    // On discrete GPU, the buffer lives in device-local VRAM; size_bytes is
    // the device memory committed, tracked against weight_cache_budget_.
    //
    // When budget is set and a new weight won't fit, evict LRU unpinned
    // entries until it does. Pinned entries (e.g. lm_head) never evict.
    struct CachedWeight {
        GpuBuffer buf;
        size_t    size_bytes = 0;   // device VRAM cost (0 for imported APU buffers)
        uint64_t  last_use   = 0;   // monotonic counter; higher = more recent
        bool      pinned     = false;
    };
    std::unordered_map<const void*, CachedWeight> weight_cache_;
    size_t   weight_cache_used_   = 0;   // sum of size_bytes for non-imported entries
    size_t   weight_cache_budget_ = 0;   // 0 = unlimited (default; backward-compat)
    uint64_t weight_cache_clock_  = 0;   // monotonic LRU counter

    // ── State ───────────────────────────────────────────────────────────────
    bool     ready_           = false;
    bool     has_host_import_ = false;
    bool     host_only_weights_ = false;   // see set_host_only_weights()
    int      n_gpu_layers_    = 0;
    int      gpu_device_index_= -1;   // -1 = auto (prefer discrete), >=0 = pin to that index
    GpuInfo  info_;

    size_t   max_weight_size_ = 0;   // largest weight buffer seen
    size_t   act_buffer_size_ = 0;   // activation buffer capacity

    // ── Internal helpers ───────────────────────────────────────────────────
    bool create_instance();
    bool pick_device();
    bool create_device();
    bool create_command_pool();
    bool create_descriptor_pool();
    bool create_pipelines();
    bool allocate_reuse_resources();

    Pipeline*  pipeline_for(QuantType qt);
    bool       size_for_raw(QuantType qt, size_t rows, size_t cols,
                            size_t& W_bytes, uint32_t& n_blocks);
    bool       size_for(QuantType qt, size_t rows, size_t cols,
                        size_t& W_bytes, uint32_t& n_blocks);

    Pipeline create_pipeline(const uint32_t* spv, size_t spv_size, uint32_t n_bindings);
    GpuBuffer alloc_buffer(size_t, VkBufferUsageFlags, VkMemoryPropertyFlags);
    GpuBuffer import_host_buffer(const void*, size_t);
    GpuBuffer& get_weight_buffer(const void*, size_t, QuantType, bool pin_on_create = false);
    void ensure_activation_buffers(size_t x_size, size_t y1_size, size_t y2_size = 0);
    void ensure_y3_buffer(size_t y3_size);

    // Low-level: record one dispatch into cmd (does NOT submit)
    bool record_one(VkCommandBuffer cmd, VkDescriptorSet ds,
                    Pipeline& pipe, GpuBuffer& W_buf,
                    GpuBuffer& x_buf, GpuBuffer& y_buf,
                    uint32_t rows, uint32_t cols, uint32_t n_blocks);

    void dispatch_mat_vec(Pipeline&, GpuBuffer&, GpuBuffer&, GpuBuffer&,
                          uint32_t, uint32_t, uint32_t);

    uint32_t find_memory_type(uint32_t, VkMemoryPropertyFlags);
    void     free_buffer(GpuBuffer&);
    bool     upload(GpuBuffer& dst, const void* src, size_t size);
    bool     upload_oneshot(GpuBuffer& dst, const void* src, size_t size);   // old path: one staging buffer per tensor (small tensors / fallback)
    void     download(const GpuBuffer& src, void* dst, size_t size);

    VkCommandBuffer begin_one_shot();
    void            end_one_shot(VkCommandBuffer);
};

// ─────────────────────────────────────────────────────────────────────────────
//  Implementation
// ─────────────────────────────────────────────────────────────────────────────

#define VK_CHECK(expr) do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "[vulkan] %s failed: %d at %s:%d\n", #expr, _r, __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

#define VK_CHECK_V(expr) do { \
    VkResult _r = (expr); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "[vulkan] %s failed: %d\n", #expr, _r); \
        return; \
    } \
} while(0)

inline bool VulkanBackend::init(bool verbose) {
    fprintf(stderr, "[vulkan] Initializing...\n"); fflush(stderr);

    // Vulkan is not linked; find the system's loader at runtime.  If there is none
    // (bare-bones OS, no GPU driver stack) the engine simply runs on CPU.
    if (!vkl::load()) {
        fprintf(stderr, "[vulkan] No usable Vulkan loader -- running on CPU.\n%s",
                vkl::error().c_str());
        fflush(stderr);
        return false;
    }
    fprintf(stderr, "[vulkan] Loader: %s\n", vkl::lib_path().c_str()); fflush(stderr);

#ifdef _WIN32
    // On Windows, diagnose common Vulkan loader failures before we even try.
    // The loader finds ICDs via registry: HKLM\SOFTWARE\Khronos\Vulkan\Drivers
    // If vulkan-1.dll exists but devices don't enumerate, it's usually:
    //   a) The ICD JSON path in the registry points to a missing .dll
    //   b) The NVIDIA/Intel ICD dll isn't in PATH or System32
    //   c) VK_ICD_FILENAMES is set to something wrong in the environment
    {
        // Check for overriding env var that bypasses registry (common mistake)
        char icd_env[512] = {};
        if (GetEnvironmentVariableA("VK_ICD_FILENAMES", icd_env, sizeof(icd_env)) > 0) {
            fprintf(stderr, "[vulkan] WARNING: VK_ICD_FILENAMES=%s\n"
                            "         This overrides registry ICD discovery.\n"
                            "         Unset it if devices are missing.\n", icd_env);
            fflush(stderr);
        }
        char driver_env[512] = {};
        if (GetEnvironmentVariableA("VK_DRIVER_FILES", driver_env, sizeof(driver_env)) > 0) {
            fprintf(stderr, "[vulkan] WARNING: VK_DRIVER_FILES=%s\n", driver_env);
            fflush(stderr);
        }
    }
#endif

    if (!create_instance()) {
        fprintf(stderr, "[vulkan] FAILED: Instance creation\n"); fflush(stderr);
        return false;
    }
    fprintf(stderr, "[vulkan] Instance OK\n"); fflush(stderr);

    if (!pick_device()) {
        fprintf(stderr, "[vulkan] FAILED: No suitable GPU found\n"); fflush(stderr);
        shutdown();
        return false;
    }
    fprintf(stderr, "[vulkan] Device selected: %s\n", info_.device_name.c_str()); fflush(stderr);

    if (!create_device()) {
        fprintf(stderr, "[vulkan] FAILED: Logical device creation\n"); fflush(stderr);
        shutdown();
        return false;
    }
    fprintf(stderr, "[vulkan] Logical device OK\n"); fflush(stderr);

    if (!create_command_pool()) {
        fprintf(stderr, "[vulkan] FAILED: Command pool\n"); fflush(stderr);
        shutdown();
        return false;
    }
    fprintf(stderr, "[vulkan] Command pool OK\n"); fflush(stderr);

    if (!create_descriptor_pool()) {
        fprintf(stderr, "[vulkan] FAILED: Descriptor pool\n"); fflush(stderr);
        shutdown();
        return false;
    }
    fprintf(stderr, "[vulkan] Descriptor pool OK\n"); fflush(stderr);

    if (!create_pipelines()) {
        fprintf(stderr, "[vulkan] FAILED: Pipelines (shaders not compiled?)\n");
        fprintf(stderr, "[vulkan]   Fix: bash build.sh --vulkan\n"); fflush(stderr);
        shutdown();
        return false;
    }
    fprintf(stderr, "[vulkan] Pipelines OK — GPU ready\n"); fflush(stderr);

    if (!allocate_reuse_resources()) {
        fprintf(stderr, "[vulkan] FAILED: Reuse resource allocation\n"); fflush(stderr);
        shutdown();
        return false;
    }

    ready_ = true;
    if (has_int8_dot_) int8_autotune();
    if (verbose) print_info();
    return true;
}

// Decide int8 vs regular prompt kernels for THIS card. Trust the driver only when it says the hardware is there; otherwise
// measure: one 2048 x 5120 NQ4 matrix times 512 tokens, best of 3 each way, and keep int8 only if it is clearly faster.
inline void VulkanBackend::int8_autotune() {
    if (int8_card_override(info_.device_name.c_str()) == 1) {
        fprintf(stderr, "[vulkan] 8-bit prompt math on %s: ON (set for this card in the launcher)\n", info_.device_name.c_str());
        fflush(stderr); return;
    }
    if (std::getenv("NEXUS_MM_INT8")) {
        fprintf(stderr, "[vulkan] 8-bit prompt math: ON (forced: NEXUS_MM_INT8=1 / launcher)\n"); fflush(stderr); return;
    }
    if (int8_hw_reported_) {
        fprintf(stderr, "[vulkan] 8-bit prompt math: ON (the driver reports hardware for it)\n"); fflush(stderr); return;
    }
    const size_t rows = 2048, cols = 5120, ntok = 512, bpr = cols / 64, bsz = 34;      // NQ4_G64 block: f16 scale + 32 bytes
    std::vector<uint8_t> W(rows * bpr * bsz);
    uint32_t r = 12345u;
    for (size_t i = 0; i < W.size(); ++i) { r = r * 1664525u + 1013904223u; W[i] = (uint8_t)(r >> 24); }
    for (size_t b = 0; b < rows * bpr; ++b) { W[b * bsz] = 0x00; W[b * bsz + 1] = 0x1c; }   // scale 1/256 (f16 0x1C00)
    std::vector<float> X(cols * ntok), Y(rows * ntok);
    for (size_t i = 0; i < X.size(); ++i) { r = r * 1664525u + 1013904223u; X[i] = ((int)(r >> 20) - 2048) / 2048.0f; }
    auto timed = [&](bool on) -> double {
        int8_mm_on_ = on;
        if (!mat_mat(W.data(), X.data(), Y.data(), rows, cols, ntok, QuantType::NQ4_G64)) return -1.0;   // warm-up (+ upload)
        double best = 1e30;
        for (int k = 0; k < 3; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            if (!mat_mat(W.data(), X.data(), Y.data(), rows, cols, ntok, QuantType::NQ4_G64)) return -1.0;
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    const double t_i8 = timed(true), t_f = timed(false);
    evict_weight_public(W.data());
    const bool keep = t_i8 > 0 && (t_f <= 0 || t_i8 < t_f * 0.95);
    int8_mm_on_ = keep;
    if (!keep) dec_int8_ = false;          // the int8 decode kernels run on the same (missing or emulated) hardware
    fprintf(stderr, "[vulkan] 8-bit prompt math: measured on this card (the driver doesn't say whether it has hardware for it): "
                    "8-bit %.1f ms, regular %.1f ms -> %s (launcher: \"8-bit prompt math\" to force)\n",
            t_i8, t_f, keep ? "8-bit ON" : "regular kernels");
    fflush(stderr);
}

inline bool VulkanBackend::create_instance() {
    // Query available instance extensions before requesting them.
    // On Windows the Vulkan loader reads ICDs from the registry; enumerate
    // extensions FIRST so the loader has a chance to find all ICDs.
    uint32_t n_inst_ext = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n_inst_ext, nullptr);
    std::vector<VkExtensionProperties> inst_exts(n_inst_ext);
    vkEnumerateInstanceExtensionProperties(nullptr, &n_inst_ext, inst_exts.data());

    std::vector<const char*> req_exts;
    bool has_props2_ext = false;
    for (auto& e : inst_exts) {
        if (strcmp(e.extensionName, "VK_KHR_get_physical_device_properties2") == 0) {
            has_props2_ext = true; break;
        }
    }
    if (has_props2_ext)
        req_exts.push_back("VK_KHR_get_physical_device_properties2");

    // macOS / MoltenVK (and any other "portability" driver) is hidden from device
    // enumeration unless the instance opts in to portability enumeration.
    bool has_portability = false;
    for (auto& e : inst_exts)
        if (strcmp(e.extensionName, "VK_KHR_portability_enumeration") == 0) { has_portability = true; break; }
    if (has_portability)
        req_exts.push_back("VK_KHR_portability_enumeration");

    // Try Vulkan 1.1 first; fall back to 1.0 for older Windows drivers.
    // Some NVIDIA drivers on Windows report "layer not present" for 1.1
    // if the runtime DLL hasn't been restarted after driver install.
    uint32_t api_versions[] = { VK_API_VERSION_1_1, VK_API_VERSION_1_0 };
    for (uint32_t api_ver : api_versions) {
        VkApplicationInfo app{};
        app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName   = "nexus";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.apiVersion         = api_ver;

        VkInstanceCreateInfo ci{};
        ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo        = &app;
        ci.enabledExtensionCount   = (uint32_t)req_exts.size();
        ci.ppEnabledExtensionNames = req_exts.empty() ? nullptr : req_exts.data();
        if (has_portability) ci.flags |= 0x00000001;  // VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR

        VkResult r = vkCreateInstance(&ci, nullptr, &instance_);
        if (r == VK_SUCCESS) {
            inst_api_ = api_ver;
            fprintf(stderr, "[vulkan] Instance created (API %u.%u)\n",
                VK_VERSION_MAJOR(api_ver), VK_VERSION_MINOR(api_ver));
            fflush(stderr);
            return true;
        }
        fprintf(stderr, "[vulkan] vkCreateInstance (API %u.%u) failed: %d — trying next\n",
            VK_VERSION_MAJOR(api_ver), VK_VERSION_MINOR(api_ver), (int)r);
        fflush(stderr);
    }
    fprintf(stderr, "[vulkan] vkCreateInstance failed on all API versions\n");
    return false;
}

inline bool VulkanBackend::pick_device() {
    // Windows Vulkan loader: on first launch after driver install the ICD
    // registry key may not be visible yet — retry enumeration a few times.
    // vkEnumeratePhysicalDevices can also return VK_INCOMPLETE if the
    // device vector was too small on the first call; re-query in that case.
    uint32_t n = 0;
    VkResult enum_r = VK_SUCCESS;
    int retries = 0;
    do {
        n = 0;
        enum_r = vkEnumeratePhysicalDevices(instance_, &n, nullptr);
        if (n > 0) break;
        if (retries < 3) {
#ifdef _WIN32
            Sleep(200);   // Windows: give loader time to find ICDs
#endif
        }
    } while (n == 0 && ++retries <= 3);

    if (n == 0) {
        fprintf(stderr, "[vulkan] No physical devices found after %d attempts\n", retries);
        fflush(stderr);
        return false;
    }

    std::vector<VkPhysicalDevice> devs(n);
    // Handle VK_INCOMPLETE: keep growing the vector until fully populated
    for (int attempt = 0; attempt < 3; ++attempt) {
        enum_r = vkEnumeratePhysicalDevices(instance_, &n, devs.data());
        if (enum_r == VK_SUCCESS) break;
        if (enum_r == VK_INCOMPLETE) {
            n++;  devs.resize(n);  // one more slot and retry
        } else {
            fprintf(stderr, "[vulkan] vkEnumeratePhysicalDevices error: %d\n", (int)enum_r);
            fflush(stderr);
            break;
        }
    }

    // Log all available devices so the user can see what's present
    fprintf(stderr, "[vulkan] Available devices (%u):\n", n);
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        const char* dtype =
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? "discrete"   :
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU  ? "integrated" :
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU     ? "virtual"    :
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU             ? "cpu"        : "other";
        fprintf(stderr, "[vulkan]   [%u] %s (%s)\n", i, p.deviceName, dtype);
    }
    fflush(stderr);

    // Scoring: discrete GPU is best (it has dedicated VRAM and more compute),
    // integrated is second (shared memory but still a GPU), CPU/other last.
    // Override: if gpu_device_hint_ >= 0, pick that index directly.
    int   best_score = -1;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devs[i], &props);

        // Must have compute queue
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, qprops.data());

        bool has_compute = false;
        uint32_t fam = 0;
        for (uint32_t j = 0; j < qn; ++j) {
            if (qprops[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                has_compute = true; fam = j; break;
            }
        }
        if (!has_compute) continue;
        // AMD on Linux: the kernel driver's watchdog allows a job 10 s on the graphics queue but 60 s on a compute-only
        // queue. A long prompt's attention step on an older card can pass 10 s, and then the driver resets the card
        // ("context is lost ... soft recovery") and everything after runs on the CPU. So AMD cards use their
        // compute-only queue when they have one. NEXUS_VK_GFX_QUEUE=1 keeps the old choice.
        if (props.vendorID == 0x1002 && !std::getenv("NEXUS_VK_GFX_QUEUE")) {
            for (uint32_t j = 0; j < qn; ++j) {
                const VkQueueFlags fl = qprops[j].queueFlags;
                if ((fl & VK_QUEUE_COMPUTE_BIT) && !(fl & VK_QUEUE_GRAPHICS_BIT) && qprops[j].queueCount > 0) {
                    if (j != fam) fprintf(stderr, "[vulkan]   [%u] %s: AMD card, using its compute-only queue (family %u) -- the driver allows "
                                                  "compute jobs 60 s there instead of 10 s (NEXUS_VK_GFX_QUEUE=1 for the old choice)\n",
                                          i, props.deviceName, j);
                    fam = j;
                    break;
                }
            }
        }

        // Hard requirements of our shaders (workgroup size, shared memory, bindings...).
        {
            std::string why;
            if (!vkl::device_meets_requirements(props.limits, why)) {
                fprintf(stderr, "[vulkan]   [%u] %s rejected: %s\n", i, props.deviceName, why.c_str());
                continue;
            }
            // Software Vulkan (llvmpipe/SwiftShader) is slower than the CPU kernels: only on request.
            if (vkl::device_is_software(props) && gpu_device_index_ < 0 && !vkl::allow_software()) {
                fprintf(stderr, "[vulkan]   [%u] %s skipped: software renderer (set NEXUS_VULKAN_ALLOW_SOFTWARE=1 to use)\n",
                        i, props.deviceName);
                continue;
            }
        }

        // Prefer a transfer-only queue so host->device weight copies can overlap
        // compute on the main queue. If the driver exposes only a combined queue,
        // fall back safely; it still works, just without true copy/compute overlap.
        uint32_t tfam = fam;
        bool dedicated_tfam = false;
        for (uint32_t j = 0; j < qn; ++j) {
            const VkQueueFlags flags = qprops[j].queueFlags;
            if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT)) {
                tfam = j;
                dedicated_tfam = true;
                break;
            }
        }
        if (!dedicated_tfam) {
            for (uint32_t j = 0; j < qn; ++j) {
                if (qprops[j].queueFlags & VK_QUEUE_TRANSFER_BIT) { tfam = j; break; }
            }
        }

        // If user pinned a specific device index, use it exclusively
        if (gpu_device_index_ >= 0) {
            if ((int)i != gpu_device_index_) continue;
            best_score = 100;  // force selection
        } else {
            // Auto: discrete > integrated > cpu/other
            // Both discrete types get high score; we prefer NVIDIA/AMD discrete over Intel integrated
            int score = 0;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score = 10;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)  score =  5;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)             score =  1;
            if (score <= best_score) continue;
            best_score = score;
        }

        phys_device_    = devs[i];
        {   // largest buffer we may bind: min(storage range, allocation size when known)
            size_t lim = (size_t)props.limits.maxStorageBufferRange;
            VkPhysicalDeviceMaintenance3Properties m3{};
            m3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
            VkPhysicalDeviceProperties2 p2{};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &m3;
            auto fn2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2");
            if (fn2) { fn2(devs[i], &p2); if (m3.maxMemoryAllocationSize) lim = std::min(lim, (size_t)m3.maxMemoryAllocationSize); }
            max_weight_bytes_ = lim;
            fprintf(stderr, "[vulkan] max weight buffer for this device: %.0f MB\n", lim / 1048576.0);
        }
        compute_family_ = fam;
        transfer_family_ = tfam;
        dedicated_transfer_queue_ = dedicated_tfam && (tfam != fam);

        VkPhysicalDeviceMemoryProperties mprops{};
        vkGetPhysicalDeviceMemoryProperties(devs[i], &mprops);

        info_.device_name    = props.deviceName;
        info_.vendor_id      = props.vendorID;
        info_.backend        = GpuBackend::VULKAN;
        info_.unified_memory = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
        info_.vram_bytes     = 0;

        for (uint32_t j = 0; j < mprops.memoryHeapCount; ++j) {
            if (mprops.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                info_.vram_bytes = std::max(info_.vram_bytes, mprops.memoryHeaps[j].size);
        }
    }

    if (phys_device_ != VK_NULL_HANDLE) {
        fprintf(stderr, "[vulkan] Selected: %s (%s, %.0f MB VRAM) compute_q=%u transfer_q=%u%s\n",
            info_.device_name.c_str(),
            info_.unified_memory ? "integrated" : "discrete",
            info_.vram_bytes / 1048576.0,
            compute_family_,
            transfer_family_,
            dedicated_transfer_queue_ ? " dedicated-transfer" : " shared-transfer");
        fflush(stderr);
    }
    return phys_device_ != VK_NULL_HANDLE;
}

inline bool VulkanBackend::create_device() {
    // Check for VK_EXT_external_memory_host
    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(phys_device_, nullptr, &n_ext, nullptr);
    std::vector<VkExtensionProperties> exts(n_ext);
    vkEnumerateDeviceExtensionProperties(phys_device_, nullptr, &n_ext, exts.data());

    // Check which extensions are actually available
    bool have_ext_memory_host = false;
    bool have_khr_ext_memory  = false;
    for (auto& e : exts)       // read-only physical-device query: supported is enough, nothing to enable
        if (strcmp(e.extensionName, "VK_EXT_memory_budget") == 0) has_mem_budget_ = true;
    for (auto& e : exts) {
        if (strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0)
            have_ext_memory_host = true;
        if (strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) == 0)
            have_khr_ext_memory = true;
    }

    std::vector<const char*> dev_exts;

    // Spec requirement: a device that advertises VK_KHR_portability_subset (MoltenVK)
    // must have it enabled when the device is created.
    for (auto& e : exts)
        if (strcmp(e.extensionName, "VK_KHR_portability_subset") == 0) {
            dev_exts.push_back("VK_KHR_portability_subset");
            break;
        }

    // VK_EXT_external_memory_host gives us zero-copy APU weight access.
    // VK_KHR_external_memory is required by it — only enable if both exist,
    // AND it's not core (Vulkan 1.1 promotes it to core so extension may not be listed).
    has_host_import_ = false;
    if (have_ext_memory_host) {
        dev_exts.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        // Only request VK_KHR_external_memory if driver exposes it as an extension.
        // On Vulkan 1.1+ it is core and should NOT be listed as an extension.
        if (have_khr_ext_memory)
            dev_exts.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        has_host_import_ = true;
        info_.host_import = true;

        // Query minimum host pointer alignment
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props{};
        host_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &host_props;
        // vkGetPhysicalDeviceProperties2 is core in Vulkan 1.1
        auto fn2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
            instance_, "vkGetPhysicalDeviceProperties2");
        if (!fn2) {
            // Fall back to KHR extension variant
            fn2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(
                instance_, "vkGetPhysicalDeviceProperties2KHR");
        }
        if (fn2) {
            fn2(phys_device_, &props2);
            if (host_props.minImportedHostPointerAlignment > 0)
                min_host_align_ = (uint32_t)host_props.minImportedHostPointerAlignment;
        }
        fprintf(stderr, "[vulkan] Zero-copy host import: active (align=%u) — APU will skip weight copy\n", min_host_align_);
        fflush(stderr);
    } else {
        fprintf(stderr, "[vulkan] Zero-copy host import: NO (will copy weights)\n");
        fflush(stderr);
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci[2]{};
    qci[0].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci[0].queueFamilyIndex = compute_family_;
    qci[0].queueCount       = 1;
    qci[0].pQueuePriorities = &prio;

    uint32_t qci_count = 1;
    if (transfer_family_ != compute_family_) {
        qci[1].sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci[1].queueFamilyIndex = transfer_family_;
        qci[1].queueCount       = 1;
        qci[1].pQueuePriorities = &prio;
        qci_count = 2;
    }

    // int8 dot product (dp4a): enable only if the driver lists the extension AND reports the feature.  NEXUS_MM_NOINT8=1 keeps it off.
    bool want_dot = false;
    VkPhysicalDeviceShaderIntegerDotProductFeatures dotf{};
    dotf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    const int card_i8 = int8_card_override(info_.device_name.c_str());
    if (card_i8 == 1 || (card_i8 != 0 && !std::getenv("NEXUS_MM_NOINT8"))) {
        for (auto& e : exts) if (strcmp(e.extensionName, "VK_KHR_shader_integer_dot_product") == 0) want_dot = true;
        if (want_dot) {
            auto ff2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFeatures2");
            if (!ff2) ff2 = (PFN_vkGetPhysicalDeviceFeatures2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFeatures2KHR");
            VkPhysicalDeviceFeatures2 pf2{}; pf2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2; pf2.pNext = &dotf;
            if (ff2) ff2(phys_device_, &pf2);
            want_dot = ff2 && dotf.shaderIntegerDotProduct == VK_TRUE;
            // The feature can be there without hardware behind it: drivers may offer int8 dot products in software on
            // older cards (e.g. AMD GCN 1/2 under RADV: an R9 290 read prompts at 13 tok/s with them, 30 without). But the
            // driver's "accelerated" flag can't be trusted the other way either: NVIDIA leaves it unset on cards that DO have
            // the hardware (a 3 GB NVIDIA card fell from ~50 to ~30 tok/s when it was required). So: the flag is noted here,
            // and when it isn't set, int8_autotune() times both kinds of prompt math on the card at startup and keeps the
            // faster one. NEXUS_MM_INT8=1 forces int8, NEXUS_MM_NOINT8=1 forces the regular kernels (launcher: "8-bit prompt math").
            if (want_dot) {
                VkPhysicalDeviceShaderIntegerDotProductProperties dpp{};
                dpp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES;
                VkPhysicalDeviceProperties2 pp2{}; pp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2; pp2.pNext = &dpp;
                auto fp2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2");
                if (!fp2) fp2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2KHR");
                if (fp2) {
                    fp2(phys_device_, &pp2);
                    int8_hw_reported_ = dpp.integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE ||
                                        dpp.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated == VK_TRUE;
                }
            }
        }
    }
    const size_t n_exts_base = dev_exts.size();
    if (want_dot) dev_exts.push_back("VK_KHR_shader_integer_dot_product");
    VkPhysicalDeviceShaderIntegerDotProductFeatures dotf_on{};
    dotf_on.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    dotf_on.shaderIntegerDotProduct = VK_TRUE;

    VkDeviceCreateInfo dci{};
    dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    if (want_dot) dci.pNext     = &dotf_on;
    dci.queueCreateInfoCount    = qci_count;
    dci.pQueueCreateInfos       = qci;
    dci.enabledExtensionCount   = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();

    VkResult r = vkCreateDevice(phys_device_, &dci, nullptr, &device_);
    if (r != VK_SUCCESS && want_dot) {          // retry without the optional int8 feature
        want_dot = false; dev_exts.resize(n_exts_base);
        dci.pNext = nullptr; dci.enabledExtensionCount = (uint32_t)dev_exts.size();
        dci.ppEnabledExtensionNames = dev_exts.empty() ? nullptr : dev_exts.data();
        r = vkCreateDevice(phys_device_, &dci, nullptr, &device_);
    }
    has_int8_dot_ = want_dot && r == VK_SUCCESS;
    if (r != VK_SUCCESS) {
        fprintf(stderr, "[vulkan] vkCreateDevice failed: %d\n", r);
        fflush(stderr);
        return false;
    }
    vkGetDeviceQueue(device_, compute_family_, 0, &compute_queue_);
    vkGetDeviceQueue(device_, transfer_family_, 0, &transfer_queue_);
    if (transfer_queue_ == VK_NULL_HANDLE) transfer_queue_ = compute_queue_;
    return true;
}

inline bool VulkanBackend::create_command_pool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = compute_family_;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &cmd_pool_));

    VkCommandPoolCreateInfo tci{};
    tci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    tci.queueFamilyIndex = transfer_family_;
    tci.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device_, &tci, nullptr, &transfer_cmd_pool_));
    return true;
}

inline bool VulkanBackend::create_descriptor_pool() {
    // Each mat_vec dispatch needs 1 descriptor set with 3 bindings (W, x, y).
    // A 7B model with 32 GPU layers needs ~32*7 = 224 sets per token.
    // We allocate and free per-dispatch so we need FREE_DESCRIPTOR_SET_BIT,
    // otherwise vkFreeDescriptorSets is a no-op and the pool exhausts silently.
    VkDescriptorPoolSize pool_size{};
    pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 1024;  // 3 bindings × up to 341 sets

    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // REQUIRED for per-dispatch free
    ci.maxSets       = 512;
    ci.poolSizeCount = 1;
    ci.pPoolSizes    = &pool_size;
    VK_CHECK(vkCreateDescriptorPool(device_, &ci, nullptr, &ds_pool_));
    return true;
}

inline Pipeline VulkanBackend::create_pipeline(const uint32_t* spv, size_t spv_size,
                                                uint32_t n_bindings) {
    Pipeline p{};
    if (!spv || spv_size == 0) return p;

    // Shader module
    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv_size;
    smci.pCode    = spv;
    VkShaderModule shader;
    if (vkCreateShaderModule(device_, &smci, nullptr, &shader) != VK_SUCCESS) return p;

    // Descriptor set layout — n_bindings storage buffers
    std::vector<VkDescriptorSetLayoutBinding> bindings(n_bindings);
    for (uint32_t i = 0; i < n_bindings; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = n_bindings;
    dslci.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &p.ds_layout) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, shader, nullptr); return p;
    }

    // Push constants: 128 bytes — large enough for all shaders.
    // RoPE needs 32 bytes, attn needs 32 bytes, mat_vec needs 16 bytes.
    // Vulkan requires the range to be at least as large as any vkCmdPushConstants call.
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = 128;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &p.ds_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(device_, &plci, nullptr, &p.layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(device_, p.ds_layout, nullptr);
        vkDestroyShaderModule(device_, shader, nullptr); return p;
    }

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = p.layout;
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName  = "main";
    if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline) != VK_SUCCESS) {
        vkDestroyPipelineLayout(device_, p.layout, nullptr);
        vkDestroyDescriptorSetLayout(device_, p.ds_layout, nullptr);
        vkDestroyShaderModule(device_, shader, nullptr); return p;
    }

    vkDestroyShaderModule(device_, shader, nullptr);
    return p;
}

inline bool VulkanBackend::create_pipelines() {
#ifdef NEXUS_SHADERS_EMBEDDED
    pipe_f32_  = create_pipeline(kSpvMatVecF32,  kSpvMatVecF32Size,  3);
    pipe_f16_  = create_pipeline(kSpvMatVecF16,  kSpvMatVecF16Size,  3);
    pipe_q8_0_ = create_pipeline(kSpvMatVecQ8_0, kSpvMatVecQ8_0Size, 3);
    pipe_mm_q8_0_ = create_pipeline(kSpvMatMatQ8_0, kSpvMatMatQ8_0Size, 3);
    pipe_q4k_  = create_pipeline(kSpvMatVecQ4K,  kSpvMatVecQ4KSize,  3);
    pipe_mm_q4k_  = create_pipeline(kSpvMatMatQ4K,  kSpvMatMatQ4KSize,  3);
    pipe_q5k_  = create_pipeline(kSpvMatVecQ5K,  kSpvMatVecQ5KSize,  3);
    pipe_mm_q5k_  = create_pipeline(kSpvMatMatQ5K,  kSpvMatMatQ5KSize,  3);
    pipe_q6k_  = create_pipeline(kSpvMatVecQ6K,  kSpvMatVecQ6KSize,  3);
    pipe_mm_q6k_  = create_pipeline(kSpvMatMatQ6K,  kSpvMatMatQ6KSize,  3);
    pipe_q2k_     = create_pipeline(kSpvMatVecQ2K,  kSpvMatVecQ2KSize,  3);
    pipe_mm_q2k_  = create_pipeline(kSpvMatMatQ2K,  kSpvMatMatQ2KSize,  3);
    pipe_q3k_     = create_pipeline(kSpvMatVecQ3K,  kSpvMatVecQ3KSize,  3);
    pipe_mm_q3k_  = create_pipeline(kSpvMatMatQ3K,  kSpvMatMatQ3KSize,  3);
    pipe_q2k_.rows_per_group = 16;       // mat_vec_q2k.comp v3
    pipe_q3k_.rows_per_group = 8;        // mat_vec_q3k.comp v3
    pipe_nq4g64_ = create_pipeline(kSpvMatVecNQ4G64, kSpvMatVecNQ4G64Size, 3);
    pipe_nq4g64_.rows_per_group = 16;    // mat_vec_nq4_g64.comp v3: 16 rows per 128-thread workgroup (must equal ROWS in the shader)
    pipe_q4k_.rows_per_group = 4;        // mat_vec_q4k.comp v3 (ROWS=16, 128 threads)
    pipe_q6k_.rows_per_group = 2;        // mat_vec_q6k.comp v3
    pipe_q5k_.rows_per_group = 2;        // mat_vec_q5k.comp v3
    pipe_q8_0_.rows_per_group = 8;       // mat_vec_q8_0.comp v3
    pipe_f16_.rows_per_group = 16;       // mat_vec_f16.comp v3 (in_cols must be even)
    pipe_mm_nq4g64_ = create_pipeline(kSpvMatMatNQ4G64, kSpvMatMatNQ4G64Size, 3);
    pipe_mm_nq4g64_big_ = create_pipeline(kSpvMatMatNQ4G64Big, kSpvMatMatNQ4G64BigSize, 3);
    pipe_mm_nq4g64_big_.mm_tile_rows = 128; pipe_mm_nq4g64_big_.mm_tile_toks = 128;   // mat_mat_nq4_g64_big.comp (v4): 128 rows x 128 tokens per workgroup
    if (has_int8_dot_) {
        pipe_mm_nq4g64_dp4_ = create_pipeline(kSpvMatMatNQ4G64Dp4, kSpvMatMatNQ4G64Dp4Size, 4);
        pipe_mm_nq4g64_dp4_.mm_tile_rows = 128; pipe_mm_nq4g64_dp4_.mm_tile_toks = 128;
        pipe_quant_q8_ = create_pipeline(kSpvQuantActQ8, kSpvQuantActQ8Size, 3);
        if (!pipe_mm_nq4g64_dp4_.pipeline || !pipe_quant_q8_.pipeline) has_int8_dot_ = false;
        fprintf(stderr, "[vulkan] int8 dot-product prompt kernels: %s\n", has_int8_dot_ ? "ON (NEXUS_MM_NOINT8=1 disables)" : "unavailable (pipeline creation failed)");
        if (has_int8_dot_) {
            const char* de = std::getenv("NEXUS_DECODE_INT8");
            if (de && de[0] == '0') fprintf(stderr, "[vulkan] int8 dot-product decode kernels: off (NEXUS_DECODE_INT8=0)\n");
            else {
                pipe_q4k_dp4_ = create_pipeline(kSpvMatVecQ4KDp4, kSpvMatVecQ4KDp4Size, 4); pipe_q4k_dp4_.rows_per_group = 4;   // = ROWS in mat_vec_q4k_dp4.comp
                pipe_q5k_dp4_ = create_pipeline(kSpvMatVecQ5KDp4, kSpvMatVecQ5KDp4Size, 4); pipe_q5k_dp4_.rows_per_group = 2;
                pipe_q6k_dp4_ = create_pipeline(kSpvMatVecQ6KDp4, kSpvMatVecQ6KDp4Size, 4); pipe_q6k_dp4_.rows_per_group = 2;
                pipe_quant_q8n_ = create_pipeline(kSpvQuantActQ8N, kSpvQuantActQ8NSize, 3);
                dq_xq_ = alloc_buffer(kDqMaxCols, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                dq_xs_ = alloc_buffer(kDqMaxCols / 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                VkDescriptorPoolSize psz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 };
                VkDescriptorPoolCreateInfo pci{}; pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; pci.maxSets = 4; pci.poolSizeCount = 1; pci.pPoolSizes = &psz;
                bool ok = pipe_q4k_dp4_.pipeline && pipe_q5k_dp4_.pipeline && pipe_q6k_dp4_.pipeline && pipe_quant_q8n_.pipeline && dq_xq_.buffer && dq_xs_.buffer &&
                          vkCreateDescriptorPool(device_, &pci, nullptr, &dq_pool_) == VK_SUCCESS;
                if (ok) {
                    const Pipeline* ps[4] = { &pipe_quant_q8n_, &pipe_q4k_dp4_, &pipe_q5k_dp4_, &pipe_q6k_dp4_ };
                    for (int i = 0; i < 4 && ok; ++i) { VkDescriptorSetAllocateInfo dai{}; dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; dai.descriptorPool = dq_pool_; dai.descriptorSetCount = 1; dai.pSetLayouts = &ps[i]->ds_layout;
                        ok = vkAllocateDescriptorSets(device_, &dai, &dq_ds_[i]) == VK_SUCCESS; }
                }
                dec_int8_ = ok;
                fprintf(stderr, "[vulkan] int8 dot-product decode kernels (Q4_K/Q5_K/Q6_K): %s\n", ok ? "ON (NEXUS_DECODE_INT8=0 disables)" : "unavailable (setup failed)");
            }
        }
    } else {
        fprintf(stderr, "[vulkan] int8 dot-product prompt kernels: off (%s)\n", std::getenv("NEXUS_MM_NOINT8") ? "NEXUS_MM_NOINT8 set" : "device lacks VK_KHR_shader_integer_dot_product");
    }
    pipe_mm_nq4g64_.mm_tile_rows = 64; pipe_mm_nq4g64_.mm_tile_toks = 64;   // mat_mat_nq4_g64.comp v3 (tiled, 4x4 register tile): 64 rows x 64 tokens per workgroup
    pipe_nq5g64_ = create_pipeline(kSpvMatVecNQ5G64, kSpvMatVecNQ5G64Size, 3);
    pipe_mm_nq5g64_ = create_pipeline(kSpvMatMatNQ5G64, kSpvMatMatNQ5G64Size, 3);
    pipe_nq6g64_ = create_pipeline(kSpvMatVecNQ6G64, kSpvMatVecNQ6G64Size, 3);
    pipe_mm_nq6g64_ = create_pipeline(kSpvMatMatNQ6G64, kSpvMatMatNQ6G64Size, 3);
    pipe_nq8g64_ = create_pipeline(kSpvMatVecNQ8G64, kSpvMatVecNQ8G64Size, 3);
    pipe_nq5g64_.rows_per_group = 16;    // mat_vec_nq5/6/8_g64.comp v3 (set AFTER create_pipeline)
    pipe_nq6g64_.rows_per_group = 16;
    pipe_nq8g64_.rows_per_group = 16;
    pipe_mm_nq8g64_ = create_pipeline(kSpvMatMatNQ8G64, kSpvMatMatNQ8G64Size, 3);
    // prompt-reading kernels for every non-NQ4 format are generated from one large-tile skeleton (tools/gen_mm_shaders.py): 128 rows x 128 tokens per workgroup
    for (Pipeline* mp : { &pipe_mm_q8_0_, &pipe_mm_q4k_, &pipe_mm_q5k_, &pipe_mm_q6k_, &pipe_mm_q2k_, &pipe_mm_q3k_, &pipe_mm_nq5g64_, &pipe_mm_nq6g64_, &pipe_mm_nq8g64_ }) { mp->mm_tile_rows = 128; mp->mm_tile_toks = 128; }
    pipe_q35_conv_ = create_pipeline(kSpvQwen35Conv1D, kSpvQwen35Conv1DSize, 4);
    pipe_q35_scan_ = create_pipeline(kSpvQwen35Scan, kSpvQwen35ScanSize, 9);
    pipe_q35_scan_r_ = create_pipeline(kSpvQwen35ScanReg, kSpvQwen35ScanRegSize, 9);
    pipe_q35_conv_step_ = create_pipeline(kSpvQwen35ConvStep, kSpvQwen35ConvStepSize, 4);
    pipe_q35_aprep_ = create_pipeline(kSpvQwen35AttnPrep, kSpvQwen35AttnPrepSize, 11);
    pipe_q35_apart_ = create_pipeline(kSpvQwen35AttnPart, kSpvQwen35AttnPartSize, 4);
    {   // low-register variant: needs Vulkan 1.1 subgroup shuffle with subgroups of at least 32 lanes (NVIDIA, AMD wave32/64); otherwise stays off
        const char* e3 = getenv("NEXUS_ATTN3"); bool ok3 = !(e3 && e3[0] == '0');
        if (ok3) {
            VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &sgp;
            auto f2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2");
            if (f2) f2(phys_device_, &p2);
            ok3 = f2 && sgp.subgroupSize >= 32 && (sgp.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) && (sgp.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
            fprintf(stderr, "[nexus] attention kernel: subgroup size %u, shuffle %s -> low-register kernel %s\n", f2 ? sgp.subgroupSize : 0u, (f2 && (sgp.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) ? "yes" : "no", ok3 ? "enabled" : "off");
        }
        if (ok3) pipe_q35_apart3_ = create_pipeline(kSpvQwen35AttnPart3, kSpvQwen35AttnPart3Size, 4);
    }
    { const char* e = getenv("NEXUS_ATTN2"); if (!(e && e[0] == '0')) pipe_q35_apart2_ = create_pipeline(kSpvQwen35AttnPart2, kSpvQwen35AttnPart2Size, 4); }   // optional; the old kernel stays as fallback
    pipe_q35_acomb_ = create_pipeline(kSpvQwen35AttnComb, kSpvQwen35AttnCombSize, 3);
    pipe_attn_ = create_pipeline(kSpvAttnScore,  kSpvAttnScoreSize,  4);
    pipe_attn_batch_ = create_pipeline(kSpvAttnScoreBatch, kSpvAttnScoreBatchSize, 4);
    pipe_attn_stream_ = create_pipeline(kSpvAttnScoreStream, kSpvAttnScoreStreamSize, 5);
    pipe_attn_flash_ = create_pipeline(kSpvAttnFlash, kSpvAttnFlashSize, 4);
    pipe_vision_attn_full_ = create_pipeline(kSpvVisionAttnFull, kSpvVisionAttnFullSize, 4);
    pipe_rope_ = create_pipeline(kSpvRope,       kSpvRopeSize,       2);
    pipe_rms_  = create_pipeline(kSpvRmsNorm,    kSpvRmsNormSize,    3);
    pipe_silu_ = create_pipeline(kSpvSiluMul,    kSpvSiluMulSize,    2);
    pipe_add_  = create_pipeline(kSpvAddResidual,kSpvAddResidualSize,2);
    // Report which pipelines loaded
    fprintf(stderr,
        "[vulkan] Pipelines: F32=%s F16=%s Q8_0=%s MM_Q8_0=%s Q4K=%s MM_Q4K=%s Q5K=%s MM_Q5K=%s Q6K=%s MM_Q6K=%s NQ4G64=%s MM_NQ4G64=%s NQ5G64=%s MM_NQ5G64=%s NQ6G64=%s MM_NQ6G64=%s NQ8G64=%s MM_NQ8G64=%s Q35_CONV=%s Q35_SCAN=%s Attn=%s VisionAttn=%s RoPE=%s RMS=%s SiLU=%s Add=%s\n",
        pipe_f32_.pipeline       ? "OK" : "--",
        pipe_f16_.pipeline       ? "OK" : "--",
        pipe_q8_0_.pipeline      ? "OK" : "--",
        pipe_mm_q8_0_.pipeline   ? "OK" : "--",
        pipe_q4k_.pipeline       ? "OK" : "--",
        pipe_mm_q4k_.pipeline    ? "OK" : "--",
        pipe_q5k_.pipeline       ? "OK" : "--",
        pipe_mm_q5k_.pipeline    ? "OK" : "--",
        pipe_q6k_.pipeline       ? "OK" : "--",
        pipe_mm_q6k_.pipeline    ? "OK" : "--",
        pipe_nq4g64_.pipeline    ? "OK" : "--",
        pipe_mm_nq4g64_.pipeline ? "OK" : "--",
        pipe_nq5g64_.pipeline    ? "OK" : "--",
        pipe_mm_nq5g64_.pipeline ? "OK" : "--",
        pipe_nq6g64_.pipeline    ? "OK" : "--",
        pipe_mm_nq6g64_.pipeline ? "OK" : "--",
        pipe_nq8g64_.pipeline    ? "OK" : "--",
        pipe_mm_nq8g64_.pipeline ? "OK" : "--",
        pipe_q35_conv_.pipeline  ? "OK" : "--",
        pipe_q35_scan_.pipeline  ? "OK" : "--",
        pipe_attn_.pipeline      ? "OK" : "--",
        pipe_vision_attn_full_.pipeline ? "OK" : "--",
        pipe_rope_.pipeline      ? "OK" : "--",
        pipe_rms_.pipeline       ? "OK" : "--",
        pipe_silu_.pipeline      ? "OK" : "--",
        pipe_add_.pipeline       ? "OK" : "--");
    fflush(stderr);
    return pipe_f32_.pipeline != VK_NULL_HANDLE;
#else
    fprintf(stderr, "[vulkan] Shaders not compiled — run: bash build.sh --vulkan\n");
    return false;
#endif
}

// ── Helper: array of all pipelines in a fixed order ──────────────────────────
inline Pipeline* VulkanBackend::pipeline_for(QuantType qt) {
    switch (qt) {
    case QuantType::F32:  return &pipe_f32_;
    case QuantType::F16:  return &pipe_f16_;
    case QuantType::Q8_0: return &pipe_q8_0_;
    case QuantType::Q4_K: return &pipe_q4k_;
    case QuantType::Q5_K: return &pipe_q5k_;
    case QuantType::Q6_K: return &pipe_q6k_;
    case QuantType::Q2_K: return &pipe_q2k_;
    case QuantType::Q3_K: return &pipe_q3k_;
    case QuantType::NQ4_G64: return &pipe_nq4g64_;
    case QuantType::NQ5_G64: return &pipe_nq5g64_;
    case QuantType::NQ6_G64: return &pipe_nq6g64_;
    case QuantType::NQ8_G64: return &pipe_nq8g64_;
    default:              return nullptr;
    }
}

inline bool VulkanBackend::mat_mat_supported(QuantType qt) const {
    switch (qt) {
    case QuantType::Q8_0:    return pipe_mm_q8_0_.pipeline   != VK_NULL_HANDLE;
    case QuantType::Q4_K:    return pipe_mm_q4k_.pipeline    != VK_NULL_HANDLE;
    case QuantType::Q5_K:    return pipe_mm_q5k_.pipeline    != VK_NULL_HANDLE;
    case QuantType::Q6_K:    return pipe_mm_q6k_.pipeline    != VK_NULL_HANDLE;
    case QuantType::Q2_K:    return pipe_mm_q2k_.pipeline    != VK_NULL_HANDLE;
    case QuantType::Q3_K:    return pipe_mm_q3k_.pipeline    != VK_NULL_HANDLE;
    case QuantType::NQ4_G64: return pipe_mm_nq4g64_.pipeline != VK_NULL_HANDLE;
    case QuantType::NQ5_G64: return pipe_mm_nq5g64_.pipeline != VK_NULL_HANDLE;
    case QuantType::NQ6_G64: return pipe_mm_nq6g64_.pipeline != VK_NULL_HANDLE;
    case QuantType::NQ8_G64: return pipe_mm_nq8g64_.pipeline != VK_NULL_HANDLE;
    default: return false;
    }
}

inline bool VulkanBackend::size_for_raw(QuantType qt, size_t rows, size_t cols,
                                     size_t& W_bytes, uint32_t& n_blocks) {
    n_blocks = 0;
    switch (qt) {
    case QuantType::F32:  W_bytes = rows * cols * 4;                                      return true;
    case QuantType::F16:  W_bytes = rows * cols * 2;                                      return true;
    case QuantType::Q8_0: W_bytes = rows * (cols/32)  * sizeof(quant::BlockQ8_0);
                          n_blocks = (uint32_t)(cols / 32);                               return true;
    case QuantType::Q4_K: W_bytes = rows * (cols/256) * sizeof(quant::BlockQ4_K);
                          n_blocks = (uint32_t)(cols / 256);                              return true;
    case QuantType::Q5_K: W_bytes = rows * (cols/256) * sizeof(quant::BlockQ5_K);
                          n_blocks = (uint32_t)(cols / 256);                              return true;
    case QuantType::Q6_K: W_bytes = rows * (cols/256) * sizeof(quant::BlockQ6_K);
                          n_blocks = (uint32_t)(cols / 256);                              return true;
    case QuantType::Q2_K: W_bytes = rows * (cols/256) * sizeof(quant::BlockQ2_K);
                          n_blocks = (uint32_t)(cols / 256);                              return true;
    case QuantType::Q3_K: W_bytes = rows * (cols/256) * sizeof(quant::BlockQ3_K);
                          n_blocks = (uint32_t)(cols / 256);                              return true;
    case QuantType::NQ4_G64: {
                          // Native NQ is grouped by 64 columns, but some vision matrices
                          // are NOT 64-aligned (for example 4304 = 67*64 + 16).
                          // The old floor(cols/64) row pitch truncated the final partial
                          // block and misaligned every following row, corrupting visual
                          // MLP/projector matmuls. Use ceil groups and let the shader skip
                          // padded columns.
                          const size_t nb = (cols + 63) / 64;
                          W_bytes = rows * nb * 34;  // 2B f16 scale + 32B packed nibbles
                          n_blocks = (uint32_t)nb;                                      return true;
    }
    case QuantType::NQ5_G64: {
                          const size_t nb = (cols + 63) / 64;
                          W_bytes = rows * nb * 42;
                          n_blocks = (uint32_t)nb;                                      return true;
    }
    case QuantType::NQ6_G64: {
                          const size_t nb = (cols + 63) / 64;
                          W_bytes = rows * nb * 50;
                          n_blocks = (uint32_t)nb;                                      return true;
    }
    case QuantType::NQ8_G64: {
                          const size_t nb = (cols + 63) / 64;
                          W_bytes = rows * nb * 66;
                          n_blocks = (uint32_t)nb;                                      return true;
    }
    default:              W_bytes = 0;                                                    return false;
    }
}

// Every weight matrix goes through here before it is uploaded.  A matrix larger than the
// device can bind in ONE storage buffer (maxStorageBufferRange / maxMemoryAllocationSize;
// only 128 MiB is guaranteed by the spec) is refused -> the caller uses the CPU for it,
// instead of the GPU silently reading past the end of the binding.
inline bool VulkanBackend::size_for(QuantType qt, size_t rows, size_t cols,
                                     size_t& W_bytes, uint32_t& n_blocks) {
    if (!size_for_raw(qt, rows, cols, W_bytes, n_blocks)) return false;
    if (W_bytes > max_weight_bytes_) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 3)
            fprintf(stderr, "[vulkan] weight %zux%zu is %.0f MB > device limit %.0f MB per buffer -- using CPU for it\n",
                    rows, cols, W_bytes / 1048576.0, max_weight_bytes_ / 1048576.0);
        return false;
    }
    return true;
}

// ── Pre-allocate reusable command buffer + descriptor sets ────────────────────
inline bool VulkanBackend::allocate_reuse_resources() {
    // One persistent command buffer — reset between uses
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmd_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device_, &ai, &reuse_cmd_) != VK_SUCCESS) return false;
    VkFenceCreateInfo reuse_fci{};
    reuse_fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_, &reuse_fci, nullptr, &reuse_fence_) != VK_SUCCESS) return false;

    // Two descriptor sets per pipeline: one for single op, one for second op in pair
    Pipeline* pipes[N_PIPE] = {
        &pipe_f32_, &pipe_f16_, &pipe_q8_0_, &pipe_q4k_, &pipe_q5k_, &pipe_q6k_, &pipe_nq4g64_, &pipe_nq5g64_, &pipe_nq6g64_, &pipe_nq8g64_, &pipe_q2k_, &pipe_q3k_
    };
    for (int i = 0; i < N_PIPE; ++i) {
        if (!pipes[i]->pipeline) { reuse_ds_[i] = VK_NULL_HANDLE; reuse_ds2_[i] = VK_NULL_HANDLE; reuse_ds3_[i] = VK_NULL_HANDLE; continue; }
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = ds_pool_;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &pipes[i]->ds_layout;
        vkAllocateDescriptorSets(device_, &dsai, &reuse_ds_[i]);
        vkAllocateDescriptorSets(device_, &dsai, &reuse_ds2_[i]);
        vkAllocateDescriptorSets(device_, &dsai, &reuse_ds3_[i]);
    }
    ds_allocated_ = true;
    return true;
}

// ── Updated ensure_activation_buffers (y2 for paired dispatch) ───────────────
inline void VulkanBackend::ensure_activation_buffers(size_t x_size,
                                                       size_t y1_size,
                                                       size_t y2_size) {
    auto recreate = [&](GpuBuffer& b, size_t needed) {
        if (needed == 0 || b.size >= needed) return;
        free_buffer(b);
        b = alloc_buffer(needed,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer && b.memory && !b.mapped_ptr)
            vkMapMemory(device_, b.memory, 0, b.size, 0, &b.mapped_ptr);
    };
    recreate(act_x_,  x_size);
    recreate(act_y_,  y1_size);
    if (y2_size > 0) recreate(act_y2_, y2_size);
}

// Ensure act_y3_ exists (used by QKV triple dispatch)
inline void VulkanBackend::ensure_y3_buffer(size_t y3_size) {
    if (y3_size == 0 || act_y3_.size >= y3_size) return;
    free_buffer(act_y3_);
    act_y3_ = alloc_buffer(y3_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT   |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (act_y3_.buffer && act_y3_.memory && !act_y3_.mapped_ptr)
        vkMapMemory(device_, act_y3_.memory, 0, act_y3_.size, 0, &act_y3_.mapped_ptr);
}

// ── record_one: write descriptor bindings and record one dispatch ─────────────
// Does NOT submit. cmd must already be in recording state.
inline bool VulkanBackend::record_one(VkCommandBuffer cmd, VkDescriptorSet ds,
                                       Pipeline& pipe, GpuBuffer& W_buf,
                                       GpuBuffer& x_buf, GpuBuffer& y_buf,
                                       uint32_t rows, uint32_t cols, uint32_t n_blocks) {
    if (pipe.pipeline == VK_NULL_HANDLE || ds == VK_NULL_HANDLE) return false;

    VkDescriptorBufferInfo bufs[3] = {
        {W_buf.buffer, 0, VK_WHOLE_SIZE},
        {x_buf.buffer, 0, VK_WHOLE_SIZE},
        {y_buf.buffer, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = ds;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufs[i];
    }
    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    PushConstants pc{};
    pc.out_rows         = rows;
    pc.in_cols          = cols;
    pc.n_blocks_per_row = n_blocks;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipe.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    uint32_t groups = (rows + pipe.rows_per_group - 1) / pipe.rows_per_group;
    vkCmdDispatch(cmd, groups, 1, 1);
    return true;
}

// ── mat_vec: single op via reusable command buffer ────────────────────────────
inline bool VulkanBackend::mat_vec(const void* W, const float* x, float* y,
                                    size_t out_rows, size_t in_cols, QuantType qtype) {
    if (!ready_) return false;
    if (dec_int8_ && out_rows >= 1024 && dq_mat_vec(W, x, y, out_rows, in_cols, qtype)) return true;
    Pipeline* pipe = pipeline_for(qtype);
    if (!pipe || !pipe->pipeline) return false;

    size_t W_size; uint32_t n_blocks;
    if (!size_for(qtype, out_rows, in_cols, W_size, n_blocks)) return false;

    size_t x_size = in_cols  * sizeof(float);
    size_t y_size = out_rows * sizeof(float);
    ensure_activation_buffers(x_size, y_size);
    if (!act_x_.buffer || !act_y_.buffer) return false;
    if (!act_x_.mapped_ptr || !act_y_.mapped_ptr) return false;

    // Get pipeline index for reuse_ds_
    int pi = 0;
    switch (qtype) {
    case QuantType::F32:  pi=0; break; case QuantType::F16:  pi=1; break;
    case QuantType::Q8_0: pi=2; break; case QuantType::Q4_K: pi=3; break;
    case QuantType::Q5_K: pi=4; break; case QuantType::Q6_K: pi=5; break;
    case QuantType::NQ4_G64: pi=6; break;
    case QuantType::NQ5_G64: pi=7; break;
    case QuantType::NQ6_G64: pi=8; break;
    case QuantType::NQ8_G64: pi=9; break;
    case QuantType::Q2_K: pi=10; break; case QuantType::Q3_K: pi=11; break;
    default: return false;
    }
    VkDescriptorSet ds = ds_allocated_ ? reuse_ds_[pi] : VK_NULL_HANDLE;
    if (ds == VK_NULL_HANDLE) return false;

    // Upload x via persistent mapping — no vkMapMemory overhead
    memcpy(act_x_.mapped_ptr, x, x_size);

    GpuBuffer& W_buf = get_weight_buffer(W, W_size, qtype);
    if (!W_buf.buffer) return false;

    // Reset and record
    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &bi);
    record_one(reuse_cmd_, ds, *pipe, W_buf, act_x_, act_y_,
               (uint32_t)out_rows, (uint32_t)in_cols, n_blocks);
    vkEndCommandBuffer(reuse_cmd_);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();

    // Read y via persistent mapping
    memcpy(y, act_y_.mapped_ptr, y_size);
    return true;
}

// ── int8 dot-product (dp4a) prompt path helpers ──────────────────────────────
inline bool VulkanBackend::use_int8_mm(QuantType q, size_t ntok, size_t cols) const {
    return has_int8_dot_ && int8_mm_on_ && pipe_mm_nq4g64_dp4_.pipeline != VK_NULL_HANDLE && pipe_quant_q8_.pipeline != VK_NULL_HANDLE &&
           q == QuantType::NQ4_G64 && ntok >= 192 && (cols % 64) == 0;
}
inline bool VulkanBackend::int8_ensure(size_t ntok, size_t cols) {
    const size_t qb = ntok * cols, sb = ntok * (cols / 32) * sizeof(float);
    auto ens = [&](GpuBuffer& b, size_t need) -> bool {
        if (b.buffer != VK_NULL_HANDLE && b.size >= need) return true;
        free_buffer(b);
        b = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return b.buffer != VK_NULL_HANDLE && b.memory != VK_NULL_HANDLE;
    };
    return ens(mm_xq_, qb) && ens(mm_xs_, sb);
}
inline bool VulkanBackend::int8_alloc_ds(const Pipeline& p, VkBuffer* bufs, uint32_t n, VkDescriptorSet& out) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = ds_pool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &p.ds_layout;
    if (vkAllocateDescriptorSets(device_, &ai, &out) != VK_SUCCESS || out == VK_NULL_HANDLE) { out = VK_NULL_HANDLE; return false; }
    VkDescriptorBufferInfo bi[4]; VkWriteDescriptorSet ws[4]{};
    for (uint32_t i = 0; i < n; ++i) {
        bi[i] = { bufs[i], 0, VK_WHOLE_SIZE };
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; ws[i].dstSet = out; ws[i].dstBinding = i;
        ws[i].descriptorCount = 1; ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, n, ws, 0, nullptr);
    return true;
}
// Records: quantise X (f32 [ntok][cols]) into mm_xq_/mm_xs_, then a compute->compute barrier.  Caller must have called int8_ensure(ntok, cols).
inline bool VulkanBackend::int8_rec_quant(VkCommandBuffer cmd, VkBuffer xbuf, size_t ntok, size_t cols, VkDescriptorSet& ds) {
    VkBuffer b[3] = { xbuf, mm_xq_.buffer, mm_xs_.buffer };
    if (!int8_alloc_ds(pipe_quant_q8_, b, 3, ds)) return false;
    struct { uint32_t n_blocks, p0, p1, p2; } pc{ (uint32_t)(ntok * cols / 32), 0, 0, 0 };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_quant_q8_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_quant_q8_.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_quant_q8_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (pc.n_blocks + 255u) / 256u, 1, 1);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    return true;
}
// Records Y = W * Xq (int8 path) using the activations currently in mm_xq_/mm_xs_.  No barrier after: the caller adds what it needs.
inline bool VulkanBackend::int8_rec_mm(VkCommandBuffer cmd, VkBuffer wbuf, VkBuffer ybuf, size_t rows, size_t cols, uint32_t nb, size_t ntok, VkDescriptorSet& ds) {
    VkBuffer b[4] = { wbuf, mm_xq_.buffer, mm_xs_.buffer, ybuf };
    if (!int8_alloc_ds(pipe_mm_nq4g64_dp4_, b, 4, ds)) return false;
    struct { uint32_t out_rows, in_cols, n_blocks_per_row, n_tokens; } pc{ (uint32_t)rows, (uint32_t)cols, nb, (uint32_t)ntok };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_mm_nq4g64_dp4_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_mm_nq4g64_dp4_.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipe_mm_nq4g64_dp4_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, ((uint32_t)rows + 127u) / 128u, ((uint32_t)ntok + 127u) / 128u, 1);
    return true;
}

// ── Batched GEMM: Y = W * X over n_tokens columns, chunked to TOK_TILE ────────
// Standalone (not part of layer recording). Allocates its own HOST_VISIBLE X/Y
// staging buffers (separate from act_x_/act_y_ so single-token paths are
// untouched). One submit per token-chunk. This is the prefill primitive.
inline bool VulkanBackend::mat_mat(const void* W, const float* X, float* Y,
                                    size_t out_rows, size_t in_cols,
                                    size_t n_tokens, QuantType qtype) {
    ProfScope _prof_scope_(this, P_MAT_MAT);
    if (!ready_ || !W || !X || !Y || n_tokens == 0)
        return false;

    // Bound every dispatch and every activation buffer: long prompts are run in
    // token tiles against the SAME resident weight buffer. Without this, X/Y
    // device buffers grow with prompt length (hundreds of MB at ffn width) and
    // one huge dispatch can exceed the driver watchdog / free VRAM.
    {
        constexpr size_t MM_TOK_TILE = 512;
        if (n_tokens > MM_TOK_TILE) {
            static const bool no_overlap = std::getenv("NEXUS_MM_NOOVERLAP") != nullptr;
            if (!no_overlap && mat_mat_multi(W, X, Y, out_rows, in_cols, n_tokens, qtype))
                return true;
            for (size_t t0 = 0; t0 < n_tokens; t0 += MM_TOK_TILE) {
                const size_t cnt = std::min(MM_TOK_TILE, n_tokens - t0);
                if (!mat_mat(W, X + t0 * in_cols, Y + t0 * out_rows,
                             out_rows, in_cols, cnt, qtype))
                    return false;
            }
            return true;
        }
    }

    // Native NQ*_G64 batched (full-prompt / multi-token) pipelines.
    Pipeline* pipe = nullptr;
    switch (qtype) {
    case QuantType::Q8_0:    pipe = &pipe_mm_q8_0_; break;
    case QuantType::Q4_K:    pipe = &pipe_mm_q4k_; break;
    case QuantType::Q5_K:    pipe = &pipe_mm_q5k_; break;
    case QuantType::Q6_K:    pipe = &pipe_mm_q6k_; break;
    case QuantType::Q2_K:    pipe = &pipe_mm_q2k_; break;
    case QuantType::Q3_K:    pipe = &pipe_mm_q3k_; break;
    case QuantType::NQ4_G64:
        pipe = &pipe_mm_nq4g64_;
        { static const bool no_big = std::getenv("NEXUS_MM_NOBIG") != nullptr;
          if (!no_big && n_tokens >= 192 && (in_cols % 64) == 0 && pipe_mm_nq4g64_big_.pipeline != VK_NULL_HANDLE) pipe = &pipe_mm_nq4g64_big_; }
        break;
    case QuantType::NQ5_G64: pipe = &pipe_mm_nq5g64_; break;
    case QuantType::NQ6_G64: pipe = &pipe_mm_nq6g64_; break;
    case QuantType::NQ8_G64: pipe = &pipe_mm_nq8g64_; break;
    default: return false;
    }
    if (!pipe || pipe->pipeline == VK_NULL_HANDLE)
        return false;

    size_t W_size = 0;
    uint32_t n_blocks = 0;
    if (!size_for(qtype, out_rows, in_cols, W_size, n_blocks))
        return false;

    GpuBuffer* W_buf_ptr = &get_weight_buffer(W, W_size, qtype);
    if (W_buf_ptr->buffer == VK_NULL_HANDLE) {
        static int mm_w_fail_log = 0;
        if (mm_w_fail_log < 12) {
            fprintf(stderr,
                "[vulkan:mm] RETRY after weight upload/cache failure rows=%zu cols=%zu ntok=%zu type=%d w=%.1fMB cache=%zu/%zuMB\n",
                out_rows, in_cols, n_tokens, (int)qtype,
                (double)W_size / 1048576.0,
                weight_cache_used_ / 1048576,
                weight_cache_budget_ / 1048576);
            fflush(stderr);
            ++mm_w_fail_log;
        }
        evict_unpinned_weights();
        W_buf_ptr = &get_weight_buffer(W, W_size, qtype);
    }
    if (W_buf_ptr->buffer == VK_NULL_HANDLE)
        return false;
    GpuBuffer& W_buf = *W_buf_ptr;

    const size_t x_bytes = in_cols  * n_tokens * sizeof(float);
    const size_t y_bytes = out_rows * n_tokens * sizeof(float);
    bool i8 = use_int8_mm(qtype, n_tokens, in_cols);
    Int8DsGuard i8g(this);

    auto ensure_gpu_buffer = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (need == 0) return false;
        if (b.buffer != VK_NULL_HANDLE && b.size >= need) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return b.buffer != VK_NULL_HANDLE && b.memory != VK_NULL_HANDLE;
    };

    auto ensure_stage_buffer = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (need == 0) return false;
        if (b.buffer != VK_NULL_HANDLE && b.size >= need && b.mapped_ptr != nullptr) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE || b.memory == VK_NULL_HANDLE) return false;
        if (!b.mapped_ptr) {
            if (vkMapMemory(device_, b.memory, 0, b.size, 0, &b.mapped_ptr) != VK_SUCCESS)
                return false;
        }
        return b.mapped_ptr != nullptr;
    };

    if (!ensure_stage_buffer(mm_x_stage_, x_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
        return false;
    if (!ensure_stage_buffer(mm_y_stage_, y_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT))
        return false;

    if (!ensure_gpu_buffer(mm_x_, x_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        static int mm_x_fail_log = 0;
        if (mm_x_fail_log < 12) {
            fprintf(stderr,
                "[vulkan:mm] RETRY after mm_x allocation failure rows=%zu cols=%zu ntok=%zu x=%.1fMB cache=%zu/%zuMB\n",
                out_rows, in_cols, n_tokens, (double)x_bytes / 1048576.0,
                weight_cache_used_ / 1048576,
                weight_cache_budget_ / 1048576);
            fflush(stderr);
            ++mm_x_fail_log;
        }
        evict_unpinned_weights();
        if (!ensure_gpu_buffer(mm_x_, x_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT))
            return false;
    }
    if (!ensure_gpu_buffer(mm_y_, y_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        static int mm_y_fail_log = 0;
        if (mm_y_fail_log < 12) {
            fprintf(stderr,
                "[vulkan:mm] RETRY after mm_y allocation failure rows=%zu cols=%zu ntok=%zu y=%.1fMB cache=%zu/%zuMB\n",
                out_rows, in_cols, n_tokens, (double)y_bytes / 1048576.0,
                weight_cache_used_ / 1048576,
                weight_cache_budget_ / 1048576);
            fflush(stderr);
            ++mm_y_fail_log;
        }
        evict_unpinned_weights();
        if (!ensure_gpu_buffer(mm_y_, y_bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT))
            return false;
    }

    if (i8 && !int8_ensure(n_tokens, in_cols)) i8 = false;     // no room for the int8 copy: use the float kernel

    // Upload the entire prompt activation block once for this matrix.
    memcpy(mm_x_stage_.mapped_ptr, X, x_bytes);

    VkDescriptorSet ds = VK_NULL_HANDLE;
    if (!i8) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = ds_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &pipe->ds_layout;
    if (vkAllocateDescriptorSets(device_, &ai, &ds) != VK_SUCCESS || ds == VK_NULL_HANDLE)
        return false;

    VkDescriptorBufferInfo bi[3] = {
        { W_buf.buffer, 0, VK_WHOLE_SIZE },
        { mm_x_.buffer, 0, VK_WHOLE_SIZE },
        { mm_y_.buffer, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet ws[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = ds;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 3, ws, 0, nullptr);
    }

    struct MMPush {
        uint32_t out_rows;
        uint32_t in_cols;
        uint32_t n_blocks_per_row;
        uint32_t n_tokens;
    };

    vkResetCommandBuffer(reuse_cmd_, 0);

    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) {
        vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
        return false;
    }

    // stage X -> device-local X
    VkBufferCopy xcopy{0, 0, x_bytes};
    vkCmdCopyBuffer(reuse_cmd_, mm_x_stage_.buffer, mm_x_.buffer, 1, &xcopy);

    VkBufferMemoryBarrier xbar{};
    xbar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    xbar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    xbar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    xbar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    xbar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    xbar.buffer = mm_x_.buffer;
    xbar.offset = 0;
    xbar.size = x_bytes;
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &xbar, 0, nullptr);

    MMPush pc{
        (uint32_t)out_rows,
        (uint32_t)in_cols,
        n_blocks,
        (uint32_t)n_tokens
    };

    if (!i8) {
    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe->layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    }

    // Full-prompt dispatch: one invocation computes one (row, token) dot product.
    // This replaces the old 16-token loop that did one submit+wait per tile.
    const uint32_t WG_ROWS = pipe->mm_tile_rows;
    const uint32_t WG_TOKS = pipe->mm_tile_toks;
    uint32_t gx = ((uint32_t)out_rows + WG_ROWS - 1u) / WG_ROWS;
    uint32_t gy = ((uint32_t)n_tokens + WG_TOKS - 1u) / WG_TOKS;
    if (i8) {
        if (!int8_rec_quant(reuse_cmd_, mm_x_.buffer, n_tokens, in_cols, i8g.s[0]) ||
            !int8_rec_mm(reuse_cmd_, W_buf.buffer, mm_y_.buffer, out_rows, in_cols, n_blocks, n_tokens, i8g.s[1])) {
            vkEndCommandBuffer(reuse_cmd_);
            return false;
        }
        gx = ((uint32_t)out_rows + 127u) / 128u; gy = ((uint32_t)n_tokens + 127u) / 128u;
    } else {
        vkCmdDispatch(reuse_cmd_, gx, gy, 1);
    }

    VkBufferMemoryBarrier ybar{};
    ybar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    ybar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ybar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ybar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ybar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ybar.buffer = mm_y_.buffer;
    ybar.offset = 0;
    ybar.size = y_bytes;
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &ybar, 0, nullptr);

    // device-local Y -> host staging Y
    VkBufferCopy ycopy{0, 0, y_bytes};
    vkCmdCopyBuffer(reuse_cmd_, mm_y_.buffer, mm_y_stage_.buffer, 1, &ycopy);

    if (vkEndCommandBuffer(reuse_cmd_) != VK_SUCCESS) {
        vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &reuse_cmd_;

    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) {
            vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
            return false;
        }
    }
    const auto mm_w0 = std::chrono::steady_clock::now();
    if (wait_reuse_fence() != VK_SUCCESS) {
        vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
        return false;
    }
    if (prof_on_) {
        const double w_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mm_w0).count();
        std::lock_guard<std::mutex> lk(mm_shape_mu_);
        auto& st = mm_shape_[std::make_tuple(out_rows, in_cols, (int)qtype)];
        st.ms += w_ms; st.calls += 1; st.tokens += n_tokens;
    }

    memcpy(Y, mm_y_stage_.mapped_ptr, y_bytes);
    vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);

    static int mm_done = 0;
    if (mm_done < 20) {
        fprintf(stderr,
            "[vulkan:mm] FULL_DISPATCH rows=%zu cols=%zu ntok=%zu groups=(%u,%u) x=%.1fMB y=%.1fMB\n",
            out_rows, in_cols, n_tokens, gx, gy,
            (double)x_bytes / 1048576.0,
            (double)y_bytes / 1048576.0);
        fflush(stderr);
        ++mm_done;
    }

    return true;
}

// ── Fused FFN for prompt reading (NQ4_G64): Y = Wdown * (silu(Wgate X) * (Wup X)) ──
// One submit per call (<= 512 tokens).  The gate/up/activation intermediates (n_tokens x ffn floats each) stay in VRAM, so there is no
// host activation pass and no PCIe round trip for them; only X goes up and Y comes back.
inline bool VulkanBackend::ffn_prefill(const void* Wg, const void* Wu, const void* Wd,
                                       const float* X, float* Y,
                                       size_t ffn, size_t hidden, size_t n_tokens, QuantType qtype) {
    ProfScope _prof_scope_(this, P_MAT_MAT);
    constexpr size_t TILE = 512;
    if (!ready_ || !Wg || !Wu || !Wd || !X || !Y || n_tokens == 0 || n_tokens > TILE) return false;
    if (qtype != QuantType::NQ4_G64 || (hidden % 64) || (ffn % 64) || !pipe_silu_.pipeline) return false;
    auto pick = [&]() -> Pipeline* {
        Pipeline* p = &pipe_mm_nq4g64_;
        static const bool no_big = std::getenv("NEXUS_MM_NOBIG") != nullptr;
        if (!no_big && n_tokens >= 192 && pipe_mm_nq4g64_big_.pipeline != VK_NULL_HANDLE) p = &pipe_mm_nq4g64_big_;
        return p->pipeline != VK_NULL_HANDLE ? p : nullptr;
    };
    Pipeline* pipe = pick();
    if (!pipe) return false;

    size_t wg_size = 0, wd_size = 0; uint32_t nb_h = 0, nb_f = 0;       // nb_h: blocks per row for cols=hidden; nb_f: cols=ffn
    if (!size_for(qtype, ffn, hidden, wg_size, nb_h)) return false;
    if (!size_for(qtype, hidden, ffn, wd_size, nb_f)) return false;

    const size_t x_bytes = hidden * n_tokens * sizeof(float);
    const size_t g_bytes = ffn    * n_tokens * sizeof(float);
    const size_t o_bytes = hidden * n_tokens * sizeof(float);

    auto ensure_dev = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (b.buffer != VK_NULL_HANDLE && b.size >= need) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return b.buffer != VK_NULL_HANDLE && b.memory != VK_NULL_HANDLE;
    };
    auto ensure_stage = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (b.buffer != VK_NULL_HANDLE && b.size >= need && b.mapped_ptr != nullptr) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE || b.memory == VK_NULL_HANDLE) return false;
        if (!b.mapped_ptr && vkMapMemory(device_, b.memory, 0, b.size, 0, &b.mapped_ptr) != VK_SUCCESS) return false;
        return b.mapped_ptr != nullptr;
    };
    const VkBufferUsageFlags xu = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags yu = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkBufferUsageFlags su = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (!ensure_stage(mm_x_stage_, x_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(mm_y_stage_, o_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(mm_x_, x_bytes, xu) || !ensure_dev(mm_y_, g_bytes, yu) ||
        !ensure_dev(ffn_u_, g_bytes, su) || !ensure_dev(ffn_o_, o_bytes, yu)) {
        evict_unpinned_weights();
        if (!ensure_dev(mm_x_, x_bytes, xu) || !ensure_dev(mm_y_, g_bytes, yu) ||
            !ensure_dev(ffn_u_, g_bytes, su) || !ensure_dev(ffn_o_, o_bytes, yu)) return false;
    }
    const bool i8 = use_int8_mm(qtype, n_tokens, hidden) && use_int8_mm(qtype, n_tokens, ffn) && int8_ensure(n_tokens, std::max(hidden, ffn));
    Int8DsGuard i8g(this);
    GpuBuffer* pg = &get_weight_buffer(Wg, wg_size, qtype);
    GpuBuffer* pu = &get_weight_buffer(Wu, wg_size, qtype);
    GpuBuffer* pd = &get_weight_buffer(Wd, wd_size, qtype);
    if (pg->buffer == VK_NULL_HANDLE || pu->buffer == VK_NULL_HANDLE || pd->buffer == VK_NULL_HANDLE) return false;

    memcpy(mm_x_stage_.mapped_ptr, X, x_bytes);

    VkDescriptorSet ds[4] = {};     // ds[] are freed by free_ds(); int8 sets by i8g
    auto free_ds = [&]() { for (auto& d : ds) if (d) { vkFreeDescriptorSets(device_, ds_pool_, 1, &d); d = VK_NULL_HANDLE; } };
    auto alloc_ds = [&](VkDescriptorSet& out, VkDescriptorSetLayout lay, VkBuffer* bufs, uint32_t n) -> bool {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = ds_pool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &lay;
        if (vkAllocateDescriptorSets(device_, &ai, &out) != VK_SUCCESS || out == VK_NULL_HANDLE) { out = VK_NULL_HANDLE; return false; }
        VkDescriptorBufferInfo bi[3]; VkWriteDescriptorSet ws[3]{};
        for (uint32_t i = 0; i < n; ++i) {
            bi[i] = { bufs[i], 0, VK_WHOLE_SIZE };
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; ws[i].dstSet = out; ws[i].dstBinding = i;
            ws[i].descriptorCount = 1; ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device_, n, ws, 0, nullptr);
        return true;
    };
    if (!i8) {
    { VkBuffer b[3] = { pg->buffer, mm_x_.buffer, mm_y_.buffer };   if (!alloc_ds(ds[0], pipe->ds_layout, b, 3)) { free_ds(); return false; } }
    { VkBuffer b[3] = { pu->buffer, mm_x_.buffer, ffn_u_.buffer };  if (!alloc_ds(ds[1], pipe->ds_layout, b, 3)) { free_ds(); return false; } }
    { VkBuffer b[3] = { pd->buffer, mm_y_.buffer, ffn_o_.buffer };  if (!alloc_ds(ds[3], pipe->ds_layout, b, 3)) { free_ds(); return false; } }
    }
    { VkBuffer b[2] = { mm_y_.buffer, ffn_u_.buffer };              if (!alloc_ds(ds[2], pipe_silu_.ds_layout, b, 2)) { free_ds(); return false; } }

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) { free_ds(); return false; }

    VkBufferCopy xcopy{0, 0, x_bytes};
    vkCmdCopyBuffer(reuse_cmd_, mm_x_stage_.buffer, mm_x_.buffer, 1, &xcopy);
    auto mem_barrier = [&](VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags sa, VkAccessFlags da) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER; mb.srcAccessMask = sa; mb.dstAccessMask = da;
        vkCmdPipelineBarrier(reuse_cmd_, src, dst, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    mem_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    const VkPipelineStageFlags CS = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    const VkAccessFlags WR = VK_ACCESS_SHADER_WRITE_BIT, RW = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    struct MMPush { uint32_t out_rows, in_cols, n_blocks_per_row, n_tokens; };
    auto mm = [&](VkDescriptorSet d, uint32_t rows, uint32_t cols, uint32_t nb) {
        MMPush pc{ rows, cols, nb, (uint32_t)n_tokens };
        vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
        vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout, 0, 1, &d, 0, nullptr);
        vkCmdPushConstants(reuse_cmd_, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(reuse_cmd_, (rows + pipe->mm_tile_rows - 1u) / pipe->mm_tile_rows,
                                  ((uint32_t)n_tokens + pipe->mm_tile_toks - 1u) / pipe->mm_tile_toks, 1);
    };
    if (i8) {       // quantise X once, then gate and up both read it
        if (!int8_rec_quant(reuse_cmd_, mm_x_.buffer, n_tokens, hidden, i8g.s[0]) ||
            !int8_rec_mm(reuse_cmd_, pg->buffer, mm_y_.buffer, ffn, hidden, nb_h, n_tokens, i8g.s[1]) ||
            !int8_rec_mm(reuse_cmd_, pu->buffer, ffn_u_.buffer, ffn, hidden, nb_h, n_tokens, i8g.s[2])) {
            vkEndCommandBuffer(reuse_cmd_); free_ds(); return false;
        }
    } else {
        mm(ds[0], (uint32_t)ffn, (uint32_t)hidden, nb_h);          // gate -> mm_y_
        mm(ds[1], (uint32_t)ffn, (uint32_t)hidden, nb_h);          // up   -> ffn_u_   (independent of gate: no barrier between)
    }
    mem_barrier(CS, CS, WR, RW);
    {
        struct { uint32_t n; uint32_t p[3]; } pc{ (uint32_t)(ffn * n_tokens), {0, 0, 0} };
        vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_silu_.pipeline);
        vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_silu_.layout, 0, 1, &ds[2], 0, nullptr);
        vkCmdPushConstants(reuse_cmd_, pipe_silu_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(reuse_cmd_, (uint32_t)((ffn * n_tokens + 255) / 256), 1, 1);
    }
    mem_barrier(CS, CS, WR, RW);
    if (i8) {
        if (!int8_rec_quant(reuse_cmd_, mm_y_.buffer, n_tokens, ffn, i8g.s[3]) ||
            !int8_rec_mm(reuse_cmd_, pd->buffer, ffn_o_.buffer, hidden, ffn, nb_f, n_tokens, i8g.s[4])) {
            vkEndCommandBuffer(reuse_cmd_); free_ds(); return false;
        }
    } else {
        mm(ds[3], (uint32_t)hidden, (uint32_t)ffn, nb_f);          // down: mm_y_ -> ffn_o_
    }
    mem_barrier(CS, VK_PIPELINE_STAGE_TRANSFER_BIT, WR, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy ycopy{0, 0, o_bytes};
    vkCmdCopyBuffer(reuse_cmd_, ffn_o_.buffer, mm_y_stage_.buffer, 1, &ycopy);
    if (vkEndCommandBuffer(reuse_cmd_) != VK_SUCCESS) { free_ds(); return false; }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) { free_ds(); return false; }
    }
    const auto w0 = std::chrono::steady_clock::now();
    const VkResult wr = wait_reuse_fence();
    free_ds();
    if (wr != VK_SUCCESS) return false;
    if (prof_on_) {
        const double w_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        std::lock_guard<std::mutex> lk(mm_shape_mu_);
        auto& st = mm_shape_[std::make_tuple(ffn, hidden, 9999)];      // 9999 = fused FFN (gate+up+act+down)
        st.ms += w_ms; st.calls += 1; st.tokens += n_tokens;
    }
    memcpy(Y, mm_y_stage_.mapped_ptr, o_bytes);
    return true;
}

// ── Pipelined multi-chunk GEMM: host copies overlap GPU compute ──────────────
// Same maths as running mat_mat() on 512-token chunks one after another, but while chunk i runs on the GPU the host copies
// chunk i+1's inputs into the second staging buffer and chunk i-1's results out of the other one.  Device buffers are shared
// (the GPU runs the chunks in order anyway), so VRAM use is unchanged; only host staging memory is doubled.
// Returns false (with no GPU work pending) if anything cannot be set up; the caller then falls back to the plain loop.
inline bool VulkanBackend::mat_mat_multi(const void* W, const float* X, float* Y,
                                         size_t out_rows, size_t in_cols,
                                         size_t n_tokens, QuantType qtype) {
    constexpr size_t TILE = 512;
    if (!ready_ || !W || !X || !Y || n_tokens <= TILE) return false;

    auto pick = [&](size_t nt) -> Pipeline* {
        Pipeline* p = nullptr;
        switch (qtype) {
        case QuantType::Q8_0:    p = &pipe_mm_q8_0_; break;
        case QuantType::Q4_K:    p = &pipe_mm_q4k_; break;
        case QuantType::Q5_K:    p = &pipe_mm_q5k_; break;
        case QuantType::Q6_K:    p = &pipe_mm_q6k_; break;
        case QuantType::Q2_K:    p = &pipe_mm_q2k_; break;
        case QuantType::Q3_K:    p = &pipe_mm_q3k_; break;
        case QuantType::NQ4_G64:
            p = &pipe_mm_nq4g64_;
            { static const bool no_big = std::getenv("NEXUS_MM_NOBIG") != nullptr;
              if (!no_big && nt >= 192 && (in_cols % 64) == 0 && pipe_mm_nq4g64_big_.pipeline != VK_NULL_HANDLE) p = &pipe_mm_nq4g64_big_; }
            break;
        case QuantType::NQ5_G64: p = &pipe_mm_nq5g64_; break;
        case QuantType::NQ6_G64: p = &pipe_mm_nq6g64_; break;
        case QuantType::NQ8_G64: p = &pipe_mm_nq8g64_; break;
        default: return nullptr;
        }
        return (p && p->pipeline != VK_NULL_HANDLE) ? p : nullptr;
    };
    if (!pick(TILE) || !pick(1)) return false;

    size_t W_size = 0; uint32_t n_blocks = 0;
    if (!size_for(qtype, out_rows, in_cols, W_size, n_blocks)) return false;
    GpuBuffer* Wp = &get_weight_buffer(W, W_size, qtype);
    if (Wp->buffer == VK_NULL_HANDLE) { evict_unpinned_weights(); Wp = &get_weight_buffer(W, W_size, qtype); }
    if (Wp->buffer == VK_NULL_HANDLE) return false;

    const size_t x_max = in_cols * TILE * sizeof(float);
    const size_t y_max = out_rows * TILE * sizeof(float);

    auto ensure_dev = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (b.buffer != VK_NULL_HANDLE && b.size >= need) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return b.buffer != VK_NULL_HANDLE && b.memory != VK_NULL_HANDLE;
    };
    auto ensure_stage = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (b.buffer != VK_NULL_HANDLE && b.size >= need && b.mapped_ptr != nullptr) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (b.buffer == VK_NULL_HANDLE || b.memory == VK_NULL_HANDLE) return false;
        if (!b.mapped_ptr && vkMapMemory(device_, b.memory, 0, b.size, 0, &b.mapped_ptr) != VK_SUCCESS) return false;
        return b.mapped_ptr != nullptr;
    };
    if (!ensure_stage(mm_x_stage_,   x_max, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(mm_x_stage_b_, x_max, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(mm_y_stage_,   y_max, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_stage(mm_y_stage_b_, y_max, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    const VkBufferUsageFlags xu = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkBufferUsageFlags yu = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!ensure_dev(mm_x_, x_max, xu)) { evict_unpinned_weights(); if (!ensure_dev(mm_x_, x_max, xu)) return false; }
    if (!ensure_dev(mm_y_, y_max, yu)) { evict_unpinned_weights(); if (!ensure_dev(mm_y_, y_max, yu)) return false; }
    // evicting may have invalidated the weight buffer reference: fetch it again now that the activation buffers exist
    Wp = &get_weight_buffer(W, W_size, qtype);
    if (Wp->buffer == VK_NULL_HANDLE) return false;
    GpuBuffer& W_buf = *Wp;

    const bool i8_ok = use_int8_mm(qtype, TILE, in_cols) && int8_ensure(TILE, in_cols);
    GpuBuffer* xs[2] = { &mm_x_stage_, &mm_x_stage_b_ };
    GpuBuffer* ys[2] = { &mm_y_stage_, &mm_y_stage_b_ };
    const size_t nchunks = (n_tokens + TILE - 1) / TILE;

    memcpy(xs[0]->mapped_ptr, X, std::min(TILE, n_tokens) * in_cols * sizeof(float));

    for (size_t c = 0; c < nchunks; ++c) {
        const size_t t0  = c * TILE;
        const size_t cnt = std::min(TILE, n_tokens - t0);
        const size_t xb  = cnt * in_cols  * sizeof(float);
        const size_t yb  = cnt * out_rows * sizeof(float);
        Pipeline* pipe = pick(cnt);
        const bool i8 = i8_ok && use_int8_mm(qtype, cnt, in_cols);
        Int8DsGuard i8g(this);
        GpuBuffer& xst = *xs[c & 1];
        GpuBuffer& yst = *ys[c & 1];

        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!i8) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &pipe->ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &ds) != VK_SUCCESS || ds == VK_NULL_HANDLE) return false;
        VkDescriptorBufferInfo bi[3] = {
            { W_buf.buffer, 0, VK_WHOLE_SIZE }, { mm_x_.buffer, 0, VK_WHOLE_SIZE }, { mm_y_.buffer, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet ws[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet = ds; ws[i].dstBinding = i; ws[i].descriptorCount = 1;
            ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(device_, 3, ws, 0, nullptr);
        }

        struct MMPush { uint32_t out_rows, in_cols, n_blocks_per_row, n_tokens; };
        MMPush pc{ (uint32_t)out_rows, (uint32_t)in_cols, n_blocks, (uint32_t)cnt };

        vkResetCommandBuffer(reuse_cmd_, 0);
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }

        VkBufferCopy xcopy{0, 0, xb};
        vkCmdCopyBuffer(reuse_cmd_, xst.buffer, mm_x_.buffer, 1, &xcopy);
        VkBufferMemoryBarrier xbar{};
        xbar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        xbar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; xbar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        xbar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; xbar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        xbar.buffer = mm_x_.buffer; xbar.offset = 0; xbar.size = xb;
        vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &xbar, 0, nullptr);

        if (i8) {
            if (!int8_rec_quant(reuse_cmd_, mm_x_.buffer, cnt, in_cols, i8g.s[0]) ||
                !int8_rec_mm(reuse_cmd_, W_buf.buffer, mm_y_.buffer, out_rows, in_cols, n_blocks, cnt, i8g.s[1])) {
                vkEndCommandBuffer(reuse_cmd_);
                return false;
            }
        } else {
            vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->pipeline);
            vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe->layout, 0, 1, &ds, 0, nullptr);
            vkCmdPushConstants(reuse_cmd_, pipe->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const uint32_t gx = ((uint32_t)out_rows + pipe->mm_tile_rows - 1u) / pipe->mm_tile_rows;
            const uint32_t gy = ((uint32_t)cnt + pipe->mm_tile_toks - 1u) / pipe->mm_tile_toks;
            vkCmdDispatch(reuse_cmd_, gx, gy, 1);
        }

        VkBufferMemoryBarrier ybar{};
        ybar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        ybar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; ybar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ybar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; ybar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ybar.buffer = mm_y_.buffer; ybar.offset = 0; ybar.size = yb;
        vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &ybar, 0, nullptr);
        VkBufferCopy ycopy{0, 0, yb};
        vkCmdCopyBuffer(reuse_cmd_, mm_y_.buffer, yst.buffer, 1, &ycopy);
        // make the device-side copy visible to the host reads that follow the fence
        if (vkEndCommandBuffer(reuse_cmd_) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1; si.pCommandBuffers = &reuse_cmd_;
        vkResetFences(device_, 1, &reuse_fence_);
        {
            std::lock_guard<std::mutex> qlk(queue_mu_);
            if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) {
                vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
                return false;
            }
        }

        // ---- host work that overlaps the GPU running chunk c ----
        if (c + 1 < nchunks) {
            const size_t nt0 = (c + 1) * TILE;
            const size_t ncnt = std::min(TILE, n_tokens - nt0);
            memcpy(xs[(c + 1) & 1]->mapped_ptr, X + nt0 * in_cols, ncnt * in_cols * sizeof(float));
        }
        if (c > 0) {
            const size_t pt0 = (c - 1) * TILE;
            memcpy(Y + pt0 * out_rows, ys[(c - 1) & 1]->mapped_ptr, TILE * out_rows * sizeof(float));
        }

        const auto w0 = std::chrono::steady_clock::now();
        const VkResult wr = wait_reuse_fence();
        vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
        if (wr != VK_SUCCESS) return false;
        if (prof_on_) {
            const double w_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
            std::lock_guard<std::mutex> lk(mm_shape_mu_);
            auto& st = mm_shape_[std::make_tuple(out_rows, in_cols, (int)qtype)];
            st.ms += w_ms; st.calls += 1; st.tokens += cnt;
        }
    }
    {
        const size_t lt0 = (nchunks - 1) * TILE;
        const size_t lcnt = n_tokens - lt0;
        memcpy(Y + lt0 * out_rows, ys[(nchunks - 1) & 1]->mapped_ptr, lcnt * out_rows * sizeof(float));
    }
    return true;
}

// ── Qwen3Next full-prompt depthwise causal conv ─────────────────────────────
inline bool VulkanBackend::qwen35_conv1d(const float* mixed,
                                          const float* conv_state,
                                          const float* conv_weight,
                                          float* conv_out,
                                          uint32_t n_tokens,
                                          uint32_t conv_dim,
                                          uint32_t kernel) {
    ProfScope _prof_scope_(this, P_CONV);
    if (!ready_ || !pipe_q35_conv_.pipeline || !mixed || !conv_state || !conv_weight || !conv_out)
        return false;
    if (n_tokens == 0 || conv_dim == 0 || kernel == 0)
        return false;

    const size_t mixed_bytes = (size_t)n_tokens * (size_t)conv_dim * sizeof(float);
    const size_t state_bytes = (size_t)conv_dim * (size_t)(kernel > 0 ? kernel - 1 : 0) * sizeof(float);
    const size_t weight_bytes = (size_t)conv_dim * (size_t)kernel * sizeof(float);
    const size_t out_bytes = mixed_bytes;

    auto ensure_dev = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (need == 0) return true;
        if (b.buffer && b.size >= need) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return b.buffer != VK_NULL_HANDLE && b.memory != VK_NULL_HANDLE;
    };
    auto ensure_stage = [&](GpuBuffer& b, size_t need, VkBufferUsageFlags usage) -> bool {
        if (need == 0) return true;
        if (b.buffer && b.size >= need && b.mapped_ptr) return true;
        free_buffer(b);
        b = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!b.buffer || !b.memory) return false;
        if (!b.mapped_ptr && vkMapMemory(device_, b.memory, 0, b.size, 0, &b.mapped_ptr) != VK_SUCCESS) return false;
        return b.mapped_ptr != nullptr;
    };

    if (!ensure_stage(q35_conv_mixed_stage_, mixed_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_conv_state_stage_, state_bytes ? state_bytes : 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_conv_out_stage_, out_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_conv_mixed_, mixed_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_conv_state_, state_bytes ? state_bytes : 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_conv_out_, out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;

    std::memcpy(q35_conv_mixed_stage_.mapped_ptr, mixed, mixed_bytes);
    if (state_bytes) std::memcpy(q35_conv_state_stage_.mapped_ptr, conv_state, state_bytes);

    GpuBuffer& wbuf = get_weight_buffer(conv_weight, weight_bytes, QuantType::F32);
    if (!wbuf.buffer) return false;

    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = ds_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &pipe_q35_conv_.ds_layout;
    if (vkAllocateDescriptorSets(device_, &ai, &ds) != VK_SUCCESS || !ds) return false;

    VkDescriptorBufferInfo bi[4] = {
        { q35_conv_mixed_.buffer, 0, VK_WHOLE_SIZE },
        { q35_conv_state_.buffer, 0, VK_WHOLE_SIZE },
        { wbuf.buffer, 0, VK_WHOLE_SIZE },
        { q35_conv_out_.buffer, 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet ws[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = ds;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, ws, 0, nullptr);

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }

    VkBufferCopy c1{0,0,mixed_bytes};
    vkCmdCopyBuffer(reuse_cmd_, q35_conv_mixed_stage_.buffer, q35_conv_mixed_.buffer, 1, &c1);
    if (state_bytes) {
        VkBufferCopy c2{0,0,state_bytes};
        vkCmdCopyBuffer(reuse_cmd_, q35_conv_state_stage_.buffer, q35_conv_state_.buffer, 1, &c2);
    }
    VkBufferMemoryBarrier inbars[2]{};
    inbars[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    inbars[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    inbars[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    inbars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inbars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    inbars[0].buffer = q35_conv_mixed_.buffer;
    inbars[0].offset = 0;
    inbars[0].size = mixed_bytes;
    inbars[1] = inbars[0];
    inbars[1].buffer = q35_conv_state_.buffer;
    inbars[1].size = state_bytes ? state_bytes : 4;
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 2, inbars, 0, nullptr);

    struct ConvPC { uint32_t n_tokens, conv_dim, kernel, pad0; } pc{ n_tokens, conv_dim, kernel, 0 };
    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_q35_conv_.pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_q35_conv_.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, pipe_q35_conv_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(reuse_cmd_, (n_tokens + 15u) / 16u, (conv_dim + 15u) / 16u, 1);

    VkBufferMemoryBarrier ob{};
    ob.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    ob.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    ob.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    ob.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ob.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ob.buffer = q35_conv_out_.buffer;
    ob.offset = 0;
    ob.size = out_bytes;
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 1, &ob, 0, nullptr);
    VkBufferCopy co{0,0,out_bytes};
    vkCmdCopyBuffer(reuse_cmd_, q35_conv_out_.buffer, q35_conv_out_stage_.buffer, 1, &co);

    if (vkEndCommandBuffer(reuse_cmd_) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    { std::lock_guard<std::mutex> qlk(queue_mu_); if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; } }
    if (wait_reuse_fence() != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }
    std::memcpy(conv_out, q35_conv_out_stage_.mapped_ptr, out_bytes);
    vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);

    static int printed = 0;
    if (printed < 20) { fprintf(stderr, "[vulkan:q35_conv] GPU ntok=%u conv_dim=%u kernel=%u bytes=%.1fMB\n", n_tokens, conv_dim, kernel, (double)out_bytes/1048576.0); fflush(stderr); ++printed; }
    return true;
}

// ── Qwen3Next full-prompt GatedDeltaNet recurrent scan ──────────────────────
inline bool VulkanBackend::qwen35_scan(const float* conv_out,
                                        const float* a,
                                        const float* b,
                                        const float* z,
                                        const float* A_log,
                                        const float* dt,
                                        const float* norm,
                                        float* rec_state_inout,
                                        float* core_out,
                                        uint32_t n_tokens,
                                        uint32_t conv_dim,
                                        uint32_t key_dim,
                                        uint32_t value_dim,
                                        uint32_t kh,
                                        uint32_t vh,
                                        uint32_t hd,
                                        uint32_t repeat,
                                        bool grouped_layout,
                                        float eps) {
    ProfScope _prof_scope_(this, P_SCAN);
    if (!ready_ || !pipe_q35_scan_.pipeline || !conv_out || !a || !b || !z || !A_log || !dt || !norm || !rec_state_inout || !core_out)
        return false;
    if (n_tokens == 0 || conv_dim == 0 || key_dim == 0 || value_dim == 0 || kh == 0 || vh == 0 || hd == 0 || hd > 256 || repeat == 0)
        return false;
    // The register-resident scan (v3, 4 threads per state column) is ~6.5x faster than the original on a GTX 1060 (19.5 vs 128 us/token).
    // Default on; NEXUS_SCAN_NOREG=1 falls back to the original kernel.
    static const bool scan_reg = std::getenv("NEXUS_SCAN_NOREG") == nullptr;
    const Pipeline& scan_pipe = (scan_reg && hd == 128 && pipe_q35_scan_r_.pipeline) ? pipe_q35_scan_r_ : pipe_q35_scan_;

    const size_t conv_bytes  = (size_t)n_tokens * conv_dim * sizeof(float);
    const size_t a_bytes     = (size_t)n_tokens * vh * sizeof(float);
    const size_t b_bytes     = a_bytes;
    const size_t z_bytes     = (size_t)n_tokens * value_dim * sizeof(float);
    const size_t A_bytes     = (size_t)vh * sizeof(float);
    const size_t dt_bytes    = A_bytes;
    const size_t norm_bytes  = (size_t)hd * sizeof(float);
    const size_t state_bytes = (size_t)vh * hd * hd * sizeof(float);
    const size_t core_bytes  = (size_t)n_tokens * value_dim * sizeof(float);

    auto ensure_dev = [&](GpuBuffer& gb, size_t need, VkBufferUsageFlags usage) -> bool {
        if (gb.buffer && gb.size >= need) return true;
        free_buffer(gb);
        gb = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return gb.buffer != VK_NULL_HANDLE && gb.memory != VK_NULL_HANDLE;
    };
    auto ensure_stage = [&](GpuBuffer& gb, size_t need, VkBufferUsageFlags usage) -> bool {
        if (gb.buffer && gb.size >= need && gb.mapped_ptr) return true;
        free_buffer(gb);
        gb = alloc_buffer(need, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!gb.buffer || !gb.memory) return false;
        if (!gb.mapped_ptr && vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped_ptr) != VK_SUCCESS) return false;
        return gb.mapped_ptr != nullptr;
    };

    if (!ensure_stage(q35_scan_conv_stage_, conv_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_a_stage_, a_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_b_stage_, b_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_z_stage_, z_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_A_stage_, A_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_dt_stage_, dt_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_norm_stage_, norm_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;
    if (!ensure_stage(q35_scan_state_stage_, state_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_stage(q35_scan_core_stage_, core_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;

    if (!ensure_dev(q35_scan_conv_, conv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_a_, a_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_b_, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_z_, z_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_A_, A_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_dt_, dt_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_norm_, norm_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_state_, state_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) return false;
    if (!ensure_dev(q35_scan_core_, core_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) return false;

    std::memcpy(q35_scan_conv_stage_.mapped_ptr, conv_out, conv_bytes);
    std::memcpy(q35_scan_a_stage_.mapped_ptr, a, a_bytes);
    std::memcpy(q35_scan_b_stage_.mapped_ptr, b, b_bytes);
    std::memcpy(q35_scan_z_stage_.mapped_ptr, z, z_bytes);
    std::memcpy(q35_scan_A_stage_.mapped_ptr, A_log, A_bytes);
    std::memcpy(q35_scan_dt_stage_.mapped_ptr, dt, dt_bytes);
    std::memcpy(q35_scan_norm_stage_.mapped_ptr, norm, norm_bytes);
    std::memcpy(q35_scan_state_stage_.mapped_ptr, rec_state_inout, state_bytes);

    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; ai.descriptorPool = ds_pool_; ai.descriptorSetCount = 1; ai.pSetLayouts = &scan_pipe.ds_layout;
    if (vkAllocateDescriptorSets(device_, &ai, &ds) != VK_SUCCESS || !ds) return false;
    VkDescriptorBufferInfo db[9] = {
        {q35_scan_conv_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_a_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_b_.buffer,0,VK_WHOLE_SIZE},
        {q35_scan_z_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_A_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_dt_.buffer,0,VK_WHOLE_SIZE},
        {q35_scan_norm_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_state_.buffer,0,VK_WHOLE_SIZE}, {q35_scan_core_.buffer,0,VK_WHOLE_SIZE}
    };
    VkWriteDescriptorSet ws[9]{};
    for (uint32_t i=0;i<9;++i){ ws[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; ws[i].dstSet=ds; ws[i].dstBinding=i; ws[i].descriptorCount=1; ws[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws[i].pBufferInfo=&db[i]; }
    vkUpdateDescriptorSets(device_, 9, ws, 0, nullptr);

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{}; cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }

    struct CopyPair { GpuBuffer* st; GpuBuffer* dev; size_t bytes; } copies[] = {
        {&q35_scan_conv_stage_, &q35_scan_conv_, conv_bytes}, {&q35_scan_a_stage_, &q35_scan_a_, a_bytes}, {&q35_scan_b_stage_, &q35_scan_b_, b_bytes},
        {&q35_scan_z_stage_, &q35_scan_z_, z_bytes}, {&q35_scan_A_stage_, &q35_scan_A_, A_bytes}, {&q35_scan_dt_stage_, &q35_scan_dt_, dt_bytes},
        {&q35_scan_norm_stage_, &q35_scan_norm_, norm_bytes}, {&q35_scan_state_stage_, &q35_scan_state_, state_bytes}
    };
    std::vector<VkBufferMemoryBarrier> bars;
    bars.reserve(8);
    for (auto& cp : copies) {
        VkBufferCopy bc{0,0,cp.bytes}; vkCmdCopyBuffer(reuse_cmd_, cp.st->buffer, cp.dev->buffer, 1, &bc);
        VkBufferMemoryBarrier br{}; br.sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; br.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; br.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT; br.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; br.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; br.buffer=cp.dev->buffer; br.offset=0; br.size=cp.bytes; bars.push_back(br);
    }
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, (uint32_t)bars.size(), bars.data(), 0, nullptr);

    struct ScanPC { uint32_t n_tokens, conv_dim, key_dim, value_dim, kh, vh, hd, repeat, grouped; float eps; } pc{
        n_tokens, conv_dim, key_dim, value_dim, kh, vh, hd, repeat, grouped_layout ? 1u : 0u, eps
    };
    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, scan_pipe.pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, scan_pipe.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, scan_pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(reuse_cmd_, vh, 1, 1);

    VkBufferMemoryBarrier outbars[2]{};
    outbars[0].sType=VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; outbars[0].srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; outbars[0].dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT; outbars[0].srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; outbars[0].dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; outbars[0].buffer=q35_scan_core_.buffer; outbars[0].offset=0; outbars[0].size=core_bytes;
    outbars[1]=outbars[0]; outbars[1].buffer=q35_scan_state_.buffer; outbars[1].size=state_bytes;
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 2, outbars, 0, nullptr);
    VkBufferCopy corecopy{0,0,core_bytes}; vkCmdCopyBuffer(reuse_cmd_, q35_scan_core_.buffer, q35_scan_core_stage_.buffer, 1, &corecopy);
    VkBufferCopy statecopy{0,0,state_bytes}; vkCmdCopyBuffer(reuse_cmd_, q35_scan_state_.buffer, q35_scan_state_stage_.buffer, 1, &statecopy);

    if (vkEndCommandBuffer(reuse_cmd_) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    { std::lock_guard<std::mutex> qlk(queue_mu_); if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; } }
    if (wait_reuse_fence() != VK_SUCCESS) { vkFreeDescriptorSets(device_, ds_pool_, 1, &ds); return false; }
    std::memcpy(core_out, q35_scan_core_stage_.mapped_ptr, core_bytes);
    std::memcpy(rec_state_inout, q35_scan_state_stage_.mapped_ptr, state_bytes);
    vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);

    static int printed = 0;
    if (printed < 20) { fprintf(stderr, "[vulkan:q35_scan] GPU ntok=%u vh=%u hd=%u core=%.1fMB state=%.1fMB\n", n_tokens, vh, hd, (double)core_bytes/1048576.0, (double)state_bytes/1048576.0); fflush(stderr); ++printed; }
    return true;
}

// ── mat_vec_pair: gate + up share x — one upload, two dispatches, one submit ──
inline bool VulkanBackend::mat_vec_pair(
    const void*  W1, float* y1, size_t rows1, QuantType qt1,
    const void*  W2, float* y2, size_t rows2, QuantType qt2,
    const float* x,  size_t in_cols)
{
    if (!ready_) return false;
    Pipeline* pipe1 = pipeline_for(qt1);
    Pipeline* pipe2 = pipeline_for(qt2);
    if (!pipe1 || !pipe1->pipeline || !pipe2 || !pipe2->pipeline) return false;

    size_t W1_size, W2_size; uint32_t nb1, nb2;
    if (!size_for(qt1, rows1, in_cols, W1_size, nb1)) return false;
    if (!size_for(qt2, rows2, in_cols, W2_size, nb2)) return false;

    int pi1=0, pi2=0;
    switch (qt1) { case QuantType::F32:pi1=0;break; case QuantType::F16:pi1=1;break;
                   case QuantType::Q8_0:pi1=2;break; case QuantType::Q4_K:pi1=3;break;
                   case QuantType::Q5_K:pi1=4;break; case QuantType::Q6_K:pi1=5;break;
                   case QuantType::NQ4_G64:pi1=6;break; case QuantType::NQ5_G64:pi1=7;break; case QuantType::NQ6_G64:pi1=8;break; case QuantType::NQ8_G64:pi1=9;break; case QuantType::Q2_K:pi1=10;break; case QuantType::Q3_K:pi1=11;break; default:return false; }
    switch (qt2) { case QuantType::F32:pi2=0;break; case QuantType::F16:pi2=1;break;
                   case QuantType::Q8_0:pi2=2;break; case QuantType::Q4_K:pi2=3;break;
                   case QuantType::Q5_K:pi2=4;break; case QuantType::Q6_K:pi2=5;break;
                   case QuantType::NQ4_G64:pi2=6;break; case QuantType::NQ5_G64:pi2=7;break; case QuantType::NQ6_G64:pi2=8;break; case QuantType::NQ8_G64:pi2=9;break; case QuantType::Q2_K:pi2=10;break; case QuantType::Q3_K:pi2=11;break; default:return false; }

    if (!ds_allocated_ || !reuse_ds_[pi1] || !reuse_ds2_[pi2]) return false;

    size_t x_size  = in_cols * sizeof(float);
    size_t y1_size = rows1   * sizeof(float);
    size_t y2_size = rows2   * sizeof(float);
    ensure_activation_buffers(x_size, y1_size, y2_size);
    if (!act_x_.buffer || !act_y_.buffer || !act_y2_.buffer) return false;
    if (!act_x_.mapped_ptr || !act_y_.mapped_ptr || !act_y2_.mapped_ptr) return false;

    // Upload x ONCE — no vkMapMemory, just direct memcpy to persistent mapping
    memcpy(act_x_.mapped_ptr, x, x_size);

    GpuBuffer& W1_buf = get_weight_buffer(W1, W1_size, qt1);
    GpuBuffer& W2_buf = get_weight_buffer(W2, W2_size, qt2);
    if (!W1_buf.buffer || !W2_buf.buffer) return false;

    // Record TWO dispatches in ONE command buffer — one submit
    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &bi);

    record_one(reuse_cmd_, reuse_ds_[pi1],  *pipe1, W1_buf, act_x_, act_y_,
               (uint32_t)rows1, (uint32_t)in_cols, nb1);

    // Memory barrier between the two dispatches (they write to different buffers, but same x)
    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    record_one(reuse_cmd_, reuse_ds2_[pi2], *pipe2, W2_buf, act_x_, act_y2_,
               (uint32_t)rows2, (uint32_t)in_cols, nb2);

    vkEndCommandBuffer(reuse_cmd_);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();

    // Read both results via persistent mapping — no vkMapMemory overhead
    memcpy(y1, act_y_.mapped_ptr,  y1_size);
    memcpy(y2, act_y2_.mapped_ptr, y2_size);
    return true;
}

// ── mat_vec_qkv: Q, K, V projections — three dispatches, ONE submit ──────────
inline bool VulkanBackend::mat_vec_qkv(
    const void*  wq, float* yq, size_t rows_q, QuantType tq,
    const void*  wk, float* yk, size_t rows_k, QuantType tk,
    const void*  wv, float* yv, size_t rows_v, QuantType tv,
    const float* x,  size_t in_cols)
{
    if (!ready_) return false;
    Pipeline* pq = pipeline_for(tq);
    Pipeline* pk = pipeline_for(tk);
    Pipeline* pv = pipeline_for(tv);
    if (!pq || !pq->pipeline || !pk || !pk->pipeline || !pv || !pv->pipeline) return false;

    size_t Wq_size, Wk_size, Wv_size; uint32_t nbq, nbk, nbv;
    if (!size_for(tq, rows_q, in_cols, Wq_size, nbq)) return false;
    if (!size_for(tk, rows_k, in_cols, Wk_size, nbk)) return false;
    if (!size_for(tv, rows_v, in_cols, Wv_size, nbv)) return false;

    auto qtype_to_pi = [](QuantType qt) -> int {
        switch (qt) {
        case QuantType::F32:  return 0; case QuantType::F16:  return 1;
        case QuantType::Q8_0: return 2; case QuantType::Q4_K: return 3;
        case QuantType::Q5_K: return 4; case QuantType::Q6_K: return 5;
        case QuantType::Q2_K: return 10; case QuantType::Q3_K: return 11;
        default: return -1;
        }
    };
    int piq = qtype_to_pi(tq), pik = qtype_to_pi(tk), piv = qtype_to_pi(tv);
    if (piq < 0 || pik < 0 || piv < 0) return false;
    if (!ds_allocated_ || !reuse_ds_[piq] || !reuse_ds2_[pik] || !reuse_ds3_[piv]) return false;

    size_t x_size  = in_cols * sizeof(float);
    size_t yq_size = rows_q  * sizeof(float);
    size_t yk_size = rows_k  * sizeof(float);
    size_t yv_size = rows_v  * sizeof(float);

    ensure_activation_buffers(x_size, yq_size, yk_size);
    ensure_y3_buffer(yv_size);
    if (!act_x_.buffer || !act_y_.buffer || !act_y2_.buffer || !act_y3_.buffer) return false;
    if (!act_x_.mapped_ptr || !act_y_.mapped_ptr || !act_y2_.mapped_ptr || !act_y3_.mapped_ptr) return false;

    memcpy(act_x_.mapped_ptr, x, x_size);  // upload x once

    GpuBuffer& Wq_buf = get_weight_buffer(wq, Wq_size, tq);
    GpuBuffer& Wk_buf = get_weight_buffer(wk, Wk_size, tk);
    GpuBuffer& Wv_buf = get_weight_buffer(wv, Wv_size, tv);
    if (!Wq_buf.buffer || !Wk_buf.buffer || !Wv_buf.buffer) return false;

    // Record THREE dispatches into ONE command buffer — single submit
    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &bi);

    VkMemoryBarrier barrier{};
    barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    record_one(reuse_cmd_, reuse_ds_[piq],  *pq, Wq_buf, act_x_, act_y_,
               (uint32_t)rows_q, (uint32_t)in_cols, nbq);
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    record_one(reuse_cmd_, reuse_ds2_[pik], *pk, Wk_buf, act_x_, act_y2_,
               (uint32_t)rows_k, (uint32_t)in_cols, nbk);
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

    record_one(reuse_cmd_, reuse_ds3_[piv], *pv, Wv_buf, act_x_, act_y3_,
               (uint32_t)rows_v, (uint32_t)in_cols, nbv);

    vkEndCommandBuffer(reuse_cmd_);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();

    memcpy(yq, act_y_.mapped_ptr,  yq_size);
    memcpy(yk, act_y2_.mapped_ptr, yk_size);
    memcpy(yv, act_y3_.mapped_ptr, yv_size);
    return true;
}

inline uint32_t VulkanBackend::find_memory_type(uint32_t type_bits,
                                                  VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

inline GpuBuffer VulkanBackend::alloc_buffer(size_t size, VkBufferUsageFlags usage,
                                               VkMemoryPropertyFlags mem_flags) {
    GpuBuffer b{};
    b.size = size;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // Shaders read quantised weights as 32-bit words.  A tensor whose byte size is not a multiple of 4 (34-byte NQ4 records x an odd
    // row count) would end in a half-covered word, and with bounds-checked buffer access the card then returns 0 for that whole word --
    // the last row would silently lose its last bytes.  Round the buffer itself up (b.size stays the caller's size).
    bci.size        = (size + 15) & ~(size_t)15;
    bci.usage       = usage;
    uint32_t queue_families[2] = { compute_family_, transfer_family_ };
    if (transfer_family_ != compute_family_) {
        bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bci.queueFamilyIndexCount = 2;
        bci.pQueueFamilyIndices = queue_families;
    } else {
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    if (vkCreateBuffer(device_, &bci, nullptr, &b.buffer) != VK_SUCCESS) return b;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);

    // Host-visible memory the CPU READS (results coming back from the card, buffers the shaders write and the CPU then copies out) must
    // be CACHED memory.  The plain host-visible type on NVIDIA/AMD discrete cards is write-combined: fast to fill, but every CPU read
    // goes uncached over PCIe (~1 GB/s), which put ~240 ms per layer of a 354-token prompt into copying results back.  Pure upload
    // staging (TRANSFER_SRC only) stays write-combined, which is what it wants.  NEXUS_NO_CACHED_HOST=1 turns this off.
    uint32_t mt = UINT32_MAX;
    {
        static const bool no_cached = (std::getenv("NEXUS_NO_CACHED_HOST") != nullptr);
        const bool plain_host = (mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                                !(mem_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        const bool upload_only = (usage == VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (plain_host && !upload_only && !no_cached) {
            mt = find_memory_type(req.memoryTypeBits, mem_flags | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) == 0)
                fprintf(stderr, "[vulkan] host-visible result buffers: %s memory\n", mt != UINT32_MAX ? "CACHED" : "uncached (no cached type offered)");
        }
    }
    if (mt == UINT32_MAX) mt = find_memory_type(req.memoryTypeBits, mem_flags);
    if (mt == UINT32_MAX) { vkDestroyBuffer(device_, b.buffer, nullptr); b.buffer = VK_NULL_HANDLE; return b; }

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(device_, &ai, nullptr, &b.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, b.buffer, nullptr); b.buffer = VK_NULL_HANDLE; return b;
    }
    vkBindBufferMemory(device_, b.buffer, b.memory, 0);
    return b;
}

inline GpuBuffer VulkanBackend::import_host_buffer(const void* ptr, size_t size) {
    GpuBuffer b{};
    if (!has_host_import_) return b;

    // Align pointer and size to min_host_align_
    uintptr_t base = (uintptr_t)ptr & ~((uintptr_t)min_host_align_ - 1);
    size_t offset  = (uintptr_t)ptr - base;
    size_t aligned_size = (size + offset + min_host_align_ - 1) & ~(size_t)(min_host_align_ - 1);

    // The imported buffer starts at the PAGE boundary below `ptr`, but every shader binding in this file
    // uses offset 0.  If `ptr` is not itself page-aligned the shader would read from the start of the page
    // (somebody else's data) instead of the start of the tensor.  Refuse the import in that case: the caller
    // then copies the tensor into a normal GPU buffer, which is always correct.
    if (offset != 0) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 8)
            fprintf(stderr, "[vulkan] zero-copy import skipped: tensor at %p (%zu KB) is not page-aligned (+%zu bytes) -- copying it instead\n",
                    ptr, size / 1024, offset);
        return b;
    }

    // Import the host memory
    VkImportMemoryHostPointerInfoEXT import_info{};
    import_info.sType        = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    import_info.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    import_info.pHostPointer = (void*)base;

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = aligned_size;
    ai.pNext           = &import_info;

    // Find a suitable host-visible memory type.  The spec only allows the types the driver lists for THIS pointer
    // (vkGetMemoryHostPointerPropertiesEXT); take the first host-visible one among them.
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_device_, &mp);
    uint32_t allowed_types = 0xFFFFFFFFu;
    {
        auto fn = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetInstanceProcAddr(instance_, "vkGetMemoryHostPointerPropertiesEXT");
        if (fn) {
            VkMemoryHostPointerPropertiesEXT hp{};
            hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
            if (fn(device_, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void*)base, &hp) == VK_SUCCESS && hp.memoryTypeBits)
                allowed_types = hp.memoryTypeBits;
        }
    }
    ai.memoryTypeIndex = 0;  // RADV uses type 0 for host-visible on APU
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((allowed_types & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            ai.memoryTypeIndex = i; break;
        }
    }

    if (vkAllocateMemory(device_, &ai, nullptr, &b.memory) != VK_SUCCESS) return b;

    // Create buffer view into imported memory
    VkExternalMemoryBufferCreateInfo ext_bci{};
    ext_bci.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext_bci.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext       = &ext_bci;
    bci.size        = aligned_size;
    bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bci, nullptr, &b.buffer) != VK_SUCCESS) {
        vkFreeMemory(device_, b.memory, nullptr); b.memory = VK_NULL_HANDLE; return b;
    }
    vkBindBufferMemory(device_, b.buffer, b.memory, 0);

    b.size      = aligned_size;
    b.imported  = true;
    b.host_ptr  = ptr;
    return b;
}

inline GpuBuffer& VulkanBackend::get_weight_buffer(const void* W, size_t size, QuantType /*qtype*/,
                                                    bool pin_on_create) {
    std::lock_guard<std::mutex> clk(weight_cache_mu_);
    auto it = weight_cache_.find(W);
    if (it != weight_cache_.end() && it->second.buf.buffer && it->second.buf.size < size) {
        // Cached smaller than what is asked for now (different code paths size the same matrix differently).  Reading past the
        // end of a buffer gives garbage, so throw the short copy away and fall through to a fresh, full-size upload.
        static std::atomic<int> healed{0};
        if (healed.fetch_add(1) < 8)
            fprintf(stderr, "[vulkan] weight %p was cached too small (%zu KB, %zu KB needed) -- re-uploading it at full size\n",
                    W, it->second.buf.size / 1024, size / 1024);
        pin_on_create = pin_on_create || it->second.pinned;
        device_idle_locked();
        if (it->second.size_bytes > 0) weight_cache_used_ -= it->second.size_bytes;
        free_buffer(it->second.buf);
        weight_cache_.erase(it);
        it = weight_cache_.end();
    }
    if (it != weight_cache_.end()) {
        it->second.last_use = ++weight_cache_clock_;
        if (pin_on_create) it->second.pinned = true;
        return it->second.buf;
    }

    // APU / unified-memory zero-copy path: no device allocation needed.
    if (info_.unified_memory && has_host_import_) {
        GpuBuffer imported = import_host_buffer(W, size);
        if (imported.buffer != VK_NULL_HANDLE) {
            CachedWeight cw;
            cw.buf        = imported;
            cw.size_bytes = 0;                       // doesn't count against budget
            cw.last_use   = ++weight_cache_clock_;
            cw.pinned     = pin_on_create;
            auto [it2, _] = weight_cache_.emplace(W, cw);
            return it2->second.buf;
        }
        fprintf(stderr, "[vulkan] host-import failed for %zu KB, falling back to copy\n",
                size / 1024);
    }

    // "Read from system RAM" mode: map the CPU's own copy into the card instead of copying it into VRAM.
    if (host_only_weights_) {
        GpuBuffer imported = has_host_import_ ? import_host_buffer(W, size) : GpuBuffer{};
        if (imported.buffer != VK_NULL_HANDLE) {
            CachedWeight cw;
            cw.buf        = imported;
            cw.size_bytes = 0;                       // lives in system RAM: does not count against VRAM
            cw.last_use   = ++weight_cache_clock_;
            cw.pinned     = pin_on_create;
            auto [it3, _] = weight_cache_.emplace(W, cw);
            return it3->second.buf;
        }
        // Could not map it (typically a small tensor that sits inside the model file at an odd address -- norms, conv kernels, tiny
        // projections).  Small ones are simply copied into VRAM (a few MB at most); only a big tensor that cannot be mapped fails the layer.
        const size_t kSmallCopyLimit = 32ull * 1024 * 1024;
        if (size > kSmallCopyLimit || (weight_cache_budget_ > 0 && weight_cache_used_ + size > weight_cache_budget_)) {
            static GpuBuffer empty_host{};
            return empty_host;
        }
        // fall through to the normal VRAM copy below
    }

    // Discrete GPU path: copy weights into device-local VRAM. Evict first if
    // the budget would be exceeded. Never evict to fit weights larger than
    // the entire budget — bail and let the caller fall back to CPU.
    if (weight_cache_budget_ > 0 && size > weight_cache_budget_) {
        fprintf(stderr,
            "[vulkan] weight (%zu MB) exceeds entire cache budget (%zu MB) — "
            "GPU upload refused, caller should CPU-fall-back\n",
            size / (1024*1024), weight_cache_budget_ / (1024*1024));
        static GpuBuffer empty{};
        return empty;
    }
    bool idled_for_evict = false;
    while (weight_cache_budget_ > 0 &&
           weight_cache_used_ + size > weight_cache_budget_) {
        // Find LRU unpinned victim.
        // Never a weight fetched in the last few requests: an operation fetches its weights one after another (gate, up, down; a
        // whole layer) and keeps references to the ones it already has. Evicting one of those -- which happened when the card was
        // nearly all pinned (Maximize GPU layers + MTP), so the weight just fetched was the only one left to evict -- left the
        // caller holding a freed buffer, and the graphics driver crashed on it (SIGSEGV in ffn_prefill).
        static const uint64_t kRecent = [] { const char* e = std::getenv("NEXUS_EVICT_RECENT"); int v = e ? std::atoi(e) : 0; return (uint64_t)(v > 0 ? v : 24); }();   // one step uses up to ~16 weights
        auto victim = weight_cache_.end();
        uint64_t oldest = UINT64_MAX;
        for (auto cit = weight_cache_.begin(); cit != weight_cache_.end(); ++cit) {
            if (cit->second.pinned)             continue;
            if (cit->second.last_use + kRecent > weight_cache_clock_) continue;   // in use by the operation asking right now
            if (cit->second.size_bytes == 0)    continue;  // imported, can't evict to reclaim VRAM
            if (cit->second.last_use < oldest) { oldest = cit->second.last_use; victim = cit; }
        }
        if (victim == weight_cache_.end()) {
            static std::atomic<int> full_logs{0};
            if (full_logs.fetch_add(1) < 8)
                fprintf(stderr,
                    "[vulkan] cache full and nothing it can evict (pinned layers, or weights the current step is using) -- "
                    "upload of %zu MB goes over the budget; if the card has no room left it runs on the CPU instead\n",
                    size / (1024*1024));
            break;  // proceed with upload anyway; OOM is the OS's problem
        }
        // Never free a buffer a queued command may still reference.
        if (!idled_for_evict) { device_idle_locked(); idled_for_evict = true; }
        weight_cache_used_ -= victim->second.size_bytes;
        free_buffer(victim->second.buf);
        weight_cache_.erase(victim);
    }

    GpuBuffer b = alloc_buffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (b.buffer == VK_NULL_HANDLE) {
        static GpuBuffer empty{};
        return empty;
    }
    if (!upload(b, W, size)) {
        // Never cache a device buffer that was not actually filled.
        fprintf(stderr, "[vulkan] weight upload FAILED (%zu MB) - not cached\n", size / (1024*1024));
        free_buffer(b);
        static GpuBuffer empty_failed{};
        return empty_failed;
    }

    CachedWeight cw;
    cw.buf        = b;
    cw.size_bytes = size;
    cw.last_use   = ++weight_cache_clock_;
    cw.pinned     = pin_on_create;
    weight_cache_used_ += size;
    auto [it2, _] = weight_cache_.emplace(W, cw);
    return it2->second.buf;
}

inline bool VulkanBackend::prefetch_weight_public(const void* W, size_t size, QuantType qt) {
    if (!ready_ || !W || size == 0) return false;

    std::lock_guard<std::mutex> clk(weight_cache_mu_);

    auto it = weight_cache_.find(W);
    if (it != weight_cache_.end()) {
        it->second.last_use = ++weight_cache_clock_;
        return it->second.buf.buffer != VK_NULL_HANDLE;
    }

    // Unified-memory path: import is cheap and does not count against VRAM.
    if (info_.unified_memory && has_host_import_) {
        GpuBuffer imported = import_host_buffer(W, size);
        if (imported.buffer != VK_NULL_HANDLE) {
            CachedWeight cw;
            cw.buf        = imported;
            cw.size_bytes = 0;
            cw.last_use   = ++weight_cache_clock_;
            cw.pinned     = false;
            weight_cache_.emplace(W, cw);
            return true;
        }
    }

    // Prefetch is opportunistic and must never evict weights that may be in use
    // by the main compute path. If it does not fit cleanly, skip it and let the
    // normal foreground get_weight_buffer() path handle eviction at the point of use.
    if (weight_cache_budget_ > 0) {
        if (size > weight_cache_budget_) return false;
        if (weight_cache_used_ + size > weight_cache_budget_) return false;
    }

    GpuBuffer b = alloc_buffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (b.buffer == VK_NULL_HANDLE) return false;

    if (!upload(b, W, size)) { free_buffer(b); return false; }

    CachedWeight cw;
    cw.buf        = b;
    cw.size_bytes = size;
    cw.last_use   = ++weight_cache_clock_;
    cw.pinned     = false;
    weight_cache_used_ += size;
    weight_cache_.emplace(W, cw);

    static int pre_logs = 0;
    if (pre_logs < 120) {
        fprintf(stderr, "[vulkan:prefetch] uploaded %zu MB qtype=%d cache=%zu/%zu MB\n",
                size / (1024*1024), (int)qt,
                weight_cache_used_ / (1024*1024),
                weight_cache_budget_ / (1024*1024));
        fflush(stderr);
        ++pre_logs;
    }
    return true;
}


inline bool VulkanBackend::load_weight_force_public(const void* W, size_t size, QuantType qt, bool pin_after_load) {
    if (!ready_ || !W || size == 0) return false;
    GpuBuffer& b = get_weight_buffer(W, size, qt, pin_after_load);
    if (b.buffer == VK_NULL_HANDLE) return false;
    return true;
}

inline bool VulkanBackend::load_weight_async_public(const void* W, size_t size, QuantType qt, bool pin_after_load) {
    if (!ready_ || !W || size == 0) return false;
    // Unified memory: import is cheap and needs no copy, so the ordinary path is fine.
    if (info_.unified_memory && has_host_import_)
        return load_weight_force_public(W, size, qt, pin_after_load);

    {   // 1. already cached? otherwise reserve budget while holding the lock briefly
        std::lock_guard<std::mutex> clk(weight_cache_mu_);
        auto it = weight_cache_.find(W);
        if (it != weight_cache_.end()) {
            it->second.last_use = ++weight_cache_clock_;
            if (pin_after_load) it->second.pinned = true;
            return it->second.buf.buffer != VK_NULL_HANDLE;
        }
        if (weight_cache_budget_ > 0 && weight_cache_used_ + size > weight_cache_budget_) {
            static std::atomic<int> warned{0};
            if (warned.fetch_add(1) < 3)
                fprintf(stderr, "[vulkan:async] %zu MB does not fit the weight budget (%zu/%zu MB) - "
                                "will load it synchronously when the layer starts\n",
                        size / (1024*1024), weight_cache_used_ / (1024*1024), weight_cache_budget_ / (1024*1024));
            return false;
        }
        weight_cache_used_ += size;     // reserve
    }
    // 2. allocate + copy with NO cache lock held (this is the slow part)
    GpuBuffer b = alloc_buffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bool ok = (b.buffer != VK_NULL_HANDLE) && upload(b, W, size);
    // 3. publish
    std::lock_guard<std::mutex> clk(weight_cache_mu_);
    if (!ok) {
        weight_cache_used_ -= size;
        if (b.buffer != VK_NULL_HANDLE) free_buffer(b);
        return false;
    }
    if (weight_cache_.find(W) != weight_cache_.end()) {   // someone else loaded it meanwhile
        weight_cache_used_ -= size;
        free_buffer(b);
        return true;
    }
    CachedWeight cw;
    cw.buf        = b;
    cw.size_bytes = size;
    cw.last_use   = ++weight_cache_clock_;
    cw.pinned     = pin_after_load;
    weight_cache_.emplace(W, cw);
    return true;
}

inline void VulkanBackend::pin_weight_public(const void* W) {
    if (!W) return;
    std::lock_guard<std::mutex> clk(weight_cache_mu_);
    auto it = weight_cache_.find(W);
    if (it != weight_cache_.end()) it->second.pinned = true;
}

inline void VulkanBackend::unpin_weight_public(const void* W) {
    if (!W) return;
    std::lock_guard<std::mutex> clk(weight_cache_mu_);
    auto it = weight_cache_.find(W);
    if (it != weight_cache_.end()) it->second.pinned = false;
}

inline bool VulkanBackend::evict_weight_public(const void* W) {
    if (!W) return false;
    std::lock_guard<std::mutex> clk(weight_cache_mu_);
    auto it = weight_cache_.find(W);
    if (it == weight_cache_.end()) return false;
    if (it->second.pinned) return false;
    device_idle_locked();   // buffer may still be referenced by queued work
    if (it->second.size_bytes > 0) weight_cache_used_ -= it->second.size_bytes;
    free_buffer(it->second.buf);
    weight_cache_.erase(it);
    return true;
}

inline void VulkanBackend::evict_unpinned_weights() {
    if (weight_cache_.empty()) return;
    // Wait for any in-flight work to finish before freeing buffers it may reference.
    device_idle_locked();
    size_t freed_bytes = 0;
    size_t freed_count = 0;
    for (auto it = weight_cache_.begin(); it != weight_cache_.end();) {
        if (it->second.pinned) { ++it; continue; }
        freed_bytes += it->second.size_bytes;
        if (it->second.size_bytes > 0) weight_cache_used_ -= it->second.size_bytes;
        free_buffer(it->second.buf);
        it = weight_cache_.erase(it);
        ++freed_count;
    }
    fprintf(stderr, "[vulkan] evicted %zu cache entries (%zu MB freed)\n",
            freed_count, freed_bytes / (1024*1024));
}

// ── GPU Attention ─────────────────────────────────────────────────────────────
// Per-kv-head K and V pointers from the contiguous KVCache.
// APU path: import_host_buffer → zero-copy, no staging at all.
// Discrete GPU path: copy K+V into a flat staging buffer (skip at long context).
inline bool VulkanBackend::attention_gpu(
    const float* q, float* out,
    int n_heads, int n_kv_heads, int head_dim,
    int /*layer*/, int n_ctx,
    const void* k_flat, const void* v_flat,
    size_t kv_bytes)
{
    ProfScope _prof_scope_(this, P_ATTN);
    if (!pipe_attn_.pipeline || !device_) return false;

    const size_t q_bytes = (size_t)n_heads * head_dim * sizeof(float);

    auto ensure_host = [&](GpuBuffer& buf, size_t need) -> bool {
        if (buf.buffer && buf.size >= need) return true;
        if (buf.buffer) { vkDestroyBuffer(device_,buf.buffer,nullptr); vkFreeMemory(device_,buf.memory,nullptr); buf={}; }
        buf = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        return buf.buffer != VK_NULL_HANDLE;
    };
    if (!ensure_host(attn_q_buf_,    q_bytes))   return false;
    if (!ensure_host(attn_out_buf_,  q_bytes))   return false;
    if (!ensure_host(attn_k_stage_,  kv_bytes))  return false;
    if (!ensure_host(attn_v_stage_,  kv_bytes))  return false;
    if (!attn_q_buf_.mapped_ptr || !attn_out_buf_.mapped_ptr) return false;
    if (!attn_k_stage_.mapped_ptr || !attn_v_stage_.mapped_ptr) return false;

    memcpy(attn_q_buf_.mapped_ptr,   q,       q_bytes);
    memcpy(attn_k_stage_.mapped_ptr, k_flat,  kv_bytes);
    memcpy(attn_v_stage_.mapped_ptr, v_flat,  kv_bytes);

    if (attn_ds_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &pipe_attn_.ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &attn_ds_) != VK_SUCCESS) return false;
    }

    GpuBuffer* bufs[4] = { &attn_q_buf_, &attn_k_stage_, &attn_v_stage_, &attn_out_buf_ };
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet   ws[4];
    for (int i = 0; i < 4; ++i) {
        bi[i] = { bufs[i]->buffer, 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = attn_ds_;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, ws, 0, nullptr);

    struct AttnPC {
        uint32_t n_heads, n_kv_heads, head_dim, n_ctx;
        float scale; uint32_t pad[3];
    } pc{ (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)n_ctx,
          1.0f/sqrtf((float)head_dim), {0,0,0} };

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &cbi);
    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_.pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_attn_.layout, 0, 1, &attn_ds_, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, pipe_attn_.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(reuse_cmd_, (uint32_t)n_heads, 1, 1);
    vkEndCommandBuffer(reuse_cmd_);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();

    memcpy(out, attn_out_buf_.mapped_ptr, q_bytes);
    return true;
}

inline bool VulkanBackend::attention_batch_gpu(
    const float* q_batch, float* out_batch,
    int n_q_heads, int n_kv_heads, int head_dim,
    int n_ctx, int n_query, int q_base_pos, int sliding_window,
    float scale, float softcap,
    const void* k_flat, const void* v_flat, size_t kv_bytes)
{
    ProfScope _prof_scope_(this, P_ATTN);
    if (!pipe_attn_batch_.pipeline || !device_) return false;

    // Query-tiled: each dispatch handles at most ATT_Q_TILE queries (causal masking
    // uses absolute positions, so slicing queries is exactly equivalent). Keeps
    // single dispatches short enough for the driver watchdog on long prompts.
    {
        constexpr int ATT_Q_TILE = 256;
        if (n_query > ATT_Q_TILE) {
            for (int q0 = 0; q0 < n_query; q0 += ATT_Q_TILE) {
                const int cnt = std::min(ATT_Q_TILE, n_query - q0);
                const size_t off = (size_t)q0 * n_q_heads * head_dim;
                if (!attention_batch_gpu(q_batch + off, out_batch + off,
                                         n_q_heads, n_kv_heads, head_dim,
                                         n_ctx, cnt, q_base_pos + q0, sliding_window,
                                         scale, softcap, k_flat, v_flat, kv_bytes))
                    return false;
            }
            return true;
        }
    }

    const size_t q_bytes = (size_t)n_query * n_q_heads * head_dim * sizeof(float);

    auto ensure_host = [&](GpuBuffer& buf, size_t need) -> bool {
        if (buf.buffer && buf.size >= need) return true;
        if (buf.buffer) {
            if (buf.mapped_ptr) { vkUnmapMemory(device_, buf.memory); buf.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            vkFreeMemory(device_, buf.memory, nullptr);
            buf = {};
        }
        buf = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (buf.buffer && buf.memory && !buf.mapped_ptr)
            vkMapMemory(device_, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped_ptr);
        return buf.buffer != VK_NULL_HANDLE;
    };
    // Reuses the SAME staging buffers as the single-token path — grows, never leaks.
    if (!ensure_host(attn_q_buf_,   q_bytes))  return false;
    if (!ensure_host(attn_out_buf_, q_bytes))  return false;
    if (!ensure_host(attn_k_stage_, kv_bytes)) return false;
    if (!ensure_host(attn_v_stage_, kv_bytes)) return false;
    if (!attn_q_buf_.mapped_ptr || !attn_out_buf_.mapped_ptr ||
        !attn_k_stage_.mapped_ptr || !attn_v_stage_.mapped_ptr) return false;

    memcpy(attn_q_buf_.mapped_ptr,   q_batch, q_bytes);
    memcpy(attn_k_stage_.mapped_ptr, k_flat,  kv_bytes);
    memcpy(attn_v_stage_.mapped_ptr, v_flat,  kv_bytes);

    if (attn_batch_ds_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &pipe_attn_batch_.ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &attn_batch_ds_) != VK_SUCCESS) return false;
    }

    GpuBuffer* bufs[4] = { &attn_q_buf_, &attn_k_stage_, &attn_v_stage_, &attn_out_buf_ };
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet   ws[4];
    for (int i = 0; i < 4; ++i) {
        bi[i] = { bufs[i]->buffer, 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = attn_batch_ds_;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, ws, 0, nullptr);

    struct AttnBatchPC {
        uint32_t n_q_heads, n_kv_heads, head_dim, n_ctx;
        uint32_t n_query, q_base_pos, sliding_window, pad0;
        float    scale, softcap; uint32_t pad1, pad2;
    } pc{ (uint32_t)n_q_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)n_ctx,
          (uint32_t)n_query, (uint32_t)q_base_pos, (uint32_t)sliding_window, 0u,
          (scale != 0.0f ? scale : 1.0f / sqrtf((float)head_dim)), softcap, 0u, 0u };

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &cbi);
    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_batch_.pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_attn_batch_.layout, 0, 1, &attn_batch_ds_, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, pipe_attn_batch_.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(reuse_cmd_, (uint32_t)n_q_heads, (uint32_t)n_query, 1);   // 2D grid
    vkEndCommandBuffer(reuse_cmd_);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();

    memcpy(out_batch, attn_out_buf_.mapped_ptr, q_bytes);
    return true;
}


inline bool VulkanBackend::vision_attention_full_gpu(const float* q, const float* k, const float* v, float* out,
                                                     int n_heads, int head_dim, int n_tokens, float scale)
{
    if (!pipe_vision_attn_full_.pipeline || !device_) return false;
    if (!q || !k || !v || !out) return false;
    if (n_heads <= 0 || head_dim <= 0 || n_tokens <= 0) return false;
    if (n_tokens > 65535 || n_heads > 65535) return false;

    // The current visual-attention shader keeps one head vector in registers.
    // Qwen visual uses hd=72.  Reject larger heads cleanly instead of compiling
    // a shader path that silently writes zeros for d >= MAX_HD.
    if (head_dim > 128) {
        fprintf(stderr,
                "[vision-gpu:attn] CPU-GPUFAIL reason=head_dim_gt_shader_max hd=%d max=128\n",
                head_dim);
        fflush(stderr);
        return false;
    }

    const size_t bytes = (size_t)n_tokens * (size_t)n_heads * (size_t)head_dim * sizeof(float);

    auto ensure_host = [&](GpuBuffer& buf, size_t need) -> bool {
        if (buf.buffer && buf.size >= need) return true;
        if (buf.buffer) {
            if (buf.mapped_ptr) { vkUnmapMemory(device_, buf.memory); buf.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            vkFreeMemory(device_, buf.memory, nullptr);
            buf = {};
        }
        buf = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (buf.buffer && buf.memory && !buf.mapped_ptr)
            vkMapMemory(device_, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped_ptr);
        return buf.buffer != VK_NULL_HANDLE;
    };

    if (!ensure_host(attn_q_buf_, bytes)) return false;
    if (!ensure_host(attn_k_stage_, bytes)) return false;
    if (!ensure_host(attn_v_stage_, bytes)) return false;
    if (!ensure_host(attn_out_buf_, bytes)) return false;
    if (!attn_q_buf_.mapped_ptr || !attn_k_stage_.mapped_ptr ||
        !attn_v_stage_.mapped_ptr || !attn_out_buf_.mapped_ptr) return false;

    memcpy(attn_q_buf_.mapped_ptr, q, bytes);
    memcpy(attn_k_stage_.mapped_ptr, k, bytes);
    memcpy(attn_v_stage_.mapped_ptr, v, bytes);

    if (attn_visual_full_ds_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &pipe_vision_attn_full_.ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &attn_visual_full_ds_) != VK_SUCCESS) return false;
    }

    GpuBuffer* bufs[4] = { &attn_q_buf_, &attn_k_stage_, &attn_v_stage_, &attn_out_buf_ };
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet ws[4];
    for (int i = 0; i < 4; ++i) {
        bi[i] = { bufs[i]->buffer, 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = attn_visual_full_ds_;
        ws[i].dstBinding = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, ws, 0, nullptr);

    struct VisionAttnPC {
        uint32_t n_heads, head_dim, n_tokens, d_model;
        float scale;
        uint32_t pad0, pad1, pad2;
    } pc{ (uint32_t)n_heads, (uint32_t)head_dim, (uint32_t)n_tokens,
          (uint32_t)(n_heads * head_dim),
          (scale != 0.0f ? scale : 1.0f / sqrtf((float)head_dim)), 0u, 0u, 0u };

    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(reuse_cmd_, &cbi) != VK_SUCCESS) return false;

    // Host wrote Q/K/V into mapped buffers immediately before submit.  Make that
    // visibility explicit for NVIDIA/Vulkan instead of relying on lucky coherent
    // behavior.  This mirrors the barrier used by the working layer-record path.
    VkMemoryBarrier host_to_shader{};
    host_to_shader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_to_shader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &host_to_shader, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_vision_attn_full_.pipeline);
    vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipe_vision_attn_full_.layout, 0, 1, &attn_visual_full_ds_, 0, nullptr);
    vkCmdPushConstants(reuse_cmd_, pipe_vision_attn_full_.layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(reuse_cmd_, (uint32_t)n_heads, (uint32_t)n_tokens, 1);

    // Make shader writes to the host-visible output buffer visible before the
    // CPU copies attn_out back after the fence.
    VkMemoryBarrier shader_to_host{};
    shader_to_host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    shader_to_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    shader_to_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(reuse_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &shader_to_host, 0, nullptr, 0, nullptr);

    VkResult end_res = vkEndCommandBuffer(reuse_cmd_);
    if (end_res != VK_SUCCESS) {
        fprintf(stderr, "[vision-gpu:attn] VK_ERROR end command buffer failed for vision_attn_full res=%d\n", (int)end_res);
        fflush(stderr);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        VkResult submit_res = vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
        if (submit_res != VK_SUCCESS) {
            fprintf(stderr, "[vision-gpu:attn] VK_ERROR queue submit failed for vision_attn_full res=%d\n", (int)submit_res);
            fflush(stderr);
            return false;
        }
    }
    VkResult wait_res = wait_reuse_fence();
    if (wait_res != VK_SUCCESS) {
        fprintf(stderr, "[vision-gpu:attn] VK_ERROR fence wait failed for vision_attn_full res=%d\n", (int)wait_res);
        fflush(stderr);
        return false;
    }

    memcpy(out, attn_out_buf_.mapped_ptr, bytes);
    return true;
}


inline bool VulkanBackend::attention_stream_gpu(
    const float* q_batch, float* out_batch,
    int n_q_heads, int n_kv_heads, int head_dim,
    int n_ctx, int n_query, int q_base_pos,
    float scale, float softcap,
    const void* k_flat, const void* v_flat, int chunk_tokens)
{
    ProfScope _prof_scope_(this, P_ATTN);
    if (!pipe_attn_stream_.pipeline || !device_ || n_ctx <= 0) return false;

    // Query-tiled (see attention_batch_gpu): bounded dispatch time and state size.
    {
        constexpr int ATT_Q_TILE = 256;
        if (n_query > ATT_Q_TILE) {
            for (int q0 = 0; q0 < n_query; q0 += ATT_Q_TILE) {
                const int cnt = std::min(ATT_Q_TILE, n_query - q0);
                const size_t off = (size_t)q0 * n_q_heads * head_dim;
                if (!attention_stream_gpu(q_batch + off, out_batch + off,
                                          n_q_heads, n_kv_heads, head_dim,
                                          n_ctx, cnt, q_base_pos + q0,
                                          scale, softcap, k_flat, v_flat, chunk_tokens))
                    return false;
            }
            return true;
        }
    }

    const size_t q_bytes  = (size_t)n_query * n_q_heads * head_dim * sizeof(float);
    const size_t st_elems = (size_t)n_query * n_q_heads * ((size_t)head_dim + 2);
    const size_t st_bytes = st_elems * sizeof(float);

    // Auto-size chunk to a VRAM target if not specified. KV chunk bytes for one
    // chunk = n_kv_heads * chunk * head_dim * 2 (f16) for each of K and V.
    int chunk = chunk_tokens;
    if (chunk <= 0) {
        const size_t target = (size_t)64 * 1024 * 1024;          // ~64MB per K (and per V)
        const size_t per_tok = (size_t)n_kv_heads * head_dim * sizeof(uint16_t);
        chunk = per_tok ? (int)(target / per_tok) : n_ctx;
        if (chunk < 256)   chunk = 256;
        if (chunk > n_ctx) chunk = n_ctx;
    }
    if (const char* e = std::getenv("NEXUS_KV_CHUNK_TOKENS")) {
        long v = std::atol(e);
        if (v > 0) chunk = (int)std::min<long>(v, n_ctx);
    }

    const size_t chunk_kv_bytes = (size_t)n_kv_heads * (size_t)chunk * head_dim * sizeof(uint16_t);

    auto ensure_host = [&](GpuBuffer& buf, size_t need) -> bool {
        if (buf.buffer && buf.size >= need) return true;
        if (buf.buffer) {
            if (buf.mapped_ptr) { vkUnmapMemory(device_, buf.memory); buf.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            vkFreeMemory(device_, buf.memory, nullptr);
            buf = {};
        }
        buf = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (buf.buffer && buf.memory && !buf.mapped_ptr)
            vkMapMemory(device_, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped_ptr);
        return buf.buffer != VK_NULL_HANDLE;
    };

    if (!ensure_host(attn_q_buf_,      q_bytes))        return false;
    if (!ensure_host(attn_out_buf_,    q_bytes))        return false;
    if (!ensure_host(attn_state_buf_,  st_bytes))       return false;
    if (!ensure_host(attn_kchunk_buf_, chunk_kv_bytes)) return false;
    if (!ensure_host(attn_vchunk_buf_, chunk_kv_bytes)) return false;
    if (!attn_q_buf_.mapped_ptr || !attn_out_buf_.mapped_ptr || !attn_state_buf_.mapped_ptr ||
        !attn_kchunk_buf_.mapped_ptr || !attn_vchunk_buf_.mapped_ptr) return false;

    // Upload Q once. Zero the state (m=-inf via large negative, l=0, acc=0).
    memcpy(attn_q_buf_.mapped_ptr, q_batch, q_bytes);
    {
        float* st = (float*)attn_state_buf_.mapped_ptr;
        const size_t per = (size_t)head_dim + 2;
        for (size_t i = 0; i < (size_t)n_query * n_q_heads; ++i) {
            float* s = st + i * per;
            for (int d = 0; d < head_dim; ++d) s[d] = 0.0f;
            s[head_dim]     = -1e30f;   // m
            s[head_dim + 1] = 0.0f;     // l
        }
    }

    if (attn_stream_ds_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &pipe_attn_stream_.ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &attn_stream_ds_) != VK_SUCCESS) return false;
    }

    GpuBuffer* bufs[5] = { &attn_q_buf_, &attn_kchunk_buf_, &attn_vchunk_buf_,
                           &attn_out_buf_, &attn_state_buf_ };
    VkDescriptorBufferInfo bi[5];
    VkWriteDescriptorSet   ws[5];
    for (int i = 0; i < 5; ++i) {
        bi[i] = { bufs[i]->buffer, 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = attn_stream_ds_;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 5, ws, 0, nullptr);

    struct StreamPC {
        uint32_t n_q_heads, n_kv_heads, head_dim, n_query;
        uint32_t q_base_pos, chunk_start, chunk_len, mode;
        float    scale, softcap; uint32_t pad0, pad1;
    };
    const float eff_scale = (scale != 0.0f) ? scale : 1.0f / sqrtf((float)head_dim);

    auto dispatch = [&](const StreamPC& pc) -> bool {
        vkResetCommandBuffer(reuse_cmd_, 0);
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(reuse_cmd_, &cbi);
        vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_stream_.pipeline);
        vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_attn_stream_.layout, 0, 1, &attn_stream_ds_, 0, nullptr);
        vkCmdPushConstants(reuse_cmd_, pipe_attn_stream_.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(reuse_cmd_, (uint32_t)n_q_heads, (uint32_t)n_query, 1);
        vkEndCommandBuffer(reuse_cmd_);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &reuse_cmd_;
        vkResetFences(device_, 1, &reuse_fence_);
        {
            std::lock_guard<std::mutex> qlk(queue_mu_);
            if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) return false;
        }
        return wait_reuse_fence() == VK_SUCCESS;
    };

    // Accumulate over KV chunks. Each chunk: repack the [chunk_start,chunk_end)
    // position slice contiguously per kv-head into the chunk staging buffer,
    // upload (it's host-visible, so the memcpy IS the upload), dispatch accumulate.
    const uint16_t* K = (const uint16_t*)k_flat;  // [n_kv_heads × n_ctx × head_dim]
    const uint16_t* V = (const uint16_t*)v_flat;
    uint16_t* Kdst = (uint16_t*)attn_kchunk_buf_.mapped_ptr;
    uint16_t* Vdst = (uint16_t*)attn_vchunk_buf_.mapped_ptr;

    for (int c0 = 0; c0 < n_ctx; c0 += chunk) {
        const int clen = std::min(chunk, n_ctx - c0);
        // Gather this position slice for every kv-head into contiguous chunk layout
        // [n_kv_heads × clen × head_dim].
        for (int kvh = 0; kvh < n_kv_heads; ++kvh) {
            const size_t src_base = ((size_t)kvh * n_ctx + c0) * head_dim;
            const size_t dst_base = (size_t)kvh * clen * head_dim;
            memcpy(Kdst + dst_base, K + src_base, (size_t)clen * head_dim * sizeof(uint16_t));
            memcpy(Vdst + dst_base, V + src_base, (size_t)clen * head_dim * sizeof(uint16_t));
        }
        StreamPC pc{ (uint32_t)n_q_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)n_query,
                     (uint32_t)q_base_pos, (uint32_t)c0, (uint32_t)clen, 0u,
                     eff_scale, softcap, 0u, 0u };
        if (!dispatch(pc)) return false;
    }

    // Finalize: out = acc / l.
    StreamPC fin{ (uint32_t)n_q_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)n_query,
                  (uint32_t)q_base_pos, 0u, 0u, 1u, eff_scale, softcap, 0u, 0u };
    if (!dispatch(fin)) return false;

    memcpy(out_batch, attn_out_buf_.mapped_ptr, q_bytes);
    return true;
}


inline bool VulkanBackend::attention_flash_gpu(
    const float* q_batch, float* out_batch,
    int n_q_heads, int n_kv_heads, int head_dim,
    int n_ctx, int n_query, int q_base_pos, int sliding_window,
    float scale, float softcap,
    const void* k_flat, const void* v_flat)
{
    ProfScope _prof_scope_(this, P_ATTN);
    static const bool disabled = std::getenv("NEXUS_ATTN_NOFLASH") != nullptr;
    if (disabled || !pipe_attn_flash_.pipeline || !device_) return false;
    if (head_dim <= 0 || head_dim > 256 || (head_dim % 64) != 0) return false;
    if (n_q_heads <= 0 || n_kv_heads <= 0 || (n_q_heads % n_kv_heads) != 0) return false;
    if (n_ctx <= 0 || n_query <= 0 || !q_batch || !out_batch || !k_flat || !v_flat) return false;

    const size_t kv_bytes = (size_t)n_kv_heads * (size_t)n_ctx * (size_t)head_dim * sizeof(uint16_t);
    size_t kv_limit = (size_t)384 * 1024 * 1024;   // K+V together
    if (const char* e = std::getenv("NEXUS_FLASH_KV_MB")) { long v = std::atol(e); if (v > 0) kv_limit = (size_t)v * 1024 * 1024; }
    if (2 * kv_bytes > kv_limit) return false;

    static const bool flash_log = std::getenv("NEXUS_FLASH_LOG") != nullptr;
    const auto fl_t0 = std::chrono::steady_clock::now();
    GpuBuffer kdev = alloc_buffer(kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    GpuBuffer vdev = alloc_buffer(kv_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::function<void()> release = [&]() { free_buffer(kdev); free_buffer(vdev); };
    if (!kdev.buffer || !kdev.memory || !vdev.buffer || !vdev.memory) { release(); return false; }
    if (!upload(kdev, k_flat, kv_bytes) || !upload(vdev, v_flat, kv_bytes)) { release(); return false; }

    const auto fl_t1 = std::chrono::steady_clock::now();
    double fl_gpu_ms = 0.0;
    // Query tiles bound the length of one dispatch (driver watchdog): roughly <= 2^26 (query, key) pairs per head-row of the grid.
    int q_tile = (int)(((size_t)1 << 26) / (size_t)n_ctx) & ~63;
    q_tile = std::max(64, std::min(1024, q_tile));
    const size_t q_tile_bytes = (size_t)q_tile * n_q_heads * head_dim * sizeof(float);

    auto ensure_host = [&](GpuBuffer& buf, size_t need) -> bool {
        if (buf.buffer && buf.size >= need) return true;
        if (buf.buffer) {
            if (buf.mapped_ptr) { vkUnmapMemory(device_, buf.memory); buf.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, buf.buffer, nullptr);
            vkFreeMemory(device_, buf.memory, nullptr);
            buf = {};
        }
        buf = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (buf.buffer && buf.memory && !buf.mapped_ptr)
            vkMapMemory(device_, buf.memory, 0, VK_WHOLE_SIZE, 0, &buf.mapped_ptr);
        return buf.buffer != VK_NULL_HANDLE;
    };
    const size_t first_tile_bytes = (size_t)std::min(q_tile, n_query) * n_q_heads * head_dim * sizeof(float);
    (void)q_tile_bytes;
    // Q lives in VRAM: the kernel re-reads it for every key tile, which would be PCIe traffic from host-visible memory.
    GpuBuffer qdev = alloc_buffer(first_tile_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // The kernel writes its result to VRAM (scattered 4-byte stores straight into host-visible memory crawl over PCIe); a copy then moves it to the host buffer.
    GpuBuffer odev = alloc_buffer(first_tile_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    release = [&]() { free_buffer(kdev); free_buffer(vdev); free_buffer(qdev); free_buffer(odev); };
    if (!qdev.buffer || !qdev.memory || !odev.buffer || !odev.memory || !ensure_host(attn_flash_host_, first_tile_bytes) || !attn_flash_host_.mapped_ptr ||
        !(attn_flash_host_.size >= first_tile_bytes))
        { release(); return false; }
    if (!(odev.buffer)) { release(); return false; }

    if (attn_flash_ds_ == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = ds_pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &pipe_attn_flash_.ds_layout;
        if (vkAllocateDescriptorSets(device_, &ai, &attn_flash_ds_) != VK_SUCCESS) { release(); return false; }
    }
    GpuBuffer* bufs[4] = { &qdev, &kdev, &vdev, &odev };
    VkDescriptorBufferInfo bi[4];
    VkWriteDescriptorSet   ws[4];
    for (int i = 0; i < 4; ++i) {
        bi[i] = { bufs[i]->buffer, 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = attn_flash_ds_;
        ws[i].dstBinding      = (uint32_t)i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(device_, 4, ws, 0, nullptr);

    struct FlashPC {
        uint32_t n_q_heads, n_kv_heads, head_dim, n_ctx;
        uint32_t n_query, q_base_pos, sliding_window, pad0;
        float    scale, softcap; uint32_t pad1, pad2;
    };
    const float eff_scale = (scale != 0.0f) ? scale : 1.0f / sqrtf((float)head_dim);

    bool ok = true;
    for (int t0 = 0; t0 < n_query && ok; t0 += q_tile) {
        const int cnt = std::min(q_tile, n_query - t0);
        const size_t qoff  = (size_t)t0 * n_q_heads * head_dim;
        const size_t qbytes = (size_t)cnt * n_q_heads * head_dim * sizeof(float);
        if (!upload(qdev, q_batch + qoff, qbytes)) { ok = false; break; }
        FlashPC pc{ (uint32_t)n_q_heads, (uint32_t)n_kv_heads, (uint32_t)head_dim, (uint32_t)n_ctx,
                    (uint32_t)cnt, (uint32_t)(q_base_pos + t0), (uint32_t)std::max(0, sliding_window), 0u,
                    eff_scale, softcap, 0u, 0u };
        vkResetCommandBuffer(reuse_cmd_, 0);
        VkCommandBufferBeginInfo cbi{};
        cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(reuse_cmd_, &cbi);
        vkCmdBindPipeline(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_flash_.pipeline);
        vkCmdBindDescriptorSets(reuse_cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipe_attn_flash_.layout, 0, 1, &attn_flash_ds_, 0, nullptr);
        vkCmdPushConstants(reuse_cmd_, pipe_attn_flash_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(reuse_cmd_, (uint32_t)((cnt + 63) / 64), (uint32_t)n_q_heads, 1);
        {
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
            VkBufferCopy oc{0, 0, qbytes};
            vkCmdCopyBuffer(reuse_cmd_, odev.buffer, attn_flash_host_.buffer, 1, &oc);
        }
        vkEndCommandBuffer(reuse_cmd_);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &reuse_cmd_;
        const auto fl_d0 = std::chrono::steady_clock::now();
        vkResetFences(device_, 1, &reuse_fence_);
        {
            std::lock_guard<std::mutex> qlk(queue_mu_);
            if (vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_) != VK_SUCCESS) { ok = false; break; }
        }
        if (wait_reuse_fence() != VK_SUCCESS) { ok = false; break; }
        fl_gpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fl_d0).count();
        memcpy(out_batch + qoff, attn_flash_host_.mapped_ptr, qbytes);
    }
    release();
    if (flash_log) {
        const auto fl_t2 = std::chrono::steady_clock::now();
        const double fl_flops = 4.0 * n_q_heads * head_dim * ((double)n_query * q_base_pos + 0.5 * (double)n_query * n_query);
        fprintf(stderr, "[flash-attn] ctx=%d nq=%d base=%d kv-upload=%.1f ms  kernel+readback=%.1f ms (%.0f GFLOP/s)  total=%.1f ms\n", n_ctx, n_query, q_base_pos,
                std::chrono::duration<double, std::milli>(fl_t1 - fl_t0).count(), fl_gpu_ms,
                fl_gpu_ms > 0 ? fl_flops / (fl_gpu_ms * 1e-3) / 1e9 : 0.0,
                std::chrono::duration<double, std::milli>(fl_t2 - fl_t0).count());
        fflush(stderr);
    }
    return ok;
}

// ── GPU layer pipeline: init, recording, helpers ─────────────────────────────

inline void VulkanBackend::init_activation_bufs(
    size_t hidden, size_t q_dim, size_t kv_dim, size_t ffn_dim)
{
    if (!device_) return;
    // HOST_VISIBLE|HOST_COHERENT so CPU can read/write directly.
    // On APU with unified memory, same memory is also DEVICE_LOCAL (free).
    auto mk = [&](GpuBuffer& b, size_t bytes) {
        if (b.buffer) {
            if (b.mapped_ptr) { vkUnmapMemory(device_, b.memory); b.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, b.buffer, nullptr);
            vkFreeMemory(device_, b.memory, nullptr); b = {};
        }
        b = alloc_buffer(bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        // Persistently map so CPU can read/write without vkMapMemory overhead.
        // VK_WHOLE_SIZE = map entire allocation (b.size may differ from actual
        // alloc size due to alignment padding — partial-range map can fail).
        if (b.buffer && b.memory) {
            VkResult map_res = vkMapMemory(device_, b.memory, 0,
                                           VK_WHOLE_SIZE, 0, &b.mapped_ptr);
            if (map_res != VK_SUCCESS) {
                fprintf(stderr, "[vulkan] WARNING: vkMapMemory failed (%d) for %zu bytes "
                        "— batched path will be disabled\n", (int)map_res, bytes);
                fflush(stderr);
                b.mapped_ptr = nullptr;
            }
        }
    };
    const size_t f = sizeof(float);
    mk(act_hidden_,    hidden  * f);
    mk(act_xb_,        hidden  * f);
    mk(act_q_,         q_dim   * f);
    mk(act_k_,         kv_dim  * f);
    mk(act_v_,         kv_dim  * f);
    mk(act_attn_out_2_, q_dim  * f);
    mk(act_gate_,      ffn_dim * f);
    mk(act_up_,        ffn_dim * f);
    mk(act_ffn_out_2_, hidden  * f);
    // Dummy 16-byte buffer for unused bindings
    mk(rec_dummy_, 16);

    // Allocate layer_cmd_ (separate from reuse_cmd_)
    if (layer_cmd_ == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmd_pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &ai, &layer_cmd_);
    }

    // Recording descriptor pool — reset each layer
    if (rec_pool_ == VK_NULL_HANDLE) {
        VkDescriptorPoolSize sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 }
        };
        VkDescriptorPoolCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.maxSets       = 64;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = sizes;
        vkCreateDescriptorPool(device_, &ci, nullptr, &rec_pool_);
    }

    // Require mapped_ptr too — batched path does CPU reads/writes on these buffers.
    // If vkMapMemory failed (e.g. driver limitation), disable batched path cleanly.
    act_bufs_ready_ = (act_hidden_.buffer   != VK_NULL_HANDLE &&
                       act_hidden_.mapped_ptr != nullptr        &&
                       act_xb_.mapped_ptr     != nullptr        &&
                       act_q_.mapped_ptr      != nullptr        &&
                       act_ffn_out_2_.mapped_ptr != nullptr);
}

// Allocate one descriptor set from rec_pool_ during recording
static VkDescriptorSet rec_alloc_ds(VkDevice dev, VkDescriptorPool pool,
                                    VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &layout;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &ai, &ds);
    return ds;
}

// Update storage buffer bindings for a descriptor set
static void rec_bind_bufs(VkDevice dev, VkDescriptorSet ds,
                          VkBuffer* bufs, uint32_t count)
{
    VkDescriptorBufferInfo bi[16];
    VkWriteDescriptorSet   ws[16];
    if (count > 16) count = 16;
    for (uint32_t i = 0; i < count; i++) {
        bi[i] = { bufs[i], 0, VK_WHOLE_SIZE };
        ws[i] = {};
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = ds;
        ws[i].dstBinding      = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ws[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(dev, count, ws, 0, nullptr);
}

inline void VulkanBackend::ops_prof_init()
{
    if (ops_prof_n_ != -1) return;
    ops_prof_n_ = 0;
    const char* e = getenv("NEXUS_OPS_PROF");
    if (!e || atoi(e) <= 0) return;
    VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(phys_device_, &p);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &nq, qf.data());
    if (compute_family_ >= nq || qf[compute_family_].timestampValidBits < 32 || p.limits.timestampPeriod <= 0.f) { fprintf(stderr, "[ops-prof] timestamps not supported on this queue\n"); return; }
    VkQueryPoolCreateInfo qi{}; qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO; qi.queryType = VK_QUERY_TYPE_TIMESTAMP; qi.queryCount = 32;
    if (vkCreateQueryPool(device_, &qi, nullptr, &ops_qp_) != VK_SUCCESS) return;
    ops_period_ns_ = p.limits.timestampPeriod;
    ops_prof_n_ = atoi(e);
    fprintf(stderr, "[ops-prof] per-phase timing on: averages printed every %d layers of each kind\n", ops_prof_n_);
}

inline void VulkanBackend::ops_prof_collect()
{
    uint32_t n = ops_ts_; if (n < 2) return;
    uint64_t t[32] = {};
    if (vkGetQueryPoolResults(device_, ops_qp_, 0, n, sizeof(t), t, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) return;
    int k = ops_kind_ ? 1 : 0;
    const uint32_t nl = n - 1;
    for (uint32_t i = 0; i < nl && i < 31; ++i) { ops_lab_[k][i] = ops_lab_cur_[i]; ops_acc_[k][i] += (double)(t[i + 1] - t[i]) * ops_period_ns_ * 1e-3; }   // us
    ops_acc_[k][31] += (double)(t[n - 1] - t[0]) * ops_period_ns_ * 1e-3;                                                                                       // total
    if (++ops_calls_[k] >= ops_prof_n_) {
        double c = (double)ops_calls_[k];
        fprintf(stderr, "[ops-prof] %s layer, average over %d layers (GPU time between timestamps, us):\n", k ? "full-attention" : "linear", ops_calls_[k]);
        for (uint32_t i = 0; i < nl; ++i) fprintf(stderr, "[ops-prof]   %-24s %7.1f us\n", ops_lab_[k][i] ? ops_lab_[k][i] : "?", ops_acc_[k][i] / c);
        fprintf(stderr, "[ops-prof]   %-24s %7.1f us  (phases: %u)\n", "TOTAL on GPU", ops_acc_[k][31] / c, n - 1);
        fflush(stderr);
        for (double& v : ops_acc_[k]) v = 0;
        ops_calls_[k] = 0;
    }
}

inline bool VulkanBackend::chain_begin(const float* x, size_t hidden)
{
    if (chain_open_ || layer_recording_ || !x || !ready_ || !act_hidden_.mapped_ptr) return false;
    ops_prof_init();
    if (ops_prof_n_ > 0) return false;                                 // per-phase timing needs the synchronous path
    static const bool off = [] { const char* e = std::getenv("NEXUS_GPU_CHAIN"); return e && e[0] == '0'; }();
    if (off || !layer_cmd_ || !rec_pool_) return false;
    if (!chain_ready_) {
        cslot_cmd_[0] = layer_cmd_; cslot_pool_[0] = rec_pool_;
        VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device_, &ai, &cslot_cmd_[1]) != VK_SUCCESS) return false;
        VkDescriptorPoolSize sz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 };
        VkDescriptorPoolCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; ci.maxSets = 64; ci.poolSizeCount = 1; ci.pPoolSizes = &sz;
        if (vkCreateDescriptorPool(device_, &ci, nullptr, &cslot_pool_[1]) != VK_SUCCESS) return false;
        VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        for (int i = 0; i < 2; ++i) if (vkCreateFence(device_, &fci, nullptr, &cslot_fence_[i]) != VK_SUCCESS) return false;
        chain_ready_ = true;
    }
    cslot_busy_[0] = cslot_busy_[1] = false; cslot_cur_ = 0; chain_tab_pos_ = -1;
    std::memcpy(act_hidden_.mapped_ptr, x, hidden * sizeof(float));
    chain_open_ = true;
    return true;
}

inline void VulkanBackend::chain_wait_all()
{
    if (!chain_ready_) return;
    for (int i = 0; i < 2; ++i) if (cslot_busy_[i]) { vkWaitForFences(device_, 1, &cslot_fence_[i], VK_TRUE, UINT64_MAX); cslot_busy_[i] = false; }
}

inline bool VulkanBackend::chain_end(float* x_out, size_t hidden)
{
    if (!chain_open_) return false;
    chain_wait_all();
    chain_open_ = false;
    layer_cmd_ = cslot_cmd_[0]; rec_pool_ = cslot_pool_[0];             // back to the synchronous slot
    if (x_out && act_hidden_.mapped_ptr) std::memcpy(x_out, act_hidden_.mapped_ptr, hidden * sizeof(float));
    return true;
}

inline bool VulkanBackend::chain_submit_layer() { return chain_submit_cmd(layer_cmd_, true); }

inline void VulkanBackend::lin_cache_drop(Q35LinCache& c)
{
    if (c.cmd) vkFreeCommandBuffers(device_, cmd_pool_, 1, &c.cmd);
    if (c.pool) vkDestroyDescriptorPool(device_, c.pool, nullptr);
    c = {};
}

inline bool VulkanBackend::chain_submit_cmd(VkCommandBuffer cmd, bool end)
{
    const int s = cslot_cur_;
    layer_recording_ = false;
    if (end) vkEndCommandBuffer(cmd);
    else if (cslot_busy_[s]) { vkWaitForFences(device_, 1, &cslot_fence_[s], VK_TRUE, UINT64_MAX); cslot_busy_[s] = false; }
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkResetFences(device_, 1, &cslot_fence_[s]);
    { std::lock_guard<std::mutex> qlk(queue_mu_); if (vkQueueSubmit(compute_queue_, 1, &si, cslot_fence_[s]) != VK_SUCCESS) return false; }
    cslot_busy_[s] = true; cslot_cur_ ^= 1;
    return true;
}

inline void VulkanBackend::begin_layer_record(const float* x_in, size_t hidden)
{
    if (chain_async_ && chain_open_) {                                 // async slot: wait only for the layer that used this slot two steps ago
        const int s = cslot_cur_;
        if (cslot_busy_[s]) { vkWaitForFences(device_, 1, &cslot_fence_[s], VK_TRUE, UINT64_MAX); cslot_busy_[s] = false; }
        VkCommandBufferBeginInfo cbi{}; cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (chain_rec_cmd_) {                                           // recording a reusable (cached) layer into its own fresh buffer + pool
            layer_cmd_ = chain_rec_cmd_; rec_pool_ = chain_rec_pool_;
            chain_rec_cmd_ = VK_NULL_HANDLE; chain_rec_pool_ = VK_NULL_HANDLE;
            cbi.flags = 0;
        } else {
            layer_cmd_ = cslot_cmd_[s]; rec_pool_ = cslot_pool_[s];
            vkResetDescriptorPool(device_, rec_pool_, 0);
            vkResetCommandBuffer(layer_cmd_, 0);
            cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        }
        vkBeginCommandBuffer(layer_cmd_, &cbi);
        rec_q8_src_ = VK_NULL_HANDLE;
        // everything the previous layer(s) wrote must be visible to this one (activations, hidden state, per-layer states)
        VkMemoryBarrier hb{}; hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(layer_cmd_, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
        layer_recording_ = true;
        return;
    }
    if (!layer_cmd_ || !rec_pool_) return;
    // Upload hidden state to GPU buffer
    if (!act_hidden_.mapped_ptr) {
        fprintf(stderr, "[vulkan] BUG: begin_layer_record called with null mapped_ptr\n");
        fflush(stderr); return;
    }
    if (x_in) memcpy(act_hidden_.mapped_ptr, x_in, hidden * sizeof(float));
    // Reset descriptor pool so all sets from last layer are freed
    vkResetDescriptorPool(device_, rec_pool_, 0);
    // Start recording
    vkResetCommandBuffer(layer_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(layer_cmd_, &cbi);
    rec_q8_src_ = VK_NULL_HANDLE;
    ops_prof_init();
    if (ops_prof_n_ > 0) { vkCmdResetQueryPool(layer_cmd_, ops_qp_, 0, 32); ops_ts_ = 0; vkCmdWriteTimestamp(layer_cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ops_qp_, ops_ts_++); }

    // Declare that the CPU (HOST) has written to activation buffers.
    // Without VK_ACCESS_HOST_WRITE_BIT in srcAccessMask, NVIDIA drivers
    // are allowed to assume host-visible buffers are unchanged and serve
    // stale cache. This barrier makes CPU writes visible to all shaders.
    VkMemoryBarrier host_barrier{};
    host_barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    host_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(layer_cmd_,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &host_barrier, 0, nullptr, 0, nullptr);

    layer_recording_ = true;
}

inline void VulkanBackend::partial_submit()
{
    if (!layer_recording_) return;
    vkEndCommandBuffer(layer_cmd_);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &layer_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();
    // Reset pool and restart recording
    vkResetDescriptorPool(device_, rec_pool_, 0);
    vkResetCommandBuffer(layer_cmd_, 0);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(layer_cmd_, &cbi);
    // HOST_WRITE barrier: CPU may write to mapped buffers between batches
    // Replace the existing HOST_WRITE barrier with one that covers
    // both CPU writes (for K/V staging) AND prior compute writes (for RoPE Q/K output)
    VkMemoryBarrier hb{};
    hb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(layer_cmd_,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &hb, 0, nullptr, 0, nullptr);
}

inline void VulkanBackend::end_layer_record(float* x_out, size_t hidden)
{
    if (!layer_recording_) return;
    layer_recording_ = false;
    if (x_out == nullptr) {
        // Abort: discard whatever was recorded without submitting
        vkResetCommandBuffer(layer_cmd_, 0);
        return;
    }
    if (ops_prof_n_ > 0 && ops_ts_ < 32) ops_lab_cur_[ops_ts_ - 1] = "residual add";
    if (ops_prof_n_ > 0 && ops_ts_ < 32) vkCmdWriteTimestamp(layer_cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ops_qp_, ops_ts_++);
    vkEndCommandBuffer(layer_cmd_);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &layer_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_);
    }
    wait_reuse_fence();
    if (ops_prof_n_ > 0) ops_prof_collect();
    // Download updated hidden state
    if (act_hidden_.mapped_ptr && x_out)
        memcpy(x_out, act_hidden_.mapped_ptr, hidden * sizeof(float));
}

inline void VulkanBackend::layer_barrier(const char* what)
{
    if (!layer_recording_) return;
    if (ops_prof_n_ > 0 && ops_ts_ < 31) { ops_lab_cur_[ops_ts_ - 1] = what ? what : "?"; vkCmdWriteTimestamp(layer_cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ops_qp_, ops_ts_++); }
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(layer_cmd_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Helper: record one compute dispatch into layer_cmd_
static void layer_dispatch(VkCommandBuffer cmd, VkPipeline pipe, VkPipelineLayout layout,
                            VkDescriptorSet ds, const void* pc, uint32_t pc_size,
                            uint32_t gx, uint32_t gy = 1, uint32_t gz = 1)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    if (pc && pc_size)
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(cmd, gx, gy, gz);
}

inline bool VulkanBackend::record_rms_norm_g(
    GpuBuffer& x_in, const void* w_ptr, size_t w_bytes,
    GpuBuffer& x_out, uint32_t n, float eps)
{
    rec_q8_src_ = VK_NULL_HANDLE;
    if (!layer_recording_ || !pipe_rms_.pipeline || !rec_pool_) return false;
    GpuBuffer& wbuf = get_weight_buffer(w_ptr, w_bytes, QuantType::F32);
    if (!wbuf.buffer) return false;
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_rms_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[3] = { x_in.buffer, wbuf.buffer, x_out.buffer };
    rec_bind_bufs(device_, ds, bufs, 3);
    struct { uint32_t n; float eps; uint32_t p[2]; } pc{ n, eps, {0,0} };
    // RMS norm shader uses ONE workgroup only — shared memory reduction requires
    // all threads to see all elements. The shader loops with stride 256 internally.
    layer_dispatch(layer_cmd_, pipe_rms_.pipeline, pipe_rms_.layout, ds, &pc, sizeof(pc), 1);
    return true;
}

inline bool VulkanBackend::record_mat_vec_g(
    const void* W, QuantType qt, GpuBuffer& x_in, GpuBuffer& y_out,
    size_t rows, size_t cols)
{
    if (!layer_recording_ || !rec_pool_) return false;
    Pipeline* pipe = pipeline_for(qt);
    if (!pipe || !pipe->pipeline) return false;
    size_t W_size; uint32_t n_blocks;
    if (!size_for(qt, rows, cols, W_size, n_blocks)) return false;
    GpuBuffer& wbuf = get_weight_buffer(W, W_size, qt);
    if (!wbuf.buffer) return false;
    if (rec_q8_src_ != VK_NULL_HANDLE && rec_q8_src_ == x_in.buffer && rec_q8_cols_ == cols) {   // input already quantised: int8 dot-product kernel
        Pipeline* dp = pipeline_dp4_for(qt);
        if (dp && dp->pipeline) {
            VkDescriptorSet dds = rec_alloc_ds(device_, rec_pool_, dp->ds_layout);
            if (!dds) return false;
            VkBuffer dbufs[4] = { wbuf.buffer, dq_xq_.buffer, dq_xs_.buffer, y_out.buffer };
            rec_bind_bufs(device_, dds, dbufs, 4);
            struct { uint32_t out_rows; uint32_t in_cols; uint32_t n_blocks; float pad; } dpc{ (uint32_t)rows, (uint32_t)cols, n_blocks, 0.f };
            layer_dispatch(layer_cmd_, dp->pipeline, dp->layout, dds, &dpc, sizeof(dpc), ((uint32_t)rows + dp->rows_per_group - 1) / dp->rows_per_group);
            return true;
        }
    }
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe->ds_layout);
    if (!ds) return false;
    VkBuffer bufs[3] = { wbuf.buffer, x_in.buffer, y_out.buffer };
    rec_bind_bufs(device_, ds, bufs, 3);
    struct { uint32_t out_rows; uint32_t in_cols; uint32_t n_blocks; float pad; }
        pc{ (uint32_t)rows, (uint32_t)cols, n_blocks, 0.f };
    uint32_t groups = ((uint32_t)rows + pipe->rows_per_group - 1) / pipe->rows_per_group;
    layer_dispatch(layer_cmd_, pipe->pipeline, pipe->layout, ds, &pc, sizeof(pc), groups);
    return true;
}

inline bool VulkanBackend::record_quant_act_g(GpuBuffer& x, size_t cols)
{
    rec_q8_src_ = VK_NULL_HANDLE;
    if (!dec_int8_ || !layer_recording_ || !rec_pool_ || cols == 0 || (cols & 255u) || cols > kDqMaxCols || !x.buffer) return false;
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_quant_q8n_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[3] = { x.buffer, dq_xq_.buffer, dq_xs_.buffer };
    rec_bind_bufs(device_, ds, bufs, 3);
    struct { uint32_t n_blocks, p0, p1, p2; } pc{ (uint32_t)(cols / 32), 0, 0, 0 };
    layer_dispatch(layer_cmd_, pipe_quant_q8n_.pipeline, pipe_quant_q8n_.layout, ds, &pc, sizeof(pc), (pc.n_blocks + 31u) / 32u);
    layer_barrier("quantise x");
    rec_q8_src_ = x.buffer; rec_q8_cols_ = cols;
    return true;
}

// Standalone int8 matvec (own submit): used for the lm_head.  Same contract as mat_vec().
inline bool VulkanBackend::dq_mat_vec(const void* W, const float* x, float* y, size_t out_rows, size_t in_cols, QuantType qt)
{
    Pipeline* dp = pipeline_dp4_for(qt);
    if (!dp || !dp->pipeline || !ready_ || layer_recording_ || (in_cols & 255u) || in_cols > kDqMaxCols) return false;
    size_t W_size; uint32_t n_blocks;
    if (!size_for(qt, out_rows, in_cols, W_size, n_blocks)) return false;
    ensure_activation_buffers(in_cols * sizeof(float), out_rows * sizeof(float));
    if (!act_x_.buffer || !act_y_.buffer || !act_x_.mapped_ptr || !act_y_.mapped_ptr) return false;
    GpuBuffer& W_buf = get_weight_buffer(W, W_size, qt);
    if (!W_buf.buffer) return false;
    memcpy(act_x_.mapped_ptr, x, in_cols * sizeof(float));
    const int di = qt == QuantType::Q4_K ? 1 : (qt == QuantType::Q5_K ? 2 : 3);
    VkBuffer qb[3] = { act_x_.buffer, dq_xq_.buffer, dq_xs_.buffer };
    VkBuffer mb4[4] = { W_buf.buffer, dq_xq_.buffer, dq_xs_.buffer, act_y_.buffer };
    rec_bind_bufs(device_, dq_ds_[0], qb, 3);
    rec_bind_bufs(device_, dq_ds_[di], mb4, 4);
    vkResetCommandBuffer(reuse_cmd_, 0);
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(reuse_cmd_, &bi);
    VkMemoryBarrier hb{}; hb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER; hb.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT; hb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
    struct { uint32_t n_blocks, p0, p1, p2; } qpc{ (uint32_t)(in_cols / 32), 0, 0, 0 };
    layer_dispatch(reuse_cmd_, pipe_quant_q8n_.pipeline, pipe_quant_q8n_.layout, dq_ds_[0], &qpc, sizeof(qpc), (qpc.n_blocks + 31u) / 32u);
    VkMemoryBarrier mb{}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(reuse_cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    struct { uint32_t out_rows, in_cols, n_blocks; float pad; } pc{ (uint32_t)out_rows, (uint32_t)in_cols, n_blocks, 0.f };
    layer_dispatch(reuse_cmd_, dp->pipeline, dp->layout, dq_ds_[di], &pc, sizeof(pc), ((uint32_t)out_rows + dp->rows_per_group - 1) / dp->rows_per_group);
    vkEndCommandBuffer(reuse_cmd_);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &reuse_cmd_;
    vkResetFences(device_, 1, &reuse_fence_);
    { std::lock_guard<std::mutex> qlk(queue_mu_); vkQueueSubmit(compute_queue_, 1, &si, reuse_fence_); }
    wait_reuse_fence();
    memcpy(y, act_y_.mapped_ptr, out_rows * sizeof(float));
    return true;
}

inline bool VulkanBackend::record_mat_vec_pair_g(
    const void* W1, QuantType qt1, GpuBuffer& y1,
    const void* W2, QuantType qt2, GpuBuffer& y2,
    size_t rows, GpuBuffer& x_in, size_t cols)
{
    // Two independent mat_vecs sharing x_in, recorded as consecutive dispatches
    bool ok = record_mat_vec_g(W1, qt1, x_in, y1, rows, cols);
    layer_barrier();
    ok &= record_mat_vec_g(W2, qt2, x_in, y2, rows, cols);
    return ok;
}

inline bool VulkanBackend::record_rope_g(
    GpuBuffer& q, GpuBuffer& k,
    uint32_t n_q, uint32_t n_kv, uint32_t hd,
    uint32_t pos, float theta, float scale, uint32_t rope_layout)
{
    if (!layer_recording_ || !pipe_rope_.pipeline || !rec_pool_) return false;
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_rope_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[2] = { q.buffer, k.buffer };
    rec_bind_bufs(device_, ds, bufs, 2);
    struct { uint32_t n_q, n_kv, hd, pos; float theta, scale; uint32_t p[2]; }
        pc{ n_q, n_kv, hd, pos, theta, scale, {rope_layout,0} };
    uint32_t groups = n_q + n_kv;
    layer_dispatch(layer_cmd_, pipe_rope_.pipeline, pipe_rope_.layout, ds, &pc, sizeof(pc), groups);
    return true;
}

inline bool VulkanBackend::record_attention_g(
    GpuBuffer& q, GpuBuffer& out,
    int n_heads, int n_kv_heads, int hd, int n_ctx,
    const uint16_t* k_flat, const uint16_t* v_flat, size_t kv_bytes)
{
    if (!layer_recording_ || !pipe_attn_.pipeline || !rec_pool_) return false;
    // Stage K and V (use the existing attn staging buffers)
    auto ensure_stage = [&](GpuBuffer& sb, size_t sz) -> bool {
        if (sb.buffer && sb.size >= sz) return true;
        if (sb.buffer) {
            if (sb.mapped_ptr) { vkUnmapMemory(device_, sb.memory); sb.mapped_ptr = nullptr; }
            vkDestroyBuffer(device_, sb.buffer, nullptr);
            vkFreeMemory(device_, sb.memory, nullptr);
            sb = {};
        }
        sb = alloc_buffer(sz,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (sb.buffer && sb.memory)   // ← add this
            vkMapMemory(device_, sb.memory, 0, VK_WHOLE_SIZE, 0, &sb.mapped_ptr);
        return sb.buffer != VK_NULL_HANDLE;
    };
    if (!ensure_stage(attn_k_stage_, kv_bytes)) return false;
    if (!ensure_stage(attn_v_stage_, kv_bytes)) return false;
    if (attn_k_stage_.mapped_ptr) memcpy(attn_k_stage_.mapped_ptr, k_flat, kv_bytes);
    if (attn_v_stage_.mapped_ptr) memcpy(attn_v_stage_.mapped_ptr, v_flat, kv_bytes);
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_attn_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[4] = { q.buffer, attn_k_stage_.buffer, attn_v_stage_.buffer, out.buffer };
    rec_bind_bufs(device_, ds, bufs, 4);
    struct { uint32_t n_heads, n_kv_heads, head_dim, n_ctx; float scale; uint32_t p[3]; }
        pc{ (uint32_t)n_heads, (uint32_t)n_kv_heads, (uint32_t)hd, (uint32_t)n_ctx,
            1.0f / sqrtf((float)hd), {0,0,0} };
    layer_dispatch(layer_cmd_, pipe_attn_.pipeline, pipe_attn_.layout, ds, &pc, sizeof(pc),
                   (uint32_t)n_heads);
    return true;
}

inline bool VulkanBackend::record_silu_mul_g(GpuBuffer& gate, GpuBuffer& up, uint32_t n)
{
    rec_q8_src_ = VK_NULL_HANDLE;
    if (!layer_recording_ || !pipe_silu_.pipeline || !rec_pool_) return false;
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_silu_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[2] = { gate.buffer, up.buffer };
    rec_bind_bufs(device_, ds, bufs, 2);
    struct { uint32_t n; uint32_t p[3]; } pc{ n, {0,0,0} };
    layer_dispatch(layer_cmd_, pipe_silu_.pipeline, pipe_silu_.layout, ds, &pc, sizeof(pc),
                   (n + 255) / 256);
    return true;
}

inline bool VulkanBackend::record_add_g(GpuBuffer& x, GpuBuffer& y, uint32_t n)
{
    if (!layer_recording_ || !pipe_add_.pipeline || !rec_pool_) return false;
    VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_add_.ds_layout);
    if (!ds) return false;
    VkBuffer bufs[2] = { x.buffer, y.buffer };
    rec_bind_bufs(device_, ds, bufs, 2);
    struct { uint32_t n; uint32_t p[3]; } pc{ n, {0,0,0} };
    layer_dispatch(layer_cmd_, pipe_add_.pipeline, pipe_add_.layout, ds, &pc, sizeof(pc),
                   (n + 255) / 256);
    return true;
}

// ── Qwen3.5 linear-attention layer: one token, one submit ──────────────────────────────────────────────────────────────────────────
inline bool VulkanBackend::q35_state_resident(const void* owner, int layer) const {
    return q35_dev_state_.find({owner, layer}) != q35_dev_state_.end();
}

inline void VulkanBackend::q35_state_drop(const void* owner, int layer) {
    if (!device_) return;
    for (auto it = q35_dev_state_.begin(); it != q35_dev_state_.end();) {
        if (it->first.first == owner && (layer < 0 || it->first.second == layer)) {
            device_idle_locked();
            free_buffer(it->second.conv); free_buffer(it->second.rec);
            it = q35_dev_state_.erase(it);
        } else ++it;
    }
}

inline bool VulkanBackend::q35_state_download(const void* owner, int layer, float* conv_out, float* rec_out) {
    auto it = q35_dev_state_.find({owner, layer});
    if (it == q35_dev_state_.end() || !ready_) return false;
    Q35DevState& st = it->second;
    const size_t cb = st.conv_n * sizeof(float), rb = st.rec_n * sizeof(float);
    auto ensure_stage = [&](GpuBuffer& gb, size_t need) -> bool {
        if (gb.buffer && gb.size >= need && gb.mapped_ptr) return true;
        free_buffer(gb);
        gb = alloc_buffer(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!gb.buffer || !gb.memory) return false;
        if (!gb.mapped_ptr && vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped_ptr) != VK_SUCCESS) return false;
        return gb.mapped_ptr != nullptr;
    };
    if (cb && !ensure_stage(q35_t_stage_conv_, cb)) return false;
    if (rb && !ensure_stage(q35_t_stage_rec_, rb)) return false;
    VkCommandBuffer cmd = begin_one_shot();
    VkMemoryBarrier mb{}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER; mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (cb) { VkBufferCopy c{0, 0, cb}; vkCmdCopyBuffer(cmd, st.conv.buffer, q35_t_stage_conv_.buffer, 1, &c); }
    if (rb) { VkBufferCopy c{0, 0, rb}; vkCmdCopyBuffer(cmd, st.rec.buffer,  q35_t_stage_rec_.buffer, 1, &c); }
    end_one_shot(cmd);
    if (cb && conv_out) std::memcpy(conv_out, q35_t_stage_conv_.mapped_ptr, cb);
    if (rb && rec_out)  std::memcpy(rec_out,  q35_t_stage_rec_.mapped_ptr,  rb);
    return true;
}

inline bool VulkanBackend::q35_linear_layer(const Q35LinearDesc& d, float* x, const float* conv_state_host, const float* rec_state_host, bool async) {
    if (!ready_ || layer_recording_ || !x || !rec_pool_ || !layer_cmd_) return false;
    if (async && ops_prof_n_ > 0) return false;
    if (!pipe_q35_conv_step_.pipeline || !pipe_rms_.pipeline || !pipe_silu_.pipeline || !pipe_add_.pipeline) return false;
    const bool reg = (d.hd == 128 && pipe_q35_scan_r_.pipeline);
    const Pipeline& scan_pipe = reg ? pipe_q35_scan_r_ : pipe_q35_scan_;
    if (!scan_pipe.pipeline || d.hd == 0 || d.hd > 256 || d.vh == 0 || d.kh == 0 || d.vh % d.kh != 0 || d.kernel < 1) return false;
    if (d.hidden == 0 || d.conv_dim == 0 || d.value_dim == 0 || d.key_dim == 0 || d.ffn == 0) return false;
    if (d.conv_dim != 2 * d.key_dim + d.value_dim || d.value_dim != d.vh * d.hd || d.key_dim != d.kh * d.hd) return false;
    if (!d.attn_norm || !d.ffn_norm || !d.qkv || !d.z || !d.a || !d.b || !d.out || !d.gate || !d.up || !d.down || !d.conv_w || !d.A || !d.dt || !d.gnorm) return false;
    if (!act_bufs_ok() || !act_hidden_.mapped_ptr) return false;
    if (act_xb_.size < (size_t)d.hidden * 4 || act_gate_.size < (size_t)d.ffn * 4 || act_up_.size < (size_t)d.ffn * 4 || act_ffn_out_2_.size < (size_t)d.hidden * 4) return false;

    const size_t hist = d.kernel - 1;
    const size_t conv_n = (size_t)d.conv_dim * hist, rec_n = (size_t)d.vh * d.hd * d.hd;

    // Every weight must be resident BEFORE recording starts (uploading mid-recording would wait on the queue we are recording for).
    struct W { const void* p; size_t rows, cols; QuantType t; };
    const W mats[] = { {d.qkv, d.conv_dim, d.hidden, d.qkv_t}, {d.z, d.value_dim, d.hidden, d.z_t}, {d.a, d.vh, d.hidden, d.a_t}, {d.b, d.vh, d.hidden, d.b_t},
                       {d.out, d.hidden, d.value_dim, d.out_t}, {d.gate, d.ffn, d.hidden, d.gate_t}, {d.up, d.ffn, d.hidden, d.up_t}, {d.down, d.hidden, d.ffn, d.down_t} };
    for (const W& m : mats) {
        Pipeline* pp = pipeline_for(m.t);
        size_t bytes = 0; uint32_t nb = 0;
        if (!pp || !pp->pipeline || !size_for(m.t, m.rows, m.cols, bytes, nb)) return false;
        if (!get_weight_buffer(m.p, bytes, m.t).buffer) return false;
    }
    const struct { const void* p; size_t n; } vecs[] = { {d.attn_norm, (size_t)d.hidden}, {d.ffn_norm, (size_t)d.hidden}, {d.conv_w, (size_t)d.conv_dim * d.kernel},
                                                         {d.A, d.vh}, {d.dt, d.vh}, {d.gnorm, d.hd} };
    for (auto& v : vecs) if (!get_weight_buffer(v.p, v.n * sizeof(float), QuantType::F32).buffer) return false;

    // scratch (device local) and per-layer state
    auto ensure_dev = [&](GpuBuffer& gb, size_t need) -> bool {
        need = std::max<size_t>(need, 16);
        if (gb.buffer && gb.size >= need) return true;
        chain_wait_all();                                   // never free a buffer an in-flight chained layer may still use
        free_buffer(gb);
        gb = alloc_buffer(need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return gb.buffer != VK_NULL_HANDLE && gb.memory != VK_NULL_HANDLE;
    };
    if (!ensure_dev(q35_t_mixed_, (size_t)d.conv_dim * 4) || !ensure_dev(q35_t_cout_, (size_t)d.conv_dim * 4) || !ensure_dev(q35_t_z_, (size_t)d.value_dim * 4) ||
        !ensure_dev(q35_t_a_, (size_t)d.vh * 4) || !ensure_dev(q35_t_b_, (size_t)d.vh * 4) || !ensure_dev(q35_t_core_, (size_t)d.value_dim * 4)) return false;

    const std::pair<const void*, int> key{d.owner, d.layer};
    bool need_upload = false;
    auto it = q35_dev_state_.find(key);
    if (async && it == q35_dev_state_.end()) return false;        // first use of this layer's state needs the synchronous upload path
    if (it == q35_dev_state_.end()) {
        if (!conv_state_host && conv_n) return false;
        if (!rec_state_host) return false;
        Q35DevState ns; ns.conv_n = conv_n; ns.rec_n = rec_n;
        if (!ensure_dev(ns.conv, conv_n * 4) || !ensure_dev(ns.rec, rec_n * 4)) { free_buffer(ns.conv); free_buffer(ns.rec); return false; }
        auto ensure_stage = [&](GpuBuffer& gb, size_t need) -> bool {
            if (gb.buffer && gb.size >= need && gb.mapped_ptr) return true;
            free_buffer(gb);
            gb = alloc_buffer(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!gb.buffer || !gb.memory) return false;
            if (!gb.mapped_ptr && vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped_ptr) != VK_SUCCESS) return false;
            return gb.mapped_ptr != nullptr;
        };
        if ((conv_n && !ensure_stage(q35_t_stage_conv_, conv_n * 4)) || !ensure_stage(q35_t_stage_rec_, rec_n * 4)) { free_buffer(ns.conv); free_buffer(ns.rec); return false; }
        if (conv_n) std::memcpy(q35_t_stage_conv_.mapped_ptr, conv_state_host, conv_n * 4);
        std::memcpy(q35_t_stage_rec_.mapped_ptr, rec_state_host, rec_n * 4);
        it = q35_dev_state_.emplace(key, ns).first;
        need_upload = true;
    } else if (it->second.conv_n != conv_n || it->second.rec_n != rec_n) {
        return false;
    }
    Q35DevState& st = it->second;

    ops_kind_ = 0;
    if (async && !chain_open_ && !chain_begin(x, d.hidden)) return false;
    chain_async_ = async && chain_open_;
    bool cache_new = false;
    Q35LinCache* cslot = nullptr;
    if (chain_async_ && !need_upload) {
        uint64_t sig = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { sig = (sig ^ v) * 1099511628211ull; };
        const void* ptrs[] = { d.owner, d.attn_norm, d.ffn_norm, d.qkv, d.z, d.a, d.b, d.out, d.gate, d.up, d.down, d.conv_w, d.A, d.dt, d.gnorm };
        for (const void* q : ptrs) mix((uint64_t)(uintptr_t)q);
        const uint64_t nums[] = { (uint64_t)d.layer, d.hidden, d.conv_dim, d.value_dim, d.key_dim, d.ffn, d.kh, d.vh, d.hd, d.kernel, (uint64_t)d.qkv_t, (uint64_t)d.z_t, (uint64_t)d.a_t, (uint64_t)d.b_t,
                                  (uint64_t)d.out_t, (uint64_t)d.gate_t, (uint64_t)d.up_t, (uint64_t)d.down_t };
        for (uint64_t v : nums) mix(v);
        uint32_t eb; std::memcpy(&eb, &d.norm_eps, 4); mix(eb);
        Q35LinCache& c = q35_lin_cache_[key];
        if (c.cmd && c.gen == buf_gen_ && c.sig == sig) {                 // hit: resubmit the recorded buffer
            ++lin_cache_hits_;
            chain_async_ = false;
            return chain_submit_cmd(c.cmd, false);
        }
        ++lin_cache_miss_;
        chain_wait_all();                                                 // the old buffer may still be in flight
        lin_cache_drop(c);
        VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; ai.commandPool = cmd_pool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkDescriptorPoolSize psz{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512 };
        VkDescriptorPoolCreateInfo ci{}; ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO; ci.maxSets = 64; ci.poolSizeCount = 1; ci.pPoolSizes = &psz;
        if (vkAllocateCommandBuffers(device_, &ai, &c.cmd) == VK_SUCCESS && vkCreateDescriptorPool(device_, &ci, nullptr, &c.pool) == VK_SUCCESS) {
            c.sig = sig; c.gen = buf_gen_;                                // stamped BEFORE recording: if recording frees anything the entry is stale next token
            chain_rec_cmd_ = c.cmd; chain_rec_pool_ = c.pool; cache_new = true; cslot = &c;
        } else { lin_cache_drop(c); }
    }
    begin_layer_record(chain_async_ ? nullptr : x, d.hidden);
    if (need_upload) {
        if (conv_n) { VkBufferCopy c{0, 0, conv_n * 4}; vkCmdCopyBuffer(layer_cmd_, q35_t_stage_conv_.buffer, st.conv.buffer, 1, &c); }
        VkBufferCopy c{0, 0, rec_n * 4}; vkCmdCopyBuffer(layer_cmd_, q35_t_stage_rec_.buffer, st.rec.buffer, 1, &c);
        VkMemoryBarrier mb{}; mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER; mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(layer_cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    auto wbuf = [&](const void* p, size_t bytes) -> GpuBuffer& { return get_weight_buffer(p, bytes, QuantType::F32); };
    bool ok = true;
    const size_t hb = (size_t)d.hidden * 4;

    ok &= record_rms_norm_g(act_hidden_, d.attn_norm, hb, act_xb_, d.hidden, d.norm_eps);
    layer_barrier("attn norm");
    record_quant_act_g(act_xb_, d.hidden);
    ok &= record_mat_vec_g(d.qkv, d.qkv_t, act_xb_, q35_t_mixed_, d.conv_dim, d.hidden);
    ok &= record_mat_vec_g(d.z,   d.z_t,   act_xb_, q35_t_z_,     d.value_dim, d.hidden);
    ok &= record_mat_vec_g(d.a,   d.a_t,   act_xb_, q35_t_a_,     d.vh, d.hidden);
    ok &= record_mat_vec_g(d.b,   d.b_t,   act_xb_, q35_t_b_,     d.vh, d.hidden);
    layer_barrier("qkv+z+a+b matvec");
    {   // conv step (updates st.conv in place)
        VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_q35_conv_step_.ds_layout);
        if (!ds) ok = false; else {
            GpuBuffer& cw = wbuf(d.conv_w, (size_t)d.conv_dim * d.kernel * 4);
            VkBuffer bufs[4] = { q35_t_mixed_.buffer, st.conv.buffer, cw.buffer, q35_t_cout_.buffer };
            rec_bind_bufs(device_, ds, bufs, 4);
            struct { uint32_t conv_dim, kernel, p0, p1; } pc{ d.conv_dim, d.kernel, 0, 0 };
            layer_dispatch(layer_cmd_, pipe_q35_conv_step_.pipeline, pipe_q35_conv_step_.layout, ds, &pc, sizeof(pc), (d.conv_dim + 255u) / 256u);
        }
    }
    layer_barrier("conv step");
    {   // delta-rule step + gated rms norm (state in place)
        VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, scan_pipe.ds_layout);
        if (!ds) ok = false; else {
            GpuBuffer& A = wbuf(d.A, (size_t)d.vh * 4); GpuBuffer& dt = wbuf(d.dt, (size_t)d.vh * 4); GpuBuffer& gn = wbuf(d.gnorm, (size_t)d.hd * 4);
            VkBuffer bufs[9] = { q35_t_cout_.buffer, q35_t_a_.buffer, q35_t_b_.buffer, q35_t_z_.buffer, A.buffer, dt.buffer, gn.buffer, st.rec.buffer, q35_t_core_.buffer };
            rec_bind_bufs(device_, ds, bufs, 9);
            struct { uint32_t n_tokens, conv_dim, key_dim, value_dim, kh, vh, hd, repeat, grouped; float eps; } pc{
                1u, d.conv_dim, d.key_dim, d.value_dim, d.kh, d.vh, d.hd, d.vh / d.kh, 0u, d.norm_eps };
            layer_dispatch(layer_cmd_, scan_pipe.pipeline, scan_pipe.layout, ds, &pc, sizeof(pc), d.vh);
        }
    }
    layer_barrier("delta-rule scan");
    record_quant_act_g(q35_t_core_, d.value_dim);
    ok &= record_mat_vec_g(d.out, d.out_t, q35_t_core_, act_ffn_out_2_, d.hidden, d.value_dim);
    layer_barrier("out matvec");
    ok &= record_add_g(act_hidden_, act_ffn_out_2_, d.hidden);
    layer_barrier("residual add");
    ok &= record_rms_norm_g(act_hidden_, d.ffn_norm, hb, act_xb_, d.hidden, d.norm_eps);
    layer_barrier("ffn norm");
    record_quant_act_g(act_xb_, d.hidden);
    ok &= record_mat_vec_g(d.gate, d.gate_t, act_xb_, act_gate_, d.ffn, d.hidden);
    ok &= record_mat_vec_g(d.up,   d.up_t,   act_xb_, act_up_,   d.ffn, d.hidden);
    layer_barrier("gate+up matvec");
    ok &= record_silu_mul_g(act_gate_, act_up_, d.ffn);
    layer_barrier("silu*mul");
    record_quant_act_g(act_gate_, d.ffn);
    ok &= record_mat_vec_g(d.down, d.down_t, act_gate_, act_ffn_out_2_, d.hidden, d.ffn);
    layer_barrier("down matvec");
    ok &= record_add_g(act_hidden_, act_ffn_out_2_, d.hidden);

    if (!ok) {
        const bool was_async = chain_async_; chain_async_ = false;
        if (was_async) { layer_recording_ = false; vkResetCommandBuffer(layer_cmd_, 0); if (cache_new && cslot) lin_cache_drop(*cslot); }
        else end_layer_record(nullptr, 0);                            // discard: nothing was submitted, x untouched
        if (need_upload) { free_buffer(st.conv); free_buffer(st.rec); q35_dev_state_.erase(key); }
        return false;
    }
    if (chain_async_) {
        chain_async_ = false;
        return chain_submit_layer();
    }
    end_layer_record(x, d.hidden);
    return true;
}

// ── Qwen3.5 full-attention layer: one token, one submit, GPU-resident KV mirror ───────────────────────────────────────────────────
inline size_t VulkanBackend::kv_mirror_cap_bytes() const {
    const int mb = gpu_kv_mb_;
    if (mb == 0) return 0;
    if (mb > 0) return (size_t)mb * 1048576ull;
    // auto: only on discrete cards; leave room for weights still to be uploaded (lm_head, other layers), scratch buffers and the desktop.
    if (info_.unified_memory || info_.vram_bytes == 0) return 0;
    const size_t vram = info_.vram_bytes;
    size_t used = weight_cache_used_;
    if (weight_cache_budget_ > 0) used = std::max(used, weight_cache_budget_);
    const size_t margin = std::max<size_t>(768ull << 20, vram / 5);
    const size_t other  = 64ull << 20;                  // activation / scratch buffers
    if (vram <= used + margin + other) return 0;
    return vram - used - margin - other;
}

inline void VulkanBackend::q35_kv_drop(const void* owner, int layer) {
    if (!device_) return;
    for (auto it = q35_kv_mirror_.begin(); it != q35_kv_mirror_.end();) {
        if (it->first.first == owner && (layer < 0 || it->first.second == layer)) {
            device_idle_locked();
            kv_mirror_used_ -= std::min(kv_mirror_used_, it->second.bytes);
            free_buffer(it->second.k); free_buffer(it->second.v);
            it = q35_kv_mirror_.erase(it);
        } else ++it;
    }
}

inline bool VulkanBackend::q35_full_layer(const Q35FullDesc& d, float* x, float* kv_out_host, uint32_t* mirror_valid_out, int async_kv_slot) {
    const bool async = async_kv_slot >= 0;
    if (!ready_ || layer_recording_ || !x || (!kv_out_host && !async) || !rec_pool_ || !layer_cmd_) return nx_decline("!ready_ || layer_recording_ || !x || (!kv_out_host && !async) || !rec_pool_ || !layer_cmd_");
    if (async && (ops_prof_n_ > 0 || async_kv_slot >= kChainKvSlots)) return nx_decline("async && (ops_prof_n_ > 0 || async_kv_slot >= kChainKvSlots)");
    if (!pipe_q35_aprep_.pipeline || !pipe_q35_apart_.pipeline || !pipe_q35_acomb_.pipeline) return nx_decline("!pipe_q35_aprep_.pipeline || !pipe_q35_apart_.pipeline || !pipe_q35_acomb_.pipeline");
    if (!pipe_rms_.pipeline || !pipe_silu_.pipeline || !pipe_add_.pipeline) return nx_decline("!pipe_rms_.pipeline || !pipe_silu_.pipeline || !pipe_add_.pipeline");
    if (gpu_kv_mb_ == -2) {
        gpu_kv_mb_ = -1;
        if (const char* e = std::getenv("NEXUS_GPU_KV")) {
            if (!std::strcmp(e, "off") || !std::strcmp(e, "0")) gpu_kv_mb_ = 0;
            else if (!std::strcmp(e, "auto") || !*e) gpu_kv_mb_ = -1;
            else { const int v = std::atoi(e); gpu_kv_mb_ = v > 0 ? v : -1; }
        }
    }
    const size_t cap_bytes = kv_mirror_cap_bytes();
    if (cap_bytes == 0) return nx_decline("cap_bytes == 0");

    const uint32_t hd = d.hd, nh = d.n_heads, nkv = d.n_kv;
    if (hd == 0 || hd % 64 != 0 || hd > 256 || nh == 0 || nkv == 0 || nh % nkv != 0) return nx_decline("hd == 0 || hd % 64 != 0 || hd > 256 || nh == 0 || nkv == 0 || nh % nkv != 0");
    const uint32_t G = nh / nkv;
    if (G > 8) return nx_decline("G > 8");
    const uint32_t KG = 256u / (hd >> 1);                    // key groups in the P*V stage
    const bool use_apart3 = pipe_q35_apart3_.pipeline && hd == 256 && G <= 4;
    const bool use_apart2 = !use_apart3 && pipe_q35_apart2_.pipeline && (hd == 128 || hd == 256) && G <= 4;     // register-q kernel (own shared-memory layout)
    if (!use_apart2 && !use_apart3 && (size_t)(KG - 1) * G * hd > 3584) return nx_decline("(size_t)(KG - 1) * G * hd > 3584");      // shared-memory pool of the partial kernel (qwen35_attn_part.comp qs[])
    if (d.rot == 0 || d.rot > hd || (d.rot & 1) || !d.rope_tab) return nx_decline("d.rot == 0 || d.rot > hd || (d.rot & 1) || !d.rope_tab");
    if (d.hidden == 0 || d.ffn == 0 || !d.k_host || !d.v_host || d.host_tokens < d.pos) return nx_decline("d.hidden == 0 || d.ffn == 0 || !d.k_host || !d.v_host || d.host_tokens < d.pos");
    if (!d.attn_norm || !d.ffn_norm || !d.wq || !d.wk || !d.wv || !d.wo || !d.gate || !d.up || !d.down) return nx_decline("!d.attn_norm || !d.ffn_norm || !d.wq || !d.wk || !d.wv || !d.wo || !d.gate || !d.up || !d.down");
    if (d.has_gate && (!d.q_norm || !d.k_norm)) return nx_decline("gated layer without q/k norm");
    if (!act_bufs_ok() || !act_hidden_.mapped_ptr) return nx_decline("!act_bufs_ok() || !act_hidden_.mapped_ptr");
    const size_t q_dim = (size_t)nh * hd, kv_dim = (size_t)nkv * hd;
    if (act_xb_.size < (size_t)d.hidden * 4 || act_gate_.size < (size_t)d.ffn * 4 || act_up_.size < (size_t)d.ffn * 4 || act_ffn_out_2_.size < (size_t)d.hidden * 4) return nx_decline("act_xb_.size < (size_t)d.hidden * 4 || act_gate_.size < (size_t)d.ffn * 4 || act_up_.size < (size_t)d.ffn *...");

    const uint32_t need = d.pos + 1;
    const uint32_t CHK = 128;                                // keys per partial workgroup (must match qwen35_attn_part.comp)
    const uint32_t nchunks = (need + CHK - 1) / CHK;
    if (nchunks > 4096) return nx_decline("nchunks > 4096");                        // merge kernel limit (~524k tokens)

    // weights resident BEFORE recording
    struct W { const void* p; size_t rows, cols; QuantType t; };
    const size_t wq_rows = d.has_gate ? 2 * q_dim : q_dim;
    const W mats[] = { {d.wq, wq_rows, d.hidden, d.wq_t}, {d.wk, kv_dim, d.hidden, d.wk_t}, {d.wv, kv_dim, d.hidden, d.wv_t}, {d.wo, d.hidden, q_dim, d.wo_t},
                       {d.gate, d.ffn, d.hidden, d.gate_t}, {d.up, d.ffn, d.hidden, d.up_t}, {d.down, d.hidden, d.ffn, d.down_t} };
    for (const W& m : mats) {
        Pipeline* pp = pipeline_for(m.t);
        size_t bytes = 0; uint32_t nb = 0;
        if (!pp || !pp->pipeline || !size_for(m.t, m.rows, m.cols, bytes, nb)) return nx_decline("!pp || !pp->pipeline || !size_for(m.t, m.rows, m.cols, bytes, nb)");
        if (!get_weight_buffer(m.p, bytes, m.t).buffer) return nx_decline("!get_weight_buffer(m.p, bytes, m.t).buffer");
    }
    const struct { const void* p; size_t n; } vecs[] = { {d.attn_norm, (size_t)d.hidden}, {d.ffn_norm, (size_t)d.hidden}, {d.q_norm, hd}, {d.k_norm, hd},
                                                         {d.bq, q_dim}, {d.bk, kv_dim}, {d.bv, kv_dim}, {d.bo, (size_t)d.hidden} };
    for (auto& v : vecs) if (v.p && !get_weight_buffer(v.p, v.n * sizeof(float), QuantType::F32).buffer) return nx_decline("!get_weight_buffer(v.p, v.n * sizeof(float), QuantType::F32).buffer");

    auto ensure_dev = [&](GpuBuffer& gb, size_t sz) -> bool {
        sz = std::max<size_t>(sz, 16);
        if (gb.buffer && gb.size >= sz) return true;
        chain_wait_all();                                   // never free a buffer an in-flight chained layer may still use
        free_buffer(gb);
        gb = alloc_buffer(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        return gb.buffer != VK_NULL_HANDLE && gb.memory != VK_NULL_HANDLE;
    };
    auto ensure_host = [&](GpuBuffer& gb, size_t sz) -> bool {
        sz = std::max<size_t>(sz, 16);
        if (gb.buffer && gb.size >= sz && gb.mapped_ptr) return true;
        chain_wait_all();
        free_buffer(gb);
        gb = alloc_buffer(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!gb.buffer || !gb.memory) return nx_decline("!gb.buffer || !gb.memory");
        if (!gb.mapped_ptr && vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped_ptr) != VK_SUCCESS) return nx_decline("!gb.mapped_ptr && vkMapMemory(device_, gb.memory, 0, gb.size, 0, &gb.mapped_ptr) != VK_SUCCESS");
        return gb.mapped_ptr != nullptr;
    };
    if (!ensure_dev(q35_a_qg_, 2 * q_dim * 4) || !ensure_dev(q35_a_k_, kv_dim * 4) || !ensure_dev(q35_a_v_, kv_dim * 4) || !ensure_dev(q35_a_q_, q_dim * 4) ||
        !ensure_dev(q35_a_gate_, q_dim * 4) || !ensure_dev(q35_a_out_, q_dim * 4) || !ensure_dev(q35_a_part_, (size_t)nh * nchunks * (hd + 2) * 4) ||
        !ensure_host(q35_a_kvout_, 2 * kv_dim * 4) || !ensure_host(q35_a_tab_, (size_t)d.rot * 4)) return false;
    if (async && !ensure_host(q35_a_kvring_[async_kv_slot], 2 * kv_dim * 4)) return nx_decline("async && !ensure_host(q35_a_kvring_[async_kv_slot], 2 * kv_dim * 4)");

    // ---- KV mirror: find / grow / fill --------------------------------------------------------------------------------------------
    const std::pair<const void*, int> key{d.owner, d.layer};
    auto mit = q35_kv_mirror_.find(key);
    if (mit == q35_kv_mirror_.end()) mit = q35_kv_mirror_.emplace(key, Q35KvMirror{}).first;
    if (mit->second.cap && (mit->second.n_kv != nkv || mit->second.hd != hd)) { q35_kv_drop(d.owner, d.layer); mit = q35_kv_mirror_.emplace(key, Q35KvMirror{}).first; }
    Q35KvMirror& mm = mit->second;
    size_t eff_valid = std::min<size_t>(mm.valid, d.host_synced);
    if (async && (mm.cap < need || eff_valid < d.pos)) return nx_decline("async && (mm.cap < need || eff_valid < d.pos)");      // growing / filling the mirror uses the synchronous path
    if (mm.cap < need) {
        if (mm.refused_need && need < mm.refused_need) return nx_decline("mm.refused_need && need < mm.refused_need");      // already refused a bigger-than-this ask recently: don't retry every token
        size_t newcap = std::max<size_t>(((size_t)need * 5 / 4 + 255) & ~(size_t)255, 2048);
        const size_t per_tok = (size_t)nkv * hd * 2 * 2;                    // K+V f16
        size_t newbytes = newcap * per_tok;
        const size_t others = kv_mirror_used_ - std::min(kv_mirror_used_, mm.bytes);
        if (others + newbytes > cap_bytes) {                                // try exactly what is needed (+ a little) before giving up
            newcap = (((size_t)need + 255) & ~(size_t)255); newbytes = newcap * per_tok;
        }
        if (others + newbytes > cap_bytes) {
            if (mm.cap) { q35_kv_drop(d.owner, d.layer); }
            else q35_kv_mirror_.erase(key);
            auto again = q35_kv_mirror_.emplace(key, Q35KvMirror{}).first; again->second.refused_need = need + 1024;
            static bool said = false;
            if (!said) { said = true; fprintf(stderr, "[vulkan] GPU KV cache: %.0f MB would exceed the budget (%.0f MB, NEXUS_GPU_KV / launcher option) -- full-attention layers use the CPU for this context\n",
                                              (others + newbytes) / 1048576.0, cap_bytes / 1048576.0); fflush(stderr); }
            return false;
        }
        device_idle_locked();
        free_buffer(mm.k); free_buffer(mm.v);
        kv_mirror_used_ -= std::min(kv_mirror_used_, mm.bytes);
        const size_t half = newcap * (size_t)nkv * hd * 2;
        mm.k = alloc_buffer(half, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        mm.v = alloc_buffer(half, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!mm.k.buffer || !mm.v.buffer) {
            free_buffer(mm.k); free_buffer(mm.v); mm.cap = 0; mm.bytes = 0; mm.valid = 0; mm.refused_need = need + 1024;
            return false;
        }
        mm.cap = newcap; mm.bytes = newbytes; mm.valid = 0; mm.n_kv = nkv; mm.hd = hd; mm.refused_need = 0;
        kv_mirror_used_ += newbytes;
        eff_valid = 0;
        fprintf(stderr, "[vulkan] GPU KV cache layer %d: %zu tokens capacity (%.0f MB, total %.0f of %.0f MB)\n", d.layer, newcap, newbytes / 1048576.0,
                kv_mirror_used_ / 1048576.0, cap_bytes / 1048576.0); fflush(stderr);
    }
    // fill tokens [eff_valid, pos) from the host
    if (eff_valid < d.pos) {
        const size_t stage_bytes = 8ull << 20;
        if (!q35_kv_stage_.buffer || q35_kv_stage_.size < stage_bytes || !q35_kv_stage_.mapped_ptr) {
            free_buffer(q35_kv_stage_);
            q35_kv_stage_ = alloc_buffer(stage_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!q35_kv_stage_.buffer || !q35_kv_stage_.memory) return nx_decline("!q35_kv_stage_.buffer || !q35_kv_stage_.memory");
            if (vkMapMemory(device_, q35_kv_stage_.memory, 0, q35_kv_stage_.size, 0, &q35_kv_stage_.mapped_ptr) != VK_SUCCESS) return nx_decline("vkMapMemory(device_, q35_kv_stage_.memory, 0, q35_kv_stage_.size, 0, &q35_kv_stage_.mapped_ptr) != VK_SUCCESS");
        }
        const size_t row_bytes = (size_t)hd * 2;
        const size_t rows_per_piece = std::max<size_t>(1, q35_kv_stage_.size / row_bytes);
        for (int which = 0; which < 2; ++which) {
            GpuBuffer& dst = which == 0 ? mm.k : mm.v;
            const uint16_t* const* host = which == 0 ? d.k_host : d.v_host;
            for (uint32_t h = 0; h < nkv; ++h) {
                if (!host[h]) return nx_decline("!host[h]");
                for (size_t t = eff_valid; t < d.pos; t += rows_per_piece) {
                    const size_t n = std::min<size_t>(rows_per_piece, d.pos - t);
                    std::memcpy(q35_kv_stage_.mapped_ptr, host[h] + t * hd, n * row_bytes);
                    VkCommandBuffer cmd = begin_one_shot();
                    VkBufferCopy c{0, ((size_t)h * mm.cap + t) * row_bytes, n * row_bytes};
                    vkCmdCopyBuffer(cmd, q35_kv_stage_.buffer, dst.buffer, 1, &c);
                    end_one_shot(cmd);
                }
            }
        }
        mm.valid = d.pos;
    }

    // ---- record -----------------------------------------------------------------------------------------------------------------------
    if (async && !chain_open_ && !chain_begin(x, d.hidden)) return nx_decline("async && !chain_open_ && !chain_begin(x, d.hidden)");
    chain_async_ = async && chain_open_;
    if (!chain_async_ || chain_tab_pos_ != (int64_t)d.pos) {            // the cos/sin table is identical for every full-attention layer of a token
        std::memcpy(q35_a_tab_.mapped_ptr, d.rope_tab, (size_t)d.rot * 4);
        chain_tab_pos_ = chain_async_ ? (int64_t)d.pos : -1;
    }
    ops_kind_ = 1;
    begin_layer_record(chain_async_ ? nullptr : x, d.hidden);
    GpuBuffer& kvdst = chain_async_ ? q35_a_kvring_[async_kv_slot] : q35_a_kvout_;
    auto wbuf = [&](const void* p, size_t bytes) -> GpuBuffer& { return get_weight_buffer(p, bytes, QuantType::F32); };
    bool ok = true;
    const size_t hb = (size_t)d.hidden * 4;
    ok &= record_rms_norm_g(act_hidden_, d.attn_norm, hb, act_xb_, d.hidden, d.norm_eps);
    layer_barrier("attn norm");
    record_quant_act_g(act_xb_, d.hidden);
    ok &= record_mat_vec_g(d.wq, d.wq_t, act_xb_, q35_a_qg_, (uint32_t)wq_rows, d.hidden);
    ok &= record_mat_vec_g(d.wk, d.wk_t, act_xb_, q35_a_k_,  kv_dim,    d.hidden);
    ok &= record_mat_vec_g(d.wv, d.wv_t, act_xb_, q35_a_v_,  kv_dim,    d.hidden);
    layer_barrier("q/k/v matvec");
    if (d.bq || d.bk || d.bv) {                         // biases go on before the head norms and RoPE, as on the CPU
        if (d.bq) ok &= record_add_g(q35_a_qg_, wbuf(d.bq, q_dim * 4), (uint32_t)q_dim);
        if (d.bk) ok &= record_add_g(q35_a_k_,  wbuf(d.bk, kv_dim * 4), (uint32_t)kv_dim);
        if (d.bv) ok &= record_add_g(q35_a_v_,  wbuf(d.bv, kv_dim * 4), (uint32_t)kv_dim);
        layer_barrier("q/k/v bias");
    }
    {   // prep: head norms, rope, cache store
        VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_q35_aprep_.ds_layout);
        if (!ds) ok = false; else {
            GpuBuffer& qn = d.q_norm ? wbuf(d.q_norm, (size_t)hd * 4) : q35_a_tab_;     // not read when the norm is off (flags)
            GpuBuffer& kn = d.k_norm ? wbuf(d.k_norm, (size_t)hd * 4) : q35_a_tab_;
            VkBuffer bufs[11] = { q35_a_qg_.buffer, q35_a_k_.buffer, q35_a_v_.buffer, qn.buffer, kn.buffer, q35_a_tab_.buffer,
                                  q35_a_q_.buffer, q35_a_gate_.buffer, mm.k.buffer, mm.v.buffer, kvdst.buffer };
            rec_bind_bufs(device_, ds, bufs, 11);
            struct { uint32_t n_heads, n_kv, hd, rot, pos, cap, flags, pad; float eps; } pc{
                nh, nkv, hd, d.rot, d.pos, (uint32_t)mm.cap,
                (d.q_zc ? 1u : 0u) | (d.k_zc ? 2u : 0u) | (d.has_gate ? 0u : 4u) | (d.q_norm ? 0u : 8u) | (d.k_norm ? 0u : 16u) | (d.rope_interleaved ? 32u : 0u),
                0u, d.norm_eps };
            layer_dispatch(layer_cmd_, pipe_q35_aprep_.pipeline, pipe_q35_aprep_.layout, ds, &pc, sizeof(pc), nh + 2 * nkv);
        }
    }
    layer_barrier("prep (norm,rope,store)");
    {   // split-K partials
        const Pipeline& ap = use_apart3 ? pipe_q35_apart3_ : use_apart2 ? pipe_q35_apart2_ : pipe_q35_apart_;
        VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, ap.ds_layout);
        if (!ds) ok = false; else {
            VkBuffer bufs[4] = { q35_a_q_.buffer, mm.k.buffer, mm.v.buffer, q35_a_part_.buffer };
            rec_bind_bufs(device_, ds, bufs, 4);
            struct { uint32_t n_heads, n_kv, hd, n_ctx, cap, nchunks, p0, p1; float scale; } pc{
                nh, nkv, hd, need, (uint32_t)mm.cap, nchunks, 0u, 0u, d.scale > 0.f ? d.scale : 1.0f / std::sqrt((float)hd) };
            layer_dispatch(layer_cmd_, ap.pipeline, ap.layout, ds, &pc, sizeof(pc), nkv, nchunks);
        }
    }
    layer_barrier("attn partials");
    {   // merge chunks + output gate
        VkDescriptorSet ds = rec_alloc_ds(device_, rec_pool_, pipe_q35_acomb_.ds_layout);
        if (!ds) ok = false; else {
            VkBuffer bufs[3] = { q35_a_part_.buffer, q35_a_gate_.buffer, q35_a_out_.buffer };
            rec_bind_bufs(device_, ds, bufs, 3);
            struct { uint32_t hd, nchunks, p0, p1; } pc{ hd, nchunks, d.has_gate ? 0u : 1u, 0u };      // p0=1: no output gate
            layer_dispatch(layer_cmd_, pipe_q35_acomb_.pipeline, pipe_q35_acomb_.layout, ds, &pc, sizeof(pc), nh);
        }
    }
    layer_barrier("attn merge+gate");
    record_quant_act_g(q35_a_out_, q_dim);
    ok &= record_mat_vec_g(d.wo, d.wo_t, q35_a_out_, act_ffn_out_2_, d.hidden, q_dim);
    layer_barrier("wo matvec");
    if (d.bo) { ok &= record_add_g(act_ffn_out_2_, wbuf(d.bo, hb), d.hidden); layer_barrier("wo bias"); }
    ok &= record_add_g(act_hidden_, act_ffn_out_2_, d.hidden);
    layer_barrier("residual add");
    ok &= record_rms_norm_g(act_hidden_, d.ffn_norm, hb, act_xb_, d.hidden, d.norm_eps);
    layer_barrier("ffn norm");
    record_quant_act_g(act_xb_, d.hidden);
    ok &= record_mat_vec_g(d.gate, d.gate_t, act_xb_, act_gate_, d.ffn, d.hidden);
    ok &= record_mat_vec_g(d.up,   d.up_t,   act_xb_, act_up_,   d.ffn, d.hidden);
    layer_barrier("gate+up matvec");
    ok &= record_silu_mul_g(act_gate_, act_up_, d.ffn);
    layer_barrier("silu*mul");
    record_quant_act_g(act_gate_, d.ffn);
    ok &= record_mat_vec_g(d.down, d.down_t, act_gate_, act_ffn_out_2_, d.hidden, d.ffn);
    layer_barrier("down matvec");
    ok &= record_add_g(act_hidden_, act_ffn_out_2_, d.hidden);
    if (!ok) {
        const bool was_async = chain_async_; chain_async_ = false;
        if (was_async) { layer_recording_ = false; vkResetCommandBuffer(layer_cmd_, 0); } else end_layer_record(nullptr, 0);
        return false;
    }
    if (chain_async_) {
        chain_async_ = false;
        if (!chain_submit_layer()) return nx_decline("!chain_submit_layer()");
    } else {
        end_layer_record(x, d.hidden);
        std::memcpy(kv_out_host, q35_a_kvout_.mapped_ptr, 2 * kv_dim * 4);
    }
    mm.valid = need;
    if (mirror_valid_out) *mirror_valid_out = need;
    return true;
}

inline VkCommandBuffer VulkanBackend::begin_one_shot() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = cmd_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

inline void VulkanBackend::end_one_shot(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    // Use a local fence here. Reusing reuse_fence_ races with mat_mat() when a
    // background prefetch thread uploads a future weight while the main thread
    // is waiting on a compute dispatch.
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_, &fci, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
        return;
    }

    {
        std::lock_guard<std::mutex> qlk(queue_mu_);
        vkQueueSubmit(compute_queue_, 1, &si, fence);
    }
    vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
}

// Weight upload.  Big tensors (the normal case: every matrix of every streamed layer) go through two persistent 16 MB staging chunks:
// the CPU copies chunk N+1 out of system RAM (with several threads, since one core cannot saturate memory bandwidth) while the card
// DMAs chunk N into VRAM.  The old path allocated a staging buffer as large as the whole tensor, copied it on one thread, waited, and
// freed it -- per tensor, ~10 tensors per layer.  NEXUS_UPLOAD_SIMPLE=1 forces the old path for A/B comparison.
inline bool VulkanBackend::upload(GpuBuffer& dst, const void* src, size_t size) {
    std::lock_guard<std::mutex> ulk(upload_mu_);
    if (!dst.buffer || !src || size == 0) return false;

    static const bool force_simple = (std::getenv("NEXUS_UPLOAD_SIMPLE") != nullptr);
    constexpr size_t kChunk = 16u << 20;
    if (force_simple || size < (2u << 20)) return upload_oneshot(dst, src, size);

    VkCommandPool pool = transfer_cmd_pool_ ? transfer_cmd_pool_ : cmd_pool_;
    if (!up_slots_ready_) {
        bool ok = true;
        for (auto& sl : up_slot_) {
            sl.buf = alloc_buffer(kChunk, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (!sl.buf.buffer || !sl.buf.memory) { ok = false; break; }
            if (!sl.buf.mapped_ptr && vkMapMemory(device_, sl.buf.memory, 0, kChunk, 0, &sl.buf.mapped_ptr) != VK_SUCCESS) { ok = false; break; }
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; ai.commandPool = pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
            if (vkAllocateCommandBuffers(device_, &ai, &sl.cmd) != VK_SUCCESS) { ok = false; break; }
            VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            if (vkCreateFence(device_, &fci, nullptr, &sl.fence) != VK_SUCCESS) { ok = false; break; }
        }
        if (!ok) {
            for (auto& sl : up_slot_) {
                if (sl.fence) vkDestroyFence(device_, sl.fence, nullptr);
                if (sl.cmd)   vkFreeCommandBuffers(device_, pool, 1, &sl.cmd);
                free_buffer(sl.buf);
                sl = UpSlot{};
            }
            fprintf(stderr, "[vulkan] pipelined upload unavailable -- using one staging buffer per tensor\n");
            return upload_oneshot(dst, src, size);
        }
        up_slots_ready_ = true;
    }

    // How many CPU threads copy one chunk.  Capped: more than ~4 stops helping and just steals cores from the CPU layers.
    int copy_threads = 4;
    if (const char* e = std::getenv("NEXUS_UPLOAD_THREADS")) { int v = std::atoi(e); if (v >= 1 && v <= 16) copy_threads = v; }

    const bool same_q = (!transfer_queue_ || transfer_queue_ == compute_queue_);
    std::mutex& qmu = same_q ? queue_mu_ : transfer_queue_mu_;
    VkQueue q = same_q ? compute_queue_ : transfer_queue_;

    const uint8_t* sp = (const uint8_t*)src;
    bool ok = true;
    size_t off = 0; unsigned idx = 0;
    while (off < size && ok) {
        UpSlot& sl = up_slot_[idx & 1u];
        // The previous use of this slot must have finished before its staging memory is overwritten.
        if (vkWaitForFences(device_, 1, &sl.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) { ok = false; break; }
        const size_t n = std::min(kChunk, size - off);

        // CPU: system RAM -> staging (parallel)
        if (copy_threads > 1 && n >= (4u << 20)) {
            std::vector<std::thread> th;
            const size_t per = (n + (size_t)copy_threads - 1) / (size_t)copy_threads;
            for (int t = 1; t < copy_threads; ++t) {
                const size_t a = (size_t)t * per; if (a >= n) break;
                const size_t b = std::min(n, a + per);
                th.emplace_back([&, a, b]() { memcpy((uint8_t*)sl.buf.mapped_ptr + a, sp + off + a, b - a); });
            }
            memcpy(sl.buf.mapped_ptr, sp + off, std::min(per, n));
            for (auto& t : th) t.join();
        } else {
            memcpy(sl.buf.mapped_ptr, sp + off, n);
        }

        // GPU: staging -> VRAM
        vkResetCommandBuffer(sl.cmd, 0);
        VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(sl.cmd, &bi);
        VkBufferCopy copy{0, (VkDeviceSize)off, (VkDeviceSize)n};
        vkCmdCopyBuffer(sl.cmd, sl.buf.buffer, dst.buffer, 1, &copy);
        vkEndCommandBuffer(sl.cmd);
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &sl.cmd;
        vkResetFences(device_, 1, &sl.fence);
        VkResult sr;
        { std::lock_guard<std::mutex> qlk(qmu); sr = vkQueueSubmit(q, 1, &si, sl.fence); }
        if (sr != VK_SUCCESS) {
            // The fence was reset but never submitted: re-signal it so a later wait cannot hang.
            vkDestroyFence(device_, sl.fence, nullptr);
            VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(device_, &fci, nullptr, &sl.fence);
            ok = false; break;
        }
        off += n; ++idx;
    }
    // Everything issued must be finished before the caller uses `dst`.
    for (auto& sl : up_slot_)
        if (sl.fence && vkWaitForFences(device_, 1, &sl.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) ok = false;
    return ok;
}

inline bool VulkanBackend::upload_oneshot(GpuBuffer& dst, const void* src, size_t size) {
    if (!dst.buffer || !src || size == 0) return false;

    // True transfer path: copies run on transfer_queue_ / transfer_cmd_pool_.
    // On GPUs with a dedicated transfer family this can overlap with compute_queue_.
    // On drivers exposing only one queue family it safely degenerates to serialized copy.
    GpuBuffer staging = alloc_buffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (!staging.buffer) return false;

    void* mapped = nullptr;
    if (vkMapMemory(device_, staging.memory, 0, size, 0, &mapped) != VK_SUCCESS || !mapped) {
        free_buffer(staging);
        return false;
    }
    memcpy(mapped, src, size);
    vkUnmapMemory(device_, staging.memory);

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = transfer_cmd_pool_ ? transfer_cmd_pool_ : cmd_pool_;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS || cmd == VK_NULL_HANDLE) {
        free_buffer(staging);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device_, &fci, nullptr, &fence) != VK_SUCCESS) {
        vkFreeCommandBuffers(device_, ai.commandPool, 1, &cmd);
        free_buffer(staging);
        return false;
    }

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    VkResult sr;
    {
        // If the transfer queue is the compute queue (or absent), the same VkQueue
        // must be externally synchronised via queue_mu_.
        const bool same_q = (!transfer_queue_ || transfer_queue_ == compute_queue_);
        std::mutex& qmu = same_q ? queue_mu_ : transfer_queue_mu_;
        std::lock_guard<std::mutex> qlk(qmu);
        sr = vkQueueSubmit(same_q ? compute_queue_ : transfer_queue_, 1, &si, fence);
    }
    bool ok = (sr == VK_SUCCESS);
    if (ok) ok = (vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS);

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, ai.commandPool, 1, &cmd);
    free_buffer(staging);
    return ok;
}

inline void VulkanBackend::download(const GpuBuffer& src, void* dst, size_t size) {
    if (!src.buffer) return;
    void* mapped;
    vkMapMemory(device_, src.memory, 0, size, 0, &mapped);
    memcpy(dst, mapped, size);
    vkUnmapMemory(device_, src.memory);
}

inline void VulkanBackend::dispatch_mat_vec(Pipeline& pipe,
                                             GpuBuffer& W_buf,
                                             GpuBuffer& x_buf,
                                             GpuBuffer& y_buf,
                                             uint32_t out_rows,
                                             uint32_t in_cols,
                                             uint32_t n_blocks) {
    if (pipe.pipeline == VK_NULL_HANDLE) return;

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = ds_pool_;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &pipe.ds_layout;
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(device_, &dsai, &ds) != VK_SUCCESS) return;

    // Bind buffers: binding 0=W, 1=x, 2=y
    VkDescriptorBufferInfo bufs[3] = {
        {W_buf.buffer, 0, VK_WHOLE_SIZE},
        {x_buf.buffer, 0, VK_WHOLE_SIZE},
        {y_buf.buffer, 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet writes[3]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = ds;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufs[i];
    }
    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    // Push constants
    PushConstants pc{};
    pc.out_rows       = out_rows;
    pc.in_cols        = in_cols;
    pc.n_blocks_per_row = n_blocks;

    // Record and submit
    auto cmd = begin_one_shot();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipe.layout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // Work groups: each workgroup handles pipe.rows_per_group output rows (64 for the older kernels)
    uint32_t groups = (out_rows + pipe.rows_per_group - 1) / pipe.rows_per_group;
    vkCmdDispatch(cmd, groups, 1, 1);
    end_one_shot(cmd);

    // Free descriptor set back to pool
    vkFreeDescriptorSets(device_, ds_pool_, 1, &ds);
}

inline void VulkanBackend::free_buffer(GpuBuffer& b) {
    if (b.buffer) ++buf_gen_;                                           // invalidates every cached recorded layer
    if (b.mapped_ptr && b.memory) vkUnmapMemory(device_, b.memory);
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
    b = {};
}

inline size_t VulkanBackend::release_prefill_buffers() {
    if (!device_) return 0;
    device_idle_locked();
    GpuBuffer* bufs[] = {
        &mm_x_, &mm_y_, &mm_x_stage_, &mm_y_stage_, &mm_x_stage_b_, &mm_y_stage_b_, &ffn_u_, &ffn_o_, &mm_xq_, &mm_xs_, &attn_flash_host_,
        &q35_conv_mixed_, &q35_conv_state_, &q35_conv_out_,
        &q35_conv_mixed_stage_, &q35_conv_state_stage_, &q35_conv_out_stage_,
        &q35_scan_conv_, &q35_scan_a_, &q35_scan_b_, &q35_scan_z_, &q35_scan_A_,
        &q35_scan_dt_, &q35_scan_norm_, &q35_scan_state_, &q35_scan_core_,
        &q35_scan_conv_stage_, &q35_scan_a_stage_, &q35_scan_b_stage_, &q35_scan_z_stage_,
        &q35_scan_A_stage_, &q35_scan_dt_stage_, &q35_scan_norm_stage_,
        &q35_scan_state_stage_, &q35_scan_core_stage_,
        &attn_q_buf_, &attn_out_buf_, &attn_state_buf_, &attn_kchunk_buf_, &attn_vchunk_buf_,
        &attn_k_stage_, &attn_v_stage_,
    };
    size_t freed = 0;
    for (GpuBuffer* b : bufs) {
        if (b->buffer || b->memory) { freed += b->size; free_buffer(*b); }
    }
    return freed;
}

inline void VulkanBackend::shutdown() {
    if (!device_) return;
    vkDeviceWaitIdle(device_);
    if (reuse_cmd_) vkFreeCommandBuffers(device_, cmd_pool_, 1, &reuse_cmd_);
    reuse_cmd_ = VK_NULL_HANDLE;
    for (auto& [k, cw] : weight_cache_) {
        if (cw.buf.buffer) vkDestroyBuffer(device_, cw.buf.buffer, nullptr);
        if (cw.buf.memory) vkFreeMemory(device_, cw.buf.memory, nullptr);
    }
    weight_cache_.clear();
    free_buffer(act_x_);
    free_buffer(act_y_);
    free_buffer(act_y2_);
    free_buffer(mm_x_);
    free_buffer(mm_y_);
    for (auto& sl : up_slot_) {
        if (sl.fence) vkDestroyFence(device_, sl.fence, nullptr);
        free_buffer(sl.buf);          // its command buffer goes away with the pool
        sl = UpSlot{};
    }
    up_slots_ready_ = false;
    free_buffer(mm_x_stage_);
    free_buffer(mm_y_stage_);
    free_buffer(mm_x_stage_b_);
    free_buffer(mm_y_stage_b_);
    free_buffer(ffn_u_);
    free_buffer(ffn_o_);
    free_buffer(q35_conv_mixed_);
    free_buffer(q35_conv_state_);
    free_buffer(q35_conv_out_);
    free_buffer(q35_conv_mixed_stage_);
    free_buffer(q35_conv_state_stage_);
    free_buffer(q35_conv_out_stage_);
    free_buffer(q35_scan_conv_);
    free_buffer(q35_scan_a_);
    free_buffer(q35_scan_b_);
    free_buffer(q35_scan_z_);
    free_buffer(q35_scan_A_);
    free_buffer(q35_scan_dt_);
    free_buffer(q35_scan_norm_);
    free_buffer(q35_scan_state_);
    free_buffer(q35_scan_core_);
    free_buffer(q35_scan_conv_stage_);
    free_buffer(q35_scan_a_stage_);
    free_buffer(q35_scan_b_stage_);
    free_buffer(q35_scan_z_stage_);
    free_buffer(q35_scan_A_stage_);
    free_buffer(q35_scan_dt_stage_);
    free_buffer(q35_scan_norm_stage_);
    free_buffer(q35_scan_state_stage_);
    free_buffer(q35_scan_core_stage_);
    auto destroy_pipe = [&](Pipeline& p) {
        if (p.pipeline)  vkDestroyPipeline(device_, p.pipeline, nullptr);
        if (p.layout)    vkDestroyPipelineLayout(device_, p.layout, nullptr);
        if (p.ds_layout) vkDestroyDescriptorSetLayout(device_, p.ds_layout, nullptr);
        p = {};
    };
    destroy_pipe(pipe_f32_);
    destroy_pipe(pipe_f16_);
    destroy_pipe(pipe_q8_0_);
    destroy_pipe(pipe_q4k_);
    destroy_pipe(pipe_q5k_);
    destroy_pipe(pipe_q6k_);
    destroy_pipe(pipe_q2k_);
    destroy_pipe(pipe_mm_q2k_);
    destroy_pipe(pipe_q3k_);
    destroy_pipe(pipe_mm_q3k_);
    destroy_pipe(pipe_nq4g64_);
    destroy_pipe(pipe_mm_nq4g64_);
    destroy_pipe(pipe_mm_nq4g64_big_);
    destroy_pipe(pipe_nq5g64_);
    destroy_pipe(pipe_mm_nq5g64_);
    destroy_pipe(pipe_nq6g64_);
    destroy_pipe(pipe_mm_nq6g64_);
    destroy_pipe(pipe_nq8g64_);
    destroy_pipe(pipe_mm_nq8g64_);
    destroy_pipe(pipe_q35_conv_);
    destroy_pipe(pipe_q35_scan_);
    destroy_pipe(pipe_q35_scan_r_);
    destroy_pipe(pipe_q35_conv_step_);
    destroy_pipe(pipe_q35_aprep_); destroy_pipe(pipe_q35_apart_); destroy_pipe(pipe_q35_apart2_); destroy_pipe(pipe_q35_apart3_); destroy_pipe(pipe_q35_acomb_);
    for (auto& kv : q35_kv_mirror_) { free_buffer(kv.second.k); free_buffer(kv.second.v); }
    q35_kv_mirror_.clear(); kv_mirror_used_ = 0;
    for (GpuBuffer* b : { &q35_a_qg_, &q35_a_k_, &q35_a_v_, &q35_a_q_, &q35_a_gate_, &q35_a_out_, &q35_a_part_, &q35_a_kvout_, &q35_a_tab_, &q35_kv_stage_ }) free_buffer(*b);
    for (auto& kv : q35_dev_state_) { free_buffer(kv.second.conv); free_buffer(kv.second.rec); }
    q35_dev_state_.clear();
    for (GpuBuffer* b : { &q35_t_mixed_, &q35_t_cout_, &q35_t_z_, &q35_t_a_, &q35_t_b_, &q35_t_core_, &q35_t_stage_conv_, &q35_t_stage_rec_ }) free_buffer(*b);
    destroy_pipe(pipe_attn_);
    destroy_pipe(pipe_attn_batch_);
    destroy_pipe(pipe_attn_stream_);
    destroy_pipe(pipe_attn_flash_);
    destroy_pipe(pipe_vision_attn_full_);
    destroy_pipe(pipe_rope_);
    destroy_pipe(pipe_rms_);
    destroy_pipe(pipe_silu_);
    destroy_pipe(pipe_add_);
    if (ops_qp_)        vkDestroyQueryPool(device_, ops_qp_, nullptr);
    if (chain_ready_) chain_wait_all();
    for (auto& kv : q35_lin_cache_) lin_cache_drop(kv.second);
    q35_lin_cache_.clear();
    if (chain_ready_) { for (int i = 0; i < 2; ++i) if (cslot_fence_[i]) vkDestroyFence(device_, cslot_fence_[i], nullptr); if (cslot_pool_[1]) vkDestroyDescriptorPool(device_, cslot_pool_[1], nullptr); }
    for (GpuBuffer& b : q35_a_kvring_) free_buffer(b);
    destroy_pipe(pipe_q4k_dp4_); destroy_pipe(pipe_q5k_dp4_); destroy_pipe(pipe_q6k_dp4_); destroy_pipe(pipe_quant_q8n_);
    if (dq_pool_)       vkDestroyDescriptorPool(device_, dq_pool_, nullptr);
    free_buffer(dq_xq_); free_buffer(dq_xs_);
    if (ds_pool_)       vkDestroyDescriptorPool(device_, ds_pool_, nullptr);
    if (transfer_cmd_pool_) vkDestroyCommandPool(device_, transfer_cmd_pool_, nullptr);
    if (cmd_pool_)      vkDestroyCommandPool(device_, cmd_pool_, nullptr);
    if (device_)        vkDestroyDevice(device_, nullptr);
    if (instance_)      vkDestroyInstance(instance_, nullptr);
    ready_ = false;
}

inline void VulkanBackend::print_info() const {
    fprintf(stderr, "\n[vulkan] --------------------------------------\n");
    fprintf(stderr, "[vulkan]  GPU: %s\n", info_.device_name.c_str());
    fprintf(stderr, "[vulkan]  Memory: %s", info_.unified_memory ? "Unified (APU)" : "Dedicated");
    if (info_.vram_bytes > 0)
        fprintf(stderr, " — %.0f MB reserved\n", info_.vram_bytes / 1048576.0);
    else fprintf(stderr, "\n");
    fprintf(stderr, "[vulkan]  Weight storage:   %s\n",
            (info_.unified_memory && info_.host_import)
                ? "Zero-copy (APU unified memory)"
                : info_.host_import
                    ? "DEVICE_LOCAL copy (discrete GPU)"
                    : "DEVICE_LOCAL copy");
    fprintf(stderr, "[vulkan]  GPU layers: %d\n", n_gpu_layers_);
    fprintf(stderr, "[vulkan] --------------------------------------\n\n");
}

#else  // !NEXUS_VULKAN

// Minimal GpuBuffer stub — only the fields transformer.h touches
struct GpuBuffer {
    void*  buffer     = nullptr;  // non-null = "valid" (pointer stands in for VkBuffer)
    size_t size       = 0;
    void*  mapped_ptr = nullptr;
    bool   imported   = false;
    const void* host_ptr = nullptr;
};

// ── Stub when Vulkan not compiled in ─────────────────────────────────────────
class VulkanBackend {
public:
    bool init(bool = false)  { return false; }
    void set_device_index(int) {}
    void shutdown()          {}
    void wait_idle()         {}
    size_t release_prefill_buffers() { return 0; }
    bool ready()       const { return false; }
    int  n_gpu_layers()const { return 0; }
    GpuInfo info()     const { return {}; }
    void set_gpu_layers(int) {}
    void print_info()  const { fprintf(stderr,"[vulkan] Not compiled in\n"); }
    bool mat_vec(const void*, const float*, float*, size_t, size_t, QuantType) { return false; }
    bool mat_mat(const void*, const float*, float*, size_t, size_t, size_t, QuantType) { return false; }
    bool mat_mat_supported(QuantType) const { return false; }
    bool ffn_prefill(const void*,const void*,const void*,const float*,float*,size_t,size_t,size_t,QuantType){ return false; }
    bool qwen35_conv1d(const float*, const float*, const float*, float*, uint32_t, uint32_t, uint32_t) { return false; }
    bool qwen35_scan(const float*, const float*, const float*, const float*, const float*, const float*, const float*, float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool, float) { return false; }
    bool mat_vec_pair(const void*, float*, size_t, QuantType,
                      const void*, float*, size_t, QuantType,
                      const float*, size_t) { return false; }
    bool mat_vec_qkv(const void*, float*, size_t, QuantType,
                     const void*, float*, size_t, QuantType,
                     const void*, float*, size_t, QuantType,
                     const float*, size_t) { return false; }
    bool attention_gpu(const float*, float*, int, int, int, int, int,
                       const void*, const void*, size_t) { return false; }
    bool attention_batch_gpu(const float*, float*, int, int, int, int, int, int, int,
                             float, float, const void*, const void*, size_t) { return false; }
    bool attention_stream_gpu(const float*, float*, int, int, int, int, int, int,
                              float, float, const void*, const void*, int) { return false; }
    bool attention_flash_gpu(const float*, float*, int, int, int, int, int, int, int,
                             float, float, const void*, const void*) { return false; }
    bool vision_attention_full_gpu(const float*, const float*, const float*, float*,
                                   int, int, int, float) { return false; }
    void init_activation_bufs(size_t,size_t,size_t,size_t) {}
    bool act_bufs_ok()  const { return false; }
    bool pipe_rms_ok()  const { return false; }
    bool pipe_rope_ok() const { return false; }
    bool pipe_silu_ok() const { return false; }
    bool pipe_add_ok()  const { return false; }
    bool record_quant_act_g(GpuBuffer&, size_t) { return false; }
    bool decode_int8_active() const { return false; }
    void begin_layer_record(const float*, size_t) {}
    void partial_submit() {}
    void end_layer_record(float*, size_t) {}
    void layer_barrier() {}
    GpuBuffer& act_hidden_buf(){ static GpuBuffer d; return d; }
    GpuBuffer& act_xb_buf()    { static GpuBuffer d; return d; }
    GpuBuffer& act_q_buf()     { static GpuBuffer d; return d; }
    GpuBuffer& act_k_buf()     { static GpuBuffer d; return d; }
    GpuBuffer& act_v_buf()     { static GpuBuffer d; return d; }
    GpuBuffer& act_attn_out_buf(){ static GpuBuffer d; return d; }
    GpuBuffer& act_gate_buf()  { static GpuBuffer d; return d; }
    GpuBuffer& act_up_buf()    { static GpuBuffer d; return d; }
    GpuBuffer& act_ffn_out_buf(){ static GpuBuffer d; return d; }
    bool record_rms_norm_g(GpuBuffer&,const void*,size_t,GpuBuffer&,uint32_t,float){return false;}
    bool record_mat_vec_g(const void*,QuantType,GpuBuffer&,GpuBuffer&,size_t,size_t){return false;}
    bool record_mat_vec_pair_g(const void*,QuantType,GpuBuffer&,const void*,QuantType,GpuBuffer&,size_t,GpuBuffer&,size_t){return false;}
    bool record_rope_g(GpuBuffer&,GpuBuffer&,uint32_t,uint32_t,uint32_t,uint32_t,float,float,uint32_t=0){return false;}
    bool record_attention_g(GpuBuffer&,GpuBuffer&,int,int,int,int,const uint16_t*,const uint16_t*,size_t){return false;}
    bool record_silu_mul_g(GpuBuffer&,GpuBuffer&,uint32_t){return false;}
    bool record_add_g(GpuBuffer&,GpuBuffer&,uint32_t){return false;}
    struct Q35LinearDesc { const void* owner = nullptr; int layer = 0; };
    bool q35_linear_layer(const Q35LinearDesc&, float*, const float*, const float*, bool = false) { return false; }
    bool chain_is_open() const { return false; }
    bool chain_end(float*, size_t) { return false; }
    uint64_t lin_cache_hits() const { return 0; }
    uint64_t lin_cache_misses() const { return 0; }
    void invalidate_layer_cache() {}
    const float* chain_kv_ptr(int) const { return nullptr; }
    static constexpr int kChainKvSlots = 16;
    bool q35_state_resident(const void*, int) const { return false; }
    bool q35_state_download(const void*, int, float*, float*) { return false; }
    void q35_state_drop(const void*, int) {}
    struct Q35FullDesc { const void* owner = nullptr; int layer = 0; };
    bool q35_full_layer(const Q35FullDesc&, float*, float*, uint32_t*, int = -1) { return false; }
    void q35_kv_drop(const void*, int) {}
    void set_gpu_kv_mb(int) {}
    int  gpu_kv_mb() const { return 0; }
    size_t kv_mirror_used() const { return 0; }
    size_t kv_mirror_cap_bytes() const { return 0; }
    bool vram_free_now(size_t& f) const { f = 0; return false; }
    // Helpers for lm_head eager upload (no-op in stub)
    bool size_for_public(QuantType, size_t, size_t, size_t&, uint32_t&) { return false; }
    void ensure_activation_buffers_public(size_t, size_t) {}
    bool get_weight_buffer_public(const void*, size_t, QuantType) { return false; }
    size_t weight_cache_used() const { return 0; }
    bool prefetch_weight_public(const void*, size_t, QuantType) { return false; }
    bool load_weight_force_public(const void*, size_t, QuantType, bool = false) { return false; }
    bool load_weight_async_public(const void*, size_t, QuantType, bool = true) { return false; }
    bool prof_enabled() const { return false; }
    void prof_reset() {}
    void pin_weight_public(const void*) {}
    void unpin_weight_public(const void*) {}
    bool evict_weight_public(const void*) { return false; }
    size_t weight_size_public(const void*) { return 0; }
    void set_host_only_weights(bool) {}
    bool host_import_available() const { return false; }
};

#endif  // NEXUS_VULKAN

// ─────────────────────────────────────────────────────────────────────────────
//  Global GPU instance (one per process)
// ─────────────────────────────────────────────────────────────────────────────
inline VulkanBackend& gpu() {
    static VulkanBackend instance;
    return instance;
}

}  // namespace gpu
}  // namespace nexus
