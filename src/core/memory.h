/*
 * nexus/src/core/memory.h
 *
 * The memory manager. This is where llama.cpp goes wrong and we go right.
 *
 * Design goals:
 *   1. Huge pages for weight storage (2MB pages = 512x fewer TLB misses)
 *   2. Explicit pools: weights never mix with KV cache never mix with activations
 *   3. Layer pinning: mlock layers that are about to compute, unlock when done
 *   4. Pressure response: when RAM is tight, we handle it — we don't hang
 *   5. Every allocation is named and tracked — full visibility into what's where
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <functional>

namespace nexus {

// ─── Allocation types ──────────────────────────────────────────────────────

enum class MemPool : uint8_t {
    WEIGHTS,      // Model weight tensors. Huge pages. Long-lived.
    KV_CACHE,     // Attention KV cache. Chunked. Dynamic lifetime.
    ACTIVATIONS,  // Per-token intermediate values. Reused constantly.
    SCRATCH,      // Temporary computation space. Very short-lived.
};

enum class PageType : uint8_t {
    STANDARD,     // 4KB pages — fallback only
    HUGE_2MB,     // 2MB transparent huge pages — preferred for weights
    HUGE_1GB,     // 1GB huge pages — for very large weight blocks if available
};

// ─── Allocation record ─────────────────────────────────────────────────────

struct Allocation {
    void*       ptr;
    size_t      size;
    MemPool     pool;
    PageType    page_type;
    bool        locked;       // mlock'd in RAM?
    bool        on_gpu;       // mirrored to GPU?
    std::string name;         // "layer_12_attn_q" etc — for diagnostics
    uint64_t    alloc_time_ns;
    uint64_t    last_access_ns;
    int         layer_id;     // -1 if not layer-specific
};

// ─── Memory report ─────────────────────────────────────────────────────────

struct MemoryReport {
    size_t weights_total_bytes;
    size_t weights_locked_bytes;
    size_t kv_cache_total_bytes;
    size_t kv_pages_allocated;
    size_t kv_pages_free;
    size_t activation_pool_bytes;
    size_t scratch_pool_bytes;
    size_t system_ram_available_bytes;
    size_t system_ram_total_bytes;
    size_t huge_pages_used;
    size_t huge_pages_available;
    std::vector<Allocation> allocations;
};

// ─── KV Cache Page ─────────────────────────────────────────────────────────

struct KVPage {
    void*    k_data;           // [page_tokens × n_kv_heads × head_dim] f16
    void*    v_data;           // same shape
    uint32_t token_start;
    uint32_t token_count;
    uint32_t page_capacity;    // max tokens this page holds
    bool     locked;
    bool     on_gpu;
    uint64_t last_access_ns;
    int      page_index;
};

// ─── Memory Manager ────────────────────────────────────────────────────────

class MemoryManager {
public:
    struct Config {
        size_t   kv_page_tokens    = 256;    // tokens per KV cache page
        size_t   activation_pool   = 64*1024*1024;  // 64MB activation ring
        size_t   scratch_pool      = 32*1024*1024;  // 32MB scratch space
        float    pressure_threshold = 0.85f;  // trigger pressure response at 85% RAM use
        bool     prefer_huge_pages = true;
        bool     lock_active_layers = true;
        int      max_locked_layers = 4;       // keep N layers locked at once
    };

    MemoryManager();                          // uses default Config
    explicit MemoryManager(const Config& cfg);
    ~MemoryManager();

    // Non-copyable, non-movable (owns raw memory)
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    // ── Weight allocation ─────────────────────────────────────────────────

    // Allocate weight tensor memory. Returns huge-page aligned pointer.
    // Name is for diagnostics: "layer_0_attn_q_weight"
    void* alloc_weights(size_t bytes, const std::string& name, int layer_id = -1);

    // Free a weight allocation
    void free_weights(void* ptr);

    // Pin a layer's weights in RAM before it computes.
    // Evicts oldest pinned layer if at max_locked_layers.
    void pin_layer(int layer_id);

    // Unpin layer — weights may be paged out under pressure
    void unpin_layer(int layer_id);

    // ── KV cache ─────────────────────────────────────────────────────────

    // Allocate a new KV page for tokens [token_start, token_start+page_tokens)
    // n_layers × n_kv_heads × head_dim defines the shape
    KVPage* alloc_kv_page(uint32_t token_start,
                          int n_layers, int n_kv_heads, int head_dim);

    // Free a specific KV page
    void free_kv_page(KVPage* page);

    // Free ALL KV pages (context clear / new conversation)
    void clear_kv_cache();

    // Get KV page containing token position
    KVPage* get_kv_page(uint32_t token_pos);

    // ── Activation pool ───────────────────────────────────────────────────

    // Borrow scratch memory from the activation ring buffer.
    // Must be returned before next forward pass layer.
    void* borrow_activation(size_t bytes);
    void  return_activation(void* ptr);

    // ── Pressure handling ─────────────────────────────────────────────────

    // Poll system memory. Returns true if under pressure.
    bool check_pressure();

    // Called when pressure detected. Frees what it can.
    // Returns bytes freed.
    size_t handle_pressure();

    // Register callback for pressure events
    void on_pressure(std::function<void(size_t available_bytes)> cb);

    // ── Diagnostics ───────────────────────────────────────────────────────

    MemoryReport report() const;
    void print_report() const;

    // Human-readable size string
    static std::string fmt_bytes(size_t bytes);

private:
    Config cfg_;
    mutable std::mutex mutex_;

    // Weight allocations
    std::vector<Allocation> weight_allocs_;
    std::unordered_map<void*, size_t> ptr_to_alloc_idx_;

    // Layer pin tracking
    std::vector<int>  pinned_layers_;   // currently locked layer IDs
    std::unordered_map<int, std::vector<void*>> layer_ptrs_; // layer_id → its allocs

    // KV cache
    std::vector<KVPage*> kv_pages_;
    size_t kv_page_tokens_;

    // Activation ring buffer
    uint8_t* activation_pool_  = nullptr;
    size_t   activation_size_  = 0;
    std::atomic<size_t> activation_head_{0};

    // Scratch pool
    uint8_t* scratch_pool_     = nullptr;
    size_t   scratch_size_     = 0;

    // Pressure callbacks
    std::vector<std::function<void(size_t)>> pressure_callbacks_;

    // Internal allocation
    void* alloc_huge(size_t bytes, const std::string& name);
    void  free_huge(void* ptr, size_t bytes);
    bool  try_lock(void* ptr, size_t bytes);
    void  do_unlock(void* ptr, size_t bytes);

    size_t system_available_bytes() const;
    size_t system_total_bytes() const;
    uint64_t now_ns() const;
};

// ─── RAII layer pin guard ──────────────────────────────────────────────────

class LayerPinGuard {
public:
    LayerPinGuard(MemoryManager& mm, int layer_id)
        : mm_(mm), layer_id_(layer_id) {
        mm_.pin_layer(layer_id_);
    }
    ~LayerPinGuard() {
        mm_.unpin_layer(layer_id_);
    }
private:
    MemoryManager& mm_;
    int layer_id_;
};

} // namespace nexus
