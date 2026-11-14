/*
 * nexus/src/model/kvcache.h
 *
 * Contiguous KV cache — single growing allocation per (layer, kv_head).
 *
 * LAYOUT: [layer * n_kv_heads + kv_head] → flat f16 array
 *   K: [capacity × head_dim] f16  — tokens 0..pos written contiguously
 *   V: same
 *
 * WHY CONTIGUOUS (vs old paged design):
 *   1. GPU attention shader can read K/V directly with ONE buffer binding —
 *      no pack_kv memcpy needed. On APU (unified memory) this is zero-copy.
 *   2. At 50K token context, old pack_kv copied 3.1 GB per token. Now: 0 bytes.
 *   3. Single allocation per head is cache-friendlier for CPU attention too.
 *
 * GROWTH: capacity doubles when full (like std::vector).
 *   Initial capacity: 512 tokens. Reallocates transparently.
 *   For GPU: gpu_dirty_ flag marks when the GPU buffer handle needs refresh.
 *
 * ALIGNMENT: each K/V buffer is aligned to 64 bytes so import_host_buffer
 *   works on any Vulkan driver (min alignment typically 64-4096 bytes).
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#if defined(__linux__)
#  include <sys/mman.h>
#  include <unistd.h>
#endif
#include "../gguf/quant.h"
#include "../core/memguard.h"

#if defined(__AVX2__)
#  include <immintrin.h>
#  define KV_AVX2 1
#else
#  define KV_AVX2 0
#endif

// Legacy compat: old code references KV_PAGE_SIZE
static constexpr int KV_PAGE_SIZE = 256;

namespace nexus {
namespace model {

// One slot per (layer × kv_head): contiguous f16 K and V arrays.
struct KVSlot {
    uint16_t* k        = nullptr;  // [capacity × head_dim] f16
    uint16_t* v        = nullptr;  // same
    size_t    capacity = 0;        // tokens allocated
    size_t    bytes    = 0;        // bytes mapped per buffer (k and v are equal)
    size_t    mirror_valid = 0;    // tokens [0,mirror_valid) unchanged since the GPU-side mirror (Qwen3.5 decode) was last in sync
    bool      gpu_dirty = true;    // true → GPU buffer handle needs refresh
};

class KVCache {
public:
    KVCache() = default;
    ~KVCache() { clear(); }

    void init(int n_layers, int n_kv_heads, int head_dim) {
        n_layers_   = n_layers;
        n_kv_heads_ = n_kv_heads;
        head_dim_   = head_dim;
        per_layer_  = false;
        n_kv_heads_layer_.clear();
        head_dim_layer_.clear();
        layer_slot_offset_.clear();
        slots_.resize((size_t)n_layers * n_kv_heads);
    }

    // Per-layer init for architectures with variable KV geometry (Gemma 4).
    // n_kv_heads_per_layer[L] and head_dim_per_layer[L] describe layer L.
    //
    // After this returns, slots_ is sized to sum(n_kv_heads_per_layer) and
    // layer_slot_offset_ is the prefix sum so slot(L, h) lands in the right
    // place. n_kv_heads_ / head_dim_ are populated with the MAX across layers
    // so callers asking for "the global kv geometry" (memory_bytes(), some
    // diag prints, save/load headers) get sensible answers.
    void init_per_layer(const std::vector<int>& n_kv_heads_per_layer,
                        const std::vector<int>& head_dim_per_layer) {
        const int L = (int)n_kv_heads_per_layer.size();
        n_layers_         = L;
        per_layer_        = true;
        n_kv_heads_layer_ = n_kv_heads_per_layer;
        head_dim_layer_   = head_dim_per_layer;

        layer_slot_offset_.assign((size_t)L, 0);
        size_t total_slots = 0;
        int max_h = 0, max_d = 0;
        for (int l = 0; l < L; ++l) {
            layer_slot_offset_[l] = total_slots;
            total_slots += (size_t)n_kv_heads_per_layer[l];
            if (n_kv_heads_per_layer[l] > max_h) max_h = n_kv_heads_per_layer[l];
            if (head_dim_per_layer[l]   > max_d) max_d = head_dim_per_layer[l];
        }
        n_kv_heads_ = max_h;
        head_dim_   = max_d;
        slots_.resize(total_slots);
    }

    // v26: lightweight per-layer init for TRUE single-layer streaming (hybrid prefill
    // ghost engines). Sizes the bookkeeping arrays to the full layer count -- so
    // indexing by a real, not-yet-reached layer number is always valid -- but does NOT
    // touch any model/file data and allocates no slots yet. Each layer's real geometry
    // and its slots get added one at a time, in order, by add_layer_geometry() below,
    // the first time that layer is genuinely streamed in for real.
    void init_per_layer_streaming(int n_layers) {
        n_layers_   = n_layers;
        per_layer_  = true;
        n_kv_heads_layer_.assign((size_t)n_layers, 0);
        head_dim_layer_.assign((size_t)n_layers, 0);
        layer_slot_offset_.assign((size_t)n_layers, 0);
        layer_geom_known_.assign((size_t)n_layers, false);
        slots_.clear();
        n_kv_heads_ = 0;
        head_dim_   = 0;
        n_tokens_   = 0;
    }
    // v26: records layer `layer`'s real KV geometry and appends its slots to the end of
    // the flat slots_ array. MUST be called with layer indices in non-decreasing order --
    // each call's offset is simply "however many slots exist so far", which only stays
    // correct if every later layer is added after this one, never before (true of the
    // one real per-layer streaming loop that calls this). Idempotent: a layer already
    // recorded is left alone instead of being appended a second time.
    void add_layer_geometry(int layer, int nh, int hd) {
        if (layer < 0 || layer >= n_layers_) return;
        if ((size_t)layer < layer_geom_known_.size() && layer_geom_known_[(size_t)layer]) return;
        n_kv_heads_layer_[(size_t)layer]  = nh;
        head_dim_layer_[(size_t)layer]    = hd;
        layer_slot_offset_[(size_t)layer] = slots_.size();
        slots_.resize(slots_.size() + (size_t)std::max(0, nh));
        if (nh > n_kv_heads_) n_kv_heads_ = nh;
        if (hd > head_dim_)   head_dim_   = hd;
        if ((size_t)layer < layer_geom_known_.size()) layer_geom_known_[(size_t)layer] = true;
    }

    // Store K and V for position pos in layer layer.
    // k_data, v_data: float32 [n_kv_heads_for_layer * head_dim_for_layer]
    void store(int layer, int pos, const float* k_data, const float* v_data,
               int kv_dim /* = n_kv_heads_for_layer * head_dim_for_layer */)
    {
        const int hd = head_dim_for(layer);
        const int nh = kv_heads_for(layer);
        if (nh == 0) {
            static bool warned0 = false;
            if (!warned0) { warned0 = true;
                fprintf(stderr, "[kv] WARNING: something tried to store K/V for layer %d, which has no KV (ignored)\n", layer); }
        }
        for (int h = 0; h < nh; ++h) {
            KVSlot& s = slot(layer, h);
            ensure_capacity_layer(s, pos + 1, hd);
            if ((size_t)(pos + 1) > s.capacity) {
                static bool warned = false;
                if (!warned) { warned = true;
                    fprintf(stderr, "[kv] OUT OF MEMORY growing KV cache at pos %d (layer %d)\n", pos, layer); }
                continue;   // do not write past the allocation
            }
            if ((size_t)pos < s.mirror_valid) s.mirror_valid = (size_t)pos;   // overwriting an old position invalidates the GPU mirror from there
            const float* ksrc = k_data + (size_t)h * hd;
            const float* vsrc = v_data + (size_t)h * hd;
            uint16_t* kdst = s.k + (size_t)pos * hd;
            uint16_t* vdst = s.v + (size_t)pos * hd;
            for (int d = 0; d < hd; ++d) {
                kdst[d] = f32_to_f16(ksrc[d]);
                vdst[d] = f32_to_f16(vsrc[d]);
            }
        }
        if (pos + 1 > n_tokens_) n_tokens_ = pos + 1;
    }

    // Raw f16 pointer to K[layer][kv_head] — entire token sequence, contiguous.
    // Layout: [token 0..n_ctx-1][head_dim] f16
    // No copy — direct pointer into the allocation.
    const uint16_t* k_ptr(int layer, int kv_head) const {
        return slot(layer, kv_head).k;
    }
    const uint16_t* v_ptr(int layer, int kv_head) const {
        return slot(layer, kv_head).v;
    }

    // Mutable raw K/V access for RoPE position surgery. This does not change
    // token count or allocation shape; callers must keep positions valid.
    uint16_t* k_mut_ptr(int layer, int kv_head) {
        return slot(layer, kv_head).k;
    }
    uint16_t* v_mut_ptr(int layer, int kv_head) {
        return slot(layer, kv_head).v;
    }
    int kv_heads_for_layer(int layer) const { return kv_heads_for(layer); }
    int head_dim_for_layer(int layer) const { return head_dim_for(layer); }
    void mark_slot_gpu_dirty(int layer, int kv_head) {
        slot(layer, kv_head).gpu_dirty = true; slot(layer, kv_head).mirror_valid = 0;
    }
    void set_k_f32(int layer, int pos, int kv_head, const float* in) {
        const int hd = head_dim_for(layer);
        KVSlot& s = slot(layer, kv_head);
        if (!s.k || pos < 0 || pos >= n_tokens_) return;
        uint16_t* p = s.k + (size_t)pos * hd;
        for (int d = 0; d < hd; ++d) p[d] = f32_to_f16_rne(in[d]);
        s.gpu_dirty = true; s.mirror_valid = 0;
    }

    // ── Raw range access, for context editing (archive / restore part of a conversation) ──
    // Copy positions [pos0, pos0+n) of one (layer, kv_head) slot out as raw f16.
    bool read_raw(int layer, int kv_head, int pos0, int n, uint16_t* k_out, uint16_t* v_out) const {
        const int hd = head_dim_for(layer);
        if (pos0 < 0 || n < 0 || pos0 + n > n_tokens_ || hd <= 0) return false;
        const KVSlot& s = slot(layer, kv_head);
        const size_t off = (size_t)pos0 * hd, cnt = (size_t)n * hd;
        if (s.k && (size_t)(pos0 + n) <= s.capacity) std::memcpy(k_out, s.k + off, cnt * sizeof(uint16_t)); else std::memset(k_out, 0, cnt * sizeof(uint16_t));
        if (s.v && (size_t)(pos0 + n) <= s.capacity) std::memcpy(v_out, s.v + off, cnt * sizeof(uint16_t)); else std::memset(v_out, 0, cnt * sizeof(uint16_t));
        return true;
    }
    // Write raw f16 K/V for positions [pos0, pos0+n) of one slot, growing it if needed.
    // Does NOT change n_tokens() -- call set_n_tokens() once every slot has been written, so a
    // failure part-way (out of memory) leaves the cache exactly as it was.
    bool write_raw(int layer, int kv_head, int pos0, int n, const uint16_t* k_in, const uint16_t* v_in) {
        const int hd = head_dim_for(layer);
        if (pos0 < 0 || n < 0 || hd <= 0) return false;
        KVSlot& s = slot(layer, kv_head);
        ensure_capacity_layer(s, pos0 + n, hd);
        if ((size_t)(pos0 + n) > s.capacity || !s.k || !s.v) return false;
        const size_t off = (size_t)pos0 * hd, cnt = (size_t)n * hd;
        std::memcpy(s.k + off, k_in, cnt * sizeof(uint16_t));
        std::memcpy(s.v + off, v_in, cnt * sizeof(uint16_t));
        if ((size_t)pos0 < s.mirror_valid) s.mirror_valid = (size_t)pos0;
        s.gpu_dirty = true;
        return true;
    }
    // Set the token count directly (only for context editing: after write_raw() above, or to drop
    // stale positions past the real end, e.g. a rejected speculative draft).
    void set_n_tokens(int n) { if (n >= 0) n_tokens_ = n; }
    // How many tokens fit without any buffer growing: the smallest capacity of any allocated slot.
    size_t min_capacity_tokens() const {
        size_t m = (size_t)-1; bool any = false;
        for (const auto& s : slots_) if (s.k) { any = true; m = std::min(m, s.capacity); }
        return any ? m : 0;
    }
    // Bytes of K+V one token takes across every layer that holds KV.
    size_t bytes_per_token() const {
        size_t b = 0;
        for (int L = 0; L < n_layers_; ++L) b += (size_t)kv_heads_for(L) * (size_t)head_dim_for(L) * 2 * sizeof(uint16_t);
        return b;
    }

    // f32 -> f16 with round-to-nearest-even (the store path keeps its own fast truncating conversion).
    static uint16_t f32_to_f16_rne(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        const uint32_t ax = x & 0x7FFFFFFFu;
        if (ax >= 0x7F800000u) return (uint16_t)(sign | 0x7C00u | (ax > 0x7F800000u ? 0x200u : 0u));   // inf / nan
        if (ax >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u);                                       // too big -> inf
        if (ax < 0x38800000u) {                                                                         // f16 subnormal or zero
            if (ax < 0x33000000u) return (uint16_t)sign;
            const uint32_t e = ax >> 23, m = (ax & 0x7FFFFFu) | 0x800000u, shift = 126u - e;
            uint32_t h = m >> shift;
            const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1u);
            if (rem > half || (rem == half && (h & 1u))) ++h;
            return (uint16_t)(sign | h);
        }
        uint32_t h = (ax - 0x38000000u) >> 13;
        const uint32_t rem = ax & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
        return (uint16_t)(sign | h);
    }

    // GPU mirror bookkeeping (per layer = min over its heads): how many leading tokens the mirror still matches.
    size_t mirror_valid(int layer) const {
        const int nh = kv_heads_for(layer); size_t v = (size_t)-1;
        for (int h = 0; h < nh; ++h) v = std::min(v, slot(layer, h).mirror_valid);
        return nh > 0 ? v : 0;
    }
    void set_mirror_valid(int layer, size_t n) {
        const int nh = kv_heads_for(layer);
        for (int h = 0; h < nh; ++h) slot(layer, h).mirror_valid = n;
    }

    // True if the GPU buffer handle for this slot is stale (capacity grew).
    bool gpu_dirty(int layer, int kv_head) const {
        return slot(layer, kv_head).gpu_dirty;
    }
    void gpu_synced(int layer, int kv_head) {
        slot(layer, kv_head).gpu_dirty = false;
    }
    size_t slot_capacity(int layer, int kv_head) const {
        return slot(layer, kv_head).capacity;
    }

    void mark_all_gpu_dirty() {
        for (auto& s : slots_) { s.gpu_dirty = true; s.mirror_valid = 0; }
    }

    // Legacy paged-API compat: returns pointer to a contiguous run of tokens.
    // page_idx selects which 256-token window. n_out = tokens in that window.
    const uint16_t* get_k_page(int layer, int kv_head, int page_idx, int* n_out) const {
        return get_page(true, layer, kv_head, page_idx, n_out);
    }
    const uint16_t* get_v_page(int layer, int kv_head, int page_idx, int* n_out) const {
        return get_page(false, layer, kv_head, page_idx, n_out);
    }

    void clear() {
        for (auto& s : slots_) free_slot(s);
        if (!slots_.empty()) slots_.assign(slots_.size(), KVSlot{});
        n_tokens_ = 0;
    }

    int    n_tokens()      const { return n_tokens_; }
    int    n_kv_heads()    const { return n_kv_heads_; }
    int    head_dim()      const { return head_dim_; }

    size_t memory_bytes() const {      // allocated (capacity)
        size_t total = 0;
        for (const auto& s : slots_) if (s.k) total += 2 * s.bytes;
        return total;
    }
    // Bytes actually holding tokens, and how many layers hold any KV at all.
    size_t live_bytes() const {
        size_t total = 0;
        for (int L = 0; L < n_layers_; ++L) {
            const size_t dim = (size_t)head_dim_for(L);
            const int    nh  = kv_heads_for(L);
            for (int h = 0; h < nh; ++h)
                if (slot(L, h).k) total += 2 * (size_t)n_tokens_ * dim * sizeof(uint16_t);
        }
        return total;
    }
    int layers_with_kv() const {
        int n = 0;
        for (int L = 0; L < n_layers_; ++L) {
            const int nh = kv_heads_for(L);
            for (int h = 0; h < nh; ++h) if (slot(L, h).k) { ++n; break; }
        }
        return n;
    }
    int total_layers() const { return n_layers_; }

    // ── Persistence ────────────────────────────────────────────────────────
    // File format: magic(8) + header(n_layers,n_kv_heads,head_dim,n_tokens)(4xi32)
    // Then for each slot: k data [n_tokens*head_dim f16] then v data same.
    // ~200MB for 7B model at 2K context (acceptable for session save).

    bool save(const std::string& path) const {
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) return false;
        const char magic[8] = {'N','X','K','V','C','A','C','H'};
        fwrite(magic, 1, 8, f);
        int32_t hdr[4] = {n_layers_, n_kv_heads_, head_dim_, n_tokens_};
        fwrite(hdr, sizeof(int32_t), 4, f);
        for (int l = 0; l < n_layers_; ++l) {
            const size_t hd = (size_t)head_dim_for(l);
            const int    nh = kv_heads_for(l);
            for (int h = 0; h < nh; ++h) {
                const KVSlot& s = slot(l, h);
                size_t n = (size_t)n_tokens_ * hd;
                fwrite(s.k ? s.k : nullptr, sizeof(uint16_t), s.k ? n : 0, f);
                if (!s.k) { std::vector<uint16_t> z(n,0); fwrite(z.data(),sizeof(uint16_t),n,f); }
                fwrite(s.v ? s.v : nullptr, sizeof(uint16_t), s.v ? n : 0, f);
                if (!s.v) { std::vector<uint16_t> z(n,0); fwrite(z.data(),sizeof(uint16_t),n,f); }
            }
        }
        fclose(f);
        return true;
    }

    // Serialize KV cache directly to a memory buffer — no temp files.
    // Free just one layer's K/V memory (used by hybrid prefill right after that layer's data has
    // been handed to another machine and is no longer needed here). Leaves every other layer alone;
    // n_tokens_ is unchanged since other layers may still hold real data at that length.
    void free_layer(int l) {
        if (l < 0 || l >= n_layers_) return;
        const int nh = kv_heads_for(l);
        for (int h = 0; h < nh; ++h) { KVSlot& s = slot(l, h); free_slot(s); s = KVSlot{}; }
    }

    // Serialize JUST ONE layer's K/V to a buffer (header carries the layer index, so this is
    // self-contained and order-independent -- layers can be sent and applied one at a time, as
    // each finishes, instead of waiting for the whole cache). An empty return means this layer
    // genuinely has no KV at all (e.g. a linear-attention layer) -- nothing to send for it.
    std::vector<uint8_t> save_layer_to_buffer(int l) const {
        std::vector<uint8_t> buf;
        if (l < 0 || l >= n_layers_) return buf;
        const int nh = kv_heads_for(l);
        if (nh == 0) return buf;
        const size_t hd = (size_t)head_dim_for(l);
        const char magic[8] = {'N','X','K','V','L','Y','R','1'};
        int32_t hdr[4] = {l, nh, (int32_t)hd, n_tokens_};
        buf.insert(buf.end(), magic, magic+8);
        buf.insert(buf.end(), (uint8_t*)hdr, (uint8_t*)(hdr+4));
        for (int h = 0; h < nh; ++h) {
            const KVSlot& s = slot(l, h);
            size_t n = (size_t)n_tokens_ * hd;
            if (s.k) buf.insert(buf.end(),(uint8_t*)s.k,(uint8_t*)(s.k+n));
            else buf.resize(buf.size() + n*sizeof(uint16_t), 0);
            if (s.v) buf.insert(buf.end(),(uint8_t*)s.v,(uint8_t*)(s.v+n));
            else buf.resize(buf.size() + n*sizeof(uint16_t), 0);
        }
        return buf;
    }
    // Apply one layer's K/V, as produced by save_layer_to_buffer() above (possibly on a totally
    // different machine). Grows this cache's token count if this layer's buffer covers more
    // tokens than have been stored here yet; never shrinks it.
    bool load_layer_from_buffer(const uint8_t* data, size_t size) {
        if (size < 24 || memcmp(data, "NXKVLYR1", 8) != 0) return false;
        const int32_t* hdr = (const int32_t*)(data + 8);
        const int l = hdr[0], nh = hdr[1], hd_in = hdr[2], n_tok = hdr[3];
        if (l < 0 || l >= n_layers_ || nh != kv_heads_for(l) || hd_in != head_dim_for(l)) return false;
        const size_t hd = (size_t)hd_in;
        const size_t slot_bytes = (size_t)n_tok * hd * sizeof(uint16_t);
        const uint8_t* p = data + 8 + 4 * sizeof(int32_t);
        for (int h = 0; h < nh; ++h) {
            KVSlot& s = slot(l, h);
            ensure_capacity_layer(s, n_tok, (int)hd);
            if ((int)s.capacity < n_tok) return false;
            if (p + slot_bytes > data + size) return false;
            memcpy(s.k, p, slot_bytes); p += slot_bytes;
            if (p + slot_bytes > data + size) return false;
            memcpy(s.v, p, slot_bytes); p += slot_bytes;
            s.gpu_dirty = true; s.mirror_valid = 0;
        }
        if (n_tok > n_tokens_) n_tokens_ = n_tok;
        return true;
    }

    std::vector<uint8_t> save_to_buffer() const {
        std::vector<uint8_t> buf;
        const char magic[8] = {'N','X','K','V','C','A','C','H'};
        int32_t hdr[4] = {n_layers_, n_kv_heads_, head_dim_, n_tokens_};
        buf.insert(buf.end(), magic, magic+8);
        buf.insert(buf.end(), (uint8_t*)hdr, (uint8_t*)(hdr+4));
        for (int l = 0; l < n_layers_; ++l) {
            const size_t hd = (size_t)head_dim_for(l);
            const int    nh = kv_heads_for(l);
            for (int h = 0; h < nh; ++h) {
                const KVSlot& s = slot(l, h);
                size_t n = (size_t)n_tokens_ * hd;
                if (s.k) {
                    buf.insert(buf.end(),(uint8_t*)s.k,(uint8_t*)(s.k+n));
                } else {
                    buf.resize(buf.size() + n*sizeof(uint16_t), 0);
                }
                if (s.v) {
                    buf.insert(buf.end(),(uint8_t*)s.v,(uint8_t*)(s.v+n));
                } else {
                    buf.resize(buf.size() + n*sizeof(uint16_t), 0);
                }
            }
        }
        return buf;
    }

    // Total serialized size for n_tok tokens. legacy=true means the file was
    // written when every layer had the file-header geometry (fh1 heads x fh2 dim),
    // including the all-zero blocks of layers that never hold KV (DeltaNet).
    size_t expected_bytes(int n_tok, bool legacy, int fh1, int fh2) const {
        size_t total = 8 + 4 * sizeof(int32_t);
        for (int l = 0; l < n_layers_; ++l) {
            const size_t hd = legacy ? (size_t)fh2 : (size_t)head_dim_for(l);
            const int    nh = legacy ? fh1 : kv_heads_for(l);
            total += (size_t)nh * (size_t)n_tok * hd * sizeof(uint16_t) * 2;
        }
        return total;
    }
    bool is_legacy_layout(int n_tok, int fh1, int fh2, size_t actual) const {
        if (!per_layer_) return false;
        const size_t cur = expected_bytes(n_tok, false, fh1, fh2);
        const size_t old = expected_bytes(n_tok, true,  fh1, fh2);
        return old != cur && actual == old;
    }

public:
    bool load_from_buffer(const uint8_t* data, size_t data_size) {
        if (data_size < 24) return false;
        if (memcmp(data, "NXKVCACH", 8) != 0) return false;
        const int32_t* hdr = (const int32_t*)(data + 8);
        if (hdr[0] != n_layers_ || hdr[1] != n_kv_heads_ || hdr[2] != head_dim_)
            return false;
        int n_tok = hdr[3];
        const bool legacy = is_legacy_layout(n_tok, hdr[1], hdr[2], data_size);
        if (!legacy && data_size < expected_bytes(n_tok, false, hdr[1], hdr[2])) return false;
        clear();
        n_tokens_ = n_tok;
        const uint8_t* p = data + 8 + 4*sizeof(int32_t);
        for (int l = 0; l < n_layers_; ++l) {
            const size_t hd = (size_t)head_dim_for(l);
            const int    nh = kv_heads_for(l);
            if (legacy && nh == 0) {
                p += (size_t)hdr[1] * n_tok * (size_t)hdr[2] * sizeof(uint16_t) * 2;
                continue;
            }
            const size_t slot_bytes = (size_t)n_tok * hd * sizeof(uint16_t);
            for (int h = 0; h < nh; ++h) {
                KVSlot& s = slot(l, h);
                ensure_capacity_layer(s, n_tok, (int)hd);
                if ((int)s.capacity < n_tok) return false;
                if (p + slot_bytes > data + data_size) return false;
                memcpy(s.k, p, slot_bytes); p += slot_bytes;
                if (p + slot_bytes > data + data_size) return false;
                memcpy(s.v, p, slot_bytes); p += slot_bytes;
                s.gpu_dirty = true; s.mirror_valid = 0;
            }
        }
        return true;
    }

    bool load(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;
        char magic[8];
        if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "NXKVCACH", 8) != 0) { fclose(f); return false; }
        int32_t hdr[4];
        if (fread(hdr, sizeof(int32_t), 4, f) != 4) { fclose(f); return false; }
        if (hdr[0] != n_layers_ || hdr[1] != n_kv_heads_ || hdr[2] != head_dim_) {
            fclose(f); return false;  // model mismatch
        }
        int n_tok = hdr[3];
        size_t fsz = 0;
        { long cur = ftell(f); fseek(f, 0, SEEK_END); fsz = (size_t)ftell(f); fseek(f, cur, SEEK_SET); }
        const bool legacy = is_legacy_layout(n_tok, hdr[1], hdr[2], fsz);
        if (!legacy && fsz < expected_bytes(n_tok, false, hdr[1], hdr[2])) { fclose(f); return false; }
        clear();
        n_tokens_ = n_tok;
        for (int l = 0; l < n_layers_; ++l) {
            const size_t hd = (size_t)head_dim_for(l);
            const int    nh = kv_heads_for(l);
            if (legacy && nh == 0) {
                fseek(f, (long)((size_t)hdr[1] * n_tok * (size_t)hdr[2] * sizeof(uint16_t) * 2), SEEK_CUR);
                continue;
            }
            for (int h = 0; h < nh; ++h) {
                KVSlot& s = slot(l, h);
                ensure_capacity_layer(s, n_tok, (int)hd);
                if ((int)s.capacity < n_tok) { fclose(f); return false; }
                size_t n = (size_t)n_tok * hd;
                if (fread(s.k, sizeof(uint16_t), n, f) != n) { fclose(f); return false; }
                if (fread(s.v, sizeof(uint16_t), n, f) != n) { fclose(f); return false; }
                s.gpu_dirty = true; s.mirror_valid = 0;
            }
        }
        fclose(f);
        return true;
    }

    // Evict a specific range of tokens [start, end) from the KV cache.
    // Everything before start and everything from end onward is preserved.
    // Used for smart context trimming — protects the system prompt (pinned
    // prefix) and recent context while discarding stale middle content.
    //
    // RoPE note: K vectors have position-dependent rotation baked in at store
    // time. After eviction the absolute positions of kept tokens are wrong —
    // but attention still works because the RELATIVE positions between kept
    // tokens are roughly preserved. In practice this produces coherent output
    // far better than stopping cold at the context limit.
    //
    // Returns bytes freed.
    size_t evict_range(int start, int end) {
        if (start < 0) start = 0;
        if (end > n_tokens_) end = n_tokens_;
        if (start >= end) return 0;

        const int evict_count = end - start;
        const int keep_tail   = n_tokens_ - end;

        size_t freed = 0;
        for (int L = 0; L < n_layers_; ++L) {
            const size_t dim = (size_t)head_dim_for(L);
            const int    nh  = kv_heads_for(L);
            for (int h = 0; h < nh; ++h) {
                KVSlot& s = slot(L, h);
                if (!s.k) continue;
                if (keep_tail > 0) {
                    memmove(s.k + (size_t)start * dim,
                            s.k + (size_t)end   * dim,
                            (size_t)keep_tail * dim * sizeof(uint16_t));
                    memmove(s.v + (size_t)start * dim,
                            s.v + (size_t)end   * dim,
                            (size_t)keep_tail * dim * sizeof(uint16_t));
                }
                s.gpu_dirty = true; s.mirror_valid = 0;
                freed += (size_t)evict_count * dim * 2 * sizeof(uint16_t);
            }
        }
        n_tokens_ -= evict_count;
        release_unused_tail();
        return freed;
    }

    // After tokens are removed, hand the memory past the last token back to the system (RAM drops right away).
    // The buffers keep their size and address, so nothing else has to change; those pages come back as zeros
    // if the conversation grows into them again.
    void release_unused_tail() {
#if defined(__linux__)
        const size_t ps = page_size();
        for (int L = 0; L < n_layers_; ++L) {
            const size_t dim = (size_t)head_dim_for(L);
            const int    nh  = kv_heads_for(L);
            for (int h = 0; h < nh; ++h) {
                KVSlot& s = slot(L, h);
                if (!s.k || !s.v || s.bytes == 0) continue;
                const size_t used = ((size_t)n_tokens_ * dim * sizeof(uint16_t) + ps - 1) / ps * ps;
                if (used >= s.bytes) continue;
                madvise((uint8_t*)s.k + used, s.bytes - used, MADV_DONTNEED);
                madvise((uint8_t*)s.v + used, s.bytes - used, MADV_DONTNEED);
            }
        }
#endif
    }
    // This is O(remaining_tokens) but only called on overflow — rare.
    size_t evict_oldest(int n_tokens_to_evict) {
        if (n_tokens_to_evict <= 0 || n_tokens_ == 0) return 0;
        int evict = std::min(n_tokens_to_evict, n_tokens_);
        size_t freed = 0;
        for (int L = 0; L < n_layers_; ++L) {
            const size_t dim = (size_t)head_dim_for(L);
            const int    nh  = kv_heads_for(L);
            for (int h = 0; h < nh; ++h) {
                KVSlot& s = slot(L, h);
                if (!s.k) continue;
                size_t keep = (size_t)(n_tokens_ - evict);
                memmove(s.k, s.k + (size_t)evict * dim, keep * dim * sizeof(uint16_t));
                memmove(s.v, s.v + (size_t)evict * dim, keep * dim * sizeof(uint16_t));
                s.gpu_dirty = true; s.mirror_valid = 0;
                freed += (size_t)evict * dim * 2 * sizeof(uint16_t);
            }
        }
        n_tokens_ -= evict;
        release_unused_tail();
        return freed;
    }

    // Compatibility: get single K/V vector as f32
    void get_k_f32(int layer, int pos, int kv_head, float* out) const {
        const int hd = head_dim_for(layer);
        const uint16_t* p = slot(layer, kv_head).k + (size_t)pos * hd;
        for (int d = 0; d < hd; ++d) out[d] = quant::f16_to_f32(p[d]);
    }
    void get_v_f32(int layer, int pos, int kv_head, float* out) const {
        const int hd = head_dim_for(layer);
        const uint16_t* p = slot(layer, kv_head).v + (size_t)pos * hd;
        for (int d = 0; d < hd; ++d) out[d] = quant::f16_to_f32(p[d]);
    }

    // pack_k / pack_v: legacy compat for code that still uses the flat staging approach.
    // On the new contiguous layout this is a no-op redirect — just return the pointer.
    // (The GPU attention path no longer calls these — it uses k_ptr/v_ptr directly.)
    void pack_k(int layer, int n_ctx, uint16_t* dst) const {
        const int hd = head_dim_for(layer);
        const int nh = kv_heads_for(layer);
        for (int h = 0; h < nh; ++h) {
            const uint16_t* src = slot(layer, h).k;
            memcpy(dst + (size_t)h * n_ctx * hd, src,
                   (size_t)n_ctx * hd * sizeof(uint16_t));
        }
    }
    void pack_v(int layer, int n_ctx, uint16_t* dst) const {
        const int hd = head_dim_for(layer);
        const int nh = kv_heads_for(layer);
        for (int h = 0; h < nh; ++h) {
            const uint16_t* src = slot(layer, h).v;
            memcpy(dst + (size_t)h * n_ctx * hd, src,
                   (size_t)n_ctx * hd * sizeof(uint16_t));
        }
    }

private:
    int n_layers_   = 0;
    int n_kv_heads_ = 0;
    int head_dim_   = 0;
    int n_tokens_   = 0;

    // ─── Per-layer geometry (Gemma 4 etc.) ──────────────────────────────────
    //
    // Most architectures have uniform (n_kv_heads, head_dim) for every layer.
    // For those we leave per-layer vectors empty and use the scalar fields
    // above — zero behavior change vs the old code.
    //
    // Gemma 4 alternates local layers (n_kv_heads=16, head_dim=256) with
    // global layers (n_kv_heads=4, head_dim=512). Different KV byte width per
    // layer means a single global stride can't address them all, so we keep
    // per-layer arrays and a precomputed slot offset table.
    //
    // Indexing: slot(layer, h) = slots_[layer_slot_offset_[layer] + h]
    //
    // When per_layer_ is false:
    //   - kv_heads_for(L)  ≡ n_kv_heads_
    //   - head_dim_for(L)  ≡ head_dim_
    //   - layer_slot_offset_[L] = L * n_kv_heads_  (lazily populated)
    bool                  per_layer_ = false;
    std::vector<int>      n_kv_heads_layer_;     // size = n_layers_ when per_layer_
    std::vector<int>      head_dim_layer_;       // size = n_layers_ when per_layer_
    std::vector<size_t>   layer_slot_offset_;    // prefix sum into slots_
    std::vector<bool>     layer_geom_known_;     // v26: which layers add_layer_geometry() has already recorded (streaming mode only)

    std::vector<KVSlot> slots_;  // see indexing comment above

    int kv_heads_for(int layer) const {
        return per_layer_ ? n_kv_heads_layer_[layer] : n_kv_heads_;
    }
    int head_dim_for(int layer) const {
        return per_layer_ ? head_dim_layer_[layer] : head_dim_;
    }
    size_t slot_offset_for(int layer) const {
        return per_layer_ ? layer_slot_offset_[layer]
                          : (size_t)layer * (size_t)n_kv_heads_;
    }

    KVSlot& slot(int layer, int kv_head) {
        return slots_[slot_offset_for(layer) + (size_t)kv_head];
    }
    const KVSlot& slot(int layer, int kv_head) const {
        return slots_[slot_offset_for(layer) + (size_t)kv_head];
    }

    static constexpr size_t INITIAL_CAP = 512;   // tokens
    static constexpr size_t ALIGN       = 64;    // bytes — satisfies Vulkan import requirement

    // ── Buffer allocation ───────────────────────────────────────────────────
    // Linux: anonymous mmap + mremap. KV buffers are large (tens of MB each,
    // 128 of them growing in lockstep). With malloc, every growth step
    // allocated a new buffer, copied, and freed the old one; glibc cannot give
    // those interior holes back and the next, larger buffer never fits in
    // them, so process memory ended up several times the live KV size.
    // mmap/mremap has no copy, no transient 2x spike, no fragmentation, hands
    // pages back to the OS on free, and the kernel supplies zeroed pages
    // (so stale data can never be read) without touching them up front.
    // Other platforms keep the aligned-malloc + copy path.

    static size_t page_size() {
#if defined(__linux__)
        static const size_t ps = (size_t)sysconf(_SC_PAGESIZE) > 0 ? (size_t)sysconf(_SC_PAGESIZE) : 4096;
        return ps;
#else
        return ALIGN;
#endif
    }
    static size_t round_bytes(size_t n_elems) {
        const size_t ps = page_size();
        const size_t b  = n_elems * sizeof(uint16_t);
        return (b + ps - 1) / ps * ps;
    }

    static uint16_t* alloc_buf(size_t bytes) {
        void* p = nullptr;
#if defined(__linux__)
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) p = nullptr;
#elif defined(_WIN32)
        p = _aligned_malloc(bytes, ALIGN);
        if (p) std::memset(p, 0, bytes);
#else
        if (posix_memalign(&p, ALIGN, bytes) != 0) p = nullptr;
        if (p) std::memset(p, 0, bytes);
#endif
        return static_cast<uint16_t*>(p);
    }

    static void free_buf(uint16_t* p, size_t bytes) {
        if (!p) return;
#if defined(__linux__)
        munmap(p, bytes);
#elif defined(_WIN32)
        _aligned_free(p);
#else
        free(p);
#endif
    }

    // Grow a buffer from old_bytes to new_bytes, preserving contents; the
    // new tail is zero. Returns nullptr (old buffer untouched) on failure.
    static uint16_t* grow_buf(uint16_t* old, size_t old_bytes, size_t new_bytes) {
        if (!old) return alloc_buf(new_bytes);
#if defined(__linux__)
        void* p = mremap(old, old_bytes, new_bytes, MREMAP_MAYMOVE);
        return p == MAP_FAILED ? nullptr : static_cast<uint16_t*>(p);
#else
        uint16_t* n = alloc_buf(new_bytes);
        if (!n) return nullptr;
        std::memcpy(n, old, old_bytes);
        free_buf(old, old_bytes);
        return n;
#endif
    }

    void free_slot(KVSlot& s) {
        free_buf(s.k, s.bytes); s.k = nullptr;
        free_buf(s.v, s.bytes); s.v = nullptr;
        s.capacity = 0;
        s.bytes = 0;
        s.gpu_dirty = true; s.mirror_valid = 0;
    }

    // Resize one slot to exactly new_cap tokens (never shrinks).
    // On failure the slot is left exactly as it was.
    bool resize_slot(KVSlot& s, size_t new_cap, int hd) {
        if (new_cap <= s.capacity) return true;
        const size_t nb = round_bytes(new_cap * (size_t)hd);
        // Not enough free memory for the buffer: stop the whole step with an exception (caught where a reply/prompt is processed).
        // Returning false here is not enough - the attention code would then read past the end of a cache that never grew.
        nexus::memguard::require(2 * (nb > s.bytes ? nb - s.bytes : 0), "the conversation memory (KV cache)");
        uint16_t* nk = grow_buf(s.k, s.bytes, nb);
        if (!nk) return false;
        // k may have moved; record it before touching v so a v failure
        // cannot leave a dangling pointer.
        s.k = nk;
        uint16_t* nv = grow_buf(s.v, s.bytes, nb);
        if (!nv) {
            // k is now larger than v's bookkeeping says. Shrink k back so the
            // slot stays consistent (mremap shrink cannot fail meaningfully).
            if (s.bytes == 0) { free_buf(s.k, nb); s.k = nullptr; }
            else { uint16_t* back = grow_buf_shrink(s.k, nb, s.bytes); if (back) s.k = back; }
            return false;
        }
        s.v = nv;
        s.bytes    = nb;
        s.capacity = nb / ((size_t)hd * sizeof(uint16_t));
        if (s.capacity < new_cap) s.capacity = new_cap;  // rounding never reduces
        s.gpu_dirty = true;
        return true;
    }

    static uint16_t* grow_buf_shrink(uint16_t* p, size_t from_bytes, size_t to_bytes) {
        if (!p || to_bytes == 0) return p;
#if defined(__linux__)
        void* q = mremap(p, from_bytes, to_bytes, MREMAP_MAYMOVE);
        return q == MAP_FAILED ? nullptr : static_cast<uint16_t*>(q);
#else
        (void)from_bytes; (void)to_bytes;
        return p;
#endif
    }

    void ensure_capacity(KVSlot& s, int needed_tokens) {
        // Legacy path — uses global head_dim_. Still correct when per_layer_
        // is false. Per-layer callers use ensure_capacity_layer().
        ensure_capacity_layer(s, needed_tokens, head_dim_);
    }

    // Growth strategy: double while small (cheap, infrequent reallocs for short
    // sessions), then switch to fixed-size increments once buffers get large.
    // Doubling a multi-GB buffer forces a transient spike where BOTH the old
    // and new buffers are resident at once (see ensure_capacity_layer) --
    // fine at 512->1024 tokens, dangerous at 32768->65536. GROWTH_DOUBLE_LIMIT
    // is the token capacity past which we stop doubling and grow by
    // GROWTH_STEP_TOKENS instead, bounding the transient spike to one step's
    // worth of memory no matter how long the session runs.
    static constexpr size_t GROWTH_DOUBLE_LIMIT = 8192;   // tokens
    static constexpr size_t GROWTH_STEP_TOKENS  = 4096;   // tokens, once past the limit

    void ensure_capacity_layer(KVSlot& s, int needed_tokens, int hd) {
        if ((size_t)needed_tokens <= s.capacity) return;
        size_t new_cap;
        if (s.capacity == 0) {
            new_cap = INITIAL_CAP;
        } else if (s.capacity < GROWTH_DOUBLE_LIMIT) {
            new_cap = s.capacity * 2;
        } else {
            new_cap = s.capacity + GROWTH_STEP_TOKENS;
        }
        while (new_cap < (size_t)needed_tokens) {
            new_cap += (new_cap < GROWTH_DOUBLE_LIMIT) ? new_cap : GROWTH_STEP_TOKENS;
        }
        resize_slot(s, new_cap, hd);   // on failure capacity is unchanged; store() checks
    }

    // Pre-size every slot for an expected session length, so no growth event
    // (and thus no transient spike) happens during generation at all. Safe to
    // call after init()/init_per_layer(); a no-op for slots already >= tokens.
    // Returns false if any allocation failed (existing slots are left as-is).
    bool reserve_tokens(int tokens) {
        if (tokens <= 0) return true;
        for (int L = 0; L < n_layers_; ++L) {
            const int hd = head_dim_for(L);
            const int nh = kv_heads_for(L);
            for (int h = 0; h < nh; ++h) {
                if (!resize_slot(slot(L, h), (size_t)tokens, hd)) return false;
            }
        }
        return true;
    }

    const uint16_t* get_page(bool is_k, int layer, int kv_head,
                              int page_idx, int* n_out) const {
        const KVSlot& s = slot(layer, kv_head);
        int start = page_idx * KV_PAGE_SIZE;
        if (start >= n_tokens_ || !s.k) { *n_out = 0; return nullptr; }
        int end = std::min(start + KV_PAGE_SIZE, n_tokens_);
        *n_out = end - start;
        const uint16_t* base = is_k ? s.k : s.v;
        const int hd = head_dim_for(layer);
        return base + (size_t)start * hd;
    }

    static uint16_t f32_to_f16(float f) {
        uint32_t x; memcpy(&x, &f, 4);
        uint32_t sign = (x >> 31) & 1;
        int32_t  expo = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (x >> 13) & 0x3FF;
        if (expo <= 0)  return (uint16_t)(sign << 15);
        if (expo >= 31) return (uint16_t)((sign << 15) | 0x7C00);
        return (uint16_t)((sign << 15) | ((uint32_t)expo << 10) | mant);
    }
};

} // namespace model
} // namespace nexus
