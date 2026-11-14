/*
 * nexus/src/gguf/parser.h + parser.cpp
 *
 * Reads GGUF files directly. No llama.cpp. No dependencies.
 * Loads metadata and maps tensor data into our memory manager.
 *
 * GGUF Format (v3):
 *   [magic u32][version u32][n_tensors u64][n_kv u64]
 *   [kv pairs...][tensor infos...][padding][tensor data...]
 */

#pragma once

#include "../core/memory.h"

#include <cstdint>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cctype>
#include <cstdlib>
#include <regex>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <sstream>

// --- Platform file/mmap abstraction -----------------------------------------
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
   // Windows file mapping - replaces mmap/munmap/open/close/fstat
   struct NxMmapFile {
       HANDLE   hFile    = INVALID_HANDLE_VALUE;
       HANDLE   hMap     = nullptr;
       void*    ptr      = nullptr;
       size_t   size     = 0;
       int      fd       = -1;   // not used on Windows, kept for struct compat
       bool     borrowed = false; // true = ptr is not owned, skip cleanup
       bool     writable = false; // Windows: weights are always copied (direct-to-GPU loading is Linux only)
       bool open(const char* path) {
           hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
           if (hFile == INVALID_HANDLE_VALUE) return false;
           LARGE_INTEGER sz; GetFileSizeEx(hFile, &sz); size = (size_t)sz.QuadPart;
           hMap = CreateFileMapping(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
           if (!hMap) { CloseHandle(hFile); hFile=INVALID_HANDLE_VALUE; return false; }
           ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
           if (!ptr) { CloseHandle(hMap); CloseHandle(hFile); hMap=nullptr; hFile=INVALID_HANDLE_VALUE; return false; }
           return true;
       }
       // Borrow an existing memory buffer -- no file, no cleanup, zero disk I/O.
       void from_memory(const void* p, size_t n) {
           ptr = const_cast<void*>(p); size = n;
           hFile = INVALID_HANDLE_VALUE; hMap = nullptr; borrowed = true;
       }
       void close() {
           if (!borrowed) {
               if (ptr)   { UnmapViewOfFile(ptr);  ptr  = nullptr; }
               if (hMap)  { CloseHandle(hMap);     hMap = nullptr; }
               if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
           }
           ptr = nullptr; hMap = nullptr; hFile = INVALID_HANDLE_VALUE;
           size = 0; borrowed = false;
       }
   };
#  define NX_MMAP_FAILED nullptr
   inline void nx_madvise(void*, size_t, int) {}  // no-op on Windows
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
   struct NxMmapFile {
       void*  ptr      = (void*)-1; // MAP_FAILED
       size_t size     = 0;
       int    fd       = -1;
       bool   borrowed = false; // true = ptr is not owned, skip cleanup
       bool   writable = false; // true = private copy-on-write mapping (the file itself is never modified)
       bool open(const char* path) {
           fd = ::open(path, O_RDONLY);
           if (fd < 0) return false;
           struct stat st;
           if (fstat(fd, &st) != 0) { ::close(fd); fd=-1; return false; }
           size = (size_t)st.st_size;
           // MAP_PRIVATE + PROT_WRITE: reads come straight from the file; a write only changes this process's own
           // copy of a page.  Lets weights that are read in place be pinned into RAM later (nx_pin_direct_layers_from).
           ptr  = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
           writable = (ptr != MAP_FAILED);
           if (ptr == MAP_FAILED) ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
           if (ptr == MAP_FAILED) { ::close(fd); fd=-1; ptr=(void*)-1; return false; }
           madvise(ptr, size, MADV_SEQUENTIAL);
           return true;
       }
       // Borrow an existing memory buffer -- no file, no cleanup, zero disk I/O.
       void from_memory(const void* p, size_t n) {
           ptr = const_cast<void*>(p); size = n; fd = -1; borrowed = true;
       }
       void close() {
           if (!borrowed) {
               if (ptr != (void*)-1) { munmap(ptr, size); ptr=(void*)-1; }
               if (fd >= 0)          { ::close(fd);       fd=-1; }
           }
           ptr = (void*)-1; fd = -1; size = 0; borrowed = false; writable = false;
       }
   };
#  define NX_MMAP_FAILED ((void*)-1)
   inline void nx_madvise(void* p, size_t n, int advice) { madvise(p,n,advice); }
#  ifndef MADV_SEQUENTIAL
#    define MADV_SEQUENTIAL 2
#  endif
#  ifndef MADV_RANDOM
#    define MADV_RANDOM 1
#  endif
#endif
// -----------------------------------------------------------------------------

namespace nexus {
namespace gguf {

// --- GGUF value types ------------------------------------------------------

enum class GGUFType : uint32_t {
    UINT8   = 0,  INT8    = 1,
    UINT16  = 2,  INT16   = 3,
    UINT32  = 4,  INT32   = 5,
    FLOAT32 = 6,  BOOL    = 7,
    STRING  = 8,  ARRAY   = 9,
    UINT64  = 10, INT64   = 11,
    FLOAT64 = 12,
};

// Quantization types (ggml_type in ggml.h)
enum class GGMLType : int32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,

    // Nexus-native .nxm row/group quant types.
    NQ4_G64 = 1004,
    NQ5_G64 = 1005,
    NQ6_G64 = 1006,
    NQ8_G64 = 1008,
    COUNT,
};

// Returns bytes per "block" and elements per block for a quant type
struct QuantInfo {
    size_t block_size;   // bytes per block
    int    elements;     // elements per block
    bool   is_quantized;
    const char* name;
};
QuantInfo get_quant_info(GGMLType type);

// --- Value variant ---------------------------------------------------------

struct GGUFArray;  // forward

using GGUFValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    float, bool,
    std::string,
    uint64_t, int64_t,
    double,
    std::shared_ptr<GGUFArray>
>;

struct GGUFArray {
    GGUFType                elem_type;
    std::vector<GGUFValue>  values;
};

// --- Tensor descriptor -----------------------------------------------------

struct TensorDesc {
    std::string    name;
    GGMLType       type;
    std::vector<uint64_t> shape;  // dimensions, n_dims elements
    uint64_t       offset;        // byte offset from THIS FILE's data section start
    uint64_t       n_elements;    // total element count
    uint64_t       data_size;     // bytes on disk
    int            file_idx = 0;  // which part file (0 = file 1, 1 = file 2, etc.)

    // Filled by loader:
    void*          data = nullptr;
    bool           loaded = false;
};

// --- Model metadata extracted from KV pairs --------------------------------

struct ModelMeta {
    std::string arch;          // "llama", "qwen2", "mistral", etc.
    std::string name;
    uint32_t    n_layers       = 0;
    uint32_t    n_heads        = 0;
    uint32_t    n_kv_heads     = 0;
    uint32_t    head_dim       = 0;
    uint32_t    hidden_dim     = 0;     // n_heads - head_dim
    uint32_t    ffn_dim        = 0;     // feed-forward hidden dim
    uint32_t    vocab_size     = 0;
    uint32_t    context_length = 0;
    float       rms_norm_eps   = 1e-5f;
    float       rope_freq_base = 10000.0f;
    float       rope_scale     = 1.0f;
    int32_t     rope_type      = 0;
    uint32_t    n_experts      = 0;     // for MoE models (Mixtral)
    uint32_t    n_experts_used = 0;
    // MoE routing metadata used by newer GGUFs: Qwen/Gemma/Mixtral/Nemotron-style.
    uint32_t    expert_ffn_dim        = 0;
    uint32_t    expert_shared_ffn_dim = 0;
    float       expert_weights_scale  = 1.0f;
    bool        expert_weights_norm   = false;
    std::string expert_gating_func;
};

// --- Parser ----------------------------------------------------------------

struct LoadProgress {
    size_t tensors_loaded;
    size_t tensors_total;
    size_t bytes_loaded;
    size_t bytes_total;
    std::string current_tensor;
};

class Parser {
public:
    using ProgressCB = std::function<void(const LoadProgress&)>;

    Parser() = default;
    ~Parser() { close_files(); }

    // Open a single GGUF file (or the first part of a split set).
    // Auto-detects split naming and loads all parts transparently.
    bool open(const std::string& path);
    // Load GGUF from an in-memory buffer -- zero disk I/O on all platforms.
    // The caller owns the buffer and must keep it alive while the parser is open.
    bool open_from_memory(const void* data, size_t size);

    bool load_weights(MemoryManager& mm, ProgressCB cb = nullptr);
    bool load_layer(int layer_id, MemoryManager& mm);
    // Frees one layer's tensor memory (the opposite of load_layer()) and resets
    // those tensors' `loaded`/`data` fields so a later load_layer() for the same
    // layer_id actually reloads them instead of silently skipping already-loaded
    // tensors. Does not touch global tensors (embeddings, norms, lm_head) or any
    // other layer's tensors.
    void free_layer(int layer_id, MemoryManager& mm);

    const ModelMeta&                          meta()     const { return meta_; }
    const std::unordered_map<std::string, TensorDesc>& tensors() const { return tensors_; }
    const std::unordered_map<std::string, GGUFValue>&  kv()      const { return kv_pairs_; }

    template<typename T>
    T get(const std::string& key, T default_val = T{}) const;

    std::string get_str(const std::string& key, const std::string& def = "") const;
    uint64_t    get_u64(const std::string& key, uint64_t def = 0)            const;
    uint32_t    get_u32(const std::string& key, uint32_t def = 0)            const;
    float       get_f32(const std::string& key, float def = 0.0f)            const;

    TensorDesc* find_tensor(const std::string& name_contains);

    void print_meta() const;
    void print_tensors() const;

    bool is_split() const { return part_files_.size() > 1; }
    int  n_parts()  const { return (int)part_files_.size(); }

    // Direct access to tensor data from the mmap -- no MemoryManager needed.
    // Returns pointer to start of tensor data section.
    // Use as: data_ptr() + td.offset to get a specific tensor's bytes.
    // Valid as long as the parser is open. nullptr if not open.
    const uint8_t* data_ptr() const {
        if (!mmap_ptr_) return nullptr;
        return (const uint8_t*)mmap_ptr_ + data_offset_;
    }

private:
    // Per-file state (one entry per part file)
    struct PartFile {
        std::string path;
        int         fd          = -1;
        void*       mmap_ptr    = nullptr;  // points into NxMmapFile::ptr
        NxMmapFile  nx_file;               // owns the OS file mapping
        size_t      file_size   = 0;
        uint64_t    data_offset = 0;   // where tensor data starts in this file
    };

    std::string  path_;                 // path of file passed to open()
    std::vector<PartFile> part_files_;  // [0] = file 1, [1] = file 2, etc.

    // Legacy single-file accessors (file 0)
    void*    mmap_ptr_    = nullptr;
    size_t   file_size_   = 0;
    uint64_t data_offset_ = 0;
    int      fd_          = -1;

    ModelMeta    meta_;
    std::unordered_map<std::string, TensorDesc>   tensors_;
    std::unordered_map<std::string, GGUFValue>    kv_pairs_;
    bool native_nxm_ = false;

    bool open_nxm(const std::string& path);

    bool read_header(const uint8_t* buf, size_t& pos);
    bool read_kv_pairs(const uint8_t* buf, size_t& pos, uint64_t count);
    bool read_tensor_infos(const uint8_t* buf, size_t& pos, uint64_t count);
    void extract_meta();

    GGUFValue read_value(const uint8_t* buf, size_t& pos, GGUFType type);
    std::string read_str(const uint8_t* buf, size_t& pos);

    uint32_t find_u32(std::initializer_list<std::string> keys, uint32_t def = 0) const;
    float    find_f32(std::initializer_list<std::string> keys, float def = 0) const;

    // Split-GGUF helpers
    bool open_part(const std::string& path, int idx);
    void close_files();
    static std::vector<std::string> find_split_parts(const std::string& first_path);
};

} // namespace gguf
} // namespace nexus


// -----------------------------------------------------------------------------
//  Implementation
// -----------------------------------------------------------------------------

#include <cstring>
#include <cstdio>
#include <cassert>
#include <sys/stat.h>

namespace nexus {
namespace gguf {

// ---- Direct-to-GPU weights (Linux) -------------------------------------------------------------------------------
// NEXUS_DIRECT_GPU_LAYERS=N (set by the launcher when the model goes onto a discrete card and the prompt speed-up is
// off): the big weight matrices of layers 0..N-1 are NOT copied into RAM.  They are read straight from the model file
// when they are uploaded to the card, so the model lives in VRAM only.  Whatever has to stay on the CPU (layers that
// did not fit, Qwen's linear-attention layers, embeddings, norms) is still copied exactly as before.
struct NxDirectRange { const uint8_t* p; size_t n; int layer; };
inline std::vector<NxDirectRange>& nx_direct_ranges() { static std::vector<NxDirectRange> v; return v; }
inline int nx_direct_gpu_layers() {
#ifdef _WIN32
    return 0;
#else
    static const int n = [] { const char* e = std::getenv("NEXUS_DIRECT_GPU_LAYERS"); return e ? std::atoi(e) : 0; }();
    return n;
#endif
}
// layer index from a tensor name ("blk.12.ffn_up.weight" / "model.language_model.layers.12....weight"), -1 if none
inline int nx_layer_of_tensor(const std::string& nm) {
    static const char* pre[] = {"blk.", "model.language_model.layers.", "model.layers.", "language_model.model.layers."};
    for (const char* pr : pre) {
        size_t L = std::strlen(pr);
        if (nm.compare(0, L, pr) == 0 && L < nm.size() && std::isdigit((unsigned char)nm[L])) return std::atoi(nm.c_str() + L);
    }
    return -1;
}
// weights the CPU also computes with every token -> always keep in RAM
inline bool nx_keep_in_ram_name(const std::string& nm) {
    static const char* bad[] = {"linear_attn", "ssm", "conv1d", "mamba", "exps", "expert", "shexp", "router", "gate_inp"};
    for (const char* b : bad) if (nm.find(b) != std::string::npos) return true;
    return false;
}
// How many leading layers fit in the card's memory?  NEXUS_DIRECT_VRAM_MB (set by the launcher from the hardware scan) is the card's
// size.  Only those layers are read in place from the file (they get copied into VRAM and need no RAM copy).  Layers beyond that are
// loaded into our own page-aligned memory, which the GPU can map directly -- so layers that do not fit in VRAM can still be computed on
// the GPU from system RAM with no second copy.  Without the variable every requested layer is read in place (older behaviour).
template <class TensorMap>
inline int nx_direct_vram_layer_limit(const TensorMap& tensors) {
    const char* e = std::getenv("NEXUS_DIRECT_VRAM_MB");
    const long long mb = e ? atoll(e) : 0;
    if (mb <= 0) return 1 << 30;
    const long long keep = 640;                                   // room for KV cache and activation buffers
    if (mb <= keep) return 0;
    const unsigned long long budget = (unsigned long long)(mb - keep) << 20;
    std::vector<unsigned long long> per_layer(1024, 0);
    int top = -1;
    for (auto& kv : tensors) {
        const int l = nx_layer_of_tensor(kv.first);
        if (l < 0 || l >= 1024) continue;
        per_layer[(size_t)l] += (unsigned long long)kv.second.data_size;
        top = std::max(top, l);
    }
    unsigned long long used = 0; int fit = 0;
    for (int l = 0; l <= top; ++l) { if (used + per_layer[(size_t)l] > budget) break; used += per_layer[(size_t)l]; ++fit; }
    fprintf(stderr, "[load] card memory %lld MB: the first %d layers are read in place; later layers are loaded into page-aligned RAM\n", mb, fit);
    return fit;
}
// Turn the in-place (file-backed) weights of layers >= first_layer into ordinary RAM copies.  Used for layers that did
// not fit on the card and so run on the CPU: each page is touched once (copy-on-write); pointers stay valid.
inline void nx_pin_direct_layers_from(int first_layer) {
#ifndef _WIN32
    size_t bytes = 0; int tensors = 0;
    for (auto& r : nx_direct_ranges()) {
        if (r.layer < 0 || r.layer < first_layer) continue;
        const uintptr_t b = (uintptr_t)r.p, e = b + r.n;
        for (uintptr_t a = b & ~(uintptr_t)4095; a < e; a += 4096) {
            volatile uint8_t* q = (volatile uint8_t*)(a < b ? b : a);
            *q = *q;
        }
        bytes += r.n; ++tensors; r.layer = -2;
    }
    if (tensors) { fprintf(stderr, "[load] %d weight tensors (%zu MB) belong to CPU layers -> copied into RAM\n", tensors, bytes >> 20); fflush(stderr); }
#else
    (void)first_layer;
#endif
}


static constexpr uint32_t GGUF_MAGIC   = 0x46554747u; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

// Human-readable name of any ggml tensor type id (used in messages only)
inline const char* ggml_type_name(int32_t t) {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1"; case 6: return "Q5_0"; case 7: return "Q5_1";
        case 8: return "Q8_0"; case 9: return "Q8_1"; case 10: return "Q2_K"; case 11: return "Q3_K"; case 12: return "Q4_K"; case 13: return "Q5_K";
        case 14: return "Q6_K"; case 15: return "Q8_K"; case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 19: return "IQ1_S";
        case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S"; case 23: return "IQ4_XS"; case 24: return "I8"; case 25: return "I16";
        case 26: return "I32"; case 27: return "I64"; case 28: return "F64"; case 29: return "IQ1_M"; case 30: return "BF16";
        case 34: return "TQ1_0"; case 35: return "TQ2_0"; case 39: return "MXFP4";
        case 1004: return "NQ4_G64"; case 1005: return "NQ5_G64"; case 1006: return "NQ6_G64"; case 1008: return "NQ8_G64";
        default: return "unknown";
    }
}

// --- Quant info table ---------------------------------------------------------

inline QuantInfo get_quant_info(GGMLType type) {
    // block_size = bytes per block, elements = scalars per block
    switch (type) {
        case GGMLType::F32:   return {4,   1,  false, "F32"};
        case GGMLType::F16:   return {2,   1,  false, "F16"};
        case GGMLType::BF16:  return {2,   1,  false, "BF16"};
        case GGMLType::NQ4_G64: return {34,  64, true,  "NQ4_G64"};
        case GGMLType::NQ5_G64: return {42,  64, true,  "NQ5_G64"};
        case GGMLType::NQ6_G64: return {50,  64, true,  "NQ6_G64"};
        case GGMLType::NQ8_G64: return {66,  64, true,  "NQ8_G64"};
        case GGMLType::Q4_0:  return {18,  32, true,  "Q4_0"};
        case GGMLType::Q4_1:  return {20,  32, true,  "Q4_1"};
        case GGMLType::Q5_0:  return {22,  32, true,  "Q5_0"};
        case GGMLType::Q5_1:  return {24,  32, true,  "Q5_1"};
        case GGMLType::Q8_0:  return {34,  32, true,  "Q8_0"};
        case GGMLType::Q8_1:  return {36,  32, true,  "Q8_1"};
        // K-quants (superblock structure, 256 elements per superblock)
        case GGMLType::Q2_K:  return {84,  256, true, "Q2_K"};
        case GGMLType::Q3_K:  return {110, 256, true, "Q3_K"};
        case GGMLType::Q4_K:  return {144, 256, true, "Q4_K"};
        case GGMLType::Q5_K:  return {176, 256, true, "Q5_K"};
        case GGMLType::Q6_K:  return {210, 256, true, "Q6_K"};
        case GGMLType::Q8_K:  return {292, 256, true, "Q8_K"};
        default:              return {0,   0,   true,  "UNSUPPORTED"};
    }
}

// --- Binary reader helpers ----------------------------------------------------

template<typename T>
static T read_le(const uint8_t* buf, size_t& pos) {
    T val;
    memcpy(&val, buf + pos, sizeof(T));
    pos += sizeof(T);
    return val;
}

// --- Parser::open -------------------------------------------------------------

inline void Parser::close_files() {
    for (auto& pf : part_files_) {
        pf.nx_file.close();
        pf.mmap_ptr = nullptr;
    }
    part_files_.clear();
    mmap_ptr_ = nullptr;
    fd_ = -1;
}

// Detect split naming. Handles any digit width:
//   model-00001-of-00003.gguf
//   model-1-of-3.gguf
// Returns all part paths, or just {first_path} if not a split file.
inline std::vector<std::string> Parser::find_split_parts(const std::string& first_path) {
    // Scan backwards for the pattern -DIGITS-of-DIGITS.gguf
    // Must end with .gguf
    if (first_path.size() < 5 ||
        first_path.substr(first_path.size()-5) != ".gguf")
        return {first_path};

    const std::string s = first_path;
    const std::string ext = ".gguf";

    // Find the -of- separator working backwards from the extension
    size_t dot = s.size() - ext.size();  // position of '.'

    // Find "-of-" somewhere before the extension
    size_t of_pos = s.rfind("-of-");
    if (of_pos == std::string::npos || of_pos >= dot) return {first_path};

    // After "-of-" should be all digits up to "."
    std::string total_str = s.substr(of_pos + 4, dot - (of_pos + 4));
    for (char c : total_str) if (!isdigit(c)) return {first_path};

    // Before "-of-" should be all digits (the part number), preceded by "-"
    size_t num_end = of_pos;          // exclusive
    size_t num_start = num_end;
    while (num_start > 0 && isdigit(s[num_start - 1])) --num_start;
    if (num_start == num_end) return {first_path};  // no digits
    if (num_start == 0 || s[num_start - 1] != '-') return {first_path};

    std::string part_num_str = s.substr(num_start, num_end - num_start);
    int width = (int)part_num_str.size();   // preserve zero-padding width
    int total_width = (int)total_str.size();

    int total = 0;
    try { total = std::stoi(total_str); } catch(...) { return {first_path}; }
    if (total <= 1) return {first_path};

    // prefix = everything up to and including the "-" before the part number
    std::string prefix = s.substr(0, num_start);    // e.g. "model-"
    std::string suffix = ext;

    std::vector<std::string> parts;
    for (int p = 1; p <= total; ++p) {
        char num[64], tot[64];
        snprintf(num, sizeof(num), "%0*d", width, p);
        snprintf(tot, sizeof(tot), "%0*d", total_width, total);
        parts.push_back(prefix + num + "-of-" + tot + suffix);
    }

    // Use the parts that are present.  A computer that only holds some layers of a split model (a donor) has only some of the parts;
    // tensors in a missing part are simply not available here, and whatever needs them reports it.
    std::vector<std::string> have; size_t missing = 0;
    for (const auto& pp : parts) {
        struct stat st;
        if (stat(pp.c_str(), &st) == 0) have.push_back(pp);
        else { ++missing; fprintf(stderr, "[gguf] Split part not on this computer (fine if it only holds some layers): %s\n", pp.c_str()); }
    }
    if (have.empty()) return {first_path};
    if (have.front() != parts.front())
        fprintf(stderr, "[gguf] WARNING: part 1 (%s) is not here. It carries the model's description (layers, vocabulary), so this computer cannot use the other parts without it.\n", parts.front().c_str());
    if (missing) fprintf(stderr, "[gguf] Using %zu of %zu split parts\n", have.size(), parts.size());
    return have;
}

// Open a GGUF from an in-memory buffer. Zero disk I/O. No mmap, no file.
// The buffer is treated as borrowed -- Parser will NOT free it on close.
// Supports only single-file GGUFs (split models must use open() with files).

// --- Nexus native .nxm helpers ------------------------------------------------

static inline bool nx_ends_with_ci(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    for (size_t i = 0; i < suf.size(); ++i) {
        char a = (char)std::tolower((unsigned char)s[s.size() - suf.size() + i]);
        char b = (char)std::tolower((unsigned char)suf[i]);
        if (a != b) return false;
    }
    return true;
}

static inline std::string nx_json_unescape(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c != '\\' || i + 1 >= s.size()) { out.push_back(c); continue; }
        char e = s[++i];
        switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u':
                if (i + 4 < s.size()) {
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k) {
                        char h = s[++i];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') code |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code |= (unsigned)(h - 'A' + 10);
                    }
                    out.push_back(code < 0x80 ? (char)code : '?');
                }
                break;
            default: out.push_back(e); break;
        }
    }
    return out;
}

static inline bool nx_json_find_key(const std::string& js, const std::string& key, size_t& pos) {
    std::string needle = "\"" + key + "\"";
    pos = js.find(needle, pos);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < js.size() && std::isspace((unsigned char)js[pos])) ++pos;
    if (pos >= js.size() || js[pos] != ':') return false;
    ++pos;
    while (pos < js.size() && std::isspace((unsigned char)js[pos])) ++pos;
    return true;
}

static inline bool nx_json_string_at(const std::string& js, size_t& pos, std::string& out) {
    if (pos >= js.size() || js[pos] != '"') return false;
    ++pos;
    std::string raw;
    while (pos < js.size()) {
        char c = js[pos++];
        if (c == '"') { out = nx_json_unescape(raw); return true; }
        if (c == '\\' && pos < js.size()) { raw.push_back('\\'); raw.push_back(js[pos++]); }
        else raw.push_back(c);
    }
    return false;
}

static inline std::string nx_json_get_string(const std::string& js, const std::string& key, const std::string& def = "") {
    size_t pos = 0; if (!nx_json_find_key(js, key, pos)) return def;
    std::string out; return nx_json_string_at(js, pos, out) ? out : def;
}

static inline uint64_t nx_json_get_u64(const std::string& js, const std::string& key, uint64_t def = 0) {
    size_t pos = 0; if (!nx_json_find_key(js, key, pos)) return def;
    char* endp = nullptr; unsigned long long v = std::strtoull(js.c_str()+pos, &endp, 10);
    return endp == js.c_str()+pos ? def : (uint64_t)v;
}

static inline double nx_json_get_num(const std::string& js, const std::string& key, double def = 0.0) {
    size_t pos = 0; if (!nx_json_find_key(js, key, pos)) return def;
    char* endp = nullptr; double v = std::strtod(js.c_str()+pos, &endp);
    return endp == js.c_str()+pos ? def : v;
}

static inline std::string nx_json_extract_bracketed(const std::string& js, const std::string& key, char open, char close) {
    size_t pos = 0; if (!nx_json_find_key(js, key, pos)) return {};
    if (pos >= js.size() || js[pos] != open) return {};
    size_t start = pos; int depth = 0; bool in_str = false, esc = false;
    for (; pos < js.size(); ++pos) {
        char c = js[pos];
        if (in_str) {
            if (esc) esc = false; else if (c == '\\') esc = true; else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == open) ++depth;
        else if (c == close) { --depth; if (depth == 0) return js.substr(start, pos-start+1); }
    }
    return {};
}

static inline std::vector<std::string> nx_json_objects_in_array(const std::string& arr) {
    std::vector<std::string> out; bool in_str=false, esc=false; int depth=0; size_t start=std::string::npos;
    for (size_t i=0;i<arr.size();++i) {
        char c=arr[i];
        if (in_str) { if (esc) esc=false; else if (c=='\\') esc=true; else if (c=='"') in_str=false; continue; }
        if (c=='"') { in_str=true; continue; }
        if (c=='{') { if (depth==0) start=i; ++depth; }
        else if (c=='}') { --depth; if (depth==0 && start!=std::string::npos) { out.push_back(arr.substr(start,i-start+1)); start=std::string::npos; } }
    }
    return out;
}

static inline std::vector<uint64_t> nx_json_get_u64_array(const std::string& js, const std::string& key) {
    std::vector<uint64_t> out; std::string arr=nx_json_extract_bracketed(js,key,'[',']'); if (arr.empty()) return out;
    size_t i=1; while (i<arr.size()) {
        while (i<arr.size() && (std::isspace((unsigned char)arr[i]) || arr[i]==',')) ++i;
        if (i>=arr.size() || arr[i]==']') break;
        char* endp=nullptr; unsigned long long v=std::strtoull(arr.c_str()+i,&endp,10);
        if (endp==arr.c_str()+i) break;
        out.push_back((uint64_t)v); i=(size_t)(endp-arr.c_str());
    }
    return out;
}

static inline GGMLType nxm_storage_to_type(const std::string& st) {
    if (st=="F32") return GGMLType::F32;
    if (st=="F16") return GGMLType::F16;
    if (st=="BF16") return GGMLType::BF16;
    if (st=="NQ4_G64" || st=="Q4G64") return GGMLType::NQ4_G64;
    if (st=="NQ5_G64" || st=="Q5G64") return GGMLType::NQ5_G64;
    if (st=="NQ6_G64" || st=="Q6G64") return GGMLType::NQ6_G64;
    if (st=="NQ8_G64" || st=="Q8G64") return GGMLType::NQ8_G64;
    return GGMLType::F16;
}

static inline void nxm_add_string_kv(std::unordered_map<std::string, GGUFValue>& kv, const std::string& k, const std::string& v) { if (!v.empty()) kv[k]=v; }
static inline void nxm_add_u32_kv(std::unordered_map<std::string, GGUFValue>& kv, const std::string& k, uint64_t v) { if (v) kv[k]=(uint32_t)v; }
static inline void nxm_add_f32_kv(std::unordered_map<std::string, GGUFValue>& kv, const std::string& k, double v) { if (v != 0.0) kv[k]=(float)v; }

static inline bool nxm_parse_vocab_to_kv(const std::string& tok_json, std::unordered_map<std::string, GGUFValue>& kv) {
    std::string vocab = nx_json_extract_bracketed(tok_json, "vocab", '{', '}');
    if (vocab.empty()) return false;

    std::vector<std::string> by_id;
    size_t pos = 1;

    // tokenizer.json model.vocab: {"token": id, ...}
    while (pos < vocab.size()) {
        while (pos < vocab.size() && (std::isspace((unsigned char)vocab[pos]) || vocab[pos] == ',')) ++pos;
        if (pos >= vocab.size() || vocab[pos] == '}') break;

        std::string token;
        if (!nx_json_string_at(vocab, pos, token)) break;

        while (pos < vocab.size() && std::isspace((unsigned char)vocab[pos])) ++pos;
        if (pos >= vocab.size() || vocab[pos] != ':') break;
        ++pos;
        while (pos < vocab.size() && std::isspace((unsigned char)vocab[pos])) ++pos;

        char* endp = nullptr;
        unsigned long long id = std::strtoull(vocab.c_str() + pos, &endp, 10);
        if (endp == vocab.c_str() + pos) break;
        pos = (size_t)(endp - vocab.c_str());

        if (id >= by_id.size()) by_id.resize((size_t)id + 1);
        by_id[(size_t)id] = token;
    }

    // tokenizer.json added_tokens: [{"id":151644,"content":"<|im_start|>",...}, ...]
    //
    // This is required for Qwen/Qwen3 chat templates. Without it, the prompt log
    // shows <|im_start|> tokenized as "<", "|i", "m", "_start", ... instead of
    // one special token, poisoning the entire context before math even matters.
    std::string added = nx_json_extract_bracketed(tok_json, "added_tokens", '[', ']');
    size_t added_count = 0;
    if (!added.empty()) {
        for (const auto& obj : nx_json_objects_in_array(added)) {
            std::string content = nx_json_get_string(obj, "content", "");
            uint64_t id = nx_json_get_u64(obj, "id", UINT64_MAX);
            if (!content.empty() && id != UINT64_MAX) {
                if (id >= by_id.size()) by_id.resize((size_t)id + 1);
                by_id[(size_t)id] = content;
                ++added_count;
            }
        }
    }

    // Some tokenizer JSONs store added tokens under added_tokens_decoder instead:
    // {"151644":{"content":"<|im_start|>",...}, ...}
    std::string decoder = nx_json_extract_bracketed(tok_json, "added_tokens_decoder", '{', '}');
    if (!decoder.empty()) {
        size_t dp = 1;
        while (dp < decoder.size()) {
            while (dp < decoder.size() && (std::isspace((unsigned char)decoder[dp]) || decoder[dp] == ',')) ++dp;
            if (dp >= decoder.size() || decoder[dp] == '}') break;

            std::string id_s;
            if (!nx_json_string_at(decoder, dp, id_s)) break;

            while (dp < decoder.size() && std::isspace((unsigned char)decoder[dp])) ++dp;
            if (dp >= decoder.size() || decoder[dp] != ':') break;
            ++dp;
            while (dp < decoder.size() && std::isspace((unsigned char)decoder[dp])) ++dp;
            if (dp >= decoder.size() || decoder[dp] != '{') break;

            size_t obj_start = dp;
            int depth = 0;
            bool in_str = false, esc = false;
            for (; dp < decoder.size(); ++dp) {
                char c = decoder[dp];
                if (in_str) {
                    if (esc) esc = false;
                    else if (c == '\\') esc = true;
                    else if (c == '"') in_str = false;
                    continue;
                }
                if (c == '"') { in_str = true; continue; }
                if (c == '{') ++depth;
                else if (c == '}') {
                    --depth;
                    if (depth == 0) { ++dp; break; }
                }
            }

            std::string obj = decoder.substr(obj_start, dp - obj_start);
            std::string content = nx_json_get_string(obj, "content", "");
            char* endp = nullptr;
            unsigned long long id = std::strtoull(id_s.c_str(), &endp, 10);
            if (!content.empty() && endp != id_s.c_str()) {
                if (id >= by_id.size()) by_id.resize((size_t)id + 1);
                by_id[(size_t)id] = content;
                ++added_count;
            }
        }
    }

    if (by_id.empty()) return false;

    // Qwen/Qwen-VL native tokenizer JSON sometimes gives us most added tokens
    // but the lightweight parser can miss one object when the list has nested
    // metadata. The model log proved this exact failure mode: <|im_start|>
    // became one token, while <|im_end|> was still split as normal text. That
    // corrupts every role boundary. If we see Qwen's start token but not its
    // end token, force the canonical adjacent end token slot.
    auto find_token_id = [&](const std::string& tok)->int64_t {
        for (size_t i = 0; i < by_id.size(); ++i) {
            if (by_id[i] == tok) return (int64_t)i;
        }
        return -1;
    };

    const int64_t qwen_im_start = find_token_id("<|im_start|>");
    const int64_t qwen_im_end   = find_token_id("<|im_end|>");
    if (qwen_im_start >= 0 && qwen_im_end < 0) {
        const size_t end_id = (size_t)qwen_im_start + 1;
        if (end_id >= by_id.size()) by_id.resize(end_id + 1);
        // In Qwen tokenizers this slot is reserved for <|im_end|>. Only replace
        // an empty/reserved slot; otherwise append as a last-resort duplicate.
        if (by_id[end_id].empty() || by_id[end_id].rfind("<|", 0) == 0) {
            by_id[end_id] = "<|im_end|>";
            ++added_count;
            fprintf(stderr, "[nxm] tokenizer forced missing Qwen special token: <|im_end|> id=%zu\n", end_id);
        } else {
            by_id.push_back("<|im_end|>");
            ++added_count;
            fprintf(stderr, "[nxm] tokenizer appended missing Qwen special token: <|im_end|> id=%zu\n", by_id.size() - 1);
        }
    }

    const int64_t qwen_im_start_after = find_token_id("<|im_start|>");
    const int64_t qwen_im_end_after   = find_token_id("<|im_end|>");
    if (qwen_im_start_after >= 0 || qwen_im_end_after >= 0) {
        fprintf(stderr, "[nxm] tokenizer Qwen specials: <|im_start|>=%lld <|im_end|>=%lld\n",
                (long long)qwen_im_start_after, (long long)qwen_im_end_after);
    }

    auto arr = std::make_shared<GGUFArray>();
    arr->elem_type = GGUFType::STRING;
    arr->values.reserve(by_id.size());
    for (const auto& s : by_id) arr->values.push_back(s);
    kv["tokenizer.ggml.tokens"] = arr;

    if (added_count > 0) {
        fprintf(stderr, "[nxm] tokenizer added/special tokens loaded: %zu\n", added_count);
        fflush(stderr);
    }

    return true;
}

// ── BPE merge rules from tokenizer.json / merges.txt ─────────────────────────
// Native .nxm models embed the full tokenizer.json, but the vocab loader above only
// exposed "tokenizer.ggml.tokens". Without "tokenizer.ggml.merges" the engine cannot
// run real BPE and falls back to greedy longest-match, which splits code into token
// pieces the model never saw in training. Publish the merges as GGUF-style
// "left right" strings so Tokenizer::load() can use them.

// Locate the model.merges array. Must skip vocab entries that happen to be the
// token "merges" ("merges": 12345) -- only accept `"merges"` followed by `: [`.
static inline bool nxm_find_merges_array(const std::string& js, size_t& arr_pos) {
    const std::string needle = "\"merges\"";
    size_t from = 0;
    while (true) {
        size_t p = js.find(needle, from);
        if (p == std::string::npos) return false;
        size_t q = p + needle.size();
        while (q < js.size() && std::isspace((unsigned char)js[q])) ++q;
        if (q < js.size() && js[q] == ':') {
            ++q;
            while (q < js.size() && std::isspace((unsigned char)js[q])) ++q;
            if (q < js.size() && js[q] == '[') { arr_pos = q; return true; }
        }
        from = p + needle.size();
    }
}

static inline bool nxm_parse_merges_to_kv(const std::string& tok_json,
                                          std::unordered_map<std::string, GGUFValue>& kv) {
    size_t i = 0;
    if (!nxm_find_merges_array(tok_json, i)) return false;
    ++i;  // past '['

    auto arr = std::make_shared<GGUFArray>();
    arr->elem_type = GGUFType::STRING;
    arr->values.reserve(250000);

    while (i < tok_json.size()) {
        while (i < tok_json.size() &&
               (std::isspace((unsigned char)tok_json[i]) || tok_json[i] == ',')) ++i;
        if (i >= tok_json.size() || tok_json[i] == ']') break;

        if (tok_json[i] == '"') {                 // "left right"
            std::string rule;
            if (!nx_json_string_at(tok_json, i, rule)) break;
            if (!rule.empty()) arr->values.push_back(rule);
        } else if (tok_json[i] == '[') {          // ["left","right"] (newer tokenizers)
            ++i;
            std::string a, b;
            while (i < tok_json.size() && std::isspace((unsigned char)tok_json[i])) ++i;
            if (!nx_json_string_at(tok_json, i, a)) break;
            while (i < tok_json.size() && (std::isspace((unsigned char)tok_json[i]) || tok_json[i] == ',')) ++i;
            if (!nx_json_string_at(tok_json, i, b)) break;
            while (i < tok_json.size() && tok_json[i] != ']') ++i;
            if (i < tok_json.size()) ++i;
            arr->values.push_back(a + " " + b);
        } else {
            break;                                // unexpected token: stop cleanly
        }
    }

    if (arr->values.empty()) return false;
    kv["tokenizer.ggml.merges"] = arr;
    return true;
}

// Fallback: merges.txt packed next to the model (one "left right" rule per line;
// an optional "#version: ..." header line is ignored).
static inline bool nxm_parse_merges_txt_to_kv(const std::string& text,
                                              std::unordered_map<std::string, GGUFValue>& kv) {
    auto arr = std::make_shared<GGUFArray>();
    arr->elem_type = GGUFType::STRING;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        if (nl == std::string::npos) nl = text.size();
        size_t end = nl;
        if (end > pos && text[end - 1] == '\r') --end;
        if (end > pos && !(text[pos] == '#' && text.compare(pos, 8, "#version") == 0))
            arr->values.push_back(text.substr(pos, end - pos));
        pos = nl + 1;
    }
    if (arr->values.empty()) return false;
    kv["tokenizer.ggml.merges"] = arr;
    return true;
}

inline bool Parser::open_nxm(const std::string& path) {
    native_nxm_ = true;
    fprintf(stderr, "[nxm] Opening native model: %s\n", path.c_str()); fflush(stderr);
    if (!open_part(path, 0)) return false;
    mmap_ptr_=part_files_[0].mmap_ptr; file_size_=part_files_[0].file_size; fd_=part_files_[0].fd;
    const uint8_t* buf=(const uint8_t*)mmap_ptr_;
    if (file_size_ < 40) { fprintf(stderr,"[nxm] file too small\n"); close_files(); return false; }
    const char magic_ref[16]={'N','E','X','U','S','_','N','X','M','_','V','1','\0','\0','\0','\0'};
    if (std::memcmp(buf, magic_ref, 16) != 0) { fprintf(stderr,"[nxm] bad magic\n"); close_files(); return false; }
    size_t pos=16;
    uint64_t manifest_len=read_le<uint64_t>(buf,pos);
    uint64_t table_len=read_le<uint64_t>(buf,pos);
    (void)read_le<uint64_t>(buf,pos);
    if (pos+manifest_len+table_len > file_size_) { fprintf(stderr,"[nxm] header out of bounds\n"); close_files(); return false; }
    std::string manifest((const char*)buf+pos,(size_t)manifest_len); pos+=(size_t)manifest_len;
    std::string file_table((const char*)buf+pos,(size_t)table_len); pos+=(size_t)table_len;
    uint64_t payload_start=(uint64_t)pos;
    data_offset_=payload_start; part_files_[0].data_offset=payload_start;

    std::unordered_map<std::string,uint64_t> file_offsets, file_sizes;
    std::string files_arr=nx_json_extract_bracketed(file_table,"files",'[',']');
    for (const auto& obj: nx_json_objects_in_array(files_arr)) {
        std::string pth=nx_json_get_string(obj,"path"); uint64_t off=nx_json_get_u64(obj,"offset",0); uint64_t nb=nx_json_get_u64(obj,"nbytes",0);
        if (!pth.empty()) { file_offsets[pth]=off; file_sizes[pth]=nb; }
    }

    std::string model_obj=nx_json_extract_bracketed(manifest,"model",'{','}');
    std::string name=nx_json_get_string(model_obj,"name","Nexus native model");
    std::string model_type=nx_json_get_string(model_obj,"model_type","qwen3next");
    std::string arch=model_type.empty() ? "qwen3next" : model_type;
    // Qwen3.5 / Qwen3-Next (gated delta-net hybrids, zero-centred norms) run through the qwen3next path.  PLAIN dense Qwen2 / Qwen3 are
    // ordinary llama-style transformers (direct norms, q/k norm for Qwen3) and must keep their own architecture name -- treating
    // them as qwen3next forced zero-centred (1+w) norms and the wrong attention layout, which produces garbage.
    if (arch == "qwen3" || arch == "qwen2") { /* keep: dense Qwen */ }
    else if (arch.find("qwen") != std::string::npos) arch="qwen3next";
    nxm_add_string_kv(kv_pairs_,"general.name",name);
    nxm_add_string_kv(kv_pairs_,"general.architecture",arch);
    const std::string pfx=arch+".";
    nxm_add_u32_kv(kv_pairs_,pfx+"block_count",nx_json_get_u64(model_obj,"num_hidden_layers",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"embedding_length",nx_json_get_u64(model_obj,"hidden_size",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"feed_forward_length",nx_json_get_u64(model_obj,"intermediate_size",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"attention.head_count",nx_json_get_u64(model_obj,"num_attention_heads",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"attention.head_count_kv",nx_json_get_u64(model_obj,"num_key_value_heads",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"context_length",nx_json_get_u64(model_obj,"max_position_embeddings",0));
    nxm_add_u32_kv(kv_pairs_,pfx+"vocab_size",nx_json_get_u64(model_obj,"vocab_size",0));
    nxm_add_f32_kv(kv_pairs_,pfx+"rope.freq_base",nx_json_get_num(model_obj,"rope_theta",0.0));
    if (kv_pairs_.find(pfx+"attention.head_count")==kv_pairs_.end()) kv_pairs_[pfx+"attention.head_count"]=(uint32_t)24;
    if (kv_pairs_.find(pfx+"attention.head_count_kv")==kv_pairs_.end()) kv_pairs_[pfx+"attention.head_count_kv"]=(uint32_t)4;

    // Special-token IDs (bos/eos) must come from the actual source model, never
    // from a hardcoded guess. Without these, Tokenizer::load() falls back to
    // legacy GGUF defaults (bos=1, eos=2), which are wrong for any modern large-
    // vocab tokenizer and cause the engine to stop generation the moment it
    // happens to sample whatever ordinary token occupies id 2 in that vocab.
    {
        const uint64_t NX_MISSING = UINT64_MAX;
        uint64_t bos_tok = nx_json_get_u64(model_obj,"bos_token_id",NX_MISSING);
        uint64_t eos_tok = nx_json_get_u64(model_obj,"eos_token_id",NX_MISSING);
        if (bos_tok != NX_MISSING) {
            nxm_add_u32_kv(kv_pairs_,"tokenizer.ggml.bos_token_id",bos_tok);
        } else {
            fprintf(stderr,"[nxm] warning: model manifest has no bos_token_id; tokenizer will fall back to a legacy default\n");
        }
        if (eos_tok != NX_MISSING) {
            nxm_add_u32_kv(kv_pairs_,"tokenizer.ggml.eos_token_id",eos_tok);
        } else {
            fprintf(stderr,"[nxm] warning: model manifest has no eos_token_id; tokenizer will fall back to a legacy default\n");
        }
    }

    std::string tensors_arr=nx_json_extract_bracketed(manifest,"tensors",'[',']');
    auto objs=nx_json_objects_in_array(tensors_arr);
    tensors_.reserve(objs.size());
    for (const auto& obj: objs) {
        std::string name=nx_json_get_string(obj,"name");
        std::string st=nx_json_get_string(obj,"storage_type");
        std::string part=nx_json_get_string(obj,"part");
        uint64_t off=nx_json_get_u64(obj,"offset",0), nb=nx_json_get_u64(obj,"nbytes",0);
        if (name.empty() || part.empty() || nb==0) continue;
        TensorDesc td; td.name=name; td.type=nxm_storage_to_type(st); td.shape=nx_json_get_u64_array(obj,"shape"); td.n_elements=1;
        for (auto d: td.shape) td.n_elements*=d;
        td.offset=file_offsets[part]+off; td.data_size=nb; td.file_idx=0;
        tensors_[td.name]=std::move(td);
    }

    auto get_embedded_text = [&](const std::string& rel)->std::string {
        auto it=file_offsets.find(rel); if (it==file_offsets.end()) return {};
        uint64_t off=payload_start+it->second, nb=file_sizes[rel];
        if (off+nb > file_size_) return {};
        return std::string((const char*)buf+off,(size_t)nb);
    };
    // Prefer an explicit chat_template.jinja file when the HF model ships one.
    // tokenizer_config.json often embeds the same text, but the standalone file
    // is the cleanest source for native NXM packages created from HF folders.
    std::string tmpl_file=get_embedded_text("metadata/chat_template.jinja");
    if (!tmpl_file.empty()) {
        kv_pairs_["tokenizer.chat_template"]=tmpl_file;
        fprintf(stderr,"[nxm] tokenizer chat_template loaded from metadata/chat_template.jinja (%zu bytes)\n", tmpl_file.size());
    }
    std::string tok_cfg=get_embedded_text("metadata/tokenizer_config.json");
    if (kv_pairs_.find("tokenizer.chat_template") == kv_pairs_.end() && !tok_cfg.empty()) {
        std::string tmpl=nx_json_get_string(tok_cfg,"chat_template","");
        if (!tmpl.empty()) {
            kv_pairs_["tokenizer.chat_template"]=tmpl;
            fprintf(stderr,"[nxm] tokenizer chat_template loaded from tokenizer_config.json (%zu bytes)\n", tmpl.size());
        }
    }
    // ---- model geometry that the builder's manifest may not carry: read it from the HF config.json that travels inside the package ----
    {
        std::string cfg = get_embedded_text("metadata/config.json");
        std::string tc  = nx_json_extract_bracketed(cfg, "text_config", '{', '}');
        const std::string& c = tc.empty() ? cfg : tc;
        uint64_t nh = nx_json_get_u64(model_obj, "num_attention_heads", 0), hs = nx_json_get_u64(model_obj, "hidden_size", 0);
        uint64_t hd = nx_json_get_u64(model_obj, "head_dim", 0);
        if (!hd) hd = nx_json_get_u64(c, "head_dim", 0);
        const char* how = "manifest/config head_dim";
        if (!hd) {
            // only the Qwen3.5 / Qwen3-Next family may assume 256; everything else is hidden / heads
            if (model_type.rfind("qwen3_5", 0) == 0 || model_type.rfind("qwen3_next", 0) == 0 || model_type.rfind("qwen3next", 0) == 0) { hd = 256; how = "Qwen3.5 family default"; }
            else if (nh && hs) { hd = hs / nh; how = "hidden_size / num_attention_heads"; }
        }
        if (hd) { nxm_add_u32_kv(kv_pairs_, pfx+"attention.key_length", hd); fprintf(stderr, "[nxm] head_dim = %llu (%s)\n", (unsigned long long)hd, how); }
        else if (kv_pairs_.find(pfx+"attention.key_length")==kv_pairs_.end()) kv_pairs_[pfx+"attention.key_length"]=(uint32_t)256;
        double eps = nx_json_get_num(c, "rms_norm_eps", 0.0);
        if (eps > 0.0) nxm_add_f32_kv(kv_pairs_, pfx+"attention.layer_norm_rms_epsilon", eps);
        if (nx_json_get_num(model_obj, "rope_theta", 0.0) <= 0.0) {
            double th = nx_json_get_num(c, "rope_theta", 0.0);
            if (th <= 0.0) {   // newer configs (Qwen3.5, Qwen3-Next, ...) nest it: "rope_parameters": {"rope_theta": ...}
                std::string rp = nx_json_extract_bracketed(c, "rope_parameters", '{', '}');
                if (!rp.empty()) th = nx_json_get_num(rp, "rope_theta", 0.0);
            }
            if (th > 0.0) { nxm_add_f32_kv(kv_pairs_, pfx+"rope.freq_base", th); fprintf(stderr, "[nxm] rope_theta = %g (from config.json)\n", th); }
        }
        // conventions the builder labelled from the HF model_type (rope layout / norm style)
        std::string conv = nx_json_extract_bracketed(model_obj, "conventions", '{', '}');
        std::string rl = nx_json_get_string(conv, "rope_layout", "");
        if (!rl.empty() && rl != "unknown") nxm_add_string_kv(kv_pairs_, "nexus.rope_layout", rl);
        nxm_add_string_kv(kv_pairs_, "nexus.native_hf_weights", "1");
    }

    std::string tok_json=get_embedded_text("metadata/tokenizer.json");
    if (!tok_json.empty()) {
        if (nxm_parse_vocab_to_kv(tok_json, kv_pairs_)) fprintf(stderr,"[nxm] tokenizer vocab loaded from embedded tokenizer.json\n");
        else fprintf(stderr,"[nxm] tokenizer.json present but vocab parse failed\n");
    }
    // BPE merges: embedded tokenizer.json first, then an embedded merges.txt.
    {
        bool have_merges = false;
        if (!tok_json.empty() && nxm_parse_merges_to_kv(tok_json, kv_pairs_)) {
            have_merges = true;
            fprintf(stderr,"[nxm] tokenizer BPE merges loaded from embedded tokenizer.json\n");
        }
        if (!have_merges) {
            std::string merges_txt=get_embedded_text("metadata/merges.txt");
            if (!merges_txt.empty() && nxm_parse_merges_txt_to_kv(merges_txt, kv_pairs_)) {
                have_merges = true;
                fprintf(stderr,"[nxm] tokenizer BPE merges loaded from embedded merges.txt\n");
            }
        }
        if (!have_merges)
            fprintf(stderr,"[nxm] warning: no BPE merges found in model package; tokenizer will use legacy greedy matching\n");
    }

    extract_meta();
    fprintf(stderr,"[nxm] Parsed native model: tensors=%zu arch=%s layers=%u hidden=%u vocab=%u payload_start=%llu\n",
            tensors_.size(), meta_.arch.c_str(), meta_.n_layers, meta_.hidden_dim, meta_.vocab_size, (unsigned long long)payload_start);
    fflush(stderr);
    return true;
}

inline bool Parser::open_from_memory(const void* data, size_t size) {
    path_ = "<memory>";
    close_files();
    PartFile pf;
    pf.nx_file.from_memory(data, size);   // borrowed -- no cleanup on close
    pf.mmap_ptr  = const_cast<void*>(data);
    pf.file_size = size;
    part_files_.push_back(std::move(pf));
    mmap_ptr_  = part_files_[0].mmap_ptr;
    file_size_ = size;
    fd_        = -1;
    const uint8_t* buf = (const uint8_t*)mmap_ptr_;
    size_t pos = 0;
    if (!read_header(buf, pos)) { close_files(); return false; }
    part_files_[0].data_offset = data_offset_;
    for (auto& [name, td] : tensors_) td.file_idx = 0;
    nx_madvise((uint8_t*)mmap_ptr_ + data_offset_, file_size_ - data_offset_, 1);
    return true;
}


inline bool Parser::open_part(const std::string& path, int idx) {
    fprintf(stderr, native_nxm_ ? "[nxm] Opening file: %s\n" : "[gguf] Opening file: %s\n", path.c_str()); fflush(stderr);
    PartFile pf;
    pf.path = path;
    if (!pf.nx_file.open(path.c_str())) {
        fprintf(stderr, "[gguf] Cannot open part %d: %s\n", idx+1, path.c_str());
        return false;
    }
    fprintf(stderr, native_nxm_ ? "[nxm] File opened: %.1f MB\n" : "[gguf] File opened: %.1f MB\n", pf.nx_file.size/1048576.0); fflush(stderr);
    pf.file_size = pf.nx_file.size;
    pf.mmap_ptr  = pf.nx_file.ptr;
    pf.fd        = pf.nx_file.fd;
    nx_madvise(pf.mmap_ptr, pf.file_size, 2 /* MADV_SEQUENTIAL */);
    part_files_.push_back(pf);
    fprintf(stderr, native_nxm_ ? "[nxm] Memory mapped OK\n" : "[gguf] Memory mapped OK\n"); fflush(stderr);
    return true;
}


static inline bool nx_path_magic_is_nxm(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    unsigned char m[16] = {0};
    size_t n = std::fread(m, 1, 16, f);
    std::fclose(f);
    const unsigned char ref[16] = {'N','E','X','U','S','_','N','X','M','_','V','1',0,0,0,0};
    return n == 16 && std::memcmp(m, ref, 16) == 0;
}

// --- Parser::open -------------------------------------------------------------

inline bool Parser::open(const std::string& path) {
    path_ = path;
    close_files();
    native_nxm_ = false;

    // Native Nexus model: detect by extension AND by magic. The magic sniff
    // avoids falling through to the GGUF reader if the path has odd casing or
    // the launcher/server passed an unusual suffix.
    if (nx_ends_with_ci(path, ".nxm") || nx_path_magic_is_nxm(path)) {
        fprintf(stderr, "[nxm] native route selected in parser.h (%s)\n", path.c_str()); fflush(stderr);
        return open_nxm(path);
    }

    auto parts = find_split_parts(path);
    int n_parts = (int)parts.size();
    if (n_parts > 1)
        fprintf(stderr, "[gguf] Split model: %d parts\n", n_parts);

    for (int i = 0; i < n_parts; ++i)
        if (!open_part(parts[i], i)) return false;

    // Keep legacy single-file pointers pointing at file 0
    mmap_ptr_  = part_files_[0].mmap_ptr;
    file_size_ = part_files_[0].file_size;
    fd_        = part_files_[0].fd;

    // -- Parse file 1 fully: KV pairs + its tensor infos ----------------------
    {
        const uint8_t* buf = (const uint8_t*)part_files_[0].mmap_ptr;
        size_t pos = 0;
        if (!read_header(buf, pos)) { close_files(); return false; }
        part_files_[0].data_offset = data_offset_;
        // Tag all tensors from file 1
        for (auto& [name, td] : tensors_) td.file_idx = 0;
    }

    // -- Parse files 2+: each has its own header + its own tensor infos -------
    // In the llama.cpp split format:
    //   - n_kv = 0 for parts 2+ (no metadata, it's all in file 1)
    //   - n_tensors > 0: each file lists only ITS tensors, with offsets
    //     relative to THAT FILE's data section (not cumulative)
    for (int i = 1; i < n_parts; ++i) {
        const uint8_t* buf = (const uint8_t*)part_files_[i].mmap_ptr;
        size_t pos = 0;

        uint32_t magic = read_le<uint32_t>(buf, pos);
        if (magic != GGUF_MAGIC) {
            fprintf(stderr, "[gguf] Bad magic in part %d\n", i+1);
            close_files(); return false;
        }
        read_le<uint32_t>(buf, pos);            // version (skip)
        uint64_t n_tensors = read_le<uint64_t>(buf, pos);
        uint64_t n_kv      = read_le<uint64_t>(buf, pos);

        // n_kv should be 0 for parts 2+; read_kv_pairs handles 0 safely
        if (!read_kv_pairs(buf, pos, n_kv)) { close_files(); return false; }

        // Temporarily clear tensors_ so read_tensor_infos only adds THIS file's
        auto saved = std::move(tensors_);
        tensors_.clear();

        if (!read_tensor_infos(buf, pos, n_tensors)) { close_files(); return false; }

        part_files_[i].data_offset = (pos + 31) & ~(size_t)31;
        fprintf(stderr, "[gguf]   Part %d: %zu tensors  data_offset=%llu\n",
                i+1, tensors_.size(),
                (unsigned long long)part_files_[i].data_offset);

        // Tag and merge
        for (auto& [name, td] : tensors_) {
            td.file_idx = i;
            saved[name] = std::move(td);
        }
        tensors_ = std::move(saved);
    }

    data_offset_ = part_files_[0].data_offset;  // for legacy single-file code paths
    nx_madvise((uint8_t*)mmap_ptr_ + data_offset_,
             file_size_ - data_offset_, 1 /* MADV_RANDOM */);

    extract_meta();

    if (n_parts > 1) {
        // Report distribution
        std::vector<int> counts(n_parts, 0);
        for (auto& [name, td] : tensors_) {
            if (td.file_idx < n_parts) ++counts[td.file_idx];
        }
        fprintf(stderr, "[gguf] Tensor distribution: ");
        for (int i = 0; i < n_parts; ++i)
            fprintf(stderr, "part%d=%d ", i+1, counts[i]);
        fprintf(stderr, "  total=%zu\n", tensors_.size());
    }
    return true;
}

inline bool Parser::read_header(const uint8_t* buf, size_t& pos) {
    uint32_t magic = read_le<uint32_t>(buf, pos);
    if (magic != GGUF_MAGIC) {
        if (magic == 0x5558454Eu) { // little-endian "NEXU"
            fprintf(stderr,
                "[gguf] Bad magic: 0x%08X (this is NXM magic). "
                "If you see this, the running binary did NOT route .nxm through Parser::open_nxm. "
                "Rebuild the exact nexus_server/nexus_server_native binary the launcher starts.\n",
                magic);
        } else {
            fprintf(stderr, "[gguf] Bad magic: 0x%08X\n", magic);
        }
        return false;
    }

    uint32_t version = read_le<uint32_t>(buf, pos);
    if (version < 2 || version > 3) {
        fprintf(stderr, "[gguf] Unsupported version: %u\n", version);
        return false;
    }

    uint64_t n_tensors = read_le<uint64_t>(buf, pos);
    uint64_t n_kv      = read_le<uint64_t>(buf, pos);

    if (!read_kv_pairs(buf, pos, n_kv))         return false;
    if (!read_tensor_infos(buf, pos, n_tensors)) return false;

    // Data section starts at next 32-byte alignment after tensor infos
    data_offset_ = (pos + 31) & ~(size_t)31;

    return true;
}

inline std::string Parser::read_str(const uint8_t* buf, size_t& pos) {
    uint64_t len = read_le<uint64_t>(buf, pos);
    std::string s((const char*)(buf + pos), len);
    pos += len;
    return s;
}

inline GGUFValue Parser::read_value(const uint8_t* buf, size_t& pos, GGUFType type) {
    switch (type) {
        case GGUFType::UINT8:   return read_le<uint8_t>(buf, pos);
        case GGUFType::INT8:    return read_le<int8_t>(buf, pos);
        case GGUFType::UINT16:  return read_le<uint16_t>(buf, pos);
        case GGUFType::INT16:   return read_le<int16_t>(buf, pos);
        case GGUFType::UINT32:  return read_le<uint32_t>(buf, pos);
        case GGUFType::INT32:   return read_le<int32_t>(buf, pos);
        case GGUFType::FLOAT32: return read_le<float>(buf, pos);
        case GGUFType::BOOL:    return (bool)read_le<uint8_t>(buf, pos);
        case GGUFType::STRING:  return read_str(buf, pos);
        case GGUFType::UINT64:  return read_le<uint64_t>(buf, pos);
        case GGUFType::INT64:   return read_le<int64_t>(buf, pos);
        case GGUFType::FLOAT64: return read_le<double>(buf, pos);
        case GGUFType::ARRAY: {
            auto arr = std::make_shared<GGUFArray>();
            arr->elem_type = (GGUFType)read_le<uint32_t>(buf, pos);
            uint64_t count = read_le<uint64_t>(buf, pos);
            arr->values.reserve(count);
            for (uint64_t i = 0; i < count; ++i) {
                arr->values.push_back(read_value(buf, pos, arr->elem_type));
            }
            return arr;
        }
        default:
            fprintf(stderr, "[gguf] Unknown value type %u\n", (uint32_t)type);
            return uint32_t(0);
    }
}

inline bool Parser::read_kv_pairs(const uint8_t* buf, size_t& pos, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        std::string key = read_str(buf, pos);
        GGUFType type   = (GGUFType)read_le<uint32_t>(buf, pos);
        GGUFValue val   = read_value(buf, pos, type);
        kv_pairs_[key]  = std::move(val);
    }
    return true;
}

inline bool Parser::read_tensor_infos(const uint8_t* buf, size_t& pos, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        TensorDesc t;
        t.name = read_str(buf, pos);

        uint32_t n_dims = read_le<uint32_t>(buf, pos);
        t.shape.resize(n_dims);
        t.n_elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            t.shape[d]    = read_le<uint64_t>(buf, pos);
            t.n_elements *= t.shape[d];
        }

        t.type   = (GGMLType)read_le<uint32_t>(buf, pos);
        t.offset = read_le<uint64_t>(buf, pos);

        // Compute on-disk tensor byte size row-by-row.
        // GGUF quantized tensors are stored with each row rounded up to the quant block.
        // This matters for PhiMoE expert-down tensors where ne0/cols = 960 and K-block = 256.
        auto qi = get_quant_info(t.type);
        if (qi.block_size == 0 || qi.elements == 0) {
            fprintf(stderr,
                "[gguf] FATAL: tensor %s uses unsupported GGML type %d. Refusing unsafe size guess.\n",
                t.name.c_str(),
                (int)t.type);
            fprintf(stderr,
                "[nexus] UNSUPPORTED MODEL FORMAT: this file stores weights as %s (type %d, first seen in tensor \"%s\"), which Nexus cannot run yet.\n"
                "[nexus]   Supported weight formats: F32, F16, BF16, Q8_0, Q4_0, Q5_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Nexus NQ4/5/6/8.\n"
                "[nexus]   Please pick a different quantization of this model (for example Q4_K_M, Q5_K_M, Q6_K or Q8_0).\n",
                ggml_type_name((int)t.type), (int)t.type, t.name.c_str());
            return false;
        }

        if (!t.shape.empty() && qi.elements > 0) {
            const uint64_t ne0 = t.shape[0];

            uint64_t rows = 1;
            for (size_t d = 1; d < t.shape.size(); ++d) {
                rows *= t.shape[d];
            }

            const uint64_t blocks_per_row =
                (ne0 + (uint64_t)qi.elements - 1) / (uint64_t)qi.elements;

            t.data_size = rows * blocks_per_row * (uint64_t)qi.block_size;
        } else {
            t.data_size = t.n_elements * 4; // fallback f32
        }

        tensors_[t.name] = std::move(t);
    }
    return true;
}

// --- Extract model metadata from KV pairs ------------------------------------

inline void Parser::extract_meta() {
    meta_.arch = get_str("general.architecture", "llama");
    meta_.name = get_str("general.name", "");

    const std::string& a = meta_.arch;

    auto try_u32 = [&](std::initializer_list<std::string> suffixes, uint32_t def) -> uint32_t {
        for (const auto& s : suffixes) {
            // Try with arch prefix
            for (const auto& key : {a + "." + s, s}) {
                auto it = kv_pairs_.find(key);
                if (it != kv_pairs_.end()) {
                    if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
                    if (auto* v = std::get_if<uint64_t>(&it->second)) return (uint32_t)*v;
                    if (auto* v = std::get_if<int32_t>(&it->second))  return (uint32_t)*v;
                }
            }
        }
        return def;
    };
    
    auto try_bool = [&](std::initializer_list<std::string> suffixes, bool def) -> bool {
        for (const auto& s : suffixes) {
            for (const auto& key : {a + "." + s, s}) {
                auto it = kv_pairs_.find(key);
                if (it != kv_pairs_.end()) {
                    if (auto* v = std::get_if<bool>(&it->second)) return *v;
                    if (auto* v = std::get_if<uint8_t>(&it->second)) return *v != 0;
                    if (auto* v = std::get_if<int32_t>(&it->second)) return *v != 0;
                    if (auto* v = std::get_if<uint32_t>(&it->second)) return *v != 0;
                }
            }
        }
        return def;
    };

    auto try_str = [&](std::initializer_list<std::string> suffixes, const std::string& def) -> std::string {
        for (const auto& s : suffixes) {
            for (const auto& key : {a + "." + s, s}) {
                auto it = kv_pairs_.find(key);
                if (it != kv_pairs_.end()) {
                    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
                }
            }
        }
        return def;
    };

    auto try_f32 = [&](std::initializer_list<std::string> suffixes, float def) -> float {
        for (const auto& s : suffixes) {
            for (const auto& key : {a + "." + s, s}) {
                auto it = kv_pairs_.find(key);
                if (it != kv_pairs_.end()) {
                    if (auto* v = std::get_if<float>(&it->second))  return *v;
                    if (auto* v = std::get_if<double>(&it->second)) return (float)*v;
                }
            }
        }
        return def;
    };

    meta_.n_layers       = try_u32({"block_count"}, 32);
    meta_.n_heads        = try_u32({"attention.head_count"}, 32);
    meta_.n_kv_heads     = try_u32({"attention.head_count_kv"}, meta_.n_heads);
    meta_.context_length = try_u32({"context_length"}, 4096);
    meta_.vocab_size     = try_u32({"vocab_size"}, 32000);
    meta_.ffn_dim        = try_u32({"feed_forward_length"}, 0);
    meta_.n_experts      = try_u32({"expert_count"}, 0);
    meta_.n_experts_used = try_u32({"expert_used_count"}, 0);
    meta_.expert_ffn_dim        = try_u32({"expert_feed_forward_length"}, 0);
    meta_.expert_shared_ffn_dim = try_u32({"expert_shared_feed_forward_length"}, 0);
    meta_.expert_weights_scale  = try_f32({"expert_weights_scale", "routed_scaling_factor"}, 1.0f);
    meta_.expert_weights_norm   = try_bool({"expert_weights_norm", "norm_topk_prob"}, false);
    meta_.expert_gating_func    = try_str({"expert_gating_func"}, "");

    meta_.rms_norm_eps   = try_f32({"attention.layer_norm_rms_epsilon",
                                     "layer_norm_rms_epsilon"}, 1e-5f);
    // Try every known key name for rope base - models differ on this
    meta_.rope_freq_base = try_f32({"rope.freq_base",
                                     "rope_freq_base",
                                     "attention.rope_theta",
                                     "rope.theta"}, 10000.0f);

    // -- Hidden dim: read directly first, infer from tensor shapes as fallback --
    meta_.hidden_dim = try_u32({"embedding_length",    // standard GGUF key
                                  "hidden_size",
                                  "d_model"}, 0);

    // head_dim: try explicit key first
    meta_.head_dim = try_u32({"attention.key_length",
                               "attention.head_dim"}, 0);

    // If hidden_dim or head_dim missing, infer from tensor shapes.
    // Check Q, K, or fused QKV tensor (Phi uses attn_qkv.weight, others use attn_q/k.weight)
    if (meta_.hidden_dim == 0 || meta_.head_dim == 0) {
        for (auto& [name, t] : tensors_) {
            bool is_q     = name.find("attn_q.weight")   != std::string::npos;
            bool is_k     = name.find("attn_k.weight")   != std::string::npos;
            bool is_qkv   = name.find("attn_qkv.weight") != std::string::npos;
            if (!(is_q || is_k || is_qkv) || t.shape.size() < 2) continue;

            uint32_t h_dim = (uint32_t)t.shape[1];  // hidden_dim (input cols)
            uint32_t rows  = (uint32_t)t.shape[0];  // output rows

            if (meta_.hidden_dim == 0 && h_dim > 0)
                meta_.hidden_dim = h_dim;

            if (meta_.head_dim == 0 && meta_.n_heads > 0) {
                if (is_q || is_k) {
                    // Q/K shape: [n_heads * head_dim, hidden_dim]
                    meta_.head_dim = rows / meta_.n_heads;
                } else if (is_qkv && meta_.n_kv_heads > 0) {
                    // Fused QKV shape: [(n_heads + 2*n_kv_heads) * head_dim, hidden_dim]
                    // So head_dim = rows / (n_heads + 2*n_kv_heads)
                    uint32_t total_heads = meta_.n_heads + 2 * meta_.n_kv_heads;
                    if (total_heads > 0) meta_.head_dim = rows / total_heads;
                }
            }
            if (meta_.hidden_dim > 0 && meta_.head_dim > 0) break;
        }
    }

    // Final fallback
    if (meta_.hidden_dim == 0) meta_.hidden_dim = meta_.n_heads * meta_.head_dim;
    if (meta_.head_dim   == 0 && meta_.n_heads > 0)
        meta_.head_dim = meta_.hidden_dim / meta_.n_heads;

    // -- Validate/infer FFN + MoE metadata from tensor shapes -------------------
    // Newer GGUF MoE models may not expose the old feed_forward_length key, and
    // their FFN tensors are named *_exps / *_shexp instead of ffn_up/ffn_gate.
    if (meta_.ffn_dim == 0) {
        for (auto& [name, t] : tensors_) {
            const bool dense_ffn =
                name.find("ffn_up.weight")   != std::string::npos ||
                name.find("ffn_gate.weight") != std::string::npos ||
                name.find("mlp.up_proj.weight")   != std::string::npos ||
                name.find("mlp.gate_proj.weight") != std::string::npos;

            const bool moe_ffn =
                name.find("ffn_up_exps.weight")      != std::string::npos ||
                name.find("ffn_gate_exps.weight")    != std::string::npos ||
                name.find("ffn_down_exps.weight")    != std::string::npos ||
                name.find("ffn_up_shexp.weight")     != std::string::npos ||
                name.find("ffn_gate_shexp.weight")   != std::string::npos ||
                name.find("ffn_down_shexp.weight")   != std::string::npos ||
                name.find("ffn_gate_up_exps.weight") != std::string::npos;

            if (!(dense_ffn || moe_ffn) || t.shape.empty()) continue;

            if (name.find("ffn_down") != std::string::npos && t.shape.size() >= 2)
                meta_.ffn_dim = (uint32_t)t.shape[1];
            else
                meta_.ffn_dim = (uint32_t)t.shape[0];

            break;
        }
    }

    if (meta_.ffn_dim == 0 && meta_.expert_ffn_dim > 0) {
        meta_.ffn_dim = meta_.expert_ffn_dim;
    }

    // Infer expert count from merged expert tensors if metadata did not expose it.
    if (meta_.n_experts == 0) {
        for (auto& [name, t] : tensors_) {
            const bool merged_expert =
                name.find("_exps.weight") != std::string::npos ||
                name.find("ffn_gate_up_exps.weight") != std::string::npos;

            if (merged_expert && t.shape.size() >= 3) {
                meta_.n_experts = (uint32_t)t.shape[2];
                break;
            }
        }
    }
}

// --- Parser::load_weights ----------------------------------------------------

inline bool Parser::load_weights(MemoryManager& mm, ProgressCB cb) {
    if (part_files_.empty() || !part_files_[0].mmap_ptr) {
        fprintf(stderr, "[gguf] load_weights: no mapped file\n");
        return false;
    }
    fprintf(stderr, "[gguf] load_weights: %zu tensors\n", tensors_.size()); fflush(stderr);

    size_t total        = tensors_.size();
    size_t loaded       = 0;
    size_t bytes_total  = 0;
    size_t bytes_loaded = 0;
    size_t n_direct = 0, bytes_direct = 0;
    for (auto& [n, t] : tensors_) bytes_total += t.data_size;

    // Build flat list to avoid iterator invalidation issues
    std::vector<std::string> names;
    names.reserve(total);
    for (auto& [n, t] : tensors_) names.push_back(n);

    int direct_limit = -1;   // layers below this may be read in place (the rest are copied page-aligned so the card can still map them)
    for (size_t ti = 0; ti < names.size(); ++ti) {
        const std::string& name = names[ti];
        TensorDesc& t = tensors_.at(name);
        if (t.loaded) { ++loaded; continue; }

        int fidx = (t.file_idx >= 0 && t.file_idx < (int)part_files_.size()) ? t.file_idx : 0;
        const PartFile& pf = part_files_[fidx];
        if (!pf.mmap_ptr) {
            fprintf(stderr, "[gguf] null mmap for tensor %s\n", name.c_str());
            return false;
        }
        if (pf.data_offset + t.offset + t.data_size > pf.file_size) {
            fprintf(stderr, "[gguf] tensor %s out of bounds\n", name.c_str());
            return false;
        }

        // Direct-to-GPU: big matrices of the layers that go to the card are read in place from the file.
        if (direct_limit < 0) direct_limit = nx_direct_vram_layer_limit(tensors_);
        if (nx_direct_gpu_layers() > 0 && pf.nx_file.writable && t.data_size >= (256u << 10) &&
            name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0 && !nx_keep_in_ram_name(name)) {
            const int dl = nx_layer_of_tensor(name);
            if (dl >= 0 && dl < nx_direct_gpu_layers() && dl < direct_limit) {
                t.data = (void*)((const uint8_t*)pf.mmap_ptr + pf.data_offset + t.offset);
                nx_direct_ranges().push_back({(const uint8_t*)t.data, t.data_size, dl});
                t.loaded = true;
                ++loaded;
                bytes_loaded += t.data_size;
                ++n_direct; bytes_direct += t.data_size;
                if (cb) cb({loaded, total, bytes_loaded, bytes_total, name});
                continue;
            }
        }

        // Allocate in our memory pool and copy - Nexus owns this memory,
        // the OS cannot evict it under pressure (unlike mmap file pages).
        t.data = mm.alloc_weights(t.data_size, name, -1);
        if (!t.data) {
            fprintf(stderr, "[gguf] Failed to allocate %s (%zu bytes)\n",
                    name.c_str(), t.data_size);
            return false;
        }
        const uint8_t* src = (const uint8_t*)pf.mmap_ptr + pf.data_offset + t.offset;
        memcpy(t.data, src, t.data_size);

        t.loaded = true;
        ++loaded;
        bytes_loaded += t.data_size;

        if (cb) {
            cb({loaded, total, bytes_loaded, bytes_total, name});
        }
    }

    fprintf(stderr, "[gguf] load_weights done: %zu/%zu tensors\n", loaded, total); fflush(stderr);
    if (n_direct) { fprintf(stderr, "[load] direct-to-GPU: %zu weight tensors (%zu MB) are read from the model file when uploaded -- no RAM copy\n", n_direct, bytes_direct >> 20); fflush(stderr); }
    else if (nx_direct_gpu_layers() > 0) { fprintf(stderr, "[load] direct-to-GPU not possible for this file/system -- loading normally\n"); fflush(stderr); }
    nx_madvise((void*)part_files_[0].mmap_ptr, part_files_[0].data_offset, 4);
    return true;
}

inline bool Parser::load_layer(int layer_id, MemoryManager& mm) {
    if (!mmap_ptr_) return false;

    const uint8_t* file_base = (const uint8_t*)mmap_ptr_;
    std::string prefix = "blk." + std::to_string(layer_id) + ".";

    for (auto& [name, t] : tensors_) {
        if (nx_layer_of_tensor(name) != layer_id) continue;   // blk.N. / model.language_model.layers.N. / model.layers.N. / language_model.model.layers.N.
        if (t.loaded) continue;

        t.data = mm.alloc_weights(t.data_size, name, layer_id);
        if (!t.data) return false;

        memcpy(t.data, file_base + data_offset_ + t.offset, t.data_size);
        t.loaded = true;
    }
    return true;
}

inline void Parser::free_layer(int layer_id, MemoryManager& mm) {
    for (auto& [name, t] : tensors_) {
        if (nx_layer_of_tensor(name) != layer_id) continue;
        if (!t.loaded || !t.data) continue;

        mm.free_weights(t.data);
        t.data   = nullptr;
        t.loaded = false;
    }
}

// --- Accessors ----------------------------------------------------------------

inline std::string Parser::get_str(const std::string& key, const std::string& def) const {
    auto it = kv_pairs_.find(key);
    if (it == kv_pairs_.end()) return def;
    if (auto* v = std::get_if<std::string>(&it->second)) return *v;
    return def;
}

inline uint32_t Parser::get_u32(const std::string& key, uint32_t def) const {
    auto it = kv_pairs_.find(key);
    if (it == kv_pairs_.end()) return def;
    if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return (uint32_t)*v;
    if (auto* v = std::get_if<int32_t>(&it->second))  return (uint32_t)*v;
    return def;
}

inline uint64_t Parser::get_u64(const std::string& key, uint64_t def) const {
    auto it = kv_pairs_.find(key);
    if (it == kv_pairs_.end()) return def;
    if (auto* v = std::get_if<uint64_t>(&it->second)) return *v;
    if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
    return def;
}

inline float Parser::get_f32(const std::string& key, float def) const {
    auto it = kv_pairs_.find(key);
    if (it == kv_pairs_.end()) return def;
    if (auto* v = std::get_if<float>(&it->second)) return *v;
    if (auto* v = std::get_if<double>(&it->second)) return (float)*v;
    return def;
}

inline TensorDesc* Parser::find_tensor(const std::string& substr) {
    for (auto& [name, t] : tensors_) {
        if (name.find(substr) != std::string::npos) return &t;
    }
    return nullptr;
}

inline void Parser::print_meta() const {
    printf("\n-- Model Metadata -----------------------------\n");
    printf("  Name:    %s\n", meta_.name.c_str());
    printf("  Arch:    %s\n", meta_.arch.c_str());
    printf("  Layers:  %u\n", meta_.n_layers);
    printf("  Heads:   %u  (KV: %u)\n", meta_.n_heads, meta_.n_kv_heads);
    printf("  HeadDim: %u%s\n", meta_.head_dim, meta_.head_dim == 0 ? " - WARN: 0!" : "");
    printf("  Hidden:  %u%s\n", meta_.hidden_dim, meta_.hidden_dim == 0 ? " - WARN: 0!" : "");
    printf("  FFN dim: %u%s\n", meta_.ffn_dim, meta_.ffn_dim == 0 ? " - WARN: 0!" : "");
    printf("  Vocab:   %u\n", meta_.vocab_size);
    printf("  Ctx max: %u\n", meta_.context_length);
    printf("  RMSnorm eps: %g\n", meta_.rms_norm_eps);
    printf("  RoPE base:   %g\n", meta_.rope_freq_base);
    if (meta_.n_experts > 0) {
        printf("  MoE experts: %u (use %u)\n", meta_.n_experts, meta_.n_experts_used);
    }
    printf("  Tensors: %zu\n", tensors_.size());
    printf("----------------------------------------------\n\n");
}

inline void Parser::print_tensors() const {
    printf("\n-- Tensors (%zu) ------------------------------\n", tensors_.size());
    for (const auto& [name, t] : tensors_) {
        auto qi = get_quant_info(t.type);
        printf("  %-50s  %-6s  [", name.c_str(), qi.name);
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i > 0) printf(", ");
            printf("%llu", (unsigned long long)t.shape[i]);
        }
        printf("]  %s\n", MemoryManager::fmt_bytes(t.data_size).c_str());
    }
    printf("----------------------------------------------\n\n");
}

} // namespace gguf
} // namespace nexus
