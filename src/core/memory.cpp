/*
 * nexus/src/core/memory.cpp
 *
 * Memory manager implementation.
 * Linux-specific for now: mmap, mlock, madvise, huge pages.
 */

#include "memory.h"

#include <cstdio>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <iomanip>

#include <fstream>
#include <string>

// ─── Platform memory abstraction ─────────────────────────────────────────────
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <psapi.h>
#  pragma comment(lib, "psapi.lib")
   // mmap/madvise/mlock shims for Windows
   inline void* nx_mmap(size_t bytes) {
       return VirtualAlloc(nullptr, bytes, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
   }
   inline void  nx_munmap(void* p, size_t) { if(p) VirtualFree(p,0,MEM_RELEASE); }
   inline void  nx_madvise(void*, size_t, int) {}   // no-op
   inline bool  nx_mlock(void* p, size_t n)  { return VirtualLock(p,n) != 0; }
   inline void  nx_munlock(void* p, size_t n){ VirtualUnlock(p,n); }
   inline size_t nx_available_ram() {
       MEMORYSTATUSEX ms; ms.dwLength=sizeof(ms); GlobalMemoryStatusEx(&ms);
       return (size_t)ms.ullAvailPhys;
   }
   inline size_t nx_total_ram() {
       MEMORYSTATUSEX ms; ms.dwLength=sizeof(ms); GlobalMemoryStatusEx(&ms);
       return (size_t)ms.ullTotalPhys;
   }
   // Raise working set so VirtualLock works
   inline void nx_raise_mlock_limit() {
       SIZE_T minWS, maxWS;
       GetProcessWorkingSetSize(GetCurrentProcess(), &minWS, &maxWS);
       SetProcessWorkingSetSize(GetCurrentProcess(), minWS+256*1024*1024, maxWS+256*1024*1024);
   }
#  define MAP_FAILED nullptr
   // MADV constants — unused on Windows (nx_madvise is a no-op) but must compile
#  define MADV_HUGEPAGE   14
#  define MADV_SEQUENTIAL  2
#  define MADV_RANDOM      1
#  define MADV_DONTNEED    4
#else
#  include <sys/mman.h>
#  include <sys/resource.h>
#  include <unistd.h>
#  ifndef MAP_HUGETLB
#    define MAP_HUGETLB 0x40000
#  endif
#  ifndef MAP_HUGE_2MB
#    define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#    define MAP_HUGE_SHIFT 26
#  endif
#  ifndef MADV_HUGEPAGE
#    define MADV_HUGEPAGE 14
#  endif
#  ifndef MADV_SEQUENTIAL
#    define MADV_SEQUENTIAL 2
#  endif
#  ifndef MADV_RANDOM
#    define MADV_RANDOM 1
#  endif
#  ifndef MADV_DONTNEED
#    define MADV_DONTNEED 4
#  endif
   inline void* nx_mmap(size_t bytes) {
       return mmap(nullptr, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
   }
   inline void  nx_munmap(void* p, size_t n) { munmap(p,n); }
   inline void  nx_madvise(void* p, size_t n, int adv) { madvise(p,n,adv); }
   inline bool  nx_mlock(void* p, size_t n)   { return mlock(p,n)==0; }
   inline void  nx_munlock(void* p, size_t n)  { munlock(p,n); }
   inline size_t nx_available_ram() {
       std::ifstream f("/proc/meminfo");
       std::string line;
       while (std::getline(f, line))
           if (line.find("MemAvailable:") == 0) {
               size_t kb=0; sscanf(line.c_str(),"MemAvailable: %zu kB",&kb);
               return kb*1024;
           }
       long p=sysconf(_SC_AVPHYS_PAGES), ps=sysconf(_SC_PAGE_SIZE);
       return (p>0&&ps>0)?(size_t)p*ps:0;
   }
   inline size_t nx_total_ram() {
       long p=sysconf(_SC_PHYS_PAGES), ps=sysconf(_SC_PAGE_SIZE);
       return (p>0&&ps>0)?(size_t)p*ps:0;
   }
   inline void nx_raise_mlock_limit() {
       struct rlimit rl;
       if (getrlimit(RLIMIT_MEMLOCK, &rl)==0 && rl.rlim_cur!=RLIM_INFINITY) {
           rl.rlim_cur=rl.rlim_max; setrlimit(RLIMIT_MEMLOCK, &rl);
       }
   }
#  // huge page mmap helper (Linux only)
   inline void* nx_mmap_huge(size_t bytes) {
       void* p = mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                      MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_2MB, -1, 0);
       if (p == MAP_FAILED) {
           p = nx_mmap(bytes);
           if (p != MAP_FAILED) nx_madvise(p, bytes, MADV_HUGEPAGE);
       }
       return p;
   }
#endif
// ─────────────────────────────────────────────────────────────────────────────

namespace nexus {

// ─── Utilities ──────────────────────────────────────────────────────────────

static size_t align_up(size_t val, size_t align) {
    return (val + align - 1) & ~(align - 1);
}

uint64_t MemoryManager::now_ns() const {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
}

std::string MemoryManager::fmt_bytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double val = (double)bytes;
    int unit = 0;
    while (val >= 1024.0 && unit < 4) { val /= 1024.0; ++unit; }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(unit > 0 ? 2 : 0) << val << " " << units[unit];
    return ss.str();
}

size_t MemoryManager::system_available_bytes() const { return nx_available_ram(); }
size_t MemoryManager::system_total_bytes() const { return nx_total_ram(); }

// ─── Constructor / Destructor ────────────────────────────────────────────────

MemoryManager::MemoryManager() : MemoryManager(Config{}) {}

MemoryManager::MemoryManager(const Config& cfg)
    : cfg_(cfg), kv_page_tokens_(cfg.kv_page_tokens)
{
    // Activation pool — anonymous mmap with huge page hint
    activation_size_ = cfg.activation_pool;
    activation_pool_ = (uint8_t*)nx_mmap(activation_size_);
    if (activation_pool_ == MAP_FAILED || !activation_pool_) {
        throw std::runtime_error("Failed to allocate activation pool");
    }
    nx_madvise(activation_pool_, activation_size_, MADV_HUGEPAGE);
    nx_mlock(activation_pool_, activation_size_);

    // Scratch pool
    scratch_size_ = cfg.scratch_pool;
    scratch_pool_ = (uint8_t*)nx_mmap(scratch_size_);
    if (scratch_pool_ == MAP_FAILED || !scratch_pool_) {
        nx_munmap(activation_pool_, activation_size_);
        throw std::runtime_error("Failed to allocate scratch pool");
    }
    nx_madvise(scratch_pool_, scratch_size_, MADV_HUGEPAGE);
    nx_raise_mlock_limit();
}

MemoryManager::~MemoryManager() {
    // Free all weight allocations
    for (auto& alloc : weight_allocs_) {
        if (alloc.locked) nx_munlock(alloc.ptr, alloc.size);
        nx_munmap(alloc.ptr, alloc.size);
    }

    // Free KV pages
    for (auto* page : kv_pages_) {
        if (page->k_data) nx_munmap(page->k_data,
            page->page_capacity * /* TODO: store total_bytes in KVPage */ 0);
        delete page;
    }

    if (activation_pool_ && activation_pool_ != (uint8_t*)MAP_FAILED) {
        nx_munlock(activation_pool_, activation_size_);
        nx_munmap(activation_pool_, activation_size_);
    }
    if (scratch_pool_ && scratch_pool_ != (uint8_t*)MAP_FAILED) {
        nx_munmap(scratch_pool_, scratch_size_);
    }
}

// ─── Huge page allocation ────────────────────────────────────────────────────

void* MemoryManager::alloc_huge(size_t bytes, const std::string& name) {
    // Round up to 2MB boundary for huge pages
    size_t aligned = align_up(bytes, 2 * 1024 * 1024);

    void* ptr = nullptr;

#ifdef _WIN32
    ptr = nx_mmap(aligned);
#else
    if (cfg_.prefer_huge_pages) ptr = nx_mmap_huge(aligned);
    else                        ptr = nx_mmap(aligned);
#endif

    // VirtualAlloc (Windows) returns NULL on failure; mmap (Linux) returns MAP_FAILED.
    // Check both so the error is caught on all platforms.
    if (!ptr || ptr == MAP_FAILED) {
        fprintf(stderr, "[memory] FATAL: Failed to allocate %s (%s)\n",
                name.c_str(), fmt_bytes(bytes).c_str());
        fflush(stderr);
        return nullptr;
    }

    nx_madvise(ptr, aligned, MADV_SEQUENTIAL);

    return ptr;
}

void MemoryManager::free_huge(void* ptr, size_t bytes) {
    if (!ptr) return;
    size_t aligned = align_up(bytes, 2 * 1024 * 1024);
    nx_munmap(ptr, aligned);
}

bool MemoryManager::try_lock(void* ptr, size_t bytes) { return nx_mlock(ptr, bytes); }
void MemoryManager::do_unlock(void* ptr, size_t bytes) { nx_munlock(ptr, bytes); }

// ─── Weight allocation ────────────────────────────────────────────────────────

void* MemoryManager::alloc_weights(size_t bytes, const std::string& name, int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we have headroom
    size_t available = system_available_bytes();
    if (bytes > available * 0.9) {
        fprintf(stderr, "[memory] WARNING: Allocating %s (%s) but only %s available\n",
                name.c_str(), fmt_bytes(bytes).c_str(), fmt_bytes(available).c_str());
    }

    // Tensors smaller than one huge page (norm weights, biases, small projections, the whole vision tower's tiny tensors...) used
    // to get a private 2 MB huge page EACH -- a 4 KB tensor then cost 2 MB of RAM (hundreds of MB to over 1 GB on a big model).
    // They now get ordinary 4 KB pages, sized to the tensor.  The pointer is still page-aligned, so the GPU can still map it.
    const size_t kHugePage = 2 * 1024 * 1024;
    const bool small_tensor = bytes < kHugePage;
    void* ptr = nullptr;
    size_t alloc_bytes;
    if (small_tensor) {
        alloc_bytes = align_up(bytes ? bytes : 1, 4096);
        ptr = nx_mmap(alloc_bytes);
        if (!ptr || ptr == MAP_FAILED) {
            fprintf(stderr, "[memory] FATAL: Failed to allocate %s (%s)\n", name.c_str(), fmt_bytes(bytes).c_str());
            fflush(stderr);
            return nullptr;
        }
    } else {
        ptr = alloc_huge(bytes, name);
        alloc_bytes = align_up(bytes, kHugePage);
    }
    if (!ptr) return nullptr;

    Allocation alloc{};
    alloc.ptr       = ptr;
    alloc.size      = alloc_bytes;
    alloc.pool      = MemPool::WEIGHTS;
    alloc.page_type = small_tensor ? PageType::STANDARD : PageType::HUGE_2MB;
    alloc.locked    = false;
    alloc.on_gpu    = false;
    alloc.name      = name;
    alloc.layer_id  = layer_id;
    alloc.alloc_time_ns = now_ns();
    alloc.last_access_ns = alloc.alloc_time_ns;

    size_t idx = weight_allocs_.size();
    weight_allocs_.push_back(alloc);
    ptr_to_alloc_idx_[ptr] = idx;

    if (layer_id >= 0) {
        layer_ptrs_[layer_id].push_back(ptr);
    }

    return ptr;
}

void MemoryManager::free_weights(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = ptr_to_alloc_idx_.find(ptr);
    if (it == ptr_to_alloc_idx_.end()) return;

    size_t idx = it->second;
    Allocation& alloc = weight_allocs_[idx];

    if (alloc.locked) do_unlock(ptr, alloc.size);
    if (alloc.page_type == PageType::HUGE_2MB) free_huge(ptr, alloc.size);
    else                                       nx_munmap(ptr, alloc.size);   // small tensor: exactly the 4 KB-rounded size that was mapped

    ptr_to_alloc_idx_.erase(it);
    weight_allocs_.erase(weight_allocs_.begin() + idx);

    // Rebuild index (simple approach for now)
    ptr_to_alloc_idx_.clear();
    for (size_t i = 0; i < weight_allocs_.size(); ++i) {
        ptr_to_alloc_idx_[weight_allocs_[i].ptr] = i;
    }
}

// ─── Layer pinning ────────────────────────────────────────────────────────────

void MemoryManager::pin_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Already pinned?
    if (std::find(pinned_layers_.begin(), pinned_layers_.end(), layer_id)
        != pinned_layers_.end()) return;

    // At capacity — evict oldest pinned layer
    if ((int)pinned_layers_.size() >= cfg_.max_locked_layers) {
        int evict = pinned_layers_.front();
        pinned_layers_.erase(pinned_layers_.begin());

        // Unlock evicted layer's memory
        auto it = layer_ptrs_.find(evict);
        if (it != layer_ptrs_.end()) {
            for (void* ptr : it->second) {
                auto ait = ptr_to_alloc_idx_.find(ptr);
                if (ait != ptr_to_alloc_idx_.end()) {
                    auto& alloc = weight_allocs_[ait->second];
                    if (alloc.locked) {
                        do_unlock(ptr, alloc.size);
                        alloc.locked = false;
                    }
                }
            }
        }
    }

    // Lock this layer's weights
    auto it = layer_ptrs_.find(layer_id);
    if (it != layer_ptrs_.end()) {
        for (void* ptr : it->second) {
            auto ait = ptr_to_alloc_idx_.find(ptr);
            if (ait != ptr_to_alloc_idx_.end()) {
                auto& alloc = weight_allocs_[ait->second];
                if (!alloc.locked) {
                    alloc.locked = try_lock(ptr, alloc.size);
                    // Change access hint for random access during computation
                    nx_madvise(ptr, alloc.size, MADV_RANDOM);
                }
            }
        }
    }

    pinned_layers_.push_back(layer_id);
}

void MemoryManager::unpin_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto pos = std::find(pinned_layers_.begin(), pinned_layers_.end(), layer_id);
    if (pos == pinned_layers_.end()) return;
    pinned_layers_.erase(pos);

    auto it = layer_ptrs_.find(layer_id);
    if (it != layer_ptrs_.end()) {
        for (void* ptr : it->second) {
            auto ait = ptr_to_alloc_idx_.find(ptr);
            if (ait != ptr_to_alloc_idx_.end()) {
                auto& alloc = weight_allocs_[ait->second];
                if (alloc.locked) {
                    do_unlock(ptr, alloc.size);
                    alloc.locked = false;
                    // Back to sequential hint — next time we read it, it's sequential
                    nx_madvise(ptr, alloc.size, MADV_SEQUENTIAL);
                }
            }
        }
    }
}

// ─── KV Cache ─────────────────────────────────────────────────────────────────

KVPage* MemoryManager::alloc_kv_page(uint32_t token_start,
                                      int n_layers, int n_kv_heads, int head_dim)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Each KV page holds page_tokens entries per layer per head
    // k: [page_tokens × n_kv_heads × head_dim] float16  (2 bytes each)
    // v: same
    // But KV is typically stored per-layer, so we allocate for all layers
    size_t entry_bytes = (size_t)kv_page_tokens_ * n_kv_heads * head_dim * 2; // float16
    size_t layer_bytes = entry_bytes * n_layers;
    size_t total_k = layer_bytes;
    size_t total_v = layer_bytes;
    size_t total   = total_k + total_v;

    // Check memory pressure before allocating
    size_t available = system_available_bytes();
    if (total * 2 > available) { // 2x safety margin
        // Try to free some KV pages
        handle_pressure();
        available = system_available_bytes();
        if (total > available) {
            fprintf(stderr, "[kvcache] Cannot allocate page: need %s, have %s\n",
                    fmt_bytes(total).c_str(), fmt_bytes(available).c_str());
            return nullptr;
        }
    }

    void* mem = nx_mmap(total);
    if (!mem || mem == MAP_FAILED) return nullptr;
    nx_madvise(mem, total, MADV_RANDOM);

    KVPage* page       = new KVPage{};
    page->k_data       = mem;
    page->v_data       = (uint8_t*)mem + total_k;
    page->token_start  = token_start;
    page->token_count  = 0;
    page->page_capacity = (uint32_t)kv_page_tokens_;
    page->locked       = false;
    page->on_gpu       = false;
    page->last_access_ns = now_ns();
    page->page_index   = (int)kv_pages_.size();

    kv_pages_.push_back(page);
    return page;
}

void MemoryManager::free_kv_page(KVPage* page) {
    if (!page) return;
    std::lock_guard<std::mutex> lock(mutex_);

    // Compute size to unmap
    // We store k and v contiguously; size tracked implicitly
    // For proper tracking we'd store size in KVPage — add in next iteration
    if (page->k_data) {
        // Find in list and remove
        auto it = std::find(kv_pages_.begin(), kv_pages_.end(), page);
        if (it != kv_pages_.end()) kv_pages_.erase(it);

        // We'd need to know the size — add size field to KVPage
        // For now: munmap(page->k_data, computed_size);
        // TODO: store total_bytes in KVPage struct
    }
    delete page;
}

void MemoryManager::clear_kv_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Free all KV pages — this is what happens on conversation reset
    // Much faster and more thorough than llama.cpp's memset approach
    for (auto* page : kv_pages_) {
        if (page->k_data) {
            // munmap — pages returned to OS immediately
            // (size must be tracked; see TODO above)
        }
        delete page;
    }
    kv_pages_.clear();
}

KVPage* MemoryManager::get_kv_page(uint32_t token_pos) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto* page : kv_pages_) {
        if (token_pos >= page->token_start &&
            token_pos < page->token_start + page->token_count) {
            page->last_access_ns = now_ns();
            return page;
        }
    }
    return nullptr;
}

// ─── Activation pool ─────────────────────────────────────────────────────────

void* MemoryManager::borrow_activation(size_t bytes) {
    // Simple bump allocator in the ring. Fast — no locking for the common case.
    // Assumes single-threaded forward pass (which is true for inference)
    bytes = align_up(bytes, 64); // cache line align
    size_t head = activation_head_.load(std::memory_order_relaxed);
    if (head + bytes > activation_size_) {
        // Wrap around
        activation_head_.store(0, std::memory_order_relaxed);
        head = 0;
    }
    activation_head_.store(head + bytes, std::memory_order_relaxed);
    return activation_pool_ + head;
}

void MemoryManager::return_activation(void* ptr) {
    // With ring buffer design, we don't free individual allocations.
    // The ring resets between forward passes.
    (void)ptr;
}

// ─── Pressure handling ────────────────────────────────────────────────────────

bool MemoryManager::check_pressure() {
    size_t available = system_available_bytes();
    size_t total     = system_total_bytes();
    if (total == 0) return false;
    float used_ratio = 1.0f - (float)available / total;
    return used_ratio > cfg_.pressure_threshold;
}

size_t MemoryManager::handle_pressure() {
    // Strategy:
    // 1. Advise kernel that unlocked weight pages can be dropped
    // 2. Free oldest KV pages (lose oldest context)
    // 3. Report what we freed

    size_t freed = 0;
    size_t before = system_available_bytes();

    // Step 1: Tell kernel it can reclaim unlocked weight pages
    for (auto& alloc : weight_allocs_) {
        if (!alloc.locked) {
            nx_madvise(alloc.ptr, alloc.size, MADV_DONTNEED);
            freed += alloc.size;
        }
    }

    // Step 2: Free oldest KV pages (up to half)
    if (!kv_pages_.empty()) {
        // Sort by last access time
        std::sort(kv_pages_.begin(), kv_pages_.end(),
            [](const KVPage* a, const KVPage* b) {
                return a->last_access_ns < b->last_access_ns;
            });

        size_t to_free = kv_pages_.size() / 2;
        for (size_t i = 0; i < to_free; ++i) {
            // munmap page data
            // freed += page_size;
            delete kv_pages_[i];
        }
        kv_pages_.erase(kv_pages_.begin(), kv_pages_.begin() + to_free);

        fprintf(stderr, "[memory] Pressure: freed %zu KV pages (oldest context lost)\n", to_free);
    }

    // Notify callbacks
    size_t after = system_available_bytes();
    freed = after - before;
    for (auto& cb : pressure_callbacks_) {
        cb(after);
    }

    return freed;
}

void MemoryManager::on_pressure(std::function<void(size_t)> cb) {
    pressure_callbacks_.push_back(std::move(cb));
}

// ─── Report ───────────────────────────────────────────────────────────────────

MemoryReport MemoryManager::report() const {
    std::lock_guard<std::mutex> lock(mutex_);

    MemoryReport r{};
    r.system_ram_available_bytes = system_available_bytes();
    r.system_ram_total_bytes     = system_total_bytes();
    r.kv_pages_allocated = kv_pages_.size();
    r.activation_pool_bytes = activation_size_;
    r.scratch_pool_bytes    = scratch_size_;

    for (const auto& alloc : weight_allocs_) {
        r.weights_total_bytes += alloc.size;
        if (alloc.locked) r.weights_locked_bytes += alloc.size;
        r.allocations.push_back(alloc);
    }

    for (const auto* page : kv_pages_) {
        // Approximate — page size tracking TODO
        r.kv_cache_total_bytes += page->page_capacity * 512; // rough
    }

    return r;
}

void MemoryManager::print_report() const {
    auto r = report();
    printf("\n── Memory Report ──────────────────────────────\n");
    printf("  System RAM: %s available / %s total\n",
           fmt_bytes(r.system_ram_available_bytes).c_str(),
           fmt_bytes(r.system_ram_total_bytes).c_str());
    printf("  Weights:    %s total, %s mlock'd\n",
           fmt_bytes(r.weights_total_bytes).c_str(),
           fmt_bytes(r.weights_locked_bytes).c_str());
    printf("  KV Cache:   %zu pages, ~%s\n",
           r.kv_pages_allocated,
           fmt_bytes(r.kv_cache_total_bytes).c_str());
    printf("  Activations: %s pool\n",
           fmt_bytes(r.activation_pool_bytes).c_str());
    printf("\n  Allocations (%zu):\n", r.allocations.size());
    for (const auto& a : r.allocations) {
        printf("    %-40s  %s  layer=%-3d  %s\n",
               a.name.c_str(),
               fmt_bytes(a.size).c_str(),
               a.layer_id,
               a.locked ? "[LOCKED]" : "");
    }
    printf("──────────────────────────────────────────────\n\n");
    fflush(stdout);
    fflush(stderr);
}

} // namespace nexus
