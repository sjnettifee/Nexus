/*
 * memguard.h -- keep a safety buffer of free RAM so a machine without swap does not freeze.
 *
 * When free memory (what the OS says is available without squeezing the page cache hard) would drop below the buffer, big allocations are refused
 * (nexus::memguard::OutOfMemory is thrown, or a cache growth simply fails) and generation is stopped with a clear message, instead of the whole
 * computer grinding to a halt.
 *
 *   NEXUS_MEM_GUARD=0        turn the guard off (run to the limit, at your own risk)
 *   NEXUS_MEM_BUFFER_MB=N    size of the buffer in MB (default 1024)
 */
#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__APPLE__)
  #include <mach/mach.h>
  #include <sys/sysctl.h>
#endif

namespace nexus { namespace memguard {

struct OutOfMemory : std::bad_alloc {
    std::string msg;
    explicit OutOfMemory(std::string m) : msg(std::move(m)) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

// RAM that can be handed out right now without starving the system, in bytes; -1 when the OS gives no answer.
inline long long available_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return (long long)ms.ullAvailPhys;
    return -1;
#elif defined(__APPLE__)
    vm_statistics64_data_t vm; mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) != KERN_SUCCESS) return -1;
    vm_size_t ps = 0; host_page_size(mach_host_self(), &ps);
    return (long long)(vm.free_count + vm.inactive_count + vm.speculative_count) * (long long)ps;
#else
    FILE* f = std::fopen("/proc/meminfo", "r"); if (!f) return -1;
    char line[256]; long long kb = -1;
    while (std::fgets(line, sizeof line, f)) if (std::strncmp(line, "MemAvailable:", 13) == 0) { kb = std::atoll(line + 13); break; }
    std::fclose(f);
    return kb < 0 ? -1 : kb * 1024;
#endif
}

struct State {
    std::atomic<bool> enabled{true};
    std::atomic<long long> buffer_bytes{1024LL << 20};
    std::atomic<bool> tripped{false};
    std::atomic<bool> inited{false};
    std::string reason; std::atomic<bool> reason_lock{false};
};
inline State& state() { static State s; return s; }

// Time (ms, steady clock) of the last "[progress]" log line from anywhere in the engine/server; lets the server add its own
// "still working" line only when nothing else has been printed for a while (e.g. while other computers do the work).
inline std::atomic<long long>& progress_stamp() { static std::atomic<long long> t{0}; return t; }
inline long long now_ms() { return (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

inline void init_from_env() {
    State& s = state();
    if (s.inited.exchange(true)) return;
    const char* g = std::getenv("NEXUS_MEM_GUARD"); if (g && g[0] == '0') s.enabled = false;
    const char* b = std::getenv("NEXUS_MEM_BUFFER_MB"); if (b && *b) { long long mb = std::atoll(b); if (mb >= 0 && mb < (1LL << 22)) s.buffer_bytes = mb << 20; }
}
inline bool enabled() { init_from_env(); return state().enabled.load(); }
inline long long buffer_mb() { init_from_env(); return state().buffer_bytes.load() >> 20; }
inline bool tripped() { return state().tripped.load(); }
inline std::string reason() { State& s = state(); while (s.reason_lock.exchange(true)) {} std::string r = s.reason; s.reason_lock = false; return r; }
inline void clear() { State& s = state(); s.tripped = false; while (s.reason_lock.exchange(true)) {} s.reason.clear(); s.reason_lock = false; }

// Would taking `extra_bytes` more leave at least the buffer free?  On "no" the guard is marked tripped and a message is kept for the UI.
inline bool allow(size_t extra_bytes, const char* what) {
    init_from_env(); State& s = state();
    if (!s.enabled.load()) return true;
    const long long av = available_bytes();
    if (av < 0) return true;                                           // unknown: do not block anything
    const long long need = (long long)extra_bytes + s.buffer_bytes.load();
    if (av >= need) return true;
    char msg[320];
    std::snprintf(msg, sizeof msg, "Stopped to protect your computer: only %lld MB of memory is free, and %s needs %lld MB more while keeping %lld MB spare "
                  "(memory safety buffer). Free some memory, use a smaller context, or change/turn off the buffer in the launcher (Advanced).",
                  av >> 20, what ? what : "this step", (long long)(extra_bytes >> 20), s.buffer_bytes.load() >> 20);
    if (!s.tripped.exchange(true)) {
        while (s.reason_lock.exchange(true)) {} s.reason = msg; s.reason_lock = false;
        std::fprintf(stderr, "[memguard] %s\n", msg); std::fflush(stderr);
    }
    return false;
}
// Cheap periodic check (no extra allocation): is free memory still above the buffer?
inline bool still_ok(const char* what) { return allow(0, what); }
// Same as allow(), but throws OutOfMemory (caught where a reply or prompt is processed).
inline void require(size_t extra_bytes, const char* what) { if (!allow(extra_bytes, what)) throw OutOfMemory(reason()); }

}} // namespace nexus::memguard
