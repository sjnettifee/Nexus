/*
 * crashlog.h -- if the program crashes (segmentation fault, abort, illegal instruction ...), print WHERE to the log before it dies,
 * so a crash on another computer is no longer just "exited with code 139".  Each line is  module + offset  (resolve with
 *    addr2line -f -C -e <the exact server binary> 0x<offset>      (Linux)      or the .pdb / "dumpbin" on Windows).
 * Nothing here runs unless a crash happens.
 */
#pragma once
#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__linux__) || defined(__APPLE__)
  #include <csignal>
  #include <execinfo.h>
  #include <unistd.h>
#endif

namespace nexus { namespace crashlog {

#if defined(_WIN32)
inline LONG WINAPI filter(EXCEPTION_POINTERS* ep) {
    std::fprintf(stderr, "\n[crash] unhandled exception 0x%08lx at %p\n", (unsigned long)ep->ExceptionRecord->ExceptionCode, ep->ExceptionRecord->ExceptionAddress);
    void* frames[48]; USHORT n = CaptureStackBackTrace(0, 48, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE mod = nullptr; char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)frames[i], &mod) && mod) GetModuleFileNameA(mod, name, sizeof name);
        std::fprintf(stderr, "[crash]   #%u %s +0x%llx\n", (unsigned)i, name, (unsigned long long)((char*)frames[i] - (char*)mod));
    }
    std::fprintf(stderr, "[crash] send these lines (and the lines above them) with the exact program version.\n"); std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
inline void install() { SetUnhandledExceptionFilter(filter); }
#elif defined(__linux__) || defined(__APPLE__)
inline void handler(int sig) {
    static volatile sig_atomic_t busy = 0;
    if (!busy) {
        busy = 1;
        const char* nm = sig == SIGSEGV ? "SIGSEGV (invalid memory access)" : sig == SIGBUS ? "SIGBUS (bad memory access)" : sig == SIGILL ? "SIGILL (illegal instruction)" :
                         sig == SIGFPE ? "SIGFPE (arithmetic error)" : sig == SIGABRT ? "SIGABRT (abort)" : "signal";
        char head[160]; int k = std::snprintf(head, sizeof head, "\n[crash] %s - the program is about to stop. Where it was:\n", nm);
        if (k > 0) { ssize_t r = write(2, head, (size_t)k); (void)r; }
        void* frames[48]; int n = backtrace(frames, 48);
        backtrace_symbols_fd(frames, n, 2);
        const char* tail = "[crash] send these lines (and the lines above them) with the exact program version.\n";
        ssize_t r2 = write(2, tail, std::strlen(tail)); (void)r2;
    }
    std::signal(sig, SIG_DFL); raise(sig);                       // keep the normal exit code (139 for a segfault)
}
inline void install() {
    static char altstack[64 * 1024];
    stack_t ss; ss.ss_sp = altstack; ss.ss_size = sizeof altstack; ss.ss_flags = 0; sigaltstack(&ss, nullptr);   // survives a stack overflow
    struct sigaction sa; std::memset(&sa, 0, sizeof sa); sa.sa_handler = handler; sa.sa_flags = SA_ONSTACK; sigemptyset(&sa.sa_mask);
    for (int s : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT}) sigaction(s, &sa, nullptr);
}
#else
inline void install() {}
#endif

}} // namespace nexus::crashlog
