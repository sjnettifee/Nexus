// nexus_dispatch.cpp -- the program users actually start.  Installed as `nexus_server`.
//
// The engine is built once per CPU tier (same source, different -march):
//     nexus_server_v2   libnexus_v2   x86-64-v2  (SSE4.2; every x86 CPU since ~2009)
//     nexus_server_v3   libnexus_v3   x86-64-v3  (AVX2+FMA+BMI2+F16C; Haswell 2013+, Zen)
//     nexus_server_v4   libnexus_v4   x86-64-v4  (AVX-512)             [optional]
//     nexus_server_arm64 ...          Apple Silicon / ARM Linux        [when built]
// This tiny program is compiled for the lowest baseline, so it starts on ANY CPU.  It
// reads the CPU's capabilities once, picks the best tier whose files are present next to
// it, and hands over: execv() on Linux/macOS (same PID, same pipes), and a
// kill-with-parent child process on Windows.  Everything after the program name is passed
// through unchanged, so the launcher / scripts keep calling plain `nexus_server`.
//
// Extra switches (consumed here, never forwarded):
//     --which-tier        print the tier that would be used and exit
//     --tier vN           force a tier (testing; refuses a tier this CPU cannot run)
//     env NEXUS_CPU_TIER  same as --tier
//
// Build this file WITHOUT any -march flag.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/stat.h>
#  include <limits.h>
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#  endif
#endif

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64)) && !defined(_MSC_VER)
#  include <cpuid.h>
#  define NX_X86 1
#else
#  define NX_X86 0
#endif

// ─── CPU detection ────────────────────────────────────────────────────────────

#if NX_X86
static void cpuid(unsigned leaf, unsigned sub, unsigned& a, unsigned& b, unsigned& c, unsigned& d) {
    a = b = c = d = 0;
    __cpuid_count(leaf, sub, a, b, c, d);
}
static unsigned long long xgetbv0() {
    unsigned eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}

struct X86Caps { bool v2 = false, v3 = false, v4 = false; };

static X86Caps detect_x86() {
    X86Caps r;
    unsigned a, b, c, d;
    unsigned maxleaf, ebx0, ecx0, edx0;
    cpuid(0, 0, maxleaf, ebx0, ecx0, edx0);
    if (maxleaf < 1) return r;
    cpuid(1, 0, a, b, c, d);
    const bool sse3 = c & (1u << 0), ssse3 = c & (1u << 9), sse41 = c & (1u << 19),
               sse42 = c & (1u << 20), popcnt = c & (1u << 23), cx16 = c & (1u << 13);
    r.v2 = sse3 && ssse3 && sse41 && sse42 && popcnt && cx16;
    if (!r.v2) return r;

    const bool osxsave = c & (1u << 27), avx = c & (1u << 28), fma = c & (1u << 12),
               f16c = c & (1u << 29), movbe = c & (1u << 22);
    bool ymm_ok = false, zmm_ok = false;
    if (osxsave && avx) {                           // the OS must save YMM (and ZMM) state
        unsigned long long xcr0 = xgetbv0();
        ymm_ok = (xcr0 & 0x6) == 0x6;
        zmm_ok = ymm_ok && (xcr0 & 0xE0) == 0xE0;
    }
    bool avx2 = false, bmi1 = false, bmi2 = false, avx512f = false, avx512bw = false,
         avx512dq = false, avx512vl = false;
    if (maxleaf >= 7) {
        cpuid(7, 0, a, b, c, d);
        avx2 = b & (1u << 5); bmi1 = b & (1u << 3); bmi2 = b & (1u << 8);
        avx512f = b & (1u << 16); avx512dq = b & (1u << 17);
        avx512bw = b & (1u << 30); avx512vl = b & (1u << 31);
    }
    bool lzcnt = false;
    cpuid(0x80000000u, 0, a, b, c, d);
    if (a >= 0x80000001u) { cpuid(0x80000001u, 0, a, b, c, d); lzcnt = c & (1u << 5); }

    r.v3 = ymm_ok && avx && avx2 && fma && f16c && movbe && bmi1 && bmi2 && lzcnt;
    r.v4 = r.v3 && zmm_ok && avx512f && avx512bw && avx512dq && avx512vl;
    return r;
}
#endif

// Tiers best-first for this machine.
static std::vector<std::string> tiers_for_this_cpu() {
    std::vector<std::string> t;
#if NX_X86
    X86Caps c = detect_x86();
    if (c.v4) t.push_back("v4");
    if (c.v3) t.push_back("v3");
    if (c.v2) t.push_back("v2");
#elif defined(__aarch64__) || defined(_M_ARM64)
    t.push_back("arm64");
#endif
    return t;
}

// ─── filesystem helpers ───────────────────────────────────────────────────────

static std::string self_dir_and_path(std::string& self) {
#if defined(_WIN32)
    char b[MAX_PATH * 2];
    DWORD n = GetModuleFileNameA(nullptr, b, sizeof(b));
    self.assign(b, n);
    size_t k = self.find_last_of("\\/");
    return k == std::string::npos ? "." : self.substr(0, k);
#elif defined(__APPLE__)
    char b[PATH_MAX]; uint32_t n = sizeof(b);
    if (_NSGetExecutablePath(b, &n) != 0) { self = "."; return "."; }
    char r[PATH_MAX];
    self = realpath(b, r) ? r : b;
    size_t k = self.find_last_of('/');
    return k == std::string::npos ? "." : self.substr(0, k);
#else
    char b[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", b, sizeof(b) - 1);
    if (n <= 0) { self = "."; return "."; }
    b[n] = 0; self = b;
    size_t k = self.find_last_of('/');
    return k == std::string::npos ? "." : self.substr(0, k);
#endif
}

static bool file_exists(const std::string& p) {
#if defined(_WIN32)
    DWORD a = GetFileAttributesA(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode) && access(p.c_str(), X_OK) == 0;
#endif
}

static std::string tier_exe(const std::string& dir, const std::string& tier) {
#if defined(_WIN32)
    return dir + "\\nexus_server_" + tier + ".exe";
#else
    return dir + "/nexus_server_" + tier;
#endif
}

#if defined(_WIN32)
// Inverse of CommandLineToArgvW.
static void append_quoted(std::string& cl, const std::string& a) {
    if (!cl.empty()) cl += ' ';
    if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) { cl += a; return; }
    cl += '"';
    for (size_t i = 0;; ++i) {
        size_t bs = 0;
        while (i < a.size() && a[i] == '\\') { ++i; ++bs; }
        if (i == a.size()) { cl.append(bs * 2, '\\'); break; }
        if (a[i] == '"') { cl.append(bs * 2 + 1, '\\'); cl += '"'; }
        else             { cl.append(bs, '\\'); cl += a[i]; }
    }
    cl += '"';
}
#endif

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string forced;
    if (const char* e = std::getenv("NEXUS_CPU_TIER")) forced = e;
    bool which_only = false;
    std::vector<std::string> fwd;                       // arguments passed to the real server
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--which-tier") which_only = true;
        else if (a == "--tier" && i + 1 < argc) forced = argv[++i];
        else fwd.push_back(a);
    }

    std::string self, dir = self_dir_and_path(self);
    std::vector<std::string> usable = tiers_for_this_cpu();
    if (usable.empty()) {
        fprintf(stderr, "[dispatch] This CPU is too old (needs SSE4.2/POPCNT on x86, or ARM64).\n");
        return 2;
    }
    std::vector<std::string> order = usable;
    if (!forced.empty()) {
        size_t k = 0;
        while (k < usable.size() && usable[k] != forced) ++k;
        if (k == usable.size()) {
            fprintf(stderr, "[dispatch] Tier '%s' cannot run on this CPU (usable:", forced.c_str());
            for (auto& u : usable) fprintf(stderr, " %s", u.c_str());
            fprintf(stderr, ")\n");
            return 2;
        }
        order.assign(usable.begin() + k, usable.end());   // forced tier, then its fallbacks
    }

    std::string chosen, exe;
    for (auto& t : order) {
        std::string p = tier_exe(dir, t);
        if (file_exists(p)) { chosen = t; exe = p; break; }
    }
    if (exe.empty()) {
        fprintf(stderr, "[dispatch] No engine build found next to %s for this CPU.\n[dispatch] Looked for:", self.c_str());
        for (auto& t : order) fprintf(stderr, " %s", tier_exe(dir, t).c_str());
        fprintf(stderr, "\n");
        return 3;
    }
    if (which_only) { printf("%s\n", chosen.c_str()); return 0; }

    fprintf(stderr, "[dispatch] CPU tier %s -> %s\n", chosen.c_str(), exe.c_str());
    fflush(stderr);

#if defined(_WIN32)
    std::string cl;
    append_quoted(cl, exe);
    for (auto& a : fwd) append_quoted(cl, a);

    // The child must die if this process is killed (the launcher kills the PID it started).
    HANDLE job = CreateJobObjectA(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
    }
    STARTUPINFOA si{}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cl.begin(), cl.end()); buf.push_back(0);
    if (!CreateProcessA(exe.c_str(), buf.data(), nullptr, nullptr, TRUE,
                        CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
        fprintf(stderr, "[dispatch] CreateProcess failed (%lu) for %s\n", GetLastError(), exe.c_str());
        return 4;
    }
    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return (int)code;
#else
    std::vector<char*> av;
    av.push_back(const_cast<char*>(exe.c_str()));
    for (auto& a : fwd) av.push_back(const_cast<char*>(a.c_str()));
    av.push_back(nullptr);
    setenv("NEXUS_CPU_TIER_USED", chosen.c_str(), 1);
    execv(exe.c_str(), av.data());
    perror("[dispatch] execv");
    return 4;
#endif
}
