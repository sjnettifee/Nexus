// core.h -- everything the launcher does that is not drawing: JSON, files, model scan, process
// control, hardware probe.  No GUI, no HTTP.  Included by exactly one translation unit (app.cpp).
#pragma once
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <shellapi.h>
   typedef SOCKET sock_t;
   static void sock_close(sock_t s) { closesocket(s); }
#  define NX_INVALID_SOCK INVALID_SOCKET
#else
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <dirent.h>
#  include <netinet/in.h>
#  include <netdb.h>
#  include <signal.h>
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <sys/wait.h>
#  include <unistd.h>
#  include <limits.h>
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#  endif
   typedef int sock_t;
   static void sock_close(sock_t s) { ::close(s); }
#  define NX_INVALID_SOCK (-1)
#endif

// ───────────────────────────── tiny JSON ─────────────────────────────────────

struct JV {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false; double n = 0; std::string s;
    std::vector<JV> a; std::vector<std::pair<std::string, JV>> o;
    const JV* get(const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
struct JP {
    const std::string& s; size_t i = 0; int depth = 0; bool ok = true;
    explicit JP(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\r'||s[i]=='\t')) ++i; }
    static void put_utf8(std::string& o, unsigned cp) {
        if (cp < 0x80) o += (char)cp;
        else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
        else { o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 63)); o += (char)(0x80 | ((cp >> 6) & 63)); o += (char)(0x80 | (cp & 63)); }
    }
    std::string str() {
        std::string r; ++i;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') { r += c; continue; }
            if (i >= s.size()) { ok = false; break; }
            char e = s[i++];
            switch (e) {
                case 'n': r += '\n'; break; case 't': r += '\t'; break; case 'r': r += '\r'; break;
                case 'b': r += '\b'; break; case 'f': r += '\f'; break;
                case 'u': {
                    if (i + 4 > s.size()) { ok = false; return r; }
                    unsigned cp = (unsigned)strtoul(s.substr(i, 4).c_str(), nullptr, 16); i += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i+1] == 'u') {
                        unsigned lo = (unsigned)strtoul(s.substr(i + 2, 4).c_str(), nullptr, 16);
                        if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); i += 6; }
                    }
                    put_utf8(r, cp); break;
                }
                default: r += e;
            }
        }
        if (i >= s.size()) ok = false; else ++i;
        return r;
    }
    JV val() {
        JV v; ws();
        if (i >= s.size() || ++depth > 64) { ok = false; return v; }
        char c = s[i];
        if (c == '{') {
            v.t = JV::Obj; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; --depth; return v; }
            while (ok) {
                ws(); if (i >= s.size() || s[i] != '"') { ok = false; break; }
                std::string k = str(); ws();
                if (i >= s.size() || s[i] != ':') { ok = false; break; }
                ++i; v.o.emplace_back(k, val()); ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; break; }
                ok = false;
            }
        } else if (c == '[') {
            v.t = JV::Arr; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; --depth; return v; }
            while (ok) {
                v.a.push_back(val()); ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; break; }
                ok = false;
            }
        } else if (c == '"') { v.t = JV::Str; v.s = str(); }
        else if (s.compare(i, 4, "true") == 0)  { v.t = JV::Bool; v.b = true;  i += 4; }
        else if (s.compare(i, 5, "false") == 0) { v.t = JV::Bool; v.b = false; i += 5; }
        else if (s.compare(i, 4, "null") == 0)  { i += 4; }
        else {
            char* end = nullptr; v.n = strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) ok = false; else { v.t = JV::Num; i = (size_t)(end - s.c_str()); }
        }
        --depth; return v;
    }
};
static bool json_parse(const std::string& s, JV& out) {
    JP p(s); out = p.val(); p.ws();
    return p.ok && p.i == s.size();
}
static std::string jesc(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

// ───────────────────────────── paths / files ─────────────────────────────────

static std::string g_dir;          // directory of this program
static std::string g_cfg_path;     // nexus_launcher.json

#if defined(_WIN32)
static const char SEP = '\\';
#else
static const char SEP = '/';
#endif

static std::string self_dir() {
#if defined(_WIN32)
    char b[MAX_PATH * 2]; DWORD n = GetModuleFileNameA(nullptr, b, sizeof b);
    std::string p(b, n); size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? "." : p.substr(0, k);
#elif defined(__APPLE__)
    char b[PATH_MAX]; uint32_t n = sizeof b;
    if (_NSGetExecutablePath(b, &n) != 0) return ".";
    char r[PATH_MAX]; std::string p = realpath(b, r) ? r : b; size_t k = p.find_last_of('/');
    return k == std::string::npos ? "." : p.substr(0, k);
#else
    char b[PATH_MAX]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n <= 0) return ".";
    b[n] = 0; std::string p = b; size_t k = p.find_last_of('/');
    return k == std::string::npos ? "." : p.substr(0, k);
#endif
}
static std::string join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char l = a.back();
    return (l == '/' || l == '\\') ? a + b : a + SEP + b;
}
static bool is_dir(const std::string& p) {
#if defined(_WIN32)
    DWORD a = GetFileAttributesA(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st; return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}
static bool is_file(const std::string& p) {
#if defined(_WIN32)
    DWORD a = GetFileAttributesA(p.c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st; return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}
static bool read_file(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf(); out = ss.str(); return true;
}
static bool write_file_atomic(const std::string& p, const std::string& data) {
    std::string tmp = p + ".tmp";
    { std::ofstream f(tmp, std::ios::binary | std::ios::trunc); if (!f) return false; f << data; if (!f) return false; }
#if defined(_WIN32)
    return MoveFileExA(tmp.c_str(), p.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(tmp.c_str(), p.c_str()) == 0;
#endif
}

static std::string lower(std::string s) { for (auto& c : s) c = (char)tolower((unsigned char)c); return s; }
struct FileEnt { std::string name, dir; unsigned long long size; };
static void list_dir(const std::string& d, std::vector<std::pair<std::string, bool>>& out) {
#if defined(_WIN32)
    WIN32_FIND_DATAA fd; HANDLE h = FindFirstFileA(join(d, "*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do { std::string n = fd.cFileName; if (n != "." && n != "..") out.emplace_back(n, (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0); }
    while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* dp = opendir(d.c_str()); if (!dp) return;
    while (dirent* e = readdir(dp)) {
        std::string n = e->d_name; if (n == "." || n == "..") continue;
        out.emplace_back(n, is_dir(join(d, n)));
    }
    closedir(dp);
#endif
}
static bool ends_ci(const std::string& s, const char* suf) {
    size_t n = strlen(suf); if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) if (tolower((unsigned char)s[s.size() - n + i]) != suf[i]) return false;
    return true;
}
static unsigned long long file_size(const std::string& p) {
#if defined(_WIN32)
    WIN32_FILE_ATTRIBUTE_DATA d; if (!GetFileAttributesExA(p.c_str(), GetFileExInfoStandard, &d)) return 0;
    return ((unsigned long long)d.nFileSizeHigh << 32) | d.nFileSizeLow;
#else
    struct stat st; return stat(p.c_str(), &st) == 0 ? (unsigned long long)st.st_size : 0;
#endif
}
static std::vector<FileEnt> g_unpacked;                            // *.nxmodel FOLDERS found while scanning: the engine opens .nxm files, so they must be packed first
static void scan_models(const std::string& dir, int depth, std::vector<FileEnt>& out, bool gguf_only = false) {
    std::vector<std::pair<std::string, bool>> es; list_dir(dir, es);
    for (auto& e : es) {
        std::string full = join(dir, e.first);
        if (e.second && ends_ci(e.first, ".nxmodel")) { if (!gguf_only) g_unpacked.push_back({e.first, dir, 0}); continue; }
        if (e.second) { if (depth > 0 && !(gguf_only == false && lower(e.first) == "loras")) scan_models(full, depth - 1, out, gguf_only); }
        else if (ends_ci(e.first, ".gguf") || (!gguf_only && ends_ci(e.first, ".nxm"))) out.push_back({e.first, dir, file_size(full)});
    }
}

// ───────────────────────────── child process ─────────────────────────────────

struct Child {
    std::mutex mu;
    std::deque<std::string> lines; size_t first_idx = 0;       // absolute index of lines.front()
    std::string partial;
    std::atomic<int> state{0};                                  // 0 idle, 1 running, 2 exited
    std::atomic<int> exit_code{0};
    std::atomic<int> port{0};
    std::string model;
    std::thread reader;
#if defined(_WIN32)
    HANDLE proc = nullptr, job = nullptr, rd = nullptr;
#else
    pid_t pid = -1; int rd = -1;
#endif
};

// Plain-words reason for a server that stopped by itself, from the exit code and the last lines of its log ("" = nothing useful to add).
static std::string explain_exit(Child& c, int code) {
    std::string tail;
    { std::lock_guard<std::mutex> lk(c.mu); size_t n = c.lines.size(), from = n > 400 ? n - 400 : 0; for (size_t i = from; i < n; ++i) { tail += c.lines[i]; tail += '\n'; } }
    std::string low = tail; for (auto& ch : low) ch = (char)tolower((unsigned char)ch);
    auto has = [&](const char* w) { return low.find(w) != std::string::npos; };
    const std::string mem = "The computer ran out of memory. Close other programs, choose a smaller model or a smaller context, or put fewer layers in RAM. "
                            "(Advanced > memory safety buffer protects against this while a reply is running; it cannot help while the model is still loading.)";
    if (code == 0) return "";
    if (code == 143 || code == 130 || code == 129 || code == 15) return "";                 // stopped on purpose
    if (has("address already in use") || has("failed to bind") || has("bind failed")) return "Another program is already using this server port. Give the model a different port (next to its name) and start again.";
    if (has("bad_alloc") || has("out of memory") || has("cannot allocate memory") || code == 137) return mem;
    if (has("invalid magic") || has("not a gguf") || has("failed to open model") || has("cannot open model") || has("unsupported gguf") || has("failed to parse")) return "The model file could not be read. It may be incomplete (re-download it) or in a format this program does not support yet.";
    if (has("vkcreateinstance") || has("vkcreatedevice") || has("no vulkan") || has("vulkan") && (has("failed") || has("error"))) return "The graphics driver would not start. Update your graphics driver, or set \"Graphics card 1\" to \"CPU only\" and start again.";
    if (code == 127) return "The server program could not be started. Make sure the server file sits next to this program and is allowed to run.";
    if (code == 139 || code == 134 || code == -1073741819 || code == -1073740791) return "The server crashed. Copy the log (Copy all) and send it with the model name and your graphics card; as a quick workaround choose \"CPU only\" or a smaller model.";
    return "";
}

static const int kMaxChildren = 12;               // slots 0..7 are models, 8 is the donor/worker
static Child g_children[kMaxChildren];
static const int kMaxSlots = 8, kWorkerChild = 8;

static void log_line(Child& c, const std::string& l) {
    std::lock_guard<std::mutex> lk(c.mu);
    c.lines.push_back(l);
    if (c.lines.size() > 6000) { c.lines.pop_front(); ++c.first_idx; }
}
static void feed(Child& c, const char* d, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        char ch = d[i];
        if (ch == '\n') {
            std::string l; l.swap(c.partial);
            if (!l.empty() && l.back() == '\r') l.pop_back();
            log_line(c, l);
        } else if (ch == '\r') { /* progress bars: ignore bare CR */ }
        else { c.partial += ch; if (c.partial.size() > 8192) { log_line(c, c.partial); c.partial.clear(); } }
    }
}

// The server may sit next to the launcher or in a folder beside it (USB layout: launcher on top, server + engine libraries in "server").
// The first place that really contains it wins; the server program then finds its engine libraries in its OWN folder.
static std::vector<std::string> server_dirs() {
    return { g_dir, join(g_dir, "server"), join(g_dir, "bin"), join(g_dir, "engine"), join(join(g_dir, ".."), "server") };
}
static std::string server_exe() {
#if defined(_WIN32)
    const char* nm = "nexus_server.exe";
#else
    const char* nm = "nexus_server";
#endif
    for (auto& d : server_dirs()) { std::string p = join(d, nm); if (is_file(p)) return p; }
    return join(g_dir, nm);                                      // not found anywhere: the usual "not found next to the launcher" message names this path
}

static bool child_alive(Child& c) { return c.state.load() == 1; }

static void child_join_reader(Child& c) { if (c.reader.joinable()) c.reader.join(); }

static bool child_start(Child& c, const std::vector<std::string>& args, const std::vector<std::pair<std::string, std::string>>& env,
                        std::string& err) {
    if (child_alive(c)) { err = "already running"; return false; }
    child_join_reader(c);
    std::string exe = server_exe();
    if (!is_file(exe)) { err = "The server program (nexus_server) was not found. Put it next to this launcher or in a folder named \"server\" beside it. Looked for: " + exe; return false; }
    {
        std::lock_guard<std::mutex> lk(c.mu);
        c.partial.clear();
        c.lines.push_back("[launcher] starting: " + exe);
        std::string a; for (auto& x : args) a += " " + x;
        c.lines.push_back("[launcher] args:" + a);
        for (auto& e : env) c.lines.push_back("[launcher] env " + e.first + "=" + e.second);
    }
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE r = nullptr, w = nullptr;
    if (!CreatePipe(&r, &w, &sa, 0)) { err = "CreatePipe failed"; return false; }
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    std::string cl = "\"" + exe + "\"";
    for (auto& a : args) {                       // inverse of CommandLineToArgvW
        cl += ' ';
        if (!a.empty() && a.find_first_of(" \t\n\v\"") == std::string::npos) { cl += a; continue; }
        cl += '"';
        for (size_t i = 0;; ++i) {
            size_t bs = 0; while (i < a.size() && a[i] == '\\') { ++i; ++bs; }
            if (i == a.size()) { cl.append(bs * 2, '\\'); break; }
            if (a[i] == '"') { cl.append(bs * 2 + 1, '\\'); cl += '"'; } else { cl.append(bs, '\\'); cl += a[i]; }
        }
        cl += '"';
    }
    std::string envblock;
    if (char* cur = GetEnvironmentStringsA()) {
        for (char* p = cur; *p; p += strlen(p) + 1) {
            std::string kv = p; std::string k = kv.substr(0, kv.find('='));
            bool over = false; for (auto& e : env) if (_stricmp(e.first.c_str(), k.c_str()) == 0) over = true;
            if (!over) { envblock += kv; envblock += '\0'; }
        }
        FreeEnvironmentStringsA(cur);
    }
    for (auto& e : env) { envblock += e.first + "=" + e.second; envblock += '\0'; }
    envblock += '\0';
    c.job = CreateJobObjectA(nullptr, nullptr);
    if (c.job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{}; li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(c.job, JobObjectExtendedLimitInformation, &li, sizeof li);
    }
    STARTUPINFOA si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = w; si.hStdError = w; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<char> buf(cl.begin(), cl.end()); buf.push_back(0);
    if (!CreateProcessA(exe.c_str(), buf.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED | CREATE_NO_WINDOW,
                        envblock.data(), g_dir.c_str(), &si, &pi)) {
        err = "CreateProcess failed (" + std::to_string(GetLastError()) + ")";
        CloseHandle(r); CloseHandle(w); return false;
    }
    if (c.job) AssignProcessToJobObject(c.job, pi.hProcess);
    ResumeThread(pi.hThread); CloseHandle(pi.hThread); CloseHandle(w);
    c.proc = pi.hProcess; c.rd = r;
    c.state = 1;
    c.reader = std::thread([&c] {
        char b[4096]; DWORD n;
        while (ReadFile(c.rd, b, sizeof b, &n, nullptr) && n) feed(c, b, n);
        DWORD code = 0; WaitForSingleObject(c.proc, INFINITE); GetExitCodeProcess(c.proc, &code);
        CloseHandle(c.rd); CloseHandle(c.proc); c.rd = nullptr; c.proc = nullptr;
        if (c.job) { CloseHandle(c.job); c.job = nullptr; }
        if (!c.partial.empty()) { log_line(c, c.partial); c.partial.clear(); }
        c.exit_code = (int)code; c.state = 2;
        log_line(c, "[launcher] server exited with code " + std::to_string((int)code));
        { std::string why = explain_exit(c, (int)code); if (!why.empty()) log_line(c, "[launcher] What probably happened: " + why); }
    });
#else
    int p[2]; if (pipe(p) != 0) { err = "pipe failed"; return false; }
    pid_t pid = fork();
    if (pid < 0) { err = "fork failed"; close(p[0]); close(p[1]); return false; }
    if (pid == 0) {
        setpgid(0, 0);
        dup2(p[1], 1); dup2(p[1], 2); close(p[0]); close(p[1]);
        if (chdir(g_dir.c_str()) != 0) {}
        for (auto& e : env) setenv(e.first.c_str(), e.second.c_str(), 1);
        std::vector<char*> av; av.push_back(const_cast<char*>(exe.c_str()));
        for (auto& a : args) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        execv(exe.c_str(), av.data());
        fprintf(stderr, "[launcher] exec failed: %s\n", strerror(errno));
        _exit(127);
    }
    close(p[1]);
    c.pid = pid; c.rd = p[0]; c.state = 1;
    c.reader = std::thread([&c] {
        char b[4096]; ssize_t n;
        while ((n = read(c.rd, b, sizeof b)) > 0) feed(c, b, (size_t)n);
        close(c.rd); c.rd = -1;
        int st = 0; waitpid(c.pid, &st, 0);
        if (!c.partial.empty()) { log_line(c, c.partial); c.partial.clear(); }
        int code = WIFEXITED(st) ? WEXITSTATUS(st) : (WIFSIGNALED(st) ? 128 + WTERMSIG(st) : -1);
        c.exit_code = code; c.state = 2;
        log_line(c, "[launcher] server exited with code " + std::to_string(code));
        { std::string why = explain_exit(c, code); if (!why.empty()) log_line(c, "[launcher] What probably happened: " + why); }
    });
#endif
    return true;
}

#if !defined(_WIN32)
// Ctrl-C / terminate / hangup on the launcher also takes the server (its own process group) down.
static void on_term(int) {
    for (int i = 0; i < kMaxChildren; ++i) if (g_children[i].pid > 0) kill(-g_children[i].pid, SIGTERM);
    _exit(0);
}
#endif

static void child_stop(Child& c) {
    if (!child_alive(c)) { child_join_reader(c); c.state = 0; return; }
    log_line(c, "[launcher] stopping server...");
#if defined(_WIN32)
    if (c.proc) TerminateProcess(c.proc, 1);
#else
    if (c.pid > 0) {
        kill(-c.pid, SIGTERM);
        for (int i = 0; i < 40 && child_alive(c); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (child_alive(c)) kill(-c.pid, SIGKILL);
    }
#endif
    child_join_reader(c);
    c.state = 0;
}

// ───────────────────────────── probe ─────────────────────────────────────────

// Runs `nexus_server --probe` (ours, so no shell involved) and returns its stdout.
static std::string run_probe() {
    std::string exe = server_exe();
    if (!is_file(exe)) return "{\"error\":\"nexus_server not found next to the launcher\"}";
    std::string out;
#if defined(_WIN32)
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE}; HANDLE r, w;
    if (!CreatePipe(&r, &w, &sa, 0)) return "{\"error\":\"pipe\"}";
    SetHandleInformation(r, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = w; si.hStdError = GetStdHandle(STD_ERROR_HANDLE); si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{}; std::string cl = "\"" + exe + "\" --probe"; std::vector<char> b(cl.begin(), cl.end()); b.push_back(0);
    if (!CreateProcessA(exe.c_str(), b.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, g_dir.c_str(), &si, &pi)) {
        CloseHandle(r); CloseHandle(w); return "{\"error\":\"could not run nexus_server --probe\"}";
    }
    CloseHandle(w);
    char buf[4096]; DWORD n; while (ReadFile(r, buf, sizeof buf, &n, nullptr) && n) out.append(buf, n);
    WaitForSingleObject(pi.hProcess, 20000); CloseHandle(pi.hProcess); CloseHandle(pi.hThread); CloseHandle(r);
#else
    int p[2]; if (pipe(p) != 0) return "{\"error\":\"pipe\"}";
    pid_t pid = fork();
    if (pid == 0) {
        dup2(p[1], 1); close(p[0]); close(p[1]);
        if (chdir(g_dir.c_str()) != 0) {}
        execl(exe.c_str(), exe.c_str(), "--probe", (char*)nullptr); _exit(127);
    }
    close(p[1]);
    if (pid < 0) { close(p[0]); return "{\"error\":\"fork\"}"; }
    char buf[4096]; ssize_t n; while ((n = read(p[0], buf, sizeof buf)) > 0) out.append(buf, (size_t)n);
    close(p[0]); int st; waitpid(pid, &st, 0);
#endif
    // keep only the JSON object (the dispatcher may print a "[dispatch]" line to stderr, not stdout)
    size_t a = out.find('{'), z = out.rfind('}');
    if (a == std::string::npos || z == std::string::npos || z < a) return "{\"error\":\"probe produced no JSON\"}";
    return out.substr(a, z - a + 1);
}

static bool port_open(int port) {
    sock_t s = socket(AF_INET, SOCK_STREAM, 0); if (s == NX_INVALID_SOCK) return false;
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
#if defined(_WIN32)
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
    connect(s, (sockaddr*)&a, sizeof a);
    fd_set wf; FD_ZERO(&wf); FD_SET(s, &wf); timeval tv{0, 200000};
    bool ok = select(0, nullptr, &wf, nullptr, &tv) > 0;
    if (ok) { int err = 0; int l = sizeof err; getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&err, &l); ok = err == 0; }
#else
    int fl = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, fl | O_NONBLOCK);
    connect(s, (sockaddr*)&a, sizeof a);
    fd_set wf; FD_ZERO(&wf); FD_SET(s, &wf); timeval tv{0, 200000};
    bool ok = select(s + 1, nullptr, &wf, nullptr, &tv) > 0;
    if (ok) { int err = 0; socklen_t l = sizeof err; getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &l); ok = err == 0; }
#endif
    sock_close(s); return ok;
}




// ───────────────────────────── additions for the native app ─────────────────────────────────

// JSON writer (pretty, 2 spaces) -- used for nexus_launcher.json
static void jdump(const JV& v, std::string& o, int ind = 0) {
    auto pad = [&](int n) { o.append((size_t)n * 2, ' '); };
    switch (v.t) {
        case JV::Null: o += "null"; break;
        case JV::Bool: o += v.b ? "true" : "false"; break;
        case JV::Num: { char b[40]; if (v.n == (double)(long long)v.n) snprintf(b, sizeof b, "%lld", (long long)v.n); else snprintf(b, sizeof b, "%g", v.n); o += b; break; }
        case JV::Str: o += "\"" + jesc(v.s) + "\""; break;
        case JV::Arr:
            if (v.a.empty()) { o += "[]"; break; }
            o += "[\n";
            for (size_t i = 0; i < v.a.size(); ++i) { pad(ind + 1); jdump(v.a[i], o, ind + 1); o += i + 1 < v.a.size() ? ",\n" : "\n"; }
            pad(ind); o += "]"; break;
        case JV::Obj:
            if (v.o.empty()) { o += "{}"; break; }
            o += "{\n";
            for (size_t i = 0; i < v.o.size(); ++i) {
                pad(ind + 1); o += "\"" + jesc(v.o[i].first) + "\": "; jdump(v.o[i].second, o, ind + 1);
                o += i + 1 < v.o.size() ? ",\n" : "\n";
            }
            pad(ind); o += "}"; break;
    }
}
static JV* jset(JV& obj, const std::string& k) {          // get-or-create member of an object
    if (obj.t != JV::Obj) { obj = JV(); obj.t = JV::Obj; }
    for (auto& kv : obj.o) if (kv.first == k) return &kv.second;
    obj.o.emplace_back(k, JV()); return &obj.o.back().second;
}
static std::string jstr(const JV* v, const std::string& def = "") {
    if (!v) return def;
    if (v->t == JV::Str) return v->s;
    if (v->t == JV::Bool) return v->b ? "1" : "";
    if (v->t == JV::Num) { char b[40]; if (v->n == (double)(long long)v->n) snprintf(b, sizeof b, "%lld", (long long)v->n); else snprintf(b, sizeof b, "%g", v->n); return b; }
    return def;
}

static std::string base_name(const std::string& p) { size_t k = p.find_last_of("/\\"); return k == std::string::npos ? p : p.substr(k + 1); }

struct BrowseResult {
    std::string path, parent;
    std::vector<std::string> dirs, roots;
    std::vector<FileEnt> files;
    bool ok = true;
};
// List one folder for the in-app file browser.  path=="" -> start place (home, or drives on Windows).
static BrowseResult browse_dir(std::string path, const std::vector<std::string>& exts) {
    BrowseResult r;
    auto add_root = [&](const std::string& d) { if (!d.empty() && is_dir(d) && std::find(r.roots.begin(), r.roots.end(), d) == r.roots.end()) r.roots.push_back(d); };
    add_root(g_dir); add_root(join(g_dir, "models"));
#if defined(_WIN32)
    DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) if (mask & (1u << i)) add_root(std::string(1, (char)('A' + i)) + ":\\");
    const char* home = getenv("USERPROFILE");
#else
    const char* home = getenv("HOME");
    for (const char* d : {"/", "/mnt", "/media", "/run/media"}) add_root(d);
#endif
    if (home) add_root(home);
    if (path.empty()) {
#if defined(_WIN32)
        DWORD m2 = GetLogicalDrives();
        for (int i = 0; i < 26; ++i) if (m2 & (1u << i)) r.dirs.push_back(std::string(1, (char)('A' + i)) + ":\\");
        return r;                                   // top level = the list of drives
#else
        path = home ? home : "/";
#endif
    }
    if (!is_dir(path)) { r.ok = false; r.path = path; return r; }
    r.path = path;
    std::vector<std::pair<std::string, bool>> es; list_dir(path, es);
    for (auto& e : es) {
        if (e.first.empty() || e.first[0] == '.') continue;
        if (e.second) r.dirs.push_back(e.first);
        else {
            bool ok = exts.empty();
            for (auto& x : exts) if (ends_ci(e.first, x.c_str())) ok = true;
            if (ok && r.files.size() < 4000) r.files.push_back({e.first, path, file_size(join(path, e.first))});
        }
    }
    std::sort(r.dirs.begin(), r.dirs.end(), [](const std::string& a, const std::string& b) { return lower(a) < lower(b); });
    std::sort(r.files.begin(), r.files.end(), [](const FileEnt& a, const FileEnt& b) { return lower(a.name) < lower(b.name); });
    std::string t = path;
    while (t.size() > 1 && (t.back() == '/' || t.back() == '\\') && !(t.size() == 3 && t[1] == ':')) t.pop_back();
    size_t k = t.find_last_of("/\\");
#if defined(_WIN32)
    if (t.size() == 3 && t[1] == ':') r.parent = "";
    else if (t.size() == 2 && t[1] == ':') r.parent = "";
    else if (k != std::string::npos) r.parent = (k == 2 && t[1] == ':') ? t.substr(0, 3) : t.substr(0, k);
#else
    if (t != "/") r.parent = (k == 0 || k == std::string::npos) ? "/" : t.substr(0, k);
#endif
    return r;
}

// Open a URL in the user's default browser (used for the web chat page).
static std::string g_browser_cmd;   // optional: the user's own browser program (e.g. an AppImage on a USB stick)
static void open_browser(const std::string& url) {
#if defined(_WIN32)
    if (!g_browser_cmd.empty()) { ShellExecuteA(nullptr, "open", g_browser_cmd.c_str(), url.c_str(), nullptr, SW_SHOWNORMAL); return; }
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        int dn = open("/dev/null", O_RDWR); if (dn >= 0) { dup2(dn, 0); dup2(dn, 1); dup2(dn, 2); }
        if (!g_browser_cmd.empty()) { std::string c = g_browser_cmd + " '" + url + "'"; execlp("sh", "sh", "-c", c.c_str(), (char*)nullptr); }
#  ifdef __APPLE__
        execlp("open", "open", url.c_str(), (char*)nullptr);
#  else
        execlp("xdg-open", "xdg-open", url.c_str(), (char*)nullptr);
        for (const char* b : {"firefox", "chromium", "google-chrome", "chromium-browser", "epiphany", "falkon", "midori"})
            execlp(b, b, url.c_str(), (char*)nullptr);
#  endif
        _exit(127);
    }
    if (pid > 0) { std::thread([pid] { int st; waitpid(pid, &st, 0); }).detach(); }
#endif
}

// ───────────────────────────── model metadata (layers, name, chat template) ──────────────

struct ModelMeta {
    bool ok = false; std::string arch, name, tmpl; int layers = 0;
};

static bool rd_exact(std::ifstream& f, void* p, size_t n) { f.read((char*)p, (std::streamsize)n); return (size_t)f.gcount() == n; }
static bool gguf_skip(std::ifstream& f, uint32_t t, int depth = 0) {
    static const int sz[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (t > 12 || depth > 4) return false;
    if (t == 8) { uint64_t n; if (!rd_exact(f, &n, 8) || n > (1u << 28)) return false; f.seekg((std::streamoff)n, std::ios::cur); return (bool)f; }
    if (t == 9) {
        uint32_t et; uint64_t n; if (!rd_exact(f, &et, 4) || !rd_exact(f, &n, 8)) return false;
        if (et != 8 && et != 9) { if (et > 12) return false; f.seekg((std::streamoff)(n * (uint64_t)sz[et]), std::ios::cur); return (bool)f; }
        for (uint64_t i = 0; i < n; ++i) if (!gguf_skip(f, et, depth + 1)) return false;
        return true;
    }
    f.seekg(sz[t], std::ios::cur); return (bool)f;
}
static ModelMeta read_gguf_meta(const std::string& path) {
    ModelMeta m; std::ifstream f(path, std::ios::binary); if (!f) return m;
    char magic[4]; uint32_t ver; uint64_t nt, nkv;
    if (!rd_exact(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !rd_exact(f, &ver, 4) || !rd_exact(f, &nt, 8) || !rd_exact(f, &nkv, 8) || nkv > 100000) return m;
    for (uint64_t i = 0; i < nkv; ++i) {
        uint64_t kl; if (!rd_exact(f, &kl, 8) || kl > 4096) break;
        std::string key(kl, '\0'); if (kl && !rd_exact(f, &key[0], kl)) break;
        uint32_t t; if (!rd_exact(f, &t, 4)) break;
        auto ends = [&](const char* suf) { size_t n = strlen(suf); return key.size() >= n && key.compare(key.size() - n, n, suf) == 0; };
        if (t == 8 && (key == "general.architecture" || key == "general.name" || key == "tokenizer.chat_template")) {
            uint64_t n; if (!rd_exact(f, &n, 8) || n > (1u << 24)) break;
            std::string v(n, '\0'); if (n && !rd_exact(f, &v[0], n)) break;
            if (key == "general.architecture") m.arch = v; else if (key == "general.name") m.name = v; else m.tmpl = v;
        } else if ((t == 4 || t == 5 || t == 10 || t == 11) && ends(".block_count")) {
            if (t == 4 || t == 5) { uint32_t v; if (!rd_exact(f, &v, 4)) break; m.layers = (int)v; }
            else { uint64_t v; if (!rd_exact(f, &v, 8)) break; m.layers = (int)v; }
        } else if (!gguf_skip(f, t)) break;
        if (!m.arch.empty() && !m.name.empty() && !m.tmpl.empty() && m.layers) break;
    }
    m.ok = !m.arch.empty() || m.layers > 0; return m;
}
static ModelMeta read_nxm_meta(const std::string& path);   // defined after the JSON parser helpers below

static ModelMeta read_model_meta(const std::string& path) { return ends_ci(path, ".nxm") ? read_nxm_meta(path) : read_gguf_meta(path); }

static std::string jnum_or_str(const JV* v) {          // number or string -> text
    if (!v) return "";
    if (v->t == JV::Num) { char b[32]; snprintf(b, sizeof b, "%.0f", v->n); return b; }
    return v->t == JV::Str ? v->s : "";
}
static ModelMeta read_nxm_meta(const std::string& path) {
    ModelMeta m; std::ifstream f(path, std::ios::binary); if (!f) return m;
    char magic[16]; uint64_t hdr[3];
    if (!rd_exact(f, magic, 16) || memcmp(magic, "NEXUS_NXM_V1", 12) != 0 || !rd_exact(f, hdr, 24)) return m;
    if (hdr[0] > (64u << 20) || hdr[1] > (256u << 20)) return m;
    std::string man(hdr[0], '\0'), tab(hdr[1], '\0');
    if (hdr[0] && !rd_exact(f, &man[0], hdr[0])) return m;
    if (hdr[1] && !rd_exact(f, &tab[0], hdr[1])) return m;
    JV jm; if (!json_parse(man, jm)) return m;
    const JV* model = jm.get("model");
    m.arch = model ? jnum_or_str(model->get("model_type")) : "";
    m.name = model ? jnum_or_str(model->get("name")) : "";
    if (m.name.empty()) m.name = base_name(path);
    for (auto& ch : m.arch) ch = (char)tolower((unsigned char)ch);
    if (m.arch.find("qwen") != std::string::npos) m.arch = "qwen3next";      // same mapping the old launcher used
    if (m.arch.empty()) m.arch = "qwen3next";
    std::string n = model ? jnum_or_str(model->get("num_hidden_layers")) : "";
    if (n.empty()) n = jnum_or_str(jm.get("n_layers"));
    m.layers = atoi(n.c_str());
    // embedded chat template: metadata/chat_template.jinja, else tokenizer_config.json["chat_template"]
    JV jt; if (json_parse(tab, jt)) if (const JV* files = jt.get("files")) {
        uint64_t payload = 16 + 24 + hdr[0] + hdr[1];
        auto embedded = [&](const char* rel) -> std::string {
            for (auto& r : files->a) if (jnum_or_str(r.get("path")) == rel) {
                uint64_t off = (uint64_t)atof(jnum_or_str(r.get("offset")).c_str()), nb = (uint64_t)atof(jnum_or_str(r.get("nbytes")).c_str());
                if (nb == 0 || nb > (64u << 20)) return "";
                f.clear(); f.seekg((std::streamoff)(payload + off)); std::string s(nb, '\0');
                if (!rd_exact(f, &s[0], nb)) return "";
                return s;
            }
            return "";
        };
        m.tmpl = embedded("metadata/chat_template.jinja");
        if (m.tmpl.empty()) { std::string tc = embedded("metadata/tokenizer_config.json"); JV jc; if (!tc.empty() && json_parse(tc, jc)) m.tmpl = jnum_or_str(jc.get("chat_template")); }
    }
    m.ok = true; return m;
}

// the address other computers should use to reach this one (no packet is sent: a UDP "connect" only picks a route)
static std::string local_ip() {
    sock_t s = socket(AF_INET, SOCK_DGRAM, 0); if (s == NX_INVALID_SOCK) return "127.0.0.1";
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(80); inet_pton(AF_INET, "192.0.2.1", &a.sin_addr);
    std::string out = "127.0.0.1";
    if (connect(s, (sockaddr*)&a, sizeof a) == 0) {
        sockaddr_in me{};
#if defined(_WIN32)
        int l = sizeof me;
#else
        socklen_t l = sizeof me;
#endif
        if (getsockname(s, (sockaddr*)&me, &l) == 0) { char b[64]; if (inet_ntop(AF_INET, &me.sin_addr, b, sizeof b)) out = b; }
    }
    sock_close(s); return out;
}

// Can we open a TCP connection to host:port within `ms`?  (used by the "Test" button for donor computers)
static bool tcp_reachable(const std::string& host, int port, int ms) {
    addrinfo hints{}, *res = nullptr; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) return false;
    sock_t s = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == NX_INVALID_SOCK) { freeaddrinfo(res); return false; }
#if defined(_WIN32)
    u_long nb = 1; ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
    connect(s, res->ai_addr, (int)res->ai_addrlen); freeaddrinfo(res);
    fd_set wf; FD_ZERO(&wf); FD_SET(s, &wf); timeval tv{ms / 1000, (ms % 1000) * 1000};
    bool ok = select((int)s + 1, nullptr, &wf, nullptr, &tv) > 0;
    if (ok) {
#if defined(_WIN32)
        int err = 0, l = sizeof err; getsockopt(s, SOL_SOCKET, SO_ERROR, (char*)&err, &l);
#else
        int err = 0; socklen_t l = sizeof err; getsockopt(s, SOL_SOCKET, SO_ERROR, &err, &l);
#endif
        ok = err == 0;
    }
    sock_close(s); return ok;
}
