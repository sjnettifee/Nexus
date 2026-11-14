/*
 * nexus/src/api/dist_session.h
 *
 * Distributed inference across two machines via TCP.
 *
 * HOW IT WORKS
 * -------------------------------------------------------------------
 * Machine A (primary)   -- runs layers 0..split_layer
 * Machine B (secondary) -- runs layers split_layer..n_layers
 *
 * Each machine loads ONLY its half of the model. A 40GB model becomes
 * two 20GB halves -- each fits in 16GB RAM with a quantized (Q4_K) model.
 *
 * Per-token decode flow:
 *   A: embed -> layers 0..split -> x_scratch_ (20KB for 14B)
 *   A: send HIDDEN msg to B
 *   B: recv hidden -> set x_scratch_ -> layers split..N -> lm_head -> sample
 *   B: send TOKEN msg back to A
 *   A: emit token to user
 *
 * Prefill flow (pipelined):
 *   A processes position t while B processes position t-1 simultaneously.
 *   Both CPUs run in parallel -- ~2x prefill speedup.
 *
 * KV cache: never crosses the wire. Machine A owns KV for layers 0..split,
 * Machine B owns KV for layers split..N. Each machine builds its own KV
 * naturally as forward_layer_range() runs.
 *
 * SETUP
 * -------------------------------------------------------------------
 * On Machine B (secondary):
 *   DistPeerServer server(engine_b, 20, 40);  // split_layer=20, n_layers=40
 *   server.listen_and_serve(9876);  // blocks -- run in a thread or process
 *
 * On Machine A (primary):
 *   DistSession session(engine_a, 20);  // split_layer=20
 *   session.connect("192.168.1.50", 9876);
 *   session.generate_stream(prompt, cfg, token_cb);
 *
 * MODEL LOADING
 * -------------------------------------------------------------------
 * Machine A: load full GGUF, then call set_active_layers(0, split_layer).
 * Machine B: load full GGUF, then call set_active_layers(split_layer, n_layers).
 * (Or use two separate split GGUFs if you have a tool to split them.)
 *
 * REQUIREMENTS
 * -------------------------------------------------------------------
 * - transformer.h must include forward_layer_range (see transformer_layer_range.h)
 * - Link with nothing extra on Linux/macOS (POSIX sockets)
 * - On Windows: link ws2_32, call WSAStartup before use
 */

#pragma once

#include "../core/memguard.h"
#include "engine.h"
#include "../model/transformer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <cassert>
#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#ifndef _WIN32
#  include <unistd.h>
#  include <sys/syscall.h>
#endif
#include <atomic>
#include <chrono>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <memory>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   using socket_t = SOCKET;
#  define INVALID_SOCK INVALID_SOCKET
#  define sock_close(s) closesocket(s)
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <fcntl.h>
   using socket_t = int;
#  define INVALID_SOCK (-1)
#  define sock_close(s) ::close(s)
#endif

namespace nexus {
namespace dist {

// --- Wire protocol ------------------------------------------------------------
//
// All messages are little-endian. No framing library needed -- messages are
// short and sent/received synchronously one at a time.
//
// A -> B  HIDDEN:  [uint8 MSG_HIDDEN=1] [uint32 pos] [uint32 n_floats] [n_floats x float32]
// B -> A  TOKEN:   [uint8 MSG_TOKEN=2]  [int32 token_id]  (legacy)
// B -> A  LOGITS:  [uint8 MSG_LOGITS=12] [uint32 n] [n x float32]
// A -> B  RESET:   [uint8 MSG_RESET=3]   (clear KV, prepare for new conversation)
// A -> B  PREFILL: [uint8 MSG_PREFILL=4] [uint32 pos] [uint32 n_floats] [n_floats x float]
//                  (like HIDDEN but B does NOT sample -- just stores KV and ACKs)
// B -> A  ACK:     [uint8 MSG_ACK=5]     (B finished prefill for this position)
// A -> B  BYE:     [uint8 MSG_BYE=6]     (close connection cleanly)
// A -> B  INIT:    [uint8 MSG_INIT=7] [uint32 layer_start] [uint32 layer_end]
//                  FIRST message after connecting. Primary tells worker its layer range.
//                  Worker acknowledges with MSG_ACK. Required before any inference.

enum class Msg : uint8_t {
    HIDDEN   = 1,
    TOKEN    = 2,
    RESET    = 3,
    PREFILL  = 4,
    ACK      = 5,
    BYE      = 6,
    INIT     = 7,
    // Primary streams layer weights to worker after INIT.
    // Worker receives a small header-only GGUF and then raw tensor bytes.
    // It loads tensors directly into final owned RAM; no temp file or fake GGUF blob.
    // Format: [MSG_WEIGHTS][uint64 header_size][header-only GGUF][raw tensor bytes...]
    WEIGHTS  = 8,
    KV_SAVE  = 9,   // primary->worker: "send me your KV bytes"
    KV_LOAD  = 10,  // primary->worker: "here are your KV bytes to restore"
    LOCAL_LOAD = 11, // primary->worker: load assigned shard from worker's local GGUF
    LOGITS = 12,     // worker->primary: final logits for primary-side sampling
    PROGRESS = 14,     // worker->primary while it reads a prompt: [u32 layers_done][u32 layers_total]
    PREFILL_BATCH = 13, // primary->worker: [u32 pos][u32 n][u32 hidden][n*hidden floats] -> worker runs its layers over all n tokens at once, replies ACK
    KV_LAYER_LOAD = 15  // primary->worker (hybrid prefill): [u32 layer][u32 n] [n bytes] -> worker stores this one layer's KV (+ state, if any), replies ACK. Sent once per layer as each finishes, instead of waiting for the whole range.
};

// --- XorCipher ---------------------------------------------------------------
//
// Derives a 256-byte key from the passphrase using FNV-1a hashing.
// Applied to all bytes on the wire AFTER the initial connection.
// The passphrase is NEVER transmitted -- both sides derive the key independently.
// A wrong passphrase produces garbage on the remote end.
//
class XorCipher {
    uint8_t key_[256];
    size_t  pos_ = 0;      // bytes sent so far (the other side's receive position)
    size_t  pos_r_ = 0;    // bytes received so far (the other side's send position)
public:
    XorCipher() { memset(key_, 0, 256); }  // null cipher when no passphrase
    explicit XorCipher(const std::string& passphrase) {
        if (passphrase.empty()) { memset(key_, 0, 256); return; }
        uint64_t h = 14695981039346656037ULL;
        for (unsigned char c : passphrase) { h ^= c; h *= 1099511628211ULL; }
        for (int i = 0; i < 256; ++i) {
            h ^= (uint64_t)i; h *= 1099511628211ULL;
            key_[i] = (uint8_t)(h >> 56);
        }
    }
    // Each direction has its own position, so both computers can send and receive at the same time
    // (needed to overlap work) without their positions drifting apart.
    void apply(void* buf, size_t n) {          // outgoing
        uint8_t* p = (uint8_t*)buf;
        for (size_t i = 0; i < n; ++i) p[i] ^= key_[pos_++ & 255];
    }
    void apply_rx(void* buf, size_t n) {       // incoming
        uint8_t* p = (uint8_t*)buf;
        for (size_t i = 0; i < n; ++i) p[i] ^= key_[pos_r_++ & 255];
    }
    void advance(size_t n) { pos_ = (uint8_t)((size_t)pos_ + n); }
    void advance_rx(size_t n) { pos_r_ = (uint8_t)((size_t)pos_r_ + n); }
};

// Auth magic -- sent encrypted. Worker decrypts and checks against this constant.
// If passphrase doesn't match, decrypted bytes are garbage and auth fails.
static const uint8_t NEXUS_MAGIC[8] = {'N','E','X','U','S','D','S','T'};

static int nx_dist_clamp_trace_level(int v) {
    if (v < 0) return 0;
    if (v > 3) return 3;
    return v;
}

static int nx_dist_trace_level_from_env() {
    const char* s = std::getenv("NEXUS_TRACE");
    if (!s || !*s) return 0;
    return nx_dist_clamp_trace_level(std::atoi(s));
}

static void nx_dist_set_trace_level_env(int level) {
    level = nx_dist_clamp_trace_level(level);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", level);

#ifdef _WIN32
    _putenv_s("NEXUS_TRACE", buf);
#else
    setenv("NEXUS_TRACE", buf, 1);
#endif
}

// ─── Override-knob env getters/setters ─────────────────────────────────────
//
// Mirror the trace-level helpers so the launcher's auto/manual selection on
// the primary can be forwarded over the wire and installed in the worker's
// environment before transformer.h reads them at load time.
//
// Empty string means "no override" (leave whatever the primary's environment
// already had alone, or unset).
static std::string nx_dist_str_from_env(const char* name) {
    const char* s = std::getenv(name);
    return s ? std::string(s) : std::string();
}

static void nx_dist_set_str_env(const char* name, const std::string& v) {
#ifdef _WIN32
    // _putenv_s with empty value unsets on Windows.
    _putenv_s(name, v.c_str());
#else
    if (v.empty()) {
        unsetenv(name);
    } else {
        setenv(name, v.c_str(), 1);
    }
#endif
}

// --- Low-level socket helpers -------------------------------------------------

static bool sock_init() {
#ifdef _WIN32
    WSADATA wd; return WSAStartup(MAKEWORD(2,2), &wd) == 0;
#else
    return true;
#endif
}

// True for an error worth a short retry instead of treating it as a dead connection: the syscall
// was merely interrupted, or (should these ever end up non-blocking) momentarily would have
// blocked. A real drop (connection reset, broken pipe, the peer closing its end) is NOT transient
// and falls through to the normal failure path below.
static bool sock_err_is_transient() {
#ifdef _WIN32
    int e = WSAGetLastError();
    return e == WSAEINTR || e == WSAEWOULDBLOCK;
#else
    return errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

// Send exactly n bytes. A brief network flicker (a dropped packet or two, an interface bouncing
// for a second) surfaces here as a transient, retryable error rather than the connection actually
// dying -- retried with a short sleep for up to ~20s total before finally giving up, so one such
// hiccup in the middle of a long prefill or a chat turn doesn't tear down the whole dist session.
// A genuine disconnect still fails immediately, same as before.
static bool sock_send_all(socket_t fd, const void* buf, size_t n) {
    const uint8_t* p = (const uint8_t*)buf;
    int retries = 0;
    const int kMaxRetries = 200;   // ~20s at the 100ms backoff below
    while (n > 0) {
        int sent = send(fd, (const char*)p, (int)n, 0);
        if (sent <= 0) {
            if (sock_err_is_transient() && retries < kMaxRetries) {
                ++retries;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            return false;
        }
        retries = 0;
        p += sent; n -= sent;
    }
    return true;
}

// Receive exactly n bytes. Same transient-retry treatment as sock_send_all above. got == 0 (the
// peer cleanly closed its end) is never retried -- that's a real, intentional disconnect.
static bool sock_recv_all(socket_t fd, void* buf, size_t n) {
    uint8_t* p = (uint8_t*)buf;
    int retries = 0;
    const int kMaxRetries = 200;
    while (n > 0) {
        int got = recv(fd, (char*)p, (int)n, 0);
        if (got <= 0) {
            if (got < 0 && sock_err_is_transient() && retries < kMaxRetries) {
                ++retries;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            return false;
        }
        retries = 0;
        p += got; n -= got;
    }
    return true;
}

static bool sock_send_u8(socket_t fd, uint8_t v)   { return sock_send_all(fd, &v, 1); }
static bool sock_send_u32(socket_t fd, uint32_t v)  { return sock_send_all(fd, &v, 4); }
static bool sock_send_i32(socket_t fd, int32_t v)   { return sock_send_all(fd, &v, 4); }
static bool sock_recv_u8(socket_t fd, uint8_t& v)   { return sock_recv_all(fd, &v, 1); }
static bool sock_recv_u32(socket_t fd, uint32_t& v) { return sock_recv_all(fd, &v, 4); }
static bool sock_recv_i32(socket_t fd, int32_t& v)  { return sock_recv_all(fd, &v, 4); }

static bool sock_send_floats(socket_t fd, const float* f, size_t n) {
    return sock_send_all(fd, f, n * sizeof(float));
}
static bool sock_recv_floats(socket_t fd, float* f, size_t n) {
    return sock_recv_all(fd, f, n * sizeof(float));
}

// Encrypted variants -- apply XOR cipher in-place before send / after recv
static bool sock_send_enc(socket_t fd, const void* buf, size_t n, XorCipher& c) {
    static thread_local std::vector<uint8_t> enc_buf;
    if (enc_buf.size() < n) enc_buf.resize(n);
    memcpy(enc_buf.data(), buf, n);
    c.apply(enc_buf.data(), n);
    return sock_send_all(fd, enc_buf.data(), n);
}

static bool sock_recv_dec(socket_t fd, void* buf, size_t n, XorCipher& c) {
    if (!sock_recv_all(fd, buf, n)) return false;
    c.apply_rx(buf, n);
    return true;
}
// Convenience wrappers for common types
static bool sock_send_u8_enc(socket_t fd, uint8_t v, XorCipher& c)
    { return sock_send_enc(fd, &v, 1, c); }
static bool sock_send_u32_enc(socket_t fd, uint32_t v, XorCipher& c)
    { return sock_send_enc(fd, &v, 4, c); }
static bool sock_send_u64_enc(socket_t fd, uint64_t v, XorCipher& c)
    { return sock_send_enc(fd, &v, 8, c); }
static bool sock_recv_u64_dec(socket_t fd, uint64_t& v, XorCipher& c)
    { return sock_recv_dec(fd, &v, 8, c); }
static bool sock_send_i32_enc(socket_t fd, int32_t v, XorCipher& c)
    { return sock_send_enc(fd, &v, 4, c); }
static bool sock_recv_u8_dec(socket_t fd, uint8_t& v, XorCipher& c)
    { return sock_recv_dec(fd, &v, 1, c); }
static bool sock_recv_u32_dec(socket_t fd, uint32_t& v, XorCipher& c)
    { return sock_recv_dec(fd, &v, 4, c); }
static bool sock_recv_i32_dec(socket_t fd, int32_t& v, XorCipher& c)
    { return sock_recv_dec(fd, &v, 4, c); }
static bool sock_send_floats_enc(socket_t fd, const float* f, size_t n, XorCipher& c) {
    // Sent in pieces so a whole-prompt activation block (can be GBs) is not copied a second time in memory.
    const size_t piece = 1u << 20;                       // floats per piece (4 MB)
    std::vector<uint8_t> buf(std::min(n, piece) * sizeof(float));
    for (size_t off = 0; off < n; off += piece) {
        const size_t cnt = std::min(piece, n - off);
        memcpy(buf.data(), f + off, cnt * sizeof(float));
        c.apply(buf.data(), cnt * sizeof(float));
        if (!sock_send_all(fd, buf.data(), cnt * sizeof(float))) return false;
    }
    return true;
}
static bool sock_recv_floats_dec(socket_t fd, float* f, size_t n, XorCipher& c) {
    if (!sock_recv_all(fd, f, n * sizeof(float))) return false;
    c.apply_rx(f, n * sizeof(float));
    return true;
}

// ─── Short string helpers (length-prefixed, max 64 chars) ──────────────────
//
// Used to carry the override knobs (NEXUS_ROPE_LAYOUT, NEXUS_NORM_CONVENTION)
// from primary to worker on MSG_INIT. Length is u8 to keep wire format tight;
// 64 chars is more than enough for short config strings like "half_split" or
// "zero_centered". Empty string ("") means "auto / no override".
static bool sock_send_str_enc(socket_t fd, const std::string& s, XorCipher& c) {
    if (s.size() > 64) return false;
    uint8_t n = (uint8_t)s.size();
    if (!sock_send_u8_enc(fd, n, c)) return false;
    if (n == 0) return true;
    return sock_send_enc(fd, s.data(), n, c);
}
static bool sock_recv_str_dec(socket_t fd, std::string& s, XorCipher& c) {
    uint8_t n = 0;
    if (!sock_recv_u8_dec(fd, n, c)) return false;
    s.clear();
    if (n == 0) return true;
    if (n > 64) return false;
    char buf[64];
    if (!sock_recv_dec(fd, buf, n, c)) return false;
    s.assign(buf, n);
    return true;
}

// Enable TCP_NODELAY (disable Nagle) -- critical for low-latency small messages
static void sock_set_nodelay(socket_t fd) {
    int on = 1;
#ifdef _WIN32
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&on, sizeof(on));
#else
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
#endif
}

// Ask the OS to actively probe an idle connection so a peer that's actually gone (wifi dropped,
// cable pulled, the other machine lost power) is noticed in well under a minute instead of a
// send/recv hanging indefinitely on an interface that still *looks* up locally. Tuned noticeably
// shorter than most OS defaults (which can be hours).
static void sock_set_keepalive(socket_t fd) {
    int on = 1;
#ifdef _WIN32
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, (const char*)&on, sizeof(on));
    // TCP_KEEPIDLE/INTVL/CNT aren't portable pre-Windows 10; the real knob there is
    // WSAIoctl(SIO_KEEPALIVE_VALS), which needs more plumbing than is worth it here. Plain
    // SO_KEEPALIVE still uses the (longer) system default timers -- still better than nothing.
#else
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
    int idle = 10, intvl = 5, cnt = 4;   // probe after 10s idle, every 5s, give up after 4 misses (~30s to notice)
#  ifdef __APPLE__
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof(idle));
#  else
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE,  &idle,  sizeof(idle));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
    setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT,   &cnt,   sizeof(cnt));
#  endif
#endif
}


// --- DistPeerServer -----------------------------------------------------------
//
// Runs on Machine B. Listens for connections from Machine A.
// Owns layers [split_layer, n_layers) of the model.
//
// Usage:
//   Engine engine_b;
//   engine_b.load("model.gguf", 8);
//   // Tell the engine to only forward layers [split_layer, n_layers)
//   // (we call forward_layer_range internally -- no weight stripping needed)
//   DistPeerServer server(engine_b, split_layer, n_layers);
//   server.listen_and_serve(9876);  // blocks
//
class DistPeerServer {
public:
    // split_layer / n_total_layers: -1 means "receive from primary via MSG_INIT".
    // passphrase: key for XOR encryption. NEVER sent over wire.
    //   Both sides derive the same key independently.
    //   Wrong passphrase = garbage data = connection dies.
    DistPeerServer(Engine& engine, int split_layer, int n_total_layers,
                   const std::string& passphrase = "",
                   const std::string& local_model_path = "",
                   int n_threads = 0)
        : engine_(engine)
        , split_layer_(split_layer)
        , n_layers_(n_total_layers)
        , passphrase_(passphrase)
        , local_model_path_(local_model_path)
        , n_threads_(n_threads)
    {}

    // Blocks until stopped. Call from a dedicated thread if needed.
    // host: bind interface -- "0.0.0.0" for all interfaces, or specific IP.
    // Uses the SAME binding method as the normal HTTP server so external
    // machines can reach this port.
    void listen_and_serve(int port, const std::string& host = "0.0.0.0") {
        sock_init();
#ifdef _WIN32
        SOCKET srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv == INVALID_SOCKET) {
            fprintf(stderr, "[dist-server] socket() failed\n"); return;
        }
#else
        int srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) { perror("socket"); return; }
#endif
        int yes = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons((uint16_t)port);
        addr.sin_addr.s_addr = INADDR_ANY;  // bind all interfaces -- works on Linux + Windows
        fprintf(stderr, "[dist-server] Binding to 0.0.0.0:%d\n", port);
        fflush(stderr);
        if (bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind");
            sock_close(srv); return;
        }
        if (listen(srv, 64) < 0) { perror("listen"); sock_close(srv); return; }
        fprintf(stderr, "[dist-server] Listening on port %d (layers %d-%d)\n",
                port, split_layer_, n_layers_-1);
        // "listening on port N" -- donor_launcher.py detects this exact phrase
        fprintf(stderr, "[dist-server] listening on port %d\n", port);
        fflush(stderr);

        running_ = true;
        while (running_) {
            sockaddr_in cli_addr{};
            socklen_t cli_len = sizeof(cli_addr);
            socket_t conn = accept(srv, (sockaddr*)&cli_addr, &cli_len);
            if (conn == INVALID_SOCK) continue;
            sock_set_nodelay(conn);
            sock_set_keepalive(conn);
            fprintf(stderr, "[dist-server] Client connected: %s\n",
                    inet_ntoa(cli_addr.sin_addr));
            handle_client(conn);
            sock_close(conn);
            fprintf(stderr, "[dist-server] Client disconnected\n");
        }
        sock_close(srv);
    }

    void stop() { running_ = false; }

private:
    Engine&           engine_;
    int               split_layer_;
    int               n_layers_;
    std::atomic<bool> running_{false};
    std::string        passphrase_;  // stored so each connection gets a fresh cipher
    std::string        local_model_path_;
    int                n_threads_ = 0;
    std::vector<float> hidden_buf_;
    std::vector<float> batch_buf_;
    uint32_t prog_total_ = 0;

    void handle_client(socket_t conn) {
        // CRITICAL: create a fresh cipher for EVERY connection so pos_ starts at 0.
        // A shared/member cipher would advance pos_ on the first connection and
        // all subsequent connections would auth-fail on the very first bytes.
        XorCipher cipher_(passphrase_);

        // No model loaded yet. We auth first, then receive weights from primary.
        // -- Step 1: XOR auth -- primary sends 8 magic bytes encrypted ----------
        // Primary encrypts NEXUS_MAGIC with its key derived from the passphrase.
        // We decrypt with our key derived from the SAME passphrase.
        // If passphrases match: decrypted bytes == NEXUS_MAGIC -> auth OK.
        // If passphrases differ: decrypted bytes are garbage -> auth FAIL.
        // The passphrase is NEVER sent over the wire.
        {
            uint8_t buf[8];
            if (!sock_recv_all(conn, buf, 8)) {
                fprintf(stderr, "[dist-server] Auth: connection dropped during magic recv\n");
                return;
            }
            cipher_.apply_rx(buf, 8);  // decrypt
            if (memcmp(buf, NEXUS_MAGIC, 8) != 0) {
                fprintf(stderr, "[dist-server] Auth FAILED -- wrong passphrase or bad client\n");
                return;  // no response -- attacker gets nothing
            }
            fprintf(stderr, "[dist-server] Auth OK (XOR passphrase verified)\n");
        }

        // All subsequent traffic is XOR-encrypted.
        // Both sides use independent but identical ciphers derived from the passphrase.

        // -- Step 2: MSG_INIT -- primary sets our layer range -------------------
        {
            uint8_t msg = 0;
            if (!sock_recv_u8_dec(conn, msg, cipher_) || (Msg)msg != Msg::INIT) {
                fprintf(stderr, "[dist-server] Expected MSG_INIT, got %d\n", (int)msg);
                return;
            }
            uint32_t ls = 0, le = 0, trace_level = 0;
            if (!sock_recv_u32_dec(conn, ls, cipher_) ||
                !sock_recv_u32_dec(conn, le, cipher_) ||
                !sock_recv_u32_dec(conn, trace_level, cipher_)) return;

            // Extended INIT fields: override knobs for forward-pass auto-detect.
            // Empty string means "no override on the worker side".
            std::string rope_layout_str, norm_conv_str;
            if (!sock_recv_str_dec(conn, rope_layout_str, cipher_) ||
                !sock_recv_str_dec(conn, norm_conv_str,   cipher_)) {
                fprintf(stderr, "[dist-server] Failed to receive override knobs\n");
                return;
            }

            split_layer_ = (int)ls;
            n_layers_    = (int)le;

            trace_level = (uint32_t)nx_dist_clamp_trace_level((int)trace_level);
            nx_dist_set_trace_level_env((int)trace_level);

            // Install the override knobs into the worker's environment BEFORE
            // it loads the model — transformer.h reads them during load().
            nx_dist_set_str_env("NEXUS_ROPE_LAYOUT",       rope_layout_str);
            nx_dist_set_str_env("NEXUS_NORM_CONVENTION",   norm_conv_str);

            fprintf(stderr, "[dist-server] Layer range: %d - %d\n", split_layer_, n_layers_ - 1);
            fprintf(stderr, "[dist-server] NEXUS_TRACE=%u\n", trace_level);
            fprintf(stderr, "[dist-server] NEXUS_ROPE_LAYOUT=%s\n",
                    rope_layout_str.empty() ? "(unset/auto)" : rope_layout_str.c_str());
            fprintf(stderr, "[dist-server] NEXUS_NORM_CONVENTION=%s\n",
                    norm_conv_str.empty() ? "(unset/auto)" : norm_conv_str.c_str());
            fflush(stderr);

            sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_);
        }

        // -- Step 3: load worker shard -------------------------------------------
        // Primary chooses one of two modes:
        //   MSG_LOCAL_LOAD: worker opens its own local GGUF and loads only its assigned layers.
        //   MSG_WEIGHTS:   primary streams header-only shard metadata + raw tensor bytes.
        {
            uint8_t msg = 0;
            if (!sock_recv_u8_dec(conn, msg, cipher_)) {
                fprintf(stderr, "[dist-server] Expected load-mode message, connection dropped\n");
                return;
            }

            int thr = n_threads_ > 0 ? n_threads_
                                     : std::max(1, (int)std::thread::hardware_concurrency() - 1);

            if ((Msg)msg == Msg::LOCAL_LOAD) {
                if (local_model_path_.empty()) {
                    fprintf(stderr, "[dist-server] LOCAL_LOAD requested but no local model path was configured\n");
                    return;
                }
                fprintf(stderr, "[dist-server] Loading local model shard: %s\n",
                        local_model_path_.c_str());
                fprintf(stderr, "[dist-server] Local shard layers: %d-%d\n",
                        split_layer_, n_layers_ - 1);
                fflush(stderr);

                // *** v31: this used to pass nullptr here -- the donor's OWN load had no progress
                // reporting at all, silent the whole time its model loads, while the primary's own
                // load already prints "[load] NN%  <stage>" lines the launcher's log view already
                // knows how to turn into a bar (same draw_log()/track_load() every log view uses).
                // Same format, same callback shape as the primary's lcb in server.cpp -- the
                // launcher needs no changes to pick this up on the donor's own log page.
                auto donor_lcb = [](const LoadProgress& p) {
                    fprintf(stderr, "[load] %.0f%%  %s\n", p.fraction * 100.0f, p.stage.substr(0, 60).c_str());
                    fflush(stderr);
                };
                if (!engine_.load_range(local_model_path_, split_layer_, n_layers_, thr, donor_lcb)) {
                    fprintf(stderr, "[dist-server] Failed to load local model shard\n");
                    return;
                }
                fprintf(stderr, "[dist-server] Local model shard loaded. Ready.\n");
                fflush(stderr);
                sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_);
            }
            else if ((Msg)msg == Msg::WEIGHTS) {
                uint64_t header_size = 0;
                if (!sock_recv_u64_dec(conn, header_size, cipher_) || header_size == 0) {
                    fprintf(stderr, "[dist-server] MSG_WEIGHTS: invalid streamed header size\n");
                    return;
                }
                // Header contains GGUF metadata + selected tensor descriptors only.
                // Tensor data is NOT assembled into a fake GGUF blob. Keep a hard cap
                // so a corrupted sender cannot force huge temporary RAM use.
                const uint64_t MAX_STREAM_HEADER = 1024ULL * 1024ULL * 1024ULL; // 1GB absolute cap
                if (header_size > MAX_STREAM_HEADER) {
                    fprintf(stderr, "[dist-server] MSG_WEIGHTS: streamed header too large: %llu MB\n",
                            (unsigned long long)header_size / (1024*1024));
                    return;
                }

                fprintf(stderr, "[dist-server] Receiving streamed shard header: %llu MB\n",
                        (unsigned long long)header_size / (1024*1024));
                fflush(stderr);

                std::vector<uint8_t> header((size_t)header_size);
                if (!sock_recv_all(conn, header.data(), (size_t)header_size)) { // raw, no XOR
                    cipher_.advance_rx((size_t)header_size);
                    fprintf(stderr, "[dist-server] MSG_WEIGHTS: header recv failed\n");
                    return;
                }
                cipher_.advance_rx((size_t)header_size);

                fprintf(stderr, "[dist-server] Streaming tensor bytes directly into owned weight buffers...\n");
                fflush(stderr);

                auto read_tensor_bytes = [&](void* dst, size_t n) -> bool {
                    if (n == 0) return true;
                    // Raw bulk payload for speed. This writes directly into the final
                    // tensor allocation supplied by Transformer::load_streamed_shard().
                    if (!sock_recv_all(conn, dst, n)) {
                        cipher_.advance_rx(n);
                        return false;
                    }
                    cipher_.advance_rx(n);
                    return true;
                };

                // *** v31: same silent-load fix as the LOCAL_LOAD branch above, for the streamed-
                // weights path (--dist-peer without --dist-peer-local-load).
                auto donor_lcb2 = [](const LoadProgress& p) {
                    fprintf(stderr, "[load] %.0f%%  %s\n", p.fraction * 100.0f, p.stage.substr(0, 60).c_str());
                    fflush(stderr);
                };
                bool loaded = engine_.load_streamed_shard(
                    header.data(), header.size(), thr, read_tensor_bytes, donor_lcb2);

                // Header is no longer needed after the parser has copied/parsed metadata.
                header.clear(); header.shrink_to_fit();

                if (!loaded) {
                    fprintf(stderr, "[dist-server] Failed to load streamed layer tensors\n");
                    return;
                }
                fprintf(stderr, "[dist-server] Worker streamed layers loaded. Ready.\n");
                fflush(stderr);
                sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_);
            }
            else {
                fprintf(stderr, "[dist-server] Expected MSG_WEIGHTS or MSG_LOCAL_LOAD, got %d\n", (int)msg);
                return;
            }
        }

        const auto info = engine_.model_info();
        const int hidden_dim2 = info.hidden_dim;
        if (hidden_dim2 <= 0) {
            fprintf(stderr, "[dist-server] hidden_dim=0 after weight load\n");
            return;
        }
        hidden_buf_.resize(hidden_dim2);

        while (true) {
            uint8_t msg_type = 0;
            if (!sock_recv_u8_dec(conn, msg_type, cipher_)) break;

            switch ((Msg)msg_type) {
            case Msg::RESET: {
                engine_.reset_context();
                fprintf(stderr, "[dist-server] new conversation - context cleared\n"); fflush(stderr);
                break;
            }
            case Msg::PREFILL: {
                uint32_t pos = 0, n_floats = 0;
                if (!sock_recv_u32_dec(conn, pos, cipher_) ||
                    !sock_recv_u32_dec(conn, n_floats, cipher_)) return;
                if (n_floats != (uint32_t)hidden_dim2) {
                    fprintf(stderr, "[dist-server] PREFILL dim mismatch: got %u want %d\n",
                            n_floats, hidden_dim2);
                    return;
                }
                if (!sock_recv_floats_dec(conn, hidden_buf_.data(), hidden_dim2, cipher_)) return;
                run_second_half(hidden_buf_.data(), (int)pos, false);
                if (!sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_)) return;
                break;
            }
            case Msg::PREFILL_BATCH: {
                uint32_t pos = 0, n_tok = 0, hd = 0;
                if (!sock_recv_u32_dec(conn, pos, cipher_) ||
                    !sock_recv_u32_dec(conn, n_tok, cipher_) ||
                    !sock_recv_u32_dec(conn, hd, cipher_)) return;
                if (hd != (uint32_t)hidden_dim2 || n_tok == 0 || (uint64_t)n_tok * hd > (1ull << 33)) {
                    fprintf(stderr, "[dist-server] PREFILL_BATCH bad header: hidden %u (want %d), tokens %u\n", hd, hidden_dim2, n_tok);
                    return;
                }
                try { batch_buf_.resize((size_t)n_tok * hd); }
                catch (...) { fprintf(stderr, "[dist-server] not enough memory to hold %u prompt tokens of activations (%.1f GB)\n", n_tok, (double)n_tok * hd * 4 / 1e9); return; }
                if (!sock_recv_floats_dec(conn, batch_buf_.data(), (size_t)n_tok * hd, cipher_)) return;
                {
                    static std::vector<int32_t> no_tokens;
                    const int my_layers = n_layers_ - split_layer_;
                    fprintf(stderr, "[dist-server] reading %u prompt tokens through layers %d-%d (each layer loaded once, whole prompt per layer)\n",
                            n_tok, split_layer_, n_layers_ - 1);
                    fflush(stderr);
                    auto t0 = std::chrono::steady_clock::now();
                    // Tell the main computer how far along we are while we work (it is waiting on us).
                    std::atomic<bool> stop{false};
                    std::thread reporter([&] {
                        int last_sent = -1; auto tl = std::chrono::steady_clock::now() - std::chrono::seconds(10);
                        while (!stop.load()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(200));
                            const int done = engine_.range_prefill_layers_done();
                            const auto now = std::chrono::steady_clock::now();
                            if (stop.load()) break;
                            if (done != last_sent && std::chrono::duration<double>(now - tl).count() >= 2.0) {
                                last_sent = done; tl = now;
                                fprintf(stderr, "[progress] reading prompt: %.0f%%  (this computer: layer %d of %d, tokens %u-%u)\n",
                                        100.0 * done / std::max(my_layers, 1), done + 1, my_layers, pos, pos + n_tok - 1);
                                fflush(stderr);
                                nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
                                if (!sock_send_u8_enc(conn, (uint8_t)Msg::PROGRESS, cipher_) ||
                                    !sock_send_u32_enc(conn, (uint32_t)done, cipher_) ||
                                    !sock_send_u32_enc(conn, (uint32_t)my_layers, cipher_)) break;
                            }
                        }
                    });
                    bool ok = false;
                    try { ok = engine_.prefill_layer_range(no_tokens, (int)pos, (int)n_tok, split_layer_, n_layers_, batch_buf_.data(), batch_buf_.data()); }
                    catch (const std::exception& e) { stop = true; reporter.join(); fprintf(stderr, "[dist-server] stopped reading the prompt: %s\n", e.what()); return; }
                    stop = true; reporter.join();
                    if (!ok) {   // not supported here: do it one token at a time, same result, slower
                        for (uint32_t t = 0; t < n_tok; ++t) run_second_half(batch_buf_.data() + (size_t)t * hd, (int)(pos + t), false);
                    }
                    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    fprintf(stderr, "[dist-server] prompt done: %u tokens through layers %d-%d in %.1fs (%.1f tok/s)%s\n",
                            n_tok, split_layer_, n_layers_ - 1, sec, n_tok / std::max(sec, 1e-9), ok ? "" : " [per-token fallback]");
                    fflush(stderr);
                    nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
                }
                if (!sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_)) return;
                break;
            }
            case Msg::HIDDEN: {
                uint32_t pos = 0, n_floats = 0;
                if (!sock_recv_u32_dec(conn, pos, cipher_) ||
                    !sock_recv_u32_dec(conn, n_floats, cipher_)) return;
                if (n_floats != (uint32_t)hidden_dim2) {
                    fprintf(stderr, "[dist-server] HIDDEN dim mismatch: got %u want %d\n",
                            n_floats, hidden_dim2);
                    return;
                }
                if (!sock_recv_floats_dec(conn, hidden_buf_.data(), hidden_dim2, cipher_)) return;
                fprintf(stderr, "[dist-server] processing pos %u through worker layers...\n", pos);
                fflush(stderr);
                run_second_half(hidden_buf_.data(), (int)pos, true);

                // Worker owns the final layers/lm_head, but the primary owns the
                // sampler settings, repetition history, tokenizer/UI, and stop logic.
                // Send logits back; primary samples the next token locally.
                const auto& logits = engine_.logits();
                if (logits.empty()) {
                    fprintf(stderr, "[dist-server] HIDDEN produced empty logits\n");
                    return;
                }
                if (!sock_send_u8_enc(conn, (uint8_t)Msg::LOGITS, cipher_)) return;
                if (!sock_send_u32_enc(conn, (uint32_t)logits.size(), cipher_)) return;
                if (!sock_send_floats_enc(conn, logits.data(), logits.size(), cipher_)) return;
                break;
            }
            case Msg::KV_SAVE: {
                // Primary wants our KV cache bytes (to save conversation).
                // Pure RAM: kvcache_ref().save_to_buffer() -- no disk.
                std::vector<uint8_t> kv_buf;
                engine_.save_kv_to_buffer(kv_buf);
                fprintf(stderr, "[dist-server] conversation saved (%.1f MB of context sent to the main computer)\n", kv_buf.size() / 1048576.0); fflush(stderr);
                uint32_t kv_sz = (uint32_t)kv_buf.size();
                sock_send_u32_enc(conn, kv_sz, cipher_);
                if (kv_sz > 0)
                    sock_send_enc(conn, kv_buf.data(), kv_sz, cipher_);
                break;
            }
            case Msg::KV_LOAD: {
                // Primary is restoring a conversation -- load KV into RAM.
                // Pure RAM: kvcache_ref().load_from_buffer() -- no disk.
                uint32_t kv_sz = 0;
                if (!sock_recv_u32_dec(conn, kv_sz, cipher_)) break;
                if (kv_sz > 0) {
                    std::vector<uint8_t> kv_buf(kv_sz);
                    if (sock_recv_dec(conn, kv_buf.data(), kv_sz, cipher_))
                        engine_.load_kv_from_buffer(kv_buf.data(), kv_sz);
                    fprintf(stderr, "[dist-server] saved conversation loaded (%.1f MB of context)\n", kv_sz / 1048576.0); fflush(stderr);
                }
                sock_send_u8_enc(conn, (uint8_t)Msg::ACK, cipher_);
                break;
            }
            case Msg::KV_LAYER_LOAD: {
                // Hybrid prefill: primary computed this one layer's KV (and state, if the
                // architecture has one) on its own card and is handing it over as soon as it
                // finished -- not waiting for the rest of its layers. Store it and ACK right away
                // so the primary can keep streaming the next layer while this one lands.
                uint32_t layer = 0, sz = 0;
                if (!sock_recv_u32_dec(conn, layer, cipher_) || !sock_recv_u32_dec(conn, sz, cipher_)) break;
                // Heartbeat for every layer received, success or fail -- there was NO log line at
                // all here before on the success path, which is most of why this screen goes
                // silent for minutes at a time with zero indication anything is happening. Printed
                // BEFORE the (possibly slow) apply call below, so a stall during apply itself still
                // shows which layer it's stuck on.
                fprintf(stderr, "[dist-server] hybrid prefill: received layer %u (%u bytes), applying...\n", layer, sz);
                fflush(stderr);
                bool ok = true;
                std::string fail_reason;
                if (sz > 0) {
                    std::vector<uint8_t> buf(sz);
                    ok = sock_recv_dec(conn, buf.data(), sz, cipher_);
                    if (!ok) fail_reason = "socket read of the KV bytes failed/was cut short";
                    if (ok) {
                        // engine_.load_kv_layer_from_buffer() can throw nexus::memguard::OutOfMemory
                        // if this computer is genuinely out of room for the buffer (see kvcache.h's
                        // resize_slot) -- catch it here so a layer that doesn't fit is a clean,
                        // reported failure for THIS layer, not an uncaught exception that would take
                        // the whole worker process down mid-prefill.
                        try { ok = engine_.load_kv_layer_from_buffer((int)layer, buf.data(), sz); }
                        catch (const nexus::memguard::OutOfMemory& e) {
                            fail_reason = std::string("out of memory: ") + e.what();
                            ok = false;
                        }
                        catch (const std::exception& e) {
                            fail_reason = std::string("exception: ") + e.what();
                            ok = false;
                        }
                        if (!ok && fail_reason.empty())
                            fail_reason = "load_kv_layer_from_buffer() returned false (likely a size/layer-geometry mismatch between what was sent and what this computer's KV cache expects for this layer -- see kvcache.h's load_layer_from_buffer)";
                    }
                }
                // *** THE BUG THIS REPLACES: the old code sent Msg::ACK here UNCONDITIONALLY, even
                // when ok was false. The primary's send_worker_kv_layer() only checks the reply byte
                // against Msg::ACK, so a failed layer was invisibly reported as a SUCCESS -- the
                // primary kept right on streaming further layers believing this one landed, while
                // this computer silently had a missing/corrupt KV layer from that point on. That is
                // almost certainly why things went strange *after* a "could not apply layer N" line
                // instead of stopping right there: the failure was already being hidden from the
                // side that could have stopped and reported it cleanly. Now a failed layer replies
                // with a byte that is NOT Msg::ACK, so send_worker_kv_layer() correctly sees it as a
                // failure and the primary's hybrid prefill attempt stops right there instead of
                // pressing on with a broken donor.
                if (!ok) {
                    fprintf(stderr, "[dist-server] hybrid prefill: could not apply layer %u's KV (%u bytes) -- %s\n",
                            layer, sz, fail_reason.c_str());
                    fflush(stderr);
                } else {
                    fprintf(stderr, "[dist-server] hybrid prefill: layer %u applied OK\n", layer);
                    fflush(stderr);
                }
                sock_send_u8_enc(conn, ok ? (uint8_t)Msg::ACK : (uint8_t)0, cipher_);
                break;
            }
            case Msg::BYE:
                return;
            default:
                fprintf(stderr, "[dist-server] Unknown message type %d\n", (int)msg_type);
                return;
            }
        }
    }

    // Run layers [split_layer_, n_layers_) with the given hidden state.
    // If sample=true, also runs lm_head so the caller can call engine_.sample().
    // This is the KEY method -- it uses forward_layer_range() from the transformer.
    //
    // ENGINE INTEGRATION NOTE:
    // ----------------------------------------------------------------------------
    // The cleanest integration is to add these two methods to Engine in engine.h/.cpp:
    //
    //   // Set hidden state directly (for distributed inference -- Machine B entry point)
    //   void set_hidden_state(const float* h, size_t n_floats);
    //
    //   // Run [layer_start, layer_end) + optionally lm_head
    //   void forward_layer_range(const std::vector<int32_t>& dummy_tokens,
    //                            int pos, int layer_start, int layer_end);
    //
    // These are just thin wrappers around Transformer::set_hidden_data() and
    // Transformer::forward_layer_range() which we add in transformer_layer_range.h.
    //
    // With those engine methods, this becomes:
    //   engine_.set_hidden_state(hidden, hidden_dim);
    //   engine_.forward_layer_range({0}, pos, split_layer_, n_layers_);
    //
    // Without patching engine.h yet, we use the workaround below:
    void run_second_half(const float* hidden, int pos, bool /*do_lm_head*/) {
        const int hidden_dim = (int)engine_.model_info().hidden_dim;
        engine_.set_hidden_data(hidden, (size_t)hidden_dim);

        // Empty token vector is intentional: this worker starts from an activation
        // produced by the primary. Keep ORIGINAL layer IDs: if this worker owns
        // source layers 20-39, run 20-40, not 0-20. That keeps KV slots, final
        // lm_head detection, hybrid layer numbering, and all per-layer logic aligned
        // with the full model's actual execution order.
        static const std::vector<int32_t> no_tokens;
        engine_.forward_layer_range(no_tokens, pos, split_layer_, n_layers_);
    }
};

// --- DistSession --------------------------------------------------------------
//
// Runs on Machine A. Manages the TCP connection to Machine B.
// Owns layers [0, split_layer) of the model.
//
// Usage:
//   DistSession session(engine_a, split_layer);
//   session.connect("192.168.1.50", 9876);
//   auto result = session.generate_stream(prompt, cfg, token_cb);
//
class DistSession {
public:
    DistSession(Engine& engine, int split_layer)
        : engine_(engine)
        , split_layer_(split_layer)
        , sock_(INVALID_SOCK)
    {}

    ~DistSession() { disconnect(); }

    // passphrase: must match the worker's passphrase (or empty for no auth)
    // worker_layer_end: the layer_end the worker will run up to (= n_total_layers)
    bool connect(const std::string& host, int port,
                 const std::string& passphrase = "",
                 int worker_layer_end = -1,
                 bool worker_uses_local_model = false) {
        sock_init();
        sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ == INVALID_SOCK) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons((uint16_t)port);
#ifdef _WIN32
        InetPtonA(AF_INET, host.c_str(), &addr.sin_addr);
#else
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
#endif
        // A network flicker right at dial time (the other computer's wifi/link bouncing, or it
        // still finishing its own startup) shows up as this connect() failing outright, not as
        // something sock_send_all/sock_recv_all's retry loop below can help with -- so retry the
        // dial itself a few times with a short backoff before giving up.
        bool connected = false;
        for (int attempt = 0; attempt < 5 && !connected; ++attempt) {
            if (attempt > 0) {
                fprintf(stderr, "[dist] Connect to %s:%d failed, retrying (%d/5)...\n", host.c_str(), port, attempt + 1);
                fflush(stderr);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                sock_close(sock_);
                sock_ = socket(AF_INET, SOCK_STREAM, 0);
                if (sock_ == INVALID_SOCK) break;
            }
            connected = ::connect(sock_, (sockaddr*)&addr, sizeof(addr)) >= 0;
        }
        if (!connected) {
            fprintf(stderr, "[dist] Failed to connect to %s:%d\n", host.c_str(), port);
            sock_close(sock_);
            sock_ = INVALID_SOCK;
            return false;
        }
        sock_set_nodelay(sock_);
        sock_set_keepalive(sock_);
        worker_ready_ = false;
        worker_ready_pending_ = false;
        fprintf(stderr, "[dist] Connected to %s:%d\n", host.c_str(), port);

        // Derive cipher from passphrase -- same key as worker, never transmitted
        cipher_ = XorCipher(passphrase);

        // -- Auth: send NEXUS_MAGIC XOR-encrypted with our key -----------------
        // Worker decrypts with its own key (derived from same passphrase).
        // If keys match -> worker sees NEXUS_MAGIC -> auth passes.
        // If keys differ -> worker sees garbage -> drops connection.
        {
            uint8_t buf[8];
            memcpy(buf, NEXUS_MAGIC, 8);
            cipher_.apply(buf, 8);
            if (!sock_send_all(sock_, buf, 8)) {
                fprintf(stderr, "[dist] Auth send failed\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
        }

        // All subsequent messages are XOR-encrypted.

        // -- MSG_INIT: tell worker its layer range ------------------------------
        int n_layers_total = worker_layer_end > 0 ? worker_layer_end
                           : (int)engine_.model_info().n_layers;
        n_layers_total_ = n_layers_total;   // kept for hybrid prefill, which needs it after connect() returns
        int trace_level = nx_dist_trace_level_from_env();

        // Override knobs: same env vars the primary read at its own load().
        // Forwarded verbatim so the worker installs them before its load.
        std::string rope_layout_str = nx_dist_str_from_env("NEXUS_ROPE_LAYOUT");
        std::string norm_conv_str   = nx_dist_str_from_env("NEXUS_NORM_CONVENTION");

        if (!sock_send_u8_enc(sock_, (uint8_t)Msg::INIT, cipher_) ||
            !sock_send_u32_enc(sock_, (uint32_t)split_layer_, cipher_) ||
            !sock_send_u32_enc(sock_, (uint32_t)n_layers_total, cipher_) ||
            !sock_send_u32_enc(sock_, (uint32_t)trace_level, cipher_) ||
            !sock_send_str_enc(sock_, rope_layout_str, cipher_) ||
            !sock_send_str_enc(sock_, norm_conv_str,   cipher_)) {
            fprintf(stderr, "[dist] Failed to send MSG_INIT\n");
            sock_close(sock_); sock_ = INVALID_SOCK; return false;
        }

        fprintf(stderr, "[dist] Worker NEXUS_TRACE=%d\n", trace_level);
        fprintf(stderr, "[dist] Worker NEXUS_ROPE_LAYOUT=%s\n",
                rope_layout_str.empty() ? "(unset/auto)" : rope_layout_str.c_str());
        fprintf(stderr, "[dist] Worker NEXUS_NORM_CONVENTION=%s\n",
                norm_conv_str.empty() ? "(unset/auto)" : norm_conv_str.c_str());
        fflush(stderr);
        
        {
            uint8_t ack = 0;
            if (!sock_recv_u8_dec(sock_, ack, cipher_) || (Msg)ack != Msg::ACK) {
                fprintf(stderr, "[dist] Worker did not ACK MSG_INIT\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
        }

        if (worker_uses_local_model) {
            fprintf(stderr, "[dist] Requesting worker local-load for layers %d-%d...\n",
                    split_layer_, n_layers_total - 1);
            fflush(stderr);
            if (!sock_send_u8_enc(sock_, (uint8_t)Msg::LOCAL_LOAD, cipher_)) {
                fprintf(stderr, "[dist] Failed to send MSG_LOCAL_LOAD\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
            // Do NOT wait here. The worker may spend minutes mmap/loading its shard.
            // Return to server.cpp so the primary can load its own shard in parallel.
            worker_ready_pending_ = true;
            fprintf(stderr, "[dist] Worker local-load command sent; primary may load in parallel.\n");
            fflush(stderr);
            return true;
        }

        // -- MSG_WEIGHTS: stream layer weights to worker -----------------------
        // Keep this path. This is the no-local-GGUF mode.
        // Important: do NOT stage the shard in a giant RAM blob before sending.
        // Engine::send_layer_subset_gguf already walks the source GGUF mmap and
        // calls write_fn() for header/tensor bytes. Send those bytes directly.
        {
            if (!sock_send_u8_enc(sock_, (uint8_t)Msg::WEIGHTS, cipher_)) {
                fprintf(stderr, "[dist] Failed to send MSG_WEIGHTS\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
            fprintf(stderr, "[dist] Streaming layers %d-%d directly from source GGUF...\n",
                    split_layer_, n_layers_total - 1);
            fflush(stderr);

            bool first_write = true;   // first write_fn call is header_size -- must stay encrypted
            auto t0 = std::chrono::steady_clock::now();
            uint64_t total_sent = 0;
            uint64_t next_log = 256ULL * 1024 * 1024; // log every 256MB

            bool send_ok = engine_.send_layer_subset_gguf(split_layer_, n_layers_total,
                [&](const void* data, size_t n) -> bool {
                    if (first_write) {
                        // header_size prefix (8 bytes) -- receiver reads this with
                        // sock_recv_u64_dec so it MUST be encrypted, not raw.
                        first_write = false;
                        return sock_send_enc(sock_, data, n, cipher_);
                    }

                    // Bulk payload is intentionally raw for speed. Keep cipher in
                    // sync because the receiver also advances by exactly n bytes.
                    cipher_.advance(n);
                    if (!sock_send_all(sock_, data, n)) return false;

                    total_sent += (uint64_t)n;
                    if (total_sent >= next_log) {
                        double sec = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
                        fprintf(stderr, "[dist] %llu MB sent  %.1f MB/s\n",
                            (unsigned long long)total_sent / (1024*1024),
                            (total_sent / (1024.0*1024.0)) / std::max(sec, 0.001));
                        fflush(stderr);
                        next_log = total_sent + 256ULL * 1024 * 1024;
                    }
                    return true;
                });

            if (!send_ok) {
                fprintf(stderr, "[dist] Failed to stream layer weights\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
            uint8_t ack = 0;
            if (!sock_recv_u8_dec(sock_, ack, cipher_) || (Msg)ack != Msg::ACK) {
                fprintf(stderr, "[dist] Worker did not ACK MSG_WEIGHTS\n");
                sock_close(sock_); sock_ = INVALID_SOCK; return false;
            }
            worker_ready_ = true;
            worker_ready_pending_ = false;
            fprintf(stderr, "[dist] Worker ready. Inference can begin.\n");
        }
        return true;
    }

    bool wait_until_ready() {
        if (!is_connected()) return false;
        if (worker_ready_) return true;
        if (!worker_ready_pending_) return true;

        fprintf(stderr, "[dist] Waiting for worker local-load ACK...\n");
        fflush(stderr);
        uint8_t ack = 0;
        if (!sock_recv_u8_dec(sock_, ack, cipher_) || (Msg)ack != Msg::ACK) {
            fprintf(stderr, "[dist] Worker did not ACK MSG_LOCAL_LOAD\n");
            sock_close(sock_);
            sock_ = INVALID_SOCK;
            worker_ready_pending_ = false;
            worker_ready_ = false;
            return false;
        }
        worker_ready_pending_ = false;
        worker_ready_ = true;
        fprintf(stderr, "[dist] Worker loaded local shard. Inference can begin.\n");
        fflush(stderr);
        return true;
    }

    void disconnect() {
        if (sock_ != INVALID_SOCK) {
            sock_send_u8_enc(sock_, (uint8_t)Msg::BYE, cipher_);
            sock_close(sock_);
            sock_ = INVALID_SOCK;
        }
    }

    bool is_connected() const { return sock_ != INVALID_SOCK; }

    // Save worker KV cache to a buffer on the primary (for conversation persistence).
    // Call before save_session() on primary. Pure RAM -- no disk on either side.
    bool save_worker_kv(std::vector<uint8_t>& out) {
        if (!is_connected()) return false;
        if (!sock_send_u8_enc(sock_, (uint8_t)Msg::KV_SAVE, cipher_)) return false;
        uint32_t sz = 0;
        if (!sock_recv_u32_dec(sock_, sz, cipher_) || sz == 0) return false;
        out.resize(sz);
        return sock_recv_dec(sock_, out.data(), sz, cipher_);
    }

    // Restore worker KV cache from buffer (after loading a conversation on primary).
    // Call after load_session() on primary. Pure RAM -- no disk on either side.
    bool load_worker_kv(const std::vector<uint8_t>& data) {
        if (!is_connected() || data.empty()) return false;
        uint32_t sz = (uint32_t)data.size();
        if (!sock_send_u8_enc(sock_, (uint8_t)Msg::KV_LOAD, cipher_)) return false;
        if (!sock_send_u32_enc(sock_, sz, cipher_)) return false;
        if (!sock_send_enc(sock_, (void*)data.data(), sz, cipher_)) return false;
        uint8_t ack = 0;
        sock_recv_u8_dec(sock_, ack, cipher_);
        return ack == (uint8_t)Msg::ACK;
    }

    // Hybrid prefill: hand over ONE layer's KV (buf may be empty -- some layers genuinely have
    // nothing to send, e.g. a linear-attention layer with no state, though that's rare). Blocking:
    // waits for the worker's ACK, same as every other message on this connection. Used from the
    // sender thread in prefill_range_hybrid() below so it can run while the main thread moves on
    // to computing the NEXT layer, instead of the two waiting on each other.
    bool send_worker_kv_layer(int layer, const std::vector<uint8_t>& buf) {
        if (!is_connected()) return false;
        uint32_t sz = (uint32_t)buf.size();
        if (!sock_send_u8_enc(sock_, (uint8_t)Msg::KV_LAYER_LOAD, cipher_)) return false;
        if (!sock_send_u32_enc(sock_, (uint32_t)layer, cipher_)) return false;
        if (!sock_send_u32_enc(sock_, sz, cipher_)) return false;
        if (sz > 0 && !sock_send_enc(sock_, (void*)buf.data(), sz, cipher_)) return false;
        uint8_t ack = 0;
        sock_recv_u8_dec(sock_, ack, cipher_);
        return ack == (uint8_t)Msg::ACK;
    }

    int context_size() const { return (int)tokens_.size(); }

    // Distributed equivalent of Engine::extend_context().
    // No reset, no replay. This advances BOTH KV caches using only new text.
    bool extend_context(const std::string& new_text, bool add_bos = false) {
        if (new_text.empty()) return true;
        if (!is_connected()) return false;
        if (!wait_until_ready()) return false;

        // If generation emitted a token but stopped before that token was used as
        // input to the next forward pass, commit it into both KVs before appending
        // the next user/tool turn.
        if (!commit_pending_token()) return false;

        auto new_toks = engine_.tokenize(new_text, add_bos);
        if (new_toks.empty()) return true;

        const int start = (int)tokens_.size();
        tokens_.insert(tokens_.end(), new_toks.begin(), new_toks.end());

        // All newly appended tokens except the final one are prefill-only.
        // The final token is left as the pending decode input so continue_stream()
        // can run it and get the first assistant logits.
        const int last = (int)tokens_.size() - 1;
        if (!prefill_range(start, last)) return false;

        decode_pos_ = last;
        pending_token_uncommitted_ = false;

        fprintf(stderr, "[dist] Extended distributed context: +%zu token(s), total=%zu, next_pos=%d\n",
                new_toks.size(), tokens_.size(), decode_pos_);
        fflush(stderr);
        return true;
    }

    // Distributed equivalent of Engine::begin_session_warm().run().
    // No reset. No replay. Primary samples from worker-returned LOGITS.
    GenerateResult continue_stream(
        std::function<bool(const std::string&, int32_t)> token_cb,
        const GenerateConfig& cfg)
    {
        GenerateResult result{};
        if (!is_connected()) return result;
        if (!wait_until_ready()) return result;
        if (tokens_.empty() || decode_pos_ < 0 || decode_pos_ >= (int)tokens_.size()) {
            fprintf(stderr, "[dist] continue_stream called with no warm context\n");
            return result;
        }

        const int info_hidden = engine_.model_info().hidden_dim;
        hidden_buf_.resize(info_hidden);

        auto t_start = std::chrono::steady_clock::now();
        std::string generated;
        size_t sent_chars = 0;

        int max_stop = 1;
        for (const auto& s : cfg.stop_strings)
            max_stop = std::max(max_stop, (int)s.size());

        for (int i = 0; i < cfg.max_tokens; ++i) {
            const int pos = decode_pos_;

            // Run the current pending token through the primary slice.
            forward_first_half(tokens_, pos);
            const float* hidden = engine_hidden_data();
            if (!send_hidden(Msg::HIDDEN, (uint32_t)pos, hidden, info_hidden)) {
                fprintf(stderr, "[dist] Send hidden failed at pos %d\n", pos);
                break;
            }

            // Worker owns final layers/lm_head. It returns logits; primary samples.
            uint8_t msg_type = 0;
            if (!sock_recv_u8_dec(sock_, msg_type, cipher_) || (Msg)msg_type != Msg::LOGITS) {
                fprintf(stderr, "[dist] Expected LOGITS, got %d\n", (int)msg_type);
                break;
            }
            uint32_t n_logits = 0;
            if (!sock_recv_u32_dec(sock_, n_logits, cipher_) || n_logits == 0) {
                fprintf(stderr, "[dist] Invalid LOGITS payload\n");
                break;
            }
            std::vector<float> logits(n_logits);
            if (!sock_recv_floats_dec(sock_, logits.data(), logits.size(), cipher_)) {
                fprintf(stderr, "[dist] Failed to receive LOGITS payload\n");
                break;
            }
            int32_t next_tok = sample_logits_on_primary(logits, cfg, tokens_);

            // The token at decode_pos_ has now been committed into both KV caches.
            pending_token_uncommitted_ = false;

            if (next_tok == engine_.eos_token()) {
                result.stopped_eos = true;
                break;
            }

            generated += engine_.detokenize(next_tok);
            ++result.n_tokens;

            bool stop = false;
            size_t stop_at = std::string::npos;
            for (const auto& s : cfg.stop_strings) {
                if (s.empty()) continue;
                size_t from = sent_chars >= s.size() ? sent_chars - s.size() : 0;
                size_t found = generated.find(s, from);
                if (found != std::string::npos) {
                    stop_at = stop_at == std::string::npos ? found : std::min(stop_at, found);
                    stop = true;
                }
            }
            if (stop) {
                if (token_cb && stop_at > sent_chars)
                    token_cb(generated.substr(sent_chars, stop_at - sent_chars), -1);
                generated = generated.substr(0, stop_at);
                result.stopped_string = true;
                break;
            }

            if (token_cb) {
                size_t safe = generated.size() > (size_t)max_stop
                              ? generated.size() - max_stop : 0;
                while (safe > sent_chars && (generated[safe] & 0xC0) == 0x80) --safe;
                if (safe > sent_chars) {
                    bool cont = token_cb(generated.substr(sent_chars, safe - sent_chars), next_tok);
                    sent_chars = safe;
                    if (!cont) break;
                }
            }

            // Append sampled token but do not process it yet. The next iteration
            // processes this token and samples the following one. If generation
            // stops before that happens, extend_context() will commit it first.
            tokens_.push_back(next_tok);
            decode_pos_ = (int)tokens_.size() - 1;
            pending_token_uncommitted_ = true;

            if ((int)tokens_.size() >= engine_.max_context() - 1) {
                result.stopped_limit = true;
                break;
            }
        }

        if (token_cb && !result.stopped_string && sent_chars < generated.size())
            token_cb(generated.substr(sent_chars), -1);

        if (!result.stopped_eos && !result.stopped_string && !result.stopped_limit)
            result.stopped_limit = true;

        auto t_end = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t_end - t_start).count();
        result.elapsed_sec    = sec;
        result.tokens_per_sec = result.n_tokens / std::max(sec, 1e-9);
        result.text           = generated;
        return result;
    }

    // Cold distributed generate for a brand-new prompt.
    // Reset once, prefill all prompt tokens except the final one, then warm-continue.
    GenerateResult generate_stream(
        const std::string& prompt,
        std::function<bool(const std::string&, int32_t)> token_cb,
        const GenerateConfig& cfg)
    {
        GenerateResult result{};
        if (!is_connected()) {
            fprintf(stderr, "[dist] Not connected -- call connect() first\n");
            return result;
        }
        if (!wait_until_ready()) {
            fprintf(stderr, "[dist] Worker is not ready -- generation aborted\n");
            return result;
        }

        bool add_bos = engine_.should_add_bos();
        tokens_ = engine_.tokenize(prompt, add_bos);
        if (tokens_.empty()) return result;

        // ------ DIAG: dump tokenization on the distributed path -----------
        // We're hunting a smooth-degradation generation bug in Qwen3.
        // First confirm whether tokenization is correct (special tokens as
        // single high-id tokens, user content properly BPE-merged).
        if (tokens_.size() <= 64) {
            fprintf(stderr,
                "[diag-dist] PROMPT (%zu chars, add_bos=%d):\n--BEGIN--\n%s\n--END--\n",
                prompt.size(), add_bos ? 1 : 0, prompt.c_str());

            fprintf(stderr,
                "[diag-dist] TOKENS (%zu total):\n", tokens_.size());
            for (size_t i = 0; i < tokens_.size(); ++i) {
                std::string ts = engine_.detokenize(tokens_[i]);
                std::string vis;
                for (char c : ts) {
                    if (c == '\n')      vis += "\\n";
                    else if (c == '\t') vis += "\\t";
                    else if (c == '\r') vis += "\\r";
                    else if ((unsigned char)c < 0x20) {
                        char buf[8];
                        snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
                        vis += buf;
                    } else vis += c;
                }
                fprintf(stderr,
                    "[diag-dist]   [%4zu] id=%6d \"%s\"\n",
                    i, tokens_[i], vis.c_str());
            }
            fflush(stderr);
        }
        // ------ END DIAG --------------------------------------------------

        engine_.reset_context();
        sock_send_u8_enc(sock_, (uint8_t)Msg::RESET, cipher_);

        const int info_hidden = engine_.model_info().hidden_dim;
        hidden_buf_.resize(info_hidden);

        const int last = (int)tokens_.size() - 1;
        fprintf(stderr, "[dist] Cold distributed prefill: %d token(s), leaving pos %d for first logits...",
                last, last);
        fflush(stderr);
        auto t0 = std::chrono::steady_clock::now();

        if (!prefill_range(0, last)) return result;

        decode_pos_ = last;
        pending_token_uncommitted_ = false;

        double prefill_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, " done (%.0f ms, %.1f tok/s)\n",
                prefill_sec * 1000.0,
                last / std::max(prefill_sec, 1e-9));
        fflush(stderr);

        return continue_stream(std::move(token_cb), cfg);
    }

    // Set before connect(). "path" is this computer's own local copy of the model file (the same
    // file the other computer loads its layers from) -- required for hybrid prefill, which loads a
    // temporary second copy of the OTHER computer's layers here. Leave unset / pass "" to keep the
    // normal distributed prefill path (hybrid mode is then simply never attempted). Called from
    // tools/server.cpp right after construction, so these two need to be public.
    void set_local_model_path(const std::string& path) { local_model_path_ = path; }
    void set_hybrid_prefill(bool on) { hybrid_prefill_ = on; }

private:
    Engine&         engine_;
    int             split_layer_;
    int             n_layers_total_ = 0;
    std::string     local_model_path_;             // this computer's own copy of the model file; set via set_local_model_path()
    bool            hybrid_prefill_ = false;        // set via set_hybrid_prefill()
    socket_t        sock_;
    XorCipher       cipher_;
    std::vector<float> hidden_buf_;
    std::vector<float> batch_buf_;
    uint32_t prog_total_ = 0;
    std::mt19937     rng_{std::random_device{}()};
    bool            worker_ready_ = false;
    bool            worker_ready_pending_ = false;

    // Distributed warm-session state. Mirrors the primary Engine warm KV position,
    // but keeps the worker KV in lockstep instead of replaying the whole prompt.
    std::vector<int32_t> tokens_;
    int                  decode_pos_ = -1;
    bool                 pending_token_uncommitted_ = false;

    int32_t sample_logits_on_primary(const std::vector<float>& remote_logits,
                                     const GenerateConfig& cfg,
                                     const std::vector<int32_t>& recent_tokens) {
        if (remote_logits.empty()) return 0;
        if (cfg.temperature <= 0.0f || cfg.temperature < 1e-6f) {
            return (int32_t)(std::max_element(remote_logits.begin(), remote_logits.end()) - remote_logits.begin());
        }

        std::vector<float> logits = remote_logits;
        const int vsz = (int)logits.size();

        const int n_rep = std::min((int)recent_tokens.size(), cfg.repeat_last_n);
        if (n_rep > 0 && (cfg.repeat_penalty != 1.0f || cfg.frequency_penalty != 0.0f || cfg.presence_penalty != 0.0f)) {
            std::unordered_map<int32_t, int> freq;
            for (int i = (int)recent_tokens.size() - n_rep; i < (int)recent_tokens.size(); ++i) {
                int32_t tok = recent_tokens[i];
                if (tok >= 0 && tok < vsz) freq[tok]++;
            }
            for (auto& kv : freq) {
                int32_t tok = kv.first;
                int cnt = kv.second;
                if (cfg.repeat_penalty != 1.0f) {
                    if (logits[tok] > 0.0f) logits[tok] /= cfg.repeat_penalty;
                    else                    logits[tok] *= cfg.repeat_penalty;
                }
                if (cfg.frequency_penalty != 0.0f) logits[tok] -= cfg.frequency_penalty * cnt;
                if (cfg.presence_penalty  != 0.0f) logits[tok] -= cfg.presence_penalty;
            }
        }

        const float inv_temp = 1.0f / std::max(cfg.temperature, 1e-6f);
        for (float& v : logits) v *= inv_temp;

        if (cfg.top_k > 0 && cfg.top_k < vsz) {
            std::vector<int> idx(vsz);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + cfg.top_k, idx.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });
            float thresh = logits[idx[cfg.top_k - 1]];
            for (float& v : logits) if (v < thresh) v = -1e9f;
        }

        float max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        for (float& v : logits) {
            v = std::exp(v - max_logit);
            sum += v;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) {
            return (int32_t)(std::max_element(remote_logits.begin(), remote_logits.end()) - remote_logits.begin());
        }
        for (float& v : logits) v = (float)(v / sum);

        if (cfg.min_p > 0.0f) {
            float max_p = *std::max_element(logits.begin(), logits.end());
            float cutoff = cfg.min_p * max_p;
            for (float& p : logits) if (p < cutoff) p = 0.0f;
        }

        if (cfg.top_p < 1.0f) {
            std::vector<std::pair<float,int>> sorted;
            sorted.reserve(vsz);
            for (int i = 0; i < vsz; ++i) if (logits[i] > 0.0f) sorted.push_back({logits[i], i});
            std::sort(sorted.begin(), sorted.end(), [](auto& a, auto& b){ return a.first > b.first; });
            float cumsum = 0.0f;
            size_t cutoff = sorted.size();
            for (size_t i = 0; i < sorted.size(); ++i) {
                cumsum += sorted[i].first;
                if (cumsum >= cfg.top_p) { cutoff = i + 1; break; }
            }
            std::fill(logits.begin(), logits.end(), 0.0f);
            for (size_t i = 0; i < cutoff; ++i) logits[sorted[i].second] = sorted[i].first;
        }

        if (cfg.seed >= 0) rng_.seed((unsigned)cfg.seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        float r = uniform(rng_);
        float cumsum = 0.0f;
        int32_t chosen = 0;
        for (int32_t i = 0; i < vsz; ++i) {
            if (logits[i] <= 0.0f) continue;
            cumsum += logits[i];
            if (r <= cumsum) { chosen = i; break; }
            chosen = i;
        }
        return chosen;
    }

    // Read positions [from, to) on both machines. Whole chunks of tokens go through this machine's layers
    // together (batched, GPU-accelerated when prompt acceleration is on), the hidden states are shipped in one
    // message, and the other machine does the same - while this one already works on the next chunk.
    bool prefill_range(int from, int to) {
        if (to <= from) return true;
        // Hybrid prefill has real fixed costs of its own (spinning up a loader thread, holding at
        // least one of the donor's layers in RAM, a background sender thread) that a short prompt
        // never earns back -- a 20-token prompt gains nothing from any of this. Below the
        // threshold, just skip straight to the normal distributed path for THIS call, without
        // touching hybrid_prefill_ itself, so a short early message in a conversation doesn't
        // permanently turn hybrid prefill off for a long one that comes later.
        if (hybrid_prefill_ && !local_model_path_.empty() && (to - from) >= hybrid_prefill_min_tokens()) {
            // *** v28: "we should never have to reprocess -- we process the layers and hold them
            // until we are ready for them. We are trying to optimize compute, not throw it away."
            // prefill_range_hybrid() now hands back the hidden states it already computed for
            // THIS computer's own layers [0, split_layer_) across the whole [from,to) range,
            // plus whether that part actually finished -- even when the function as a whole
            // returns false because something AFTER that (opening the ghost engine, streaming a
            // donor layer, the donor rejecting a layer's KV) failed. That own-layer work is the
            // expensive part (real GPU/CPU compute across the whole prompt); if it's already
            // done, there is no reason to burn it and recompute it a second time. Hand the SAME
            // buffer straight to the other computer over the ordinary network path instead of
            // restarting the whole range from scratch via prefill_range_normal().
            std::vector<float> precomputed; bool own_layers_done = false;
            if (prefill_range_hybrid(from, to, precomputed, own_layers_done)) return true;
            if (own_layers_done && precomputed.size() == (size_t)(to - from) * (size_t)std::max(0, (int)engine_.model_info().hidden_dim)) {
                fprintf(stderr, "[dist] hybrid prefill: the handoff to the other computer did not finish, but this computer's own layers for these %d tokens were already computed -- sending that work over instead of recomputing it\n",
                        to - from);
                fflush(stderr);
                if (send_buf_to_donor(from, to, precomputed.data())) { hybrid_prefill_ = false; return true; }
                fprintf(stderr, "[dist] hybrid prefill: handing off the already-computed layers failed too -- reprocessing this range normally\n");
            } else {
                fprintf(stderr, "[dist] hybrid prefill is not available for this prompt -- using the normal distributed path instead\n");
            }
            fflush(stderr);
            hybrid_prefill_ = false;   // this model/prompt/memory combination didn't work; don't retry it every chunk
        }
        return prefill_range_normal(from, to);
    }

    // *** v28: used when hybrid prefill already computed this computer's own layers for
    // [from,to) (see prefill_range_hybrid()'s own_layers_done out-param) but failed afterward --
    // sends those SAME hidden states to the other computer over the ordinary chunked
    // PREFILL_BATCH path, instead of recomputing them with engine_.prefill_layer_range() again.
    // Deliberately self-contained (does not reuse prefill_range_normal()'s private lambdas) so
    // it can be called from prefill_range() without restructuring that function's locals.
    bool send_buf_to_donor(int from, int to, const float* buf) {
        if (to <= from) return true;
        const int H = engine_.model_info().hidden_dim;
        const int total = to - from;
        int CH = 1024;
        if (const char* e = std::getenv("NEXUS_DIST_PREFILL_CHUNK")) { if (*e) { int v = atoi(e); CH = v <= 0 ? total : std::max(v, 2); } }
        CH = std::min(CH, total);
        const auto t0 = std::chrono::steady_clock::now();
        fprintf(stderr, "[dist] sending %d already-computed token(s) to the other computer in pieces of up to %d (no recompute on this side)\n", total, CH);
        fflush(stderr);
        auto drain_one = [&]() -> bool {
            uint8_t m = 0;
            for (;;) {
                if (!sock_recv_u8_dec(sock_, m, cipher_)) { fprintf(stderr, "[dist] lost the other computer while handing off already-computed layers\n"); return false; }
                if ((Msg)m == Msg::PROGRESS) { uint32_t d = 0, t = 0; if (!sock_recv_u32_dec(sock_, d, cipher_) || !sock_recv_u32_dec(sock_, t, cipher_)) return false; continue; }
                if ((Msg)m == Msg::ACK) return true;
                fprintf(stderr, "[dist] unexpected reply %d handing off already-computed layers\n", (int)m); return false;
            }
        };
        int outstanding = 0;
        for (int p = from; p < to; p += CH) {
            int n = std::min(CH, to - p);
            const float* chunk = buf + (size_t)(p - from) * (size_t)H;
            if (!(sock_send_u8_enc(sock_, (uint8_t)Msg::PREFILL_BATCH, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)p, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)n, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)H, cipher_)
                  && sock_send_floats_enc(sock_, chunk, (size_t)n * (size_t)H, cipher_))) return false;
            ++outstanding;
            if (outstanding >= 2) { if (!drain_one()) return false; --outstanding; }
        }
        while (outstanding > 0) { if (!drain_one()) return false; --outstanding; }
        fprintf(stderr, "[dist] already-computed layers handed off (%.0fs)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        fflush(stderr);
        return true;
    }

    // NEXUS_HYBRID_PREFILL_MIN_TOKENS: how many tokens a read has to be before hybrid prefill is
    // even considered worth it. Unset = 2048, a conservative "clearly a real prompt, not a quick
    // question" cutoff -- adjustable either way: set it lower if even medium prompts are worth it
    // on your hardware, or higher if you'd rather only ever use this for the big ones.
    static int hybrid_prefill_min_tokens() {
        if (const char* e = std::getenv("NEXUS_HYBRID_PREFILL_MIN_TOKENS")) {
            int v = atoi(e);
            if (v >= 0) return v;
        }
        return 2048;
    }

    // HYBRID PREFILL: instead of shipping this computer's hidden states to the other computer for
    // every chunk during prefill, stream TEMPORARY copies of the OTHER computer's layers here (from
    // the same local model file this computer already has) and run that range locally too -- on
    // this computer's own graphics card, whatever its speed relative to the other computer's. The
    // two ranges are run back to back, in the same process, with no network round trip for either
    // of them. As each of the other computer's layers finishes, its KV (and, for architectures with
    // a recurrent state such as Qwen3.5's linear-attention layers, that layer's small fixed-size
    // state) is handed off right away over a per-layer message (Msg::KV_LAYER_LOAD) on a background
    // sender thread, so that network transfer of layer L overlaps with this computer still computing
    // layer L+1 -- no waiting for the whole donor range to finish before anything crosses the wire,
    // and no single giant blob at the end. The other computer's existing decode path doesn't know or
    // care that its layers arrived one at a time instead of all at once.
    // Decode afterwards is completely unchanged: the other computer already has a correctly
    // populated cache for its own layers, computed on this computer's card, and answers per-token
    // requests exactly as it does in the normal distributed path.
    //
    // *** v20: REPLACES v16/v17/v19's windowing entirely. Every earlier version of this had to
    // guess (or measure, then guess how to apply the measurement to) how many of the donor's
    // layers to hold in memory at once as a "window" or "batch" -- v17 guessed too small a fixed
    // cost and multiplied it by a deep prefetch queue; v19 fixed that guess but still double-
    // counted memory during the overlap between one window finishing and the next one prefetching.
    // Both were windowing variants of the same idea, and both nearly froze the machine on real
    // hardware. There is no window anymore. Engine::open_for_layer_streaming() opens the donor
    // model file ONCE -- global tensors (embed_tokens/lm_head/norm) + tokenizer, paid once for the
    // whole prefill, not once per window -- and Engine::swap_layer(l) streams in exactly ONE
    // layer's weights at a time, freeing whichever layer was resident before it. The loop below
    // calls swap_layer() once per donor layer: load layer l's weights (a single memcpy from the
    // still-open mmap, no file reopen, no tokenizer reload, no MemoryManager re-creation), run it,
    // extract and queue its KV for sending, then move straight to layer l+1 -- which frees layer l's
    // weights as part of loading layer l+1's. Never more than one donor layer's weights resident at
    // any instant, by construction, not by a sizing formula. The only "big" cost left -- opening the
    // ghost Engine at all -- is measured directly against real free memory (nexus::memguard::
    // available_bytes() before and after the open call) instead of estimated, for the same reason
    // probe_hybrid_layer_cost() existed in v19: a model's global tensors can run into the GB range,
    // and guessing that number is exactly what caused both previous near-crashes.
    // *** v28: buf/own_layers_done are now out-params instead of a local variable, so a caller
    // whose later steps fail can still recover the hidden states already computed for this
    // computer's own layers (see prefill_range()'s "never reprocess" fallback above) instead of
    // the whole [from,to) range being silently discarded and recomputed from scratch.
    bool prefill_range_hybrid(int from, int to, std::vector<float>& buf, bool& own_layers_done) {
        own_layers_done = false;
        if (!engine_.range_prefill_ok()) return false;      // this architecture can't batch-prefill at all (e.g. Gemma PLE) -- hybrid needs it on both halves
        const int all_layers = n_layers_total_ > 0 ? n_layers_total_ : (int)engine_.model_info().n_layers;
        if (split_layer_ <= 0 || split_layer_ >= all_layers) return false;   // nothing on the other computer to take over, or nothing here to run first
        const int total = to - from;
        if (total <= 0) return true;

        // One-time hidden-state handoff buffer, sized for the WHOLE prompt (no chunking here on
        // purpose: with both ranges running on this ONE computer there is no second computer to
        // overlap with by pipelining pieces -- that's what NEXUS_DIST_PREFILL_CHUNK is for in the
        // normal path).
        const int H = engine_.model_info().hidden_dim;
        try { buf.resize((size_t)total * (size_t)H); }
        catch (...) {
            fprintf(stderr, "[dist] hybrid prefill: not enough memory to hold activations for all %d tokens at once (%.1f GB)\n",
                    total, (double)total * H * 4 / 1e9); fflush(stderr);
            return false;
        }

        fprintf(stderr, "[dist] hybrid prefill: reading %d prompt token(s), ALL %d layers on this computer (0-%d as usual, %d-%d standing in for the other computer, streamed in one real layer at a time and sent off as each finishes)\n",
                total, all_layers, split_layer_ - 1, split_layer_, all_layers - 1);
        fflush(stderr);
        // *** v30: this used to be quiet(true) on the theory that "the distributed code prints
        // its own" progress -- true for the donor-streaming loop further down (it prints a line
        // per layer), but NOT true for this computer's OWN layers, computed right below, which
        // can easily be the biggest single silent stretch of a hybrid-prefill prompt (it's
        // everything up to split_layer_, computed before any donor-streaming progress line is
        // possible). Quiet here meant total silence for that whole stretch -- long enough to
        // repeatedly trip the server's ">14s with no progress" heartbeat ("still working on your
        // prompt ... no first word yet"), which is the flashing, no-information bar the launcher
        // showed instead of real progress. False lets the engine's own per-layer gauge (same one
        // the single-computer case already uses) print real "[progress] reading prompt: NN% ..."
        // lines through this phase too -- both ending the long silence AND giving a real layer
        // count/percentage to look at, not just a moving animation.
        engine_.set_range_prefill_quiet(false);
        struct QuietOff { Engine& a; ~QuietOff() { a.set_range_prefill_quiet(false); } } quiet_off{engine_};
        const auto t0 = std::chrono::steady_clock::now();

        if (!engine_.prefill_layer_range(tokens_, from, total, 0, split_layer_, nullptr, buf.data())) {
            fprintf(stderr, "[dist] hybrid prefill: this computer's own range failed\n"); fflush(stderr);
            return false;
        }
        // From here on, buf holds real, correct hidden states for THIS computer's own layers
        // across the whole [from,to) range -- any failure past this point no longer needs to
        // throw that work away; the caller can hand buf straight to the donor instead. IMPORTANT:
        // the streaming loop below advances activations THROUGH each donor layer it streams in
        // (ghost.prefill_layer_range() below both reads and writes its working buffer in place),
        // so buf itself must stop being that working buffer right here, or by the time a later
        // layer fails, buf would hold activations already carried a few donor layers forward --
        // sending THAT to the real donor would apply those layers a second time. `work` below is
        // the mutable copy the ghost loop advances; buf stays exactly as it is at this moment:
        // the clean, correct hand-off point for layer split_layer_, untouched by anything after.
        own_layers_done = true;
        std::vector<float> work = buf;
        fprintf(stderr, "[dist] hybrid prefill: this computer's own layers done (%.0fs). Streaming layers %d-%d to the other computer, one real layer in memory at a time...\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), split_layer_, all_layers - 1);
        fflush(stderr);

        // Open the ghost Engine ONCE: file header + global tensors (embed_tokens, lm_head, norms) +
        // tokenizer -- no donor layer's weights yet. This is the only cost here bigger than a
        // single layer, and it's measured against real free memory, not guessed.
        unsigned thr = std::max(1u, std::thread::hardware_concurrency());
        const long long before_open = nexus::memguard::available_bytes();
        // *** THE ACTUAL CAUSE of CPU cores pegging and minutes-per-layer: a freshly
        // constructed Engine defaults turbo_prefill_enabled=false. Engine::prefill_layer_range()
        // computes `turbo = impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream()`
        // -- with that false, EVERY ghost layer took the CPU path (prefill_cpu_prompt_batch()),
        // never the GPU path, no matter how fast this computer's card is. The whole point of
        // hybrid prefill is running the donor's layers on THIS computer's GPU; without this, it
        // was quietly running them on the CPU instead, which is exactly consistent with cores
        // pegging the instant donor-layer compute started while the network transfer itself
        // (confirmed separately) was only ~30 seconds. Mirrors whatever turbo setting the
        // primary's own engine_ is already using, so hybrid prefill behaves consistently with
        // however this computer's own prefill is configured.
        Engine ghost;
        ghost.set_turbo_prefill(engine_.turbo_prefill_enabled());
        // Pass the REAL donor range so open_for_layer_streaming() can probe each layer's
        // actual KV geometry (linear/recurrent vs full-attention, real kv_heads/head_dim)
        // before sizing the KV cache -- without this, an architecture that mixes attention
        // types per layer (Qwen3Next/Qwen3.6's linear-attention layers) gets a uniform,
        // WRONG cache shape and every layer whose real shape differs fails to apply on the
        // donor's side. See the v22 patch note on Transformer::open_for_layer_streaming().
        if (!ghost.open_for_layer_streaming(local_model_path_, (int)thr, split_layer_, all_layers)) {
            fprintf(stderr, "[dist] hybrid prefill: could not open the other computer's model for layer streaming\n"); fflush(stderr);
            return false;
        }
        double measured_fixed_bytes = 0.0;
        if (before_open >= 0) {
            const long long after_open = nexus::memguard::available_bytes();
            if (after_open >= 0) {
                measured_fixed_bytes = (double)(before_open - after_open);
                fprintf(stderr, "[dist] hybrid prefill: ghost engine opened (global tensors + tokenizer) -- measured %.0f MB, real not guessed\n",
                        measured_fixed_bytes / 1e6);
                fflush(stderr);
            }
        }

        // Final check against the standard memory guard: the measured fixed cost just paid above,
        // PLUS one donor layer's weights (never more than one resident -- that is the whole point
        // of streaming layer-by-layer instead of in windows), PLUS a full KV cache for one layer
        // across the whole prompt. A no-op when the guard is disabled, same as everywhere else.
        //
        // *** v29: used to just give up here if it didn't fit. The layer-by-layer streaming
        // framework underneath this (swap_layer_direct()'s free/load pair) is solid enough now to
        // lean on for this too, not just for the ghost: this computer already owns layers [0,
        // split_layer_) loaded and resident for its own use. If there genuinely isn't room, free
        // some of THOSE -- starting from the one closest to the donor's range and working
        // backward -- to make room for the donor layers' streaming+compute, then reload every one
        // of them before this function returns (see freed_primary_layers below; the reload is
        // timed to overlap with the tail of the network handoff rather than sit idle). Decode
        // afterward needs every one of this computer's own layers back, so this function does not
        // return -- success OR failure -- with any of them still missing.
        std::vector<int> freed_primary_layers;
        auto reload_freed_primary_layers_now = [&]() -> bool {
            bool all_ok = true;
            for (int l : freed_primary_layers) {
                if (!engine_.reload_own_layer(l)) {
                    fprintf(stderr, "[dist] hybrid prefill: FATAL -- could not reload this computer's own layer %d after freeing it to make room; decode on this computer will fail until it is restarted\n", l);
                    all_ok = false;
                }
            }
            fflush(stderr);
            freed_primary_layers.clear();
            return all_ok;
        };
        {
            const auto& mi = engine_.model_info();
            const double bytes_per_layer_weights = mi.n_layers > 0 ? (double)mi.file_size_bytes / mi.n_layers : 0.0;
            const double kv_per_token_per_layer = (double)mi.n_kv_heads * mi.head_dim * 2.0 /*K+V*/ * 2.0 /*f16 bytes*/;
            const size_t estimate = (size_t)(measured_fixed_bytes + bytes_per_layer_weights
                                              + kv_per_token_per_layer * std::max(1, total));
            bool guard_ok = false;
            try { nexus::memguard::require(estimate, "hybrid prefill: the ghost engine plus one streamed donor layer"); guard_ok = true; }
            catch (const nexus::memguard::OutOfMemory&) {}
            for (int l = split_layer_ - 1; !guard_ok && l >= 0; --l) {
                if (!engine_.free_own_layer(l)) continue;   // couldn't free this one (already free, or not actually ours) -- try the next
                freed_primary_layers.push_back(l);
                fprintf(stderr, "[dist] hybrid prefill: memory is tight -- freed this computer's own layer %d to make room (%zu freed so far)\n", l, freed_primary_layers.size());
                fflush(stderr);
                try { nexus::memguard::require(estimate, "hybrid prefill: the ghost engine plus one streamed donor layer"); guard_ok = true; }
                catch (const nexus::memguard::OutOfMemory&) {}
            }
            if (!guard_ok) {
                fprintf(stderr, "[dist] hybrid prefill: memory guard refused even after freeing every one of this computer's own layers -- not enough free memory for even one streamed layer right now\n");
                fflush(stderr);
                reload_freed_primary_layers_now();
                return false;
            }
        }

        // A separate thread sends each finished layer while THIS thread moves straight on to
        // streaming in and computing the next one -- the network transfer of layer L overlaps with
        // the GPU work on layer L+1 instead of happening only after everything is done.
        struct SendJob { int layer; std::vector<uint8_t> buf; };
        std::deque<SendJob> q; std::mutex qm; std::condition_variable qcv;
        bool producer_done = false; std::atomic<bool> send_failed{false}; int failed_layer = -1;
        std::thread sender([&] {
            for (;;) {
                SendJob job;
                { std::unique_lock<std::mutex> lk(qm);
                  qcv.wait(lk, [&] { return !q.empty() || producer_done; });
                  if (q.empty() && producer_done) return;
                  job = std::move(q.front()); q.pop_front(); }
                if (send_failed.load()) continue;   // already broken -- just drain so the producer never blocks on a full queue
                if (!send_worker_kv_layer(job.layer, job.buf)) { send_failed.store(true); failed_layer = job.layer; }
            }
        });

        bool ok = true;
        for (int l = split_layer_; l < all_layers && ok; ++l) {
            if (send_failed.load()) { ok = false; break; }
            // Streams in layer l's weights, freeing whichever layer was resident before it (a
            // no-op the very first time through). Never more than one donor layer's weights alive.
            if (!ghost.swap_layer(l)) {
                fprintf(stderr, "[dist] hybrid prefill: could not stream in layer %d\n", l); fflush(stderr);
                ok = false; break;
            }
            // keep_prefill_scratch=true for every layer except the last: the prompt-sized
            // scratch buffers (and GPU staging buffers) get grown ONCE and freed ONCE, the
            // same as if this whole donor range had been done in a single batched call,
            // instead of being torn down and regrown on every single layer -- which is what
            // was actually making each streamed layer take minutes instead of seconds.
            const bool keep_scratch = (l != all_layers - 1);
            if (!ghost.prefill_layer_range(tokens_, from, total, l, l + 1, work.data(), work.data(), keep_scratch)) {
                fprintf(stderr, "[dist] hybrid prefill: layer %d failed on the streamed copy\n", l); fflush(stderr);
                ok = false; break;
            }
            std::vector<uint8_t> kv_buf;
            ghost.save_kv_layer_to_buffer(l, kv_buf);
            ghost.free_kv_layer(l);   // sent (or queued to be) -- this layer's KV is no longer needed here
            { std::lock_guard<std::mutex> lk(qm); q.push_back({l, std::move(kv_buf)}); }
            qcv.notify_one();
            fprintf(stderr, "[progress] reading prompt: %.0f%%  (layer %d of %d done here, handed to the other computer, %.0fs elapsed)\n",
                    100.0 * (l - split_layer_ + 1) / (all_layers - split_layer_), l - split_layer_ + 1, all_layers - split_layer_,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            fflush(stderr);
            nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
        }

        { std::lock_guard<std::mutex> lk(qm); producer_done = true; } qcv.notify_one();
        // *** v29: if any of this computer's own layers were freed above to make room, bring them
        // back now, on a thread of their own, running WHILE sender.join() below still waits for
        // the last layer or two to finish crossing the wire / being applied on the other computer
        // -- exactly the dead time the donor's own KV-apply lag otherwise leaves this computer's
        // GPU idle for. Still always joined (success or failure, ok or not) before this function
        // returns by any path below: decode right after this needs every one of this computer's
        // own layers back, with nothing missing.
        std::thread reload_thread;
        if (!freed_primary_layers.empty()) {
            fprintf(stderr, "[dist] hybrid prefill: reloading %zu of this computer's own layer(s) now, while the last of the handoff finishes\n", freed_primary_layers.size());
            fflush(stderr);
            reload_thread = std::thread([&] { reload_freed_primary_layers_now(); });
        }
        sender.join();
        if (reload_thread.joinable()) reload_thread.join();
        if (!ok || send_failed.load()) {
            if (send_failed.load()) fprintf(stderr, "[dist] hybrid prefill: the other computer did not accept layer %d's KV\n", failed_layer);
            fflush(stderr);
            return false;
        }
        fprintf(stderr, "[dist] hybrid prefill: done. The other computer has every layer it needs (%.0fs total).\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        fflush(stderr);
        return true;
    }

    bool prefill_range_normal(int from, int to) {
        if (to <= from) return true;
        if (!engine_.range_prefill_ok() || to - from == 1) {
            for (int pos = from; pos < to; ++pos) if (!prefill_token(pos)) return false;
            return true;
        }
        const int H = engine_.model_info().hidden_dim;
        const int total = to - from;
        // Chunk size (tokens). NEXUS_DIST_PREFILL_CHUNK: 0 = whole prompt in one piece (every layer loads once per
        // computer); N = pieces of N tokens, so the second computer starts while the first works on the next piece.
        int CH = 1024;
        if (const char* e = std::getenv("NEXUS_DIST_PREFILL_CHUNK")) { if (*e) { int v = atoi(e); CH = v <= 0 ? total : std::max(v, 2); } }
        CH = std::min(CH, total);
        const int n_chunks = (total + CH - 1) / CH;
        const int my_layers = split_layer_;
        const int all_layers = (int)engine_.model_info().n_layers;
        const auto t0 = std::chrono::steady_clock::now();
        auto t_print = t0 - std::chrono::seconds(10);
        std::vector<float> buf;
        try { buf.resize((size_t)CH * (size_t)H); }
        catch (...) { fprintf(stderr, "[dist] not enough memory to hold %d prompt tokens of activations (%.1f GB)\n", CH, (double)CH * H * 4 / 1e9); return false; }
        fprintf(stderr, "[dist] reading %d prompt tokens in %d piece(s) of up to %d: layers 0-%d here, layers %d-%d on the other computer%s\n",
                total, n_chunks, CH, my_layers - 1, my_layers, all_layers - 1, n_chunks > 1 ? " (both work at the same time)" : "");
        fflush(stderr);
        engine_.set_range_prefill_quiet(n_chunks > 1);     // per-piece layer progress would just jump back to 0 each piece
        struct QuietOff { Engine& e; ~QuietOff() { e.set_range_prefill_quiet(false); } } quiet_off{engine_};
        int outstanding = 0, sent_tok = 0, acked_tok = 0, chunk_no = 0;
        std::deque<int> inflight;      // sizes of pieces sent but not yet finished by the other computer
        auto show = [&](int donor_layer, int donor_tot, bool force) {
            const auto now = std::chrono::steady_clock::now();
            if (!force && std::chrono::duration<double>(now - t_print).count() < 2.0) return;
            t_print = now;
            // Two stages: a token is half done once it left this computer, done when the other computer has it.
            double frac = 0.5 * (double)sent_tok / total + 0.5 * (double)acked_tok / total;
            if (n_chunks == 1 && donor_tot > 0) frac = ((double)my_layers + donor_layer) / (double)std::max(all_layers, 1);
            const double el = std::chrono::duration<double>(now - t0).count();
            char eta[48] = "";
            if (frac > 0.02 && el > 4.0) { long long rem = (long long)(el / frac * (1.0 - frac));
                if (rem >= 3600) snprintf(eta, sizeof eta, ", about %lldh %lldm left", rem / 3600, (rem % 3600) / 60);
                else if (rem >= 60) snprintf(eta, sizeof eta, ", about %lldm %llds left", rem / 60, rem % 60); else snprintf(eta, sizeof eta, ", about %llds left", rem); }
            fprintf(stderr, "[progress] reading prompt: %.0f%%  (%d of %d tokens passed this computer, %d done on the other, %.0fs elapsed%s)\n",
                    frac * 100.0, sent_tok, total, acked_tok, el, eta);
            fflush(stderr);
            nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
        };
        // Wait for the oldest outstanding piece to be finished by the other computer (relaying its progress meanwhile).
        auto wait_one = [&]() -> bool {
            for (;;) {
                uint8_t m = 0;
                if (!sock_recv_u8_dec(sock_, m, cipher_)) { fprintf(stderr, "[dist] lost the other computer while it was reading the prompt\n"); return false; }
                if ((Msg)m == Msg::PROGRESS) {
                    uint32_t done = 0, tot = 0;
                    if (!sock_recv_u32_dec(sock_, done, cipher_) || !sock_recv_u32_dec(sock_, tot, cipher_)) return false;
                    show((int)done, (int)tot, false);
                    continue;
                }
                if ((Msg)m == Msg::ACK) break;
                fprintf(stderr, "[dist] unexpected reply %d while waiting for the other computer\n", (int)m);
                return false;
            }
            --outstanding; if (!inflight.empty()) { acked_tok += inflight.front(); inflight.pop_front(); } return true;
        };
        for (int p = from; p < to; p += CH) {
            int n = std::min(CH, to - p);
            ++chunk_no;
            if (n == 1) {                                  // lone token: plain path (same order, so drain first)
                while (outstanding > 0) { if (!wait_one()) return false; }
                if (!prefill_token(p)) return false;
                sent_tok += 1; acked_tok += 1; continue;
            }
            if (!engine_.prefill_layer_range(tokens_, p, n, 0, split_layer_, nullptr, buf.data())) {
                fprintf(stderr, "[dist] batched prefill not available for this model here; falling back to one token at a time\n");
                while (outstanding > 0) { if (!wait_one()) return false; }
                for (int q = p; q < to; ++q) if (!prefill_token(q)) return false;
                return true;
            }
            sent_tok += n;
            show(0, 0, chunk_no == 1);
            if (!(sock_send_u8_enc(sock_, (uint8_t)Msg::PREFILL_BATCH, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)p, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)n, cipher_)
                  && sock_send_u32_enc(sock_, (uint32_t)H, cipher_)
                  && sock_send_floats_enc(sock_, buf.data(), (size_t)n * (size_t)H, cipher_))) return false;
            inflight.push_back(n); ++outstanding;
            if (outstanding >= 2) { if (!wait_one()) return false; }   // keep one piece in flight
        }
        while (outstanding > 0) { if (!wait_one()) return false; }
        show(0, 0, true);
        return true;
    }

    bool prefill_token(int pos) {
        const int info_hidden = engine_.model_info().hidden_dim;
        if (pos < 0 || pos >= (int)tokens_.size()) return false;
        forward_first_half(tokens_, pos);
        const float* hidden = engine_hidden_data();
        if (!send_hidden(Msg::PREFILL, (uint32_t)pos, hidden, info_hidden)) return false;
        uint8_t ack = 0;
        if (!sock_recv_u8_dec(sock_, ack, cipher_) || (Msg)ack != Msg::ACK) {
            fprintf(stderr, "[dist] Prefill ACK failed at pos %d\n", pos);
            return false;
        }
        return true;
    }

    bool commit_pending_token() {
        if (!pending_token_uncommitted_) return true;
        if (!prefill_token(decode_pos_)) return false;
        pending_token_uncommitted_ = false;
        return true;
    }

    // Run Machine A's slice of the forward pass (layers 0..split_layer_).
    // After this, engine_hidden_data() returns the hidden state to send to B.
    //
    // ENGINE INTEGRATION NOTE (same as in DistPeerServer):
    // Replace with: engine_.forward_layer_range(tokens, pos, 0, split_layer_);
    // For now this calls the FULL forward (all layers) as a stub.
    // Machine A's answer will be wrong until forward_layer_range is wired up.
    void forward_first_half(const std::vector<int32_t>& tokens, int pos) {
        engine_.forward_layer_range(tokens, pos, 0, split_layer_);
    }

    const float* engine_hidden_data() {
        return engine_.hidden_data();
    }

    bool send_hidden(Msg type, uint32_t pos, const float* hidden, int n_floats) {
        return sock_send_u8_enc(sock_, (uint8_t)type, cipher_)
            && sock_send_u32_enc(sock_, pos, cipher_)
            && sock_send_u32_enc(sock_, (uint32_t)n_floats, cipher_)
            && sock_send_floats_enc(sock_, hidden, (size_t)n_floats, cipher_);
    }
};

} // namespace dist
} // namespace nexus


/*
 * --- ENGINE INTEGRATION PATCH ------------------------------------------------
 *
 * Add these 4 methods to Engine (engine.h public section + engine.cpp):
 *
 * IN engine.h:
 *   // Distributed inference -- expose hidden state and layer-range forward
 *   const float* hidden_data() const;
 *   void set_hidden_data(const float* h, size_t n_floats);
 *   void forward_layer_range(const std::vector<int32_t>& tokens, int pos,
 *                            int layer_start, int layer_end);
 *
 * IN engine.cpp:
 *
 *   const float* Engine::hidden_data() const {
 *       return impl_->transformer.hidden_data();
 *   }
 *
 *   void Engine::set_hidden_data(const float* h, size_t n_floats) {
 *       impl_->transformer.set_hidden_data(h, n_floats);
 *   }
 *
 *   void Engine::forward_layer_range(const std::vector<int32_t>& tokens,
 *                                    int pos, int layer_start, int layer_end) {
 *       impl_->transformer.forward_layer_range(tokens, pos, layer_start, layer_end);
 *   }
 *
 * That's the complete integration -- 12 lines of boilerplate.
 * After that, replace the STUB comments in DistSession and DistPeerServer with
 * the direct calls shown in their respective ENGINE INTEGRATION NOTE blocks.
 *
 * --- WIRING THE KV CACHE ON MACHINE B ----------------------------------------
 *
 * Machine B needs to initialize its KV cache starting at layer split_layer_.
 * The KVCache stores K/V per (layer_id, pos). When B calls forward_layer_range
 * with layer_start=split_layer_, the attention in each layer calls:
 *
 *   kvcache_.store(l, pos, k, v, kv_dim);   // l ranges from split_layer_ to N-1
 *
 * This is already correct -- kvcache_ has n_layers slots even if we only write
 * to the upper half. The lower half (0..split_layer_) simply stays empty on B.
 * No special initialization needed.
 *
 * --- PIPELINED PREFILL --------------------------------------------------------
 *
 * To get the 2x prefill speedup, remove the sync ACK after each PREFILL send
 * and instead drain all ACKs at the end of the prefill loop:
 *
 *   // In generate_stream() prefill loop -- remove ACK recv, just send:
 *   for (int i = 0; i < tokens.size(); ++i) {
 *       forward_first_half(tokens, i);
 *       send_hidden(Msg::PREFILL, i, engine_hidden_data(), hidden_dim);
 *       // NO ACK here -- B processes asynchronously
 *   }
 *   // Drain all ACKs at the end
 *   for (int i = 0; i < tokens.size(); ++i) {
 *       uint8_t ack; sock_recv_u8(sock_, ack);
 *   }
 *
 * Machine B's socket buffer will absorb the queued PREFILL messages at line rate.
 * At 1 Gbit/s and 20KB per hidden state: 1000 tokens = 20MB, transfer time < 200ms.
 * Both CPUs run simultaneously -- A on pos t, B on pos t-1. True pipeline.
 *
 * --- FOUR MACHINE CHAIN -------------------------------------------------------
 *
 * For A->B->C->D (150B model split 4 ways):
 *
 *   // Machine C acts as BOTH a server (for B) and a client (for D)
 *   // Server role: DistPeerServer(engine_c, split_B, split_C)
 *   // Client role: DistSession(engine_c, split_C) connected to D
 *
 *   // In DistPeerServer::handle_client(), after running its own layers,
 *   // forward the hidden state to the next machine instead of sampling.
 *   // Only the LAST machine in the chain samples and returns the token.
 *
 * The hidden state size stays constant through the chain: hidden_dim floats.
 * Each machine adds its own KV. Token flows back through the chain as a single int32.
 *
 * --- BANDWIDTH REALITY CHECK -------------------------------------------------
 *
 * 14B model (hidden_dim=5120):
 *   hidden state = 5120 x 4 = 20,480 bytes ~= 20KB per token
 *   At 1 Gbit/s ethernet: 20KB / 125MB/s = 0.16ms transfer time
 *   At 5 GT/s USB3:        20KB / 400MB/s = 0.05ms transfer time
 *   Typical decode speed: 1-5 tok/s -> 200-1000ms/tok compute
 *   Network overhead: 0.16ms / 500ms = 0.03% -- genuinely negligible
 *
 * The ONLY time network would matter: extreme quantization (Q2) on very fast hardware
 * running a small model. For the 14B-40B models you described: network is never the limit.
 */
