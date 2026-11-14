// chat.h -- talks to a running nexus_server over its normal HTTP API, exactly like the web page does:
//   POST /api/chat (server-sent events), /api/stop, /api/reset, /api/start_state/{save,list,load,delete}.
// The web UI and this window are two clients of the same server.  One Chat object per model slot.
#pragma once
#include "core.h"

struct ChatMsg { std::string role, text; };
struct KvState { std::string name, path, created; int ntok = 0; };

struct Chat {
    std::mutex mu;
    std::vector<ChatMsg> msgs;            // finished messages (user + assistant)
    std::string live;                     // assistant reply being streamed
    std::string err;                      // last problem, shown under the transcript
    std::atomic<bool> busy{false}, stop_req{false};
    std::atomic<int> ctx{0}, ntok{0};
    std::atomic<double> tps{0};
    std::string conv_id;
    bool single_turn = false;             // after loading a saved state: the server holds the history, send only the new message
    std::thread th;
    int rev = 0;                          // bumped whenever the transcript changes
    // saved memory (KV) states
    std::atomic<bool> kv_busy{false};
    std::string kv_msg;                   // result of the last save/load/delete (guarded by mu)
    std::vector<KvState> kv_list;         // guarded by mu
};

static std::string chat_new_id() {
    static const char* h = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::string s; unsigned long long t = (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count() ^ 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 16; ++i) { t = t * 6364136223846793005ULL + 1442695040888963407ULL; s += h[(t >> 33) % 36]; }
    return s;
}

static sock_t chat_connect(int port, int recv_ms = 300) {
    sock_t s = socket(AF_INET, SOCK_STREAM, 0); if (s == NX_INVALID_SOCK) return s;
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons((unsigned short)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (connect(s, (sockaddr*)&a, sizeof a) != 0) { sock_close(s); return NX_INVALID_SOCK; }
#if defined(_WIN32)
    DWORD tv = (DWORD)recv_ms; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
#else
    timeval tv{recv_ms / 1000, (recv_ms % 1000) * 1000}; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
    return s;
}
static bool chat_send_all(sock_t s, const std::string& d) {
    size_t off = 0;
    while (off < d.size()) {
        int n = (int)send(s, d.data() + off, (int)(d.size() - off), 0);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}
static bool sock_timed_out() {
#if defined(_WIN32)
    return WSAGetLastError() == WSAETIMEDOUT;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
#endif
}

// One short request, whole answer.  Returns the response body; ok=false when the server cannot be reached.
static std::string http_json(int port, const char* method, const std::string& path, const std::string& body, bool& ok, int total_ms = 20000) {
    ok = false; sock_t s = chat_connect(port); if (s == NX_INVALID_SOCK) return "";
    std::string req = std::string(method) + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
    if (!chat_send_all(s, req)) { sock_close(s); return ""; }
    std::string raw; char tmp[8192]; auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        int n = (int)recv(s, tmp, sizeof tmp, 0);
        if (n > 0) { raw.append(tmp, (size_t)n); continue; }
        if (n < 0 && sock_timed_out() && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(total_ms)) continue;
        break;
    }
    sock_close(s);
    size_t e = raw.find("\r\n\r\n"); if (e == std::string::npos) return "";
    std::string head = lower(raw.substr(0, e)), b = raw.substr(e + 4);
    if (head.find("transfer-encoding: chunked") != std::string::npos) {
        std::string out; size_t p = 0;
        while (p < b.size()) { size_t le = b.find("\r\n", p); if (le == std::string::npos) break; size_t sz = strtoul(b.c_str() + p, nullptr, 16); if (!sz) break; out += b.substr(le + 2, sz); p = le + 2 + sz + 2; }
        b = out;
    }
    ok = true; return b;
}
static void chat_post_quick(int port, const char* path) { bool ok; http_json(port, "POST", path, "{}", ok, 4000); }

static std::string chat_body(Chat& c, const std::string& system, const std::string& think) {
    std::string j = "{\"messages\":[";
    bool first = true;
    auto add = [&](const std::string& role, const std::string& text) {
        if (!first) j += ",";
        first = false;
        j += "{\"role\":\"" + role + "\",\"content\":\"" + jesc(text) + "\"}";
    };
    if (c.single_turn) {                                           // server already holds the earlier turns
        add("user", c.msgs.back().text);
    } else {
        if (!system.empty()) add("system", system);
        for (auto& m : c.msgs) add(m.role, m.text);
    }
    j += "],\"conv_id\":\"" + c.conv_id + "\",\"images\":[]";
    if (!think.empty()) j += ",\"think\":\"" + think + "\"";
    return j + "}";
}

static void chat_finish(Chat& c, const std::string& err_in) {
    std::string err = err_in;
    std::lock_guard<std::mutex> lk(c.mu);
    if (c.live.empty() && err.empty() && !c.stop_req) err = "The model sent no reply. The Log tab may say why.";
    if (!c.live.empty()) c.msgs.push_back({"assistant", c.live});
    c.live.clear();
    if (!err.empty()) c.err = err;
    c.rev++; c.busy = false;
}

static void chat_worker(Chat* cp, int port, std::string body) {
    Chat& c = *cp;
    sock_t s = chat_connect(port);
    if (s == NX_INVALID_SOCK) { chat_finish(c, "Could not reach the server on port " + std::to_string(port) + "."); return; }
    std::string req = "POST /api/chat HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
                      "\r\nConnection: close\r\n\r\n" + body;
    if (!chat_send_all(s, req)) { sock_close(s); chat_finish(c, "Could not send the message to the server."); return; }
    std::string buf, err; bool headers = false, chunked = false, done = false;
    std::string raw;
    auto t_first = std::chrono::steady_clock::time_point(); int ntok = 0;
    auto stop_at = std::chrono::steady_clock::time_point();
    char tmp[8192];
    while (!done) {
        int n = (int)recv(s, tmp, sizeof tmp, 0);
        if (n == 0) break;
        if (n < 0) {
            if (!sock_timed_out()) { err = "Connection to the server was lost."; break; }
            if (c.stop_req) {
                if (stop_at == std::chrono::steady_clock::time_point()) stop_at = std::chrono::steady_clock::now();
                else if (std::chrono::steady_clock::now() - stop_at > std::chrono::seconds(4)) break;
            }
            continue;
        }
        std::string data(tmp, (size_t)n);
        if (!headers) {
            raw += data; size_t e = raw.find("\r\n\r\n"); if (e == std::string::npos) continue;
            std::string h = lower(raw.substr(0, e));
            if (h.compare(0, 12, "http/1.1 200") != 0 && h.compare(0, 12, "http/1.0 200") != 0) {
                err = "The server refused the request (" + raw.substr(0, raw.find('\r')) + ")."; break;
            }
            chunked = h.find("transfer-encoding: chunked") != std::string::npos;
            headers = true; data = raw.substr(e + 4); raw.clear();
        }
        if (chunked) {
            raw += data; data.clear();
            for (;;) {
                size_t le = raw.find("\r\n"); if (le == std::string::npos) break;
                size_t sz = strtoul(raw.c_str(), nullptr, 16);
                if (sz == 0) { done = true; break; }
                if (raw.size() < le + 2 + sz + 2) break;
                data += raw.substr(le + 2, sz); raw.erase(0, le + 2 + sz + 2);
            }
        }
        buf += data;
        for (;;) {
            size_t nl = buf.find('\n'); if (nl == std::string::npos) break;
            std::string line = buf.substr(0, nl); buf.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.compare(0, 5, "data:") != 0) continue;
            std::string d = line.substr(5); size_t a = d.find_first_not_of(' '); d = a == std::string::npos ? "" : d.substr(a);
            if (d == "[DONE]") { done = true; break; }
            JV v; if (!json_parse(d, v)) continue;
            if (const JV* t = v.get("token")) {
                if (t_first == std::chrono::steady_clock::time_point()) t_first = std::chrono::steady_clock::now();
                ++ntok; c.ntok = ntok;
                { std::lock_guard<std::mutex> lk(c.mu); c.live += jstr(t); c.rev++; }
                double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_first).count();
                if (el > 0.25) c.tps = ntok / el;
            }
            if (const JV* cx = v.get("context_tokens")) c.ctx = (int)cx->n;
            if (const JV* e = v.get("error")) err = jstr(e);
        }
    }
    sock_close(s);
    chat_finish(c, err);
}

static void chat_join(Chat& c) { if (c.th.joinable()) c.th.join(); }
static void chat_detach(Chat& c) { if (c.th.joinable()) c.th.detach(); }

static bool chat_send(Chat& c, int port, const std::string& text, const std::string& system, const std::string& think) {
    if (c.busy.load() || text.empty()) return false;
    chat_join(c);
    std::string body;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        if (c.conv_id.empty()) c.conv_id = chat_new_id();
        c.msgs.push_back({"user", text}); c.err.clear(); c.live.clear(); c.rev++;
        body = chat_body(c, system, think);
    }
    c.busy = true; c.stop_req = false; c.ntok = 0; c.tps = 0;
    c.th = std::thread(chat_worker, &c, port, body);
    return true;
}
static void chat_stop(Chat& c, int port) { if (!c.busy.load()) return; c.stop_req = true; std::thread([port] { chat_post_quick(port, "/api/stop"); }).detach(); }
static void chat_new(Chat& c, int port) {
    if (c.busy.load()) chat_stop(c, port);
    chat_join(c);
    { std::lock_guard<std::mutex> lk(c.mu); c.msgs.clear(); c.live.clear(); c.err.clear(); c.conv_id.clear(); c.single_turn = false; c.rev++; }
    c.ctx = 0; c.ntok = 0; c.tps = 0;
    std::thread([port] { chat_post_quick(port, "/api/reset"); }).detach();
}

// ── saved memory states (the model's KV cache, no text) ──────────────────────────────────────
static void kv_set_msg(Chat& c, const std::string& m) { std::lock_guard<std::mutex> lk(c.mu); c.kv_msg = m; }

static void kv_refresh(Chat* cp, int port) {
    Chat& c = *cp; bool ok; std::string r = http_json(port, "GET", "/api/start_state/list", "", ok);
    JV v; std::vector<KvState> out;
    if (ok && json_parse(r, v) && v.t == JV::Arr) for (auto& x : v.a) {
        KvState k; k.name = jstr(x.get("name")); k.path = jstr(x.get("path")); k.created = jstr(x.get("created"));
        if (const JV* n = x.get("n_tokens")) k.ntok = (int)n->n;
        if (!k.path.empty()) out.push_back(k);
    }
    std::lock_guard<std::mutex> lk(c.mu);
    if (!ok) c.kv_msg = "Could not reach the model."; else c.kv_list = out;
    c.kv_busy = false;
}
static void kv_save(Chat* cp, int port, std::string name, std::string conv) {
    Chat& c = *cp; bool ok;
    std::string r = http_json(port, "POST", "/api/start_state/save", "{\"name\":\"" + jesc(name) + "\",\"conv_id\":\"" + jesc(conv) + "\"}", ok, 120000);
    JV v; bool good = ok && json_parse(r, v) && v.get("ok") && v.get("ok")->b;
    kv_set_msg(c, good ? "Saved \"" + name + "\"." : "Saving failed: " + (ok ? jstr(v.get("error"), "the model said no") : "could not reach the model"));
    kv_refresh(cp, port);
}
static void kv_load(Chat* cp, int port, std::string path, std::string name) {
    Chat& c = *cp; bool ok;
    std::string r = http_json(port, "POST", "/api/start_state/load", "{\"path\":\"" + jesc(path) + "\"}", ok, 120000);
    JV v; bool good = ok && json_parse(r, v) && v.get("ok") && v.get("ok")->b;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        if (good) {
            c.msgs.clear(); c.live.clear(); c.err.clear(); c.conv_id = jstr(v.get("conv_id")); c.single_turn = true;
            c.ctx = v.get("context_size") ? (int)v.get("context_size")->n : 0; c.rev++;
            c.kv_msg = "Loaded \"" + name + "\" - the model remembers it. New messages continue from there.";
        } else c.kv_msg = "Loading failed: " + (ok ? jstr(v.get("error"), "the model said no") : "could not reach the model");
    }
    c.kv_busy = false;
}
static void kv_delete(Chat* cp, int port, std::string path) {
    Chat& c = *cp; bool ok; http_json(port, "POST", "/api/start_state/delete", "{\"path\":\"" + jesc(path) + "\"}", ok);
    kv_set_msg(c, ok ? "Deleted." : "Could not reach the model.");
    kv_refresh(cp, port);
}
// run one of the above off the UI thread
template <class F, class... A> static void kv_run(Chat& c, F f, A... a) {
    if (c.kv_busy.exchange(true)) return;
    std::thread(f, &c, a...).detach();
}
