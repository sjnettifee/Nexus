/* 
 * nexus/tools/server.cpp  v2
 *
 * Local HTTP + SSE server with embedded chat UI
 * Zero external dependencies beyond the nexus engine.
 *
 * New in v2:
 *   - Projects + saved chats (filesystem in ./nexus-data/)
 *   - Template picker: auto-detect + manual override per conversation
 *   - Custom system-prompt library (save / load / delete)
 *   - Repeat penalty 1.0-1.5  (1.0 = off, correct for coder models)
 *   - Max tokens: slider + manual numeric input, default 2048
 *   - Chat auto-title from first user message
 *   - Full state restore when loading a saved chat
 *
 * Storage layout:
 *   ./nexus-data/projects/{id}.json
 *   ./nexus-data/chats/{id}.json
 *   ./nexus-data/templates/{id}.json
 *
 * API:
 *   GET  /                           - embedded UI
 *   GET  /api/info                   - model info JSON
 *   POST /api/chat                   - SSE token stream
 *   POST /api/reset                  - clear KV cache
 *   GET  /api/projects               - list projects
 *   POST /api/projects               - create project
 *   DELETE /api/projects/:id         - delete project
 *   GET  /api/projects/:id/chats     - list chats in project
 *   POST /api/projects/:id/chats     - save / update chat
 *   GET  /api/chats/:id              - load chat (with messages)
 *   DELETE /api/chats/:id            - delete chat
 *   GET  /api/templates              - list prompt templates
 *   POST /api/templates              - create template
 *   DELETE /api/templates/:id        - delete template
 */

#include "api/engine.h"
#include "api/config.h"
#include "api/chat_template.h"
#include "api/dist_session.h"
#include "core/memguard.h"
#include "core/crashlog.h"
#ifdef NEXUS_VULKAN
#include "gpu/vulkan_backend.h"
#endif
#include "api/probe.h"          // --probe: hardware/GPU report as JSON

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cctype>
#include <iomanip>
#include <thread>
#ifndef _WIN32
#include <sys/statvfs.h>
#endif
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <ctime>
#ifndef _WIN32
#include <netinet/tcp.h>
#endif
#include <atomic>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <utility>
#include <cerrno>

// --- Platform socket / OS abstraction ----------------------------------------
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <windows.h>
#  include <direct.h>
#  include <io.h>
#  pragma comment(lib, "ws2_32.lib")
   using ssize_t   = SSIZE_T;
   using socklen_t = int;
#  define MSG_NOSIGNAL 0
#  define close(s)     closesocket(s)
#  define SHUT_RDWR    SD_BOTH
#  include <sys/stat.h>
   struct dirent { char d_name[MAX_PATH]; };
   struct DIR_impl;
   typedef DIR_impl DIR;
   inline DIR*    opendir(const char*);
   inline dirent* readdir(DIR*);
   inline int     closedir(DIR*);
#else
#  include <sys/socket.h>
#  include <sys/stat.h>
#  include <netinet/in.h>
#  include <unistd.h>
#  include <arpa/inet.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <dirent.h>
#  include <sys/wait.h>
#endif
// -----------------------------------------------------------------------------

// --- Cross-platform helpers --------------------------------------------------
#ifdef _WIN32
// Non-blocking socket toggle
inline void sock_nonblock(SOCKET fd, bool on) {
    u_long mode = on ? 1 : 0;
    ioctlsocket(fd, FIONBIO, &mode);
}
// mkdir without mode arg
inline int mkdir_compat(const char* p) { return _mkdir(p); }

// Bounded blocking sends: a wedged/half-dead client must never stall generation
inline void sock_sndtimeo(SOCKET fd, int ms) { DWORD t = (DWORD)ms; setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&t, sizeof(t)); }
// Minimal dirent emulation
#include <string>
struct DIR_impl {
    HANDLE           hFind = INVALID_HANDLE_VALUE;
    WIN32_FIND_DATAA fdata{};
    dirent           ent{};
    bool             first = true;
};
inline DIR* opendir(const char* path) {
    auto* d = new DIR_impl;
    std::string pat = std::string(path) + "\\*";
    d->hFind = FindFirstFileA(pat.c_str(), &d->fdata);
    if (d->hFind == INVALID_HANDLE_VALUE) { delete d; return nullptr; }
    return d;
}
inline dirent* readdir(DIR* d) {
    if (!d) return nullptr;
    if (d->first) d->first = false;
    else if (!FindNextFileA(d->hFind, &d->fdata)) return nullptr;
    strncpy(d->ent.d_name, d->fdata.cFileName, MAX_PATH-1);
    return &d->ent;
}
inline int closedir(DIR* d) {
    if (d && d->hFind != INVALID_HANDLE_VALUE) FindClose(d->hFind);
    delete d; return 0;
}
#else
inline void sock_nonblock(int fd, bool on) {
    int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, on ? fl|O_NONBLOCK : fl&~O_NONBLOCK);
}
inline int mkdir_compat(const char* p) { return mkdir(p, 0755); }
inline void sock_sndtimeo(int fd, int ms) { struct timeval tv; tv.tv_sec = ms/1000; tv.tv_usec = (ms%1000)*1000; setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)); }
#endif
// -----------------------------------------------------------------------------

static std::string nx_log_visible_text(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            char b[8];
            snprintf(b, sizeof(b), "\\x%02x", c);
            out += b;
        } else {
            out.push_back((char)c);
        }
    }
    return out;
}



static int nx_clamp_trace_level(int v) {
    if (v < 0) return 0;
    if (v > 3) return 3;
    return v;
}

static void nx_set_trace_level_env(int level) {
    level = nx_clamp_trace_level(level);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d", level);

#ifdef _WIN32
    _putenv_s("NEXUS_TRACE", buf);
#else
    setenv("NEXUS_TRACE", buf, 1);
#endif
}

// --- JSON helpers -------------------------------------------------------------

static std::string jstr(const std::string& s) {
    std::string o="\"";
    for (unsigned char c:s){
        if(c=='"') o+="\\\""; else if(c=='\\') o+="\\\\";
        else if(c=='\n') o+="\\n"; else if(c=='\r') o+="\\r";
        else if(c=='\t') o+="\\t";
        else if(c<0x20){char b[8];snprintf(b,8,"\\u%04x",c);o+=b;}
        else o+=c;
    }
    return o+"\"";
}
static std::string jnum(uint64_t v) { return std::to_string(v); }
static std::string jnum(uint32_t v) { return std::to_string(v); }
static std::string jnum(int v)      { return std::to_string(v); }

static std::string jp_str(const std::string& j, const std::string& key, const std::string& def="") {
    auto needle="\""+key+"\"";
    size_t p=j.find(needle); if(p==std::string::npos) return def;
    p=j.find(':',p); if(p==std::string::npos) return def;
    while(p<j.size()&&(j[p]==':'||j[p]==' ')) ++p;
    if(p>=j.size()||j[p]!='"') return def;
    ++p; std::string v;
    while(p<j.size()&&j[p]!='"'){
        if(j[p]=='\\'&&p+1<j.size()){++p;
            switch(j[p]){case 'n':v+='\n';break;case 't':v+='\t';break;
                         case 'r':v+='\r';break;case 'b':v+='\b';break;case 'f':v+='\f';break;
                         case 'u': {   // \uXXXX (and surrogate pairs) -> UTF-8, so attached documents arrive byte-for-byte
                             auto hex4=[&](size_t at, unsigned& out)->bool{ if(at+4>j.size())return false; unsigned c=0;
                                 for(int k=0;k<4;++k){ char h=j[at+k]; c<<=4; if(h>='0'&&h<='9')c|=h-'0'; else if(h>='a'&&h<='f')c|=h-'a'+10; else if(h>='A'&&h<='F')c|=h-'A'+10; else return false; } out=c; return true; };
                             unsigned cp=0;
                             if(!hex4(p+1,cp)){ v+='u'; break; }
                             p+=4;
                             if(cp>=0xD800&&cp<=0xDBFF&&p+6<j.size()&&j[p+1]=='\\'&&j[p+2]=='u'){ unsigned lo=0; if(hex4(p+3,lo)&&lo>=0xDC00&&lo<=0xDFFF){ cp=0x10000+((cp-0xD800)<<10)+(lo-0xDC00); p+=6; } }
                             if(cp<0x80) v+=(char)cp;
                             else if(cp<0x800){ v+=(char)(0xC0|(cp>>6)); v+=(char)(0x80|(cp&0x3F)); }
                             else if(cp<0x10000){ v+=(char)(0xE0|(cp>>12)); v+=(char)(0x80|((cp>>6)&0x3F)); v+=(char)(0x80|(cp&0x3F)); }
                             else { v+=(char)(0xF0|(cp>>18)); v+=(char)(0x80|((cp>>12)&0x3F)); v+=(char)(0x80|((cp>>6)&0x3F)); v+=(char)(0x80|(cp&0x3F)); }
                             break; }
                         default:v+=j[p];break;}}
        else { v+=j[p]; } ++p;
    }
    return v;
}

// Per-request thinking switch: "think":"on"|"off" in the request body.
// -1 = not specified (use the launcher / model default).
static int jp_think(const std::string& j) {
    std::string v = jp_str(j, "think", "");
    for (auto& c : v) c = (char)tolower((unsigned char)c);
    if (v == "on"  || v == "1" || v == "true")  return 1;
    if (v == "off" || v == "0" || v == "false") return 0;
    return -1;
}
static double jp_num(const std::string& j, const std::string& key, double def=0) {
    auto needle="\""+key+"\"";
    size_t p=j.find(needle); if(p==std::string::npos) return def;
    p=j.find(':',p); if(p==std::string::npos) return def;
    while(p<j.size()&&!isdigit((unsigned char)j[p])&&j[p]!='-') ++p;
    if(p>=j.size()) return def;
    return atof(j.c_str()+p);
}

static std::vector<nexus::ChatMessage> jp_messages(const std::string& json) {
    std::vector<nexus::ChatMessage> out;

    // Find the "messages" array
    const std::string key = "\"messages\"";
    size_t km = json.find(key);
    if (km == std::string::npos) return out;
    size_t ab = json.find('[', km + key.size());
    if (ab == std::string::npos) return out;

    size_t p = ab + 1;
    while (p < json.size()) {
        // Skip whitespace and commas between objects
        while (p < json.size() && (json[p]==',' || json[p]==' ' ||
               json[p]=='\n' || json[p]=='\r' || json[p]=='\t')) ++p;
        if (p >= json.size() || json[p] == ']') break;
        if (json[p] != '{') { ++p; continue; }

        // Find the matching closing '}' - must correctly skip string values
        // so a '}' inside a "content" field doesn't end the object early.
        size_t ob = p;
        int depth = 0;
        while (p < json.size()) {
            char c = json[p];
            if (c == '"') {
                // Skip over entire quoted string
                ++p;
                while (p < json.size()) {
                    if (json[p] == '\\') { p += 2; continue; } // skip escape + next char
                    if (json[p] == '"') { ++p; break; }        // end of string
                    ++p;
                }
                continue;  // don't fall through to brace counting
            }
            if (c == '{') ++depth;
            else if (c == '}') { if (--depth == 0) { ++p; break; } }
            ++p;
        }

        std::string obj = json.substr(ob, p - ob);
        nexus::ChatMessage m;
        m.role    = jp_str(obj, "role");
        m.content = jp_str(obj, "content");
        if (!m.role.empty()) out.push_back(m);
    }
    return out;
}


struct ImagePayload {
    std::string name;
    int width = 0;
    int height = 0;
    // Compatibility path: old JSON carried raw RGB as base64. New UI path uploads
    // raw RGB as a separate binary POST and only sends this id in chat JSON.
    std::string rgb_b64;
    std::string upload_id;
};

static std::vector<uint8_t> b64_decode(const std::string& s) {
    int table[256];
    for (int i = 0; i < 256; ++i) table[i] = -1;
    const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (int i = 0; alphabet[i]; ++i) table[(unsigned char)alphabet[i]] = i;
    std::vector<uint8_t> out;
    int val = 0, valb = -8;
    for (unsigned char c : s) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        int d = table[c];
        if (d < 0) continue;
        val = (val << 6) + d;
        valb += 6;
        if (valb >= 0) {
            out.push_back((uint8_t)((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}


static std::string nx_url_decode(const std::string& s) {
    std::string out; out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '+' ) { out.push_back(' '); continue; }
        if (c == '%' && i + 2 < s.size()) {
            auto hx = [](char ch)->int {
                if (ch >= '0' && ch <= '9') return ch - '0';
                if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
                if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
                return -1;
            };
            int a = hx(s[i+1]), b = hx(s[i+2]);
            if (a >= 0 && b >= 0) { out.push_back((char)((a << 4) | b)); i += 2; continue; }
        }
        out.push_back((char)c);
    }
    return out;
}

static std::string nx_query_param(const std::string& path, const std::string& key, const std::string& defv = "") {
    size_t q = path.find('?');
    if (q == std::string::npos) return defv;
    std::string needle = key + "=";
    size_t p = q + 1;
    while (p <= path.size()) {
        size_t amp = path.find('&', p);
        std::string part = path.substr(p, amp == std::string::npos ? std::string::npos : amp - p);
        if (part.rfind(needle, 0) == 0) return nx_url_decode(part.substr(needle.size()));
        if (amp == std::string::npos) break;
        p = amp + 1;
    }
    return defv;
}

static std::string nx_image_upload_dir() {
#ifdef _WIN32
    _mkdir("nexus-data");
    _mkdir("nexus-data\\image_uploads");
    return "nexus-data\\image_uploads";
#else
    mkdir("nexus-data", 0755);
    mkdir("nexus-data/image_uploads", 0755);
    return "nexus-data/image_uploads";
#endif
}

static std::string nx_safe_id() {
    static std::atomic<unsigned long long> ctr{0};
    unsigned long long c = ++ctr;
    char b[96];
    std::snprintf(b, sizeof(b), "img_%llu_%llu", (unsigned long long)std::time(nullptr), c);
    return b;
}

static std::string nx_upload_path_for_id(const std::string& id) {
    std::string clean;
    clean.reserve(id.size());
    for (char ch : id) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') clean.push_back(ch);
    }
    if (clean.empty()) clean = "bad";
#ifdef _WIN32
    return nx_image_upload_dir() + "\\" + clean + ".rgb";
#else
    return nx_image_upload_dir() + "/" + clean + ".rgb";
#endif
}

static bool nx_read_file_bytes(const std::string& path, std::vector<uint8_t>& out) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamoff sz = f.tellg();
    if (sz < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    if (!out.empty()) f.read((char*)out.data(), (std::streamsize)out.size());
    return (bool)f || f.gcount() == (std::streamsize)out.size();
}

static std::vector<ImagePayload> jp_images(const std::string& json) {
    std::vector<ImagePayload> out;
    const std::string key = "\"images\"";
    size_t km = json.find(key);
    if (km == std::string::npos) return out;
    size_t ab = json.find('[', km + key.size());
    if (ab == std::string::npos) return out;
    size_t p = ab + 1;
    while (p < json.size()) {
        while (p < json.size() && (json[p]==',' || json[p]==' ' || json[p]=='\n' || json[p]=='\r' || json[p]=='\t')) ++p;
        if (p >= json.size() || json[p] == ']') break;
        if (json[p] != '{') { ++p; continue; }
        size_t ob = p;
        int depth = 0;
        while (p < json.size()) {
            char c = json[p];
            if (c == '"') {
                ++p;
                while (p < json.size()) {
                    if (json[p] == '\\') { p += 2; continue; }
                    if (json[p] == '"') { ++p; break; }
                    ++p;
                }
                continue;
            }
            if (c == '{') ++depth;
            else if (c == '}') { if (--depth == 0) { ++p; break; } }
            ++p;
        }
        std::string obj = json.substr(ob, p - ob);
        ImagePayload im;
        im.name = jp_str(obj, "name", "image");
        im.width = (int)jp_num(obj, "width", 0);
        im.height = (int)jp_num(obj, "height", 0);
        im.rgb_b64 = jp_str(obj, "rgb_b64", "");
        im.upload_id = jp_str(obj, "id", "");
        if (im.upload_id.empty()) im.upload_id = jp_str(obj, "upload_id", "");
        if (im.width > 0 && im.height > 0 && (!im.rgb_b64.empty() || !im.upload_id.empty())) out.push_back(std::move(im));
    }
    return out;
}


static std::string nx_image_marker(const ImagePayload& im) {
    return std::string("[Image: ") + im.name + "]";
}


static int32_t nx_image_pad_token_id(nexus::Engine& eng) {
    // Qwen3.6 official config.json: image_token_id = 248056.
    // Prefer tokenizer lookup so non-Qwen models are not hardwired when their
    // tokenizer actually exposes <|image_pad|>.
    try {
        std::vector<int32_t> t = eng.tokenize("<|image_pad|>", false);
        if (t.size() == 1 && t[0] > 0) return t[0];
    } catch(...) {}
    return 248056;
}

struct NxVisionSettings {
    // single = encode the uploaded image as one model view.
    // tiles  = encode native/detail crops only. No automatic overview is added.
    std::string mode = "single";
    int tile_size = 0;
    int tile_stride = 0;
    int max_tiles = 0; // 0 = unlimited
};

static int nx_env_int(const char* key, int defv) {
    const char* v = std::getenv(key);
    if (!v || !*v) return defv;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v) return defv;
    if (x > 2147483647L) return 2147483647;
    if (x < -2147483647L) return -2147483647;
    return (int)x;
}

static NxVisionSettings nx_vision_settings() {
    NxVisionSettings s;
    if (const char* m = std::getenv("NEXUS_VISION_MODE")) {
        std::string x = m;
        std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c){ return (char)std::tolower(c); });
        if (x == "tile" || x == "tiles" || x == "tiles_only" || x == "tiles-only") s.mode = "tiles";
        else s.mode = "single";
    }
    s.tile_size = nx_env_int("NEXUS_VISION_TILE_SIZE", 0);
    s.tile_stride = nx_env_int("NEXUS_VISION_TILE_STRIDE", 0);
    s.max_tiles = nx_env_int("NEXUS_VISION_MAX_TILES", 0);
    if (s.tile_size < 0) s.tile_size = 0;
    if (s.tile_stride < 0) s.tile_stride = 0;
    if (s.max_tiles < 0) s.max_tiles = 0;
    return s;
}

static std::vector<int> nx_tile_starts(int dim, int tile, int stride) {
    std::vector<int> out;
    if (dim <= 0) return out;
    if (tile <= 0 || tile >= dim) { out.push_back(0); return out; }
    if (stride <= 0) stride = tile;
    for (int p = 0; p + tile < dim; p += stride) {
        if (out.empty() || out.back() != p) out.push_back(p);
        if (stride <= 0) break;
    }
    int last = std::max(0, dim - tile);
    if (out.empty() || out.back() != last) out.push_back(last);
    return out;
}

static bool nx_append_encoded_view(nexus::Engine& eng,
                                   const std::string& label,
                                   const uint8_t* rgb,
                                   int w,
                                   int h,
                                   std::vector<float>& emb_accum,
                                   size_t& n_tok_accum,
                                   std::string& err) {
    const int hidden = (int)eng.model_info().hidden_dim;
    if (!rgb || w <= 0 || h <= 0) {
        err = "invalid vision view dimensions";
        return false;
    }
    std::vector<float> e = eng.encode_image(rgb, w, h);
    if (e.empty() || (e.size() % (size_t)hidden) != 0) {
        char b[256];
        snprintf(b, sizeof(b), "vision encoder returned invalid embeddings for %s: floats=%zu hidden=%d view=%dx%d",
                 label.c_str(), e.size(), hidden, w, h);
        err = b;
        return false;
    }
    size_t nt = e.size() / (size_t)hidden;
    emb_accum.insert(emb_accum.end(), e.begin(), e.end());
    n_tok_accum += nt;
    fprintf(stderr,"[vision-place] encoded view %s -> %zu token(s)\n", label.c_str(), nt);
    return true;
}

static bool nx_append_encoded_tile(nexus::Engine& eng,
                                   const std::string& name,
                                   const std::vector<uint8_t>& rgb,
                                   int src_w,
                                   int src_h,
                                   int x,
                                   int y,
                                   int tw,
                                   int th,
                                   int tile_idx,
                                   std::vector<float>& emb_accum,
                                   size_t& n_tok_accum,
                                   std::string& err) {
    if (src_w <= 0 || src_h <= 0 || x < 0 || y < 0 || tw <= 0 || th <= 0 ||
        x + tw > src_w || y + th > src_h) {
        err = "invalid tile crop geometry";
        return false;
    }
    std::vector<uint8_t> tile((size_t)tw * (size_t)th * 3u);
    const size_t src_stride = (size_t)src_w * 3u;
    const size_t dst_stride = (size_t)tw * 3u;
    for (int yy = 0; yy < th; ++yy) {
        const uint8_t* sp = rgb.data() + ((size_t)(y + yy) * src_stride) + (size_t)x * 3u;
        uint8_t* dp = tile.data() + (size_t)yy * dst_stride;
        std::memcpy(dp, sp, dst_stride);
    }
    char label[320];
    snprintf(label, sizeof(label), "%s tile#%d x=%d y=%d %dx%d", name.c_str(), tile_idx, x, y, tw, th);
    return nx_append_encoded_view(eng, label, tile.data(), tw, th, emb_accum, n_tok_accum, err);
}

static bool nx_encode_image_embeddings(nexus::Engine& eng,
                                       const ImagePayload& im,
                                       std::vector<float>& emb,
                                       size_t& n_tok,
                                       std::string& err) {
    emb.clear();
    n_tok = 0;
    if (!eng.has_vision()) {
        err = "image attached but no native/embedded vision encoder is loaded";
        return false;
    }
    const int hidden = (int)eng.model_info().hidden_dim;
    if (hidden <= 0) {
        err = "invalid model hidden size for vision injection";
        return false;
    }
    std::vector<uint8_t> rgb;
    const size_t need = (size_t)im.width * (size_t)im.height * 3u;
    const char* source = im.upload_id.empty() ? "json-b64" : "binary-upload";
    if (!im.upload_id.empty()) {
        const std::string path = nx_upload_path_for_id(im.upload_id);
        if (!nx_read_file_bytes(path, rgb)) {
            err = std::string("uploaded image payload not found: ") + im.upload_id;
            return false;
        }
        // One-shot upload cache: delete after reading so full-size images do not
        // accumulate on disk across turns.
        std::remove(path.c_str());
    } else {
        rgb = b64_decode(im.rgb_b64);
    }
    if (rgb.size() != need) {
        char b[256];
        snprintf(b, sizeof(b), "bad image RGB payload for %s: got=%zu need=%zu (%dx%d)",
                 im.name.c_str(), rgb.size(), need, im.width, im.height);
        err = b;
        return false;
    }

    NxVisionSettings vs = nx_vision_settings();
    fprintf(stderr,"[vision-place] encoding %s (%dx%d RGB, %zu bytes, source=%s, mode=%s)\n",
            im.name.c_str(), im.width, im.height, rgb.size(), source, vs.mode.c_str());

    if (vs.mode != "tiles") {
        return nx_append_encoded_view(eng, im.name, rgb.data(), im.width, im.height, emb, n_tok, err);
    }

    int tile = vs.tile_size;
    if (tile <= 0) tile = nx_env_int("NEXUS_QWEN_NATIVE_IMAGE_SIZE", 0);
    if (tile <= 0) tile = std::min(im.width, im.height);
    if (tile <= 0) tile = std::max(im.width, im.height);
    int stride = vs.tile_stride > 0 ? vs.tile_stride : tile;
    if (stride <= 0) stride = tile;

    std::vector<int> xs = nx_tile_starts(im.width, tile, stride);
    std::vector<int> ys = nx_tile_starts(im.height, tile, stride);
    int planned = (int)((size_t)xs.size() * (size_t)ys.size());
    int limit = vs.max_tiles > 0 ? std::min(planned, vs.max_tiles) : planned;
    fprintf(stderr,"[vision-tiles] %s source=%dx%d tile=%d stride=%d planned=%d max_tiles=%d encoding=%d overview=0\n",
            im.name.c_str(), im.width, im.height, tile, stride, planned, vs.max_tiles, limit);

    int encoded = 0;
    for (int y : ys) {
        for (int x : xs) {
            if (vs.max_tiles > 0 && encoded >= vs.max_tiles) break;
            int tw = std::min(tile, im.width);
            int th = std::min(tile, im.height);
            int xx = std::max(0, std::min(x, im.width  - tw));
            int yy = std::max(0, std::min(y, im.height - th));
            if (!nx_append_encoded_tile(eng, im.name, rgb, im.width, im.height,
                                        xx, yy, tw, th, encoded + 1, emb, n_tok, err)) return false;
            ++encoded;
        }
        if (vs.max_tiles > 0 && encoded >= vs.max_tiles) break;
    }
    if (encoded == 0) {
        err = "tile mode produced zero tiles";
        return false;
    }
    fprintf(stderr,"[vision-tiles] done %s views=%d total_tokens=%zu overview=0\n",
            im.name.c_str(), encoded, n_tok);
    return true;
}

static bool nx_append_text_tokens_for_mixed(nexus::Engine& eng,
                                            const std::string& text,
                                            std::vector<int32_t>& tokens,
                                            std::vector<uint8_t>& mask,
                                            std::vector<float>& embeds,
                                            int hidden) {
    if (text.empty()) return true;
    std::vector<int32_t> t = eng.tokenize(text, false);
    if (t.empty()) return true;
    tokens.insert(tokens.end(), t.begin(), t.end());
    mask.insert(mask.end(), t.size(), 0);
    embeds.resize(embeds.size() + t.size() * (size_t)hidden, 0.0f);
    return true;
}

static bool nx_append_vision_tokens_for_mixed(nexus::Engine& eng,
                                              const std::vector<float>& emb,
                                              int32_t image_pad_token,
                                              std::vector<int32_t>& tokens,
                                              std::vector<uint8_t>& mask,
                                              std::vector<float>& embeds,
                                              int hidden,
                                              size_t& n_tok_out,
                                              std::string& err) {
    n_tok_out = 0;
    if (hidden <= 0 || emb.empty() || (emb.size() % (size_t)hidden) != 0) {
        char b[256];
        snprintf(b, sizeof(b), "bad mixed vision embedding payload: floats=%zu hidden=%d",
                 emb.size(), hidden);
        err = b;
        return false;
    }
    const size_t n_tok = emb.size() / (size_t)hidden;
    const size_t old = tokens.size();
    tokens.insert(tokens.end(), n_tok, image_pad_token);
    mask.insert(mask.end(), n_tok, 1);
    embeds.resize(embeds.size() + n_tok * (size_t)hidden);
    std::memcpy(embeds.data() + old * (size_t)hidden,
                emb.data(), emb.size() * sizeof(float));
    n_tok_out = n_tok;
    return true;
}

static bool nx_extend_context_with_images_in_place(nexus::Engine& eng,
                                                   const std::string& formatted_turn,
                                                   const std::vector<ImagePayload>& images,
                                                   size_t& vision_tokens_injected,
                                                   std::string& err) {
    vision_tokens_injected = 0;
    if (images.empty()) {
        eng.extend_context(formatted_turn, false);
        return true;
    }
    if (!eng.has_vision()) {
        err = "image attached but no native/embedded vision encoder is loaded";
        return false;
    }

    const int hidden = (int)eng.model_info().hidden_dim;
    if (hidden <= 0) {
        err = "invalid model hidden size for mixed vision prefill";
        return false;
    }
    const int32_t image_pad_token = nx_image_pad_token_id(eng);

    std::vector<int32_t> mixed_tokens;
    std::vector<uint8_t> embed_mask;
    std::vector<float> embedded_inputs; // dense [mixed_tokens * hidden]; only mask=1 entries are meaningful now

    size_t cur = 0;
    for (const auto& im : images) {
        const std::string mark = nx_image_marker(im);
        size_t pos = formatted_turn.find(mark, cur);
        if (pos == std::string::npos) {
            char b[256];
            snprintf(b, sizeof(b), "image marker not found in formatted turn for %s", im.name.c_str());
            err = b;
            return false;
        }

        // Build the full body first. Do not prefill text-before-image yet.
        // That prevents the main model from seeing/processing the sentence as a
        // separate context update before visual embeddings exist in the same pass.
        if (pos > cur) {
            nx_append_text_tokens_for_mixed(eng, formatted_turn.substr(cur, pos - cur),
                                            mixed_tokens, embed_mask, embedded_inputs, hidden);
        }
        nx_append_text_tokens_for_mixed(eng, "<|vision_start|>",
                                        mixed_tokens, embed_mask, embedded_inputs, hidden);

        std::vector<float> emb;
        size_t n_tok = 0;
        if (!nx_encode_image_embeddings(eng, im, emb, n_tok, err)) return false;
        size_t added = 0;
        if (!nx_append_vision_tokens_for_mixed(eng, emb, image_pad_token,
                                               mixed_tokens, embed_mask, embedded_inputs,
                                               hidden, added, err)) return false;
        vision_tokens_injected += added;

        nx_append_text_tokens_for_mixed(eng, "<|vision_end|>",
                                        mixed_tokens, embed_mask, embedded_inputs, hidden);
        cur = pos + mark.size();
        fprintf(stderr,"[vision-mixed] staged marker %s as %zu native visual token(s); no main-model prefill yet\n",
                mark.c_str(), added);
    }

    if (cur < formatted_turn.size()) {
        nx_append_text_tokens_for_mixed(eng, formatted_turn.substr(cur),
                                        mixed_tokens, embed_mask, embedded_inputs, hidden);
    }

    fprintf(stderr,"[vision-mixed] single main-model prefill body: text+vision tokens=%zu vision_tokens=%zu ctx_before=%d\n",
            mixed_tokens.size(), vision_tokens_injected, eng.context_size());
    if (!eng.extend_context_mixed_embeddings(mixed_tokens, embed_mask, embedded_inputs)) {
        err = "mixed text+vision prefill failed; see server log";
        return false;
    }
    fprintf(stderr,"[vision-mixed] done ctx=%d\n", eng.context_size());
    return true;
}
// --- Utilities ----------------------------------------------------------------

static std::string gen_id() {
    static std::atomic<uint64_t> ctr{0};
    char b[32]; snprintf(b,32,"%llx%04llx",(unsigned long long)time(nullptr),(unsigned long long)(ctr++&0xFFFF));
    return b;
}
static std::string now_iso() {
    time_t t=time(nullptr); struct tm tm;
#ifdef _WIN32
    localtime_s(&tm,&t);
#else
    localtime_r(&t,&tm);
#endif
    char b[32]; strftime(b,32,"%Y-%m-%dT%H:%M:%S",&tm); return b;
}
static bool dir_exists(const std::string& p){struct stat s;return stat(p.c_str(),&s)==0&&S_ISDIR(s.st_mode);}
static void mkdir_p(const std::string& p){for(size_t i=1;i<=p.size();++i)if(i==p.size()||p[i]=='/')mkdir_compat(p.substr(0,i).c_str());}
static std::string read_file(const std::string& path){std::ifstream f(path);if(!f)return"";return std::string(std::istreambuf_iterator<char>(f),{});}
static bool write_file(const std::string& path,const std::string& data){std::ofstream f(path);if(!f)return false;f<<data;return true;}
static std::vector<std::string> list_dir(const std::string& path){
    std::vector<std::string> out; DIR* d=opendir(path.c_str());if(!d)return out;
    dirent* e; while((e=readdir(d))){std::string n=e->d_name;if(n!="."&&n!="..")out.push_back(n);}
    closedir(d); std::sort(out.begin(),out.end()); return out;
}

// --- Save folders (extra places a saved session may live; user-added, persisted in <data>/folders.txt) ------------
static std::mutex g_folders_mu;
static std::vector<std::string> g_folders;
static std::string norm_dir(std::string p){
    for(char& c:p) if(c=='\\') c='/';
    while(p.size()>1&&p.back()=='/') p.pop_back();
    return p;
}
static bool has_dotdot(const std::string& p){
    size_t i=0; while(i<=p.size()){ size_t j=p.find('/',i); if(j==std::string::npos) j=p.size(); if(p.compare(i,j-i,"..")==0&&j-i==2) return true; i=j+1; }
    return false;
}
static void folders_load(const std::string& base){
    std::lock_guard<std::mutex> l(g_folders_mu); g_folders.clear();
    std::istringstream is(read_file(base+"/folders.txt")); std::string ln;
    while(std::getline(is,ln)){ while(!ln.empty()&&(ln.back()=='\r'||ln.back()==' ')) ln.pop_back(); if(!ln.empty()) g_folders.push_back(norm_dir(ln)); }
}
static void folders_save_locked(const std::string& base){ std::string o; for(auto& f:g_folders) o+=f+"\n"; write_file(base+"/folders.txt",o); }
static bool session_dir_ok(const std::string& dir,const std::string& base){
    std::string d=norm_dir(dir);
    if(d.empty()||has_dotdot(d)) return false;
    if(d=="nexus-data/sessions"||d==norm_dir(base)+"/sessions") return true;
    std::lock_guard<std::mutex> l(g_folders_mu);
    for(auto& f:g_folders) if(f==d) return true;
    return false;
}
// a session path must sit directly in the default sessions folder or in a folder the user added
static bool session_path_ok(const std::string& path,const std::string& base){
    std::string p=norm_dir(path); size_t k=p.find_last_of('/');
    if(k==std::string::npos) return !has_dotdot(p);                        // bare name: current directory (legacy)
    return session_dir_ok(p.substr(0,k),base);
}
static long long free_mb_of(const std::string& dir){
#ifdef _WIN32
    ULARGE_INTEGER fb,tb,tf; if(GetDiskFreeSpaceExA(dir.c_str(),&fb,&tb,&tf)) return (long long)(fb.QuadPart>>20); return -1;
#else
    struct statvfs sv; if(statvfs(dir.c_str(),&sv)==0) return (long long)((unsigned long long)sv.f_bavail*sv.f_frsize>>20); return -1;
#endif
}

// --- Storage ------------------------------------------------------------------

struct Project { std::string id,name,color,created_at; };
struct Chat {
    std::string id,project_id,title,created_at,updated_at,model_name,chat_format,system_prompt;
    std::vector<nexus::ChatMessage> messages;
};
struct PTemplate { std::string id,name,description,content; };

class Storage {
public:
    std::string base;
    void init(const std::string& dir) {
        base=dir; mkdir_p(base+"/projects"); mkdir_p(base+"/chats"); mkdir_p(base+"/templates"); mkdir_p(base+"/sessions"); mkdir_p(base+"/states"); mkdir_p(base+"/checkpoints");
        if(list_projects().empty()){Project p;p.id="default";p.name="General";p.color="#7c6af7";p.created_at=now_iso();save_project(p);}
    }
    std::vector<Project> list_projects(){
        std::vector<Project> out;
        for(auto& f:list_dir(base+"/projects")){
            if(f.size()<5||f.substr(f.size()-5)!=".json")continue;
            auto r=read_file(base+"/projects/"+f); if(r.empty())continue;
            Project p{jp_str(r,"id"),jp_str(r,"name","Untitled"),jp_str(r,"color","#7c6af7"),jp_str(r,"created_at")};
            if(!p.id.empty())out.push_back(p);
        }
        std::sort(out.begin(),out.end(),[](const Project&a,const Project&b){return a.created_at<b.created_at;});
        return out;
    }
    bool save_project(const Project& p){
        return write_file(base+"/projects/"+p.id+".json",
            "{\"id\":"+jstr(p.id)+",\"name\":"+jstr(p.name)+",\"color\":"+jstr(p.color)+",\"created_at\":"+jstr(p.created_at)+"}");
    }
    bool delete_project(const std::string& id){if(id=="default")return false;remove((base+"/projects/"+id+".json").c_str());return true;}

    std::vector<Chat> list_chats(const std::string& pid=""){
        std::vector<Chat> out;
        for(auto& f:list_dir(base+"/chats")){
            if(f.size()<5||f.substr(f.size()-5)!=".json")continue;
            auto r=read_file(base+"/chats/"+f); if(r.empty())continue;
            Chat c; c.id=jp_str(r,"id"); c.project_id=jp_str(r,"project_id","default");
            c.title=jp_str(r,"title","New Chat"); c.updated_at=jp_str(r,"updated_at");
            c.model_name=jp_str(r,"model_name"); c.chat_format=jp_str(r,"chat_format");
            if(c.id.empty())continue;
            if(!pid.empty()&&c.project_id!=pid)continue;
            out.push_back(c);
        }
        std::sort(out.begin(),out.end(),[](const Chat&a,const Chat&b){return a.updated_at>b.updated_at;});
        return out;
    }
    Chat load_chat(const std::string& id){
        Chat c; auto r=read_file(base+"/chats/"+id+".json"); if(r.empty())return c;
        c.id=jp_str(r,"id"); c.project_id=jp_str(r,"project_id","default");
        c.title=jp_str(r,"title"); c.model_name=jp_str(r,"model_name");
        c.chat_format=jp_str(r,"chat_format"); c.system_prompt=jp_str(r,"system_prompt");
        c.created_at=jp_str(r,"created_at"); c.updated_at=jp_str(r,"updated_at");
        // jp_messages searches for "messages": key - pass full file content, not just the array
        c.messages=jp_messages(r);
        return c;
    }
    bool save_chat(Chat& c){
        if(c.id.empty())c.id=gen_id();
        if(c.created_at.empty())c.created_at=now_iso();
        c.updated_at=now_iso();
        std::string j="{\"id\":"+jstr(c.id)+",\"project_id\":"+jstr(c.project_id)+
            ",\"title\":"+jstr(c.title)+",\"created_at\":"+jstr(c.created_at)+
            ",\"updated_at\":"+jstr(c.updated_at)+",\"model_name\":"+jstr(c.model_name)+
            ",\"chat_format\":"+jstr(c.chat_format)+",\"system_prompt\":"+jstr(c.system_prompt)+
            ",\"messages\":[";
        for(size_t i=0;i<c.messages.size();++i){
            if(i)j+=",";
            j+="{\"role\":"+jstr(c.messages[i].role)+",\"content\":"+jstr(c.messages[i].content)+"}";
        }
        j+="]}";
        return write_file(base+"/chats/"+c.id+".json",j);
    }
    bool delete_chat(const std::string& id){return remove((base+"/chats/"+id+".json").c_str())==0;}

    std::vector<PTemplate> list_templates(){
        std::vector<PTemplate> out;
        for(auto& f:list_dir(base+"/templates")){
            if(f.size()<5||f.substr(f.size()-5)!=".json")continue;
            auto r=read_file(base+"/templates/"+f); if(r.empty())continue;
            PTemplate t{jp_str(r,"id"),jp_str(r,"name"),jp_str(r,"description"),jp_str(r,"content")};
            if(!t.id.empty())out.push_back(t);
        }
        return out;
    }
    bool save_template(PTemplate& t){
        if(t.id.empty())t.id=gen_id();
        return write_file(base+"/templates/"+t.id+".json",
            "{\"id\":"+jstr(t.id)+",\"name\":"+jstr(t.name)+
            ",\"description\":"+jstr(t.description)+",\"content\":"+jstr(t.content)+"}");
    }
    bool delete_template(const std::string& id){return remove((base+"/templates/"+id+".json").c_str())==0;}
};

// --- HTTP helpers -------------------------------------------------------------

#ifdef _WIN32
static void send_all(SOCKET fd,const std::string& s){
#else
static void send_all(int fd,const std::string& s){
#endif
    size_t sent=0;
    while(sent<s.size()){ssize_t n=send(fd,s.c_str()+sent,s.size()-sent,MSG_NOSIGNAL);if(n<=0)break;sent+=n;}
}
#ifdef _WIN32
static std::string read_body(SOCKET fd,size_t len){
#else
static std::string read_body(int fd,size_t len){
#endif
    std::string b(len,'\0');size_t got=0;
    while(got<len){ssize_t n=recv(fd,&b[got],len-got,0);if(n<=0)break;got+=n;}
    b.resize(got);return b;
}
struct Req{std::string method,path,body;std::unordered_map<std::string,std::string>headers;bool valid=false;};
#ifdef _WIN32
static Req parse_req(SOCKET fd){
#else
static Req parse_req(int fd){
#endif
    Req r;std::string raw;char buf[8192];
    while(true){ssize_t n=recv(fd,buf,sizeof(buf),0);if(n<=0)return r;
        raw.append(buf,n);if(raw.find("\r\n\r\n")!=std::string::npos)break;if(raw.size()>2*1024*1024)return r;}
    std::istringstream ss(raw);std::string line;
    if(!std::getline(ss,line))return r;
    {std::istringstream l(line);std::string proto;l>>r.method>>r.path>>proto;}
    int clen=0;
    while(std::getline(ss,line)&&line!="\r"){
        auto c=line.find(':');if(c==std::string::npos)continue;
        std::string k=line.substr(0,c),v=line.substr(c+1);
        while(!v.empty()&&(v.front()==' '||v.front()=='\t'))v.erase(0,1);
        while(!v.empty()&&(v.back()=='\r'||v.back()=='\n'||v.back()==' '))v.pop_back();
        std::transform(k.begin(),k.end(),k.begin(),::tolower);
        r.headers[k]=v;if(k=="content-length")clen=atoi(v.c_str());
    }
    if(clen>0){size_t he=raw.find("\r\n\r\n");std::string have=raw.substr(he+4);
        if((int)have.size()<clen)have+=read_body(fd,clen-have.size());
        r.body=have.substr(0,clen);}
    r.valid=true;return r;
}
// Access-Control-Allow-Origin for the request this thread is answering: the caller's own origin when
// the access policy allows it (see nx_access_check), nothing otherwise -- never "*" unless --cors any.
static thread_local std::string tl_acao;
static std::string nx_acao(){ return tl_acao; }

#ifdef _WIN32
static void resp_json(SOCKET fd,const std::string& body,int code=200){
#else
static void resp_json(int fd,const std::string& body,int code=200){
#endif
    std::string s=code==200?"200 OK":code==404?"404 Not Found":code==401?"401 Unauthorized":code==403?"403 Forbidden":"400 Bad Request";
    send_all(fd,"HTTP/1.1 "+s+"\r\nContent-Type: application/json\r\n"+nx_acao()+"Connection: close\r\n\r\n"+body);
}
#ifdef _WIN32
static void resp_html(SOCKET fd,const std::string& body){
#else
static void resp_html(int fd,const std::string& body){
#endif
    send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"+body);
}
// --- Document readers for the web chat (PDF + OCR, Word, spreadsheets, slides, e-books) -------------------------
// They run in the browser and are served read-only from a "web/vendor" folder shipped next to the server program
// (NEXUS_WEB_DIR overrides), so dragging a document into the chat works offline on every platform.
#ifdef __APPLE__
#  include <mach-o/dyld.h>
#endif
static std::string nx_exe_dir(){
    std::string p;
#ifdef _WIN32
    char buf[4096]; DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    if(n > 0 && n < sizeof(buf)) p.assign(buf, n);
#elif defined(__APPLE__)
    char buf[4096]; uint32_t sz = (uint32_t)sizeof(buf);
    if(_NSGetExecutablePath(buf, &sz) == 0){ char rp[4096]; p = realpath(buf, rp) ? rp : buf; }
#else
    char buf[4096]; ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if(n > 0){ buf[n] = 0; p = buf; }
#endif
    const size_t k = p.find_last_of("/\\");
    return k == std::string::npos ? std::string(".") : p.substr(0, k);
}
static bool nx_file_exists(const std::string& f){ std::ifstream t(f, std::ios::binary); return (bool)t; }
static const std::string& nx_web_dir(){
    static const std::string d = [](){
        std::vector<std::string> c;
        if(const char* e = std::getenv("NEXUS_WEB_DIR")) if(*e) c.push_back(e);
        const std::string x = nx_exe_dir();
        c.push_back(x + "/web"); c.push_back(x + "/../web"); c.push_back("web");
        for(auto& w : c) if(nx_file_exists(w + "/vendor/VERSIONS.txt")) return w;
        return std::string();
    }();
    return d;
}
static const char* nx_vendor_mime(const std::string& f){
    auto ends = [&](const char* e){ const size_t n = strlen(e); return f.size() >= n && f.compare(f.size() - n, n, e) == 0; };
    if(ends(".js"))   return "application/javascript; charset=utf-8";
    if(ends(".wasm")) return "application/wasm";
    if(ends(".json")) return "application/json";
    if(ends(".txt") || ends(".md")) return "text/plain; charset=utf-8";
    return "application/octet-stream";   // .gz language data (unpacked by the OCR worker itself), .bcmap, .pfb, .ttf
}

static std::string pid_from_path(const std::string& p){
    // /api/projects/XXXXX/chats - XXXXX
    size_t a=p.find("/api/projects/");if(a==std::string::npos)return"";
    std::string r=p.substr(a+14);size_t sl=r.find('/');
    return sl==std::string::npos?r:r.substr(0,sl);
}

// --- Embedded UI --------------------------------------------------------------

static const char* UI = R"HTMLPAGE(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nexus</title>
<style>
:root{
  --bg:#09090f;--bg2:#0f0f1a;--bg3:#141425;--bg4:#1a1a2e;
  --brd:#1e1e35;--brd2:#28284a;
  --tx:#d0d0e8;--tx2:#7878a0;--tx3:#3a3a60;
  --acc:#7c6af7;--acc2:#5a47d4;--acc3:rgba(124,106,247,.13);
  --grn:#3dba7a;--red:#e05555;--ylw:#d4a520;
  --mono:'JetBrains Mono','Fira Mono',monospace;
  --ui:system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--tx);font-family:var(--ui);font-size:14px;overflow:hidden;-webkit-font-smoothing:antialiased}
#layout{display:flex;height:100vh}

/* -- Sidebar -- */
#sidebar{width:270px;min-width:270px;background:var(--bg2);border-right:1px solid var(--brd);display:flex;flex-direction:column;overflow:hidden}
#sb-logo{padding:12px 14px 10px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--brd);flex-shrink:0}
#sb-logo span{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--acc);letter-spacing:.12em}
#sb-status{margin-left:auto;font-size:9px;padding:2px 6px;border-radius:8px;background:var(--acc3);color:var(--acc);font-weight:600}
#sb-tabs{display:flex;border-bottom:1px solid var(--brd);flex-shrink:0}
.stab{flex:1;padding:7px 4px;font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;text-align:center;color:var(--tx3);cursor:pointer;border-bottom:2px solid transparent;transition:all .15s}
.stab:hover{color:var(--tx2)}.stab.on{color:var(--acc);border-color:var(--acc)}
#sb-body{flex:1;overflow-y:auto;padding:10px}

/* Sidebar sections */
.sec{margin-bottom:16px}
.sec-h{font-size:9px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--tx3);margin-bottom:7px;padding-bottom:4px;border-bottom:1px solid var(--brd)}
.sbtn{display:flex;align-items:center;gap:7px;width:100%;padding:7px 9px;border-radius:6px;border:1px solid var(--brd2);background:var(--bg3);color:var(--tx2);font-size:11.5px;cursor:pointer;font-family:var(--ui);margin-bottom:5px;transition:all .12s;text-align:left}
.sbtn:hover{border-color:var(--acc);color:var(--tx)}.sbtn .ic{font-size:13px;flex-shrink:0}
.sbtn.danger:hover{border-color:var(--red);color:var(--red)}
.sinput{width:100%;background:var(--bg);border:1px solid var(--brd2);border-radius:5px;padding:5px 8px;color:var(--tx);font-size:11px;font-family:var(--mono);outline:none;margin-bottom:4px}
.sinput:focus{border-color:var(--acc)}
.shint{font-size:10px;color:var(--tx3);margin-bottom:8px;line-height:1.5}
.srow{display:flex;align-items:center;gap:7px;margin-bottom:6px}
.srow label{font-size:11px;color:var(--tx2);flex:0 0 90px}
.srow input,.srow select{flex:1;background:var(--bg);border:1px solid var(--brd2);border-radius:5px;padding:4px 7px;color:var(--tx);font-size:11px;font-family:var(--ui);outline:none}
.srow input:focus,.srow select:focus{border-color:var(--acc)}
.stat{display:flex;justify-content:space-between;margin-bottom:3px;font-size:11px}
.stat .k{color:var(--tx3)}.stat .v{font-family:var(--mono);color:var(--tx2)}
#sess-list{margin-top:6px}
.sess-entry{display:flex;align-items:center;gap:5px;padding:5px 7px;background:var(--bg3);border:1px solid var(--brd);border-radius:6px;margin-bottom:4px}
.sess-name{flex:1;font-size:11.5px;font-weight:500;color:var(--tx);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sess-date{font-size:9px;color:var(--tx3);font-family:var(--mono)}
.sess-load{padding:3px 8px;border-radius:4px;border:1px solid var(--brd2);background:none;color:var(--tx2);font-size:10px;cursor:pointer;font-family:var(--ui);transition:all .12s}
.sess-load:hover{border-color:var(--acc);color:var(--acc)}
.sess-del{padding:3px 6px;border-radius:4px;border:1px solid transparent;background:none;color:var(--tx3);font-size:10px;cursor:pointer;font-family:var(--ui);transition:all .12s}
.sess-del:hover{border-color:var(--red);color:var(--red)}

/* -- Main area -- */
#main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}
#topbar{display:flex;align-items:center;gap:8px;padding:7px 14px;border-bottom:1px solid var(--brd);background:var(--bg2);flex-shrink:0}
#tb-sb{width:26px;height:26px;border-radius:5px;border:1px solid var(--brd2);background:none;color:var(--tx3);cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all .12s}
#tb-sb:hover{border-color:var(--acc);color:var(--acc)}
#tb-name{font-size:12px;font-weight:500;color:var(--tx2)}
#tb-fmt{font-size:9px;padding:2px 5px;border-radius:3px;background:var(--acc3);color:var(--acc);font-family:var(--mono);font-weight:600;margin-left:4px}
#tb-right{margin-left:auto;display:flex;align-items:center;gap:6px}
.tb-pill{font-family:var(--mono);font-size:10px;color:var(--tx3);padding:2px 7px;border-radius:4px;background:var(--bg3);border:1px solid var(--brd)}
.tbtn{padding:4px 10px;border-radius:5px;border:1px solid var(--brd2);background:var(--bg3);color:var(--tx2);font-size:11px;cursor:pointer;font-family:var(--ui);transition:all .12s}
.tbtn:hover{border-color:var(--acc);color:var(--tx)}
.tbtn.red:hover{border-color:var(--red);color:var(--red)}

/* -- Messages -- */
#msgs{flex:1;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;animation:fi .18s ease}
@keyframes fi{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.msg.user{flex-direction:row-reverse}
.av{width:26px;height:26px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;font-family:var(--mono);margin-top:3px}
.msg.user .av{background:rgba(124,106,247,.2);color:var(--acc)}
.msg.ai .av{background:rgba(61,186,122,.15);color:var(--grn)}
.bub{max-width:74%}
.msg.user .bub{margin-left:auto}
.bub-inner{padding:10px 13px;border-radius:10px;line-height:1.7;word-break:break-word;font-size:13.5px}
.msg.user .bub-inner{background:rgba(124,106,247,.1);border:1px solid rgba(124,106,247,.2);border-radius:10px 2px 10px 10px}
.msg.ai .bub-inner{background:var(--bg3);border:1px solid var(--brd);border-radius:2px 10px 10px 10px}
.msg-stats{font-family:var(--mono);font-size:9px;color:var(--tx3);padding:3px 2px 0;display:flex;gap:8px}
.cursor{display:inline-block;width:2px;height:13px;background:var(--acc);margin-left:2px;vertical-align:text-bottom;animation:bl .75s step-end infinite}
@keyframes bl{0%,100%{opacity:1}50%{opacity:0}}

/* -- Markdown -- */
.md p{margin:0 0 6px}.md p:last-child{margin:0}
.md h1{font-size:1.2em;font-weight:700;margin:10px 0 4px;color:#e8e8ff}
.md h2{font-size:1.08em;font-weight:600;margin:8px 0 3px;color:#e0e0f8}
.md h3{font-size:1em;font-weight:600;margin:6px 0 2px}
.md strong{font-weight:600;color:#e8e8ff}.md em{font-style:italic;color:var(--tx2)}
.md ul{margin:4px 0 4px 18px;list-style:disc}.md ol{margin:4px 0 4px 18px;list-style:decimal}
.md li{margin:2px 0;line-height:1.6}
.md blockquote{border-left:3px solid var(--acc2);padding:2px 0 2px 10px;margin:5px 0;color:var(--tx2)}
.md hr{border:none;border-top:1px solid var(--brd);margin:8px 0}
.md a{color:var(--acc);text-decoration:none}.md a:hover{text-decoration:underline}
.md code:not(pre code){background:rgba(0,0,0,.5);border:1px solid var(--brd2);border-radius:3px;padding:1px 5px;font-family:var(--mono);font-size:11px;color:#c8b8ff}
.md table{border-collapse:collapse;margin:6px 0;font-size:12px;width:100%}
.md th,.md td{border:1px solid var(--brd2);padding:4px 9px;text-align:left}
.md th{background:var(--bg4);font-weight:600}
.md tr:nth-child(even) td{background:rgba(255,255,255,.02)}
.code-wrap{position:relative;margin:7px 0;border-radius:7px;overflow:hidden;border:1px solid var(--brd2)}
.code-hdr{display:flex;align-items:center;justify-content:space-between;padding:4px 10px;background:rgba(0,0,0,.4);border-bottom:1px solid var(--brd)}
.code-lang{font-family:var(--mono);font-size:9px;color:var(--tx3);font-weight:600;text-transform:uppercase}
.code-copy{font-size:10px;padding:2px 7px;border-radius:3px;border:1px solid var(--brd2);background:none;color:var(--tx3);cursor:pointer;font-family:var(--ui);transition:all .12s}
.code-copy:hover{border-color:var(--acc);color:var(--acc)}.code-copy.ok{color:var(--grn);border-color:var(--grn)}
.code-wrap pre{padding:9px 11px;overflow-x:auto;background:rgba(0,0,0,.55);margin:0}
.code-wrap code{font-family:var(--mono);font-size:11.5px;line-height:1.5;color:#c8c8e8;white-space:pre}

/* -- Attachments -- */
.att-doc{margin-bottom:6px;border:1px solid var(--brd2);border-radius:6px;background:var(--bg4);font-size:11px;color:var(--tx2);max-width:100%}
.att-doc>summary{cursor:pointer;padding:3px 8px}
.att-doc pre{margin:0;padding:6px 8px;max-height:260px;overflow:auto;white-space:pre-wrap;word-break:break-word;border-top:1px solid var(--brd2);color:var(--tx);font-size:11px}
.att-img{max-width:220px;max-height:160px;border-radius:6px;margin-bottom:6px;display:block;border:1px solid var(--brd2)}
.att-file{display:inline-flex;align-items:center;gap:5px;padding:2px 7px;border-radius:4px;background:var(--bg4);border:1px solid var(--brd2);font-size:11px;color:var(--tx2);margin-bottom:5px}

/* -- Status bar -- */
#statusbar{display:flex;align-items:center;gap:10px;padding:3px 14px;background:var(--bg2);border-top:1px solid var(--brd);font-size:10px;font-family:var(--mono);color:var(--tx3);flex-shrink:0}
#sb-ctx{}.#sb-tps{color:#2ab0cc}.#sb-sess{margin-left:auto}

/* -- Input -- */
#input-area{padding:10px 14px 12px;border-top:1px solid var(--brd);background:var(--bg2);flex-shrink:0}
#att-row{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px}
.att-thumb{position:relative;border-radius:5px;overflow:hidden;border:1px solid var(--brd2)}
.att-thumb img{width:48px;height:48px;object-fit:cover;display:block}
.att-thumb .att-ico{width:48px;height:48px;display:flex;align-items:center;justify-content:center;background:var(--bg4);font-size:18px}
.att-thumb .att-nm{position:absolute;bottom:0;left:0;right:0;font-size:8px;background:rgba(0,0,0,.75);padding:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#fff;text-align:center}
.att-thumb.att-wait{border-color:var(--acc)}
.att-thumb.att-wait .att-nm{white-space:normal;font-size:8px;line-height:1.15}
.att-thumb .att-x{position:absolute;top:2px;right:2px;width:14px;height:14px;border-radius:50%;background:rgba(0,0,0,.8);border:none;color:#fff;font-size:9px;cursor:pointer;display:flex;align-items:center;justify-content:center;line-height:1;padding:0}
#irow{display:flex;gap:7px;align-items:flex-end}
#btn-att{width:34px;height:34px;border-radius:7px;border:1px solid var(--brd2);background:var(--bg3);color:var(--tx3);font-size:16px;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;transition:all .12s}
#btn-att:hover{border-color:var(--acc);color:var(--acc)}
#btn-dir{height:34px;padding:0 7px;border-radius:7px;border:1px solid var(--brd2);background:var(--bg3);color:var(--tx3);font-size:10px;font-weight:600;cursor:pointer;flex-shrink:0;transition:all .12s}
#btn-dir:hover{border-color:var(--acc);color:var(--acc)}
details.att-dir>summary{font-weight:600}
details.att-dir>.att-dir-body{padding-left:12px}
#fpick{display:none}
#msginp{flex:1;min-height:74px;background:var(--bg3);border:1px solid var(--brd2);border-radius:8px;padding:9px 12px;color:var(--tx);font-size:13.5px;font-family:var(--ui);outline:none;resize:none;max-height:160px;line-height:1.55;transition:border .12s}
#msginp:focus{border-color:var(--acc)}#msginp::placeholder{color:var(--tx3)}
#btn-stop{display:none;padding:9px 12px;border-radius:7px;border:1px solid var(--red);background:rgba(224,85,85,.1);color:var(--red);font-size:12px;font-weight:600;cursor:pointer;font-family:var(--ui);flex-shrink:0}
#btn-stop:hover{background:rgba(224,85,85,.2)}
#send-col{display:flex;flex-direction:column;gap:6px;flex-shrink:0}
#btn-think{display:none;padding:6px 16px;border-radius:7px;border:1px solid var(--acc);background:transparent;color:var(--acc);font-size:12px;font-weight:600;cursor:pointer;font-family:var(--ui);transition:background .12s}
#btn-think:hover:not(:disabled){background:rgba(124,106,247,.14)}
#btn-think:disabled{border-color:var(--bg4);color:var(--tx3);cursor:not-allowed}
#btn-send{padding:9px 16px;border-radius:7px;border:none;background:var(--acc);color:#fff;font-size:13px;font-weight:600;cursor:pointer;font-family:var(--ui);flex-shrink:0;transition:background .12s}
#btn-send:hover:not(:disabled){background:var(--acc2)}
#btn-send:disabled{background:var(--bg4);color:var(--tx3);cursor:not-allowed}

/* -- Drop overlay -- */
#drop-ov{position:fixed;inset:0;background:rgba(9,9,15,.88);z-index:200;display:none;align-items:center;justify-content:center;border:2px dashed var(--acc);margin:10px;border-radius:12px}
#drop-ov.show{display:flex}
#drop-ov div{text-align:center;color:var(--acc)}
#drop-ov .di{font-size:48px;margin-bottom:10px}
#drop-ov p{font-size:16px;font-weight:500}

/* -- Per-message tools & conversation memory -- */
.msg-tools{display:flex;gap:4px;align-items:center;margin-top:3px;opacity:0;transition:opacity .12s;min-height:18px}
.msg:hover .msg-tools,.msg.sel .msg-tools,.msg.kv-out .msg-tools{opacity:1}
.msg.user .msg-tools{justify-content:flex-end}
.mt-btn{font-size:10px;padding:1px 7px;border-radius:4px;border:1px solid var(--brd2);background:none;color:var(--tx3);cursor:pointer;font-family:var(--ui);line-height:1.5}
.mt-btn:hover{border-color:var(--acc);color:var(--acc)}.mt-btn.ok{color:var(--grn);border-color:var(--grn)}
.kv-tag{font-family:var(--mono);font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid var(--brd2);color:var(--tx3)}
.kv-tag.arch{border-color:var(--acc);color:var(--acc)}
.msg.kv-out .bub-inner{opacity:.45;border-style:dashed}
.msg.sel .bub-inner{outline:2px solid var(--acc);outline-offset:1px}
.kv-menu{position:fixed;z-index:300;background:var(--bg3);border:1px solid var(--brd2);border-radius:7px;padding:4px;min-width:230px;max-width:320px;box-shadow:0 6px 24px rgba(0,0,0,.45)}
.kv-menu .it{padding:6px 10px;font-size:12px;border-radius:5px;cursor:pointer;color:var(--tx)}
.kv-menu .it:hover{background:var(--bg4)}
.kv-menu .hint{font-size:10.5px;color:var(--tx3);padding:5px 10px;line-height:1.45}
.kv-menu .sep{border-top:1px solid var(--brd);margin:3px 0}
#kv-selbar{position:fixed;left:50%;transform:translateX(-50%);bottom:120px;z-index:250;display:none;gap:6px;align-items:center;background:var(--bg3);border:1px solid var(--acc);border-radius:8px;padding:6px 10px;font-size:12px;box-shadow:0 6px 24px rgba(0,0,0,.45)}
.mem-bar{height:6px;border-radius:3px;background:var(--bg4);overflow:hidden;margin:4px 0 6px}
.mem-bar>div{height:100%;background:var(--acc);width:0}
.mem-item{border:1px solid var(--brd2);border-radius:6px;padding:6px 8px;margin-bottom:5px;font-size:11px}
.mem-item .t{font-weight:600;color:var(--tx);word-break:break-word}
.mem-item .d{color:var(--tx3);font-size:10px;margin:2px 0 4px}
.mem-item button{font-size:10px;padding:2px 7px;margin-right:4px}

/* -- Toast -- */
#toasts{position:fixed;bottom:55px;right:14px;z-index:400;display:flex;flex-direction:column;gap:5px;align-items:flex-end}
.toast{padding:7px 12px;border-radius:7px;font-size:12px;font-weight:500;background:var(--bg4);border:1px solid var(--brd2);color:var(--tx);animation:fi .2s ease;max-width:300px}
.toast.ok{border-color:var(--grn);color:var(--grn)}.toast.err{border-color:var(--red);color:var(--red)}

/* -- Scrollbar -- */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--brd2);border-radius:2px}
</style>
</head>
<body>
<div id="drop-ov"><div><div class="di">-</div><p>Drop to attach (files or whole folders)</p></div></div>
<div id="toasts"></div>

<div id="conn-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:50;align-items:center;justify-content:center" onclick="if(event.target===this)closeConnect()">
  <div style="background:var(--bg2);border:1px solid var(--brd);border-radius:12px;width:min(560px,92vw);max-height:80vh;display:flex;flex-direction:column;padding:16px">
    <div style="display:flex;align-items:center;margin-bottom:6px"><div style="font-weight:600;flex:1">Connect to a session</div><button class="tbtn" onclick="refreshConnect()">Refresh</button><button class="tbtn" onclick="closeConnect()" style="margin-left:6px">Close</button></div>
    <div style="font-size:12px;color:var(--tx2);margin-bottom:8px">Chats the server is holding. A chat can only be open in one window at a time. If a reply was still being written when the page closed, it is shown from the start and continues live.</div>
    <div id="conn-list" style="overflow:auto"></div>
  </div>
</div>
<div id="kv-selbar"><span id="kv-selcount">0 selected</span>
  <button class="mt-btn" onclick="kvSelAction('forget')">Forget</button>
  <button class="mt-btn" onclick="kvSelAction('archive')">Archive</button>
  <button class="mt-btn" onclick="kvSelClear()">Cancel</button></div>
<div id="layout">
<!-- SIDEBAR -->
<div id="sidebar">
  <div id="sb-logo">
    <span>NEXUS</span>
    <span id="sb-status">connecting-</span>
  </div>
  <div id="sb-tabs">
    <div class="stab on" data-tab="sessions">Sessions</div>
    <div class="stab" data-tab="memory">Memory</div>
    <div class="stab" data-tab="settings">Settings</div>
  </div>
  <div id="sb-body">

    <!-- Sessions tab -->
    <div id="tab-sessions">
      <div class="sec">
        <div class="sec-h">Save / Load</div>
        <div class="shint">Type a name and Save. Load restores KV context + messages exactly.</div>
        <div style="display:flex;gap:5px;margin-bottom:5px">
          <input class="sinput" id="sess-name" placeholder="session name" style="flex:1;margin:0">
          <button class="sbtn" onclick="saveSession()" style="flex:0;padding:5px 10px;margin:0;white-space:nowrap">- Save</button>
        </div>
        <div style="display:flex;gap:5px;margin-bottom:5px">
          <select class="sinput" id="sess-folder" onchange="folderChanged()" style="flex:1;margin:0;min-width:0"></select>
          <button class="sbtn" onclick="addFolder()" title="Add another folder to save in or load from" style="flex:0;padding:5px 8px;margin:0;white-space:nowrap">+ Folder</button>
          <button class="sbtn" id="sess-folder-rm" onclick="removeFolder()" title="Forget this folder (files are kept)" style="flex:0;padding:5px 8px;margin:0;display:none">&minus;</button>
        </div>
        <div id="sess-list"></div>
      </div>
      <div class="sec">
        <div class="sec-h">Warm States</div>
        <div class="shint">Warm the prompt below, then save. Loading restores the conditioned KV - next message extends from that point.</div>
        <button class="sbtn" onclick="saveWarmState()"><span class="ic">-</span> Save Current Warm State</button>
        <div id="wstate-list" style="margin-top:6px"></div>
      </div>
      <div class="sec">
        <div class="sec-h">Checkpoints</div>
        <button class="sbtn" onclick="saveCheckpoint()"><span class="ic">-</span> Save Checkpoint</button>
        <div id="chk-list"></div>
      </div>
      <div class="sec">
        <div class="sec-h">Export</div>
        <div class="shint">Clean copies of the chat on screen, made to be posted or shared. (To keep a chat you can continue later, use Save above.)</div>
        <button class="sbtn" onclick="exportChatHTML()"><span class="ic">-</span> Export as web page (.html)</button>
        <button class="sbtn" onclick="exportChatMarkdown()"><span class="ic">-</span> Export as Markdown (.md)</button>

      </div>
    </div>

    <!-- Memory tab: what the model currently remembers, its limit, and saved (archived) parts -->
    <div id="tab-memory" style="display:none">
      <div class="sec">
        <div class="sec-h">Conversation memory</div>
        <div class="stat"><span class="k">In memory</span><span class="v" id="mem-used">-</span></div>
        <div class="mem-bar"><div id="mem-fill"></div></div>
        <div class="shint" id="mem-note">Hover a message and use its &hellip; button to forget it, archive it, or restore it.</div>
        <div class="sec-h" style="margin-top:8px">Limit</div>
        <select class="sinput" id="mem-limit-mode" onchange="document.getElementById('mem-limit-tok').style.display=this.value==='n'?'':'none'">
          <option value="max">The model's maximum</option>
          <option value="auto">What fits in memory (no limit)</option>
          <option value="n">A set number of tokens</option>
        </select>
        <input class="sinput" id="mem-limit-tok" type="number" min="1024" step="1024" placeholder="tokens, e.g. 32768" style="display:none">
        <label class="shint" style="display:flex;gap:6px;align-items:flex-start;cursor:pointer"><input type="checkbox" id="mem-autotrim" checked style="margin-top:2px"> When memory is full, drop the oldest messages and keep going (the system prompt is always kept). Off = stop.</label>
        <button class="sbtn" onclick="kvSaveSettings()">Apply</button>
      </div>
      <div class="sec">
        <div class="sec-h">Saved memories</div>
        <div class="shint">Parts of conversations you archived (A1, A2, &hellip;). Restore puts one back at the end of the current chat, so the model remembers it. They only fit the model they were saved from.</div>
        <div id="mem-arch"></div>
      </div>
    </div>

    <!-- Settings tab -->
    <div id="tab-settings" style="display:none">
      <div class="sec">
        <div class="sec-h">Model</div>
        <div class="stat"><span class="k">Name</span><span class="v" id="si-name">-</span></div>
        <div class="stat"><span class="k">Format</span><span class="v" id="si-fmt">-</span></div>
        <div class="stat"><span class="k">Layers</span><span class="v" id="si-lay">-</span></div>
        <div class="stat"><span class="k">Context</span><span class="v" id="si-ctx">-</span></div>
        <div class="stat"><span class="k">GPU</span><span class="v" id="si-gpu">-</span></div>
      </div>
      <div class="sec">
        <div class="sec-h">Documents</div>
        <div class="srow"><label>OCR for scanned PDF pages</label><select id="sd-ocr" onchange="try{localStorage.setItem('nexus_ocr',this.value)}catch(e){}"><option value="1">On</option><option value="0">Off</option></select></div>
        <div class="srow"><label>OCR language</label><input id="sd-ocrlang" type="text" placeholder="eng" onchange="try{localStorage.setItem('nexus_ocr_lang',this.value.trim()||'eng')}catch(e){}"></div>
        <div style="font-size:10px;color:var(--tx3);margin-top:3px;line-height:1.35">Drop PDFs, Word (.docx, .odt), spreadsheets (.xlsx, .xls, .ods, .csv), slides (.pptx, .odp), e-books (.epub), .rtf or web pages into the chat: they're read as text. Pages of a PDF without text are read with OCR. English is included; for other languages put e.g. <code>deu.traineddata.gz</code> into <code>web/vendor/tesseract/lang/</code> and enter <code>eng+deu</code>.</div>
      </div>
      <div class="sec">
        <div class="sec-h">Sampler (live - saves to nexus.json)</div>
        <div class="srow"><label>Temperature</label><input id="sc-temp" type="number" min="0" max="2" step="0.05" onchange="patchCfg()"></div>
        <div class="srow"><label>Min-P</label><input id="sc-minp" type="number" min="0" max="1" step="0.01" onchange="patchCfg()"></div>
        <div class="srow"><label>Top-P</label><input id="sc-topp" type="number" min="0" max="1" step="0.01" onchange="patchCfg()"></div>
        <div class="srow"><label>Top-K</label><input id="sc-topk" type="number" min="0" max="200" step="1" onchange="patchCfg()"></div>
        <div class="srow"><label>Rep Penalty</label><input id="sc-rep" type="number" min="1" max="2" step="0.05" onchange="patchCfg()"></div>
        <div class="srow"><label>Rep Last N</label><input id="sc-repn" type="number" min="0" max="2048" step="16" onchange="patchCfg()"></div>
        <div class="srow"><label>Freq Penalty</label><input id="sc-freq" type="number" min="0" max="2" step="0.05" onchange="patchCfg()"></div>
        <div class="srow"><label>Presence Penalty</label><input id="sc-pres" type="number" min="0" max="2" step="0.05" onchange="patchCfg()"></div>
        <div class="srow"><label>TFS-Z</label><input id="sc-tfsz" type="number" min="0" max="1" step="0.01" onchange="patchCfg()"></div>
        <div class="srow"><label>Max Tokens</label><input id="sc-maxt" type="number" min="64" max="131072" step="128" onchange="patchCfg()"></div>
        <div class="srow"><label></label><button id="sc-save" type="button" onclick="saveCfgDefaults()">Save as defaults</button></div>
      </div>
      <div class="sec">
        <div class="sec-h">Display</div>
        <label style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--tx2);cursor:pointer;margin-bottom:5px">
          <input type="checkbox" id="opt-md" checked style="accent-color:var(--acc)"> Render markdown
        </label>
        <label style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--tx2);cursor:pointer">
          <input type="checkbox" id="opt-ts" checked style="accent-color:var(--acc)"> Show timestamps
        </label>
      </div>
      <div class="sec">
        <div class="sec-h">System Prompt</div>
        <textarea id="sysprompt" style="width:100%;background:var(--bg);border:1px solid var(--brd2);border-radius:5px;padding:6px 8px;color:var(--tx);font-size:11px;font-family:var(--mono);outline:none;resize:vertical;min-height:65px" placeholder="Custom system prompt (applied to next message)-"></textarea>
        <button class="sbtn" style="margin-top:6px" onclick="warmPrompt()"><span class="ic">-</span> Warm Prompt</button>
        <div id="warm-hint" style="font-size:10px;color:var(--tx2);margin-top:3px"></div>
      </div>
    </div>

  </div>
</div>

<!-- MAIN -->
<div id="main">
  <div id="topbar">
    <button id="tb-sb" onclick="toggleSB()" title="Toggle sidebar">-</button>
    <span id="tb-name">Loading-</span><span id="tb-fmt"></span>
    <div id="tb-right">
      <span class="tb-pill" id="tb-tok">0 tok</span>
      <span class="tb-pill" id="tb-time">-</span>
      <button class="tbtn" onclick="newChat()">New Chat</button>
      <button class="tbtn" onclick="openConnect()" title="Reconnect to a chat the server is still holding, for example after closing or reloading this page">Connect to session</button>
      <button class="tbtn red" onclick="newChat()">Clear</button>
    </div>
  </div>
  <div id="taken-bar" style="display:none;background:rgba(224,85,85,.15);color:var(--red);padding:7px 14px;font-size:12px;border-bottom:1px solid var(--brd)">This chat was taken over by another window. This window is no longer connected to it - open "Connect to session" to take it back.</div>
  <div id="msgs"></div>
  <div id="statusbar">
    <span id="sb-ctx" title="Tokens the model has in memory right now (from the engine)">in memory: 0 tok</span>
    <span id="sb-tps"></span>
    <span id="sb-time2"></span>
    <span id="sb-sess" style="margin-left:auto"></span>
  </div>
  <div id="input-area">
    <div id="att-row"></div>
    <div id="irow">
      <button id="btn-att" onclick="document.getElementById('fpick').click()" title="Attach">-</button>
      <input id="fpick" type="file" multiple onchange="onPick(this)">
      <button id="btn-dir" onclick="document.getElementById('dpick').click()" title="Attach a folder: every text file in it, labeled with its path (you can also drag a folder in)">DIR</button>
      <input id="dpick" type="file" webkitdirectory multiple style="display:none" onchange="onPickDir(this)">
      <textarea id="msginp" placeholder="Message- (Shift+Enter = new line)" rows="2" onkeydown="onKey(event)" oninput="autoH(this)"></textarea>
      <div id="send-col">
        <button id="btn-send" onclick="generating?stopGen():send()">Send</button>
        <button id="btn-think" onclick="if(!generating)send('on')" title="Send this message and ask the model to reason before answering (this message only)">Send + Think</button>
      </div>
    </div>
  </div>
</div>
</div>

<script>
// API key: when the server asks for one (it was started with --api-key and this page is open on another
// computer), ask once, remember it in this browser, and send it with every request to this server.
(function(){
  var of = window.fetch.bind(window), asking = null;
  function key(){ try { return localStorage.getItem('nexus_api_key') || ''; } catch(e){ return ''; } }
  function same(u){ return typeof u === 'string' ? (u.charAt(0) === '/' || u.indexOf(location.origin) === 0) : (u && u.url && u.url.indexOf(location.origin) === 0); }
  function withKey(o){ var k = key(); if(!k) return o; o = Object.assign({}, o || {}); var h = new Headers(o.headers || {}); h.set('Authorization', 'Bearer ' + k); o.headers = h; return o; }
  window.fetch = function(u, o){
    if(!same(u)) return of(u, o);
    var k0 = key();
    return of(u, withKey(o)).then(function(r){
      if(r.status !== 401) return r;
      if(key() && key() !== k0) return of(u, withKey(o));       // a key was entered meanwhile (parallel request): just retry
      if(!asking) asking = Promise.resolve(window.prompt('This Nexus server needs its API key (it is set in the launcher on the computer running the model):', '')).then(function(k){
        asking = null; if(k){ try { localStorage.setItem('nexus_api_key', k.trim()); } catch(e){} } return !!k; });
      return asking.then(function(got){ return got ? of(u, withKey(o)) : r; });
    });
  };
})();
window.NEXUS_IMAGE_MAX_EDGE = __NEXUS_IMAGE_MAX_EDGE__;
window.NEXUS_IMAGE_MAX_PIXELS = __NEXUS_IMAGE_MAX_PIXELS__;
'use strict';
/* --- State ------------------------------------------------------------------- */
var msgs = [];          // [{role, content, time, atts, stats}]
var atts = [];          // pending attachments
var generating = false;
var BOOT = '';   // id of the server run this page last talked to; if it changes, the server was restarted
function fetchBoot(){ return fetch('/api/boot').then(function(r){return r.json();}).then(function(d){ return (d&&d.boot)||''; }); }
try{ fetchBoot().then(function(b){ if(!BOOT) BOOT=b; }).catch(function(){}); }catch(e){}
var abortCtrl = null;
var sessTokens = 0;     // tokens the model has generated in this chat (real engine count)
var lastCtxN = 0;       // tokens in memory now, as last reported by the engine
function showCtx(n){    // tokens in memory, always the engine's own count
  lastCtxN = n||0;
  var lim = (typeof kvState!=='undefined' && kvState && kvState.limit) ? (kvState.limit>=1e9 ? ' (no limit)' : ' / '+kvState.limit.toLocaleString()) : '';
  document.getElementById('sb-ctx').textContent = 'in memory: '+(n||0).toLocaleString()+lim+' tok';
}
var useExtend = false;  // true after loadSession - use extend_turn+continue
var convId = '';
var loadedSessionName = '';        // KV cache reuse ID - server returns on first /api/chat

/* --- Init --------------------------------------------------------------------- */
var SKEY='nexus_sessions_v1';
loadInfo();
setInterval(loadInfo, 15000);
// Defer so DOM is fully painted and localStorage is accessible
setTimeout(function(){ loadFolders(syncSessions); loadCkList(); loadWarmList(); }, 100);
setTimeout(function(){ apiPost('/api/live/list',{ui_id:UI_ID}).then(function(arr){ if(Array.isArray(arr)&&arr.some(function(c){return c.running&&!c.connected;})) toast('A reply is still being written on the server. Click "Connect to session" to pick it up.'); }).catch(function(){}); }, 900);
document.getElementById('msginp').focus();

function loadInfo(){
  fetch('/api/info').then(r=>r.json()).then(d=>{
    lastInfo = d;
    var n = (d.name||'Unknown').replace(/\.gguf$/i,'').replace(/[-_]/g,' ').slice(0,36);
    document.getElementById('tb-name').textContent = n;
    document.getElementById('tb-fmt').textContent  = d.chat_format||'';
    document.getElementById('btn-think').style.display = d.think_control ? 'block' : 'none';
    document.getElementById('si-name').textContent = d.name||'-';
    document.getElementById('si-fmt').textContent  = d.chat_format||'-';
    document.getElementById('si-lay').textContent  = (d.n_layers||'-')+' (GPU: '+(d.n_gpu_layers||0)+')';
    document.getElementById('si-ctx').textContent  = (d.context_length||'-')+' tok';
    if(d.context_length) window.NX_MODEL_MAX = d.context_length;
    if(typeof kvState!=='undefined' && !kvState && !generating) kvSync();   // limit settings, for the warning before sending
    document.getElementById('si-gpu').textContent  = d.gpu_device||'CPU only';
    document.getElementById('sb-status').textContent = '- '+d.arch;
    document.getElementById('sb-status').style.background = 'rgba(61,186,122,.15)';
    document.getElementById('sb-status').style.color = '#3dba7a';
  }).catch(()=>{
    document.getElementById('sb-status').textContent = '- offline';
    document.getElementById('sb-status').style.background = 'rgba(224,85,85,.1)';
    document.getElementById('sb-status').style.color = '#e05555';
  });
  fetch('/api/config').then(r=>r.json()).then(d=>{
    var sv = (id,v) => { var e=document.getElementById(id); if(document.activeElement!==e) e.value=v; };
    var nv = (x,def) => (x===undefined||x===null||isNaN(Number(x))) ? def : Number(x);
    sv('sc-temp', nv(d.temperature,.7).toFixed(2));
    sv('sc-minp', nv(d.min_p,.05).toFixed(3));
    sv('sc-topp', nv(d.top_p,.95).toFixed(2));
    sv('sc-topk', nv(d.top_k,40));
    sv('sc-rep',  nv(d.repeat_penalty,1.1).toFixed(2));
    sv('sc-repn', nv(d.repeat_last_n,64));
    sv('sc-freq', nv(d.frequency_penalty,0).toFixed(3));
    sv('sc-pres', nv(d.presence_penalty,0).toFixed(3));
    sv('sc-tfsz', nv(d.tfs_z,1).toFixed(3));
    sv('sc-maxt', nv(d.max_tokens,2048));
  }).catch(()=>{});
}

function saveCfgDefaults(){ apiPost('/api/config/save',{}).then(()=>{var b=document.getElementById('sc-save'); if(b){b.textContent='Saved'; setTimeout(()=>b.textContent='Save as defaults',1500);}}); }
function patchCfg(){
  var p = {};
  var nf=(id,k)=>{ var v=parseFloat(document.getElementById(id).value); if(!isNaN(v)) p[k]=v; };
  var ni=(id,k)=>{ var v=parseInt(document.getElementById(id).value);   if(!isNaN(v)) p[k]=v; };
  nf('sc-temp','temperature'); nf('sc-minp','min_p'); nf('sc-topp','top_p');
  ni('sc-topk','top_k'); nf('sc-rep','repeat_penalty'); ni('sc-repn','repeat_last_n');
  nf('sc-freq','frequency_penalty'); nf('sc-pres','presence_penalty');
  nf('sc-tfsz','tfs_z'); ni('sc-maxt','max_tokens');
  if(Object.keys(p).length) apiPost('/api/config',p);
}

/* --- New chat ---------------------------------------------------------------- */
function newChat(){
  try{ atts.forEach(releaseAttachment); msgs.forEach(m=>(m.atts||[]).forEach(releaseAttachment)); }catch(e){}
  msgs = []; atts = []; useExtend = false; sessTokens = 0;
  renderMsgs();
  renderAtts();
  document.getElementById('sb-sess').textContent = '';
  document.getElementById('tb-tok').textContent = '0 tok';
  document.getElementById('tb-time').textContent = '-';
  showCtx(0);
  document.getElementById('sb-tps').textContent = '';
  convId = ''; takenOver = false; lastTurnId = '';
  var tbr=document.getElementById('taken-bar'); if(tbr) tbr.style.display='none';
  kvSelClear();
  fetch('/api/reset',{method:'POST'}).catch(()=>{}).then(function(){ kvSync(); });
}


function imageScaleFor(srcW, srcH){
  var maxEdge = Number(window.NEXUS_IMAGE_MAX_EDGE || 0);
  var maxPixels = Number(window.NEXUS_IMAGE_MAX_PIXELS || 0);
  var sc = 1;
  if(maxEdge > 0) sc = Math.min(sc, maxEdge / Math.max(srcW, srcH));
  if(maxPixels > 0 && srcW > 0 && srcH > 0){
    var pixScale = Math.sqrt(maxPixels / (srcW * srcH));
    if(isFinite(pixScale)) sc = Math.min(sc, pixScale);
  }
  sc = Math.min(1, sc);
  return {maxEdge:maxEdge, maxPixels:maxPixels, scale:sc,
          w:Math.max(1, Math.round(srcW * sc)),
          h:Math.max(1, Math.round(srcH * sc))};
}

function uploadRgbPayload(name, width, height, rgb){
  var url = '/api/image/upload?name='+encodeURIComponent(name)+
            '&w='+encodeURIComponent(width)+'&h='+encodeURIComponent(height);
  return fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/octet-stream'},
    body:rgb
  }).then(function(r){return r.json();}).then(function(d){
    if(!d.ok) throw new Error(d.error || 'image upload failed');
    return {name:d.name || name, width:d.width || width, height:d.height || height, id:d.id};
  });
}

function imagePayloadFromSource(name, src){
  return new Promise(function(resolve,reject){
    var im=new Image();
    im.onload=function(){
      var c=null, ctx=null, rgba=null, rgb=null;
      try{
        var srcW=im.naturalWidth||im.width, srcH=im.naturalHeight||im.height;
        var sc=imageScaleFor(srcW, srcH);
        c=document.createElement('canvas');
        c.width=sc.w; c.height=sc.h;
        ctx=c.getContext('2d',{willReadFrequently:true});
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(im,0,0,sc.w,sc.h);
        rgba=ctx.getImageData(0,0,sc.w,sc.h).data;
        rgb=new Uint8Array(sc.w*sc.h*3);
        for(var i=0,j=0;i<rgba.length;i+=4){
          rgb[j++]=rgba[i]; rgb[j++]=rgba[i+1]; rgb[j++]=rgba[i+2];
        }
        var limits=[];
        if(sc.maxEdge > 0) limits.push('edge '+sc.maxEdge);
        if(sc.maxPixels > 0) limits.push('pixels '+sc.maxPixels);
        var mode = limits.length ? ('limit '+limits.join(',')) : 'full-size';
        console.log('[vision-upload] '+name+' '+srcW+'x'+srcH+' -> '+sc.w+'x'+sc.h+' '+mode+' raw_rgb='+rgb.byteLength);
        uploadRgbPayload(name, sc.w, sc.h, rgb).then(resolve).catch(reject).finally(function(){
          // Release canvas backing store and typed-array references ASAP.
          try{ c.width=0; c.height=0; }catch(e){}
          rgba=null; rgb=null; ctx=null; c=null;
        });
      }catch(e){
        try{ if(c){c.width=0;c.height=0;} }catch(_){}
        reject(e);
      }
    };
    im.onerror=function(){reject(new Error('image decode failed: '+name));};
    im.src=src;
  });
}

function imagePayloadFromAttachment(a){
  if(a && a.id && a.width && a.height)
    return Promise.resolve({name:a.name,width:a.width,height:a.height,id:a.id});
  if(a && a.objectUrl) return imagePayloadFromSource(a.name, a.objectUrl).then(function(x){releaseAttachment(a); a.objectUrl=''; return x;});
  if(a && a.dataUrl)   return imagePayloadFromSource(a.name, a.dataUrl);
  return Promise.resolve(null);
}

function collectImagePayloads(myAtts){
  var imgs=myAtts.filter(a=>a.type==='image');
  // Sequential processing keeps peak RAM bounded: one decoded image/canvas/RGB
  // payload at a time, never N images expanded together.
  var out=[];
  var p=Promise.resolve();
  imgs.forEach(function(a){
    p=p.then(function(){return imagePayloadFromAttachment(a);}).then(function(x){if(x)out.push(x);});
  });
  return p.then(function(){return out;});
}

/* --- Send --------------------------------------------------------------------- */
// Before sending: if this message takes the conversation past the model's own context limit (or the limit you set),
// say so and ask. It never blocks -- you decide. With "What fits in memory" it asks once, when crossing the line.
function ctxWarnOk(text, list){
  var d=(typeof kvState!=='undefined')?kvState:null;
  var mx=(d&&d.model_max)||window.NX_MODEL_MAX||0;
  if(!mx) return true;
  var chars=(text||'').length, imgs=0;
  (list||[]).forEach(function(a){ if(a.type==='image') imgs++; else if(typeof a.content==='string') chars+=a.content.length+(a.name||'').length+12; });
  var est=Math.round(chars/3.5)+imgs*576, now=lastCtxN, after=now+est;
  var cfg=d?d.limit_cfg:0, lim=(d&&d.limit)||mx, trim=d&&d.auto_trim;
  var pastMax=after>mx && now<=mx, pastLim=lim<1e9 && after>lim;
  if(!pastMax && !pastLim) return true;
  var f=function(n){ return Math.round(n).toLocaleString(); };
  var m='This message is about '+f(est)+' tokens, so the conversation will be about '+f(after)+' tokens. ';
  if(pastLim){
    m+='That is past '+(cfg>0?'the limit you set ('+f(lim)+' tokens)':"the model's own context limit ("+f(mx)+' tokens)')+'. ';
    m+=trim?'The oldest messages will be dropped to make room (the system prompt is kept).':'Auto-trim is off, so the reply will stop when the conversation reaches the limit, possibly right away.';
  } else {
    m+="That is past the model's own context limit ("+f(mx)+' tokens). ';
    m+=(cfg<0?'The limit is set to "What fits in memory", so it will keep going until memory runs out.':'Your limit is '+f(lim)+' tokens, so it will keep going.')+
       ' Past its limit a model often starts repeating itself or falls apart.';
  }
  return confirm(m+'\n\nSend it anyway?');
}
function send(mode){
  var think = (mode === 'on') ? 'on' : '';
  var inp = document.getElementById('msginp');
  var text = inp.value.trim();
  if((!text && !atts.length) || generating) return;
  if(takenOver){ toast('This chat is open in another window. Use Connect to session to take it back.','err'); return; }
  var pend = attachPending();
  if(pend.length){ toast('Still reading '+pend[0].name+(pend[0].status?' ('+pend[0].status+')':'')+' -- send again when its chip shows the name','err'); return; }
  if(!ctxWarnOk(text, atts)) return;              // kept in the box: nothing is sent
  inp.value = ''; inp.style.height = '';

  var myAtts = atts.slice(); atts = []; renderAtts();
  var userMsgObj = {role:'user', content:text, time:new Date(), atts:myAtts};
  msgs.push(userMsgObj);
  document.getElementById('msgs').appendChild(bubbleFor(userMsgObj));

  var aiBub = buildStreamBub();
  var t0 = Date.now();
  setBusy(true);

  collectImagePayloads(myAtts).then(function(imgPayloads){
    if(convId){
      // We have a conv_id - the model's KV cache is warm.
      // Append ONLY the new user turn, then generate from cache position.
      var ep = {text: userContent(text, myAtts), conv_id: convId, images: imgPayloads};
      if(think) ep.think = think;
      // Every turn has an id. The server remembers it, so repeating this request can
      // never add the prompt to the KV twice. We therefore keep retrying (slowly, with
      // no time limit) until the server answers - a wifi outage of any length is fine.
      // The server keeps prefilling the whole time; nothing is lost. Stop ends the retries.
      var turnId = Math.random().toString(36).slice(2) + Date.now().toString(36);
      ep.turn_id = turnId; lastTurnId = turnId; pushSnap();
      var started = false;
      var bootAtSend = BOOT;
      function startContinue(){
        if(started) return; started = true;
        doStream('/api/continue', {conv_id: convId, turn_id: turnId}, aiBub, t0);
      }
      // After a connection loss, never re-send this message to a server that has been restarted since: nothing may
      // happen on a fresh server until the person presses Send on a new message.
      function extendTry(n){
        if(started) return;
        if(!generating){ return; }                    // user pressed Stop
        if(n > 0 && bootAtSend){
          fetchBoot().then(function(b){
            if(b && b !== bootAtSend){
              toast('The server was restarted, so this message was NOT sent. Press Send again if you still want it sent.','err');
              hideConnBar(); setBusy(false); BOOT = b;
            } else extendGo(n);
          }).catch(function(){ scheduleRetry(function(){ extendTry(n+1); }, 3000); });
          return;
        }
        extendGo(n);
      }
      function extendGo(n){
        if(started) return;
        if(!generating){ return; }
        // The server sends a heartbeat byte every 10 s while it works. If nothing arrives
        // for 60 s the link is dead even though the browser has not noticed: drop it and retry.
        var ac = new AbortController(), lastByte = Date.now();
        var wd = setInterval(function(){ if(Date.now()-lastByte > 60000) ac.abort(); }, 5000);
        fetch('/api/session/extend_turn',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify(ep),
          signal:ac.signal
        }).then(function(r){
          var rd = r.body.getReader(), dec = new TextDecoder(), acc = '';
          function pump(){
            return rd.read().then(function(x){
              if(x.done) return acc;
              lastByte = Date.now(); acc += dec.decode(x.value,{stream:true});
              return pump();
            });
          }
          return pump();
        }).then(function(txt){
          clearInterval(wd);
          var ext; try{ ext = JSON.parse(txt.trim()); }catch(e){ throw new Error('incomplete reply'); }
          hideConnBar();
          if(n > 0) toast('Reconnected','ok');
          if(!ext.ok){
            toast('Could not continue this chat on the server ('+(ext.error||'no reason given')+'). Try sending again.','err');
            setBusy(false);
            return;
          }
          startContinue();
        }).catch(function(e){
          clearInterval(wd);
          if(n === 0) toast('Connection lost - the server keeps working; retrying until it answers...','err');
          var wait = Math.min(2000*Math.pow(1.5,n), 15000);
          scheduleRetry(function(){ extendTry(n+1); }, wait);
        });
      }
      extendTry(0);
      return;
    }

    // First message - generate a conversation ID right now, client-side.
    // Don't wait for the server to send one back. The server accepts conv_id
    // in the request body and uses it for KV cache tracking.
    // This guarantees convId is set before message 2 fires, with zero race conditions.
    convId = Math.random().toString(36).slice(2) + Date.now().toString(36);

    var sysP = document.getElementById('sysprompt').value.trim();
    var messages = [];
    if(sysP) messages.push({role:'system', content:sysP});
    msgs.forEach(function(m){
      if(m.role === 'note') return;
      if(m.kv === 'removed' || m.kv === 'archived' || m.kv === 'trimmed') return;   // forgotten: the model must not see it again
      messages.push({role:m.role, content:(m.role==='user'?userContent(m.content,m.atts||[]):m.content)});
    });
    // turn_id makes this request safely repeatable: if the link drops, the page re-attaches to the running reply (or re-sends it) without a double run.
    var chatBody = {messages:messages, conv_id:convId, images: imgPayloads, turn_id: Math.random().toString(36).slice(2) + Date.now().toString(36)};
    lastTurnId = chatBody.turn_id; pushSnap();
    if(think) chatBody.think = think;
    doStream('/api/chat', chatBody, aiBub, t0);
  }).catch(function(e){
    toast('Could not prepare that image: '+e.message,'err');
    setBusy(false);
  });
}

// doStream: single streaming function used by both /api/chat and /api/continue
// Captures conv_id from /api/chat response for all future turns.
function doStream(url, payload, aiBub, t0){
  var t1=0, t_gen=0, tc=0, full='', done=false, rt=null;

  function finish(){
    if(done) return; done=true;
    if(rt){ clearTimeout(rt); rt=null; }
    setBusy(false);
    var ttft    = t1 ? (t1/1000).toFixed(2)+'s' : '-';
    var gen_ms  = t_gen ? (Date.now() - t_gen) : 0;
    var tps     = (tc>0 && gen_ms>0) ? (tc/(gen_ms/1000)).toFixed(1) : '0';
    var elapsed = gen_ms > 0 ? (gen_ms/1000).toFixed(1)+'s' : '-';
    if(attachMode){ tps='-'; ttft='-'; }          // timings of a replayed reply would be meaningless
    finalizeAiBub(aiBub, full, tc, tps, ttft);
    sessTokens += tc;
    document.getElementById('sb-sess').textContent = 'generated this chat: '+sessTokens+' tok';
    document.getElementById('tb-tok').textContent  = tc+' tok';
    document.getElementById('tb-time').textContent = elapsed;
    scrollBot();
  }

  function sched(){
    if(!rt && !done)
      rt = setTimeout(function(){
        rt = null;
        if(!done) liveRender(aiBub, full);
      }, 80);
  }

  // /api/continue replies are buffered on the server, so if the connection drops we
  // re-attach with "everything after event N". The reply keeps generating meanwhile.
  // Backoff 2s,3s,5s,...,15s and at most 60 tries (~12 min), never a tight loop.
  var nev=0, gotDone=false, tries=0, nojob=false;
  var attachMode = (url === '/api/continue/resume');
  var bootAtStart = BOOT;   // this message's send-time boot id -- used below to tell a real
                            // server restart apart from an ordinary same-request retry race
  var resumable = (url === '/api/continue' || url === '/api/chat' || attachMode);   // both are buffered server-side and re-attachable

  function ended(){
    if(done) return;
    if(gotDone || !resumable){ finish(); return; }
    tries++;
    if(tries === 1) toast('Connection lost - the reply keeps generating; reconnecting...','err');
    var wait = Math.min(2000*Math.pow(1.5,tries-1), 15000);
    scheduleRetry(function(){
      if(done) return;
      if(!generating){ finish(); return; }            // user pressed Stop while we were waiting
      connect('/api/continue/resume', {from:String(nev), turn_id:(payload&&payload.turn_id)||'', model:(payload&&payload.model)||'1'});
    }, wait);
  }

  function connect(u, pl){
    var myCtrl = new AbortController(); abortCtrl = myCtrl;
    // Whether THIS SPECIFIC attempt is a resume/reconnect -- not the outer attachMode, which is
    // frozen from whatever url doStream() started with and stays false even once ended() has
    // switched to reconnecting via /api/continue/resume. Using that stale value here was the bug:
    // a reconnect whose job the server no longer holds was being treated as a fresh send to retry,
    // not as "that reply is gone."
    var thisAttach = (u === '/api/continue/resume');
    // Silent-link watchdog: the server sends something at least every 10 s (tokens or a ping). 45 s of nothing means the link is dead
    // even though the browser has not noticed (wifi flicker): abort it and re-attach instead of sitting there forever.
    var lastByte = Date.now(), wdFired = false;
    var wd = setInterval(function(){
      if(done || abortCtrl !== myCtrl){ clearInterval(wd); return; }
      if(Date.now() - lastByte > 45000){ wdFired = true; clearInterval(wd); try{ myCtrl.abort(); }catch(e){} }
    }, 5000);
    function aborted(){ clearInterval(wd); if(wdFired){ wdFired = false; ended(); } else finish(); }
    fetch(u, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(pl),
      signal: myCtrl.signal
    }).then(function(res){
      if(!res.ok) throw new Error('HTTP '+res.status);
      lastByte = Date.now();
      hideConnBar();
      if(tries > 0){ toast('Reconnected','ok'); tries = 0; }
      var reader = res.body.getReader();
      var dec    = new TextDecoder();
      var buf    = '';

      function read(){
        reader.read().then(function(r){
          lastByte = Date.now();
          if(r.done){ clearInterval(wd); ended(); return; }
          buf += dec.decode(r.value, {stream:true});
          var lines = buf.split('\n');
          buf = lines.pop();
          var stop = false;
          lines.forEach(function(line){
            if(stop || !line.startsWith('data:')) return;
            var d = line.slice(5).trim();
            if(d === '[DONE]'){ stop=true; gotDone=true; return; }
            try{
              var ev = JSON.parse(d);
              if(ev.nojob){ stop=true; nojob=true; return; }
              if(ev.conv_id === undefined && ev.progress === undefined) nev++;    // every event the server buffers counts, so a resume starts at the right place
              if(ev.progress && !t1){                // still reading the prompt
                var pg=ev.progress, eta=pg.eta>=0 ? (pg.eta>=60 ? ' \u00b7 about '+Math.floor(pg.eta/60)+'m '+(pg.eta%60)+'s left' : ' \u00b7 about '+pg.eta+'s left') : '';
                document.getElementById('sb-ctx').textContent = pg.moving_total
                  ? 'prompt read \u00b7 moving layers onto the card '+pg.moving_done+' of '+pg.moving_total
                  : 'reading prompt '+Math.round(pg.pct)+'% \u00b7 '+(pg.tps>0 ? pg.tps.toFixed(1)+' tok/s' : 'measuring speed')+eta;
              }
              if(ev.notice){ toast(ev.notice,'err'); }
              if(ev.final){                         // exact numbers from the engine at the end of the reply
                if(ev.gen !== undefined) tc = ev.gen;
                if(ev.context_tokens) showCtx(ev.context_tokens);
              }
              if(ev.token){
                if(!t1){ t1 = Date.now()-t0; t_gen = Date.now(); }
                full += ev.token;
                if(ev.gen !== undefined) tc = ev.gen; else tc++;   // real tokens (several per piece with MTP)
                if(ev.context_tokens) showCtx(ev.context_tokens);
                var gen_el = t_gen ? (Date.now()-t_gen)/1000 : 0;
                document.getElementById('sb-tps').textContent  = tc>0&&gen_el>0 ? (tc/gen_el).toFixed(1)+' t/s' : '';
                document.getElementById('sb-time2').textContent = gen_el > 0 ? gen_el.toFixed(1)+'s' : '';
                sched();
                scrollIfNear();
              }
            }catch(e){}
          });
          if(stop){
            clearInterval(wd);
            if(nojob && thisAttach){      // the reply is no longer held by the server (includes: the
                                           // server was restarted since this reply was last seen)
              done = true; setBusy(false);
              { var wr=aiBub && aiBub.parentNode; if(wr && wr.parentNode) wr.parentNode.removeChild(wr); }
              toast('That reply is no longer on the server','err'); return;
            }
            if(nojob){
              // This is NOT a resume attempt, so the server just told us (on the original send
              // endpoint) that it hasn't claimed this job yet -- normally a same-request timing
              // race, safe to retry as-is. But if the server was RESTARTED since this message was
              // sent, "hasn't claimed it" actually means "is a brand-new process with no memory of
              // this conversation at all" -- retrying would silently start an unrequested new
              // generation. Check the boot id before resending; only a person pressing Send again
              // should ever do that.
              nojob = false;
              fetchBoot().then(function(b){
                if(b && bootAtStart && b !== bootAtStart){
                  done = true; setBusy(false);
                  { var wr=aiBub && aiBub.parentNode; if(wr && wr.parentNode) wr.parentNode.removeChild(wr); }
                  toast('The server was restarted, so this message was NOT sent. Press Send again if you still want it sent.','err');
                  BOOT = b;
                  return;
                }
                setTimeout(function(){ if(!done && generating) connect(url, payload); else finish(); }, 500);
              }).catch(function(){
                // Could not confirm the server's boot id right now -- do not silently resend on
                // a guess either way; let the person decide.
                done = true; setBusy(false);
                toast('Lost contact with the server. Press Send again if you still want this message sent.','err');
              });
              return;
            }
            finish(); return;
          }
          read();
        }).catch(function(e){
          if(e && e.name === 'AbortError'){ aborted(); return; }   // user pressed Stop, or the watchdog fired
          clearInterval(wd); ended();
        });
      }
      read();
    }).catch(function(e){
      if(e && e.name === 'AbortError'){ aborted(); return; }
      clearInterval(wd);
      if(!resumable) toast('Something went wrong talking to the server ('+e.message+'). If the server window shows it has stopped, start the model again.','err');
      ended();
    });
  }
  connect(url, payload);
}

function streamContinue(aiBub, t0, t1, tc, full){
  // Legacy - kept for session/state load paths that call it directly
  doStream('/api/continue', convId ? {conv_id:convId} : {}, aiBub, t0);
}

function stopGen(){
  if(abortCtrl){abortCtrl.abort();abortCtrl=null;}
  // Also signal server-side stop
  fetch('/api/stop',{method:'POST'}).catch(()=>{});
  setBusy(false);
}

function setBusy(on){
  generating=on;
  if(!on) hideConnBar();
  var btn=document.getElementById('btn-send');
  var tb=document.getElementById('btn-think'); if(tb) tb.disabled=!!on;
  if(on){btn.textContent='- Stop';btn.style.background='rgba(224,85,85,.2)';btn.style.border='1px solid var(--red)';btn.style.color='var(--red)';}
  else{btn.textContent='Send';btn.style.background='';btn.style.border='';btn.style.color='';}
}

/* --- Bubble helpers ----------------------------------------------------------- */
function buildStreamBub(){
  var wrap=mk('div','msg ai'); var av=mk('div','av'); av.textContent='AI';
  var bub=mk('div','bub'); var inner=mk('div','bub-inner md');
  inner.innerHTML='<span class="cursor"></span>';
  bub.appendChild(inner); wrap.appendChild(av); wrap.appendChild(bub);
  document.getElementById('msgs').appendChild(wrap);
  scrollBot(); return bub;
}

function liveRender(bub, text){
  var inner=bub.querySelector('.bub-inner');
  if(!inner) return;
  if(document.getElementById('opt-md').checked)
    inner.innerHTML = renderMD(partialMD(text))+'<span class="cursor"></span>';
  else {
    inner.textContent=text;
    inner.appendChild(mk('span','cursor'));
  }
}

function finalizeAiBub(bub, text, tc, tps, ttft){
  var inner=bub.querySelector('.bub-inner');
  if(!inner) return;
  if(document.getElementById('opt-md').checked)
    inner.innerHTML = renderMD(text);
  else inner.textContent=text;
  var stats=mk('div','msg-stats');
  stats.textContent=tc+' tok - '+tps+' t/s - ttft '+ttft;
  bub.appendChild(stats);
  var am={role:'assistant',content:text,time:new Date(),stats:{tc,tps,ttft}};
  msgs.push(am);
  if(bub.parentNode) kvAttachTools(bub.parentNode,bub,am);
  pushSnap();
  setTimeout(function(){ kvSync(null,true); }, 400);   // see what the model now remembers (and whether old turns were dropped)
}

function readyImagePayloads(myAtts){
  // Compatibility helper for already-prepared image payloads.
  return myAtts.filter(a=>a.type==='image'&&a.id&&a.width&&a.height)
              .map(a=>({name:a.name,width:a.width,height:a.height,id:a.id}));
}

function userContent(text, myAtts){
  var parts=[], seen={};
  myAtts.forEach(a=>{
    if(a.group && !seen[a.group]){
      // an attached folder: list every file first, so the model sees the layout before reading them
      seen[a.group]=1;
      var mine=myAtts.filter(b=>b.group===a.group);
      parts.push('[Folder: '+a.groupName+'/ -- '+mine.length+' file'+(mine.length===1?'':'s')+', each attached below under its path]\n'
                 + mine.map(b=>'  '+b.name).join('\n'));
    }
    if(a.type==='image') parts.push('[Image: '+a.name+']');
    else if(a.content) parts.push('[File: '+a.name+']\n'+a.content);
  });
  if(text) parts.push(text);
  return parts.join('\n\n');
}

/* --- Render messages ----------------------------------------------------------- */
function renderMsgs(){
  var el=document.getElementById('msgs'); el.innerHTML='';
  msgs.forEach(m=>el.appendChild(bubbleFor(m)));
  scrollBot();
}

function bubbleFor(m){
  if(m.role==='note'){var n=mk('div');n.style.cssText='font-size:11px;color:var(--tx3);text-align:center;padding:8px;font-style:italic;';n.textContent=m.content;return n;}
  var wrap=mk('div','msg '+(m.role==='user'?'user':'ai'));
  var av=mk('div','av'); av.textContent=m.role==='user'?'You':'AI';
  var bub=mk('div','bub'); var inner=mk('div','bub-inner md');

  var dirBox={};
  if(m.atts&&m.atts.length) m.atts.forEach(a=>{
    var into=inner;
    if(a.group){
      if(!dirBox[a.group]){
        var n=m.atts.filter(b=>b.group===a.group).length;
        var dd0=document.createElement('details');dd0.className='att-doc att-dir';
        var sm0=document.createElement('summary');sm0.textContent='DIR '+a.groupName+'/ ('+n+' file'+(n===1?'':'s')+')';dd0.appendChild(sm0);
        dirBox[a.group]=mk('div','att-dir-body');dd0.appendChild(dirBox[a.group]);inner.appendChild(dd0);
      }
      into=dirBox[a.group];
    }
    if(a.type==='image'&&a.dataUrl){
      var img=document.createElement('img');img.src=a.dataUrl;img.className='att-img';inner.appendChild(img);
    } else if(a.type==='file'&&typeof a.content==='string'){
      // attached document: kept in full, collapsed by default (click the arrow to read it); filled in on first open so a huge file does not slow rendering
      var dd=document.createElement('details');dd.className='att-doc';
      var sm=document.createElement('summary');sm.textContent='- '+a.name+' ('+a.content.length.toLocaleString()+' chars)';dd.appendChild(sm);
      var pr=document.createElement('pre');dd.appendChild(pr);
      dd.addEventListener('toggle',function(){ if(dd.open&&!pr.dataset.f){ pr.textContent=a.content; pr.dataset.f='1'; } });
      into.appendChild(dd);
    } else {
      var f=mk('div','att-file');f.textContent='- '+a.name;inner.appendChild(f);
    }
  });

  if(document.getElementById('opt-md').checked&&m.role==='assistant')
    inner.innerHTML=(inner.innerHTML||'')+renderMD(m.content||'');
  else { var p=mk('p');p.style.whiteSpace='pre-wrap';p.textContent=m.content||'';inner.appendChild(p); }

  if(document.getElementById('opt-ts').checked&&m.time){
    var ts=mk('div','msg-stats');ts.style.opacity='.5';
    ts.textContent=fmtTime(m.time);
    if(m.stats) ts.textContent+=' - '+m.stats.tc+' tok - '+m.stats.tps+' t/s - ttft '+m.stats.ttft;
    bub.appendChild(inner);bub.appendChild(ts);
  } else bub.appendChild(inner);

  if(m.role==='user'){wrap.appendChild(bub);wrap.appendChild(av);}
  else {wrap.appendChild(av);wrap.appendChild(bub);}
  kvAttachTools(wrap,bub,m);
  return wrap;
}

/* --- Session save/load -------------------------------------------------------- */
// Session list loaded from disk via server. localStorage not used for sessions.
var _sessCache = [];  // in-memory cache, refreshed from disk
function getSaved(){ return _sessCache; }
function putSaved(a){ _sessCache = a; }
function syncSessions(cb){
  fetch("/api/sessions/list_in",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({folder:curFolder})}).then(function(r){return r.json();})
  .then(function(arr){
    _sessCache = Array.isArray(arr) ? arr.map(function(it){
      var m = it.meta || {name: String(it.base).replace(/_/g,' ')};
      m.path = curFolder + '/' + it.base + '.kvc'; m.mb = it.mb || 0; return m;
    }) : [];
    renderSessList();
    if(cb) cb();
  }).catch(function(){ if(cb) cb(); });
}

function saveSession(){
  var typedName=document.getElementById('sess-name').value.trim();
  var name = typedName || loadedSessionName;
  if(!name){toast('Enter a session name','err');return;}
  var safeName=name.replace(/[^a-zA-Z0-9_-]/g,'_');
  var path=curFolder+'/'+safeName+'.kvc';
  apiPost('/api/session/save',{path:path}).then(d=>{
    if(!d.ok){toast(d.error||'Save failed - is the model running?','err');return;}
    // Build metadata - stored on disk as path.json
    var slimMsgs=msgs.map(function(m){return {role:m.role,content:m.content,time:m.time,kv:m.kv,arch:m.arch,atts:(m.atts||[]).map(function(a){return a.type==='file'&&typeof a.content==='string'?{type:a.type,name:a.name,content:a.content,group:a.group,groupName:a.groupName}:{type:a.type,name:a.name};}),stats:m.stats||null};});
    var meta={name:name,path:path,date:new Date().toISOString(),count:msgs.length,msgs:slimMsgs};
    // Save metadata to disk via server
    apiPost('/api/sessions/meta',{path:path,meta:JSON.stringify(meta)}).then(function(){
      syncSessions();
      document.querySelectorAll('.stab').forEach(function(t){
        t.classList.toggle('on', t.dataset.tab==='sessions');
      });
      ['sessions','settings','memory'].forEach(function(n){
        document.getElementById('tab-'+n).style.display=(n==='sessions')?'':'none';
      });
      toast('Saved "'+name+'" ('+msgs.length+' msgs, '+(d.context_tokens||0)+' tok)');
    });
  });
}

function loadSession(name){
  // If not in cache yet, sync from disk first
  var arr=getSaved(); var entry=arr.find(s=>s.name===name);
  if(!entry){toast('Not found - try refreshing','err');return;}
  apiPost('/api/session/load',{path:entry.path}).then(d=>{
    if(!d.ok){toast(d.error||'Load failed','err');return;}
    // Set convId - send() uses extend_turn+continue, NOT /api/chat
    convId = 'loaded_' + Math.random().toString(36).slice(2) + Date.now().toString(36); takenOver = false;
    loadedSessionName = name;
    if(entry.msgs&&entry.msgs.length){
      msgs=entry.msgs.map(function(m){return Object.assign({},m,{time:m.time?new Date(m.time):new Date()});});
    } else {
      msgs=[{role:'note',content:'Session "'+name+'" loaded - context restored. Message history unavailable (older save).',time:new Date(),atts:[]}];
    }
    renderMsgs();
    sessTokens=0;
    document.getElementById('sb-sess').textContent='';
    showCtx(d.context_tokens||0);
    loadedSessionName = name;
    document.querySelectorAll('.stab').forEach(function(t){
      t.classList.toggle('on', t.dataset.tab==='sessions');
    });
    ['sessions','settings','memory'].forEach(function(n){
      document.getElementById('tab-'+n).style.display=(n==='sessions')?'':'none';
    });
    renderSessList();
    toast('- "'+name+'" loaded - '+(d.context_tokens||0)+' tok in memory. Continue typing.');
    kvSync(null,false);
  });
}

function deleteSession(name){
  if(!confirm('Delete saved session "'+name+'"?')) return;
  var entry=getSaved().find(s=>s.name===name);
  if(!entry){renderSessList();return;}
  apiPost('/api/session/delete',{path:entry.path}).then(function(d){
    if(!d.ok){toast(d.error||'Delete failed','err');return;}
    syncSessions();
    toast('Deleted "'+name+'"');
  }).catch(function(){toast('Could not delete that save. Is its folder still there?','err');});
}

function renderSessList(){
  var el=document.getElementById('sess-list'); el.innerHTML='';
  var arr=getSaved();
  if(!arr.length){
    var em=mk('div');em.style.cssText='font-size:10px;color:var(--tx3);padding:3px 2px';
    em.textContent='No saved sessions.';el.appendChild(em);return;
  }
  arr.forEach(s=>{
    var row=mk('div','sess-entry');
    var info=mk('div');info.style.flex='1';info.style.minWidth='0';
    var nm=mk('div','sess-name');nm.textContent=s.name;
    var dt=mk('div','sess-date');
    dt.textContent=(s.date?s.date.slice(0,16).replace('T',' '):'')+(s.count?' - '+s.count+' msgs':'')+(s.mb?' - '+(s.mb>=1024?(s.mb/1024).toFixed(1)+' GB':s.mb+' MB'):'');
    info.appendChild(nm);info.appendChild(dt);
    var lb=mk('button','sess-load');lb.textContent='Load';lb.onclick=()=>loadSession(s.name);
    var db=mk('button','sess-del');db.textContent='-';db.onclick=()=>deleteSession(s.name);
    row.appendChild(info);row.appendChild(lb);row.appendChild(db);
    el.appendChild(row);
  });
}

/* --- Warm state --------------------------------------------------------------- */
function saveWarmState(){
  if(!convId){toast('Warm the prompt first (- Warm Prompt below)','err');return;}
  var name=prompt('Name for this warm state:','');
  if(!name) return;
  apiPost('/api/start_state/save',{conv_id:convId,name:name}).then(d=>{
    if(d.ok){toast('Warm state saved: '+name);loadWarmList();}
    else toast(d.error||'Save failed','err');
  });
}
function loadWarmList(){
  fetch('/api/start_state/list').then(r=>r.json()).then(list=>{
    var el=document.getElementById('wstate-list');el.innerHTML='';
    if(!Array.isArray(list)||!list.length){el.innerHTML='<div style="color:var(--tx2);font-size:11px;padding:4px 0">No warm states saved</div>';return;}
    list.forEach(function(ws){
      var row=mk('div');row.style.cssText='display:flex;align-items:center;gap:5px;margin-bottom:4px;font-size:11px';
      var info=mk('span');info.style.flex='1';info.style.color='var(--tx2)';
      info.textContent=ws.name+' ('+ws.n_tokens+' tok)';
      var lb=mk('button','sess-load');lb.textContent='Load';lb.style.fontSize='10px';
      lb.onclick=function(){ apiPost('/api/start_state/load',{path:ws.path}).then(function(d){
        if(d.ok){convId=d.conv_id;useExtend=true;toast('Warm state loaded: '+ws.name+' ('+d.context_size+' tok)');}
        else toast(d.error||'Load failed','err');
      }); };
      var db=mk('button','sbtn');db.textContent='-';db.style.cssText='font-size:10px;padding:2px 6px;background:#c0392b;border-color:#c0392b';
      db.onclick=function(){ apiPost('/api/start_state/delete',{path:ws.path}).then(function(d){if(d.ok)loadWarmList();}); };
      row.appendChild(info);row.appendChild(lb);row.appendChild(db);el.appendChild(row);
    });
  }).catch(function(){});
}

function warmPrompt(){
  var sysP = document.getElementById('sysprompt').value.trim();
  var hint  = document.getElementById('warm-hint');
  var messages = sysP ? [{role:'system',content:sysP}] : [];
  hint.textContent = 'Warming-';
  apiPost('/api/preload',{messages:messages,reset:1}).then(d=>{
    if(d.ok){
      convId    = d.conv_id;
      useExtend = true;
      hint.textContent = 'Ready - '+d.context_size+' tokens in KV';
      toast('Prompt warmed ('+d.context_size+' tok)');
    } else {
      hint.textContent = '';
      toast(d.error||'Warm failed','err');
    }
  });
}

/* --- Checkpoints -------------------------------------------------------------- */
function saveCheckpoint(){
  var label=prompt('Checkpoint name (optional):','');
  if(label===null) return;
  apiPost('/api/checkpoint/save',{name:label||('chk_'+new Date().toLocaleTimeString())}).then(d=>{
    if(d.ok){toast('Checkpoint saved');loadCkList();}
    else toast(d.error||'Failed','err');
  });
}
function loadCkList(){
  fetch('/api/checkpoint/list').then(r=>r.json()).then(list=>{
    var el=document.getElementById('chk-list');el.innerHTML='';
    if(!Array.isArray(list)||!list.length){el.innerHTML='<div style="color:var(--tx2);font-size:11px;padding:4px 0">No checkpoints saved</div>';return;}
    list.forEach(c=>{
      var row=mk('div');row.style.cssText='display:flex;align-items:center;gap:5px;margin-bottom:4px;font-size:11px';
      var info=mk('span');info.style.flex='1';info.style.color='var(--tx2)';
      info.textContent=c.name+' ('+c.n_tokens+' tok)';
      var rb=mk('button','sess-load');rb.textContent='Restore';rb.style.fontSize='10px';
      rb.onclick=()=>apiPost('/api/checkpoint/restore',{path:c.path}).then(d=>{
        if(d.ok){useExtend=true;loadMsgs();toast('Restored checkpoint');}else toast('Failed','err');
      });
      var db=mk('button','sbtn');db.textContent='-';db.style.cssText='font-size:10px;padding:2px 6px;background:#c0392b;border-color:#c0392b';
      db.onclick=()=>apiPost('/api/checkpoint/delete',{path:c.path}).then(d=>{if(d.ok)loadCkList();});
      row.appendChild(info);row.appendChild(rb);row.appendChild(db);el.appendChild(row);
    });
  }).catch(()=>{});
}

/* --- Export ------------------------------------------------------------------- */

/* --- Save folders ------------------------------------------------------------- */
var DEF_FOLDER='nexus-data/sessions', curFolder=DEF_FOLDER, _folders=[];
try{ var _sf=localStorage.getItem('nx_folder'); if(_sf) curFolder=_sf; }catch(e){}
function normFolder(p){ p=(p||'').replace(/\\/g,'/'); while(p.length>1&&p.charAt(p.length-1)==='/') p=p.slice(0,-1); return p; }
function fmtFree(mb){ return mb>=1024 ? (mb/1024).toFixed(1)+' GB' : mb+' MB'; }
function loadFolders(cb){
  fetch('/api/folders').then(function(r){return r.json();}).then(function(a){
    _folders=Array.isArray(a)?a:[];
    var sel=document.getElementById('sess-folder'); if(!sel){ if(cb) cb(); return; }
    sel.innerHTML='';
    _folders.forEach(function(f){
      var o=document.createElement('option'); o.value=f.path;
      o.textContent=(f.default?'Nexus folder (default)':f.path)+(f.free_mb>=0?'  -  '+fmtFree(f.free_mb)+' free':'')+(f.exists===false?'  (missing)':'');
      sel.appendChild(o);
    });
    if(!_folders.some(function(f){return f.path===curFolder;})) curFolder=DEF_FOLDER;
    sel.value=curFolder;
    var rm=document.getElementById('sess-folder-rm'); if(rm) rm.style.display=(curFolder===DEF_FOLDER)?'none':'';
    if(cb) cb();
  }).catch(function(){ if(cb) cb(); });
}
function folderChanged(){
  curFolder=document.getElementById('sess-folder').value;
  try{ localStorage.setItem('nx_folder',curFolder); }catch(e){}
  loadFolders(syncSessions);
}
function addFolder(){
  var p=prompt('Folder to keep saved sessions in (full path), for example\nD:\\Nexus saves     or     /mnt/usb/saves\nIt is created if it does not exist.');
  if(!p||!p.trim()) return;
  apiPost('/api/folders/add',{path:p.trim()}).then(function(d){
    if(!d.ok){ toast(d.error||'Could not add that folder','err'); return; }
    curFolder=normFolder(p.trim());
    try{ localStorage.setItem('nx_folder',curFolder); }catch(e){}
    loadFolders(syncSessions); toast('Folder added');
  });
}
function removeFolder(){
  if(curFolder===DEF_FOLDER) return;
  if(!confirm('Remove "'+curFolder+'" from the list?\nThe saved files in it are NOT deleted.')) return;
  apiPost('/api/folders/remove',{path:curFolder}).then(function(){ curFolder=DEF_FOLDER; try{ localStorage.setItem('nx_folder',curFolder); }catch(e){} loadFolders(syncSessions); });
}

/* --- Export / import a chat --------------------------------------------------- */
var lastInfo={};
function escH(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function chatTitle(){
  if(loadedSessionName) return loadedSessionName;
  var f=msgs.find(function(m){return m.role==='user'&&m.content;});
  var t=f?f.content.split('\n')[0].trim():''; return t?t.slice(0,90):'Nexus chat';
}
function chatSummary(){
  var n=0,tok=0,tps=0,tn=0,ttft='';
  msgs.forEach(function(m){ if(m.role==='note') return; n++;
    if(m.role==='assistant'&&m.stats){ tok+=Number(m.stats.tc)||0; var v=parseFloat(m.stats.tps); if(!isNaN(v)){tps+=v;tn++;} if(!ttft&&m.stats.ttft) ttft=m.stats.ttft; } });
  return {messages:n, tokens:tok, tps:tn?(tps/tn).toFixed(1):'', ttft:ttft};
}
function downloadText(name,mime,text){
  var a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type:mime}));
  a.download=name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function(){URL.revokeObjectURL(a.href);},2000);
}
function fileStamp(){ var d=new Date(); function p(x){return String(x).padStart(2,'0');} return d.getFullYear()+p(d.getMonth()+1)+p(d.getDate())+'_'+p(d.getHours())+p(d.getMinutes()); }
function safeFile(s){ return (s||'chat').replace(/[^a-zA-Z0-9_-]+/g,'_').replace(/^_+|_+$/g,'').slice(0,50)||'chat'; }
function mdWithThink(text){
  // <think>...</think> blocks become collapsed sections
  var out='', parts=String(text||'').split(/(<think>[\s\S]*?<\/think>)/);
  parts.forEach(function(p){
    var m=p.match(/^<think>([\s\S]*?)<\/think>$/);
    if(m) out+='<details class="think"><summary>Thinking</summary><div class="md">'+renderMD(m[1].trim())+'</div></details>';
    else if(p.trim()) out+=renderMD(p);
  });
  return out.replace(/<button class="code-copy"[^>]*>Copy<\/button>/g,'');
}
function exportChatHTML(){
  if(!msgs.length){ toast('Nothing to export','err'); return; }
  var title=chatTitle(), sm=chatSummary(), info=lastInfo||{};
  var meta=[];
  if(info.name) meta.push('<span><b>Model</b> '+escH(info.name)+'</span>');
  if(info.chat_format) meta.push('<span><b>Format</b> '+escH(info.chat_format)+'</span>');
  meta.push('<span><b>Messages</b> '+sm.messages+'</span>');
  if(sm.tokens) meta.push('<span><b>Generated</b> '+sm.tokens.toLocaleString()+' tokens</span>');
  if(sm.tps) meta.push('<span><b>Speed</b> '+sm.tps+' tok/s</span>');
  if(sm.ttft) meta.push('<span><b>First token</b> '+escH(sm.ttft)+'</span>');
  meta.push('<span><b>Exported</b> '+escH(new Date().toLocaleString())+'</span>');
  var body='';
  msgs.forEach(function(m){
    if(m.role==='note') return;
    var who=m.role==='user'?'You':'Assistant', cls=m.role==='user'?'user':'ai', h='';
    (m.atts||[]).forEach(function(a){
      if(a.type==='image'&&a.dataUrl) h+='<img class="att-img" src="'+a.dataUrl+'" alt="'+escH(a.name)+'">';
      else if(a.type==='file'&&typeof a.content==='string') h+='<details class="att"><summary>'+escH(a.name)+' ('+a.content.length.toLocaleString()+' characters)</summary><pre>'+escH(a.content)+'</pre></details>';
      else if(a.name) h+='<div class="attname">'+escH(a.name)+'</div>';
    });
    if(m.role==='assistant') h+=mdWithThink(m.content);
    else h+='<p class="plain">'+escH(m.content||'').replace(/\n/g,'<br>')+'</p>';
    var st=(m.role==='assistant'&&m.stats)?'<div class="st">'+escH(m.stats.tc)+' tok &middot; '+escH(m.stats.tps)+' tok/s &middot; first token '+escH(m.stats.ttft)+'</div>':'';
    body+='<section class="msg '+cls+'"><div class="who">'+who+'</div><div class="bub md">'+h+st+'</div></section>';
  });
  var css='*{box-sizing:border-box}body{margin:0;font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f6f9;color:#1d1d2b}'
   +'@media(prefers-color-scheme:dark){body{background:#101018;color:#dcdcec}.bub{background:#1a1a28!important;border-color:#2a2a40!important}.msg.user .bub{background:#232344!important}pre,code{background:#0b0b12!important}details.att,details.think{border-color:#2a2a40!important}}'
   +'main{max-width:860px;margin:0 auto;padding:28px 18px 60px}h1.t{font-size:1.5em;margin:0 0 8px}.meta{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;opacity:.75;margin-bottom:24px;padding-bottom:14px;border-bottom:1px solid #8884}.meta b{font-weight:600;margin-right:4px}'
   +'.msg{margin:16px 0}.who{font-size:11px;letter-spacing:.08em;text-transform:uppercase;opacity:.55;margin:0 0 4px 4px}.bub{background:#fff;border:1px solid #dcdce8;border-radius:12px;padding:12px 16px;overflow-wrap:anywhere}.msg.user .bub{background:#ecebff}'
   +'.bub p{margin:0 0 8px}.bub p:last-child{margin:0}.bub h1,.bub h2,.bub h3{margin:14px 0 6px;line-height:1.3}.bub ul,.bub ol{margin:6px 0 6px 22px}.bub blockquote{border-left:3px solid #8a7cf0;margin:8px 0;padding:2px 12px;opacity:.85}'
   +'pre{background:#f0f0f6;border-radius:8px;padding:10px 12px;overflow:auto;font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace;white-space:pre}code{font:12.5px ui-monospace,Menlo,Consolas,monospace;background:#8882;padding:1px 4px;border-radius:4px}pre code{background:none;padding:0}'
   +'.code-hdr{font-size:11px;opacity:.6;margin:8px 0 2px}.code-wrap pre{margin-top:0}table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #8886;padding:4px 9px}'
   +'details.att,details.think{border:1px solid #dcdce8;border-radius:8px;padding:6px 10px;margin:0 0 8px}summary{cursor:pointer;font-size:13px;opacity:.8}details.think{opacity:.8}.att-img{max-width:100%;border-radius:8px;margin-bottom:8px}.st{font-size:11.5px;opacity:.5;margin-top:8px}'
   +'footer{margin-top:36px;font-size:12px;opacity:.5;text-align:center}';
  var html='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escH(title)+'</title><style>'+css+'</style></head><body><main><h1 class="t">'+escH(title)+'</h1><div class="meta">'+meta.join('')+'</div>'+body+'<footer>Generated locally with Nexus</footer></main></body></html>';
  downloadText('nexus_'+safeFile(title)+'_'+fileStamp()+'.html','text/html',html);
  toast('Exported as a web page');
}
function exportChatMarkdown(){
  if(!msgs.length){ toast('Nothing to export','err'); return; }
  var title=chatTitle(), sm=chatSummary(), info=lastInfo||{};
  function fence(t){ var run=0, mx=0; (t.match(/`+/g)||[]).forEach(function(r){ mx=Math.max(mx,r.length); }); return '`'.repeat(Math.max(3,mx+1)); }
  var o='# '+title.replace(/\n/g,' ')+'\n\n';
  var bits=[]; if(info.name) bits.push('Model: '+info.name); bits.push(sm.messages+' messages'); if(sm.tokens) bits.push(sm.tokens+' tokens generated'); if(sm.tps) bits.push(sm.tps+' tok/s'); if(sm.ttft) bits.push('first token '+sm.ttft);
  o+='_'+bits.join(' | ')+' | exported '+new Date().toLocaleString()+'_\n\n---\n\n';
  msgs.forEach(function(m){
    if(m.role==='note') return;
    o+='## '+(m.role==='user'?'You':'Assistant')+'\n\n';
    (m.atts||[]).forEach(function(a){
      if(a.type==='file'&&typeof a.content==='string'){ var f=fence(a.content); o+='<details><summary>'+a.name+' ('+a.content.length+' characters)</summary>\n\n'+f+'\n'+a.content+'\n'+f+'\n\n</details>\n\n'; }
      else if(a.name) o+='_[attachment: '+a.name+']_\n\n';
    });
    var c=m.content||'';
    if(m.role==='assistant') c=c.replace(/<think>([\s\S]*?)<\/think>/g,function(_,t){ return '<details><summary>Thinking</summary>\n\n'+t.trim()+'\n\n</details>\n\n'; });
    o+=c.trim()+'\n\n';
  });
  downloadText('nexus_'+safeFile(title)+'_'+fileStamp()+'.md','text/markdown',o);
  toast('Exported as Markdown');
}

/* --- Connect to a session on the server (after the page was closed / reloaded) ------------------------------------- */
var UI_ID = Math.random().toString(36).slice(2)+Date.now().toString(36), lastTurnId = '', takenOver = false;
function slimForServer(){
  return msgs.filter(function(m){return m.role!=='note';}).map(function(m){
    return {role:m.role, content:m.content, time:m.time, stats:m.stats, kv:m.kv, arch:m.arch,
            atts:(m.atts||[]).map(function(a){ return (a.type==='file'&&typeof a.content==='string') ? {type:'file',name:a.name,content:a.content,group:a.group,groupName:a.groupName} : {type:a.type,name:a.name}; })};
  });
}
function pushSnap(){
  if(!convId||takenOver) return;
  var slim=slimForServer(), last=slim.length?slim[slim.length-1].role:'';
  apiPost('/api/live/snap',{ui_id:UI_ID,conv_id:convId,title:chatTitle(),tid:lastTurnId,model:'1',n:slim.length,last_role:last,msgs_json:JSON.stringify(slim)})
    .then(function(d){ if(d&&d.mine===false) markTakenOver(); }).catch(function(){});
}
function markTakenOver(){
  if(takenOver) return; takenOver=true;
  var b=document.getElementById('taken-bar'); if(b) b.style.display='block';
}
setInterval(function(){
  if(!convId||takenOver) return;
  apiPost('/api/live/hb',{ui_id:UI_ID,conv_id:convId}).then(function(d){ if(d&&d.mine===false) markTakenOver(); }).catch(function(){});
}, 5000);
function openConnect(){
  var m=document.getElementById('conn-modal'); m.style.display='flex'; refreshConnect();
}
function closeConnect(){ document.getElementById('conn-modal').style.display='none'; }
function refreshConnect(){
  var box=document.getElementById('conn-list'); box.textContent='Looking...';
  apiPost('/api/live/list',{ui_id:UI_ID}).then(function(arr){
    box.innerHTML='';
    if(!Array.isArray(arr)||!arr.length){ var e=mk('div'); e.style.cssText='color:var(--tx2);font-size:13px;padding:6px 2px'; e.textContent='No chats are held by the server right now. A chat appears here after you send a message in it.'; box.appendChild(e); return; }
    arr.sort(function(a,b){return a.ago_s-b.ago_s;});
    arr.forEach(function(c){
      var row=mk('div','sess-entry'); row.style.cssText='display:flex;align-items:center;gap:8px;padding:8px 6px;border-bottom:1px solid var(--brd)';
      var info=mk('div'); info.style.cssText='flex:1;min-width:0';
      var nm=mk('div','sess-name'); nm.textContent=c.title||'(untitled)';
      var st=mk('div','sess-date');
      var state=c.running?('reply being written ('+c.tokens+' tokens so far)'):'reply finished';
      var who=c.mine?'open in this window':(c.connected?'open in another window':'not open anywhere');
      st.textContent=state+' - '+c.n+' messages - '+who+' - '+(c.ago_s<90?c.ago_s+' s':Math.round(c.ago_s/60)+' min')+' ago';
      info.appendChild(nm); info.appendChild(st);
      var b=mk('button','sess-load'); b.textContent=c.mine?'This window':(c.connected?'In use':'Connect'); b.disabled=!!(c.connected&&!c.mine)||!!c.mine; if(b.disabled) b.style.opacity='.45';
      if(c.connected&&!c.mine) b.title='Another window is connected to this chat';
      b.onclick=function(){ claimChat(c.conv_id); };
      row.appendChild(info); row.appendChild(b); box.appendChild(row);
    });
  }).catch(function(){ box.textContent='Could not reach the server.'; });
}
function claimChat(conv){
  if(generating){ toast('Wait for the current reply to finish first','err'); return; }
  apiPost('/api/live/claim',{ui_id:UI_ID,conv_id:conv}).then(function(d){
    if(!d||!d.ok){ toast((d&&d.error)||'Could not connect','err'); refreshConnect(); return; }
    var arr; try{ arr=JSON.parse(d.msgs_json||'[]'); }catch(e){ arr=[]; }
    msgs=arr.map(function(m){ return Object.assign({},m,{time:m.time?new Date(m.time):new Date()}); });
    convId=conv; loadedSessionName=''; takenOver=false; sessTokens=0; lastTurnId=d.tid||'';
    var tb=document.getElementById('taken-bar'); if(tb) tb.style.display='none';
    renderMsgs(); closeConnect();
    if(d.resume){
      var aiBub=buildStreamBub(); setBusy(true);
      doStream('/api/continue/resume',{from:'0',turn_id:d.tid,model:d.model||'1'},aiBub,Date.now());
      toast('Connected. Showing the reply from the start...');
    } else toast('Connected to "'+(d.title||'chat')+'".');
  });
}

/* --- Markdown renderer -------------------------------------------------------- */
function partialMD(text){
  var fences=(text.match(/^```/gm)||[]).length;
  return fences%2!==0 ? text+'\n```' : text;
}
function renderMD(s){
  // Strip Mistral tokenizer artifacts: <0x0A>, <unk>, <s>, </s>, -, \u200b
  s=s.replace(/<0x[0-9A-Fa-f]{2}>/g,'').replace(/<\/s>|<s>|<unk>/g,'');
  s=s.replace(/\ufffd/g,'').replace(/\u200b/g,'').replace(/\u200c/g,'').replace(/\u200d/g,'');
  // HTML-escape the three dangerous chars only - leave all other Unicode as-is.
  // Mistral Unicode symbols (- - - - -  etc.) render fine in UTF-8 HTML.
  s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Code blocks - pulled out into placeholders FIRST, before any other markdown
  // transform touches the string. Every later regex (bold/italic/tables/the
  // paragraph splitter) operates on the whole string with no awareness of tag
  // boundaries, so if the finished <pre><code> HTML were left inline: (a) a
  // literal '*' in code (e.g. "10 * 1020") gets consumed as italic markup and
  // vanishes, and (b) a blank line inside the code (any multi-line function
  // with normal spacing) gets split by the paragraph step and tears the
  // <pre><code> tag in half, corrupting the DOM enough that the copy button's
  // querySelector('code') grabs a truncated/wrong element. Stashing the
  // rendered block as an opaque placeholder keeps it immune to all of that;
  // it's spliced back in verbatim at the very end.
  var codeBlocks=[];
  s=s.replace(/```([\w+-]*)\n?([\s\S]*?)```/g,(_,lang,code)=>{
    var id='cb'+Math.random().toString(36).slice(2);
    var html='<div class="code-wrap"><div class="code-hdr"><span class="code-lang">'+(lang||'text')+'</span>'
      +'<button class="code-copy" id="'+id+'" onclick="cpCode(\''+id+'\',this)">Copy</button></div>'
      +'<pre><code>'+code.replace(/\n$/,'')+'</code></pre></div>';
    var idx=codeBlocks.length;
    codeBlocks.push(html);
    return '\x00CODEBLOCK'+idx+'\x00';
  });
  // Tables
  s=s.replace(/^\|(.+)\|\n\|[-| :]+\|\n((?:\|.+\|\n?)+)/gm,(_,hdr,rows)=>{
    var th=hdr.split('|').map(c=>'<th>'+c.trim()+'</th>').join('');
    var tr=rows.trim().split('\n').map(r=>'<tr>'+r.replace(/^\||\|$/g,'').split('|').map(c=>'<td>'+c.trim()+'</td>').join('')+'</tr>').join('');
    return '<table><thead><tr>'+th+'</tr></thead><tbody>'+tr+'</tbody></table>';
  });
  s=s.replace(/^&gt;\s?(.+)$/gm,'<blockquote>$1</blockquote>');
  s=s.replace(/^#{3}\s+(.+)$/gm,'<h3>$1</h3>');
  s=s.replace(/^#{2}\s+(.+)$/gm,'<h2>$1</h2>');
  s=s.replace(/^#\s+(.+)$/gm,'<h1>$1</h1>');
  s=s.replace(/^---+$/gm,'<hr>');
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*\*(.+?)\*\*\*/g,'<strong><em>$1</em></strong>');
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/\*([^\n*]+?)\*/g,'<em>$1</em>');
  s=s.replace(/~~(.+?)~~/g,'<del>$1</del>');
  s=s.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener">$1</a>');
  s=s.replace(/^[-*+]\s+(.+)$/gm,'<li>$1</li>');
  // Wrap consecutive <li> runs in <ul> - handles multiple list items correctly
  s=s.replace(/(<li>[\s\S]*?<\/li>)(\n<li>[\s\S]*?<\/li>)*/g, function(m){ return '<ul>'+m+'</ul>'; });
  var blocks=s.split(/\n\n+/);
  var out = blocks.map(b=>{
    b=b.trim();if(!b)return'';
    if(/^\x00CODEBLOCK\d+\x00$/.test(b))return b;
    if(/^<(h[1-6]|ul|ol|pre|hr|table|blockquote|div)/.test(b))return b;
    return '<p>'+b.replace(/\n/g,'<br>')+'</p>';
  }).filter(Boolean).join('\n');
  // Splice the untouched code blocks back in.
  out=out.replace(/\x00CODEBLOCK(\d+)\x00/g,(_,i)=>codeBlocks[+i]);
  return out;
}
function cpCode(id,btn){
  var code=document.getElementById(id).closest('.code-wrap').querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(()=>{
    btn.textContent='Copied!';btn.classList.add('ok');
    setTimeout(()=>{btn.textContent='Copy';btn.classList.remove('ok');},1500);
  });
}

/* --- Attachments -------------------------------------------------------------- */
function onPick(input){Array.from(input.files).forEach(addFile);input.value='';}
function addFile(file){
  if(file.type.startsWith('image/')){
    // Do NOT decode/canvas/base64 on drag/drop. That expanded a small compressed
    // image into massive raw RGB + base64 copies before Send was pressed.
    // Keep the browser File/Blob and a lightweight object URL for preview only.
    var url = URL.createObjectURL(file);
    atts.push({type:'image',name:file.name,file:file,objectUrl:url,size:file.size});
    renderAtts();
  } else if(DOC_RE.test(file.name)){
    attachDocument(file);                                            // PDF, Word, spreadsheet, slides, e-book, RTF, web page
  } else if(OLD_OFFICE_RE.test(file.name)){
    toast(file.name+': the old Word / PowerPoint format can\'t be read here. Save it as .docx / .pptx (or PDF) and attach that.','err');
  } else if(BIN_EXT.test(file.name)){
    toast(file.name+' is a binary file -- not something the model can read as text','err');
  } else {
    // anything else is tried as text, but never sent if it turns out to be binary (that used to send a PDF's raw
    // bytes: hundreds of thousands of tokens of noise)
    var r2=new FileReader();r2.onload=e=>{
      var t=e.target.result;
      if(looksBinary(t)){ toast(file.name+' is a binary file -- not something the model can read as text','err'); return; }
      atts.push({type:'file',name:file.name,content:t});renderAtts();
    };r2.readAsText(file);
  }
}
function releaseAttachment(a){
  try{ if(a && a.objectUrl) URL.revokeObjectURL(a.objectUrl); }catch(e){}
}
function removeAtt(i){
  var g=atts[i]&&atts[i].group;
  if(g){ atts=atts.filter(a=>a.group!==g); renderAtts(); return; }   // a folder chip removes the whole folder
  releaseAttachment(atts[i]);
  atts.splice(i,1);
  renderAtts();
}
function renderAtts(){
  var el=document.getElementById('att-row');el.innerHTML='';
  var shown={};
  atts.forEach((a,i)=>{
    if(a.group){
      if(shown[a.group]) return; shown[a.group]=1;
      var mine=atts.filter(b=>b.group===a.group), chars=mine.reduce((s,b)=>s+b.content.length,0);
      var t=mk('div','att-thumb');t.title=mine.map(b=>b.name).join('\n');
      var ic=mk('div','att-ico');ic.style.fontSize='11px';ic.textContent=mine.length;t.appendChild(ic);
      var nm=mk('div','att-nm');var est=Math.round(chars/3.5); nm.textContent=a.groupName+'/ ~'+(est<1000?est:Math.round(est/1000)+'k')+' tok';t.appendChild(nm);
      var x=mk('button','att-x');x.textContent='x';x.onclick=()=>removeAtt(i);t.appendChild(x);
      el.appendChild(t); return;
    }
    var t=mk('div','att-thumb');
    if(a.type==='image'){var ic=mk('div','att-ico');ic.textContent='IMG';t.appendChild(ic);}
    else{var ic=mk('div','att-ico');var dk=docKind(a.name);ic.style.fontSize=dk?'10px':'';
         ic.textContent=a.pending?'...':({pdf:'PDF',docx:'DOC',odf:'ODF',sheet:'XLS',pptx:'PPT',epub:'BOOK',rtf:'RTF',html:'WEB'}[dk]||'-');t.appendChild(ic);}
    if(a.pending){ t.classList.add('att-wait'); t.title=a.name+' -- '+(a.status||'reading'); }
    else if(a.type==='file'&&typeof a.content==='string') t.title=a.name+' -- about '+Math.round(a.content.length/3.5).toLocaleString()+' tokens';
    var nm=mk('div','att-nm');nm.textContent=a.pending?(a.status||'reading'):a.name;t.appendChild(nm);
    var x=mk('button','att-x');x.textContent='-';x.onclick=()=>removeAtt(i);t.appendChild(x);
    el.appendChild(t);
  });
}

/* --- Documents: PDF (+ OCR for scanned pages), Word, spreadsheets, slides, e-books, RTF, web pages ----------
   Everything is read here in the browser, with readers served offline from the server's web/vendor folder, and
   attached as plain text: "[Page 3]", "[Sheet: Budget]", "[Slide 2]", headings as "#", lists as "-", tables as
   "| a | b |". Images still go to the model itself (models that can see). */
var DOC_RE=/\.(pdf|docx|docm|odt|xlsx|xlsm|xlsb|xls|ods|pptx|pptm|odp|epub|rtf|html?|xhtml)$/i;
var OLD_OFFICE_RE=/\.(doc|ppt|pps|pot|dot)$/i;
var DOC_MAX=300*1024*1024;
var _nxLibs={};
function nxLib(src){
  if(_nxLibs[src]) return _nxLibs[src];
  return _nxLibs[src]=new Promise(function(res,rej){
    var s=document.createElement('script'); s.src=src; s.async=true;
    s.onload=function(){res();};
    s.onerror=function(){ delete _nxLibs[src]; rej(new Error('the document reader '+src.split('/').pop()+' is missing -- the web/vendor folder must sit next to the Nexus server')); };
    document.head.appendChild(s);
  });
}
function docKind(name){
  var m=(name.match(/\.([a-z0-9]+)$/i)||['',''])[1].toLowerCase();
  if(m==='pdf') return 'pdf';
  if(m==='docx'||m==='docm') return 'docx';
  if(m==='odt'||m==='odp') return 'odf';
  if(m==='xlsx'||m==='xlsm'||m==='xlsb'||m==='xls'||m==='ods') return 'sheet';
  if(m==='pptx'||m==='pptm') return 'pptx';
  if(m==='epub') return 'epub';
  if(m==='rtf') return 'rtf';
  if(m==='html'||m==='htm'||m==='xhtml') return 'html';
  return '';
}
function ocrOn(){ try{ return localStorage.getItem('nexus_ocr')!=='0'; }catch(e){ return true; } }
function ocrLang(){ try{ return (localStorage.getItem('nexus_ocr_lang')||'eng').replace(/[^a-z_+]/gi,'')||'eng'; }catch(e){ return 'eng'; } }
function tidyText(s){ return s.replace(/[ \t]+\n/g,'\n').replace(/\n{3,}/g,'\n\n').replace(/(\S)[ \t]{2,}/g,'$1 ').replace(/^\s*\n/,'').replace(/\s+$/,''); }

/* HTML (Word via mammoth, web pages, e-book chapters) -> readable text with markdown-ish structure */
function htmlToText(root){
  var out=[], listDepth=0;
  function inline(n){
    var s='';
    n.childNodes.forEach(function(c){
      if(c.nodeType===3) s+=c.nodeValue.replace(/\s+/g,' ');
      else if(c.nodeType===1){
        var t=c.tagName.toLowerCase();
        if(t==='br') s+='\n';
        else if(t==='script'||t==='style'||t==='noscript'||t==='template') {}
        else if(t==='img'){ var alt=c.getAttribute('alt'); if(alt) s+='[image: '+alt+']'; }
        else if(t==='a'){ var tx=inline(c).trim(), h=c.getAttribute('href')||''; s+= (h && /^https?:/i.test(h) && h!==tx) ? tx+' ('+h+')' : tx; }
        else s+=inline(c);
      }
    });
    return s;
  }
  function listLines(lst,depth,acc){
    var ord=lst.tagName.toLowerCase()==='ol', k=0;
    lst.childNodes.forEach(function(li){
      if(li.nodeType!==1||li.tagName.toLowerCase()!=='li') return; k++;
      var own=''; li.childNodes.forEach(function(x){ if(!(x.nodeType===1&&/^(ul|ol)$/i.test(x.tagName))) own+= x.nodeType===1?inline(x):(x.nodeValue||''); });
      acc.push('  '.repeat(depth)+(ord?k+'. ':'- ')+own.replace(/\s+/g,' ').trim());
      li.childNodes.forEach(function(x){ if(x.nodeType===1&&/^(ul|ol)$/i.test(x.tagName)) listLines(x,depth+1,acc); });
    });
  }
  function table(tb){
    var rows=[];
    tb.querySelectorAll('tr').forEach(function(tr){
      var cells=[]; tr.querySelectorAll('th,td').forEach(function(td){ cells.push(inline(td).replace(/\|/g,'/').replace(/\s+/g,' ').trim()); });
      if(cells.length) rows.push('| '+cells.join(' | ')+' |');
    });
    if(rows.length) out.push(rows.join('\n'));
  }
  function block(n){
    n.childNodes.forEach(function(c){
      if(c.nodeType===3){ var t=c.nodeValue.trim(); if(t) out.push(t.replace(/\s+/g,' ')); return; }
      if(c.nodeType!==1) return;
      var tg=c.tagName.toLowerCase();
      if(tg==='script'||tg==='style'||tg==='noscript'||tg==='template'||tg==='head') return;
      var hm=tg.match(/^h([1-6])$/);
      if(hm){ var ht=inline(c).trim(); if(ht) out.push('#'.repeat(+hm[1])+' '+ht); return; }
      if(tg==='p'||tg==='caption'||tg==='figcaption'||tg==='dt'||tg==='dd'){ var pt=inline(c).trim(); if(pt) out.push(pt); return; }
      if(tg==='pre'){ out.push('```\n'+c.textContent.replace(/\n+$/,'')+'\n```'); return; }
      if(tg==='blockquote'){ var q=inline(c).trim(); if(q) out.push(q.split('\n').map(function(l){return '> '+l;}).join('\n')); return; }
      if(tg==='table'){ table(c); return; }
      if(tg==='ul'||tg==='ol'){ var ll=[]; listLines(c,0,ll); if(ll.length) out.push(ll.join('\n')); return; }
      if(tg==='hr'){ out.push('---'); return; }
      block(c);
    });
  }
  block(root);
  var merged=[];
  out.forEach(function(b){
    var isList=/^\s*(- |\d+\. )/.test(b), prev=merged[merged.length-1];
    if(isList && prev!==undefined && /^\s*(- |\d+\. )/.test(prev.split('\n').pop())) merged[merged.length-1]=prev+'\n'+b; else merged.push(b);
  });
  return tidyText(merged.join('\n\n'));
}

/* PDF: the real text of every page; pages without text (scans) are rendered and read with OCR */
async function readPdf(file,onp){
  await nxLib('/vendor/pdfjs/pdf.min.js');
  var lib=window.pdfjsLib; lib.GlobalWorkerOptions.workerSrc='/vendor/pdfjs/pdf.worker.min.js';
  var pdf=await lib.getDocument({data:new Uint8Array(await file.arrayBuffer()),cMapUrl:'/vendor/pdfjs/cmaps/',cMapPacked:true,
                                 standardFontDataUrl:'/vendor/pdfjs/standard_fonts/',isEvalSupported:false}).promise;
  var n=pdf.numPages, pages=[], scans=[];
  for(var p=1;p<=n;p++){
    onp('reading page '+p+' of '+n);
    var page=await pdf.getPage(p), tc=await page.getTextContent(), s='', lastY=null, lastEnd=null;
    tc.items.forEach(function(it){
      if(typeof it.str!=='string') return;
      var x=it.transform[4], y=it.transform[5], h=Math.abs(it.transform[3])||Math.abs(it.transform[0])||10;
      if(lastY!==null && Math.abs(y-lastY)>h*0.6) s+='\n';
      else if(lastEnd!==null && x-lastEnd>h*0.15 && !/\s$/.test(s) && !/^\s/.test(it.str)) s+=' ';
      s+=it.str; lastY=y; lastEnd=x+(it.width||0);
      if(it.hasEOL){ s+='\n'; lastY=null; lastEnd=null; }
    });
    s=tidyText(s);
    if(s.replace(/\s/g,'').length<16) scans.push(p);
    pages[p]=s;
    page.cleanup();
  }
  var ocrDone=[], ocrNote='';
  if(scans.length && ocrOn()){
    await nxLib('/vendor/tesseract/tesseract.min.js');
    onp('starting OCR for '+scans.length+' page'+(scans.length===1?'':'s')+' without text');
    var w=await Tesseract.createWorker(ocrLang().split('+'),1,{workerPath:'/vendor/tesseract/worker.min.js',corePath:'/vendor/tesseract/core',
                                         langPath:'/vendor/tesseract/lang',gzip:true,workerBlobURL:false,cacheMethod:'none'});
    try{
      for(var i=0;i<scans.length;i++){
        var pn=scans[i]; onp('page '+pn+' of '+n+' has no text: OCR '+(i+1)+' of '+scans.length);
        var pg=await pdf.getPage(pn), v1=pg.getViewport({scale:1}), sc=Math.min(3,2400/Math.max(v1.width,v1.height)), vp=pg.getViewport({scale:sc});
        var cv=document.createElement('canvas'); cv.width=Math.ceil(vp.width); cv.height=Math.ceil(vp.height);
        var cx=cv.getContext('2d'); cx.fillStyle='#fff'; cx.fillRect(0,0,cv.width,cv.height);
        await pg.render({canvasContext:cx,viewport:vp}).promise;
        var r=await w.recognize(cv); var t=tidyText((r&&r.data&&r.data.text)||'');
        if(t){ pages[pn]=(pages[pn]?pages[pn]+'\n':'')+t; ocrDone.push(pn); }
        cv.width=cv.height=0; pg.cleanup();
      }
    } finally { await w.terminate(); }
  } else if(scans.length) ocrNote='; OCR is off in Settings, so '+scans.length+' page'+(scans.length===1?'':'s')+' without text '+(scans.length===1?'was':'were')+' left empty';
  await pdf.destroy();
  var head='(PDF, '+n+' page'+(n===1?'':'s')+(ocrDone.length?'; read with OCR: page'+(ocrDone.length===1?' ':'s ')+ocrDone.join(', '):'')+ocrNote+')';
  var body=[]; for(var q=1;q<=n;q++) body.push('[Page '+q+']'+(ocrDone.indexOf(q)>=0?' (OCR)':'')+'\n'+(pages[q]||'(no text on this page)'));
  return head+'\n\n'+body.join('\n\n');
}

async function readDocx(file,onp){
  await nxLib('/vendor/mammoth.browser.min.js');
  onp('reading');
  var r=await mammoth.convertToHtml({arrayBuffer:await file.arrayBuffer()});
  var d=new DOMParser().parseFromString('<body>'+r.value+'</body>','text/html');
  return '(Word document)\n\n'+htmlToText(d.body);
}

async function readSheet(file,onp){
  await nxLib('/vendor/xlsx.full.min.js');
  onp('reading');
  var wb=XLSX.read(new Uint8Array(await file.arrayBuffer()),{type:'array',cellDates:true,dense:true});
  var parts=[];
  wb.SheetNames.forEach(function(nm,i){
    onp('sheet '+(i+1)+' of '+wb.SheetNames.length);
    var csv=XLSX.utils.sheet_to_csv(wb.Sheets[nm],{blankrows:false,strip:true});
    parts.push('[Sheet: '+nm+']\n'+(csv.trim()?csv.trim():'(empty)'));
  });
  return '(Spreadsheet, '+wb.SheetNames.length+' sheet'+(wb.SheetNames.length===1?'':'s')+', each as comma-separated rows)\n\n'+parts.join('\n\n');
}

function xmlDoc(txt){ return new DOMParser().parseFromString(txt,'application/xml'); }
function byLocal(root,name){ var r=[]; var all=root.getElementsByTagName('*'); for(var i=0;i<all.length;i++) if(all[i].localName===name) r.push(all[i]); return r; }

async function readPptx(file,onp){
  await nxLib('/vendor/jszip.min.js');
  var zip=await JSZip.loadAsync(await file.arrayBuffer());
  var rels={}, relsX=zip.file('ppt/_rels/presentation.xml.rels');
  if(relsX) byLocal(xmlDoc(await relsX.async('string')),'Relationship').forEach(function(r){ rels[r.getAttribute('Id')]=r.getAttribute('Target'); });
  var order=[], pres=zip.file('ppt/presentation.xml');
  if(pres) byLocal(xmlDoc(await pres.async('string')),'sldId').forEach(function(s){
    var id=s.getAttribute('r:id')||s.getAttributeNS('http://schemas.openxmlformats.org/officeDocument/2006/relationships','id'); if(rels[id]) order.push('ppt/'+rels[id].replace(/^\/?ppt\//,'').replace(/^\.\.\//,'')); });
  if(!order.length){ zip.forEach(function(p){ if(/^ppt\/slides\/slide\d+\.xml$/.test(p)) order.push(p); });
                     order.sort(function(a,b){ return +a.match(/(\d+)\.xml$/)[1] - +b.match(/(\d+)\.xml$/)[1]; }); }
  function paras(x){ return byLocal(x,'p').map(function(p){ return byLocal(p,'t').map(function(t){return t.textContent;}).join(''); }).filter(function(l){return l.trim();}); }
  var out=[];
  for(var i=0;i<order.length;i++){
    onp('slide '+(i+1)+' of '+order.length);
    var f=zip.file(order[i]); if(!f) continue;
    var lines=paras(xmlDoc(await f.async('string'))), notes=[];
    var rf=zip.file(order[i].replace(/slides\/(slide\d+\.xml)$/,'slides/_rels/$1.rels'));
    if(rf){ var nt=byLocal(xmlDoc(await rf.async('string')),'Relationship').filter(function(r){return /notesSlide$/.test(r.getAttribute('Type')||'');})[0];
            if(nt){ var nf=zip.file('ppt/'+nt.getAttribute('Target').replace(/^\.\.\//,'')); if(nf) notes=paras(xmlDoc(await nf.async('string'))).filter(function(l){return !/^\d+$/.test(l.trim());}); } }
    out.push('[Slide '+(i+1)+']\n'+(lines.length?lines.join('\n'):'(no text)')+(notes.length?'\nNotes: '+notes.join(' '):''));
  }
  return '(Slides, '+order.length+')\n\n'+out.join('\n\n');
}

/* OpenDocument text (.odt) and slides (.odp); .ods goes through the spreadsheet reader */
async function readOdf(file,onp){
  await nxLib('/vendor/jszip.min.js');
  var zip=await JSZip.loadAsync(await file.arrayBuffer()), cf=zip.file('content.xml');
  if(!cf) throw new Error('no content.xml inside -- not an OpenDocument file');
  onp('reading');
  var x=xmlDoc(await cf.async('string')), out=[], slide=0;
  function inl(n){ var s=''; n.childNodes.forEach(function(c){
    if(c.nodeType===3) s+=c.nodeValue;
    else if(c.nodeType===1){ var l=c.localName; if(l==='s') s+=' '.repeat(+(c.getAttribute('text:c')||1)); else if(l==='tab') s+='\t'; else if(l==='line-break') s+='\n'; else if(l!=='note'&&l!=='annotation') s+=inl(c); } });
    return s; }
  function odfList(lst,depth,acc){ lst.childNodes.forEach(function(it){ if(it.nodeType!==1) return; var t3=[];
    it.childNodes.forEach(function(q){ if(q.nodeType===1&&(q.localName==='p'||q.localName==='h')) t3.push(inl(q).trim()); });
    if(t3.join('').trim()) acc.push('  '.repeat(depth)+'- '+t3.join(' '));
    it.childNodes.forEach(function(q){ if(q.nodeType===1&&q.localName==='list') odfList(q,depth+1,acc); }); }); }
  function walk(n,depth){ n.childNodes.forEach(function(c){
    if(c.nodeType!==1) return; var l=c.localName;
    if(l==='page'){ slide++; out.push('[Slide '+slide+']'); walk(c,depth); return; }
    if(l==='h'){ var lv=+(c.getAttribute('text:outline-level')||1); var t=inl(c).trim(); if(t) out.push('#'.repeat(Math.min(6,lv))+' '+t); return; }
    if(l==='p'){ var t2=inl(c).trim(); if(t2) out.push(t2); return; }
    if(l==='list'){ var ll=[]; odfList(c,depth,ll); if(ll.length){ var pv=out[out.length-1]; if(pv!==undefined && /^\s*- /.test(pv.split('\n').pop())) out[out.length-1]=pv+'\n'+ll.join('\n'); else out.push(ll.join('\n')); } return; }
    if(l==='table'){ var rows=[]; byLocal(c,'table-row').forEach(function(r){ var cells=byLocal(r,'table-cell').map(function(td){ return inl(td).replace(/\|/g,'/').replace(/\s+/g,' ').trim(); }); if(cells.join('').trim()) rows.push('| '+cells.join(' | ')+' |'); }); if(rows.length) out.push(rows.join('\n')); return; }
    if(l==='notes'||l==='annotation') return;
    walk(c,depth); }); }
  var body=byLocal(x,'body')[0]||x.documentElement; walk(body,0);
  return '('+(slide?'Slides, '+slide:'OpenDocument text')+')\n\n'+tidyText(out.join('\n\n').replace(/(\[Slide \d+\])\n\n/g,'$1\n'));
}

async function readEpub(file,onp){
  await nxLib('/vendor/jszip.min.js');
  var zip=await JSZip.loadAsync(await file.arrayBuffer());
  var cont=zip.file('META-INF/container.xml'); if(!cont) throw new Error('not an EPUB (no META-INF/container.xml)');
  var opfPath=byLocal(xmlDoc(await cont.async('string')),'rootfile')[0].getAttribute('full-path'), base=opfPath.replace(/[^\/]*$/,'');
  var opf=xmlDoc(await zip.file(opfPath).async('string')), man={};
  byLocal(opf,'item').forEach(function(it){ man[it.getAttribute('id')]=it.getAttribute('href'); });
  var title=(byLocal(opf,'title')[0]||{}).textContent||'', spine=byLocal(opf,'itemref').map(function(r){return man[r.getAttribute('idref')];}).filter(Boolean), out=[];
  for(var i=0;i<spine.length;i++){
    onp('chapter '+(i+1)+' of '+spine.length);
    var f=zip.file(decodeURIComponent(base+spine[i].split('#')[0])); if(!f) continue;
    var d=new DOMParser().parseFromString(await f.async('string'),'text/html'), t=htmlToText(d.body||d.documentElement);
    if(t) out.push(t);
  }
  return '(E-book'+(title?': '+title.trim():'')+', '+out.length+' parts)\n\n'+out.join('\n\n---\n\n');
}

/* RTF: text only (fonts, colours, pictures and other metadata left out) */
function rtfToText(r){
  var out='', i=0, stack=[], skip=false, uc=1, pendingSkip=0;
  var SKIPDEST={fonttbl:1,colortbl:1,stylesheet:1,info:1,pict:1,header:1,footer:1,headerl:1,headerr:1,footerl:1,footerr:1,listtable:1,listoverridetable:1,rsidtbl:1,generator:1,themedata:1,colorschememapping:1,datastore:1,latentstyles:1,xmlnstbl:1,object:1,fldinst:1};
  var cp=[8364,0,8218,402,8222,8230,8224,8225,710,8240,352,8249,338,0,381,0,0,8216,8217,8220,8221,8226,8211,8212,732,8482,353,8250,339,0,382,376];
  function put(ch){ if(pendingSkip>0){ pendingSkip--; return; } if(!skip) out+=ch; }
  while(i<r.length){
    var c=r[i];
    if(c==='{'){ stack.push([skip,uc]); i++; if(r.substr(i,2)==='\\*') skip=true; continue; }
    if(c==='}'){ var st=stack.pop()||[false,1]; skip=st[0]; uc=st[1]; i++; continue; }
    if(c==='\\'){
      var n2=r[i+1];
      if(n2==='\\'||n2==='{'||n2==='}'){ put(n2); i+=2; continue; }
      if(n2==="'"){ var code=parseInt(r.substr(i+2,2),16); put(String.fromCharCode(code>=128&&code<160&&cp[code-128]?cp[code-128]:code)); i+=4; continue; }
      if(n2==='~'){ put(' '); i+=2; continue; }
      if(n2==='-'||n2==='_'){ i+=2; continue; }
      var m=/^\\([a-z]+)(-?\d+)? ?/i.exec(r.substr(i,40));
      if(!m){ i+=2; continue; }
      var w=m[1], num=m[2]!==undefined?+m[2]:null; i+=m[0].length;
      if(SKIPDEST[w]) { skip=true; continue; }
      if(w==='par'||w==='line'||w==='sect'||w==='page') put('\n'); else if(w==='tab'||w==='cell') put('\t'); else if(w==='row') put('\n');
      else if(w==='uc') uc=num||0; else if(w==='u'){ var cc=num<0?num+65536:num; if(!skip) out+=String.fromCharCode(cc); pendingSkip=uc; }
      else if(w==='emdash') put('—'); else if(w==='endash') put('–'); else if(w==='bullet') put('•');
      else if(w==='lquote') put('‘'); else if(w==='rquote') put('’'); else if(w==='ldblquote') put('“'); else if(w==='rdblquote') put('”');
      continue;
    }
    if(c==='\r'||c==='\n'){ i++; continue; }
    put(c); i++;
  }
  return tidyText(out);
}

async function readDocument(file,onp){
  var k=docKind(file.name);
  if(k==='pdf') return readPdf(file,onp);
  if(k==='docx') return readDocx(file,onp);
  if(k==='odf') return readOdf(file,onp);
  if(k==='sheet') return readSheet(file,onp);
  if(k==='pptx') return readPptx(file,onp);
  if(k==='epub') return readEpub(file,onp);
  if(k==='rtf'){ onp('reading'); return '(RTF document)\n\n'+rtfToText(await file.text()); }
  if(k==='html'){ onp('reading'); var d=new DOMParser().parseFromString(await file.text(),'text/html'), ti=(d.title||'').trim();
                  return '(Web page'+(ti?': '+ti:'')+')\n\n'+htmlToText(d.body||d.documentElement); }
  throw new Error('not a document type this chat can read');
}

/* A chip shows "reading page 3 of 40" while a document is read; Send waits until every chip is ready. */
function attachDocument(file){
  if(file.size>DOC_MAX){ toast(file.name+' is over '+(DOC_MAX>>20)+' MB -- too big to read here','err'); return; }
  var a={type:'file',name:file.name,content:'',pending:true,status:'waiting'}; atts.push(a); renderAtts();
  readDocument(file,function(msg){ a.status=msg; renderAtts(); }).then(function(txt){
    if(atts.indexOf(a)<0) return;                                  // removed while reading
    a.content=txt; a.pending=false; delete a.status; renderAtts();
    toast('Read '+file.name+': about '+Math.round(txt.length/3.5).toLocaleString()+' tokens');
  }).catch(function(e){
    var i=atts.indexOf(a); if(i>=0){ atts.splice(i,1); renderAtts(); }
    toast('Could not read '+file.name+': '+(e&&e.message?e.message:e),'err');
  });
}
function looksBinary(t){
  var head=t.slice(0,16384); if(head.indexOf('\u0000')>=0) return true;
  var bad=(head.match(/�/g)||[]).length; return head.length>200 && bad/head.length>0.05;
}
function attachPending(){ return atts.filter(function(a){return a.pending;}); }
setTimeout(function(){ try{ var o=document.getElementById('sd-ocr'); if(o) o.value=ocrOn()?'1':'0';
  var l=document.getElementById('sd-ocrlang'); if(l) l.value=localStorage.getItem('nexus_ocr_lang')||''; }catch(e){} },0);

/* --- Folders ------------------------------------------------------------------- */
// Drop a folder (or pick one with DIR) and every text file in it is attached under its path inside the folder,
// e.g. "myproj/src/app.py", with a list of all the paths first -- so the model knows the layout and can tell
// two files with the same name apart. Build output, dependencies, binaries and lock files are left out.
var DIR_SKIP={'.git':1,'.hg':1,'.svn':1,'node_modules':1,'bower_components':1,'__pycache__':1,'.venv':1,'venv':1,'.env':1,
  '.mypy_cache':1,'.pytest_cache':1,'.ruff_cache':1,'.tox':1,'.cache':1,'dist':1,'build':1,'out':1,'target':1,'.next':1,'.nuxt':1,
  '.idea':1,'.vs':1,'.vscode':1,'.gradle':1,'obj':1,'coverage':1,'.terraform':1,'vendor':1,'Pods':1,'DerivedData':1};
var FILE_SKIP={'package-lock.json':1,'yarn.lock':1,'pnpm-lock.yaml':1,'Cargo.lock':1,'poetry.lock':1,'composer.lock':1,'Gemfile.lock':1,'.DS_Store':1,'Thumbs.db':1};
var BIN_EXT=/\.(png|jpe?g|gif|bmp|ico|webp|tiff?|psd|svgz|mp[34]|m4a|wav|flac|ogg|avi|mov|mkv|webm|zip|gz|tgz|bz2|xz|7z|rar|zst|tar|jar|war|exe|dll|so|dylib|a|lib|o|obj|pdb|class|pyc|pyo|wasm|bin|dat|db|sqlite3?|gguf|safetensors|pt|pth|onnx|npy|npz|pkl|h5|ckpt|pdf|docx?|xlsx?|pptx?|ttf|otf|woff2?|eot|iso|dmg|apk|ipa)$/i;
var DIR_MAX_FILE=2*1024*1024;
function dirReadAll(d){ return new Promise(function(res){ var r=d.createReader(),all=[];
  (function more(){ r.readEntries(function(b){ if(!b.length) return res(all); all=all.concat(Array.from(b)); more(); }, function(){ res(all); }); })(); }); }
function dirEntryFile(fe){ return new Promise(function(res){ fe.file(res, function(){ res(null); }); }); }
async function dirWalk(ent, path, out, sk){
  if(ent.isFile){ var f=await dirEntryFile(ent); if(f) out.push({file:f,path:path}); return; }
  if(!ent.isDirectory) return;
  if(DIR_SKIP[ent.name] && path.indexOf('/')>=0){ sk.dirs.push(path+'/'); return; }
  var kids=await dirReadAll(ent); kids.sort(function(a,b){ return a.name<b.name?-1:a.name>b.name?1:0; });
  for(var k of kids) await dirWalk(k, path+'/'+k.name, out, sk);
}
function dirNewSkips(){ return {dirs:[],bin:0,big:[],lock:0}; }
async function addFolder(root, list, sk){
  var gid='d'+Date.now().toString(36)+Math.random().toString(36).slice(2,6), add=[], chars=0;
  var ndocs=list.filter(function(it){ return DOC_RE.test(it.path); }).length, di=0, chip=null, failed=[];
  if(ndocs){ chip={type:'file',name:root+'/',content:'',pending:true,status:'reading documents'}; atts.push(chip); renderAtts(); }
  for(var it of list){
    var f=it.file, p=it.path, base=p.split('/').pop();
    if(FILE_SKIP[base]){ sk.lock++; continue; }
    if(DOC_RE.test(base)){
      di++;
      if(f.size>DOC_MAX){ sk.big.push(p); continue; }
      try{
        var dt=await readDocument(f,function(m){ chip.status='document '+di+' of '+ndocs+': '+base+' -- '+m; renderAtts(); });
        add.push({type:'file',name:p,content:dt,group:gid,groupName:root}); chars+=dt.length;
      }catch(e){ failed.push(base); }
      continue;
    }
    if(BIN_EXT.test(base)){ sk.bin++; continue; }
    if(f.size>DIR_MAX_FILE){ sk.big.push(p); continue; }
    var txt; try{ txt=await f.text(); }catch(e){ sk.bin++; continue; }
    if(txt.slice(0,8192).indexOf('\u0000')>=0){ sk.bin++; continue; }      // binary after all
    add.push({type:'file',name:p,content:txt,group:gid,groupName:root}); chars+=txt.length;
  }
  var skipTxt=[]; if(sk.dirs.length) skipTxt.push(sk.dirs.length+' folder(s) like '+sk.dirs.slice(0,3).join(', '));
  if(sk.bin) skipTxt.push(sk.bin+' binary file(s)'); if(sk.lock) skipTxt.push(sk.lock+' lock file(s)');
  if(sk.big.length) skipTxt.push(sk.big.length+' file(s) over 2 MB');
  if(chip){ var ci=atts.indexOf(chip); if(ci>=0) atts.splice(ci,1); else { renderAtts(); return; } renderAtts(); }   // chip removed = cancelled
  if(failed.length) skipTxt.push(failed.length+' document(s) that could not be read ('+failed.slice(0,3).join(', ')+')');
  if(!add.length){ toast('No text files found in '+root+'/'+(skipTxt.length?' (left out: '+skipTxt.join(', ')+')':''),'err'); return; }
  var est=Math.round(chars/3.5);
  if(add.length>500 && !confirm('Attach all '+add.length+' files from '+root+'/?')) return;
  atts=atts.concat(add); renderAtts();
  toast('Attached '+root+'/: '+add.length+' file'+(add.length===1?'':'s')+', about '+est.toLocaleString()+' tokens'+(skipTxt.length?'. Left out: '+skipTxt.join(', '):''));
}
function onPickDir(input){
  var files=Array.from(input.files); input.value='';
  var byRoot={}, skipped={};
  files.forEach(function(f){
    var p=f.webkitRelativePath||f.name, segs=p.split('/');
    for(var i=1;i<segs.length-1;i++) if(DIR_SKIP[segs[i]]){                   // inside a left-out folder
      var d=segs.slice(0,i+1).join('/')+'/'; (skipped[segs[0]]=skipped[segs[0]]||{})[d]=1; return; }
    (byRoot[segs[0]]=byRoot[segs[0]]||[]).push({file:f,path:p});
  });
  Object.keys(byRoot).forEach(function(r){
    byRoot[r].sort(function(a,b){ return a.path<b.path?-1:a.path>b.path?1:0; });
    var sk=dirNewSkips(); sk.dirs=Object.keys(skipped[r]||{}).sort();
    addFolder(r, byRoot[r], sk);
  });
}
async function onDropItems(dt){
  // the entries must be taken before the first await: the browser empties the drop data after this event
  var ents=Array.from(dt.items||[]).map(function(it){ return it.webkitGetAsEntry?it.webkitGetAsEntry():null; });
  var plain=Array.from(dt.files||[]);
  if(!ents.some(function(e){ return e&&e.isDirectory; })){ plain.forEach(addFile); return; }
  for(var i=0;i<ents.length;i++){
    var e=ents[i]; if(!e) continue;
    if(e.isDirectory){ var list=[], sk=dirNewSkips(); await dirWalk(e, e.name, list, sk); await addFolder(e.name, list, sk); }
    else if(plain[i]) addFile(plain[i]);
    else { var f=await dirEntryFile(e); if(f) addFile(f); }
  }
}

/* --- Drag & drop -------------------------------------------------------------- */
var _dc=0;
document.addEventListener('dragenter',e=>{if(![].slice.call(e.dataTransfer.types).includes('Files'))return;_dc++;document.getElementById('drop-ov').classList.add('show');e.preventDefault();});
document.addEventListener('dragleave',()=>{_dc=Math.max(0,_dc-1);if(!_dc)document.getElementById('drop-ov').classList.remove('show');});
document.addEventListener('dragover',e=>e.preventDefault());
document.addEventListener('drop',e=>{e.preventDefault();_dc=0;document.getElementById('drop-ov').classList.remove('show');onDropItems(e.dataTransfer);});

/* --- Tabs & sidebar ----------------------------------------------------------- */
document.querySelectorAll('.stab').forEach(t=>t.addEventListener('click',()=>{
  document.querySelectorAll('.stab').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  ['sessions','settings','memory'].forEach(n=>document.getElementById('tab-'+n).style.display=n===t.dataset.tab?'':'none');
  if(t.dataset.tab==='sessions'){renderSessList();loadCkList();loadWarmList();}
  if(t.dataset.tab==='memory'){kvSync();kvLoadArchive();}
}));
function toggleSB(){document.getElementById('sidebar').style.display=document.getElementById('sidebar').style.display==='none'?'':'none';}

/* --- Scroll ------------------------------------------------------------------- */
function scrollBot(){var m=document.getElementById('msgs');m.scrollTop=m.scrollHeight;}
function scrollIfNear(){var m=document.getElementById('msgs');if(m.scrollHeight-m.scrollTop-m.clientHeight<120)m.scrollTop=m.scrollHeight;}

/* --- Misc --------------------------------------------------------------------- */
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function autoH(ta){ta.style.height='';ta.style.height=Math.min(ta.scrollHeight,160)+'px';}
function fmtTime(d){if(!d)return'';var t=new Date(d);return t.getHours().toString().padStart(2,'0')+':'+t.getMinutes().toString().padStart(2,'0');}
/* --- Conversation memory: copy, forget, archive, restore -------------------------------- */
// The server splits what the model remembers into turns using the model's own chat template;
// kvSync() lines those up with the chat bubbles from the END (the newest turn is always the
// newest bubble), so each bubble knows which tokens are its own.
var kvState = null;
function kvFmt(n){ return (n||0).toLocaleString(); }
function kvCopyText(t, btn){
  function ok(){ if(btn){ var o=btn.textContent; btn.textContent='Copied'; btn.classList.add('ok'); setTimeout(function(){ btn.textContent=o; btn.classList.remove('ok'); },1300); } }
  if(navigator.clipboard && window.isSecureContext){ navigator.clipboard.writeText(t).then(ok, function(){ kvCopyFallback(t)&&ok(); }); }
  else if(kvCopyFallback(t)) ok(); else toast('Copy failed - select the text by hand','err');
}
function kvCopyFallback(t){     // plain http on another computer: no clipboard API, use a hidden text box
  var ta=document.createElement('textarea'); ta.value=t; ta.style.cssText='position:fixed;left:-9999px;top:0';
  document.body.appendChild(ta); ta.select(); var r=false; try{ r=document.execCommand('copy'); }catch(e){} ta.remove(); return r;
}
function kvMsgMarkdown(m){
  var c=m.content||'';
  if(m.role==='assistant') c=c.replace(/<think>[\s\S]*?<\/think>\s*/g,'');   // the answer, not the thinking
  return c.trim()+'\n';
}
function kvAttachTools(wrap,bub,m){
  if(!wrap||!bub||m.role==='note') return;
  m._el=wrap;
  var old=bub.querySelector('.msg-tools'); if(old) old.remove();
  var bar=mk('div','msg-tools');
  var cp=mk('button','mt-btn'); cp.textContent='Copy'; cp.title='Copy this message as Markdown (code blocks included)';
  cp.onclick=function(e){ e.stopPropagation(); kvCopyText(kvMsgMarkdown(m),cp); };
  var more=mk('button','mt-btn'); more.textContent='⋯'; more.title='Memory: forget, archive or restore this message';
  more.onclick=function(e){ e.stopPropagation(); kvMenu(m,more); };
  var tag=mk('span','kv-tag'); tag.style.display='none';
  bar.appendChild(cp); bar.appendChild(more); bar.appendChild(tag);
  bub.appendChild(bar);
  m._tag=tag;
  kvDecorate(m);
}
function kvDecorate(m){
  if(!m._el) return;
  var out = m.kv==='removed' || m.kv==='archived' || m.kv==='trimmed';
  m._el.classList.toggle('kv-out', out);
  m._el.classList.toggle('sel', !!m._sel);
  var t=m._tag; if(!t) return;
  t.className='kv-tag'+(m.kv==='archived'?' arch':'');
  var txt='';
  if(m.kv==='removed') txt='forgotten';
  else if(m.kv==='trimmed') txt='dropped (memory full)';
  else if(m.kv==='archived') txt='archived '+(m.arch||'');
  else if(m.restoredFrom) txt='restored '+m.restoredFrom;
  else if(m._kv && m._kv.protected) txt='kept';
  t.textContent=txt; t.style.display=txt?'':'none';
  t.title = m._kv ? ('tokens '+kvFmt(m._kv.start)+' - '+kvFmt(m._kv.end)) : '';
}
function kvSync(cb, announce){
  if(generating){ if(cb) cb(null); return; }
  fetch('/api/kv/turns?model=1').then(function(r){ return r.json(); }).then(function(d){
    if(!d.ok){ if(d.busy) setTimeout(function(){ kvSync(cb,announce); },1500); else if(cb) cb(null); return; }
    kvApply(d, announce); if(cb) cb(d);
  }).catch(function(){ if(cb) cb(null); });
}
function kvApply(d, announce){
  kvState=d;
  showCtx(d.context);
  if((d.turns||[]).length===1 && d.turns[0].role==='turn'){   // a chat format without turn markers: single messages can't be picked out
    msgs.forEach(function(m){ delete m._kv; kvDecorate(m); }); kvRenderTab(); return;
  }
  var T=[];
  (d.turns||[]).forEach(function(t){
    if(t.role==='banner'||t.role==='system'||t.role==='tool') return;
    if(t.role==='exchange'){ T.push({role:'user',t:t}); T.push({role:'assistant',t:t}); }
    else T.push({role:(t.role==='turn'?'?':t.role),t:t});
  });
  var U=msgs.filter(function(m){ return (m.role==='user'||m.role==='assistant') && (!m.kv||m.kv==='in'); });
  var i=U.length-1, j=T.length-1, mismatch=false;
  while(i>=0 && j>=0){
    if(T[j].role!=='?' && T[j].role!==U[i].role){ mismatch=true; break; }
    U[i]._kv=T[j].t; U[i].kv='in'; i--; j--;
  }
  var dropped=0;
  for(; i>=0; i--){
    delete U[i]._kv;
    if(!mismatch && j<0 && U[i].kv==='in'){ U[i].kv='trimmed'; dropped++; }
  }
  if(mismatch) console.warn('[memory] chat bubbles and the model memory do not line up; older bubbles have no memory controls');
  msgs.forEach(kvDecorate);
  if(dropped && announce) toast('Memory was full: the oldest '+dropped+' message'+(dropped>1?'s were':' was')+' dropped (the system prompt is kept).');
  if(dropped) pushSnap();
  kvRenderTab();
}
function kvRenderTab(){
  var d=kvState; if(!d) return;
  var u=document.getElementById('mem-used'); if(!u) return;
  var unl=d.limit>=1e9, ref=unl?d.model_max:d.limit;
  u.textContent=kvFmt(d.context)+(unl?' tokens (no limit)':' / '+kvFmt(d.limit)+' tokens');
  document.getElementById('mem-fill').style.width=Math.min(100, ref? d.context*100/ref : 0)+'%';
  var mb=(d.context*d.bytes_per_token/1048576);
  var note=kvFmt(Math.round(mb))+' MB in memory';
  if(unl) note+=' (the bar is the model maximum, '+kvFmt(d.model_max)+(d.context>d.model_max?': now past it':'')+')';
  else if(d.limit!==d.model_max) note+=' (model maximum '+kvFmt(d.model_max)+')';
  note+='. ';
  if(!d.exact) note+='This model also keeps a running summary in some of its layers; forgetting or restoring is exact everywhere else, but that summary still holds a trace of what was removed.';
  document.getElementById('mem-note').textContent=note;
  var mode=document.getElementById('mem-limit-mode'), tok=document.getElementById('mem-limit-tok');
  if(document.activeElement!==mode && document.activeElement!==tok){
    mode.value = d.limit_cfg<0 ? 'auto' : (d.limit_cfg>0 ? 'n' : 'max');
    tok.style.display = mode.value==='n' ? '' : 'none';
    if(d.limit_cfg>0) tok.value=d.limit_cfg;
    document.getElementById('mem-autotrim').checked=!!d.auto_trim;
  }
}
function kvSaveSettings(){
  var mode=document.getElementById('mem-limit-mode').value;
  var lim = mode==='n' ? String(Math.max(1024, parseInt(document.getElementById('mem-limit-tok').value,10)||0)) : mode;
  apiPost('/api/kv/settings',{model:'1',limit:lim,auto_trim:document.getElementById('mem-autotrim').checked?'on':'off'}).then(function(d){
    if(!d.ok){ toast(d.error||'Could not change the limit','err'); return; }
    kvApply(d,false); toast('Memory limit: '+kvFmt(d.limit)+' tokens');
  });
}
function kvCloseMenu(){ var m=document.getElementById('kv-menu'); if(m) m.remove(); }
document.addEventListener('click', kvCloseMenu);
function kvGroup(m){   // messages that share this message's tokens (question+answer stored as one block in [INST]-style formats)
  if(!m._kv) return [m];
  return msgs.filter(function(x){ return x._kv && x._kv.start===m._kv.start && x.kv==='in'; });
}
function kvMenu(m, anchor){
  kvCloseMenu();
  var menu=mk('div','kv-menu'); menu.id='kv-menu';
  menu.addEventListener('click',function(e){ e.stopPropagation(); });
  function item(label, fn){ var it=mk('div','it'); it.textContent=label; it.onclick=function(){ kvCloseMenu(); fn(); }; menu.appendChild(it); }
  function hint(t){ var h=mk('div','hint'); h.textContent=t; menu.appendChild(h); }
  function sep(){ menu.appendChild(mk('div','sep')); }
  item('Copy as Markdown', function(){ kvCopyText(kvMsgMarkdown(m)); });
  sep();
  if(generating) hint('The model is writing a reply. Memory can be edited once it has finished (or been stopped).');
  else if(m.kv==='archived'){
    item('Restore into memory ('+(m.arch||'?')+')', function(){ kvRestore(m.arch); });
    hint('Puts it back at the end of the conversation, so the model remembers it again.');
  } else if(m.kv==='removed') hint('Forgotten: the model no longer sees this message. (Only archived messages can be restored.)');
  else if(m.kv==='trimmed') hint('Dropped when memory was full: the model no longer sees this message.');
  else if(!m._kv){
    hint('Not matched to the model memory yet.');
    item('Refresh', function(){ kvSync(null,true); });
  } else if(m._kv.protected) hint('Part of the protected start of the conversation (it holds the system prompt), so it is always kept.');
  else {
    var g=kvGroup(m);
    item('Forget (remove from memory)', function(){ kvForget(g); });
    item('Archive (save it, then forget it)', function(){ kvArchive(g); });
    item(m._sel?'Unselect':'Select (to forget or archive several)', function(){ kvToggleSel(m); });
    if(g.length>1) hint('This model stores the question and its answer together, so both are affected.');
    hint((m._kv.end-m._kv.start).toLocaleString()+' tokens');
  }
  document.body.appendChild(menu);
  var r=anchor.getBoundingClientRect(), w=menu.offsetWidth, h=menu.offsetHeight;
  menu.style.left=Math.max(6, Math.min(window.innerWidth-w-6, r.left))+'px';
  menu.style.top=(r.bottom+h+6>window.innerHeight ? Math.max(6, r.top-h-4) : r.bottom+4)+'px';
}
function kvToggleSel(m){
  kvGroup(m).forEach(function(x){ x._sel=!m._sel; kvDecorate(x); });
  kvSelBar();
}
function kvSelected(){ return msgs.filter(function(m){ return m._sel && m.kv==='in' && m._kv; }); }
function kvSelBar(){
  var n=kvSelected().length, b=document.getElementById('kv-selbar');
  b.style.display=n?'flex':'none';
  document.getElementById('kv-selcount').textContent=n+' selected';
}
function kvSelClear(){ msgs.forEach(function(m){ if(m._sel){ m._sel=false; kvDecorate(m); } }); kvSelBar(); }
function kvSelAction(what){ var s=kvSelected(); if(!s.length) return; if(what==='forget') kvForget(s); else kvArchive(s); }
function kvRanges(list){   // distinct token ranges, oldest first
  var seen={}, out=[];
  list.forEach(function(m){ if(m._kv && !seen[m._kv.start]){ seen[m._kv.start]=1; out.push({start:m._kv.start,end:m._kv.end}); } });
  return out.sort(function(a,b){ return a.start-b.start; });
}
function kvReady(){
  if(generating){ toast('Wait for the reply to finish (or stop it) first','err'); return false; }
  if(!kvState){ toast('Memory not loaded yet - try again in a moment','err'); kvSync(); return false; }
  return true;
}
function kvForget(list){
  if(!kvReady()) return;
  var rs=kvRanges(list).reverse();   // newest first: earlier positions stay valid while later parts go
  var ctx=kvState.context, last=null;
  function next(k){
    if(k>=rs.length){ if(last) kvApply(last,false); kvSelClear(); pushSnap(); toast('Forgotten'); return; }
    apiPost('/api/kv/remove',{model:'1',start:rs[k].start,end:rs[k].end,ctx:ctx}).then(function(d){
      if(!d.ok){ toast(d.error||'Could not forget that','err'); if(last) kvApply(last,false); else kvSync(); return; }
      list.forEach(function(m){ if(m._kv && m._kv.start===rs[k].start){ m.kv='removed'; m._sel=false; delete m._kv; } });
      last=d; ctx=d.context; next(k+1);
    });
  }
  next(0);
}
function kvArchive(list){
  if(!kvReady()) return;
  var rs=kvRanges(list);
  for(var k=1;k<rs.length;k++) if(rs[k].start!==rs[k-1].end){ toast('To archive several messages together they must be next to each other in the chat','err'); return; }
  var first=list.filter(function(m){ return m.content; })[0] || list[0];
  var def=(first.content||'').replace(/\s+/g,' ').trim().slice(0,50);
  var label=window.prompt('Name for this saved memory (for example "app.py rev 1"):', def);
  if(label===null) return;
  var payload=list.map(function(m){ return {role:m.role, content:m.content, time:m.time}; });
  apiPost('/api/kv/archive',{model:'1',start:rs[0].start,end:rs[rs.length-1].end,ctx:kvState.context,label:label,msgs:JSON.stringify(payload)}).then(function(d){
    if(!d.ok){ toast(d.error||'Could not archive that','err'); kvSync(); return; }
    var id=d.item&&d.item.id;
    list.forEach(function(m){ m.kv='archived'; m.arch=id; m._sel=false; delete m._kv; });
    kvApply(d,false); kvSelClear(); pushSnap(); kvLoadArchive();
    toast('Archived as '+id+' and removed from memory');
  });
}
function kvRestore(id){
  if(!kvReady()) return;
  if(!convId){ toast('Start the chat first (send a message), then restore a saved memory into it','err'); return; }
  var before=kvState.context;
  apiPost('/api/kv/archive/restore',{model:'1',id:id,ctx:kvState.context}).then(function(d){
    if(!d.ok){ toast(d.error||'Could not restore that','err'); kvSync(); return; }
    var it=d.item||{}, saved=[];
    try{ saved=JSON.parse(it.msgs||'[]'); }catch(e){ saved=[]; }
    var add=[{role:'note',content:'Restored saved memory '+id+(it.label?' - '+it.label:'')+' (the model remembers it again)',time:new Date(),atts:[]}];
    if(saved.length) saved.forEach(function(x){ add.push({role:x.role,content:x.content,time:new Date(),atts:[],restoredFrom:id}); });
    else (d.turns||[]).forEach(function(t){   // no chat text kept with it: show what the model got back
      if(t.start<before) return;
      if(t.role==='exchange'){ add.push({role:'user',content:t.preview,time:new Date(),atts:[],restoredFrom:id}); add.push({role:'assistant',content:'(restored)',time:new Date(),atts:[],restoredFrom:id}); }
      else if(t.role==='user'||t.role==='assistant') add.push({role:t.role,content:t.preview,time:new Date(),atts:[],restoredFrom:id});
    });
    var el=document.getElementById('msgs');
    add.forEach(function(m){ msgs.push(m); el.appendChild(bubbleFor(m)); });
    kvApply(d,false); pushSnap(); scrollBot();
    toast('Restored '+id+' ('+kvFmt(d.added)+' tokens)');
    if(d.warning) toast(d.warning,'err');
  });
}
function kvLoadArchive(){
  fetch('/api/kv/archive/list').then(function(r){ return r.json(); }).then(function(d){
    var box=document.getElementById('mem-arch'); if(!box) return;
    box.innerHTML='';
    var curFile=d.current_file||'';
    var items=(d.items||[]).sort(function(a,b){ return (parseInt((b.id||'').slice(1),10)||0)-(parseInt((a.id||'').slice(1),10)||0); });
    if(!items.length){ var e=mk('div','shint'); e.textContent='Nothing saved yet.'; box.appendChild(e); return; }
    items.forEach(function(it){
      var row=mk('div','mem-item');
      var t=mk('div','t'); t.textContent=it.id+'  '+(it.label||''); row.appendChild(t);
      var dd=mk('div','d'); dd.textContent=kvFmt(it.tokens)+' tokens, '+(+it.mb||0).toFixed(1)+' MB - '+(it.model||'?')+' - '+String(it.created||'').replace('T',' ');
      dd.title=it.preview||''; row.appendChild(dd);
      if(it.file && curFile && it.file!==curFile){
        var wf=mk('div','d'); wf.style.color='var(--red)'; wf.textContent='Saved with a different model file: '+it.file;
        wf.title='Restoring still works if it is the same model (for example another quantization), but it may be remembered less accurately.'; row.appendChild(wf);
      }
      function b(lbl,fn){ var x=mk('button','mt-btn'); x.textContent=lbl; x.onclick=fn; row.appendChild(x); }
      b('Restore',function(){ kvRestore(it.id); });
      b('Rename',function(){ var l=window.prompt('New name for '+it.id+':', it.label||''); if(l===null) return;
        apiPost('/api/kv/archive/rename',{id:it.id,label:l}).then(function(){ kvLoadArchive(); }); });
      b('Copy text',function(){ var a=[]; try{ a=JSON.parse(it.msgs||'[]'); }catch(e){}
        kvCopyText(a.map(function(m){ return '## '+(m.role==='user'?'You':'Assistant')+'\n\n'+kvMsgMarkdown(m); }).join('\n') || (it.preview||'')); });
      b('Delete',function(){ if(!confirm('Delete saved memory '+it.id+' ('+(it.label||'')+')?')) return;
        apiPost('/api/kv/archive/delete',{id:it.id}).then(function(){ kvLoadArchive(); }); });
      box.appendChild(row);
    });
  }).catch(function(){});
}

function mk(tag,cls){var e=document.createElement(tag);if(cls)e.className=cls;return e;}
var connRetry = null;   // runs the pending retry immediately (used by the Retry now button)
function hideConnBar(){ connRetry=null; var b=document.getElementById('connbar'); if(b) b.remove(); }
function showConnBar(){
  if(document.getElementById('connbar')) return;
  var b=document.createElement('div'); b.id='connbar';
  b.style.cssText='position:fixed;left:50%;bottom:110px;transform:translateX(-50%);z-index:9999;background:#3a1d1d;color:#fff;border:1px solid #e05555;border-radius:8px;padding:8px 14px;font-size:13px;display:flex;gap:12px;align-items:center;box-shadow:0 4px 16px rgba(0,0,0,.4);max-width:92vw';
  var t=document.createElement('span'); t.textContent='Connection lost - the server is still working. Retrying automatically.';
  var btn=document.createElement('button'); btn.textContent='Retry now';
  btn.style.cssText='cursor:pointer;padding:4px 10px;border-radius:6px;border:1px solid #e05555;background:transparent;color:#fff;white-space:nowrap';
  btn.onclick=function(){ if(connRetry) connRetry(); };
  b.appendChild(t); b.appendChild(btn); document.body.appendChild(b);
}
// Run fn after `wait` ms, or right away if the user presses Retry now.
function scheduleRetry(fn, wait){
  var timer = null;
  function run(){ if(timer){ clearTimeout(timer); timer=null; } if(connRetry===run) connRetry=null; fn(); }
  connRetry = run;
  showConnBar();
  timer = setTimeout(run, wait);
}
function toast(msg,type){var t=mk('div','toast '+(type||'ok'));t.textContent=msg;document.getElementById('toasts').appendChild(t);setTimeout(()=>t.remove(),Math.min(10000,3000+String(msg).length*40));}
function apiPost(path,body){return fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json()).catch(e=>({ok:false,error:e.message}));}
</script>
</body>
</html>
)HTMLPAGE";

// --- Server state -------------------------------------------------------------

// One reply buffer per model. The generation thread appends every SSE event here, so a
// browser whose connection dropped can re-attach with "give me everything after event N"
// and the running (or just-finished) reply is never lost. RAM only, cleared by the next reply.
struct GenJob {
    std::mutex m; std::condition_variable cv;
    std::vector<std::string> evs;
    bool running=false, done=true;
    std::string tid;            // turn id of the reply held here
    std::atomic<long long> last_attach_ms{0};   // last time a browser was attached/replaying (steady clock, ms)
};
static long long nx_now_ms(){ return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const char* NX_SSE_DONE = "data: [DONE]\n\n";
template<class FD>
static void nx_job_replay(FD fd, GenJob& J, size_t from, const std::string& want_tid=std::string()){
    const std::string hdr="HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    +nx_acao()+"X-Accel-Buffering: no\r\nConnection: close\r\n\r\n";
    if(::send(fd,hdr.c_str(),hdr.size(),MSG_NOSIGNAL)<=0) return;
    {   // Asked for a specific turn that this buffer does not hold (its /api/continue never arrived):
        // tell the page so it can start that reply instead of waiting for one that does not exist.
        std::lock_guard<std::mutex> l(J.m);
        if(!want_tid.empty() && J.tid!=want_tid){
            const char* nj="data: {\"nojob\":true}\n\n";
            ::send(fd,nj,strlen(nj),MSG_NOSIGNAL);
            return;
        }
    }
    size_t i=from;
    for(;;){
        J.last_attach_ms = nx_now_ms();
        std::vector<std::string> batch; bool fin=false, got=false;
        {
            std::unique_lock<std::mutex> l(J.m);
            got=J.cv.wait_for(l,std::chrono::seconds(10),[&]{return J.evs.size()>i||J.done;});
            for(size_t k=i;k<J.evs.size();++k) batch.push_back(J.evs[k]);
            fin=J.done;
        }
        i+=batch.size();
        std::string out;
        for(auto& e:batch) out+=e;
        if(!got && batch.empty()) out=": ping\n\n";               // keep the link alive while waiting
        if(fin && (batch.empty() || batch.back()!=NX_SSE_DONE)) out+=NX_SSE_DONE;   // stale/no job: just say done
        size_t sent=0;
        while(sent<out.size()){
            ssize_t n=::send(fd,out.c_str()+sent,out.size()-sent,MSG_NOSIGNAL);
            if(n<=0) return;
            sent+=(size_t)n;
        }
        if(fin) return;
    }
}

// Chats a browser has told the server about, so a closed/crashed page can reconnect ("Connect to session").  One window owns a chat at a time.
struct LiveChat { std::string conv,title,tid,owner,last_role;   // the chat text itself lives in a small temp file (live_file), not in RAM
    int model=1, n=0; long long upd=0, owner_ms=0; };
static const long long NX_OWNER_FRESH_MS = 15000;
static std::string live_safe(const std::string& c){ std::string o; for(char ch:c) if(isalnum((unsigned char)ch)||ch=='_'||ch=='-') o+=ch; return o.substr(0,80); }      // an owner that has not sent a heartbeat for this long is considered gone

struct ServerState {
    GenJob cjob[2];
    std::mutex live_mu; std::map<std::string,LiveChat> live;
    // Last extend_turn, so a retried request (same turn_id) returns the stored result
    // instead of adding the prompt to the KV cache a second time.
    std::mutex turn_m; std::condition_variable turn_cv;
    std::string turn_id; bool turn_done=true; std::string turn_reply;
    nexus::Engine engine;
    nexus::Engine engine2;          // optional secondary model (GPU-pinned)
    std::unique_ptr<nexus::dist::DistSession> dist_sess;  // null = local-only mode
    Storage       storage;
    std::mutex    gen_mutex;
    std::mutex    gen_mutex2;
    std::atomic<bool> stop_flag{false};
    std::atomic<bool> stop_flag2{false};
    bool          has_engine2 = false;

    // -- Pausing sessions -----------------------------------------------------
    struct ActiveSession {
        nexus::GenerateSession  session;
        std::condition_variable cv;
        std::mutex              mu;
        std::string             inject_text;
        bool                    inject_ready = false;
        bool                    aborted      = false;
        explicit ActiveSession(nexus::GenerateSession s) : session(std::move(s)) {}
    };
    std::mutex                                          sessions_mu;
    std::unordered_map<std::string, ActiveSession*>    sessions;

    struct ConvState { std::string last_prompt; int n_tokens = 0; int model = 1; };
    // Conversation length limit (both models): 0 = the model's own maximum, -1 = what fits in free
    // memory (worked out once at startup), N = N tokens. auto_trim: at the limit drop the oldest
    // turns (the system prompt is never dropped) instead of stopping.
    int  ctx_limit_cfg = 0;
    bool auto_trim     = true;
    std::string model_file[2];       // the model files loaded (model 1, model 2): saved memories record which one they came from
    bool web_tools = false;          // /api/search* (fetch web pages for a model): off unless --enable-web-tools
    // Who may use the API (see nx_access_check):
    //   cors_mode 0 "lan" (default): web pages from this computer and the local network (private addresses,
    //     plain computer names, *.local) -- not internet websites; 1 "local": this computer only; 2 "any": every website.
    //   allow_origins: extra web addresses allowed (--allow-origin), also accepted as host names.
    //   api_key: when set, requests from OTHER computers must send it (this computer never needs it).
    int cors_mode = 0;
    std::vector<std::string> allow_origins;
    std::string api_key;
    std::unordered_map<std::string, ConvState>         conversations;
    std::string                                         warm_conv_id;   // conv_id registered on --start-state load

    int port=8080;
    std::string host="127.0.0.1";
    std::unordered_map<std::string,nexus::ChatFormat> tmpl_map{
        {"ChatML",          nexus::ChatFormat::CHATML},
        {"Qwen3",           nexus::ChatFormat::QWEN3},
        {"Mistral",         nexus::ChatFormat::MISTRAL},
        {"Mistral V3",      nexus::ChatFormat::MISTRAL_V3},
        {"Llama-3",         nexus::ChatFormat::LLAMA3},
        {"DeepSeek",         nexus::ChatFormat::DEEPSEEK},
        {"DeepSeek-R1",      nexus::ChatFormat::DEEPSEEK_R1},
        {"DeepSeek-Coder",   nexus::ChatFormat::DEEPSEEK_CODER},
        {"Phi-3",            nexus::ChatFormat::PHI3},
        {"Phi-4",            nexus::ChatFormat::PHI4},
        {"Gemma",            nexus::ChatFormat::GEMMA},
        {"Zephyr",           nexus::ChatFormat::ZEPHYR},
        {"Alpaca",           nexus::ChatFormat::ALPACA},
        {"Granite",          nexus::ChatFormat::GRANITE},
        {"OLMoE",            nexus::ChatFormat::OLMOE},
        {"Custom",           nexus::ChatFormat::CUSTOM},
        {"Raw",              nexus::ChatFormat::RAW},
    };
    // Active generation settings loaded at startup from nexus.json
    float   temperature        = 0.7f;
    float   top_p              = 0.9f;
    float   min_p              = 0.05f;
    float   tfs_z              = 1.0f;
    float   repeat_penalty     = 1.1f;
    int     repeat_last_n      = 128;
    float   frequency_penalty  = 0.05f;
    float   presence_penalty   = 0.0f;
    int     top_k              = 40;
    int     max_tokens         = 32000;
    int     n_threads      = 1;
    int     n_gpu_layers   = 0;   // 0 = CPU only; >0 = offload first N layers to GPU
    int     engine2_n_gpu_layers = 0;
    bool    mtp_enabled    = false;   // multi-token prediction (off by default)
    int     mtp_n_predict  = 1;       // tokens MTP proposes per step
    bool    mtp_gpu        = false;    // run MTP layer on GPU if available
    std::string roster_file = "nexus-roster.json";
};

// --- Dispatcher ---------------------------------------------------------------

// A conversation is only worth holding while the engine still has its KV in memory.  When the engine's context is replaced or reset, everything we
// kept for the old conversation (prompt text, chat snapshot file, finished-reply buffer) is released.
static std::string live_file(ServerState& S,const std::string& conv){ return S.storage.base+"/live/"+live_safe(conv)+".json"; }
// ─── Who may use the API ──────────────────────────────────────────────────────────────────────
// Programs (Python, curl, the hardener, another Nexus) don't send Origin and are never limited by this --
// only by the API key, when one is set and they're on another computer. Web pages are, because a page
// from ANY website you visit can make your browser send requests to 127.0.0.1 or your local network:
//   - its Origin must be this server itself, this computer, the local network (lan mode), or --allow-origin;
//   - the Host it addressed must be an IP / plain computer name / *.local (or an --allow-origin host), which
//     stops "DNS rebinding" (a website pointing its own name at your computer to look same-origin);
//   - a cross-site request with no Origin (an <img> or link) is refused unless its Referer is allowed.
static std::string nx_lc(std::string s){ for(auto& c:s) c=(char)std::tolower((unsigned char)c); return s; }
static std::string nx_host_of(const std::string& origin_or_host){      // "http://h:1234/x" or "h:1234" -> "h"
    std::string h=origin_or_host; size_t p=h.find("://"); if(p!=std::string::npos) h=h.substr(p+3);
    size_t sl=h.find('/'); if(sl!=std::string::npos) h=h.substr(0,sl);
    if(!h.empty()&&h[0]=='['){ size_t e=h.find(']'); return nx_lc(e==std::string::npos?h:h.substr(0,e+1)); }
    size_t c=h.find(':'); if(c!=std::string::npos) h=h.substr(0,c);
    return nx_lc(h);
}
static bool nx_ipv4(const std::string& h, int o[4]){
    int n=0; unsigned a=0; int dots=0; bool dig=false;
    for(char c:h){ if(c>='0'&&c<='9'){ a=a*10+(c-'0'); if(a>255) return false; dig=true; }
                   else if(c=='.'){ if(!dig||dots>=3) return false; o[dots++]=(int)a; a=0; dig=false; } else return false; }
    if(!dig||dots!=3) return false; o[3]=(int)a; (void)n; return true;
}
static bool nx_host_local(const std::string& h){
    int o[4];
    if(h=="localhost"||h=="[::1]") return true;
    if(nx_ipv4(h,o)) return o[0]==127;
    return false;
}
static bool nx_host_lan(const std::string& h){
    int o[4];
    if(nx_host_local(h)) return true;
    if(nx_ipv4(h,o)) return o[0]==10 || (o[0]==172&&o[1]>=16&&o[1]<=31) || (o[0]==192&&o[1]==168) ||
                            (o[0]==169&&o[1]==254) || (o[0]==100&&o[1]>=64&&o[1]<=127);   // private, link-local, CGNAT/Tailscale
    if(!h.empty()&&h[0]=='[') return h.compare(0,3,"[fd")==0 || h.compare(0,3,"[fc")==0 || h.compare(0,5,"[fe80")==0;
    if(h.empty()) return false;
    if(h.find('.')==std::string::npos) return true;                                      // plain computer name ("desktop")
    auto ends=[&](const char* x){ size_t n=strlen(x); return h.size()>n && h.compare(h.size()-n,n,x)==0; };
    return ends(".local") || ends(".lan") || ends(".home.arpa") || ends(".internal");
}
static bool nx_listed(const ServerState& S, const std::string& origin){
    const std::string o=nx_lc(origin), h=nx_host_of(origin);
    for(const auto& a:S.allow_origins){ const std::string al=nx_lc(a);
        if(al==o || al==h || al=="http://"+h || al=="https://"+h) return true; }   // exact origin, or a bare host name (any port)
    return false;
}
static bool nx_origin_ok(const ServerState& S, const std::string& origin){
    if(S.cors_mode==2 || nx_listed(S,origin)) return true;
    const std::string h=nx_host_of(origin);
    return S.cors_mode==1 ? nx_host_local(h) : nx_host_lan(h);
}
static bool nx_peer_local(const std::string& peer){ int o[4]; return peer.empty() || (nx_ipv4(peer,o) && o[0]==127); }
static bool nx_key_eq(const std::string& a, const std::string& b){       // constant time
    unsigned char d = (unsigned char)(a.size()!=b.size()); size_t n=std::max(a.size(),b.size());
    for(size_t i=0;i<n;++i) d |= (unsigned char)((i<a.size()?a[i]:0) ^ (i<b.size()?b[i]:0));
    return d==0;
}
// Returns 0 = go on, otherwise the HTTP status to refuse with (why = message). Sets tl_acao.
static int nx_access_check(const ServerState& S, const Req& r, const std::string& peer, std::string& why){
    tl_acao.clear();
    auto h=[&](const char* k){ auto it=r.headers.find(k); return it==r.headers.end()?std::string():it->second; };
    const std::string origin=h("origin"), host=h("host"), sfs=h("sec-fetch-site");
    const bool from_browser = !origin.empty() || !sfs.empty();
    const bool is_api = r.path.compare(0,5,"/api/")==0;
    if(from_browser && S.cors_mode!=2){
        const std::string hh=nx_host_of(host);
        int o4[4];
        const bool ip_literal = nx_ipv4(hh,o4) || (!hh.empty() && hh[0]=='[');          // rebinding needs a NAME; any IP is fine
        if(!hh.empty() && !ip_literal && !nx_host_lan(hh) && !nx_listed(S,"http://"+host) && !nx_listed(S,"https://"+host)){
            why="this server was addressed as \""+hh+"\", which is not an address on this computer or network "
                "(add it with --allow-origin if that name is really yours)";
            return 403;
        }
    }
    if(!origin.empty()){
        const bool same = nx_lc(origin)==nx_lc("http://"+host) || nx_lc(origin)==nx_lc("https://"+host);
        if(!same && !nx_origin_ok(S,origin)){
            why="web pages from "+origin+" may not use this server (allow it with --allow-origin "+origin+", or --cors any)";
            return 403;
        }
        tl_acao = S.cors_mode==2 && !same ? "Access-Control-Allow-Origin: *\r\n" : "Access-Control-Allow-Origin: "+origin+"\r\nVary: Origin\r\n";
    } else if(is_api && (sfs=="cross-site"||sfs=="same-site") && S.cors_mode!=2){
        const std::string ref=h("referer");
        if(ref.empty() || !nx_origin_ok(S,ref)){ why="refused: a request another website made your browser send"; return 403; }
    }
    if(r.method=="OPTIONS") return 0;                           // preflights never carry the key
    if(is_api && !S.api_key.empty() && !nx_peer_local(peer)){
        std::string k=h("x-api-key");
        const std::string au=h("authorization");
        if(k.empty() && au.size()>7 && nx_lc(au.substr(0,7))=="bearer ") k=au.substr(7);
        if(k.empty()){ size_t q=r.path.find("key="); if(q!=std::string::npos && (r.path[q-1]=='?'||r.path[q-1]=='&')){
            size_t e=r.path.find('&',q); k=r.path.substr(q+4,e==std::string::npos?std::string::npos:e-q-4); } }
        if(!nx_key_eq(k,S.api_key)){ why="this server needs its API key from other computers (set in the launcher on the computer running it)"; return 401; }
    }
    return 0;
}

// ─── Web fetch for the search endpoints: curl run DIRECTLY (no shell), http/https only ──────────────
// The URL is one argument to curl, never part of a shell command line, so nothing in it ($(), `...`,
// ;, quotes) can run a command. --proto/--proto-redir stop file://, ftp://, etc., including on redirects.
static bool nx_url_ok(const std::string& u){
    if(u.size() > 4096) return false;
    if(u.compare(0,7,"http://") != 0 && u.compare(0,8,"https://") != 0) return false;
    for(unsigned char c : u)
        if(c <= 0x20 || c == 0x7f || c=='"' || c=='\'' || c=='`' || c=='\\' || c=='<' || c=='>' || c=='^' || c=='|' || c=='{' || c=='}') return false;
    return true;
}
static std::string nx_fetch_url(const std::string& url, int max_time_s){
    std::string out;
    if(!nx_url_ok(url)) return out;
    const std::string mt = std::to_string(max_time_s);
    const char* argv[] = {"curl", "-s", "-L", "--max-time", mt.c_str(), "--max-filesize", "20000000",
                          "--proto", "=http,https", "--proto-redir", "=http,https",
                          "-A", "Mozilla/5.0", "--url", url.c_str(), nullptr};
#ifdef _WIN32
    // CreateProcess runs curl.exe itself (no cmd.exe). nx_url_ok already refused quotes and spaces, so every
    // argument can be wrapped in plain double quotes.
    std::string cl;
    for(int i = 0; argv[i]; ++i){ if(i) cl += ' '; cl += '"'; cl += argv[i]; cl += '"'; }
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if(!CreatePipe(&rd, &wr, &sa, 0)) return out;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr; si.hStdError = nullptr; si.hStdInput = nullptr;
    PROCESS_INFORMATION pi{};
    std::vector<char> clbuf(cl.begin(), cl.end()); clbuf.push_back('\0');
    if(!CreateProcessA(nullptr, clbuf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)){
        CloseHandle(rd); CloseHandle(wr); return out;
    }
    CloseHandle(wr);
    char buf[8192]; DWORD n = 0;
    while(ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n > 0){ out.append(buf, n); if(out.size() > 24u*1024*1024) break; }
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, (DWORD)(max_time_s + 5) * 1000);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
#else
    int pfd[2];
    if(pipe(pfd) != 0) return out;
    pid_t pid = fork();
    if(pid < 0){ ::close(pfd[0]); ::close(pfd[1]); return out; }
    if(pid == 0){
        dup2(pfd[1], 1);
        int dn = open("/dev/null", O_WRONLY); if(dn >= 0){ dup2(dn, 2); dup2(dn, 0); }
        ::close(pfd[0]); ::close(pfd[1]);
        execvp("curl", (char* const*)argv);
        _exit(127);
    }
    ::close(pfd[1]);
    char buf[8192]; ssize_t n;
    while((n = ::read(pfd[0], buf, sizeof(buf))) > 0){ out.append(buf, (size_t)n); if(out.size() > 24u*1024*1024) break; }
    ::close(pfd[0]);
    int status = 0; waitpid(pid, &status, 0);
#endif
    return out;
}
// The web endpoints are off unless --enable-web-tools, and even then only answer this server's own
// pages or a program -- never a request another website's page makes the browser send (an <img> or a
// fetch pointed at 127.0.0.1): browsers mark those with Sec-Fetch-Site: cross-site / an Origin header.
static bool nx_web_tools_allowed(ServerState& S, const Req& r, std::string& why){
    if(!S.web_tools){ why = "web tools are off on this server (start it with --enable-web-tools to use them)"; return false; }
    auto h = [&](const char* k){ auto it = r.headers.find(k); return it == r.headers.end() ? std::string() : it->second; };
    const std::string sfs = h("sec-fetch-site");
    if(!sfs.empty() && sfs != "same-origin" && sfs != "none"){ why = "refused: request came from another website"; return false; }
    const std::string org = h("origin"), host = h("host");
    if(!org.empty() && !host.empty()){
        const size_t p = org.find("://");
        if(p == std::string::npos || org.substr(p + 3) != host){ why = "refused: request came from another website"; return false; }
    }
    return true;
}

// ─── Conversation memory (KV) editing helpers ──────────────────────────────────────────
// Turn the configured limit into tokens for one engine: a set number (used as is, even above the model's
// maximum), the model's maximum ("max"), or no limit at all ("auto": until memory runs out).
static void nx_apply_ctx_limit(ServerState& S, nexus::Engine& e, const char* tag){
    e.set_auto_trim(S.auto_trim);
    int lim = 0;
    // "What fits in memory" (auto) has NO limit, not even the model's own maximum: the conversation grows until
    // memory actually runs out -- or, only if the memory safety buffer is on, until that buffer stops it. Past the
    // model's maximum it will likely start repeating itself or fall apart; that's yours to watch. (It used to be a
    // guess made once at startup, which cut people off far below what their machine could hold.)
    if (S.ctx_limit_cfg > 0) lim = S.ctx_limit_cfg;
    else if (S.ctx_limit_cfg < 0) lim = -1;
    e.set_context_limit(lim);
    if (S.ctx_limit_cfg < 0)
        fprintf(stderr, "[ctx] %s: no conversation limit -- grows until memory runs out, past the model's maximum of %d if it gets there (%.1f KB of memory per token; memory safety buffer %s)\n",
                tag, e.max_context(), e.kv_bytes_per_token() / 1024.0, nexus::memguard::enabled() ? "ON: it stops a reply before memory runs out" : "off");
    else
        fprintf(stderr, "[ctx] %s: conversation limit %d tokens (model maximum %d, %.1f KB of memory per token), when full: %s\n",
                tag, e.context_limit(), e.max_context(), e.kv_bytes_per_token() / 1024.0,
                S.auto_trim ? "drop the oldest turns (system prompt kept)" : "stop");
    fflush(stderr);
}
// Before adding text to the conversation: drop the oldest turns if it would not fit.
static int nx_make_room(nexus::Engine& e, const std::string& text, int n_images, int max_tokens){
    int incoming = 0;
    try { incoming = (int)e.tokenize(text, false).size(); } catch (...) {}
    incoming += n_images * 576;                                   // rough size of one image's tokens
    const int reserve = std::max(64, std::min(max_tokens > 0 ? max_tokens : 1024, e.context_limit() / 16));
    return e.context_make_room(incoming, reserve);
}
// The conversation limit, said plainly: a reply that ends because memory is full must never look like
// the model chose to stop.
static std::string nx_limit_desc(nexus::Engine& e){
    char b[160]; const int lim = e.context_limit(), mx = e.max_context();
    if (lim == mx) snprintf(b, sizeof b, "%d tokens (the model's maximum)", lim);
    else snprintf(b, sizeof b, "%d tokens (the model's maximum is %d)", lim, mx);
    return b;
}
static const char* NX_RAISE_LIMIT =
    "To keep going past it, tick auto-trim or raise the limit (web chat Memory tab, or the launcher's Conversation memory card).";
// After a reply: if it ended because the conversation hit its limit with nothing older to drop, the notice to show.
static std::string nx_full_notice(nexus::Engine& e){
    if (e.context_size() < e.context_limit() - 1) return "";
    const std::string lim = nx_limit_desc(e);
    fprintf(stderr, "[ctx] reply stopped: the conversation reached its limit of %s and nothing older could be dropped\n", lim.c_str());
    fflush(stderr);
    return "Memory full: the conversation reached its limit of " + lim +
           " and nothing older could be dropped, so the reply stopped. " + NX_RAISE_LIMIT;
}
static std::string nx_jstr(const std::string& t){            // JSON string body for these messages
    std::string o; for (char c : t) { if (c == '"' || c == '\\') { o += '\\'; o += c; } else if ((unsigned char)c >= 0x20) o += c; else o += ' '; }
    return o;
}
// Protect the system prompt at the start of a fresh conversation, whatever the template does
// with it (some put it in its own turn, Llama-2/Mistral/Gemma fold it into the first user turn).
static void nx_pin_banner(nexus::Engine& e, const std::vector<nexus::ChatMessage>& messages){
    bool has_sys = false;
    for (const auto& m : messages) if (m.role == "system" && !m.content.empty()) { has_sys = true; break; }
    if (!has_sys) return;
    try {
        const std::string pre = e.format_system_prefix(messages);
        if (!pre.empty()) e.context_pin_tokens((int)e.tokenize(pre, true).size());
    } catch (...) {}
}
// Prompt-reading progress line every 2 s while a prompt is read on this computer: percentage
// (layers done plus how far through the current layer), tokens per second and time left. Steady
// lines (not one per finished layer) so a long layer never looks like a stall -- the keep-alive's
// "still working" line only appears when nothing at all is reported for 14 s.
// sink (optional): also gets each report as JSON, {"pct":34.2,"tokens":98304,"tps":104.2,"eta":580}
// (tps 0 = still measuring, eta -1 = unknown) -- /api/chat streams it to the client as a progress event.
// The latest report per engine, for GET /api/progress (clients whose request doesn't stream, like
// /api/session/extend_turn, poll it while they wait). Read-only: nothing here touches the prompt.
static std::mutex g_prog_mu;
static std::map<const nexus::Engine*, std::string> g_prog;      // engine -> {"active":..,"pct":..,...}
static void nx_prog_set(const nexus::Engine* e, const std::string& js){ std::lock_guard<std::mutex> l(g_prog_mu); g_prog[e] = js; }
static std::string nx_prog_get(const nexus::Engine* e){
    std::lock_guard<std::mutex> l(g_prog_mu); auto it = g_prog.find(e);
    return it == g_prog.end() ? std::string("{\"active\":false}") : it->second;
}
struct NxPrefillReporter {
    nexus::Engine& e; size_t ntok; std::atomic<bool> stop{false}; std::thread th; bool on = false;
    std::function<void(const std::string&)> sink;
    NxPrefillReporter(nexus::Engine& eng, size_t n, std::function<void(const std::string&)> sk = nullptr) : e(eng), ntok(n), sink(std::move(sk)) {
        if (n < 256) return;
        on = true; e.prefill_progress_begin(true);
        nx_prog_set(&e, "{\"active\":true,\"pct\":0,\"tokens\":" + std::to_string(n) + ",\"tps\":0,\"eta\":-1}");
        th = std::thread([this]{
            const auto t0 = std::chrono::steady_clock::now();
            auto tlast = t0 - std::chrono::seconds(10);
            double f0 = -1.0; std::chrono::steady_clock::time_point tf0;   // first report: speed is measured from there
            double last_tps = 0;
            while (!stop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                if (stop.load()) break;
                int mv_done = 0, mv_total = 0;
                if (e.park_moving(mv_done, mv_total)) {             // prompt read; parked layers going back onto the card
                    const auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration<double>(now - tlast).count() < 2.0) continue;
                    tlast = now;
                    char js[200];
                    snprintf(js, sizeof js, "{\"pct\":100,\"tokens\":%zu,\"tps\":%.1f,\"eta\":-1,\"moving_done\":%d,\"moving_total\":%d}",
                             ntok, last_tps, mv_done, mv_total);
                    nx_prog_set(&e, std::string("{\"active\":true,") + (js + 1));
                    if (sink) sink(js);
                    nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
                    continue;
                }
                const double f = e.prefill_fraction();
                if (f < 0) continue;
                const auto now = std::chrono::steady_clock::now();
                if (f0 < 0) { f0 = f; tf0 = now; }
                if (std::chrono::duration<double>(now - tlast).count() < 2.0) continue;
                tlast = now;
                const double el = std::chrono::duration<double>(now - tf0).count();
                const double tps = el > 0.5 && f > f0 ? (double)ntok * (f - f0) / el : 0.0;
                if (tps > 0) last_tps = tps;
                char eta[48] = "";
                long long rem = -1;
                if (tps > 0 && f < 1.0) {
                    rem = (long long)((double)ntok * (1.0 - f) / tps);
                    if (rem >= 3600) snprintf(eta, sizeof eta, ", about %lldh %lldm left", rem / 3600, (rem % 3600) / 60);
                    else if (rem >= 60) snprintf(eta, sizeof eta, ", about %lldm %llds left", rem / 60, rem % 60);
                    else snprintf(eta, sizeof eta, ", about %llds left", rem);
                }
                char sp[32] = "measuring speed";
                if (tps > 0) snprintf(sp, sizeof sp, "%.1f tok/s", tps);
                fprintf(stderr, "[progress] reading prompt: %.0f%%  (%zu tokens, %s%s)\n", f * 100.0, ntok, sp, eta);
                fflush(stderr);
                char js[160];
                snprintf(js, sizeof js, "{\"pct\":%.1f,\"tokens\":%zu,\"tps\":%.1f,\"eta\":%lld}", f * 100.0, ntok, tps, rem);
                nx_prog_set(&e, std::string("{\"active\":true,") + (js + 1));
                if (sink) sink(js);
                nexus::memguard::progress_stamp().store(nexus::memguard::now_ms());
            }
        });
    }
    ~NxPrefillReporter() { stop = true; if (th.joinable()) th.join(); if (on) { e.prefill_progress_begin(false); nx_prog_set(&e, "{\"active\":false}"); } }
};

static std::string nx_kv_dir(ServerState& S){ return S.storage.base + "/kv_archive"; }
// "name.gguf (12345678 bytes)" -- the file name plus its size tells two quantizations of one model apart
static std::string nx_model_file_tag(const std::string& path){
    if(path.empty()) return "";
    size_t sl = path.find_last_of("/\\");
    std::string base = sl == std::string::npos ? path : path.substr(sl + 1);
    struct stat st; long long sz = (stat(path.c_str(), &st) == 0) ? (long long)st.st_size : -1;
    return sz >= 0 ? base + " (" + std::to_string(sz) + " bytes)" : base;
}
static std::string nx_kv_safe_id(const std::string& id){ std::string o; for(char c:id) if(isalnum((unsigned char)c)) o+=c; return o.substr(0,24); }

static void nx_forget_conversations(ServerState& S,int model,const std::string& keep){
    std::vector<std::string> gone_tids;
    { std::lock_guard<std::mutex> cl(S.sessions_mu);
      for(auto it=S.conversations.begin();it!=S.conversations.end();){ if(it->second.model==model&&it->first!=keep) it=S.conversations.erase(it); else ++it; } }
    { std::lock_guard<std::mutex> l(S.live_mu);
      for(auto it=S.live.begin();it!=S.live.end();){
          if(it->second.model==model&&it->first!=keep){ remove(live_file(S,it->first).c_str()); gone_tids.push_back(it->second.tid); it=S.live.erase(it); } else ++it; } }
    GenJob& J=S.cjob[model==2?1:0];
    { std::lock_guard<std::mutex> jl(J.m);
      if(!J.running){ for(auto& t:gone_tids) if(!t.empty()&&J.tid==t){ J.evs.clear(); J.evs.shrink_to_fit(); J.tid.clear(); break; } } }
}

#ifdef _WIN32
static void handle(SOCKET fd, ServerState& S, const std::string& peer){
#else
static void handle(int fd, ServerState& S, const std::string& peer){
#endif
    auto r=parse_req(fd);
    if(!r.valid){
        // Log only non-empty connections (browsers open speculative connections)
        close(fd);return;
    }
    {   // who may use the API: web pages from this computer / the local network, programs, the API key
        std::string why; const int code = nx_access_check(S, r, peer, why);
        if(code){
            static std::atomic<int> nlog{0};
            if(nlog.fetch_add(1) < 50){ fprintf(stderr,"[access] refused %s %s from %s: %s\n", r.method.c_str(), r.path.substr(0,80).c_str(), peer.c_str(), why.c_str()); fflush(stderr); }
            resp_json(fd, "{\"ok\":false,\"error\":"+jstr(why)+(code==401?",\"need_key\":true":"")+"}", code);
            close(fd); return;
        }
    }
    // Only log actionable requests - suppress high-frequency polling paths
    // that would spam the log and scroll it continuously.
    {
        bool noisy = (r.path == "/api/live/hb"            ||
                      r.path.rfind("/api/kv/turns",0) == 0 ||
                      r.path.rfind("/api/progress",0) == 0 ||
                      r.path == "/api/kv/archive/list"    ||
                      r.path == "/api/boot"               ||
                      r.path == "/api/live/list"          ||
                      r.path == "/api/live/snap"          ||
                      r.path == "/api/folders"            ||
                      r.path == "/api/roster"             ||
                      r.path == "/api/info"              ||
                      r.path == "/api/config"             ||
                      r.path == "/api/sessions/list"      ||
                      r.path == "/api/checkpoint/list"    ||
                      r.path == "/api/start_state/list"   ||
                      r.path == "/favicon.ico"            ||
                      (r.method == "GET" && r.path == "/") ||
                      (r.method == "GET" && r.path == "/index.html"));
        if (!noisy) {
            char _ts[16]; { time_t _t=time(nullptr); struct tm _tm;
#ifdef _WIN32
                localtime_s(&_tm,&_t);
#else
                localtime_r(&_t,&_tm);
#endif
                strftime(_ts,sizeof(_ts),"%H:%M:%S",&_tm); }
            fprintf(stderr,"[req %s] %s %s (body %zu)\n",
                    _ts,r.method.c_str(),r.path.c_str(),r.body.size());
            fflush(stderr);
        }
    }

    // UI
    if(r.method=="GET"&&(r.path=="/"||r.path=="/index.html")){
        int image_max_edge = 0;
        if(const char* env = std::getenv("NEXUS_WEB_IMAGE_MAX_EDGE")) {
            image_max_edge = std::atoi(env);
            if(image_max_edge < 0) image_max_edge = 0;
        }
        int image_max_pixels = 0;
        if(const char* env = std::getenv("NEXUS_WEB_IMAGE_MAX_PIXELS")) {
            image_max_pixels = std::atoi(env);
            if(image_max_pixels < 0) image_max_pixels = 0;
        }
        std::string ui = UI;
        const std::string token = "__NEXUS_IMAGE_MAX_EDGE__";
        size_t pos = ui.find(token);
        if(pos != std::string::npos) ui.replace(pos, token.size(), std::to_string(image_max_edge));
        const std::string token2 = "__NEXUS_IMAGE_MAX_PIXELS__";
        pos = ui.find(token2);
        if(pos != std::string::npos) ui.replace(pos, token2.size(), std::to_string(image_max_pixels));
        resp_html(fd,ui);close(fd);return;
    }

    // Document reader files (see nx_web_dir). Read-only, only below web/vendor, only plain path characters.
    if(r.method=="GET" && r.path.rfind("/vendor/", 0) == 0){
        std::string rel = r.path.substr(8);
        const size_t q = rel.find_first_of("?#"); if(q != std::string::npos) rel.resize(q);
        bool ok = !rel.empty() && rel.size() < 256 && rel.find("..") == std::string::npos && rel[0] != '/';
        for(char ch : rel) if(!(isalnum((unsigned char)ch) || ch=='.' || ch=='_' || ch=='-' || ch=='/')) { ok = false; break; }
        const std::string& wd = nx_web_dir();
        std::string body;
        if(ok && !wd.empty()){
            std::ifstream f(wd + "/vendor/" + rel, std::ios::binary);
            if(f){ std::ostringstream ss; ss << f.rdbuf(); body = ss.str(); } else ok = false;
        } else ok = false;
        if(!ok){
            send_all(fd, std::string("HTTP/1.1 404 Not Found\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n") +
                         (wd.empty() ? "The document readers (web/vendor) were not found next to the Nexus server." : "not found"));
            close(fd); return;
        }
        send_all(fd, std::string("HTTP/1.1 200 OK\r\nContent-Type: ") + nx_vendor_mime(rel) +
                     "\r\nContent-Length: " + std::to_string(body.size()) +
                     "\r\nCache-Control: max-age=604800\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n" + body);
        close(fd); return;
    }

    // Handle CORS preflight (browser sends OPTIONS before fetch with custom headers)
    if(r.method=="OPTIONS"){
        send_all(fd,"HTTP/1.1 204 No Content\r\n"+nx_acao()+
                    "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
                    "Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key\r\nAccess-Control-Max-Age: 600\r\nConnection: close\r\n\r\n");
        close(fd);return;
    }


    // Binary RGB image upload. The browser no longer base64-encodes raw image
    // payloads inside JSON, because that made small compressed files balloon
    // into massive JS strings and duplicate copies. This endpoint receives the
    // raw RGB bytes, stores them once, and /api/chat or /api/session/extend_turn
    // references them by id.
    if(r.method=="POST" && r.path.rfind("/api/image/upload", 0) == 0){
        std::string name = nx_query_param(r.path, "name", "image");
        int w = std::atoi(nx_query_param(r.path, "w", "0").c_str());
        int h = std::atoi(nx_query_param(r.path, "h", "0").c_str());
        const size_t need = (w > 0 && h > 0) ? (size_t)w * (size_t)h * 3u : 0u;
        if(w <= 0 || h <= 0 || need == 0 || r.body.size() != need){
            char b[256];
            std::snprintf(b, sizeof(b), "{\"ok\":false,\"error\":\"bad RGB upload: got=%zu need=%zu w=%d h=%d\"}",
                          r.body.size(), need, w, h);
            resp_json(fd, b, 400); close(fd); return;
        }
        std::string id = nx_safe_id();
        std::string path = nx_upload_path_for_id(id);
        std::ofstream f(path, std::ios::binary);
        if(!f){ resp_json(fd, "{\"ok\":false,\"error\":\"could not create image upload file\"}", 400); close(fd); return; }
        f.write(r.body.data(), (std::streamsize)r.body.size());
        (f.close)();   // parenthesised: on Windows `close(s)` is a macro for closesocket(s)
        if(!f){ resp_json(fd, "{\"ok\":false,\"error\":\"could not write image upload file\"}", 400); close(fd); return; }
        fprintf(stderr,"[vision-upload] stored %s id=%s %dx%d RGB %zu bytes\n",
                name.c_str(), id.c_str(), w, h, r.body.size());
        std::string j = "{\"ok\":true,\"id\":" + jstr(id) +
                        ",\"name\":" + jstr(name) +
                        ",\"width\":" + std::to_string(w) +
                        ",\"height\":" + std::to_string(h) +
                        ",\"bytes\":" + std::to_string(r.body.size()) + "}";
        resp_json(fd, j); close(fd); return;
    }

    // Info
    if(r.method=="GET"&&(r.path=="/api/info"||r.path.rfind("/api/info?",0)==0)){   // a query string (?model=2) must not make it a different page
        auto i=S.engine.model_info();
        std::string sys=S.engine.default_system_prompt();
        std::string gpu_name=S.engine.gpu_device_name();
        bool zero_copy=S.engine.gpu_zero_copy();
        std::string j="{\"name\":"+jstr(i.name)+",\"arch\":"+jstr(i.arch)+
            ",\"chat_format\":"+jstr(i.chat_format)+",\"n_layers\":"+jnum(i.n_layers)+
            ",\"think_control\":"+(S.engine.supports_think_control()?"true":"false")+
            ",\"n_heads\":"+jnum(i.n_heads)+",\"n_kv_heads\":"+jnum(i.n_kv_heads)+
            ",\"hidden_dim\":"+jnum(i.hidden_dim)+",\"context_length\":"+jnum(i.context_length)+
            ",\"vocab_size\":"+jnum(i.vocab_size)+
            ",\"n_gpu_layers\":"+jnum((uint32_t)S.n_gpu_layers)+
            ",\"gpu_device\":"+jstr(gpu_name)+
            ",\"gpu_zero_copy\":"+(zero_copy?"true":"false")+
            ",\"default_system_prompt\":"+jstr(sys)+
            ",\"has_model2\":"+( S.has_engine2?"true":"false")+
            ",\"has_vision\":"+(S.engine.has_vision()?"true":"false")+
            ",\"has_audio\":"+(S.engine.has_audio()?"true":"false")+
            ",\"lora_count\":"+std::to_string(S.engine.lora_count())+
            ",\"context_size\":"+std::to_string(S.engine.context_size())+
            ",\"max_context\":"+std::to_string(S.engine.max_context())+",\"warm_conv_id\":"+jstr(S.warm_conv_id)+"}";
        if(S.has_engine2){
            auto i2=S.engine2.model_info();
            std::string gpu2=S.engine2.gpu_device_name();
            bool zc2=S.engine2.gpu_zero_copy();
            // strip trailing } and append model2 fields
            j.pop_back();
            j+=",\"model2_name\":"+jstr(i2.name)+
                ",\"model2_arch\":"+jstr(i2.arch)+
                ",\"model2_chat_format\":"+jstr(i2.chat_format)+
                ",\"model2_n_layers\":"+jnum(i2.n_layers)+
                ",\"model2_hidden_dim\":"+jnum(i2.hidden_dim)+
                ",\"model2_context_length\":"+jnum(i2.context_length)+
                ",\"model2_gpu_device\":"+jstr(gpu2)+
                ",\"model2_gpu_zero_copy\":"+(zc2?"true":"false")+
                ",\"model2_default_system_prompt\":"+jstr(S.engine2.default_system_prompt())+"}";
        }
        resp_json(fd,j);close(fd);return;
    }

    // Active config (read-only - set by launcher via nexus.json)
    if(r.method=="POST"&&r.path=="/api/config"){
        if(r.body.find("temperature")!=std::string::npos)        S.temperature       =(float)jp_num(r.body,"temperature",S.temperature);
        if(r.body.find("repeat_penalty")!=std::string::npos)     S.repeat_penalty    =(float)jp_num(r.body,"repeat_penalty",S.repeat_penalty);
        if(r.body.find("repeat_last_n")!=std::string::npos)      S.repeat_last_n     =(int)  jp_num(r.body,"repeat_last_n",S.repeat_last_n);
        if(r.body.find("frequency_penalty")!=std::string::npos)  S.frequency_penalty =(float)jp_num(r.body,"frequency_penalty",S.frequency_penalty);
        if(r.body.find("presence_penalty")!=std::string::npos)   S.presence_penalty  =(float)jp_num(r.body,"presence_penalty",S.presence_penalty);
        if(r.body.find("top_p")!=std::string::npos)              S.top_p             =(float)jp_num(r.body,"top_p",S.top_p);
        if(r.body.find("top_k")!=std::string::npos)              S.top_k             =(int)  jp_num(r.body,"top_k",S.top_k);
        if(r.body.find("max_tokens")!=std::string::npos)         S.max_tokens        =(int)  jp_num(r.body,"max_tokens",S.max_tokens);
        if(r.body.find("mtp_enabled")!=std::string::npos)        S.mtp_enabled       = jp_num(r.body,"mtp_enabled",S.mtp_enabled?1:0)!=0.0;
        if(r.body.find("mtp_n_predict")!=std::string::npos)      S.mtp_n_predict     =(int)  jp_num(r.body,"mtp_n_predict",S.mtp_n_predict);
        if(r.body.find("min_p")!=std::string::npos)              S.min_p             =(float)jp_num(r.body,"min_p",S.min_p);
        if(r.body.find("tfs_z")!=std::string::npos)              S.tfs_z             =(float)jp_num(r.body,"tfs_z",S.tfs_z);
        resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/config/save"){
        // Write current in-memory settings back to nexus.json so they survive restart
        FILE* f=fopen("nexus.json","r");
        std::string existing;
        if(f){fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);existing.resize(sz);fread(&existing[0],1,sz,f);fclose(f);}
        // Parse existing json and overwrite sampler keys
        auto set_num=[&](std::string& j, const std::string& key, double val){
            // Simple key replace: find "key": and replace its value
            auto pos=j.find("\""+key+"\":");
            if(pos==std::string::npos) return;
            auto vstart=j.find_first_not_of(" \t",pos+key.size()+3);
            if(vstart==std::string::npos) return;
            auto vend=j.find_first_of(",}\n",vstart);
            if(vend==std::string::npos) return;
            char buf[32]; snprintf(buf,sizeof(buf),"%.6g",val);
            j=j.substr(0,vstart)+buf+j.substr(vend);
        };
        if(!existing.empty()){
            set_num(existing,"temperature",   S.temperature);
            set_num(existing,"top_p",         S.top_p);
            set_num(existing,"min_p",         S.min_p);
            set_num(existing,"tfs_z",         S.tfs_z);
            set_num(existing,"repeat_penalty",S.repeat_penalty);
            set_num(existing,"repeat_last_n", S.repeat_last_n);
            set_num(existing,"frequency_penalty",S.frequency_penalty);
            set_num(existing,"presence_penalty", S.presence_penalty);
            set_num(existing,"top_k",         S.top_k);
            set_num(existing,"max_tokens",    S.max_tokens);
            set_num(existing,"mtp_enabled",   S.mtp_enabled?1:0);
            set_num(existing,"mtp_n_predict", S.mtp_n_predict);
            FILE* wf=fopen("nexus.json","w");
            if(wf){fwrite(existing.c_str(),1,existing.size(),wf);fclose(wf);}
        }
        resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    if(r.method=="GET"&&r.path=="/api/config"){
        auto i=S.engine.model_info();
        std::string j="{\"chat_format\":"+jstr(i.chat_format)+
            ",\"temperature\":"+std::to_string(S.temperature)+
            ",\"top_p\":"+std::to_string(S.top_p)+
            ",\"repeat_penalty\":"+std::to_string(S.repeat_penalty)+
            ",\"repeat_last_n\":"+std::to_string(S.repeat_last_n)+
            ",\"frequency_penalty\":"+std::to_string(S.frequency_penalty)+
            ",\"presence_penalty\":"+std::to_string(S.presence_penalty)+
            ",\"top_k\":"+std::to_string(S.top_k)+
            ",\"max_tokens\":"+std::to_string(S.max_tokens)+
            ",\"mtp_enabled\":"+std::to_string(S.mtp_enabled?1:0)+
            ",\"mtp_n_predict\":"+std::to_string(S.mtp_n_predict)+
            ",\"min_p\":"+std::to_string(S.min_p)+
            ",\"tfs_z\":"+std::to_string(S.tfs_z)+
            ",\"threads\":"+std::to_string(S.n_threads)+"}";
        resp_json(fd,j);close(fd);return;
    }

    // Web search - GET /api/search?q=query
    // Returns {"ok":true,"results":"markdown text"}
    // Uses DuckDuckGo HTML scrape via system curl
    if(r.method=="GET"&&r.path.substr(0,12)=="/api/search?"){
        { std::string why; if(!nx_web_tools_allowed(S, r, why)){ resp_json(fd, "{\"ok\":false,\"error\":" + jstr(why) + "}"); close(fd); return; } }
        std::string qs=r.path.substr(12);
        std::string query;
        for(size_t i=0;i<qs.size();i+=2){
            if(qs.substr(i,2)=="q="){size_t amp=qs.find('&',i+2);
                query=qs.substr(i+2,amp==std::string::npos?std::string::npos:amp-i-2); break;}
        }
        // URL-decode the query (basic)
        std::string dq;
        for(size_t i=0;i<query.size();i++){
            if(query[i]=='%'&&i+2<query.size()){
                int v=strtol(query.substr(i+1,2).c_str(),nullptr,16);
                dq+=(char)v;i+=2;
            } else if(query[i]=='+') dq+=' ';
            else dq+=query[i];
        }
        if(dq.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"empty query\"}");close(fd);return;}
        // Fetch DDG results (curl run directly, no shell: see nx_fetch_url)
        std::string ddg="https://html.duckduckgo.com/html/?q=";
        for(char c:dq){
            if(isalnum((unsigned char)c)||c=='-'||c=='_'||c=='.'||c=='~') ddg+=c;
            else{char buf[4];snprintf(buf,4,"%%%02X",(unsigned char)c);ddg+=buf;}
        }
        std::string raw = nx_fetch_url(ddg, 10);
        // Extract results: find result__a links and result__snippet
        std::vector<std::string> result_items;
        size_t pos2=0; int found=0;
        while(found<5&&pos2<raw.size()){
            size_t ta=raw.find("result__a",pos2);
            if(ta==std::string::npos) break;
            // Extract href
            size_t h1=raw.rfind("href=\"",ta);
            size_t h2=h1!=std::string::npos?raw.find("\"",h1+6):std::string::npos;
            // Extract title text
            size_t t1=raw.find(">",ta+9);
            size_t t2=t1!=std::string::npos?raw.find("</a>",t1):std::string::npos;
            // Extract snippet
            size_t s1=raw.find("result__snippet",ta);
            size_t s2=s1!=std::string::npos?raw.find(">",s1+15):std::string::npos;
            size_t s3=s2!=std::string::npos?raw.find("</",s2+1):std::string::npos;
            if(h1!=std::string::npos&&h2!=std::string::npos&&
               t1!=std::string::npos&&t2!=std::string::npos){
                std::string url=raw.substr(h1+6,h2-h1-6);
                std::string title=raw.substr(t1+1,t2-t1-1);
                // Strip HTML tags from title
                std::string clean_title;
                bool in_tag=false;
                for(char c:title){if(c=='<')in_tag=true;else if(c=='>')in_tag=false;else if(!in_tag)clean_title+=c;}
                std::string snip=s3!=std::string::npos?raw.substr(s2+1,s3-s2-1):"";
                std::string clean_snip;
                in_tag=false;
                for(char c:snip){if(c=='<')in_tag=true;else if(c=='>')in_tag=false;else if(!in_tag)clean_snip+=c;}
                if(!clean_title.empty()&&!url.empty()&&url.find("duckduckgo")==std::string::npos){
                    std::string host=url, display_url=url;
                    size_t scheme=host.find("://");
                    if(scheme!=std::string::npos) host=host.substr(scheme+3);
                    size_t slash=host.find('/');
                    if(slash!=std::string::npos) host=host.substr(0,slash);

                    std::string item =
                        std::string("{\"id\":")+std::to_string(found+1)+
                        ",\"title\":"+jstr(clean_title)+
                        ",\"url\":"+jstr(url)+
                        ",\"display_url\":"+jstr(display_url)+
                        ",\"host\":"+jstr(host)+
                        ",\"snippet\":"+jstr(clean_snip)+"}";

                    result_items.push_back(item);
                    found++;
                }
            }
            pos2=ta+10;
        }
        std::string joined;
        for(size_t i=0;i<result_items.size();++i){
            if(i) joined += ",";
            joined += result_items[i];
        }

        resp_json(fd,
            std::string("{\"ok\":true,\"query\":")+jstr(dq)+
            ",\"results\":["+joined+"]}");
        close(fd); return;
    }
    
    
    // Search detail by query/result index - GET /api/search_detail?q=...&n=...
    if(r.method=="GET" && r.path.substr(0,19)=="/api/search_detail?"){
        { std::string why; if(!nx_web_tools_allowed(S, r, why)){ resp_json(fd, "{\"ok\":false,\"error\":" + jstr(why) + "}"); close(fd); return; } }
        std::string qs=r.path.substr(19);
        std::string q_enc, n_str="1";

        for(size_t i=0;i<qs.size();) {
            size_t amp=qs.find('&',i);
            std::string kv=qs.substr(i, amp==std::string::npos ? std::string::npos : amp-i);
            size_t eq=kv.find('=');
            if(eq!=std::string::npos){
                std::string k=kv.substr(0,eq);
                std::string v=kv.substr(eq+1);
                if(k=="q") q_enc=v;
                else if(k=="n") n_str=v;
            }
            if(amp==std::string::npos) break;
            i=amp+1;
        }

        auto url_decode = [](const std::string& s)->std::string{
            std::string out;
            for(size_t i=0;i<s.size();++i){
                if(s[i]=='%' && i+2<s.size()){
                    int h=0;
                    std::stringstream ss;
                    ss << std::hex << s.substr(i+1,2);
                    ss >> h;
                    out.push_back((char)h);
                    i+=2;
                } else if(s[i]=='+') {
                    out.push_back(' ');
                } else {
                    out.push_back(s[i]);
                }
            }
            return out;
        };

        std::string dq = url_decode(q_enc);
        int want = std::max(1, atoi(n_str.c_str()));

        if(dq.empty()){
            resp_json(fd,"{\"ok\":false,\"error\":\"missing query\"}");
            close(fd); return;
       }

       // Same DDG fetch as /api/search (curl run directly, no shell)
        std::string ddg="https://html.duckduckgo.com/html/?q=";
        for(char c:dq){
            if(isalnum((unsigned char)c)||c=='-'||c=='_'||c=='.'||c=='~') ddg+=c;
            else{char buf[4];snprintf(buf,4,"%%%02X",(unsigned char)c);ddg+=buf;}
        }
        std::string raw = nx_fetch_url(ddg, 10);

        // Parse results same way as /api/search and pick nth URL
        size_t pos2=0; int found=0;
        std::string picked_url;
        while(found<want && pos2<raw.size()){
            size_t ta=raw.find("result__a",pos2);
            if(ta==std::string::npos) break;

            size_t h1=raw.rfind("href=\"",ta);
            size_t h2=h1!=std::string::npos?raw.find("\"",h1+6):std::string::npos;
            size_t t1=raw.find(">",ta+9);
            size_t t2=t1!=std::string::npos?raw.find("</a>",t1):std::string::npos;

            if(h1!=std::string::npos&&h2!=std::string::npos&&
               t1!=std::string::npos&&t2!=std::string::npos){
                std::string url=raw.substr(h1+6,h2-h1-6);
                std::string title=raw.substr(t1+1,t2-t1-1);

                std::string clean_title;
                bool in_tag=false;
                for(char c:title){
                    if(c=='<') in_tag=true;
                    else if(c=='>') in_tag=false;
                    else if(!in_tag) clean_title+=c;
                }

                if(!clean_title.empty() && !url.empty() && url.find("duckduckgo")==std::string::npos){
                    found++;
                    if(found==want){
                        picked_url=url;
                        break;
                    }
                }
            }
            pos2=ta+10;
        }

        if(picked_url.empty()){
            resp_json(fd,"{\"ok\":false,\"error\":\"result not found\"}");
            close(fd); return;
        }

        // Hand off to the URL detail route logic
        std::string enc;
        for(unsigned char c: picked_url){
            if(isalnum(c)||c=='-'||c=='_'||c=='.'||c=='~') enc.push_back((char)c);
            else { char buf[4]; snprintf(buf,4,"%%%02X",c); enc += buf; }
        }

        std::string fake = "/api/search_detail_by_url?url=" + enc;
        r.path = fake;
    }
    
    // Search detail by URL - GET /api/search_detail_by_url?url=...
    if(r.method=="GET" && r.path.substr(0,26)=="/api/search_detail_by_url?"){
        { std::string why; if(!nx_web_tools_allowed(S, r, why)){ resp_json(fd, "{\"ok\":false,\"error\":" + jstr(why) + "}"); close(fd); return; } }
        std::string qs=r.path.substr(26);
        std::string enc_url;
        for(size_t i=0;i<qs.size();) {
            size_t amp=qs.find('&',i);
            std::string kv=qs.substr(i, amp==std::string::npos ? std::string::npos : amp-i);
            size_t eq=kv.find('=');
            if(eq!=std::string::npos){
                std::string k=kv.substr(0,eq);
                std::string v=kv.substr(eq+1);
                if(k=="url"){ enc_url=v; break; }
            }
            if(amp==std::string::npos) break;
            i=amp+1;
        }

        std::string url;
        for(size_t i=0;i<enc_url.size();++i){
            if(enc_url[i]=='%' && i+2<enc_url.size()){
                int h=0;
                std::stringstream ss;
                ss << std::hex << enc_url.substr(i+1,2);
                ss >> h;
                url.push_back((char)h);
                i+=2;
            } else if(enc_url[i]=='+') {
                url.push_back(' ');
            } else {
                url.push_back(enc_url[i]);
            }
        }

        if(url.empty()){
            resp_json(fd,"{\"ok\":false,\"error\":\"missing url\"}");
            close(fd); return;
        }

        // This used to build a shell command line around the URL, where $(...) or `...` in it ran as a
        // command. Now curl runs directly with the URL as one argument, and only http/https URLs are accepted.
        if(!nx_url_ok(url)){
            resp_json(fd,"{\"ok\":false,\"error\":\"only plain http:// or https:// addresses can be fetched\"}");
            close(fd); return;
        }
        std::string html = nx_fetch_url(url, 20);

        if(html.empty()){
            resp_json(fd,"{\"ok\":false,\"error\":\"empty page\"}");
            close(fd); return;
        }

        std::string text;
        bool in_tag=false;
        for(char c:html){
            if(c=='<') in_tag=true;
            else if(c=='>'){ in_tag=false; text.push_back('\n'); }
            else if(!in_tag) text.push_back(c);
        }

        // compact whitespace
        std::string clean;
        bool prev_nl=false, prev_sp=false;
        for(char c:text){
            if(c=='\r') continue;
            if(c=='\n'){
                if(!prev_nl) clean.push_back('\n');
                prev_nl=true; prev_sp=false;
            } else if(c==' ' || c=='\t'){
                if(!prev_sp && !prev_nl) clean.push_back(' ');
                prev_sp=true;
            } else {
                clean.push_back(c);
                prev_nl=false; prev_sp=false;
            }
        }

        if(clean.size()>12000) clean=clean.substr(0,12000);
    
        std::string out="[PAGE]\nURL: "+url+"\n\n"+clean;
        resp_json(fd, std::string("{\"ok\":true,\"result\":")+jstr(out)+"}");
        close(fd); return;
    }
    
    // POST /api/tokenize  {"text":"...","add_bos":true,"model":"1"}
    // Returns {"ok":true,"count":N,"tokens":[...]}
    if(r.method=="POST"&&r.path=="/api/tokenize"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::string text=jp_str(r.body,"text","");
        bool add_bos=jp_str(r.body,"add_bos","true")!="false";
        auto toks=e.tokenize(text,add_bos);
        std::string j="{\"ok\":true,\"count\":"+std::to_string(toks.size())+",\"tokens\":[";
        for(size_t i=0;i<toks.size();++i){if(i)j+=",";j+=std::to_string(toks[i]);}
        j+="]}"; resp_json(fd,j); close(fd); return;
    }
    // POST /api/detokenize  {"tokens":[1,2,3],"model":"1"}
    // Returns {"ok":true,"text":"..."}
    if(r.method=="POST"&&r.path=="/api/detokenize"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::vector<int32_t> toks;
        size_t p=r.body.find("tokens"); if(p!=std::string::npos) p=r.body.find('[',p);
        while(p!=std::string::npos&&p<r.body.size()&&r.body[p]!=']'){
            p++; char* end; long v=strtol(r.body.c_str()+p,&end,10);
            if(end==r.body.c_str()+p) break;
            toks.push_back((int32_t)v); p=(size_t)(end-r.body.c_str());
            while(p<r.body.size()&&(r.body[p]==','||r.body[p]==' ')) p++;
        }
        std::string text=e.detokenize(toks);
        resp_json(fd,"{\"ok\":true,\"text\":"+jstr(text)+"}"); close(fd); return;
    }
    // GET /api/context  - current and max context token counts
    // GET /api/progress?model=1 -> {"active":true,"pct":34.2,"tokens":98304,"tps":104.2,"eta":580} while a prompt is
    // being read on this computer ({"active":false} otherwise). For clients waiting on a request that doesn't stream.
    if(r.method=="GET"&&(r.path=="/api/progress"||r.path.rfind("/api/progress?",0)==0)){
        const bool p2 = S.has_engine2 && r.path.find("model=2")!=std::string::npos;
        resp_json(fd, nx_prog_get(p2 ? &S.engine2 : &S.engine)); close(fd); return;
    }
    if(r.method=="GET"&&(r.path=="/api/context"||r.path.rfind("/api/context?",0)==0)){   // ?model=2 is read below
        bool u2=S.has_engine2&&r.path.find("model=2")!=std::string::npos;
        nexus::Engine& e=u2?S.engine2:S.engine;
        resp_json(fd,"{\"context_size\":"+std::to_string(e.context_size())+
                     ",\"max_context\":"+std::to_string(e.max_context())+
                     ",\"remaining\":"+std::to_string(e.context_limit()-e.context_size())+
                     ",\"limit\":"+std::to_string(e.context_limit())+"}"); close(fd); return;
    }
    // GET /api/lora/list
    if(r.method=="GET"&&r.path=="/api/lora/list"){
        resp_json(fd,"{\"ok\":true,\"count\":"+std::to_string(S.engine.lora_count())+"}"); close(fd); return;
    }
    // POST /api/lora/scale  {"index":0,"scale":0.8,"model":"1"}
    if(r.method=="POST"&&r.path=="/api/lora/scale"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        int idx=(int)jp_num(r.body,"index",0); float sc=(float)jp_num(r.body,"scale",1.0);
        std::lock_guard<std::mutex> lg(mu); e.lora_set_scale(idx,sc);
        resp_json(fd,"{\"ok\":true}"); close(fd); return;
    }
    // GET /api/vision/status
    if(r.method=="GET"&&r.path=="/api/vision/status"){
        resp_json(fd,std::string("{\"loaded\":")+( S.engine.has_vision()?"true":"false")+"}"); close(fd); return;
    }
    // POST /api/vision/inject  {"tokens":[...floats...]}
    if(r.method=="POST"&&r.path=="/api/vision/inject"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::vector<float> toks;
        size_t p=r.body.find("tokens"); if(p!=std::string::npos) p=r.body.find('[',p);
        while(p!=std::string::npos&&p<r.body.size()&&r.body[p]!=']'){
            p++; char* end; float v=strtof(r.body.c_str()+p,&end);
            if(end==r.body.c_str()+p) break;
            toks.push_back(v); p=(size_t)(end-r.body.c_str());
            while(p<r.body.size()&&(r.body[p]==','||r.body[p]==' ')) p++;
        }
        if(toks.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"no tokens\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu); e.inject_vision_tokens(toks);
        resp_json(fd,"{\"ok\":true,\"injected\":"+std::to_string(toks.size())+"}"); close(fd); return;
    }
    // GET /api/audio/status
    if(r.method=="GET"&&r.path=="/api/audio/status"){
        resp_json(fd,std::string("{\"loaded\":")+( S.engine.has_audio()?"true":"false")+"}"); close(fd); return;
    }
    // POST /api/audio/inject  {"tokens":[...floats...]}
    if(r.method=="POST"&&r.path=="/api/audio/inject"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::vector<float> toks;
        size_t p=r.body.find("tokens"); if(p!=std::string::npos) p=r.body.find('[',p);
        while(p!=std::string::npos&&p<r.body.size()&&r.body[p]!=']'){
            p++; char* end; float v=strtof(r.body.c_str()+p,&end);
            if(end==r.body.c_str()+p) break;
            toks.push_back(v); p=(size_t)(end-r.body.c_str());
            while(p<r.body.size()&&(r.body[p]==','||r.body[p]==' ')) p++;
        }
        if(toks.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"no tokens\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu); e.inject_audio_tokens(toks);
        resp_json(fd,"{\"ok\":true,\"injected\":"+std::to_string(toks.size())+"}"); close(fd); return;
    }
    // GET /api/slot/list  - list all context slots with token counts
    if(r.method=="GET"&&r.path=="/api/slot/list"){
        int n=S.engine.context_slot_count();
        std::string j="{\"ok\":true,\"count\":"+std::to_string(n)+",\"slots\":[";
        for(int i=0;i<n;++i){if(i)j+=","; j+="{\"slot\":"+std::to_string(i)+",\"tokens\":"+std::to_string(S.engine.context_slot_size(i))+"}";}
        j+="]}"; resp_json(fd,j); close(fd); return;
    }
    // POST /api/slot/clear  {"slot":0}
    if(r.method=="POST"&&r.path=="/api/slot/clear"){
        int slot=(int)jp_num(r.body,"slot",0);
        std::lock_guard<std::mutex> lg(S.gen_mutex); S.engine.context_slot_clear(slot);
        resp_json(fd,"{\"ok\":true}"); close(fd); return;
    }
    // GET /api/top_logits?n=10  - top N token log-probs from last forward pass
    if(r.method=="GET"&&r.path.size()>15&&r.path.substr(0,15)=="/api/top_logits"){
        int n=10; auto eq=r.path.find("n=");
        if(eq!=std::string::npos) n=atoi(r.path.c_str()+eq+2);
        n=std::max(1,std::min(n,200));
        auto top=S.engine.top_logits(n);
        std::string j="{\"ok\":true,\"tokens\":[";
        for(size_t i=0;i<top.size();++i){if(i)j+=","; char buf[32]; snprintf(buf,32,"%.4f",top[i].second); j+="{\"id\":"+std::to_string(top[i].first)+",\"logit\":"+buf+"}";}
        j+="]}"; resp_json(fd,j); close(fd); return;
    }
    // GET /api/session/tokens  - current context size (token count)
    if(r.method=="GET"&&r.path=="/api/session/tokens"){
        resp_json(fd,"{\"ok\":true,\"context_size\":"+std::to_string(S.engine.context_size())+"}"); close(fd); return;
    }
    // Stop current generation
    if(r.method=="POST"&&r.path=="/api/stop"){
        S.stop_flag  = true;
        S.stop_flag2 = true;
        resp_json(fd,"{\"ok\":true}"); close(fd); return;
    }
    
    // Reset
    if(r.method=="POST"&&r.path=="/api/reset"){
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        if(use2){ std::lock_guard<std::mutex> lg(S.gen_mutex2); S.engine2.reset_context(); }
        else    { std::lock_guard<std::mutex> lg(S.gen_mutex);  S.engine.reset_context();  }
        nx_forget_conversations(S, use2?2:1, "");
        resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    
    // GET /api/turbo — current turbo-prefill state for each model
    if(r.method=="GET" && r.path=="/api/turbo"){
        auto stat = [](nexus::Engine& e) {
            std::string j = "{\"enabled\":";
            j += e.turbo_prefill_enabled() ? "true" : "false";
            j += ",\"gpu_stream\":";
            j += e.turbo_gpu_stream() ? "true" : "false";
            j += ",\"vram_budget_mb\":";
            j += std::to_string(e.turbo_vram_budget_mb());
            j += "}";
            return j;
        };
        std::string j = "{\"ok\":true,\"model1\":" + stat(S.engine);
        if (S.has_engine2) { j += ",\"model2\":" + stat(S.engine2); }
        j += "}";
        resp_json(fd, j); close(fd); return;
    }
    // POST /api/turbo  {"enabled":1, "gpu_stream":0|1, "vram_budget_mb":2500, "model":"1"|"2"}
    // Any field is optional; only present fields are updated.
    if(r.method=="POST" && r.path=="/api/turbo"){
        bool use2 = S.has_engine2 && jp_str(r.body, "model", "1") == "2";
        nexus::Engine& eng = use2 ? S.engine2 : S.engine;
        std::mutex& mu     = use2 ? S.gen_mutex2 : S.gen_mutex;
        std::lock_guard<std::mutex> lg(mu);

        const bool enabled_present = r.body.find("\"enabled\"") != std::string::npos;
        const bool stream_present  = r.body.find("\"gpu_stream\"") != std::string::npos;
        const bool enabled = enabled_present
            ? (jp_num(r.body, "enabled", 0) != 0)
            : eng.turbo_prefill_enabled();

        // Live toggle policy:
        //   enabled=true  -> request GPU stream and then enable turbo.
        //   enabled=false -> disable turbo; stream may also be disabled if requested.
        // The Engine refuses to leave turbo enabled without an active GPU stream,
        // so Turbo cannot silently become CPU/layer-major.
        const bool stream_req = enabled_present
            ? enabled
            : (stream_present ? (jp_num(r.body, "gpu_stream", 0) != 0) : eng.turbo_gpu_stream());

        if (r.body.find("\"gpu_device\"") != std::string::npos)
            eng.set_turbo_gpu_device((int)jp_num(r.body, "gpu_device", -1));
        if (r.body.find("\"vram_budget_mb\"") != std::string::npos)
            eng.set_turbo_vram_budget_mb((int)jp_num(r.body, "vram_budget_mb", 0));

        if (enabled_present || stream_present)
            eng.set_turbo_gpu_stream(stream_req);

        if (enabled_present)
            eng.set_turbo_prefill(enabled);

        std::string j = "{\"ok\":true,\"enabled\":";
        j += eng.turbo_prefill_enabled() ? "true" : "false";
        j += ",\"gpu_stream\":";
        j += eng.turbo_gpu_stream() ? "true" : "false";
        j += ",\"vram_budget_mb\":";
        j += std::to_string(eng.turbo_vram_budget_mb());
        j += "}";
        resp_json(fd, j); close(fd); return;
    }

    if(r.method=="POST" && r.path=="/api/preload"){
    bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
    nexus::Engine& eng = use2 ? S.engine2 : S.engine;
    std::mutex& eng_mu = use2 ? S.gen_mutex2 : S.gen_mutex;

    auto messages = jp_messages(r.body);
    bool do_reset = jp_num(r.body, "reset", 1) != 0;

    std::lock_guard<std::mutex> lg(eng_mu);

    if(do_reset) eng.reset_context();

    std::string prompt = eng.format_system_prefix(messages);
    fprintf(stderr, "[preload%s] %zu messages - prompt (%zu chars): %.200s%s\n",
            use2 ? "2" : "", messages.size(), prompt.size(),
            prompt.c_str(), prompt.size()>200?"...":"");
    fflush(stderr);

    nexus::GenerateConfig cfgp;
    cfgp.temperature       = S.temperature;
    cfgp.max_tokens        = S.max_tokens;
    cfgp.top_p             = S.top_p;
    cfgp.mtp_enabled       = S.mtp_enabled;
    cfgp.mtp_n_predict     = S.mtp_n_predict;
    cfgp.mtp_gpu           = S.mtp_gpu;
    cfgp.top_k             = S.top_k;
    cfgp.min_p             = S.min_p;
    cfgp.tfs_z             = S.tfs_z;
    cfgp.repeat_penalty    = S.repeat_penalty;
    cfgp.repeat_last_n     = S.repeat_last_n;
    cfgp.frequency_penalty = S.frequency_penalty;
    cfgp.presence_penalty  = S.presence_penalty;
    cfgp.stop_strings      = eng.stop_strings();
    cfgp.stop_strings.push_back("\nHuman:");
    cfgp.stop_strings.push_back("\nUser:");
    cfgp.stop_strings.push_back("\nAssistant:");

    auto sess = eng.begin_session(prompt, cfgp, {});
    eng.sync_session_tokens(sess.tokens(), sess.pos());

    // Register warm KV in conversations so the next /api/chat
    // knows it can extend rather than re-prefill from scratch.
    char wid[32];
    snprintf(wid, sizeof(wid), "warm_%lx", (unsigned long)time(nullptr));
    std::string warm_conv_id = wid;
    {
        std::lock_guard<std::mutex> cl(S.sessions_mu);
        S.conversations[warm_conv_id].last_prompt = prompt;
        S.conversations[warm_conv_id].n_tokens    = eng.context_size();
        S.conversations[warm_conv_id].model       = use2 ? 2 : 1;
    }
    nx_forget_conversations(S, use2 ? 2 : 1, warm_conv_id);

    resp_json(fd,
      std::string("{\"ok\":true,\"conv_id\":\"") + warm_conv_id +
      "\",\"context_size\":" + std::to_string(eng.context_size()) + "}");
    close(fd);
    return;
}  

    // Roster - list of running model instances written by launcher
    if(r.method=="GET"&&r.path=="/api/roster"){
        std::string j="[]";
        FILE* rf=fopen(S.roster_file.c_str(),"r");
        if(rf){
            fseek(rf,0,SEEK_END);long sz=ftell(rf);rewind(rf);
            if(sz>0&&sz<65536){j.assign(sz,'\0');(void)fread(&j[0],1,sz,rf);}
            fclose(rf);
        }
        resp_json(fd,j);close(fd);return;
    }

    // Generate SSE
    // ── Conversation memory (KV) editing ───────────────────────────────────────────────
    // GET  /api/kv/turns?model=1            the turns in memory (start/end tokens, role, preview, protected)
    // POST /api/kv/remove   {model,start,end,ctx}           forget part of the conversation
    // POST /api/kv/archive  {model,start,end,ctx,label,msgs} save it to <data>/kv_archive/A<n>.nxctx, then forget it
    // GET  /api/kv/archive/list                             saved parts (id, label, model, tokens, size, chat text)
    // POST /api/kv/archive/restore {model,id,ctx}           put a saved part back at the end of the conversation
    // POST /api/kv/archive/delete  {id}   /rename {id,label}
    // POST /api/kv/settings {model,limit:"max"|"auto"|N,auto_trim:"on"|"off"}
    // "ctx" is the context size the page last saw: if it changed meanwhile, the edit is refused
    // rather than applied to the wrong tokens.
    if(r.path.rfind("/api/kv/",0)==0){
        const bool k2 = S.has_engine2 && (jp_str(r.body,"model","1")=="2" || r.path.find("model=2")!=std::string::npos);
        nexus::Engine& ke = k2 ? S.engine2 : S.engine;
        std::mutex& kmu  = k2 ? S.gen_mutex2 : S.gen_mutex;
        auto kerr=[&](const std::string& m, bool busy=false){
            resp_json(fd,std::string("{\"ok\":false")+(busy?",\"busy\":true":"")+",\"error\":"+jstr(m)+"}"); close(fd); };
        if(!k2 && S.dist_sess && S.dist_sess->is_connected() && r.path!="/api/kv/archive/list" && r.path!="/api/kv/archive/delete" && r.path!="/api/kv/archive/rename"){
            kerr("this model is split across computers; editing its memory only works when it runs on one computer"); return;
        }
        auto turns_json=[&](){
            std::string o="{\"ok\":true,\"model\":"+std::string(k2?"2":"1")+
                ",\"context\":"+std::to_string(ke.context_size())+",\"limit\":"+std::to_string(ke.context_limit())+
                ",\"model_max\":"+std::to_string(ke.max_context())+",\"limit_cfg\":"+std::to_string(S.ctx_limit_cfg)+
                ",\"auto_trim\":"+(ke.auto_trim()?"true":"false")+",\"exact\":"+(ke.context_edits_exact()?"true":"false")+
                ",\"bytes_per_token\":"+std::to_string(ke.kv_bytes_per_token())+
                ",\"protected\":"+std::to_string(ke.context_protected_tokens())+",\"turns\":[";
            bool first=true;
            for(const auto& t: ke.context_turns()){
                if(!first) o+=","; first=false;
                o+="{\"start\":"+std::to_string(t.start)+",\"end\":"+std::to_string(t.end)+",\"role\":"+jstr(t.role)+
                   ",\"protected\":"+(t.protected_part?"true":"false")+",\"preview\":"+jstr(t.preview)+"}";
            }
            return o+"]}";
        };
        if(r.method=="GET" && r.path.rfind("/api/kv/turns",0)==0){
            std::unique_lock<std::mutex> l(kmu, std::try_to_lock);
            if(!l.owns_lock()){ kerr("the model is busy (a reply is being written)", true); return; }
            resp_json(fd, turns_json()); close(fd); return;
        }
        if(r.method=="GET" && r.path=="/api/kv/archive/list"){
            std::string o="{\"ok\":true,\"items\":["; bool first=true;
            for(const auto& f: list_dir(nx_kv_dir(S))){
                if(f.size()<6 || f.substr(f.size()-5)!=".json") continue;
                std::string m=read_file(nx_kv_dir(S)+"/"+f);
                if(m.empty()||m[0]!='{') continue;
                if(!first) o+=","; first=false; o+=m;
            }
            o+="],\"current_file\":"+jstr(nx_model_file_tag(S.model_file[0]))+",\"current_file2\":"+jstr(nx_model_file_tag(S.model_file[1]))+"}";
            resp_json(fd,o); close(fd); return;
        }
        if(r.method=="POST" && (r.path=="/api/kv/archive/delete" || r.path=="/api/kv/archive/rename")){
            const std::string id=nx_kv_safe_id(jp_str(r.body,"id",""));
            const std::string base=nx_kv_dir(S)+"/"+id;
            std::string meta=read_file(base+".json");
            if(id.empty()||meta.empty()){ kerr("that saved memory was not found"); return; }
            if(r.path=="/api/kv/archive/delete"){ std::remove((base+".nxctx").c_str()); std::remove((base+".json").c_str()); }
            else {
                const std::string old_label=jp_str(meta,"label","");
                const std::string key="\"label\":"+jstr(old_label);
                const size_t at=meta.find(key);
                if(at!=std::string::npos) meta.replace(at,key.size(),"\"label\":"+jstr(jp_str(r.body,"label",old_label)));
                write_file(base+".json",meta);
            }
            resp_json(fd,"{\"ok\":true}"); close(fd); return;
        }
        if(r.method=="POST" && r.path=="/api/kv/settings"){
            std::lock_guard<std::mutex> l(kmu);
            const std::string lim=jp_str(r.body,"limit","");
            if(lim=="max") S.ctx_limit_cfg=0; else if(lim=="auto") S.ctx_limit_cfg=-1;
            else if(!lim.empty()) S.ctx_limit_cfg=std::max(0,atoi(lim.c_str()));
            else if(jp_num(r.body,"limit",-2)>=0) S.ctx_limit_cfg=(int)jp_num(r.body,"limit",0);
            const std::string at=jp_str(r.body,"auto_trim","");
            if(!at.empty()) S.auto_trim=(at=="on"||at=="1"||at=="true");
            nx_apply_ctx_limit(S, S.engine, "model 1");
            if(S.has_engine2) nx_apply_ctx_limit(S, S.engine2, "model 2");
            resp_json(fd, turns_json()); close(fd); return;
        }
        if(r.method=="POST" && (r.path=="/api/kv/remove" || r.path=="/api/kv/archive" || r.path=="/api/kv/archive/restore")){
            std::unique_lock<std::mutex> l(kmu, std::try_to_lock);
            if(!l.owns_lock()){ kerr("the model is busy writing a reply; stop it or wait, then try again", true); return; }
            const int seen=(int)jp_num(r.body,"ctx",-1);
            if(seen>=0 && seen!=ke.context_size()){ kerr("the conversation changed since this page last looked; refresh and try again"); return; }
            std::string err;
            if(r.path=="/api/kv/archive/restore"){
                const std::string id=nx_kv_safe_id(jp_str(r.body,"id",""));
                const std::string base=nx_kv_dir(S)+"/"+id;
                const std::string meta=read_file(base+".json");
                if(id.empty()||meta.empty()){ kerr("that saved memory was not found"); return; }
                const int n=(int)jp_num(meta,"tokens",0);
                if(ke.context_size()+n > ke.context_limit()-64){
                    if(ke.context_make_room(n, 256) <= 0 || ke.context_size()+n > ke.context_limit()-64){
                        kerr("there is no room for it: the conversation is "+std::to_string(ke.context_size())+" of "+std::to_string(ke.context_limit())+" tokens and this part is "+std::to_string(n)); return; }
                }
                const int added=ke.context_restore(base+".nxctx",&err);
                if(added<0){ kerr(err); return; }
                std::string warn;
                const std::string saved_file=jp_str(meta,"file",""), cur_file=nx_model_file_tag(S.model_file[k2?1:0]);
                if(!saved_file.empty() && !cur_file.empty() && saved_file!=cur_file){
                    warn="This memory was saved with "+saved_file+" and the model running now is "+cur_file+". It fits the same model shape, so it was restored, but a different quantization or fine-tune can remember it less accurately.";
                    fprintf(stderr,"[ctx] restore %s: %s\n", id.c_str(), warn.c_str()); fflush(stderr);
                }
                std::string o=turns_json(); o.pop_back();
                o+=",\"added\":"+std::to_string(added)+(warn.empty()?"":",\"warning\":"+jstr(warn))+",\"item\":"+meta+"}";
                resp_json(fd,o); close(fd); return;
            }
            const int a=(int)jp_num(r.body,"start",-1), b=(int)jp_num(r.body,"end",-1);
            if(r.path=="/api/kv/remove"){
                if(!ke.context_remove(a,b,&err)){ kerr(err); return; }
                resp_json(fd,turns_json()); close(fd); return;
            }
            // archive: next free id A1, A2, ...
            mkdir_p(nx_kv_dir(S));
            int next=1;
            for(const auto& f: list_dir(nx_kv_dir(S)))
                if(f.size()>1 && f[0]=='A' && isdigit((unsigned char)f[1])) next=std::max(next, atoi(f.c_str()+1)+1);
            const std::string id="A"+std::to_string(next);
            const std::string base=nx_kv_dir(S)+"/"+id;
            std::string preview;
            for(const auto& t: ke.context_turns()) if(t.start>=a && t.start<b && !t.preview.empty()){ preview=t.preview; break; }
            const size_t kb_tok=ke.kv_bytes_per_token();
            if(!ke.context_archive(a,b,base+".nxctx",&err)){ kerr(err); return; }
            std::string label=jp_str(r.body,"label",""); if(label.empty()) label=preview.substr(0,60);
            const std::string meta="{\"id\":"+jstr(id)+",\"label\":"+jstr(label)+",\"model\":"+jstr(ke.model_info().name)+
                ",\"file\":"+jstr(nx_model_file_tag(S.model_file[k2?1:0]))+
                ",\"tokens\":"+std::to_string(b-a)+",\"mb\":"+std::to_string((double)kb_tok*(b-a)/1048576.0)+
                ",\"created\":"+jstr(now_iso())+",\"preview\":"+jstr(preview)+",\"msgs\":"+jstr(jp_str(r.body,"msgs",""))+"}";
            write_file(base+".json",meta);
            std::string o=turns_json(); o.pop_back();
            resp_json(fd,o+",\"item\":"+meta+"}"); close(fd); return;
        }
        kerr("unknown memory request"); return;
    }

    if(r.method=="POST"&&r.path=="/api/chat"){
        fprintf(stderr,"[chat] POST /api/chat - body %zu bytes\n", r.body.size());
        fflush(stderr);
        // Route to engine2 if request contains "model":"2" and engine2 is loaded
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& eng      = use2 ? S.engine2  : S.engine;
        std::mutex&    eng_mu   = use2 ? S.gen_mutex2 : S.gen_mutex;
        std::atomic<bool>& sflag = use2 ? S.stop_flag2 : S.stop_flag;

        // The reply is generated independently of this connection and buffered in a GenJob (same as /api/continue), so a dropped browser
        // connection never loses or aborts it: the page re-attaches with /api/continue/resume. A retry of the same turn_id just re-attaches.
        GenJob& J = S.cjob[use2?1:0];
        const std::string chat_tid = jp_str(r.body,"turn_id","");
        bool job_claimed = false;
        {
            std::lock_guard<std::mutex> l(J.m);
            if(!chat_tid.empty() && J.tid==chat_tid){
                fprintf(stderr,"[chat] RETRY of turn %s - re-attaching to the existing reply\n",chat_tid.c_str()); fflush(stderr);
            } else if(!J.running){
                J.evs.clear(); J.done=false; J.running=true; J.tid=chat_tid; job_claimed=true;   // claim now so a retry sees it even while we wait for the engine
            }
        }
        if(!job_claimed){
            bool same; { std::lock_guard<std::mutex> l(J.m); same = !chat_tid.empty() && J.tid==chat_tid; }
            if(same){ nx_job_replay(fd,J,0,chat_tid); close(fd); return; }
        }
        // Whatever happens below (exception included), a job we own must end up 'done', or later replies would wait on it forever.
        std::shared_ptr<void> job_guard(nullptr, [&J, chat_tid](void*){
            { std::lock_guard<std::mutex> l(J.m); if(J.running && J.tid==chat_tid){ J.running=false; J.done=true; } }
            J.cv.notify_all();
        });
        sock_sndtimeo(fd, 3000);     // blocking sends with a bound: a wedged client is treated as disconnected, generation carries on
        send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"+nx_acao()+
                    "X-Accel-Buffering: no\r\nConnection: keep-alive\r\n\r\n");
        auto messages=jp_messages(r.body);
        auto image_payloads=jp_images(r.body);
        fprintf(stderr, "[chat%s] Parsed %zu messages, images=%zu:", use2?"2":"", messages.size(), image_payloads.size());
        for(const auto& m:messages) fprintf(stderr," [%s:%zu]",m.role.c_str(),m.content.size());
        fprintf(stderr,"\n"); fflush(stderr);
        nexus::GenerateConfig cfg;
        float req_temp=(float)jp_num(r.body,"temperature",-1.f);
        cfg.temperature    = req_temp>=0.f ? req_temp : S.temperature;
        cfg.max_tokens     = S.max_tokens;
        cfg.top_p              = (float)jp_num(r.body,"top_p",        S.top_p);
        cfg.top_k              = (int)  jp_num(r.body,"top_k",        S.top_k);
        cfg.min_p              = (float)jp_num(r.body,"min_p",        S.min_p);
        cfg.tfs_z              = (float)jp_num(r.body,"tfs_z",        S.tfs_z);
        cfg.repeat_penalty     = (float)jp_num(r.body,"repeat_penalty",S.repeat_penalty);
        cfg.repeat_last_n      = (int)  jp_num(r.body,"repeat_last_n",S.repeat_last_n);
        cfg.frequency_penalty  = (float)jp_num(r.body,"frequency_penalty",S.frequency_penalty);
        cfg.presence_penalty   = (float)jp_num(r.body,"presence_penalty",S.presence_penalty);
        cfg.mtp_enabled        = (jp_num(r.body,"mtp_enabled",   S.mtp_enabled?1:0) != 0.0);
        cfg.mtp_n_predict      = (int)  jp_num(r.body,"mtp_n_predict", S.mtp_n_predict);
        cfg.mtp_gpu            = (jp_num(r.body,"mtp_gpu",       S.mtp_gpu?1:0) != 0.0);
        cfg.n_gpu_layers       = use2 ? S.engine2_n_gpu_layers : S.n_gpu_layers;
        cfg.stop_strings   = eng.stop_strings();
        cfg.stop_strings.push_back("\nHuman:");
        cfg.stop_strings.push_back("\nUser:");
        cfg.stop_strings.push_back("\nAssistant:");
        std::string prompt;
        const int think_req = jp_think(r.body);   // per-message thinking switch (-1 = default)

        // Persistent conversation: conv_id lets us skip re-prefilling old tokens
        std::string conv_id   = jp_str(r.body, "conv_id", "");
        std::string new_tail;
        bool        is_continuation = false;

        if (!conv_id.empty()) {
            std::lock_guard<std::mutex> cl(S.sessions_mu);
            auto it = S.conversations.find(conv_id);
            if (it != S.conversations.end() && it->second.n_tokens > 0) {
                const std::string& prev = it->second.last_prompt;

                if (messages.size() == 1 && messages[0].role == "user") {
                    new_tail = eng.format_user_turn(messages[0].content, think_req);
                    prompt = prev + new_tail;
                    is_continuation = true;
                    fprintf(stderr, "[chat] conv %s: single-msg extend +%zu chars\n",
                            conv_id.c_str(), new_tail.size());
                } else {
                    prompt = eng.format_chat(messages, think_req);
                    if (!prev.empty() && prompt.size() > prev.size() &&
                        prompt.substr(0, prev.size()) == prev) {
                        new_tail = prompt.substr(prev.size());
                        is_continuation = true;
                        fprintf(stderr, "[chat] conv %s: prefix-extend +%zu chars\n",
                                conv_id.c_str(), new_tail.size());
                    }
                }
            }
        }

        if (prompt.empty()) {
            prompt = eng.format_chat(messages, think_req);
        }

        fprintf(stderr, "[chat%s] %zu messages - prompt (%zu chars): %.200s%s\n",
                use2?"2":"", messages.size(), prompt.size(), prompt.c_str(),
                prompt.size() > 200 ? "..." : "");
        fflush(stderr);
        // Generate a session ID for this request
        char sid_buf[32];
        snprintf(sid_buf,32,"%lx",(unsigned long)(
            std::hash<std::string>{}(prompt) ^ (unsigned long)time(nullptr)));
        std::string session_id = sid_buf;
        if (conv_id.empty()) conv_id = session_id;  // stable ID for KV cache reuse

        // Keep-alive while we wait for the engine / prefill (can be minutes): without bytes a router or the browser may drop the link.
        std::mutex wmu; bool fd_dead=false; long long dead_since=0; std::atomic<bool> any_ev{false};
        std::atomic<bool> ka_stop{false}; std::mutex ka_m; std::condition_variable ka_cv;
        std::thread ka_thr([&]{
            std::unique_lock<std::mutex> l(ka_m);
            const long long ka_t0 = nexus::memguard::now_ms();
            while(!ka_stop.load() && !any_ev.load()){
                if(ka_cv.wait_for(l,std::chrono::seconds(10),[&]{return ka_stop.load()||any_ev.load();})) break;
                {   // nothing else printed for a while: say so, so the log never just sits there.
                    // *** v30: reworded -- "no first word yet" read as flashy/uncertain ("like we
                    // don't know"), and this is now rare instead of routine (the long silent
                    // stretch during hybrid prefill's own-layer phase got its own real progress
                    // lines this round -- see dist_session.h). What's left mostly IS a real wait
                    // on the other computer (network lag, it applying a KV handoff, etc.), so say
                    // that plainly when this is a distributed session, instead of a generic line
                    // that reads the same whether something is actually wrong or everything's fine.
                    const long long now = nexus::memguard::now_ms();
                    if(now - nexus::memguard::progress_stamp().load() > 14000){
                        const long long s = (now - ka_t0) / 1000;
                        const bool dist_now = S.dist_sess && S.dist_sess->is_connected();
                        if (dist_now) fprintf(stderr,"[progress] still working: waiting on the other computer (%lldm %02llds so far)\n", s/60, s%60);
                        else          fprintf(stderr,"[progress] still working: no progress info yet (%lldm %02llds so far)\n", s/60, s%60);
                        fflush(stderr);
                    }
                }
                std::lock_guard<std::mutex> w(wmu);
                if(!fd_dead && ::send(fd,": ping\n\n",8,MSG_NOSIGNAL)!=8){ fd_dead=true; dead_since=nx_now_ms(); }
            }
        });
        std::shared_ptr<void> ka_guard(nullptr, [&](void*){
            { std::lock_guard<std::mutex> l(ka_m); ka_stop=true; }
            ka_cv.notify_all(); if(ka_thr.joinable()) ka_thr.join();
        });
        std::unique_lock<std::mutex> lg(eng_mu);
        if(!job_claimed){ std::lock_guard<std::mutex> l(J.m); J.evs.clear(); J.done=false; J.running=true; J.tid=chat_tid; }
        sflag=false;
        int ctx=0; int ctx_base=0;   // ctx_base = tokens already in the context (prompt + history); ctx = streamed text pieces this reply
        int live_ctx=-1, live_gen=-1; // REAL counts from the engine (tokens in memory, tokens generated) -- used whenever known

        // JSON-escape helper
        auto jesc=[](const std::string& s)->std::string{
            std::string e;
            for(unsigned char c:s){
                if(c=='"')  e+="\\\"";
                else if(c=='\\') e+="\\\\";
                else if(c=='\n') e+="\\n";
                else if(c=='\r') e+="\\r";
                else if(c<0x20){char b[8];snprintf(b,8,"\\u%04x",c);e+=b;}
                else e+=c;
            }
            return e;
        };

        // Every event goes into the job buffer first (so it can be replayed), then best-effort to this socket. A dead socket never stops generation.
        auto emit=[&](const std::string& ev){
            { std::lock_guard<std::mutex> l(J.m); J.evs.push_back(ev); }
            J.cv.notify_all();
            any_ev=true; ka_cv.notify_all();
            std::lock_guard<std::mutex> w(wmu);
            if(fd_dead) return;
            size_t sent=0;
            while(sent<ev.size()){
                ssize_t n=::send(fd,ev.c_str()+sent,ev.size()-sent,MSG_NOSIGNAL);
                if(n<=0){
                    fd_dead=true; dead_since=nx_now_ms();
                    ::shutdown(fd,SHUT_RDWR);       // make the page notice at once and re-attach
                    fprintf(stderr,"[chat] browser connection lost - generation continues, reply is buffered for reconnect\n"); fflush(stderr);
                    return;
                }
                sent+=(size_t)n;
            }
        };
        auto _send=[&](const std::string& d)->bool{ emit(d); return true; };
        auto send_tok=[&](const std::string& tok)->bool{
            if(sflag)return false;
            ++ctx;
            if((ctx&7)==0 && !nexus::memguard::still_ok("the next words")){      // memory safety buffer: stop cleanly, keep the conversation
                _send("data: {\"notice\":\""+jesc(nexus::memguard::reason())+"\"}\n\n");
                return false;
            }
            fprintf(stderr,"[sse%s] chunk=%d chars=%zu text=\"%s\"\n",
                    use2?"2":"", ctx, tok.size(), nx_log_visible_text(tok).c_str());
            fflush(stderr);
            const int c_now = live_ctx >= 0 ? live_ctx : ctx_base+ctx, g_now = live_gen >= 0 ? live_gen : ctx;
            _send("data: {\"token\":\""+jesc(tok)+"\",\"context_tokens\":"+std::to_string(c_now)+",\"gen\":"+std::to_string(g_now)+"}\n\n");
            return true;                      // a reply is never abandoned because nobody is watching: it finishes and waits in the buffer
        };

        auto begin_session_with_images = [&](const std::string& full_prompt)->nexus::GenerateSession{
            if(image_payloads.empty()) return eng.begin_session(full_prompt, cfg, {});
            if(!eng.has_vision()){
                fprintf(stderr,"[vision-mixed] %zu image(s) attached but no vision encoder loaded\n", image_payloads.size());
                _send("data: {\"token\":\"[vision encoder not loaded for attached image]\"}\n\n");
                return eng.begin_session(full_prompt, cfg, {});
            }
            std::string verr;
            size_t vt = 0;
            eng.reset_context();
            if(!nx_extend_context_with_images_in_place(eng, full_prompt, image_payloads, vt, verr)){
                fprintf(stderr,"[vision-mixed] fresh prompt placement failed: %s\n", verr.c_str());
                _send("data: {\"token\":\"[vision mixed prefill failed; falling back to text-only]\"}\n\n");
                return eng.begin_session(full_prompt, cfg, {});
            }
            fprintf(stderr,"[vision-mixed] fresh prompt ready via warm session; vision_tokens=%zu ctx=%d\n", vt, eng.context_size());
            return eng.begin_session_warm(cfg, {});
        };

        try {
            // Create session - NO pause_tags. Python detects tags and stops via /api/stop.
            // Continuation: extend with new tail then warm-start. Fresh: full prefill.
            // Tell client the conv_id so next message can continue from warm cache
            { std::string ce = "data: {\"conv_id\":\"" + jesc(conv_id) + "\"}\n\n";   // not buffered: resume offsets count token events only
              std::lock_guard<std::mutex> w(wmu); if(!fd_dead && ::send(fd,ce.c_str(),ce.size(),MSG_NOSIGNAL)<=0){ fd_dead=true; dead_since=nx_now_ms(); } }

            // Use distributed inference if worker is connected, otherwise local
            if (!use2 && S.dist_sess && S.dist_sess->is_connected() && image_payloads.empty()) {
                fprintf(stderr, "[dist] Distributed generate: %zu chars\n", prompt.size());
                fflush(stderr);
                // the whole prompt is sent each time, so the context after prefill is the prompt's token count
                try { ctx_base = (int)eng.tokenize(prompt, true).size(); } catch(...) { ctx_base = 0; }
                S.dist_sess->generate_stream(prompt,
                    [&](const std::string& t, int32_t)->bool{ return send_tok(t); },
                    cfg);
                // DistSession manages its own state -- no local sync needed
                emit("data: {\"final\":true,\"context_tokens\":"+std::to_string(S.dist_sess->context_size())+",\"gen\":"+std::to_string(ctx)+"}\n\n");
            } else {
                // *** v28 prefill speed gauge (single-computer case): see this script's header
                // comment. Reads already-public Engine counters only -- range_prefill_layers_done()
                // resets per prefill_layer_range() call and counts up to the active layer range's
                // size (transformer.h's pr_layers_done_), which for a local, non-distributed engine
                // IS the whole model, so model_info().n_layers is the right total here.
                size_t pr_prompt_tok_est = 0;
                try { pr_prompt_tok_est = is_continuation ? eng.tokenize(new_tail, false).size() : eng.tokenize(prompt, true).size(); } catch(...) {}
                // Progress also goes to the client as {"progress":{...}} events while the prompt is read. Not buffered for
                // resume (like conv_id): clients don't count them, and a resumed stream doesn't need old progress.
                std::unique_ptr<NxPrefillReporter> pr_guard(new NxPrefillReporter(eng, pr_prompt_tok_est, [&](const std::string& js){
                    const std::string ev = "data: {\"progress\":" + js + "}\n\n";
                    std::lock_guard<std::mutex> w(wmu);
                    if(!fd_dead && ::send(fd, ev.c_str(), ev.size(), MSG_NOSIGNAL) <= 0){ fd_dead=true; dead_since=nx_now_ms(); }
                }));

                nexus::GenerateSession sess = is_continuation
                    ? [&]()->nexus::GenerateSession{
                        nx_make_room(eng, new_tail, (int)image_payloads.size(), cfg.max_tokens);
                        if(!image_payloads.empty()){
                            size_t vt = 0;
                            std::string verr;
                            if(!nx_extend_context_with_images_in_place(eng, new_tail, image_payloads, vt, verr)){
                                fprintf(stderr,"[vision-place] continuation image placement failed: %s\n", verr.c_str());
                                eng.extend_context(new_tail, false);
                            }
                        } else {
                            eng.extend_context(new_tail, false);
                        }
                        return eng.begin_session_warm(cfg);
                      }()
                    : begin_session_with_images(prompt);
                if (!is_continuation) nx_pin_banner(eng, messages);
                pr_guard.reset();   // prefill is over the moment sess exists -- stop the reporter before decode starts
                ctx_base = sess.pos() + 1;   // the prompt is already prefilled here; +1 for the last prompt token the first step feeds
                sess.run([&](const std::string& t,int32_t)->bool{
                    live_ctx = sess.pos() + 1; live_gen = sess.generated() + 1;   // the token being streamed is not committed yet
                    return send_tok(t); });
                // Sync engine session_tokens with what was generated
                eng.sync_session_tokens(sess.tokens(), sess.pos());
                { const std::string fn = nx_full_notice(eng); if (!fn.empty()) emit("data: {\"notice\":\""+jesc(fn)+"\"}\n\n"); }
                emit("data: {\"final\":true,\"context_tokens\":"+std::to_string(eng.context_size())+",\"gen\":"+std::to_string(sess.generated())+"}\n\n");
            }

            // Update conversation state for KV cache reuse next turn
            if (!conv_id.empty()) {
                std::lock_guard<std::mutex> cl(S.sessions_mu);
                S.conversations[conv_id].last_prompt = prompt;
                S.conversations[conv_id].n_tokens    = eng.context_size();
                S.conversations[conv_id].model       = use2 ? 2 : 1;
            }
            nx_forget_conversations(S, use2 ? 2 : 1, conv_id);

        } catch(const std::exception& ex){
            fprintf(stderr,"[chat] session EXCEPTION: %s\n",ex.what()); fflush(stderr);
            if(nexus::memguard::tripped()) emit("data: {\"notice\":\""+jesc(nexus::memguard::reason())+"\"}\n\n");
        } catch(...){
            fprintf(stderr,"[chat] session UNKNOWN EXCEPTION\n"); fflush(stderr);
        }
        nexus::memguard::clear();
        emit(NX_SSE_DONE);
        { std::lock_guard<std::mutex> l(J.m); J.running=false; J.done=true; }
        J.cv.notify_all();
        { std::lock_guard<std::mutex> l(ka_m); ka_stop=true; }
        ka_cv.notify_all(); if(ka_thr.joinable()) ka_thr.join();
        close(fd);return;
    }
    // -- Extend KV cache with a new formatted user turn --------------------------
    // POST /api/session/extend_turn  body: {"text":"user message","conv_id":"...","model":"1"}
    // Formats the message with chat template and appends to KV cache, no generation.
    if(r.method=="POST" && r.path=="/api/session/extend_turn"){
        bool use2e = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& enge = use2e ? S.engine2 : S.engine;
        std::mutex& mue = use2e ? S.gen_mutex2 : S.gen_mutex;
        std::string user_text = jp_str(r.body,"text","");
        auto image_payloads = jp_images(r.body);
        if(user_text.empty()){ resp_json(fd,"{\"ok\":false,\"error\":\"empty text\"}"); close(fd); return; }
        // Headers go out NOW, then one space of JSON whitespace every 10 s while we
        // wait for the engine lock / prefill. Prefill can be minutes to hours; a
        // silent connection can be dropped by the browser, OS or a router.
        send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-cache\r\n"
                    +nx_acao()+"Connection: close\r\n\r\n");
        std::string tid = jp_str(r.body,"turn_id","");
        std::atomic<bool> ka_stop{false}, ka_failed{false};
        std::mutex ka_m; std::condition_variable ka_cv;
        std::thread ka_thr([&]{
            std::unique_lock<std::mutex> l(ka_m);
            while(!ka_stop.load()){
                if(ka_cv.wait_for(l,std::chrono::seconds(10),[&]{return ka_stop.load();})) break;
                if(::send(fd," ",1,MSG_NOSIGNAL)<=0){ ka_failed=true; break; }
            }
        });
        auto ext_reply=[&](const std::string& body){
            if(!tid.empty()){   // remember the outcome so a retry of this turn gets the same answer
                { std::lock_guard<std::mutex> l(S.turn_m); S.turn_reply=body; S.turn_done=true; }
                S.turn_cv.notify_all();
            }
            { std::lock_guard<std::mutex> l(ka_m); ka_stop=true; }
            ka_cv.notify_all(); if(ka_thr.joinable()) ka_thr.join();
            size_t sent=0; bool ok=!ka_failed.load();
            while(ok && sent<body.size()){
                ssize_t n=::send(fd,body.c_str()+sent,body.size()-sent,MSG_NOSIGNAL);
                if(n<=0){ ok=false; break; } sent+=(size_t)n;
            }
            fprintf(stderr,"[extend_turn] reply %s%s\n", ok?"DELIVERED":"NOT DELIVERED - browser connection was already gone",
                    (!ok && ka_failed.load())?" (keepalive write failed)":"");
            fflush(stderr);
        };
        if(!tid.empty()){
            bool is_retry=false;
            { std::lock_guard<std::mutex> l(S.turn_m);
              if(S.turn_id==tid) is_retry=true;
              else { S.turn_id=tid; S.turn_done=false; S.turn_reply.clear(); } }
            if(is_retry){
                fprintf(stderr,"[extend_turn] RETRY of turn %s - not re-adding the prompt, waiting for the original to finish\n",tid.c_str()); fflush(stderr);
                std::string rep;
                { std::unique_lock<std::mutex> l(S.turn_m);
                  S.turn_cv.wait(l,[&]{return S.turn_done;}); rep=S.turn_reply; }
                { std::lock_guard<std::mutex> l(ka_m); ka_stop=true; }
                ka_cv.notify_all(); if(ka_thr.joinable()) ka_thr.join();
                size_t sent=0; while(sent<rep.size()){ ssize_t n=::send(fd,rep.c_str()+sent,rep.size()-sent,MSG_NOSIGNAL); if(n<=0)break; sent+=(size_t)n; }
                close(fd); return;
            }
        }
        std::lock_guard<std::mutex> lge(mue);
        // Format only the next user turn continuation, not a standalone full prompt
        std::string turn = enge.format_user_turn(user_text, jp_think(r.body));
        int ctx_before = (!use2e && S.dist_sess && S.dist_sess->is_connected())
            ? S.dist_sess->context_size() : enge.context_size();
        fprintf(stderr,"[extend_turn] +%zu chars, images=%zu (ctx before=%d)%s\n",
                (size_t)turn.size(), image_payloads.size(), ctx_before,
                (!use2e && S.dist_sess && S.dist_sess->is_connected()) ? " [dist]" : "");
        fflush(stderr);

        int ctx_after = ctx_before;
        size_t vision_tokens_injected = 0;
        if (!use2e && S.dist_sess && S.dist_sess->is_connected()) {
            if (!image_payloads.empty()) {
                fprintf(stderr,"[extend_turn] refusing %zu image(s): distributed extend cannot inject vision embeddings yet\n",
                        image_payloads.size());
                ext_reply("{\"ok\":false,\"error\":\"image attachments are not supported in distributed warm-extend yet; run local or load image in a fresh local chat\"}");
                close(fd); return;
            }
            if (!S.dist_sess->extend_context(turn, false)) {
                ext_reply("{\"ok\":false,\"error\":\"distributed extend failed\"}");
                close(fd); return;
            }
            ctx_after = S.dist_sess->context_size();
        } else {
            const int dropped = nx_make_room(enge, turn, (int)image_payloads.size(), S.max_tokens);
            if (dropped > 0) { fprintf(stderr,"[extend_turn] made room: dropped the oldest %d token(s)\n", dropped); fflush(stderr); }
            if (!image_payloads.empty()) {
                fprintf(stderr,"[extend_turn][vision] %zu image attachment(s) received; injecting at [Image: ...] marker inside user turn\n",
                        image_payloads.size());
                std::string verr;
                if (!nx_extend_context_with_images_in_place(enge, turn, image_payloads, vision_tokens_injected, verr)) {
                    fprintf(stderr,"[extend_turn][vision] FAILED: %s\n", verr.c_str());
                    ext_reply("{\"ok\":false,\"error\":\"vision image insertion failed; see server log\"}");
                    close(fd); return;
                }
            } else {
                { size_t nt = 0; try { nt = enge.tokenize(turn, false).size(); } catch (...) {}
                  NxPrefillReporter rep(enge, nt);
                  enge.extend_context(turn, false); }
            }
            ctx_after = enge.context_size();
        }

        fprintf(stderr,"[extend_turn] done ctx=%d vision_tokens=%zu\n",ctx_after, vision_tokens_injected); fflush(stderr);
        ext_reply("{\"ok\":true,\"context_tokens\":" + std::to_string(ctx_after) + ",\"vision_tokens\":" + std::to_string(vision_tokens_injected) + "}");
        close(fd); return;
    }
    
    // -- System / tool-result injection ----------------------------------------
    // POST /api/inject  {"content":"...","conv_id":"...","model":"1"}
    if(r.method=="POST" && r.path=="/api/inject"){
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& enge = use2 ? S.engine2 : S.engine;
        std::mutex&    mue  = use2 ? S.gen_mutex2 : S.gen_mutex;

        std::string content = jp_str(r.body, "content", "");
        std::string conv_id = jp_str(r.body, "conv_id", "");

        if(content.empty()){
            resp_json(fd,"{\"ok\":false,\"error\":\"content required\"}");
            close(fd); return;
        }

        std::lock_guard<std::mutex> lge(mue);

        // -- Format the system block + assistant turn-opener -------------------
        // Each chat template has different system-role token sequences.
        // We also append the assistant opener so /api/continue immediately
        // generates in assistant role; without it the model might re-emit a role
        // header or produce garbage on the first token.
        std::string chat_fmt = enge.model_info().chat_format;
        std::string formatted;

        if(chat_fmt == "Llama-3"){
            // Llama-3 natively supports mid-conversation system turns.
            formatted = "<|start_header_id|>system<|end_header_id|>\n\n"
                      + content + "<|eot_id|>\n"
                      + "<|start_header_id|>assistant<|end_header_id|>\n\n";

        } else if(chat_fmt.substr(0,6) == "ChatML"
               || chat_fmt == "DeepSeek"
               || chat_fmt == "DeepSeek-R1"
               || chat_fmt == "DeepSeek-Coder"){
            // ChatML and DeepSeek variants all share the same special tokens.
            formatted = "<|im_start|>system\n"
                      + content + "<|im_end|>\n"
                      + "<|im_start|>assistant\n";

        } else if(chat_fmt == "Phi-3"){
            formatted = "<|system|>\n" + content + "<|end|>\n"
                      + "<|assistant|>\n";

        } else if(chat_fmt == "Zephyr"){
            formatted = "<|system|>\n" + content + "</s>\n"
                      + "<|assistant|>\n";

        } else if(chat_fmt == "Gemma"){
            // Gemma has no mid-conversation system role - inject as a special
            // user turn with an unmistakable structural marker.
            formatted = "<start_of_turn>user\n"
                        "[TOOL RESULT]\n" + content + "\n[END TOOL RESULT]"
                        "<end_of_turn>\n"
                        "<start_of_turn>model\n";

        } else if(chat_fmt == "Mistral" || chat_fmt == "Mistral V3"){
            // Mistral also has no mid-conv system role.  Use [TOOL_RESULT] inside
            // an existing [INST] context so the model at least sees it as data,
            // not a hallucinated instruction.
            formatted = " [TOOL_RESULT]\n" + content + "\n[/TOOL_RESULT][/INST] ";

        } else if(chat_fmt == "Alpaca"){
            formatted = "\n### Input:\n" + content + "\n\n### Response:\n";
            
        } else if(chat_fmt == "Granite"){
            formatted = "<|start_of_role|>system<|end_of_role|>"
                      + content + "<|end_of_text|>\n"
                      + "<|start_of_role|>assistant<|end_of_role|>";

        } else {
            // Generic fallback: format as a clearly-labelled user turn and hope
            // the model's fine-tuning treats bracketed sections appropriately.
            formatted = enge.format_user_turn(
                "<<SYSTEM RESULT>>\n" + content + "\n<</SYSTEM RESULT>>");
        }

        // -- Prefill the new tokens into KV cache ------------------------------
        int inject_ctx = 0;
        if (!use2 && S.dist_sess && S.dist_sess->is_connected()) {
            if (!S.dist_sess->extend_context(formatted, /*add_bos=*/false)) {
                resp_json(fd,"{\"ok\":false,\"error\":\"distributed inject failed\"}");
                close(fd); return;
            }
            inject_ctx = S.dist_sess->context_size();
        } else {
            enge.extend_context(formatted, /*add_bos=*/false);
            inject_ctx = enge.context_size();
        }

        // -- Keep conversations table in sync ----------------------------------
        // We deliberately do NOT update last_prompt here - that string is used
        // by /api/chat's prefix-extend optimisation and must stay as the
        // formatted chat prompt.  We only update the token count so subsequent
        // calls know how full the context is.
        if(!conv_id.empty()){
            std::lock_guard<std::mutex> cl(S.sessions_mu);
            auto it = S.conversations.find(conv_id);
            if(it != S.conversations.end())
                it->second.n_tokens = inject_ctx;
        }

        fprintf(stderr,"[inject] +%zu chars (%s template) - ctx=%d%s\n",
                content.size(), chat_fmt.c_str(), inject_ctx,
                (!use2 && S.dist_sess && S.dist_sess->is_connected()) ? " [dist]" : "");
        fflush(stderr);

        resp_json(fd, "{\"ok\":true,\"context_tokens\":"
                    + std::to_string(inject_ctx) + "}");
        close(fd); return;
    }    

    // -- Text embeddings --------------------------------------------------------
    // POST /api/embed  {"text":"...","normalize":true,"model":"1"}
    // Returns {"embedding":[...], "dim":N} - use for RAG/semantic search
    if(r.method=="POST"&&r.path=="/api/embed"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string text=jp_str(r.body,"text","");
        bool norm = jp_str(r.body,"normalize","true")!="false";
        if(text.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"text required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        auto emb=e.embed_text(text,norm);
        std::string j="{\"ok\":true,\"dim\":"+std::to_string(emb.size())+",\"embedding\":[";
        for(size_t i=0;i<emb.size();++i){if(i)j+=","; char buf[32]; snprintf(buf,32,"%.6f",emb[i]); j+=buf;}
        j+="]}"; resp_json(fd,j); close(fd); return;
    }
    // POST /api/lora/load  {"path":"/path/to/adapter.gguf","scale":1.0,"model":"1"}
    if(r.method=="POST"&&r.path=="/api/lora/load"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string path=jp_str(r.body,"path","");
        float scale=(float)jp_num(r.body,"scale",1.0);
        if(path.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"path required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.lora_load(path,scale);
        resp_json(fd,ok?("{\"ok\":true,\"count\":"+std::to_string(e.lora_count())+"}"):
                       "{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");
        close(fd);return;
    }
    // POST /api/lora/unload  {"model":"1"}
    if(r.method=="POST"&&r.path=="/api/lora/unload"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::lock_guard<std::mutex> lg(mu);
        e.lora_unload_all(); resp_json(fd,"{\"ok\":true}"); close(fd);return;
    }

    // -- Multimodal -------------------------------------------------------------
    // POST /api/vision/load  {"path":"/path/to/mmproj.gguf","model":"1"}
    if(r.method=="POST"&&r.path=="/api/vision/load"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string path=jp_str(r.body,"path","");
        if(path.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"path required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.load_vision_encoder(path,[](const std::string&,float){});
        resp_json(fd,ok?"{\"ok\":true}":"{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");
        close(fd);return;
    }
    // POST /api/audio/load  {"path":"/path/to/whisper.gguf","model":"1"}
    if(r.method=="POST"&&r.path=="/api/audio/load"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string path=jp_str(r.body,"path","");
        if(path.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"path required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.load_audio_encoder(path,[](const std::string&,float){});
        resp_json(fd,ok?"{\"ok\":true}":"{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");
        close(fd);return;
    }
    // POST /api/audio/encode_wav  {"path":"/audio.wav","model":"1"}
    // Returns base64-encoded float32 embeddings
    if(r.method=="POST"&&r.path=="/api/audio/encode_wav"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine;
        std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string path=jp_str(r.body,"path","");
        if(!e.has_audio()){resp_json(fd,"{\"ok\":false,\"error\":\"no audio encoder loaded\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        auto emb=e.encode_audio_wav(path);
        resp_json(fd,"{\"ok\":true,\"n_frames\":"+std::to_string(emb.size()/512)+
                     ",\"n_floats\":"+std::to_string(emb.size())+"}");
        close(fd);return;
    }

    // -- Checkpoints and start states -------------------------------------------
    if(r.method=="POST"&&r.path=="/api/checkpoint/save"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string name=jp_str(r.body,"name","");
        std::lock_guard<std::mutex> lg(mu);
        std::string safe=name;
        for(auto& c:safe) if(!isalnum((unsigned char)c)&&c!='-'&&c!='_'&&c!=' ') c='_';
        if(safe.empty()) safe="checkpoint";
        std::string ts=std::to_string((long long)time(nullptr));
        std::string cbase=S.storage.base+"/checkpoints/"+safe+"_"+ts;
        std::string kvc=cbase+".kvc", cmeta=cbase+".json";
        int n_tok=e.context_size();
        bool csaved=e.save_session(kvc);
        if(csaved){
            FILE* mf=fopen(cmeta.c_str(),"w");
            if(mf){
                std::string mj="{\"name\":"+jstr(name)+",\"path\":"+jstr(kvc)+",\"n_tokens\":"+std::to_string(n_tok)+",\"created\":\""+ts+"\"}";
                fwrite(mj.c_str(),1,mj.size(),mf); fclose(mf);
            }
        }
        resp_json(fd,csaved?("{\"ok\":true,\"path\":"+jstr(kvc)+"}"):"{\"ok\":false,\"error\":\"Could not save. Check that the model is still running and that the folder has free space.\"}");
        close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/checkpoint/restore"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string cpath=jp_str(r.body,"path","");
        if(cpath.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"path required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.load_session(cpath);
        resp_json(fd,ok?("{\"ok\":true,\"context_tokens\":"+std::to_string(e.context_size())+"}"):"{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");
        close(fd);return;
    }
    if(r.method=="GET"&&r.path=="/api/checkpoint/list"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        std::string cdir=S.storage.base+"/checkpoints";
        std::string j="["; bool cfirst=true;
        for(auto& cf:list_dir(cdir)){
            if(cf.size()<5||cf.substr(cf.size()-5)!=".json") continue;
            std::string cp=cdir+"/"+cf;
            FILE* mf=fopen(cp.c_str(),"r"); if(!mf) continue;
            fseek(mf,0,SEEK_END);long csz=ftell(mf);rewind(mf);
            if(csz<=0||csz>4096){fclose(mf);continue;}
            std::string jbuf(csz,'\0');fread(&jbuf[0],1,csz,mf);fclose(mf);
            if(!cfirst) j+=","; cfirst=false; j+=jbuf;
        }
        j+="]"; resp_json(fd,j); close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/checkpoint/delete"){
        std::string dpath=jp_str(r.body,"path","");
        if(!dpath.empty()){
            remove(dpath.c_str());
            if(dpath.size()>4&&dpath.substr(dpath.size()-4)==".kvc"){
                std::string dm=dpath.substr(0,dpath.size()-4)+".json"; remove(dm.c_str());
            }
        }
        resp_json(fd,"{\"ok\":true}"); close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/start_state/save"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string name=jp_str(r.body,"name","state");
        std::string cid =jp_str(r.body,"conv_id","");
        std::string safe=name; for(auto& c:safe) if(!isalnum((unsigned char)c)&&c!='-'&&c!='_'&&c!=' ') c='_';
        if(safe.empty()) safe="state";
        std::string ts=std::to_string((long long)time(nullptr));
        std::string base=S.storage.base+"/states/"+safe+"_"+ts;
        std::string kvc=base+".kvc", meta=base+".json";
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.save_start_state(kvc);
        if(ok){
            // Look up the prompt from conversations if we have a conv_id
            std::string last_prompt;
            if(!cid.empty()){
                std::lock_guard<std::mutex> cl(S.sessions_mu);
                auto it=S.conversations.find(cid);
                if(it!=S.conversations.end()) last_prompt=it->second.last_prompt;
            }
            int n_tok=e.context_size();
            FILE* mf=fopen(meta.c_str(),"w");
            if(mf){
                std::string mj="{\"name\":"+jstr(name)+",\"path\":"+jstr(kvc)+",\"prompt\":"+jstr(last_prompt)+",\"n_tokens\":"+std::to_string(n_tok)+",\"created\":\""+ts+"\"}";
                fwrite(mj.c_str(),1,mj.size(),mf); fclose(mf);
            }
            fprintf(stderr,"[server] Warm state saved: %s (%d tok)\n",kvc.c_str(),n_tok); fflush(stderr);
        }
        resp_json(fd,ok?("{\"ok\":true,\"path\":"+jstr(kvc)+"}"):"{\"ok\":false,\"error\":\"Could not save. Check that the model is still running and that the folder has free space.\"}");
        close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/start_state/load"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        std::string path=jp_str(r.body,"path","");
        if(path.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"path required\"}");close(fd);return;}
        std::lock_guard<std::mutex> lg(mu);
        bool ok=e.load_start_state(path);
        if(!ok){resp_json(fd,"{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");close(fd);return;}
        // Try to read sidecar .json for the prompt (so continuation knows the prefix)
        std::string last_prompt;
        if(path.size()>4&&path.substr(path.size()-4)==".kvc"){
            std::string meta=path.substr(0,path.size()-4)+".json";
            FILE* mf=fopen(meta.c_str(),"r");
            if(mf){
                fseek(mf,0,SEEK_END);long msz=ftell(mf);rewind(mf);
                if(msz>0&&msz<65536){std::string mb(msz,'\0');fread(&mb[0],1,msz,mf);last_prompt=jp_str(mb,"prompt","");}
                fclose(mf);
            }
        }
        // Register warm KV so next /api/chat extends instead of re-prefilling
        char wid[40]; snprintf(wid,sizeof(wid),"ws_%lx",(unsigned long)time(nullptr));
        std::string wconv=wid;
        {
            std::lock_guard<std::mutex> cl(S.sessions_mu);
            S.conversations[wconv].last_prompt=last_prompt;
            S.conversations[wconv].n_tokens=e.context_size();
            S.conversations[wconv].model=u2?2:1;
        }
        nx_forget_conversations(S, u2?2:1, wconv);
        fprintf(stderr,"[server] Warm state loaded: %s (%d tok) conv=%s\n",path.c_str(),e.context_size(),wconv.c_str()); fflush(stderr);
        resp_json(fd,"{\"ok\":true,\"conv_id\":\""+wconv+"\",\"context_size\":"+std::to_string(e.context_size())+"}");
        close(fd);return;
    }
    if(r.method=="GET"&&r.path=="/api/start_state/list"){
        std::string sdir=S.storage.base+"/states";
        std::string j="["; bool sfirst=true;
        for(auto& sf:list_dir(sdir)){
            if(sf.size()<5||sf.substr(sf.size()-5)!=".json") continue;
            std::string sp=sdir+"/"+sf;
            FILE* mf=fopen(sp.c_str(),"r"); if(!mf) continue;
            fseek(mf,0,SEEK_END);long ssz=ftell(mf);rewind(mf);
            if(ssz<=0||ssz>65536){fclose(mf);continue;}
            std::string jbuf(ssz,'\0');fread(&jbuf[0],1,ssz,mf);fclose(mf);
            if(!sfirst) j+=","; sfirst=false; j+=jbuf;
        }
        j+="]"; resp_json(fd,j); close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/start_state/delete"){
        std::string spath=jp_str(r.body,"path","");
        if(!spath.empty()){
            remove(spath.c_str());
            if(spath.size()>4&&spath.substr(spath.size()-4)==".kvc"){
                std::string sm=spath.substr(0,spath.size()-4)+".json"; remove(sm.c_str());
            }
        }
        resp_json(fd,"{\"ok\":true}"); close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/slot/select"){
        bool u2=S.has_engine2&&jp_str(r.body,"model","1")=="2";
        nexus::Engine& e=u2?S.engine2:S.engine; std::mutex& mu=u2?S.gen_mutex2:S.gen_mutex;
        int slot=(int)jp_num(r.body,"slot",0);
        std::lock_guard<std::mutex> lg(mu);
        e.context_slot_select(slot);
        resp_json(fd,"{\"ok\":true,\"slot\":"+std::to_string(slot)+"}"); close(fd);return;
    }
    // -- Continue from warm KV cache --------------------------------------------
    // POST /api/continue  body: {"conv_id":"...", "model":"1"}
    // No prefill - generates from wherever extend_context left off.
    if(r.method=="POST" && r.path=="/api/continue/resume"){
        bool use2r = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        GenJob& Jr = S.cjob[use2r?1:0];
        long from = atol(jp_str(r.body,"from","0").c_str()); if(from<0) from=0;
        fprintf(stderr,"[continue/resume] browser re-attached, replaying from event %ld\n",from); fflush(stderr);
        nx_job_replay(fd,Jr,(size_t)from,jp_str(r.body,"turn_id","")); close(fd); return;
    }
    if(r.method=="POST" && r.path=="/api/continue"){
        fprintf(stderr,"[continue] conv=%s\n",jp_str(r.body,"conv_id","(none)").c_str()); fflush(stderr);
        bool use2c = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& engc = use2c ? S.engine2 : S.engine;
        std::mutex& muc = use2c ? S.gen_mutex2 : S.gen_mutex;
        std::atomic<bool>& sfc = use2c ? S.stop_flag2 : S.stop_flag;
        GenJob& J = S.cjob[use2c?1:0];
        {   // A reply is already being generated (e.g. the page retried): attach instead of starting a second one.
            bool busy; { std::lock_guard<std::mutex> l(J.m); busy=J.running; }
            if(busy){
                fprintf(stderr,"[continue] a reply is already running - attaching to it from the start\n"); fflush(stderr);
                nx_job_replay(fd,J,0); close(fd); return;
            }
        }
        sock_sndtimeo(fd, 3000);
        send_all(fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\n"+nx_acao()+
                    "X-Accel-Buffering: no\r\nConnection: keep-alive\r\n\r\n");
        nexus::GenerateConfig cfgc;          // same sampling fields as /api/chat; server defaults when not given
        { const float rt=(float)jp_num(r.body,"temperature",-1.f); cfgc.temperature = rt>=0.f ? rt : S.temperature; }
        cfgc.max_tokens        = S.max_tokens;
        cfgc.top_p             = (float)jp_num(r.body,"top_p",        S.top_p);
        cfgc.top_k             = (int)  jp_num(r.body,"top_k",        S.top_k);
        cfgc.min_p             = (float)jp_num(r.body,"min_p",        S.min_p);
        cfgc.tfs_z             = (float)jp_num(r.body,"tfs_z",        S.tfs_z);
        cfgc.repeat_penalty    = (float)jp_num(r.body,"repeat_penalty",S.repeat_penalty);
        cfgc.repeat_last_n     = (int)  jp_num(r.body,"repeat_last_n",S.repeat_last_n);
        cfgc.frequency_penalty = (float)jp_num(r.body,"frequency_penalty",S.frequency_penalty);
        cfgc.presence_penalty  = (float)jp_num(r.body,"presence_penalty",S.presence_penalty);
        cfgc.mtp_enabled       = (jp_num(r.body,"mtp_enabled",   S.mtp_enabled?1:0) != 0.0);
        cfgc.mtp_n_predict     = (int)  jp_num(r.body,"mtp_n_predict", S.mtp_n_predict);
        cfgc.mtp_gpu           = (jp_num(r.body,"mtp_gpu",       S.mtp_gpu?1:0) != 0.0);
        cfgc.stop_strings   = engc.stop_strings();
        cfgc.stop_strings.push_back("\nHuman:");
        cfgc.stop_strings.push_back("\nUser:");
        cfgc.stop_strings.push_back("\nAssistant:");
        std::lock_guard<std::mutex> lgc(muc);
        sfc = false; int ctxc = 0; int ctxc_base = 0;
        int live_ctx = -1, live_gen = -1;   // REAL counts from the engine; ctxc only counts streamed text pieces
        { std::lock_guard<std::mutex> l(J.m); J.evs.clear(); J.done=false; J.running=true; J.tid=jp_str(r.body,"turn_id",""); }
        bool fd_dead=false;
        // Server-side speed log: the real generation rate, independent of the browser.
        const auto gen_t0 = std::chrono::steady_clock::now();
        auto win_t0 = gen_t0; int win_n = 0, total_n = 0, win_g0 = 0;
        auto emit=[&](const std::string& ev){
            if(ev != NX_SSE_DONE && ev.compare(0, 14, "data: {\"final\"") != 0){
                auto now = std::chrono::steady_clock::now();
                ++total_n;
                const int g = live_gen >= 0 ? live_gen : total_n;      // real tokens (MTP can commit several per piece)
                if(total_n == 1)
                    fprintf(stderr,"[gen] first token %.2f s after the continue request started\n",
                            std::chrono::duration<double>(now-gen_t0).count());
                if(++win_n == 16){
                    double dt = std::chrono::duration<double>(now-win_t0).count();
                    fprintf(stderr,"[gen] %d tokens | last %d in %.1f s = %.2f tok/s\n", g, g-win_g0, dt, dt>0 ? (g-win_g0)/dt : 0.0);
                    fflush(stderr); win_t0 = now; win_n = 0; win_g0 = g;
                }
            }
            { std::lock_guard<std::mutex> l(J.m); J.evs.push_back(ev); }
            J.cv.notify_all();
            if(fd_dead) return;
            size_t sent=0;
            while(sent<ev.size()){
                ssize_t n=::send(fd,ev.c_str()+sent,ev.size()-sent,MSG_NOSIGNAL);
                if(n<=0){ fd_dead=true; ::shutdown(fd,SHUT_RDWR); fprintf(stderr,"[continue] browser connection lost at event %d - generation continues, reply is buffered for reconnect\n",ctxc); fflush(stderr); break; }
                sent+=(size_t)n;
            }
        };
        auto jesc2=[](const std::string& s)->std::string{
            std::string e; for(unsigned char c:s){
                if(c=='"')e+="\\\""; else if(c=='\\')e+="\\\\";
                else if(c=='\n')e+="\\n"; else if(c=='\r')e+="\\r";
                else if(c<0x20){char b[8];snprintf(b,8,"\\u%04x",c);e+=b;} else e+=c;
            } return e;
        };
        try {
            if (!use2c && S.dist_sess && S.dist_sess->is_connected()) {
                fprintf(stderr,"[continue/dist] warm distributed continue at ctx=%d\n",
                        S.dist_sess->context_size());
                fflush(stderr);
                ctxc_base = S.dist_sess->context_size();
                auto dr = S.dist_sess->continue_stream([&](const std::string& t,int32_t)->bool{
                    if(sfc)return false;
                    ++ctxc;
                    fprintf(stderr,"[sse/continue/dist] chunk=%d chars=%zu text=\"%s\"\n",
                            ctxc, t.size(), nx_log_visible_text(t).c_str());
                    fflush(stderr);
                    std::string ev="data: {\"token\":\""+jesc2(t)+"\",\"context_tokens\":"+std::to_string(ctxc_base+ctxc)+",\"gen\":"+std::to_string(ctxc)+"}\n\n";
                    emit(ev); return true;
                }, cfgc);
                (void)dr;
                emit("data: {\"final\":true,\"context_tokens\":"+std::to_string(S.dist_sess->context_size())+",\"gen\":"+std::to_string(ctxc)+"}\n\n");
            } else {
                auto sess2 = engc.begin_session_warm(cfgc);
                ctxc_base = sess2.pos();
                sess2.run([&](const std::string& t,int32_t)->bool{
                    if(sfc)return false;
                    ++ctxc;
                    live_ctx = sess2.pos() + 1; live_gen = sess2.generated() + 1;
                    fprintf(stderr,"[sse/continue] chunk=%d chars=%zu text=\"%s\"\n",
                            ctxc, t.size(), nx_log_visible_text(t).c_str());
                    fflush(stderr);
                    std::string ev="data: {\"token\":\""+jesc2(t)+"\",\"context_tokens\":"+std::to_string(live_ctx)+",\"gen\":"+std::to_string(live_gen)+"}\n\n";
                    emit(ev); return true;
                });
                engc.sync_session_tokens(sess2.tokens(), sess2.pos());
                fprintf(stderr,"[gen] reply finished: %d tokens generated, %d in memory\n", sess2.generated(), engc.context_size()); fflush(stderr);
                { const std::string fn = nx_full_notice(engc); if (!fn.empty()) emit("data: {\"notice\":\""+nx_jstr(fn)+"\"}\n\n"); }
                emit("data: {\"final\":true,\"context_tokens\":"+std::to_string(engc.context_size())+",\"gen\":"+std::to_string(sess2.generated())+"}\n\n");
            }
        } catch(const std::exception& ex){
            fprintf(stderr,"[continue] EXCEPTION: %s\n",ex.what());
        }
        emit(NX_SSE_DONE);
        { std::lock_guard<std::mutex> l(J.m); J.running=false; J.done=true; }
        J.cv.notify_all();
        close(fd); return;
    }

    // -- Session inject ---------------------------------------------------------
    // POST /api/session/:id/inject  body: {"text":"<result>..."}
    if(r.method=="POST" && r.path.size()>14 && r.path.substr(0,13)=="/api/session/"){
        size_t last_slash = r.path.rfind('/');
        std::string route_end = r.path.substr(last_slash+1);
        if(route_end=="inject"){
            std::string sid = r.path.substr(13, last_slash-13);
            std::string inject_text = jp_str(r.body,"text","");
            bool found = false;
            {
                std::lock_guard<std::mutex> sl(S.sessions_mu);
                auto it = S.sessions.find(sid);
                if(it != S.sessions.end()){
                    auto* as = it->second;
                    {
                        std::lock_guard<std::mutex> ul(as->mu);
                        as->inject_text  = inject_text;
                        as->inject_ready = true;
                    }
                    as->cv.notify_one();
                    found = true;
                }
            }
            if(found){
                resp_json(fd,"{\"ok\":true,\"session_id\":\""+sid+"\"}");
                fprintf(stderr,"[session] injected %zu chars into session %s\n",
                        inject_text.size(), sid.c_str());
            } else {
                resp_json(fd,"{\"ok\":false,\"error\":\"session not found\"}");
            }
            close(fd); return;
        }
        // DELETE /api/session/:id  - abort a session
        if(r.method=="DELETE"){
            std::string sid2 = r.path.substr(13);
            std::lock_guard<std::mutex> sl(S.sessions_mu);
            auto it = S.sessions.find(sid2);
            if(it != S.sessions.end()){
                auto* as = it->second;
                { std::lock_guard<std::mutex> ul(as->mu); as->aborted=true; }
                as->cv.notify_one();
            }
            resp_json(fd,"{\"ok\":true}"); close(fd); return;
        }
    }

    // -- Session save/load/extend ----------------------------------------------
    if(r.method=="GET"&&r.path=="/api/sessions/list"){
        std::string dir=S.storage.base+"/sessions";
        auto files=list_dir(dir);
        std::string j="[";
        bool first=true;
        for(auto& fn:files){
            const std::string tsuf=".kvc.tokens";
            if(fn.size()<=tsuf.size()||fn.substr(fn.size()-tsuf.size())!=tsuf)continue;
            std::string base=fn.substr(0,fn.size()-tsuf.size());
            std::string meta="";
            FILE* mf=fopen((dir+"/"+base+".kvc.json").c_str(),"r");
            if(mf){fseek(mf,0,SEEK_END);long sz=ftell(mf);rewind(mf);
                if(sz>0&&sz<1048576){meta.resize(sz);(void)fread(&meta[0],1,sz,mf);}
                fclose(mf);}
            if(meta.empty()){
                std::string name=base; std::replace(name.begin(),name.end(),'_',' ');
                meta="{\"name\":\""+name+"\",\"path\":\"nexus-data/sessions/"+base+".kvc\"}";
            }
            if(!first)j+=","; j+=meta; first=false;
        }
        j+="]";
        resp_json(fd,j);close(fd);return;
    }
    if(r.method=="GET"&&r.path=="/api/folders"){
        std::string defd=S.storage.base+"/sessions";
        std::string j="[{\"path\":\"nexus-data/sessions\",\"label\":\"Nexus folder (default)\",\"default\":true,\"free_mb\":"+std::to_string(free_mb_of(defd))+"}";
        std::lock_guard<std::mutex> l(g_folders_mu);
        for(auto& f:g_folders) j+=",{\"path\":"+jstr(f)+",\"label\":"+jstr(f)+",\"default\":false,\"exists\":"+(dir_exists(f)?"true":"false")+",\"free_mb\":"+std::to_string(free_mb_of(f))+"}";
        j+="]"; resp_json(fd,j); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/folders/add"){
        std::string d=norm_dir(jp_str(r.body,"path",""));
        if(d.empty()||has_dotdot(d)){resp_json(fd,"{\"ok\":false,\"error\":\"enter a folder path\"}");close(fd);return;}
        mkdir_p(d);
        if(!dir_exists(d)){resp_json(fd,"{\"ok\":false,\"error\":\"cannot create or reach that folder\"}");close(fd);return;}
        std::string probe=d+"/.nexus_write_test";
        if(!write_file(probe,"x")){resp_json(fd,"{\"ok\":false,\"error\":\"that folder is not writable\"}");close(fd);return;}
        remove(probe.c_str());
        { std::lock_guard<std::mutex> l(g_folders_mu);
          bool have=false; for(auto& f:g_folders) if(f==d) have=true;
          if(!have){ g_folders.push_back(d); folders_save_locked(S.storage.base); } }
        resp_json(fd,"{\"ok\":true}"); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/folders/remove"){          // forgets the folder; the files in it are not touched
        std::string d=norm_dir(jp_str(r.body,"path",""));
        { std::lock_guard<std::mutex> l(g_folders_mu);
          g_folders.erase(std::remove(g_folders.begin(),g_folders.end(),d),g_folders.end()); folders_save_locked(S.storage.base); }
        resp_json(fd,"{\"ok\":true}"); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/sessions/list_in"){          // body {"folder":"..."}  -> [{"base":..,"meta":{..}|null,"mb":N}]
        std::string dir=norm_dir(jp_str(r.body,"folder","nexus-data/sessions"));
        if(!session_dir_ok(dir,S.storage.base)){resp_json(fd,"{\"ok\":false,\"error\":\"folder not allowed\"}");close(fd);return;}
        std::string real = (dir=="nexus-data/sessions") ? S.storage.base+"/sessions" : dir;
        std::string j="["; bool first=true;
        for(auto& fn:list_dir(real)){
            const std::string tsuf=".kvc.tokens";
            if(fn.size()<=tsuf.size()||fn.substr(fn.size()-tsuf.size())!=tsuf) continue;
            std::string base=fn.substr(0,fn.size()-tsuf.size());
            std::string meta; FILE* mf=fopen((real+"/"+base+".kvc.json").c_str(),"r");
            if(mf){fseek(mf,0,SEEK_END);long sz=ftell(mf);rewind(mf);
                if(sz>0&&sz<16777216){meta.resize(sz);(void)fread(&meta[0],1,sz,mf);} fclose(mf);}
            long long kvb=0; { struct stat st; if(::stat((real+"/"+base+".kvc.kv").c_str(),&st)==0) kvb=(long long)st.st_size; }
            if(!first) j+=","; first=false;
            j+="{\"base\":"+jstr(base)+",\"mb\":"+std::to_string(kvb>>20)+",\"meta\":"+(meta.empty()?std::string("null"):meta)+"}";
        }
        j+="]"; resp_json(fd,j); close(fd); return;
    }
    // -- Live chats: reconnect a closed page to a running or finished reply --------------------------------------------
    if(r.method=="POST"&&r.path=="/api/live/snap"){
        std::string conv=jp_str(r.body,"conv_id",""), ui=jp_str(r.body,"ui_id","");
        if(conv.empty()||ui.empty()){resp_json(fd,"{\"ok\":false}");close(fd);return;}
        long long now=nx_now_ms(); bool mine=true;
        { std::lock_guard<std::mutex> l(S.live_mu);
          LiveChat& L=S.live[conv]; L.conv=conv;
          if(L.owner.empty()||L.owner==ui||now-L.owner_ms>NX_OWNER_FRESH_MS){
              L.owner=ui; L.owner_ms=now; L.upd=now;
              L.title=jp_str(r.body,"title",""); L.tid=jp_str(r.body,"tid",""); L.last_role=jp_str(r.body,"last_role","");
              L.model=(int)jp_num(r.body,"model",1)==2?2:1; L.n=(int)jp_num(r.body,"n",0);
              write_file(live_file(S,conv),jp_str(r.body,"msgs_json",""));
          } else mine=false;
          while(S.live.size()>8){ auto old=S.live.begin(); for(auto it=S.live.begin();it!=S.live.end();++it) if(it->second.upd<old->second.upd) old=it; S.live.erase(old); } }
        resp_json(fd,std::string("{\"ok\":true,\"mine\":")+(mine?"true":"false")+"}"); close(fd); return;
    }
    if(r.path=="/api/boot"){   // a new random id every time the server starts: a page that sees it change knows the server was restarted
        static const std::string boot_id=[]{ std::random_device rd; char b[40]; snprintf(b,sizeof b,"%08x%08x%08x",(unsigned)rd(),(unsigned)rd(),(unsigned)std::chrono::steady_clock::now().time_since_epoch().count()); return std::string(b); }();
        resp_json(fd,std::string("{\"boot\":\"")+boot_id+"\"}"); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/live/hb"){
        std::string conv=jp_str(r.body,"conv_id",""), ui=jp_str(r.body,"ui_id","");
        long long now=nx_now_ms(); bool mine=true;
        { std::lock_guard<std::mutex> l(S.live_mu);
          auto it=S.live.find(conv);
          if(it!=S.live.end()){
              LiveChat& L=it->second;
              if(L.owner.empty()||L.owner==ui||now-L.owner_ms>NX_OWNER_FRESH_MS){ L.owner=ui; L.owner_ms=now; } else mine=false;
          } }
        resp_json(fd,std::string("{\"ok\":true,\"mine\":")+(mine?"true":"false")+"}"); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/live/list"){
        std::string ui=jp_str(r.body,"ui_id",""); long long now=nx_now_ms();
        std::string j="["; bool first=true;
        std::lock_guard<std::mutex> l(S.live_mu);
        for(auto& kv:S.live){
            LiveChat& L=kv.second; GenJob& J=S.cjob[L.model==2?1:0];
            bool running=false; size_t evs=0; { std::lock_guard<std::mutex> jl(J.m); running=J.running&&!L.tid.empty()&&J.tid==L.tid; evs=J.evs.size(); }
            bool connected=!L.owner.empty()&&L.owner!=ui&&now-L.owner_ms<=NX_OWNER_FRESH_MS;
            if(!first) j+=","; first=false;
            j+="{\"conv_id\":"+jstr(L.conv)+",\"title\":"+jstr(L.title)+",\"n\":"+std::to_string(L.n)+",\"ago_s\":"+std::to_string((now-L.upd)/1000)+
               ",\"running\":"+(running?"true":"false")+",\"tokens\":"+std::to_string(evs)+",\"connected\":"+(connected?"true":"false")+",\"mine\":"+(L.owner==ui?"true":"false")+"}";
        }
        j+="]"; resp_json(fd,j); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/live/claim"){
        std::string conv=jp_str(r.body,"conv_id",""), ui=jp_str(r.body,"ui_id",""); long long now=nx_now_ms();
        std::string out;
        { std::lock_guard<std::mutex> l(S.live_mu);
          auto it=S.live.find(conv);
          if(it==S.live.end()){ out="{\"ok\":false,\"error\":\"That chat is no longer on the server.\"}"; }
          else {
              LiveChat& L=it->second;
              if(!L.owner.empty()&&L.owner!=ui&&now-L.owner_ms<=NX_OWNER_FRESH_MS){
                  out="{\"ok\":false,\"error\":\"Another window is connected to this chat right now. Close it (or wait about 15 seconds after closing it) and try again.\"}";
              } else {
                  L.owner=ui; L.owner_ms=now;
                  GenJob& J=S.cjob[L.model==2?1:0]; bool has_job=false, running=false; { std::lock_guard<std::mutex> jl(J.m); has_job=!L.tid.empty()&&J.tid==L.tid; running=has_job&&J.running; }
                  bool resume=has_job&&L.last_role=="user";
                  out="{\"ok\":true,\"title\":"+jstr(L.title)+",\"msgs_json\":"+jstr(read_file(live_file(S,L.conv)))+",\"tid\":"+jstr(L.tid)+",\"model\":\""+std::to_string(L.model)+
                      "\",\"running\":"+(running?"true":"false")+",\"resume\":"+(resume?"true":"false")+"}";
              }
          } }
        resp_json(fd,out); close(fd); return;
    }
    if(r.method=="POST"&&r.path=="/api/session/delete"){
        std::string path=jp_str(r.body,"path","");
        if(path.empty()){resp_json(fd,"{\"ok\":false,\"error\":\"no path\"}");close(fd);return;}
        if(!session_path_ok(path,S.storage.base)){resp_json(fd,"{\"ok\":false,\"error\":\"folder not allowed\"}");close(fd);return;}
        // Delete all three files: .kv .tokens .json
        int deleted=0;
        for(auto& suf:{std::string(".kv"),std::string(".tokens"),std::string(".json")}){
            std::string fp=path+suf;
            if(remove(fp.c_str())==0) ++deleted;
        }
        fprintf(stderr,"[delete] %s - removed %d files\n",path.c_str(),deleted);
        resp_json(fd,deleted>0?"{\"ok\":true}":"{\"ok\":false,\"error\":\"Could not find that save's files. Were they moved or deleted?\"}");
        close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/sessions/meta"){
        std::string path=jp_str(r.body,"path","");
        std::string meta=jp_str(r.body,"meta","");
        if(path.empty()||meta.empty()){resp_json(fd,"{\"ok\":false}");close(fd);return;}
        if(!session_path_ok(path,S.storage.base)){resp_json(fd,"{\"ok\":false,\"error\":\"folder not allowed\"}");close(fd);return;}
        std::string jpath=path+".json";
        FILE* f=fopen(jpath.c_str(),"w");
        if(!f){resp_json(fd,"{\"ok\":false,\"error\":\"Could not write to that folder (disk full or read-only?)\"}");close(fd);return;}
        fwrite(meta.c_str(),1,meta.size(),f);fclose(f);
        resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    if(r.method=="POST" && r.path=="/api/session/save"){
        std::string path = jp_str(r.body,"path","nexus_session");
        if(!session_path_ok(path,S.storage.base)){resp_json(fd,"{\"ok\":false,\"error\":\"that folder is not in the list of save folders\"}");close(fd);return;}
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& eng = use2 ? S.engine2 : S.engine;
        std::lock_guard<std::mutex> lg(use2?S.gen_mutex2:S.gen_mutex);
        bool ok = eng.save_session(path);
        // Distributed: save worker KV alongside primary session.
        // Worker bytes transferred over network (RAM) -> stored on primary disk.
        if (ok && !use2 && S.dist_sess && S.dist_sess->is_connected()) {
            std::vector<uint8_t> wkv;
            if (S.dist_sess->save_worker_kv(wkv) && !wkv.empty()) {
                FILE* wf = fopen((path + ".worker_kv").c_str(), "wb");
                if (wf) { fwrite(wkv.data(), 1, wkv.size(), wf); fclose(wf); }
                fprintf(stderr, "[session] Saved worker KV: %zu bytes\n", wkv.size());
                fflush(stderr);
            }
        }
        resp_json(fd, ok ? "{\"ok\":true,\"path\":\""+path+"\"}"
                         : "{\"ok\":false,\"error\":\"Could not save. Check that the model is still running and that the folder has free space.\"}");
        close(fd); return;
    }
    if(r.method=="POST" && r.path=="/api/session/load"){
        std::string path = jp_str(r.body,"path","nexus_session");
        if(!session_path_ok(path,S.storage.base)){resp_json(fd,"{\"ok\":false,\"error\":\"that folder is not in the list of save folders\"}");close(fd);return;}
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& eng = use2 ? S.engine2 : S.engine;
        fprintf(stderr,"[session/load] path=%s\n",path.c_str()); fflush(stderr);
        std::lock_guard<std::mutex> lg(use2?S.gen_mutex2:S.gen_mutex);
        bool ok = eng.load_session(path);
        // Distributed: restore worker KV from file saved on primary.
        // Bytes sent to worker over network (pure RAM) -- no disk on worker.
        if (ok && !use2 && S.dist_sess && S.dist_sess->is_connected()) {
            FILE* wf = fopen((path + ".worker_kv").c_str(), "rb");
            if (wf) {
                fseek(wf,0,SEEK_END); long wsz=ftell(wf); rewind(wf);
                if (wsz > 0) {
                    std::vector<uint8_t> wkv((size_t)wsz);
                    (void)fread(wkv.data(),1,(size_t)wsz,wf); fclose(wf);
                    bool wok = S.dist_sess->load_worker_kv(wkv);
                    fprintf(stderr,"[session] Restored worker KV: %ld bytes: %s\n",
                            wsz, wok?"OK":"FAILED"); fflush(stderr);
                } else { fclose(wf); }
            }
        }
        fprintf(stderr,"[session/load] result=%s ctx=%d\n",ok?"OK":"FAILED",eng.context_size()); fflush(stderr);
        if(ok) nx_forget_conversations(S, use2?2:1, "");          // the loaded KV replaced whatever chat was in memory
        resp_json(fd, ok ? "{\"ok\":true,\"context_tokens\":"+std::to_string(eng.context_size())+"}"
                         : "{\"ok\":false,\"error\":\"Could not load that save. It may belong to a different model, or its files are incomplete.\"}");
        close(fd); return;
    }
    if(r.method=="POST" && r.path=="/api/session/extend"){
        // Inject new text into context without reprocessing existing tokens
        std::string text = jp_str(r.body,"text","");
        bool use2 = S.has_engine2 && jp_str(r.body,"model","1")=="2";
        nexus::Engine& eng = use2 ? S.engine2 : S.engine;
        std::lock_guard<std::mutex> lg(use2?S.gen_mutex2:S.gen_mutex);
        eng.extend_context(text, false);
        resp_json(fd, "{\"ok\":true,\"context_tokens\":"+std::to_string(eng.context_size())+"}");
        close(fd); return;
    }

    // Projects list
    if(r.method=="GET"&&r.path=="/api/projects"){
        auto ps=S.storage.list_projects();
        std::string j="[";
        for(size_t i=0;i<ps.size();++i){if(i)j+=",";
            j+="{\"id\":"+jstr(ps[i].id)+",\"name\":"+jstr(ps[i].name)+
               ",\"color\":"+jstr(ps[i].color)+",\"created_at\":"+jstr(ps[i].created_at)+"}";}
        j+="]";resp_json(fd,j);close(fd);return;
    }
    // Create project
    if(r.method=="POST"&&r.path=="/api/projects"){
        Project p{gen_id(),jp_str(r.body,"name","Untitled"),jp_str(r.body,"color","#7c6af7"),now_iso()};
        S.storage.save_project(p);resp_json(fd,"{\"id\":"+jstr(p.id)+"}");close(fd);return;
    }
    // Delete project
    if(r.method=="DELETE"&&r.path.find("/api/projects/")==0&&r.path.find("/chats")==std::string::npos){
        S.storage.delete_project(r.path.substr(14));resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    // List chats in project
    if(r.method=="GET"&&r.path.find("/api/projects/")==0&&r.path.find("/chats")!=std::string::npos){
        auto cs=S.storage.list_chats(pid_from_path(r.path));
        std::string j="[";
        for(size_t i=0;i<cs.size();++i){if(i)j+=",";
            j+="{\"id\":"+jstr(cs[i].id)+",\"project_id\":"+jstr(cs[i].project_id)+
               ",\"title\":"+jstr(cs[i].title)+",\"model_name\":"+jstr(cs[i].model_name)+
               ",\"chat_format\":"+jstr(cs[i].chat_format)+",\"updated_at\":"+jstr(cs[i].updated_at)+"}";}
        j+="]";resp_json(fd,j);close(fd);return;
    }
    // Save chat
    if(r.method=="POST"&&r.path.find("/api/projects/")==0&&r.path.find("/chats")!=std::string::npos){
        Chat c;c.id=jp_str(r.body,"id");c.project_id=pid_from_path(r.path);
        c.title=jp_str(r.body,"title","Untitled");c.model_name=jp_str(r.body,"model_name");
        c.chat_format=jp_str(r.body,"chat_format");c.system_prompt=jp_str(r.body,"system_prompt");
        c.messages=jp_messages(r.body);
        S.storage.save_chat(c);resp_json(fd,"{\"id\":"+jstr(c.id)+"}");close(fd);return;
    }
    // Load chat
    if(r.method=="GET"&&r.path.find("/api/chats/")==0){
        auto c=S.storage.load_chat(r.path.substr(11));
        if(c.id.empty()){resp_json(fd,"{\"error\":\"not found\"}",404);close(fd);return;}
        std::string j="{\"id\":"+jstr(c.id)+",\"project_id\":"+jstr(c.project_id)+
            ",\"title\":"+jstr(c.title)+",\"model_name\":"+jstr(c.model_name)+
            ",\"chat_format\":"+jstr(c.chat_format)+",\"system_prompt\":"+jstr(c.system_prompt)+
            ",\"messages\":[";
        for(size_t i=0;i<c.messages.size();++i){if(i)j+=",";
            j+="{\"role\":"+jstr(c.messages[i].role)+",\"content\":"+jstr(c.messages[i].content)+"}";}
        j+="]}";resp_json(fd,j);close(fd);return;
    }
    // Delete chat
    if(r.method=="DELETE"&&r.path.find("/api/chats/")==0){
        S.storage.delete_chat(r.path.substr(11));resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    // Templates
    if(r.method=="GET"&&r.path=="/api/templates"){
        auto ts=S.storage.list_templates();
        std::string j="[";
        for(size_t i=0;i<ts.size();++i){if(i)j+=",";
            j+="{\"id\":"+jstr(ts[i].id)+",\"name\":"+jstr(ts[i].name)+
               ",\"description\":"+jstr(ts[i].description)+",\"content\":"+jstr(ts[i].content)+"}";}
        j+="]";resp_json(fd,j);close(fd);return;
    }
    if(r.method=="POST"&&r.path=="/api/templates"){
        PTemplate t;t.name=jp_str(r.body,"name");t.description=jp_str(r.body,"description");
        t.content=jp_str(r.body,"content");S.storage.save_template(t);
        resp_json(fd,"{\"id\":"+jstr(t.id)+"}");close(fd);return;
    }
    if(r.method=="DELETE"&&r.path.find("/api/templates/")==0){
        S.storage.delete_template(r.path.substr(15));resp_json(fd,"{\"ok\":true}");close(fd);return;
    }
    resp_json(fd,"{\"error\":\"not found\"}",404);close(fd);
}

// --- main ---------------------------------------------------------------------

static void usage(const char* p){
    fprintf(stderr,"Usage: %s [--model <path>] [--port <n>] [--host <addr>] [--data <dir>] [--threads <n>] [--chat-format <fmt>]\n"
        "\n  --model2 <path>            load a second model (runs parallel to primary)"
        "\n  --model2-gpu-layers <n>    GPU layers for model2 (-1 = all, good for small GPU-only models)"
        "\n  --model2-gpu-device <n>    GPU device index for model2"
        "\n  --model2-threads <n>       CPU threads for model2"
        "\n  --model2-chat-format <fmt> chat template for model2"
        "\n  --temperature <f>          generation temperature (overrides nexus.json)"
        "\n  --top-p <f>               top-p sampling"
        "\n  --rep-penalty <f>         repetition penalty (1.0=off, try 1.1)"
        "\n  --min-p <f>               min-p sampling (0.05 default, improves coherence)"
        "\n  --tfs-z <f>               tail-free sampling z (1.0=off, try 0.95)"
        "\n  --vision-encoder <path>   load CLIP/SigLIP mmproj GGUF for image input"
        "\n  --audio-encoder <path>    load Whisper GGUF for audio/speech input"
        "\n  --start-state <path>      load saved KV cache state before serving"
        "\n  --park-layers <N|auto>     Maximize GPU: N layers (auto = as many as fit) live on the prompt speed-up card for writing, with no copy in RAM; when a prompt needs the card's room they move off it, then go back before the reply (needs --turbo-prefill, Linux)"
        "\n  --park-below-mb <MB>       oversized-prompt protection: while a prompt is read, whenever free RAM is under this, drop layers from RAM; they go onto the card for writing (needs --turbo-prefill, Linux)"
        "\n  --park-prefill-free-mb <MB|auto> card memory the prompt speed-up gets while a prompt is read (default auto = what it needs)"
        "\n  --park-keep-mb <MB>        card memory always left free when putting layers on it (default 768); also what --gpu2-layers auto / --gpuN-layers auto leave free"
        "\n  --gpu2-layers <N|auto>     layers on the second card (auto = as many as fit); --gpu3-layers, --gpu4-layers, ... the same for more cards"
        "\n  --park-min-tokens <N>      optional: prompts shorter than N never move layers and are read on the CPU (default 0 = no such rule)"
        "\n  --ctx-limit <n|auto|max>  conversation length limit: max = the model's maximum (default); n = that many tokens, even above the model's maximum; auto = no limit at all, grows until memory runs out"
        "\n  --api-key <key>           require this key from OTHER computers (Authorization: Bearer <key>, X-API-Key, or ?key=); this computer never needs it. Also NEXUS_API_KEY"
        "\n  --cors <lan|local|any>     which web pages may use the API: lan (default: this computer + local network, not internet sites), local, any"
        "\n  --allow-origin <url>      also allow web pages from this address (repeatable or comma-separated), e.g. http://192.168.1.20:5173"
        "\n  --enable-web-tools        turn on /api/search, /api/search_detail, /api/search_detail_by_url (fetch web pages; off by default)"
        "\n  --auto-trim <on|off>      when the conversation is full: drop the oldest turns, keeping the system prompt (on, default) or stop (off)"
        "\n  --lora <path>             load LoRA adapter GGUF"
        "\n  --lora-scale <f>          LoRA adapter scale (default 1.0)"
        "\n  --top-k <n>               top-k sampling"
        "\n  --max-tokens <n>          max output tokens"
        "\n  --roster-file <path>      roster JSON written by launcher (default: nexus-roster.json)"
        "\n  --host 0.0.0.0       expose on local network"
        "\n  --chat-format <fmt>  override auto-detect (ChatML, OLMoE, Mistral, Mistral V3, Llama-3, DeepSeek, DeepSeek-R1, DeepSeek-Coder, Phi-3, Phi-4, Gemma, Zephyr, Alpaca, Granite, Custom, Raw)"
        "\n  --chat-custom-pre-prompt <s>  custom template prefix before all turns"
        "\n  --chat-custom-pre-system <s>  custom template prefix before system text"
        "\n  --chat-custom-end-system <s>  custom template suffix after system text"
        "\n  --chat-custom-pre-user <s>    custom template prefix before user text"
        "\n  --chat-custom-end-user <s>    custom template suffix after user text"
        "\n  --chat-custom-pre-model <s>   custom template prefix before assistant/model text"
        "\n  --chat-custom-end-model <s>   custom template suffix after assistant/model history text"
        "\n  --dist-worker                   run as a remote worker node (no HTTP UI)"
        "\n  --dist-local-model <path>       worker mode: load assigned shard from this local GGUF"
        "\n  --dist-passphrase <word>        shared secret - must match primary and all workers"
        "\n  --dist-layer-start <n>          first layer index this node runs"
        "\n  --dist-layer-end <n>            last layer index + 1 this node runs"
        "\n  --dist-peer <host:port>         (primary) add a worker node"
        "\n  --dist-peer-layers <start:end>  layer range for the preceding --dist-peer"
        "\n  --dist-peer-local-load          use worker's local GGUF instead of streaming weights"
        "\n  --dist-required                 refuse full local load if distributed setup is incomplete/fails"
        "\n  --trace-level <0|1|2|3>        set NEXUS_TRACE for this server and distributed worker"
        "\n\n  Once running: http://localhost:8080\n",p);
}
// A graphics card can be given as its number, or as  name:<card name>[#n]  (n-th card of that name, default 1).  The number is the position
// in THIS program's own Vulkan list; on laptops with two graphics systems that list can be in a different order in different programs/starts,
// so the launcher sends the card's name and each program finds the card itself.
static int nx_resolve_gpu_arg(const std::string& v){
    if(v.empty()) return -1;
    if(v.compare(0,5,"name:")!=0) return atoi(v.c_str());
    std::string name=v.substr(5); int ord=1;
    size_t h=name.rfind('#');
    if(h!=std::string::npos && h+1<name.size() && name.find_first_not_of("0123456789",h+1)==std::string::npos){ ord=std::max(1,atoi(name.c_str()+h+1)); name.erase(h); }
    auto low=[](std::string s){ for(auto& c:s) c=(char)tolower((unsigned char)c); return s; };
    const std::string want=low(name);
#ifdef NEXUS_VULKAN
    int seen=0;
    for(const std::string& e: nexus::gpu::VulkanBackend::list_devices()){          // "[2] NVIDIA GeForce 940MX (discrete, 2048 MB)"
        size_t a=e.find('['), b=e.find("] "); if(a!=0||b==std::string::npos) continue;
        size_t p=e.rfind(" ("); std::string nm=e.substr(b+2, p==std::string::npos? std::string::npos : p-(b+2));
        if(low(nm)==want && ++seen==ord){ int idx=atoi(e.c_str()+1); fprintf(stderr,"[gpu] \"%s\" is graphics card %d on this computer\n",name.c_str(),idx); return idx; }
    }
#endif
    fprintf(stderr,"[gpu] WARNING: graphics card \"%s\" was not found here; letting the program choose\n",name.c_str());
    return -1;
}

int main(int argc,char* argv[]){
    nexus::crashlog::install();
#ifdef _WIN32
    // On Windows, stderr/stdout are block-buffered when connected to a pipe.
    // Use a small static buffer with line-buffering so every \n flushes immediately.
    // DO NOT use nullptr buffer (_IONBF) - that crashes MinGW on pipe handles.
    { static char _sb[256]; setvbuf(stderr, _sb, _IOLBF, sizeof(_sb)); }
    { static char _sb[256]; setvbuf(stdout, _sb, _IOLBF, sizeof(_sb)); }
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2,2), &wsa_data) != 0) {
        fprintf(stderr, "[server] WSAStartup failed\n"); return 1;
    }
#endif
    // --probe: print a JSON description of this machine (CPU, SIMD, RAM, Vulkan GPUs)
    // and exit.  Needs no license and no model; the launcher uses it to pick defaults.
    for(int i=1;i<argc;++i){
        if(std::string(argv[i])=="--probe"){
            std::string j=nexus::probe::run();
            fwrite(j.data(),1,j.size(),stdout); fflush(stdout);
            return 0;
        }
    }
    std::string model,cfgpath,host="127.0.0.1",data="./nexus-data";
    std::string chat_format_override;
    nexus::ChatTemplate::CustomTemplateParts custom_chat_template;
    bool custom_chat_template_set = false;
    std::string model2;
    std::string model2_chat_format;
    // -- Distributed inference --------------------------------------------
    bool        dist_worker_mode = false;
    std::string dist_passphrase;
    std::string dist_local_model;
    int         dist_layer_start = -1;  // -1 = run all layers
    int         dist_layer_end   = -1;
    struct DistPeer { std::string host; int port; int layer_start; int layer_end; bool local_load; };
    std::vector<DistPeer> dist_peers;
    bool        dist_required = false;  // safety fuse: never local-fallback when remote split was requested
    std::string roster_file_cli;
    std::string vision_enc_path;
    std::string audio_enc_path;
    std::string start_state_path;
    std::string lora_path;
    float lora_scale = 1.0f;
    int port=8080,threads=-1,gpu_layers_cli=-1,gpu_device_cli=-1,gpu2_layers_cli=-1,gpu2_device_cli=-1;
    std::map<int,int> gpux_layers_cli, gpux_device_cli;   // --gpu3-layers / --gpu3-device, --gpu4-... (any number of extra cards)
    int port_cli=-1,threads_cli=-1;  // -1 = not set on CLI; used to re-apply after nexus.json load
    std::string host_cli;            // empty = not set on CLI
    int model2_gpu_layers=-1,model2_gpu_device=-1,model2_threads=-1;
    int turbo_gpu_device_cli = -1;   // NEW    
    float temp_cli=-1.f,topp_cli=-1.f,rep_cli=-1.f,minp_cli=-1.f,tfsz_cli=-1.f;
    float freqpen_cli=-1.f,prespen_cli=-1.f;
    int topk_cli=-1,maxtok_cli=-1,replastn_cli=-1;
    int ctx_limit_cli=0, auto_trim_cli=1;   // --ctx-limit (0=model max, -1=auto) / --auto-trim
    bool web_tools_cli=false;               // --enable-web-tools
    std::string api_key_cli; int cors_cli=0; std::vector<std::string> allow_origins_cli;   // --api-key / --cors / --allow-origin
    int mtp_enabled_cli=-1, mtp_npredict_cli=-1, mtp_gpu_cli=-1;
    int trace_level_cli = -1;
    bool turbo_prefill_cli = false;
    // Layer parking: "Maximize GPU" (--park-layers) and oversized-prompt protection (--park-below-mb). See park_setup.
    int park_layers_cli = 0, park_below_mb_cli = 0, park_keep_mb_cli = 768, park_prefill_free_mb_cli = 0, park_min_tokens_cli = 0;
    bool turbo_gpu_stream_cli = false;       // NEW
    bool prefill_token_by_token_cli = false;
    int  turbo_vram_budget_mb_cli = 0;       // NEW
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        if(a=="--model"&&i+1<argc)model=argv[++i];
        else if(a=="--port"&&i+1<argc){port=atoi(argv[++i]);port_cli=port;}
        else if(a=="--host"&&i+1<argc){host=argv[++i];host_cli=host;}
        else if(a=="--data"&&i+1<argc)data=argv[++i];
        else if(a=="--threads"&&i+1<argc){threads=atoi(argv[++i]);threads_cli=threads;}
        else if(a=="--config"&&i+1<argc)cfgpath=argv[++i];
        else if(a=="--chat-format"&&i+1<argc)chat_format_override=argv[++i];
        else if(a=="--chat-custom-pre-prompt"&&i+1<argc){ custom_chat_template.pre_prompt=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-pre-system"&&i+1<argc){ custom_chat_template.pre_system=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-end-system"&&i+1<argc){ custom_chat_template.end_system=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-pre-user"&&i+1<argc){ custom_chat_template.pre_user=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-end-user"&&i+1<argc){ custom_chat_template.end_user=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-pre-model"&&i+1<argc){ custom_chat_template.pre_model=argv[++i]; custom_chat_template_set=true; }
        else if(a=="--chat-custom-end-model"&&i+1<argc){ custom_chat_template.end_model=argv[++i]; custom_chat_template_set=true; }        
        else if(a=="--gpu-layers"&&i+1<argc)gpu_layers_cli=atoi(argv[++i]);
        else if(a=="--gpu-device"&&i+1<argc)gpu_device_cli=nx_resolve_gpu_arg(argv[++i]);
        else if(a=="--gpu2-layers"&&i+1<argc){ std::string v=argv[++i]; gpu2_layers_cli = (v=="auto"||v=="max") ? -2 : std::max(0, atoi(v.c_str())); }   // -2 = auto (-1 = not given)
        else if(a.size()>14&&a.compare(0,5,"--gpu")==0&&isdigit((unsigned char)a[5])&&i+1<argc
                &&(a.size()>=7&&(a.substr(a.find('-',5))=="-layers"||a.substr(a.find('-',5))=="-device"))
                &&atoi(a.c_str()+5)>=3){
            const int gn=atoi(a.c_str()+5); const bool is_lay=a.substr(a.find('-',5))=="-layers";
            if(is_lay){ std::string v=argv[++i]; gpux_layers_cli[gn] = (v=="auto"||v=="max") ? -1 : atoi(v.c_str()); }   // auto = Maximize: as many as fit
            else gpux_device_cli[gn]=nx_resolve_gpu_arg(argv[++i]);
        }
        else if(a=="--gpu2-device"&&i+1<argc)gpu2_device_cli=nx_resolve_gpu_arg(argv[++i]);
        else if(a=="--model2"&&i+1<argc)model2=argv[++i];
        else if(a=="--trace-level"&&i+1<argc) trace_level_cli=nx_clamp_trace_level(atoi(argv[++i]));
        else if(a=="--model2-gpu-layers"&&i+1<argc)model2_gpu_layers=atoi(argv[++i]);
        else if(a=="--model2-gpu-device"&&i+1<argc)model2_gpu_device=nx_resolve_gpu_arg(argv[++i]);
        else if(a=="--model2-threads"&&i+1<argc)model2_threads=atoi(argv[++i]);
        else if(a=="--model2-chat-format"&&i+1<argc)model2_chat_format=argv[++i];
        else if(a=="--temperature"&&i+1<argc)  temp_cli  =(float)atof(argv[++i]);
        else if(a=="--top-p"&&i+1<argc)        topp_cli  =(float)atof(argv[++i]);
        else if(a=="--rep-penalty"&&i+1<argc)  rep_cli   =(float)atof(argv[++i]);
        else if(a=="--top-k"&&i+1<argc)        topk_cli  =atoi(argv[++i]);
        else if(a=="--max-tokens"&&i+1<argc)   maxtok_cli=atoi(argv[++i]);
        else if(a=="--enable-web-tools")       web_tools_cli=true;
        else if(a=="--api-key"&&i+1<argc)      api_key_cli=argv[++i];
        else if(a=="--cors"&&i+1<argc){ std::string v=argv[++i]; cors_cli = v=="any"?2:(v=="local"?1:0); }
        else if(a=="--allow-origin"&&i+1<argc){ std::string v=argv[++i]; size_t p0=0;          // repeatable, or comma-separated
            while(p0<=v.size()){ size_t c=v.find(',',p0); std::string o=v.substr(p0,c==std::string::npos?std::string::npos:c-p0);
                while(!o.empty()&&(o.back()=='/'||o.back()==' ')) o.pop_back(); while(!o.empty()&&o[0]==' ') o.erase(0,1);
                if(o=="*") cors_cli=2; else if(!o.empty()) allow_origins_cli.push_back(o);
                if(c==std::string::npos) break; p0=c+1; } }
        else if(a=="--ctx-limit"&&i+1<argc){ std::string v=argv[++i]; ctx_limit_cli = v=="auto" ? -1 : (v=="max" ? 0 : std::max(0,atoi(v.c_str()))); }
        else if(a=="--auto-trim"&&i+1<argc){ std::string v=argv[++i]; auto_trim_cli=(v=="on"||v=="1"||v=="true")?1:0; }
        else if(a=="--min-p"&&i+1<argc)         minp_cli  =(float)atof(argv[++i]);
        else if(a=="--tfs-z"&&i+1<argc)         tfsz_cli  =(float)atof(argv[++i]);
        else if(a=="--vision-encoder"&&i+1<argc) vision_enc_path=argv[++i];
        else if(a=="--audio-encoder"&&i+1<argc)  audio_enc_path =argv[++i];
        else if(a=="--start-state"&&i+1<argc)    start_state_path=argv[++i];
        else if(a=="--lora"&&i+1<argc)           lora_path=argv[++i];
        else if(a=="--lora-scale"&&i+1<argc)     lora_scale=(float)atof(argv[++i]);
        else if(a=="--roster-file"&&i+1<argc)  roster_file_cli=argv[++i];
        else if(a=="--repeat-last-n"&&i+1<argc) replastn_cli=atoi(argv[++i]);
        else if(a=="--freq-penalty"&&i+1<argc)  freqpen_cli=(float)atof(argv[++i]);
        else if(a=="--presence-penalty"&&i+1<argc) prespen_cli=(float)atof(argv[++i]);
        else if(a=="--turbo-prefill") turbo_prefill_cli = true;
        else if(a=="--park-layers"&&i+1<argc){ std::string v=argv[++i]; park_layers_cli = (v=="auto"||v=="max") ? -1 : std::max(0, atoi(v.c_str())); }
        else if(a=="--park-below-mb"&&i+1<argc) park_below_mb_cli = std::max(0, atoi(argv[++i]));
        else if(a=="--park-keep-mb"&&i+1<argc) park_keep_mb_cli = std::max(0, atoi(argv[++i]));
        else if(a=="--park-prefill-free-mb"&&i+1<argc){ std::string v=argv[++i]; park_prefill_free_mb_cli = v=="auto" ? 0 : std::max(0, atoi(v.c_str())); }
        else if(a=="--park-min-tokens"&&i+1<argc) park_min_tokens_cli = std::max(0, atoi(argv[++i]));
        else if(a=="--turbo-gpu-stream") turbo_gpu_stream_cli = true;                              // NEW
        else if(a=="--prefill-token-by-token") prefill_token_by_token_cli = true;
        else if(a=="--turbo-vram-budget-mb"&&i+1<argc) turbo_vram_budget_mb_cli=atoi(argv[++i]);  // NEW
        else if(a=="--mtp"&&i+1<argc){ std::string v=argv[++i]; mtp_enabled_cli=(v=="on"||v=="1"||v=="true")?1:0; }
        else if(a=="--mtp-n-predict"&&i+1<argc) mtp_npredict_cli=atoi(argv[++i]);
        else if(a=="--mtp-gpu"&&i+1<argc){ std::string v=argv[++i]; mtp_gpu_cli=(v=="on"||v=="1"||v=="true")?1:0; }
        else if(a=="--list-devices"){
#ifdef NEXUS_VULKAN
            auto devs=nexus::Engine::list_gpu_devices();
#else
            std::vector<std::string> devs;
#endif
            if(devs.empty()){printf("No Vulkan devices found.\n");return 0;}
            printf("Vulkan devices:\n");
            for(auto& d:devs) printf("  %s\n",d.c_str());
            return 0;
        }
        else if(a=="--dist-worker")                        dist_worker_mode=true;
        else if(a=="--dist-local-model"&&i+1<argc)         dist_local_model=argv[++i];
        else if(a=="--dist-passphrase"&&i+1<argc)          dist_passphrase=argv[++i];
        else if(a=="--dist-layer-start"&&i+1<argc)         dist_layer_start=atoi(argv[++i]);
        else if(a=="--dist-layer-end"&&i+1<argc)           dist_layer_end=atoi(argv[++i]);
        else if(a=="--dist-peer"&&i+1<argc){
            std::string hp=argv[++i];
            size_t col=hp.rfind(':');
            std::string ph=(col!=std::string::npos)?hp.substr(0,col):hp;
            int         pp=(col!=std::string::npos)?atoi(hp.c_str()+col+1):9876;
            dist_peers.push_back({ph,pp,0,0,false});
        }
        else if(a=="--dist-peer-layers"&&i+1<argc){
            std::string lr=argv[++i];
            size_t col=lr.find(':');
            if(!dist_peers.empty()){
                dist_peers.back().layer_start=atoi(lr.c_str());
                dist_peers.back().layer_end=(col!=std::string::npos)?atoi(lr.c_str()+col+1):0;
            }
        }
        else if(a=="--dist-peer-local-load"){
            if(!dist_peers.empty()) dist_peers.back().local_load=true;
        }
        else if(a=="--dist-required")                    dist_required=true;
        else if(a=="--help"||a=="-h"){usage(argv[0]);return 0;}
    }
    
    if (trace_level_cli >= 0) {
        nx_set_trace_level_env(trace_level_cli);
        fprintf(stderr, "[server] NEXUS_TRACE=%d\n", trace_level_cli);
        fflush(stderr);
    }
    
    nexus::Config cfg;
    if(!cfgpath.empty())cfg.load(cfgpath);else cfg.load_default();
    if(model.empty())model=cfg.get_str("model.path");
    if(threads<0)threads=cfg.get_int("system.threads",0);
    port=cfg.get_int("server.port",port);
    host=cfg.get_str("server.host",host);
    data=cfg.get_str("server.data_dir",data);

    // Also try loading nexus.json (launcher saves here)
    // These override nexus.conf values when set
    auto load_json_cfg=[&](){
        FILE* f=fopen("nexus.json","r");if(!f)return;
        fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
        std::string j(sz,'\0');(void)fread(&j[0],1,sz,f);fclose(f);
        if(!model.empty()){} // command-line model takes priority
        else {auto v=jp_str(j,"model_path");if(!v.empty())model=v;}
        // Only read host/port/threads from nexus.json if NOT set on CLI.
        // This prevents stale nexus.json values from silently overriding
        // --port 8081 / --host etc. when running multiple server instances.
        if(host_cli.empty()){auto ph=jp_str(j,"host");if(!ph.empty())host=ph;}
        if(port_cli    < 0){auto pp=(int)jp_num(j,"port",0);if(pp>0)port=pp;}
        if(threads_cli < 0){auto pt=(int)jp_num(j,"threads",0);if(pt>0)threads=pt;}
        return;
    };
    load_json_cfg();
    // CLI args take priority over nexus.json - re-apply if they were explicitly set
    if(port_cli    > 0)    port    = port_cli;
    if(threads_cli > 0)    threads = threads_cli;
    if(!host_cli.empty())  host    = host_cli;

    // -- Distributed worker mode ---------------------------------------------
    // Worker loads the model directly (same GGUF as primary, or a network path),
    // runs only its assigned layer range, and serves hidden states via TCP.
    // No HTTP server is started. The donor_launcher.py handles the UI.
    if (dist_worker_mode) {
        // Worker mode: bind port, wait. NO model loaded at startup.
        // Primary connects -> XOR auth -> MSG_INIT (layer range).
        // Then either MSG_WEIGHTS streams a mini-GGUF, or MSG_LOCAL_LOAD tells
        // this worker to load only the assigned shard from its own local GGUF.
        // The passphrase is an XOR key -- never sent over the wire.
        if (threads <= 0) {
            threads = (int)std::thread::hardware_concurrency();
            if (threads > 1) --threads;
            if (threads < 1) threads = 1;
        }
        fprintf(stderr, "[dist-worker] Worker node starting\n");
        fprintf(stderr, "[dist-worker] Port       : %d\n", port);
        fprintf(stderr, "[dist-worker] Threads    : %d\n", threads);
        fprintf(stderr, "[dist-worker] Passphrase : %s\n",
                dist_passphrase.empty() ? "(none -- open)" : "set (XOR key)");
        if(!dist_local_model.empty())
            fprintf(stderr, "[dist-worker] Local GGUF : %s\n", dist_local_model.c_str());
        else
            fprintf(stderr, "[dist-worker] Local GGUF : none (primary will stream weights)\n");
        fprintf(stderr, "[dist-worker] Binding port %d -- no weights loaded yet, waiting for primary...\n", port);

        nexus::Engine worker_engine;
        if (gpu_layers_cli > 0) {
            if (gpu_device_cli >= 0) worker_engine.set_gpu_device(gpu_device_cli);
            worker_engine.set_gpu_layers(gpu_layers_cli);
        }
        if ((turbo_prefill_cli || turbo_gpu_stream_cli) && !prefill_token_by_token_cli) {      // prompt acceleration on the donor
            const int dev = (turbo_gpu_device_cli >= 0) ? turbo_gpu_device_cli : gpu_device_cli;
            if (dev >= 0) worker_engine.set_turbo_gpu_device(dev);
            worker_engine.set_turbo_gpu_stream(true);
            worker_engine.set_turbo_prefill(true);
            fprintf(stderr, "[dist-worker] Prompt acceleration requested (graphics card %d)\n", dev);
        }

        nexus::dist::DistPeerServer server(worker_engine, -1, -1, dist_passphrase, dist_local_model, threads);
        server.listen_and_serve(port, host);  // blocks until process is killed
        return 0;
    }

    // -- Primary distributed mode - log peer chain ---------------------------
    if (!dist_peers.empty()) {
        fprintf(stderr, "[dist] Primary mode: %zu worker node(s)\n", dist_peers.size());
        fprintf(stderr, "[dist]   --dist-layer-start=%d  --dist-layer-end=%d\n",
                dist_layer_start, dist_layer_end);
        for (auto& p : dist_peers)
            fprintf(stderr, "[dist]   worker %s:%d  peer-layers %d-%d  mode=%s\n",
                    p.host.c_str(), p.port, p.layer_start, p.layer_end - 1,
                    p.local_load ? "local-load" : "stream-weights");
        // Validate -- these must be set by launcher for partial loading to work
        if (dist_layer_start < 0 || dist_layer_end <= dist_layer_start) {
            fprintf(stderr, "[dist] WARNING: --dist-layer-start/end not set correctly!\n");
            fprintf(stderr, "[dist] Will load FULL model -- check launcher is sending these args\n");
        }
        fflush(stderr);
    }

    if(model.empty()){fprintf(stderr,"Error: no model file given. Start with  --model <file.gguf>  (or pick a model in the launcher).\n\n");usage(argv[0]);return 1;}
    if(threads<=0){threads=(int)std::thread::hardware_concurrency();if(threads>1)--threads;if(threads<1)threads=1;}

    ServerState S;S.port=port;S.host=host;

    // Load generation settings from nexus.json
    {FILE* f=fopen("nexus.json","r");
    if(f){
        fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
        std::string j(sz,'\0');(void)fread(&j[0],1,sz,f);fclose(f);
        S.temperature        =(float)jp_num(j,"temperature",         0.7);
        S.top_p              =(float)jp_num(j,"top_p",               0.95);
        S.min_p              =(float)jp_num(j,"min_p",               0.05);
        S.tfs_z              =(float)jp_num(j,"tfs_z",               1.0);
        S.repeat_penalty     =(float)jp_num(j,"repeat_penalty",      1.1);
        S.repeat_last_n      =(int)  jp_num(j,"repeat_last_n",       64);
        S.frequency_penalty  =(float)jp_num(j,"frequency_penalty",   0.0);
        S.presence_penalty   =(float)jp_num(j,"presence_penalty",    0.0);
        S.top_k              =(int)  jp_num(j,"top_k",               40);
        S.max_tokens         =(int)  jp_num(j,"max_tokens",          32000);
        S.n_threads      =threads;
        // GPU layers: NOT read from nexus.json - must be passed explicitly via
        // --gpu-layers on the CLI. Reading it from saved config caused stale
        // values to silently activate Vulkan across restarts.
        S.n_gpu_layers   = 0;
    }}
    // CLI overrides nexus.json generation params
    if(temp_cli     >= 0.f) S.temperature       = temp_cli;
    if(topp_cli     >= 0.f) S.top_p             = topp_cli;
    if(rep_cli      >= 0.f) S.repeat_penalty    = rep_cli;
    if(replastn_cli >= 0)   S.repeat_last_n     = replastn_cli;
    if(freqpen_cli  >= 0.f) S.frequency_penalty = freqpen_cli;
    if(prespen_cli  >= 0.f) S.presence_penalty  = prespen_cli;
    if(topk_cli     >= 0)   S.top_k             = topk_cli;
    if(maxtok_cli   >= 0)   S.max_tokens        = maxtok_cli;
    S.ctx_limit_cfg = ctx_limit_cli; S.auto_trim = auto_trim_cli != 0;
    S.web_tools = web_tools_cli;
    { const char* ek = getenv("NEXUS_API_KEY"); if(api_key_cli.empty() && ek && *ek) api_key_cli = ek; }   // keeps the key off the command line if preferred
    S.api_key = api_key_cli; S.cors_mode = cors_cli; S.allow_origins = allow_origins_cli;
    fprintf(stderr,"[access] web pages allowed: %s%s; API key for other computers: %s\n",
            S.cors_mode==2 ? "any website (--cors any)" : S.cors_mode==1 ? "this computer only" : "this computer and the local network",
            S.allow_origins.empty() ? "" : (" + " + std::to_string(S.allow_origins.size()) + " added with --allow-origin").c_str(),
            S.api_key.empty() ? "off" : "required");
    if(S.web_tools) fprintf(stderr,"[web] web search / page fetch endpoints are ON (--enable-web-tools)\n");
    if(mtp_enabled_cli >= 0) S.mtp_enabled       = (mtp_enabled_cli != 0);
    if(mtp_npredict_cli > 0) S.mtp_n_predict     = mtp_npredict_cli;
    if(mtp_gpu_cli  >= 0)   S.mtp_gpu           = (mtp_gpu_cli != 0);
    if(minp_cli     >= 0.f) S.min_p             = minp_cli;
    if(tfsz_cli     >= 0.f) S.tfs_z             = tfsz_cli;
    fprintf(stderr,"[server] Config: temp=%.2f top_p=%.2f top_k=%d min_p=%.3f rep=%.2f rep_n=%d max_tok=%d (launcher/CLI values applied)\n",
        S.temperature,S.top_p,S.top_k,S.min_p,S.repeat_penalty,S.repeat_last_n,S.max_tokens);
    fprintf(stderr, "[server] MTP startup setting: enabled=%d tokens/step=%d gpu=%d\n",
            S.mtp_enabled?1:0, S.mtp_n_predict, S.mtp_gpu?1:0);
    // SIMD build status — if AVX2 is NOT compiled in, the NQ matmul kernels fall
    // back to a scalar bit-unpack path that is ~8x slower. This is the #1 cause
    // of slow CPU decode. If this prints AVX2=0 on an AVX2-capable CPU, rebuild
    // with -mavx2 -mfma (or -march=native).
#if defined(__AVX2__)
    fprintf(stderr, "[server] SIMD: AVX2=1 FMA=%d  (NQ kernels use fast path)\n",
#  if defined(__FMA__)
            1
#  else
            0
#  endif
    );
#else
    fprintf(stderr, "[server] SIMD: AVX2=0  *** WARNING: scalar matmul fallback — "
                    "decode will be ~8x slower. Rebuild with -mavx2 -mfma ***\n");
#endif
    fprintf(stderr, "[server] CPU threads in use: %d\n", S.n_threads);
    if(minp_cli     >= 0.f) S.min_p             = minp_cli;
    if(tfsz_cli     >= 0.f) S.tfs_z             = tfsz_cli;
    if(!roster_file_cli.empty()) S.roster_file = roster_file_cli;
    if(!vision_enc_path.empty()){
        fprintf(stderr,"[server] Loading vision encoder: %s\n",vision_enc_path.c_str());
        if(!S.engine.load_vision_encoder(vision_enc_path))
            fprintf(stderr,"[server] WARNING: vision encoder load failed\n");
    }
    if(!audio_enc_path.empty()){
        fprintf(stderr,"[server] Loading audio encoder: %s\n",audio_enc_path.c_str());
        if(!S.engine.load_audio_encoder(audio_enc_path))
            fprintf(stderr,"[server] WARNING: audio encoder load failed\n");
    }

    if(!lora_path.empty()){
        fprintf(stderr,"[server] Loading LoRA adapter: %s (scale=%.2f)\n",lora_path.c_str(),lora_scale);
        std::lock_guard<std::mutex> lg(S.gen_mutex);
        if(!S.engine.lora_load(lora_path,lora_scale))
            fprintf(stderr,"[server] WARNING: LoRA load failed\n");
    }
    // CLI --gpu-layers overrides nexus.json value
    if(gpu_layers_cli>=0) S.n_gpu_layers=gpu_layers_cli;

    printf("\n  ----------------------------------------\n");
    printf("  -   NEXUS  -  Open Inference Engine   -\n");
    printf("  ----------------------------------------\n\n");
    printf("  Loading: %s\n\n",model.c_str());

    S.storage.init(data);
    nexus::memguard::init_from_env();
    if(nexus::memguard::enabled()){ long long av=nexus::memguard::available_bytes(); fprintf(stderr,"[nexus] memory safety buffer: %lld MB kept free (free now: %lld MB)\n",nexus::memguard::buffer_mb(),av<0?-1LL:av>>20); }
    else fprintf(stderr,"[nexus] memory safety buffer: OFF (running to the limit)\n");
    { std::string ld=S.storage.base+"/live"; mkdir_p(ld); for(auto& f:list_dir(ld)) remove((ld+"/"+f).c_str()); }   // chat snapshots are per server run
    folders_load(S.storage.base);

    // Parking needs every big layer weight mapped in place from the model file (so a layer's RAM copy can be thrown
    // away and read back from the file); the loader does that when NEXUS_DIRECT_GPU_LAYERS covers every layer.
    const bool park_wanted = (park_layers_cli != 0 || park_below_mb_cli > 0) && dist_peers.empty();
#ifndef _WIN32
    if (park_wanted && turbo_prefill_cli && !prefill_token_by_token_cli) {
        setenv("NEXUS_DIRECT_GPU_LAYERS", "1000000", 1);
        unsetenv("NEXUS_DIRECT_VRAM_MB");
        if (gpu_layers_cli > 0)
            fprintf(stderr, "[park] note: %d layers were also asked to live on the card the usual way; Maximize GPU replaces that "
                            "(use --gpu-layers 0 with --park-layers)\n", gpu_layers_cli);
    }
#endif
    if (park_wanted && !turbo_prefill_cli)
        fprintf(stderr, "[park] Maximize GPU / oversized-prompt protection need the prompt speed-up (--turbo-prefill) on a card: ignored\n");

    auto lcb=[](const nexus::LoadProgress& p){
        // Use stderr + newlines (no \r) so launcher shows clean log on all platforms
        fprintf(stderr, "[load] %.0f%%  %s\n",
                p.fraction * 100.0f, p.stage.substr(0,60).c_str());
        fflush(stderr);
    };
    if (dist_required && (dist_peers.empty() || dist_layer_start < 0 || dist_layer_end <= dist_layer_start)) {
        fprintf(stderr, "[dist] HARD STOP: distributed launch was required, but dist args are incomplete. Refusing full local load.\n");
        fflush(stderr);
        return 1;
    }

    // -- Model loading: partial for dist primary, full otherwise ----------------
    // -- Distributed: stream worker shard FIRST, then primary loads its own shard --
    // This is the correct order: primary has zero weight RAM loaded when it
    // streams to the worker. After the worker has its data, primary loads itself.
    if (!dist_peers.empty() && dist_layer_start >= 0 && dist_layer_end > dist_layer_start) {
        auto& p = dist_peers[0];
        int split_layer = dist_layer_end;

        // Step 1: register model path so send_layer_subset_gguf can open source GGUF
        S.engine.init_model_path(model);

        // Step 2: connect + prepare worker shard (primary has NO weights loaded yet)
        fprintf(stderr, "[dist] Connecting to worker %s:%d -- %s layers %d-%d...\n",
                p.host.c_str(), p.port,
                p.local_load ? "local-load" : "streaming",
                split_layer, p.layer_end - 1);
        fflush(stderr);
        S.dist_sess = std::make_unique<nexus::dist::DistSession>(S.engine, split_layer);
        // Hybrid prefill: this computer reads the SAME full local model file ("model", already used
        // below for its own shard) to also stand in for the other computer's layers during prefill,
        // handing over the finished KV cache in one shot instead of sending hidden states for every
        // chunk. Off by default -- opt in with NEXUS_HYBRID_PREFILL=1 (also settable from the launcher's
        // distributed-setup menu). Needs this computer's own copy of the model, same as local_load.
        if (const char* e = std::getenv("NEXUS_HYBRID_PREFILL")) {
            if (*e && *e != '0') {
                S.dist_sess->set_local_model_path(model);
                S.dist_sess->set_hybrid_prefill(true);
                fprintf(stderr, "[dist] hybrid prefill enabled: this computer will prefill BOTH layer ranges and hand the other computer its KV cache once prompt reading is done.\n");
            }
        }
        if (!S.dist_sess->connect(p.host, p.port, dist_passphrase, p.layer_end, p.local_load)) {
            fprintf(stderr, "[dist] CONNECT FAILED -- aborting distributed launch. No local-only fallback.\n");
            fflush(stderr);
            S.dist_sess.reset();
            return 1;
        }

        if (p.local_load) {
            fprintf(stderr, "[dist] Worker is loading local shard; primary loading its shard in parallel...\n");
        } else {
            fprintf(stderr, "[dist] Worker has its streamed shard. Now loading primary shard...\n");
        }
        fflush(stderr);

        // Step 3: primary loads its own shard. In local-load mode this happens
        // while the worker is also loading its own shard. After local load, we
        // wait for the worker ACK before marking the model ready.
        if (!S.engine.load_range(model, dist_layer_start, dist_layer_end, threads, lcb)) {
            fprintf(stderr, "Load failed.\n"); return 1;
        }
        if (!S.dist_sess || !S.dist_sess->wait_until_ready()) {
            fprintf(stderr, "[dist] Worker failed to become ready -- aborting launch.\n");
            return 1;
        }
        fprintf(stderr, "[dist] Distributed model ready.\n");
        fflush(stderr);
    } else {
        if(!S.engine.load(model,threads,lcb)){fprintf(stderr,"Error: the model could not be loaded. Check that the file exists, is complete (not a partly finished download) and is a supported GGUF/NXM model; the lines above say what went wrong.\n");return 1;}
    }
    fprintf(stderr,"[server] MODEL READY\n");
    fflush(stderr);

    // Apply --chat-format override if specified (CLI takes priority over nexus.json)
    if(chat_format_override.empty()){
        // Try nexus.json for format override
        FILE* f=fopen("nexus.json","r");
        if(f){
            fseek(f,0,SEEK_END);long sz=ftell(f);rewind(f);
            std::string j(sz,'\0');(void)fread(&j[0],1,sz,f);fclose(f);
            auto v=jp_str(j,"chat_format");
            if(!v.empty()&&v!="auto")chat_format_override=v;
        }
    }
    if (custom_chat_template_set || chat_format_override == "Custom") {
        S.engine.set_custom_chat_template(custom_chat_template);
        fprintf(stderr,"[server] Chat format override: Custom\n");
    } else if(!chat_format_override.empty()) {
        S.engine.override_chat_format(chat_format_override);
        fprintf(stderr,"[server] Chat format override: %s\n",chat_format_override.c_str());
    }

    // Apply GPU layer offloading (after model is loaded, before serving)
    int actual_gpu_layers = 0;
    if(S.n_gpu_layers > 0){
        if(gpu_device_cli >= 0)
            S.engine.set_gpu_device(gpu_device_cli);
        actual_gpu_layers = S.engine.set_gpu_layers(S.n_gpu_layers);
        if(actual_gpu_layers == 0)
            fprintf(stderr,"[server] GPU offload requested (%d layers) but not available - CPU only\n", S.n_gpu_layers);
    }

    // Secondary GPU (iGPU / second device) - layers after the primary GPU's share
    int actual_gpu2_layers = 0;
    S.engine.set_card_fill_keep_mb(park_keep_mb_cli);              // what "Maximize" leaves free on cards 2, 3, ...
    if((gpu2_layers_cli > 0 || gpu2_layers_cli == -2) && gpu2_device_cli >= 0){
        actual_gpu2_layers = S.engine.set_gpu2_layers(gpu2_layers_cli == -2 ? -1 : gpu2_layers_cli, gpu2_device_cli,
                                                       actual_gpu_layers);
        if(actual_gpu2_layers > 0)
            fprintf(stderr,"[server] Secondary GPU: %d layers on device %d\n",
                    actual_gpu2_layers, gpu2_device_cli);
    }
    // Third, fourth, ... graphics cards: each takes the layers right after the previous card
    int actual_gpux_total = 0; std::vector<std::pair<int,int>> actual_gpux;   // {device, layers} per extra card
    {
        int next_first = actual_gpu_layers + actual_gpu2_layers;
        for(auto& kv : gpux_layers_cli){
            auto dv = gpux_device_cli.find(kv.first);
            if(kv.second == 0 || kv.second < -1 || dv == gpux_device_cli.end() || dv->second < 0) continue;
            int got = S.engine.add_gpu_layers(kv.second, dv->second, next_first);
            if(got > 0){
                fprintf(stderr,"[server] GPU %d: %d layers on device %d (layers %d-%d)\n", kv.first, got, dv->second, next_first, next_first+got-1);
                actual_gpux.push_back({dv->second, got}); next_first += got; actual_gpux_total += got;
            } else if (kv.second > 0) fprintf(stderr,"[server] GPU %d: could not place %d layers on device %d\n", kv.first, kv.second, dv->second);
            else fprintf(stderr,"[server] GPU %d: no layers fit on device %d\n", kv.first, dv->second);
        }
    }

    // Prompt prefill policy:
    //   default                      -> CPU prompt batch, no turbo GPU stream/evict
    //   --turbo-prefill              -> turbo GPU stream/layer resident path
    //   --prefill-token-by-token     -> old/debug forward() loop
    S.engine.set_prefill_token_by_token(prefill_token_by_token_cli);
    if(S.has_engine2) S.engine2.set_prefill_token_by_token(prefill_token_by_token_cli);

    {
        // Plain-language summary so nobody has to guess where the work happens.
        const int gpu_total = actual_gpu_layers + actual_gpu2_layers + actual_gpux_total;
        if(prefill_token_by_token_cli)
            fprintf(stderr,"[server] Prompt reading: one token at a time (debug mode). Layers run where they live.\n");
        else if(turbo_prefill_cli)
            fprintf(stderr,"[server] Prompt reading: Turbo Prefill Acceleration ON -- the GPU does the heavy math for ALL layers by borrowing them from system RAM.\n");
        else if(gpu_total > 0)
            fprintf(stderr,"[server] Prompt reading: normal -- the %d layers on the GPU are read on the GPU, any other layers on the CPU.\n", gpu_total);
        else
            fprintf(stderr,"[server] Prompt reading: normal -- no layers are on a GPU, so everything runs on the CPU.\n");
    }

    const bool turbo_stream_startup = turbo_prefill_cli || turbo_gpu_stream_cli;
    if(turbo_vram_budget_mb_cli > 0){
        S.engine.set_turbo_vram_budget_mb(turbo_vram_budget_mb_cli);
        if(S.has_engine2) S.engine2.set_turbo_vram_budget_mb(turbo_vram_budget_mb_cli);
    }

    if(prefill_token_by_token_cli){
        S.engine.set_turbo_prefill(false);
        S.engine.set_turbo_gpu_stream(false);
        if(S.has_engine2){
            S.engine2.set_turbo_prefill(false);
            S.engine2.set_turbo_gpu_stream(false);
        }
    } else {
        int dev = (turbo_gpu_device_cli >= 0) ? turbo_gpu_device_cli : gpu_device_cli;
        if(dev >= 0){
            S.engine.set_turbo_gpu_device(dev);
            if(S.has_engine2) S.engine2.set_turbo_gpu_device(dev);
        }

        if(turbo_stream_startup){
            S.engine.set_turbo_gpu_stream(true);
            if(S.has_engine2) S.engine2.set_turbo_gpu_stream(true);
            fprintf(stderr,"[server] Turbo GPU streaming requested on startup (device %d)\n", dev);
        }

        if(turbo_prefill_cli){
            S.engine.set_turbo_prefill(true);
            if(S.has_engine2) S.engine2.set_turbo_prefill(true);

            if(S.engine.turbo_prefill_enabled() && S.engine.turbo_gpu_stream()){
                fprintf(stderr,"[server] Turbo prefill ENABLED on startup using GPU stream\n");
            } else {
                fprintf(stderr,
                    "[server] Turbo prefill requested but GPU stream is OFF; CPU prompt-batch remains active\n");
            }
        }
    }

    if (park_wanted && turbo_prefill_cli && !prefill_token_by_token_cli) {
        if (gpu_layers_cli > 0)
            fprintf(stderr, "[park] OFF: layers already live on the card the usual way (--gpu-layers %d); set it to 0 to use Maximize GPU\n", gpu_layers_cli);
        else {
            S.engine.set_mtp_card_reserve(S.mtp_enabled && S.mtp_gpu);
            S.engine.park_setup(park_layers_cli, park_below_mb_cli, park_keep_mb_cli, park_prefill_free_mb_cli, park_min_tokens_cli);
        }
    }

    auto info=S.engine.model_info();
    printf("\n  Model:   %s\n",info.name.empty()?model.c_str():info.name.c_str());
    printf("  Format:  %s\n",info.chat_format.c_str());
    printf("  Threads: %d\n",threads);
    if(actual_gpu_layers > 0){
        std::string gpu_name = S.engine.gpu_device_name();
        bool zero_copy = S.engine.gpu_zero_copy();
        printf("  GPU 1:   %s - layers 0-%d%s\n",
               gpu_name.empty()?"(unknown)":gpu_name.c_str(),
               actual_gpu_layers-1,
               zero_copy?" (shared RAM, zero-copy -)":"");
    }
    if(actual_gpu2_layers > 0){
        std::string gpu2_name = S.engine.gpu2_device_name();
        printf("  GPU 2:   %s - layers %d-%d (shared RAM)\n",
               gpu2_name.empty()?"(iGPU)":gpu2_name.c_str(),
               actual_gpu_layers, actual_gpu_layers+actual_gpu2_layers-1);
    }
    { int at = actual_gpu_layers + actual_gpu2_layers;
      for(size_t xi=0; xi<actual_gpux.size(); ++xi){
          std::string nm = S.engine.gpux_device_name((int)xi);
          printf("  GPU %d:   %s - layers %d-%d\n", (int)xi+3, nm.empty()?"(unknown)":nm.c_str(), at, at+actual_gpux[xi].second-1);
          at += actual_gpux[xi].second;
      } }
    int cpu_layers = (int)info.n_layers - actual_gpu_layers - actual_gpu2_layers - actual_gpux_total;
    if(cpu_layers > 0)
        printf("  CPU:     layers %d-%d (%d threads)\n",
               actual_gpu_layers+actual_gpu2_layers+actual_gpux_total, (int)info.n_layers-1,
               threads);
    printf("  Data:    %s\n\n",data.c_str());

    // Load warm start state AFTER model and GPU layers are fully initialised.
    // Loading before S.engine.load() is a no-op - engine reinitialises KV cache on load.
    if(!start_state_path.empty()){
        fprintf(stderr,"[server] Loading warm state: %s\n",start_state_path.c_str());
        std::lock_guard<std::mutex> lg(S.gen_mutex);
        if(!S.engine.load_start_state(start_state_path))
            fprintf(stderr,"[server] WARNING: warm state load failed - cold start\n");
        else{
            // Register conv_id so extend_turn+continue work without /api/chat
            // (mirrors exactly what /api/start_state/load does via the API)
            char _wid[40]; snprintf(_wid,sizeof(_wid),"ws_%lx",(unsigned long)time(nullptr));
            S.warm_conv_id = _wid;
            {
                std::lock_guard<std::mutex> cl(S.sessions_mu);
                // Try to read the sidecar .json for last_prompt
                std::string last_prompt;
                if(start_state_path.size()>4 && start_state_path.substr(start_state_path.size()-4)==".kvc"){
                    std::string meta=start_state_path.substr(0,start_state_path.size()-4)+".json";
                    FILE* mf=fopen(meta.c_str(),"r");
                    if(mf){
                        fseek(mf,0,SEEK_END);long msz=ftell(mf);rewind(mf);
                        if(msz>0&&msz<65536){std::string mb(msz,'\0');fread(&mb[0],1,msz,mf);last_prompt=jp_str(mb,"prompt","");}
                        fclose(mf);
                    }
                }
                S.conversations[S.warm_conv_id].last_prompt = last_prompt;
                S.conversations[S.warm_conv_id].n_tokens    = S.engine.context_size();
            }
            fprintf(stderr,"[server] Warm state loaded - context at %d tokens, conv_id=%s\n",
                    S.engine.context_size(), S.warm_conv_id.c_str());
            fflush(stderr);
        }
    }

    // -- Load secondary model if specified ------------------------------------
    if(!model2.empty()){
        printf("  Loading model 2: %s\n\n",model2.c_str());
        int m2_threads = model2_threads > 0 ? model2_threads : threads;
        auto lcb2=[](const nexus::LoadProgress& p){
            int pct=(int)(p.fraction*100),bars=pct/5;
            printf("\r  [2]");for(int i=0;i<20;++i)printf(i<bars?"#":"-");
            printf("] %3d%%  %-40s",pct,p.stage.substr(0,40).c_str());
            fflush(stdout);if(p.fraction>=1.f)printf("\n");
        };
        S.model_file[1] = model2;
        if(!S.engine2.load(model2, m2_threads, lcb2)){
            fprintf(stderr,"[server] WARNING: model2 load failed - running without it\n");
        } else {
            S.has_engine2 = true;
            if(!model2_chat_format.empty())
                S.engine2.override_chat_format(model2_chat_format);
            int m2_actual_gpu = 0;
            if(model2_gpu_layers != 0){
                int req = (model2_gpu_layers < 0) ? -1 : model2_gpu_layers;
                if(model2_gpu_device >= 0) S.engine2.set_gpu_device(model2_gpu_device);
                m2_actual_gpu = S.engine2.set_gpu_layers(req);
                S.engine2_n_gpu_layers = m2_actual_gpu;
            }
            auto i2=S.engine2.model_info();
            printf("  Model 2: %s\n",i2.name.empty()?model2.c_str():i2.name.c_str());
            printf("  Format:  %s\n",i2.chat_format.c_str());
            printf("  Threads: %d\n",m2_threads);
            if(m2_actual_gpu > 0){
                std::string gn=S.engine2.gpu_device_name();
                bool zc=S.engine2.gpu_zero_copy();
                printf("  GPU:     %s - all %d layers%s\n",
                       gn.empty()?"(unknown)":gn.c_str(), m2_actual_gpu,
                       zc?" (unified RAM)":"");
            } else {
                printf("  GPU:     none (CPU only)\n");
            }
            printf("\n");
        }
    }

    S.model_file[0] = model;
    if(!(S.dist_sess && S.dist_sess->is_connected())) nx_apply_ctx_limit(S, S.engine, "model 1");
    if(S.has_engine2) nx_apply_ctx_limit(S, S.engine2, "model 2");
    printf("  ------------------------------------------\n");
    printf("  -  Open - http://%s:%-5d              -\n",host.c_str(),port);
    printf("  ------------------------------------------\n\n");
    printf("  Ctrl+C to stop.\n\n");

#ifndef _WIN32
    signal(SIGPIPE,SIG_IGN);
#endif
#ifdef _WIN32
    SOCKET srv=socket(AF_INET,SOCK_STREAM,0);
    if(srv==INVALID_SOCKET){fprintf(stderr,"socket failed: %d\n",WSAGetLastError());return 1;}
#else
    int srv=socket(AF_INET,SOCK_STREAM,0);if(srv<0){perror("socket");return 1;}
#endif
    int yes=1;
#ifdef _WIN32
    // On Windows SO_REUSEADDR lets a second process bind the SAME port and silently
    // steal connections.  SO_EXCLUSIVEADDRUSE gives the Unix behaviour: a busy port
    // fails to bind with a clear error.
    setsockopt(srv,SOL_SOCKET,SO_EXCLUSIVEADDRUSE,(const char*)&yes,sizeof(yes));
#else
    setsockopt(srv,SOL_SOCKET,SO_REUSEADDR,(const char*)&yes,sizeof(yes));
#endif
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons((uint16_t)port);
    inet_pton(AF_INET,host.c_str(),&addr.sin_addr);
    fprintf(stderr,"[server] Binding to %s:%d\n",host.c_str(),port); fflush(stderr);
#ifdef _WIN32
    // Winsock errors are NOT in errno, so perror() prints "No error".  Use WSAGetLastError.
    auto nx_net_fail=[&](const char* what){
        int e=WSAGetLastError();
        fprintf(stderr,"[server] %s FAILED on %s:%d  (Winsock error %d)\n",what,host.c_str(),port,e);
        if(e==WSAEADDRINUSE)
            fprintf(stderr,"[server]   Port %d is already in use. Another Nexus (or program) is running on it.\n"
                           "[server]   Close it, or pick another port in the launcher.\n",port);
        else if(e==WSAEACCES)
            fprintf(stderr,"[server]   Windows refused access to port %d. It is usually reserved by Windows/Hyper-V/Docker/WSL\n"
                           "[server]   or held exclusively by another program.  Pick a different port (e.g. 18080) or run:\n"
                           "[server]     netsh interface ipv4 show excludedportrange protocol=tcp\n",port);
        else if(e==WSAEADDRNOTAVAIL)
            fprintf(stderr,"[server]   Address %s is not available on this machine. Use 127.0.0.1 or 0.0.0.0.\n",host.c_str());
        fflush(stderr);
    };
    if(bind(srv,(sockaddr*)&addr,sizeof(addr))==SOCKET_ERROR){nx_net_fail("bind");return 1;}
    if(listen(srv,64)==SOCKET_ERROR){nx_net_fail("listen");return 1;}
#else
    if(bind(srv,(sockaddr*)&addr,sizeof(addr))<0){perror("bind");return 1;}
    if(listen(srv,64)<0){perror("listen");return 1;}
#endif
    fprintf(stderr,"[server] Listening on %s:%d\n",host.c_str(),port); fflush(stderr);
    while(true){
        sockaddr_in ca{};socklen_t cl=sizeof(ca);
#ifdef _WIN32
        SOCKET fd=accept(srv,(sockaddr*)&ca,&cl);if(fd==INVALID_SOCKET)continue;
#else
        int fd=accept(srv,(sockaddr*)&ca,&cl);if(fd<0)continue;
#endif
#ifndef _WIN32
        { int ka=1; setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&ka,sizeof(ka));
#ifdef TCP_KEEPIDLE
          int idle=30,intv=10,cnt=6;
          setsockopt(fd,IPPROTO_TCP,TCP_KEEPIDLE,&idle,sizeof(idle));
          setsockopt(fd,IPPROTO_TCP,TCP_KEEPINTVL,&intv,sizeof(intv));
          setsockopt(fd,IPPROTO_TCP,TCP_KEEPCNT,&cnt,sizeof(cnt));
#endif
        }
#endif
        char pip[64]="";
#ifdef _WIN32
        inet_ntop(AF_INET,&ca.sin_addr,pip,sizeof(pip));
#else
        inet_ntop(AF_INET,&ca.sin_addr,pip,sizeof(pip));
#endif
        std::string peer=pip;
        std::thread([fd,&S,peer]{handle(fd,S,peer);}).detach();
    }
    close(srv);return 0;
}
#ifdef _WIN32
// WSACleanup called at process exit automatically
#endif
