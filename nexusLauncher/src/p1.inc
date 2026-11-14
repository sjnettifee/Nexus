// app.cpp -- the Nexus launcher: a native window (Dear ImGui drawn by a software renderer).
// Several model "slots" (each is its own nexus_server on its own port / card / threads), optional donor computers,
// a built-in chat, and a donor (worker) mode for the secondary computers.
//
//   nexus_launcher                       normal start
//   nexus_launcher --donor               start on the Donor tab (for a computer that only lends its power)
//   nexus_launcher --shot out.ppm        render one frame to a file and exit (testing; no window)
//        [--tab setup|chat|log|donor] [--slot N] [--w N --h N] [--expand] [--scale F]
#include "core.h"
#include <set>
#include "detect.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_stdlib.h"
#include "platform.h"
#include "sw_render.h"

#include <cmath>
#include <functional>
#include <regex>

#include "font_roboto.inc"

namespace {

struct Choice { std::string value, label; bool disabled = false; };
typedef std::map<std::string, std::string> SMap;

// ───────────────────────────── settings fields ───────────────────────────────
// "model" fields are remembered per model file name; "hw" fields belong to the slot (which card / how many threads).

const char* const MODEL_FIELDS[] = {
    "temp", "topp", "topk", "minp", "tfsz", "rep", "repn", "freq", "pres", "maxtok",
    "imgSize", "upEdge", "upPx", "tileSize", "tileStride", "maxTiles", "loraScale", "mtpN", "extraArgs", "extraEnv", "startState",
    "ctPre", "ctPreSys", "ctEndSys", "ctPreUser", "ctEndUser", "ctPreModel", "ctEndModel",
    "chatFmt", "rope", "norm", "thinking", "preserve", "vMode", "mtp", "mtpGpu", "visionGpu"};
const char* const HW_FIELDS[] = {
    "gpuSel", "gpuLayers", "gpu2Sel", "gpu2Layers", "threads", "myLayers", "prefillMode", "prefillGpu", "vram", "gpuWin", "attnPath", "tokByTok", "profile", "trace"};

// graphics cards: card 1 and 2 keep their old setting names, card 3, 4, ... are gpu3Sel / gpu3Layers ... (no upper limit that matters)
const int kMaxCards = 16;
std::string cardSel(int n) { return n == 1 ? "gpuSel" : n == 2 ? "gpu2Sel" : "gpu" + std::to_string(n) + "Sel"; }
std::string cardLay(int n) { return n == 1 ? "gpuLayers" : n == 2 ? "gpu2Layers" : "gpu" + std::to_string(n) + "Layers"; }

// what every box holds until the user changes it -- shown in the box, and sent to the server explicitly
std::string dflt(const std::string& k) {
    static const std::map<std::string, std::string> d = {
        {"temp", "0.2"}, {"topp", "0.95"}, {"topk", "40"}, {"minp", "0.05"}, {"tfsz", "1.0"}, {"rep", "1.1"}, {"repn", "128"}, {"freq", "0.0"}, {"pres", "0.0"}, {"maxtok", "32000"},
        {"imgSize", "768"}, {"upEdge", "0"}, {"upPx", "0"}, {"tileSize", "768"}, {"tileStride", "768"}, {"maxTiles", "0"}, {"loraScale", "1.0"}, {"mtpN", "1"},
        {"vMode", "single"}, {"mtp", "off"}, {"mtpGpu", "off"}, {"visionGpu", "1"},
        {"gpuSel", "cpu"}, {"gpuLayers", "0"}, {"gpu2Sel", "none"}, {"gpu2Layers", "0"}, {"vram", "0"}, {"gpuWin", "1"}, {"attnPath", "auto"}, {"trace", "0"},
        {"prefillMode", "auto"}, {"tokByTok", "0"}, {"profile", "0"}};
    auto it = d.find(k); return it == d.end() ? std::string() : it->second;
}

struct Peer { std::string host, port = "9876", pass, layers = "0"; bool local = false; };

struct LogView {
    std::vector<std::string> v; size_t next = 0; bool follow = true; int load_pct = -1; std::string load_stage;   // load_pct: -1 = not loading
    int sa_l = -1, sa_c = 0, sb_l = -1, sb_c = 0; bool dragging = false;                                           // mouse text selection: anchor / cursor = (line, byte)
    void clear_sel() { sa_l = sb_l = -1; sa_c = sb_c = 0; dragging = false; }
};

struct Slot {
    SMap S;                                         // every setting of this slot, as text ("1"/"0" for check boxes)
    std::string cur, custom, note, port;            // model file name, typed/browsed path, small note, server port
    int sel = -1;                                   // index into g_models for the drop-down
    int lora_sel = -1; std::string lora_custom;     // -1 none, -2 browsed, else index into g_loras
    bool dist_on = false; std::string dist_mode = "auto"; std::vector<Peer> peers;
    ModelMeta meta; Detected det{"ChatML", "low"};  // read from the model file when it is picked
    std::atomic<bool> ready{false}; bool was_ready = false;
    LogView log;
    std::string profile;                            // name of the profile in use for this model ("" = none)
};

JV g_cfg;                                           // whole nexus_launcher.json
Slot g_slots[kMaxSlots]; int g_ns = 1, g_cs = 0;
Slot& cs() { return g_slots[g_cs]; }
SMap* g_m = &g_slots[0].S;                          // the settings the widgets edit right now
SMap g_donor;                                       // settings of the Donor tab (passphrase is never saved)
LogView g_wlog;

std::vector<FileEnt> g_models, g_loras;
std::vector<std::string> g_dirs;                    // folders the user added
int g_lan = 0;
JV g_hw; std::mutex g_hw_mu; std::atomic<int> g_hw_state{0};   // 0 detecting, 1 ok, 2 failed
bool g_hw_applied = false;
std::vector<Choice> g_gpus, g_pgpus;                // all choices (with "CPU only") / cards only

double g_now = 0, g_save_at = -1;
std::string g_error;                                // text of the error popup
int g_req_tab = -1;                                 // 0 setup, 2 log, 3 donor
bool g_expand_all = false;
std::mutex g_peer_mu; std::map<std::string, int> g_peer_state;     // "host:port" -> 0 unknown, 1 reachable, 2 not reachable, 3 testing

struct Browse {
    bool open_req = false, open = false, folder = false;
    std::string title, path_edit; std::vector<std::string> exts;
    BrowseResult r; std::function<void(const std::string&)> cb;
} g_br;

// ───────────────────────────── config ────────────────────────────────────────

JV jstrv(const std::string& s) { JV v; v.t = JV::Str; v.s = s; return v; }
JV jboolv(bool b) { JV v; v.t = JV::Bool; v.b = b; return v; }
JV jnumv(double n) { JV v; v.t = JV::Num; v.n = n; return v; }
std::string trim(const std::string& s) { size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n"); return a == std::string::npos ? "" : s.substr(a, b - a + 1); }

std::string path_of(Slot& sl, int idx_unused = 0) {
    (void)idx_unused;
    return !sl.custom.empty() ? sl.custom : (sl.sel >= 0 && sl.sel < (int)g_models.size() ? join(g_models[sl.sel].dir, g_models[sl.sel].name) : std::string());
}
std::string lora_path(Slot& sl) {
    if (sl.lora_sel == -2) return sl.lora_custom;
    return sl.lora_sel >= 0 && sl.lora_sel < (int)g_loras.size() ? join(g_loras[sl.lora_sel].dir, g_loras[sl.lora_sel].name) : std::string();
}
// value of a setting: what is in the box, or its stated default when the box was emptied
std::string val(Slot& sl, const char* k) { std::string v = trim(sl.S[k]); return v.empty() ? dflt(k) : v; }

bool is_set(const SMap& m, const char* k) { auto it = m.find(k); return it != m.end(); }

bool g_dirty = false;                               // something changed since the last "Save current settings"
void save_file() {
    std::string out; jdump(g_cfg, out); out += "\n";
    if (!write_file_atomic(g_cfg_path, out)) fprintf(stderr, "[launcher] cannot write %s\n", g_cfg_path.c_str());
}
void save_profiles_of_active();                     // below: the profile in use is updated whenever the settings are saved
void save_now() {
    JV& g = *jset(g_cfg, "global");
    *jset(g, "lan") = jnumv(g_lan); *jset(g, "browser_cmd") = jstrv(g_browser_cmd); *jset(g, "nslots") = jnumv(g_ns); *jset(g, "cur_slot") = jnumv(g_cs);
    JV arr; arr.t = JV::Arr; for (auto& d : g_dirs) arr.a.push_back(jstrv(d));
    *jset(g, "model_dirs") = arr;
    *jset(g_cfg, "version") = jnumv(3);
    JV slots; slots.t = JV::Arr;
    for (int i = 0; i < g_ns; ++i) {
        Slot& sl = g_slots[i]; JV s; s.t = JV::Obj;
        *jset(s, "path") = jstrv(path_of(sl)); *jset(s, "port") = jstrv(sl.port);
        JV hw; hw.t = JV::Obj;
        for (auto* f : HW_FIELDS) *jset(hw, f) = jstrv(sl.S[f]);
        for (int n = 3; n <= kMaxCards; ++n) { *jset(hw, cardSel(n).c_str()) = jstrv(sl.S[cardSel(n)]); *jset(hw, cardLay(n).c_str()) = jstrv(sl.S[cardLay(n)]); }
        *jset(s, "hw") = hw;
        JV d; d.t = JV::Obj; *jset(d, "on") = jboolv(sl.dist_on); *jset(d, "mode") = jstrv(sl.dist_mode);
        JV pa; pa.t = JV::Arr;
        for (auto& p : sl.peers) { JV x; x.t = JV::Obj; *jset(x, "host") = jstrv(p.host); *jset(x, "port") = jstrv(p.port); *jset(x, "pass") = jstrv(p.pass); *jset(x, "layers") = jstrv(p.layers); *jset(x, "local") = jboolv(p.local); pa.a.push_back(x); }
        *jset(d, "peers") = pa; *jset(s, "dist") = d;
        slots.a.push_back(s);
        if (!sl.cur.empty()) {
            JV& m = *jset(*jset(g_cfg, "models"), sl.cur);
            for (auto* f : MODEL_FIELDS) *jset(m, f) = jstrv(sl.S[f]);
            std::string lp = lora_path(sl);
            *jset(m, "loraPath") = jstrv(lp); *jset(m, "loraName") = jstrv(lp.empty() ? "" : base_name(lp));
        }
    }
    *jset(g_cfg, "slots") = slots;
    JV dn; dn.t = JV::Obj;                                            // donor settings, but never the passphrase
    for (auto& kv : g_donor) if (kv.first != "pass") *jset(dn, kv.first) = jstrv(kv.second);
    *jset(g_cfg, "donor") = dn;
    save_profiles_of_active();
    save_file(); g_dirty = false;
}
void changed() { g_dirty = true; }

// ───────────────────────────── hardware ──────────────────────────────────────

int physical_cores() {
    std::lock_guard<std::mutex> lk(g_hw_mu);
    const JV* cpu = g_hw.get("cpu"); if (!cpu) return std::max(1, (int)std::thread::hardware_concurrency());
    const JV* p = cpu->get("physical"); const JV* l = cpu->get("logical");
    if (p && p->n > 0) return (int)p->n;
    return l && l->n > 0 ? (int)l->n : std::max(1, (int)std::thread::hardware_concurrency());
}
std::map<std::string, long long> g_gpu_vram_mb;       // card value -> its own memory in MB (from the hardware scan)
std::set<std::string> g_gpu_integrated;             // card values that are integrated graphics (shared memory), e.g. a laptop APU
void rebuild_gpu_list() {
    g_gpus.clear(); g_pgpus.clear(); g_gpu_integrated.clear(); g_gpu_vram_mb.clear(); g_gpus.push_back({"cpu", "CPU only", false});
    std::lock_guard<std::mutex> lk(g_hw_mu);
    const JV* v = g_hw.get("vulkan"); const JV* d = v ? v->get("devices") : nullptr;
    if (d) for (auto& x : d->a) {
        const JV* u = x.get("usable"); bool usable = u && u->b;
        std::string why = jstr(x.get("reason"));
        Choice c{jstr(x.get("index")), "[" + jstr(x.get("index")) + "] " + jstr(x.get("name")) + (usable ? "" : "  (auto-detect says no" + (why.empty() ? std::string() : ": " + why) + " - you can still use it)"), false};
        g_gpus.push_back(c); g_pgpus.push_back(c);
        { const JV* vb = x.get("vram_bytes"); if (vb) g_gpu_vram_mb[c.value] = (long long)(vb->n / 1048576.0); }
        if (lower(jstr(x.get("type"))).find("discrete") == std::string::npos) g_gpu_integrated.insert(c.value);
    }
}
std::string gbs(double bytes) { char b[32]; snprintf(b, sizeof b, bytes >= 1e10 ? "%.0f GB" : "%.1f GB", bytes / 1073741824.0); return b; }

// make sure every slot's hardware boxes hold real values and name cards that exist on this computer
void fix_hw(Slot& sl) {
    if (trim(sl.S["threads"]).empty()) sl.S["threads"] = std::to_string(physical_cores());
    for (auto* f : HW_FIELDS) if (trim(sl.S[f]).empty() && !dflt(f).empty()) sl.S[f] = dflt(f);
    auto exists = [&](const std::vector<Choice>& l, const std::string& v) { for (auto& c : l) if (c.value == v) return true; return false; };
    const bool probed = g_hw_state.load() == 1;       // before the hardware check has finished (or if it failed) the saved choices are left alone
    if (probed && !exists(g_gpus, sl.S["gpuSel"])) sl.S["gpuSel"] = "cpu";
    for (int n = 2; n <= kMaxCards; ++n) {
        std::string& v = sl.S[cardSel(n)]; if (v.empty() || (probed && !exists(g_pgpus, v))) v = "none";
        if (trim(sl.S[cardLay(n)]).empty()) sl.S[cardLay(n)] = "0";
    }
    if (sl.S["prefillMode"] != "1" && sl.S["prefillMode"] != "0") sl.S["prefillMode"] = "auto";
    if (probed && !exists(g_pgpus, sl.S["prefillGpu"])) { sl.S["prefillGpu"] = g_pgpus.empty() ? std::string() : g_pgpus[0].value; for (auto& c : g_pgpus) if (!g_gpu_integrated.count(c.value)) { sl.S["prefillGpu"] = c.value; break; } }
}

// layers placed on graphics cards by the "Run on graphics card(s)" section
int card_layers(Slot& sl) {
    if (sl.S["gpuSel"] == "cpu") return 0;
    int t = std::max(0, atoi(sl.S["gpuLayers"].c_str()));
    for (int n = 2; n <= kMaxCards; ++n) { if (sl.S[cardSel(n)] == "none" || sl.S[cardSel(n)].empty()) break; t += std::max(0, atoi(sl.S[cardLay(n)].c_str())); }
    return t;
}
// Prefill acceleration only has to fit ONE layer on a card, so almost any card will do - but it is pointless when the whole
// model already lives on the cards (nothing to swap), and integrated graphics (shared memory) gave wrong answers in testing.
// "" = Auto turns it on; otherwise the reason it stays off.
std::string prefill_auto_reason(Slot& sl) {
    if (g_pgpus.empty() || sl.S["prefillGpu"].empty()) return "no graphics card found";
    if (sl.meta.layers > 0 && card_layers(sl) >= sl.meta.layers) return "the whole model is already on the card(s), nothing to swap";
    if (g_gpu_integrated.count(sl.S["prefillGpu"])) return "that card is integrated graphics - choose it by hand if you want to try";
    return "";
}
bool prefill_on(Slot& sl) {
    if (g_pgpus.empty() || sl.S["prefillGpu"].empty()) return false;
    const std::string m = sl.S["prefillMode"];
    if (m == "1") return true;
    if (m == "0") return false;
    return prefill_auto_reason(sl).empty();
}

// ───────────────────────────── models / attach ───────────────────────────────

void scan_all() {
    g_models.clear(); g_loras.clear();
    std::vector<std::string> roots;                                  // user folders first: their choice wins
    for (auto& d : g_dirs) if (is_dir(d)) roots.push_back(d);
    roots.push_back(join(g_dir, "models")); roots.push_back(join(join(g_dir, ".."), "models"));
    for (auto& d : roots) if (is_dir(d)) {
        size_t from = g_models.size(); scan_models(d, 2, g_models);
        std::sort(g_models.begin() + (long)from, g_models.end(), [](const FileEnt& a, const FileEnt& b) { return lower(a.dir + "/" + a.name) < lower(b.dir + "/" + b.name); });
    }
    scan_models(join(g_dir, "loras"), 1, g_loras, true);
    std::sort(g_loras.begin(), g_loras.end(), [](const FileEnt& a, const FileEnt& b) { return lower(a.name) < lower(b.name); });
}

void refresh_meta(Slot& sl) {
    std::string p = path_of(sl); sl.meta = ModelMeta();
    if (!p.empty() && is_file(p)) sl.meta = read_model_meta(p);
    sl.det = p.empty() ? Detected{"ChatML", "low"} : detect_format(sl.meta, p);
}

// ───────────────────────────── profiles ──────────────────────────────────────
// A profile = every setting of one model (sampling, graphics cards, threads, extra variables ...) under a name.
// Kept in nexus_launcher.json: "profiles" { model file name : { profile name : {...} } }, "auto_profile" { model : profile }.
std::vector<std::string> profile_names(const std::string& model) {
    std::vector<std::string> r; const JV* ps = g_cfg.get("profiles"); const JV* m = ps ? ps->get(model) : nullptr;
    if (m) for (auto& kv : m->o) r.push_back(kv.first);
    std::sort(r.begin(), r.end()); return r;
}
JV profile_snapshot(Slot& sl) {
    JV p; p.t = JV::Obj;
    for (auto* f : MODEL_FIELDS) *jset(p, f) = jstrv(sl.S[f]);
    for (auto* f : HW_FIELDS)    *jset(p, f) = jstrv(sl.S[f]);
    for (int n = 3; n <= kMaxCards; ++n) { *jset(p, cardSel(n).c_str()) = jstrv(sl.S[cardSel(n)]); *jset(p, cardLay(n).c_str()) = jstrv(sl.S[cardLay(n)]); }
    *jset(p, "port") = jstrv(sl.port);
    return p;
}
void profile_apply(Slot& sl, const JV& p) {
    for (auto* f : MODEL_FIELDS) { const JV* v = p.get(f); if (v) { std::string x = jstr(v); sl.S[f] = x.empty() ? dflt(f) : x; } }
    for (auto* f : HW_FIELDS)    { const JV* v = p.get(f); if (v) sl.S[f] = jstr(v); }
    for (int n = 3; n <= kMaxCards; ++n) { const JV* a = p.get(cardSel(n).c_str()); const JV* b = p.get(cardLay(n).c_str()); if (a) sl.S[cardSel(n)] = jstr(a); if (b) sl.S[cardLay(n)] = jstr(b); }
    const JV* pt = p.get("port"); if (pt && !jstr(pt).empty()) sl.port = jstr(pt);
    fix_hw(sl);
}
bool profile_load(Slot& sl, const std::string& name) {
    const JV* ps = g_cfg.get("profiles"); const JV* m = ps ? ps->get(sl.cur) : nullptr; const JV* p = m ? m->get(name) : nullptr;
    if (!p) return false;
    profile_apply(sl, *p); sl.profile = name; return true;
}
void profile_save(Slot& sl, const std::string& name) {   // writes only the profile; nothing else is saved
    if (sl.cur.empty() || name.empty()) return;
    *jset(*jset(*jset(g_cfg, "profiles"), sl.cur), name) = profile_snapshot(sl);
    sl.profile = name; save_file();
}
void profile_delete(Slot& sl, const std::string& name) {
    JV* ps = jset(g_cfg, "profiles"); JV* m = jset(*ps, sl.cur);
    m->o.erase(std::remove_if(m->o.begin(), m->o.end(), [&](const std::pair<std::string, JV>& kv) { return kv.first == name; }), m->o.end());
    JV* au = jset(g_cfg, "auto_profile"); const JV* a = au->get(sl.cur);
    if (a && jstr(a) == name) au->o.erase(std::remove_if(au->o.begin(), au->o.end(), [&](const std::pair<std::string, JV>& kv) { return kv.first == sl.cur; }), au->o.end());
    if (sl.profile == name) sl.profile.clear();
    save_file();
}
std::string profile_auto_of(const std::string& model) { const JV* au = g_cfg.get("auto_profile"); const JV* a = au ? au->get(model) : nullptr; return a ? jstr(a) : std::string(); }
void profile_set_auto(Slot& sl, bool on) {
    JV* au = jset(g_cfg, "auto_profile");
    au->o.erase(std::remove_if(au->o.begin(), au->o.end(), [&](const std::pair<std::string, JV>& kv) { return kv.first == sl.cur; }), au->o.end());
    if (on && !sl.profile.empty()) *jset(*au, sl.cur) = jstrv(sl.profile);
    save_file();
}
void save_profiles_of_active() {                    // "Save current settings" also updates the profile that is in use
    for (int i = 0; i < g_ns; ++i) { Slot& sl = g_slots[i]; if (!sl.profile.empty() && !sl.cur.empty()) *jset(*jset(*jset(g_cfg, "profiles"), sl.cur), sl.profile) = profile_snapshot(sl); }
}

void load_model_settings(Slot& sl) {
    const JV* m = nullptr;
    if (!sl.cur.empty()) { const JV* ms = g_cfg.get("models"); if (ms) m = ms->get(sl.cur); }
    for (auto* f : MODEL_FIELDS) { const JV* v = m ? m->get(f) : nullptr; std::string x = v ? jstr(v) : ""; sl.S[f] = x.empty() ? dflt(f) : x; }
    sl.lora_sel = -1; sl.lora_custom.clear();
    std::string lp = m ? jstr(m->get("loraPath")) : "", ln = m ? jstr(m->get("loraName")) : "";
    if (!lp.empty()) {
        for (size_t i = 0; i < g_loras.size(); ++i) if (join(g_loras[i].dir, g_loras[i].name) == lp) sl.lora_sel = (int)i;
        if (sl.lora_sel < 0 && is_file(lp)) { sl.lora_sel = -2; sl.lora_custom = lp; }
        if (sl.lora_sel < 0) for (size_t i = 0; i < g_loras.size(); ++i) if (g_loras[i].name == ln) sl.lora_sel = (int)i;
    }
    refresh_meta(sl);
    sl.profile.clear();
    const std::string ap = profile_auto_of(sl.cur);
    if (!ap.empty()) profile_load(sl, ap);          // this model has a profile set to apply automatically
}

// 1) the saved path, if that file exists on THIS computer -> the user's own choice always wins
// 2) the saved model NAME in any searched folder (user folders first)  -> a USB stick moved to another computer
// 3) slot 1 only: the first model found
void attach(Slot& sl, int idx, const std::string& saved_path) {
    sl.cur.clear(); sl.custom.clear(); sl.sel = -1; sl.note.clear();
    if (!saved_path.empty() && is_file(saved_path)) {
        for (size_t i = 0; i < g_models.size(); ++i) if (join(g_models[i].dir, g_models[i].name) == saved_path) sl.sel = (int)i;
        if (sl.sel < 0) sl.custom = saved_path;
        sl.cur = base_name(saved_path);
    } else {
        std::string name = saved_path.empty() ? "" : base_name(saved_path); int hit = -1;
        for (size_t i = 0; i < g_models.size(); ++i) if (!name.empty() && g_models[i].name == name) { hit = (int)i; break; }
        if (hit >= 0) { sl.sel = hit; sl.cur = g_models[hit].name; sl.note = "Your saved location was not found on this computer, so \"" + sl.cur + "\" was picked up from " + g_models[hit].dir + "."; }
        else if (idx == 0 && !g_models.empty()) { sl.sel = 0; sl.cur = g_models[0].name; if (!name.empty()) sl.note = "Saved model \"" + name + "\" is not on this computer. Showing the first model found."; }
    }
    load_model_settings(sl);
}

void on_model_picked(Slot& sl) {              // user changed the model by hand
    std::string p = path_of(sl);
    sl.cur = p.empty() ? "" : base_name(p); sl.note.clear();
    load_model_settings(sl); changed();
}

void init_slot(int i, const JV* rec) {
    Slot& sl = g_slots[i];
    sl.S.clear(); sl.peers.clear(); sl.dist_on = false; sl.dist_mode = "auto";
    sl.port = std::to_string(8080 + i);
    std::string path;
    if (rec) {
        path = jstr(rec->get("path")); std::string p = jstr(rec->get("port")); if (!p.empty()) sl.port = p;
        if (const JV* hw = rec->get("hw")) { for (auto* f : HW_FIELDS) sl.S[f] = jstr(hw->get(f)); for (int n = 3; n <= kMaxCards; ++n) { sl.S[cardSel(n)] = jstr(hw->get(cardSel(n).c_str())); sl.S[cardLay(n)] = jstr(hw->get(cardLay(n).c_str())); } }
        if (const JV* d = rec->get("dist")) {
            sl.dist_on = d->get("on") && d->get("on")->b; std::string mo = jstr(d->get("mode")); if (!mo.empty()) sl.dist_mode = mo;
            if (const JV* pa = d->get("peers")) for (auto& x : pa->a) {
                Peer p2; p2.host = jstr(x.get("host")); p2.port = jstr(x.get("port"), "9876"); p2.pass = jstr(x.get("pass")); p2.layers = jstr(x.get("layers"), "0"); p2.local = x.get("local") && x.get("local")->b;
                sl.peers.push_back(p2);
            }
        }
    }
    attach(sl, i, path); fix_hw(sl);
}

// ───────────────────────────── starting a server ─────────────────────────────

std::vector<std::string> split_args(const std::string& s) {
    std::vector<std::string> out; std::string cur; bool in = false; char q = 0;
    for (char c : s) {
        if (q) { if (c == q) q = 0; else cur += c; }
        else if (c == '"' || c == '\'') { q = c; in = true; }
        else if (isspace((unsigned char)c)) { if (in || !cur.empty()) { out.push_back(cur); cur.clear(); in = false; } }
        else { cur += c; in = true; }
    }
    if (in || !cur.empty()) out.push_back(cur);
    return out;
}

// donors found on the local network (port scan of this computer's /24)
std::vector<std::string> g_found; std::mutex g_found_mu; std::atomic<int> g_scan_state{0};   // 0 idle, 1 scanning, 2 done
void scan_lan(int port) {
    if (g_scan_state.load() == 1) return;
    g_scan_state = 1; { std::lock_guard<std::mutex> lk(g_found_mu); g_found.clear(); }
    std::thread([port] {
        std::string ip = local_ip(); size_t d = ip.rfind('.');
        if (d != std::string::npos) {
            std::string base = ip.substr(0, d + 1); std::vector<std::thread> th; std::atomic<int> next{1};
            for (int t = 0; t < 32; ++t) th.emplace_back([&] {
                for (int i; (i = next.fetch_add(1)) <= 254;) {
                    std::string h = base + std::to_string(i);
                    if (h != ip && tcp_reachable(h, port, 400)) { std::lock_guard<std::mutex> lk(g_found_mu); g_found.push_back(h); }
                }
            });
            for (auto& x : th) x.join();
        }
        g_scan_state = 2;
    }).detach();
}

// how the layers are shared: [start,end) for this computer first, then each donor
bool compute_splits(Slot& sl, std::vector<std::pair<int, int>>& out, std::string& err) {
    int n = sl.meta.layers; size_t np = sl.peers.size();
    if (n <= 0) { err = "The number of layers in this model could not be read, so it cannot be shared between computers."; return false; }
    if ((int)np + 1 > n) { err = "There are more computers than layers."; return false; }
    std::vector<int> cnt(np + 1, 0);
    if (sl.dist_mode == "manual") {
        int sum = 0; for (size_t i = 0; i < np; ++i) { cnt[i + 1] = atoi(trim(sl.peers[i].layers).c_str()); if (cnt[i + 1] < 1) { err = "Manual split: every donor computer needs at least 1 layer."; return false; } sum += cnt[i + 1]; }
        if (sum >= n) { err = "Manual split: the donors take " + std::to_string(sum) + " layers but the model only has " + std::to_string(n) + ". This computer needs at least 1."; return false; }
        const int mine = atoi(trim(sl.S["myLayers"]).c_str());
        if (mine > 0 && mine + sum != n) { err = "Manual split: this computer " + std::to_string(mine) + " + donors " + std::to_string(sum) + " = " + std::to_string(mine + sum) + ", but the model has " + std::to_string(n) + " layers. (Clear the This computer box to let it take the rest.)"; return false; }
        cnt[0] = n - sum;
    } else {
        int base = n / (int)(np + 1); int sum = 0;
        for (size_t i = 0; i < np; ++i) { cnt[i + 1] = base; sum += base; }
        cnt[0] = n - sum;
    }
    int at = 0; out.clear();
    for (size_t i = 0; i <= np; ++i) { out.push_back({at, at + cnt[i]}); at += cnt[i]; }
    return true;
}
std::string split_text(const std::vector<std::pair<int, int>>& sp) {
    std::string t;
    for (size_t i = 0; i < sp.size(); ++i) { if (i) t += "   |   "; t += (i == 0 ? "This computer: " : "Donor " + std::to_string(i) + ": ") + std::to_string(sp[i].first) + "-" + std::to_string(sp[i].second - 1) + " (" + std::to_string(sp[i].second - sp[i].first) + " layers)"; }
    return t;
}

bool build_start(int idx, std::vector<std::string>& a, std::vector<std::pair<std::string, std::string>>& env, std::string& err) {
    Slot& sl = g_slots[idx];
    std::string path = path_of(sl);
    if (path.empty()) { err = "Model " + std::to_string(idx + 1) + ": choose a model first."; return false; }
    if (!is_file(path)) { err = "That model file does not exist:\n" + path; return false; }
    auto V = [&](const char* k) { return val(sl, k); };
    auto E = [&](const char* k, const std::string& v) { env.emplace_back(k, v); };
    int port = atoi(sl.port.c_str()); if (port < 1 || port > 65535) { err = "Model " + std::to_string(idx + 1) + ": the server port must be 1-65535."; return false; }
    for (int j = 0; j < g_ns; ++j) if (j != idx && child_alive(g_children[j]) && atoi(g_slots[j].port.c_str()) == port) { err = "Port " + std::to_string(port) + " is already used by model " + std::to_string(j + 1) + ". Give each model its own port."; return false; }
    a = {"--model", path, "--port", std::to_string(port), "--host", g_lan ? "0.0.0.0" : "127.0.0.1", "--roster-file", join(g_dir, "nexus-roster.json")};
    a.push_back("--threads"); a.push_back(std::to_string(std::max(1, atoi(V("threads").c_str()))));
    // graphics cards
    const std::string gpu = sl.S["gpuSel"];
    const bool pf = prefill_on(sl);
    const std::string dev = gpu != "cpu" ? gpu : (pf ? sl.S["prefillGpu"] : std::string());
    a.push_back("--gpu-layers"); a.push_back(gpu == "cpu" ? "0" : std::to_string(std::max(0, atoi(V("gpuLayers").c_str()))));
    if (!dev.empty()) { a.push_back("--gpu-device"); a.push_back(dev); }
    if (sl.S["gpu2Sel"] != "none" && atoi(V("gpu2Layers").c_str()) > 0) { a.push_back("--gpu2-layers"); a.push_back(std::to_string(atoi(V("gpu2Layers").c_str()))); a.push_back("--gpu2-device"); a.push_back(sl.S["gpu2Sel"]); }
    for (int n = 3; n <= kMaxCards; ++n)          // cards 3, 4, ...: sent in the same style; the engine decides what it can place
        if (sl.S[cardSel(n)] != "none" && !sl.S[cardSel(n)].empty() && atoi(V(cardLay(n).c_str()).c_str()) > 0) {
            a.push_back("--gpu" + std::to_string(n) + "-layers"); a.push_back(std::to_string(atoi(V(cardLay(n).c_str()).c_str())));
            a.push_back("--gpu" + std::to_string(n) + "-device"); a.push_back(sl.S[cardSel(n)]);
        }
    if (pf) {
        a.push_back("--turbo-prefill"); a.push_back("--turbo-gpu-stream");
        if (atoi(V("gpuWin").c_str()) > 1) E("NEXUS_TURBO_GPU_LAYERS", std::to_string(atoi(V("gpuWin").c_str())));
        if (V("attnPath") != "auto") E("NEXUS_ATTN_PATH", V("attnPath"));
    }
    bool tbt = sl.S["tokByTok"] == "1";
#ifndef _WIN32
    // Layers sent to a discrete card live in the card's memory ONLY (no second copy in RAM).  The RAM copy is kept only
    // for the prompt speed-up (prefill acceleration), which is the one thing that needs it.  Without a RAM copy the prompt
    // is read through the card, token by token.
    if (!pf && gpu != "cpu" && !g_gpu_integrated.count(gpu) && atoi(V("gpuLayers").c_str()) > 0) {
        E("NEXUS_DIRECT_GPU_LAYERS", std::to_string(atoi(V("gpuLayers").c_str())));
        // The card's size lets the loader read only the layers that fit in place; the rest are loaded page-aligned so the card can still compute them from RAM.
        auto vm = g_gpu_vram_mb.find(gpu); if (vm != g_gpu_vram_mb.end() && vm->second > 0) E("NEXUS_DIRECT_VRAM_MB", std::to_string(vm->second));
    }
#endif
    if (tbt) a.push_back("--prefill-token-by-token");
    if (sl.S["profile"] == "1") E("NEXUS_PREFILL_PROFILE", "1");
    // chat template: always stated explicitly, so the server never guesses differently from what the window shows
    std::string fmt = trim(sl.S["chatFmt"]); if (fmt.empty()) fmt = sl.det.fmt;
    a.push_back("--chat-format"); a.push_back(fmt);
    if (fmt == "Custom") {
        const char* k[] = {"ctPre", "ctPreSys", "ctEndSys", "ctPreUser", "ctEndUser", "ctPreModel", "ctEndModel"};
        const char* fl[] = {"--chat-custom-pre-prompt", "--chat-custom-pre-system", "--chat-custom-end-system", "--chat-custom-pre-user", "--chat-custom-end-user", "--chat-custom-pre-model", "--chat-custom-end-model"};
        for (int i = 0; i < 7; ++i) { a.push_back(fl[i]); a.push_back(sl.S[k[i]]); }
    }
    // sampling: every value stated explicitly
    const char* sf[][2] = {{"temp", "--temperature"}, {"topp", "--top-p"}, {"topk", "--top-k"}, {"minp", "--min-p"}, {"tfsz", "--tfs-z"}, {"rep", "--rep-penalty"},
                           {"repn", "--repeat-last-n"}, {"freq", "--freq-penalty"}, {"pres", "--presence-penalty"}, {"maxtok", "--max-tokens"}};
    for (auto& f : sf) { a.push_back(f[1]); a.push_back(V(f[0])); }
    a.push_back("--mtp"); a.push_back(V("mtp")); a.push_back("--mtp-n-predict"); a.push_back(V("mtpN")); a.push_back("--mtp-gpu"); a.push_back(V("mtpGpu"));
    std::string lp = lora_path(sl);
    if (!lp.empty()) { a.push_back("--lora"); a.push_back(lp); a.push_back("--lora-scale"); a.push_back(V("loraScale")); }
    std::string ss = trim(sl.S["startState"]);
    if (!ss.empty()) { if (!is_file(ss)) { err = "The saved memory file does not exist:\n" + ss; return false; } a.push_back("--start-state"); a.push_back(ss); }
    a.push_back("--trace-level"); a.push_back(V("trace")); E("NEXUS_TRACE", V("trace"));
    if (!trim(sl.S["rope"]).empty()) E("NEXUS_ROPE_LAYOUT", sl.S["rope"]);
    if (!trim(sl.S["norm"]).empty()) E("NEXUS_NORM_CONVENTION", sl.S["norm"]);
    if (supports_thinking(sl.meta, fmt)) {
        if (!trim(sl.S["thinking"]).empty()) { E("NEXUS_ENABLE_THINKING", sl.S["thinking"]); E("NEXUS_QWEN36_THINKING", sl.S["thinking"]); }
        if (!trim(sl.S["preserve"]).empty()) { E("NEXUS_PRESERVE_THINKING", sl.S["preserve"]); E("NEXUS_QWEN36_PRESERVE_THINKING", sl.S["preserve"]); }
    }
    { int v = atoi(V("imgSize").c_str()); v = std::max(64, v / 32 * 32); E("NEXUS_QWEN_NATIVE_IMAGE_SIZE", std::to_string(v)); }
    E("NEXUS_WEB_IMAGE_MAX_EDGE", V("upEdge")); E("NEXUS_WEB_IMAGE_MAX_PIXELS", V("upPx"));
    E("NEXUS_WEB_IMAGE_FULL_SIZE", (V("upEdge") == "0" && V("upPx") == "0") ? "1" : "0");
    E("NEXUS_VISION_MODE", V("vMode")); E("NEXUS_VISION_TILE_SIZE", V("tileSize")); E("NEXUS_VISION_TILE_STRIDE", V("tileStride")); E("NEXUS_VISION_MAX_TILES", V("maxTiles"));
    E("NEXUS_VISION_GPU", (sl.S["visionGpu"] == "1" || pf) ? "1" : "0");
    // other computers
    if (sl.dist_on) {
        std::vector<Peer> ok; for (auto& p : sl.peers) if (!trim(p.host).empty()) ok.push_back(p);
        // Safety: if sharing is switched on we must never quietly load the whole model here.
        if (ok.empty()) { err = "Model " + std::to_string(idx + 1) + ": sharing between computers is switched on, but no donor address is filled in. Nothing was started, so this computer does not load the whole model by mistake."; return false; }
        for (auto& p : ok) if (trim(p.pass).empty()) { err = "The donor " + p.host + " has no passphrase. Nothing was started."; return false; }
        std::vector<Peer> keep = sl.peers; sl.peers = ok;
        std::vector<std::pair<int, int>> sp; bool good = compute_splits(sl, sp, err); sl.peers = keep;
        if (!good) return false;
        a.push_back("--dist-required"); a.push_back("--dist-layer-start"); a.push_back(std::to_string(sp[0].first)); a.push_back("--dist-layer-end"); a.push_back(std::to_string(sp[0].second));
        for (size_t i = 0; i < ok.size(); ++i) {
            a.push_back("--dist-peer"); a.push_back(trim(ok[i].host) + ":" + std::to_string(atoi(ok[i].port.c_str()) > 0 ? atoi(ok[i].port.c_str()) : 9876));
            a.push_back("--dist-passphrase"); a.push_back(trim(ok[i].pass));
            a.push_back("--dist-peer-layers"); a.push_back(std::to_string(sp[i + 1].first) + ":" + std::to_string(sp[i + 1].second));
            if (ok[i].local) a.push_back("--dist-peer-local-load");
        }
    }
    for (auto& x : split_args(sl.S["extraArgs"])) a.push_back(x);
    std::istringstream es(sl.S["extraEnv"]); std::string line;
    while (std::getline(es, line)) {
        line = trim(line); if (line.empty() || line[0] == '#') continue;
        size_t eq = line.find('='); if (eq == std::string::npos || eq < 1) continue;
        std::string k = trim(line.substr(0, eq));
        bool okname = k.compare(0, 6, "NEXUS_") == 0 && k.size() > 6;
        for (char c : k) if (!(isalnum((unsigned char)c) || c == '_')) okname = false;
        if (!okname) { err = "Only NEXUS_... variables are allowed in the environment box: " + k; return false; }
        env.emplace_back(k, trim(line.substr(eq + 1)));
    }
    return true;
}

void write_roster() {                                  // the list the web page uses to switch between running models
    JV arr; arr.t = JV::Arr;
    for (int i = 0; i < g_ns; ++i) if (child_alive(g_children[i])) {
        std::string bn = g_slots[i].cur; size_t dot = bn.rfind('.'); if (dot != std::string::npos) bn = bn.substr(0, dot);
        for (auto& c : bn) if (c == '-' || c == '_') c = ' ';
        if (bn.size() > 30) bn.resize(30); if (bn.empty()) bn = "Model " + std::to_string(i + 1);
        JV r; r.t = JV::Obj; *jset(r, "slot") = jnumv(i); *jset(r, "name") = jstrv(bn); *jset(r, "port") = jnumv(atoi(g_slots[i].port.c_str()));
        *jset(r, "url") = jstrv("http://localhost:" + g_slots[i].port); *jset(r, "status") = jstrv("running");
        arr.a.push_back(r);
    }
    std::string out; jdump(arr, out); write_file_atomic(join(g_dir, "nexus-roster.json"), out + "\n");
}

bool start_slot(int i) {
    std::vector<std::string> a; std::vector<std::pair<std::string, std::string>> env; std::string err;
    if (child_alive(g_children[i])) return true;
    if (!build_start(i, a, env, err)) { g_error = err; return false; }
    g_children[i].port = atoi(g_slots[i].port.c_str()); g_children[i].model = g_slots[i].cur;
    if (!child_start(g_children[i], a, env, err)) { g_error = err; return false; }
    g_slots[i].ready = false; g_slots[i].was_ready = false; write_roster();
    return true;
}
void stop_slot_async(int i) { std::thread([i] { child_stop(g_children[i]); write_roster(); }).detach(); }
bool any_running() { for (int i = 0; i < kMaxChildren; ++i) if (child_alive(g_children[i])) return true; return false; }
void stop_everything() { for (int i = 0; i < kMaxChildren; ++i) if (child_alive(g_children[i])) child_stop(g_children[i]); }

