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
    "gpuSel", "gpuLayers", "gpu2Sel", "gpu2Layers", "threads", "myLayers", "prefillMode", "prefillGpu", "vram", "gpuWin", "attnPath", "tokByTok", "profile", "trace", "gpuKv"};

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
        {"gpuKv", "auto"}, {"gpuSel", "cpu"}, {"gpuLayers", "0"}, {"gpu2Sel", "none"}, {"gpu2Layers", "0"}, {"vram", "0"}, {"gpuWin", "1"}, {"attnPath", "auto"}, {"trace", "0"},
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
        JV cf; cf.t = JV::Obj; for (auto* f : MODEL_FIELDS) *jset(cf, f) = jstrv(sl.S[f]);      // the boxes exactly as they are now (restored at start-up)
        *jset(s, "cfg") = cf;
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

// restore == true : program start-up -- the sampling / template boxes come from what was saved for this slot (cfg), else from the model's last saved values.
// restore == false: the user picked / switched a model -- the boxes are LEFT EXACTLY AS THEY ARE.  Only a profile set to "use automatically" for
//                   the new model changes them.  (Per-model settings are what profiles are for; nothing is applied behind the user's back.)
void load_model_settings(Slot& sl, bool restore = false, const JV* cfg = nullptr) {
    const JV* m = nullptr;
    if (!sl.cur.empty()) { const JV* ms = g_cfg.get("models"); if (ms) m = ms->get(sl.cur); }
    if (restore) {
        const JV* base = cfg ? cfg : m;
        for (auto* f : MODEL_FIELDS) { const JV* v = base ? base->get(f) : nullptr; std::string x = v ? jstr(v) : ""; sl.S[f] = x.empty() ? dflt(f) : x; }
    } else {
        for (auto* f : MODEL_FIELDS) if (sl.S[f].empty()) sl.S[f] = dflt(f);          // only fills boxes that were never set
    }
    sl.lora_sel = -1; sl.lora_custom.clear();
    std::string lp = (restore && m) ? jstr(m->get("loraPath")) : "", ln = (restore && m) ? jstr(m->get("loraName")) : "";   // an adapter belongs to one base model: never carried over
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
void attach(Slot& sl, int idx, const std::string& saved_path, bool restore = false, const JV* cfg = nullptr) {
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
    load_model_settings(sl, restore, cfg);
}

void on_model_picked(Slot& sl) {              // user changed the model by hand
    std::string p = path_of(sl);
    sl.cur = p.empty() ? "" : base_name(p); sl.note.clear();
    load_model_settings(sl, false); changed();
}

void init_slot(int i, const JV* rec) {
    Slot& sl = g_slots[i];
    sl.S.clear(); sl.peers.clear(); sl.dist_on = false; sl.dist_mode = "auto";
    sl.port = std::to_string(8080 + i);
    std::string path; const JV* cfg = nullptr;
    if (rec) {
        cfg = rec->get("cfg");
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
    attach(sl, i, path, true, cfg); fix_hw(sl);
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
    if (gpu != "cpu") E("NEXUS_GPU_KV", V("gpuKv"));      // Qwen3.5: attention memory (KV cache) kept on the card: auto / 0 = off / N megabytes
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

// ───────────────────────────── widgets ───────────────────────────────────────

const ImVec4 kMute(0.55f, 0.60f, 0.67f, 1.0f);
const ImVec4 kGreen(0.3f, 0.76f, 0.54f, 1), kAmber(0.88f, 0.69f, 0.29f, 1), kRed(1, 0.42f, 0.37f, 1);

void Label(const char* t) { ImGui::PushStyleColor(ImGuiCol_Text, kMute); ImGui::TextUnformatted(t); ImGui::PopStyleColor(); }
void Hint(const char* t) { ImGui::PushStyleColor(ImGuiCol_Text, kMute); ImGui::TextWrapped("%s", t); ImGui::PopStyleColor(); }

// ───────────────────────────── right-click menu for text boxes ───────────────
// Every text box gets Cut / Copy / Paste / Delete / Select all on the right mouse button (no keyboard shortcuts needed).
ImGuiID g_selall_id = 0; int g_selall_tries = 0;

void text_menu(ImGuiID id, std::string* s, ImGuiInputTextFlags fl, bool& edited, bool multiline) {
    static ImGuiID menu_id = 0; static int sa = 0, sb = 0, cur = 0;
    const bool ro = (fl & ImGuiInputTextFlags_ReadOnly) != 0, pw = (fl & ImGuiInputTextFlags_Password) != 0;
    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {           // remember what was selected at the moment of the click
        menu_id = id; sa = sb = 0; cur = (int)s->size();
        if (ImGuiInputTextState* st = ImGui::GetInputTextState(id)) {
            int a = st->GetSelectionStart(), b = st->GetSelectionEnd(); sa = std::min(a, b); sb = std::max(a, b); cur = st->GetCursorPos();
        }
        sa = std::max(0, std::min(sa, (int)s->size())); sb = std::max(sa, std::min(sb, (int)s->size())); cur = std::max(0, std::min(cur, (int)s->size()));
    }
    if (!ImGui::BeginPopupContextItem(nullptr, ImGuiPopupFlags_MouseButtonRight)) return;      // popup id = this text box
    if (menu_id != id) { sa = sb = 0; cur = (int)s->size(); menu_id = id; }
    const bool has_sel = sb > sa;
    const char* clip = ImGui::GetClipboardText();
    if (ImGui::MenuItem("Cut", nullptr, false, !ro && !pw && has_sel)) { ImGui::SetClipboardText(s->substr((size_t)sa, (size_t)(sb - sa)).c_str()); s->erase((size_t)sa, (size_t)(sb - sa)); edited = true; }
    if (ImGui::MenuItem("Copy", nullptr, false, !pw && has_sel)) ImGui::SetClipboardText(s->substr((size_t)sa, (size_t)(sb - sa)).c_str());
    if (ImGui::MenuItem("Paste", nullptr, false, !ro && clip && *clip)) {
        std::string c = clip;
        if (!multiline) for (auto& ch : c) if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';   // one-line boxes cannot hold line breaks
        if (has_sel) { s->erase((size_t)sa, (size_t)(sb - sa)); s->insert((size_t)sa, c); } else s->insert((size_t)cur, c);
        edited = true;
    }
    if (ImGui::MenuItem("Delete", nullptr, false, !ro && has_sel)) { s->erase((size_t)sa, (size_t)(sb - sa)); edited = true; }
    ImGui::Separator();
    if (ImGui::MenuItem("Select all", nullptr, false, !pw && !s->empty())) { g_selall_id = id; g_selall_tries = 0; }
    ImGui::EndPopup();
}
// the three text-box kinds used in this program, all with the menu above
bool InText(const char* label, std::string* s, ImGuiInputTextFlags fl = 0, const char* hint = nullptr) {
    const ImGuiID id = ImGui::GetID(label);
    if (g_selall_id == id) ImGui::SetKeyboardFocusHere();
    bool c = hint ? ImGui::InputTextWithHint(label, hint, s, fl) : ImGui::InputText(label, s, fl);
    if (g_selall_id == id) { if (ImGuiInputTextState* st = ImGui::GetInputTextState(id)) { st->SelectAll(); g_selall_id = 0; } else if (++g_selall_tries > 3) g_selall_id = 0; }
    text_menu(id, s, fl, c, false); return c;
}
bool InTextMulti(const char* label, std::string* s, const ImVec2& size, ImGuiInputTextFlags fl = 0) {
    const ImGuiID id = ImGui::GetID(label);
    if (g_selall_id == id) ImGui::SetKeyboardFocusHere();
    bool c = ImGui::InputTextMultiline(label, s, size, fl);
    if (g_selall_id == id) { if (ImGuiInputTextState* st = ImGui::GetInputTextState(id)) { st->SelectAll(); g_selall_id = 0; } else if (++g_selall_tries > 3) g_selall_id = 0; }
    text_menu(id, s, fl, c, true); return c;
}

bool Txt(const char* id, const char* label, const char* hint = "", const char* note = nullptr, ImGuiInputTextFlags fl = 0) {
    ImGui::PushID(id); Label(label); ImGui::SetNextItemWidth(-FLT_MIN);
    std::string ph = hint && *hint ? std::string(hint) : dflt(id);
    bool c = InText("##v", &(*g_m)[id], fl, ph.c_str());
    if (note) Hint(note);
    ImGui::PopID(); if (c) changed(); return c;
}
bool Sel(const char* id, const char* label, const std::vector<Choice>& items) {
    ImGui::PushID(id); Label(label); ImGui::SetNextItemWidth(-FLT_MIN);
    std::string& cur = (*g_m)[id]; const char* prev = items.empty() ? "" : items[0].label.c_str();
    for (auto& it : items) if (it.value == cur) prev = it.label.c_str();
    bool c = false;
    if (ImGui::BeginCombo("##v", prev)) {
        for (auto& it : items) {
            bool sel = it.value == cur;
            if (ImGui::Selectable(it.label.c_str(), sel, it.disabled ? ImGuiSelectableFlags_Disabled : 0)) { cur = it.value; c = true; }
            if (sel) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopID(); if (c) changed(); return c;
}
bool Chk(const char* id, const char* label, const char* note = nullptr) {
    ImGui::PushID(id); bool b = (*g_m)[id] == "1"; bool c = ImGui::Checkbox(label, &b);
    if (note) { ImGui::Indent(ImGui::GetFrameHeight() + 4); Hint(note); ImGui::Unindent(ImGui::GetFrameHeight() + 4); }
    ImGui::PopID(); if (c) { (*g_m)[id] = b ? "1" : "0"; changed(); } return c;
}
bool Card(const char* title, bool open_default = false) {
    if (g_expand_all) ImGui::SetNextItemOpen(true, ImGuiCond_Always);
    else if (open_default) ImGui::SetNextItemOpen(true, ImGuiCond_Once);
    return ImGui::CollapsingHeader(title);
}
struct Grid {
    Grid(const char* id, int cols) { ok = ImGui::BeginTable(id, cols, ImGuiTableFlags_SizingStretchSame); }
    ~Grid() { if (ok) ImGui::EndTable(); }
    void next() { ImGui::TableNextColumn(); }
    bool ok;
};
std::vector<Choice> gpu_choices(bool with_cpu, bool with_none) {
    std::vector<Choice> r;
    if (with_none) r.push_back({"none", "None", false});
    for (auto& c : (with_cpu ? g_gpus : g_pgpus)) r.push_back(c);
    return r;
}

void open_browse(const char* title, bool folder, std::vector<std::string> exts, const std::string& start, std::function<void(const std::string&)> cb) {
    g_br.title = title; g_br.folder = folder; g_br.exts = std::move(exts); g_br.cb = std::move(cb);
    g_br.r = browse_dir(start, g_br.exts); if (!g_br.r.ok) g_br.r = browse_dir("", g_br.exts);
    g_br.path_edit = g_br.r.path; g_br.open_req = true;
}
void browse_go(const std::string& p) {
    BrowseResult r = browse_dir(p, g_br.exts);
    if (r.ok) { g_br.r = r; g_br.path_edit = r.path; }
}

void draw_browse() {
    if (g_br.open_req) { ImGui::OpenPopup("##browse"); g_br.open_req = false; g_br.open = true; }
    ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(std::min(vp->Size.x - 40.0f, 760.0f * ImGui::GetFontSize() / 15.0f), std::min(vp->Size.y - 40.0f, 560.0f * ImGui::GetFontSize() / 15.0f)), ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal("##browse", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings)) return;
    ImGui::TextUnformatted(g_br.title.c_str()); ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("Cancel").x - ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetCursorPosX());
    if (ImGui::Button("Cancel")) { g_br.open = false; ImGui::CloseCurrentPopup(); }
    if (ImGui::Button("Up")) browse_go(g_br.r.parent);
    ImGui::SameLine(); ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Go").x - ImGui::GetStyle().FramePadding.x * 4);
    if (InText("##bp", &g_br.path_edit, ImGuiInputTextFlags_EnterReturnsTrue)) browse_go(g_br.path_edit);
    ImGui::SameLine(); if (ImGui::Button("Go")) browse_go(g_br.path_edit);
    for (size_t i = 0; i < g_br.r.roots.size(); ++i) {
        const std::string& d = g_br.r.roots[i];
        if (i) ImGui::SameLine();
        float w = ImGui::CalcTextSize(d.c_str()).x + ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetStyle().ItemSpacing.x;
        if (i && ImGui::GetContentRegionAvail().x < w) ImGui::NewLine();
        ImGui::PushID((int)i); if (ImGui::SmallButton(d.c_str())) browse_go(d); ImGui::PopID();
    }
    ImGui::Separator();
    const float footer = ImGui::GetFrameHeightWithSpacing() + 4;
    ImGui::BeginChild("##blist", ImVec2(0, -footer), ImGuiChildFlags_Borders);
    std::string go_to;
    for (auto& d : g_br.r.dirs) {
        std::string label = "[ ] " + d;
        if (ImGui::Selectable(label.c_str())) {
            const std::string& base = g_br.r.path; bool drive = base.empty();
            go_to = drive ? d : (base.back() == '/' || base.back() == '\\' ? base + d : base + SEP + d);
        }
    }
    if (!g_br.folder) for (auto& f : g_br.r.files) {
        char sz[32]; if (f.size >= 1073741824ULL) snprintf(sz, sizeof sz, "%.2f GB", f.size / 1073741824.0); else if (f.size >= 1048576ULL) snprintf(sz, sizeof sz, "%.1f MB", f.size / 1048576.0); else snprintf(sz, sizeof sz, "%.0f KB", f.size / 1024.0);
        ImGui::PushID(f.name.c_str());
        if (ImGui::Selectable(f.name.c_str())) {
            std::string p = join(g_br.r.path, f.name); auto cb = g_br.cb; g_br.open = false; ImGui::CloseCurrentPopup(); if (cb) cb(p);
        }
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(sz).x); Label(sz);
        ImGui::PopID();
        if (!g_br.open) break;
    }
    if (g_br.r.dirs.empty() && (g_br.folder || g_br.r.files.empty())) Hint("Nothing here.");
    ImGui::EndChild();
    if (!go_to.empty()) browse_go(go_to);
    if (g_br.folder) {
        if (ImGui::Button("Use this folder") && !g_br.r.path.empty()) { auto cb = g_br.cb; std::string p = g_br.r.path; g_br.open = false; ImGui::CloseCurrentPopup(); if (cb) cb(p); }
        ImGui::SameLine(); Hint("Open a folder, then press the button.");
    } else Hint("Click a file to choose it.");
    ImGui::EndPopup();
}

// a path box with a Browse button (for files that are not models)
void PathField(const char* id, const char* label, const char* title, std::vector<std::string> exts) {
    ImGui::PushID(id); Label(label);
    ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetStyle().ItemSpacing.x));
    if (InText("##v", &(*g_m)[id])) changed();
    ImGui::SameLine();
    SMap* m = g_m; std::string key = id;
    if (ImGui::Button("Browse...")) open_browse(title, false, exts, "", [m, key](const std::string& f) { (*m)[key] = f; changed(); });
    ImGui::PopID();
}

// ───────────────────────────── setup screen ──────────────────────────────────

void draw_model_card(Slot& sl) {
    if (!Card("Model", true)) return;
    Label("Model file");
    std::string prev = sl.sel >= 0 && sl.sel < (int)g_models.size() ? g_models[sl.sel].name : (!sl.custom.empty() ? "(using the path typed below)" : "(no model selected)");
    float bw = ImGui::CalcTextSize("Browse...").x + ImGui::CalcTextSize("Rescan").x + ImGui::GetStyle().FramePadding.x * 4 + ImGui::GetStyle().ItemSpacing.x * 2;
    ImGui::SetNextItemWidth(-bw);
    if (ImGui::BeginCombo("##model", prev.c_str())) {
        if (g_models.empty()) Hint("No models found. Put one in the \"models\" folder next to this program, or add a folder.");
        for (int i = 0; i < (int)g_models.size(); ++i) {
            std::string l = g_models[i].name + "   " + gbs((double)g_models[i].size) + "   " + g_models[i].dir;
            if (ImGui::Selectable(l.c_str(), i == sl.sel && sl.custom.empty())) { sl.sel = i; sl.custom.clear(); on_model_picked(sl); }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    Slot* sp = &sl;
    if (ImGui::Button("Browse...")) {
        std::string p = path_of(sl); size_t k = p.find_last_of("/\\"); std::string start = k == std::string::npos ? "" : p.substr(0, k);
        open_browse("Choose a model", false, {"gguf", "nxm"}, start, [sp](const std::string& f) { sp->custom = f; sp->sel = -1; on_model_picked(*sp); });
    }
    ImGui::SameLine(); if (ImGui::Button("Rescan")) { scan_all(); for (int i = 0; i < g_ns; ++i) { Slot& rs = g_slots[i]; std::string p = path_of(rs); SMap keep = rs.S; std::string keep_prof = rs.profile, keep_cur = rs.cur;
            attach(rs, i, p, true); fix_hw(rs);                                  // re-find the model file; the boxes and the profile stay as they were
            if (rs.cur == keep_cur) { for (auto* f : MODEL_FIELDS) rs.S[f] = keep[f]; rs.profile = keep_prof; } } }
    if (!sl.note.empty()) Hint(sl.note.c_str());
    Label("Or type / paste a full path");
    ImGui::SetNextItemWidth(-FLT_MIN);
    std::string typed = sl.custom;
    if (InText("##path", &typed, 0, "/path/to/model.gguf   or   D:\\models\\model.nxm")) { sl.custom = typed; if (!typed.empty()) sl.sel = -1; on_model_picked(sl); }
    if (sl.meta.ok) {
        char b[200]; snprintf(b, sizeof b, "%s  -  %d layers  -  chat format: %s (%s confidence)", sl.meta.arch.c_str(), sl.meta.layers, sl.det.fmt.c_str(), sl.det.conf.c_str());
        Hint(b);
    }
    if (!sl.cur.empty()) {                                                        // profiles of this model
        ImGui::Spacing(); Label("Profile for this model");
        static std::string new_name; static std::map<std::string, int> del_ask;
        const std::vector<std::string> names = profile_names(sl.cur);
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        if (ImGui::BeginCombo("##prof", sl.profile.empty() ? "(none - keep my current settings)" : sl.profile.c_str())) {
            if (ImGui::Selectable("(none - keep my current settings)", sl.profile.empty())) { sl.profile.clear(); changed(); }
            for (auto& n : names) if (ImGui::Selectable(n.c_str(), n == sl.profile)) { std::string keep = n; profile_load(sl, keep); changed(); break; }
            ImGui::EndCombo();
        }
        bool autoon = !sl.profile.empty() && profile_auto_of(sl.cur) == sl.profile;
        ImGui::SameLine(); ImGui::BeginDisabled(sl.profile.empty());
        if (ImGui::Checkbox("Use automatically for this model", &autoon)) profile_set_auto(sl, autoon);
        ImGui::EndDisabled();
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
        InText("##newprof", &new_name, 0, "name for a new profile, e.g.  Fast  or  Test-1");
        ImGui::SameLine(); ImGui::BeginDisabled(trim(new_name).empty());
        if (ImGui::Button("Save as new profile")) { profile_save(sl, trim(new_name)); new_name.clear(); }
        ImGui::EndDisabled();
        if (!sl.profile.empty()) {
            ImGui::SameLine(); if (ImGui::Button("Update this profile")) profile_save(sl, sl.profile);
            ImGui::SameLine(); if (ImGui::Button("Delete")) ImGui::OpenPopup("##delprof");
            if (ImGui::BeginPopup("##delprof")) {
                ImGui::Text("Delete profile \"%s\"?", sl.profile.c_str());
                if (ImGui::Button("Delete")) { profile_delete(sl, sl.profile); ImGui::CloseCurrentPopup(); }
                ImGui::SameLine(); if (ImGui::Button("Keep")) ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
            }
        }
        Hint("With no profile selected, every box stays exactly as you set it, also when you switch models. A profile remembers all settings of this model (sampling, graphics cards, threads, extra variables) and is applied only when you pick it here, or when it is set to \"Use automatically for this model\". Settings are kept for the next start when you press \"Save current settings\".");
    }
    ImGui::Spacing(); Label("Model folders");
    for (size_t i = 0; i < g_dirs.size(); ++i) {
        ImGui::PushID((int)i);
        if (ImGui::SmallButton("x")) { g_dirs.erase(g_dirs.begin() + (long)i); ImGui::PopID(); scan_all(); changed(); break; }
        ImGui::SameLine(); ImGui::TextUnformatted(g_dirs[i].c_str()); ImGui::PopID();
    }
    if (ImGui::Button("Add folder...")) open_browse("Add a folder with models", true, {}, g_dirs.empty() ? "" : g_dirs[0], [](const std::string& d) {
        if (!d.empty() && std::find(g_dirs.begin(), g_dirs.end(), d) == g_dirs.end()) { g_dirs.insert(g_dirs.begin(), d); scan_all(); changed(); } });
    ImGui::SameLine(); Hint("The \"models\" folder next to this program is always searched too. Your own folders come first, and your choice is never replaced.");
}

void recheck_hw() {
    if (g_hw_state.load() == 0) return;
    g_hw_state = 0; g_hw_applied = false;
    std::thread([] {
        std::string j = run_probe(); JV v; bool ok = json_parse(j, v) && v.get("cpu");
        { std::lock_guard<std::mutex> lk(g_hw_mu); g_hw = ok ? v : JV(); }
        g_hw_state = ok ? 1 : 2;
    }).detach();
}

void draw_hw_card() {
    if (!Card("This computer")) return;
    int st = g_hw_state.load();
    if (st == 0) { Hint("Detecting hardware..."); return; }
    if (ImGui::Button("Re-check hardware")) { recheck_hw(); return; }
    ImGui::SameLine(); Hint("Graphics detection can differ from one start-up to the next on some computers.");
    std::lock_guard<std::mutex> lk(g_hw_mu);
    const JV* cpu = g_hw.get("cpu"); const JV* ram = g_hw.get("ram");
    if (st == 2 || !cpu) { Hint("Could not detect hardware. You can still start a model; set things by hand."); return; }
    ImGui::Text("%s  -  %d cores / %d threads, %s memory (%s free)", jstr(cpu->get("brand")).c_str(), atoi(jstr(cpu->get("physical")).c_str()), atoi(jstr(cpu->get("logical")).c_str()),
                gbs(ram ? ram->get("total_bytes")->n : 0).c_str(), gbs(ram ? ram->get("available_bytes")->n : 0).c_str());
    const JV* v = g_hw.get("vulkan"); const JV* d = v ? v->get("devices") : nullptr;
    if (!v || !v->get("loaded") || !v->get("loaded")->b) { ImGui::TextColored(kAmber, "No graphics driver found"); ImGui::SameLine(); Hint("The CPU will be used."); }
    else if (!d || d->a.empty()) ImGui::TextColored(kAmber, "No graphics card found");
    else for (auto& x : d->a) {
        const JV* u = x.get("usable"); bool usable = u && u->b;
        ImGui::TextColored(usable ? kGreen : kRed, usable ? "[usable]" : "[not auto-picked]");
        ImGui::SameLine(); ImGui::Text("[%s] %s", jstr(x.get("index")).c_str(), jstr(x.get("name")).c_str());
        ImGui::SameLine(); std::string extra = (x.get("vram_bytes") && x.get("vram_bytes")->n > 0 ? gbs(x.get("vram_bytes")->n) + "  -  " : "") + jstr(x.get("type")) + (usable ? "" : "  -  " + jstr(x.get("reason")));
        Hint(extra.c_str());
    }
}

void draw_peers(Slot& sl) {
    if (ImGui::Checkbox("Share this model with other computers", &sl.dist_on)) changed();
    Hint("Each donor computer runs this program in Donor mode (or the Donor tab) and holds part of the model's layers. Activations travel over the network, so a wired connection is best.");
    if (!sl.dist_on) return;
    std::vector<Choice> modes = {{"auto", "Equal share for everyone"}, {"manual", "I choose how many layers each donor takes"}};
    ImGui::Spacing(); Label("Split"); ImGui::SetNextItemWidth(-FLT_MIN);
    { std::string pv = sl.dist_mode == "manual" ? modes[1].label : modes[0].label;
      if (ImGui::BeginCombo("##dmode", pv.c_str())) { for (auto& m : modes) if (ImGui::Selectable(m.label.c_str(), sl.dist_mode == m.value)) { sl.dist_mode = m.value; changed(); } ImGui::EndCombo(); } }
    const bool manual = sl.dist_mode == "manual";
    int remove = -1;
    if (!sl.peers.empty() && ImGui::BeginTable("peers", manual ? 7 : 6, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Donor address", ImGuiTableColumnFlags_WidthStretch, 3.0f); ImGui::TableSetupColumn("Port", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Passphrase", ImGuiTableColumnFlags_WidthStretch, 2.5f); ImGui::TableSetupColumn("Has the model", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        if (manual) ImGui::TableSetupColumn("Layers", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Test", ImGuiTableColumnFlags_WidthFixed, 0); ImGui::TableSetupColumn("##x", ImGuiTableColumnFlags_WidthFixed, 0);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < sl.peers.size(); ++i) {
            Peer& p = sl.peers[i]; ImGui::PushID((int)i); ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-FLT_MIN); if (InText("##h", &p.host, 0, "192.168.1.20")) changed();
            ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-FLT_MIN); if (InText("##p", &p.port, ImGuiInputTextFlags_CharsDecimal)) changed();
            ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-FLT_MIN); if (InText("##k", &p.pass, ImGuiInputTextFlags_Password)) changed();
            ImGui::TableNextColumn(); if (ImGui::Checkbox("##l", &p.local)) changed();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Tick when the donor already has this same model file on its own disk,\nso the weights are not sent over the network.");
            if (manual) { ImGui::TableNextColumn(); ImGui::SetNextItemWidth(-FLT_MIN); if (InText("##n", &p.layers, ImGuiInputTextFlags_CharsDecimal)) changed(); }
            ImGui::TableNextColumn();
            std::string key = trim(p.host) + ":" + p.port; int st = 0; { std::lock_guard<std::mutex> lk(g_peer_mu); auto it = g_peer_state.find(key); if (it != g_peer_state.end()) st = it->second; }
            if (ImGui::SmallButton("Test") && !trim(p.host).empty() && st != 3) {
                { std::lock_guard<std::mutex> lk(g_peer_mu); g_peer_state[key] = 3; }
                std::string h = trim(p.host); int po = atoi(p.port.c_str());
                std::thread([key, h, po] { bool ok = tcp_reachable(h, po > 0 ? po : 9876, 2500); std::lock_guard<std::mutex> lk(g_peer_mu); g_peer_state[key] = ok ? 1 : 2; }).detach();
            }
            ImGui::SameLine(); ImGui::TextColored(st == 1 ? kGreen : st == 2 ? kRed : st == 3 ? kAmber : kMute, st == 1 ? "reachable" : st == 2 ? "no answer" : st == 3 ? "testing" : "-");
            ImGui::TableNextColumn(); if (ImGui::SmallButton("x")) remove = (int)i;
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (remove >= 0) { sl.peers.erase(sl.peers.begin() + remove); changed(); }
    if (ImGui::Button("Add donor computer")) { sl.peers.push_back(Peer()); changed(); }
    ImGui::SameLine(); ImGui::BeginDisabled(g_scan_state.load() == 1);
    if (ImGui::Button(g_scan_state.load() == 1 ? "Looking..." : "Find donors on my network")) scan_lan(atoi(trim(g_donor["port"]).c_str()) > 0 ? atoi(trim(g_donor["port"]).c_str()) : 9876);
    ImGui::EndDisabled();
    if (g_scan_state.load() == 2) {
        std::vector<std::string> f; { std::lock_guard<std::mutex> lk(g_found_mu); f = g_found; }
        if (f.empty()) Hint("No donor answered on port 9876. Is the Donor tab started on the other computer?");
        for (auto& h : f) {
            bool have = false; for (auto& p : sl.peers) if (trim(p.host) == h) have = true;
            ImGui::PushID(h.c_str()); ImGui::BulletText("%s answers", h.c_str()); ImGui::SameLine();
            ImGui::BeginDisabled(have); if (ImGui::SmallButton(have ? "Added" : "Add")) { Peer p; p.host = h; sl.peers.push_back(p); changed(); } ImGui::EndDisabled(); ImGui::PopID();
        }
    }
    std::vector<std::pair<int, int>> sp; std::string err;
    if (!sl.peers.empty()) {
        ImGui::Spacing(); Label(sl.meta.layers > 0 ? ("Layer plan  -  this model has " + std::to_string(sl.meta.layers) + " layers").c_str() : "Layer plan");
        const bool ok = compute_splits(sl, sp, err);
        if (ImGui::BeginTable("plan", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Computer", ImGuiTableColumnFlags_WidthStretch, 3.0f); ImGui::TableSetupColumn("Layers", ImGuiTableColumnFlags_WidthStretch, 2.0f); ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableHeadersRow();
            for (size_t i = 0; ok && i < sp.size(); ++i) {
                ImGui::TableNextRow(); ImGui::TableNextColumn();
                if (i == 0) ImGui::TextUnformatted("This computer"); else ImGui::Text("Donor %d  (%s)", (int)i, trim(sl.peers[i - 1].host).empty() ? "no address yet" : trim(sl.peers[i - 1].host).c_str());
                ImGui::TableNextColumn(); ImGui::Text("%d - %d", sp[i].first, sp[i].second - 1);
                ImGui::TableNextColumn();
                if (i == 0 && manual) { ImGui::SetNextItemWidth(-FLT_MIN); ImGui::PushID("mine"); if (InText("##m", &sl.S["myLayers"], ImGuiInputTextFlags_CharsDecimal, std::to_string(sp[0].second - sp[0].first).c_str())) changed(); ImGui::PopID(); }
                else ImGui::Text("%d", sp[i].second - sp[i].first);
            }
            ImGui::EndTable();
        }
        if (!ok) ImGui::TextColored(kAmber, "%s", err.c_str());
    }
    Hint("If sharing is on and a donor is missing or has no passphrase, nothing starts - this computer never quietly loads the whole model.");
}

void draw_setup(Slot& sl, int idx) {
    g_m = &sl.S;
    draw_model_card(sl);
    draw_hw_card();

    if (Card("Run on graphics card(s)")) {
        { Grid g("compute", 3);
          g.next(); Sel("gpuSel", "Graphics card 1", g_gpus);
          g.next(); Txt("gpuLayers", "Layers kept on card 1", "", "The first N layers stay on this card for good. 0 = none.");
          g.next(); Txt("threads", "CPU threads", "", "For the layers that run on the CPU."); }
        // every time a card is chosen, a place for the next one appears; layers are placed in order: card 1, then card 2, then card 3 ...
        int placed = 0; std::string plan; bool any_card = sl.S["gpuSel"] != "cpu";
        if (any_card) { int l0 = std::max(0, atoi(sl.S["gpuLayers"].c_str())); if (l0 > 0) { plan += "card 1: layers " + std::to_string(placed) + "-" + std::to_string(placed + l0 - 1); placed += l0; } }
        for (int n = 2; n <= kMaxCards && any_card; ++n) {
            const std::string ks = cardSel(n), kl = cardLay(n); const std::string prev = sl.S[cardSel(n - 1)];
            if (n > 2 && (prev == "none" || prev.empty())) break;
            std::string t1 = "Graphics card " + std::to_string(n), t2 = "Layers kept on card " + std::to_string(n);
            { Grid g(("compute" + std::to_string(n)).c_str(), 3);
              g.next(); Sel(ks.c_str(), t1.c_str(), gpu_choices(false, true));
              g.next(); ImGui::BeginDisabled(sl.S[ks] == "none"); Txt(kl.c_str(), t2.c_str(), "", n == 2 ? "Placed right after the layers of card 1." : nullptr); ImGui::EndDisabled(); }
            if (sl.S[ks] == "none") break;
            int ln = std::max(0, atoi(sl.S[kl].c_str()));
            if (ln > 0) { if (!plan.empty()) plan += "   |   "; plan += "card " + std::to_string(n) + ": layers " + std::to_string(placed) + "-" + std::to_string(placed + ln - 1); placed += ln; }
        }
        if (any_card) {
            Sel("gpuKv", "Context memory on the card (KV cache)", {{"auto", "Auto  (free card memory, with a safety margin)"}, {"0", "Off  (context stays in RAM, processor does attention)"},
                                                                  {"512", "Up to 512 MB"}, {"1024", "Up to 1 GB"}, {"2048", "Up to 2 GB"}, {"4096", "Up to 4 GB"}, {"8192", "Up to 8 GB"}});
            Hint("Qwen3.5 models: keeps the attention memory on the card so every generated word needs no copying from RAM - this is what makes long conversations fast. If it would not fit (very long context, small card) the engine says so in the log and the processor does attention for that context instead, so long prompts always work. A copy always stays in RAM.");
        }
        if (any_card && placed > 0) {
            std::string t = "Plan: " + plan;
            if (sl.meta.layers > 0) t += placed >= sl.meta.layers ? "   |   the whole model (" + std::to_string(sl.meta.layers) + " layers) is on cards" : "   |   layers " + std::to_string(placed) + "-" + std::to_string(sl.meta.layers - 1) + ": CPU";
            ImGui::TextColored(kGreen, "%s", t.c_str());
        }
        Hint("Mix any cards you like, in any amounts - even more than fits; the engine uses what it can and says so in the log. Pick \"CPU only\" for card 1 to keep every layer on the processor. Card 3 and up are sent as --gpu3-layers / --gpu3-device (needs the engine build that includes them).");
    }
    if (Card("Prompt speed-up (prefill acceleration)")) {
        { const std::string why = prefill_auto_reason(sl);
          Sel("prefillMode", "Prefill acceleration", {{"auto", why.empty() ? "Auto  (ON)" : "Auto  (off: " + why + ")"}, {"1", "On"}, {"0", "Off"}}); }
        Hint("Separate from the card above. Layers are swapped through the card one at a time while a long prompt is read, then the CPU generates. It only needs room for a single layer, so almost any card works.");
        { Grid g("pf", 3);
          g.next(); Sel("prefillGpu", "Card to use", g_pgpus);
          g.next(); Txt("gpuWin", "Layers swapped through the card at once", "");
          g.next(); Sel("attnPath", "Attention path", {{"auto", "Auto (short prompts batch)"}, {"stream", "Stream"}, {"batch", "Batch"}}); }
        if (g_pgpus.empty()) Hint("No graphics card was found on this computer.");
        else if (prefill_on(sl) && sl.S["gpuSel"] != "cpu" && sl.S["gpuSel"] != sl.S["prefillGpu"]) Hint("Note: the engine uses one card for both jobs right now, so the card chosen above is the one used.");
    }
    if (Card("Share with other computers (donors)")) draw_peers(sl);
    if (Card("Model behaviour")) {
        std::vector<Choice> fm = {{"", "Auto-detect  ->  " + sl.det.fmt + "  (" + sl.det.conf + " confidence)"}};
        for (const char* f : {"ChatML", "Qwen3", "Llama-3", "Mistral", "Mistral V3", "Gemma", "Gemma 4", "Phi-3", "Phi-4", "DeepSeek", "DeepSeek-R1", "DeepSeek-Coder", "OLMoE", "Zephyr", "Alpaca", "Granite", "Raw", "Custom"}) fm.push_back({f, f});
        const std::string eff = trim(sl.S["chatFmt"]).empty() ? sl.det.fmt : sl.S["chatFmt"];
        const bool think_ok = supports_thinking(sl.meta, eff);
        { Grid g("beh", 3);
          g.next(); Sel("chatFmt", "Chat format", fm);
          g.next(); Sel("rope", "RoPE layout", {{"", "From the model file"}, {"interleaved", "Interleaved"}, {"half_split", "Half-split"}});
          g.next(); Sel("norm", "Norm convention", {{"", "From the model file"}, {"direct", "Direct  (x*w)"}, {"zero_centered", "Zero-centered  (x*(1+w))"}});
          ImGui::BeginDisabled(!think_ok);
          g.next(); Sel("thinking", "Thinking", {{"", "Model decides"}, {"1", "Always on"}, {"0", "Off"}});
          g.next(); Sel("preserve", "Keep earlier thinking", {{"", "Model decides"}, {"1", "Keep"}, {"0", "Drop"}});
          ImGui::EndDisabled(); }
        if (!think_ok) Hint("This model's chat template has no thinking switch, so the two thinking settings do nothing.");
        if (eff == "Custom") {
            Label("Custom chat template - the text placed around each part of the conversation");
            { Grid g("ct", 2);
              g.next(); Txt("ctPre", "Before everything", ""); g.next(); Txt("ctPreSys", "Before system text", "");
              g.next(); Txt("ctEndSys", "After system text", ""); g.next(); Txt("ctPreUser", "Before user text", "");
              g.next(); Txt("ctEndUser", "After user text", ""); g.next(); Txt("ctPreModel", "Before model text", "");
              g.next(); Txt("ctEndModel", "After model text", ""); }
        }
        Hint("Saved separately for each model. \"From the model file\" means the file itself decides (GGUF: its architecture rules; .nxm: what the quantizer wrote). If a model answers with gibberish, try changing RoPE or Norm - the Log tab shows what was chosen.");
    }
    if (Card("Sampling")) {
        Grid g("samp", 4);
        const struct { const char* id; const char* l; } f[] = {{"temp", "Temperature"}, {"topp", "Top-p"}, {"topk", "Top-k"}, {"minp", "Min-p"}, {"tfsz", "Tail-free z (1 = off)"},
            {"rep", "Repeat penalty (1 = off)"}, {"repn", "Repeat window"}, {"freq", "Frequency penalty"}, {"pres", "Presence penalty"}, {"maxtok", "Max tokens"}};
        for (auto& x : f) { g.next(); Txt(x.id, x.l, ""); }
    }
    if (Card("Images (models that can see)")) {
        Hint("Only for models with built-in image understanding.");
        { Grid g("img", 3);
          g.next(); Txt("imgSize", "Image size the model sees (px)", "");
          g.next(); Txt("upEdge", "Largest uploaded edge (px, 0 = original)", "");
          g.next(); Txt("upPx", "Largest uploaded area (pixels, 0 = original)", "");
          g.next(); Sel("vMode", "Large images", {{"single", "Shrink to one image"}, {"tiles", "Split into tiles"}});
          g.next(); Txt("tileSize", "Tile size", "");
          g.next(); Txt("tileStride", "Tile step", "");
          g.next(); Txt("maxTiles", "Most tiles (0 = no limit)", ""); }
        Chk("visionGpu", "Run the image part on the graphics card");
    }
    if (Card("LoRA adapter")) {
        Label("Adapter file (.gguf)");
        std::string prev = sl.lora_sel == -2 ? base_name(sl.lora_custom) + " (browsed)" : (sl.lora_sel >= 0 && sl.lora_sel < (int)g_loras.size() ? g_loras[sl.lora_sel].name : "None");
        ImGui::SetNextItemWidth(-(ImGui::CalcTextSize("Browse...").x + ImGui::GetStyle().FramePadding.x * 2 + ImGui::GetStyle().ItemSpacing.x));
        if (ImGui::BeginCombo("##lora", prev.c_str())) {
            if (ImGui::Selectable("None", sl.lora_sel == -1)) { sl.lora_sel = -1; changed(); }
            for (int i = 0; i < (int)g_loras.size(); ++i) if (ImGui::Selectable(g_loras[i].name.c_str(), sl.lora_sel == i)) { sl.lora_sel = i; changed(); }
            ImGui::EndCombo();
        }
        ImGui::SameLine(); Slot* sp = &sl;
        if (ImGui::Button("Browse...##lora")) open_browse("Choose a LoRA adapter", false, {"gguf"}, "", [sp](const std::string& f) { sp->lora_custom = f; sp->lora_sel = -2; changed(); });
        Hint("Put adapters in a \"loras\" folder next to this program to list them here, or browse to one anywhere.");
        Grid g("lora", 3); g.next(); Txt("loraScale", "Strength", "");
    }
    if (Card("Speculative decoding (MTP)")) {
        Grid g("mtp", 3);
        g.next(); Sel("mtp", "Use the model's own draft heads", {{"off", "Off"}, {"on", "On"}});
        g.next(); Txt("mtpN", "Tokens drafted per step", "");
        g.next(); Sel("mtpGpu", "Draft on", {{"off", "CPU"}, {"on", "Graphics card"}});
    }
    if (Card("Saved state (warm state) at startup")) {
        PathField("startState", "A saved state (KV only, no text) to load before the model starts serving", "Choose a saved state file", {"kvc"});
        Hint("Optional. Feed the model a big system prompt or teach it something in the web chat, save the state there, and it can start every new conversation already knowing it. States saved in the web chat live in the model's states folder.");
    }
    if (Card("Network")) {
        { Grid g("net", 2);
          g.next(); ImGui::PushID("port"); Label(("Port for model " + std::to_string(idx + 1)).c_str()); ImGui::SetNextItemWidth(-FLT_MIN); if (InText("##v", &sl.port, ImGuiInputTextFlags_CharsDecimal)) changed(); ImGui::PopID();
          g.next(); ImGui::PushID("lan"); Label("Who can connect (all models)"); ImGui::SetNextItemWidth(-FLT_MIN);
          if (ImGui::BeginCombo("##v", g_lan ? "Anyone on my network" : "Only this computer")) {
              if (ImGui::Selectable("Only this computer", !g_lan)) { g_lan = 0; changed(); }
              if (ImGui::Selectable("Anyone on my network", g_lan)) { g_lan = 1; changed(); }
              ImGui::EndCombo();
          }
          ImGui::PopID(); }
        { Grid g("br", 2); g.next(); ImGui::PushID("br"); Label("Browser program (optional)"); ImGui::SetNextItemWidth(-FLT_MIN);
          if (InText("##v", &g_browser_cmd)) changed(); ImGui::PopID(); }
        Hint("Used by \"Open chat in browser\". Leave empty for the system default. Example for a portable browser:  /mnt/usb/Firefox.AppImage");
        Hint(g_lan ? "Other devices can open  http://<this computer's address>:PORT  in a browser. There is no password - use only on a network you trust."
                   : "Only programs on this computer can connect.");
    }
    if (Card("Diagnostics")) {
        Chk("tokByTok", "Read prompts one token at a time (slow, diagnostic)");
        Chk("profile", "Print prompt-speed profile in the log");
        Grid g("diag", 3); g.next(); Sel("trace", "Log detail", {{"0", "0 - quiet"}, {"1", "1"}, {"2", "2"}, {"3", "3 - everything"}});
    }
    if (Card("Advanced")) {
        Txt("extraArgs", "Extra command-line options for the server", "anything not listed above");
        ImGui::PushID("env"); Label("Extra environment variables (NEXUS_... only, one per line, NAME=value)");
        if (InTextMulti("##v", &sl.S["extraEnv"], ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 4.5f))) changed();
        ImGui::PopID();
    }
    ImGui::Dummy(ImVec2(0, 8));
}

// ───────────────────────────── log ───────────────────────────────────────────

ImVec4 log_color(const std::string& l) {
    auto has_w = [&](const char* w) { std::string a = lower(l); return a.find(w) != std::string::npos; };
    if (has_w("error") || has_w("fail") || has_w("cannot") || has_w("abort") || has_w("fatal")) return ImVec4(1, 0.55f, 0.49f, 1);
    if (has_w("warn")) return ImVec4(0.9f, 0.75f, 0.42f, 1);
    if (l.rfind("[launcher]", 0) == 0) return ImVec4(0.5f, 0.69f, 1, 1);
    return ImVec4(0.84f, 0.87f, 0.9f, 1);
}

bool g_hide_load = true;                       // hide the per-tensor "[load] 12% Loading: ..." progress lines (thousands of them)
static bool is_load_noise(const std::string& l) {
    if (l.compare(0, 7, "[load] ") == 0 && l.find("%  Loading:") != std::string::npos) return true;
    if (l.size() > 6 && l[0] == ' ' && l[1] == ' ' && l[2] == '[' && l.find("%] ") != std::string::npos && l.find("%] ") < 9) return true;
    return false;
}
// "[load] 45%  Loading weights into memory" -> progress for the status bar. Later stages without a percentage still update the text.
static void track_load(LogView& lv, const std::string& l) {
    if (l.rfind("[launcher] starting", 0) == 0) { lv.load_pct = 0; lv.load_stage = "Starting"; return; }
    if (l.compare(0, 7, "[load] ") == 0) {
        size_t p = l.find('%'); if (p == std::string::npos || p < 7 || p > 11) return;
        int pct = atoi(l.c_str() + 7); if (pct < 0 || pct > 100) return;
        std::string txt = l.substr(p + 1); size_t a = txt.find_first_not_of(' '); txt = a == std::string::npos ? "" : txt.substr(a);
        if (txt.compare(0, 8, "Loading:") == 0) txt = "Loading weights";
        lv.load_pct = pct; lv.load_stage = txt; return;
    }
    if (l.find("[server] Binding to") != std::string::npos) { lv.load_pct = -1; lv.load_stage.clear(); return; }
    if (l.find("[vulkan] Initializing") != std::string::npos) { lv.load_pct = 100; lv.load_stage = "Starting graphics card"; }
    else if (l.find("Uploading") != std::string::npos && l.find("GPU layer") != std::string::npos) { lv.load_pct = 100; lv.load_stage = "Copying layers to the graphics card"; }
}

bool pull_log(Child& ch, LogView& lv, const std::function<void(const std::string&)>& on_line = nullptr) {
    std::lock_guard<std::mutex> lk(ch.mu);
    size_t total = ch.first_idx + ch.lines.size();
    if (lv.next < ch.first_idx) lv.next = ch.first_idx;
    bool any = lv.next < total;
    for (; lv.next < total; ++lv.next) { const std::string& ln = ch.lines[lv.next - ch.first_idx]; track_load(lv, ln); if (g_hide_load && is_load_noise(ln)) continue; lv.v.push_back(ln); if (on_line) on_line(lv.v.back()); }
    if (lv.v.size() > 8000) {
        const int gone = (int)(lv.v.size() - 6000); lv.v.erase(lv.v.begin(), lv.v.begin() + (long)gone);
        if (lv.sa_l >= 0) { lv.sa_l -= gone; lv.sb_l -= gone; if (lv.sa_l < 0 || lv.sb_l < 0) lv.clear_sel(); }
    }
    return any;
}

// ---- log text selection with the mouse ----------------------------------------------------------------------------
// byte offset in `line` that is under horizontal position x (pixels from the start of the line)
int log_col_at(const std::string& line, float x) {
    const int n = (int)line.size();
    if (x <= 0 || n == 0) return 0;
    ImFont* f = ImGui::GetFont(); const float fs = ImGui::GetFontSize();
    int prevk = 0; float prevw = 0;
    for (int k = 1; k <= n; ++k) {
        if (k < n && (line[(size_t)k] & 0xC0) == 0x80) continue;                  // not on a character boundary
        float w = f->CalcTextSizeA(fs, FLT_MAX, 0.0f, line.c_str(), line.c_str() + k).x;
        if (w >= x) return (x - prevw < w - x) ? prevk : k;                        // the nearer side of the character
        prevk = k; prevw = w;
    }
    return n;
}
float log_x_of(const std::string& line, int col) {
    if (col <= 0) return 0; col = std::min(col, (int)line.size());
    return ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(), FLT_MAX, 0.0f, line.c_str(), line.c_str() + col).x;
}
bool log_sel_ordered(const LogView& lv, int& l0, int& c0, int& l1, int& c1) {
    if (lv.sa_l < 0 || lv.sb_l < 0) return false;
    bool fwd = lv.sa_l < lv.sb_l || (lv.sa_l == lv.sb_l && lv.sa_c <= lv.sb_c);
    l0 = fwd ? lv.sa_l : lv.sb_l; c0 = fwd ? lv.sa_c : lv.sb_c; l1 = fwd ? lv.sb_l : lv.sa_l; c1 = fwd ? lv.sb_c : lv.sa_c;
    return !(l0 == l1 && c0 == c1);
}
std::string log_sel_text(const LogView& lv) {
    int l0, c0, l1, c1; if (!log_sel_ordered(lv, l0, c0, l1, c1)) return "";
    std::string r; l1 = std::min(l1, (int)lv.v.size() - 1);
    for (int i = l0; i <= l1 && i >= 0; ++i) {
        const std::string& ln = lv.v[(size_t)i];
        int a = (i == l0) ? std::min(c0, (int)ln.size()) : 0, b = (i == l1) ? std::min(c1, (int)ln.size()) : (int)ln.size();
        r += ln.substr((size_t)a, (size_t)std::max(0, b - a)); if (i != l1) r += '\n';
    }
    return r;
}

void draw_log(Child& ch, LogView& lv, bool new_lines, const char* id) {
    ImGui::PushID(id);
    if (ImGui::Button("Clear")) { lv.v.clear(); lv.clear_sel(); std::lock_guard<std::mutex> lk(ch.mu); ch.first_idx += ch.lines.size(); ch.lines.clear(); lv.next = ch.first_idx; }
    ImGui::SameLine();
    if (ImGui::Button("Copy all")) { std::string all; for (auto& l : lv.v) { all += l; all += '\n'; } ImGui::SetClipboardText(all.c_str()); }
    ImGui::SameLine(); ImGui::Checkbox("Hide model-loading progress lines", &g_hide_load);
    if (!lv.follow) { ImGui::SameLine(); if (ImGui::Button("Jump to latest")) lv.follow = true; }
    ImGui::BeginChild("##log", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
    const float lh = ImGui::GetTextLineHeightWithSpacing();
    int l0 = 0, c0 = 0, l1 = 0, c1 = 0; const bool has_sel = log_sel_ordered(lv, l0, c0, l1, c1);
    // where the mouse is, as (line, byte): lines are fixed-height, so the line comes from the y position
    const ImVec2 origin = ImGui::GetCursorScreenPos();            // top-left of line 0 (already includes scrolling)
    const ImVec2 mp = ImGui::GetIO().MousePos;
    const bool child_hov = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    auto pos_at = [&](int& L, int& C) {
        L = (int)std::floor((mp.y - origin.y) / lh); L = std::max(0, std::min(L, (int)lv.v.size() - 1));
        C = lv.v.empty() ? 0 : log_col_at(lv.v[(size_t)L], mp.x - origin.x);
    };
    if (!lv.v.empty()) {
        const bool any_popup = ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel);   // never start/clear a selection while the context menu is up
        if (child_hov && !any_popup && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mp.x < ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize
            && mp.y < ImGui::GetWindowPos().y + ImGui::GetWindowHeight() - ImGui::GetStyle().ScrollbarSize) {
            int L, C; pos_at(L, C);
            if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) { lv.sa_l = lv.sb_l = L; lv.sa_c = 0; lv.sb_c = (int)lv.v[(size_t)L].size(); lv.dragging = false; }   // double click = whole line
            else { lv.sa_l = lv.sb_l = L; lv.sa_c = lv.sb_c = C; lv.dragging = true; }
        }
        if (lv.dragging) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                int L, C; pos_at(L, C); lv.sb_l = L; lv.sb_c = C;
                const ImVec2 wp = ImGui::GetWindowPos(); const float wh = ImGui::GetWindowHeight();   // dragging past the edge scrolls
                if (mp.y < wp.y + 4) ImGui::SetScrollY(ImGui::GetScrollY() - lh * 0.6f);
                else if (mp.y > wp.y + wh - 4) ImGui::SetScrollY(ImGui::GetScrollY() + lh * 0.6f);
            } else lv.dragging = false;
        }
    }
    ImGuiListClipper clip; clip.Begin((int)lv.v.size(), lh);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    while (clip.Step()) for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
        const std::string& ln = lv.v[(size_t)i];
        const ImVec2 p = ImGui::GetCursorScreenPos();
        if (has_sel && i >= l0 && i <= l1) {                      // highlight the selected part of this line
            float xa = (i == l0) ? log_x_of(ln, c0) : 0.0f, xb = (i == l1) ? log_x_of(ln, c1) : log_x_of(ln, (int)ln.size()) + 6.0f;
            if (xb > xa) dl->AddRectFilled(ImVec2(p.x + xa, p.y), ImVec2(p.x + xb, p.y + lh), IM_COL32(60, 110, 200, 120));
        }
        ImGui::PushStyleColor(ImGuiCol_Text, log_color(ln)); ImGui::TextUnformatted(ln.c_str()); ImGui::PopStyleColor();
    }
    clip.End();
    // right mouse button: Copy / Select all / Clear selection
    static std::string s_ctx_sel;                                  // selection as it was when the menu opened
    if (child_hov && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) s_ctx_sel = log_sel_text(lv);
    if (ImGui::BeginPopupContextWindow("##logctx", ImGuiPopupFlags_MouseButtonRight)) {
        if (ImGui::MenuItem("Copy", nullptr, false, !s_ctx_sel.empty() || has_sel)) ImGui::SetClipboardText((s_ctx_sel.empty() ? log_sel_text(lv) : s_ctx_sel).c_str());
        if (ImGui::MenuItem("Select all", nullptr, false, !lv.v.empty())) { lv.sa_l = 0; lv.sa_c = 0; lv.sb_l = (int)lv.v.size() - 1; lv.sb_c = (int)lv.v.back().size(); }
        if (ImGui::MenuItem("Copy all", nullptr, false, !lv.v.empty())) { std::string all; for (auto& l : lv.v) { all += l; all += '\n'; } ImGui::SetClipboardText(all.c_str()); }
        ImGui::Separator();
        if (ImGui::MenuItem("Clear selection", nullptr, false, has_sel)) lv.clear_sel();
        ImGui::EndPopup();
    }
    if (has_sel && ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows) && ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C, false)) ImGui::SetClipboardText(log_sel_text(lv).c_str());
    // follow new lines only while the reader is at (or within a few lines of) the bottom -- never while the user is selecting
    if (lv.follow && new_lines && !lv.dragging) ImGui::SetScrollHereY(1.0f);
    float at = ImGui::GetScrollY(), mx = ImGui::GetScrollMaxY();
    if (!new_lines || !lv.follow) {
        bool at_bottom = at >= mx - ImGui::GetTextLineHeight() * 2;
        if (at_bottom) lv.follow = true; else if (ImGui::IsWindowHovered() && ImGui::GetIO().MouseWheel > 0) lv.follow = false;
        else if (!at_bottom && (lv.dragging || ImGui::IsMouseDragging(ImGuiMouseButton_Left)) && ImGui::IsWindowHovered()) lv.follow = false;
        else if (!at_bottom && ImGui::IsWindowFocused()) lv.follow = false;
    }
    ImGui::EndChild();
    ImGui::PopID();
}

// ───────────────────────────── donor (worker) tab ────────────────────────────

struct WorkerStatus { std::string state = "stopped", primary = "-", layers = "-", activity = "idle"; long tokens = 0; } g_ws;

void parse_worker_line(const std::string& line) {
    std::string lo = lower(line); std::smatch m;
    static const std::regex r_layers(R"(layers?\s*[range:]*\s*(\d+)\s*[-\xe2\x80\x93]\s*(\d+))"), r_primary(R"((?:primary|connection from|connected[^:]*:)\s*([\d.]+))"),
                            r_tokens(R"(tokens?[:\s]+(\d+))"), r_pct(R"((\d+)%)");
    if (lo.find("receiving weights") != std::string::npos || lo.find("weight stream") != std::string::npos) { g_ws.state = "receiving weights"; g_ws.activity = "Receiving model weights..."; }
    if (lo.find("loading local model shard") != std::string::npos || lo.find("local shard layers") != std::string::npos) { g_ws.state = "loading local shard"; g_ws.activity = "Loading local model shard..."; }
    if (lo.find("local model shard loaded") != std::string::npos) { g_ws.state = "waiting"; g_ws.activity = "local shard ready - waiting for tokens"; }
    if (std::regex_search(lo, m, r_layers)) { int s = atoi(m[1].str().c_str()), e = atoi(m[2].str().c_str()); g_ws.layers = std::to_string(s) + " - " + std::to_string(e) + "  (" + std::to_string(e - s) + " layers)"; }
    if (std::regex_search(lo, m, r_primary)) { g_ws.primary = m[1].str(); g_ws.state = "connected"; g_ws.activity = "waiting for tokens..."; }
    if (std::regex_search(lo, m, r_tokens)) g_ws.tokens = atol(m[1].str().c_str());
    if (lo.find("processing pos") != std::string::npos || (lo.find("forward") != std::string::npos && lo.find("layer") != std::string::npos)) { g_ws.state = "processing"; g_ws.activity = "processing hidden state..."; }
    if (lo.find("waiting for") != std::string::npos && lo.find("primary") == std::string::npos && g_ws.state == "processing") { g_ws.state = "connected"; g_ws.activity = "idle - waiting for next token"; }
    if (std::regex_search(line, m, r_pct) && lo.find("weight") != std::string::npos) {
        int pct = atoi(m[1].str().c_str()); g_ws.activity = "Loading weights: " + std::to_string(pct) + "%";
        if (pct >= 100) { g_ws.state = "waiting"; g_ws.activity = "weights ready - waiting for primary"; }
    }
    if (lo.find("started on port") != std::string::npos || lo.find("listening on port") != std::string::npos) { g_ws.state = "waiting"; g_ws.activity = "listening - tell the primary: " + local_ip() + ":" + (g_donor["port"].empty() ? std::string("9876") : g_donor["port"]); }
}

bool start_worker() {
    SMap& D = g_donor; std::string pass = trim(D["pass"]);
    if (pass.empty()) { g_error = "Enter a passphrase. The primary computer must use the same word to connect."; return false; }
    int port = atoi(trim(D["port"]).c_str()); if (port < 1 || port > 65535) { g_error = "The listen port must be 1-65535."; return false; }
    std::vector<std::string> a = {"--dist-worker", "--dist-passphrase", pass, "--port", std::to_string(port), "--host", D["bind"].empty() ? "0.0.0.0" : D["bind"],
                                  "--threads", std::to_string(std::max(1, atoi(trim(D["threads"]).c_str())))};
    if (D["useLocal"] == "1") {
        std::string p = trim(D["model"]);
        if (!is_file(p)) { g_error = "\"I already have the model file\" is ticked, but that file was not found:\n" + p; return false; }
        a.push_back("--dist-local-model"); a.push_back(p);
    }
    const int gl = atoi(trim(D["gpuLayers"]).c_str());
    if (D["gpuSel"] != "cpu" && !D["gpuSel"].empty() && gl > 0) { a.push_back("--gpu-layers"); a.push_back(std::to_string(gl)); a.push_back("--gpu-device"); a.push_back(D["gpuSel"]); }
    Child& w = g_children[kWorkerChild]; std::string err;
    g_ws = WorkerStatus(); g_ws.state = "waiting";
    if (!child_start(w, a, {}, err)) { g_error = err; return false; }
    return true;
}

void draw_donor(bool new_lines) {
    g_m = &g_donor; Child& w = g_children[kWorkerChild]; const bool run = child_alive(w);
    ImGui::Dummy(ImVec2(0, 4));
    ImGui::TextWrapped("Use this on a second computer that should lend its processor and graphics card. The main computer connects to it from \"Share with other computers\" on the Setup tab.");
    ImGui::Spacing();
    ImGui::TextColored(kMute, "This computer's address:"); ImGui::SameLine(); static std::string ip = local_ip(); ImGui::TextColored(ImVec4(0.4f, 0.85f, 1, 1), "%s", ip.c_str());
    ImGui::SameLine(); if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(ip.c_str());
    ImGui::BeginDisabled(run);
    if (Card("Connection", true)) {
        { Grid g("dn", 3);
          g.next(); Txt("port", "Listen port", "9876");
          g.next(); Sel("bind", "Who can connect", {{"0.0.0.0", "Anyone on my network"}, {"127.0.0.1", "Only this computer"}});
          g.next(); Txt("pass", "Passphrase (the same word on both computers)", "", nullptr, ImGuiInputTextFlags_Password); }
        Hint("The passphrase is never saved to disk.");
    }
    if (Card("Model", true)) {
        Chk("useLocal", "I already have the model file on this computer", "Then the main computer sends only instructions, not gigabytes of weights. The file must be the same model.");
        ImGui::BeginDisabled(g_donor["useLocal"] != "1"); PathField("model", "Model file", "Choose the model file", {"gguf", "nxm"}); ImGui::EndDisabled();
    }
    if (Card("Hardware", true)) {
        { Grid g("dh", 3);
          g.next(); Txt("threads", "CPU threads", "");
          g.next(); Sel("gpuSel", "Graphics card", g_gpus);
          g.next(); Txt("gpuLayers", "Layers kept on the card (0 = none)", ""); }
    }
    ImGui::EndDisabled();
    ImGui::Spacing();
    if (!run) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.43f, 0.92f, 1));
        if (ImGui::Button("  Start donor  ")) start_worker();
        ImGui::PopStyleColor();
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.65f, 0.2f, 0.18f, 1));
        if (ImGui::Button("  Stop donor  ")) std::thread([] { child_stop(g_children[kWorkerChild]); g_ws = WorkerStatus(); }).detach();
        ImGui::PopStyleColor();
    }
    const std::string st = run ? g_ws.state : "stopped";
    ImGui::SameLine(0, 20); ImGui::TextColored(st == "stopped" ? kRed : st == "connected" || st == "processing" ? kGreen : kAmber, "o"); ImGui::SameLine(); ImGui::TextUnformatted(st.c_str());
    ImGui::Text("Main computer: %s     Layers: %s     Tokens handled: %ld", g_ws.primary.c_str(), g_ws.layers.c_str(), g_ws.tokens);
    ImGui::TextColored(kMute, "%s", run ? g_ws.activity.c_str() : "idle");
    ImGui::Spacing();
    ImGui::BeginChild("##wlogbox", ImVec2(0, 0), ImGuiChildFlags_None);
    draw_log(w, g_wlog, new_lines, "wlog");
    ImGui::EndChild();
}

// ───────────────────────────── frame ─────────────────────────────────────────

bool g_quit_ask = false; int g_active_tab = 0;

// loading progress for one model (nothing is drawn unless it is starting up)
static bool loading_now(Slot& sl) { return child_alive(g_children[&sl - g_slots]) && !sl.ready.load() && sl.log.load_pct >= 0; }
void draw_progress(Slot& sl) {
    if (!loading_now(sl)) return;
    char t[160]; snprintf(t, sizeof t, "%d%%  %s", sl.log.load_pct, sl.log.load_stage.c_str());
    ImGui::ProgressBar(sl.log.load_pct / 100.0f, ImVec2(-FLT_MIN, 0), t); ImGui::Spacing();
}

void draw_slot_strip() {
    for (int i = 0; i < g_ns; ++i) {
        Slot& sl = g_slots[i]; Child& ch = g_children[i]; ImGui::PushID(i);
        const bool run = child_alive(ch), ready = run && sl.ready.load();
        ImGui::AlignTextToFramePadding(); ImGui::TextColored(run ? (ready ? kGreen : kAmber) : (ch.state.load() == 2 ? kRed : kMute), "o"); ImGui::SameLine(0, 4);
        std::string nm = sl.cur.empty() ? "no model" : sl.cur; if (nm.size() > 22) nm = nm.substr(0, 20) + "..";
        std::string label = "Model " + std::to_string(i + 1) + ": " + nm; if (loading_now(sl)) label += "  " + std::to_string(sl.log.load_pct) + "%";
        if (i == g_cs) { ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.28f, 0.45f, 1)); }
        if (ImGui::Button(label.c_str())) { g_cs = i; changed(); }
        if (i == g_cs) ImGui::PopStyleColor();
        ImGui::SameLine(0, 14); ImGui::PopID();
    }
    if (g_ns < kMaxSlots && ImGui::Button("+ Add a model")) { init_slot(g_ns, nullptr); g_cs = g_ns; ++g_ns; changed(); }
    if (g_ns > 1) {
        ImGui::SameLine(0, 14);
        const bool last = g_cs == g_ns - 1;
        ImGui::BeginDisabled(!last || child_alive(g_children[g_cs]));
        if (ImGui::Button("Remove this model")) { --g_ns; g_cs = std::min(g_cs, g_ns - 1); changed(); }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("Only the last model can be removed, and only when it is stopped.");
    }
}

std::string g_last_url; double g_url_until = 0;

void draw_ui(int W, int H, bool new_lines) {
    ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(ImVec2((float)W, (float)H));
    ImGui::Begin("##main", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);
    {   // logo: a diamond with a smaller solid diamond inside (the Python launcher's accent mark)
        ImDrawList* dl = ImGui::GetWindowDrawList(); ImVec2 p = ImGui::GetCursorScreenPos(); float h = ImGui::GetFrameHeight(), r = h * 0.5f;
        ImVec2 c(p.x + r, p.y + r); ImU32 col = IM_COL32(0x7c, 0x6a, 0xf7, 255);
        auto dia = [&](float k) { return std::array<ImVec2, 4>{ImVec2(c.x, c.y - r * k), ImVec2(c.x + r * k, c.y), ImVec2(c.x, c.y + r * k), ImVec2(c.x - r * k, c.y)}; };
        auto o = dia(0.95f); dl->AddPolyline(o.data(), 4, col, ImDrawFlags_Closed, 2.0f);
        auto n = dia(0.5f); dl->AddConvexPolyFilled(n.data(), 4, col);
        ImGui::Dummy(ImVec2(h, h)); ImGui::SameLine(0, 6);
        ImGui::AlignTextToFramePadding(); ImGui::TextUnformatted("Nexus"); ImGui::SameLine(0, 0);
        if (W >= 1500) { ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.55f, 0.57f, 0.65f, 1)); ImGui::TextUnformatted("  Multi-Model Inference Launcher"); ImGui::PopStyleColor(); ImGui::SameLine(0, 24); }
        else ImGui::SameLine(0, 18);
        // settings are written only when this button is pressed -- an experiment (a test variable, a changed number) is gone after a restart
        const bool was_dirty = g_dirty;                          // saving clears g_dirty, so remember it for the colour pop below
        if (was_dirty) { ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.40f, 0.08f, 1)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.70f, 0.52f, 0.12f, 1)); }
        const bool save_clicked = ImGui::Button(was_dirty ? "Save current settings *" : "Save current settings");
        if (was_dirty) ImGui::PopStyleColor(2);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(was_dirty ? "You changed something. It is used now, but forgotten after a restart unless you save." : "Everything is saved.");
        if (save_clicked) save_now();
        ImGui::SameLine(0, 18);
    }
    if (ImGui::BeginTabBar("##tabs", ImGuiTabBarFlags_NoTabListScrollingButtons)) {
        auto tab = [&](const char* name, int idx, std::function<void()> body) {
            ImGuiTabItemFlags fl = g_req_tab == idx ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem(name, nullptr, fl)) {
                g_active_tab = idx;
                ImGui::BeginChild("##body", ImVec2(0, 0), ImGuiChildFlags_None, (idx == 2 || idx == 3) ? ImGuiWindowFlags_NoScrollbar : 0);
                body(); ImGui::EndChild(); ImGui::EndTabItem();
            }
        };
        tab("Setup", 0, [&] { ImGui::Dummy(ImVec2(0, 2)); draw_slot_strip(); ImGui::Spacing(); draw_progress(cs()); draw_setup(cs(), g_cs); });
        tab("Log", 2, [&] { ImGui::Dummy(ImVec2(0, 2)); draw_slot_strip(); ImGui::Spacing(); draw_progress(cs()); draw_log(g_children[g_cs], cs().log, new_lines, "slotlog"); });
        tab("Donor", 3, [&] { draw_donor(new_lines); });
        g_req_tab = -1;
        ImGui::EndTabBar();
    }
    ImGui::End();

    // status + Start/Stop for the model being shown: a small floating window pinned to the top right
    if (g_active_tab != 3) {
        Child& ch = g_children[g_cs]; Slot& sl = cs();
        const int st = ch.state.load(); const bool running = st == 1, ready = running && sl.ready.load();
        std::string status = running ? (ready ? "Running - " + ch.model : "Loading " + ch.model + "...") : (st == 2 ? "Stopped (exit " + std::to_string(ch.exit_code.load()) + ")" : "Stopped");
        ImVec4 dot = running ? (ready ? kGreen : kAmber) : (st == 2 ? kRed : kMute);
        ImGui::SetNextWindowPos(ImVec2((float)W - 8, 6), ImGuiCond_Always, ImVec2(1, 0));
        ImGui::Begin("##btn", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoFocusOnAppearing);
        ImGui::AlignTextToFramePadding(); ImGui::TextColored(dot, "o"); ImGui::SameLine(); ImGui::TextUnformatted(status.c_str()); ImGui::SameLine(0, 16);
        if (!running) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.43f, 0.92f, 1)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.5f, 0.98f, 1));
            if (ImGui::Button("  Start  ") && start_slot(g_cs)) g_req_tab = 2;
            ImGui::PopStyleColor(2);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.65f, 0.2f, 0.18f, 1)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.8f, 0.27f, 0.23f, 1));
            if (ImGui::Button("  Stop  ")) stop_slot_async(g_cs);
            ImGui::PopStyleColor(2);
        }
        if (ready) { ImGui::SameLine(0, 10); if (ImGui::Button("Open chat in browser")) { g_last_url = "http://127.0.0.1:" + sl.port + "/"; ImGui::SetClipboardText(g_last_url.c_str()); open_browser(g_last_url); g_url_until = ImGui::GetTime() + 12; } }
        if (g_ns > 1) {
            ImGui::SameLine(0, 10); if (ImGui::Button("Start all")) for (int i = 0; i < g_ns; ++i) if (!path_of(g_slots[i]).empty()) start_slot(i);
            ImGui::SameLine(); if (ImGui::Button("Stop all")) for (int i = 0; i < g_ns; ++i) if (child_alive(g_children[i])) stop_slot_async(i);
        }
        ImGui::End();
        if (ImGui::GetTime() < g_url_until && !g_last_url.empty()) {
            ImGui::SetNextWindowPos(ImVec2((float)W - 8, 44), ImGuiCond_Always, ImVec2(1, 0));
            ImGui::Begin("##url", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing);
            ImGui::TextUnformatted("Address (also copied):"); ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::CalcTextSize(g_last_url.c_str()).x + 16); InText("##u", &g_last_url, ImGuiInputTextFlags_ReadOnly);
            if (ImGui::IsItemActive() || ImGui::IsItemHovered()) g_url_until = ImGui::GetTime() + 12;
            ImGui::End();
        }
    }

    draw_browse();

    ImGuiViewport* vp = ImGui::GetMainViewport();
    if (!g_error.empty()) ImGui::OpenPopup("Problem");
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Problem", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 28); ImGui::TextUnformatted(g_error.c_str()); ImGui::PopTextWrapPos(); ImGui::Spacing();
        if (ImGui::Button("OK", ImVec2(120, 0))) { g_error.clear(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
    if (g_quit_ask) ImGui::OpenPopup("Quit?");
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal("Quit?", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::TextUnformatted("Something is still running. Stop everything and quit?"); ImGui::Spacing();
        if (ImGui::Button("Stop and quit", ImVec2(150, 0))) { stop_everything(); write_roster(); plat::destroy(); exit(0); }
        ImGui::SameLine(); if (ImGui::Button("Keep running", ImVec2(150, 0))) { g_quit_ask = false; ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }
}

void set_style(float scale) {
    ImGuiStyle& s = ImGui::GetStyle(); ImGui::StyleColorsDark(&s);
    s.WindowRounding = 0; s.FrameRounding = 6; s.GrabRounding = 6; s.PopupRounding = 8; s.ChildRounding = 8; s.TabRounding = 6; s.ScrollbarRounding = 8;
    s.FramePadding = ImVec2(9, 6); s.ItemSpacing = ImVec2(10, 7); s.WindowPadding = ImVec2(16, 12); s.ScrollbarSize = 13; s.FrameBorderSize = 1; s.TabBarBorderSize = 0;
    ImVec4* c = s.Colors;
    c[ImGuiCol_WindowBg] = ImVec4(0.071f, 0.086f, 0.109f, 1); c[ImGuiCol_ChildBg] = ImVec4(0.071f, 0.086f, 0.109f, 1); c[ImGuiCol_PopupBg] = ImVec4(0.102f, 0.125f, 0.161f, 1);
    c[ImGuiCol_Border] = ImVec4(0.165f, 0.196f, 0.239f, 1); c[ImGuiCol_Text] = ImVec4(0.90f, 0.92f, 0.94f, 1);
    c[ImGuiCol_FrameBg] = ImVec4(0.102f, 0.125f, 0.161f, 1); c[ImGuiCol_FrameBgHovered] = ImVec4(0.14f, 0.17f, 0.21f, 1); c[ImGuiCol_FrameBgActive] = ImVec4(0.16f, 0.2f, 0.25f, 1);
    c[ImGuiCol_Header] = ImVec4(0.102f, 0.125f, 0.161f, 1); c[ImGuiCol_HeaderHovered] = ImVec4(0.15f, 0.19f, 0.25f, 1); c[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.23f, 0.3f, 1);
    c[ImGuiCol_Button] = ImVec4(0.14f, 0.17f, 0.21f, 1); c[ImGuiCol_ButtonHovered] = ImVec4(0.2f, 0.25f, 0.32f, 1); c[ImGuiCol_ButtonActive] = ImVec4(0.25f, 0.31f, 0.4f, 1);
    c[ImGuiCol_Tab] = ImVec4(0.071f, 0.086f, 0.109f, 1); c[ImGuiCol_TabHovered] = ImVec4(0.15f, 0.19f, 0.25f, 1); c[ImGuiCol_TabSelected] = ImVec4(0.14f, 0.17f, 0.21f, 1);
    c[ImGuiCol_CheckMark] = ImVec4(0.3f, 0.55f, 1, 1); c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0); c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.5f);
    s.ScaleAllSizes(scale);
}

}  // namespace

// ───────────────────────────── main ──────────────────────────────────────────

static void load_fonts(float px) {
    ImGuiIO& io = ImGui::GetIO();
    ImFontGlyphRangesBuilder b; b.AddRanges(io.Fonts->GetGlyphRangesDefault());
    static const ImWchar extra[] = {0x2010, 0x2027, 0x2190, 0x2193, 0};
    b.AddRanges(extra);
    static ImVector<ImWchar> ranges; b.BuildRanges(&ranges);
    ImFontConfig cfg; cfg.OversampleH = 2; cfg.OversampleV = 1; cfg.PixelSnapH = true;
    io.Fonts->AddFontFromMemoryCompressedBase85TTF(RobotoMedium_compressed_data_base85, px, &cfg, ranges.Data);
}

int main(int argc, char** argv) {
    std::string shot, tab = "setup"; int sw = 1100, sh = 820; float force_scale = 0; int shot_slot = 0; bool donor_mode = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shot = argv[++i];
        else if (a == "--tab" && i + 1 < argc) tab = argv[++i];
        else if (a == "--slot" && i + 1 < argc) shot_slot = atoi(argv[++i]);
        else if (a == "--w" && i + 1 < argc) sw = atoi(argv[++i]);
        else if (a == "--h" && i + 1 < argc) sh = atoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) force_scale = (float)atof(argv[++i]);
        else if (a == "--expand") g_expand_all = true;
        else if (a == "--donor") donor_mode = true;
        else if (a == "--dir" && i + 1 < argc) g_dir = argv[++i];
        else if (a == "--help" || a == "-h") { printf("nexus_launcher            open the launcher window\nnexus_launcher --donor    open on the Donor tab\nnexus_launcher --shot F.ppm [--tab setup|log|donor] [--slot N] [--w N --h N] [--expand] [--scale F]\n"); return 0; }
    }
    if (g_dir.empty()) g_dir = self_dir();
    g_cfg_path = join(g_dir, "nexus_launcher.json");
#if defined(_WIN32)
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#else
    signal(SIGPIPE, SIG_IGN); signal(SIGINT, on_term); signal(SIGTERM, on_term); signal(SIGHUP, on_term);
#endif
    const bool headless = !shot.empty();
    if (!headless && !plat::create(donor_mode ? "Nexus Donor" : "Nexus", 1100, 820)) {
        fprintf(stderr, "[launcher] Could not open a window. Start the server directly instead:  nexus_server --model <file> --port 8080\n");
        return 1;
    }
    float scale = force_scale > 0 ? force_scale : (headless ? 1.0f : plat::dpi_scale());

    IMGUI_CHECKVERSION(); ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); io.IniFilename = nullptr; io.LogFilename = nullptr;
    load_fonts(15.0f * scale); set_style(scale);
    ImGuiPlatformIO& pio = ImGui::GetPlatformIO();
    if (!headless) {
        pio.Platform_SetClipboardTextFn = [](ImGuiContext*, const char* t) { plat::set_clipboard(t); };
        pio.Platform_GetClipboardTextFn = [](ImGuiContext*) -> const char* { static std::string buf; buf = plat::get_clipboard(); return buf.c_str(); };
    }
    unsigned char* tp; int tw, th; io.Fonts->GetTexDataAsRGBA32(&tp, &tw, &th);
    swr::Tex font{(const uint32_t*)tp, tw, th};

    // config
    { std::string txt; JV v; if (read_file(g_cfg_path, txt) && json_parse(txt, v) && v.t == JV::Obj) g_cfg = v; else g_cfg.t = JV::Obj; }
    const JV* gl = g_cfg.get("global");
    if (gl) {
        g_lan = atoi(jstr(gl->get("lan")).c_str()) ? 1 : 0;
        g_browser_cmd = jstr(gl->get("browser_cmd"));
        if (const JV* d = gl->get("model_dirs")) for (auto& x : d->a) if (x.t == JV::Str) g_dirs.push_back(x.s);
    }
    rebuild_gpu_list(); scan_all();
    const JV* sj = g_cfg.get("slots");
    if (sj && sj->t == JV::Arr && !sj->a.empty()) {
        g_ns = std::min((int)sj->a.size(), kMaxSlots);
        for (int i = 0; i < g_ns; ++i) init_slot(i, &sj->a[(size_t)i]);
        g_cs = std::max(0, std::min(g_ns - 1, atoi(jstr(gl ? gl->get("cur_slot") : nullptr, "0").c_str())));
    } else {                                                    // first run, or a config from the single-model launcher
        JV old; old.t = JV::Obj; if (gl) *jset(old, "path") = jstrv(jstr(gl->get("last_path"))); init_slot(0, &old); g_ns = 1; g_cs = 0;
    }
    if (const JV* dn = g_cfg.get("donor")) for (auto& kv : dn->o) g_donor[kv.first] = jstr(&kv.second);
    if (g_donor["port"].empty()) g_donor["port"] = "9876";
    if (g_donor["bind"].empty()) g_donor["bind"] = "0.0.0.0";
    if (g_donor["gpuSel"].empty()) g_donor["gpuSel"] = "cpu";
    if (g_donor["gpuLayers"].empty()) g_donor["gpuLayers"] = "0";
    g_m = &cs().S;

    std::thread([] {
        std::string j = run_probe(); JV v; bool ok = json_parse(j, v) && v.get("cpu");
        { std::lock_guard<std::mutex> lk(g_hw_mu); g_hw = ok ? v : JV(); }
        g_hw_state = ok ? 1 : 2;
    }).detach();
    std::thread([] {                                      // is each server answering yet?
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
            for (int i = 0; i < kMaxSlots; ++i) { int st = g_children[i].state.load(), port = g_children[i].port.load(); g_slots[i].ready = st == 1 && port > 0 && port_open(port); }
        }
    }).detach();
    auto apply_hw = [&] {
        rebuild_gpu_list();
        for (int i = 0; i < g_ns; ++i) fix_hw(g_slots[i]);
        if (trim(g_donor["threads"]).empty()) g_donor["threads"] = std::to_string(physical_cores());
        bool ok = false; for (auto& c : g_gpus) if (c.value == g_donor["gpuSel"]) ok = true; if (!ok) g_donor["gpuSel"] = "cpu";
    };

    std::vector<uint32_t> fb;
    int frames = 4; double last_draw = -1;
    auto t0 = std::chrono::steady_clock::now();
    auto pull_all = [&](bool& cur_new, bool& worker_new) {
        bool any = false; cur_new = false;
        for (int i = 0; i < g_ns; ++i) { bool n = pull_log(g_children[i], g_slots[i].log); any |= n; if (i == g_cs) cur_new = n; }
        worker_new = pull_log(g_children[kWorkerChild], g_wlog, parse_worker_line);
        return any || worker_new;
    };
    auto render_frame = [&](int W, int H, bool new_lines) {
        io.DisplaySize = ImVec2((float)W, (float)H); io.DeltaTime = 1.0f / 60.0f;
        ImGui::NewFrame(); draw_ui(W, H, new_lines); ImGui::Render();
        fb.resize((size_t)W * H);
        swr::render(ImGui::GetDrawData(), fb.data(), W, H, 0x00121620, font);
    };
    if (headless) {
        g_req_tab = tab == "log" ? 2 : tab == "donor" ? 3 : 0;
        g_cs = std::max(0, std::min(g_ns - 1, shot_slot)); g_m = &cs().S;
        for (int i = 0; i < 20 && g_hw_state.load() == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (g_hw_state.load() != 0) apply_hw();
        for (int i = 0; i < 4; ++i) { bool a, b; pull_all(a, b); render_frame(sw, sh, false); }
        FILE* f = fopen(shot.c_str(), "wb"); if (!f) return 1;
        fprintf(f, "P6\n%d %d\n255\n", sw, sh);
        for (size_t i = 0; i < (size_t)sw * sh; ++i) { unsigned char px[3] = {(unsigned char)(fb[i] >> 16), (unsigned char)(fb[i] >> 8), (unsigned char)fb[i]}; fwrite(px, 1, 3, f); }
        fclose(f); return 0;
    }
    if (donor_mode) g_req_tab = 3;

    for (;;) {
        bool got = false;
        if (!plat::pump(frames > 0 ? 0 : 60, got)) {
            if (any_running()) { plat::cancel_close(); g_quit_ask = true; frames = 4; }
            else break;
        }
        g_now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (got) frames = 4;
        bool cur_new = false, worker_new = false; bool any_new = pull_all(cur_new, worker_new);
        if (g_hw_state.load() != 0 && !g_hw_applied) { g_hw_applied = true; apply_hw(); frames = 3; }
        for (int i = 0; i < g_ns; ++i) {
            bool ready = g_slots[i].ready.load();
            if (ready != g_slots[i].was_ready) { g_slots[i].was_ready = ready; frames = 3; write_roster(); }
        }
        if (any_new) frames = std::max(frames, 2);
        if (frames <= 0 && g_now - last_draw < 0.5) continue;       // idle: redraw only twice a second (status, log)
        int W, H; plat::size(W, H); if (W < 100 || H < 100) continue;
        render_frame(W, H, cur_new || worker_new);
        plat::present(fb.data(), W, H);
        last_draw = g_now; if (frames > 0) --frames;
    }
    stop_everything();
    plat::destroy();
    ImGui::DestroyContext();
    return 0;
}
