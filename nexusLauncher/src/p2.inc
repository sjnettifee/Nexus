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
    ImGui::SameLine(); if (ImGui::Button("Rescan")) { scan_all(); for (int i = 0; i < g_ns; ++i) { std::string p = path_of(g_slots[i]); attach(g_slots[i], i, p); fix_hw(g_slots[i]); } }
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
        if (ImGui::BeginCombo("##prof", sl.profile.empty() ? "(none - model's own settings)" : sl.profile.c_str())) {
            if (ImGui::Selectable("(none - model's own settings)", sl.profile.empty())) { sl.profile.clear(); load_model_settings(sl); sl.profile.clear(); changed(); }
            for (auto& n : names) if (ImGui::Selectable(n.c_str(), n == sl.profile)) { std::string keep = n; load_model_settings(sl); profile_load(sl, keep); changed(); break; }
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
        Hint("A profile remembers every setting of this model (sampling, graphics cards, threads, extra variables). Settings are only kept when you press \"Save current settings\" at the top, or save them into a profile here.");
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

