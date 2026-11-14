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
    ImGui::SameLine(); Hint("Drag with the mouse to select text, then right-click to copy.");
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
        if (child_hov && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mp.x < ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize
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
    if (ImGui::BeginPopupContextWindow("##logctx", ImGuiPopupFlags_MouseButtonRight)) {
        if (ImGui::MenuItem("Copy", nullptr, false, has_sel)) ImGui::SetClipboardText(log_sel_text(lv).c_str());
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
