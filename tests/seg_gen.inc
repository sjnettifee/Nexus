namespace {
struct NxTurnMarkers {
    std::vector<int32_t> starts;      // tokens that begin a turn
    int32_t assistant = -1;           // a distinct assistant-turn marker (<|assistant|>), if the template has one
    int32_t system    = -1;           // a distinct system-turn marker ([SYSTEM_PROMPT]), if any
    bool    exchange  = false;        // question and answer share one block ([INST] ... [/INST] answer)
};
bool nx_ctrl_text(const std::string& t) {
    return t.size() >= 3 && ((t.front() == '<' && t.back() == '>') || (t.front() == '[' && t.back() == ']'));
}
std::string nx_lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }
}

// Turn markers come from the model's own chat template: format a user turn around a marker
// string, and the first token before it is what starts every turn in this format.
static NxTurnMarkers nx_turn_markers(const MockEngine& e) {
    NxTurnMarkers m;
    const std::string MARK = "NXTURNMARK";
    std::string ut;
    try { ut = e.format_user_turn(MARK, -1); } catch (...) { return m; }
    const size_t i = ut.find(MARK);
    if (i == std::string::npos || i == 0) return m;
    const std::vector<int32_t> tb = e.tokenize(ut.substr(0, i), false);
    if (tb.empty() || !nx_ctrl_text(e.detokenize(tb[0]))) return m;    // e.g. Alpaca "###": not a reliable marker
    const int32_t user_start = tb[0];
    m.starts.push_back(user_start);
    const std::vector<int32_t> ta = e.tokenize(ut.substr(i + MARK.size()), false);
    int ctrl_seen = 0;
    for (int32_t t : ta) {                       // first control token after the user turn's own closing one
        if (!nx_ctrl_text(e.detokenize(t))) continue;
        if (++ctrl_seen == 1 && t != user_start) continue;
        if (t != user_start) { m.assistant = t; m.starts.push_back(t); }
        break;
    }
    if (m.assistant < 0 && std::find(ta.begin(), ta.end(), user_start) == ta.end()) m.exchange = true;
    try {
        const std::string sp = e.format_system_prefix(std::vector<nexus::ChatMessage>{nexus::ChatMessage{"system", "NXSYSMARK"}});
        // A distinct system-turn marker ([SYSTEM_PROMPT], <|system|>) is the first control token of
        // the system prefix -- unless that is a beginning-of-text token (<bos>, <s>, <|begin_of_text|>),
        // recognisable because a normal turn marker follows it straight away.
        std::vector<int32_t> ctrl;
        for (int32_t t : e.tokenize(sp, false)) if (nx_ctrl_text(e.detokenize(t))) ctrl.push_back(t);
        auto is_start = [&](int32_t t) { return std::find(m.starts.begin(), m.starts.end(), t) != m.starts.end(); };
        if (!ctrl.empty() && !is_start(ctrl[0]) && !(ctrl.size() > 1 && is_start(ctrl[1]))) {
            m.system = ctrl[0]; m.starts.push_back(ctrl[0]);
        }
    } catch (...) {}
    return m;
}

static std::string nx_turn_role(const MockEngine& e, const NxTurnMarkers& m,
                                const std::vector<int32_t>& toks, int a, int b) {
    const int32_t t = toks[(size_t)a];
    if (t == m.assistant) return "assistant";
    if (t == m.system)    return "system";
    const int vocab = (int)e.vocab_size();
    std::string head = nx_lower(e.detokenize(t));                 // e.g. <|user|>, <start_of_turn>
    if (head.find("assistant") != std::string::npos || head.find("model") != std::string::npos) return "assistant";
    if (head.find("system") != std::string::npos) return "system";
    if (head.find("user") != std::string::npos) return m.exchange ? "exchange" : "user";
    if (m.exchange) return "exchange";
    std::string next;
    for (int k = a + 1; k < b && k < a + 16 && next.size() < 24; ++k)
        if (toks[(size_t)k] >= 0 && toks[(size_t)k] < vocab) next += e.detokenize(toks[(size_t)k]);
    size_t q = 0;
    while (q < next.size() && (std::isspace((unsigned char)next[q]) || next[q] == '|' || next[q] == '>')) ++q;
    const std::string w = nx_lower(next.substr(q));
    auto starts = [&](const char* x) { return w.compare(0, strlen(x), x) == 0; };
    if (starts("user") || starts("human")) return "user";
    if (starts("assistant") || starts("model") || starts("gpt") || starts("bot")) return "assistant";
    if (starts("system") || starts("developer")) return "system";
    if (starts("tool") || starts("ipython") || starts("function")) return "tool";
    return "turn";
}

static std::string nx_turn_preview(const MockEngine& e, const std::vector<int32_t>& toks, int a, int b, const std::string& role) {
    std::string out;
    const int vocab = (int)e.vocab_size();
    for (int k = a; k < b && out.size() < 400; ++k) {
        const int32_t t = toks[(size_t)k];
        if (t < 0 || t >= vocab) { out += " [image] "; continue; }
        const std::string x = e.detokenize(t);
        if (nx_ctrl_text(x)) { out += ' '; continue; }
        out += x;
    }
    std::string c; bool sp = true;                    // collapse whitespace
    for (char ch : out) { if (std::isspace((unsigned char)ch)) { if (!sp) c += ' '; sp = true; } else { c += ch; sp = false; } }
    while (!c.empty() && c.back() == ' ') c.pop_back();
    for (const char* w : {"user", "assistant", "model", "system"})     // drop the role word of the header
        if (role != "turn" && nx_lower(c).compare(0, strlen(w), w) == 0) { c.erase(0, strlen(w)); break; }
    while (!c.empty() && (c[0] == ' ' || c[0] == ':')) c.erase(0, 1);
    if (c.size() > 120) { c.resize(120); while (!c.empty() && (c.back() & 0xC0) == 0x80) c.pop_back(); c += "..."; }
    return c;
}


std::vector<ContextTurn> turns_of(const MockEngine& self_, const std::vector<int32_t>& toks_, int pinned) {
    std::vector<ContextTurn> out;
    const auto& toks = toks_;
    const int n = (int)toks.size();
    if (n == 0) return out;
    const NxTurnMarkers m = nx_turn_markers(self_);
    std::vector<int> starts;
    if (!m.starts.empty())
        for (int i = 0; i < n; ++i)
            if (std::find(m.starts.begin(), m.starts.end(), toks[(size_t)i]) != m.starts.end()) starts.push_back(i);
    auto add = [&](int a, int b, const std::string& role) {
        ContextTurn t; t.start = a; t.end = b; t.role = role;
        t.preview = nx_turn_preview(self_, toks, a, b, role);
        out.push_back(t);
    };
    if (starts.empty()) { add(0, n, "turn"); return out; }   // no turn markers in this chat format: one block
    else {
        if (starts[0] > 0) add(0, starts[0], "banner");
        for (size_t k = 0; k < starts.size(); ++k) {
            const int a = starts[k], b = k + 1 < starts.size() ? starts[k + 1] : n;
            add(a, b, nx_turn_role(self_, m, toks, a, b));
        }
    }
    // Protected start: whatever comes before the first turn (BOS ...), leading system turns,
    // and anything pinned -- rounded up to whole turns.
    int prot = 0;
    for (const auto& t : out) { if (t.role == "banner" || t.role == "system") prot = t.end; else break; }
    if (pinned > prot)
        for (const auto& t : out) if (t.start < pinned) prot = std::max(prot, t.end);
    for (auto& t : out) {
        t.protected_part = t.end <= prot;
        for (const auto& sec : std::vector<Sec>{})        // sections pinned with context_pin("name")
            if (sec.pinned && sec.start_token < t.end && sec.end_token > t.start) t.protected_part = true;
    }
    return out;
}

// Length of the protected START of the conversation (banner/system prompt/pinned prefix).
