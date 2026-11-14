// detect.h -- which chat template a model uses and whether it has a thinking switch.
// A direct port of the rules in the old Python launcher (detect_format / model_supports_thinking), so the launcher
// passes --chat-format explicitly instead of letting the server guess.
#pragma once
#include "core.h"

struct Detected { std::string fmt, conf; };

static bool has(const std::string& s, const char* w) { return s.find(w) != std::string::npos; }
static bool in_set(const std::string& a, std::initializer_list<const char*> l) { for (auto* x : l) if (a == x) return true; return false; }

static Detected detect_format(const ModelMeta& m, const std::string& filename) {
    std::string arch = lower(m.arch), name = lower(m.name.empty() ? filename : m.name), fname = lower(base_name(filename)), tmpl = lower(m.tmpl);
    std::string n = arch + " " + name + " " + fname;
    auto Q3 = [&] { return in_set(arch, {"qwen3", "qwen3moe", "qwen3_5", "qwen35moe", "qwen3_6", "qwen36moe", "qwen3next"}); };
    if (!tmpl.empty()) {
        if (has(tmpl, "<|start_of_role|>") || has(tmpl, "<|end_of_role|>")) return {"Granite", "high"};
        if (has(tmpl, "<|start_header_id|>") || has(tmpl, "<|eot_id|>") || has(tmpl, "<|begin_of_text|>")) return {"Llama-3", "high"};
        if (has(tmpl, "<|turn>") || has(tmpl, "<turn|>") || has(tmpl, "<|channel>thought")) return {"Gemma 4", "high"};
        if (has(tmpl, "<start_of_turn>") || has(tmpl, "<end_of_turn>")) return {"Gemma", "high"};
        if (arch == "olmoe" || has(n, "olmoe") || (has(tmpl, "<|endoftext|>") && has(tmpl, "<|user|>") && has(tmpl, "<|assistant|>"))) return {"OLMoE", "high"};
        if (has(tmpl, "<|user|>") && (has(tmpl, "<|assistant|>") || has(tmpl, "<|end|>"))) return {"Phi-3", "high"};
        if (has(tmpl, "[system_prompt]")) return {"Mistral V3", "high"};
        if (has(tmpl, "[inst]")) return {"Mistral", "high"};
        if (has(tmpl, "<\xef\xbd\x9cuser\xef\xbd\x9c>") || has(tmpl, "<|user|>")) if (has(n, "deepseek")) return {(has(tmpl, "think") || has(n, "r1")) ? "DeepSeek-R1" : "DeepSeek", "high"};
        if (has(tmpl, "<|system|>") && !has(tmpl, "<|im_start|>")) return {"Zephyr", "high"};
        if (has(tmpl, "<|im_start|>") || has(tmpl, "<|im_end|>")) {
            bool think = has(tmpl, "<think>") || has(tmpl, "</think>") || has(tmpl, "enable_thinking") || has(tmpl, "reasoning_content");
            bool q3 = has(n, "qwen3") || has(n, "qwen-3") || has(n, "qwen 3") || has(n, "qwen-next");
            if (Q3() || think || q3) return {"Qwen3", "high"};
            return {"ChatML", "high"};
        }
        if (has(tmpl, "### instruction") || has(tmpl, "###instruction")) return {"Alpaca", "high"};
    }
    if (arch == "olmoe" || has(n, "olmoe")) return {"OLMoE", "high"};
    if (arch == "granitemoe" || has(n, "granite")) return {"Granite", arch == "granitemoe" ? "high" : "medium"};
    if (Q3()) return {"Qwen3", "high"};
    if (has(n, "qwen3") || has(n, "qwen-3") || has(n, "qwen 3") || has(n, "qwen-next")) return {"Qwen3", "high"};
    if (in_set(arch, {"qwen", "qwen2", "qwen2_5", "qwen2moe"}) || has(n, "qwen")) return {"ChatML", "medium"};
    if (has(n, "deepseek")) {
        if (has(n, "coder") || has(n, "code")) return {"DeepSeek-Coder", "medium"};
        if (has(n, "distill")) {
            if (Q3() || has(n, "qwen3") || has(n, "qwen-3")) return {"Qwen3", "medium"};
            if (has(n, "qwen") || in_set(arch, {"qwen", "qwen2", "qwen2_5", "qwen2moe"})) return {"ChatML", "medium"};
            if (has(n, "llama") || arch == "llama") return {"Llama-3", "medium"};
        }
        return {"DeepSeek", "low"};
    }
    if (has(n, "phi") || arch == "phi3" || arch == "phimoe") return {"Phi-3", "medium"};
    if (arch == "gemma4" || arch == "gemma4_text" || has(n, "gemma-4") || has(n, "gemma4")) return {"Gemma 4", "high"};
    struct R { std::initializer_list<const char*> k; const char* f; };
    const R rules[] = {{{"llama-3", "llama3", "llama3.1", "llama3.2", "llama3.3"}, "Llama-3"}, {{"ministral", "mistral", "llama-2", "llama2"}, "Mistral"},
                       {{"hermes", "openchat", "neuralbeagle"}, "ChatML"}, {{"gemma"}, "Gemma"}, {{"zephyr"}, "Zephyr"}, {{"vicuna", "alpaca", "wizard"}, "Alpaca"}};
    for (auto& r : rules) for (auto* k : r.k) if (has(n, k)) return {r.f, "medium"};
    return {"ChatML", "low"};
}

static bool supports_thinking(const ModelMeta& m, const std::string& fmt) {
    std::string t = lower(m.tmpl), arch = lower(m.arch);
    if (fmt == "Qwen3") return true;
    if (has(t, "enable_thinking") || has(t, "preserve_thinking") || has(t, "reasoning_content") || has(t, "<think>") || has(t, "</think>")) return true;
    return in_set(arch, {"qwen3", "qwen3moe", "qwen3_5", "qwen35moe", "qwen3_6", "qwen36moe", "qwen3next"});
}
