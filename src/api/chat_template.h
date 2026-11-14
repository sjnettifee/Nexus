/*
 * nexus/src/api/chat_template.h
 *
 * Auto-detects the correct chat format from GGUF metadata and formats
 * conversation history into the token string each model expects.
 *
 * Most instruct models store their template in tokenizer.chat_template
 * as a Jinja2 string. We don't implement full Jinja2 — instead we detect
 * the template family by fingerprinting the template string and model name,
 * then apply the correct hardcoded format.
 *
 * Supported formats:
 *   ChatML      — Qwen, Hermes, OpenChat, NeuralBeagle, many HF finetunes
 *   Mistral     — Mistral 7B/8x7B instruct
 *   Llama-3     — Meta Llama-3 / Llama-3.1
 *   DeepSeek    — DeepSeek-V2/V3/R1 (also DeepSeek-Coder)
 *   Phi-3/4     — Microsoft Phi
 *   Gemma       — Google Gemma 2
 *   Zephyr      — HuggingFace Zephyr
 *   Alpaca      — generic instruction-following fallback
 */

#pragma once
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>

namespace nexus {

enum class ChatFormat {
    CHATML,          // <|im_start|>role\ncontent<|im_end|>
    QWEN3,           // Qwen3/Qwen3.6/Qwen3-Next ChatML + thinking controls
    MISTRAL,         // [INST] content [/INST]  (v1/v2 — adds BOS <s>)
    MISTRAL_V3,      // [INST] content [/INST]  (v3 Tekken — no BOS, [SYSTEM_PROMPT])
    LLAMA3,          // <|begin_of_text|>...<|start_header_id|>role<|end_header_id|>
    DEEPSEEK,        // <|User|>content<|Assistant|>  (V2/V3/R1 chat)
    DEEPSEEK_R1,     // <|User|>content<|Assistant|>  (R1 with <think>)
    DEEPSEEK_CODER,  // ### Instruction:\ncontent\n### Response:  (Coder 1.x/2.x)
    PHI3,            // <|user|>\ncontent<|end|>\n<|assistant|>
    PHI4,            // <|user|>\ncontent<|im_end|>\n<|assistant|>  (Phi-4)
    GEMMA,           // <start_of_turn>user\ncontent<end_of_turn>   (Gemma 1/2/3)
    GEMMA4,          // <|turn>user\ncontent<turn|>  (Gemma 4 — different tokens!)
    ZEPHYR,          // <|system|>content</s><|user|>content</s><|assistant|>
    ALPACA,          // ### Instruction:\ncontent\n\n### Response:
    GRANITE,         // <|start_of_role|>user<|end_of_role|>...<|end_of_text|>
    OLMOE,           // <|endoftext|><|user|>\ncontent\n<|assistant|>
    CUSTOM,          // User-supplied role wrappers from launcher
    RAW,             // No template — just pass text through
};

struct ChatMessage {
    std::string role;     // "system", "user", "assistant"
    std::string content;
};

class ChatTemplate {
public:
    ChatFormat format = ChatFormat::CHATML;
    std::string model_name;

    struct CustomTemplateParts {
        std::string pre_prompt;
        std::string pre_system;
        std::string end_system;
        std::string pre_user;
        std::string end_user;
        std::string pre_model;
        std::string end_model;
    };

    CustomTemplateParts custom_template;

    void set_custom_template(const CustomTemplateParts& parts) {
        custom_template = parts;
        format = ChatFormat::CUSTOM;
    }

    // Detect format from GGUF metadata
    void detect(const std::string& template_str, const std::string& arch,
                const std::string& name)
    {
        model_name = name;
        std::string t = template_str;
        std::string n = name;
        std::string a = arch;
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        std::transform(n.begin(), n.end(), n.begin(), ::tolower);
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);

        // ── ARCHITECTURE is the most reliable signal ──────────────────────────
        // If GGUF has a chat_template string, that overrides everything.
        // If it doesn't, we fall back to arch + name heuristics.

        // Qwen3 / Qwen3.6 / Qwen3-Next needs a separate format from generic ChatML
        // because it owns <think> handling / thinking preservation. Keep Qwen2.x
        // and generic ChatML finetunes on plain CHATML.
        const bool is_qwen3_name =
            n.find("qwen3") != std::string::npos ||
            n.find("qwen-3") != std::string::npos ||
            n.find("qwen 3") != std::string::npos ||
            n.find("qwen3.6") != std::string::npos ||
            n.find("qwen3-6") != std::string::npos ||
            n.find("qwen3_6") != std::string::npos ||
            n.find("qwen3-next") != std::string::npos ||
            n.find("qwen3_next") != std::string::npos ||
            n.find("qwen-next") != std::string::npos;

        if (is_qwen3_name) {
            format = ChatFormat::QWEN3;
            return;
        }

        // DeepSeek-Coder (1.x / 2.x instruct): uses ### Instruction / ### Response
        // Detect by name BEFORE checking for distill or other deepseek variants
        if (n.find("deepseek") != std::string::npos &&
            (n.find("coder") != std::string::npos ||
             n.find("code") != std::string::npos)) {
            // Coder V2 uses ChatML; Coder 1.x / 1.5 use the Instruction format
            // Check the template string first
            if (t.find("<|im_start|>") != std::string::npos) {
                if (is_qwen3_name ||
                    t.find("enable_thinking") != std::string::npos ||
                    t.find("preserve_thinking") != std::string::npos)
                    { format = ChatFormat::QWEN3; return; }
                { format = ChatFormat::CHATML; return; }
            }
            if (t.find("### instruction") != std::string::npos ||
                t.find("###instruction") != std::string::npos)
                { format = ChatFormat::DEEPSEEK_CODER; return; }
            // No template? Use arch as tiebreaker
            if (arch == "qwen2" || arch == "qwen2_5" || arch == "qwen2moe")
                { format = ChatFormat::CHATML; return; }  // Coder V2
            // Default for coder 1.x models
            { format = ChatFormat::DEEPSEEK_CODER; return; }
        }

        // DeepSeek distill models: despite "deepseek" in the name they use the
        // base model's chat format. Distill-Qwen → ChatML, Distill-Llama → Llama3.
        // Check BEFORE the generic deepseek name match.
        if (n.find("deepseek") != std::string::npos) {
            if (n.find("distill") != std::string::npos) {
                if (n.find("qwen") != std::string::npos ||
                    arch == "qwen2" || arch == "qwen2_5")
                    { format = ChatFormat::CHATML; return; }
                if (n.find("llama") != std::string::npos ||
                    arch == "llama")
                    { format = ChatFormat::LLAMA3; return; }
            }
            // Non-distill DeepSeek: use template string as primary signal
            if (t.find("<|user|>") != std::string::npos) {
                if (t.find("think") != std::string::npos ||
                    n.find("r1") != std::string::npos)
                    { format = ChatFormat::DEEPSEEK_R1; return; }
                { format = ChatFormat::DEEPSEEK; return; }
            }
            // DeepSeek with ChatML template (e.g. DeepSeek-V3 via some loaders)
            if (t.find("<|im_start|>") != std::string::npos)
                { format = ChatFormat::CHATML; return; }
            // DeepSeek name but unknown template → use arch
            if (arch == "qwen2" || arch == "qwen2_5")
                { format = ChatFormat::CHATML; return; }
            if (arch == "llama")
                { format = ChatFormat::LLAMA3; return; }
            // Last resort for deepseek
            format = ChatFormat::DEEPSEEK; return;
        }
        
        // OLMoE must be detected before the generic <|user|>/<|assistant|>
        // Phi rule, because OLMoE uses those role tokens too.
        if (a == "olmoe" || n.find("olmoe") != std::string::npos) {
            format = ChatFormat::OLMOE;
            return;
        }
        
        // ── Template string fingerprinting ────────────────────────────────────
        // This is the most reliable path — read what the model says it needs.
        if (!t.empty()) {
            if (t.find("<|start_header_id|>") != std::string::npos ||
                t.find("begin_of_text") != std::string::npos)
                { format = ChatFormat::LLAMA3; return; }

            if (t.find("<|im_start|>") != std::string::npos) {
                if (is_qwen3_name ||
                    t.find("enable_thinking") != std::string::npos ||
                    t.find("preserve_thinking") != std::string::npos)
                    { format = ChatFormat::QWEN3; return; }
                { format = ChatFormat::CHATML; return; }
            }

            // Gemma 4 uses a completely different token set than Gemma 2/3:
            //   bos      = <bos>            (token id 2)
            //   role     = <|turn>user / <|turn>model / <|turn>system / <turn|>
            //   thought  = <|channel>thought\n<channel|>   (empty for non-thinking)
            //   tool     = <|tool>, <|tool_call>, <|tool_response>
            //   media    = <|image|>, <|audio|>, <|video|>
            // Check for Gemma 4 markers BEFORE the Gemma 2/3 detection below,
            // since those would never appear in a real Gemma 4 template.
            if (t.find("<|turn>") != std::string::npos ||
                t.find("<turn|>") != std::string::npos ||
                t.find("<|channel>thought") != std::string::npos)
                { format = ChatFormat::GEMMA4; return; }

            if (t.find("<start_of_turn>") != std::string::npos)
                { format = ChatFormat::GEMMA; return; }

            // Mistral V3 Tekken (Ministral-3B/8B, Mistral-Nemo, Mistral-Small-24B)
            if (t.find("[system_prompt]") != std::string::npos ||
                t.find("[inst]") != std::string::npos) {
                if (t.find("[system_prompt]") != std::string::npos)
                    { format = ChatFormat::MISTRAL_V3; return; }
                { format = ChatFormat::MISTRAL; return; }
            }
            
            // OLMoE instruct: GPT-2 endoftext + <|user|>/<|assistant|>.
            // This must stay before Phi detection.
            if (t.find("<|endoftext|>") != std::string::npos &&
                t.find("<|user|>")      != std::string::npos &&
                t.find("<|assistant|>") != std::string::npos) {
                format = ChatFormat::OLMOE;
                return;
            }
                   
            // Zephyr / TinyLlama-chat: <|user|> ... </s>.  Phi-3 also uses <|user|> but ends turns with <|end|>.
            if (t.find("<|user|>") != std::string::npos && t.find("</s>") != std::string::npos &&
                t.find("<|end|>") == std::string::npos && t.find("<|im_end|>") == std::string::npos)
                { format = ChatFormat::ZEPHYR; return; }

            // Phi-3 / Phi-4: both use <|user|> but differ on end token
            if (t.find("<|user|>") != std::string::npos) {
                if (t.find("<|im_end|>") != std::string::npos)
                    { format = ChatFormat::PHI4; return; }
                { format = ChatFormat::PHI3; return; }
            }

            if (t.find("<|system|>") != std::string::npos)
                { format = ChatFormat::ZEPHYR; return; }

            // Command-R / Cohere: uses <|START_OF_TURN_TOKEN|>
            if (t.find("start_of_turn_token") != std::string::npos ||
                t.find("<|start_of_turn|>") != std::string::npos)
                { format = ChatFormat::CHATML; return; }  // close enough for now

            // Falcon / Falcon2: uses >>QUESTION<< style or User: / Assistant:
            if (t.find(">>question<<") != std::string::npos ||
                t.find("user:") != std::string::npos)
                { format = ChatFormat::ALPACA; return; }

            // SmolLM / SmolLM2: uses ChatML
            // StarCoder2-instruct: uses ChatML
            // Already caught by <|im_start|> above
        }

        // ── Architecture field — reliable for families that never changed format ─
        if (a == "olmoe")
            { format = ChatFormat::OLMOE; return; }

        if (a == "qwen3" || a == "qwen3_6" || a == "qwen3.6" ||
            a == "qwen3_next" || a == "qwen3-next" || a == "qwen_next")
            { format = ChatFormat::QWEN3; return; }

        if (a == "qwen2" || a == "qwen2_5" || a == "qwen2moe")
            { format = ChatFormat::CHATML; return; }

        // phi3 arch string from GGUF
        if (arch == "phi3" || arch == "phi-3")
            { format = ChatFormat::PHI3; return; }

        // phi4 doesn't have a distinct arch string in GGUF yet — falls to name check

        // gemma4 arch — uses different tokens than gemma 1/2/3
        if (arch == "gemma4" || arch == "gemma4_text" || arch == "gemma-4")
            { format = ChatFormat::GEMMA4; return; }

        // gemma / gemma2 / gemma3 arch strings
        if (arch == "gemma" || arch == "gemma2" || arch == "gemma3")
            { format = ChatFormat::GEMMA; return; }

        // starcoder2, falcon, cohere — these all use chatml or alpaca-like
        if (arch == "starcoder2" || arch == "falcon" || arch == "cohere")
            { format = ChatFormat::CHATML; return; }

        // ── Name heuristics (last resort) ─────────────────────────────────────
        // Llama-3.x family: covers all Meta releases and finetunes
        if (n.find("llama-3") != std::string::npos || n.find("llama3") != std::string::npos ||
            n.find("llama 3") != std::string::npos)
            { format = ChatFormat::LLAMA3; return; }

        // Ministral / Mistral-Nemo / Mistral-Small use V3 format
        if (n.find("ministral") != std::string::npos ||
            n.find("mistral-nemo") != std::string::npos ||
            n.find("mistral nemo") != std::string::npos ||
            n.find("mistral-small") != std::string::npos ||
            n.find("mistral small") != std::string::npos) {
            format = ChatFormat::MISTRAL_V3;
            return;
        }

        if (n.find("mistral") != std::string::npos ||
            n.find("llama-2") != std::string::npos ||
            n.find("llama2") != std::string::npos ||
            n.find("llama 2") != std::string::npos) {
            format = ChatFormat::MISTRAL;
            return;
        }
        
        // Granite 3.x instruct uses <|start_of_role|>/<|end_of_role|>, not ChatML.
        if (t.find("<|start_of_role|>") != std::string::npos ||
            arch == "granitemoe" ||
            n.find("granite") != std::string::npos) {
            format = ChatFormat::GRANITE;
    return;
}

        // ChatML family — lots of finetunes use this
        if (n.find("qwen") != std::string::npos || n.find("hermes") != std::string::npos ||
            n.find("openchat") != std::string::npos || n.find("neuralbeagle") != std::string::npos ||
            n.find("neural-chat") != std::string::npos || n.find("ninja") != std::string::npos ||
            n.find("smollm") != std::string::npos || n.find("smol-lm") != std::string::npos ||
            n.find("starcoder") != std::string::npos || n.find("command-r") != std::string::npos ||
            n.find("orca") != std::string::npos || n.find("dolphin") != std::string::npos ||
            n.find("nous") != std::string::npos || n.find("solar") != std::string::npos)
            { format = ChatFormat::CHATML; return; }

        // Phi family — check phi4 first since it contains "phi"
        if (n.find("phi") != std::string::npos) {
            // phi-4 / phi4 models use PHI4 format
            if (n.find("phi-4") != std::string::npos || n.find("phi4") != std::string::npos)
                { format = ChatFormat::PHI4; return; }
            // phi-3.5 uses phi3 format
            { format = ChatFormat::PHI3; return; }
        }

        if (n.find("gemma") != std::string::npos)
            { format = ChatFormat::GEMMA; return; }
        if (n.find("zephyr") != std::string::npos)
            { format = ChatFormat::ZEPHYR; return; }
        if (n.find("falcon") != std::string::npos)
            { format = ChatFormat::CHATML; return; }

        // Meta Llama without version number — safest to assume Llama3
        if (n.find("llama") != std::string::npos)
            { format = ChatFormat::LLAMA3; return; }

        // CodeLlama: Mistral-based format (CodeLlama uses Llama-2 tokenizer+format)
        if (n.find("codellama") != std::string::npos || n.find("code-llama") != std::string::npos ||
            n.find("code llama") != std::string::npos)
            { format = ChatFormat::MISTRAL; return; }

        // OmniCoder and similar code models that use ChatML
        if (n.find("omni") != std::string::npos || n.find("coder") != std::string::npos ||
            n.find("code") != std::string::npos)
            { format = ChatFormat::CHATML; return; }

        if (arch == "llama")
            { format = ChatFormat::LLAMA3; return; }  // modern default — was CHATML

        format = ChatFormat::CHATML;  // safe default
    }

    // Format a conversation into the model's expected input string
    // Per-turn "think" control. Only some formats have a switch for it
    // (Qwen3-family and Gemma 4); for everything else the UI hides the button
    // and the value is ignored.
    //   think: -1 = leave as configured (launcher setting / model default)
    //           1 = ask the model to reason before answering, this turn only
    //           0 = ask it not to, this turn only
    bool supports_think_control() const {
        return format == ChatFormat::QWEN3 || format == ChatFormat::GEMMA4;
    }

    std::string format_prompt(const std::vector<ChatMessage>& messages, int think = -1) const {
        switch (format) {
            case ChatFormat::CHATML:        return fmt_chatml(messages);
            case ChatFormat::QWEN3:         return fmt_qwen3(messages, think);
            case ChatFormat::MISTRAL:       return fmt_mistral(messages);
            case ChatFormat::MISTRAL_V3:    return fmt_mistral_v3(messages);
            case ChatFormat::LLAMA3:        return fmt_llama3(messages);
            case ChatFormat::DEEPSEEK:      return fmt_deepseek(messages, false);
            case ChatFormat::DEEPSEEK_R1:   return fmt_deepseek(messages, true);
            case ChatFormat::DEEPSEEK_CODER:return fmt_deepseek_coder(messages);
            case ChatFormat::PHI3:          return fmt_phi3(messages);
            case ChatFormat::PHI4:          return fmt_phi4(messages);
            case ChatFormat::GEMMA:         return fmt_gemma(messages);
            case ChatFormat::GEMMA4:        return fmt_gemma4(messages, think);
            case ChatFormat::ZEPHYR:        return fmt_zephyr(messages);
            case ChatFormat::ALPACA:        return fmt_alpaca(messages);
            case ChatFormat::GRANITE:       return fmt_granite(messages);
            case ChatFormat::OLMOE:         return fmt_olmoe(messages);
            case ChatFormat::CUSTOM:        return fmt_custom(messages);
            case ChatFormat::RAW:           return fmt_raw(messages);
        }
        return fmt_chatml(messages);
    }
    
    std::string format_system_prefix(const std::vector<ChatMessage>& messages) const {
        switch (format) {
            case ChatFormat::CHATML:         return fmt_chatml_system_prefix(messages);
            case ChatFormat::QWEN3:          return fmt_qwen3_system_prefix(messages);
            case ChatFormat::MISTRAL:        return fmt_mistral_system_prefix(messages);
            case ChatFormat::MISTRAL_V3:     return fmt_mistral_v3_system_prefix(messages);
            case ChatFormat::LLAMA3:         return fmt_llama3_system_prefix(messages);
            case ChatFormat::DEEPSEEK:       return fmt_deepseek_system_prefix(messages);
            case ChatFormat::DEEPSEEK_R1:    return fmt_deepseek_system_prefix(messages);
            case ChatFormat::DEEPSEEK_CODER: return fmt_deepseek_coder_system_prefix(messages);
            case ChatFormat::PHI3:           return fmt_phi3_system_prefix(messages);
            case ChatFormat::PHI4:           return fmt_phi4_system_prefix(messages);
            case ChatFormat::GEMMA:          return fmt_gemma_system_prefix(messages);
            case ChatFormat::GEMMA4:         return fmt_gemma4_system_prefix(messages);
            case ChatFormat::ZEPHYR:         return fmt_zephyr_system_prefix(messages);
            case ChatFormat::ALPACA:         return fmt_alpaca_system_prefix(messages);
            case ChatFormat::GRANITE:        return fmt_granite_system_prefix(messages);
            case ChatFormat::OLMOE:          return fmt_olmoe_system_prefix(messages);
            case ChatFormat::CUSTOM:         return fmt_custom_system_prefix(messages);
            case ChatFormat::RAW:            return fmt_raw_system_prefix(messages);
        }
        return fmt_chatml_system_prefix(messages);
    }
    std::string format_user_turn(const std::string& user_text, int think = -1) const {
        switch (format) {
            case ChatFormat::CHATML:         return fmt_chatml_user_turn(user_text);
            case ChatFormat::QWEN3:          return fmt_qwen3_user_turn(user_text, think);
            case ChatFormat::MISTRAL:        return fmt_mistral_user_turn(user_text);
            case ChatFormat::MISTRAL_V3:     return fmt_mistral_v3_user_turn(user_text);
            case ChatFormat::LLAMA3:         return fmt_llama3_user_turn(user_text);
            case ChatFormat::DEEPSEEK:       return fmt_deepseek_user_turn(user_text);
            case ChatFormat::DEEPSEEK_R1:    return fmt_deepseek_user_turn(user_text);
            case ChatFormat::DEEPSEEK_CODER: return fmt_deepseek_coder_user_turn(user_text);
            case ChatFormat::PHI3:           return fmt_phi3_user_turn(user_text);
            case ChatFormat::PHI4:           return fmt_phi4_user_turn(user_text);
            case ChatFormat::GEMMA:          return fmt_gemma_user_turn(user_text);
            case ChatFormat::GEMMA4:         return fmt_gemma4_user_turn(user_text, think);
            case ChatFormat::ZEPHYR:         return fmt_zephyr_user_turn(user_text);
            case ChatFormat::ALPACA:         return fmt_alpaca_user_turn(user_text);
            case ChatFormat::GRANITE:        return fmt_granite_user_turn(user_text);
            case ChatFormat::OLMOE:          return fmt_olmoe_user_turn(user_text);
            case ChatFormat::CUSTOM:         return fmt_custom_user_turn(user_text);
            case ChatFormat::RAW:            return fmt_raw_user_turn(user_text);
        }
        return fmt_chatml_user_turn(user_text);
    }

    // ── Continuation marker ───────────────────────────────────────────────────
    // ~15-20 tokens injected every N turns to reset the model's turn-position
    // sense. The model's fine-tuning distribution treats long conversations as
    // anomalous past ~30-40 turns. This marker appears at the current position
    // (which the model attends to heavily) and signals "active task, continue."
    //
    // The KV cache is NEVER cleared. All prior tokens remain fully intact.
    // This adds tokens, it never removes them.
    std::string continuation_marker() const {
        static const char* msg = "Active session — continue current task.";
        switch (format) {
            case ChatFormat::CHATML:
            case ChatFormat::QWEN3:
            case ChatFormat::PHI4:
                return std::string("<|im_start|>system\n") + msg + "<|im_end|>\n";
            case ChatFormat::MISTRAL:
                return std::string("[INST] ") + msg + " [/INST] Understood. ";
            case ChatFormat::MISTRAL_V3:
                return std::string("[SYSTEM_PROMPT] ") + msg + " [/SYSTEM_PROMPT]";
            case ChatFormat::LLAMA3:
                return std::string("<|start_header_id|>system<|end_header_id|>\n\n")
                     + msg + "<|eot_id|>";
            case ChatFormat::DEEPSEEK:
            case ChatFormat::DEEPSEEK_R1:
                return std::string("<|System|>") + msg + "\n";
            case ChatFormat::DEEPSEEK_CODER:
                return std::string("### System:\n") + msg + "\n\n";
            case ChatFormat::PHI3:
                return std::string("<|system|>\n") + msg + "<|end|>\n";
            case ChatFormat::GEMMA:
                return std::string("<start_of_turn>user\n[") + msg
                     + "]<end_of_turn>\n<start_of_turn>model\nUnderstood.<end_of_turn>\n";
            case ChatFormat::GEMMA4:
                // Gemma 4 puts system into a dedicated <|turn>system block.
                return std::string("<|turn>system\n") + msg + "<turn|>\n";
            case ChatFormat::ZEPHYR:
                return std::string("<|system|>\n") + msg + "</s>\n";
            case ChatFormat::ALPACA:
                return std::string("### Note:\n") + msg + "\n\n";
            case ChatFormat::GRANITE:
                return std::string("<|start_of_role|>system<|end_of_role|>")
                     + msg + "<|end_of_text|>\n";
            case ChatFormat::OLMOE:
                return std::string("<|endoftext|><|system|>\n")
                     + msg + "\n";
            case ChatFormat::CUSTOM:
                return custom_template.pre_system + msg + custom_template.end_system;
            default:
                return std::string("\n---\n[") + msg + "]\n---\n";
        }
    }

    // Stop strings for this format — used to terminate generation cleanly
    std::vector<std::string> stop_strings() const {
        switch (format) {
            case ChatFormat::CHATML:
                return {"<|im_end|>", "<|endoftext|>"};
            case ChatFormat::QWEN3:
                return {"<|im_end|>", "<|endoftext|>"};
            case ChatFormat::MISTRAL:
                return {"</s>", "[/INST]"};
            case ChatFormat::MISTRAL_V3:
                return {"</s>", "[/INST]"};
            case ChatFormat::LLAMA3:
                return {"<|eot_id|>", "<|end_of_text|>"};
            case ChatFormat::DEEPSEEK:
            case ChatFormat::DEEPSEEK_R1:
                // ▁ is U+2581 (UTF-8: E2 96 81) — include both raw and decoded forms
                return {"<|User|>",
                        "<|end\xe2\x96\x81of\xe2\x96\x81sentence|>",
                        "<|end of sentence|>",
                        "</s>"};
            case ChatFormat::PHI3:
                return {"<|end|>", "<|endoftext|>", "<|user|>"};
            case ChatFormat::PHI4:
                // Phi-4 uses <|im_end|> — same as ChatML but with different role tokens
                return {"<|im_end|>", "<|endoftext|>", "<|user|>"};
            case ChatFormat::GEMMA:
                return {"<end_of_turn>", "<eos>"};
            case ChatFormat::GEMMA4:
                // Per Gemma 4 GGUF metadata: eos_token = <turn|>.
                return {"<turn|>", "<eos>"};
            case ChatFormat::ZEPHYR:
                return {"</s>", "<|endoftext|>"};
            case ChatFormat::ALPACA:
                return {"\n### ", "\n\n###"};
            case ChatFormat::DEEPSEEK_CODER:
                return {"\n### Instruction:", "<|EOT|>", "### Instruction:"};
            case ChatFormat::GRANITE:
                return {"<|end_of_text|>", "<|start_of_role|>user<|end_of_role|>"};
            case ChatFormat::OLMOE:
                return {"<|endoftext|>", "<|user|>", "<|system|>"};
            case ChatFormat::CUSTOM: {
                std::vector<std::string> s;
                if (!custom_template.end_model.empty())  s.push_back(custom_template.end_model);
                if (!custom_template.pre_user.empty())   s.push_back(custom_template.pre_user);
                if (!custom_template.pre_system.empty()) s.push_back(custom_template.pre_system);
                s.push_back("<|endoftext|>");
                return s;
            }
            default:
                return {"</s>", "<|endoftext|>"};
        }
    }

    // Recommended default system prompt for this format/model family
    std::string default_system_prompt() const {
        switch (format) {
            case ChatFormat::CHATML:
                return "You are a helpful assistant.";
            case ChatFormat::QWEN3:
                return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
            case ChatFormat::DEEPSEEK:
            case ChatFormat::DEEPSEEK_R1:
                return "";
            case ChatFormat::DEEPSEEK_CODER:
                // The exact system prompt from DeepSeek-Coder docs
                return "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company.";
            case ChatFormat::MISTRAL:
            case ChatFormat::MISTRAL_V3:
                return "";
            case ChatFormat::LLAMA3:
                return "You are a helpful, respectful and honest assistant.";
            case ChatFormat::PHI3:
            case ChatFormat::PHI4:
                return "You are a helpful AI assistant.";
            case ChatFormat::GEMMA:
                // Gemma inlines system into first user message — keep short
                return "You are a helpful assistant.";
            case ChatFormat::GEMMA4:
                return "You are a helpful assistant.";
            case ChatFormat::GRANITE:
                return "You are a helpful AI assistant.";
            case ChatFormat::OLMOE:
                return "You are a helpful assistant.";
            case ChatFormat::CUSTOM:
                return "";
            default:
                return "You are a helpful assistant.";
        }
    }

    // Whether to prepend BOS token before the formatted prompt.
    // Most chat formats embed their own sequence-start token in the template
    // and do NOT want a separate BOS prepended.
    // Llama-2/Mistral are the exception — they expect <s> (BOS) before [INST].
    bool should_add_bos() const {
        switch (format) {
            case ChatFormat::MISTRAL:
                return true;   // expects <s>[INST] ...
            case ChatFormat::MISTRAL_V3:
                return false;  // V3 Tekken tokenizer — BOS embedded in [INST] token
            case ChatFormat::LLAMA3:
                return false;  // template starts with <|begin_of_text|> token
            case ChatFormat::CHATML:
                return false;  // starts with <|im_start|>, no BOS needed
            case ChatFormat::QWEN3:
                return false;  // Qwen3/Qwen3.6 uses ChatML role tokens, no separate BOS
            case ChatFormat::DEEPSEEK:
            case ChatFormat::DEEPSEEK_R1:
            case ChatFormat::DEEPSEEK_CODER:
                return false;  // BOS causes confusion in DeepSeek formats
            case ChatFormat::ZEPHYR:
                return true;   // TinyLlama / Zephyr: Llama tokenizer, <s> starts every sequence
            case ChatFormat::PHI3:
            case ChatFormat::PHI4:
                return false;  // Phi template is self-contained
            case ChatFormat::GEMMA:
                return false;  // Gemma embeds <bos> in the template
            case ChatFormat::GEMMA4:
                return false;  // Gemma 4 also embeds <bos> in the template (we emit it in fmt_gemma4)
            case ChatFormat::GRANITE:
            case ChatFormat::OLMOE:
            case ChatFormat::CUSTOM:
                return false;
            default:
                return false;  // safe default for unknown formats
        }
    }

    // Whether to prepend BOS token before the formatted prompt.
    const char* name() const {
        switch (format) {
            case ChatFormat::CHATML:         return "ChatML";
            case ChatFormat::QWEN3:          return "Qwen3";
            case ChatFormat::MISTRAL:        return "Mistral [INST]";
            case ChatFormat::MISTRAL_V3:     return "Mistral V3 [INST]";
            case ChatFormat::LLAMA3:         return "Llama-3";
            case ChatFormat::DEEPSEEK:       return "DeepSeek";
            case ChatFormat::DEEPSEEK_R1:    return "DeepSeek-R1";
            case ChatFormat::DEEPSEEK_CODER: return "DeepSeek-Coder";
            case ChatFormat::PHI3:           return "Phi-3";
            case ChatFormat::PHI4:           return "Phi-4";
            case ChatFormat::GEMMA:          return "Gemma";
            case ChatFormat::GEMMA4:         return "Gemma 4";
            case ChatFormat::ZEPHYR:         return "Zephyr";
            case ChatFormat::ALPACA:         return "Alpaca";
            case ChatFormat::GRANITE:        return "Granite";
            case ChatFormat::OLMOE:          return "OLMoE";
            case ChatFormat::CUSTOM:         return "Custom";
            case ChatFormat::RAW:            return "Raw";
        }
        return "Unknown";
    }

private:

    std::string fmt_chatml_user_turn(const std::string& user_text) const {
        return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    }

    // Qwen3 generation prefix. think==1 opens a reasoning block for this turn
    // (overrides a launcher "Disable"); think==0 closes it empty; -1 follows
    // the launcher setting / model default.
    std::string qwen3_gen_suffix(int think) const {
        if (think == 1) return "<think>\n";
        if (think == 0 || qwen_thinking_disabled()) return "<think>\n\n</think>\n\n";
        return "";
    }

    std::string fmt_qwen3_user_turn(const std::string& user_text, int think) const {
        return "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n"
             + qwen3_gen_suffix(think);
    }
    static std::string fmt_mistral_user_turn(const std::string& user_text) {
        return "[INST] " + user_text + " [/INST]";
    }

    static std::string fmt_mistral_v3_user_turn(const std::string& user_text) {
        return "[INST] " + user_text + " [/INST]";
    }

    static std::string fmt_llama3_user_turn(const std::string& user_text) {
        return "<|start_header_id|>user<|end_header_id|>\n\n" +
               user_text +
               "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    static std::string fmt_deepseek_user_turn(const std::string& user_text) {
        return "<|User|>" + user_text + "<|Assistant|>";
    }

    static std::string fmt_deepseek_coder_user_turn(const std::string& user_text) {
        return "### Instruction:\n" + user_text + "\n### Response:\n";
    }

    static std::string fmt_phi3_user_turn(const std::string& user_text) {
        return "<|user|>\n" + user_text + "<|end|>\n<|assistant|>\n";
    }

    static std::string fmt_phi4_user_turn(const std::string& user_text) {
        return "<|user|>\n" + user_text + "<|im_end|>\n<|assistant|>\n";
    }

    static std::string fmt_gemma_user_turn(const std::string& user_text) {
        return "<start_of_turn>user\n" + user_text + "<end_of_turn>\n<start_of_turn>model\n";
    }

    static std::string fmt_zephyr_user_turn(const std::string& user_text) {
        return "<|user|>\n" + user_text + "</s>\n<|assistant|>\n";
    }

    static std::string fmt_alpaca_user_turn(const std::string& user_text) {
        return "### Instruction:\n" + user_text + "\n\n### Response:\n";
    }

    static std::string fmt_raw_user_turn(const std::string& user_text) {
        return user_text;
    }

    static std::string first_system(const std::vector<ChatMessage>& msgs) {
        if (!msgs.empty() && msgs[0].role == "system") return msgs[0].content;
        for (const auto& m : msgs) {
            if (m.role == "system") return m.content;
        }
        return "";
    }

    static std::string fmt_chatml_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }

    std::string fmt_qwen3_system_prefix(const std::vector<ChatMessage>& msgs) const {
        std::string sys = first_system(msgs);
        if (sys.empty()) sys = default_system_prompt();
        if (sys.empty()) return "";
        return "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }

    static std::string fmt_mistral_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "[INST] " + sys + "\n\n";
    }

    static std::string fmt_mistral_v3_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "[SYSTEM_PROMPT] " + sys + " [/SYSTEM_PROMPT]";
    }

    static std::string fmt_llama3_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
           sys + "<|eot_id|>";
    }

    static std::string fmt_deepseek_system_prefix(const std::vector<ChatMessage>& msgs) {
        return first_system(msgs);
    }

    static std::string fmt_deepseek_coder_system_prefix(const std::vector<ChatMessage>& msgs) {
        return first_system(msgs);
    }

    static std::string fmt_phi3_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|system|>\n" + sys + "<|end|>\n";
    }

    static std::string fmt_phi4_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|system|>\n" + sys + "<|im_end|>\n";
    }

    static std::string fmt_gemma_system_prefix(const std::vector<ChatMessage>& msgs) {
        return first_system(msgs);
    }

    static std::string fmt_zephyr_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|system|>\n" + sys + "</s>\n";
    }

    static std::string fmt_alpaca_system_prefix(const std::vector<ChatMessage>& msgs) {
        return first_system(msgs);
    }

    static std::string fmt_raw_system_prefix(const std::vector<ChatMessage>& msgs) {
        return first_system(msgs);
    }
    
    static bool env_true(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s == "1" || s == "true" || s == "yes" || s == "on";
    }

    static bool env_false(const char* name) {
        const char* v = std::getenv(name);
        if (!v) return false;
        std::string s(v);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s == "0" || s == "false" || s == "no" || s == "off";
    }

    bool is_qwen_family() const {
        std::string n = model_name;
        std::transform(n.begin(), n.end(), n.begin(), ::tolower);
        return n.find("qwen") != std::string::npos;
    }

    bool qwen_thinking_disabled() const {
        return env_false("NEXUS_ENABLE_THINKING") ||
               env_false("NEXUS_QWEN36_THINKING");
    }

    bool qwen_preserve_thinking() const {
        return env_true("NEXUS_PRESERVE_THINKING") ||
               env_true("NEXUS_QWEN36_PRESERVE_THINKING");
    }

    static std::string strip_qwen_think_blocks(std::string s) {
        size_t pos = 0;
        while (true) {
            size_t a = s.find("<think>", pos);
            if (a == std::string::npos) break;

            size_t b = s.find("</think>", a + 7);
            if (b == std::string::npos) break;

            b += 8; // strlen("</think>")
            while (b < s.size() && (s[b] == '\n' || s[b] == '\r')) ++b;

            s.erase(a, b - a);
            pos = a;
        }
        return s;
    }

    // ── ChatML ────────────────────────────────────────────────────────────────
    // Used by: Qwen, Hermes, OpenChat, NeuralBeagle, most modern finetunes
    std::string fmt_chatml(const std::vector<ChatMessage>& msgs) const {
        std::string out;
        const bool qwen = is_qwen_family();
        const bool preserve = qwen_preserve_thinking();

        for (const auto& m : msgs) {
            std::string content = m.content;

            // Qwen3.6 docs say preserve_thinking is optional. Default behavior should
            // not keep every old hidden reasoning block when rebuilding a prompt.
            if (qwen && m.role == "assistant" && !preserve) {
                content = strip_qwen_think_blocks(content);
            }

            out += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
        }

        out += "<|im_start|>assistant\n";

        return out;
    }
    
    // ── Qwen3 / Qwen3.6 / Qwen3-Next ────────────────────────────────────────
    // Same role tokens as ChatML, but kept separate so thinking controls and
    // thinking preservation do not alter generic ChatML models.
    std::string fmt_qwen3(const std::vector<ChatMessage>& msgs, int think) const {
        std::string out;
        const bool preserve = qwen_preserve_thinking();

        bool has_system = false;
        for (const auto& m : msgs) {
            if (m.role == "system") { has_system = true; break; }
        }

        // Qwen3/Qwen3.6 official ChatML template expects a system turn.  If the
        // request did not provide one, inject Qwen's canonical default here.
        // Keep this inside QWEN3 only so generic ChatML models are untouched.
        if (!has_system) {
            const std::string sys = default_system_prompt();
            if (!sys.empty()) {
                out += "<|im_start|>system\n" + sys + "<|im_end|>\n";
            }
        }

        for (const auto& m : msgs) {
            std::string content = m.content;
            if (m.role == "assistant" && !preserve) {
                content = strip_qwen_think_blocks(content);
            }
            out += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
        }

        out += "<|im_start|>assistant\n";

        // Qwen's thinking switch is a Qwen-specific template behavior.
        // Do not put this in generic ChatML.
        out += qwen3_gen_suffix(think);

        return out;
    }

    //---Granite----------------------------------------------------------------
    static std::string fmt_granite_user_turn(const std::string& user_text) {
        return "<|start_of_role|>user<|end_of_role|>"
             + user_text
             + "<|end_of_text|>\n"
             + "<|start_of_role|>assistant<|end_of_role|>";
    }

    static std::string fmt_granite_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|start_of_role|>system<|end_of_role|>"
             + sys
             + "<|end_of_text|>\n";
    }

    static std::string fmt_granite(const std::vector<ChatMessage>& msgs) {
        std::string out;

        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += "<|start_of_role|>system<|end_of_role|>"
                     + m.content
                     + "<|end_of_text|>\n";
            } else if (m.role == "user") {
                out += "<|start_of_role|>user<|end_of_role|>"
                     + m.content
                     + "<|end_of_text|>\n";
            } else if (m.role == "assistant") {
                out += "<|start_of_role|>assistant<|end_of_role|>"
                     + m.content
                     + "<|end_of_text|>\n";
            }
        }

        out += "<|start_of_role|>assistant<|end_of_role|>";
        return out;
    }
    
    // ── OLMoE ─────────────────────────────────────────────────────────────────
    static std::string fmt_olmoe_user_turn(const std::string& user_text) {
        return "<|endoftext|><|user|>\n"
             + user_text
             + "\n<|assistant|>\n";
    }

    static std::string fmt_olmoe_system_prefix(const std::vector<ChatMessage>& msgs) {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return "<|endoftext|><|system|>\n" + sys + "\n";
    }

    static std::string fmt_olmoe(const std::vector<ChatMessage>& msgs) {
        std::string out = "<|endoftext|>";

        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += "<|system|>\n";
                out += m.content;
                out += "\n";
            } else if (m.role == "user") {
                out += "<|user|>\n";
                out += m.content;
                out += "\n";
            } else if (m.role == "assistant") {
                out += "<|assistant|>\n";
                out += m.content;
                out += "<|endoftext|>\n";
            }
        }

        out += "<|assistant|>\n";
        return out;
    }    

    // ── Mistral [INST] ────────────────────────────────────────────────────────
    // Used by: Mistral-7B-Instruct, Mixtral-Instruct, Llama-2-Chat
    static std::string fmt_mistral(const std::vector<ChatMessage>& msgs) {
        std::string out;
        std::string system_prompt;

        // Extract system prompt
        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            system_prompt = msgs[0].content;
            start = 1;
        }

        bool first = true;
        for (size_t i = start; i < msgs.size(); ++i) {
            if (msgs[i].role == "user") {
                if (first && !system_prompt.empty())
                    out += "[INST] " + system_prompt + "\n\n" + msgs[i].content + " [/INST]";
                else
                    out += "[INST] " + msgs[i].content + " [/INST]";
                first = false;
            } else if (msgs[i].role == "assistant") {
                out += " " + msgs[i].content + " </s>";
            }
        }
        return out;
    }

    // ── Mistral V3 / Tekken (Ministral-3B/8B, Mistral-Nemo, Mistral-Small-24B) ──
    // Uses [SYSTEM_PROMPT]...[/SYSTEM_PROMPT] for system, [INST]...[/INST] for user.
    // No BOS prepended — the tokenizer handles it.
    // Multi-turn: assistant turns end with </s>, no [INST] wrapper on them.
    static std::string fmt_mistral_v3(const std::vector<ChatMessage>& msgs) {
        std::string out;
        std::string sys;
        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            sys   = msgs[0].content;
            start = 1;
        }

        bool first = true;
        for (size_t i = start; i < msgs.size(); ++i) {
            if (msgs[i].role == "user") {
                std::string content = msgs[i].content;
                if (first && !sys.empty()) {
                    // V3: system prompt wraps the first user turn
                    out += "[SYSTEM_PROMPT] " + sys + " [/SYSTEM_PROMPT]";
                }
                out += "[INST] " + content + " [/INST]";
                first = false;
            } else if (msgs[i].role == "assistant") {
                out += msgs[i].content + "</s>";
            }
        }
        return out;
    }
    // Used by: Meta Llama-3, Llama-3.1, Llama-3.2
    static std::string fmt_llama3(const std::vector<ChatMessage>& msgs) {
        std::string out = "<|begin_of_text|>";
        for (const auto& m : msgs) {
            out += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n";
            out += m.content + "<|eot_id|>";
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }

    // ── DeepSeek ──────────────────────────────────────────────────────────────
    // Used by: DeepSeek-V2, DeepSeek-V3, DeepSeek-Coder
    // R1 variant: model outputs <think>...</think> before the answer
    static std::string fmt_deepseek(const std::vector<ChatMessage>& msgs, bool /*is_r1*/) {
        std::string out;
        // DeepSeek doesn't support system prompts in the same way
        // Inline the system prompt into the first user message if present
        std::string system_prefix;
        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            system_prefix = msgs[0].content + "\n\n";
            start = 1;
        }

        bool first_user = true;
        for (size_t i = start; i < msgs.size(); ++i) {
            if (msgs[i].role == "user") {
                std::string content = (first_user ? system_prefix : "") + msgs[i].content;
                out += "<|User|>" + content;
                first_user = false;
            } else if (msgs[i].role == "assistant") {
                out += "<|Assistant|>" + msgs[i].content + "<|end▁of▁sentence|>";
            }
        }
        out += "<|Assistant|>";
        return out;
    }

    // ── DeepSeek-Coder ────────────────────────────────────────────────────────
    // Used by: DeepSeek-Coder 1.x, 1.5B/6.7B/33B instruct
    // The exact format from the official DeepSeek-Coder docs.
    // System prompt is prepended once. Each turn is ### Instruction / ### Response.
    static std::string fmt_deepseek_coder(const std::vector<ChatMessage>& msgs) {
        std::string out;
        std::string sys;
        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            sys   = msgs[0].content;
            start = 1;
        }
        // System prompt goes at the very top, once
        if (!sys.empty()) out += sys + "\n";

        for (size_t i = start; i < msgs.size(); ++i) {
            if (msgs[i].role == "user") {
                out += "### Instruction:\n" + msgs[i].content + "\n";
            } else if (msgs[i].role == "assistant") {
                out += "### Response:\n" + msgs[i].content + "\n<|EOT|>\n";
            }
        }
        out += "### Response:\n";
        return out;
    }

    // ── Phi-3 / Phi-4 ────────────────────────────────────────────────────────
    // Used by: Microsoft Phi-3-mini, Phi-3.5, Phi-4
    static std::string fmt_phi3(const std::vector<ChatMessage>& msgs) {
        std::string out;
        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += "<|system|>\n" + m.content + "<|end|>\n";
            } else if (m.role == "user") {
                out += "<|user|>\n" + m.content + "<|end|>\n";
            } else if (m.role == "assistant") {
                out += "<|assistant|>\n" + m.content + "<|end|>\n";
            }
        }
        out += "<|assistant|>\n";
        return out;
    }

    // ── Phi-4 ────────────────────────────────────────────────────────────────
    // Phi-4 switched from <|end|> to <|im_end|> but kept <|user|>/<|assistant|>
    static std::string fmt_phi4(const std::vector<ChatMessage>& msgs) {
        std::string out;
        for (const auto& m : msgs) {
            if      (m.role == "system")    out += "<|system|>\n"    + m.content + "<|im_end|>\n";
            else if (m.role == "user")      out += "<|user|>\n"      + m.content + "<|im_end|>\n";
            else if (m.role == "assistant") out += "<|assistant|>\n" + m.content + "<|im_end|>\n";
        }
        out += "<|assistant|>\n";
        return out;
    }

    // ── Gemma ─────────────────────────────────────────────────────────────────
    // Used by: Google Gemma-2
    static std::string fmt_gemma(const std::vector<ChatMessage>& msgs) {
        std::string out;
        // Gemma doesn't have a system role — prepend to first user message
        std::string system_prefix;
        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            system_prefix = msgs[0].content + "\n\n";
            start = 1;
        }

        bool first_user = true;
        for (size_t i = start; i < msgs.size(); ++i) {
            if (msgs[i].role == "user") {
                std::string content = (first_user ? system_prefix : "") + msgs[i].content;
                out += "<start_of_turn>user\n" + content + "<end_of_turn>\n";
                first_user = false;
            } else if (msgs[i].role == "assistant") {
                out += "<start_of_turn>model\n" + msgs[i].content + "<end_of_turn>\n";
            }
        }
        out += "<start_of_turn>model\n";
        return out;
    }

    // ── Gemma 4 ───────────────────────────────────────────────────────────────
    //
    // Gemma 4 uses a COMPLETELY DIFFERENT token set than Gemma 1/2/3:
    //   bos                    = <bos>                       (token id 2)
    //   role-open              = <|turn>role\n               where role ∈ {system, user, model}
    //   role-close             = <turn|>\n
    //   thought-channel-open   = <|channel>thought\n         (empty body for non-thinking)
    //   thought-channel-close  = <channel|>
    //   eos                    = <turn|>
    //
    // Reference: the chat_template stored in unsloth/gemma-4-31B-it-GGUF reads:
    //   {{- bos_token -}}
    //   {%- if ... system... -%}
    //       {{- '<|turn>system\n' -}}{{- ... -}}{{- '<turn|>\n' -}}
    //   {%- endif -%}
    //   {%- for message in messages -%}
    //       {{- '<|turn>' + role + '\n' -}}{{- content -}}{{- '<turn|>\n' -}}
    //   {%- endfor -%}
    //   {%- if add_generation_prompt -%}
    //       {{- '<|turn>model\n' -}}
    //       {%- if not enable_thinking -%}
    //           {{- '<|channel>thought\n<channel|>' -}}
    //       {%- endif -%}
    //   {%- endif -%}
    //
    // The literal '<bos>' string in our output gets tokenized to the actual
    // BOS token id by the GGUF tokenizer, same way Gemma 2/3 path relied on
    // it. should_add_bos() returns false for this format so we don't double-
    // prepend.
    static std::string fmt_gemma4(const std::vector<ChatMessage>& msgs, int think) {
        std::string out;
        out += "<bos>";

        size_t start = 0;
        if (!msgs.empty() && msgs[0].role == "system") {
            out += "<|turn>system\n" + msgs[0].content + "<turn|>\n";
            start = 1;
        }

        for (size_t i = start; i < msgs.size(); ++i) {
            const std::string& role =
                (msgs[i].role == "assistant") ? "model" : msgs[i].role;
            out += "<|turn>" + role + "\n" + msgs[i].content + "<turn|>\n";
        }

        // Generation prompt: non-thinking = empty thought channel; thinking
        // (think==1 for this turn) = leave the channel for the model to open.
        out += "<|turn>model\n";
        if (think != 1) out += "<|channel>thought\n<channel|>";
        return out;
    }

    static std::string fmt_gemma4_user_turn(const std::string& user_text, int think) {
        // Single-turn helper used by some calling code paths.
        // Does not emit <bos> because incremental turns assume context is
        // already established.
        return "<|turn>user\n" + user_text
             + "<turn|>\n<|turn>model\n"
             + (think == 1 ? std::string() : std::string("<|channel>thought\n<channel|>"));
    }

    static std::string fmt_gemma4_system_prefix(const std::vector<ChatMessage>& msgs) {
        // Returns the system content extracted (plain text) — the actual
        // <|turn>system block is emitted inside fmt_gemma4 itself.
        return first_system(msgs);
    }


    // ── Zephyr ────────────────────────────────────────────────────────────────
    static std::string fmt_zephyr(const std::vector<ChatMessage>& msgs) {
        std::string out;
        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += "<|system|>\n" + m.content + "</s>\n";
            } else if (m.role == "user") {
                out += "<|user|>\n" + m.content + "</s>\n";
            } else if (m.role == "assistant") {
                out += "<|assistant|>\n" + m.content + "</s>\n";
            }
        }
        out += "<|assistant|>\n";
        return out;
    }

    // ── Alpaca ────────────────────────────────────────────────────────────────
    static std::string fmt_alpaca(const std::vector<ChatMessage>& msgs) {
        std::string out;
        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += m.content + "\n\n";
            } else if (m.role == "user") {
                out += "### Instruction:\n" + m.content + "\n\n";
            } else if (m.role == "assistant") {
                out += "### Response:\n" + m.content + "\n\n";
            }
        }
        out += "### Response:\n";
        return out;
    }

    // ── Raw passthrough ───────────────────────────────────────────────────────
    static std::string fmt_raw(const std::vector<ChatMessage>& msgs) {
        std::string out;
        for (const auto& m : msgs) {
            if (m.role != "system") out += m.content;
        }
        return out;
    }

    // ── Custom launcher-defined template ─────────────────────────────────────
    std::string fmt_custom(const std::vector<ChatMessage>& msgs) const {
        std::string out = custom_template.pre_prompt;

        for (const auto& m : msgs) {
            if (m.role == "system") {
                out += custom_template.pre_system;
                out += m.content;
                out += custom_template.end_system;
            } else if (m.role == "user") {
                out += custom_template.pre_user;
                out += m.content;
                out += custom_template.end_user;
            } else if (m.role == "assistant") {
                out += custom_template.pre_model;
                out += m.content;
                out += custom_template.end_model;
            }
        }

        out += custom_template.pre_model;
        return out;
    }

    std::string fmt_custom_user_turn(const std::string& user_text) const {
        return custom_template.end_model +
               custom_template.pre_user +
               user_text +
               custom_template.end_user +
               custom_template.pre_model;
    }

    std::string fmt_custom_system_prefix(const std::vector<ChatMessage>& msgs) const {
        std::string sys = first_system(msgs);
        if (sys.empty()) return "";
        return custom_template.pre_prompt +
               custom_template.pre_system +
               sys +
               custom_template.end_system;
    }
};

} // namespace nexus
