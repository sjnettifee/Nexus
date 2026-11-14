/*
 * nexus/tools/run.cpp
 *
 * Command-line inference tool using the Nexus engine.
 * The simplest possible test of the full stack.
 *
 * Build:
 *   cd nexus && bash build.sh
 *
 * Usage:
 *   ./build/nexus_run --model /path/to/model.gguf --prompt "Hello"
 *   ./build/nexus_run --model /path/to/model.gguf --interactive
 *   ./build/nexus_run --model /path/to/model.gguf --prompt "Hi" --temp 0.8 --max-tokens 500
 */

#include "api/engine.h"
#include "api/config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--model <path>] [options]\n"
        "\n"
        "Config file: nexus.conf (./nexus.conf or ~/.nexus.conf)\n"
        "  CLI flags override config file values.\n"
        "\n"
        "Options:\n"
        "  --model <path>       Path to .gguf model file\n"
        "  --config <path>      Config file path (default: auto-detect)\n"
        "  --show-config        Print active config and exit\n"
        "  --prompt <text>      Prompt text\n"
        "  --interactive        Interactive chat mode\n"
        "  --max-tokens <n>     Max tokens to generate   (default: 512)\n"
        "  --temp <f>           Temperature               (default: 0.7)\n"
        "  --top-p <f>          Top-P sampling            (default: 0.9)\n"
        "  --top-k <n>          Top-K sampling            (default: 50)\n"
        "  --ctx <n>            Context length override\n"
        "  --threads <n>        CPU threads               (default: auto)\n"
        "  --system <text>      System prompt for chat\n"
        "  --no-progress        Suppress loading progress\n"
        "  --memory             Print memory report after load\n"
        "  --debug              Show dimensions, tokenization, first-token info\n"
        "\n"
        "Examples:\n"
        "  %s                                  # uses nexus.conf defaults\n"
        "  %s --prompt \"Explain quantum computing\"\n"
        "  %s --model ~/models/llama3-8b.gguf --interactive\n"
        "  %s --temp 0 --max-tokens 200 --prompt \"Hello\"\n",
        prog, prog, prog, prog, prog
    );
}

struct Args {
    std::string model_path;
    std::string prompt;
    std::string system;
    std::string config_path;
    bool interactive    = false;
    bool debug          = false;
    bool show_config    = false;
    int  max_tokens     = -1;   // -1 = not set by CLI, use config
    float temperature   = -1.f; // -1 = not set by CLI
    float top_p         = -1.f;
    int   top_k         = -1;
    int   threads       = -1;
    int   ctx_length    = -1;
    float repeat_penalty = -1.f;
    bool  no_progress   = false;
    bool  memory_report = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--model"       && i+1 < argc) a.model_path  = argv[++i];
        else if (arg == "--prompt"      && i+1 < argc) a.prompt      = argv[++i];
        else if (arg == "--system"      && i+1 < argc) a.system      = argv[++i];
        else if (arg == "--config"      && i+1 < argc) a.config_path = argv[++i];
        else if (arg == "--interactive")               a.interactive  = true;
        else if (arg == "--debug")                     a.debug        = true;
        else if (arg == "--show-config")               a.show_config  = true;
        else if (arg == "--no-progress")               a.no_progress  = true;
        else if (arg == "--memory")                    a.memory_report= true;
        else if (arg == "--max-tokens"  && i+1 < argc) a.max_tokens  = atoi(argv[++i]);
        else if (arg == "--temp"        && i+1 < argc) a.temperature = (float)atof(argv[++i]);
        else if (arg == "--top-p"       && i+1 < argc) a.top_p       = (float)atof(argv[++i]);
        else if (arg == "--top-k"       && i+1 < argc) a.top_k       = atoi(argv[++i]);
        else if (arg == "--threads"     && i+1 < argc) a.threads     = atoi(argv[++i]);
        else if (arg == "--ctx"         && i+1 < argc) a.ctx_length  = atoi(argv[++i]);
        else if (arg == "--repeat-penalty" && i+1 < argc) a.repeat_penalty = (float)atof(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); exit(0); }
    }
    return a;
}

// ─── Chat template helpers (mirrors Python layer) ─────────────────────────────

static std::string build_chatml(const std::string& system,
                                  const std::vector<std::pair<std::string,std::string>>& history,
                                  const std::string& user_msg)
{
    std::string p;
    if (!system.empty())
        p += "<|im_start|>system\n" + system + "<|im_end|>\n";
    for (const auto& [u, a] : history) {
        p += "<|im_start|>user\n"      + u + "<|im_end|>\n";
        p += "<|im_start|>assistant\n" + a + "<|im_end|>\n";
    }
    p += "<|im_start|>user\n" + user_msg + "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    return p;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    // ── Load config file ──────────────────────────────────────────────────────
    nexus::Config cfg;
    bool cfg_loaded = false;
    if (!args.config_path.empty())
        cfg_loaded = cfg.load(args.config_path);
    else
        cfg_loaded = cfg.load_default();

    if (args.show_config) {
        printf("\n── Nexus Configuration ──────────────────────────\n");
        cfg.print();
        printf("─────────────────────────────────────────────────\n\n");
        return 0;
    }

    // ── Apply config defaults (CLI flags override) ────────────────────────────
    // Model path: CLI > config > nothing (error)
    if (args.model_path.empty())
        args.model_path = cfg.get_str("model.path");

    // System prompt: CLI > config
    if (args.system.empty())
        args.system = cfg.get_str("model.system_prompt");

    // Numeric: use config value only if CLI didn't set it (sentinel = -1)
    if (args.max_tokens < 0)
        args.max_tokens = cfg.get_int("generation.max_tokens", 512);
    if (args.temperature < 0)
        args.temperature = cfg.get_float("generation.temperature", 0.7f);
    if (args.top_p < 0)
        args.top_p = cfg.get_float("generation.top_p", 0.9f);
    if (args.top_k < 0)
        args.top_k = cfg.get_int("generation.top_k", 50);
    if (args.threads < 0)
        args.threads = cfg.get_int("system.threads", 0);
    if (args.ctx_length < 0)
        args.ctx_length = cfg.get_int("model.context_length", 0);
    if (args.repeat_penalty < 0)
        args.repeat_penalty = cfg.get_float("generation.repeat_penalty", 1.1f);

    if (!cfg.get_bool("system.progress", true))
        args.no_progress = true;

    // ── Validate ──────────────────────────────────────────────────────────────
    if (args.model_path.empty()) {
        fprintf(stderr, "Error: no model specified. Use --model or set model.path in nexus.conf\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!args.interactive && args.prompt.empty()) {
        // Default to interactive if model is set but no prompt given
        args.interactive = true;
    }

    // Auto-detect thread count
    if (args.threads <= 0) {
        args.threads = 1;
        // Try to detect CPU cores
        FILE* f = fopen("/proc/cpuinfo", "r");
        if (f) {
            int cores = 0;
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "processor", 9) == 0) ++cores;
            }
            fclose(f);
            args.threads = std::max(1, cores - 1);
        }
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════╗\n");
    printf("  ║   NEXUS  —  Open Inference Engine   ║\n");
    printf("  ╚══════════════════════════════════════╝\n\n");
    if (cfg_loaded)
        printf("  Config: %s\n\n", cfg.loaded_path().c_str());
    else
        printf("  Config: (none — using defaults)\n\n");

    // ── Load engine ──────────────────────────────────────────────────────────
    nexus::Engine engine;

    if (!args.no_progress) {
        printf("Loading: %s\n", args.model_path.c_str());
    }

    auto load_cb = [&](const nexus::LoadProgress& p) {
        if (args.no_progress) return;
        int pct = (int)(p.fraction * 100);
        int bars = pct / 5;
        printf("\r  [");
        for (int i = 0; i < 20; ++i) printf(i < bars ? "█" : "░");
        printf("] %3d%%  %-40s", pct, p.stage.substr(0, 40).c_str());
        fflush(stdout);
        if (p.fraction >= 1.0f) printf("\n");
    };

    if (!engine.load(args.model_path, args.threads, load_cb)) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }

    auto info = engine.model_info();
    printf("\nModel: %s  |  %u layers  |  %u ctx\n",
           info.name.empty() ? args.model_path.c_str() : info.name.c_str(),
           info.n_layers, info.context_length);
    printf("Using %d CPU threads\n\n", args.threads);

    if (args.debug || args.memory_report) {
        printf("── Model Dimensions ─────────────────────────────\n");
        printf("  hidden_dim   : %u\n",  info.hidden_dim);
        printf("  n_heads      : %u\n",  info.n_heads);
        printf("  n_kv_heads   : %u\n",  info.n_kv_heads);
        printf("  head_dim     : %u\n",  info.head_dim);
        printf("  ffn_dim      : %u\n",  info.ffn_dim);
        printf("  vocab_size   : %u\n",  info.vocab_size);
        printf("  rope_base    : %.1f\n", info.rope_base);
        printf("─────────────────────────────────────────────────\n\n");
    }

    if (args.memory_report) {
        engine.print_memory_report();
    }

    // ── Generation config ─────────────────────────────────────────────────────
    nexus::GenerateConfig gen_cfg;
    gen_cfg.max_tokens     = args.max_tokens;
    gen_cfg.temperature    = args.temperature;
    gen_cfg.top_p          = args.top_p;
    gen_cfg.top_k          = args.top_k;
    gen_cfg.repeat_penalty = args.repeat_penalty;
    gen_cfg.stop_strings = {"</s>", "[/INST]", "<|im_end|>",
                             "<|endoftext|>", "<|eot_id|>",
                             "\nHuman:", "\nUser:"};

    // Detect arch for template
    bool use_chatml = (info.arch == "qwen2" || info.arch == "phi3" ||
                       info.arch == "phi"   || info.arch == "gemma");
    (void)use_chatml;

    // ── Single prompt mode ────────────────────────────────────────────────────
    if (!args.interactive) {
        std::string prompt = args.prompt;

        if (args.debug) {
            // Show what the tokenizer produces
            auto toks = engine.tokenize(prompt, true);
            printf("── Tokenization ─────────────────────────────────\n");
            printf("  Input: \"%s\"\n", prompt.c_str());
            printf("  Tokens (%zu): ", toks.size());
            for (auto id : toks) printf("[%d='%s'] ", id, engine.detokenize(id).c_str());
            printf("\n─────────────────────────────────────────────────\n\n");
        }

        printf("Prompt: %s\n\n", prompt.c_str());
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        int token_count = 0;
        auto result = engine.generate_stream(
            prompt,
            [&](const std::string& tok, int32_t id) -> bool {
                printf("%s", tok.c_str());
                fflush(stdout);
                ++token_count;

                if (args.debug && token_count == 1) {
                    printf("\n\n── First token: id=%d '%s'\n\n", id, tok.c_str());
                }
                return true;
            },
            gen_cfg
        );

        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  %d tokens  •  %.1f tok/s  •  %.1fs\n\n",
               result.n_tokens, result.tokens_per_sec, result.elapsed_sec);
        return 0;
    }

    // ── Interactive mode ──────────────────────────────────────────────────────
    printf("Interactive mode. Commands: /reset, /memory, /quit\n");
    printf("─────────────────────────────────────────────────────\n\n");

    std::vector<std::pair<std::string,std::string>> history;
    std::string system_prompt = args.system;

    while (true) {
        printf("You: ");
        fflush(stdout);

        std::string user_input;
        if (!std::getline(std::cin, user_input)) break;

        if (user_input.empty()) continue;

        if (user_input == "/quit" || user_input == "/exit") break;

        if (user_input == "/reset") {
            history.clear();
            engine.reset_context();
            printf("  [Context cleared]\n\n");
            continue;
        }

        if (user_input == "/memory") {
            engine.print_memory_report();
            continue;
        }

        if (user_input.substr(0, 8) == "/system ") {
            system_prompt = user_input.substr(8);
            printf("  [System prompt set]\n\n");
            continue;
        }

        // Build prompt with history
        std::string prompt = build_chatml(system_prompt, history, user_input);

        printf("\nAssistant: ");
        fflush(stdout);

        std::string response;
        auto result = engine.generate_stream(
            prompt,
            [&](const std::string& tok, int32_t /*id*/) -> bool {
                printf("%s", tok.c_str());
                fflush(stdout);
                response += tok;
                return true;
            },
            gen_cfg
        );

        printf("\n  [%.1f tok/s]\n\n", result.tokens_per_sec);

        // Add to history
        history.push_back({user_input, response});

        // Trim history if getting too long (keep last 10 exchanges)
        while (history.size() > 10) {
            history.erase(history.begin());
        }
    }

    printf("\nGoodbye.\n");
    return 0;
}
