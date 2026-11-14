/*
 * nexus/src/api/engine.cpp
 *
 * Engine implementation.
 *
 * This is the one .cpp file that includes all the internal headers
 * that contain their implementations (transformer, attention, kvcache,
 * matmul, quant). They're header-only by design so they inline well
 * and we avoid cross-file linking complexity at this stage
 *
 * As the codebase grows, they can be split into separate .cpp files.
 */

#include "engine.h"
// chat_template.h already included via engine.h

// Pull in all internal implementations
#include "../gguf/quant.h"
#include "../gguf/parser.h"
#include "../cpu/matmul.h"
#include "../model/kvcache.h"
#include "../model/attention.h"
#include "../model/tokenizer.h"
#include "../model/transformer.h"

#include <cstdio>
#include <cstring>
#include <cctype>
#include <chrono>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <utility>
#include <cstdlib>

static std::string nx_visible_token_text(const std::string& ts) {
    std::string clean;
    clean.reserve(ts.size());
    for (unsigned char ch : ts) {
        if (ch == '\n') clean += "\\n";
        else if (ch == '\r') clean += "\\r";
        else if (ch == '\t') clean += "\\t";
        else if (ch < 0x20) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", ch);
            clean += buf;
        } else {
            clean.push_back((char)ch);
        }
    }
    return clean;
}

static bool nx_env_truthy_engine(const char* name) {
    const char* v = std::getenv(name);
    return v && *v && v[0] != '0' && v[0] != 'n' && v[0] != 'N' && v[0] != 'f' && v[0] != 'F';
}
#ifndef _WIN32
#  include <unistd.h>
#  include <sys/syscall.h>
#endif

namespace nexus {

// --- Pimpl (hide internal types from public header) ---------------------------

struct Engine::Impl {
    int turbo_gpu_device = -1;   // -1 = auto (prefer discrete)
    int turbo_vram_budget_mb = 0;
    bool turbo_prefill_enabled = false;   // set via Engine::set_turbo_prefill
    bool prefill_token_by_token = false;  // debug/old path only
    int n_gpu_layers_committed = 0;
    model::Transformer   transformer;
    model::Tokenizer     tokenizer;
    ChatTemplate         chat_tmpl;
    std::string          model_path;
    size_t               file_size    = 0;
    int                  n_threads    = 1;
    int                  ctx_pos      = 0;
    int                  streamed_layer = -1;  // layer currently resident via swap_layer(); -1 = none yet
    std::unique_ptr<nexus::vision::VisionEncoder> vision_enc;
    std::unique_ptr<nexus::audio::AudioEncoder>   audio_enc;
    std::vector<int32_t> session_tokens;  // full token history for persist/extend
    std::string          pinned_prefix;   // always-present system context (never evicted)
    int                  pinned_tokens = 0; // how many tokens are pinned
    int                  ctx_limit     = 0;    // conversation length limit, 0 = model maximum
    bool                 auto_trim_on  = true; // at the limit: drop oldest turns (true) or stop (false)

    // -- Section markers -------------------------------------------------------
    // Named positions in the token stream. Set via context_mark() before/after
    // injecting each logical section (system prompt, KB block, conversation turns).
    // Used by context_trim() to evict specific ranges without reprocessing.
    struct Section {
        std::string name;
        int start_token;   // inclusive
        int end_token;     // exclusive
        bool pinned;       // true = never evict this section
    };
    std::vector<Section> sections;
};

// --- Engine lifecycle ---------------------------------------------------------

Engine::Engine()  : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

bool Engine::load(const std::string& model_path, int n_threads,
                  std::function<void(const LoadProgress&)> progress_cb)
{
    // Always start from a clean engine-level state. The transformer's
    // own load() re-inits kvcache_ but doesn't know about ctx_pos etc.
    impl_->ctx_pos       = 0;
    impl_->session_tokens.clear();
    impl_->pinned_tokens = 0;
    impl_->sections.clear();
    impl_->pinned_prefix.clear();
#ifdef _WIN32
    // The DLL has its own CRT instance with its own stderr buffer - separate
    // from the exe. Force line-buffering here so every \n flushes to the pipe
    // immediately. Must be done inside the DLL, not in the exe.
    { static char _sb[256]; setvbuf(stderr, _sb, _IOLBF, sizeof(_sb)); }
    { static char _sb[256]; setvbuf(stdout, _sb, _IOLBF, sizeof(_sb)); }
#endif
    impl_->model_path = model_path;
    impl_->n_threads  = n_threads;

    // Get file size for progress
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[engine] Cannot open: %s\n", model_path.c_str());
        return false;
    }
    fseek(f, 0, SEEK_END);
    impl_->file_size = (size_t)ftell(f);
    fclose(f);

    // Translate our progress CB into transformer's callback
    std::function<void(const std::string&, float)> t_cb;
    if (progress_cb) {
        t_cb = [&](const std::string& stage, float frac) {
            LoadProgress p;
            p.stage         = stage;
            p.fraction      = frac;
            p.bytes_loaded  = (size_t)(frac * impl_->file_size);
            p.bytes_total   = impl_->file_size;
            progress_cb(p);
        };
    }

    // Load model weights via transformer
    fprintf(stderr, "[engine] Loading model: %s\n", model_path.c_str()); fflush(stderr);
    fprintf(stderr, "[engine] DIAG engine.cpp compiled %s %s\n", __DATE__, __TIME__); fflush(stderr);
    if (!impl_->transformer.load(model_path, n_threads, t_cb)) {
        return false;
    }

    // Native multimodal support: homemade .nxm files can carry model.visual.*
    // tensors inside the same model. IMPORTANT: do not re-open/re-mmap the .nxm
    // here. The Transformer has already loaded all tensors once; vision borrows
    // those TensorDesc::data pointers so RAM does not double.
    impl_->vision_enc = std::make_unique<nexus::vision::VisionEncoder>();
    if (const auto* loaded_tensors = impl_->transformer.loaded_tensor_map()) {
        if (impl_->vision_enc->bind_native_qwen3next_from_loaded_tensors(*loaded_tensors, t_cb)) {
            fprintf(stderr, "[engine] Native/embedded vision encoder ready from already-loaded model tensors\n");
        } else {
            impl_->vision_enc.reset();
            fprintf(stderr, "[engine] No embedded/native vision tensors found in loaded model\n");
        }
    } else {
        impl_->vision_enc.reset();
        fprintf(stderr, "[engine] No loaded tensor map available for embedded vision\n");
    }

    // Load tokenizer from the same GGUF (parser is inside transformer)
    // We need to re-open the parser for tokenizer - or expose it
    // For now, the transformer's internal tokenizer handles basic encoding
    // The standalone Tokenizer below gives us better BPE
    {
        gguf::Parser parser;
        if (parser.open(model_path)) {
            impl_->tokenizer.load(parser);
            impl_->tokenizer.print_info();
            // Auto-detect chat template
            std::string tmpl = parser.get_str("tokenizer.chat_template", "");
            const auto& m    = impl_->transformer.meta();
            impl_->chat_tmpl.detect(tmpl, m.arch, m.name);
            fprintf(stderr, "[chat] Template: %s\n", impl_->chat_tmpl.name());
        }
    }

    return true;
}

void Engine::unload() {
    impl_ = std::make_unique<Impl>();
}

bool Engine::is_loaded() const {
    return impl_->transformer.is_loaded();
}

// --- GPU ----------------------------------------------------------------------

int Engine::set_gpu_layers(int n) {
    if (!is_loaded()) return 0;
    return impl_->transformer.set_gpu_layers(n);
}

void Engine::set_card_fill_keep_mb(int mb) { impl_->transformer.set_card_fill_keep_mb(mb); }
int Engine::set_gpu2_layers(int n, int device_index, int first_layer) {
    if (!is_loaded() || n == 0) return 0;
    return impl_->transformer.set_gpu2_layers(n, device_index, first_layer);
}
void Engine::set_mtp_card_reserve(bool on) { impl_->transformer.set_mtp_card_reserve(on); }

int Engine::add_gpu_layers(int n, int device_index, int first_layer) {
    if (!is_loaded() || n == 0) return 0;
    return impl_->transformer.add_gpu_layers(n, device_index, first_layer);
}

std::string Engine::gpux_device_name(int i) const {
    return impl_->transformer.gpux_device_name(i);
}

void Engine::set_gpu_device(int index) {
    nexus::gpu::gpu().set_device_index(index);
}

std::vector<std::string> Engine::list_gpu_devices() {
#ifdef NEXUS_VULKAN
    return nexus::gpu::VulkanBackend::list_devices();
#else
    return {};
#endif
}

std::string Engine::gpu_device_name() const {
    auto info = impl_->transformer.gpu_info();
    return info.device_name;
}

std::string Engine::gpu2_device_name() const {
    return impl_->transformer.gpu2_device_name();
}

bool Engine::ensure_gpu_ready(int device_index) {
    auto& g = nexus::gpu::gpu();
    if (device_index >= 0) g.set_device_index(device_index);
    if (g.ready()) return true;
    bool ok = g.init(/*prefer_apu_zero_copy=*/true);
    if (ok) {
        fprintf(stderr, "[engine] GPU brought up for turbo streaming: %s\n",
                g.info().device_name.c_str());
    } else {
        fprintf(stderr, "[engine] GPU init failed — turbo streaming will fall back to CPU\n");
    }
    return ok;
}

bool Engine::gpu_zero_copy() const {
    // True only for APU/iGPU unified memory - weights genuinely live in shared RAM,
    // no PCIe copy needed. Discrete GPUs always copy even if host_import extension
    // is detected, so we don't report that as zero-copy.
    auto info = impl_->transformer.gpu_info();
    return info.unified_memory;
}

// --- Info ---------------------------------------------------------------------

ModelInfo Engine::model_info() const {
    const auto& m = impl_->transformer.meta();
    ModelInfo info{};
    info.name           = m.name;
    info.arch           = m.arch;
    info.n_layers       = m.n_layers;
    info.n_heads        = m.n_heads;
    info.n_kv_heads     = m.n_kv_heads;
    info.head_dim       = m.head_dim;
    info.hidden_dim     = m.hidden_dim;
    info.ffn_dim        = m.ffn_dim;
    info.vocab_size     = m.vocab_size;
    info.context_length = m.context_length;
    info.rope_base      = m.rope_freq_base;
    info.chat_format    = impl_->chat_tmpl.name();
    info.file_size_bytes= impl_->file_size;
    return info;
}

std::string Engine::format_chat(const std::vector<nexus::ChatMessage>& messages, int think) const {
    std::vector<nexus::ChatMessage> msgs;
    msgs.reserve(messages.size());
    for (const auto& m : messages)
        msgs.push_back({m.role, m.content});
    return impl_->chat_tmpl.format_prompt(msgs, think);
}

std::string Engine::format_system_prefix(const std::vector<nexus::ChatMessage>& messages) const {
    return impl_->chat_tmpl.format_system_prefix(messages);
}

std::string Engine::format_user_turn(const std::string& user_text, int think) const {
    return impl_->chat_tmpl.format_user_turn(user_text, think);
}


bool Engine::supports_think_control() const {
    return impl_->chat_tmpl.supports_think_control();
}

std::vector<std::string> Engine::stop_strings() const {
    return impl_->chat_tmpl.stop_strings();
}

std::string Engine::default_system_prompt() const {
    return impl_->chat_tmpl.default_system_prompt();
}

bool Engine::should_add_bos() const {
    return impl_->chat_tmpl.should_add_bos();
}

void Engine::override_chat_format(const std::string& fmt) {
    // Map human-readable name to enum - same table as ServerState::tmpl_map
    static const std::unordered_map<std::string, ChatFormat> map = {
        {"ChatML",          ChatFormat::CHATML},
        {"Qwen3",           ChatFormat::QWEN3},
        {"Mistral",         ChatFormat::MISTRAL},
        {"Mistral V3",      ChatFormat::MISTRAL_V3},
        {"Mistral [INST]",  ChatFormat::MISTRAL},
        {"Mistral V3 [INST]", ChatFormat::MISTRAL_V3},
        {"Llama-3",         ChatFormat::LLAMA3},
        {"DeepSeek",        ChatFormat::DEEPSEEK},
        {"DeepSeek-R1",     ChatFormat::DEEPSEEK_R1},
        {"DeepSeek-Coder",  ChatFormat::DEEPSEEK_CODER},
        {"Phi-3",           ChatFormat::PHI3},
        {"Phi-4",           ChatFormat::PHI4},
        {"Gemma",           ChatFormat::GEMMA},
        {"Gemma 4",         ChatFormat::GEMMA4},
        {"Gemma4",          ChatFormat::GEMMA4},
        {"Zephyr",          ChatFormat::ZEPHYR},
        {"Alpaca",          ChatFormat::ALPACA},
        {"Granite",         ChatFormat::GRANITE},
        {"OLMoE",           ChatFormat::OLMOE},
        {"Custom",          ChatFormat::CUSTOM},
        {"Raw",             ChatFormat::RAW},
    };
    auto it = map.find(fmt);
    if (it != map.end()) {
        impl_->chat_tmpl.format = it->second;
        fprintf(stderr, "[engine] Chat format overridden to: %s\n", impl_->chat_tmpl.name());
    } else {
        fprintf(stderr, "[engine] Unknown chat format: '%s' - keeping auto-detected: %s\n",
                fmt.c_str(), impl_->chat_tmpl.name());
    }
}

void Engine::set_custom_chat_template(const nexus::ChatTemplate::CustomTemplateParts& parts) {
    impl_->chat_tmpl.set_custom_template(parts);
    fprintf(stderr, "[engine] Chat format overridden to: %s\n", impl_->chat_tmpl.name());
}

void Engine::print_memory_report() const {
    impl_->transformer.print_memory_report();
}

size_t Engine::memory_used_bytes() const {
    // Rough: file size is a good proxy for weight memory
    return impl_->file_size;
}

// --- Tokenize -----------------------------------------------------------------

std::vector<int32_t> Engine::tokenize(const std::string& text, bool add_bos) const {
    if (impl_->tokenizer.vocab_size() > 0) {
        return impl_->tokenizer.encode(text, add_bos);
    }
    // Fallback to transformer's built-in tokenizer
    return impl_->transformer.tokenize(text, add_bos);
}

std::string Engine::detokenize(const std::vector<int32_t>& tokens) const {
    if (impl_->tokenizer.vocab_size() > 0) {
        return impl_->tokenizer.decode(tokens);
    }
    return impl_->transformer.detokenize(tokens);
}

std::string Engine::detokenize(int32_t token_id) const {
    if (impl_->tokenizer.vocab_size() > 0) {
        return impl_->tokenizer.decode_token(token_id);
    }
    return impl_->transformer.detokenize(token_id);
}

int32_t Engine::bos_token() const {
    return impl_->tokenizer.vocab_size() > 0
        ? impl_->tokenizer.bos_id() : 1;
}

int32_t Engine::eos_token() const {
    return impl_->tokenizer.vocab_size() > 0
        ? impl_->tokenizer.eos_id() : 2;
}

size_t Engine::vocab_size() const {
    return impl_->transformer.vocab_size();
}

// --- Low-level forward --------------------------------------------------------

int32_t Engine::forward(const std::vector<int32_t>& tokens, int pos) {
    return impl_->transformer.forward(tokens, pos);
}

int32_t Engine::sample(float temperature, float top_p, int top_k) {
    model::SamplerConfig cfg;
    cfg.temperature = temperature;
    cfg.top_p       = top_p;
    cfg.top_k       = top_k;
    return impl_->transformer.sample(cfg);
}

const float* Engine::hidden_data() const {
    return impl_->transformer.hidden_data();
}

void Engine::set_hidden_data(const float* h, size_t n_floats) {
    impl_->transformer.set_hidden_data(h, n_floats);
}

void Engine::forward_layer_range(const std::vector<int32_t>& tokens, int pos,
                                  int layer_start, int layer_end) {
    impl_->transformer.forward_layer_range(tokens, pos, layer_start, layer_end);
}

bool Engine::prefill_layer_range(const std::vector<int32_t>& tokens, int start_pos, int n,
                                 int layer_start, int layer_end, const float* x_in, float* x_out,
                                 bool keep_prefill_scratch) {
    const bool turbo = impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream();
    return impl_->transformer.prefill_layer_range(tokens, start_pos, n, layer_start, layer_end, x_in, x_out, turbo, keep_prefill_scratch);
}
void Engine::set_range_prefill_quiet(bool q) { impl_->transformer.pr_quiet_ = q; }
int Engine::range_prefill_layers_done() const { return impl_->transformer.pr_layers_done_.load(); }
double Engine::prefill_fraction() const {
    const auto& t = impl_->transformer;
    const int layers = t.pf_layers_.load(), l = t.pf_layer_.load(), tot = t.pf_tok_total_.load();
    if (layers <= 0 || l < 0) return -1.0;
    const double in_layer = tot > 0 ? (double)t.pf_tok_done_.load() / tot : 0.0;
    return std::min(1.0, ((double)l + in_layer) / layers);
}
void Engine::prefill_progress_begin(bool external_report) {
    auto& t = impl_->transformer;
    if (external_report) { t.pf_layer_.store(-1); t.pf_layers_.store(0); t.pf_tok_done_.store(0); t.pf_tok_total_.store(0); }
    t.pf_external_.store(external_report);
}
bool Engine::range_prefill_ok() const { return impl_->transformer.range_prefill_ok(); }

// ---------------------------------------------------------------------------
// Engine::load_from_memory
//
// Loads a GGUF from an in-memory blob with zero disk I/O on all platforms.
// Delegates to Transformer::load_from_memory -> Parser::open_from_memory.
// The blob is borrowed (not copied) for the duration of the load; the
// transformer's internal structures are initialized from it.
// ---------------------------------------------------------------------------
bool Engine::load_from_memory(const void* data, size_t size, int n_threads,
                               std::function<void(const LoadProgress&)> progress_cb)
{
    impl_->n_threads = n_threads;
    std::function<void(const std::string&, float)> t_cb;
    if (progress_cb) {
        t_cb = [&](const std::string& stage, float frac) {
            LoadProgress p; p.stage = stage; p.fraction = frac;
            p.bytes_loaded = (size_t)(frac * size);
            p.bytes_total  = size;
            progress_cb(p);
        };
    }
    return impl_->transformer.load_from_memory(data, size, n_threads, t_cb);
}

bool Engine::load_streamed_shard(const void* gguf_header, size_t gguf_header_size,
                                  int n_threads, std::function<bool(void*, size_t)> read_fn,
                                  std::function<void(const LoadProgress&)> progress_cb)
{
    impl_->n_threads = n_threads;
    std::function<void(const std::string&, float)> t_cb;
    if (progress_cb) {
        t_cb = [&](const std::string& stage, float frac) {
            LoadProgress p;
            p.stage = stage;
            p.fraction = frac;
            p.bytes_loaded = (size_t)(frac * gguf_header_size);
            p.bytes_total  = gguf_header_size;
            progress_cb(p);
        };
    }
    return impl_->transformer.load_streamed_shard(
        gguf_header, gguf_header_size, n_threads, std::move(read_fn), t_cb);
}

void Engine::init_model_path(const std::string& model_path) {
    impl_->model_path = model_path;
    FILE* f = fopen(model_path.c_str(), "rb");
    if (f) { fseek(f, 0, SEEK_END); impl_->file_size = (size_t)ftell(f); fclose(f); }
    fprintf(stderr, "[engine] model path set: %s\n", model_path.c_str());
    fflush(stderr);
}

// Distributed shards preserve original blk.L names. Older versions remapped
// worker source layers to blk.0.., which can break the exact layer-order chain.
static std::string nx_remap_name(const std::string& n, int /*layer_start*/) {
    return n;
}
static bool nx_is_block_count(const std::string& k) {
    return k.size() > 11 && k.substr(k.size()-11) == "block_count";
}

// KV metadata is copied verbatim from the source GGUF so the worker's
// transformer.load() sees a complete, self-consistent model file.
// ---------------------------------------------------------------------------

static void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back((uint8_t)(v >>  0));
    buf.push_back((uint8_t)(v >>  8));
    buf.push_back((uint8_t)(v >> 16));
    buf.push_back((uint8_t)(v >> 24));
}
static void write_u64_le(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)(v >> (i*8)));
}
static void write_i32_le(std::vector<uint8_t>& buf, int32_t v) {
    write_u32_le(buf, (uint32_t)v);
}
static void write_str_gguf(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64_le(buf, (uint64_t)s.size());
    buf.insert(buf.end(), s.begin(), s.end());
}

// Write a single GGUFValue back in GGUF wire format.
// Returns the GGUFType byte that precedes this value in a KV pair.
static uint32_t write_gguf_value(std::vector<uint8_t>& buf, const gguf::GGUFValue& val) {
    return std::visit([&](auto&& v) -> uint32_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, uint8_t>)  { buf.push_back(v); return 0; }
        if constexpr (std::is_same_v<T, int8_t>)   { buf.push_back((uint8_t)v); return 1; }
        if constexpr (std::is_same_v<T, uint16_t>) { buf.push_back(v&0xFF); buf.push_back(v>>8); return 2; }
        if constexpr (std::is_same_v<T, int16_t>)  { uint16_t u=(uint16_t)v; buf.push_back(u&0xFF); buf.push_back(u>>8); return 3; }
        if constexpr (std::is_same_v<T, uint32_t>) { write_u32_le(buf, v); return 4; }
        if constexpr (std::is_same_v<T, int32_t>)  { write_i32_le(buf, v); return 5; }
        if constexpr (std::is_same_v<T, float>)    { uint32_t u; memcpy(&u, &v, 4); write_u32_le(buf, u); return 6; }
        if constexpr (std::is_same_v<T, bool>)     { buf.push_back(v ? 1 : 0); return 7; }
        if constexpr (std::is_same_v<T, std::string>) { write_str_gguf(buf, v); return 8; }
        if constexpr (std::is_same_v<T, uint64_t>) { write_u64_le(buf, v); return 10; }
        if constexpr (std::is_same_v<T, int64_t>)  { write_u64_le(buf, (uint64_t)v); return 11; }
        if constexpr (std::is_same_v<T, double>)   { uint64_t u; memcpy(&u, &v, 8); write_u64_le(buf, u); return 12; }
        if constexpr (std::is_same_v<T, std::shared_ptr<gguf::GGUFArray>>) {
            // Array: type(u32) + count(u64) + elements
            if (!v) return 9;
            write_u32_le(buf, (uint32_t)v->elem_type);
            write_u64_le(buf, (uint64_t)v->values.size());
            for (auto& elem : v->values) write_gguf_value(buf, elem);
            return 9;
        }
        return 0;
    }, val);
}

// ---------------------------------------------------------------------------
// Engine::load_range
//
// Loads ONLY layers [layer_start, layer_end) from the source GGUF into owned RAM.
// Direct path: original GGUF -> filtered/remapped tensor table -> MemoryManager.
// No mini-GGUF, no memfd, no temp file, no full in-RAM shard staging.
// ---------------------------------------------------------------------------
bool Engine::load_range(const std::string& model_path, int layer_start, int layer_end,
                        int n_threads,
                        std::function<void(const LoadProgress&)> progress_cb)
{
    impl_->model_path = model_path;
    impl_->n_threads  = n_threads;
    { FILE* f = fopen(model_path.c_str(), "rb");
      if (!f) { fprintf(stderr, "[engine] Cannot open: %s\n", model_path.c_str()); return false; }
      fseek(f, 0, SEEK_END); impl_->file_size = (size_t)ftell(f); fclose(f); }

    std::function<void(const std::string&, float)> t_cb;
    if (progress_cb) {
        t_cb = [&](const std::string& stage, float frac) {
            LoadProgress p; p.stage = stage; p.fraction = frac;
            p.bytes_loaded = (size_t)(frac * impl_->file_size);
            p.bytes_total  = impl_->file_size;
            progress_cb(p);
        };
    }

    fprintf(stderr,
            "[engine] load_range: DIRECT loading source layers %d-%d from original GGUF (no mini-GGUF, no memfd/temp, no shard staging)...\n",
            layer_start, layer_end - 1);
    fflush(stderr);

    bool loaded = impl_->transformer.load_range_direct(
        model_path, layer_start, layer_end, n_threads, t_cb);

    if (!loaded) {
        fprintf(stderr, "[engine] load_range: direct transformer shard load failed\n");
        return false;
    }

    // Engine-level tokenizer/chat template stays sourced from the original full GGUF.
    { gguf::Parser parser;
      if (parser.open(model_path)) {
          impl_->tokenizer.load(parser);
          impl_->tokenizer.print_info();
          std::string tmpl = parser.get_str("tokenizer.chat_template", "");
          const auto& m = impl_->transformer.meta();
          impl_->chat_tmpl.detect(tmpl, m.arch, m.name);
          fprintf(stderr, "[chat] Template: %s\n", impl_->chat_tmpl.name());
      } }

    fprintf(stderr, "[engine] load_range: READY -- source layers %d-%d loaded directly as local shard\n",
            layer_start, layer_end - 1);
    fflush(stderr);
    return true;
}

bool Engine::open_for_layer_streaming(const std::string& model_path, int n_threads,
                                       int probe_layer_start, int probe_layer_end) {
    impl_->model_path = model_path;
    impl_->n_threads  = n_threads;
    impl_->streamed_layer = -1;
    { FILE* f = fopen(model_path.c_str(), "rb");
      if (!f) { fprintf(stderr, "[engine] Cannot open: %s\n", model_path.c_str()); return false; }
      fseek(f, 0, SEEK_END); impl_->file_size = (size_t)ftell(f); fclose(f); }

    if (!impl_->transformer.open_for_layer_streaming(model_path, n_threads, probe_layer_start, probe_layer_end)) {
        fprintf(stderr, "[engine] open_for_layer_streaming: transformer setup failed\n");
        return false;
    }

    // Tokenizer/chat template, sourced from the full GGUF, exactly as load_range() does -- loaded
    // once here, never again on a swap_layer() call.
    { gguf::Parser parser;
      if (parser.open(model_path)) {
          impl_->tokenizer.load(parser);
          impl_->tokenizer.print_info();
          std::string tmpl = parser.get_str("tokenizer.chat_template", "");
          const auto& m = impl_->transformer.meta();
          impl_->chat_tmpl.detect(tmpl, m.arch, m.name);
          fprintf(stderr, "[chat] Template: %s\n", impl_->chat_tmpl.name());
      } }

    fprintf(stderr, "[engine] open_for_layer_streaming: READY -- model opened once, no layers resident yet\n");
    return true;
}

bool Engine::swap_layer(int new_layer) {
    const int prev = impl_->streamed_layer;
    if (!impl_->transformer.swap_layer_direct(new_layer, prev)) return false;
    impl_->streamed_layer = new_layer;
    return true;
}

// *** v29: thin wrappers -- see the declaration comment in engine.h. Deliberately NOT
// routed through impl_->streamed_layer (that bookkeeping is for the one-layer-at-a-time
// ghost use case above); these two just forward to the Transformer methods that leave
// active_layer_start_/active_layer_end_ untouched.
bool Engine::park_setup(int park_layers, int below_mb, int keep_mb, int prefill_free_mb, int min_tokens) {
    return impl_->transformer.park_setup(park_layers, below_mb, keep_mb, prefill_free_mb, min_tokens);
}
bool Engine::park_active() const { return impl_->transformer.park_active(); }
bool Engine::park_moving(int& done, int& total) const { return impl_->transformer.park_moving(done, total); }
bool Engine::free_own_layer(int layer) { return impl_->transformer.free_resident_layer(layer); }
bool Engine::reload_own_layer(int layer) { return impl_->transformer.reload_resident_layer(layer); }

// ---------------------------------------------------------------------------
// KV cache buffer transfer -- used by distributed inference to save/restore
// the worker's KV cache on the primary machine.
//
// save_kv_to_buffer: serializes this engine's KV cache to a byte buffer.
//                    Pure RAM -- uses KVCache::save_to_buffer() directly.
// load_kv_from_buffer: restores KV cache from bytes received over the network.
//                      Pure RAM -- uses KVCache::load_from_buffer() directly.
// ---------------------------------------------------------------------------
bool Engine::save_kv_to_buffer(std::vector<uint8_t>& out) const {
    out = impl_->transformer.save_kv_buffer();
    return !out.empty();
}

bool Engine::load_kv_from_buffer(const uint8_t* data, size_t size) {
    return impl_->transformer.load_kv_buffer(data, size);
}

// Per-layer siblings, for hybrid prefill (see dist_session.h) -- stream one layer at a time
// instead of the whole cache at once.
bool Engine::save_kv_layer_to_buffer(int layer, std::vector<uint8_t>& out) const {
    out = impl_->transformer.save_kv_layer_buffer(layer);
    return true;   // empty IS a valid answer here (this layer has nothing to send) -- not a failure
}

bool Engine::load_kv_layer_from_buffer(int layer, const uint8_t* data, size_t size) {
    return impl_->transformer.load_kv_layer_buffer(layer, data, size);
}

void Engine::free_kv_layer(int layer) {
    impl_->transformer.free_kv_layer(layer);
}


// ---------------------------------------------------------------------------
// Engine::write_layer_subset_gguf
//
// Called by DistSession on the PRIMARY machine to extract the layer weights
// that the worker needs, formatted as a valid GGUF v3 file in memory.
//
// The worker receives this blob, writes it to a temp file, and calls
// engine.load() on it -- reusing the existing GGUF loader without any
// changes to transformer.h.
//
// Which tensors are included:
//   - "token_embd.weight"          always (worker needs embedding table)
//   - "blk.{L}.*"                  for L in [layer_start, layer_end)
//   - "output_norm.weight"          always (worker runs final norm + lm_head)
//   - "output.weight"              always
//

bool Engine::write_layer_subset_gguf(std::vector<uint8_t>& out,
                                      int layer_start, int layer_end) const
{
    if (impl_->model_path.empty()) {
        fprintf(stderr, "[engine] write_layer_subset_gguf: no model loaded\n");
        return false;
    }

    // Re-open the GGUF to get raw tensor data pointers.
    // The transformer's mmap is already live so we can open a second Parser
    // for metadata/tensor iteration without double-loading weights.
    gguf::Parser parser;
    if (!parser.open(impl_->model_path)) {
        fprintf(stderr, "[engine] write_layer_subset_gguf: cannot reopen %s\n",
                impl_->model_path.c_str());
        return false;
    }

    const auto& all_tensors = parser.tensors();
    const auto& kv          = parser.kv();
    // Build list of tensor names to include:
    //   token_embd.weight, output_norm.weight, output.weight (global)
    //   blk.{L}.* for L in [layer_start, layer_end)
    auto want_tensor = [&](const std::string& name) -> bool {
        if (name == "token_embd.weight") return true;
        if (name == "output_norm.weight") return true;
        if (name == "output.weight") return true;
        // Match "blk.L." prefix
        if (name.size() > 4 && name.substr(0, 4) == "blk.") {
            size_t dot2 = name.find('.', 4);
            if (dot2 == std::string::npos) return false;
            int L = atoi(name.c_str() + 4);
            return L >= layer_start && L < layer_end;
        }
        return false;
    };

    // Collect matching tensors (stable order: sorted by name)
    std::vector<const gguf::TensorDesc*> selected;
    selected.reserve(32);
    for (auto& [name, td] : all_tensors) {
        if (want_tensor(name)) selected.push_back(&td);
    }
    std::sort(selected.begin(), selected.end(),
              [](auto* a, auto* b){ return a->name < b->name; });

    if (selected.empty()) {
        fprintf(stderr, "[engine] write_layer_subset_gguf: no tensors matched\n");
        return false;
    }

    // -----------------------------------------------------------------------
    // Build the GGUF blob:
    //   [GGUF header: magic(4) version(4) n_tensors(8) n_kv(8)]
    //   [KV pairs verbatim]
    //   [Tensor info: name, n_dims, shape[], type, offset for each tensor]
    //   [alignment padding to 32 bytes]
    //   [tensor data blocks, each padded to 32-byte alignment]
    // -----------------------------------------------------------------------

    const uint32_t ALIGN = 32;
    out.clear();

    // Header
    write_u32_le(out, gguf::GGUF_MAGIC);
    write_u32_le(out, gguf::GGUF_VERSION);
    write_u64_le(out, (uint64_t)selected.size());   // n_tensors
    write_u64_le(out, (uint64_t)kv.size());          // n_kv

    // KV pairs -- copy verbatim from source
    // We need a deterministic order so collect into a sorted vector first
    std::vector<std::pair<std::string, gguf::GGUFValue>> kv_sorted(kv.begin(), kv.end());
    std::sort(kv_sorted.begin(), kv_sorted.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    for (auto& [key, val] : kv_sorted) {
        write_str_gguf(out, key);
        // We need to write the type byte then the value.
        // Capture current size, write placeholder type, write value, then fix up type.
        size_t type_pos = out.size();
        write_u32_le(out, 0);  // placeholder
        uint32_t actual_type = write_gguf_value(out, val);
        // Fix up the type
        out[type_pos+0] = (uint8_t)(actual_type >>  0);
        out[type_pos+1] = (uint8_t)(actual_type >>  8);
        out[type_pos+2] = (uint8_t)(actual_type >> 16);
        out[type_pos+3] = (uint8_t)(actual_type >> 24);
    }

    // Tensor info section
    // Compute data offsets (from start of data section, not from file start)
    std::vector<uint64_t> data_offsets(selected.size());
    uint64_t running_offset = 0;
    for (size_t i = 0; i < selected.size(); ++i) {
        // align to ALIGN
        running_offset = (running_offset + ALIGN - 1) / ALIGN * ALIGN;
        data_offsets[i] = running_offset;
        running_offset += selected[i]->data_size;
    }

    for (size_t i = 0; i < selected.size(); ++i) {
        const auto* td = selected[i];
        write_str_gguf(out, td->name);
        write_u32_le(out, (uint32_t)td->shape.size());  // n_dims
        for (uint64_t dim : td->shape) write_u64_le(out, dim);
        write_i32_le(out, (int32_t)td->type);           // ggml type
        write_u64_le(out, data_offsets[i]);             // offset in data section
    }

    // Pad metadata section to ALIGN boundary
    while (out.size() % ALIGN) out.push_back(0);

    // data_ptr() returns mmap_base + data_offset_ -- the start of the tensor data
    // section in the file. Valid for the parser's lifetime without load_weights().
    const uint8_t* data_section = parser.data_ptr();
    if (!data_section) {
        fprintf(stderr, "[engine] write_layer_subset_gguf: parser mmap not available\n");
        return false;
    }

    // Data section: write each tensor's raw bytes from the mmap
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto* td = selected[i];
        while (out.size() % ALIGN) out.push_back(0);
        if (td->data_size > 0) {
            // Read from mmap at td->offset from the data section start.
            // This works without load_weights() -- GGUF data is always
            // accessible via mmap after open().
            const uint8_t* tensor_bytes = data_section + td->offset;
            out.insert(out.end(), tensor_bytes, tensor_bytes + td->data_size);
        }
    }

    fprintf(stderr, "[engine] write_layer_subset_gguf: %zu tensors, %zu bytes, layers %d-%d\n",
            selected.size(), out.size(), layer_start, layer_end - 1);
    return true;
}

// Streaming version for sending to worker: never allocates a ~10GB blob.
// Calls write_fn(data, size) for the uint64 total_size prefix, then header,
// then each tensor's raw bytes directly from the source GGUF mmap.
bool Engine::send_layer_subset_gguf(int layer_start, int layer_end,
    std::function<bool(const void*, size_t)> write_fn) const
{
    if (impl_->model_path.empty()) return false;

    gguf::Parser parser;
    if (!parser.open(impl_->model_path)) return false;

    const auto& all_tensors = parser.tensors();
    const auto& kv          = parser.kv();

    // *** v27 fix: this used to only recognize the old llama.cpp-style names (a fixed
    // list of 6 exact global names plus anything starting with "blk."). Qwen3Next/
    // Qwen3.6's native GGUF export uses "model.language_model.layers.N.xxx" for every
    // layer and "model.language_model.embed_tokens.weight" / "model.language_model.
    // norm.weight" for two of the three globals -- none of which matched, so this
    // function sent a near-empty shard (lm_head.weight only, by exact-name coincidence)
    // for any model using that naming scheme. nx_layer_of_tensor() already understands
    // every naming scheme this codebase supports; the global list below is the same one
    // open_for_layer_streaming() already uses elsewhere (transformer.h's is_global).
    auto want = [&](const std::string& name) -> bool {
        static const char* const globals[] = {
            "token_embd.weight", "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight", "tok_embeddings.weight",
            "output_norm.weight", "model.norm.weight", "model.language_model.norm.weight", "norm.weight",
            "output_norm.bias", "model.norm.bias", "model.language_model.norm.bias", "norm.bias",
            "output.weight", "lm_head.weight", "model.lm_head.weight", "model.language_model.lm_head.weight",
            "output.bias", "lm_head.bias", "model.lm_head.bias", "model.language_model.lm_head.bias",
        };
        for (const char* g : globals) if (name == g) return true;
        const int L = ::nexus::gguf::nx_layer_of_tensor(name);
        return L >= 0 && L >= layer_start && L < layer_end;
    };

    std::vector<const gguf::TensorDesc*> sel;
    sel.reserve(64);
    for (auto& [n, td] : all_tensors) if (want(n)) sel.push_back(&td);

    // Critical: receiver reads raw tensor bytes in this exact order after sorting
    // its parsed, remapped tensor names. Sort by the remapped name, not source name.
    std::sort(sel.begin(), sel.end(), [&](auto* a, auto* b){
        return nx_remap_name(a->name, layer_start) < nx_remap_name(b->name, layer_start);
    });
    if (sel.empty()) return false;

    // Build a header-only GGUF: metadata + tensor descriptors, but NO tensor data.
    // The receiver parses this small header, then reads tensor bytes directly into
    // final owned allocations one tensor at a time. This is not a fake full GGUF blob.
    const uint32_t ALIGN = 32;
    std::vector<uint8_t> hdr;
    hdr.reserve(4 * 1024 * 1024);

    write_u32_le(hdr, gguf::GGUF_MAGIC);
    write_u32_le(hdr, gguf::GGUF_VERSION);
    write_u64_le(hdr, (uint64_t)sel.size());
    write_u64_le(hdr, (uint64_t)kv.size());

    std::vector<std::pair<std::string,gguf::GGUFValue>> kv_s(kv.begin(), kv.end());
    std::sort(kv_s.begin(), kv_s.end(), [](auto& a, auto& b){ return a.first < b.first; });
    for (auto& [key, val] : kv_s) {
        write_str_gguf(hdr, key);
        size_t tp = hdr.size(); write_u32_le(hdr, 0);
        // Preserve full-model block_count. The receiver indexes only the
        // selected active source-layer interval, but meta_.n_layers must remain
        // the true full layer count so layer_end==N only on the final worker.
        gguf::GGUFValue wval = val;
        uint32_t at = write_gguf_value(hdr, wval);
        hdr[tp+0]=(uint8_t)(at>>0); hdr[tp+1]=(uint8_t)(at>>8);
        hdr[tp+2]=(uint8_t)(at>>16); hdr[tp+3]=(uint8_t)(at>>24);
    }

    // Offsets are present because GGUF tensor descriptors require them, but the
    // receiver does not read tensor bytes through Parser::data_ptr(). It receives
    // bytes over the socket directly into final tensor allocations.
    uint64_t running_offset = 0;
    for (size_t i = 0; i < sel.size(); ++i) {
        const auto* td = sel[i];
        running_offset = (running_offset + ALIGN - 1) / ALIGN * ALIGN;
        write_str_gguf(hdr, nx_remap_name(td->name, layer_start));
        write_u32_le(hdr, (uint32_t)td->shape.size());
        for (uint64_t dim : td->shape) write_u64_le(hdr, dim);
        write_i32_le(hdr, (int32_t)td->type);
        write_u64_le(hdr, running_offset);
        running_offset += td->data_size;
    }
    while (hdr.size() % ALIGN) hdr.push_back(0);

    uint64_t total_tensor_bytes = 0;
    for (auto* td : sel) total_tensor_bytes += (uint64_t)td->data_size;

    fprintf(stderr,
            "[engine] send_layer_subset_gguf: header-only stream, %zu tensors, header %llu MB, tensor bytes %llu MB, layers %d-%d\n",
            sel.size(),
            (unsigned long long)hdr.size()/(1024*1024),
            (unsigned long long)total_tensor_bytes/(1024*1024),
            layer_start, layer_end-1);
    fflush(stderr);

    uint64_t header_size = (uint64_t)hdr.size();
    if (!write_fn(&header_size, sizeof(uint64_t))) return false;
    if (!write_fn(hdr.data(), hdr.size())) return false;

    const uint8_t* ds = parser.data_ptr();
    if (!ds) return false;

    uint64_t sent = 0;
    const size_t tot = sel.size();
    for (size_t i = 0; i < tot; ++i) {
        const auto* td = sel[i];
        if (td->data_size > 0 && !write_fn(ds + td->offset, td->data_size)) return false;
        sent += (uint64_t)td->data_size;
        const std::string remapped = nx_remap_name(td->name, layer_start);
        if (remapped != td->name) {
            fprintf(stderr, "[send_tensor] [%3zu/%3zu] %.0f%%  source %s -> shard %s (%zu MB)\n",
                    i+1, tot, total_tensor_bytes ? 100.0*(double)sent/(double)total_tensor_bytes : 100.0,
                    td->name.c_str(), remapped.c_str(), td->data_size/(1024ULL*1024ULL));
        } else {
            fprintf(stderr, "[send_tensor] [%3zu/%3zu] %.0f%%  %s (%zu MB)\n",
                    i+1, tot, total_tensor_bytes ? 100.0*(double)sent/(double)total_tensor_bytes : 100.0,
                    remapped.c_str(), td->data_size/(1024ULL*1024ULL));
        }
        fflush(stderr);
    }
    return true;
}


std::vector<std::pair<int32_t,float>> Engine::top_logits(int n) const {
    const auto& logits = impl_->transformer.logits();
    if (logits.empty()) return {};

    std::vector<std::pair<float,int32_t>> scored;
    scored.reserve(logits.size());
    for (int32_t i = 0; i < (int32_t)logits.size(); ++i)
        scored.push_back(std::make_pair(logits[i], i));

    int k = std::min(n, (int)scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
        [](const std::pair<float,int32_t>& a, const std::pair<float,int32_t>& b){
            return a.first > b.first; });

    std::vector<std::pair<int32_t,float>> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i)
        result.push_back(std::make_pair(scored[i].second, scored[i].first));
    return result;
}

const std::vector<float>& Engine::logits() const {
    return impl_->transformer.logits();
}

// --- Generate -----------------------------------------------------------------

GenerateResult Engine::generate(const std::string& prompt, const GenerateConfig& cfg) {
    return generate_stream(prompt, nullptr, cfg);
}

GenerateResult Engine::generate_stream(
    const std::string& prompt,
    std::function<bool(const std::string&, int32_t)> token_cb,
    const GenerateConfig& cfg)
{
    GenerateResult result{};
    if (!is_loaded()) return result;

    auto t_start = std::chrono::steady_clock::now();

    // BOS: only add for formats that genuinely need it (e.g. Mistral/Llama-2).
    // ChatML, Llama-3, Phi, DeepSeek embed their own sequence-start tokens.
    // Adding a spurious BOS to ChatML models causes small models (<7B) to
    // immediately sample EOS and produce zero output tokens.
bool add_bos = impl_->chat_tmpl.should_add_bos();
    std::vector<int32_t> tokens = tokenize(prompt, add_bos);

    // ------ DIAG: dump prompt tokenization ----------------------------------
    // Prints the formatted prompt and how the tokenizer encoded it. We are
    // hunting "smooth degradation" output corruption; the most common root
    // causes are special-token splitting, BOS mishandling, and template
    // formatting drift. This dump exposes all three at once.
    {
        fprintf(stderr,
            "[diag] PROMPT (%zu chars, add_bos=%d):\n--BEGIN--\n%s\n--END--\n",
            prompt.size(), add_bos ? 1 : 0, prompt.c_str());

        fprintf(stderr,
            "[diag] TOKENS (%zu total):\n", tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string ts;
            try {
                ts = impl_->tokenizer.decode_token(tokens[i]);
            } catch (...) { ts = "<decode_err>"; }
            // Visible-escape so newlines / control chars don't corrupt log alignment.
            std::string vis;
            for (char c : ts) {
                if (c == '\n')      vis += "\\n";
                else if (c == '\t') vis += "\\t";
                else if (c == '\r') vis += "\\r";
                else if ((unsigned char)c < 0x20) {
                    char buf[8]; snprintf(buf, sizeof(buf), "\\x%02x", (unsigned char)c);
                    vis += buf;
                } else vis += c;
            }
            fprintf(stderr,
                "[diag]   [%4zu] id=%6d \"%s\"\n",
                i, tokens[i], vis.c_str());
        }
        fflush(stderr);
    }
    // ------ END DIAG ---------------------------------------------------------
    if (tokens.empty()) return result;

    if ((int)tokens.size() >= context_limit()) {
        fprintf(stderr, "[engine] Prompt too long: %zu tokens (conversation limit %d)\n",
                tokens.size(), context_limit());
        result.stopped_limit = true;
        return result;
    }

    // Configure GPU offloading if requested
    // NOTE: GPU layers are set once at load time via Engine::set_gpu_layers().
    //       Calling it here on every generate would re-init Vulkan each time.
    //       Left as a no-op for API compatibility.

    // Prefill
    auto t_prefill_start = std::chrono::steady_clock::now();
    fprintf(stderr, "[engine] Prefill: %zu tokens...", tokens.size());
    for (int i = 0; i < (int)tokens.size(); ++i)
        impl_->transformer.forward(tokens, i);
    auto t_prefill_end = std::chrono::steady_clock::now();
    double prefill_sec = std::chrono::duration<double>(t_prefill_end - t_prefill_start).count();
    fprintf(stderr, " done (%.0f ms, %.1f tok/s)\n",
            prefill_sec * 1000.0,
            tokens.size() / std::max(prefill_sec, 1e-9));

    model::SamplerConfig scfg;
    scfg.temperature    = cfg.temperature;
    scfg.top_p          = cfg.top_p;
    scfg.top_k          = cfg.top_k;
    scfg.repeat_penalty = cfg.repeat_penalty;
    scfg.recent_tokens  = &tokens;  // point at live token list for repeat penalty
    if (cfg.seed >= 0) scfg.seed = cfg.seed;

    // Compute how many chars to hold back as a "guard window" so partial
    // stop strings that span multiple tokens never reach the callback.
    int max_stop = 0;
    for (const auto& s : cfg.stop_strings)
        max_stop = std::max(max_stop, (int)s.size());
    max_stop = std::max(max_stop, 1);

    std::string generated;   // full accumulated decoded text
    size_t sent = 0;         // how many chars have been sent to token_cb
    int pos = (int)tokens.size();
    bool first_token = true;
    auto t_gen_start = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg.max_tokens; ++i) {
        if (i == 0) impl_->transformer.park_finish_pending();            // nothing is read from the drive while writing
        if (i > 0 && (i & 127) == 0) impl_->transformer.park_watch();   // parked layers: keep card and RAM out of trouble
        int32_t next_tok = impl_->transformer.sample(scfg);

        // Debug: on first generated token, log what we got
        if (first_token) {
            first_token = false;
            std::string tok_str = detokenize(next_tok);
            fprintf(stderr, "[engine] First token: %d (%s)%s\n",
                next_tok, tok_str.c_str(),
                next_tok == eos_token() ? " - EOS! Check BOS/template." : "");
        }

        // EOS token
        if (next_tok == eos_token()) {
            // Commit EOS/EOT into KV/session history before stopping.
            // Warm chats continue from KV; dropping the closing token leaves
            // the next user turn attached to an unterminated assistant turn.
            tokens.push_back(next_tok);
            impl_->transformer.forward(tokens, pos++);
            ++result.n_tokens;
            result.stopped_eos = true;
            break;
        }

        generated += detokenize(next_tok);
        ++result.n_tokens;

        // Check all stop strings in accumulated text, searching from just
        // before the last confirmed-safe position to catch multi-token seqs
        bool stop = false;
        size_t stop_at = std::string::npos;
        for (const auto& s : cfg.stop_strings) {
            if (s.empty()) continue;
            size_t search_from = sent >= s.size() ? sent - s.size() : 0;
            size_t found = generated.find(s, search_from);
            if (found != std::string::npos) {
                stop_at = (stop_at == std::string::npos)
                          ? found : std::min(stop_at, found);
                stop = true;
            }
        }

        if (stop) {
            // Flush everything BEFORE the stop string, nothing after.
            // Still commit the sampled token that completed the stop string
            // into KV, because warm continuation relies on the structural
            // assistant terminator being present in the cached conversation.
            if (token_cb && stop_at > sent)
                token_cb(generated.substr(sent, stop_at - sent), -1);
            tokens.push_back(next_tok);
            impl_->transformer.forward(tokens, pos++);
            generated = generated.substr(0, stop_at);
            result.stopped_string = true;
            break;
        }

        // Flush the "safe" portion - hold back max_stop chars as guard window.
        // Also snap back to a UTF-8 character boundary so we never send a partial
        // multi-byte sequence (which the browser would show as a ? replacement char).
        if (token_cb) {
            size_t safe = generated.size() > (size_t)max_stop
                          ? generated.size() - max_stop : 0;
            // Snap safe back to a valid UTF-8 start byte
            while (safe > sent &&
                   (generated[safe] & 0xC0) == 0x80)  // 0x80..0xBF = continuation byte
                --safe;
            if (safe > sent) {
                bool cont = token_cb(generated.substr(sent, safe - sent), next_tok);
                sent = safe;
                if (!cont) break;
            }
        }

        tokens.push_back(next_tok);
        if (pos >= context_limit() - 1) {
            // Context full: drop the oldest whole turns (never the protected start or the turn
            // being answered) and carry on. The KV holds [0, pos); next_tok is not in it yet.
            impl_->session_tokens.assign(tokens.begin(), tokens.begin() + pos);
            impl_->ctx_pos = pos;
            int freed = context_auto_trim();
            if (freed > 0) {
                tokens = impl_->session_tokens;      // the token list now matches the trimmed KV exactly
                tokens.push_back(next_tok);
                pos = impl_->ctx_pos;
                fprintf(stderr, "[engine] Auto-trim: continuing generation from pos %d\n", pos);
                impl_->transformer.forward(tokens, pos++);
            } else {
                result.stopped_limit = true;
                break;
            }
        } else {
            impl_->transformer.forward(tokens, pos++);
        }
    }

    // Flush remaining guard window (only if we didn't stop on a stop string)
    if (token_cb && !result.stopped_string && sent < generated.size()) {
        token_cb(generated.substr(sent), -1);
    }

    if (!result.stopped_eos && !result.stopped_string && !result.stopped_limit)
        result.stopped_limit = true;

    auto t_end = std::chrono::steady_clock::now();
    result.elapsed_sec     = std::chrono::duration<double>(t_end - t_start).count();
    result.tokens_per_sec  = result.n_tokens / std::max(result.elapsed_sec, 1e-9);
    result.text            = generated;

    // Sync session_tokens so save_session() works after generate_stream()
    impl_->session_tokens = tokens;

    // Detailed perf breakdown to stderr
    double gen_sec = std::chrono::duration<double>(t_end - t_gen_start).count();
    fprintf(stderr,
        "[perf] prefill: %.0f ms (%zu tok, %.1f tok/s) | "
        "generate: %.0f ms (%d tok, %.2f tok/s)\n",
        prefill_sec * 1000.0, tokens.size() - result.n_tokens,
            (tokens.size() - result.n_tokens) / std::max(prefill_sec, 1e-9),
        gen_sec * 1000.0, result.n_tokens,
            result.n_tokens / std::max(gen_sec, 1e-9));
    return result;
}

// --- Context ------------------------------------------------------------------

void Engine::reset_context() {
    impl_->transformer.reset_context();
    impl_->ctx_pos = 0;
    impl_->sections.clear();
    impl_->pinned_tokens = 0;
}

void Engine::set_turbo_prefill(bool enabled) {
    // Turbo prefill is the GPU fast path. Do not let it silently become the
    // CPU/layer-major path. If enabling turbo and GPU streaming is not already
    // active, bring the GPU stream up here too. If that fails, turbo stays off
    // and callers will use the normal CPU prompt-batch route instead.
    if (enabled && !impl_->transformer.turbo_gpu_stream()) {
        set_turbo_gpu_stream(true);
    }

    if (enabled && !impl_->transformer.turbo_gpu_stream()) {
        impl_->turbo_prefill_enabled = false;
        fprintf(stderr,
            "[engine] turbo prefill requested but GPU stream is OFF; using CPU prompt-batch instead\n");
    } else {
        impl_->turbo_prefill_enabled = enabled;
        fprintf(stderr, "[engine] turbo prefill %s\n", enabled ? "ENABLED" : "disabled");
    }
    fflush(stderr);
}
bool Engine::turbo_prefill_enabled() const {
    return impl_->turbo_prefill_enabled;
}

void Engine::set_prefill_token_by_token(bool enabled) {
    impl_->prefill_token_by_token = enabled;
    fprintf(stderr, "[engine] token-by-token prompt prefill %s\n",
            enabled ? "ENABLED" : "disabled");
    fflush(stderr);
}

bool Engine::prefill_token_by_token() const {
    return impl_->prefill_token_by_token;
}

void Engine::set_turbo_gpu_stream(bool e) {
    if (e) {
        // Turbo streaming needs a live GPU even when n_gpu_layers == 0.
        if (!ensure_gpu_ready(impl_->turbo_gpu_device)) {
            fprintf(stderr, "[engine] turbo GPU streaming requested but GPU init failed — stream disabled\n");
            impl_->transformer.set_turbo_gpu_stream(false);
            return;
        }
    }
    impl_->transformer.set_turbo_gpu_stream(e);
    fprintf(stderr, "[engine] turbo GPU streaming %s\n", e ? "ENABLED" : "disabled");
}

void Engine::set_turbo_gpu_device(int index) {
    impl_->turbo_gpu_device = index;
    auto& g = nexus::gpu::gpu();
    if (impl_->transformer.turbo_gpu_stream() && g.ready()
        && index >= 0 && impl_->n_gpu_layers_committed == 0) {
        // Only safe to re-init the device if no standing layers depend on it.
        g.shutdown();
        ensure_gpu_ready(index);
    }
}

//int Engine::turbo_gpu_device() const { return impl_->turbo_gpu_device; }

bool Engine::turbo_gpu_stream() const {
    return impl_->transformer.turbo_gpu_stream();
}

void Engine::set_turbo_vram_budget_mb(int mb) {
    size_t bytes = (mb > 0) ? (size_t)mb * 1024 * 1024 : 0;
    gpu::gpu().set_weight_cache_budget(bytes);
    impl_->turbo_vram_budget_mb = mb;
}

int Engine::turbo_vram_budget_mb() const {
    return impl_->turbo_vram_budget_mb;
}

void Engine::set_rope_stabilizer(bool enabled, int trigger_effective_tokens, int target_effective_tokens) {
    impl_->transformer.set_rope_stabilizer(enabled, trigger_effective_tokens, target_effective_tokens);
}

bool Engine::rope_stabilizer_enabled() const {
    return impl_->transformer.rope_stabilizer_enabled();
}

int Engine::rope_stabilizer_trigger() const {
    return impl_->transformer.rope_stabilizer_trigger();
}

int Engine::rope_stabilizer_target() const {
    return impl_->transformer.rope_stabilizer_target();
}

float Engine::rope_position_scale() const {
    return impl_->transformer.rope_position_scale();
}

void Engine::prefill_batch(const std::vector<int32_t>& tokens, int start, int n) {
    // Parked layers back onto the card (or into RAM) once the prompt is read -- also when the read stopped early.
    try { impl_->transformer.prefill_batch(tokens, start, n); }
    catch (...) { impl_->transformer.park_finish_pending(); throw; }
    impl_->transformer.park_finish_pending();
}

void Engine::prefill_cpu_prompt_batch(const std::vector<int32_t>& tokens, int start, int n) {
    impl_->transformer.prefill_cpu_prompt_batch(tokens, start, n);
}

int Engine::context_size() const {
    return impl_->ctx_pos;
}

void Engine::log_state(const char* tag) const {
    const auto& st = impl_->session_tokens;
    std::string tail;
    const size_t n = std::min<size_t>(12, st.size());
    for (size_t i = st.size() - n; i < st.size(); ++i)
        tail += "[" + nx_visible_token_text(detokenize(st[i])) + "]";
    const std::string fp = impl_->transformer.state_fingerprint();
    fprintf(stderr, "[state] %s: ctx_pos=%d session_tokens=%zu | %s | last tokens: %s\n",
            tag, impl_->ctx_pos, st.size(), fp.c_str(), tail.c_str());
    fflush(stderr);
}

int Engine::max_context() const {
    return impl_->transformer.context_length();
}

// -- Section marker API --------------------------------------------------------
// Call context_mark_start("name") before injecting a logical block,
// context_mark_end("name") after. The engine records token positions.
// context_pin("name") marks a section as never-evictable.
// context_trim(target_tokens) evicts the oldest unpinned sections to fit.

void Engine::context_mark_start(const std::string& name) {
    // If a section with this name already exists, update it
    for (auto& s : impl_->sections) {
        if (s.name == name) {
            s.start_token = impl_->ctx_pos;
            s.end_token   = impl_->ctx_pos;  // will be closed by mark_end
            return;
        }
    }
    impl_->sections.push_back({name, impl_->ctx_pos, impl_->ctx_pos, false});
}

void Engine::context_mark_end(const std::string& name) {
    for (auto& s : impl_->sections) {
        if (s.name == name) {
            s.end_token = impl_->ctx_pos;
            return;
        }
    }
}

void Engine::context_pin(const std::string& name) {
    for (auto& s : impl_->sections)
        if (s.name == name) { s.pinned = true; return; }
}

// Pin all tokens up to the current position as the system/tool banner.
// These are never evicted. Call once after initial prefill.
void Engine::context_pin_prefix() {
    impl_->pinned_tokens = impl_->ctx_pos;
    // Also create/update the implicit pinned section
    bool found = false;
    for (auto& s : impl_->sections) {
        if (s.name == "__prefix__") {
            s.end_token = impl_->ctx_pos;
            s.pinned = true;
            found = true;
            break;
        }
    }
    if (!found)
        impl_->sections.push_back({"__prefix__", 0, impl_->ctx_pos, true});
    fprintf(stderr, "[ctx] Pinned prefix: %d tokens\n", impl_->pinned_tokens);
}

// ─── Conversation memory editing ─────────────────────────────────────────────────────────
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
static NxTurnMarkers nx_turn_markers(const Engine& e) {
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

static std::string nx_turn_role(const Engine& e, const NxTurnMarkers& m,
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

static std::string nx_turn_preview(const Engine& e, const std::vector<int32_t>& toks, int a, int b, const std::string& role) {
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

std::vector<Engine::ContextTurn> Engine::context_turns() const {
    std::vector<ContextTurn> out;
    const auto& toks = impl_->session_tokens;
    const int n = (int)toks.size();
    if (n == 0) return out;
    const NxTurnMarkers m = nx_turn_markers(*this);
    std::vector<int> starts;
    if (!m.starts.empty())
        for (int i = 0; i < n; ++i)
            if (std::find(m.starts.begin(), m.starts.end(), toks[(size_t)i]) != m.starts.end()) starts.push_back(i);
    auto add = [&](int a, int b, const std::string& role) {
        ContextTurn t; t.start = a; t.end = b; t.role = role;
        t.preview = nx_turn_preview(*this, toks, a, b, role);
        out.push_back(t);
    };
    if (starts.empty()) { add(0, n, "turn"); return out; }   // no turn markers in this chat format: one block
    else {
        if (starts[0] > 0) add(0, starts[0], "banner");
        for (size_t k = 0; k < starts.size(); ++k) {
            const int a = starts[k], b = k + 1 < starts.size() ? starts[k + 1] : n;
            add(a, b, nx_turn_role(*this, m, toks, a, b));
        }
    }
    // Protected start: whatever comes before the first turn (BOS ...), leading system turns,
    // and anything pinned -- rounded up to whole turns.
    int prot = 0;
    for (const auto& t : out) { if (t.role == "banner" || t.role == "system") prot = t.end; else break; }
    if (impl_->pinned_tokens > prot)
        for (const auto& t : out) if (t.start < impl_->pinned_tokens) prot = std::max(prot, t.end);
    for (auto& t : out) {
        t.protected_part = t.end <= prot;
        for (const auto& sec : impl_->sections)        // sections pinned with context_pin("name")
            if (sec.pinned && sec.start_token < t.end && sec.end_token > t.start) t.protected_part = true;
    }
    return out;
}

// Length of the protected START of the conversation (banner/system prompt/pinned prefix).
int Engine::context_protected_tokens() const {
    const auto turns = context_turns();
    if (turns.size() == 1 && turns[0].role == "turn") return std::max(1, impl_->pinned_tokens);   // marker-less format: the pinned prefix (at least BOS)
    int prot = 0;
    for (const auto& t : turns) { if (t.protected_part && t.start == prot) prot = t.end; else break; }
    return prot;
}

void Engine::context_pin_tokens(int n) {
    impl_->pinned_tokens = std::max(0, std::min(n, impl_->ctx_pos));
    fprintf(stderr, "[ctx] Protected start: %d token(s) pinned (rounded up to whole turns: %d)\n",
            impl_->pinned_tokens, context_protected_tokens());
}

const std::vector<int32_t>& Engine::context_tokens() const { return impl_->session_tokens; }
bool   Engine::context_edits_exact() const { return !impl_->transformer.has_recurrent_state(); }
size_t Engine::kv_bytes_per_token() const { return impl_->transformer.kv_bytes_per_token(); }
size_t Engine::kv_capacity_tokens() const { return impl_->transformer.kv_capacity_tokens(); }
void   Engine::set_auto_trim(bool on) { impl_->auto_trim_on = on; }
bool   Engine::auto_trim() const { return impl_->auto_trim_on; }
// tokens > 0: exactly that many, even above the model's own maximum (you asked for it).
// 0: the model's own maximum. < 0: no limit at all -- the conversation grows until memory runs out.
void   Engine::set_context_limit(int tokens) { impl_->ctx_limit = tokens < 0 ? -1 : tokens; }
int    Engine::context_limit() const {
    if (impl_->ctx_limit < 0) return 1 << 30;
    if (impl_->ctx_limit > 0) return impl_->ctx_limit;
    const int mx = impl_->transformer.context_length();
    return mx > 0 ? mx : (1 << 30);
}

// The engine's token list, position and KV must agree before anything is cut out of the middle.
bool Engine::ctx_consistent_(std::string* err) {
    auto& I = *impl_;
    auto& kv = I.transformer.kvcache_ref();
    if ((int)I.session_tokens.size() != I.ctx_pos) {
        if (err) *err = "the conversation memory is busy or out of step (finish or stop the current reply first)";
        return false;
    }
    if (kv.n_tokens() > I.ctx_pos) kv.set_n_tokens(I.ctx_pos);    // stale positions past the end (a rejected MTP draft)
    if (kv.n_tokens() < I.ctx_pos) {
        if (err) *err = "this model's memory is not all on this computer (split across computers?), so it can't be edited here";
        return false;
    }
    return true;
}

bool Engine::context_remove(int start, int end, std::string* err) {
    auto& I = *impl_;
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (!ctx_consistent_(err)) return false;
    if (start < 0 || end > I.ctx_pos || start >= end) return fail("that part is not in the conversation memory");
    if (start < context_protected_tokens()) return fail("that is part of the protected start of the conversation (the system prompt)");
    for (const auto& t : context_turns())
        if (t.protected_part && t.start < end && t.end > start)
            return fail("that includes a protected part of the conversation (the system prompt or a pinned part)");
    I.transformer.kv_remove_range(start, end);
    I.session_tokens.erase(I.session_tokens.begin() + start, I.session_tokens.begin() + end);
    I.ctx_pos = (int)I.session_tokens.size();
    for (auto& sec : I.sections) {                 // keep named sections pointing at the same text
        auto mv = [&](int p) { return p <= start ? p : (p >= end ? p - (end - start) : start); };
        sec.start_token = mv(sec.start_token);
        sec.end_token   = mv(sec.end_token);
    }
    log_state("after removing part of the conversation");
    return true;
}

// Archive file: "NXCTXAR1" | u32 version | u32 n_tokens | i32 tokens[n] | u64 kv_size | KV blob
bool Engine::context_archive(int start, int end, const std::string& path, std::string* err) {
    auto& I = *impl_;
    auto fail = [&](const char* m) { if (err) *err = m; return false; };
    if (!ctx_consistent_(err)) return false;
    if (start < 0 || end > I.ctx_pos || start >= end) return fail("that part is not in the conversation memory");
    for (const auto& t : context_turns())
        if (t.protected_part && t.start < end && t.end > start)
            return fail("that includes a protected part of the conversation (the system prompt or a pinned part)");
    std::vector<uint8_t> kvb;
    if (!I.transformer.kv_export_range(start, end, kvb)) return fail("could not read that part of the memory");
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return fail("could not create the archive file");
    const uint32_t ver = 1, n = (uint32_t)(end - start);
    const uint64_t ks = kvb.size();
    bool ok = fwrite("NXCTXAR1", 1, 8, f) == 8 && fwrite(&ver, 4, 1, f) == 1 && fwrite(&n, 4, 1, f) == 1 &&
              fwrite(I.session_tokens.data() + start, sizeof(int32_t), n, f) == n &&
              fwrite(&ks, 8, 1, f) == 1 && fwrite(kvb.data(), 1, kvb.size(), f) == kvb.size();
    ok = (fclose(f) == 0) && ok;
    if (!ok) { std::remove(tmp.c_str()); return fail("could not write the archive file (disk full?)"); }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return fail("could not save the archive file"); }
    fprintf(stderr, "[ctx] Archived tokens [%d,%d) to %s (%.1f MB)\n", start, end, path.c_str(), kvb.size() / 1048576.0);
    return context_remove(start, end, err);
}

int Engine::context_restore(const std::string& path, std::string* err) {
    auto& I = *impl_;
    auto fail = [&](const char* m) { if (err) *err = m; return -1; };
    if (!ctx_consistent_(err)) return -1;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return fail("the archive file was not found");
    char magic[8]; uint32_t ver = 0, n = 0; uint64_t ks = 0;
    std::vector<int32_t> toks; std::vector<uint8_t> kvb;
    bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "NXCTXAR1", 8) == 0 &&
              fread(&ver, 4, 1, f) == 1 && ver == 1 && fread(&n, 4, 1, f) == 1 && n > 0 && n < (1u << 28);
    if (ok) { toks.resize(n); ok = fread(toks.data(), sizeof(int32_t), n, f) == n && fread(&ks, 8, 1, f) == 1 && ks < (1ull << 40); }
    if (ok) { kvb.resize((size_t)ks); ok = fread(kvb.data(), 1, kvb.size(), f) == kvb.size(); }
    fclose(f);
    if (!ok) return fail("the archive file is damaged or not a Nexus memory archive");
    const int pos0 = I.ctx_pos;
    const int added = I.transformer.kv_import_append(kvb.data(), kvb.size(), err);
    if (added < 0) return -1;
    if (added != (int)n) {                     // should never happen; undo rather than leave the two out of step
        I.transformer.kv_remove_range(pos0, pos0 + added);
        return fail("the archive's token list and memory don't match");
    }
    I.session_tokens.insert(I.session_tokens.end(), toks.begin(), toks.end());
    I.ctx_pos = (int)I.session_tokens.size();
    log_state("after restoring an archived part");
    return added;
}

// Drop the oldest whole turns until the context is at most target_tokens. Never touches the
// protected start, the newest user turn, or anything after it (the turn being answered).
// Returns tokens freed.
int Engine::context_trim(int target_tokens) {
    auto& I = *impl_;
    const int cur = I.ctx_pos;
    if (cur <= target_tokens) return 0;
    const int need = cur - target_tokens;
    const auto turns = context_turns();
    int keep_from = cur;
    for (int k = (int)turns.size() - 1; k >= 0; --k)
        if (turns[(size_t)k].role == "user" || turns[(size_t)k].role == "exchange") { keep_from = turns[(size_t)k].start; break; }
    const bool structured = !(turns.size() == 1 && turns[0].role == "turn");
    int a = -1, b = -1;
    for (const auto& t : turns) {
        if (t.protected_part) { if (a >= 0) break; continue; }   // stop at a pinned part: removal is one contiguous cut
        if (t.start >= keep_from) break;
        if (a < 0) a = t.start;
        b = t.end;
        if (b - a >= need) break;
    }
    if (a < 0 && !structured) {                // no turn markers in this template: plain oldest-token trim
        a = context_protected_tokens();
        b = std::min(cur - 1, a + need);
        if (b <= a) a = -1;
    }
    if (a < 0 || b <= a) {
        fprintf(stderr, "[ctx] Context full but nothing can be dropped (the current turn alone fills it)\n");
        return 0;
    }
    std::string err;
    if (!context_remove(a, b, &err)) { fprintf(stderr, "[ctx] Trim failed: %s\n", err.c_str()); return 0; }
    fprintf(stderr, "[ctx] Dropped the oldest %d token(s) to make room (%d -> %d)\n", b - a, cur, I.ctx_pos);
    return b - a;
}

// Called automatically when the context reaches its limit during generation.
// Trims to 75% of the limit (whole oldest turns), keeping the protected start.
int Engine::context_auto_trim() {
    if (!impl_->auto_trim_on) return 0;
    const int lim = context_limit();
    const int cur = impl_->ctx_pos;
    if (cur < lim - 64) return 0;
    int target = (int)((long long)lim * 3 / 4);
    const int floor_tokens = context_protected_tokens() + 128;
    if (target < floor_tokens) target = floor_tokens;
    if (target >= cur) return 0;
    fprintf(stderr, "[ctx] Context full (%d of %d tokens): dropping the oldest turns\n", cur, lim);
    return context_trim(target);
}

int Engine::context_make_room(int incoming_tokens, int reserve_tokens) {
    if (!impl_->auto_trim_on) return 0;
    const int lim = context_limit();
    const long long want = (long long)impl_->ctx_pos + incoming_tokens + reserve_tokens;
    if (want <= lim) return 0;
    long long target = std::min<long long>((long long)lim - incoming_tokens - reserve_tokens, (long long)lim * 3 / 4);
    if (target < 0) target = 0;
    fprintf(stderr, "[ctx] Next turn needs %d (+%d reserved) tokens, context %d of %d: dropping the oldest turns\n",
            incoming_tokens, reserve_tokens, impl_->ctx_pos, lim);
    return context_trim((int)target);
}

// --- GenerateSession ----------------------------------------------------------

struct GenerateSession::Impl {
    Engine*                    engine;
    model::Transformer*        xfmr = nullptr;
    std::vector<int32_t>       tokens;
    int                        pos      = 0;
    int                        n_generated = 0;   // tokens this session has produced (real count, MTP drafts included)
    bool                       is_done  = false;
    GenerateConfig             cfg;
    std::vector<std::string>   pause_tags;   // e.g. {"nexus:tool","nexus:call"}
    std::string                pending_text; // generated text not yet returned
    model::SamplerConfig       scfg;

    // Build SamplerConfig from GenerateConfig
    void build_scfg() {
        scfg.temperature       = cfg.temperature;
        scfg.top_p             = cfg.top_p;
        scfg.top_k             = cfg.top_k;
        scfg.min_p             = cfg.min_p;
        scfg.tfs_z             = cfg.tfs_z;
        scfg.repeat_penalty    = cfg.repeat_penalty;
        scfg.repeat_last_n     = cfg.repeat_last_n;
        scfg.frequency_penalty = cfg.frequency_penalty;
        scfg.presence_penalty  = cfg.presence_penalty;
        scfg.seed              = cfg.seed;
        scfg.recent_tokens     = &tokens;
    }

    // Run prompt prefill for positions [start, tokens.size()).
    // Three separate routes:
    //   1) token-by-token debug = old forward() loop
    //   2) turbo              = layer-major GPU stream/resident-window path
    //   3) normal CPU         = full-prompt CPU batch, no GPU stream/eviction
    void prefill(int start) {
        const int n = (int)tokens.size() - start;
        if (n <= 0) { pos = (int)tokens.size(); return; }

        if (engine->prefill_token_by_token()) {
            for (int i = 0; i < n; ++i) engine->forward(tokens, start + i);
            fprintf(stderr, "[session] prefill mode=token-by-token +%d tokens\n", n);
        } else if (engine->turbo_prefill_enabled() && engine->turbo_gpu_stream()) {
            engine->prefill_batch(tokens, start, n);
            fprintf(stderr, "[session] prefill mode=turbo-gpu-stream +%d tokens\n", n);
        } else {
            if (engine->turbo_prefill_enabled() && !engine->turbo_gpu_stream()) {
                fprintf(stderr,
                    "[session] turbo requested but GPU stream is OFF; using cpu-prompt-batch +%d tokens\n", n);
            }
            engine->prefill_cpu_prompt_batch(tokens, start, n);
            fprintf(stderr, "[session] prefill mode=cpu-prompt-batch +%d tokens\n", n);
        }
        fflush(stderr);
        pos = (int)tokens.size();
    }

    // Check if pending_text contains a complete pause tag.
    // Returns true and fills tag_name/tag_input/tag_content if so.
    bool check_pause(std::string& tag_name, std::string& tag_input,
                     std::string& tag_content, size_t& tag_end_pos) {
        for (const auto& tname : pause_tags) {
            std::string open  = "<"  + tname;
            std::string close = "</" + tname + ">";
            size_t s = pending_text.find(open);
            if (s == std::string::npos) continue;
            size_t e = pending_text.find(close, s);
            if (e == std::string::npos) continue;
            e += close.size();
            tag_content = pending_text.substr(s, e - s);
            tag_name    = tname;
            // Extract inner content (between first > and close tag)
            size_t inner_start = tag_content.find('>');
            size_t inner_end   = tag_content.rfind("</" + tname);
            if (inner_start != std::string::npos && inner_end != std::string::npos)
                tag_input = tag_content.substr(inner_start + 1, inner_end - inner_start - 1);
            tag_end_pos = e;
            return true;
        }
        return false;
    }
};

GenerateSession::GenerateSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

GenerateSession::~GenerateSession() = default;
GenerateSession::GenerateSession(GenerateSession&&) noexcept = default;
GenerateSession& GenerateSession::operator=(GenerateSession&&) noexcept = default;

bool GenerateSession::done()  const { return impl_->is_done; }
int  GenerateSession::pos()   const { return impl_->pos; }
int  GenerateSession::generated() const { return impl_->n_generated; }
const std::vector<int32_t>& GenerateSession::tokens() const { return impl_->tokens; }

SessionChunk GenerateSession::run(
    std::function<bool(const std::string& tok, int32_t id)> token_cb)
{
    SessionChunk chunk;
    if (impl_->is_done) { chunk.done = true; return chunk; }

    auto& imp  = *impl_;
    auto& eng  = *imp.engine;
    auto& xfmr = *imp.xfmr;

    // Build a local stop-string max for guard window
    int max_stop = 0;
    for (const auto& s : imp.cfg.stop_strings)
        max_stop = std::max(max_stop, (int)s.size());
    // Also account for longest pause tag close
    for (const auto& t : imp.pause_tags)
        max_stop = std::max(max_stop, (int)(t.size() + 3)); // </ + name + >
    max_stop = std::max(max_stop, 16);

    size_t sent = 0; // chars in pending_text already in chunk.text
    std::string& acc = imp.pending_text;

    // Apply MTP settings from the request config. Safe if the model has no MTP
    // tensors (mtp_enabled() will report false and the loop skips it).
    xfmr.set_mtp(imp.cfg.mtp_enabled, imp.cfg.mtp_n_predict, imp.cfg.mtp_gpu);
    fprintf(stderr, "[mtp] generation start: cfg.mtp_enabled=%d n_predict=%d gpu=%d -> xfmr.mtp_enabled()=%d\n",
            imp.cfg.mtp_enabled?1:0, imp.cfg.mtp_n_predict, imp.cfg.mtp_gpu?1:0,
            xfmr.mtp_enabled()?1:0);

    // `pending_next`, when >= 0, is a token already sampled (by MTP's verify or a
    // prior step) that should be emitted THIS iteration without re-sampling. This
    // lets the MTP path hand the loop its next token(s) without the top of the
    // loop independently sampling and double-emitting. -1 means "sample normally".
    int32_t pending_next = -1;

    for (int i = 0; i < imp.cfg.max_tokens; ++i) {
        if (i == 0) xfmr.park_finish_pending();                          // nothing is read from the drive while writing
        if (i > 0 && (i & 127) == 0) xfmr.park_watch();                 // parked layers: keep card and RAM out of trouble
        int32_t next_tok = (pending_next >= 0) ? pending_next : xfmr.sample(imp.scfg);
        pending_next = -1;
        std::string next_txt = eng.detokenize(next_tok);

        {
            fprintf(stderr,
                "[session-gen] i=%d pos=%d tok=%d text=\"%s\"%s\n",
                i, imp.pos, next_tok,
                nx_visible_token_text(next_txt).c_str(),
                next_tok == eng.eos_token() ? " EOS" : "");
            fflush(stderr);
        }

        if (next_tok == eng.eos_token()) {
            // Flush remaining text
            std::string eos_flush = acc.substr(sent);
            chunk.text += eos_flush;
            fprintf(stderr, "[session-gen] EOS flush chars=%zu text=\"%s\"\n",
                    eos_flush.size(), nx_visible_token_text(eos_flush).c_str());
            fflush(stderr);
            if (token_cb) token_cb(eos_flush, next_tok);
            // Commit EOT to KV so the next user turn has a clean separator.
            // Without this the KV ends mid-assistant-response with no closing
            // token, causing role confusion on long multi-turn conversations.
            ++imp.n_generated; imp.tokens.push_back(next_tok);
            xfmr.forward(imp.tokens, imp.pos++);
            imp.is_done = true;
            chunk.done  = true;
            break;
        }

        acc += next_txt;

        // Check for stop strings
        bool stop = false; size_t stop_at = std::string::npos;
        for (const auto& s : imp.cfg.stop_strings) {
            if (s.empty()) continue;
            size_t search_from = sent >= s.size() ? sent - s.size() : 0;
            size_t found = acc.find(s, search_from);
            if (found != std::string::npos) {
                stop_at = (stop_at == std::string::npos) ? found : std::min(stop_at, found);
                stop = true;
                fprintf(stderr,
                    "[session-gen] STOP matched string=\"%s\" found=%zu sent=%zu acc_len=%zu\n",
                    nx_visible_token_text(s).c_str(), found, sent, acc.size());
                fflush(stderr);
            }
        }
        if (stop) {
            std::string flush = acc.substr(sent, stop_at - sent);
            chunk.text += flush;
            fprintf(stderr,
                "[session-gen] STOP flush chars=%zu text=\"%s\" commit_tok=%d commit_text=\"%s\" pos=%d\n",
                flush.size(), nx_visible_token_text(flush).c_str(),
                next_tok, nx_visible_token_text(next_txt).c_str(), imp.pos);
            fflush(stderr);
            if (token_cb) token_cb(flush, -1);
            // Commit through the token that completed the stop string.
            // This keeps KV/session_tokens aligned with the real chat template
            // boundary (for example <|im_end|> / <|eot_id|>) while still hiding
            // that boundary from streamed user-visible text.
            ++imp.n_generated; imp.tokens.push_back(next_tok);
            xfmr.forward(imp.tokens, imp.pos++);
            fprintf(stderr, "[session-gen] STOP committed tok=%d new_pos=%d tokens=%zu\n",
                    next_tok, imp.pos, imp.tokens.size());
            fflush(stderr);
            imp.is_done = true;
            chunk.done  = true;
            break;
        }

        // Check for complete pause tag
        std::string tname, tinput, tcontent;
        size_t tag_end = 0;
        if (imp.check_pause(tname, tinput, tcontent, tag_end)) {
            // Find where tag starts in acc
            size_t tag_start = acc.find("<" + tname);
            // Flush everything before the tag
            std::string pre_tag = acc.substr(sent, tag_start - sent);
            chunk.text += pre_tag;
            if (token_cb && !pre_tag.empty()) token_cb(pre_tag, -1);
            // Return pause info
            chunk.paused      = true;
            chunk.tag_name    = tname;
            chunk.tag_input   = tinput;
            chunk.tag_content = tcontent;
            // Keep everything after the tag as pending for next run()
            acc = acc.substr(tag_end);
            sent = 0;
            // pos is already advanced - KV cache is intact
            break;
        }

        // Stream safe portion (hold back guard window)
        size_t safe = acc.size() > (size_t)max_stop ? acc.size() - max_stop : sent;
        while (safe > sent && (acc[safe] & 0xC0) == 0x80) --safe;
        if (safe > sent) {
            std::string flush = acc.substr(sent, safe - sent);
            chunk.text += flush;
            fprintf(stderr, "[session-gen] flush chars=%zu text=\"%s\" via_tok=%d\n",
                    flush.size(), nx_visible_token_text(flush).c_str(), next_tok);
            fflush(stderr);
            if (token_cb) { if (!token_cb(flush, next_tok)) { imp.is_done = true; chunk.done = true; break; } }
            sent = safe;
        }

        ++imp.n_generated; imp.tokens.push_back(next_tok);

        // Context full: drop the oldest whole turns and carry on with the token just sampled.
        // The KV holds [0, pos); next_tok (just pushed) is not in it yet. Bring the engine's own
        // bookkeeping in line with the KV, trim, then rebuild this session's token list from it.
        bool trimmed_break = false;
        if (imp.pos >= eng.context_limit() - 1) {
            eng.sync_session_tokens(std::vector<int32_t>(imp.tokens.begin(), imp.tokens.begin() + imp.pos), imp.pos);
            int freed = eng.context_auto_trim();
            if (freed > 0) {
                imp.tokens = eng.context_tokens();
                imp.tokens.push_back(next_tok);
                imp.pos = eng.context_size();
                xfmr.forward(imp.tokens, imp.pos++);
                continue;  // re-enter loop cleanly after a trim
            } else {
                fprintf(stderr, "[ctx] Context full (%d of %d tokens) and nothing older can be dropped (auto-trim %s): reply stopped\n",
                        imp.pos, eng.context_limit(), eng.auto_trim() ? "on" : "off");
                fflush(stderr);
                imp.is_done = true; chunk.done = true; break;
            }
        }

        // ── MTP speculative path ─────────────────────────────────────────────
        // next_tok was already emitted (flushed) by the top of this iteration.
        // MTP proposes a CHAIN of up to n_predict drafts, then ONE batched verify
        // sweep over [next_tok, draft0, .., draftK-1] returns the model's own
        // distribution at every position. We accept the longest prefix of drafts
        // that match the model, plus one bonus token, then queue the next token.
        // Output is identical to non-MTP decoding; speed comes from learning
        // multiple tokens per weight-streaming sweep.
        bool mtp_did_step = false;
        if (xfmr.mtp_enabled() && i + 1 < imp.cfg.max_tokens) {
            std::vector<int32_t> drafts;
            auto _t0 = std::chrono::steady_clock::now();
            int np = xfmr.mtp_propose(next_tok, imp.pos, imp.scfg, drafts);
            auto _t1 = std::chrono::steady_clock::now();
            if (np >= 1) {
                auto snap = xfmr.mtp_snapshot_q35_state();
                std::vector<std::vector<float>> lg;   // lg[j] = dist after seq position j
                int filled = xfmr.forward_verifyN(next_tok, drafts, imp.pos, lg);
                auto _t2 = std::chrono::steady_clock::now();
                if (i < 24) {
                    double ms_prop = std::chrono::duration<double,std::milli>(_t1-_t0).count();
                    double ms_ver  = std::chrono::duration<double,std::milli>(_t2-_t1).count();
                    fprintf(stderr, "[mtp-time] step=%d propose=%.1fms verifyN(ntok=%d)=%.1fms\n",
                            i, ms_prop, np+1, ms_ver);
                    fflush(stderr);
                }
                // lg[0] verifies drafts[0]; lg[1] verifies drafts[1]; etc.
                // Accept the longest matching prefix.
                int accepted = 0;                      // how many drafts matched
                std::vector<int32_t> emit;             // tokens to commit/emit this round
                bool hit_eos = false;
                // The repetition penalties read imp.tokens (scfg.recent_tokens). Checking draft j must see exactly the
                // history plain decoding would have at that point -- including drafts 0..j-1 that were just accepted --
                // or MTP could pick a different token than decoding without it. They are added only while checking,
                // and taken off again below; the commit code adds the accepted ones for real.
                const size_t hist_base = imp.tokens.size();
                if (filled >= 1) {
                    for (int j = 0; j < np && j < filled; ++j) {
                        if (j > 0) imp.tokens.push_back(drafts[(size_t)(j - 1)]);
                        int32_t true_tok = xfmr.sample_from(lg[(size_t)j], imp.scfg);
                        if (i < 32) {
                            fprintf(stderr, "[mtp-diag] step=%d j=%d draft=%d true=%d %s\n",
                                    i, j, drafts[(size_t)j], true_tok,
                                    (true_tok == drafts[(size_t)j]) ? "MATCH" : "miss");
                            fflush(stderr);
                        }
                        if (true_tok == eng.eos_token()) { emit.push_back(true_tok); hit_eos = true; break; }
                        emit.push_back(true_tok);
                        if (true_tok == drafts[(size_t)j]) { accepted++; continue; }
                        break;  // first mismatch: true_tok is the correct token, stop
                    }
                }
                // If ALL drafts matched and no eos, we also get a bonus token from
                // the last position's distribution.
                if (!hit_eos && accepted == np && (int)lg.size() >= np + 1) {
                    if (np > 0) imp.tokens.push_back(drafts[(size_t)(np - 1)]);
                    int32_t bonus = xfmr.sample_from(lg[(size_t)np], imp.scfg);
                    if (bonus == eng.eos_token()) { hit_eos = true; }
                    emit.push_back(bonus);
                }
                imp.tokens.resize(hist_base);           // checking only; the commit below adds the accepted tokens

                if (emit.empty()) {
                    // Nothing learned (shouldn't happen) — fall back to normal.
                    xfmr.mtp_restore_q35_state(snap);
                    xfmr.forward(imp.tokens, imp.pos++);
                    mtp_did_step = true;
                } else if (accepted == np && !hit_eos) {
                    // ALL drafts matched. forward_verifyN already advanced the
                    // recurrent state correctly through next_tok + every draft, so
                    // there is NOTHING to roll back or re-forward. Commit next_tok
                    // and all drafts; queue the bonus token (emit.back()) for the
                    // next iteration. This is the fast path: 0 extra forwards.
                    // Sequence committed here: next_tok (already in imp.tokens) +
                    // drafts[0..np-1]. pos advanced by verifyN's internal forwards.
                    imp.pos += (np + 1);  // next_tok + np drafts now committed
                    for (int c = 0; c < np; ++c) {
                        int32_t tk = drafts[(size_t)c];
                        ++imp.n_generated; imp.tokens.push_back(tk);
                        acc += eng.detokenize(tk);
                        fprintf(stderr, "[session-gen] MTP-accepted pos=%d tok=%d text=\"%s\"\n",
                                imp.pos - np + c, tk,
                                nx_visible_token_text(eng.detokenize(tk)).c_str());
                        fflush(stderr);
                        ++i;
                        if (i + 1 >= imp.cfg.max_tokens) { hit_eos = true; break; }
                    }
                    // emit.back() is the bonus (sampled from the last position but
                    // NOT yet forwarded). Queue it for the next iteration's top.
                    if (!imp.is_done && !hit_eos) pending_next = emit.back();
                    else if (hit_eos) { /* bonus dropped if we hit the cap */ }
                    mtp_did_step = true;
                } else {
                    // MISMATCH at position `accepted` (or eos). forward_verifyN
                    // advanced state through the wrong-draft tail; roll back to
                    // before next_tok and re-forward only the CORRECT committed
                    // sequence: next_tok + the accepted drafts + the correcting
                    // token (emit[accepted]). The last emitted token is queued.
                    xfmr.mtp_restore_q35_state(snap);

                    int commit_count = (int)emit.size() - 1;  // last is queued
                    if (commit_count < 0) commit_count = 0;

                    // No second pass: the check already ran next_tok and the accepted drafts through every layer, with the
                    // right history (each position only sees the ones before it). Keep those results -- conversation memory,
                    // the hidden state MTP drafts from -- and replay only the recurrent state's cheap update for them.
                    bool keep_ok = true;
                    for (int c = 0; c < commit_count; ++c)
                        if (emit[(size_t)c] == eng.eos_token() || c >= np || emit[(size_t)c] != drafts[(size_t)c]) { keep_ok = false; break; }
                    if (keep_ok) keep_ok = xfmr.mtp_keep_checked(1 + commit_count, imp.pos);
                    if (keep_ok) {
                        ++imp.pos;                                   // next_tok
                        for (int c = 0; c < commit_count; ++c) {
                            int32_t tk = emit[(size_t)c];
                            ++imp.n_generated; imp.tokens.push_back(tk);
                            acc += eng.detokenize(tk);
                            fprintf(stderr, "[session-gen] MTP-accepted pos=%d tok=%d text=\"%s\"\n",
                                    imp.pos, tk, nx_visible_token_text(eng.detokenize(tk)).c_str());
                            fflush(stderr);
                            ++imp.pos;
                            ++i;
                            if (i + 1 >= imp.cfg.max_tokens) { hit_eos = true; break; }
                        }
                        if (!imp.is_done) pending_next = emit.back();
                        if (hit_eos && emit.back() == eng.eos_token()) { imp.is_done = true; chunk.done = true; }
                        mtp_did_step = true;
                    } else {
                    // Fallback (the check didn't record every layer): re-forward next_tok @ pos (state rolled back to before it).
                    xfmr.forward(imp.tokens, imp.pos++);

                    for (int c = 0; c < commit_count; ++c) {
                        int32_t tk = emit[(size_t)c];
                        if (tk == eng.eos_token()) { imp.is_done = true; chunk.done = true; break; }
                        ++imp.n_generated; imp.tokens.push_back(tk);
                        acc += eng.detokenize(tk);
                        fprintf(stderr, "[session-gen] MTP-accepted pos=%d tok=%d text=\"%s\"\n",
                                imp.pos, tk, nx_visible_token_text(eng.detokenize(tk)).c_str());
                        fflush(stderr);
                        xfmr.forward(imp.tokens, imp.pos++);
                        ++i;
                        if (i + 1 >= imp.cfg.max_tokens) { hit_eos = true; break; }
                    }
                    if (!imp.is_done) pending_next = emit.back();
                    if (hit_eos && emit.back() == eng.eos_token()) { imp.is_done = true; chunk.done = true; }
                    mtp_did_step = true;
                    }
                }
            }
        }

        if (!mtp_did_step) {
            // Normal single-token advance (MTP off, or no draft produced).
            auto _tb0 = std::chrono::steady_clock::now();
            xfmr.forward(imp.tokens, imp.pos++);
            auto _tb1 = std::chrono::steady_clock::now();
            if (i < 24) {
                fprintf(stderr, "[mtp-time] step=%d baseline_forward=%.1fms\n", i,
                        std::chrono::duration<double,std::milli>(_tb1-_tb0).count());
                fflush(stderr);
            }
        }

        // Flush any newly-accumulated text (covers the MTP-accepted true_after).
        {
            size_t safeM = acc.size() > (size_t)max_stop ? acc.size() - max_stop : sent;
            while (safeM > sent && (acc[safeM] & 0xC0) == 0x80) --safeM;
            if (safeM > sent) {
                std::string flush = acc.substr(sent, safeM - sent);
                chunk.text += flush;
                fprintf(stderr, "[session-gen] flushM chars=%zu text=\"%s\" via_tok=%d\n",
                        flush.size(), nx_visible_token_text(flush).c_str(), next_tok);
                fflush(stderr);
                if (token_cb) { if (!token_cb(flush, next_tok)) { imp.is_done = true; chunk.done = true; break; } }
                sent = safeM;
            }
        }
        continue;
    }

    return chunk;
}

void GenerateSession::inject(const std::string& text) {
    if (impl_->is_done || text.empty()) return;
    auto& imp = *impl_;

    // Tokenize the injected text (no BOS - it's a continuation)
    std::vector<int32_t> new_toks = imp.engine->tokenize(text, /*add_bos=*/false);
    if (new_toks.empty()) return;

    // Append to token list and prefill only the NEW tokens
    int start = (int)imp.tokens.size();
    imp.tokens.insert(imp.tokens.end(), new_toks.begin(), new_toks.end());
    imp.prefill(start);
    // pos is now at end of tokens, ready for next run()
}

void GenerateSession::inject_embeddings(const std::vector<float>& embeddings,
                                        int hidden_dim,
                                        int32_t placeholder_token) {
    if (impl_->is_done || embeddings.empty() || hidden_dim <= 0) return;
    auto& imp = *impl_;
    auto& xfmr = *imp.xfmr;
    const size_t h = (size_t)hidden_dim;
    const size_t n = embeddings.size() / h;
    if (n == 0 || embeddings.size() != n * h) {
        fprintf(stderr, "[vision] inject_embeddings rejected: floats=%zu hidden=%d\n",
                embeddings.size(), hidden_dim);
        return;
    }
    const int start = imp.pos;
    if (imp.engine->prefill_token_by_token()) {
        for (int i = 0; i < (int)n; ++i) {
            const float* e = embeddings.data() + (size_t)i * (size_t)hidden_dim;
            if (i + 1 == (int)n)
                xfmr.forward_embed(e, start + i);
            else
                xfmr.forward_embed_prefill(e, start + i);
        }
    } else if (imp.engine->turbo_prefill_enabled() && imp.engine->turbo_gpu_stream()) {
        xfmr.prefill_embeddings_stream(embeddings.data(), hidden_dim, start, (int)n);
    } else {
        if (imp.engine->turbo_prefill_enabled() && !imp.engine->turbo_gpu_stream()) {
            fprintf(stderr,
                "[session:emb] turbo requested but GPU stream is OFF; using CPU token-major embeddings\n");
        }
        xfmr.prefill_embeddings_cpu_prompt_batch(embeddings.data(), hidden_dim, start, (int)n);
    }
    imp.pos += (int)n;
    imp.tokens.insert(imp.tokens.end(), n, placeholder_token);
    const char* emb_mode = imp.engine->prefill_token_by_token()
        ? "token-by-token"
        : ((imp.engine->turbo_prefill_enabled() && imp.engine->turbo_gpu_stream()) ? "turbo-gpu-stream" : "cpu-token-major");
    fprintf(stderr, "[vision] injected %zu embedding tokens into session via %s, pos=%d\n",
            n, emb_mode, imp.pos);
}

GenerateSession Engine::begin_session(
    const std::string& prompt,
    const GenerateConfig& cfg,
    std::vector<std::string> pause_tags)
{
    auto si = std::make_unique<GenerateSession::Impl>();
    si->engine     = this;
    si->xfmr       = &impl_->transformer;
    si->cfg        = cfg;
    si->pause_tags = std::move(pause_tags);

    // Reset context and tokenize prompt
    reset_context();
    si->tokens = tokenize(prompt, /*add_bos=*/should_add_bos());

    // Add stop strings from format
    for (const auto& s : stop_strings()) {
        if (std::find(si->cfg.stop_strings.begin(), si->cfg.stop_strings.end(), s)
                == si->cfg.stop_strings.end())
            si->cfg.stop_strings.push_back(s);
    }

    si->build_scfg();

    // Session prompt/token dump: this is the streaming path used by the UI.
    // It exposes template/control-token mistakes that Engine::generate_stream()
    // diagnostics do not see.
    {
        fprintf(stderr,
            "[session] PROMPT (%zu chars, add_bos=%d):\n--BEGIN--\n%s\n--END--\n",
            prompt.size(), should_add_bos() ? 1 : 0, prompt.c_str());
        fprintf(stderr, "[session] PROMPT TOKENS (%zu total):\n", si->tokens.size());
        const size_t dump_n = std::min<size_t>(si->tokens.size(), 96);
        for (size_t ti = 0; ti < dump_n; ++ti) {
            std::string ts;
            try { ts = impl_->tokenizer.decode_token(si->tokens[ti]); }
            catch (...) { ts = "<decode_err>"; }
            fprintf(stderr, "[session]   [%4zu] id=%6d text=\"%s\"\n",
                ti, si->tokens[ti], nx_visible_token_text(ts).c_str());
        }
        if (si->tokens.size() > dump_n) {
            fprintf(stderr, "[session]   ... %zu more tokens omitted ...\n",
                si->tokens.size() - dump_n);
        }
        fprintf(stderr, "[session] STOP STRINGS (%zu):\n", si->cfg.stop_strings.size());
        for (size_t si_stop = 0; si_stop < si->cfg.stop_strings.size(); ++si_stop) {
            fprintf(stderr, "[session]   stop[%zu]=\"%s\"\n",
                si_stop, nx_visible_token_text(si->cfg.stop_strings[si_stop]).c_str());
        }
        fflush(stderr);
    }

    // Prefill the prompt (all but last token - last feeds first forward)
    fprintf(stderr, "[session] Prefill: %zu tokens...", si->tokens.size());
    si->prefill(0);
    fprintf(stderr, " done\n");
    // Sync engine session_tokens so save_session works
    impl_->session_tokens = si->tokens;

    return GenerateSession(std::move(si));
}

GenerateSession Engine::begin_session_warm(
    const GenerateConfig& cfg,
    std::vector<std::string> pause_tags)
{
    // No reset_context, no prefill - use the existing warm KV cache.
    // Call extend_context() before this to add new content.
    auto si = std::make_unique<GenerateSession::Impl>();
    si->engine     = this;
    si->xfmr       = &impl_->transformer;
    si->cfg        = cfg;
    si->pause_tags = std::move(pause_tags);
    si->tokens     = impl_->session_tokens;  // current token history
    si->pos        = impl_->ctx_pos;         // current position
    // Add format stop strings
    for (const auto& s : stop_strings()) {
        if (std::find(si->cfg.stop_strings.begin(), si->cfg.stop_strings.end(), s)
                == si->cfg.stop_strings.end())
            si->cfg.stop_strings.push_back(s);
    }
    si->build_scfg();
    fprintf(stderr, "[session] Warm resume at pos %d\n", si->pos);
    return GenerateSession(std::move(si));
}

GenerateSession Engine::begin_session_chat(
    const std::vector<nexus::ChatMessage>& messages,
    const GenerateConfig& cfg,
    std::vector<std::string> pause_tags)
{
    std::string prompt = format_chat(messages);
    return begin_session(prompt, cfg, std::move(pause_tags));
}

// --- KV cache persistence & context extension --------------------------------

bool Engine::save_session(const std::string& path) const {
    if (impl_->session_tokens.empty()) return false;
    // Save KV cache
    std::string kv_path = path + ".kv";
    if (!impl_->transformer.save_kv_file(kv_path)) return false;
    // Save token list alongside
    std::string tok_path = path + ".tokens";
    FILE* f = fopen(tok_path.c_str(), "wb");
    if (!f) return false;
    uint32_t n = (uint32_t)impl_->session_tokens.size();
    fwrite(&n, sizeof(uint32_t), 1, f);
    fwrite(impl_->session_tokens.data(), sizeof(int32_t), n, f);
    // Also save pinned count
    uint32_t pinned = (uint32_t)impl_->pinned_tokens;
    fwrite(&pinned, sizeof(uint32_t), 1, f);
    fclose(f);
    fprintf(stderr, "[session] Saved %u tokens + KV cache to %s\n", n, path.c_str());
    log_state("after save");
    return true;
}

bool Engine::load_session(const std::string& path) {
    std::string kv_path  = path + ".kv";
    std::string tok_path = path + ".tokens";
    // Load token list
    FILE* f = fopen(tok_path.c_str(), "rb");
    if (!f) return false;
    uint32_t n = 0; (void)fread(&n, sizeof(uint32_t), 1, f);
    impl_->session_tokens.resize(n);
    (void)fread(impl_->session_tokens.data(), sizeof(int32_t), n, f);
    uint32_t pinned = 0; (void)fread(&pinned, sizeof(uint32_t), 1, f);
    impl_->pinned_tokens = (int)pinned;
    fclose(f);
    // Load KV cache - this restores the model state entirely
    fprintf(stderr, "[session] Loading KV from: %s\n", kv_path.c_str()); fflush(stderr);
    // Check file exists first
    FILE* kvf = fopen(kv_path.c_str(), "rb");
    if (!kvf) {
        fprintf(stderr, "[session] KV file not found: %s\n", kv_path.c_str());
        impl_->session_tokens.clear();
        return false;
    }
    fclose(kvf);
    if (!impl_->transformer.load_kv_file(kv_path)) {
        fprintf(stderr, "[session] KV load rejected (size/format mismatch)\n");
        impl_->session_tokens.clear();
        return false;
    }
    impl_->ctx_pos = (int)n;
    fprintf(stderr, "[session] Loaded %u tokens + KV cache from %s\n", n, path.c_str());
    log_state("after load");
    return true;
}

void Engine::extend_context(const std::string& new_text, bool add_bos) {
    log_state("before your prompt");
    auto new_toks = tokenize(new_text, add_bos);
    if (new_toks.empty()) return;

    const int start = (int)impl_->session_tokens.size();
    impl_->session_tokens.insert(impl_->session_tokens.end(),
                                  new_toks.begin(), new_toks.end());

    const int n = (int)impl_->session_tokens.size() - start;
    const bool token_debug = impl_->prefill_token_by_token;
    const bool turbo_gpu   = impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream();

    if (token_debug) {
        for (int i = 0; i < n; ++i)
            impl_->transformer.forward(impl_->session_tokens, start + i);
    } else if (turbo_gpu) {
        prefill_batch(impl_->session_tokens, start, n);
    } else {
        if (impl_->turbo_prefill_enabled && !impl_->transformer.turbo_gpu_stream()) {
            fprintf(stderr,
                "[session] turbo requested but GPU stream is OFF; extend_context using CPU prompt-batch\n");
        }
        prefill_cpu_prompt_batch(impl_->session_tokens, start, n);
    }

    impl_->ctx_pos = (int)impl_->session_tokens.size();
    const char* mode = token_debug ? "token-by-token" : (turbo_gpu ? "turbo-gpu-stream" : "cpu-prompt-batch");
    fprintf(stderr, "[session] Extended context: +%zu tokens (total %d, %s)\n",
            new_toks.size(), impl_->ctx_pos, mode);
    // Same prompt on the same loaded chat must give the same memory fingerprint here.
    // If it differs between two identical attempts, prefill itself is non-deterministic.
    log_state("after your prompt (prefill done)");
}

bool Engine::extend_context_mixed_embeddings(const std::vector<int32_t>& tokens,
                                             const std::vector<uint8_t>& embed_mask,
                                             const std::vector<float>& embedded_inputs) {
    if (tokens.empty()) return true;

    const int hidden = (int)impl_->transformer.meta().hidden_dim;
    if (hidden <= 0) {
        fprintf(stderr, "[session-mixed] failed: invalid hidden=%d\n", hidden);
        return false;
    }
    if (embed_mask.size() != tokens.size()) {
        fprintf(stderr, "[session-mixed] failed: mask=%zu tokens=%zu\n",
                embed_mask.size(), tokens.size());
        return false;
    }
    const size_t need = tokens.size() * (size_t)hidden;
    if (embedded_inputs.size() != need) {
        fprintf(stderr, "[session-mixed] failed: embedded floats=%zu need=%zu tokens=%zu hidden=%d\n",
                embedded_inputs.size(), need, tokens.size(), hidden);
        return false;
    }

    std::vector<float> input(need);
    size_t overrides = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        float* dst = input.data() + i * (size_t)hidden;
        if (embed_mask[i]) {
            const float* src = embedded_inputs.data() + i * (size_t)hidden;
            std::memcpy(dst, src, (size_t)hidden * sizeof(float));
            ++overrides;
        } else {
            if (!impl_->transformer.token_embedding(tokens[i], dst)) {
                fprintf(stderr, "[session-mixed] failed: token embedding lookup tok=%d at mixed index=%zu\n",
                        (int)tokens[i], i);
                return false;
            }
        }
    }

    const int start = (int)impl_->session_tokens.size();
    impl_->session_tokens.insert(impl_->session_tokens.end(), tokens.begin(), tokens.end());

    // One main-model pass over the complete ordered body. Text tokens use their
    // normal token embeddings; image-pad positions use the precomputed visual
    // embeddings. This avoids text-before-image/text-after-image separate layer
    // streaming passes while preserving the exact token order in KV.
    if (impl_->prefill_token_by_token) {
        const int n_embed = (int)tokens.size();
        for (int i = 0; i < n_embed; ++i) {
            const float* e = input.data() + (size_t)i * (size_t)hidden;
            if (i + 1 == n_embed)
                impl_->transformer.forward_embed(e, start + i);
            else
                impl_->transformer.forward_embed_prefill(e, start + i);
        }
    } else if (impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream()) {
        impl_->transformer.prefill_embeddings_stream(input.data(), hidden, start, (int)tokens.size());
    } else {
        if (impl_->turbo_prefill_enabled && !impl_->transformer.turbo_gpu_stream()) {
            fprintf(stderr,
                "[mixed:emb] turbo requested but GPU stream is OFF; using CPU token-major embeddings\n");
        }
        impl_->transformer.prefill_embeddings_cpu_prompt_batch(input.data(), hidden, start, (int)tokens.size());
    }
    impl_->ctx_pos = (int)impl_->session_tokens.size();

    const char* mixed_mode = impl_->prefill_token_by_token
        ? "token-by-token"
        : ((impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream()) ? "turbo-gpu-stream" : "cpu-token-major-embeddings");
    fprintf(stderr, "[session-mixed] Extended context: +%zu tokens (%zu embedded override, total %d, mode=%s)\n",
            tokens.size(), overrides, impl_->ctx_pos, mixed_mode);
    return true;
}


// -- Checkpoint implementation -------------------------------------------------

struct Checkpoint {
    int                       id;
    std::string               name;
    std::vector<int32_t>      tokens;
    int                       ctx_pos;
    // KV cache snapshot - copy all cache slices at this position
    // We store the serialized binary form (same as save_session)
    std::vector<uint8_t>      kv_data;
};

static std::vector<Checkpoint> g_checkpoints;
static std::mutex               g_cp_mu;
static int                      g_cp_next_id = 0;

int Engine::checkpoint_save(const std::string& name) {
    // Serialize KV cache directly to memory - no temp files
    std::vector<uint8_t> kv_data =
        impl_->transformer.save_kv_buffer();
    if (kv_data.empty()) return -1;
    std::lock_guard<std::mutex> lg(g_cp_mu);
    int id = g_cp_next_id++;
    Checkpoint cp;
    cp.id      = id;
    cp.name    = name.empty() ? ("checkpoint_" + std::to_string(id)) : name;
    cp.tokens  = impl_->session_tokens;
    cp.ctx_pos = impl_->ctx_pos;
    cp.kv_data = std::move(kv_data);
    g_checkpoints.push_back(std::move(cp));
    fprintf(stderr, "[checkpoint] Saved '%s' (id=%d, tokens=%d)\n",
            g_checkpoints.back().name.c_str(), id, impl_->ctx_pos);
    return id;
}

bool Engine::checkpoint_restore(int id) {
    std::lock_guard<std::mutex> lg(g_cp_mu);
    for (auto& cp : g_checkpoints) {
        if (cp.id != id) continue;
        // Restore KV cache directly from memory - no temp files
        bool ok = impl_->transformer.load_kv_buffer(
                      cp.kv_data.data(), cp.kv_data.size());
        if (ok) {
            impl_->session_tokens = cp.tokens;
            impl_->ctx_pos        = cp.ctx_pos;
            fprintf(stderr, "[checkpoint] Restored '%s' (id=%d, pos=%d)\n",
                    cp.name.c_str(), id, cp.ctx_pos);
        }
        return ok;
    }
    fprintf(stderr, "[checkpoint] id=%d not found\n", id);
    return false;
}

std::vector<Engine::CheckpointInfo> Engine::checkpoint_list() const {
    std::lock_guard<std::mutex> lg(g_cp_mu);
    std::vector<CheckpointInfo> out;
    for (auto& cp : g_checkpoints)
        out.push_back({cp.id, cp.name, cp.ctx_pos});
    return out;
}

void Engine::checkpoint_delete(int id) {
    std::lock_guard<std::mutex> lg(g_cp_mu);
    g_checkpoints.erase(
        std::remove_if(g_checkpoints.begin(), g_checkpoints.end(),
                       [id](const Checkpoint& c){ return c.id == id; }),
        g_checkpoints.end());
}

// -- Start-state: save/load without client-visible history ---------------------
// Same binary format as save_session but with a different magic number.
// The model server loads it silently - client sees no prior context.

bool Engine::save_start_state(const std::string& path) const {
    // Identical to save_session - reuse it
    return save_session(path);
}

bool Engine::load_start_state(const std::string& path) {
    // load_session restores KV cache + token list.
    // The difference is purely semantic: callers know this is a "blank slate"
    // start for a sub-model, not a resumed conversation.
    return load_session(path);
}

// -- Multi-session context slots -----------------------------------------------
// Each slot holds its own token list and KV cache position.
// The underlying KV cache (in transformer) is shared but saved/restored per slot.
// Simple implementation: serialize/deserialize to temp storage.
// For production use, implement a proper paged KV cache allocator.

struct ContextSlot {
    std::vector<int32_t>  tokens;
    int                   ctx_pos = 0;
    std::vector<uint8_t>  kv_data;  // serialized KV cache for this slot
    bool                  valid = false;
};

static std::vector<ContextSlot> g_slots;
static std::mutex               g_slots_mu;
static int                      g_active_slot = 0;

void Engine::context_slot_select(int slot_id) {
    std::lock_guard<std::mutex> lg(g_slots_mu);
    if (slot_id == g_active_slot) return;

    // Save current slot directly to memory - no temp files
    if (g_active_slot < (int)g_slots.size() && g_slots[g_active_slot].valid) {
        auto& cur = g_slots[g_active_slot];
        cur.kv_data = impl_->transformer.save_kv_buffer();
        cur.tokens  = impl_->session_tokens;
        cur.ctx_pos = impl_->ctx_pos;
    }

    // Grow slot vector if needed
    if (slot_id >= (int)g_slots.size()) g_slots.resize(slot_id + 1);

    // Restore new slot directly from memory - no temp files
    auto& next = g_slots[slot_id];
    if (next.valid && !next.kv_data.empty()) {
        impl_->transformer.load_kv_buffer(
            next.kv_data.data(), next.kv_data.size());
        impl_->session_tokens = next.tokens;
        impl_->ctx_pos        = next.ctx_pos;
    } else {
        reset_context();
        next.valid = true;
    }
    g_active_slot = slot_id;
    fprintf(stderr, "[slot] Switched to slot %d (pos=%d)\n", slot_id, impl_->ctx_pos);
}

void Engine::context_slot_clear(int slot_id) {
    std::lock_guard<std::mutex> lg(g_slots_mu);
    if (slot_id < (int)g_slots.size()) {
        g_slots[slot_id] = ContextSlot{};
    }
    if (slot_id == g_active_slot) reset_context();
}

int Engine::context_slot_size(int slot_id) const {
    std::lock_guard<std::mutex> lg(g_slots_mu);
    if (slot_id == g_active_slot) return impl_->ctx_pos;
    if (slot_id < (int)g_slots.size()) return g_slots[slot_id].ctx_pos;
    return 0;
}

int Engine::context_slot_count() const {
    std::lock_guard<std::mutex> lg(g_slots_mu);
    return (int)g_slots.size();
}


// -- Text embeddings -----------------------------------------------------------

std::vector<float> Engine::embed_text(const std::string& text, bool normalize) {
    auto tokens = tokenize(text, true);
    if (tokens.empty()) return {};
    // Run forward on all tokens, last hidden state is the embedding
    for (int i = 0; i < (int)tokens.size(); ++i)
        impl_->transformer.forward(tokens, i);
    const auto& hidden = impl_->transformer.last_hidden();
    if (hidden.empty()) return {};
    std::vector<float> out(hidden.begin(), hidden.end());
    if (normalize) {
        float norm = 0.0f;
        for (float v : out) norm += v * v;
        norm = sqrtf(norm + 1e-8f);
        for (float& v : out) v /= norm;
    }
    return out;
}

// -- LoRA adapters -------------------------------------------------------------

bool Engine::lora_load(const std::string& path, float scale) {
    return impl_->transformer.lora_load(path, scale);
}
void Engine::lora_unload_all() { impl_->transformer.lora_unload_all(); }
void Engine::lora_set_scale(int idx, float scale) { impl_->transformer.lora_set_scale(idx, scale); }
int  Engine::lora_count() const { return impl_->transformer.lora_count(); }

// -- Speculative decoding session ----------------------------------------------

GenerateSession Engine::begin_session_speculative(
    const std::string&    prompt,
    Engine*               draft_engine,
    int                   n_draft,
    const GenerateConfig& cfg)
{
    // For now: fall back to normal session if no draft engine
    // Full speculative decoding requires interleaving target and draft forwards
    // which doesn't fit cleanly into the GenerateSession streaming model.
    // The speculative path is exposed via Transformer::generate_speculative()
    // for direct use, and we plan a dedicated SpeculativeSession class.
    if (!draft_engine) {
        return begin_session(prompt, cfg, {});
    }
    // Build a standard session but note that draft is available
    // TODO: SpeculativeSession that wraps both engines and interleaves
    fprintf(stderr, "[speculative] Draft engine provided (%p), falling back to standard session\n",
            (void*)draft_engine);
    fprintf(stderr, "[speculative] Use transformer.generate_speculative() directly for now\n");
    return begin_session(prompt, cfg, {});
}

// -- Vision/audio embed injection ----------------------------------------------
// Inject multimodal embeddings into the KV cache by running forward_embed()
// at each multimodal token position. This is how LLaVA works:
// vision tokens occupy positions in the token stream and their KV entries
// are computed from the projected vision embeddings, not from token lookups.




bool Engine::load_vision_encoder(const std::string& path,
    std::function<void(const std::string&,float)> cb) {
    impl_->vision_enc = std::make_unique<nexus::vision::VisionEncoder>();
    return impl_->vision_enc->load(path, cb);
}
bool Engine::has_vision() const {
    return impl_->vision_enc && impl_->vision_enc->loaded();
}
std::vector<float> Engine::encode_image(const uint8_t* rgb, int w, int h) {
    if (!has_vision()) return {};

    // Vision runs before the text prefill that follows image insertion. Make it
    // participate in the same turbo policy: when turbo is enabled, prepare the
    // Vulkan backend here too, so native visual matmuls/attention can use GPU
    // instead of silently reporting CPU-NOREADY. Unsupported visual tensor types
    // still fall back per op inside VisionEncoder.
    if (impl_->turbo_prefill_enabled && nx_env_truthy_engine("NEXUS_VISION_GPU")) {
        (void)ensure_gpu_ready(impl_->turbo_gpu_device);
    }

    return impl_->vision_enc->encode(rgb, w, h);
}
void Engine::inject_vision_tokens(const std::vector<float>& tokens) {
    if (tokens.empty()) {
        fprintf(stderr, "[vision] inject_vision_tokens skipped: empty embedding buffer\n");
        return;
    }
    const int hidden = (int)impl_->transformer.meta().hidden_dim;
    if (hidden <= 0) {
        fprintf(stderr, "[vision] inject_vision_tokens failed: hidden_dim=%d\n", hidden);
        return;
    }
    if ((tokens.size() % (size_t)hidden) != 0) {
        fprintf(stderr, "[vision] inject_vision_tokens failed: floats=%zu not divisible by hidden=%d\n",
                tokens.size(), hidden);
        return;
    }
    const size_t n_tok = tokens.size() / (size_t)hidden;
    int32_t image_pad_token = 248056; // Qwen3.6 image_token_id fallback
    try {
        std::vector<int32_t> pad = tokenize("<|image_pad|>", false);
        if (pad.size() == 1 && pad[0] > 0) image_pad_token = pad[0];
    } catch(...) {}
    fprintf(stderr, "[vision] injecting %zu embedding token(s), floats=%zu hidden=%d image_pad=%d at ctx_pos=%d\n",
            n_tok, tokens.size(), hidden, (int)image_pad_token, impl_->ctx_pos);
    const int start = impl_->ctx_pos;
    if (impl_->prefill_token_by_token) {
        for (int i = 0; i < (int)n_tok; ++i) {
            const float* e = tokens.data() + (size_t)i * (size_t)hidden;
            if (i + 1 == (int)n_tok)
                impl_->transformer.forward_embed(e, start + i);
            else
                impl_->transformer.forward_embed_prefill(e, start + i);
        }
    } else if (impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream()) {
        impl_->transformer.prefill_embeddings_stream(tokens.data(), hidden, start, (int)n_tok);
    } else {
        if (impl_->turbo_prefill_enabled && !impl_->transformer.turbo_gpu_stream()) {
            fprintf(stderr,
                "[vision] turbo requested but GPU stream is OFF; using CPU token-major embeddings\n");
        }
        impl_->transformer.prefill_embeddings_cpu_prompt_batch(tokens.data(), hidden, start, (int)n_tok);
    }
    impl_->ctx_pos += (int)n_tok;
    impl_->session_tokens.insert(impl_->session_tokens.end(), n_tok, image_pad_token);
    const char* vis_mode = impl_->prefill_token_by_token
        ? "token-by-token"
        : ((impl_->turbo_prefill_enabled && impl_->transformer.turbo_gpu_stream()) ? "turbo-gpu-stream" : "cpu-token-major");
    fprintf(stderr, "[vision] injected %zu embedding token(s) via %s; ctx_pos=%d session_tokens=%zu\n",
            n_tok, vis_mode, impl_->ctx_pos, impl_->session_tokens.size());
}
bool Engine::load_audio_encoder(const std::string& path,
    std::function<void(const std::string&,float)> cb) {
    impl_->audio_enc = std::make_unique<nexus::audio::AudioEncoder>();
    return impl_->audio_enc->load(path, cb);
}
bool Engine::has_audio() const {
    return impl_->audio_enc && impl_->audio_enc->loaded();
}
std::vector<float> Engine::encode_audio(const std::vector<float>& pcm) {
    if (!has_audio()) return {};
    return impl_->audio_enc->encode(pcm);
}
std::vector<float> Engine::encode_audio_wav(const std::string& path) {
    if (!has_audio()) return {};
    return impl_->audio_enc->encode_wav(path);
}
void Engine::inject_audio_tokens(const std::vector<float>& tokens) {
    int d = has_audio() ? impl_->audio_enc->config().n_audio_state : 512;
    fprintf(stderr, "[audio] inject %zu floats (%d frames) at pos %d\n",
            tokens.size(), (int)tokens.size()/d, impl_->ctx_pos);
    (void)tokens;
}


void Engine::sync_session_tokens(const std::vector<int32_t>& toks, int pos) {
    impl_->session_tokens = toks;
    impl_->ctx_pos        = pos;
    log_state("after reply");
}

} // namespace nexus
