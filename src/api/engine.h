/*
 * nexus/src/api/engine.h
 *
 * Public C++ API for the Nexus inference engine.
 * This is what external code (Python bindings, tools, other C++ programs)
 * talks to. It wraps Transformer + Tokenizer behind a clean interface
 * and hides all the internal headers.
 *
 * Everything below this layer is implementation detail.
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <cstdint>
#include "chat_template.h"   // defines nexus::ChatMessage, nexus::ChatTemplate, nexus::ChatFormat
// Vision and audio encoders --- included here so Engine can hold them
#include "../model/vision_encoder.h"
#include "../model/audio_encoder.h"


namespace nexus {

// --------- Generation settings ------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct GenerateConfig {
    // ------ Generation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int   max_tokens         = 512;
    bool  echo_prompt        = false;       // include prompt tokens in output

    // ------ Sampling ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    float temperature        = 0.7f;
    float top_p              = 0.95f;       // nucleus sampling (1.0 = off)
    int   top_k              = 40;          // 0 = disabled
    float min_p              = 0.05f;       // min-p (0.0 = off) --- improves coherence
    float tfs_z              = 1.0f;        // tail-free sampling (1.0 = off)

    // ------ Repetition ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    float repeat_penalty     = 1.1f;        // multiplicative, 1.0 = off
    int   repeat_last_n      = 64;          // context window for repeat penalty
    float frequency_penalty  = 0.0f;        // per-occurrence additive penalty
    float presence_penalty   = 0.0f;        // flat additive if token seen at all

    // ------ Misc ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    int   seed               = -1;          // -1 = random
    int   n_gpu_layers       = 0;           // 0 = CPU only
    std::vector<std::string> stop_strings;

    // ------ MTP (multi-token prediction) ---------------------------------------
    bool  mtp_enabled        = false;       // off by default (accuracy-first)
    int   mtp_n_predict      = 1;           // tokens MTP proposes per step
    bool  mtp_gpu            = false;       // run MTP layer on GPU if available (>=1)
};

// --------- Model metadata (no internal types exposed) ---------------------------------------------------------------------------------------------

struct ModelInfo {
    std::string name;
    std::string arch;
    std::string chat_format;    // "ChatML", "Mistral [INST]", "Llama-3", etc.
    uint32_t    n_layers;
    uint32_t    n_heads;
    uint32_t    n_kv_heads;
    uint32_t    head_dim;
    uint32_t    hidden_dim;
    uint32_t    ffn_dim;
    uint32_t    vocab_size;
    uint32_t    context_length;
    float       rope_base;
    size_t      file_size_bytes;
};

// --------- Generation result ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct GenerateResult {
    std::string text;            // full generated text
    int         n_tokens;        // tokens generated
    double      elapsed_sec;
    double      tokens_per_sec;
    bool        stopped_eos;     // stopped on EOS token
    bool        stopped_string;  // stopped on stop string
    bool        stopped_limit;   // stopped on max_tokens
};

// --------- Load progress callback ---------------------------------------------------------------------------------------------------------------------------------------------------------

struct LoadProgress {
    std::string stage;   // "Opening model", "Loading weights", "Ready"
    float       fraction; // 0.0 --- 1.0
    size_t      bytes_loaded;
    size_t      bytes_total;
};

// --------- Pausing generation session ------------------------------------------------------------------------------------------------------------------------------------------
//
// Pauses mid-generation when a complete tool/call tag appears in the output.
// KV cache is preserved --- inject() only processes NEW tokens, never re-runs
// the prompt. This is how ChatGPT-style tool use works: one continuous KV cache.
//
// FLOW:
//   session = engine.begin_session(prompt, cfg)
//   while true:
//     chunk = session.run(stream_cb)   <- generates until pause or done
//     if chunk.paused:
//       result = execute(chunk.tag_name, chunk.tag_input)
//       session.inject("<result>\n" + result + "\n</result>\n")
//     elif chunk.done: break

struct SessionChunk {
    std::string text;          // text generated in this run() call
    bool        done   = false;// true = generation finished
    bool        paused = false;// true = paused on a tag boundary
    std::string tag_name;      // e.g. "nexus:tool" (when paused)
    std::string tag_input;     // inner content of the paused tag
    std::string tag_content;   // full tag text including open/close
};

class GenerateSession {
public:
    ~GenerateSession();
    GenerateSession(GenerateSession&&) noexcept;
    GenerateSession& operator=(GenerateSession&&) noexcept;

    // Generate until done or a pause tag is hit.
    // token_cb: optional, called for each streamed token (return false to stop)
    SessionChunk run(
        std::function<bool(const std::string& tok, int32_t id)> token_cb = nullptr);

    // Inject text after a pause --- tokenize, prefill new tokens only, ready to run()
    void inject(const std::string& text);

    // Inject already-projected multimodal embeddings into the live KV stream.
    // embeddings is [n_tokens * hidden_dim]. placeholder_token is kept only for
    // repeat-penalty/session bookkeeping; the KV entries use the embeddings.
    void inject_embeddings(const std::vector<float>& embeddings,
                           int hidden_dim,
                           int32_t placeholder_token = -200);

    bool   done()   const;
    int    pos()    const;
    int    generated() const;   // tokens produced by this session so far (exact; MTP-accepted drafts count too)
    const std::vector<int32_t>& tokens() const;

private:
    friend class Engine;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit GenerateSession(std::unique_ptr<Impl>);
};

// --------- Engine ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Engine {
public:
    Engine();
    ~Engine();

    // Non-copyable
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // ------ Load ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Load a GGUF model file.
    // progress_cb: optional, called periodically during load
    // Returns true on success.
    bool load(const std::string& model_path,
              int n_threads = 1,
              std::function<void(const LoadProgress&)> progress_cb = nullptr);

    // Unload model and free all memory
    void unload();

    bool is_loaded() const;

    // Enable GPU acceleration for the first n_layers transformer blocks.
    // Call after load(). n=0 disables GPU. n=-1 = all layers.
    // Returns the number of layers actually offloaded (0 if GPU unavailable).
    int set_gpu_layers(int n);

    // Secondary GPU (iGPU / second device).
    // first_layer = layers already on primary GPU; assigns layers [first_layer, first_layer+n).
    // Returns actual layers assigned, 0 if failed.
    int set_gpu2_layers(int n, int device_index, int first_layer);
    // Third, fourth, ... graphics card (call once per card, in order, after set_gpu2_layers).
    int add_gpu_layers(int n, int device_index, int first_layer);
    // For "Maximize" on cards 2, 3, ... (n < 0 above): card memory to leave free.
    void set_card_fill_keep_mb(int mb);
    void set_mtp_card_reserve(bool on);   // Maximize GPU leaves room on the prompt card for MTP drafting there
    std::string gpux_device_name(int i) const;   // i = 0 for the third card

    // Pin GPU to a specific device index (from list_gpu_devices).
    // Call BEFORE set_gpu_layers(). -1 = auto (prefer discrete GPU).
    void set_gpu_device(int index);
    
    // Bring up the Vulkan backend without committing standing GPU layers.
    // Needed for turbo streaming when n_gpu_layers == 0 (decode on CPU,
    // GPU used only for prefill bursts). Pass device_index = -1 for auto.
    bool ensure_gpu_ready(int device_index = -1);

    // List all Vulkan-capable devices without initializing inference.
    // Returns strings like "[0] GeForce GTX 1060 3GB (discrete, 3072 MB)"
    static std::vector<std::string> list_gpu_devices();

    // GPU device info (empty if no GPU or not compiled with Vulkan)
    std::string gpu_device_name()  const;
    std::string gpu2_device_name() const;
    bool        gpu_zero_copy()    const;  // true = APU unified memory (no weight copy)

    // ------ Distributed / layer-streaming inference ------------------------------------------------------------------------------------
    // These expose the hidden state (x_scratch_) and a layer-range forward pass.
    // Used by dist_session.h to split the model across machines or load layers
    // sequentially from disk on a single RAM-constrained machine.

    // Read the current hidden state after forward_layer_range() --- returns pointer
    // to internal x_scratch_ buffer [hidden_dim floats]. Valid until next forward call.
    const float* hidden_data() const;

    // Overwrite x_scratch_ to resume from a saved hidden state.
    // Machine B calls this before running its layer chunk.
    void set_hidden_data(const float* h, size_t n_floats);

    // Run layers [layer_start, layer_end) of the forward pass.
    //   layer_start == 0        --- also does embedding lookup from tokens[pos]
    //   layer_end   == n_layers --- also does final RMSNorm + lm_head --- fills logits
    // KV cache is written normally for each layer processed.
    void forward_layer_range(const std::vector<int32_t>& tokens, int pos,
                             int layer_start, int layer_end);

    // Batched version of the above for prompt reading (see Transformer::prefill_layer_range).
    // keep_prefill_scratch: true if the caller will make another prefill_layer_range()
    // call on this same prompt right after this one (e.g. hybrid prefill streaming
    // one donor layer at a time) -- skips freeing/regrowing the prompt-sized scratch
    // buffers between calls, which otherwise turns a 32-layer streamed range into 32x
    // the teardown/regrow cost of doing it as one batched call. Default false keeps
    // every existing single-call caller's behavior unchanged.
    bool prefill_layer_range(const std::vector<int32_t>& tokens, int start_pos, int n,
                             int layer_start, int layer_end, const float* x_in, float* x_out,
                             bool keep_prefill_scratch = false);
    bool range_prefill_ok() const;
    int  range_prefill_layers_done() const;
    // Prompt-reading progress, 0..1 (-1 = nothing reported yet), for any prefill path. Call
    // prefill_progress_begin(true) before a prefill whose progress you print yourself (the engine
    // then stays quiet), prefill_progress_begin(false) after.
    double prefill_fraction() const;
    void   prefill_progress_begin(bool external_report);
    void set_range_prefill_quiet(bool q);

    // Called by DistSession on primary to stream layer weights to worker.
    // Writes a valid GGUF v3 shard stream with token_embd, blk.L.* for L in
    // [layer_start, layer_end), output_norm, and output.weight.
    //
    // load_range() loads ONLY layers [layer_start, layer_end) from the source
    // GGUF directly into owned model memory. It must not build a mini-GGUF,
    // memfd, temp file, or full in-RAM shard first.
    //
    // load_from_memory() still exists for streamed GGUF shards and compatibility.
    bool load_from_memory(const void* data, size_t size, int n_threads,
                          std::function<void(const LoadProgress&)> progress_cb = nullptr);
                          
    void init_model_path(const std::string& model_path);

    bool load_range(const std::string& path, int layer_start, int layer_end,
                    int n_threads,
                    std::function<void(const LoadProgress&)> progress_cb = nullptr);

    // True single-layer streaming pair, for hybrid prefill's "ghost" donor Engine:
    // open the model ONCE (file header, global tensors, tokenizer -- no layers yet),
    // then call swap_layer() as many times as needed; each call frees whichever
    // layer was previously streamed in and loads just the requested one. Never
    // reopens the file or reloads the tokenizer after the first call. Use this
    // instead of repeated load_range(start, start+1) calls, which reload the whole
    // model's global tensors + tokenizer every single time.
    // probe_layer_start/probe_layer_end: the layer range this ghost will actually
    // stream. Used to probe each layer's real KV geometry once before the cache is
    // sized -- see Transformer::open_for_layer_streaming()'s comment. -1 (default)
    // probes the whole model.
    bool open_for_layer_streaming(const std::string& path, int n_threads,
                                  int probe_layer_start = 0, int probe_layer_end = -1);
    bool swap_layer(int new_layer);

    // *** v29: for THIS engine's own, already-loaded, persistent layer range (not a
    // layer-streaming ghost) -- temporarily frees one layer's weights to make room for
    // something else (hybrid prefill running the donor's layers on this computer's own
    // GPU), then brings it back. KV cache and every other layer are untouched either
    // way. The caller must reload every layer it frees before using it again.
    // Layer parking ("Maximize GPU" / oversized-prompt protection). Call once after loading, with the prompt
    // speed-up already on. See Transformer::park_setup for the parameters. Returns false (and loads normally) when
    // it can't run here; the log says why.
    bool park_setup(int park_layers, int below_mb, int keep_mb, int prefill_free_mb, int min_tokens);
    bool park_active() const;
    // While parked layers are being moved onto the card after a prompt: how many of how many.
    bool park_moving(int& done, int& total) const;
    bool free_own_layer(int layer);
    bool reload_own_layer(int layer);

    // KV cache buffer transfer for distributed inference save/restore.
    // Pure RAM -- no disk I/O on either side.
    bool save_kv_to_buffer(std::vector<uint8_t>& out) const;
    bool load_kv_from_buffer(const uint8_t* data, size_t size);

    // Per-layer siblings of the two above, for hybrid prefill: stream one layer's KV (and its
    // recurrent state, if the architecture has one) as soon as it finishes, instead of waiting for
    // every layer in a range. An empty `out` from the save side means this layer genuinely has
    // nothing to send and the caller should skip it, not an error.
    bool save_kv_layer_to_buffer(int layer, std::vector<uint8_t>& out) const;
    bool load_kv_layer_from_buffer(int layer, const uint8_t* data, size_t size);
    // Frees one layer's KV memory. Call after save_kv_layer_to_buffer(layer) has been sent off and
    // is no longer needed on THIS engine -- this is what makes streaming layer-by-layer actually
    // save memory, not just defer the cost to the end.
    void free_kv_layer(int layer);

    // Set model path without loading weights. Call before connect()
    // so the worker shard can be streamed before the primary loads.
    // Streaming version for sending worker shard -- never allocates a large blob.
    // Wire format: [uint64 header_size][header-only GGUF][tensor bytes sorted by remapped tensor name].
    bool send_layer_subset_gguf(int layer_start, int layer_end,
        std::function<bool(const void*, size_t)> write_fn) const;

    // Receiver-side stream loader for remote shards. It holds only the small header
    // plus each final tensor allocation; it never assembles a fake GGUF blob.
    bool load_streamed_shard(const void* gguf_header, size_t gguf_header_size,
        int n_threads, std::function<bool(void*, size_t)> read_fn,
        std::function<void(const LoadProgress&)> progress_cb = nullptr);

    bool write_layer_subset_gguf(std::vector<uint8_t>& out,
                                  int layer_start, int layer_end) const;

    // ------ Info ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ModelInfo model_info() const;
    size_t    memory_used_bytes() const;
    void      print_memory_report() const;

    // ------ Chat template ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Format conversation history into the raw prompt string for this model
    // think: -1 = as configured, 1 = reason first (this turn), 0 = don't. Ignored by
    // formats without a thinking switch (see supports_think_control()).
    std::string format_chat(const std::vector<nexus::ChatMessage>& messages, int think = -1) const;
    std::string format_system_prefix(const std::vector<nexus::ChatMessage>& messages) const;
    std::string format_user_turn(const std::string& user_text, int think = -1) const;
    // True when the loaded model's chat format has a per-turn thinking switch.
    bool supports_think_control() const;

    // Stop strings for this format (e.g. "<|im_end|>", "</s>")
    std::vector<std::string> stop_strings() const;

    // Recommended default system prompt (empty = none needed)
    std::string default_system_prompt() const;

    // Whether BOS token should be prepended to the prompt (format-dependent)
    bool should_add_bos() const;

    // Override auto-detected chat format (e.g. from --chat-format CLI arg)
    // fmt: "ChatML", "Mistral", "Llama-3", "DeepSeek", "DeepSeek-Coder", "Phi-3", ...
    void override_chat_format(const std::string& fmt);

    // Set a launcher-defined custom chat template.
    void set_custom_chat_template(const nexus::ChatTemplate::CustomTemplateParts& parts);

    // ------ Tokenize ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    std::vector<int32_t> tokenize(const std::string& text,
                                   bool add_bos = true) const;

    std::string detokenize(const std::vector<int32_t>& tokens) const;
    std::string detokenize(int32_t token_id) const;

    int32_t bos_token() const;
    int32_t eos_token() const;
    size_t  vocab_size() const;

    // ------ Generate ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Blocking: generate full response string
    GenerateResult generate(const std::string& prompt,
                             const GenerateConfig& cfg = {});

    // Streaming: callback called once per token as it's generated.
    // token_cb(token_string, token_id) --- return false to stop early
    GenerateResult generate_stream(
        const std::string& prompt,
        std::function<bool(const std::string& token, int32_t token_id)> token_cb,
        const GenerateConfig& cfg = {}
    );

    // Low-level: run one forward step
    // tokens: full token sequence so far
    // pos:    position to compute (usually tokens.size()-1)
    int32_t forward(const std::vector<int32_t>& tokens, int pos);

    // Sample from last forward's logits
    int32_t sample(float temperature, float top_p, int top_k);

    // Get top-N (token_id, logit_value) pairs from last forward pass
    std::vector<std::pair<int32_t,float>> top_logits(int n = 10) const;

    // Full logits from last forward pass. Used by distributed primary when the
    // worker owns final layers/lm_head and sends logits back for local sampling.
    const std::vector<float>& logits() const;

    // ------ Context ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // Clear KV cache --- start fresh conversation
    void reset_context();
    
    void set_turbo_prefill(bool enabled);
    bool turbo_prefill_enabled() const;

    // Debug/old path: prompt prefill uses forward() once per token.
    // Default is false; normal CPU prefill is full-prompt batch.
    void set_prefill_token_by_token(bool enabled);
    bool prefill_token_by_token() const;
    
    void set_turbo_gpu_device(int index);
    int  turbo_gpu_device() const;

    // Turbo layer-major prefill entry point. Use only when Turbo Prefill is ON.
    void prefill_batch(const std::vector<int32_t>& tokens, int start, int n);

    // Normal CPU prompt prefill: one batched prompt prefill call, no turbo GPU
    // stream, no resident-window layer swapping, then output remains token decode.
    void prefill_cpu_prompt_batch(const std::vector<int32_t>& tokens, int start, int n);
    
    // Turbo GPU streaming: during prefill, route all layers through GPU and
    // stream weights via the bounded cache. Only active when turbo_prefill
    // is also enabled.
    void   set_turbo_gpu_stream(bool enabled);
    bool   turbo_gpu_stream() const;

    // VRAM budget for the weight cache during turbo streaming. 0 = unlimited
    // (will OOM on small GPUs). Recommended: leave ~500 MB headroom for
    // activations and the OS, then divide remaining VRAM by 2 (so two layers
    // fit at once for future overlap).
    void   set_turbo_vram_budget_mb(int mb);
    int    turbo_vram_budget_mb() const;

    // RoPE position stabilizer. Non-destructive: keeps all KV, rewrites cached K
    // positions into a compressed RoPE coordinate space when enabled.
    void  set_rope_stabilizer(bool enabled, int trigger_effective_tokens, int target_effective_tokens);
    bool  rope_stabilizer_enabled() const;
    int   rope_stabilizer_trigger() const;
    int   rope_stabilizer_target() const;
    float rope_position_scale() const;

    // How many tokens are currently in context
    int context_size() const;
    // Diagnostic: one log line with the context length, the model-state fingerprint and the
    // last tokens in the context. Used around save / load / extend / reply.
    void log_state(const char* tag) const;

    // Max context length the model supports
    int max_context() const;

    // ------ Section markers ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Mark the start/end of a named logical block in the token stream.
    // Call before and after each inject() for KB blocks, tool results, etc.
    // These positions are used by context_trim() to evict specific ranges
    // without touching the pinned prefix or recent turns.
    //
    // Typical usage:
    //   engine.context_mark_start("kb_block_1");
    //   engine.extend_context(kb_text);
    //   engine.context_mark_end("kb_block_1");
    //   engine.context_pin_prefix();  // call once after system prompt prefill
    //
    void context_mark_start(const std::string& name);
    void context_mark_end(const std::string& name);
    void context_pin(const std::string& name);  // mark a section as never-evictable

    // Pin all tokens up to current position as the system/tool banner.
    // Call once after prefilling system prompt + tool definitions.
    // These tokens are never evicted --- they stay at position 0..pinned_tokens.
    void context_pin_prefix();

    // Evict sections to reach target_tokens.
    // Oldest unpinned sections are evicted first. No reprocessing needed ---
    // the KV cache is surgically trimmed via memmove.
    // Returns number of tokens freed.
    int context_trim(int target_tokens);

    // Called automatically at context limit --- trims to 75% capacity.
    // Generation continues without stopping cold.
    // Returns tokens freed (0 if nothing could be freed safely).
    int context_auto_trim();

    // ------ Editing the conversation memory (remove / archive / restore) ------------
    // The context is split into turns with the model's OWN chat template: every turn starts at
    // the template's turn marker (<|im_start|>, <|start_header_id|>, <start_of_turn>, [INST],
    // <|user|>/<|assistant|> ...), so removing whole turns always leaves a well-formed
    // conversation, whatever the model. Trimming and removal never touch the protected start
    // (everything before the first turn, the system prompt, and anything pinned).
    // Removing part of the middle shifts the keys of everything after it back into place (RoPE),
    // so the kept text is stored exactly as if the removed part had never been there -- exact
    // for attention layers. Models with recurrent layers (Qwen3-Next / Qwen3.5) also keep a
    // running summary that can't drop individual tokens: see context_edits_exact().
    struct ContextTurn {
        int start = 0, end = 0;       // token range [start, end)
        std::string role;             // "banner" (before the first turn), "system", "user", "assistant",
                                      // "tool", "exchange" (question+answer stored as one block,
                                      // e.g. [INST] formats) or "turn" (role unknown)
        bool protected_part = false;  // part of the protected start -- never trimmed or removed
        std::string preview;          // first words of the text
    };
    std::vector<ContextTurn> context_turns() const;
    int  context_protected_tokens() const;
    // Protect at least the first n tokens (rounded up to whole turns). The protected start is
    // also found automatically (leading system turns); this is for templates that merge the
    // system prompt into the first user turn.
    void context_pin_tokens(int n);
    bool context_remove(int start, int end, std::string* err = nullptr);
    // Save [start, end) (tokens + KV) to a file, then remove it from the context.
    bool context_archive(int start, int end, const std::string& path, std::string* err = nullptr);
    // Append an archived part at the END of the current context. Returns tokens added, -1 on error.
    int  context_restore(const std::string& path, std::string* err = nullptr);
    bool context_edits_exact() const;
    // Conversation length limit. 0 = the model's own maximum. context_limit() is the limit in
    // force (never above the model maximum).
    void set_context_limit(int tokens);
    int  context_limit() const;
    // When the limit is reached: drop the oldest turns and carry on (true, default) or stop (false).
    void set_auto_trim(bool on);
    bool auto_trim() const;
    // Drop oldest turns (if auto-trim is on) so that incoming + reserve tokens fit under the limit.
    // Returns tokens dropped.
    int  context_make_room(int incoming_tokens, int reserve_tokens);
    size_t kv_bytes_per_token() const;        // K+V bytes one token takes, all layers
    size_t kv_capacity_tokens() const;        // tokens that fit before the KV buffers must grow
    const std::vector<int32_t>& context_tokens() const;

    // ------ Multi-session context slots ---------------------------------------------------------------------------------------------------------------------
    // Each slot has its own KV cache --- allows multiple concurrent conversations
    // on the same loaded model without reloading weights.
    // Slot 0 is the default (all existing API uses slot 0).
    void   context_slot_select(int slot_id);   // switch active slot
    void   context_slot_clear(int slot_id);    // clear one slot's cache
    int    context_slot_size(int slot_id) const;
    int    context_slot_count() const;         // how many slots exist

    // Sync engine session_tokens from a completed GenerateSession.
    // Call after sess.run() so extend_context starts from the right position.
    void sync_session_tokens(const std::vector<int32_t>& toks, int pos);

    // ------ KV cache persistence ---------------------------------------------------------------------------------------------------------------------------
    // Save current KV cache + token list to file.
    // Returns false if nothing in context or write fails.
    bool save_session(const std::string& path) const;

    // Load KV cache + token list from file.
    // After load, the model resumes exactly from where save_session left off.
    // Returns false on model mismatch or missing file.
    bool load_session(const std::string& path);

    // Extend an existing session: tokenize new_text, prefill only the new
    // tokens (KV cache already warm), ready for generate/begin_session.
    // Call after load_session() or after generate() to continue a conversation.
    void extend_context(const std::string& new_text, bool add_bos = false);

    // Extend context with a single mixed text+embedding prefill. tokens are the
    // bookkeeping/session token IDs; embed_mask[i] means embedded_inputs at
    // i*hidden_dim overrides the normal token embedding at that position. This
    // is for vision/audio turns so the main model sees one ordered body instead
    // of separate text-prefill, image-prefill, text-prefill passes.
    bool extend_context_mixed_embeddings(const std::vector<int32_t>& tokens,
                                         const std::vector<uint8_t>& embed_mask,
                                         const std::vector<float>& embedded_inputs);

    // ------ Text embeddings (for RAG / semantic search) ------------------------------------------------------------------------------
    // Run the model on text and return the last hidden state.
    // This is the sentence embedding --- use for similarity search, clustering, RAG.
    // Returns [hidden_dim] floats (normalized). Empty on failure.
    std::vector<float> embed_text(const std::string& text, bool normalize = true);

    // ------ LoRA adapters ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    bool lora_load(const std::string& path, float scale = 1.0f);
    void lora_unload_all();
    void lora_set_scale(int adapter_idx, float scale);
    int  lora_count() const;

    // ------ Speculative decoding ------------------------------------------------------------------------------------------------------------------------------------------------------
    // Use a second (smaller/faster) Engine as a draft model.
    // Generates tokens ~2-4x faster when draft and target are from same model family.
    // draft_engine: must have the same tokenizer and similar distribution
    // n_draft: tokens to speculate per step (4-8 is typical)
    GenerateSession begin_session_speculative(
        const std::string&   prompt,
        Engine*              draft_engine,
        int                  n_draft,
        const GenerateConfig& cfg = {});

    // ------ Multimodal: vision ------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Load a vision encoder. Usually automatic for .nxm model.visual.* tensors;
    // this remains for legacy external CLIP/SigLIP mmproj GGUF files.
    bool load_vision_encoder(const std::string& mmproj_path,
                             std::function<void(const std::string&,float)> cb = nullptr);
    bool has_vision() const;
    // Encode RGB image [h*w*3] --- vision token embeddings [n_tok * d_model]
    std::vector<float> encode_image(const uint8_t* rgb, int w, int h);
    // Inject encoded vision tokens into context before generation
    void inject_vision_tokens(const std::vector<float>& tokens);

    // ------ Multimodal: audio ------------------------------------------------------------------------------------------------------------------------------------------------------
    // Load a Whisper-compatible audio encoder from a GGUF file.
    bool load_audio_encoder(const std::string& gguf_path,
                            std::function<void(const std::string&,float)> cb = nullptr);
    bool has_audio() const;
    // Encode mono 16kHz PCM --- audio token embeddings
    std::vector<float> encode_audio(const std::vector<float>& pcm_16k);
    // Encode a WAV file (auto-resamples to 16kHz)
    std::vector<float> encode_audio_wav(const std::string& wav_path);
    // Inject encoded audio tokens into context before generation
    void inject_audio_tokens(const std::vector<float>& tokens);

    // ------ KV cache checkpoints ---------------------------------------------------------------------------------------------------------------------------------------
    // Save a named checkpoint of the current KV cache state.
    // Returns checkpoint ID (monotonically increasing).
    // These are in-memory; use save_session() to persist to disk.
    int   checkpoint_save(const std::string& name = "");

    // Restore to a previously saved checkpoint.
    // After restore, the model generates as if that was the end of the conversation.
    // Returns false if id not found.
    bool  checkpoint_restore(int id);

    // List all checkpoints: {id, name, token_count}
    struct CheckpointInfo { int id; std::string name; int n_tokens; };
    std::vector<CheckpointInfo> checkpoint_list() const;

    // Delete a checkpoint to free memory
    void  checkpoint_delete(int id);

    // Save session state WITHOUT conversation history metadata ---
    // just the KV cache + position. When loaded, the model continues
    // from that point with no prior context shown to client.
    // Use for sub-model start states and pre-primed states.
    bool  save_start_state(const std::string& path) const;
    bool  load_start_state(const std::string& path);

    // ------ Pausing session ---------------------------------------------------------------------------------------------------------------------------------------------------------
    GenerateSession begin_session(
        const std::string& prompt,
        const GenerateConfig& cfg,
        std::vector<std::string> pause_tags = {"nexus:tool","nexus:call"});

    // Like begin_session but skips reset_context and prefill.
    // Use after extend_context() to generate from the warm cache.
    GenerateSession begin_session_warm(
        const GenerateConfig& cfg,
        std::vector<std::string> pause_tags = {"nexus:tool","nexus:call"});

    GenerateSession begin_session_chat(
        const std::vector<nexus::ChatMessage>& messages,
        const GenerateConfig& cfg,
        std::vector<std::string> pause_tags = {"nexus:tool","nexus:call"});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool ctx_consistent_(std::string* err);   // token list, position and KV agree (needed before editing)
};

} // namespace nexus
