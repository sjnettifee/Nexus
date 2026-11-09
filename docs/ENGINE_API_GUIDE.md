# Nexus Engine & Server API Guide

This guide is for anyone who wants to skip the launcher entirely: run the
Nexus server headless from the command line, drive it from your own
front end (web, mobile, a script, another chat UI) over HTTP, or link
against the `Engine` C++ class directly and skip the HTTP layer too.

It assumes you already have a working Nexus build (the `nexus_server`
binary, or your own program linked against `engine.h`/`engine.cpp`) and a
GGUF or native `.nxm`/`.nxmodel` model. It does not cover building Nexus
from source or converting models — see the Quant Builder guide for that.

Everything here is read directly from `tools/server.cpp`,
`src/api/engine.h`, and `src/api/engine.cpp`. Where the server and the
engine disagree slightly in naming, both names are given.

---

## 1. Two ways to use Nexus without the launcher

**A. Run the server, talk HTTP.** This is almost always the right choice
if you're building a front end, a script, or integrating with another
tool. You get streaming generation over SSE, session persistence,
vision/audio, LoRA, checkpoints, and the distributed (donor/primary)
mode, all through plain HTTP + JSON. No linking, no C++. Section 3.

**B. Link against `Engine` directly.** For embedding Nexus inside your
own C++ application with no HTTP server in the loop at all — a game, a
native app, a different server framework. You write the event loop
yourself. Section 4.

The server itself is just a thin HTTP wrapper around the exact same
`Engine`/`GenerateSession` classes described in Section 4 — reading
Section 4 after Section 3 will make clear how every HTTP route maps to
an engine call.

---

## 2. Running the server headless

```
nexus_server --model /path/to/model.gguf --port 8080
```

Default bind is `127.0.0.1:8080`. Nothing else is required — the server
loads the model, starts listening, and prints
`http://localhost:8080` once ready.

### CLI flags (from the server's own `--help`)

**Core**
| Flag | Meaning |
|---|---|
| `--model <path>` | Model file (GGUF or native `.nxm`/`.nxmodel`) |
| `--port <n>` | Listen port (default 8080) |
| `--host <addr>` | Bind address (default 127.0.0.1) |
| `--data <dir>` | Storage directory (sessions, checkpoints, projects, etc.) |
| `--threads <n>` | CPU threads |
| `--config <path>` | Load a `nexus.conf`-style config file |

If a `nexus.json` file (the format the launcher writes) exists in the
working directory, the server also reads `model_path`/`host`/`port`/
`threads` from it — but only for whichever of those you did **not** pass
on the command line. CLI flags always win over `nexus.json`, and
`nexus.json` always wins over nothing. This lets you drop a server
binary next to a launcher-written `nexus.json` and it just works, while
still letting `--port 8081` override it for a second instance.

**Chat formatting** (only needed if auto-detection from the GGUF's chat
template guesses wrong, or the model has no template at all)
| Flag | Meaning |
|---|---|
| `--chat-format <fmt>` | Force a format: `ChatML`, `Qwen3`, `Mistral`, `Mistral V3`, `Llama-3`, `DeepSeek`, `DeepSeek-R1`, `DeepSeek-Coder`, `Phi-3`, `Phi-4`, `Gemma`, `Gemma4`, `Zephyr`, `Alpaca`, `Granite`, `OLMoE`, `Custom`, `Raw` |
| `--chat-custom-pre-prompt/-pre-system/-end-system/-pre-user/-end-user/-pre-model/-end-model <s>` | Build a one-off format out of literal fragments (only used with `--chat-format Custom`) |

**GPU placement**
| Flag | Meaning |
|---|---|
| `--gpu-layers <n>` | Layers to put on the primary GPU |
| `--gpu-device <idx or name:...>` | Which Vulkan device; accepts a raw index or `name:<card substring>[#n]` resolved against the device list |
| `--gpu2-layers` / `--gpu2-device` | Second GPU |
| `--gpu3-layers` / `--gpu3-device`, `--gpuN-layers` / `--gpuN-device` | Any further GPUs (N ≥ 3) |
| `--list-devices` | Print Vulkan devices and exit |
| `--probe` | Print a hardware/GPU JSON probe and exit — no model or license needed |

**Second model (side-by-side)**
| Flag | Meaning |
|---|---|
| `--model2 <path>` | Load a second model alongside the first |
| `--model2-gpu-layers/-device/-threads/-chat-format` | Same knobs, scoped to model 2 |

Model 2 is addressed in every chat/session API call by passing
`"model":"2"` in the JSON body (default is `"1"`). It has its own
generation mutex, its own context, its own everything — the two models
never block each other.

**Sampling defaults** (overridable per-request in `/api/chat`)
`--temperature`, `--top-p`, `--rep-penalty`, `--min-p`, `--tfs-z`,
`--top-k`, `--max-tokens`, `--repeat-last-n`, `--freq-penalty`,
`--presence-penalty`

**Multimodal / adapters**
| Flag | Meaning |
|---|---|
| `--vision-encoder <path>` | Load a vision (mmproj) encoder at startup |
| `--audio-encoder <path>` | Load an audio (Whisper-style) encoder at startup |
| `--start-state <path>` | Load a saved start-state (warm KV) at startup |
| `--lora <path>` + `--lora-scale <f>` | Load a LoRA adapter at startup |
| `--roster-file <path>` | Model roster file for the launcher's model picker |

**Prefill acceleration**
| Flag | Meaning |
|---|---|
| `--turbo-prefill` | Enable the accelerated prefill path |
| `--turbo-gpu-stream` | Stream prefill through the GPU layer-by-layer (the "hybrid" path) |
| `--prefill-token-by-token` | Disable turbo, force the plain token-at-a-time debug path |
| `--turbo-vram-budget-mb <n>` | VRAM ceiling for the turbo path |

**Speculative decoding**
`--mtp on|off`, `--mtp-n-predict <n>`, `--mtp-gpu on|off`

**Diagnostics**
`--trace-level <0-3>` — sets `NEXUS_TRACE`

**Distributed (donor/primary) mode** — see Section 5.
`--dist-worker`, `--dist-local-model <path>`, `--dist-passphrase <word>`,
`--dist-layer-start/-end <n>`, `--dist-peer <host:port>`,
`--dist-peer-layers <start:end>`, `--dist-peer-local-load`,
`--dist-required`

### Environment variables worth knowing about

These aren't CLI flags, just env vars the server itself reads:

| Variable | Effect |
|---|---|
| `NEXUS_TRACE` | Trace verbosity (same thing `--trace-level` sets) |
| `NEXUS_HYBRID_PREFILL` | Toggles the hybrid prefill path server-side |
| `NEXUS_VISION_MODE` | Overrides vision-encoder behavior |
| `NEXUS_WEB_IMAGE_MAX_EDGE` | Max image edge (pixels) the server will accept via `/api/image/upload` before downscaling |
| `NEXUS_WEB_IMAGE_MAX_PIXELS` | Max total pixel count, same purpose |

---

## 3. The HTTP API

The server is a hand-rolled HTTP/SSE server with zero external
dependencies — any HTTP client works (`curl`, `fetch`, `requests`,
whatever). All request bodies are JSON; all responses are JSON except
`/api/chat`, `/api/continue`, `/api/continue/resume`, and
`/api/session/extend_turn`, which stream.

Every endpoint below that touches "the model" takes an optional
`"model":"1"` or `"model":"2"` field to pick between the primary model
and the `--model2` model. It defaults to `"1"` and most endpoints simply
omit it in the examples below for brevity.

### 3.1 The core generation endpoint: `POST /api/chat`

This is the one endpoint that actually generates text. Everything else
is support machinery around it.

**Request body**
```json
{
  "model": "1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "images": [
    {"name": "photo.jpg", "width": 640, "height": 480, "rgb_b64": "..."}
  ],
  "temperature": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.0, "tfs_z": 1.0,
  "repeat_penalty": 1.1, "repeat_last_n": 64,
  "frequency_penalty": 0.0, "presence_penalty": 0.0,
  "mtp_enabled": true, "mtp_n_predict": 4, "mtp_gpu": true,
  "conv_id": "abc123",
  "turn_id": "unique-retry-key",
  "think": "on"
}
```

- `messages` is the full conversation (role/content pairs) — the server
  formats it through the model's chat template itself, so you always
  send plain role/content, never pre-templated text.
- `images` lets you attach images inline in the same turn. Each image
  is either raw RGB (`rgb_b64` + `width`/`height`) or a reference to an
  image already uploaded via `/api/image/upload` (`id`/`upload_id`
  field instead). Requires a vision encoder to be loaded.
- `conv_id` is the key to warm-KV reuse (see 3.2). Omit it for a
  stateless one-shot call; the server still works, it just can't skip
  re-prefilling the same prefix next time.
- `turn_id` is an idempotency key (see 3.3) — optional, but strongly
  recommended for anything going over a flaky connection (mobile,
  long-running turns).
- `think` is `"on"`/`"off"` (also accepts `"1"/"0"`/`"true"/"false"`) —
  controls reasoning-mode models that support a think/no-think switch.
- Any sampling field you omit falls back to the server's `--temperature`
  etc. CLI defaults.

**Response** — `Content-Type: text/event-stream`, standard SSE framing
(`data: {...}\n\n`):

1. First event, sent immediately and unbuffered:
   `{"conv_id":"abc123"}` — echoes back the conversation id (generating
   one if you didn't supply one). This first event is not buffered for
   replay — if your client disconnects before seeing it, a reconnect
   via `/api/continue/resume` will not re-send it.
2. Then one event per generated chunk:
   `{"token":"...","context_tokens":1234}`
3. Optionally, a `{"notice":"..."}` event if the server's memory guard
   had to intervene (e.g. trimmed context under pressure).
4. A final literal line: `data: [DONE]\n\n`

While the connection is open but nothing has streamed yet (still
prefilling), the server sends SSE comment lines (`: ping\n\n`) roughly
every 10 seconds to keep the connection alive. Prefill on a cold, long
prompt can be minutes; this prevents proxies/browsers from timing the
connection out. If the server goes more than ~14 seconds without an
actual `[progress]` line of its own, it logs (server-side only, not
sent to the client) a note that it's still working.

**A dropped client never kills the generation.** The server keeps
buffering events into an internal job (`GenJob`) regardless of whether
anyone is listening, so a page refresh or a phone going to sleep mid-
reply doesn't waste the compute — reconnect with `/api/continue/resume`
(3.3) and get the rest.

### 3.2 Conversation continuation (`conv_id`)

Pass the same `conv_id` on the next `/api/chat` call and the server
tries to avoid re-prefilling the whole conversation from scratch:

- If the new `messages` array is exactly the previous one plus one new
  user message, the server extends the existing KV cache with just that
  one message (cheap).
- If it's a longer common prefix, it extends from wherever the prefixes
  diverge.
- If nothing matches, it falls back to a full fresh prefill.

This is purely a performance optimization — correctness does not depend
on it. If you don't track `conv_id` across calls, every call is a full
prefill of the entire `messages` array, which still works, just slower.

### 3.3 Retry safety (`turn_id`)

`turn_id` is an idempotency key scoped to one in-flight turn. If your
client retries a `/api/chat` call with the same `turn_id` because the
first attempt's connection died, the server recognizes the retry,
**does not** re-run generation or re-add the prompt a second time, and
either streams the still-running job or replays its buffered result.
Use a fresh random `turn_id` per logical turn, and reuse it only when
genuinely retrying that same turn.

`/api/session/extend_turn` (3.5) has the identical mechanism.

### 3.4 Reconnecting a dropped stream

| Endpoint | Use |
|---|---|
| `POST /api/continue` | Body: `{"conv_id":"...","model":"1"}`. Generates straight from wherever the warm KV cache left off — **no prefill at all**. Use this after a `/api/session/extend`/`/api/inject` call, or any time you've already gotten the context into the state you want and just want generation to continue. |
| `POST /api/continue/resume` | Body: `{"model":"1","from":<event index>,"turn_id":"..."}`. Replays a `/api/chat` (or `/api/session/extend_turn`) job from the given event index onward — this is what a client calls after a page refresh or dropped connection to pick back up a reply that's still generating or already finished. |

### 3.5 Injecting content without a full chat turn

| Endpoint | Use |
|---|---|
| `POST /api/inject` | Body: `{"content":"...","conv_id":"...","model":"1"}`. Injects a system-role block mid-conversation (for tool results, external context, etc.) and formats it correctly per the model's chat template (Llama-3, ChatML/DeepSeek, Phi-3, Zephyr, Gemma, Mistral, Alpaca, Granite each get their own real template fragment; anything else falls back to a generic bracketed `<<SYSTEM RESULT>>` block). Also appends the assistant turn-opener, so a following `/api/continue` generates immediately in assistant role instead of risking a stray role header. This is the right tool for "give the model a tool-call result and let it keep going." |
| `POST /api/session/extend_turn` | Body: `{"text":"...","images":[...],"conv_id":"...","turn_id":"...","model":"1"}`. Formats and prefills **one more user turn** (not a full conversation re-send) into the existing KV. Supports image attachments (local only, not over distributed). Has its own keep-alive/idempotency machinery identical to `/api/chat`'s. Use this when you're managing the conversation incrementally yourself rather than re-sending the full `messages` array each time. |
| `POST /api/session/extend` | Body: `{"text":"...","model":"1"}`. The simplest, rawest version — pushes raw text into context with no chat-template formatting at all and no idempotency handling. Mostly useful for non-chat/raw-completion use. |

### 3.6 Session, checkpoint, and start-state persistence

Nexus has three distinct persistence concepts. They all save the KV
cache (so generation can resume with zero re-prefill), but differ in
intent and storage location:

| Concept | Save | Load | List | Delete | Notes |
|---|---|---|---|---|---|
| **Session** | `POST /api/session/save` `{"path":"...","model":"1"}` | `POST /api/session/load` `{"path":"..."}` | — (managed by your own file list / the `/api/sessions/*` project endpoints) | `POST /api/session/delete` `{"path":"..."}` | Writes `<path>.kv` + `<path>.tokens` (+ `.worker_kv` if a distributed worker is attached). `path` must be inside an allowed storage folder (`session_path_ok`). This is the general-purpose "save my chat state to disk" mechanism. |
| **Checkpoint** | `POST /api/checkpoint/save` `{"name":"..."}` | `POST /api/checkpoint/restore` `{"path":"<returned .kvc path>"}` | `GET /api/checkpoint/list` | `POST /api/checkpoint/delete` `{"path":"..."}` | Stored under `<data>/checkpoints/<name>_<timestamp>.kvc` with a sidecar `.json` (name, path, token count, created time). Meant for "snapshot this exact point, I might want to come back to it" — e.g. before trying something risky in the same chat. |
| **Start-state** | `POST /api/start_state/save` `{"name":"...","conv_id":"..."}` | `POST /api/start_state/load` `{"path":"..."}` | `GET /api/start_state/list` | `POST /api/start_state/delete` `{"path":"..."}` | Stored under `<data>/states/`. Semantically a "blank slate" to warm-boot new conversations from (e.g. a long, expensive system prompt prefilled once and reused). Loading one registers a brand-new `conv_id` for you (returned in the response) and primes it so the next `/api/chat` with that `conv_id` extends instead of re-prefilling. |

A session load does **not** auto-generate a `conv_id` for you the way a
start-state load does — if you want `/api/chat`'s prefix-extend
optimization to work after a session load, track the conversation
yourself and pass a `conv_id`, or just use `/api/continue` to keep going
without sending `messages` at all.

### 3.7 Multi-slot context

If you need several independent conversations resident at once against
the same loaded model (rather than reloading a session each time), use
context slots:

```
POST /api/slot/select   {"slot": 2, "model": "1"}
GET  /api/slot/list
POST /api/slot/clear
```

Slot 0 is the default. Selecting a slot switches which KV cache
subsequent calls operate on, without touching the model weights.

### 3.8 Embeddings, LoRA, vision, audio

| Endpoint | Use |
|---|---|
| `POST /api/embed` | Body: `{"text":"...","normalize":true,"model":"1"}`. Returns `{"ok":true,"dim":N,"embedding":[...]}` — the model's last hidden state, L2-normalized by default. For RAG/semantic search. |
| `POST /api/lora/load` | Body: `{"path":"adapter.gguf","scale":1.0,"model":"1"}` |
| `POST /api/lora/unload` | Body: `{"model":"1"}` — unloads all LoRA adapters on that model |
| `GET /api/lora/list` | Lists currently loaded adapters |
| `POST /api/lora/scale` | Adjust an adapter's scale without reloading |
| `POST /api/vision/load` | Body: `{"path":"mmproj.gguf","model":"1"}` — load a vision encoder at runtime (same thing `--vision-encoder` does at startup) |
| `GET /api/vision/status` | Whether a vision encoder is loaded |
| `POST /api/vision/inject` | Inject a vision embedding directly into context outside the normal `/api/chat` image-attachment path |
| `POST /api/audio/load` | Body: `{"path":"whisper.gguf","model":"1"}` |
| `GET /api/audio/status` | Whether an audio encoder is loaded |
| `POST /api/audio/inject` | Inject an audio embedding directly into context |
| `POST /api/audio/encode_wav` | Body: `{"path":"/audio.wav","model":"1"}`. Returns `{"ok":true,"n_frames":N,"n_floats":N}` — encodes a WAV file on disk through the loaded audio encoder. |

### 3.9 Info, config, context inspection

| Endpoint | Use |
|---|---|
| `GET /api/info` | Model info: name, context size, chat format, etc. |
| `GET /api/config` / `POST /api/config` / `POST /api/config/save` | Read/update/persist server config |
| `GET /api/context` | Current context size/usage |
| `GET /api/session/tokens` | Raw token list for the active session |
| `GET /api/top_logits?n=<k>` | Top-k next-token logits/probabilities at the current position — useful for building your own sampling UI or debugging |
| `POST /api/tokenize` / `POST /api/detokenize` | Convert text ↔ token ids directly through the model's tokenizer |
| `GET /api/roster` | The configured model roster (for a multi-model picker UI) |

### 3.10 Runtime control

| Endpoint | Use |
|---|---|
| `POST /api/stop` | Stop an in-progress generation |
| `POST /api/reset` | Clear context / reset a model's session |
| `GET /api/turbo` / `POST /api/turbo` | Read/toggle turbo-prefill settings at runtime |
| `POST /api/preload` | Warm up / preload a model ahead of first use |

### 3.11 Project/chat/template storage (built-in UI's own data model)

These exist to back the embedded web UI's own chat history and have
nothing to do with model inference — you only need them if you're
reimplementing chat-history/project browsing in your own front end
rather than treating Nexus purely as an inference backend:

`GET/POST /api/projects`, `DELETE /api/projects/:id`,
`GET /api/projects/:id/chats`, `POST /api/projects/:id/chats` (save a
chat into a project), `GET /api/chats/:id`, `DELETE /api/chats/:id`,
`GET/POST /api/templates`, `DELETE /api/templates/:id`,
`GET /api/folders`, `POST /api/folders/add`, `POST /api/folders/remove`,
`POST /api/sessions/list_in`, `POST /api/sessions/meta`.

If you're building a standalone front end with your own storage, ignore
this section entirely and just talk to `/api/chat` + the session/
checkpoint/start-state endpoints.

### 3.12 Image upload

`POST /api/image/upload` accepts an image and returns an id you can
reference from `/api/chat`'s `images[].id` field instead of inlining
raw RGB bytes every call. Images are downscaled server-side according
to `NEXUS_WEB_IMAGE_MAX_EDGE`/`NEXUS_WEB_IMAGE_MAX_PIXELS` if set.

### 3.13 Multi-device "live" presence (`/api/live/*`)

`POST /api/live/snap`, `/api/live/hb` (heartbeat), `/api/live/list`,
`/api/live/claim`, and the generic `/api/boot` route exist to support
the embedded UI's "same chat open on two devices" handling. Not needed
for a standalone integration.

### 3.14 Search

`GET /api/search?q=` and `GET /api/search_detail?q=&n=` shell out to
`curl` against DuckDuckGo's HTML search and return scraped results —
this is what backs the built-in UI's web-search tool-use feature, not a
general-purpose search API. It requires outbound network access from
wherever the server process runs.

---

## 4. Using the `Engine` C++ API directly (no HTTP at all)

Everything the server does above funnels through two classes:
`nexus::Engine` (model lifetime, generation, persistence) and
`nexus::GenerateSession`/`SessionChunk` (the actual token-by-token
generation loop with pause/resume support). If you're embedding Nexus
in your own C++ program, link against these directly.

### 4.1 Minimal example

```cpp
#include "api/engine.h"

nexus::Engine engine;
engine.load("model.gguf");                 // GGUF, or native .nxm/.nxmodel
engine.set_gpu_layers(28);                 // optional

nexus::GenerateConfig cfg;
cfg.max_tokens = 512;
cfg.temperature = 0.7f;

std::string prompt = engine.format_user_turn("Hello!");
engine.extend_context(prompt, /*add_bos=*/true);

auto result = engine.generate(cfg);        // blocking, full result at once
std::cout << result.text;
```

For streaming output instead of blocking for the whole reply, use
`GenerateSession`/`SessionChunk` — see 4.4.

### 4.2 Loading

`Engine::load(path)` is GGUF-only on the surface, but it also
auto-detects embedded native/.nxm multimodal (vision/audio) tensors
bundled inside the same file via `loaded_tensor_map()` and binds them
automatically (`bind_native_qwen3next_from_loaded_tensors` internally)
— you don't need a separate call for models that ship their vision
tower embedded. A second, lightweight `gguf::Parser` pass is done
purely to pull out tokenizer data and the chat-template string; chat
format is auto-detected from that template unless you call
`override_chat_format()`.

`override_chat_format(name)` accepts the same format names as the
server's `--chat-format`: `ChatML`, `Qwen3`, `Mistral`, `Mistral V3`,
`Llama-3`, `DeepSeek`, `DeepSeek-R1`, `DeepSeek-Coder`, `Phi-3`,
`Phi-4`, `Gemma`, `Gemma4`, `Zephyr`, `Alpaca`, `Granite`, `OLMoE`,
`Custom`, `Raw`.

### 4.3 GPU placement

```cpp
engine.set_gpu_layers(n);
engine.set_gpu2_layers(n);           // second device
engine.add_gpu_layers(dev_index, n); // generic, any device index
engine.gpux_device_name(idx);
nexus::Engine::list_gpu_devices();   // static — enumerate Vulkan devices
```

### 4.4 Streaming generation: `GenerateSession`

```cpp
auto session = engine.begin_session_chat(messages, cfg);
while (auto chunk = session->next()) {
    std::cout << chunk->text;        // SessionChunk
    if (chunk->is_final) break;
}
```

`begin_session`, `begin_session_warm`, and `begin_session_chat` are the
three entry points — `_warm` continues from existing KV without
re-prefilling (what the server's `/api/continue` calls), `_chat` takes
a structured message list and applies chat-template formatting for you
(what `/api/chat` calls).

Internally, `GenerateSession::Impl::prefill()` picks one of three
routes depending on engine settings: plain token-by-token (debug/
fallback), turbo CPU-batched, or turbo-GPU-stream (the hybrid path that
streams work through the GPU layer-by-layer — this is what the v28/v32
prefill-progress reporting in the server targets).

`GenerateSession::run()`'s main loop also contains the speculative-
decoding (MTP) path: `mtp_propose()` drafts ahead, `forward_verifyN()`
verifies the draft against the real model in one batched pass, and
accepted/rejected/bonus tokens are reconciled with a KV
snapshot/restore on mismatch. Enable with `cfg.mtp_enabled = true` (and
`mtp_n_predict`, `mtp_gpu`).

The pause-tag mechanism (`<nexus:tool>`/`<nexus:call>` by default, used
for tool-use-style pausing with KV preserved mid-generation) is part of
`SessionChunk`'s handling — a session can pause generation when it
emits one of these tags and be resumed later without losing context.

**Note:** `begin_session_speculative()` is currently a stub that falls
back to an ordinary session — the real speculative path lives inside
`GenerateSession::run()`'s MTP block above, not in that entry point.

### 4.5 Context management

```cpp
engine.extend_context(text, add_bos);
engine.context_size();
engine.context_mark_start(); engine.context_mark_end();
engine.context_pin(); engine.context_pin_prefix();
engine.context_trim(n);
engine.context_auto_trim();     // trims to 75% capacity, never below pinned_tokens+128
```

Section markers let you tag a range of context as "pinned" (protected
from auto-trim) — useful for keeping a system prompt or a long
document anchor alive while letting conversational turns get trimmed
under memory pressure.

### 4.6 Multi-session slots

```cpp
engine.context_slot_select(n);
engine.context_slot_clear();
engine.context_slot_size();
engine.context_slot_count();
```

Backed by a static `g_slots` vector; slot 0 is the default and what a
freshly-loaded engine starts on.

### 4.7 Persistence

```cpp
engine.save_session(path);   // writes <path>.kv + <path>.tokens
engine.load_session(path);

engine.save_start_state(path);
engine.load_start_state(path);   // same file format as save_session,
                                  // semantically "blank slate to warm-boot from"

nexus::Engine::checkpoint_save(name);     // static, backed by g_checkpoints
nexus::Engine::checkpoint_restore(name);
nexus::Engine::checkpoint_list();
nexus::Engine::checkpoint_delete(name);
```

### 4.8 Embeddings, LoRA, vision, audio

```cpp
auto vec = engine.embed_text(text, /*normalize=*/true);

engine.lora_load(path, scale);
engine.lora_unload_all();
engine.lora_set_scale(scale);
engine.lora_count();

engine.load_vision_encoder(path, progress_cb);
engine.has_vision();
engine.encode_image(...);
engine.inject_vision_tokens(...);

engine.load_audio_encoder(path, progress_cb);
engine.has_audio();
engine.encode_audio(...);
engine.encode_audio_wav(path);
engine.inject_audio_tokens(...);
```

### 4.9 Distributed / layer-streaming primitives

These are the low-level building blocks the server's `--dist-worker`/
`--dist-peer` CLI mode (Section 5) is built out of. Most integrations
will never call these directly — reach for the server's distributed CLI
flags instead unless you're building your own distributed topology:

`forward_layer_range`, `prefill_layer_range`, `load_range` (direct
partial-GGUF layer load, no temp file), `open_for_layer_streaming` /
`swap_layer` (true single-layer "ghost" streaming — loads tokenizer and
chat template once, then swaps one layer's weights at a time),
`free_own_layer` / `reload_own_layer` (free a resident layer during
hybrid prefill without touching KV, then bring it back),
`save_kv_to_buffer` / `load_kv_from_buffer` and the per-layer siblings
`save_kv_layer_to_buffer` / `load_kv_layer_from_buffer` /
`free_kv_layer`, and `send_layer_subset_gguf` / `load_streamed_shard` /
`write_layer_subset_gguf` for building and sending GGUF shards over the
wire (header-only or full).

---

## 5. Distributed (donor/primary) mode from the CLI

This is the "use a second, weaker computer to host extra layers" setup,
run without the launcher's `donor_launcher.py` UI:

**On the donor machine** (just binds a port and waits — no model loaded
at startup; it receives either streamed weights or loads its own local
GGUF once the primary tells it which layer range to serve):
```
nexus_server --dist-worker --port 9876 --dist-passphrase <shared-word> \
             [--dist-local-model /path/on/donor.gguf] \
             [--gpu-layers N]   # if the donor also has a GPU to accelerate its slice
```
A worker in this mode starts **no HTTP server** — it's a raw TCP
listener for the primary to connect to, authenticated by XORing against
`--dist-passphrase` (the passphrase itself is never sent over the wire).

**On the primary machine**:
```
nexus_server --model /path/to/model.gguf --port 8080 \
             --dist-layer-start 0 --dist-layer-end 20 \
             --dist-peer donor-host:9876 --dist-peer-layers 20:40 \
             --dist-passphrase <shared-word> \
             [--dist-peer-local-load]   # tell that peer to load its own local copy instead of receiving streamed weights
             [--dist-required]          # refuse to start/generate if the peer can't be reached
```
`--dist-layer-start`/`--dist-layer-end` is the range the primary itself
loads and computes. `--dist-peer-layers start:end` is the range that
specific peer is responsible for. If you omit the layer-range flags,
the server logs a loud warning and falls back to loading the full model
locally — these must be set correctly for partial loading to actually
happen. Multiple `--dist-peer` flags (each followed by its own
`--dist-peer-layers`/`--dist-peer-local-load`) chain several donors
together.

Once connected, ordinary `/api/chat` calls against the primary's normal
port work exactly as in local mode — the server checks
`S.dist_sess->is_connected()` internally and transparently routes
generation through the distributed path instead of local, including
for `/api/inject`, `/api/session/extend_turn`, and session save/load
(which also saves/restores the worker's KV alongside the primary's).
Image attachments are **not** currently supported through a distributed
`/api/session/extend_turn` call — only in a fresh local chat.

---

## 6. Which approach should I actually use?

- **Building any kind of chat front end, mobile app, or bot** → run the
  server, use `/api/chat` + `conv_id`. Everything else is optional.
- **Need retrieval/semantic search** → `/api/embed`.
- **Want to give the model tool results mid-conversation** →
  `/api/inject`.
- **Want warm-boot conversations from a prefilled system prompt** →
  `/api/start_state/save` once, `/api/start_state/load` on every new
  session.
- **Embedding Nexus inside an existing native C++ app with no HTTP
  server wanted at all** → link `engine.h`/`engine.cpp` directly
  (Section 4).
- **Spreading one model across two machines** → the `--dist-*` CLI
  flags (Section 5); reach for the raw layer-streaming primitives
  (4.9) only if you're building a different distributed topology than
  the one-primary/N-donors chain the server already implements.

---

## 7. Known gaps and footholds to be aware of

- `begin_session_speculative()` is a stub — use `cfg.mtp_enabled` on an
  ordinary session for real speculative decoding.
- Distributed `extend_turn` can't take image attachments yet.
- `/api/search` and `/api/search_detail` depend on outbound network
  access and scrape DuckDuckGo's HTML via a `curl` subprocess — treat
  this as a convenience feature for the built-in UI, not a stable
  search API to build on.
- A dropped SSE connection is always safe to reconnect from
  (`/api/continue/resume`), but the very first `{"conv_id":...}` event
  of a call is sent unbuffered and is not replayed — if your client
  needs the conv_id and might miss that first packet, generate and send
  your own `conv_id` on the request instead of waiting for the server
  to assign one.
- Session saves/loads and checkpoint saves/loads are restricted to
  folders the server considers "allowed" storage paths
  (`session_path_ok`) — arbitrary absolute paths outside that list are
  rejected.
