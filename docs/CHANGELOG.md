# Changelog

## 1.0.1

### Mixture-of-experts models on the graphics cards
- **MoE models (Qwen3.6-35B-A3B and others) now use the cards.** The attention part of every layer runs on its card. The experts stay in RAM, and the ones that matter get a copy on the cards. For each token, every card runs the experts it holds in one go, while the CPU runs the rest at the same time. In 1.0.0, MoE layers ran entirely on the CPU. See `docs/MOE_MODELS.md`.
- **Two ways to choose the experts** (launcher *Experts on the cards (MoE models)*, `--moe-experts`):
  - *Picked by each prompt* (default): after a prompt is read, the experts it used most move onto the cards. Re-checked every 256 written tokens.
  - *Fixed home*: placed at load and never moved. Either whole layers until the cards are full, or N experts of every layer (`--moe-home`).
  - *CPU only*.
- **Prompts on MoE models get the prompt speed-up.** Each expert runs once over all the tokens that picked it: on the card that holds it, streamed through the prompt card when enough tokens picked it, or on the CPU alongside the cards. In 1.0.0 these prompts were read one token at a time.
- The prompt card (Maximize GPU) holds experts too. They step aside while a prompt is read and come back after it.
- Measured on a Qwen3.6-35B-A3B GGUF with 25 GB of RAM: about 6 tokens/s writing on a GTX 1060 3 GB + R9 290 4 GB. On the GTX 1060 alone: over 40 tokens/s reading the prompt and about 5 tokens/s writing, sustained through a 10,000-token reply.
- Experts on the CPU run through the normal fast matrix code. Before, each one went row by row and started thousands of threads per token.
- With Maximize GPU, the second card now fills with experts. Before, a 35B-A3B's attention parts (1.3 GB) all fit on card 1, and card 2 had nothing to hold.
- Experts are always kept in RAM, even in layers on a card, because the CPU computes the ones that aren't on a card. Maximize GPU now sizes a MoE layer by its card part only, not counting its experts.

### GGUF files made by llama.cpp (Qwen 3.5 / 3.6)
- **They load.** Current llama.cpp merges two pairs of tensors in every Qwen 3.5 layer (`attn_qkv_gate`, `ssm_beta_alpha`). Nexus now reads each half in place, with no copy, for any quant type. Before, these files failed with `missing standard attention weights`.
- **9B / 27B GGUFs compute correctly.** When a model has more value heads than key heads, llama.cpp stores the value heads in a different order. Nexus detects which order a file uses and uses it on the CPU and on every card. Before, every value head would have been paired with the wrong key head. Nexus `.nxm` files and Training Ground GGUFs keep their own order. `NEXUS_QWEN35_V_ORDER=tiled|grouped` overrides the detection.
- Older Qwen3-Next GGUFs with a fused `ssm_ba` tensor are split for the card instead of sending the layer to the CPU.
- **Fixed: Q4_0 and Q5_0 weights were decoded in the wrong order**, which gave wrong results wherever they appeared. llama.cpp's Q4_K_M uses Q5_0 for some MoE expert matrices. Also fixed in Training Ground.

### Your own system prompt
- **System prompt** setting per model in the launcher (next to *Chat format*): *The model's usual one* or *My own*, a text box of any length. It's used whenever a chat doesn't send its own; the web chat's System Prompt box still wins. Works for every chat format. Server: `--system-prompt <text>`, `--system-prompt-file <path>`.
- Fixed: the Qwen3 chat format told every model using it that it was "Qwen, created by Alibaba Cloud". That line is now only used for models whose name says Qwen; other models get "You are a helpful assistant."

### Graphics cards
- **One number for card memory: *GPU prompt processing space* (MB, default 768).** Every card, in every mode, takes the whole layers that fit in what it really has free, minus this number. The same number is the room a prompt gets on the prompt card. It replaces *Card memory always kept free*, *Prompt card: memory for reading a prompt* and *Prompt working space on the card*. A number you set is used exactly.
- Fixed: with Maximize GPU on the prompt card, card 2 was filled first and took the layers card 1 should have had. Card 1 now always claims its share first.
- **The output matrix (lm_head) goes on a card** (launcher *Output matrix (lm_head)*, `--lm-head first|after|cpu`). *On the fastest card first* (default) times every card's read speed and puts it on the fastest card with room. If none has room, that card hands back just enough layers. Heads in types with no card kernel, or slow ones, are converted to 8-bit once. Before, the head ran on the CPU for every token unless the whole model was on card 1.
- Fixed: Qwen 3.5 layers on a card ran on the CPU anyway, because two small BF16 tensors had no card kernel. They're now widened once at load. Together with the output-matrix change above, writing on a two-card 9B went from about 3 to about 17 tokens/s.
- Fixed: the conversation state of layers on a second card was never read back, so the next turn could continue from stale state.
- Layers on different cards are chained without waiting, card to card.
- The log says where every layer ran and why the CPU took any (`[decode]` placement report), and prints the time per written token once writing is under way.

## 1.0.0: first public release (2026-11-14)

### Conversation memory
- Forget, archive (A1, A2, …) and restore any part of a conversation from the web chat, using the **⋯** menu on each message. There's also a **Memory** tab with usage, limit and the saved archives. See `docs/CONVERSATION_MEMORY.md`.
- Removing part of the middle re-rotates the keys of everything after it for the model's own RoPE (interleaved, half-split, partial, p-RoPE, linear, YaRN), so the rest stays exact.
- Message boundaries come from the model's own chat template (ChatML, Llama 3, Gemma, Mistral, Phi-3, …), not model-specific rules.
- Conversation length limit (model maximum / what fits in memory / a set number), with auto-trim of the oldest messages. The system prompt is never dropped. Launcher card *Conversation memory*; server flags `--ctx-limit`, `--auto-trim`.
- **Copy** button on every message (Markdown, code blocks kept).
- Conversation limits do what they say. *The model's maximum* stops there. *A set number* is used exactly, even above the model's maximum. *What fits in memory* has no limit at all and grows until memory runs out, past the model's maximum if it gets there; only the memory safety buffer, when on, stops it sooner. Before, it was a number guessed at startup that could end a reply after one token without a word.
- The web chat asks *Send it anyway?* before a message that goes past the model's maximum or your limit, and says what will happen. A reply stopped by a limit always says so.
- Forget, archive and auto-trim now give the freed memory back to the system right away. Before, RAM stayed the same until the conversation grew back into it.
- Fixed: auto-trim only triggered at the model's maximum length; after a trim the token list was cut from the end (dropping the newest tokens) and read past its end; trimmed text kept wrong RoPE positions; the system prompt could be evicted; a new conversation could see the previous one's MTP history.

### Web chat
- **Documents are read, not pasted as bytes.** Drop a PDF, Word file (.docx, .odt), spreadsheet (.xlsx, .xls, .ods), slides (.pptx, .odp), e-book (.epub), .rtf or web page, and the model gets its text: pages, sheets and slides labelled, headings, lists and tables kept. PDF pages without text are read with **OCR**. Everything runs offline in the browser, from a `web/vendor` folder shipped with Nexus. Folders read the documents inside them too. See `docs/DOCUMENTS_IN_CHAT.md`.
- Fixed: a dropped PDF (or any binary file) was sent as text: hundreds of thousands of tokens of noise. Binary files that aren't documents are now refused with a message, and old `.doc` / `.ppt` files get a hint to save them as `.docx` / `.pptx`.
- Attach a whole folder: drag it in, or use **DIR**. Each file is labeled with its path (`myproj/src/app.py`) after a list of all the paths. Build output, dependencies, binaries and lock files are left out, and it asks first if the folder won't fit in memory.

### Graphics cards
- **Several graphics cards at once.** Pick card 1, 2, 3, … in *Run on graphics card(s)* (a new row appears each time). Each card holds either a set number of layers or, with **Maximize GPU**, as many as really fit. Helper cards are filled while the model loads (shown on the loading bar) and compute their own layers while writing. See `docs/GRAPHICS_CARDS_AND_SPEED.md`.
- **The prompt speed-up card** is whichever card you pick under *Prompt speed-up*, wherever it is in the list. It can hold layers for writing too, and they stay on it.
- **8-bit prompt math per card: Auto / On / Off.** Auto uses the card's integer units when the driver reports hardware for them, and otherwise times both ways on that card at startup and keeps the faster. Older cards that only fake 8-bit (e.g. AMD R9 290: 13 → 30 tok/s) get the regular kernels; cards with the hardware keep the speed-up even when their driver doesn't report it.
- **CPU: 8-bit output math (Off / Auto / On)** for NQ models: the CPU's byte instructions do the multiplying, about 1% rounding per step. Auto times both on your CPU at the first token and keeps 8-bit only if it's at least 10% faster.
- **Split long prompt steps (Auto = AMD cards only / On / Off).** It keeps each job under an AMD Linux driver's time limit, never below 256-token steps. AMD cards also use their compute queue (a 60 s limit instead of 10 s), and a driver reset, if one happens, is reported in plain words.
- **Card memory is measured, not guessed.** While a prompt is read, its working space is capped at what the card really has free, with an attention reserve that grows with the conversation (*Prompt working space on the card* switches back to sizing by the card's total memory). *Card memory always kept free* defaults to 768 MB. While writing, every card is checked every 128 tokens and gives a layer back when it has under 256 MB free.
- Settings with long explanations have a small **?** button: hover or click to read the note.
- Fixed: layers given to a second or later card were computed on the CPU (Qwen3.5 / NQ models), and the card stayed empty.
- Fixed: a crash (SIGSEGV in the graphics driver) when a card was nearly full. The weight cache could evict a weight the current step had just fetched.

### Maximize GPU and oversized-prompt protection
- **Maximize GPU** fills a card with as many layers as fit. On the prompt card those layers keep no copy in RAM, and a long prompt clears only the room the prompt speed-up needs, plus the conversation memory at the prompt's length. They go back before the reply. With MTP drafting on the card, room for the draft head is kept.
- **Oversized-prompt protection:** while a prompt is read, layers are dropped from RAM whenever free RAM falls under a number you set, and go onto the card or back into RAM afterwards.
- The drive is only used while a prompt is read. Every layer is on a card or in RAM before the first token is written.
- Together they let a long prompt fit where RAM would run out (e.g. 100k tokens on a 27B with 25 GB of RAM), keeping the prompt speed-up. See `docs/MAXIMIZE_GPU.md`.

### Speculative decoding (MTP)
- Exactly the same tokens as normal writing, repetition penalties included.
- The real token and the guesses are checked in one pass. After a miss, nothing is run through the model again: the accepted tokens are kept from that pass, and only the linear-attention state is replayed. The draft head always starts from the model's latest state, and the output layer runs once per check.
- Each layer of a check runs where it lives. Before, every check uploaded the whole model through the prompt card, taking 5.7 s per check.
- Helps most when most layers are on graphics cards. On CPU-heavy setups it's about even.

### Speed
- CPU batches for NQ models: 4 tokens cost about 2× one token instead of 4.2×. This affects MTP checks and short prompts read on the CPU.
- Native NQ models: the FFN of layers on a graphics card runs on the card while writing. Before, it ran on the CPU and re-read its weights from the model file for every token.
- Ordinary attention layers (Llama, Mistral, Qwen2/2.5/3 dense, InternLM2, EXAONE, DeepSeek) run as one GPU job per layer, chained without waiting. `NEXUS_GPU_LAYER_ANY=0` restores the old behaviour for comparison.
- The GPU attention kernel handles 7-8 query heads per KV head (e.g. Qwen2.5-7B).

### Launcher
- The progress bar says what's actually happening: reading the prompt, moving layers onto a card, loading the model.

### Progress and token counts
- A steady prompt-reading line every 2 s (percentage, tok/s, time left) for every message, not just the first. The launcher's bar no longer flips to "no progress" on long prompts.
- `GET /api/progress[?model=2]` reports the prompt being read right now (`active`, `pct`, `tokens`, `tps`, `eta`), for callers of the blocking `/api/session/extend_turn`.
- `/api/chat` streams `progress` events while reading a prompt. Every token event carries the engine's real `gen` and `context_tokens` counts, and a closing `final` event gives exact totals.

### Security and access
- **Fixed a command injection** in `/api/search_detail_by_url`: the URL went into a shell command. Web fetch now runs curl without a shell, http/https only, and the search endpoints are **off unless `--enable-web-tools`**.
- Web pages from internet sites can no longer use the API, including DNS-rebinding tricks. Web pages from this computer and the local network still can; programs are unaffected. `--cors lan|local|any`, `--allow-origin`.
- Optional API key for other computers (`--api-key` / `NEXUS_API_KEY`, launcher Network card). The computer running Nexus never needs it, and the web chat asks for it once.
- `/api/info` and `/api/context` accept a `?model=` query.
- See `docs/ACCESS_AND_API_KEYS.md`.

### Model handling
- RoPE layout auto-detection now uses the model's architecture only, never its name (a "Dolphin" Llama was getting half-split RoPE).
- The final output norm follows the same architecture rule as the per-layer norms.

### Nexus Code Hardener (new, `nexus_harden_tool/`)
- Unattended chunked review of a whole codebase: per-chunk prompt chains; separate chunk / large-context / final-pass models, started and swapped automatically; live output with read/write speed; stop, skip, restart, reset and re-run mid-run; split oversized files for either model.
- Final report from a stronger model, and follow-up questions with files attached, shown as a chat. Each answer is saved automatically with the report. *Save report* (text) or *Save report with final KV* (with the model's memory), and continue either later.
- Chain steps 2+ and follow-ups add only the new prompt to the model's memory (no re-reading), with reading percentage and speed shown while it reads.
- Opens clean: the last project is offered at the top, and *New project* clears the page without deleting anything. Live output follows along in raw and rendered view, unless you scroll up.
- Saved prompt chains (built-ins for security audit, second look, notes & personal-info sweep, consolidation); settings carry over to new projects; folder picker and recent projects; API keys for servers on other computers.
- See `nexus_harden_tool/README.md`.

### Documentation
- **The Nexus Guide** (`docs/NEXUS_GUIDE.md`, also as PDF): the whole manual in one place: how Nexus works, setups for common hardware, every launcher setting, every server flag and environment variable, every endpoint, files it writes, the log and troubleshooting.
- New pages: `GRAPHICS_CARDS_AND_SPEED.md`, `MAXIMIZE_GPU.md`, `CONVERSATION_MEMORY.md`, `DOCUMENTS_IN_CHAT.md`, `ACCESS_AND_API_KEYS.md`.
- `README.md`, `LAUNCHER_GUIDE.md` and `ENGINE_API_GUIDE.md` updated for everything above.
