# Conversation memory: forget, archive, restore

Nexus keeps every conversation in the model's working memory (the KV cache). It never re-reads the chat history for a new message; it adds only the new text. This page covers how to control that memory: how long it can grow, what happens when it is full, and how to remove parts of it, save them, and put them back.

It works with any model and chat format. Nexus finds where each message starts from the model's own chat template, so the same controls work for ChatML (Qwen), Llama 3, Gemma, Mistral, Phi-3 and others.

---

## What you can do

| You want to | Use |
|---|---|
| Keep going when the conversation gets long, instead of starting a new chat | Auto-trim (on by default) |
| Drop the intermediate work and keep the result ("we have the answer, not all the steps") | **Forget** on those messages |
| Keep a piece of work for later: a file revision, a long manual, a worked example | **Archive** (saved as A1, A2, …) |
| Give the model a saved piece back, in this chat or another one | **Restore** |
| Copy a message, code blocks and all | **Copy** on the message |

---

## In the web chat

Hover any message to see two buttons under it.

- **Copy** copies the message as Markdown, so code blocks keep their fences. For model replies, the thinking is left out.
- **⋯** opens the memory menu:
  - **Forget (remove from memory):** the model no longer sees this message, and the RAM it used is freed right away.
  - **Archive (save it, then forget it):** asks for a name (for example *app.py rev 1*), saves the message's memory as **A1**, **A2**, … and then removes it.
  - **Select:** pick several messages, then use **Forget** or **Archive** on the bar that appears. Archiving several messages together needs them to be next to each other.
  - **Restore into memory (A3):** shown on archived messages. It puts that part back.

Messages that are no longer in memory stay on screen, dimmed and tagged *forgotten*, *archived A3* or *dropped (memory full)*. They are never sent to the model again.

The first message of a chat that holds the system prompt is tagged **kept** and can't be removed. Some formats (Gemma, Mistral, Llama 2) put the system prompt inside the first user message, so that message is kept too.

### The Memory tab

The sidebar's **Memory** tab shows:

- **How much is in memory:** tokens used and the limit, with a bar.
- **Limit:** the model's maximum, a set number of tokens, or what fits in memory (no limit). It also has the auto-trim switch (below).
- **Saved memories:** every archive with its name, size, model and date, and four buttons: **Restore**, **Rename**, **Copy text** (the chat text that was archived) and **Delete**.

The status bar shows **in memory: N / limit tok**, always the engine's own count. While a long prompt is being read it shows the percentage, speed and time left instead.

---

## When memory is full: auto-trim

When the conversation reaches its limit, Nexus drops the **oldest whole messages**, down to about 75% of the limit, and keeps going. It never drops:

- the system prompt (and anything before the first message),
- the message currently being answered.

Before each new message it also makes room, if needed, so the message and a reply both fit. When messages are dropped, the chat shows a note saying how many.

Turn auto-trim off and the reply stops when memory is full instead. With *What fits in memory* there is no limit, so auto-trim never comes into it.

Before a message that takes the conversation past the model's own maximum or past your limit, the web chat asks *Send it anyway?* and says what will happen. It never blocks; Cancel keeps the message in the box.

**Where to set it:**
- **Launcher:** the *Conversation memory* card, saved per model:
  - *Conversation length limit:*
    - *The model's maximum:* stops there.
    - *A set number of tokens:* exactly that, even above the model's maximum.
    - *What fits in memory (no limit):* grows until memory runs out, past the model's maximum if it gets there. Past it, a model will likely start repeating itself or fall apart; that's yours to try. Only the memory safety buffer, when it's on, stops it sooner.
  - *When the conversation is full, drop the oldest messages and keep going* (ticked by default).
- **Web chat:** the Memory tab (applies until the server restarts).
- **Server flags:**
  - `--ctx-limit <n|auto|max>`: `max` (default) is the model's own maximum. A number is used exactly, even above the model's maximum. `auto` is no limit at all.
  - `--auto-trim <on|off>`

---

## Saved memories (archives)

An archive is the model's memory of that part of the conversation, not just its text. Restoring it doesn't make the model re-read anything; the memory is put straight back.

- Archives are stored in `nexus-data/kv_archive/`:
  - `A1.nxctx`: the tokens and their memory.
  - `A1.json`: the name, model, model file, size, date and the chat text.
- Restoring puts the archive **at the end** of the current conversation. Its positions are adjusted so it fits there exactly, and the chat shows it as a *Restored A1* block.
- An archive isn't tied to the chat it came from. Restore it into any chat with the same model.
- **Same model only.** A different model (other architecture or layer sizes) is refused, because that memory would be meaningless to it. A different file of the same model, such as another quantization, gives a warning but still restores; it may be remembered a little less accurately.
- Start the chat first (send one message), then restore.

**Example: a big manual you only need for a while.** Paste a 40,000-token manual into a chat once and archive it right away. Whenever you need it, restore it into the chat you're working in, finish the job, then forget it again to free the memory.

**Example: code revisions.** Archive *app.py rev 1* when it's done, continue with rev 2. Later, restore rev 1 next to it and ask the model to compare.

---

## How exact is it?

Removing a part from the middle shifts everything after it back into place. The model's position encoding (RoPE) is recalculated for every layer, so the rest of the conversation is stored exactly as if the removed part had never been there. This works for every RoPE style: interleaved, half-split, partial rotary, Gemma p-RoPE, linear and YaRN scaling.

**Models with recurrent layers** are different. Qwen3.5 and Qwen3-Next, for example, mix attention layers with layers that keep a running summary of everything they've read, and that summary can't drop individual tokens. On these models:

- **Forget** is exact in the attention layers. The summary layers still hold a faint trace of the removed text, so the model may "feel" it did something it can no longer see.
- **Restore** goes into the attention layers only. The summary layers never saw the restored part, so restore-then-forget leaves no trace there.

The Memory tab says so when the loaded model is one of these.

---

## For developers: the API

All endpoints take `model` (`"1"` or `"2"`). Edits are refused while a reply is being written, and for a model split across computers.

| Endpoint | Body | Does |
|---|---|---|
| `GET /api/kv/turns?model=1` | | The messages in memory: `start`, `end` (token positions), `role`, `protected`, `preview`. Also `context`, `limit`, `model_max`, `auto_trim`, `exact`, `bytes_per_token`. |
| `POST /api/kv/remove` | `start, end, ctx` | Forget tokens `[start, end)`. |
| `POST /api/kv/archive` | `start, end, ctx, label, msgs` | Save `[start, end)` as the next `A<n>`, then forget it. `msgs` is any text you want kept with it (the chat page stores its messages as JSON). |
| `GET /api/kv/archive/list` | | All archives (the `.json` contents), plus the model files running now. |
| `POST /api/kv/archive/restore` | `id, ctx` | Put an archive back at the end. Returns `added` and, for a different model file, a `warning`. |
| `POST /api/kv/archive/rename` | `id, label` | |
| `POST /api/kv/archive/delete` | `id` | |
| `POST /api/kv/settings` | `limit: "max"\|"auto"\|N, auto_trim: "on"\|"off"` | Change the limit and auto-trim while running. |

`ctx` is the context size your app last saw, from `/api/kv/turns`. If the conversation changed since then, the edit is refused instead of hitting the wrong tokens; read the turns again and retry. Ranges must be whole messages taken from `/api/kv/turns`, and the protected start can't be removed.

Token counts in the chat stream are the engine's own. Every streamed event carries `gen` (tokens generated so far) and `context_tokens` (tokens in memory). The last event is `{"final":true,"gen":N,"context_tokens":M}`. While a prompt is read, `/api/chat` also sends progress events: `{"progress":{"pct":34.2,"tokens":98304,"tps":104.2,"eta":580}}`.
