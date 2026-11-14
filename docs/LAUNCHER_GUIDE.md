# Nexus Launcher Guide

Welcome to **Nexus**, a free, local, multi-model inference engine built
to put real LLM inference in the hands of anyone, not just people with a
multi-thousand-dollar GPU rig. Nexus runs on whatever you already have:
a modest graphics card, an old laptop, a spare desktop, or just a CPU.
It can use every graphics card in the machine, old ones included, and it
can split one large model across two or more computers you already own.
Hardware you'd normally call "too weak" becomes part of the answer
instead of a reason to give up. No subscription, no cloud, no sending
your prompts anywhere.

This guide covers the **launcher**, the window you use to pick a model,
configure it and start it. For everything in one place (every server
flag, environment variable and endpoint), see the
[Nexus Guide](NEXUS_GUIDE.md). To run the server from the command line
or your own front end, see [ENGINE_API_GUIDE.md](ENGINE_API_GUIDE.md). To
convert your own models, see the [Quant Builder Guide](../nexus_quant_builder/README.md).

---

## Quick Start

This gets you from "just installed" to "chatting with a model" in a few
steps. Everything here is covered in more depth further down.

### 1. Put a model where the launcher can find it

The simplest option: drop a `.gguf` or `.nxm` model file into the
**`models`** folder next to the launcher program (it's created the first
time you run it). The launcher scans that folder, and one folder up,
automatically.

If your models live somewhere else (another drive, a USB stick, a folder
you organize yourself), open the **Setup** tab → **Model** card →
**Add folder...** and point it at that folder. Folders you add are
searched *first*, and they're remembered.

### 2. Pick your model

On the **Model** card, use the **Model file** drop-down. It lists
everything found in your model folders, with each file's size and
location. Or type/paste a full path into the box below it (for example
`D:\models\model.nxm`).

As soon as you pick a model, the launcher reads its metadata (size,
layer count, detected chat format), shows it, and picks sensible
graphics card settings for you (step 3).

### 3. Choose where it runs

Open **Run on graphics card(s)**. The first time you pick a model, the
launcher has already decided:

- If the model fits comfortably on your best graphics card, it's placed
  there entirely.
- If it's too big for the card, it stays on the CPU, and the card speeds
  up reading prompts instead (step 4), so it isn't wasted.

You can change this. **Graphics card 1** and **Layers kept on card 1**
set how many of the model's layers live on that card (0 = none). As you
type a number, the launcher shows roughly how many GB that is and
whether it fits. Pick a second, third, … card the same way: a new row
appears each time you choose one. Each card can hold **a set number** of
layers or, with **Maximize GPU**, **as many as fit**.

### 4. Prompt speed-up (prefill acceleration)

Open **Prompt speed-up (prefill acceleration)**. This is *separate* from
step 3. While a prompt is read, it swaps the model's layers through a
graphics card one at a time, even if the model itself runs on the CPU.
It needs room for a single layer, so almost any card helps.

Leave it on **Auto**. It turns itself on when it helps (the cards don't
already hold the whole model) and says why when it's off.

### 5. Basic model settings

**Sampling** holds the usual knobs: Temperature, Top-p, Top-k, Min-p,
repeat penalty, max tokens and so on. Sensible defaults are filled in.

### 6. Launch

The top-right corner always shows a small status readout with a
**Start** button for the selected model. Click **Start**. The launcher
switches to the **Log** tab so you can watch it load (with a progress
bar). When the dot turns green and says *Running*, click **Open chat in
browser**. The built-in chat opens, pointed at that model.

That's it: you're chatting with a model running on your own computer.

---

## Detailed Reference

### The window itself

At the top: the Nexus name and a **Save current settings** button. Nexus
writes your settings to disk only when you press it. Until then,
anything you change is in effect right away (so you can experiment
freely) but is forgotten if you close the launcher without saving. The
button turns amber with a `*` when something is unsaved.

Settings are stored in `nexus-data/nexus_launcher.json` next to the
launcher (the same folder the server uses for chats, projects and
sessions). If that folder can't be written to (read-only USB stick,
permissions), a banner at the top of the Setup tab says so. The launcher
still works; it just can't remember anything between runs.

Three tabs: **Setup**, **Log**, **Donor**.

Every setting with a **?** next to its name has a note. Hover over the
**?** to read it, or click it to keep it open.

### Running more than one model at once

The row of buttons under the top bar (Setup and Log tabs) is the **model
slot strip**. Nexus can run several independent models side by side,
each its own server on its own port, each with its own settings. Click
**+ Add a model** to add a slot, click a slot's button to configure it,
and **Remove this model** to drop the last slot (only the last one, and
only while it's stopped). Each slot remembers its own model, cards,
sampling, donors, everything.

### Start / Stop / status (top-right panel)

Shows a status dot (gray = stopped, amber = loading, green = running,
red = stopped with an error) and text for the slot you're looking at,
plus:
- **Start** / **Stop** for that model
- **Open chat in browser** (once it's ready): opens and copies the chat
  address (`http://127.0.0.1:<port>/`)
- **Start all** / **Stop all**, when you have more than one slot

### Setup tab: the cards

Cards collapse and expand individually (click the header). They apply to
the selected model slot.

#### Model
- **Model file** drop-down (your scanned models), or a typed/pasted path
  underneath it for anything outside a scanned folder.
- **Browse...** / **Rescan** next to the drop-down.
- Once a model is chosen, a line of metadata: architecture, size on
  disk, layer count (and roughly how big one layer is), and the detected
  chat format with a confidence level.
- **Profile for this model**: see *Profiles* below.
- **Model folders**: the folders you've added, each removable with its
  `x`, plus **Add folder...**. The `models` folder next to the launcher
  is always searched too.

#### This computer
A read-only hardware summary: CPU and core count, total and free RAM,
and every graphics card found, each marked usable or not (with the
reason when not, such as a driver problem). **Re-check hardware** runs
the check again after you change something (added a card, plugged in an
eGPU).

#### Run on graphics card(s)

Places model layers on one or more cards. Layers go in order: card 1
gets the first ones, then card 2, and so on. The rest run on the CPU.

- **Graphics card 1** and **Layers kept on card 1**. *CPU only* keeps
  every layer on the processor.
- **CPU threads**: for the layers that run on the CPU.
- **How card N holds its layers**:
  - *A set number*: exactly the number typed in its layers field.
  - *Maximize GPU: as many as fit*: Nexus measures the card's real free
    memory and fills it. See [MAXIMIZE_GPU.md](MAXIMIZE_GPU.md).
- **8-bit prompt math on card N**: *Auto* (default), *On* or *Off*, one
  choice per card. *Auto* uses the card's 8-bit hardware when the driver
  reports it, and otherwise times both ways on that card at startup and
  keeps the faster.
- **Graphics card 2, 3, …**: a new row appears each time you pick a card,
  with the same settings.
- With Maximize on any card, two more settings appear:
  - **Card memory always kept free (MB)**, default 768. Raise it to 1024
    if your desktop runs on that card.
  - **Prompt card: memory for reading a prompt (MB)**, default *auto*.
- **CPU: 8-bit output math**: *Off* (default, full precision), *Auto*
  (measured on this CPU at the first token; kept only if at least 10%
  faster) or *On*. It applies to NQ models while writing.

A running plan is shown as you change things, for example *Plan: card 1:
layers 0-19 | card 2: as many as fit | layers 20-63: CPU | about 9.6 GB
on cards, 7.2 GB in RAM*. You can ask for more layers than fit: the
engine places what it can and says so in the log.

The details, including what each 8-bit option costs, are in
[GRAPHICS_CARDS_AND_SPEED.md](GRAPHICS_CARDS_AND_SPEED.md).

#### Prompt speed-up (prefill acceleration)

Separate from the card above: layers are swapped through a card one at a
time while a long prompt is read, then writing happens where the layers
live.

- **Prefill acceleration**: *Auto* (recommended) turns it on when it
  helps and says why when it's off: no card found, the whole model is
  already on cards, or the only card is integrated graphics (shares RAM,
  gave wrong answers in testing; choose it by hand to try it anyway).
- **Card to use**: the card that reads prompts. Pick your fastest card
  that has the best PCIe link. It can also hold layers for writing.
- **Layers swapped through the card at once**: leave it empty unless
  you're experimenting.
- **Attention path**: *Auto* (short prompts batch), *Stream* or *Batch*.
- **8-bit prompt math on this card**: only shown when the prompt card
  isn't already one of the cards above (then its row there holds it).
- **Split long prompt steps**: *Auto (AMD cards only)*, *On* or *Off*.
  Keeps each step under the AMD Linux driver's time limit.
- **Prompt working space on the card**: *Fit to what's really free*
  (default, safer) or *By the card's size* (older, fuller).
- **Oversized-prompt protection** and **Drop layers from RAM below (MB
  free)**: when free RAM falls under the number while a prompt is read,
  layers move off RAM and onto the card afterwards, instead of the
  system swapping. Linux. See [MAXIMIZE_GPU.md](MAXIMIZE_GPU.md).

#### Share with other computers (donors)
Nexus's distributed mode: one model's layers split across two or more
computers on your network. See *Multi-computer setup* below.

#### Model behaviour
- **Chat format**: detected from the model file by default (shown as
  *Auto-detect → X (confidence)*). You can force one from the full list
  (ChatML, Qwen3, Llama-3, Mistral, Mistral V3, Gemma, Gemma 4, Phi-3,
  Phi-4, DeepSeek variants, OLMoE, Zephyr, Alpaca, Granite, Raw or
  Custom).
- **RoPE layout** / **Norm convention**: leave on *From the model file*
  unless a model answers with gibberish. Then trying the other option is
  a real diagnostic step (the Log tab shows which one was used).
- **Thinking** / **Keep earlier thinking**: only enabled for models whose
  chat template has a thinking switch.
- **Custom** chat format shows seven text fields (before everything,
  before/after system, before/after user, before/after model).
- All of this is remembered **per model file**, separately from profiles.

#### Conversation memory
- **Conversation length limit**:
  - *The model's maximum* (default): stops there.
  - *A set number of tokens*: used exactly, even above the model's
    maximum.
  - *What fits in memory (no limit)*: grows until memory runs out, past
    the model's maximum if it gets there. Past it, a model will likely
    start repeating itself or fall apart.
- **When the conversation is full, drop the oldest messages and keep
  going** (ticked by default). The system prompt is always kept.
  Unticked, the reply stops when the conversation is full.

Each message in the web chat also has a **⋯** button to forget, archive
or restore it by hand. See [CONVERSATION_MEMORY.md](CONVERSATION_MEMORY.md).
Saved per model.

#### Sampling
Temperature, Top-p, Top-k, Min-p, Tail-free z (1 = off), Repeat penalty
(1 = off), Repeat window, Frequency penalty, Presence penalty, Max
tokens. Each box shows its default when empty.

#### Images (models that can see)
Only for vision-capable models. The image size the model sees, the
largest uploaded edge and area (0 = original), how large images are
handled (*Shrink to one image* or *Split into tiles*, with tile size,
step and the most tiles), and whether the image part runs on the
graphics card.

#### LoRA adapter
Pick a `.gguf` LoRA adapter from the `loras` folder next to the launcher,
or browse to one anywhere, and set its **Strength**. Adapters are
remembered per base model, so switching models never carries one over by
mistake.

#### Speculative decoding (MTP)
Uses the model's own draft heads, if it has them (Qwen3.5 / 3.6, for
example). **Use the model's own draft heads** (Off by default), **Tokens
drafted per step**, and **Draft on** CPU or graphics card. The output is
exactly the same as without MTP. Whether it's faster depends on where
the layers live: see *Speculative decoding* in
[GRAPHICS_CARDS_AND_SPEED.md](GRAPHICS_CARDS_AND_SPEED.md).

#### Saved state (warm state) at startup
Point this at a saved `.kvc` state to have the model start every
conversation already "warmed up": a long system prompt or a reference
document you've paid the reading cost for once. States saved in the web
chat land in that model's states folder.

#### Network
- **Port for model N**: each running model needs its own port.
- **Who can connect** (all models): *Only this computer* (127.0.0.1) or
  *Anyone on my network* (other devices open
  `http://<this computer's address>:PORT`).
- **Browser program** (optional): what *Open chat in browser* launches,
  for example a portable browser on the same USB stick.
- **API key for other computers** (all models): empty means no key.
  **Make a key** creates a random one, **Copy key** copies it, **No key**
  removes it. With a key, programs and pages on *other* computers must
  send it; this computer never needs it.
- **Web pages that may use it**: *This computer and my network*
  (default), *This computer only*, or *Any website (not recommended)*.
  Websites on the internet can't use your models otherwise, even though
  your browser can reach them.
- **Also allow web pages from**: extra page addresses, comma-separated.

Changes apply when a model is started or restarted. The details, and how
to call Nexus from your own app, are in
[ACCESS_AND_API_KEYS.md](ACCESS_AND_API_KEYS.md).

> **Windows note:** the first time another device tries to reach a model
> (or the first time you choose *Anyone on my network*), Windows will
> likely show a **Windows Defender Firewall** prompt for
> `nexus_server.exe` and/or `nexus_launcher.exe`. Click **Allow**. Until
> you do, nothing outside your machine can reach the server, even though
> the launcher shows it running. If you missed the prompt, go to
> **Windows Security → Firewall & network protection → Allow an app
> through firewall** and enable both programs. Also don't assume port
> 8080 is free on Windows: other software often takes it. If a model
> won't start, try a different port first.

#### Speed and accuracy
Two long-standing speed options, both on by default:
- **Store the output layer as Q8_0**: faster, with a tiny rounding change.
- **Use 8-bit math for the K-quant matrices when generating**: for GGUF
  K-quant models (Q4_K and similar), the same trade llama.cpp makes.

Untick either to run the exact, slower form. Takes effect on the next
start.

#### Diagnostics
- **Quick self-test**: sends one fixed short question to the running
  model and reports time to first token, tokens per second, and
  pass/fail/unsure depending on whether the answer was right. It replaces
  the model's current conversation memory, so run it when you don't
  mind that.
- **Read prompts one token at a time**: turns off prompt acceleration
  and batching. Slow; for diagnosis only.
- **Print prompt-speed profile in the log**: extra timing detail.
- **Log detail**: 0 (quiet) to 3 (everything).

#### Advanced
- **Memory safety buffer** (on by default): stops a reply with a clear
  message before the computer runs out of memory and freezes, keeping
  the amount you set (default 1024 MB) free. Turning it off lets the
  model use every last byte: fine on a machine with swap, risky without.
- **Extra command-line options for the server**: anything not exposed as
  its own control, passed straight to `nexus_server`.
- **Extra environment variables**: one `NEXUS_SOMETHING=value` per line.
  Only names starting with `NEXUS_` are accepted; anything else is
  refused with an error before the model starts. The
  [Nexus Guide](NEXUS_GUIDE.md#environment-variables) lists them.

### Profiles

A **profile** is a snapshot of every setting for one model file
(sampling, cards, threads, extra options, everything) under a name you
choose. Keep several configurations for the same model (say *Fast* and
*Max quality*) and switch between them from the **Model** card's
**Profile for this model** drop-down.

- **Save as new profile**: snapshots your current settings under a new
  name.
- **Update this profile**: saves your current settings into the active
  profile.
- **Use automatically for this model**: applies this profile every time
  you pick this model, in any slot.
- **Delete**: removes the active profile (after asking).

With no profile selected, every setting stays exactly as you left it,
including across a model switch. Nexus never changes your settings on
its own.

### Multi-computer setup (donors)

Nexus can split one model's layers across several computers: your main
("primary") computer plus one or more "donor" computers lending their
CPU and graphics card. The **Share with other computers (donors)** card
(Setup tab, per model) and the **Donor** tab (on the lending computer)
set it up between them.

**On the donor computer:**
1. Start Nexus there with `nexus_launcher --donor` (or open the **Donor**
   tab).
2. Under **Connection**: a listen port (default 9876), who may connect,
   and a **passphrase**: the same word you'll enter on the primary. The
   passphrase is never written to disk.
3. Under **Model**: either tick *I already have the model file on this
   computer* and pick it (only instructions cross the network, not the
   weights: much faster, and a partial file holding only this computer's
   layers works too), or leave it unticked and let the primary send the
   weights.
4. Under **Hardware**: CPU threads, optionally a graphics card and how
   many layers to keep on it, the memory safety buffer, and whether this
   donor uses its card for prompt acceleration too.
5. Click **Start donor**. It shows this computer's network address (with
   a Copy button) and waits. The status line updates live (receiving
   weights, loading, connected, working), with a progress bar while
   weights load.

**On the primary computer**, in the model's **Share with other computers
(donors)** card:
1. Tick **Share this model with other computers**.
2. Pick a **Split**: *Equal share for everyone*, or *I choose how many
   layers each donor takes* (a **Layers** box per donor, and one for
   this computer).
3. **Add donor computer** (or **Find donors on my network** to find
   anything listening on the donor port) and fill in each donor's
   address, port and passphrase. Tick *Has the model* if that donor has
   its own copy. **Test** checks that the address answers.
4. A **Layer plan** table shows which layers go to this computer and to
   each donor, and roughly how much memory each range needs.
5. Two prompt-reading settings matter on a multi-computer setup:
   - **Reading long prompts**: in pieces (1024 tokens, recommended; 512;
     2048), so both computers work at the same time, or *Whole prompt,
     one layer at a time*, if the log shows a lot of "load-wait" from
     constant layer reloads.
   - **Read long prompts entirely on THIS computer** (hybrid prefill):
     this computer reads the *whole* prompt (it needs its own full copy
     of the model) and hands the donor the finished result for its
     layers, once, at the end. Worth trying when this computer's card is
     clearly faster at reading prompts. It only kicks in from the token
     count you set (default 2048). If this computer lacks the memory to
     also hold the donor's layers for that moment, it frees some of its
     own, streams the donor's layers through, and reloads its own, all
     shown in the log.
6. If sharing is on but a donor's address or passphrase is missing,
   nothing starts. Nexus never quietly loads the whole model on the
   primary instead.

While running, the primary's **Log** tab shows percent, layer, tokens
per second and time left for its own layers and, labelled separately,
any real wait on a donor. The donor's **Donor** tab shows its own
progress bar while it loads.

### Log tab

The same model slot strip as Setup, plus the live log for the selected
model:
- **Clear** / **Copy all**.
- **Hide model-loading progress lines**: collapses the thousands of
  per-tensor loading lines into the progress bar above.
- A **Reading prompt NN%** bar appears during a long prompt, with tokens
  per second and time left. While layers move onto a card after a prompt
  (Maximize GPU), it says so. During a real wait on a donor it shows a
  plain "waiting on the other computer" bar.
- Click and drag to select text (right-click for Copy / Select all /
  Clear selection; double-click selects a line; Ctrl+C copies).
- The view follows new lines unless you've scrolled up; then a **Jump to
  latest** button appears.

The lines worth knowing are listed under *Reading the log* in
[GRAPHICS_CARDS_AND_SPEED.md](GRAPHICS_CARDS_AND_SPEED.md#reading-the-log).

### Donor tab

Covered under *Multi-computer setup* above. Only needed on the computer
lending its hardware.

### The web chat

**Open chat in browser** opens the chat built into the server. Its
sidebar has three tabs:

- **Sessions**: save and load whole conversations (memory and messages),
  warm states, checkpoints, and export the chat as a web page or
  Markdown.
- **Memory**: how much the model remembers, the limit and auto-trim, and
  your saved memories (archives). See
  [CONVERSATION_MEMORY.md](CONVERSATION_MEMORY.md).
- **Settings**: model info, OCR for documents, live sampler settings,
  display options and a system prompt.

Drag documents or whole folders into the message box: see
[DOCUMENTS_IN_CHAT.md](DOCUMENTS_IN_CHAT.md). **Send + Think** asks a
reasoning model to think before answering, for that message only.
**Connect to session** reattaches to a reply the server is still writing
after you closed or reloaded the page.

### Command-line options for the launcher itself

```
nexus_launcher                       normal start
nexus_launcher --donor               start on the Donor tab
nexus_launcher --dir <path>          use a specific folder instead of the launcher's own folder
nexus_launcher --shot out.ppm        render one frame to an image file and exit (testing only, no window)
   [--tab setup|log|donor] [--slot N] [--w N --h N] [--expand] [--scale F]
```

`--shot` is a developer and testing aid. Everyday use is just
`nexus_launcher`, or `nexus_launcher --donor` on a lending computer.

---

## Tips

- The first time you run the launcher, read the coloured notices at the
  top of the Setup tab. They catch the common first-run snags (server
  program not found next to the launcher, no model in the models folder
  yet, hardware check still running) before they turn into a confusing
  error later.
- If a model answers with gibberish right after loading, the **Model
  behaviour** card's RoPE layout / Norm convention are the first thing to
  try. The Log tab says which option the engine used.
- Settings persist across restarts only after **Save current settings**.
  That's deliberate: experiment freely (a different temperature, a test
  card split, an extra variable) without changing anything for good
  until you're sure.
- Adding a second graphics card can make prompts a little slower if your
  motherboard splits the first slot to x8. See
  [Slots and PCIe lanes](GRAPHICS_CARDS_AND_SPEED.md#slots-and-pcie-lanes).
