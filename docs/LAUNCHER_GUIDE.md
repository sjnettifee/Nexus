# Nexus Launcher Guide

Welcome to **Nexus** — a free, local, multi-model inference engine built
to put real LLM inference in the hands of anyone, not just people with a
multi-thousand-dollar GPU rig. Nexus runs on whatever you already have —
a modest graphics card, an old laptop, or just a CPU — and it can even
split one large model across two or more computers you already own, so
hardware you'd normally call "too weak" becomes part of the answer
instead of a reason to give up. No subscription, no cloud, no sending
your prompts anywhere. This is local AI, democratized.

This guide covers the **launcher** — the native window you use to pick
a model, configure it, and start it. If you want to skip the launcher
entirely and run the server from the command line or your own front
end instead, see `ENGINE_API_GUIDE.md`. If you're converting/quantizing
your own models, see `QUANT_BUILDER_GUIDE.md`.

---

## Quick Start

This gets you from "just installed" to "chatting with a model" in a
few steps. Everything mentioned here is covered in more depth further
down for when you want it.

### 1. Put a model where the launcher can find it

The simplest option: drop a `.gguf` or `.nxm` model file into the
**`models`** folder that sits right next to the launcher program (it's
created automatically the first time you run it). The launcher scans
that folder — and one folder up — automatically.

If your models live somewhere else (another drive, a USB stick, a
folder you organize yourself), open the **Setup** tab → **Model** card
→ **Add folder...** and point it at that folder instead. Your own
added folders are always searched *first*, and once you add one it's
remembered for next time.

### 2. Pick your model

Still on the **Model** card, use the **Model file** drop-down — it
lists everything found in your models folder(s), with each file's size
and location. Pick one. Or, if you'd rather not browse, type/paste a
full path directly into the box below it (works for a path on another
drive too, e.g. `D:\models\model.nxm`).

The moment you pick a model, the launcher reads its metadata (size,
layer count, detected chat format) and shows it right there, and it
tries to automatically pick sensible GPU settings for you (see step 3).

### 3. Choose where it runs (CPU vs. GPU)

Open **Run on graphics card(s)**. The first time you pick a model, the
launcher already tried to do this for you:

- If the model fits comfortably on your best graphics card, it's
  placed there entirely.
- If it's too big for the card, it stays on the CPU, and the card is
  instead used to speed up reading long prompts (see step 4) rather
  than being wasted.

You're free to override this. **Graphics card 1** + **Layers kept on
card 1** controls how many of the model's layers live on that card (0
= none, i.e. CPU only). As you type a number, the launcher tells you
roughly how many GB that is and whether it'll actually fit. You can
add a second, third, etc. card the same way — extra card pickers
appear automatically once you've filled in the one before it.

### 4. Prefill acceleration (prompt speed-up)

Open **Prompt speed-up (prefill acceleration)**. This is a *separate*
feature from step 3 — it uses a graphics card to speed up *reading*
long prompts (the "prefill" phase), even if the model itself is
running entirely on the CPU, by swapping layers through the card one
at a time. It needs only enough room for a single layer, so almost any
card can help here, even a small one.

Leave it on **Auto** — it turns itself on when it'll actually help
(i.e. when the GPU isn't already holding 100% of the model) and off
when it wouldn't. You only need to touch this if you want to force it
on or off.

### 5. Basic model settings

Open **Sampling** for the usual knobs: **Temperature**, **Top-p**,
**Top-k**, **Min-p**, repeat penalty, max tokens, etc. Sensible
defaults are already filled in (temperature 0.2, top-p 0.95, and so
on) — you don't have to touch any of this to get started, but it's all
right there when you do.

### 6. Launch

Look at the top-right corner of the window — there's always a small
floating status readout with a **Start** button for whichever model is
currently selected. Click **Start**. The launcher switches you to the
**Log** tab automatically so you can watch it load (you'll see a
progress bar while weights load). Once the status dot turns green and
says "Running," click **Open chat in browser** right next to it — this
opens the built-in chat UI in your browser, pointed at that model.

That's it — you're chatting with a locally-running model. Everything
past this point is for when you want more control.

---

## Detailed Reference

### The window itself

At the top: the Nexus logo/name, and a **Save current settings**
button. Nexus only writes your settings to disk when you press this —
until then, anything you change is in effect immediately (so you can
experiment freely) but will be forgotten if you close the launcher
without saving. The button turns amber with a `*` the moment anything
is unsaved, and the tooltip tells you which state you're in.

Settings are stored in `nexus-data/nexus_launcher.json` next to the
launcher (the same folder the server itself uses for chats, projects,
and sessions, since the server is started with that as its working
directory). If that folder can't be written to (read-only USB stick,
permissions issue), you'll see a banner at the top of the Setup tab
telling you so — the launcher still works, it just can't remember
anything between runs.

Three tabs: **Setup**, **Log**, **Donor**.

### Running more than one model at once

The row of buttons right under the top bar (on the Setup and Log tabs)
is the **model slot strip**. Nexus can run several independent models
side by side, each its own server process on its own port, each with
entirely separate settings. Click **+ Add a model** to add another
slot, click any slot's button to switch which one you're configuring,
and **Remove this model** to drop the last slot (only the last one can
be removed, and only while it's stopped). Each slot remembers its own
model, GPU placement, sampling settings, donors, everything.

### Start / Stop / status (top-right floating panel)

Visible on the Setup and Log tabs. Shows a colored status dot (gray =
stopped, amber = loading, green = running and ready, red = stopped
with a non-zero exit code) and text for whichever slot you're currently
looking at, plus:
- **Start** / **Stop** for that one model
- **Open chat in browser** (only once it's ready) — opens and copies
  the chat URL (`http://127.0.0.1:<port>/`)
- **Start all** / **Stop all**, shown whenever you have more than one
  model slot — starts or stops every slot that has a model chosen

### Setup tab — the cards

Cards collapse/expand individually (click the header). They apply to
whichever model slot is currently selected.

#### Model
- **Model file** drop-down (your scanned models) or a typed/pasted
  path underneath it for anything not in a scanned folder.
- **Browse...** / **Rescan** buttons next to the drop-down.
- Once a model is loaded, a line of metadata: architecture, size on
  disk, layer count (and roughly how big one layer is), and the
  detected chat format with a confidence level (low/medium/high).
- **Profile for this model** — see "Profiles" below.
- **Model folders** — the list of extra folders you've added, each
  removable with its little `x` button, plus **Add folder...**. The
  `models` folder next to the launcher is always searched too, in
  addition to anything you add here.

#### This computer
Read-only hardware summary: CPU brand/core count, total/free RAM, and
every graphics card the hardware probe found, each marked usable or
not (with a reason when not — e.g. a driver issue). **Re-check
hardware** re-runs the probe if you've changed something physically
(added a card, plugged in an eGPU) since starting the launcher.

#### Run on graphics card(s)
Place model layers on one or more graphics cards. **Graphics card 1** +
**Layers kept on card 1**, then **CPU threads** for whatever doesn't
live on a card. Pick a second, third, etc. card the same way — each
new card picker appears once the previous one is set. You can
overcommit (ask for more layers than fit); the engine places what it
can and logs the rest. A running total ("Plan: card 1: layers 0-19 |
card 2: layers 20-31 | layers 32-63: CPU") and an estimate of VRAM vs.
RAM usage is shown live as you adjust the numbers.

#### Prompt speed-up (prefill acceleration)
Separate mechanism from the card above — swaps layers through a card
one at a time (or a few at once, via "Layers swapped through the card
at once") purely to speed up reading long prompts, regardless of where
the model itself runs. **Auto** (recommended) turns it on exactly when
it would help and tells you why it's off otherwise (no card found, the
whole model is already on cards with nothing left to swap, or the only
card available is integrated/shared-memory graphics, which gave wrong
answers in testing and needs to be chosen by hand if you want to try
it anyway). **Attention path** lets you force `stream` or `batch`
instead of the automatic choice (short prompts batch by default).

#### Share with other computers (donors)
This is Nexus's distributed/hybrid mode — splitting one model's layers
across two or more computers on your network. See "Multi-computer
setup" below for the full walkthrough; this card is where you turn it
on and configure it for the model you're currently editing.

#### Model behaviour
- **Chat format** — auto-detected from the model file by default
  (shown as "Auto-detect → X (confidence)"); you can force a specific
  one from the full list (ChatML, Qwen3, Llama-3, Mistral, Mistral V3,
  Gemma, Gemma 4, Phi-3, Phi-4, DeepSeek variants, OLMoE, Zephyr,
  Alpaca, Granite, Raw, or Custom).
- **RoPE layout** / **Norm convention** — leave as "From the model
  file" unless a model is producing gibberish, in which case trying the
  other option here is a real diagnostic step (the Log tab shows which
  one was actually used).
- **Thinking** / **Keep earlier thinking** — only enabled for models
  whose chat template actually supports a reasoning/thinking toggle;
  grayed out otherwise.
- If you pick **Custom** as the chat format, seven raw text fields
  appear (before everything / before+after system / before+after user
  / before+after model) so you can hand-build a template fragment by
  fragment.
- All of this is remembered **per model file**, separately from
  whatever profile is active.

#### Sampling
The standard set: Temperature, Top-p, Top-k, Min-p, Tail-free z,
Repeat penalty, Repeat window, Frequency penalty, Presence penalty,
Max tokens. Each box shows its default as a placeholder when empty.

#### Images (models that can see)
Only relevant for vision-capable models. Controls the size the model
actually sees versus what gets uploaded (useful for keeping uploads
small while still giving the model full resolution, or vice versa),
how oversized images are handled (shrink to one image vs. split into
tiles, with tile size/stride/max-tile-count for the latter), and
whether the vision portion runs on the graphics card.

#### LoRA adapter
Pick a `.gguf` LoRA adapter from your `loras` folder (next to the
launcher) or browse to one anywhere, plus a **Strength** (scale)
value. Adapters are remembered per base model — switching models never
carries an adapter over by mistake.

#### Speculative decoding (MTP)
Turns on the model's own draft/speculative-decoding heads if it has
them: **Tokens drafted per step**, and whether drafting runs on the
**CPU** or the **graphics card**. Off by default.

#### Saved state (warm state) at startup
Point this at a previously-saved `.kvc` state file to have the model
start every session already "warmed up" with that context pre-loaded
— e.g. a long system prompt or reference document you've already paid
the prefill cost for once. States saved from the web chat UI itself
land in that model's own states folder automatically.

#### Network
- **Port for model N** — each running model needs its own port.
- **Who can connect** (applies to all models at once): "Only this
  computer" (127.0.0.1 — nothing outside your machine can reach it) or
  "Anyone on my network" (0.0.0.0 — any device on your LAN can open
  `http://<your IP>:PORT`; there's no password, so only use this on a
  network you trust).
- **Browser program** — optional override for what "Open chat in
  browser" launches (useful for a portable browser carried on a USB
  stick alongside Nexus itself).

> **Windows note:** the first time you set "Who can connect" to
> "Anyone on my network" (or the first time another device tries to
> reach a model that's still on "Only this computer," depending on
> your setup), Windows will likely pop up a **Windows Defender
> Firewall** prompt asking whether to allow `nexus_server.exe` (and/or
> `nexus_launcher.exe`) access on private/public networks. You have to
> click **Allow** — until you do, nothing outside (and sometimes
> nothing *inside*) your machine can actually reach the server, even
> though the launcher itself shows it as running. If you miss that
> prompt or dismiss it, go to **Windows Security → Firewall & network
> protection → Allow an app through firewall** and enable both
> programs manually. Also don't assume port 8080 is free on a Windows
> machine — other software (IIS, Skype, various dev tools) commonly
> grabs it. If a model won't start or won't start cleanly, try a
> different port in the **Network** card before assuming something
> else is wrong.
  stick alongside Nexus itself).

#### Speed and accuracy
Two toggles, both on by default: storing the output layer as Q8_0, and
using 8-bit math for K-quant matrices during generation. Both are
small, deliberate speed/accuracy trade-offs — untick either to run the
exact, slower form. Takes effect on the next start.

#### Diagnostics
- **Quick self-test** — a one-click smoke test: sends one fixed short
  question to the running model and reports time-to-first-token,
  tokens/sec, and a pass/fail/unsure verdict based on whether the
  answer was actually correct. Replaces the model's current
  conversation memory, so only run it when you don't mind that.
- **Read prompts one token at a time** — disables prefill
  acceleration/batching entirely; slow, diagnostic only.
- **Print prompt-speed profile in the log** — extra timing detail.
- **Log detail** — 0 (quiet) through 3 (everything).

#### Advanced
- **Memory safety buffer** — on by default. Stops generation with a
  clear message before the computer would actually run out of memory
  and freeze, keeping the amount you set (default 1024 MB) free at all
  times. Turning it off lets the model use every last byte — fine on a
  machine with swap space to fall back on, risky on one without.
- **Extra command-line options for the server** — anything not
  exposed as its own control, passed straight through to `nexus_server`.
- **Extra environment variables** — one `NEXUS_SOMETHING=value` per
  line; only `NEXUS_...`-prefixed names are accepted (anything else is
  rejected with an error before the model starts, so you can't
  accidentally set an unrelated environment variable).

### Profiles

A **profile** is a complete snapshot of every setting for one specific
model file — sampling, GPU placement, threads, extra args/env,
everything — saved under a name you choose. This lets you keep several
different configurations for the same model (e.g. "Fast" vs. "Max
quality") and switch between them instantly from the **Model** card's
**Profile for this model** drop-down.

- **Save as new profile** — snapshots your current settings under a
  new name.
- **Update this profile** — re-saves the active profile with your
  current settings.
- **Use automatically for this model** — when ticked, this profile is
  applied every time you pick this model again (in any slot), without
  you having to select it by hand.
- **Delete** — removes the active profile (with a confirmation).

With no profile selected, every setting simply stays exactly as you
left it, including across a model switch — profiles are opt-in, Nexus
never changes your settings behind your back on its own.

### Multi-computer setup (donors)

Nexus can split one model's layers across several computers — your
main ("primary") computer plus one or more "donor" computers lending
their CPU/GPU. This is what the **Share with other computers (donors)**
card (Setup tab, per model) and the **Donor** tab (on the lending
computer) configure between them.

**On the donor computer:**
1. Launch Nexus there with `nexus_launcher --donor` (or just open the
   **Donor** tab).
2. Fill in **Connection**: a listen port (default 9876), who's allowed
   to connect, and a **passphrase** — the same word you'll enter on the
   primary. The passphrase is never written to disk.
3. Under **Model**, either tick "I already have the model file on this
   computer" and pick/browse to it (so only instructions cross the
   network, not the weights themselves — much faster, and works with a
   partial file that holds only the layers this computer will be given),
   or leave it unticked and let the primary stream the weights over.
4. Under **Hardware**, set CPU threads, optionally a graphics card and
   how many layers to keep on it, the memory safety buffer, and whether
   this donor should also use its card for prompt acceleration.
5. Click **Start donor**. It shows this computer's network address (with
   a one-click Copy button) and sits waiting — status/activity line
   updates live (receiving weights, loading local shard, connected,
   processing, etc.), along with a load progress bar while weights come
   in.

**On the primary computer**, in the model's **Share with other
computers (donors)** card:
1. Tick **Share this model with other computers**.
2. Pick a **Split**: "Equal share for everyone" (automatic) or "I choose
   how many layers each donor takes" (manual, with a **Layers** box per
   donor and a **This computer** box for your own share).
3. Click **Add donor computer** (or **Find donors on my network** to
   auto-discover anything listening on the donor port on your LAN) and
   fill in each donor's address, port, and the same passphrase you set
   on that donor. Tick "Has the model" if that donor already has its own
   copy of the model file (matches "I already have the model file" on
   the donor side). **Test** checks whether that address/port actually
   answers.
4. A **Layer plan** table appears once you have at least one donor
   filled in, showing exactly which layer range goes to this computer
   vs. each donor, and roughly how much memory each range needs.
5. Two extra prompt-reading controls matter for performance on a
   multi-computer setup:
   - **Reading long prompts** — how the prompt is chunked and handed
     back and forth between computers while it's being read (in pieces
     so both computers stay busy at once, or as one whole pass if your
     log shows a lot of "load-wait" time from constant layer reloading).
   - **Read long prompts entirely on THIS computer** (hybrid prefill) —
     instead of splitting prompt-reading itself across computers, this
     computer reads the *entire* prompt alone (it needs its own full
     copy of the model to do this) and only hands the donor the
     finished result for the donor's own layers, once, at the end. Worth
     trying when this computer's card is clearly faster at reading
     prompts than the donor's. Only kicks in once a prompt reaches the
     token count you set here (default 2048) — short prompts gain
     nothing from it. If this computer doesn't have enough spare memory
     to also hold the donor's layers for that final handoff, it
     temporarily frees some of its own already-loaded layers to make
     room, streams the donor's layers through, then reloads its own —
     all visible in the log as it happens.
6. Nothing starts if sharing is on but a donor address or passphrase is
   missing — Nexus refuses rather than silently loading the whole model
   on the primary by itself.

Once running, the primary's **Log** tab shows real-time percent/layer/
tok-s-per-second/ETA progress for both the local-layer phase and
(labeled distinctly) any genuine wait on the donor. The donor's own
**Donor** tab shows a matching progress bar while it loads its own
share.

### Log tab

Same model-slot strip as Setup, plus the live log for whichever model
you're viewing. Useful controls:
- **Clear** / **Copy all**
- **Hide model-loading progress lines** — collapses the thousands of
  individual per-tensor loading lines into just the progress bar above
  (which is still live either way).
- A **Reading prompt NN% (...)** bar appears automatically during a
  long prefill, including tokens-per-second and an ETA once that
  information is available; during a genuine wait on a donor computer
  it switches to a plain, non-animated "waiting on the other computer"
  bar instead of an endless scrolling animation.
- Click-and-drag to select text in the log (right-click for Copy /
  Select all / Clear selection; double-click selects a whole line;
  Ctrl+C copies the current selection).
- The view auto-follows new lines unless you've scrolled up to read
  something, in which case a **Jump to latest** button appears.

### Donor tab

Covered fully under "Multi-computer setup" above. Only reachable/usable
on the computer that's lending its hardware to another's model.

### Command-line options for the launcher itself

```
nexus_launcher                       normal start
nexus_launcher --donor               start directly on the Donor tab
nexus_launcher --dir <path>          use a specific folder instead of the launcher's own folder
nexus_launcher --shot out.ppm        render one frame to an image file and exit (testing only, no window)
   [--tab setup|log|donor] [--slot N] [--w N --h N] [--expand] [--scale F]
```
`--shot` is primarily a developer/testing aid (renders the UI to a
`.ppm` image non-interactively). Everyday use is just `nexus_launcher`,
or `nexus_launcher --donor` on a lending machine.

---

## Tips

- The first time you ever run the launcher, read the colored notices at
  the top of the Setup tab — they catch the common first-run snags
  (server binary not found next to the launcher, no model in the models
  folder yet, hardware scan still running) before they turn into a
  confusing error later.
- If a model answers with gibberish right after loading, the **Model
  behaviour** card's RoPE layout / Norm convention overrides are the
  first thing to try, and the Log tab will tell you which option the
  engine actually went with.
- Settings only persist across restarts once you click **Save current
  settings** — this is deliberate, so you can experiment freely (a
  different temperature, a test GPU split, an extra env var) without
  worrying about permanently changing anything until you're sure.
