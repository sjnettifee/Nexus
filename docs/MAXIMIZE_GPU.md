# Maximize GPU and oversized-prompt protection

Two ways to use a graphics card's memory as extra room for the model, so a long prompt fits on a computer that would otherwise run out of RAM. For example, a 27B model reading a 100,000-token prompt on a 25 GB machine.

Both keep the **prompt speed-up** (prefill acceleration): while a prompt is read, the card has the room the speed-up needs, and once the prompt is read, layers are put on the card for writing. Linux only.

For setting up several cards, the 8-bit options and the other speed settings, see [GRAPHICS_CARDS_AND_SPEED.md](GRAPHICS_CARDS_AND_SPEED.md).

---

## The idea

During a long prompt, RAM fills up with the model's weights, the conversation's memory (the KV cache, several GB at 100k tokens) and the prompt's working buffers. Once RAM is full, the system starts swapping, and writing slows to a crawl.

Nexus can move some layers out of RAM entirely:

- A moved ("parked") layer has **no copy in RAM**. Its weights are read straight from the model file whenever they're needed.
- While the prompt is read, parked layers stream through the card like every other layer, reading from the drive instead of RAM.
- When the prompt is read, there is a **pause** while the parked layers load from the drive onto the card. Then writing starts, with those layers computed on the card and the rest on the CPU from RAM.

The pause depends on the drive. 2.5 GB takes a couple of seconds from an NVMe drive and about 20 seconds from a hard disk.

---

## The two modes

### Maximize GPU

*Launcher → Run on graphics card(s) → How card N holds its layers → **Maximize GPU: as many as fit***

Every card in the list has its own setting: **A set number** (the layers field, as before) or **Maximize GPU**, which fills that card with as many layers as really fit for writing. You can mix them, for example card 1 on Maximize and card 2 with 8 layers, or Maximize on all of them.

What Maximize does depends on the card:

- **Any card that isn't the prompt speed-up card:** as many layers as fit go onto it and stay there (`--gpu2-layers auto`, `--gpu3-layers auto`, …). The amount is measured from the card's real free memory, keeping *Card memory always kept free*. These layers stream through the prompt card while a prompt is read, like any other card's layers, so nothing moves on or off them.
- **The prompt speed-up card:** its layers live there for writing **with no copy in RAM**, and move off only as far as a prompt needs (below).

The cards are filled in order: the other cards first, then the prompt card takes as many of the remaining layers as fit.

**MTP:** if *Speculative decoding (MTP) → Draft on: Graphics card* is on, MTP drafts on the prompt card. Maximize leaves room there for the MTP head (the log says `[park] keeping N MB free on the card for MTP drafting`), so it doesn't fight the layers for space.

When a prompt is read, the prompt card needs room for the prompt speed-up (*Prompt card: memory for reading a prompt*: **auto** works out exactly what it needs, or set a number, e.g. 1536; a number bigger than the card clears every Maximize layer off it for each prompt). If that room is already free, nothing moves. If not, **just enough layers move off** to make it. The rest stay on the card, and the prompt is read through them there. Afterwards, the moved ones go back on, with a pause while they load from the drive.

This room is **only what the prompt speed-up uses plus this conversation's memory (KV) at its new length**, not the whole card. The layers that fit beside it stay put.

Example: on a 12 GB card, about 10 GB of layers live on the card. A prompt moves about 1.5 GB of them off, is read, and they go back. Most of the writing then happens on the fast card.

With "as many as fit", the card is full, so most prompts move a few layers and pause. To avoid the pause, set *Prompt card: memory for reading a prompt* to a number: that much stays free from the start, and nothing moves.

### Oversized-prompt protection

*Launcher → Prompt speed-up → **Oversized-prompt protection: On**, with **Drop layers from RAM below (MB free)***

Everything starts in RAM as normal, and **nothing happens until free RAM actually drops under your number** (1024 MB by default) while a prompt is read. There's no prompt-length rule. Then layers are dropped from RAM:

- first the layers this prompt is already finished with, since reading never needs them again;
- then, only if that isn't enough, layers it hasn't reached yet, which are read from the drive when their turn comes.

After the prompt, its working buffers are freed and **RAM is checked again**. Dropped layers go back into RAM first, as far as RAM allows while keeping your number free plus room for the reply's own memory (4,096 tokens' worth). Only the rest go onto the card, as far as the card really has room. **Whatever the card can't take goes into RAM, with no exceptions.**

**The rule: the drive is only a temporary home while a prompt is read.** Before the first token is written, every layer is on the card or in RAM. Nothing is read from the drive while writing. This also holds when a prompt read is stopped partway, and for any part of a card layer the card can't hold (that part keeps a RAM copy). If RAM runs out, that's your call: with the memory safety buffer off, nothing stops it; with it on, the safety buffer stops the reply first.

Layers that protection put on the card stay there for later replies. A later prompt gets the card's room back automatically, only if and when it needs it.

You can use both modes together.

---

## Card memory: what's really free

When the graphics driver supports it (`VK_EXT_memory_budget`, which most current drivers do), Nexus asks the card how much is **really** free, counting your desktop and every buffer Nexus has made. Layers only go on while that leaves *Card memory always kept free*. Filling a card past its real memory either fails or makes the driver quietly spill into system RAM, after which everything on the card crawls.

Without driver support, Nexus estimates from the card's size. The log says *(estimated)*, and it's worth raising the kept-free number if your desktop runs on the same card.

**While reading a prompt**, the prompt speed-up's working space on the card is capped at what the card *really* has free, minus room for attention. That room grows with the conversation (about 400 MB per layer at 100,000 tokens), minus a safety margin. *Prompt speed-up → Prompt working space on the card → By the card's size* switches back to the older, fuller sizing.

**While writing**, every 128 tokens:
- If the prompt card's real free memory falls under 256 MB, one parked layer moves off it into RAM.
- If card 2, 3, … falls under 256 MB really free, its last layer goes back to the CPU (those layers keep their RAM copy).
- If protection is on and free RAM falls under your number, because the reply's memory keeps growing, one layer moves from RAM onto the card, but only if the card has room for it. It never goes to the drive.

## Settings

| Setting (launcher) | Server flag | Default | Meaning |
|---|---|---|---|
| How card N holds its layers: Maximize GPU (prompt card) | `--park-layers N\|auto` | off | N layers (auto = as many as fit) live on the prompt card for writing. The launcher sends `--gpu-layers 0` with it. |
| How card N holds its layers: Maximize GPU (other cards) | `--gpu2-layers auto`, `--gpu3-layers auto`, … | set number | As many layers as fit on that card, kept there. |
| Oversized-prompt protection | `--park-below-mb MB` | off | Drop layers from RAM while reading when free RAM is under this. |
| Prompt card: memory for reading a prompt | `--park-prefill-free-mb MB\|auto` | auto | Room the prompt speed-up gets on the prompt card. Layers move off only when it isn't free already. |
| Card memory always kept free | `--park-keep-mb MB` | 768 | Left free on every Maximized card (activations, the GPU copy of the conversation memory). MTP drafting on the card gets its own room on top. |
| (server only) | `--park-min-tokens N` | 0 (off) | Optional: prompts shorter than N never move layers and are read on the CPU instead. |

Both modes need the prompt speed-up on (`--turbo-prefill`) with a discrete card. They are off for integrated graphics (it shares RAM, so nothing is freed) and when this computer runs only part of the model (shared with donors). Other cards can hold layers at the same time, set or Maximized.

If the **memory safety buffer** (launcher, Advanced) is on, set its number below the protection's number, so that layers are dropped before the safety buffer stops the reply.

---

## What the log shows

```
[park] ON: 14210 MB of layer weights in RAM; 9 layers (2304 MB) parked for the card, with no copy in RAM. ...
[park] model loaded: moving 9 layers (2304 MB) onto the card for writing -- reading them from the model file ...
[park] prompt of 86000 tokens: the prompt speed-up needs 1180 MB of the card (auto); 5 parked layers (1280 MB) moved off it, 1350 MB is free now; 4 parked layers stay on the card ...
[park] free RAM 980 MB is under 1024 MB: layer 4 (256 MB) dropped from RAM -- this prompt is done with it; it goes onto the card after the prompt ...
[progress] moving layers onto the card: 50%  (layer 4, 4 of 8, 1024 MB, 3s elapsed)
[park] prompt read; RAM has room again: 6 dropped layers (1536 MB) back into RAM (free RAM now 2310 MB, keeping 1280 MB for the reply)
[park] 2 of 2 layers (512 MB) on the card in 1.9 s; card memory free now: 690 MB
[park] the card is nearly full while writing (52 MB free): layer 9 (256 MB) moved off it, back into RAM (card free now 310 MB)
```

During the pause, the web chat's status bar shows *prompt read · moving layers onto the card 4 of 8*. The code hardener shows the same.

When it can't run, the log says why, and the model loads normally: `[park] Maximize GPU / oversized-prompt protection is OFF: <reason>`.

---

## How it works

Every big layer weight is mapped straight from the model file:

- **In RAM:** a normal layer's pages are copied into RAM once (copy-on-write).
- **Dropping a layer** (`MADV_DONTNEED`) throws that copy away. The pointers stay valid, and the next read comes from the file, so a dropped layer can still be used anywhere.
- **On the card:** layers are pinned there, so the prompt stream never pushes them out.

This depends on weights never being changed in memory after loading, which is true for Nexus.
