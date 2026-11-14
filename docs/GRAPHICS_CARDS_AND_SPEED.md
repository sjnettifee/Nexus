# Graphics cards and speed settings

Nexus splits a model between your CPU and as many graphics cards as you give it. Older cards count too: a $20 AMD R9 290 next to a 3 GB GTX 1060 runs a 27B model faster than either one can alone.

This page covers how the work is shared out, what every speed setting does, how to read the log, and how to set up the hardware. To use a card's memory as extra room for very long prompts, see [MAXIMIZE_GPU.md](MAXIMIZE_GPU.md).

---

## How the work is shared

A model is a stack of layers. Every layer lives in one of three places:

- **On a graphics card.** You choose how many layers each card holds, in *Run on graphics card(s)*. While writing, a layer is computed on the card it lives on.
- **In RAM.** Every layer that isn't on a card is computed by the CPU.
- **On the drive, briefly.** Only with Maximize GPU or oversized-prompt protection, and only while a prompt is read (see MAXIMIZE_GPU.md). Nothing is ever read from the drive while writing.

There are two jobs, and they use the cards differently:

| | Reading a prompt (prefill) | Writing the reply (output) |
|---|---|---|
| What runs | Thousands of tokens at once, layer by layer | One token at a time through every layer |
| Where | **One card**, the *prompt speed-up card*. Every layer streams through it, including layers that live on other cards or in RAM. | Each layer where it lives: card 1, card 2, … or the CPU |
| What limits speed | That card's computing power and its PCIe link | Mostly memory speed: how fast each layer's weights can be read |

So adding a second card speeds up **writing** (more layers on fast card memory, fewer on the CPU) and makes long prompts easier on the computer, because the load is split. **Prompt** reading stays on one card.

---

## Setting up the cards

### Run on graphics card(s)

- **Graphics card 1, 2, 3, …**: a new row appears each time you pick a card. Layers go in order: card 1 gets the first ones, then card 2, and so on. The rest run on the CPU.
- **How card N holds its layers**:
  - *A set number*: exactly that many layers, as typed in the layers field.
  - *Maximize GPU: as many as fit*: Nexus measures the card's real free memory and fills it, keeping *Card memory always kept free*. Details in MAXIMIZE_GPU.md.
- **8-bit prompt math on card N**: Auto / On / Off, chosen per card. See *8-bit math* below.
- **CPU threads**: for the layers that run on the CPU.
- **CPU: 8-bit output math**: Auto / Off / On. See *8-bit math* below.

When the model loads, the log shows each extra card being filled:

```
[load] 40%  Moving layers onto graphics card 2 (6 of 15)
[nexus] graphics card 2 (AMD Radeon R9 200 Series (RADV HAWAII)): 15 layers (3420 MB) on the card for writing in 4.1 s, card memory free now: 412 MB
```

If a set number doesn't fit, the card takes as many layers as fit and the log says so. The rest run on the CPU.

### Prompt speed-up (prefill acceleration)

- **Card to use**: the card that reads prompts. Pick your fastest card that has the best PCIe link (see *Slots and lanes*). It can also hold layers for writing.
- **Layers swapped through the card at once**, **Attention path**: leave these on their defaults unless you're experimenting.
- **8-bit prompt math on this card**: shown only when the prompt card isn't one of the cards in *Run on graphics card(s)*. Otherwise its row up there holds the setting.
- **Split long prompt steps** and **Prompt working space on the card**: see below.

Every setting with a **?** next to its name has a note. Hover over the **?** to read it, or click to keep it open.

---

## 8-bit math

Graphics cards and CPUs have fast integer units that multiply 8-bit numbers several times faster than full-size ones. To use them, Nexus rounds the numbers flowing *between* layers (the activations) to 8 bits. **The model's weights are never changed.** The cost is a small loss of precision: about 0.5% per matrix on a card, and about 1% on the CPU. That's the same trade llama.cpp makes, and K-quant GGUF models (Q4_K and similar) always use it on the CPU.

Whether it's actually faster depends on the hardware, so every place it applies offers **Auto**, which measures instead of guessing.

**8-bit prompt math on card N** (reading prompts):
- *Auto* (default): if the driver reports hardware for it, it's used. Otherwise Nexus times both kinds of math on that card at startup and keeps the faster one. This matters because some older cards only fake 8-bit in software and get *slower* (an R9 290: 13 tok/s with it, 30 without), while cards that do have the hardware (NVIDIA since Pascal, for example) aren't always reported as having it by their driver.
- *On* / *Off*: your choice. *Off* means full precision always.
- The log says what was chosen: `[vulkan] 8-bit prompt math: measured on this card ...: 8-bit X ms, regular Y ms -> ...`.

**CPU: 8-bit output math** (writing, for NQ models):
- *Off* (default): full precision.
- *Auto*: at the first token, Nexus times both on your CPU with your thread count. It keeps 8-bit only if it's **at least 10% faster**, because it costs a little precision. On machines where memory speed is the bottleneck, faster arithmetic buys nothing, and Auto will say so: `[cpu] 8-bit output math (Auto): measured on this CPU with 8 threads: full precision … ms, 8-bit … ms -> …`.
- *On*: always. CPUs without AVX2 ignore it.

---

## Other speed settings

| Setting | Choices (default first) | What it does | Command line |
|---|---|---|---|
| Split long prompt steps | **Auto (AMD cards only)** / On / Off | If one step of a long prompt takes over 3 s, the rest of that layer runs in smaller steps (never below 256 tokens). AMD's Linux driver resets a card after 10 s on one job. Other cards don't need it, and smaller steps are slower. | `NEXUS_PREFILL_TILE_MAX_S=3` (on) / `0` (off); `NEXUS_PREFILL_TILE_MIN` |
| Prompt working space on the card | **Fit to what's really free** / By the card's size | How much of the prompt card the prompt speed-up may fill. *Really free* leaves room for the desktop and for long chats (attention needs about 400 MB per layer at 100,000 tokens). *By the card's size* is fuller and can run out. | `NEXUS_TURBO_BUDGET_REAL=0` = by size |
| Card memory always kept free | **768** MB | Left free on every Maximized card. Raise it to 1024 if the desktop runs on that card or you see out-of-memory messages. | `--park-keep-mb` |
| Prompt card: memory for reading a prompt | **auto** / a number in MB | With Maximize on the prompt card: how much room a prompt gets. *Auto* clears just enough. A number bigger than the card clears every Maximize layer off it for each prompt. | `--park-prefill-free-mb` |
| Speculative decoding (MTP) | **Off** / On, tokens per step, draft on CPU / card | See below. | `--mtp`, `--mtp-n-predict`, `--mtp-gpu` |

---

## Speculative decoding (MTP)

Some models (Qwen3.5, for example) come with their own small draft heads. With MTP on, they guess the next few tokens. The model then checks the real token and all the guesses in **one** pass through its layers. Every guess that matches is a token you get for free.

- **Lossless.** A guess is only kept if it is exactly what the model itself produces at that position, with the same sampler and the same repetition penalties. The text is the same as without MTP.
- **Nothing is redone.** When a guess misses, the tokens before it are kept from that one pass. Only the linear-attention layers' running state is replayed (cheap arithmetic), not the layers.
- **When it helps:** when checking several tokens costs about the same as writing one. That's true when most layers are on graphics cards, where reading the weights is the cost, and the extra tokens ride along nearly free. When most of the model runs on the CPU, the extra tokens cost real arithmetic, and MTP comes out about even. Try 2 drafts instead of 3. *CPU: 8-bit output math* may tip it, so measure.
- **Draft on: Graphics card** uses the prompt card. Maximize leaves room there for the draft head: `[park] keeping N MB free on the card for MTP drafting`.

Each step logs `[mtp-time] step=… propose=…ms verifyN(ntok=…)=…ms`. Compare that with a run with MTP off.

---

## Older AMD cards (GCN 1 and 2: R9 290, R9 390, HD 7000, …)

They work, and they make good helper cards for writing. Nexus handles their quirks automatically:

- **8-bit:** these cards only fake it in software. *Auto* measures that and uses the regular prompt kernels.
- **Driver time limit:** AMD's Linux driver resets a card after 10 s on the graphics queue, and 60 s on the compute queue. Nexus uses the compute queue on AMD cards (`NEXUS_VK_GFX_QUEUE=1` restores the old choice), and *Split long prompt steps* (Auto) keeps every step short. If a reset happens anyway, the log says so in plain words.
- **Clock speeds:** an MSI R9 290 GAMING runs 977 MHz core (1007 in OC mode) and 1250 MHz memory ("5000" effective). Numbers printed on some boxes aren't real clock speeds.

---

## Slots and PCIe lanes

The prompt card gets the model's weights sent to it over PCIe for **every prompt**, so **its** link matters. A helper card's link barely matters. Its layers are sent once, when the model loads, and while writing only a tiny amount crosses the bus.

Many motherboards share 16 lanes between the first two long slots. With one card, slot 1 runs at x16. Put a card in slot 2, and both run at x8. That costs the prompt card bandwidth.

A real example: an 11,400-token prompt on a 27B model read in **251 s** with the GTX 1060 alone at x16, and **~270 s** with an R9 290 added in slot 2, both at x8. Writing was faster, and the computer stayed usable during long prompts. A worthwhile trade, but worth knowing.

Ways to keep the prompt card at x16:
- Put the helper card in a slot that **doesn't** share lanes with slot 1. On many boards that's the bottom x4 slot, wired to the chipset. Check your motherboard manual's slot table. x4 is plenty for a helper card.
- If a two-slot-wide card won't physically fit there, a **PCIe riser cable** lets it sit elsewhere in the case.

To see the current link:
```
nvidia-smi -q | grep -A3 "Link Width"        # NVIDIA: Max vs Current
sudo lspci -vv | grep -E "LnkCap|LnkSta"     # any card: capability vs actual status
```
Cards often drop to a slower link when idle to save power, so check while a prompt is being read.

---

## Memory safety on the cards

- **While reading a prompt**, the prompt speed-up only fills what the card *really* has free (the driver reports it), minus room for attention that grows with the conversation, minus a safety margin. The log says `[turbo:auto-budget] … source=real-free`.
- **While writing**, every 128 tokens: if the prompt card has less than 256 MB really free, one Maximize layer moves to RAM. If card 2, 3, … has less than 256 MB free, its last layer goes back to the CPU. The log says which.
- **MTP drafting on the card** gets its own room on the prompt card.
- The conversation memory's copy on a card has its own conservative cap, and attention for layers past that cap runs on the CPU.

---

## Reading the log

| Line | Meaning |
|---|---|
| `[load] NN%  Moving layers onto graphics card N (k of n)` | Filling a helper card while the model loads |
| `[nexus] graphics card N (...): K layers (... MB) on the card for writing ...` | How many layers each helper card got |
| `[vulkan] 8-bit prompt math: ...` | What 8-bit choice each card made, and why |
| `[cpu] 8-bit output math (Auto): ...` | What the CPU measured and picked |
| `[progress] reading prompt: NN% (layer i of n, ... tok/s, ...)` | Prompt progress, every 2 s |
| `[turbo:auto-budget] ... source=real-free` | The prompt speed-up's room on the card |
| `[prefill] layer N: a ...-token slice took ... s ...; smaller slices ...` | *Split long prompt steps* kicked in |
| `[park] ...` | Maximize GPU / protection moving layers (see MAXIMIZE_GPU.md) |
| `[nexus] graphics card ... is nearly full while writing ...` | The watcher gave a layer back to the CPU |
| `[vulkan] cache full and nothing it can evict ...` | The card is completely full. Raise *Card memory always kept free*. |
| `[mtp-time] ...`, `[mtp-diag] ... MATCH / miss` | MTP timing and which guesses were kept |

---

## Troubleshooting

- **A helper card stays empty.** Look for its `[nexus] graphics card N …` line at load. A "doesn't fit" message means it had less free memory than the layers need, so lower its layer count or use Maximize.
- **Prompts got slower after adding a card.** Check the prompt card's PCIe link (see *Slots and lanes*).
- **Out-of-memory messages, or the card sits right at its limit.** Set *Card memory always kept free* to 1024, keep *Prompt working space* on "really free", and if the desktop runs on that card, give it a little more.
- **An AMD card resets during a long prompt** ("context is lost"). Set *Split long prompt steps* to On.
- **MTP is slower than writing without it.** On CPU-heavy splits, that's expected. Try 2 drafts, try *CPU: 8-bit output math* on Auto, or turn MTP off.
- **Prompt reading is slower than expected on an NVIDIA card.** Make sure its *8-bit prompt math* isn't set to Off, and check the `[vulkan] 8-bit prompt math` line.
