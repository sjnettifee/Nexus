# Mixture-of-experts models

*New in Nexus 1.0.1.*

A mixture-of-experts (MoE) model, such as **Qwen3.6-35B-A3B**, is mostly *experts*: small blocks of weights, hundreds per layer, of which each token uses only a few. A 35B-A3B holds 35 billion weights but uses about 3 billion per token. That's why it can run fast on a modest computer, if the right weights are in the right place.

Nexus 1.0.1 puts them in the right place:

- The **attention** part of every layer runs on its graphics card, like any other model's layers.
- The **experts** all stay in RAM, and the ones that matter also get a copy on your cards. You choose which ones, in two ways (below).
- For each token, **every card runs the experts it holds in one go, while the CPU runs the rest at the same time.**
- **Reading a prompt** gets the prompt speed-up too. Each expert runs once over all the tokens that picked it, instead of once per token.

For setting up cards in general, see [GRAPHICS_CARDS_AND_SPEED.md](GRAPHICS_CARDS_AND_SPEED.md). For Maximize GPU, see [MAXIMIZE_GPU.md](MAXIMIZE_GPU.md).

---

## What it does on real hardware

Qwen3.6-35B-A3B GGUF, 25 GB of RAM, Linux, default settings:

| Cards | Reading the prompt | Writing |
|---|---|---|
| GTX 1060 3 GB + R9 290 4 GB | (not measured) | about **6 tokens/s** |
| GTX 1060 3 GB alone | over **40 tokens/s** | about **5 tokens/s**, sustained through a 10,000-token reply |

In 1.0.0, MoE layers ran on the CPU and prompts were read one token at a time.

---

## The two ways to choose experts

*Launcher → Run on graphics card(s) → **Experts on the cards (MoE models)***

### Picked by each prompt (default)

After a prompt is read, the experts that prompt used most are moved onto the cards, and experts it didn't use make room for them. The idea is that the reply is about the same thing as the prompt, so it mostly uses the same experts.

- Before the first prompt, the cards get an even spread of experts, so the first reply isn't all CPU.
- Any prompt of 16 tokens or more triggers a re-pick. Experts that are already on a card win ties, so nothing moves without a reason.
- While writing, the choice is checked again every 256 tokens, so a reply that drifts to a new topic catches up.
- Usage counts halve at every check, so a new topic takes over quickly.

Moving experts takes a moment once the prompt is read: it's a copy from RAM to the card. The log reports how long (`[moe] prompt-picked ... (1500 MB, 400 ms)`).

Best for: most chats. One conversation tends to stay on one subject.

### Fixed home

A set of experts is put on the cards when the model loads and stays there. Nothing ever moves. Next to it, **Home experts per layer** chooses which:

- **all** (default): every expert of the first layer, then the next layer, and so on, until the cards are full. Those layers then run entirely on a card.
- **a number**, like `32`: that many experts of *every* layer (experts 0 to 31 of each), until the cards are full.

Best for: comparing speeds, or a server answering many unrelated questions where re-picking would chase each one.

### CPU only

Experts never go on a card. The attention still runs on the cards. Use it to compare, or when the cards are needed for something else.

### In every mode

- The **shared expert** of each layer (one that every token uses) goes on a card first.
- **Cards are filled fastest first.** Nexus times how fast each card reads weights at startup and logs it as `how fast each card reads weights while writing`.
- **Every card's room is what's free after its layers and the output matrix, minus *GPU prompt processing space*.** Experts never take the room a prompt needs.
- **The prompt card (Maximize GPU) holds experts too.** Its experts step aside while a prompt is read (the prompt speed-up owns that card then) and come back right after it, at the same time as the layers.
- **All experts stay in RAM**, because the CPU computes every expert that isn't on a card. Plan on RAM for the whole model file. A 35B-A3B at 4 bits needs about 20 GB.

---

## Reading a prompt

A prompt is read in slices of up to 512 tokens. For each layer:

1. The attention runs on the prompt card, like any model's.
2. The router picks each token's experts.
3. Each expert runs once over just the tokens that picked it:
   - on the card that holds it, if it's on one (nothing to upload);
   - streamed through the prompt card, if at least **24** tokens of the slice picked it, so the upload pays for itself;
   - otherwise on the CPU, at the same time as the cards.

Each card gets its own worker, and the CPU works alongside them.

The log shows the split for the first few layers:

```
[moe:prompt] L=0 512 tokens: experts used 256 -- 40 on the cards that hold them, 180 streamed through the prompt card, 36 on the CPU
```

---

## Settings

### Launcher

| Setting | Choices | What it does |
|---|---|---|
| Experts on the cards (MoE models) | **Picked by each prompt** / Fixed home / CPU only | Which experts get a copy on the cards. Other models ignore it. |
| Home experts per layer | **all** / a number | Only with *Fixed home*: whole layers until the cards are full, or N experts of every layer. |

Both sit under *Run on graphics card(s)*, next to *Output matrix (lm_head)*. The **?** next to each one has the short version of this page.

### Server

| Flag | Default | What it does |
|---|---|---|
| `--moe-experts prompt\|home\|off` | `prompt` | Which experts get a copy on the cards |
| `--moe-home N\|all` | `all` | With `--moe-experts home`: N experts of every layer, or whole layers until the cards are full |

### Environment variables (for tuning and comparing)

| Variable | Default | What it does |
|---|---|---|
| `NEXUS_MOE_STREAM_MIN` | `24` | While reading a prompt: how many tokens must pick an expert before it is streamed through the prompt card instead of run on the CPU. Lower it if the CPU is the slow part of reading prompts. Raise it if your card sits on a slow PCIe link. |
| `NEXUS_MOE_RECHECK` | `256` | *Picked by each prompt*: written tokens between re-checks |
| `NEXUS_MOE_EXPERTS` | `prompt` | Same as `--moe-experts` (the flag wins) |
| `NEXUS_MOE_HOME` | `all` | Same as `--moe-home` |
| `NEXUS_MOE_BATCH` | `1` | `0` reads MoE prompts one token at a time, as 1.0.0 did |
| `NEXUS_MOE_ATTN_GPU` | `1` | `0` runs MoE attention on the CPU, as 1.0.0 did |
| `NEXUS_MOE_GPU_EXPERTS` | `0` | `1` turns the old 1.0.0 per-matrix card path back on. It made about a thousand card round trips per token, so it's off. |

---

## Reading the log

At load:

```
[moe] 120 expert tensors (19800 MB) kept in RAM: the CPU computes the experts that are not on a card
[moe] experts on the cards (--moe-experts prompt, an even spread until the first prompt): 1830 of 10280 (40 shared), 3100 MB, 1.2 s | room: NVIDIA GeForce GTX 1060 3GB 900 MB, AMD Radeon R9 290 2200 MB
```

After each prompt (*Picked by each prompt*):

```
[moe] prompt-picked (prompt read): 912 experts moved onto the cards (1500 MB, 400 ms), 900 made room, 310 back after stepping aside for the prompt | of the expert uses since the last check 18% were on a card, now 46% would be
```

- **were on a card**: how much of the prompt's expert work the cards held before the move.
- **now ... would be**: how much the cards hold after it. This is the number that shows the trick working: the higher, the faster the reply.

While writing (after the 10th token):

```
[decode] time per written token (avg of 8): layers 150.2 ms + final norm/lm_head on NVIDIA GeForce GTX 1060 3GB 9.8 ms = 160.0 ms (6.2 tokens/s before sampling)
[decode] experts run on a card so far: 44% (35200 of 80000)
```

With *Fixed home*, a prompt on the Maximize card prints `[moe] N of N home experts back on the card after the prompt`.

(The numbers above show the format. Yours depend on the model, cards and prompt.)

---

## Which models

Built and tested with **Qwen 3.5 / 3.6 MoE** models: GGUFs made by llama.cpp (Q4_K_M, Q8_0, F32) and Nexus's own. Other MoE models (OLMoE, Qwen3-MoE, Mixtral-style) also get their experts on the cards. How much of their attention runs on a card depends on the model.

An expert goes on a card only if the card has a kernel for its weight type:

- **On a card:** Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, NQ4, NQ5, NQ6, NQ8.
- **CPU only:** F32, F16, BF16, Q4_0, Q5_0.

llama.cpp's Q4_K_M stores some experts' last matrix as Q5_0 or Q8_0. The Q5_0 ones stay on the CPU, which is correct, just slower. An all-K-quant or NQ model gets the most out of the cards.

Models that store an expert's gate and up matrices as one tensor, or store experts transposed, keep their experts on the CPU for now.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `[moe] experts: CPU only -- no card has room for them` | The cards are full with layers and the output matrix, or *GPU prompt processing space* takes the rest. Lower it a little, or use fewer layers on a set-number card. |
| `(no expert went on a card: their weight types have no card kernel ...)` | The model's experts are F16/BF16/Q4_0/Q5_0. Use a K-quant or NQ version of the model. |
| The `now ... would be` percentage stays low | The cards only fit a small share of the experts. Add a card, free card memory, or try *Fixed home → all* to keep whole layers on the card. |
| The pause after the prompt is long | That's the experts moving onto the cards. Use *Fixed home* to skip it, or `NEXUS_MOE_RECHECK` to make in-reply checks rarer. |
| `[moe] <card> would not run its experts for layer N -- done on the CPU this time` | The card refused one batch (usually out of memory). The CPU did that work, so the output stays correct. If it keeps happening, raise *GPU prompt processing space*. |
| Reading prompts is slow and the CPU is busy | Lower `NEXUS_MOE_STREAM_MIN` (e.g. 12) so more experts go through the prompt card. |
