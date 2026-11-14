# Nexus Native Quant Builder — User Guide

Turns a HuggingFace model (safetensors) into a native Nexus package (`.nxm` /
`.nxmodel`) that the Nexus engine loads directly — no GGUF step, no donor
model, no renaming or fusing of tensors. Every tensor keeps its original
HuggingFace name and shape; the only thing this tool changes is how each
tensor is *stored* (full precision, or one of four quantized levels).

This guide covers the builder itself: what every control does, what the
output actually contains, and how to pick settings. It does not cover
running the resulting package (the launcher and the engine/API guides cover
that).

## Requirements

- Python 3
- `tkinter` (ships with most Python installs; on some Linux distros it's a
  separate package — `sudo apt install python3-tk` or equivalent)
- `numpy` (only needed when you actually quantize something — `KEEP_SOURCE`
  copies don't touch it, but the tool will tell you to install it the moment
  you need it: `python3 -m pip install numpy`)

Run it with:

```
python3 nexus_native_quant_builder.py                        # from inside nexus_quant_builder/
python3 nexus_quant_builder/nexus_native_quant_builder.py    # from the Nexus folder
```

## Opening a model

**Open HF Folder / Safetensors** accepts either:
- A HuggingFace model folder (one or more `.safetensors` shards, plus
  `config.json`, tokenizer files, etc.)
- A single `.safetensors` file

The scan reads only the safetensors headers (tensor names, shapes, dtypes,
byte offsets) — it does not load any weights into memory at this stage, so
scanning is fast and cheap even for large models. Once scanned, the tool
reads `config.json` (and `text_config`/`language_model_config`-style nested
configs some multimodal models use) to work out layer count, head counts,
rope theta, and so on, and looks up the architecture's RoPE layout
(`half_split` vs `interleaved`) and norm style (`direct` vs `zero_centered`)
from a small built-in table, since those two conventions aren't written down
anywhere in the weights themselves — the engine needs them and can't guess.

**Base name** fills in automatically from whatever you opened — the model
folder's name, or the file's name if you opened a single `.safetensors`
(falling back to the containing folder's name if the file is an anonymous
shard like `model-00001-of-00004.safetensors`). It's a plain text field
afterward; change it if you want something else. It resets to the new
model's name every time you open a different model.

## The tensor table (right side)

Every tensor in the model, one row each, with:

| Column | Meaning |
|---|---|
| Layer | Decode layer number, blank for global tensors (embeddings, output head, vision/audio tensors, etc.) |
| Component | `language`, `visual`, `audio`, `mtp`, or `other` |
| Class | What kind of tensor this is — see below |
| Decision | What will actually happen to it (an `*` prefix means it's a manual override) |
| Auto | What the automatic rules alone would pick (useful to compare against an override) |
| Src | Source dtype (F32/F16/BF16/etc.) |
| Src MiB / Out MiB | Size before and after |
| Shape | Tensor shape |
| Tensor | Full HuggingFace tensor name |
| Reason | Why this decision was made |

**Filter** does a substring search across name/component/class/decision/reason.
**Component** and **Decision** narrow the table to one value each. **Manual
Only** shows just the rows you've hand-overridden. Click a column header to
sort by it. Double-click a row (or select several and use the Manual
Decision panel on the left) to override it directly.

Tensor **class** is worked out from the name: `embedding`, `norm`, `bias`,
`output` (the final lm_head / output projection), `ssm_control` (DeltaNet
gate/decay vectors), `conv1d` (short depthwise kernels), `rope_freq`,
`ssm_matrix` (big SSM projection matrices), `attention`, `ffn`, or a generic
`matrix`/`other` for anything that doesn't match a known pattern. This
classification is what the per-category quant dropdowns (below) key off of.

## Auto Quant Rules (left side)

### The quant levels

| Decision | What it stores |
|---|---|
| `KEEP_SOURCE` | Byte-for-byte copy, original dtype untouched |
| `KEEP_F32` | Upconverted to plain F32 |
| `Q4G64` / `Q5G64` / `Q6G64` / `Q8G64` | Quantized: symmetric signed 4/5/6/8-bit integers, one F16 scale per 64 values, quantized along the tensor's last dimension |
| `SKIP` | Tensor dropped from the package entirely |

The number after Q is bits per value — Q4G64 is the smallest/fastest/least
accurate, Q8G64 the largest/most accurate of the quantized options. These
are Nexus's own scheme (`NQ4_G64` etc. in the manifest), not GGUF's K-quants
or any llama.cpp format — nothing to match up with external tooling here.

### The six policy dropdowns

Each one picks the quant level for one tensor class, when nothing else
overrides it first:

- **Default matrix** — anything that's a big 2-D+ matrix but doesn't fall
  into one of the more specific categories below
- **Embeddings** — the token embedding table
- **Output/lm_head** — the final output projection
- **FFN** — gate/up/down projection matrices
- **Attention** — Q/K/V/O projections (and the equivalent linear-attention
  in/out projections)
- **SSM matrix** — large DeltaNet/SSM projection matrices (not the small
  control vectors — those are handled separately, see below)

**Full precision** picks what "kept at full precision" actually means
everywhere else in this panel: `KEEP_SOURCE` (byte-identical to the
HuggingFace file) or `KEEP_F32` (upconverted to F32, useful if you want
everything comparable at the same precision regardless of what the source
checkpoint shipped in).

**Min matrix MiB** is a size floor, not a class. Below it, a tensor keeps
full precision regardless of what its class-specific dropdown says — this
exists because quantizing something tiny saves almost nothing and the
group/scale overhead can actually make it *worse* proportionally.

### Order these rules are actually applied in

This matters because several of these can apply to the same tensor, and the
first match wins. For every tensor, in order:

1. Its **component** (language/visual/audio/mtp/other) is excluded via the
   checkboxes below → `SKIP`, nothing else runs.
2. It's 1-D and **"1D tensors"** is checked → full precision.
3. It's below **Min matrix MiB** and **"Below threshold"** is checked →
   full precision.
4. It's a **norm**, **bias**, **SSM/DeltaNet control**, or **Conv1D kernel**
   and the matching checkbox is on → full precision.
5. It's a RoPE frequency table or other non-matrix oddity → full precision
   (always, not optional — there's nothing sensible to quantize here).
6. Otherwise, its **class** picks the dropdown above: embedding / output /
   ffn / attention / ssm_matrix / default matrix.

So, for example, even if you set "FFN" to Q4G64, an individual FFN tensor
under the size floor still comes out at full precision — the floor runs
before the class rule gets a turn. The **Reason** column in the tensor table
tells you exactly which rule fired for each row.

### Components

Turns off entire feature modules rather than individual tensors — useful if
you're shipping a text-only build and don't want vision/audio/speculative
weights taking up space at all. Any component unchecked here is fully
`SKIP`ped, before any of the per-class quant rules even run.

### Keep Source Precision

The five checkboxes feeding rules 2–4 above, plus **"Below threshold"**
(the Min matrix MiB floor). All five are on by default — the defaults assume
you want the small, sensitive stuff (norms, biases, control vectors, conv
kernels, anything under the size floor) left alone, and only the big matrix
multiplies actually quantized.

### Manual overrides

Select one or more rows in the tensor table, pick a decision in **Manual
Decision**, and **Set Selected** applies it to exactly those tensors — shown
with a `*` prefix in the Decision column from then on, and immune to
**Apply Auto Decisions** re-running (which only touches non-manual rows).
**Revert Selected to Auto** clears the override and **Clear Manual Overrides**
clears all of them at once, model-wide.

## Native Output (left side)

### Mode

| Mode | What you get |
|---|---|
| **Single native file** | One `.nxm` file. Builds a temporary `.nxmodel` folder, packs it into the single file, then deletes the temp folder. This is the normal choice for a package that'll be copied around as one file. |
| **Native directory package** | A `.nxmodel/` folder — manifest, `metadata/`, and one `.nxw` weights file — left as-is, not packed. |
| **Multipart by size** | Same folder layout, but the weight payload is split across multiple `.nxw` files, each capped at **Max part GiB**. This is the *only* mode that field does anything in. |
| **Remote two-computer split** | Two folders: `<name>.local.nxmodel/` and `<name>.remote.nxmodel/`. Neither is packed to a single file — both stay as open directories. |
| **Split across several computers** | N self-contained pieces, each packed to its own single `.nxm` file. |

**Base name**, **Output folder**, and **Refresh Estimate** (recomputes the
plan summary below, including auto decisions, without needing a full Apply
or Build) are available regardless of mode.

### Remote two-computer split

Produces a `local` package (language layers `0` through the split point,
plus embeddings and *all* non-language feature tensors — vision/audio/MTP/
other) and a `remote` package (the tail language layers, plus the final
output projection). **Remote tail layers** sets how many layers go to the
remote side; `0` defaults to half the model. **Duplicate embeddings to
remote** copies the embedding table into the remote package too (off by
default — only turn it on if your remote-side loading path specifically
needs its own copy). Each package's build note spells out exactly which
layers it holds and, for the remote one, that it should be copied to the
other machine.

### Split across several computers

**Pieces (computers)** sets how many pieces to make. **Layers per piece**
lets you force exact counts — e.g. `20,22,22` for three pieces — and the
last piece absorbs any shortfall or gets trimmed for any excess. Leave it
blank and pieces are balanced *by estimated byte size* rather than by raw
layer count, so a model with uneven per-layer sizes (common with MoE or
mixed attention types) still ends up roughly even — piece 1 always carries
the embeddings, output layer, and any vision/audio/MTP tensors, so it's
deliberately given a smaller layer share to compensate. Every tensor lands
in exactly one piece — nothing is duplicated here, unlike the two-computer
mode above.

### Chunk MiB

The read/convert buffer size used while streaming a tensor through an F32
upconversion or a quantization pass. Doesn't change the output, just how
much gets held in memory at once mid-tensor. The default (32 MiB) is a
reasonable middle ground; raise it if you have memory to spare and want
slightly fewer read calls on a very large model.

### USB release note

**Single native file** and **Split across several computers** each pack
down to one `.nxm` file per piece. FAT32 — still the default pre-format on a
lot of USB sticks — caps a single file at just under 4 GiB. Cross that line
and you'll get a plain "no space left" error mid-copy with the drive nowhere
near full, which is a confusing thing to debug after the fact. The builder
now flags this both in the plan estimate (before you build) and in the
build log (after, with the real file size) — if you see the note, use
exFAT or NTFS for that drive, or split into more pieces.

## Building

**Refresh Estimate** shows the plan without building anything — per-piece
layer ranges, tensor counts, and estimated output size, broken down by
component. **Build Native Model** (or the **BUILD** shortcut at the top)
re-applies the auto rules once (so any checkbox/threshold change you made
is reflected) and runs the actual build in the background — progress bar,
status line, and a running log all update live. **Cancel** stops cleanly
between tensors; nothing partially written is left in a weird state.

## After a build

**Export Policy JSON** saves everything — settings, output settings, and
every tensor's decision — to a JSON file. **Import Policy JSON** reads one
back and reapplies it to the *currently scanned* model by matching tensor
names, so you can save a policy once and reapply it (say, after a clean
re-download, or when quantizing a sibling/fine-tuned model with identical
tensor names) without redoing manual overrides from scratch. It also
recognizes a few older/alternate decision spellings (`KEEP_F16`→
`KEEP_SOURCE`, `NQ4_G64`→`Q4G64`, etc.) so a policy from an earlier version
of this tool still imports cleanly.

**Export CSV** writes the full tensor table to a spreadsheet-friendly file —
useful for review, or for sharing exactly what a given build decided without
handing over the whole policy JSON.

## Picking quant levels: a practical starting point

The shipped defaults are a reasonable starting point and match what's been
tested: `Q4G64` for FFN, attention, SSM matrices, and the generic default;
`Q5G64` for embeddings and the output head (these sit closest to the
model's actual input/output and tend to be more sensitive); everything
small/structural (norms, biases, control vectors, conv kernels,
sub-threshold tensors) left at full source precision. If you want a quick
higher-quality / larger-file variant, raising **Default matrix**, **FFN**,
and **Attention** to `Q5G64` or `Q6G64` is the first knob to try before
touching anything in Keep Source Precision — those checkboxes exist because
the tensors behind them are small enough that quantizing them barely moves
the file size, but can disproportionately affect quality.

## Known limitations

- Only this tool's own `NQ4/5/6/8_G64` scheme is supported right now — no
  GGUF-style K-quants (`Q4_K`, etc.) or the smaller legacy formats
  (`Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`, `IQ4_XS`/`IQ4_NL`). Format expansion is a
  deferred, future item, not something this build supports yet.
- Quantizing a 1-D tensor isn't supported and isn't attempted — if a manual
  override tries to force a quant level onto one anyway, it's copied at
  source precision instead rather than silently doing something wrong.
- RAM-overflow behavior (a model larger than the build or target machine's
  memory) hasn't been tested yet.

## Troubleshooting

- **"numpy is required for quantized build"** — `python3 -m pip install
  numpy`. Only needed once you quantize something; `KEEP_SOURCE`-only
  builds don't need it.
- **Scan fails / "No .safetensors files found"** — you pointed it at a
  folder with no `.safetensors` shards in it (a GGUF-only folder, or the
  wrong directory level).
