<p align="center">
  <img src="docs/readme_banner.png" alt="Nexus — the multi-model inference engine for hardware you already own" width="100%">
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-AGPLv3-7c6af7.svg" alt="License: AGPLv3">
  </a>
</p>

Nexus is a free, local, multi-model inference engine built to get real
large-language-model performance out of hardware you already own — a
single modest graphics card, an old laptop, or a CPU alone. No cloud,
no subscription, nothing leaves your machine.

📄 **[Read the full intro (PDF)](docs/Nexus_Intro.pdf)** — a short overview
of what Nexus does and why, including a real-hardware benchmark.

## What makes it different

- **Multi-model** — run several models at once, each in its own
  isolated slot, each with its own GPU placement and settings.
- **Prefill acceleration** — streams a model's layers through a
  graphics card one at a time while reading a prompt, so even a card
  far too small to hold the whole model still gives GPU-class
  prompt-reading speed.
- **Persistent KV memory** — the model's conversation state is
  retained and reusable. Read a huge document once, then every
  follow-up question skips that cost entirely. Save it to disk and a
  reload drops you right back into the conversation — no
  re-processing.
- **Multi-computer inference** — split one model's layers across two
  or more computers on your network, turning spare hardware into a
  model neither machine could run alone. Even a small (1 GB) graphics
  card on the primary computer accelerates prefill for the *whole*
  setup, while a GPU-less donor only has to carry its share of
  generating the reply.

## Getting started

1. Grab a release build (or build from source — see below).
2. Drop a `.gguf` or `.nxm` model into the `models` folder next to the
   launcher, or point it at your own model folder.
3. Pick GPU placement, hit Start, and open the chat in your browser.

Full walkthrough: **[Launcher Guide](docs/LAUNCHER_GUIDE.md)**

### Converting your own models

Nexus reads GGUF directly, or its own native `.nxm` format for the
best quantization and packaging options (including splitting a model
across several computers ahead of time). See the
**[Quant Builder Guide](docs/QUANT_BUILDER_GUIDE.md)**.

### Running headless / building your own front end

Nexus's server exposes a full HTTP/SSE API, and its `Engine` C++ class
can be linked directly with no HTTP layer at all. See the
**[Engine / API Guide](docs/ENGINE_API_GUIDE.md)**.

## Building from source

```
# see CMakeLists.txt for build instructions
```

## License

Nexus is released under the **GNU Affero General Public License v3.0**
(AGPLv3) — see [`LICENSE`](LICENSE) for the full text.
