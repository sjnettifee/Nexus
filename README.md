<p align="center">
  <img src="docs/readme_banner.png" alt="Nexus — the multi-model inference engine for hardware you already own" width="100%">
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-AGPLv3-7c6af7.svg" alt="License: AGPLv3">
  </a>
  <a href="https://github.com/sjnettifee">
    <img src="https://img.shields.io/badge/author-Shea%20Nettifee-7c6af7.svg" alt="Author: Shea Nettifee">
  </a>
</p>

<p align="center"><sub>Designed and built by <strong>Shea Nettifee</strong> — a year-long solo project.</sub></p>

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

## Troubleshooting

- **Windows: another device on my network can't connect.** Windows
  Defender Firewall will prompt the first time this happens — you
  have to click **Allow**, or it'll quietly block the connection with
  no error on the Nexus side. Missed the prompt? Go to **Windows
  Security → Firewall & network protection → Allow an app through
  firewall** and enable `nexus_server.exe`/`nexus_launcher.exe`
  manually. See the [Launcher Guide](docs/LAUNCHER_GUIDE.md#network)
  for more.
- **Port 8080 won't bind / model won't start.** Something else on
  your machine (common on Windows) may already be using it. Pick a
  different port in the launcher's Network card.

## Building from source

Requires CMake 3.18+, a C++17 compiler, [Ninja](https://ninja-build.org/),
and a Vulkan shader compiler (`glslc` or `glslangValidator`) for GPU
support. Linux/macOS/Windows (MSYS2 MinGW64 shell):

```sh
scripts/build_release.sh            # builds CPU tiers v2 v3 v4 -> ./dist
scripts/build_release.sh v3         # just one tier
OUT=/path/to/folder scripts/build_release.sh
```

This builds `nexus_server` (the small dispatcher every user actually
runs — no `-march`, works on any CPU) plus a `nexus_server_v2/v3/v4` +
`libnexus_v2/v3/v4` pair for each CPU tier (SSE4.2 / AVX2+FMA /
AVX-512); the dispatcher picks the best one your CPU supports at
startup. Everything lands in `dist/` (and your project root), ready to
zip or copy to a USB stick.

Prefer driving CMake yourself instead of the script:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DNEXUS_VULKAN=ON
cmake --build build --parallel
```

Useful CMake options: `-DNEXUS_TIER=v2|v3|v4|native`,
`-DNEXUS_VULKAN=ON` (GPU support — off by default), `-DNEXUS_CUDA=ON`
/ `-DNEXUS_ROCM=ON` (optional, off by default), `-DNEXUS_PYTHON=ON`
(pybind11 bindings, off by default).

## License

Nexus is released under the **GNU Affero General Public License v3.0**
(AGPLv3) — see [`LICENSE`](LICENSE) for the full text,
[`NOTICE`](NOTICE) for copyright and authorship, and
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for the bundled
Dear ImGui and Roboto components.

## Author

Nexus was designed and built by **Shea Nettifee**
([sjnettifee@gmail.com](mailto:sjnettifee@gmail.com)) over roughly a
year of solo development, with AI assistance (Claude, and earlier
development help from ChatGPT) used throughout as a development tool.
