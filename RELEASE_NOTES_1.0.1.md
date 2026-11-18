# Nexus 1.0.1

## Mixture-of-experts models now run on your graphics cards

MoE models like **Qwen3.6-35B-A3B** now use the cards. The attention part of every layer runs on a card. The experts each prompt uses are moved onto the cards once it's read, or you can keep a fixed set there. For each token, the cards and the CPU work at the same time. Long prompts on MoE models also get the prompt speed-up.

On a 25 GB desktop with a 3 GB GTX 1060 and a $20 R9 290, a 35B-A3B GGUF writes about **6 tokens/s**. On the GTX 1060 alone, it reads prompts at **over 40 tokens/s** and writes about **5 tokens/s**, sustained through a 10,000-token reply.

New launcher setting: *Experts on the cards (MoE models)*: Picked by each prompt / Fixed home / CPU only. Guide: [docs/MOE_MODELS.md](docs/MOE_MODELS.md)

## Also in this release

- **llama.cpp Qwen 3.5 / 3.6 GGUFs load and compute correctly.** Merged tensors are read in place. The value-head order llama.cpp uses for 9B / 27B is detected automatically.
- **Your own system prompt per model** in the launcher. Also, models are no longer told they're "Qwen, by Alibaba" unless they are.
- **Writing is much faster with cards:**
  - Qwen 3.5 layers on a card now really run there. Two small BF16 tensors had sent them to the CPU.
  - The output matrix goes on the fastest card.
  - A Qwen 3.5 9B on two small cards went from about 3 to about 17 tokens/s.
  - A 27B now writes a steady 2 tokens/s.
- **One number for card memory**, *GPU prompt processing space*, instead of three. With Maximize GPU, card 1 now gets its share before card 2.
- Fixed: Q4_0 / Q5_0 weights decoded in the wrong order.
- Fixed: on a second card, the conversation state wasn't read back between turns.

Full list: [docs/CHANGELOG.md](docs/CHANGELOG.md)
