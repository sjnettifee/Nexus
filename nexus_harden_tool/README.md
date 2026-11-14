# Nexus Code Hardener

Point it at a codebase and let local models read all of it, one chunk at a time, unattended. Every answer is kept. When the chunks are done, a bigger model can read all the results and write one report, and you can keep talking to that model about the report afterwards.

It was built to check Nexus itself before release. One run read the whole Nexus tree, about 1.64 million tokens in 20 chunks, overnight on a Qwen 9B, with a 27B writing the final report. It found a real command-injection bug that's now fixed.

Everything runs on your own computers. No code leaves them.

---

## Contents

1. [What you need](#what-you-need)
2. [Starting it](#starting-it)
3. [A run, step by step](#a-run-step-by-step)
4. [Models](#models)
5. [Prompt chains](#prompt-chains)
6. [Files that are too big](#files-that-are-too-big)
7. [Running: watching, stopping, redoing](#running-watching-stopping-redoing)
8. [The final pass](#the-final-pass)
9. [Follow-up questions to the final model](#follow-up-questions-to-the-final-model)
10. [Saved reports and exports](#saved-reports-and-exports)
11. [Settings that stick](#settings-that-stick)
12. [Models on other computers](#models-on-other-computers)
13. [A good security-audit workflow](#a-good-security-audit-workflow)
14. [Trusting the results](#trusting-the-results)
15. [Where things are stored](#where-things-are-stored)
16. [Troubleshooting](#troubleshooting)

---

## What you need

- **Python 3.8 or newer.** The standard library only; nothing to `pip install`.
- **Nexus**, with at least one model. The hardener can start models itself (it uses Nexus's `nexus_server`) or use servers you started in the launcher, on this computer or others.
- **A browser.**

## Starting it

Put the `nexus_harden_tool` folder inside your Nexus folder, or one or two levels below it. It then finds the server program and the launcher's model list on its own.

```bash
cd nexus_harden_tool
python3 server.py              # serves http://127.0.0.1:8787/
python3 server.py --port 8790  # another port
```

Open `http://127.0.0.1:8787/` in a browser. The tool only listens on this computer.

If it can't find Nexus, a note at the top says so. Move the folder next to the launcher, or use servers that are already running (see [Models](#models)).

---

## A run, step by step

The page opens clean, with nothing loaded. If you've used it before, **Pick up where you left off** at the top offers your last project: **Open it**, or choose another folder to start a new one. **New project** (next to Load) clears the page at any time; it deletes nothing, and every project keeps its results.

The page walks you through numbered cards.

1. **Project.** Pick the folder to check: type it, click **Browse…**, or choose from **Recent projects**. Then **Load / create project**. Each project keeps its own results and settings.
   - **Token cap / chunk:** how big each chunk may be. About 100k is a good spot for a 9B model.
   - **Large-context cap / part:** used when a big file is split for the large-context model.
   - **File types** and **folders to skip** have sensible defaults: `.git`, `build`, `node_modules`, `third_party`, …
2. **Models.** Choose the model for each phase (next section).
3. **Scan & tokenize.** Every file is counted with the chunk model's own tokenizer. Files bigger than the cap are listed for a decision (see [Files that are too big](#files-that-are-too-big)).
4. **Prompt chains.** What to ask about every chunk, and optionally what to ask in the final pass.
5. **Build chunks, then Run.** Files are packed into chunks that fit the cap, keeping files whole.
6. **Final pass** (optional). **Build final chunk(s) from the results**, then **Run final pass**. Or tick *Run the final pass automatically when the chunks finish* and walk away.

---

## Models

There are three phases, each with its own model:

| Phase | Reads | Typical choice |
|---|---|---|
| **Chunk model** | every normal chunk | a fast mid-size model (e.g. a 9B) |
| **Large-context model** | files or parts too big for the chunk cap | a model with a long context; can be the same as the chunk model |
| **Final-pass model** | all the chunk results together | your strongest model (e.g. a 27B) |

There are three ways to provide each model:

- **Load a model file (this tool starts and stops it).** Pick a file from the launcher's model list. Set the port, threads, graphics card, *whole model on the card*, *prompt acceleration*, thinking, sampling, max output tokens, and the advanced RoPE / norm / chat-format settings. **From the launcher** copies a model's launcher settings in one click.
- **Use a server that's already running.** Give its URL, or pick it from *Running in the launcher now*. The tool uses it but can't stop it.
- **Custom launch command.** Any command; the tool starts and stops it.

**Test** checks that a model answers and shows its context size. **Start** starts it now, so you can watch it load.

**Model swapping.** With *Swap models one at a time* ticked (the default), the tool stops one model before starting the next. So a run can go 9B → long-context → 27B even if they don't all fit in memory together. *Stop models this tool started when a run finishes* frees the memory afterwards. The final-pass model is the exception: it is kept loaded for [follow-up questions](#follow-up-questions-to-the-final-model) unless you untick that option.

**Max output tokens** cuts an answer off at that length and moves on. Useful against a small model that gets stuck repeating itself.

---

## Prompt chains

A **chain** is a list of prompts asked one after another about the same chunk. The model remembers the chunk and its earlier answers, so later prompts can say "review your findings above". Every answer is kept, and the last one is the chunk's result.

- **Per-chunk chain:** asked on every chunk. For a single sweep, leave just one prompt.
- **Final-pass chain:** asked on the final chunk (all the results together).

**Saved chains.** Above each chain is a dropdown of saved chains:

- **Load** puts a saved chain in (it asks first if that would replace what's there).
- **Save as…** stores the current chain under a name.
- **Delete** removes one of yours.

The built-in chains can't be deleted:

| Chain | For |
|---|---|
| *Built-in: security audit* (per chunk) | Vulnerabilities with file, function, quoted lines and the input that triggers them; then a self-review that drops weak findings; then "could any of this reach code execution?" |
| *Built-in: security audit + second look* (per chunk) | The same, plus a last pass looking only for what was missed: outside input reaching a shell, a path, a size or index, or another process unchecked |
| *Built-in: notes & personal-info sweep* (per chunk) | Personal information, notes to self, TODO/FIXME, debug leftovers, anything unprofessional, quoted with file and line |
| *Built-in: combine & rank findings* (final) | Findings that combine across files, then everything ranked by risk |
| *Built-in: consolidate into one report* (final) | One clean report with every distinct issue once, marking anything that needs checking |

Saved chains work in every project.

---

## Files that are too big

After a scan, each file over the cap needs a decision:

| Choice | What happens |
|---|---|
| **Exclude it** | Skipped. |
| **Split into parts for the chunk model** | Cut into line ranges that each fit the chunk cap. |
| **Split into parts for the large-context model** | Cut into parts that fit the large-context cap. A 300k file with a 250k cap becomes 2 parts, not 4. |
| **Run it whole on the large-context model** | One piece, if it fits that model. |
| **Force it on the chunk model as its own over-cap chunk** | Sent whole to the chunk model anyway (it may not fit its context). |

Splitting keeps some headroom, about 15%. Splitting is fine for sweeps (notes, personal info, local bugs), but a single part can't see the rest of its file. Large-context work runs after all the normal chunks, once the tool has switched models.

---

## Running: watching, stopping, redoing

**Watching.** A chunk being worked on shows a live line:

- **Model loading** (when this tool started it): *Starting the chunk model … 45% Loading weights*.
- **Reading the prompt:** *● reading prompt 61% · 104.3 tok/s · about 9m 40s left*.
- **Writing:** *● writing · 120/1000 tok · 38.5 tok/s*.

Under each chunk, a box shows the model's output as it's written. It follows the text down as it comes, in raw and rendered view alike; scroll up and it stays where you put it, scroll back to the bottom and it follows again. Every finished step keeps a *read … tok/s · wrote … tok/s* tag, so the morning results also show how fast each chunk went.

**Stopping.**

- **Stop after this chunk** finishes the current chunk, then stops.
- **Stop now** stops at once.
- **Stop & skip** (on the running chunk) keeps what it wrote so far and moves on.

**Changing your mind mid-run.** Settings are read again before each chunk, so you can change a model or a prompt during a run: change it, press **Save settings** or **Save prompts**, and the next chunk uses it. A model this tool started is restarted with the new settings before its next chunk. To redo work:

- **Restart with current settings** (on the running chunk) stops it and runs it again from the start with what's set now.
- **Reset** (on any chunk) clears that chunk's results so it runs again.
- **Re-run all chunk-model chunks** / **Re-run all large-context chunks** reset a whole group, for example after swapping a small model for a bigger one.

Chunks reset while a run is going are picked up before it ends.

**Order.** Move chunks up or down, or delete them, before or during a run.

---

## The final pass

**Build final chunk(s) from the results** packs every chunk's result into one final chunk, or several if they don't fit the final model. **Run final pass** asks the final-pass chain on it.

If results came from a small model you don't fully trust, re-run those chunks on a better model first. Otherwise the final model only sees those weaker reports.

---

## Follow-up questions to the final model

After a final pass, the final-pass model stays loaded and still remembers everything it just read. Below the final results is a follow-up box:

- **Type a question.** Ctrl+Enter sends it. It shows in the conversation at once, like a chat, and the answer follows under it. Only your new message is read; the report is never re-read.
- **Give it files.** **Attach files…**, drop files onto the box, or type a project file path in *add a project file* (it suggests as you type). The files go in with your message.
- While your message (and files) is read, it shows the percentage, reading speed and time left; then the answer streams live with its speed. The page follows the answer down unless you scroll up. **Stop** works.
- **Saved automatically.** After every answer, the conversation is saved with the report: the run's auto-snapshot is updated in place (*auto: after run + 3 follow-ups*) instead of piling up new ones. This is text only. Turn it off with *Save a snapshot automatically when a run finishes*.
- **Save report** saves the report and the conversation under a name you choose, text only.
- **Save report with final KV** saves the report (with the follow-up conversation) and the model's memory (into Nexus's `nexus-data/sessions/`), linked together. Saved reports with memory are marked **+ model memory**.
- **Continue later:** open the saved report (**View**), then **Continue this conversation**.
  - With model memory, it's loaded back into the final-pass model and you're where you left off, without re-reading.
  - Without it, your next message has the model read the report and the conversation again first (slow for a big report, but exact); after that it's quick again.
  - Continuing from a save you named leaves that save as it was; new answers go to a new *auto:* snapshot.
- **If the model forgot** (it was restarted, or another app used it), the box tells you instead of answering from the wrong memory. Load a saved memory, or tick **read it all again** to rebuild the conversation by re-reading it, which is slow for a big report but exact.

When there are several final chunks, the model remembers the last one.

---

## Saved reports and exports

- **Save snapshot** (or automatically at the end of a run) keeps a copy of all results. It doesn't include the model memory unless you used *Save report with final KV*.
- **Saved reports** (top of the page): **View** opens one read-only, with **Export all (.md)** / **Export final (.md)**. **Export** downloads one directly. **Delete** removes one.
- **Export all (.md)** / **Copy all** / **Export final (.md)** on the live project.

---

## Settings that stick

- Pressing **Save settings** or **Save prompts** also makes those settings the starting point for every **new** project: models, caps, options and both chains.
- An existing project keeps its own settings. **Use my saved defaults in this project** pulls the saved defaults into it, keeping its scan and results.
- **Recent projects** remembers the last 15 projects.

---

## Models on other computers

The hardener talks to model servers over the network like any other program, so a model can run anywhere:

- Choose **Use a server that's already running** and give its address, e.g. `http://192.168.1.10:8080`. In the launcher on that computer, set *Who can connect* to *Anyone on my network*.
- If that server has an **API key** (launcher → Network card), enter it in the model's **API key** box. Servers on this computer never need one. A missing or wrong key is reported as such, not as "not reachable".

The hardener itself only listens on this computer (`127.0.0.1`).

---

## A good security-audit workflow

A wide sweep finds leads, and a focused check confirms them. Models make things up too: in the Nexus run, the same report that found a real command-injection bug also named variables that don't exist.

1. **Sweep.** Per-chunk chain: *Built-in: security audit + second look*, on your best model that's still fast enough to read everything. A 9B over a 1.6M-token codebase took about 7.5 hours.
2. **Re-run weak spots.** If a small model did the large-context parts, use **Re-run all large-context chunks** on a stronger one.
3. **Consolidate.** Final chain: *Built-in: consolidate into one report* (or *combine & rank findings*) on your strongest model.
4. **Verify, one claim at a time,** in the follow-up box, with the files attached:
   > *Here are the files for finding #1. Check it against the actual code: quote the exact lines, say whether the problem is real, and if it is, what input triggers it. If it's not real, say so plainly.*

   Or ask first: *Which files do you need to check finding #1?*, then attach them.
5. **Save report with final KV** before you stop, so you can come back to the same conversation.
6. **Check the ones that matter yourself.** The model points you to the right file and line; the code is the final word.

To clean up a release (personal info, notes, professionalism), use *Built-in: notes & personal-info sweep* instead. Mid-size models are reliable at that.

---

## Trusting the results

- **Chunk reports are leads, not verdicts.** Small models (under ~3B) can lose the thread in long chunks. Use them for quick sweeps, not security calls.
- **A chunk only sees itself.** A bug that needs two files to show (input read in one, used in another) can be missed. The final pass's *combine* prompt helps, and so does the verify step.
- **Make claims specific.** Prompts that demand file, function, quoted lines and a concrete trigger make made-up findings easy to spot: if the quote isn't in the file, the finding isn't real.

---

## Where things are stored

Everything lives in `harden_data/` next to `server.py`:

| Path | Holds |
|---|---|
| `harden_data/global.json` | Saved defaults, saved prompt chains, recent projects |
| `harden_data/<project>/state.json` | That project's settings, scan, chunks and results |
| `harden_data/<project>/saves/` | Saved reports |
| `harden_data/<project>/model_*.log` | Output of models this tool started |

**Delete this project's working data** removes one project's folder. Saved model memories live with Nexus, in `nexus-data/sessions/harden-…`.

The `harden_data` folder is never scanned, even if the tool sits inside the project it checks.

---

## Troubleshooting

| Problem | What to do |
|---|---|
| *Nexus not found* | Put `nexus_harden_tool` inside the Nexus folder (or 1-2 levels below it), or use servers that are already running. |
| *the chunk model isn't reachable at …* | Start it (**Start**, or in the launcher), or check the address. If this tool started it, Scan waits for it to finish loading. |
| *needs its API key* | Enter that server's key in the model's **API key** box. |
| A model answers with gibberish | Try the model card's RoPE / Norm settings. The model's log (`harden_data/<project>/model_*.log`) shows what it chose. |
| A small model repeats itself forever | Set **Max output tokens** (e.g. 1500) for that model, or use a bigger model. |
| A chunk is too big for a model | Lower the cap and rebuild chunks, or split the big file for the large-context model. |
| The follow-up box says the model forgot | Load the saved memory (View the saved report → **Continue this conversation**), or tick **read it all again**. |
