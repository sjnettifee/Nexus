# Documents in the web chat

Drag a document into the chat (or pick it with the attach button) and the model gets its **text**, not its raw bytes. Everything is read in your browser, offline, by readers shipped with Nexus. Nothing is uploaded anywhere.

| Type | Files | What the model gets |
|---|---|---|
| PDF | `.pdf` | The real text of every page, as `[Page 1]`, `[Page 2]`, … Pages with no text (scans, photos of paper) are read with **OCR** and marked `(OCR)`. |
| Word | `.docx`, `.docm`, `.odt` | Headings as `#`, lists as `-`, tables as `\| a \| b \|`. |
| Spreadsheets | `.xlsx`, `.xlsm`, `.xlsb`, `.xls`, `.ods` | Every sheet as `[Sheet: name]` followed by comma-separated rows. |
| Slides | `.pptx`, `.pptm`, `.odp` | `[Slide 1]`, `[Slide 2]`, …, with each slide's speaker notes. |
| E-books | `.epub` | The chapters in reading order. |
| Other | `.rtf`, `.html`, `.htm` | Clean text, with the formatting codes and scripts removed. |
| Text and code | anything else that is text | As is. |
| Images | `.png`, `.jpg`, … | Shown to the model itself (models that can see). |

A chip shows progress while a document is read ("reading page 12 of 40", "page 13 has no text: OCR 1 of 3"). **Send** waits until every chip shows its file name. Hover a finished chip to see about how many tokens it adds.

**Folders** work the same way: drop a folder, and documents inside it are read too, each under its path (`project/specs/plan.pdf`).

**Not readable here:** the old binary Word and PowerPoint formats (`.doc`, `.ppt`). Save them as `.docx` / `.pptx` or PDF first. Other binary files (archives, programs, …) are refused with a message instead of being sent as noise.

## How big it gets

Text costs roughly one token per 3.5 characters. A 50-page report is usually 25,000–40,000 tokens. Before this, dropping a PDF sent its compressed bytes as if they were text, which came to hundreds of thousands of tokens of noise.

## OCR

- Used only for PDF pages that contain no text. Expect a few seconds per page.
- *Settings → Documents → OCR for scanned PDF pages* turns it off.
- **Languages:** English is included. For others, put the language file (for example `deu.traineddata.gz` from the `@tesseract.js-data/deu` package, folder `4.0.0_best_int`) into `web/vendor/tesseract/lang/`, and enter `eng+deu` under *Settings → Documents → OCR language*.

## Where the readers live

The `web/vendor/` folder ships next to the Nexus server (in the Nexus folder, or next to the server program). The server finds it by itself, wherever Nexus is started from, which also covers a USB stick. `NEXUS_WEB_DIR` points it somewhere else. Without the folder, everything else still works; dropping a document then says the readers are missing.

The readers are unmodified open-source libraries: PDF.js, Tesseract.js (with Tesseract's English data), mammoth, SheetJS and JSZip. They're under Apache-2.0, BSD-2-Clause and MIT, all compatible with Nexus's AGPLv3. Versions and license texts are in `web/vendor/VERSIONS.txt` and `web/vendor/LICENSES/`.
