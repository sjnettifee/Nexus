"""Render docs/NEXUS_GUIDE.md to a print PDF (dark cover, light pages, Nexus purple accents)."""
import pathlib, re, sys
import markdown
from playwright.sync_api import sync_playwright

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
md = src.read_text(encoding="utf-8")

# Links to sibling .md files point at GitHub-relative paths; in a PDF keep the text, drop dead relative links.
body_md = md.split("\n", 1)[1]                       # drop the "# The Nexus Guide" title: the cover has it
html_body = markdown.markdown(body_md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
html_body = html_body.replace('\\|', '|')
html_body = re.sub(r'<a href="(?!https?:|#)([^"]*)">(.*?)</a>', r'<span class="xref">\2</span>', html_body)

css = """
@page { size: letter; margin: 0.7in 0.75in 0.75in 0.75in; }
@page :first { margin: 0; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI","Helvetica Neue",Arial,sans-serif; color:#1d2030; font-size:10pt; line-height:1.5; margin:0; }
.cover { height: 11in; margin: 0; padding: 2.4in 0.9in 0 0.9in; background:#0a0d14; color:#e4e7ee;
  position:relative; overflow:hidden; page-break-after: always; }
.cover .g1 { position:absolute; width:520px; height:520px; top:-160px; left:-140px; border-radius:50%; background:#4b3fb0; filter:blur(60px); opacity:.35; }
.cover .g2 { position:absolute; width:480px; height:480px; bottom:-180px; right:-150px; border-radius:50%; background:#2e6ea6; filter:blur(60px); opacity:.28; }
.cover .in { position:relative; z-index:1; }
.cover .kicker { text-transform:uppercase; letter-spacing:.16em; font-size:10.5pt; font-weight:600; color:#9b8cfb; }
.cover h1 { font-size:54pt; margin:10px 0 6px 0; color:#f2eefe; font-weight:700; letter-spacing:-.01em; }
.cover .tag { font-size:16pt; color:#c6cbe0; font-weight:300; }
.cover .sub { margin-top:34px; font-size:11.5pt; color:#8a90a3; max-width:5.4in; line-height:1.6; }
.cover .foot { position:absolute; bottom:0.5in; left:0.9in; font-size:9pt; color:#5a6072; letter-spacing:.08em; z-index:1; }
h2 { font-size:17pt; color:#2b2560; margin:26px 0 8px 0; padding-bottom:5px; border-bottom:2px solid #7c6af7; page-break-after:avoid; }
h2:not(:first-of-type) { page-break-before: always; }
h3 { font-size:12.5pt; color:#3d348f; margin:18px 0 6px 0; page-break-after:avoid; }
p { margin:6px 0 8px 0; }
ul, ol { margin:4px 0 10px 0; padding-left:22px; }
li { margin:2px 0; }
table { border-collapse:collapse; width:100%; margin:8px 0 14px 0; font-size:8.8pt; page-break-inside:auto; }
tr { page-break-inside:avoid; }
th { background:#efecfe; color:#2b2560; text-align:left; font-weight:650; }
th, td { border:1px solid #d9d6ee; padding:4px 7px; vertical-align:top; }
tr:nth-child(even) td { background:#faf9ff; }
code { font-family: "DejaVu Sans Mono", Consolas, monospace; font-size:8.6pt; background:#f1f0f8; padding:1px 4px; border-radius:3px; color:#3b3266; }
pre { background:#11141d; color:#dfe3ee; padding:10px 12px; border-radius:6px; font-size:8.3pt; line-height:1.45; white-space:pre-wrap; page-break-inside:avoid; }
pre code { background:none; color:inherit; padding:0; }
a { color:#5a49d6; text-decoration:none; }
.xref { color:#5a49d6; }
hr { border:none; border-top:1px solid #e3e1f2; margin:18px 0; }
strong { color:#14162a; }
"""

cover = """
<div class="cover"><div class="g1"></div><div class="g2"></div><div class="in">
  <div class="kicker">Nexus &middot; multi-model inference engine</div>
  <h1>The Nexus Guide</h1>
  <div class="tag">Everything in one place</div>
  <div class="sub">How Nexus works, setups for the hardware you have, every launcher setting,
  every server flag and environment variable, every HTTP endpoint, the log, and what to do when
  something goes wrong.</div>
</div><div class="foot">FREE &middot; LOCAL &middot; AGPLv3 &middot; github.com/sjnettifee/nexus</div></div>
"""

doc = f"<!doctype html><html><head><meta charset='utf-8'><title>The Nexus Guide</title><style>{css}</style></head><body>{cover}{html_body}</body></html>"
tmp = out.with_suffix(".html")
tmp.write_text(doc, encoding="utf-8")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(tmp.as_uri())
    pg.pdf(path=str(out), format="Letter", print_background=True, display_header_footer=True, prefer_css_page_size=True,
           header_template="<span></span>",
           footer_template="<div style='width:100%;font-size:8px;color:#8a8fa3;text-align:center;font-family:Arial'>The Nexus Guide &middot; <span class='pageNumber'></span></div>")
    b.close()
print("wrote", out)
