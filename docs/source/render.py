from playwright.sync_api import sync_playwright
import pathlib

src = pathlib.Path(__file__).parent / "nexus_intro.html"
out = pathlib.Path(__file__).parent / "Nexus_Intro.pdf"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(src.as_uri())
    page.pdf(
        path=str(out),
        width="8.5in",
        height="11in",
        print_background=True,
        margin={"top": "0in", "bottom": "0in", "left": "0in", "right": "0in"},
    )
    browser.close()
print("wrote", out)
