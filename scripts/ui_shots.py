"""Screenshots of the running app for UI review (python app.py must be running).

    python scripts/ui_shots.py <out_dir>

Writes: desktop_empty.png, desktop_full.png (after a photo + a request),
desktop_hardcase.png (falafel misread as bolognese), mobile_full.png.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:7860/?__theme=light"
TEST = ROOT / "data" / "raw" / "food10" / "test"


def _browser(p):
    os.environ["TMP"] = os.environ["TEMP"] = str(ROOT / "data" / "cache" / "tmp")
    for exe in sorted(glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-*\chrome-win*\chrome.exe")),
                      reverse=True):
        try:
            return p.chromium.launch(executable_path=exe)
        except Exception:
            continue
    return p.chromium.launch(channel="msedge")


def run(page, photo: Path, query: str) -> None:
    page.goto(URL)
    page.wait_for_selector("#rf-photo")
    page.locator("#rf-photo input[type=file]").set_input_files(str(photo))
    page.wait_for_selector("#rf-pred .rf-dish", timeout=30000)
    if query:
        box = page.locator("#rf-query textarea, #rf-query input").first
        box.fill(query)
        box.press("Enter")
        page.wait_for_selector("#rf-recipes .rf-feature", timeout=30000)
        page.wait_for_selector("#rf-cmp-search .rf-cmp-4", timeout=30000)
    page.wait_for_timeout(1800)


def main(out: Path) -> None:
    # Playwright's driver reads TEMP when it starts: point it inside the repo first
    os.environ["TMP"] = os.environ["TEMP"] = str(ROOT / "data" / "cache" / "tmp")
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = _browser(p)
        d = b.new_context(viewport={"width": 1440, "height": 900}, locale="en-US").new_page()
        d.goto(URL)
        d.wait_for_selector("#rf-photo")
        d.wait_for_timeout(1200)
        d.screenshot(path=str(out / "desktop_empty.png"))
        run(d, TEST / "pizza" / "validation_00529.jpg", "vegetarian, under 30 minutes")
        d.evaluate("window.scrollTo(0, 0)")
        d.wait_for_timeout(500)
        d.screenshot(path=str(out / "desktop_full.png"), full_page=True)
        run(d, TEST / "falafel" / "validation_01766.jpg", "")
        d.screenshot(path=str(out / "desktop_hardcase.png"))
        m = b.new_context(viewport={"width": 390, "height": 844}, locale="en-US", device_scale_factor=1).new_page()
        run(m, TEST / "ramen" / "validation_00000.jpg", "vegan ramen under 30 minutes")
        m.evaluate("window.scrollTo(0, 0)")
        m.wait_for_timeout(500)
        m.screenshot(path=str(out / "mobile_full.png"), full_page=True)
        b.close()
    print("wrote", *sorted(x.name for x in out.glob("*.png")))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "shots" / "review")
