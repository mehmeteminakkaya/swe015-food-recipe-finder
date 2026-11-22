"""Record the 2-minute demo video by driving the running app with Playwright.

The app must already be running:   python app.py
Then:                               python scripts/record_demo.py

Output:
  dist/video/demo_silent.mp4      silent screen recording (1440x900)
  dist/video/timeline.json        start second of every scene, for the narration

The recording is silent on purpose: narration is added afterwards
(docs/VIDEO_NARRATION.md has the text per scene), either one clip per scene,
placed automatically at the scene's start second:
  python scripts/record_demo.py --voice-dir <folder with intro.wav, app.wav, ...>
or a single file:
  python scripts/record_demo.py --mux <narration.wav|mp3>

Extra dependencies (not needed to run the app):
  pip install playwright==1.48.0 imageio-ffmpeg==0.5.1 && python -m playwright install chromium
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "video"
URL = "http://127.0.0.1:7860/?__theme=light"
W, H = 1440, 900
TEST = ROOT / "data" / "raw" / "food10" / "test"
FALAFEL = TEST / "falafel" / "validation_01766.jpg"
PIZZA = TEST / "pizza" / "validation_00529.jpg"
RAMEN = TEST / "ramen" / "validation_00000.jpg"

# ------------------------------------------------------------------ cards ---

CARD_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Young+Serif&family=Figtree:wght@400;600;700&display=swap');
html,body{margin:0;height:100%;background:#F1F4EF;color:#17301F;font-family:Figtree,system-ui,sans-serif}
.wrap{box-sizing:border-box;height:100%;padding:80px 110px;display:flex;flex-direction:column;justify-content:center}
h1{font-family:'Young Serif',Georgia,serif;font-weight:400;font-size:68px;line-height:1.05;margin:0 0 26px;max-width:16ch}
h2{font-family:'Young Serif',Georgia,serif;font-weight:400;font-size:40px;margin:0 0 22px}
p{font-size:26px;line-height:1.45;color:#5B6A60;margin:0 0 14px;max-width:46ch}
.meta{font-size:19px;margin-top:34px}
.fig{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#fff;
  border:1px solid #D3DCD4;border-radius:14px;padding:18px}
.fig img{max-width:100%;max-height:100%;object-fit:contain}
table{border-collapse:collapse;width:100%;font-size:19px;background:#fff;border:1px solid #D3DCD4;border-radius:14px;overflow:hidden}
th,td{padding:11px 14px;text-align:left;border-top:1px solid #E3E9E4;font-variant-numeric:tabular-nums}
th{background:#17301F;color:#fff;font-weight:600;border-top:none}
td.num,th.num{text-align:right}
tr.hl td{background:#FBF0D5;font-weight:700}
.note{font-size:17px;margin-top:14px}
ul{font-size:25px;line-height:1.5;color:#17301F;padding-left:28px;margin:0;max-width:52ch}
li{margin-bottom:10px}
"""


def card(body: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CARD_CSS}</style></head><body><div class='wrap'>{body}</div></body></html>"


def _results() -> dict[str, dict]:
    with (ROOT / "results" / "results.csv").open(encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def intro_card() -> str:
    return card(
        "<h1>Photo to recipe</h1>"
        "<p>You have a photo of a dish and a wish such as “vegetarian, under 30 minutes”. "
        "The app recognises the dish and ranks five recipes that fit the wish.</p>"
        "<p class='meta'>SWE015 · Use Case 3 · Mehmet Emin Akkaya</p>"
    )


def arch_card() -> str:
    b64 = base64.b64encode((ROOT / "docs" / "architecture.png").read_bytes()).decode()
    return card(f"<h2>Architecture</h2><div class='fig'><img src='data:image/png;base64,{b64}'></div>")


def results_card() -> str:
    names = {
        "small_cnn": "Small CNN (from scratch)", "vgg16_ft": "VGG16 fine-tuned", "vgg16_int8": "VGG16 INT8",
        "tfidf": "tf-idf + cosine", "word2vec": "word2vec (trained here)", "bert_e5": "E5 Transformer",
        "bert_e5_int8": "E5 INT8",
    }
    rows = []
    for r in _results().values():
        hl = " class='hl'" if r["model"] in ("vgg16_int8", "bert_e5_int8") else ""
        gpu = "—" if r["latency_gpu_ms"] in ("", "nan") else f"{float(r['latency_gpu_ms']):.1f} ms"
        rows.append(
            f"<tr{hl}><td>{'Image' if r['task'] == 'vision' else 'Search'}</td>"
            f"<td>{names.get(r['model'], r['model'])}</td>"
            f"<td>{r['primary_metric']}</td><td class='num'>{float(r['primary_value']):.3f}</td>"
            f"<td class='num'>{float(r['size_mb']):.1f} MB</td>"
            f"<td class='num'>{float(r['latency_cpu_ms']):.1f} ms</td><td class='num'>{gpu}</td></tr>"
        )
    return card(
        "<h2>Results</h2><table><tr><th>Task</th><th>Model</th><th>Metric</th><th class='num'>Value</th>"
        "<th class='num'>Size</th><th class='num'>CPU</th><th class='num'>GPU</th></tr>"
        + "".join(rows) + "</table>"
        "<p class='note'>Same test sets for every model (2,499 photos, 40 requests). Latency is the mean of "
        "100 runs on a Ryzen 7 8845HS / RTX 4060 laptop. The two highlighted INT8 models run in the app.</p>"
    )


def outro_card() -> str:
    r = _results()
    f = lambda m, c: float(r[m][c])  # noqa: E731
    shrink = f("vgg16_ft", "size_mb") / f("vgg16_int8", "size_mb")
    speed = f("vgg16_ft", "latency_cpu_ms") / f("vgg16_int8", "latency_cpu_ms")
    return card(
        "<h2>Conclusion</h2><ul>"
        f"<li>INT8 quantization is almost free: VGG16 became {shrink:.0f}× smaller and {speed:.1f}× faster on the CPU, "
        f"while top-1 accuracy moved from {f('vgg16_ft', 'primary_value'):.3f} to {f('vgg16_int8', 'primary_value'):.3f}.</li>"
        "<li>The multilingual Transformer finds the most relevant recipes; in Turkish, our own word2vec is just as good.</li>"
        "<li>Limitation: the model can be fooled by colour and plating. Next step: more dishes and more data.</li>"
        "</ul>"
    )


# ------------------------------------------------------------- recording ---

# An arrow pointer drawn on top of the page (headless recordings have no cursor).
# Its tip is the hotspot, so left/top are the exact pointer coordinates.
CURSOR_JS = """
([x, y]) => {
  let c = document.getElementById('demo-cursor');
  if (!c) {
    c = document.createElement('div');
    c.id = 'demo-cursor';
    c.innerHTML = '<svg width="26" height="30" viewBox="0 0 26 30" xmlns="http://www.w3.org/2000/svg">' +
      '<path d="M2 2 L2 24 L8 18.5 L12.5 28 L16.5 26.2 L12 17 L20 17 Z" fill="#fff" stroke="#17301F" ' +
      'stroke-width="2" stroke-linejoin="round"/></svg>';
    c.style.cssText = 'position:fixed;width:26px;height:30px;margin:-2px 0 0 -2px;z-index:999999;' +
      'pointer-events:none;filter:drop-shadow(0 2px 3px rgba(0,0,0,.25));transform-origin:2px 2px;' +
      'transition:transform .12s ease';
    document.body.appendChild(c);
    window.__demoMove = (x, y) => { c.style.left = x + 'px'; c.style.top = y + 'px'; };
    window.__demoPress = (down) => { c.style.transform = down ? 'scale(.82)' : ''; };
  }
  window.__demoMove(x, y);
}
"""


VOICE_DIR = OUT / "voice"
VOICE_OFFSET = 0.3  # a clip starts this long after its scene (see mux_scenes)
# scenes whose narration should wait for the action on screen (seconds, on top of VOICE_OFFSET)
VOICE_DELAY = {"ramen_en": 4.6}


def voice_start(scene: str) -> float:
    return VOICE_OFFSET + VOICE_DELAY.get(scene, 0.0)


def _voice_lengths() -> dict[str, float]:
    """Seconds of each narration clip in dist/video/voice/, if they have been recorded."""
    import wave

    out = {}
    for f in VOICE_DIR.glob("*.wav"):
        try:
            with wave.open(str(f)) as w:
                out[f.stem] = w.getnframes() / w.getframerate()
        except (wave.Error, EOFError):
            pass
    return out


VOICE = _voice_lengths()


def _voice_words() -> dict[str, list[dict]]:
    """Word timestamps per clip (scripts/voice_timing.py), used to point at things as they are said."""
    f = VOICE_DIR / "words.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


WORDS = _voice_words()
EVENTS_FILE = OUT / "events.json"  # clicks, keys and camera moves, for the zoom pass and the sound effects

# smooth scrolling inside the page: one step per animation frame, eased, no jank
SCROLL_JS = """
([y, ms]) => new Promise(done => {
  const y0 = window.scrollY, t0 = performance.now();
  const step = now => {
    const k = Math.min((now - t0) / ms, 1), e = (1 - Math.cos(Math.PI * k)) / 2;
    window.scrollTo({top: y0 + (y - y0) * e, behavior: 'instant'});
    k < 1 ? requestAnimationFrame(step) : done();
  };
  requestAnimationFrame(step);
})
"""


def _norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def _ease(t: float) -> float:
    """Ease in-out: a hand starts slowly, speeds up, then slows down near the target."""
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


class Recorder:
    """Drives the page like a person: curved, unhurried mouse paths and uneven typing."""

    def __init__(self, page, seed: int = 7):
        import random

        self.page = page
        self.rng = random.Random(seed)  # same "hand" on every recording
        self.x, self.y = W * 0.62, H * 0.55
        self.t0 = time.monotonic()
        self.timeline: list[dict] = []
        self.events: list[dict] = []

    def log(self, kind: str, **kw) -> None:
        self.events.append({"t": round(time.monotonic() - self.t0, 3), "kind": kind, **kw})

    # --- camera: zoom requests, applied to the frames after recording ------------

    def focus(self, locator, zoom: float = 1.45) -> None:
        """Ask the camera to move in on an element (centre and zoom are logged, not applied here)."""
        box = locator.bounding_box()
        if box:
            self.log("zoom", cx=box["x"] + box["width"] / 2, cy=box["y"] + box["height"] / 2, z=zoom)

    def unfocus(self) -> None:
        self.log("zoom", cx=W / 2, cy=H / 2, z=1.0)

    # --- pointing at what the narration is talking about -------------------------

    def word_time(self, scene: str, word: str, nth: int = 1) -> float | None:
        """Absolute time (monotonic) at which `word` is spoken in the scene's clip."""
        if scene not in WORDS or scene not in getattr(self, "starts", {}):
            return None
        hits = [w["s"] for w in WORDS[scene] if _norm_word(w["w"]).startswith(_norm_word(word))]
        if len(hits) < nth:
            return None
        return self.starts[scene] + voice_start(scene) + hits[nth - 1]

    def point(self, scene: str, word: str, locator=None, nth: int = 1, xy: tuple | None = None,
              zoom: float | None = None) -> None:
        """Arrive at the element just as `word` is said, like a presenter pointing with the mouse."""
        if xy is None:
            locator.scroll_into_view_if_needed()
            box = locator.bounding_box()
            if not box:
                return
            xy = (box["x"] + box["width"] * self.rng.uniform(0.3, 0.55),
                  box["y"] + box["height"] * self.rng.uniform(0.4, 0.6))
        dist = ((xy[0] - self.x) ** 2 + (xy[1] - self.y) ** 2) ** 0.5
        travel = min(max(520 + dist * 0.85, 700), 1500) / 1000
        said = self.word_time(scene, word, nth)
        if said is not None:
            now = time.monotonic()
            if now > said + 1.2:          # running late: skip rather than point at the wrong moment
                return
            lead = said - 0.25 - travel - now
            if lead > 0:
                self.wait(int(lead * 1000))
        if zoom and locator is not None:
            self.focus(locator, zoom)
        self.move_to(*xy)

    def mark(self, scene: str) -> None:
        t = round(time.monotonic() - self.t0, 2)
        self.timeline.append({"scene": scene, "t": t})
        self.starts = getattr(self, "starts", {})
        self.starts[scene] = time.monotonic()
        print(f"{t:7.2f}s  {scene}")

    def until_voice(self, scene: str, fallback_ms: int, minimum_ms: int = 0, card: bool = False,
                    still: bool = False) -> None:
        """Hold the scene until its narration clip has finished (+ a breath), so the
        picture follows the voice. Without recorded clips, hold for fallback_ms."""
        if scene in VOICE:
            end = self.starts[scene] + voice_start(scene) + VOICE[scene] + 0.7
            ms = max(int((end - time.monotonic()) * 1000), minimum_ms)
        else:
            ms = fallback_ms
        if ms > 0:
            (self.wait if card or still else self.idle)(ms)   # still: keep pointing, do not drift away

    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)

    def show_cursor(self) -> None:
        self.page.evaluate(CURSOR_JS, [self.x, self.y])

    # --- mouse -----------------------------------------------------------------

    def move_to(self, x: float, y: float, speed: float = 1.0) -> None:
        """Move along an S-shaped cubic Bezier curve with ease-in-out timing and a small overshoot."""
        import math

        self.show_cursor()
        x0, y0 = self.x, self.y
        dx, dy = x - x0, y - y0
        dist = math.hypot(dx, dy)
        if dist < 2:
            return
        nx, ny = -dy / dist, dx / dist                       # unit normal to the straight line
        bend = min(dist * self.rng.uniform(0.12, 0.24), 140)  # how far the path swings out
        side = self.rng.choice((-1, 1))
        # control points on opposite sides of the line -> a gentle slalom, not a straight slide
        c1 = (x0 + dx * 0.3 + nx * bend * side, y0 + dy * 0.3 + ny * bend * side)
        c2 = (x0 + dx * 0.7 - nx * bend * 0.6 * side, y0 + dy * 0.7 - ny * bend * 0.6 * side)
        # overshoot the target a little along the direction of travel, then settle back
        over = min(dist * 0.03, 9)
        end = (x + dx / dist * over, y + dy / dist * over)
        duration = min(max(520 + dist * 0.85, 700), 1500) / speed

        start = time.monotonic()
        while True:
            t = min((time.monotonic() - start) * 1000 / duration, 1.0)
            e = _ease(t)
            u = 1 - e
            px = u ** 3 * x0 + 3 * u ** 2 * e * c1[0] + 3 * u * e ** 2 * c2[0] + e ** 3 * end[0]
            py = u ** 3 * y0 + 3 * u ** 2 * e * c1[1] + 3 * u * e ** 2 * c2[1] + e ** 3 * end[1]
            px += self.rng.uniform(-0.6, 0.6)               # a hand is never perfectly steady
            py += self.rng.uniform(-0.6, 0.6)
            self.page.mouse.move(px, py)
            self.page.evaluate("([x, y]) => window.__demoMove && window.__demoMove(x, y)", [px, py])
            if t >= 1.0:
                break
            self.page.wait_for_timeout(14)
        # settle from the overshoot onto the target
        for k in range(1, 7):
            sx, sy = end[0] + (x - end[0]) * k / 6, end[1] + (y - end[1]) * k / 6
            self.page.mouse.move(sx, sy)
            self.page.evaluate("([x, y]) => window.__demoMove(x, y)", [sx, sy])
            self.page.wait_for_timeout(18)
        self.x, self.y = x, y

    def target(self, locator) -> tuple[float, float]:
        """A point near, not exactly at, the centre of an element: people do not aim perfectly."""
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        fx = self.rng.uniform(0.35, 0.65) if box["width"] > 40 else 0.5
        fy = self.rng.uniform(0.38, 0.62) if box["height"] > 24 else 0.5
        return box["x"] + box["width"] * fx, box["y"] + box["height"] * fy

    def cursor_to(self, locator) -> None:
        self.move_to(*self.target(locator))
        self.wait(self.rng.randint(160, 320))               # a short look before clicking

    def click(self, locator) -> None:
        self.cursor_to(locator)
        self.page.evaluate("window.__demoPress && window.__demoPress(true)")
        self.log("click")
        self.page.mouse.down()
        self.wait(self.rng.randint(70, 120))
        self.page.mouse.up()
        self.page.evaluate("window.__demoPress && window.__demoPress(false)")

    def idle(self, ms: int) -> None:
        """Wait while the hand drifts a little, as it does when someone is reading."""
        end = time.monotonic() + ms / 1000
        while time.monotonic() + 1.2 < end:
            tx = min(max(self.x + self.rng.uniform(-70, 70), 40), W - 40)
            ty = min(max(self.y + self.rng.uniform(-45, 45), 40), H - 40)
            self.move_to(tx, ty, speed=0.55)
            rest = self.rng.randint(500, 1100)
            self.page.wait_for_timeout(min(rest, max(int((end - time.monotonic()) * 1000), 0)))
        left = int((end - time.monotonic()) * 1000)
        if left > 0:
            self.page.wait_for_timeout(left)

    # --- keyboard ----------------------------------------------------------------

    def type_text(self, text: str, typo_at: int | None = None) -> None:
        """Type with uneven rhythm: pauses after spaces and commas, the odd hesitation,
        and (optionally) one slip that is noticed and corrected."""
        neighbours = {"e": "r", "a": "s", "i": "o", "n": "m", "t": "r", "u": "y", "o": "p", "r": "e"}
        for i, ch in enumerate(text):
            if typo_at is not None and i == typo_at and ch.lower() in neighbours:
                self.log("key")
                self.page.keyboard.type(neighbours[ch.lower()])
                self.wait(self.rng.randint(220, 380))      # notices the slip
                self.log("key")
                self.page.keyboard.press("Backspace")
                self.wait(self.rng.randint(140, 220))
            self.log("key")
            self.page.keyboard.type(ch)
            delay = self.rng.lognormvariate(4.3, 0.35)     # ~74 ms median, long tail
            if ch == " ":
                delay += self.rng.randint(40, 130)
            elif ch in ",.":
                delay += self.rng.randint(160, 280)
            elif self.rng.random() < 0.04:
                delay += self.rng.randint(220, 420)        # thinking mid-word
            self.wait(int(min(delay, 420)))

    def scroll_y(self, y: float, ms: int = 900) -> None:
        self.page.evaluate(SCROLL_JS, [max(y, 0), ms])

    def scroll_to(self, selector: str, block: str = "start", margin: int = 24, ms: int = 900) -> None:
        """Smoothly bring an element to the top (or centre) of the viewport."""
        r = self.page.evaluate(
            "s => { const b = document.querySelector(s).getBoundingClientRect();"
            " return {top: b.top + window.scrollY, h: b.height}; }", selector)
        y = r["top"] - margin if block == "start" else r["top"] + r["h"] / 2 - H / 2
        self.scroll_y(y, ms)

    def scroll_top(self, ms: int = 900) -> None:
        self.scroll_y(0, ms)


def upload(rec: "Recorder", photo: Path) -> None:
    """Click the photo box (opens the file dialog for a user) and hand it the file."""
    page = rec.page
    remove = page.locator("#rf-photo button[aria-label='Remove Image']")
    if remove.count():
        rec.click(remove.first)
        rec.wait(400)
    rec.cursor_to(page.locator("#rf-photo"))
    rec.log("click")
    page.locator("#rf-photo input[type=file]").set_input_files(str(photo))


def wait_dish(page, prev: str) -> None:
    """The app recognises a photo by itself; wait until a new dish name is shown."""
    page.wait_for_function(
        "prev => { const d = document.querySelector('#rf-pred .rf-dish'); return d && d.innerText !== prev; }",
        arg=prev, timeout=30000)


def dish_text(page) -> str:
    el = page.query_selector("#rf-pred .rf-dish")
    return el.inner_text() if el else ""


def wait_recipes(page, before: str) -> None:
    page.wait_for_function("prev => document.querySelector('#rf-recipes').innerHTML !== prev",
                           arg=before, timeout=30000)
    page.wait_for_timeout(900)  # the app scrolls to the results by itself


def search(rec: Recorder, text: str, typo_at: int | None = None) -> None:
    """Click into the box, type the request like a person, pause, press Enter."""
    page = rec.page
    box = page.locator("#rf-query textarea, #rf-query input").first
    rec.click(box)
    rec.focus(page.locator("#rf-query"), zoom=1.7)          # the camera moves in on the text box
    if box.input_value():
        rec.wait(rec.rng.randint(150, 260))
        rec.log("key")
        page.keyboard.press("Control+A")
        rec.wait(rec.rng.randint(120, 200))
        rec.log("key")
        page.keyboard.press("Backspace")
    rec.wait(rec.rng.randint(250, 420))
    rec.type_text(text, typo_at=typo_at)
    rec.wait(rec.rng.randint(380, 560))                    # reads it once before pressing Enter
    rec.unfocus()                                          # zoom out before the page scrolls
    before = page.inner_html("#rf-recipes")
    rec.log("key", strong=True)
    page.keyboard.press("Enter")
    wait_recipes(page, before)


def glide(rec: Recorder, px: int, ms: int) -> None:
    """Scroll down by px, eased, one step per animation frame inside the page (no stutter)."""
    y = rec.page.evaluate("window.scrollY")
    rec.scroll_y(y + px, ms)


def choose(rec: Recorder, label: str) -> None:
    rec.click(rec.page.locator(".rf-models label", has_text=label).first)
    rec.wait(300)


def record() -> Path:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / "raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    # keep Playwright's temp files inside the repo (gitignored)
    tmp = ROOT / "data" / "cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(tmp)
    # Playwright's ffmpeg (video capture) lives here: python -m playwright install ffmpeg
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / "data" / "cache" / "ms-playwright"))

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            # Playwright's own Chromium is not installed: use Edge, which ships with Windows
            browser = p.chromium.launch(channel="msedge")
        # warm-up page outside the recording so fonts and the app are cached
        warm = browser.new_page(viewport={"width": W, "height": H})
        warm.goto(URL)
        warm.wait_for_selector("#rf-photo")
        warm.close()

        # Gradio translates its own widgets to the browser language: force English
        ctx = browser.new_context(viewport={"width": W, "height": H}, locale="en-US",
                                  extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                                  record_video_dir=str(raw_dir), record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        rec = Recorder(page)

        # 0:00 problem
        page.set_content(intro_card(), wait_until="networkidle")
        rec.mark("intro")
        rec.until_voice("intro", 12000, minimum_ms=5000, card=True)

        # live demo
        page.goto(URL)
        page.wait_for_selector("#rf-photo")
        rec.show_cursor()
        rec.mark("app")
        rec.idle(1600)

        pred, dish, cam = page.locator("#rf-pred"), page.locator("#rf-pred .rf-dish"), page.locator("#rf-cam")

        # easy case: clicking an example photo recognises it straight away
        rec.mark("pizza")
        upload(rec, PIZZA)
        wait_dish(page, "")
        rec.scroll_to("#rf-cam", "start", margin=70)       # heat map and prediction both in view
        rec.point("app", "says", dish, zoom=1.5)            # "...says pizza"
        rec.point("app", "heat", cam, zoom=1.6)             # "...the heat map shows where..."
        rec.until_voice("app", 4000, minimum_ms=1200, still=True)
        rec.unfocus()
        rec.scroll_top()

        # request + Enter: the app scrolls to the recipes, then we read down the best one
        rec.mark("query")
        search(rec, "vegetarian, under 30 minutes", typo_at=4)
        rec.point("query", "best", page.locator("#rf-recipes .rf-feature h3"))
        rec.wait(500)
        glide(rec, 650, 2600)
        rec.until_voice("query", 1100, minimum_ms=900)
        rec.scroll_top()

        # difficult case: falafel uploaded from disk, recognised on upload, predicted as bolognese
        rec.mark("falafel_upload")
        rec.click(page.locator("#rf-photo button[aria-label='Remove Image']").first)
        rec.wait(500)
        rec.log("click")
        page.locator("#rf-photo input[type=file]").set_input_files(str(FALAFEL))
        wait_dish(page, "")
        rec.scroll_to("#rf-cam", "start", margin=70)
        rec.point("falafel_upload", "spaghetti", dish, zoom=1.5)
        rec.point("falafel_upload", "heat", cam, zoom=1.6)  # "...the heat map sits on the red sauce"
        rec.until_voice("falafel_upload", 5000, minimum_ms=1500, still=True)
        rec.unfocus()
        rec.scroll_top()

        # ramen example + a vegan request
        rec.mark("ramen_en")
        prev = dish_text(page)
        upload(rec, RAMEN)
        wait_dish(page, prev)
        rec.wait(600)
        search(rec, "vegan ramen under 30 minutes")
        feature = page.locator("#rf-recipes .rf-feature")
        rec.point("ramen_en", "vegan", feature.locator(".rf-chip.is-veg").first, nth=2)  # "every result is vegan"
        rec.point("ramen_en", "minutes", feature.locator(".rf-facts div").first, nth=2)  # "...30 minutes or less"
        rec.wait(400)
        glide(rec, 900, 2400)
        rec.until_voice("ramen_en", 700, minimum_ms=600)

        # every model side by side for the same photo and request
        rec.mark("compare")
        page.wait_for_selector("#rf-cmp-search .rf-cmp-4", timeout=30000)
        rec.scroll_to(".rf-compare", "start", ms=1100)
        rec.point("compare", "three", page.locator("#rf-cmp-photo .rf-cmp-card").first)
        rec.point("compare", "four", page.locator("#rf-cmp-search .rf-cmp-card").first)
        rec.until_voice("compare", 2600, minimum_ms=1500, still=True)

        # architecture: point along the two paths as they are named
        page.set_content(arch_card(), wait_until="networkidle")
        rec.show_cursor()
        rec.mark("architecture")
        img = page.locator(".fig img").bounding_box()
        at = lambda fx, fy: (img["x"] + img["width"] * fx, img["y"] + img["height"] * fy)  # noqa: E731
        rec.point("architecture", "photo", xy=at(0.27, 0.43))    # image path, start
        rec.point("architecture", "vgg", xy=at(0.43, 0.43))      # VGG16 INT8 box
        rec.point("architecture", "text", xy=at(0.27, 0.73))     # text path
        rec.point("architecture", "encoder", xy=at(0.59, 0.73))  # E5 INT8 box
        rec.until_voice("architecture", 11500, minimum_ms=1500, still=True)

        # results: point at the INT8 row being described
        page.set_content(results_card(), wait_until="networkidle")
        rec.show_cursor()
        rec.mark("results")
        rec.point("results", "vgg16", page.locator("tr.hl").first)
        rec.until_voice("results", 14500, minimum_ms=3000, still=True)

        # conclusion
        page.set_content(outro_card(), wait_until="networkidle")
        rec.mark("conclusion")
        rec.until_voice("conclusion", 12000, minimum_ms=6000, card=True)
        rec.wait(800)                                     # a calm last frame
        rec.mark("end")

        video_path = Path(page.video.path())
        ctx.close()
        browser.close()

    (OUT / "timeline.json").write_text(json.dumps(rec.timeline, indent=2), encoding="utf-8")
    EVENTS_FILE.write_text(json.dumps(rec.events, indent=1), encoding="utf-8")
    return video_path


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


CAMERA = False  # zooming on the active area: off by default (it looked artificial); --camera turns it on


def to_mp4(webm: Path, out: Path) -> None:
    """Convert the capture to 25 fps H.264 (the capture rate); with --camera, apply the logged zooms."""
    events = json.loads(EVENTS_FILE.read_text(encoding="utf-8")) if EVENTS_FILE.exists() else []
    zooms = [e for e in events if e["kind"] == "zoom"] if CAMERA else []
    if not zooms:
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-r", "25", "-t", "119.5", "-movflags", "+faststart",
                        str(out)], check=True)
        return
    apply_camera(webm, out, zooms)


def _camera_path(zooms: list[dict], fps: int, n_frames: int, ease_s: float = 0.8):
    """Per-frame (cx, cy, z): each zoom request eases from the previous state over ease_s seconds."""
    state = (W / 2, H / 2, 1.0)
    keys = []  # (t_start, from_state, to_state)
    for z in sorted(zooms, key=lambda e: e["t"]):
        to = (z["cx"], z["cy"], z["z"])
        keys.append((z["t"], state, to))
        state = to
    out, k = [], 0
    cur = (W / 2, H / 2, 1.0)
    for i in range(n_frames):
        t = i / fps
        while k < len(keys) and keys[k][0] <= t:
            k += 1
        if k == 0:
            out.append(cur)
            continue
        t0, a, b = keys[k - 1]
        p = min((t - t0) / ease_s, 1.0)
        e = _ease(p)
        out.append(tuple(a[j] + (b[j] - a[j]) * e for j in range(3)))
    return out


def apply_camera(src: Path, out: Path, zooms: list[dict], fps: int = 25) -> None:
    """fps 25 = the capture rate: converting to 30 would duplicate every fifth frame (judder)."""
    import cv2
    import numpy as np

    dur = 119.5
    n = int(dur * fps)
    path = _camera_path(zooms, fps, n)
    dec = subprocess.Popen([ffmpeg(), "-loglevel", "error", "-i", str(src), "-r", str(fps), "-t", str(dur),
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    size = W * H * 3
    i = 0
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        cx, cy, z = path[min(i, n - 1)]
        if z > 1.001:
            cw, ch = W / z, H / z
            x0 = min(max(cx - cw / 2, 0), W - cw)
            y0 = min(max(cy - ch / 2, 0), H - ch)
            m = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])   # sub-pixel crop + scale in one step
            frame = cv2.warpAffine(frame, m, (W, H), flags=cv2.INTER_CUBIC)
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        i += 1
    enc.stdin.close()
    dec.wait()
    enc.wait()


def _sfx(kind: str, sr: int, rng) -> "np.ndarray":
    """Soft, synthetic UI sounds (no sample files): a mouse click or a key press."""
    import numpy as np

    if kind == "click":
        n = int(0.035 * sr)
        t = np.arange(n) / sr
        noise = rng.standard_normal(n)
        tick = np.diff(noise, prepend=0) * np.exp(-t / 0.0035)          # bright, very short
        thump = np.sin(2 * np.pi * 170 * t) * np.exp(-t / 0.008) * 0.6   # a little body
        s = tick * 0.35 + thump
        return (s / np.abs(s).max() * 0.07).astype(np.float32)
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n)
    s = np.diff(noise, prepend=0) * np.exp(-t / rng.uniform(0.0025, 0.004))
    s += np.sin(2 * np.pi * rng.uniform(380, 520) * t) * np.exp(-t / 0.006) * 0.4
    return (s / np.abs(s).max() * rng.uniform(0.025, 0.04)).astype(np.float32)


VOICE_SCENES = ["intro", "app", "query", "falafel_upload", "ramen_en", "compare",
                "architecture", "results", "conclusion"]


def mux_scenes(voice_dir: Path) -> Path:
    """Place voice_dir/<scene>.(wav|mp3|m4a|ogg) at the second its scene starts in timeline.json."""
    import numpy as np
    from scipy.io import wavfile

    sr = 44100
    timeline = {e["scene"]: e["t"] for e in json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))}
    track = np.zeros(sr * 120, dtype=np.float32)
    tmp = OUT / "voice_tmp.wav"
    found = 0
    for i, scene in enumerate(VOICE_SCENES):
        clip = next((p for ext in ("wav", "mp3", "m4a", "ogg", "flac")
                     if (p := voice_dir / f"{scene}.{ext}").exists()), None)
        if clip is None:
            print(f"  no clip for {scene}, skipped")
            continue
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(clip), "-ac", "1", "-ar", str(sr),
                        "-c:a", "pcm_s16le", str(tmp)], check=True)
        _, data = wavfile.read(tmp)
        data = data.astype(np.float32) / 32768.0
        start = int((timeline[scene] + voice_start(scene)) * sr)
        # a clip may run on until the next scene starts, not further
        nxt = VOICE_SCENES[i + 1] if i + 1 < len(VOICE_SCENES) else "end"
        limit = int(timeline.get(nxt, 120) * sr) if nxt != "end" else len(track)
        room = max(min(limit, len(track)) - start, 0)
        if len(data) > room:
            print(f"  ! {scene}: clip is {len(data) / sr:.1f}s but the scene has {room / sr:.1f}s, cut")
        n = min(len(data), room)
        track[start:start + n] += data[:n]
        found += 1
        print(f"  {scene:15s} at {start / sr:6.2f}s  {len(data) / sr:5.1f}s  <- {clip.name}")
    if not found:
        raise SystemExit(f"no audio clips found in {voice_dir}")
    # quiet clicks and key presses where they happened in the recording, well under the voice
    if EVENTS_FILE.exists():
        rng = np.random.default_rng(3)
        n_sfx = 0
        for e in json.loads(EVENTS_FILE.read_text(encoding="utf-8")):
            if e["kind"] not in ("click", "key"):
                continue
            s = _sfx(e["kind"], sr, rng) * (1.6 if e.get("strong") else 1.0)
            at = int(e["t"] * sr)
            if 0 <= at < len(track) - len(s):
                track[at:at + len(s)] += s
                n_sfx += 1
        print(f"  {n_sfx} click/key sounds mixed in")
    wavfile.write(tmp, sr, (np.clip(track, -1, 1) * 32767).astype(np.int16))

    out = OUT / "demo_final.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(OUT / "demo_silent.mp4"), "-i", str(tmp),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink(missing_ok=True)
    return out


def mux(audio: Path) -> Path:
    video = OUT / "demo_silent.mp4"
    out = OUT / "demo_final.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "160k",
                    "-t", "119.5", "-movflags", "+faststart", str(out)], check=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mux", type=Path, help="one narration file to put on demo_silent.mp4")
    ap.add_argument("--voice-dir", type=Path, help="folder with one clip per scene (intro.wav, app.wav, ...)")
    ap.add_argument("--camera", action="store_true", help="zoom in on the active area (off by default)")
    ap.add_argument("--reconvert", action="store_true",
                    help="rebuild demo_silent.mp4 from the last raw capture without recording again")
    a = ap.parse_args()
    CAMERA = a.camera
    if a.reconvert:
        webm = sorted((OUT / "raw").glob("*.webm"), key=lambda p: p.stat().st_mtime)[-1]
        to_mp4(webm, OUT / "demo_silent.mp4")
        print("wrote", OUT / "demo_silent.mp4", "from", webm.name)
        sys.exit()
    if a.voice_dir:
        print("wrote", mux_scenes(a.voice_dir))
        sys.exit()
    if a.mux:
        print("wrote", mux(a.mux))
        sys.exit()
    webm = record()
    mp4 = OUT / "demo_silent.mp4"
    to_mp4(webm, mp4)
    print("wrote", mp4)
