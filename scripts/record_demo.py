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

CURSOR_JS = """
(() => {
  if (document.getElementById('demo-cursor')) return;
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  c.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;' +
    'background:rgba(23,48,31,.35);border:2px solid #17301F;z-index:999999;pointer-events:none;' +
    'transition:left .55s ease, top .55s ease, transform .15s ease;left:720px;top:450px';
  document.body.appendChild(c);
  window.__demoClick = () => { c.style.transform = 'scale(.6)'; setTimeout(() => c.style.transform = '', 160); };
})();
"""


class Recorder:
    def __init__(self, page):
        self.page = page
        self.t0 = time.monotonic()
        self.timeline: list[dict] = []

    def mark(self, scene: str) -> None:
        t = round(time.monotonic() - self.t0, 2)
        self.timeline.append({"scene": scene, "t": t})
        print(f"{t:7.2f}s  {scene}")

    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)

    def cursor_to(self, locator) -> None:
        self.page.evaluate(CURSOR_JS)
        locator.scroll_into_view_if_needed()
        box = locator.bounding_box()
        if box:
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            self.page.evaluate(f"(() => {{ const c = document.getElementById('demo-cursor'); c.style.left='{x}px'; c.style.top='{y}px'; }})()")
            self.wait(650)

    def click(self, locator) -> None:
        self.cursor_to(locator)
        self.page.evaluate("window.__demoClick && window.__demoClick()")
        locator.click()

    def scroll_to(self, selector: str, block: str = "start") -> None:
        self.page.evaluate(
            f"document.querySelector('{selector}').scrollIntoView({{behavior:'smooth', block:'{block}'}})")
        self.wait(900)

    def scroll_top(self) -> None:
        self.page.evaluate("window.scrollTo({top:0, behavior:'smooth'})")
        self.wait(900)


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


def search(rec: Recorder, text: str) -> None:
    """Type a request and press Enter, like a user would."""
    page = rec.page
    box = page.locator("#rf-query textarea, #rf-query input").first
    rec.click(box)
    page.keyboard.press("Control+A")
    page.keyboard.press("Backspace")
    box.press_sequentially(text, delay=50)
    rec.wait(250)
    before = page.inner_html("#rf-recipes")
    page.keyboard.press("Enter")
    wait_recipes(page, before)


def glide(rec: Recorder, px: int, ms: int) -> None:
    """Scroll slowly so the viewer can read along."""
    steps = max(ms // 40, 1)
    for _ in range(steps):
        rec.page.mouse.wheel(0, px / steps)
        rec.page.wait_for_timeout(40)


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
        warm.wait_for_selector("#rf-btn-photo")
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
        rec.wait(13000)

        # 0:15 live demo
        page.goto(URL)
        page.wait_for_selector("#rf-btn-photo")
        page.evaluate(CURSOR_JS)
        rec.mark("app")
        rec.wait(2500)

        # easy case: clicking an example photo recognises it straight away
        rec.mark("pizza")
        rec.click(page.locator("#rf-examples button.gallery-item").nth(0))
        wait_dish(page, "")
        rec.wait(600)
        rec.scroll_to("#rf-pred", "center")
        rec.wait(5400)
        rec.scroll_top()

        # request + Enter: the app scrolls to the recipes, then we read down the best one
        rec.mark("query")
        search(rec, "vegetarian, under 30 minutes")
        rec.wait(1600)
        glide(rec, 650, 2600)
        rec.wait(1800)
        rec.scroll_top()

        # difficult case: falafel uploaded from disk, recognised on upload, predicted as bolognese
        rec.mark("falafel_upload")
        prev = dish_text(page)
        rec.click(page.locator("#rf-photo button[aria-label='Remove Image']").first)
        rec.wait(500)
        page.locator("#rf-photo input[type=file]").set_input_files(str(FALAFEL))
        wait_dish(page, "")
        rec.wait(600)
        rec.scroll_to("#rf-pred", "center")
        rec.wait(6800)
        rec.scroll_top()

        # ramen example + a vegan request
        rec.mark("ramen_en")
        prev = dish_text(page)
        rec.click(page.locator("#rf-examples button.gallery-item").nth(4))
        wait_dish(page, prev)
        rec.wait(900)
        search(rec, "vegan ramen under 30 minutes")
        rec.wait(1400)
        glide(rec, 900, 2400)
        rec.wait(900)

        # every model side by side for the same photo and request
        rec.mark("compare")
        page.wait_for_selector("#rf-cmp-search .rf-cmp-4", timeout=30000)
        rec.scroll_to(".rf-compare", "start")
        rec.wait(3000)
        glide(rec, 380, 1500)
        rec.wait(3500)

        # ~1:10 architecture and results
        page.set_content(arch_card(), wait_until="networkidle")
        rec.mark("architecture")
        rec.wait(12500)
        page.set_content(results_card(), wait_until="networkidle")
        rec.mark("results")
        rec.wait(15500)

        # ~1:40 conclusion
        page.set_content(outro_card(), wait_until="networkidle")
        rec.mark("conclusion")
        rec.wait(13000)
        rec.mark("end")

        video_path = Path(page.video.path())
        ctx.close()
        browser.close()

    (OUT / "timeline.json").write_text(json.dumps(rec.timeline, indent=2), encoding="utf-8")
    return video_path


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def to_mp4(webm: Path, out: Path) -> None:
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(webm), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-t", "120", "-movflags", "+faststart",
                    str(out)], check=True)


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
        start = int((timeline[scene] + 0.3) * sr)
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
    wavfile.write(tmp, sr, (np.clip(track, -1, 1) * 32767).astype(np.int16))

    out = OUT / "demo_final.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(OUT / "demo_silent.mp4"), "-i", str(tmp),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink(missing_ok=True)
    return out


def mux(audio: Path) -> Path:
    video = OUT / "demo_silent.mp4"
    out = OUT / "demo_final.mp4"
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-t", "120", "-movflags", "+faststart", str(out)], check=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mux", type=Path, help="one narration file to put on demo_silent.mp4")
    ap.add_argument("--voice-dir", type=Path, help="folder with one clip per scene (intro.wav, app.wav, ...)")
    a = ap.parse_args()
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
