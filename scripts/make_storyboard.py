"""Build a one-page storyboard of the demo video for review before the final cut.

Needs a recording first (python scripts/record_demo.py). Writes
dist/video/storyboard.html with one frame per scene, its start time and the
narration text from docs/VIDEO_NARRATION.md.
"""
from __future__ import annotations

import base64
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "video"
SCENES = [
    ("intro", "Problem"), ("app", "Easy case: pizza"), ("query", "Request and recipes"),
    ("falafel_upload", "Hard case: falafel upload"), ("ramen_en", "Ramen, vegan request"),
    ("compare", "Every model side by side"), ("architecture", "Architecture"),
    ("results", "Results table"), ("conclusion", "Conclusion"),
]


def narration() -> dict[str, str]:
    text = (ROOT / "docs" / "VIDEO_NARRATION.md").read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^### (\w+) \([^)]*\)\n\n(.+?)(?=\n\n)", text, re.M | re.S):
        out[m.group(1)] = m.group(2).strip()
    return out


def frame(scene: str, t: float) -> str:
    import imageio_ffmpeg

    path = OUT / "storyboard" / f"{scene}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
                    "-i", str(OUT / "demo_silent.mp4"), "-frames:v", "1", "-vf", "scale=960:-1",
                    "-q:v", "4", str(path)], check=True)
    return base64.b64encode(path.read_bytes()).decode()


def main() -> None:
    tl = json.loads((OUT / "timeline.json").read_text(encoding="utf-8"))
    starts = {e["scene"]: e["t"] for e in tl}
    order = [e["scene"] for e in tl]
    text = narration()
    cards = []
    for i, (scene, title) in enumerate(SCENES, 1):
        start = starts[scene]
        nxt = next((starts[s] for s in order[order.index(scene) + 1:] if s in starts), start + 14)
        if scene == "app":  # the pizza scene belongs to the same narration clip
            nxt = starts["query"]
        shot = max(start + 1, nxt - 1.6)
        words = len(text.get(scene, "").split())
        cards.append(
            f"<article><img src='data:image/jpeg;base64,{frame(scene, shot)}' alt='{html.escape(title)}'>"
            f"<div class='meta'><span class='n'>{i}</span><h2>{html.escape(title)}</h2>"
            f"<span class='t'>{int(start // 60)}:{int(start % 60):02d}–{int(nxt // 60)}:{int(nxt % 60):02d} · "
            f"{nxt - start:.0f} s · {words} words · <code>{scene}.wav</code></span></div>"
            f"<p>{html.escape(text.get(scene, '(no narration)'))}</p></article>"
        )
    total = tl[-1]["t"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Demo storyboard</title>
<style>
:root{{--paper:#F1F4EF;--ink:#17301F;--muted:#5B6A60;--line:#D3DCD4;--saffron:#E2A21B}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font:400 36px/1.1 Georgia,serif;margin:0 0 6px}} .sub{{color:var(--muted);margin:0 0 28px}}
article{{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:4px 24px;background:#fff;
  border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}}
img{{width:100%;border-radius:8px;border:1px solid var(--line);grid-row:span 2}}
.meta h2{{display:inline;font:600 19px/1.3 system-ui,sans-serif;margin:0}}
.n{{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:50%;background:var(--ink);
  color:#fff;font-size:13px;margin-right:8px}}
.t{{display:block;color:var(--muted);font-size:13.5px;margin-top:6px}}
code{{background:#FBF0D5;padding:1px 6px;border-radius:5px;font-size:12.5px}}
p{{margin:0;font-size:16.5px}}
@media (max-width:760px){{article{{grid-template-columns:1fr}} img{{grid-row:auto}}}}
</style></head><body><main>
<h1>Demo video storyboard</h1>
<p class="sub">Silent draft recorded by scripts/record_demo.py · total {int(total // 60)}:{int(total % 60):02d} ·
each frame is taken near the end of its scene, so it shows the result. Narration is the text to be spoken.</p>
{''.join(cards)}
</main></body></html>"""
    out = OUT / "storyboard.html"
    out.write_text(page, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
