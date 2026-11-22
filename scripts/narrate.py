"""Record your own narration over the silent demo video, with a teleprompter.

    python scripts/narrate.py          # then open http://127.0.0.1:8765 in Chrome

The page plays dist/video/demo_silent.mp4 and shows, under it, the sentence to
read for the current scene (from docs/VIDEO_NARRATION.md, timed by
dist/video/timeline.json) with a bar for the time left and the next sentence.
"Start" counts down 3 s, then starts the video and the microphone together, so
the voice lines up with the picture. After "Stop" you can listen, retake, or
"Save": the take is written to dist/video/voice_take.webm and muxed into
dist/video/demo_final.mp4.

Uses only the standard library plus imageio-ffmpeg (already needed by record_demo.py).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "dist" / "video"
SILENT = VIDEO_DIR / "demo_silent.mp4"
FINAL = VIDEO_DIR / "demo_final.mp4"
TAKE = VIDEO_DIR / "voice_take.webm"
PORT = 8765

SCENES = ["intro", "app", "query", "falafel_upload", "ramen_en", "compare", "architecture", "results", "conclusion"]


def script() -> list[dict]:
    text = (ROOT / "docs" / "VIDEO_NARRATION.md").read_text(encoding="utf-8")
    lines = {m.group(1): m.group(2).strip()
             for m in re.finditer(r"^### (\w+) \([^)]*\)\n\n(.+?)(?=\n\n)", text, re.M | re.S)}
    tl = {e["scene"]: e["t"] for e in json.loads((VIDEO_DIR / "timeline.json").read_text(encoding="utf-8"))}
    out = []
    for i, s in enumerate(SCENES):
        end = tl[SCENES[i + 1]] if i + 1 < len(SCENES) else 119.5
        out.append({"scene": s, "start": tl[s] + 0.3, "end": end, "text": lines.get(s, "")})
    return out


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Narration</title>
<style>
:root{--paper:#F1F4EF;--ink:#17301F;--muted:#5B6A60;--line:#D3DCD4;--saffron:#E2A21B;--tomato:#C4432B}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px 16px 24px;display:grid;gap:12px}
/* the video stays small enough that the sentence to read is always on screen */
video{width:100%;max-height:50vh;object-fit:contain;border-radius:12px;border:1px solid var(--line);background:#000}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button{font:700 16px system-ui;padding:10px 18px;border-radius:10px;border:0;cursor:pointer;background:#E5EBE6;color:var(--ink)}
button.go{background:var(--saffron)} button.rec{background:var(--tomato);color:#fff} button:disabled{opacity:.45;cursor:default}
#status{color:var(--muted)} #status b{color:var(--ink)}
.prompt{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 22px}
.now{font:400 34px/1.35 Georgia,serif;min-height:3.2em}
.scene{font-size:13px;color:var(--muted);font-weight:700}
.track{height:8px;background:#E5EBE6;border-radius:99px;overflow:hidden;margin:12px 0 10px}
.track span{display:block;height:100%;background:var(--saffron);width:0}
.next{color:var(--muted);font-size:17px} .next b{color:var(--ink)}
.count{position:fixed;inset:0;display:none;place-items:center;background:rgba(23,48,31,.55);color:#fff;font:700 160px system-ui}
.level{width:120px;height:10px;background:#E5EBE6;border-radius:99px;overflow:hidden}.level span{display:block;height:100%;background:#3D7A4C;width:0}
audio{width:100%}
</style></head><body><main>
<video id="v" src="/video" preload="auto" playsinline></video>
<div class="bar">
  <button id="start" class="go">Start (3 s countdown)</button>
  <button id="stop" class="rec" disabled>Stop</button>
  <button id="save" disabled>Save and build demo_final.mp4</button>
  <span class="level" title="microphone level"><span id="lvl"></span></span>
  <span id="status">Allow the microphone, then press Start. Read each sentence while it is shown.</span>
</div>
<section class="prompt">
  <div class="scene" id="scene">—</div>
  <div class="now" id="now">The sentence to read appears here.</div>
  <div class="track"><span id="left"></span></div>
  <div class="next" id="next"></div>
</section>
<audio id="take" controls hidden></audio>
<div class="count" id="count">3</div>
</main><script>
const S = __SCRIPT__;
const v = document.getElementById('v'), $ = id => document.getElementById(id);
let rec, chunks = [], blob = null, stream;

function show(t) {
  let i = S.findIndex(s => t >= s.start - 0.3 && t < s.end);
  if (i < 0) { $('now').textContent = t < S[0].start ? '(get ready)' : '(done — stay quiet until the end)'; $('left').style.width = '0'; return; }
  const s = S[i], n = S[i + 1];
  $('scene').textContent = `${i + 1}/${S.length} · ${s.scene} · ${Math.round(s.end - s.start)} s`;
  $('now').textContent = s.text;
  $('left').style.width = (100 * (s.end - t) / (s.end - s.start)).toFixed(1) + '%';
  $('next').innerHTML = n ? `<b>Next:</b> ${n.text}` : '';
}
v.addEventListener('timeupdate', () => show(v.currentTime));
(function loop(){ if (!v.paused) show(v.currentTime); requestAnimationFrame(loop); })();

async function mic() {
  if (stream) return stream;
  stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: true, noiseSuppression: true}});
  const ctx = new AudioContext(), an = ctx.createAnalyser(); ctx.createMediaStreamSource(stream).connect(an);
  const d = new Uint8Array(an.fftSize);
  (function meter(){ an.getByteTimeDomainData(d); let m = 0; for (const x of d) m = Math.max(m, Math.abs(x - 128));
    $('lvl').style.width = Math.min(100, m / 1.28) + '%'; requestAnimationFrame(meter); })();
  return stream;
}

$('start').onclick = async () => {
  try { await mic(); } catch (e) { $('status').textContent = 'Microphone blocked: allow it in the address bar and reload.'; return; }
  $('start').disabled = $('save').disabled = true; $('take').hidden = true;
  v.pause(); v.currentTime = 0; show(0);
  for (const n of [3, 2, 1]) { $('count').style.display = 'grid'; $('count').textContent = n; await new Promise(r => setTimeout(r, 1000)); }
  $('count').style.display = 'none';
  chunks = []; rec = new MediaRecorder(stream, {mimeType: 'audio/webm'});
  rec.ondataavailable = e => chunks.push(e.data);
  rec.onstop = () => { blob = new Blob(chunks, {type: 'audio/webm'}); $('take').src = URL.createObjectURL(blob);
    $('take').hidden = false; $('save').disabled = false; $('start').disabled = false; $('start').textContent = 'Retake';
    $('status').innerHTML = 'Listen to the take below. <b>Save</b> builds the video, <b>Retake</b> starts again.'; };
  rec.start(); await v.play();                       // picture and microphone start together
  $('stop').disabled = false; $('status').innerHTML = '<b>Recording…</b> read the sentence in the box.';
};
const finish = () => { if (rec && rec.state === 'recording') { rec.stop(); v.pause(); $('stop').disabled = true; } };
$('stop').onclick = finish; v.addEventListener('ended', finish);

$('save').onclick = async () => {
  $('save').disabled = true; $('status').textContent = 'Saving and building demo_final.mp4…';
  const r = await fetch('/save', {method: 'POST', body: blob}); const j = await r.json();
  $('status').innerHTML = j.ok ? `Done: <b>${j.path}</b>` : `Failed: ${j.error}`; $('save').disabled = false;
};
</script></body></html>"""


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_final() -> Path:
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(SILENT), "-i", str(TAKE),
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-t", "119.5", "-movflags", "+faststart",
                    str(FINAL)], check=True)
    return FINAL


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_video(self) -> None:
        """Serve the MP4 with HTTP Range support so the browser can seek and buffer."""
        data = SILENT.read_bytes()
        size = len(data)
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if not m:
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            self.wfile.write(data)
            return
        start = int(m.group(1)) if m.group(1) else max(size - int(m.group(2)), 0)
        end = int(m.group(2)) if m.group(1) and m.group(2) else size - 1
        end = min(end, size - 1)
        self.send_response(206)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start:end + 1])

    def do_GET(self):
        if self.path == "/":
            page = PAGE.replace("__SCRIPT__", json.dumps(script(), ensure_ascii=False))
            return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/video":
            return self._send_video()
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, b"not found", "text/plain")
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            TAKE.write_bytes(data)
            out = build_final()
            print(f"saved {TAKE.name} ({len(data) / 1e6:.1f} MB) -> {out}")
            body = {"ok": True, "path": str(out.relative_to(ROOT))}
        except Exception as e:  # report back to the page instead of failing silently
            body = {"ok": False, "error": str(e)}
        self._send(200, json.dumps(body).encode(), "application/json")


if __name__ == "__main__":
    if not SILENT.exists():
        sys.exit("record the silent video first: python scripts/record_demo.py")
    print(f"open http://127.0.0.1:{PORT} in Chrome (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
