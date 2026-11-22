"""Word timestamps for every narration clip in dist/video/voice/.

    python scripts/voice_timing.py      ->  dist/video/voice/words.json

scripts/record_demo.py uses them to point the cursor at what is being said
("heat map", "vegan", ...) at the moment it is said.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.config  # noqa: F401  -- keeps the Hugging Face cache inside the repo

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

VOICE = Path(__file__).resolve().parent.parent / "dist" / "video" / "voice"


def main() -> None:
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device=-1)
    out = {}
    for f in sorted(VOICE.glob("*.wav")):
        sr, x = wavfile.read(f)
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
        if x.ndim > 1:
            x = x.mean(axis=1)
        x16 = resample_poly(x, 16000, sr).astype(np.float32)
        r = asr({"raw": x16, "sampling_rate": 16000}, return_timestamps="word")
        out[f.stem] = [{"w": c["text"].strip(), "s": round(c["timestamp"][0], 2),
                        "e": round(c["timestamp"][1] or c["timestamp"][0], 2)} for c in r["chunks"]]
        print(f"{f.stem:15s} {r['text'].strip()[:70]}")
    (VOICE / "words.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", VOICE / "words.json")


if __name__ == "__main__":
    main()
