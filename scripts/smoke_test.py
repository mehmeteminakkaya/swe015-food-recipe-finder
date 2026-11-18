"""End-to-end smoke test of the app functions (no browser needed).

Usage:  python scripts/smoke_test.py [--image path/to.jpg]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="data/raw/food10/test/pizza/train_01501.jpg")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    for photo_model in app.PHOTO_MODELS:
        cam, dish, top5, info = app.predict_photo(img, photo_model)
        assert cam is not None and dish, "photo pipeline returned nothing"
        print(f"[{photo_model}] {info} -> {dish}")
        print("   ", top5.replace("\n", " | "))
        for text_model in app.TEXT_MODELS:
            rows, info2 = app.search("vejetaryen, 30 dakikadan kısa", dish, text_model)
            assert len(rows) == 5, f"{text_model}: {len(rows)} rows"
            print(f"    [{text_model}] {info2}")
            print("      ", " · ".join(r[0] for r in rows))


if __name__ == "__main__":
    main()
