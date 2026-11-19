"""Fetch the prepared artifacts so the app can start without training.

Default: the runtime weights from the GitHub release (~200 MB: small CNN,
VGG16 fp32 + INT8, tf-idf, word2vec, E5 INT8 indexes). `--with-images` adds the
10-class image subset (~720 MB) instead of running prepare_data.py for an hour.
The multilingual E5 encoder itself is pulled from Hugging Face on first use.

    python scripts/download_data.py                    # weights only
    python scripts/download_data.py --with-images      # weights + images
    python scripts/download_data.py --hf-repo X/Y      # Hugging Face repo instead
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA, IMAGES, MODELS, RAW, ROOT

REPO = "https://github.com/mehmeteminakkaya/swe015-food-recipe-finder/releases"
BASE = f"{REPO}/latest/download"
RELEASE_URL = f"{BASE}/swe015-models.zip"  # weights + indexes for the current recipe corpus
# the 10-class image subset never changes, so it stays in the first release
IMAGES_URL = f"{REPO}/download/v1.0-weights/swe015-food10.zip"


def fetch(url: str, dest: Path) -> None:
    def hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total) / total
            print(f"\r  {done:5.1%}  {min(blocks * block_size, total) / 1e6:6.1f} MB", end="")

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=hook)
    print("\ndone", f"{dest.stat().st_size / 1e6:.1f} MB")


def prefetch_encoder() -> None:
    from sentence_transformers import SentenceTransformer

    from src.retrieval.bert import MODEL_NAME

    print("caching the E5 encoder for offline use…")
    SentenceTransformer(MODEL_NAME, device="cpu")
    print("ok")


def fetch_images(url: str) -> None:
    tmp = DATA / "cache" / "swe015-food10.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    fetch(url, tmp)
    print("extracting images…")
    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp) as zf:
        zf.extractall(RAW)  # archive root is food10/
    tmp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=RELEASE_URL)
    ap.add_argument("--with-images", action="store_true",
                    help="also download the 10-class image subset (~720 MB)")
    ap.add_argument("--hf-repo", default=None,
                    help="download a Hugging Face model repo instead of the release")
    ap.add_argument("--skip-encoder", action="store_true")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)

    if args.hf_repo:
        from huggingface_hub import snapshot_download

        print(f"downloading {args.hf_repo} -> {MODELS}")
        snapshot_download(repo_id=args.hf_repo, repo_type="model", local_dir=str(MODELS),
                          allow_patterns=["*.pt", "*.onnx", "*.json", "*.npy", "*.kv", "*.pkl"])
    else:
        tmp = DATA / "cache" / "swe015-models.zip"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        fetch(args.url, tmp)
        print("extracting weights…")
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(ROOT)  # archive root is models/
        tmp.unlink()

    if args.with_images and not IMAGES.exists():
        fetch_images(IMAGES_URL)

    if not args.skip_encoder:
        prefetch_encoder()

    required = [MODELS / "vgg16_ft_int8.onnx", MODELS / "small_cnn.pt",
                MODELS / "retrieval" / "bert_e5_int8" / "encoder_int8.onnx"]
    for p in required:
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p.relative_to(ROOT)}")
    n = sum(1 for _ in IMAGES.rglob("*.jpg")) if IMAGES.exists() else 0
    print(f"  {'ok ' if n else 'MISSING'} images ({n} JPEG)")
    if not n:
        print("\nimages are missing: python scripts/download_data.py --with-images  "
              "(or prepare_data.py)")
    print("\nnext: python app.py")


if __name__ == "__main__":
    main()
