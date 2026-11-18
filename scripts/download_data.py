"""Fetch the runtime weights so the app can start without training.

Default source: the GitHub release of this repository (~200 MB zip: small CNN,
VGG16 fp32 + INT8, tf-idf, word2vec, E5 INT8 indexes). The multilingual E5
encoder itself is pulled from the Hugging Face cache on first use.

    python scripts/download_data.py                 # GitHub release
    python scripts/download_data.py --hf-repo X/Y   # Hugging Face repo instead

Images and recipes are not in the release; build them with prepare_data.py.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES, MODELS, ROOT

RELEASE_URL = ("https://github.com/mehmeteminakkaya/swe015-food-recipe-finder/"
               "releases/latest/download/swe015-models.zip")


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=RELEASE_URL)
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
        tmp = ROOT / "data" / "cache" / "swe015-models.zip"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        fetch(args.url, tmp)
        print("extracting…")
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(ROOT)
        tmp.unlink()

    if not args.skip_encoder:
        prefetch_encoder()

    required = [MODELS / "vgg16_ft_int8.onnx", MODELS / "small_cnn.pt",
                MODELS / "retrieval" / "bert_e5_int8" / "encoder_int8.onnx"]
    for p in required:
        print(f"  {'ok ' if p.exists() else 'MISSING'} {p.relative_to(ROOT)}")
    if not IMAGES.exists():
        print("\nimages are missing: python scripts/prepare_data.py --skip-recipes")
    print("\nnext: python app.py")


if __name__ == "__main__":
    main()
