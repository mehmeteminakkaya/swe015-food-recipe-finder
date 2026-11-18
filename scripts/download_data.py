"""Download the prepared artifacts (image subset + model weights) from the Hub.

The repository itself only carries code and small JSON/JSONL files: the 740 MB
image subset and the model weights live in a Hugging Face repo.

    set HF_REPO=<user>/swe015-food-recipe-finder
    python scripts/download_data.py

Without HF_REPO the script rebuilds the data locally with prepare_data.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA, IMAGES, MODELS, RECIPES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_REPO", ""),
                    help="Hugging Face repo id that holds models/ and the image tarball")
    args = ap.parse_args()

    if not args.repo:
        print("No --repo/HF_REPO given; preparing the data locally instead.")
        os.execv(sys.executable, [sys.executable, "scripts/prepare_data.py"])

    from huggingface_hub import snapshot_download

    print(f"downloading {args.repo} -> models/")
    snapshot_download(repo_id=args.repo, repo_type="model", local_dir=str(MODELS),
                      allow_patterns=["*.pt", "*.onnx", "*.json", "*.npy", "*.kv", "*.pkl"])

    if not IMAGES.exists():
        print("images missing; download them with: python scripts/prepare_data.py --skip-recipes")

    print("recipes:", RECIPES.exists(), "| model files:", len(list(MODELS.rglob('*.*'))))


if __name__ == "__main__":
    main()
