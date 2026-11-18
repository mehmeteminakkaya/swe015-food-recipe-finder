"""Publish the model weights to the Hugging Face Hub (keeps them out of git).

    huggingface-cli login                 # once
    python scripts/upload_models.py --repo <user>/swe015-food-recipe-finder

The images stay in data/raw and are rebuilt with scripts/prepare_data.py; only
models/ is uploaded, because that is what the app and the download script need.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="<user>/<repo> on the Hub")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    print(f"uploading {MODELS} -> {args.repo}")
    api.upload_folder(folder_path=str(MODELS), repo_id=args.repo, repo_type="model",
                      commit_message="SWE015 UC3 model weights")
    print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
