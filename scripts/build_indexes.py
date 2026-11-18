"""Build and save every retrieval index.

Usage:  python scripts/build_indexes.py [--skip-baselines] [--skip-bert]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS
from src.data import load_recipes
from src.retrieval.bert import BertRetriever
from src.retrieval.quantize import QuantizedBertRetriever
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.w2v import Word2VecRetriever

OUT = MODELS / "retrieval"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-bert", action="store_true")
    args = ap.parse_args()

    recipes = load_recipes()
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.skip_baselines:
        for cls in (TfidfRetriever, Word2VecRetriever):
            t = time.time()
            r = cls(recipes)
            r.save(OUT / cls.name)
            print(f"{cls.name:16s} built in {time.time() - t:5.1f}s  {r.footprint_mb():7.2f} MB")

    if not args.skip_bert:
        t = time.time()
        b = BertRetriever(recipes)
        b.save(OUT / b.name)
        print(f"{b.name:16s} built in {time.time() - t:5.1f}s  {b.footprint_mb():7.2f} MB")

        t = time.time()
        q = QuantizedBertRetriever.build(recipes, OUT / QuantizedBertRetriever.name)
        q.save(OUT / q.name)
        print(f"{q.name:16s} built in {time.time() - t:5.1f}s  {q.footprint_mb():7.2f} MB")

    for p in sorted(OUT.glob("*")):
        print(" ", p.relative_to(MODELS))


if __name__ == "__main__":
    main()
