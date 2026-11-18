"""Evaluate every model on the shared held-out test data.

Retrieval: 40 hand-written queries (20 TR / 20 EN). Constraints are applied as a
hard filter (same pool for every model); relevance = target dish + constraints.
Vision: the 2 499 Food-101 test images (top-1 / top-5 accuracy).

Results are written to results/metrics.json.

Usage:  python scripts/evaluate.py [--retrieval-only] [--vision-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, RESULTS
from src.data import candidate_recipes, load_queries, load_recipes, relevant_ids
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.w2v import Word2VecRetriever

K = 5


def recall_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(k, len(relevant))


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, rid in enumerate(ranked, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(models: dict) -> dict:
    recipes = load_recipes()
    queries = load_queries()
    out = {}
    for name, retriever in models.items():
        rows = []
        for q in queries:
            cand = candidate_recipes(q, recipes)
            ranked = [r["id"] for r, _ in retriever.search(q["text"], cand, k=len(cand))]
            rel = relevant_ids(q, recipes)
            rows.append({
                "id": q["id"], "lang": q["lang"],
                "recall@5": recall_at_k(ranked, rel),
                "rr": reciprocal_rank(ranked, rel),
                "n_relevant": len(rel),
            })
        by_lang = {}
        for lang in ("tr", "en"):
            sub = [r for r in rows if r["lang"] == lang]
            by_lang[lang] = {
                "recall@5": float(np.mean([r["recall@5"] for r in sub])),
                "mrr": float(np.mean([r["rr"] for r in sub])),
                "n": len(sub),
            }
        out[name] = {
            "recall@5": float(np.mean([r["recall@5"] for r in rows])),
            "mrr": float(np.mean([r["rr"] for r in rows])),
            "n_queries": len(rows),
            "by_lang": by_lang,
            "per_query": rows,
        }
        print(f"{name:20s} Recall@5={out[name]['recall@5']:.3f}  MRR={out[name]['mrr']:.3f}  "
              f"(TR {by_lang['tr']['recall@5']:.3f} / EN {by_lang['en']['recall@5']:.3f})")
    return out


def load_retrievers(names: list[str] | None) -> dict:
    from src.retrieval.bert import BertRetriever
    from src.retrieval.quantize import QuantizedBertRetriever

    recipes = load_recipes()
    root = MODELS / "retrieval"
    available = {
        "tfidf": lambda: TfidfRetriever.load(root / "tfidf", recipes),
        "word2vec": lambda: Word2VecRetriever.load(root / "word2vec", recipes),
        "bert_e5": lambda: BertRetriever.load(root / "bert_e5", recipes),
        "bert_e5_int8": lambda: QuantizedBertRetriever.load(root / "bert_e5_int8", recipes),
    }
    wanted = names or list(available)
    return {n: available[n]() for n in wanted}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--vision-only", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    if not args.vision_only:
        metrics["retrieval"] = evaluate_retrieval(load_retrievers(args.models))
    if not args.retrieval_only:
        from src.vision.evaluate import evaluate_vision

        metrics["vision"] = evaluate_vision()

    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
