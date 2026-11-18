"""Dataset loading and the rule-based notion of relevance.

Images live in data/raw/food10/{train,val,test}/{class}/*.jpg, recipes in
data/recipes.jsonl, test queries in data/queries.jsonl.

A recipe is *relevant* for a query when both hold:
  1. the recipe belongs to the query's target dish, and
  2. it satisfies the query's constraints (max total minutes, diet).

Constraints are extracted from the query text by src.retrieval.text and are
applied as a hard filter for every retrieval model, so the models differ only
in ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import IMAGES, QUERIES, RECIPES

Recipe = dict[str, Any]
Query = dict[str, Any]


def load_recipes(path: Path = RECIPES) -> list[Recipe]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_queries(path: Path = QUERIES) -> list[Query]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recipe_text(rec: Recipe) -> str:
    """The text a retrieval model indexes for one recipe."""
    parts = [
        rec["name"],
        rec.get("description", ""),
        rec.get("category", ""),
        " ".join(rec.get("tags") or []),
        " ".join(rec.get("ingredients") or []),
    ]
    return " ".join(p for p in parts if p)


def satisfies(rec: Recipe, query: Query) -> bool:
    """True when the recipe obeys the query's constraints."""
    max_minutes = query.get("max_minutes")
    if max_minutes is not None and rec["minutes"] > max_minutes:
        return False
    diet = query.get("diet") or []
    if "vegan" in diet and not rec["vegan"]:
        return False
    if "vegetarian" in diet and not rec["vegetarian"]:
        return False
    return True


def is_relevant(rec: Recipe, query: Query) -> bool:
    return rec["dish"] == query["target_dish"] and satisfies(rec, query)


def candidate_recipes(query: Query, recipes: list[Recipe]) -> list[Recipe]:
    """The hard-filtered pool every model ranks (same for all models)."""
    return [r for r in recipes if satisfies(r, query)]


def relevant_ids(query: Query, recipes: list[Recipe]) -> set[str]:
    return {r["id"] for r in recipes if is_relevant(r, query)}


def image_splits():
    """torchvision ImageFolder datasets for train/val/test (no transforms)."""
    from torchvision.datasets import ImageFolder

    return {s: ImageFolder(IMAGES / s) for s in ("train", "val", "test")}
