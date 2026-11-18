"""Shared retriever interface.

Every retrieval model indexes the same 300 recipes and scores a hard-filtered
candidate list produced by src.data.candidate_recipes. Only the ranking differs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..data import Recipe


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def top_k(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k best scores, best first (stable on ties)."""
    return list(np.argsort(-scores, kind="stable")[:k])


class Retriever(ABC):
    name: str = "base"

    def __init__(self, recipes: list[Recipe]):
        self.recipes = recipes
        self.doc_ids = [r["id"] for r in recipes]
        self.row = {rid: i for i, rid in enumerate(self.doc_ids)}

    @abstractmethod
    def scores(self, text: str, rows: list[int]) -> np.ndarray:
        """Similarity of `text` to the recipes at the given corpus rows."""

    def search(self, text: str, candidates: list[Recipe], k: int = 5) -> list[tuple[Recipe, float]]:
        rows = [self.row[r["id"]] for r in candidates]
        s = self.scores(text, rows)
        order = top_k(s, k)
        return [(candidates[i], float(s[i])) for i in order]

    def save(self, out: Path) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @abstractmethod
    def footprint_mb(self) -> float:
        """Size of the model assets this retriever needs on disk (float32)."""
