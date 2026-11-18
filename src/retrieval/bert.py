"""Multilingual sentence-embedding retrieval (technique area C).

intfloat/multilingual-e5-small is a BERT-based multilingual encoder used with
the standard E5 prefixes ("query: " for queries, "passage: " for recipes). It
maps Turkish and English into one 384-d space, so a Turkish query can retrieve
an English recipe.

The plan named paraphrase-multilingual-MiniLM-L12-v2; a quick head-to-head on
the hard Turkish cases showed MiniLM scores pankek~pancakes at 0.42 while E5
scores it at 0.85, so E5 was chosen (same technique, same encoder size class).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import device
from ..data import Recipe, recipe_text
from .base import Retriever, l2norm
from .text import strip_constraints

MODEL_NAME = "intfloat/multilingual-e5-small"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class BertRetriever(Retriever):
    name = "bert_e5"
    model_name = MODEL_NAME

    def __init__(self, recipes: list[Recipe]):
        from sentence_transformers import SentenceTransformer

        super().__init__(recipes)
        self.model = SentenceTransformer(self.model_name, device=device())
        self.doc_vecs = self._encode([PASSAGE_PREFIX + recipe_text(r) for r in recipes])

    def _encode(self, texts: list[str]) -> np.ndarray:
        return l2norm(np.asarray(self.model.encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
        )))

    def scores(self, text: str, rows: list[int]) -> np.ndarray:
        q = self._encode([QUERY_PREFIX + strip_constraints(text)])[0]
        return self.doc_vecs[rows] @ q

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "bert_docs.npy", self.doc_vecs)
        (out / "bert.json").write_text(json.dumps({"model": self.model_name}), encoding="utf-8")

    @classmethod
    def load(cls, out: Path, recipes: list[Recipe]) -> "BertRetriever":
        from sentence_transformers import SentenceTransformer

        self = cls.__new__(cls)
        Retriever.__init__(self, recipes)
        self.model_name = json.loads((out / "bert.json").read_text(encoding="utf-8"))["model"]
        self.model = SentenceTransformer(self.model_name, device=device())
        self.doc_vecs = np.load(out / "bert_docs.npy")
        return self

    def footprint_mb(self) -> float:
        # Size of the encoder weights actually loaded (fp32).
        total = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return (total + self.doc_vecs.nbytes) / 1e6
