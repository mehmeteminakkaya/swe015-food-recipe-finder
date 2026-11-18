"""Classical retrieval baseline: tf-idf + cosine (technique area B)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data import Recipe, recipe_text
from .base import Retriever, l2norm
from .text import analyzer, strip_constraints


class TfidfRetriever(Retriever):
    name = "tfidf"

    def __init__(self, recipes: list[Recipe]):
        super().__init__(recipes)
        corpus = [recipe_text(r) for r in recipes]
        self.vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True, min_df=1)
        self.matrix = l2norm(self.vectorizer.fit_transform(corpus).toarray())

    def scores(self, text: str, rows: list[int]) -> np.ndarray:
        q = self.vectorizer.transform([strip_constraints(text)]).toarray()[0]
        return self.matrix[rows] @ l2norm(q)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "tfidf.pkl").open("wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, out: Path, recipes: list[Recipe]) -> "TfidfRetriever":
        with (out / "tfidf.pkl").open("rb") as f:
            d = pickle.load(f)
        self = cls.__new__(cls)
        Retriever.__init__(self, recipes)
        self.vectorizer = d["vectorizer"]
        self.matrix = d["matrix"]
        return self

    def footprint_mb(self) -> float:
        vocab_bytes = sum(len(t) for t in self.vectorizer.vocabulary_)
        return (self.matrix.nbytes + vocab_bytes) / 1e6
