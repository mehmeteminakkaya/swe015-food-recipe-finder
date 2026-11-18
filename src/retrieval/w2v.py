"""Vector-space retrieval with our own word2vec (technique area B, the NN step).

A skip-gram word2vec model is trained on the 300 tokenized recipes; a recipe
(and a query) is represented by the mean of its token vectors and ranked by
cosine similarity. Nothing is pretrained -- the embeddings come from our corpus.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from ..config import SEED
from ..data import Recipe, recipe_text
from .base import Retriever, l2norm
from .text import analyzer, strip_constraints

DIM = 100


def _mean_vec(tokens: list[str], wv) -> np.ndarray:
    vecs = [wv[t] for t in tokens if t in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(DIM, dtype=np.float32)


class Word2VecRetriever(Retriever):
    name = "word2vec"

    def __init__(self, recipes: list[Recipe]):
        super().__init__(recipes)
        tokenized = [analyzer(recipe_text(r)) for r in recipes]
        model = Word2Vec(
            tokenized, vector_size=DIM, window=5, min_count=1, sg=1,
            workers=4, epochs=100, seed=SEED, negative=10, sample=1e-3,
        )
        self.wv = model.wv
        self.doc_vecs = l2norm(np.stack([_mean_vec(t, self.wv) for t in tokenized]))

    def _query_vec(self, text: str) -> np.ndarray:
        return l2norm(_mean_vec(analyzer(strip_constraints(text)), self.wv))

    def scores(self, text: str, rows: list[int]) -> np.ndarray:
        return self.doc_vecs[rows] @ self._query_vec(text)

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.wv.save(str(out / "w2v.kv"))
        np.save(out / "w2v_docs.npy", self.doc_vecs)

    @classmethod
    def load(cls, out: Path, recipes: list[Recipe]) -> "Word2VecRetriever":
        from gensim.models import KeyedVectors

        self = cls.__new__(cls)
        Retriever.__init__(self, recipes)
        self.wv = KeyedVectors.load(str(out / "w2v.kv"))
        self.doc_vecs = np.load(out / "w2v_docs.npy")
        return self

    def footprint_mb(self) -> float:
        return (self.wv.vectors.nbytes + self.doc_vecs.nbytes) / 1e6
