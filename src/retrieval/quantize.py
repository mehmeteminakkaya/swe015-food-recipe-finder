"""INT8 retrieval: the multilingual encoder exported to ONNX and dynamically
quantized with ONNX Runtime (technique area D).

Dynamic quantization stores MatMul weights as int8 and quantizes activations on
the fly, so no calibration set is needed. We measure the size, the CPU latency
and the retrieval quality before/after.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import HF_CACHE
from ..data import Recipe, recipe_text
from .base import Retriever, l2norm
from .bert import MODEL_NAME, PASSAGE_PREFIX, QUERY_PREFIX
from .text import strip_constraints

MAX_LEN = 256
ONNX_NAME = "encoder_int8.onnx"


def _export_onnx(model_name: str, out: Path) -> tuple[object, Path]:
    """Export the encoder to ONNX and return (tokenizer, onnx path)."""
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device="cpu")
    encoder = st[0].auto_model.eval()
    tokenizer = st[0].tokenizer
    dummy = tokenizer(["hello world"], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
    onnx_path = out / "encoder.onnx"
    torch.onnx.export(
        encoder,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    return tokenizer, onnx_path


def _quantize(onnx_path: Path, out_path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    return out_path


class QuantizedBertRetriever(Retriever):
    name = "bert_e5_int8"
    model_name = MODEL_NAME

    def __init__(self, recipes: list[Recipe], session: ort.InferenceSession, tokenizer,
                 doc_vecs: np.ndarray, onnx_path: Path):
        super().__init__(recipes)
        self.session = session
        self.tokenizer = tokenizer
        self.doc_vecs = doc_vecs
        self.onnx_path = onnx_path

    @classmethod
    def build(cls, recipes: list[Recipe], out: Path) -> "QuantizedBertRetriever":
        out.mkdir(parents=True, exist_ok=True)
        tokenizer, fp32_path = _export_onnx(cls.model_name, out)
        int8_path = _quantize(fp32_path, out / ONNX_NAME)
        fp32_path.unlink()
        session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        (out / "bert_q.json").write_text(
            json.dumps({"model": cls.model_name, "onnx": ONNX_NAME}), encoding="utf-8")
        doc_vecs = cls._encode_session(
            session, tokenizer, [PASSAGE_PREFIX + recipe_text(r) for r in recipes])
        return cls(recipes, session, tokenizer, doc_vecs, int8_path)

    @staticmethod
    def _encode_session(session: ort.InferenceSession, tokenizer, texts: list[str]) -> np.ndarray:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        last = session.run(None, feeds)[0]
        mask = feeds["attention_mask"][..., None]
        pooled = (last * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        return l2norm(pooled)

    def scores(self, text: str, rows: list[int]) -> np.ndarray:
        q = self._encode_session(
            self.session, self.tokenizer, [QUERY_PREFIX + strip_constraints(text)])[0]
        return self.doc_vecs[rows] @ q

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        if not (out / ONNX_NAME).exists():
            shutil.copy(self.onnx_path, out / ONNX_NAME)
        np.save(out / "bert_q_docs.npy", self.doc_vecs)
        (out / "bert_q.json").write_text(
            json.dumps({"model": self.model_name, "onnx": ONNX_NAME}), encoding="utf-8")

    @classmethod
    def load(cls, out: Path, recipes: list[Recipe]) -> "QuantizedBertRetriever":
        from transformers import AutoTokenizer

        meta = json.loads((out / "bert_q.json").read_text(encoding="utf-8"))
        onnx_path = out / meta["onnx"]
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(meta["model"], cache_dir=str(HF_CACHE / "hub"))
        doc_vecs = np.load(out / "bert_q_docs.npy")
        return cls(recipes, session, tokenizer, doc_vecs, onnx_path)

    def footprint_mb(self) -> float:
        return self.onnx_path.stat().st_size / 1e6 + self.doc_vecs.nbytes / 1e6
