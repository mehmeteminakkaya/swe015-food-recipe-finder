"""Load the trained vision models (PyTorch checkpoints and the INT8 ONNX model)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..config import MODELS
from . import small_cnn, vgg


def _checkpoint(name: str) -> dict:
    return torch.load(MODELS / name, map_location="cpu", weights_only=False)


def load_small_cnn(device: str = "cpu") -> tuple[nn.Module, dict]:
    ck = _checkpoint("small_cnn.pt")
    model = small_cnn.build(len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def load_vgg16(device: str = "cpu") -> tuple[nn.Module, dict]:
    ck = _checkpoint("vgg16_ft.pt")
    model = vgg.build(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


class Vgg16Int8:
    """ONNX Runtime INT8 VGG16 (static QDQ quantization)."""

    def __init__(self, path: Path | None = None, device: str = "cpu"):
        import onnxruntime as ort

        self.path = path or (MODELS / "vgg16_ft_int8.onnx")
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} -- run scripts/quantize.py first")
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(self.path), providers=providers)
        except Exception:
            self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        meta = json.loads((MODELS / "vgg16_ft_int8.json").read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        self.img_size = meta["img_size"]
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, batch: torch.Tensor) -> np.ndarray:
        x = batch.numpy().astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0]

    def footprint_mb(self) -> float:
        return self.path.stat().st_size / 1e6
