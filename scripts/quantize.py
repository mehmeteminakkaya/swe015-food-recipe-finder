"""Quantize VGG16 to INT8 with ONNX Runtime static QDQ quantization.

Dynamic quantization only touches MatMul/Gemm, which for a convolutional net is
the small part. Static QDQ (per-channel weights, 256 calibration images) is the
right tool for the convnet, so that is what we use here; the Transformer encoder
uses dynamic quantization in src/retrieval/quantize.py.

Usage:  python scripts/quantize.py [--calib-batches 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS
from src.vision import vgg
from src.vision.loaders import load_vgg16


def export_onnx(model: torch.nn.Module, img_size: int, out: Path) -> Path:
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, (dummy,), str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    return out


class Reader:
    def __init__(self, batches: int, batch_size: int = 32):
        from torchvision.datasets import ImageFolder

        from src.config import IMAGES
        from src.vision.data import eval_transform

        ds = ImageFolder(IMAGES / "train", transform=eval_transform(224))
        self.loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        self.it = iter(self.loader)
        self.batches = batches
        self.seen = 0

    def get_next(self):
        if self.seen >= self.batches:
            return None
        try:
            x, _ = next(self.it)
        except StopIteration:
            return None
        self.seen += 1
        return {"input": x.numpy().astype(np.float32)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-batches", type=int, default=8)  # 8 x 32 = 256 images
    args = ap.parse_args()

    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    model, ck = load_vgg16("cpu")
    fp32_path = export_onnx(model, ck["img_size"], MODELS / "vgg16_ft.onnx")
    print(f"fp32 ONNX: {fp32_path.stat().st_size / 1e6:.1f} MB")

    int8_path = MODELS / "vgg16_ft_int8.onnx"
    quantize_static(
        str(fp32_path), str(int8_path), Reader(args.calib_batches),
        quant_format=QuantFormat.QDQ, per_channel=True,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
    )
    print(f"INT8 ONNX: {int8_path.stat().st_size / 1e6:.1f} MB "
          f"({int8_path.stat().st_size / fp32_path.stat().st_size:.1%} of fp32)")

    (MODELS / "vgg16_ft_int8.json").write_text(
        json.dumps({"classes": ck["classes"], "img_size": ck["img_size"]}), encoding="utf-8")


if __name__ == "__main__":
    main()
