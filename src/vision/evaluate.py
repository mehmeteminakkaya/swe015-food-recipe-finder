"""Top-1 / top-5 accuracy of every vision model on the held-out test split."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ..config import IMAGES
from . import vgg
from .data import eval_transform
from .loaders import Vgg16Int8, load_small_cnn, load_vgg16


@torch.no_grad()
def _accuracy_pytorch(model: torch.nn.Module, loader, device: str = "cuda") -> dict:
    model.eval()
    n = c1 = c5 = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        n += len(y)
        c1 += (logits.argmax(1) == y).sum().item()
        c5 += (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
    return {"n": n, "top1": c1 / n, "top5": c5 / n}


@torch.no_grad()
def _accuracy_onnx(model: Vgg16Int8, loader) -> dict:
    n = c1 = c5 = 0
    for x, y in loader:
        logits = torch.from_numpy(model.logits(x))
        n += len(y)
        c1 += (logits.argmax(1) == y).sum().item()
        c5 += (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
    return {"n": n, "top1": c1 / n, "top5": c5 / n}


def evaluate_vision(device: str = "cuda", include_int8: bool = True) -> dict:
    out: dict = {}

    cnn, ck = load_small_cnn(device)
    ds = ImageFolder(IMAGES / "test", transform=eval_transform(ck["img_size"]))
    out["small_cnn"] = _accuracy_pytorch(cnn, DataLoader(ds, batch_size=64), device)
    out["small_cnn"]["params_m"] = sum(p.numel() for p in cnn.parameters()) / 1e6
    del cnn

    net, ck = load_vgg16(device)
    ds = ImageFolder(IMAGES / "test", transform=eval_transform(ck["img_size"]))
    out["vgg16_ft"] = _accuracy_pytorch(net, DataLoader(ds, batch_size=32), device)
    out["vgg16_ft"]["params_m"] = sum(p.numel() for p in net.parameters()) / 1e6
    del net

    if include_int8:
        try:
            q = Vgg16Int8(device=device)
            ds = ImageFolder(IMAGES / "test", transform=eval_transform(q.img_size))
            out["vgg16_int8"] = _accuracy_onnx(q, DataLoader(ds, batch_size=32))
            out["vgg16_int8"]["params_m"] = None
        except FileNotFoundError as e:
            print("skip int8:", e)

    for k, v in out.items():
        print(f"{k:12s} top1 {v['top1']:.3f}  top5 {v['top5']:.3f}  n={v['n']}")
    return out
