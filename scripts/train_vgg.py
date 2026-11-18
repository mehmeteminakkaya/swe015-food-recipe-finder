"""Train VGG16 in two stages: frozen feature extraction, then conv5 fine-tuning.

Usage:  python scripts/train_vgg.py [--head-epochs 15] [--ft-epochs 6]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, RESULTS, SEED, device
from src.vision import vgg
from src.vision.data import classes, loaders

DEVICE = device()


def top1_top5(logits: torch.Tensor, y: torch.Tensor) -> tuple[int, int]:
    top1 = (logits.argmax(1) == y).sum().item()
    top5 = (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
    return top1, top5


@torch.no_grad()
def evaluate_head(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor) -> dict:
    head.eval()
    n = c1 = c5 = 0
    for i in range(0, len(feats), 512):
        logits = head(feats[i:i + 512].to(DEVICE))
        y = labels[i:i + 512].to(DEVICE)
        a1, a5 = top1_top5(logits, y)
        n += len(y)
        c1 += a1
        c5 += a5
    head.train()
    return {"n": n, "top1": c1 / n, "top5": c5 / n}


@torch.no_grad()
def evaluate_model(model: nn.Module, loader) -> dict:
    model.eval()
    n = c1 = c5 = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        a1, a5 = top1_top5(logits, y)
        n += len(y)
        c1 += a1
        c5 += a5
    model.train()
    return {"n": n, "top1": c1 / n, "top5": c5 / n}


def train_head(head: nn.Module, tr, va, tr_lab, va_lab, epochs: int, lr: float) -> list[dict]:
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(tr))
        running, seen = 0.0, 0
        for i in range(0, len(tr), 256):
            idx = perm[i:i + 256]
            x, y = tr[idx].to(DEVICE), tr_lab[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(head(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()
        val = evaluate_head(head, va, va_lab)
        history.append({"epoch": epoch, "train_loss": running / seen, "val_top1": val["top1"], "val_top5": val["top5"]})
        print(f"  head epoch {epoch:2d}  loss {running / seen:.3f}  val top1 {val['top1']:.3f}")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--ft-epochs", type=int, default=6)
    ap.add_argument("--ft-lr-features", type=float, default=1e-5)
    ap.add_argument("--ft-lr-head", type=float, default=1e-4)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    cls = classes()
    data = loaders(args.img_size, args.batch_size)

    model = vgg.build(len(cls)).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M "
          f"(trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M)")

    # ---- stage 1: feature extraction -------------------------------------
    t = time.time()
    tr_feats, tr_lab = vgg.extract_features(model, data["train"], DEVICE)
    va_feats, va_lab = vgg.extract_features(model, data["val"], DEVICE)
    te_feats, te_lab = vgg.extract_features(model, data["test"], DEVICE)
    print(f"features extracted in {time.time() - t:.0f}s  shape {tuple(tr_feats.shape)}")

    head = model.classifier
    history_head = train_head(head, tr_feats, va_feats, tr_lab, va_lab, args.head_epochs, 1e-3)
    feat_metrics = {
        "val": evaluate_head(head, va_feats, va_lab),
        "test": evaluate_head(head, te_feats, te_lab),
    }
    print(f"feature-extraction test: top1 {feat_metrics['test']['top1']:.3f} "
          f"top5 {feat_metrics['test']['top5']:.3f}")
    torch.save({"state_dict": model.state_dict(), "classes": cls, "img_size": args.img_size,
                "stage": "feature_extraction"}, MODELS / "vgg16_feat.pt")
    del tr_feats, va_feats, te_feats

    # ---- stage 2: fine-tune the last convolutional block ------------------
    vgg.unfreeze_last_block(model)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(vgg.param_groups(model, args.ft_lr_head, args.ft_lr_features), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    best, best_state, history_ft = 0.0, None, []
    for epoch in range(1, args.ft_epochs + 1):
        t = time.time()
        model.train()
        running, seen = 0.0, 0
        for x, y in tqdm(data["train"], desc=f"ft epoch {epoch}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()
        val = evaluate_model(model, data["val"])
        history_ft.append({"epoch": epoch, "train_loss": running / seen, **val, "seconds": round(time.time() - t, 1)})
        print(f"ft epoch {epoch}  loss {running / seen:.3f}  val top1 {val['top1']:.3f}  ({time.time() - t:.0f}s)")
        if val["top1"] > best:
            best = val["top1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save({"state_dict": best_state, "classes": cls, "img_size": args.img_size,
                        "stage": "fine_tuned"}, MODELS / "vgg16_ft.pt")
            print("  saved")

    model.load_state_dict(best_state)
    test = evaluate_model(model, data["test"])
    print(f"fine-tuned test: top1 {test['top1']:.3f} top5 {test['top5']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "vgg16_history.json").write_text(
        json.dumps({"heads": history_head, "fine_tune": history_ft,
                    "feature_extraction": feat_metrics, "fine_tuned": test}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
