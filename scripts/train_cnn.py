"""Train the small CNN from scratch.

Usage:  python scripts/train_cnn.py [--epochs 20] [--img-size 128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS, RESULTS, SEED, device
from src.vision import small_cnn
from src.vision.data import classes, loaders

DEVICE = device()


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> tuple[int, int]:
    top1 = (logits.argmax(1) == y).sum().item()
    top5 = (logits.topk(5, dim=1).indices == y[:, None]).any(1).sum().item()
    return top1, top5


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> dict:
    model.eval()
    n = c1 = c5 = 0
    loss = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss += crit(logits, y).item() * len(y)
        a1, a5 = accuracy(logits, y)
        n += len(y)
        c1 += a1
        c5 += a5
    model.train()
    return {"n": n, "top1": c1 / n, "top5": c5 / n, "loss": loss / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    loaders_ = loaders(args.img_size, args.batch_size)
    cls = classes()
    model = small_cnn.build(len(cls)).to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    history, best, since_best = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        running, seen = 0.0, 0
        for x, y in tqdm(loaders_["train"], desc=f"epoch {epoch}", leave=False):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        sched.step()
        val = evaluate(model, loaders_["val"])
        rec = {"epoch": epoch, "train_loss": running / seen, **{f"val_{k}": v for k, v in val.items()},
               "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t, 1)}
        history.append(rec)
        print(f"epoch {epoch:2d}  train {rec['train_loss']:.3f}  val top1 {val['top1']:.3f}  "
              f"top5 {val['top5']:.3f}  ({rec['seconds']}s)")
        if val["top1"] > best:
            best = val["top1"]
            since_best = 0
            MODELS.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": cls, "img_size": args.img_size}, MODELS / "small_cnn.pt")
            print(f"  saved (val top1 {best:.3f})")
        else:
            since_best += 1
            if since_best >= args.patience:
                print("early stop")
                break

    test = evaluate(model, loaders_["test"])
    print("test", {k: round(v, 4) if isinstance(v, float) else v for k, v in test.items()})
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "small_cnn_history.json").write_text(json.dumps({"history": history, "test": test}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
