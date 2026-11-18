"""Measure size and latency for every model and build results/results.csv.

Accuracy numbers come from results/metrics.json (scripts/evaluate.py); this
script only measures size (MB) and latency (ms, >=100 runs, batch size 1) on
CPU and GPU and joins the two into the single results table used in the slides.

Usage:  python scripts/benchmark.py [--runs 100]
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODELS, RESULTS, device

DEVICE = device()


def cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        import os

        return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor() or "unknown CPU")


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "-"
    return torch.cuda.get_device_name(0)


def timed(fn, runs: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / runs


def bench_torch(model: torch.nn.Module, x: torch.Tensor, runs: int) -> tuple[float, float]:
    model.eval()
    x = x.to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        gpu_ms = timed(lambda: model(x), runs)
        model_cpu = model.to("cpu")
        x_cpu = x.cpu()
        cpu_ms = timed(lambda: model_cpu(x_cpu), runs)
    return cpu_ms, gpu_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-retrieval", action="store_true")
    args = ap.parse_args()

    metrics_path = RESULTS / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    rows: list[dict] = []
    hw_cpu, hw_gpu = cpu_name(), gpu_name()

    # ------------------------------------------------------------- vision ---
    if not args.skip_vision:
        from PIL import Image
        from torchvision import transforms

        from src.vision.data import eval_transform
        from src.vision.loaders import Vgg16Int8, load_small_cnn, load_vgg16

        sample = sorted(Path("data/raw/food10/test").glob("*/*.jpg"))[0]
        img = Image.open(sample).convert("RGB")
        m = metrics.get("vision", {})

        cnn, ck = load_small_cnn(DEVICE)
        x = eval_transform(ck["img_size"])(img).unsqueeze(0)
        cpu_ms, gpu_ms = bench_torch(cnn, x, args.runs)
        rows.append({
            "task": "vision", "model": "small_cnn", "precision": "fp32",
            "primary_metric": "top-1 accuracy", "primary_value": round(m.get("small_cnn", {}).get("top1", float("nan")), 4),
            "secondary_metric": "top-5 accuracy", "secondary_value": round(m.get("small_cnn", {}).get("top5", float("nan")), 4),
            "size_mb": round((MODELS / "small_cnn.pt").stat().st_size / 1e6, 2),
            "latency_cpu_ms": round(cpu_ms, 2), "latency_gpu_ms": round(gpu_ms, 2),
            "runs": args.runs, "hardware": f"{hw_gpu} / {hw_cpu}",
        })
        del cnn

        net, ck = load_vgg16(DEVICE)
        x = eval_transform(ck["img_size"])(img).unsqueeze(0)
        cpu_ms, gpu_ms = bench_torch(net, x, args.runs)
        rows.append({
            "task": "vision", "model": "vgg16_ft", "precision": "fp32",
            "primary_metric": "top-1 accuracy", "primary_value": round(m.get("vgg16_ft", {}).get("top1", float("nan")), 4),
            "secondary_metric": "top-5 accuracy", "secondary_value": round(m.get("vgg16_ft", {}).get("top5", float("nan")), 4),
            "size_mb": round((MODELS / "vgg16_ft.pt").stat().st_size / 1e6, 2),
            "latency_cpu_ms": round(cpu_ms, 2), "latency_gpu_ms": round(gpu_ms, 2),
            "runs": args.runs, "hardware": f"{hw_gpu} / {hw_cpu}",
        })
        del net

        q = Vgg16Int8(device="cpu")
        x = eval_transform(q.img_size)(img).unsqueeze(0)
        cpu_ms = timed(lambda: q.logits(x), args.runs)
        rows.append({
            "task": "vision", "model": "vgg16_int8", "precision": "int8 (ONNX QDQ)",
            "primary_metric": "top-1 accuracy", "primary_value": round(m.get("vgg16_int8", {}).get("top1", float("nan")), 4),
            "secondary_metric": "top-5 accuracy", "secondary_value": round(m.get("vgg16_int8", {}).get("top5", float("nan")), 4),
            "size_mb": round(q.footprint_mb(), 2),
            "latency_cpu_ms": round(cpu_ms, 2), "latency_gpu_ms": float("nan"),
            "runs": args.runs, "hardware": f"CPU only / {hw_cpu}",
        })

    # ---------------------------------------------------------- retrieval ---
    if not args.skip_retrieval:
        from evaluate import load_retrievers
        from src.data import load_recipes

        recipes = load_recipes()
        q = "vejetaryen pizza, 30 dakikadan kısa"
        m = metrics.get("retrieval", {})
        for name, retriever in load_retrievers(None).items():
            fn = lambda r=retriever: r.search(q, recipes, k=5)
            gpu_ms = cpu_ms = timed(fn, args.runs)
            if name.startswith("bert") and DEVICE == "cuda":
                # fp32 encoder can also run on the GPU; int8 ONNX is CPU-only
                import numpy as np

                if not name.endswith("int8"):
                    retriever.model.to("cuda")
                    gpu_ms = timed(lambda r=retriever: r.scores(q, list(range(300))), args.runs)
                    retriever.model.to("cpu")
            rows.append({
                "task": "retrieval", "model": name,
                "precision": "int8 (ONNX dynamic)" if name.endswith("int8") else "fp32",
                "primary_metric": "Recall@5", "primary_value": round(m.get(name, {}).get("recall@5", float("nan")), 4),
                "secondary_metric": "MRR", "secondary_value": round(m.get(name, {}).get("mrr", float("nan")), 4),
                "size_mb": round(retriever.footprint_mb(), 2),
                "latency_cpu_ms": round(cpu_ms, 2), "latency_gpu_ms": round(gpu_ms, 2) if name.startswith("bert") and not name.endswith("int8") else float("nan"),
                "runs": args.runs, "hardware": f"{hw_gpu} / {hw_cpu}",
            })

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.csv"
    cols = ["task", "model", "precision", "primary_metric", "primary_value",
            "secondary_metric", "secondary_value", "size_mb",
            "latency_cpu_ms", "latency_gpu_ms", "runs", "hardware"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"wrote {out}")
    for r in rows:
        print(f"{r['task']:9s} {r['model']:16s} {r['precision']:20s} "
              f"{r['primary_value']:>7} {r['size_mb']:>7} MB  cpu {r['latency_cpu_ms']:>7} ms")


if __name__ == "__main__":
    main()
