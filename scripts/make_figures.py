"""Figures for the slides: retrieval scores, vision trade-off, training curves.

Usage:  python scripts/make_figures.py
Writes docs/fig_retrieval.png, docs/fig_tradeoff.png, docs/fig_curves.png.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, ROOT

DOCS = ROOT / "docs"
C_TR = "#1f77b4"
C_EN = "#ff7f0e"
C_GREY = "#5f6368"


def retrieval() -> None:
    m = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))["retrieval"]
    order = ["tfidf", "word2vec", "bert_e5", "bert_e5_int8"]
    labels = ["tf-idf", "word2vec\n(ours)", "E5 small\nfp32", "E5 small\nINT8"]
    tr = [m[k]["by_lang"]["tr"]["recall@5"] for k in order]
    en = [m[k]["by_lang"]["en"]["recall@5"] for k in order]
    avg = [m[k]["recall@5"] for k in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.26
    ax.bar(x - w, tr, w, label="Turkish queries", color=C_TR)
    ax.bar(x, en, w, label="English queries", color=C_EN)
    ax.bar(x + w, avg, w, label="all 40 queries", color=C_GREY)
    for i, v in enumerate(avg):
        ax.text(i + w, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color="#202124")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall@5")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_retrieval.png", dpi=200, bbox_inches="tight")
    print("wrote", DOCS / "fig_retrieval.png")


def tradeoff() -> None:
    rows = list(csv.DictReader((RESULTS / "results.csv").open(encoding="utf-8")))
    v = {r["model"] + "_" + r["precision"].split()[0]: r for r in rows if r["task"] == "vision"}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    names = ["small_cnn", "vgg16_ft", "vgg16_int8"]
    pretty = ["small CNN\n(fp32)", "VGG16\n(fp32)", "VGG16\n(INT8)"]
    sizes = [float(next(r for r in rows if r["model"] == n)["size_mb"]) for n in names]
    lat = [float(next(r for r in rows if r["model"] == n)["latency_cpu_ms"]) for n in names]
    acc = [float(next(r for r in rows if r["model"] == n)["primary_value"]) for n in names]
    colors = ["#9aa0a6", "#3b6ea5", "#2e8b57"]

    axes[0].bar(pretty, sizes, color=colors)
    for i, s in enumerate(sizes):
        axes[0].text(i, s + 2, f"{s:.1f} MB", ha="center", fontsize=9)
    axes[0].set_title("Model size", fontsize=10)
    axes[0].set_ylim(0, max(sizes) * 1.25)

    axes[1].bar(pretty, lat, color=colors)
    for i, (l, a) in enumerate(zip(lat, acc)):
        axes[1].text(i, l + 2, f"{l:.1f} ms\ntop-1 {a:.3f}", ha="center", fontsize=8.5)
    axes[1].set_title("CPU latency per image", fontsize=10)
    axes[1].set_ylim(0, max(lat) * 1.35)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.suptitle("Quantization trade-off: VGG16 gets 4x smaller and 6x faster on CPU, "
                 "top-1 drops 0.926 → 0.924", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_tradeoff.png", dpi=200)
    print("wrote", DOCS / "fig_tradeoff.png")


def curves() -> None:
    cnn = json.loads((RESULTS / "small_cnn_history.json").read_text(encoding="utf-8"))["history"]
    vgg = json.loads((RESULTS / "vgg16_history.json").read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot([e["epoch"] for e in cnn], [e["val_top1"] for e in cnn], marker="o", ms=3,
            label="small CNN, val top-1", color=C_TR)
    ft = vgg["fine_tune"]
    ax.plot([len(cnn) + e["epoch"] for e in ft], [e["top1"] for e in ft], marker="s", ms=4,
            label="VGG16 after conv5 fine-tuning, val top-1", color="#2e8b57")
    ax.axhline(vgg["feature_extraction"]["val"]["top1"], color=C_GREY, linestyle="--", linewidth=1)
    ax.text(1, vgg["feature_extraction"]["val"]["top1"] + 0.012,
            "VGG16 feature extraction (frozen): 0.860", fontsize=8.5, color=C_GREY)
    ax.set_xlabel("epoch (VGG16 fine-tuning is drawn after the 18 CNN epochs)")
    ax.set_ylabel("validation top-1")
    ax.set_ylim(0.25, 0.95)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Training curves", fontsize=11)
    fig.tight_layout()
    fig.savefig(DOCS / "fig_curves.png", dpi=200)
    print("wrote", DOCS / "fig_curves.png")


if __name__ == "__main__":
    retrieval()
    tradeoff()
    curves()
