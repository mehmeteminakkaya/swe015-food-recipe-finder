"""Draw the architecture diagram (R6) from the components app.py actually uses.

Sizes and latencies are read from results/results.csv, so the diagram never
disagrees with the results table.

Usage:  python scripts/architecture.py
Writes docs/architecture.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, ROOT

INK = "#17301F"
MUTED = "#5B6A60"
EDGE = "#8A9A8E"
FILL = {
    "ui": "#E6ECF5",
    "pre": "#FBF0D5",
    "model": "#E1EEE3",
    "out": "#EFEAF6",
    "alt": "#FFFFFF",
}

# column layout: col 0 is the app, cols 1-5 the pipeline steps
X0, W0 = 0.015, 0.135
COLS = [0.185 + i * 0.163 for i in range(5)]
W = 0.149
IMG_Y, TXT_Y, H = 0.47, 0.15, 0.20


def results() -> dict[str, dict]:
    with (RESULTS / "results.csv").open(encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def box(ax, x, y, w, h, title, body, kind, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", linewidth=1.3,
        edgecolor=EDGE, facecolor=FILL[kind], linestyle=(0, (4, 3)) if dashed else "solid"))
    ax.text(x + 0.012, y + h - 0.03, title, ha="left", va="top", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(x + 0.012, y + h - 0.075, body, ha="left", va="top", fontsize=8.9,
            linespacing=1.45, color=INK)


def arrow(ax, a, b, dashed=False, label=None, lx=0.0, ly=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color=MUTED,
        linestyle=(0, (4, 3)) if dashed else "solid", shrinkA=0, shrinkB=0))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, ha=ha, va="center",
                fontsize=9, color=MUTED, style="italic",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


def main() -> None:
    r = results()
    vgg8, e58 = r["vgg16_int8"], r["bert_e5_int8"]

    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(X0, 0.985, "Photo to recipe: data flow", fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(X0, 0.935, "Solid boxes run by default in the app. Both INT8 models run on the CPU with ONNX Runtime. "
            "Sizes and latencies come from results/results.csv.", fontsize=10, color=MUTED, va="top")

    # app column ------------------------------------------------------------
    box(ax, X0, 0.73, W0, 0.14, "User", "uploads a photo,\ntypes a request", "ui")
    box(ax, X0, TXT_Y, W0, IMG_Y + H - TXT_Y, "Gradio app",
        "runs in the browser\n127.0.0.1:7860\n\ninputs\n· dish photo\n· request text\n"
        "· image model picker\n· search model picker", "ui")
    arrow(ax, (X0 + W0 / 2, 0.73), (X0 + W0 / 2, IMG_Y + H))

    # lane labels -------------------------------------------------------------
    ax.text(COLS[0], IMG_Y + H + 0.02, "Image path", fontsize=10.5, color=MUTED, fontweight="bold")
    ax.text(COLS[0], TXT_Y + H + 0.02, "Text path", fontsize=10.5, color=MUTED, fontweight="bold")

    # image lane ------------------------------------------------------------
    img = [
        ("Preprocess", "resize 256, crop 224\nImageNet normalisation", "pre"),
        ("VGG16 INT8", f"fine-tuned on 10 dishes\nONNX static QDQ INT8\n"
                       f"{float(vgg8['size_mb']):.1f} MB, {float(vgg8['latency_cpu_ms']):.1f} ms CPU", "model"),
        ("Prediction", "softmax over 10 dishes\ntop dish + top-5\nconfidences", "out"),
        ("Grad-CAM", "fp32 VGG16, conv5_3\nheat map for the\npredicted dish", "model"),
        ("Shown in app", "dish name\nconfidence bars\nheat map", "out"),
    ]
    txt = [
        ("Preprocess", "normalise EN / TR text\nregex: time and diet\nSnowball stemming", "pre"),
        ("Candidate filter", "300 recipes\nkeep: dish, time, diet\nsame pool for all\nsearch models", "pre"),
        ("E5 small INT8", f"multilingual encoder\nONNX dynamic INT8\n"
                          f"{float(e58['size_mb']):.1f} MB, {float(e58['latency_cpu_ms']):.1f} ms CPU", "model"),
        ("Cosine ranking", "request vs. recipe\nembeddings\nkeep the top 5", "model"),
        ("Shown in app", "five recipes with\ntime, diet tags\nand similarity", "out"),
    ]
    for lane_y, steps in ((IMG_Y, img), (TXT_Y, txt)):
        for x, (title, body, kind) in zip(COLS, steps):
            box(ax, x, lane_y, W, H, title, body, kind)
        mid = lane_y + H / 2
        arrow(ax, (X0 + W0, mid), (COLS[0], mid), label="photo" if lane_y == IMG_Y else "text", ly=0.025)
        for a, b in zip(COLS, COLS[1:]):
            arrow(ax, (a + W, mid), (b, mid))

    # predicted dish feeds the text path
    arrow(ax, (COLS[2] + W * 0.35, IMG_Y), (COLS[1] + W * 0.65, TXT_Y + H), dashed=True,
          label="predicted dish", lx=0.0, ly=0.0)

    # swappable models --------------------------------------------------------
    alt_w = W * 2 + (COLS[1] - COLS[0] - W)
    box(ax, COLS[1], 0.78, alt_w, 0.10, "Also selectable for photos",
        "VGG16 fp32 · small CNN trained from scratch", "alt", dashed=True)
    arrow(ax, (COLS[1] + W / 2, 0.78), (COLS[1] + W / 2, IMG_Y + H), dashed=True)
    box(ax, COLS[2], 0.0, alt_w, 0.10, "Also selectable for search",
        "tf-idf + cosine · word2vec trained here · E5 fp32", "alt", dashed=True)
    arrow(ax, (COLS[2] + W / 2, 0.10), (COLS[2] + W / 2, TXT_Y), dashed=True)

    out = ROOT / "docs" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
