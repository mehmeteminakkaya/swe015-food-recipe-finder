"""Draw the architecture diagram (R6) straight from the model choices in the code.

Usage:  python scripts/architecture.py
Writes docs/architecture.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT

COLORS = {
    "ui": "#e8f0fe",
    "pre": "#fff4e5",
    "vision": "#e6f4ea",
    "text": "#fce8e6",
    "out": "#ede7f6",
    "quant": "#d1e7dd",
}
EDGE = "#5f6368"


def box(ax, x, y, w, h, title, body, color, title_size=11, body_size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.015",
                                linewidth=1.2, edgecolor=EDGE, facecolor=COLORS[color]))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top",
            fontsize=body_size, linespacing=1.5, color="#202124")


def arrow(ax, start, end, text=None, dashed=False, tx=0.0, ty=0.015, ha="center"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color=EDGE,
        linestyle=(0, (4, 3)) if dashed else "solid", shrinkA=2, shrinkB=2))
    if text:
        ax.text((start[0] + end[0]) / 2 + tx, (start[1] + end[1]) / 2 + ty, text,
                ha=ha, va="bottom", fontsize=7.8, color="#3c4043")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- top: user + UI ----------------------------------------------------
    box(ax, 0.02, 0.80, 0.13, 0.14, "Kullanıcı", "fotoğraf yükler\nveya metin yazar", "ui")
    box(ax, 0.18, 0.80, 0.22, 0.14, "Gradio arayüzü (R1)",
        "tarayıcı · 127.0.0.1:7860\nmodel seçici (INT8 varsayılan)", "ui")

    # --- vision lane -------------------------------------------------------
    box(ax, 0.26, 0.55, 0.16, 0.15, "Ön işleme",
        "resize 256 → crop 224\nImageNet normalisation", "pre")
    box(ax, 0.45, 0.55, 0.16, 0.15, "VGG16 INT8 · ONNX",
        "conv1–conv5 + head (10 sınıf)\nONNX Runtime · CPU · 110 MB", "vision")
    box(ax, 0.64, 0.55, 0.16, 0.15, "Grad-CAM",
        "conv5_3 gradyanları\nfp32 VGG16 · GPU", "vision")
    box(ax, 0.81, 0.55, 0.17, 0.15, "Tahmin",
        "yemek adı + güven\nısı haritası", "out")

    # --- retrieval lane ----------------------------------------------------
    box(ax, 0.26, 0.22, 0.16, 0.15, "Ön işleme",
        "normalize (TR/EN)\nregex süre + diyet\nSnowball + TR→EN sözlük", "pre")
    box(ax, 0.45, 0.22, 0.16, 0.15, "Kısıt filtresi",
        "300 tarif → aday küme\nsüre ≤ X · vejetaryen/vegan\nbütün modellerde aynı", "pre")
    box(ax, 0.64, 0.22, 0.16, 0.15, "E5 small INT8",
        "ONNX dynamic INT8\nquery:/passage: gömme\ncosine · CPU · 118 MB", "quant")
    box(ax, 0.81, 0.22, 0.17, 0.15, "Sonuç",
        "en iyi 5 tarif + skor\nsüre ve diyet etiketleri", "out")

    # --- comparison row ----------------------------------------------------
    box(ax, 0.45, 0.02, 0.53, 0.12,
        "Diğer modeller — uygulamada seçilebilir, aynı aday kümede sıralanır",
        "tf-idf + cosine · word2vec (kendi eğitimimiz) · E5 fp32 · VGG16 fp32 · küçük CNN (sıfırdan)",
        "quant", title_size=10, body_size=8.4)

    # --- arrows ------------------------------------------------------------
    arrow(ax, (0.15, 0.87), (0.18, 0.87))
    arrow(ax, (0.30, 0.80), (0.34, 0.70), "fotoğraf", tx=-0.02)
    ax.plot([0.22, 0.22], [0.80, 0.295], color=EDGE, linewidth=1.2)
    arrow(ax, (0.22, 0.295), (0.26, 0.295), "metin", ty=0.015, ha="left", tx=-0.03)

    arrow(ax, (0.42, 0.625), (0.45, 0.625))
    arrow(ax, (0.61, 0.625), (0.64, 0.625))
    arrow(ax, (0.80, 0.625), (0.81, 0.625))

    arrow(ax, (0.42, 0.295), (0.45, 0.295))
    arrow(ax, (0.61, 0.295), (0.64, 0.295))
    arrow(ax, (0.80, 0.295), (0.81, 0.295))

    arrow(ax, (0.855, 0.55), (0.755, 0.37), "yemek adı sorguya eklenir", dashed=True, tx=0.03)

    arrow(ax, (0.72, 0.14), (0.72, 0.22), dashed=True)
    ax.text(0.735, 0.178, "yerine geçer", fontsize=7.8, color="#3c4043", va="center")

    ax.text(0.02, 0.985, "Food Photo → Recipe Finder — veri akışı ve teknik haritası",
            fontsize=14, fontweight="bold")
    ax.text(0.02, 0.955, "Her model aynı 10 sınıflık test setinde / aynı 40 sorguda değerlendirilir; "
                         "her sayı results/results.csv dosyasından gelir.",
            fontsize=9, color="#3c4043")

    out = ROOT / "docs" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
