"""Make a Grad-CAM figure for the slides: 4 test photos, prediction + heat map.

Usage:  python scripts/gradcam_demo.py [--n 4]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES, ROOT, device
from src.vision import vgg
from src.vision.data import eval_transform
from src.vision.loaders import load_vgg16


def main() -> None:
    DEVICE = device()
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs" / "gradcam_example.png"))
    args = ap.parse_args()

    model, ck = load_vgg16(DEVICE)
    files = sorted((IMAGES / "test").rglob("*.jpg"))
    random.Random(0).shuffle(files)
    picked = files[: args.n]

    fig, axes = plt.subplots(2, args.n, figsize=(3.2 * args.n, 6.4))
    for i, path in enumerate(picked):
        img = Image.open(path).convert("RGB")
        tensor = eval_transform(ck["img_size"])(img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)
        with torch.inference_mode():
            prob = torch.softmax(model(tensor), 1)[0]
        cls = int(prob.argmax())
        with GradCAM(model=model, target_layers=[vgg.target_layer(model)]) as cam:
            heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(cls)])[0]
        rgb = np.asarray(img.resize((ck["img_size"], ck["img_size"])), dtype=np.float32) / 255
        overlay = show_cam_on_image(rgb, heat, use_rgb=True)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"true: {path.parent.name.replace('_', ' ')}", fontsize=10)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"pred: {ck['classes'][cls].replace('_', ' ')} {float(prob[cls]):.0%}", fontsize=10)
        for ax in (axes[0, i], axes[1, i]):
            ax.axis("off")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print("wrote", out)


if __name__ == "__main__":
    main()
