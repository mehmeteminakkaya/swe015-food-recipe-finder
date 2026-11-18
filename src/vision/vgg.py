"""VGG16 transfer learning (technique area A).

Two stages, as in the plan:
  1. feature extraction -- freeze the ImageNet features, train a small head;
  2. fine-tuning       -- unfreeze the last convolutional block (conv5) and
                          continue training at a low learning rate.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

LAST_BLOCK_START = 24  # features[24:30] == conv5_1 .. conv5_3
GRADCAM_LAYER = 28     # conv5_3, the last convolutional layer


def build(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def unfreeze_last_block(model: nn.Module) -> None:
    for p in model.features[LAST_BLOCK_START:].parameters():
        p.requires_grad = True


def target_layer(model: nn.Module) -> nn.Module:
    return model.features[GRADCAM_LAYER]


def param_groups(model: nn.Module, lr_head: float, lr_features: float) -> list[dict]:
    head = list(model.classifier.parameters())
    feats = [p for p in model.features.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": lr_head}]
    if feats:
        groups.append({"params": feats, "lr": lr_features})
    return groups


@torch.no_grad()
def extract_features(model: nn.Module, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen convolutional stack once and keep the pooled features."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        f = model.features(x.to(device, non_blocking=True)).flatten(1)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)
