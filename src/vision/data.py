"""Image transforms and DataLoaders for the 10-class Food-101 subset."""
from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from ..config import IMAGES, SEED

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform(img_size: int) -> transforms.Compose:
    resize = int(round(img_size * 256 / 224))
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def loaders(img_size: int, batch_size: int = 64, num_workers: int = 0,
            train=True, val=True, test=True) -> dict[str, DataLoader]:
    out: dict[str, DataLoader] = {}
    if train:
        ds = ImageFolder(IMAGES / "train", transform=train_transform(img_size))
        out["train"] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
    if val:
        ds = ImageFolder(IMAGES / "val", transform=eval_transform(img_size))
        out["val"] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)
    if test:
        ds = ImageFolder(IMAGES / "test", transform=eval_transform(img_size))
        out["test"] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)
    return out


def classes() -> list[str]:
    return ImageFolder(IMAGES / "train").classes


def generator() -> "torch.Generator":
    import torch

    g = torch.Generator()
    g.manual_seed(SEED)
    return g
