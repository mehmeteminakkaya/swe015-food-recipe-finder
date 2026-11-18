"""Remove exact-duplicate images from data/raw/food10.

Food-101 contains a few repeated photos, some of them across the official
train/test boundary. A repeated test photo would leak into training, so each
duplicate group is collapsed to one file:

    test > val > train      (evaluation splits win)
    then lexicographically first path

Usage:  python scripts/dedupe_images.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGES

RANK = {"test": 0, "val": 1, "train": 2}


def split_of(p: Path) -> str:
    return p.relative_to(IMAGES).parts[0]


def digest(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(IMAGES.rglob("*.jpg")):
        groups[digest(p)].append(p)

    removed = []
    for paths in groups.values():
        if len(paths) < 2:
            continue
        paths.sort(key=lambda p: (RANK[split_of(p)], str(p)))
        keep, drop = paths[0], paths[1:]
        print(f"keep {keep.relative_to(IMAGES)}")
        for p in drop:
            print(f"  - {p.relative_to(IMAGES)}")
            removed.append(p)
            if not args.dry_run:
                p.unlink()

    print(f"\n{len(removed)} duplicate files {'would be ' if args.dry_run else ''}removed")
    for s in ("train", "val", "test"):
        print(f"{s}: {sum(1 for _ in (IMAGES / s).rglob('*.jpg'))} images")


if __name__ == "__main__":
    main()
