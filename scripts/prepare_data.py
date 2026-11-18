"""Download and prepare all data.

1. Food-101 (ethz/food101 on Hugging Face) -> 10-class subset saved as JPEGs:
   data/raw/food10/{train,val,test}/{class}/*.jpg
   - official Food-101 train (750/class) is split 90/10 into train/val
   - official Food-101 test  (250/class) is our held-out test set
2. Food.com recipes (AkashPS11/recipes_data_food.com, MIT) -> data/recipes.jsonl
   with up to N recipes per class, total time in minutes and diet labels.

Usage:  python scripts/prepare_data.py [--skip-images] [--per-class 30]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tqdm import tqdm

from src.config import CLASSES, IMAGES, RECIPES, SEED

# ---------------------------------------------------------------- images ---

def prepare_images() -> None:
    from datasets import load_dataset

    ds = load_dataset("ethz/food101")
    names = ds["train"].features["label"].names
    keep = {names.index(c): c for c in CLASSES}
    rng = random.Random(SEED)

    for hf_split in ("train", "validation"):
        split = ds[hf_split].filter(lambda ex: ex["label"] in keep, num_proc=1)
        rows = list(range(len(split)))
        if hf_split == "train":
            by_cls: dict[int, list[int]] = {}
            for i, lab in enumerate(split["label"]):
                by_cls.setdefault(lab, []).append(i)
            assign = {}
            for lab, idx in by_cls.items():
                rng.shuffle(idx)
                n_val = len(idx) // 10
                for j, i in enumerate(idx):
                    assign[i] = "val" if j < n_val else "train"
        else:
            assign = {i: "test" for i in rows}

        for i in tqdm(rows, desc=f"save {hf_split}"):
            ex = split[i]
            out = IMAGES / assign[i] / keep[ex["label"]] / f"{hf_split}_{i:05d}.jpg"
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            ex["image"].convert("RGB").save(out, quality=95)

    for s in ("train", "val", "test"):
        n = sum(1 for _ in (IMAGES / s).rglob("*.jpg"))
        print(f"{s}: {n} images")


# --------------------------------------------------------------- recipes ---

# Name patterns that decide which Food-101 class a recipe belongs to.
CLASS_PATTERNS = {
    "caesar_salad": r"\bcaesar\b.*\bsalad\b",
    "chocolate_cake": r"\bchocolate\b.*\bcake\b",
    "falafel": r"\bfalafel",
    "guacamole": r"\bguacamole\b",
    "hamburger": r"\b(ham)?burgers?\b",
    "pancakes": r"\bpancakes?\b",
    "pizza": r"\bpizzas?\b",
    "ramen": r"\bramen\b",
    "spaghetti_bolognese": r"\bbolognese\b",
    "sushi": r"\b(sushi|maki|nigiri|california rolls?)\b",
}
EXCLUDE = {
    "chocolate_cake": r"cupcake|cheesecake|pancake|frosting only|icing",
    "pizza": r"\bdough\b|\bsauce\b|\bcrust\b$",
    "hamburger": r"\bbun\b|\bseasoning\b|helper",
    "pancakes": r"\bsyrup\b|\bmix\b$",
    "caesar_salad": r"dressing$",
    "falafel": r"sauce|dressing|\bdip\b|seasoning|mix$",
    "guacamole": r"seasoning|\bmix$",
    "spaghetti_bolognese": r"^bolognese sauce$",
}

MEAT_FISH = re.compile(
    r"\b(beef|pork|chicken|turkey|bacon|ham|sausage|pepperoni|salami|prosciutto|"
    r"lamb|veal|duck|anchov(y|ies)|tuna|salmon|shrimp|prawns?|crab|fish|"
    r"gelatin|lard|chorizo|meat|steak|mince|ground round|crabmeat|eel|"
    r"worcestershire|fish sauce|oyster sauce|dashi|bonito|pancetta)\b")
ANIMAL = re.compile(
    r"\b(eggs?|egg yolks?|egg whites?|milk|butter|cream|cheese|yogurt|honey|"
    r"buttermilk|mayonnaise|parmesan|mozzarella|cheddar|ghee|whey)\b")


def _json_list(s) -> list:
    if not isinstance(s, str):
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


_RECIPE_ID = re.compile(r"/recipes/(\d+)/")


def prepare_recipes(per_class: int) -> None:
    """Source: Food.com Recipes (Kaggle, irkaal, CC0), parsed copy on HF:
    Karo8870/food.com-parsed-dataset. Its `steps` column is broken (holds image
    URLs), so recipes are represented by name, description, category, tags and
    ingredients."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Karo8870/food.com-parsed-dataset", "parsed_recipes.csv", repo_type="dataset")
    df = pd.read_csv(path, low_memory=False)
    df["name_l"] = df["name"].astype(str).str.lower()
    df["minutes"] = df["cooking_time"].fillna(0) + df["preparation_time"].fillna(0)
    df = df[(df["minutes"] > 0) & (df["minutes"] <= 24 * 60)]
    df = df[df["description"].astype(str).str.len() > 20]

    out = []
    for cls in CLASSES:
        sub = df[df["name_l"].str.contains(CLASS_PATTERNS[cls], regex=True)]
        if cls in EXCLUDE:
            sub = sub[~sub["name_l"].str.contains(EXCLUDE[cls], regex=True)]
        n_match = len(sub)
        sub = sub.sample(n=min(per_class, len(sub)), random_state=SEED)
        print(f"{cls:22s} matched {n_match:5d} -> kept {len(sub)}")
        for idx, r in sub.iterrows():
            ingredients = [i.get("name", "") for i in _json_list(r["ingredients"]) if isinstance(i, dict)]
            tags = [t for t in _json_list(r["tags"]) if isinstance(t, str)]
            ing_text = " ".join(ingredients).lower()
            vegetarian = not MEAT_FISH.search(ing_text)
            vegan = vegetarian and not ANIMAL.search(ing_text)
            m = _RECIPE_ID.search(str(r["images"]))
            out.append({
                "id": f"r{idx}",
                "dish": cls,
                "name": str(r["name"]).strip(),
                "minutes": int(r["minutes"]),
                "vegetarian": bool(vegetarian),
                "vegan": bool(vegan),
                "category": r["RecipeCategory"] if isinstance(r["RecipeCategory"], str) else "",
                "tags": tags,
                "ingredients": ingredients,
                "description": str(r["description"]).strip(),
                "calories": float(r["calories"]) if pd.notna(r["calories"]) else None,
                "url": f"https://www.food.com/recipe/{m.group(1)}" if m else None,
            })

    RECIPES.parent.mkdir(parents=True, exist_ok=True)
    with RECIPES.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    veg = sum(r["vegetarian"] for r in out)
    print(f"wrote {len(out)} recipes ({veg} vegetarian) -> {RECIPES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-recipes", action="store_true")
    ap.add_argument("--per-class", type=int, default=30)
    a = ap.parse_args()
    if not a.skip_recipes:
        prepare_recipes(a.per_class)
    if not a.skip_images:
        prepare_images()
