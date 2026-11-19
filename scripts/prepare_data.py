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
EXCLUDE = {  # names that match the dish word but are not that dish (sauces, sides, mixes, other dishes)
    "chocolate_cake": r"cupcake|cheesecake|pancake|frosting only|icing|cookies?|\bbars?\b",
    "pizza": r"\bdough\b|\bsauce\b|\bcrust\b$|pot pie|sausage|casserole|\bbites?\b|cookies?|salad|\bdips?\b|"
             r"soup|seasoning|pasta|lasagn|burgers?",
    "hamburger": r"\bbuns?\b|\bseasoning\b|helper|burger king|alfredo|\bbake\b|casserole|soup|\bdip\b|sauce",
    "pancakes": r"\bsyrup\b|\bmix\b|gift|latke|potato|scallion",
    "caesar_salad": r"dressing|crouton|\bmix\b|sandwich|pizza|wrap",
    "falafel": r"sauce|dressing|\bdip\b|seasoning|mix$|crusted|waffle|guacamole",
    "guacamole": r"seasoning|\bmix$|with guacamole|w/ guacamole|guacamole sauce|quesadilla|falafel",
    "ramen": r"dessert",
    "spaghetti_bolognese": r"^bolognese sauce$|pizza",
    "sushi": r"sushi rice|sauce|burgers?|quesadilla|candy|fruit|filling|almonds|sandwich|dessert",
}

MEAT_FISH = re.compile(
    r"\b(beef|pork|chicken|turkey|bacon|ham|sausage|pepperoni|salami|prosciutto|"
    r"lamb|veal|duck|anchov(y|ies)|tuna|salmon|shrimp|prawns?|crab|fish|"
    r"gelatin|lard|chorizo|meat|steak|mince|ground round|crabmeat|eel|"
    r"worcestershire|fish sauce|oyster sauce|dashi|bonito|pancetta|"
    r"sardines?|mackerel|cod|tilapia|halibut|trout|herring|haddock|pollock|snapper|catfish|swordfish|"
    r"clams?|mussels?|scallops?|lobster|squid|octopus|oysters?|surimi|imitation crab|kamaboko|"
    r"caviar|roe|tobiko|masago|venison|bison|rabbit|meatballs?|hot dogs?|frankfurters?|kielbasa|spam)\b")
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


def _norm_name(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def method_keys() -> set[tuple[str, int]]:
    """(name, total minutes) of every recipe with a step-by-step method in
    KingName1/food.com (Kaggle RAW_recipes). Used to prefer complete recipes."""
    import glob

    from huggingface_hub import snapshot_download

    from src.config import DATA

    path = snapshot_download("KingName1/food.com", repo_type="dataset", allow_patterns=["data/*.parquet"],
                             cache_dir=str(DATA / "cache" / "huggingface" / "hub"))
    raw = pd.concat([pd.read_parquet(f, columns=["name", "minutes"])
                     for f in sorted(glob.glob(path + "/data/*.parquet"))])
    return set(zip(raw["name"].map(_norm_name), raw["minutes"].astype(int)))


def prepare_recipes(per_class: int) -> None:
    """Source: Food.com Recipes (Kaggle, irkaal, CC0), parsed copy on HF:
    Karo8870/food.com-parsed-dataset. Its `steps` column is broken (holds image
    URLs), so recipes are represented by name, description, category, tags and
    ingredients; the method and photo are added by scripts/enrich_recipes.py.

    Recipes shown in the app should be complete, so within each dish we prefer
    (1) photo + method, (2) method only, (3) photo only, (4) neither, and draw
    at random (seed 42) inside each tier."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Karo8870/food.com-parsed-dataset", "parsed_recipes.csv", repo_type="dataset")
    df = pd.read_csv(path, low_memory=False)
    df["name_l"] = df["name"].astype(str).str.lower()
    df["minutes"] = df["cooking_time"].fillna(0) + df["preparation_time"].fillna(0)
    df = df[(df["minutes"] > 0) & (df["minutes"] <= 24 * 60)]
    df = df[df["description"].astype(str).str.len() > 20]
    keys = method_keys()

    out = []
    taken: set[int] = set()
    for cls in CLASSES:
        sub = df[df["name_l"].str.contains(CLASS_PATTERNS[cls], regex=True)]
        if cls in EXCLUDE:
            sub = sub[~sub["name_l"].str.contains(EXCLUDE[cls], regex=True)]
        n_match = len(sub)
        has_img = sub["images"].astype(str).str.contains("http")
        has_steps = pd.Series([(_norm_name(n), int(m)) in keys for n, m in zip(sub["name"], sub["minutes"])],
                              index=sub.index)
        tier = (~has_steps).astype(int) * 2 + (~has_img).astype(int)  # 0 both, 1 steps, 2 photo, 3 neither
        sub = sub.assign(tier=tier).sample(frac=1, random_state=SEED).sort_values("tier", kind="stable")
        sub = sub[~sub.index.isin(taken)].head(per_class)  # one recipe never belongs to two dishes
        taken.update(sub.index)
        print(f"{cls:22s} matched {n_match:5d} -> kept {len(sub)} "
              f"(photo+method {int((sub.tier == 0).sum())}, method only {int((sub.tier == 1).sum())})")
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


def diet_labels(ingredients: list[str]) -> tuple[bool, bool]:
    ing_text = " ".join(ingredients).lower()
    vegetarian = not MEAT_FISH.search(ing_text)
    return vegetarian, vegetarian and not ANIMAL.search(ing_text)


def relabel() -> None:
    """Re-apply the diet rules to data/recipes.jsonl in place (same recipes, no download)."""
    recs = [json.loads(line) for line in RECIPES.open(encoding="utf-8") if line.strip()]
    for rec in recs:
        veg, vegan = diet_labels(rec["ingredients"])
        if (veg, vegan) != (rec["vegetarian"], rec["vegan"]):
            print(f"  {rec['dish']:20s} {rec['name'][:50]:50s} veg {rec['vegetarian']}->{veg} vegan {rec['vegan']}->{vegan}")
        rec["vegetarian"], rec["vegan"] = veg, vegan
    with RECIPES.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(recs)} recipes, {sum(r['vegetarian'] for r in recs)} vegetarian, "
          f"{sum(r['vegan'] for r in recs)} vegan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-recipes", action="store_true")
    ap.add_argument("--relabel", action="store_true", help="only re-apply diet labels to data/recipes.jsonl")
    ap.add_argument("--per-class", type=int, default=30)
    a = ap.parse_args()
    if a.relabel:
        relabel()
        sys.exit()
    if not a.skip_recipes:
        prepare_recipes(a.per_class)
    if not a.skip_images:
        prepare_images()
