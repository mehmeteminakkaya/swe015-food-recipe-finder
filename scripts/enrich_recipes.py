"""Add display details to data/recipes.jsonl for the recipe cards in the app.

Adds, per recipe (the retrieval fields name / description / tags / ingredients
are left untouched, so no metric changes):
  image          first Food.com photo URL, or null
  servings       number of servings, or null
  prep_minutes   preparation time
  cook_minutes   cooking time
  nutrition      calories, fat, carbs, protein, sugar, fiber, sodium (per serving)
  ingredient_lines  [{"qty", "unit", "name", "note"}] with quantities
  steps          step-by-step method, or [] when no source has it

Sources:
  - Karo8870/food.com-parsed-dataset (same rows as data/recipes.jsonl: id "r<row>")
  - KingName1/food.com (Kaggle RAW_recipes): steps, matched on name + total minutes

Usage:  python scripts/enrich_recipes.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import RECIPES


def _num(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _json(s):
    if not isinstance(s, str):
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


_NUTRITION_START = re.compile(r"^\s*(nutrition(al)?\s+(information|info|facts|values?)|per serving\b)", re.I)
_NUTRITION_LINE = re.compile(r"%\s*dv|daily value|^\s*(total fat|saturated fat|cholesterol|sodium|carbohydrates?|"
                             r"fiber|fibre|protein|sugars?|calories|vitamin [a-z]|calcium|iron)\b|^\s*\d+\s*g\s*$", re.I)


def _clean_steps(steps: list[str]) -> list[str]:
    """Some authors pasted the nutrition panel into the method: cut it off."""
    out = []
    for s in steps:
        if _NUTRITION_START.search(s):
            break
        if _NUTRITION_LINE.search(s):
            continue
        out.append(s)
    return out


def _sentence(step: str) -> str:
    """RAW_recipes steps are lower-case with spaced punctuation."""
    s = re.sub(r"\s+([,.;:!?)])", r"\1", step.strip())
    s = re.sub(r"\(\s+", "(", s)
    return (s[:1].upper() + s[1:]).rstrip(".") + "." if s else s


def main() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    recs = [json.loads(line) for line in RECIPES.open(encoding="utf-8") if line.strip()]

    karo = pd.read_csv(hf_hub_download("Karo8870/food.com-parsed-dataset", "parsed_recipes.csv",
                                       repo_type="dataset"), low_memory=False)
    raw_dir = snapshot_download("KingName1/food.com", repo_type="dataset", allow_patterns=["data/*.parquet"],
                                cache_dir=str(Path(__file__).resolve().parent.parent / "data" / "cache" / "huggingface" / "hub"))
    raw = pd.concat([pd.read_parquet(f, columns=["name", "minutes", "steps"])
                     for f in sorted(glob.glob(raw_dir + "/data/*.parquet"))])
    raw["k"] = raw["name"].map(_norm)
    raw = raw.set_index("k")

    n_img = n_steps = n_serv = 0
    for rec in recs:
        row = karo.loc[int(rec["id"][1:])]
        if str(row["name"]).strip() != rec["name"]:
            raise SystemExit(f"row mismatch for {rec['id']}: {row['name']!r} != {rec['name']!r}")

        imgs = [u for u in _json(row["images"]) if isinstance(u, str) and u.startswith("http")]
        rec["image"] = imgs[0] if imgs else None
        serv = _num(row["servings"])
        rec["servings"] = int(serv) if serv else None
        rec["prep_minutes"] = int(_num(row["preparation_time"]) or 0)
        rec["cook_minutes"] = int(_num(row["cooking_time"]) or 0)
        rec["nutrition"] = {k: _num(row[c]) for k, c in (
            ("calories", "calories"), ("fat_g", "total_fat"), ("carbs_g", "carbohydrates"),
            ("protein_g", "protein"), ("sugar_g", "sugar"), ("fiber_g", "fiber"), ("sodium_mg", "sodium"))}
        rec["ingredient_lines"] = [
            {"qty": str(i.get("quantity") or "").strip(), "unit": str(i.get("unit") or "").strip(),
             "name": str(i.get("name") or "").strip(), "note": str(i.get("description") or "").strip()}
            for i in _json(row["ingredients"]) if isinstance(i, dict) and i.get("name")
        ]

        steps: list[str] = []
        k = _norm(rec["name"])
        if k in raw.index:
            cand = raw.loc[[k]]
            same = cand[cand["minutes"].astype(int) == rec["minutes"]]
            if len(same):
                steps = _clean_steps([_sentence(s) for s in list(same.iloc[0]["steps"]) if str(s).strip()])
        rec["steps"] = steps

        n_img += rec["image"] is not None
        n_steps += bool(steps)
        n_serv += rec["servings"] is not None

    with RECIPES.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(recs)} recipes: {n_img} with photo, {n_serv} with servings, {n_steps} with steps -> {RECIPES}")


if __name__ == "__main__":
    main()
