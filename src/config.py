"""Shared paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Hugging Face downloads must stay inside the repo: data/cache is gitignored
# and the sandbox that runs this project only allows writes under ROOT.
HF_CACHE = ROOT / "data" / "cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "cache" / "torch"))
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = RAW / "food10"          # food10/{train,val,test}/{class}/*.jpg
RECIPES = DATA / "recipes.jsonl"
QUERIES = DATA / "queries.jsonl"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

CLASSES = [
    "caesar_salad", "chocolate_cake", "falafel", "guacamole", "hamburger",
    "pancakes", "pizza", "ramen", "spaghetti_bolognese", "sushi",
]

# Human-readable names used in the UI and appended to retrieval queries.
DISPLAY = {c: c.replace("_", " ") for c in CLASSES}

SEED = 42


def device() -> str:
    """'cuda' when a GPU is present, else 'cpu' (torch imported lazily)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
