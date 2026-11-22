"""Gradio demo: photo -> dish + Grad-CAM, free text -> top-5 recipes.

Run:  python app.py
The default models are the quantized ones (INT8 VGG16 and INT8 E5 encoder);
the float models can be selected in the UI for comparison.
"""
from __future__ import annotations

import html
import re
import time
from pathlib import Path

import os
import tempfile

import src.config  # noqa: F401  -- sets HF/TORCH cache dirs before huggingface_hub loads

# Uploaded photos are cached by Gradio and served back to the page. Inside a synced
# folder (this repo lives in OneDrive) the sync client can touch a file while it is
# being served, and the photo shows up broken; keep that cache in the local temp dir.
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "swe015-gradio"))

import gradio as gr
import numpy as np
import torch
from PIL import Image

# gradio 5.6.0 bug: the API schema of gr.Image contains "additionalProperties: true"
# and gradio_client's JSON->Python type helper crashes on the boolean, which makes
# every request to / fail with a 500. Return a plain "object" for booleans instead.
import gradio_client.utils as _gc_utils  # noqa: E402

_orig_schema_to_type = _gc_utils._json_schema_to_python_type


def _safe_schema_to_type(schema, defs=None):
    if isinstance(schema, bool):
        return "object"
    return _orig_schema_to_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_schema_to_type

from src.config import MODELS, device
from src.data import candidate_recipes, load_recipes
from src.retrieval.text import extract_constraints
from src.retrieval.tfidf import TfidfRetriever
from src.retrieval.w2v import Word2VecRetriever
from src.vision.data import eval_transform
from src.vision.loaders import Vgg16Int8, load_small_cnn, load_vgg16

RECIPES = load_recipes()
ROOT = MODELS / "retrieval"
DEVICE = device()

PHOTO_MODELS = ["VGG16 INT8 (default)", "VGG16 fp32", "Small CNN (from scratch)"]
TEXT_MODELS = ["E5 INT8 (default)", "E5 fp32", "tf-idf + cosine", "word2vec (trained here)"]

_cache: dict[str, object] = {}


def get_retriever(name: str):
    if name in _cache:
        return _cache[name]
    if name.startswith("E5 INT8"):
        from src.retrieval.quantize import QuantizedBertRetriever
        r = QuantizedBertRetriever.load(ROOT / "bert_e5_int8", RECIPES)
    elif name.startswith("E5 fp32"):
        from src.retrieval.bert import BertRetriever
        r = BertRetriever.load(ROOT / "bert_e5", RECIPES)
    elif name.startswith("tf-idf"):
        r = TfidfRetriever.load(ROOT / "tfidf", RECIPES)
    else:
        r = Word2VecRetriever.load(ROOT / "word2vec", RECIPES)
    _cache[name] = r
    return r


def get_vgg():
    if "vgg" not in _cache:
        _cache["vgg"], _cache["vgg_ck"] = load_vgg16(DEVICE)
    return _cache["vgg"], _cache["vgg_ck"]


def get_vgg_int8():
    if "vgg_int8" not in _cache:
        _cache["vgg_int8"] = Vgg16Int8(device="cpu")
    return _cache["vgg_int8"]


def get_cnn():
    if "cnn" not in _cache:
        _cache["cnn"], _cache["cnn_ck"] = load_small_cnn(DEVICE)
    return _cache["cnn"], _cache["cnn_ck"]


def gradcam(img: Image.Image, cls_idx: int) -> np.ndarray:
    """Grad-CAM of the fp32 VGG16 on conv5_3, overlayed on the input photo."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    from src.vision import vgg

    model, ck = get_vgg()
    tensor = eval_transform(ck["img_size"])(img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)  # frozen weights: drive the graph from the input
    with GradCAM(model=model, target_layers=[vgg.target_layer(model)]) as cam:
        heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(cls_idx)])[0]
    rgb = np.asarray(img.resize((ck["img_size"], ck["img_size"])), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    # the model sees a square; give the overlay back the photo's own shape so the two images match
    return np.asarray(Image.fromarray(overlay).resize(img.size, Image.BILINEAR))


def classify(img: Image.Image, photo_model: str):
    """Return (class names, probabilities tensor, milliseconds)."""
    t0 = time.perf_counter()
    if photo_model.startswith("VGG16 INT8"):
        m = get_vgg_int8()
        x = eval_transform(m.img_size)(img.convert("RGB")).unsqueeze(0)
        logits = torch.from_numpy(m.logits(x))
        cls, ck = m.classes, {"img_size": m.img_size}
    elif photo_model.startswith("VGG16"):
        model, ck = get_vgg()
        x = eval_transform(ck["img_size"])(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(x).cpu()
        cls = ck["classes"]
    else:
        model, ck = get_cnn()
        x = eval_transform(ck["img_size"])(img.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            logits = model(x).cpu()
        cls = ck["classes"]

    prob = torch.softmax(logits, dim=1)[0]
    return cls, prob, (time.perf_counter() - t0) * 1000


LOW_CONFIDENCE = 0.30


def predict_photo(img: Image.Image, photo_model: str):
    if img is None:
        return None, "", "", "Upload a photo."
    cls, prob, ms = classify(img, photo_model)
    top = int(prob.argmax())
    confidence = float(prob[top])
    text = "\n".join(
        f"{cls[i].replace('_', ' ')}: {float(prob[i]):.1%}"
        for i in prob.topk(5).indices.tolist()
    )
    cam_img = gradcam(img.convert("RGB"), top)
    warn = "" if confidence >= LOW_CONFIDENCE else " · low confidence: the photo may not show one of the ten dishes"
    return (cam_img, cls[top].replace("_", " "), text,
            f"{photo_model} · {ms:.0f} ms · confidence {confidence:.1%}{warn}")


def _pool(dish_key: str, max_minutes: int | None, diet: list[str]) -> list:
    """Recipes obeying the constraints (and the predicted dish, when there is one)."""
    q = {"target_dish": dish_key, "max_minutes": max_minutes, "diet": diet}
    pool = [r for r in RECIPES if r["dish"] == dish_key] if dish_key else RECIPES
    return candidate_recipes(q, pool)


def build_pool(dish_key: str, c) -> tuple[list, str]:
    """Tightest pool that still has at least five recipes; reports what was relaxed."""
    steps = [
        (c.max_minutes, sorted(c.diet), ""),
        (None, sorted(c.diet), "the time limit was relaxed"),
        (None, [], "the time and diet filters were relaxed"),
    ]
    for max_minutes, diet, note in steps:
        pool = _pool(dish_key, max_minutes, diet)
        if len(pool) >= 5:
            return pool, note
    return RECIPES, "too few recipes, so all recipes were searched"


def search_hits(text: str, photo_dish: str, text_model: str):
    """Return (hits, constraints, pool size, relax note, ms) or None for an empty request."""
    if not text.strip() and not photo_dish:
        return None
    query = text.strip()
    if photo_dish and photo_dish.lower() not in query.lower():
        query = f"{query} {photo_dish}".strip()

    t0 = time.perf_counter()
    c = extract_constraints(query)
    dish_key = photo_dish.replace(" ", "_") if photo_dish else ""
    pool, note = build_pool(dish_key, c)
    r = get_retriever(text_model)
    hits = r.search(query, pool, k=5)
    return hits, c, len(pool), note, (time.perf_counter() - t0) * 1000


def search(text: str, photo_dish: str, text_model: str):
    res = search_hits(text, photo_dish, text_model)
    if res is None:
        return [], "Type a request or upload a photo."
    hits, c, n_pool, note, ms = res
    rows = []
    for rec, score in hits:
        badges = []
        if rec["vegetarian"]:
            badges.append("vegetarian")
        if rec["vegan"]:
            badges.append("vegan")
        rows.append([
            rec["name"],
            f"{rec['minutes']} min",
            ", ".join(badges) or "—",
            f"{score:.3f}",
        ])
    info = f"{text_model} · {ms:.0f} ms · {n_pool} candidate recipes"
    if c.max_minutes or c.diet:
        info += f" · asked: ≤{c.max_minutes or '∞'} min, diet: {', '.join(sorted(c.diet)) or 'none'}"
    if note:
        info += f" · {note}"
    return rows, info


# ----------------------------------------------------------------- UI ------

def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _pretty(name: str) -> str:
    return name.replace("_", " ")


PHOTO_EMPTY = (
    '<div class="rf-empty">Upload or paste a photo. It is recognised straight away, '
    "and the dish and where the model looked appear here.</div>"
)
RECIPES_EMPTY = (
    '<div class="rf-empty">Say what you would like to cook, or recognise a photo first. '
    "The best recipe appears here in full, with four more below it.</div>"
)


def _cam_note(photo_model: str) -> str:
    """Grad-CAM needs gradients, so it always runs on the fp32 VGG16; say so for the other models."""
    if photo_model.startswith("VGG16 fp32"):
        return "The heat map is Grad-CAM on this model's last convolutional layer (conv5_3)."
    if photo_model.startswith("VGG16 INT8"):
        return ("The heat map is Grad-CAM on the same VGG16 before quantization (conv5_3): "
                "an INT8 graph has no gradients.")
    return ("The heat map comes from the fine-tuned VGG16 (conv5_3) for the dish predicted here, "
            "not from the small CNN.")


def render_prediction(cls, prob, ms: float, photo_model: str) -> str:
    order = prob.topk(5).indices.tolist()
    top = order[0]
    conf = float(prob[top])
    bars = []
    for rank, i in enumerate(order):
        p = float(prob[i])
        bars.append(
            f'<li class="rf-bar{" is-top" if rank == 0 else ""}">'
            f'<span class="rf-bar-name">{_esc(_pretty(cls[i]))}</span>'
            f'<span class="rf-bar-track"><span style="width:{max(p * 100, 0.5):.1f}%"></span></span>'
            f'<span class="rf-bar-val">{p:.1%}</span></li>'
        )
    warn = ""
    second = float(prob[order[1]])
    # unsure = low confidence, or a close runner-up (falafel read as bolognese at 41 % vs 28 %)
    if conf < 0.5 or conf - second < 0.15:
        warn = (f'<p class="rf-warn">Not sure: only {conf:.1%} confident, and it could also be '
                f"{_esc(_pretty(cls[order[1]]))} ({second:.1%}). Check the heat map before trusting it.</p>")
    kicker = "This might be" if warn else "This looks like"
    return (
        '<div class="rf-pred">'
        f'<p class="rf-pred-kicker">{kicker}</p>'
        f'<h2 class="rf-dish">{_esc(_pretty(cls[top]))}</h2>{warn}'   # warning above the fold, right under the name
        f'<ul class="rf-bars">{"".join(bars)}</ul>'
        f'<p class="rf-meta">{_esc(photo_model)}, {ms:.0f} ms. {_cam_note(photo_model)}</p>'
        "</div>"
    )


SUMMARY_EMPTY = (
    '<div class="rf-summary is-empty">'
    '<div class="rf-fact"><span>Time limit</span><b>—</b></div>'
    '<div class="rf-fact"><span>Diet</span><b>—</b></div>'
    '<div class="rf-fact"><span>Dish from photo</span><b>—</b></div>'
    '<div class="rf-fact"><span>Candidates</span><b>—</b></div>'
    "</div>"
    '<p class="rf-meta">After you press <b>Find recipes</b>, this shows how your request was read.</p>'
)


def _txt(s) -> str:
    """Food.com text arrives HTML-encoded ("&amp;"): decode once, then escape for output."""
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), str(s or ""))  # literal "'"
    text = text.replace('\\"', '"').replace("\\'", "'")  # escaped quotes left over from the CSV
    return _esc(html.unescape(text))


def _fmt_minutes(m: int | None) -> str:
    if not m:
        return "—"
    h, mm = divmod(int(m), 60)
    return f"{h} h {mm} min" if h and mm else (f"{h} h" if h else f"{mm} min")


def _diet_chip(rec) -> str:
    if rec.get("vegan"):
        return '<span class="rf-chip is-veg">vegan</span>'
    if rec.get("vegetarian"):
        return '<span class="rf-chip is-veg">vegetarian</span>'
    return '<span class="rf-chip">contains meat or fish</span>'


def _media(rec, cls: str) -> str:
    if rec.get("image"):
        return (f'<div class="{cls}"><img src="{_esc(rec["image"])}" alt="{_txt(rec["name"])}" '
                'referrerpolicy="no-referrer"></div>')
    return (f'<div class="{cls} rf-noimg"><span>{_esc(_pretty(rec["dish"]))}</span>'
            "<small>No photo in the dataset</small></div>")


def _facts(rec) -> str:
    n = rec.get("nutrition") or {}
    cells = [
        ("Total", _fmt_minutes(rec["minutes"])),
        ("Prep", _fmt_minutes(rec.get("prep_minutes"))),
        ("Cook", _fmt_minutes(rec.get("cook_minutes"))),
        ("Serves", str(rec["servings"]) if rec.get("servings") else "—"),
        ("Calories", f"{n['calories']:.0f} kcal" if n.get("calories") else "—"),
    ]
    return '<dl class="rf-facts">' + "".join(
        f"<div><dt>{k}</dt><dd>{_esc(v)}</dd></div>" for k, v in cells) + "</dl>"


def _ingredients(rec) -> str:
    lines = rec.get("ingredient_lines") or [{"qty": "", "unit": "", "name": i, "note": ""}
                                            for i in rec.get("ingredients", [])]
    items = []
    for i in lines:
        amount = " ".join(x for x in (i["qty"], i["unit"]) if x)
        amount = re.sub(r"(\d)\s*-\s*(\d)", r"\1–\2", amount)  # "1 -2" -> "1–2"
        note = f' <span class="rf-note">{_txt(i["note"])}</span>' if i.get("note") else ""
        items.append(f'<li><span class="rf-qty">{_txt(amount)}</span><span>{_txt(i["name"])}{note}</span></li>')
    return f'<ul class="rf-ingredients">{"".join(items)}</ul>'


def _method(rec) -> str:
    if rec.get("steps"):
        # the source steps get a full stop added: drop it after "!", "?", ":" ("Bon appetit!.")
        steps = (re.sub(r"([!?:.])\s*\.$", r"\1", str(s).strip()) for s in rec["steps"])
        return '<ol class="rf-steps">' + "".join(f"<li>{_txt(s)}</li>" for s in steps) + "</ol>"
    link = (f' <a href="{_esc(rec["url"])}" target="_blank" rel="noopener">Open it on Food.com</a>.'
            if rec.get("url") else "")
    return f'<p class="rf-nomethod">The dataset has no step-by-step method for this recipe.{link}</p>'


def _nutrition(rec) -> str:
    n = rec.get("nutrition") or {}
    cells = [("Fat", n.get("fat_g"), "g"), ("Carbs", n.get("carbs_g"), "g"), ("Protein", n.get("protein_g"), "g"),
             ("Sugar", n.get("sugar_g"), "g"), ("Fibre", n.get("fiber_g"), "g"), ("Sodium", n.get("sodium_mg"), "mg")]
    return ('<dl class="rf-nutri">' + "".join(
        f"<div><dt>{k}</dt><dd>{'—' if v is None else f'{v:.0f} {u}'}</dd></div>" for k, v, u in cells)
        + "</dl>")


def _title(rec) -> str:
    name = _txt(rec["name"])
    if rec.get("url"):
        return f'<a href="{_esc(rec["url"])}" target="_blank" rel="noopener">{name}</a>'
    return name


def _score(score: float) -> str:
    return (f'<span class="rf-score" title="cosine similarity between request and recipe">'
            f'<span class="rf-score-track"><span style="width:{min(max(score, 0), 1) * 100:.0f}%"></span></span>'
            f"match {score:.2f}</span>")


_BOILERPLATE = re.compile(r"^\s*make and share this .* recipe from food\.com\.?\s*$", re.I)


def _description(rec, limit: int = 320) -> str:
    """The author's description, or '' when it is Food.com's generic filler sentence."""
    desc = html.unescape(rec.get("description") or "").strip()
    if _BOILERPLATE.match(desc):
        return ""
    return desc if len(desc) <= limit else desc[: limit - 3].rsplit(" ", 1)[0] + "…"


def render_featured(rec, score: float) -> str:
    desc = _description(rec)
    return (
        '<article class="rf-feature">'
        f'{_media(rec, "rf-feature-media")}'
        '<div class="rf-feature-head">'
        f'<div class="rf-rankline"><span class="rf-rank">1</span><span class="rf-best">Best match</span>{_score(score)}</div>'
        f"<h3>{_title(rec)}</h3>"
        f'<div class="rf-chips">{_diet_chip(rec)}<span class="rf-chip is-dish">{_esc(_pretty(rec["dish"]))}</span></div>'
        + (f'<p class="rf-desc">{_esc(desc)}</p>' if desc else "") +
        f"{_facts(rec)}"
        "</div>"
        '<div class="rf-feature-cols">'
        f'<section><h4>Ingredients</h4>{_ingredients(rec)}</section>'
        f"<section><h4>Method</h4>{_method(rec)}</section>"
        "</div>"
        f'<section class="rf-feature-nutri"><h4>Nutrition per serving</h4>{_nutrition(rec)}</section>'
        "</article>"
    )


def render_card(rec, score: float, rank: int) -> str:
    n = rec.get("nutrition") or {}
    kcal = f"{n['calories']:.0f} kcal" if n.get("calories") else ""
    serves = f"serves {rec['servings']}" if rec.get("servings") else ""
    meta = " · ".join(x for x in (_fmt_minutes(rec["minutes"]), serves, kcal) if x)
    return (
        '<article class="rf-card">'
        f'{_media(rec, "rf-card-media")}'
        '<div class="rf-card-body">'
        f'<div class="rf-rankline"><span class="rf-rank">{rank}</span>{_score(score)}</div>'
        f'<h3 title="{_txt(rec["name"])}">{_title(rec)}</h3>'
        f'<p class="rf-card-meta">{_esc(meta)}</p>'
        f'<div class="rf-chips">{_diet_chip(rec)}</div>'
        '<details><summary>Full recipe</summary>'
        f"{_facts(rec)}<h4>Ingredients</h4>{_ingredients(rec)}<h4>Method</h4>{_method(rec)}"
        f"<h4>Nutrition per serving</h4>{_nutrition(rec)}"
        "</details></div></article>"
    )


def render_summary(res, text_model: str) -> str:
    if res is None:
        return SUMMARY_EMPTY
    hits, c, n_pool, note, ms = res
    diet = "vegan" if "vegan" in c.diet else ("vegetarian" if "vegetarian" in c.diet else "any")
    dish = ""
    if hits:
        dishes = {r["dish"] for r, _ in hits}
        dish = _pretty(next(iter(dishes))).capitalize() if len(dishes) == 1 else "any"
    cells = [
        ("Time limit", f"≤ {c.max_minutes} min" if c.max_minutes else "none"),
        ("Diet", diet),
        ("Dish from photo", dish or "any"),
        ("Candidates", f"{n_pool} recipes"),
    ]
    out = '<div class="rf-summary">' + "".join(
        f'<div class="rf-fact"><span>{k}</span><b>{_esc(v)}</b></div>' for k, v in cells) + "</div>"
    if note:
        out += (f'<p class="rf-relaxed">Fewer than five recipes match exactly, so {_esc(note)}. '
                "Some results may break that limit.</p>")
    out += f'<p class="rf-meta">{_esc(text_model)} ranked them in {ms:.0f} ms.</p>'
    return out


def render_recipes(res, text_model: str) -> str:
    if res is None:
        return RECIPES_EMPTY
    hits = res[0]
    if not hits:
        return '<div class="rf-empty">No recipe matches this request. Loosen the time or diet limit and try again.</div>'
    (top, top_score), rest = hits[0], hits[1:]
    cards = "".join(render_card(r, s, i) for i, (r, s) in enumerate(rest, 2))
    more = f'<h4 class="rf-more">Other matches</h4><div class="rf-grid">{cards}</div>' if cards else ""
    return render_featured(top, top_score) + more


def ui_predict(img, photo_model: str):
    if img is None:
        return None, "", PHOTO_EMPTY, gr.update(visible=False)
    img = img.convert("RGB")
    cls, prob, ms = classify(img, photo_model)
    top = int(prob.argmax())
    cam_img = gradcam(img, top)
    return cam_img, _pretty(cls[top]), render_prediction(cls, prob, ms, photo_model), gr.update(visible=True)


def ui_search(text: str, dish: str, text_model: str):
    res = search_hits(text or "", dish or "", text_model)
    return render_summary(res, text_model), render_recipes(res, text_model)


def ui_clear():
    return None, "", PHOTO_EMPTY, gr.update(visible=False)


def ui_reset_recipes():
    return SUMMARY_EMPTY, RECIPES_EMPTY


# --- side-by-side comparison of every model ------------------------------

CMP_PHOTO_EMPTY = '<div class="rf-empty">Recognise a photo to see what every image model says about it.</div>'
CMP_SEARCH_EMPTY = '<div class="rf-empty">Find recipes to see how every search model ranks the same request.</div>'


def ui_compare_photo(img) -> str:
    if img is None:
        return CMP_PHOTO_EMPTY
    img = img.convert("RGB")
    cards = []
    for m in PHOTO_MODELS:
        cls, prob, ms = classify(img, m)
        top = int(prob.argmax())
        default = " is-default" if m == PHOTO_MODELS[0] else ""
        cards.append(
            f'<div class="rf-cmp-card{default}"><span class="rf-cmp-model">{_esc(m)}</span>'
            f'<b class="rf-cmp-dish">{_esc(_pretty(cls[top]))}</b>'
            f'<span class="rf-cmp-stat">{float(prob[top]):.1%} confidence · {ms:.0f} ms</span></div>'
        )
    return (f'<h4 class="rf-cmp-group">Image models <span>same photo</span></h4>'
            f'<div class="rf-cmp rf-cmp-3">{"".join(cards)}</div>')


def ui_compare_search(text: str, dish: str) -> str:
    runs = [(m, search_hits(text or "", dish or "", m)) for m in TEXT_MODELS]
    if runs[0][1] is None:
        return CMP_SEARCH_EMPTY
    shown = {r["id"] for r, _ in runs[0][1][0]}  # what the app shows (default model)
    cards = []
    for m, res in runs:
        hits, _, n_pool, _, ms = res
        items = "".join(
            f'<li class="{"is-shared" if r["id"] in shown and m != TEXT_MODELS[0] else ""}"><span>{_txt(r["name"])}</span>'
            f"<small>{s:.2f}</small></li>" for r, s in hits)
        default = " is-default" if m == TEXT_MODELS[0] else ""
        cards.append(
            f'<div class="rf-cmp-card{default}"><span class="rf-cmp-model">{_esc(m)}</span>'
            f'<span class="rf-cmp-stat">{f"{ms:.0f} ms" if ms >= 1 else "<1 ms"} · {n_pool} candidates</span>'
            f'<ol>{items}</ol></div>'
        )
    return (f'<h4 class="rf-cmp-group">Search models <span>same request</span></h4>'
            f'<div class="rf-cmp rf-cmp-4">{"".join(cards)}</div>'
            '<p class="rf-meta">Bold recipes are also in the default model\'s top five. '
            "The number is each model's own cosine similarity, so it is not comparable across models.</p>")


def ui_refresh_recipes(text: str, dish: str, text_model: str):
    """After a new photo: re-run the request for the new dish, or clear the results if there is none."""
    if not (text or "").strip():
        return ui_reset_recipes()
    return ui_search(text, dish, text_model)


def _noop():
    """Carries a js-only step: Gradio does not continue a .then() chain after a step without a Python fn."""
    return None


# browser-side helpers, run after the Python side has returned
SCROLL_TO_RECIPES = """() => { setTimeout(() => {
  const el = document.querySelector('.rf-results');
  if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
}, 120); }"""
FOCUS_REQUEST = """() => { setTimeout(() => {
  const t = document.querySelector('#rf-query textarea, #rf-query input');
  // desktop only: on a phone the keyboard would cover the result that just appeared
  if (matchMedia('(pointer: fine)').matches) { if (t && !t.value) t.focus({preventScroll: true}); }
  else { const r = document.querySelector('#rf-pred'); if (r) r.scrollIntoView({behavior: 'smooth', block: 'start'}); }
}, 120); }"""


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Young+Serif&family=Figtree:wght@400;500;600;700&display=swap');

:root, .dark {
  --rf-paper: #F1F4EF;
  --rf-surface: #FFFFFF;
  --rf-ink: #17301F;
  --rf-muted: #5B6A60;
  --rf-line: #D3DCD4;
  --rf-saffron: #E2A21B;
  --rf-saffron-deep: #B87C05;
  --rf-tomato: #C4432B;
  --rf-leaf: #3D7A4C;
  --rf-leaf-soft: #E1EEE3;
}
body, gradio-app, .gradio-container { background: var(--rf-paper) !important; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; padding: 28px 24px 48px !important;
  font-family: 'Figtree', system-ui, sans-serif !important; color: var(--rf-ink) !important; }
footer { display: none !important; }

#rf-head { padding: 4px 4px 18px; border-bottom: 1px solid var(--rf-line); margin-bottom: 22px; }
#rf-head h1 { font-family: 'Young Serif', Georgia, serif; font-weight: 400; font-size: 44px; line-height: 1.05;
  letter-spacing: -0.01em; margin: 0 0 8px; color: var(--rf-ink); }
#rf-head p { margin: 0; font-size: 17px; color: var(--rf-muted); max-width: 62ch; }

.rf-panel { background: var(--rf-surface) !important; border: 1px solid var(--rf-line) !important;
  border-radius: 14px !important; padding: 20px !important; }
.rf-step { display: flex; align-items: baseline; gap: 12px; margin: 0 0 4px; }
.rf-step .n { font-family: 'Young Serif', serif; font-size: 15px; width: 28px; height: 28px; border-radius: 50%;
  display: inline-grid; place-items: center; background: var(--rf-ink); color: #fff; flex: none; transform: translateY(-3px); }
.rf-step h2 { font-family: 'Young Serif', serif; font-weight: 400; font-size: 24px; margin: 0; color: var(--rf-ink); }
.rf-step-sub { color: var(--rf-muted); font-size: 14.5px; margin: 0 0 10px 40px; }
.rf-results { margin-top: 18px !important; }

.rf-models .wrap { gap: 6px !important; }
.rf-models label { border-radius: 999px !important; font-size: 13.5px !important; }
.rf-models span[data-testid="block-info"], .rf-models .block-info { font-size: 13px !important; color: var(--rf-muted) !important; }

button.primary, .rf-go button, button.rf-go { background: var(--rf-saffron) !important; color: var(--rf-ink) !important;
  border: none !important; font-weight: 700 !important; font-size: 16px !important; border-radius: 10px !important; }
button.primary:hover { background: var(--rf-saffron-deep) !important; color: #fff !important; }
button.secondary { border-radius: 10px !important; }

.rf-empty { border: 1.5px dashed var(--rf-line); border-radius: 12px; padding: 22px; color: var(--rf-muted);
  font-size: 15px; line-height: 1.5; }

.rf-pred-kicker { margin: 6px 0 0; color: var(--rf-muted); font-size: 15px; }
.rf-dish { font-family: 'Young Serif', Georgia, serif !important; font-weight: 400 !important; font-size: 52px !important;
  line-height: 1 !important; margin: 2px 0 16px !important; color: var(--rf-ink) !important; text-transform: capitalize; }
.rf-bars { list-style: none; margin: 0; padding: 0; display: grid; gap: 7px; }
.rf-bar { display: grid; grid-template-columns: 150px 1fr 56px; align-items: center; gap: 10px; font-size: 14.5px; color: var(--rf-muted); }
.rf-bar-name { text-transform: capitalize; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rf-bar-track { height: 8px; background: #E9EEE9; border-radius: 99px; overflow: hidden; }
.rf-bar-track span { display: block; height: 100%; background: #AFC0B3; border-radius: 99px; transition: width .5s ease; }
.rf-bar.is-top { color: var(--rf-ink); font-weight: 700; }
.rf-bar.is-top .rf-bar-track { height: 12px; }
.rf-bar.is-top .rf-bar-track span { background: var(--rf-saffron); }
.rf-bar-val { text-align: right; font-variant-numeric: tabular-nums; }
.rf-warn { margin: 14px 0 16px; padding: 10px 12px; border-left: 3px solid var(--rf-tomato); background: #FBEDEA;
  color: #7A2617; font-size: 14.5px; border-radius: 0 8px 8px 0; }
.rf-meta { color: var(--rf-muted); font-size: 13px; margin: 14px 0 0; line-height: 1.45; }

/* request summary: 2 x 2 facts, mirrors the confidence bars on the photo side */
.rf-summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 4px; }
.rf-fact { background: var(--rf-paper); border-radius: 10px; padding: 12px 14px; }
.rf-fact span { display: block; font-size: 13px; color: var(--rf-muted); }
.rf-fact b { display: block; font-family: 'Young Serif', serif; font-weight: 400; font-size: 24px; line-height: 1.2;
  color: var(--rf-ink); margin-top: 2px; }
.rf-summary.is-empty b { color: #B9C4BC; }
.rf-relaxed { margin: 12px 0 0; font-size: 14px; color: #6B4A00; background: #FBF0D5; padding: 8px 12px;
  border-radius: 8px; line-height: 1.45; }

.rf-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0; }
.rf-chip { display: inline-block; padding: 2px 10px; border-radius: 99px; font-size: 13px; font-weight: 600;
  line-height: 1.6; background: #ECEFEC; color: var(--rf-muted); }
.rf-chip.is-veg { background: var(--rf-leaf-soft); color: var(--rf-leaf); }
.rf-chip.is-dish { background: #FBF0D5; color: #6B4A00; text-transform: capitalize; }

.rf-rankline { display: flex; align-items: center; gap: 10px; }
.rf-rank { font-family: 'Young Serif', serif; font-size: 22px; width: 38px; height: 38px; border-radius: 50%;
  display: inline-grid; place-items: center; background: var(--rf-saffron); color: var(--rf-ink); flex: none; }
.rf-best { font-weight: 700; font-size: 14px; color: var(--rf-saffron-deep); }
.rf-score { margin-left: auto; display: inline-flex; align-items: center; gap: 8px; font-size: 13px;
  color: var(--rf-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.rf-score-track { display: inline-block; width: 64px; height: 6px; background: #E5EBE6; border-radius: 99px; overflow: hidden; }
.rf-score-track span { display: block; height: 100%; background: var(--rf-ink); border-radius: 99px; }

#rf-recipes h3 { font-family: 'Young Serif', Georgia, serif !important; font-weight: 400 !important;
  color: var(--rf-ink) !important; line-height: 1.2 !important; margin: 10px 0 0 !important; }
#rf-recipes h3 a { color: inherit !important; text-decoration: none; }
#rf-recipes h3 a:hover { text-decoration: underline; text-underline-offset: 3px; }
#rf-recipes h4 { font-family: 'Figtree', sans-serif !important; font-size: 13px !important; font-weight: 700 !important;
  color: var(--rf-muted) !important; margin: 16px 0 8px !important; }

.rf-facts { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin: 14px 0 0; }
.rf-facts div, .rf-nutri div { background: var(--rf-paper); border-radius: 10px; padding: 9px 11px; }
.rf-facts dt, .rf-nutri dt { font-size: 12px; color: var(--rf-muted); }
.rf-facts dd, .rf-nutri dd { margin: 2px 0 0; font-weight: 700; font-size: 15px; color: var(--rf-ink);
  font-variant-numeric: tabular-nums; }
.rf-nutri { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin: 0; }

.rf-ingredients { list-style: none; margin: 0; padding: 0; }
.rf-ingredients li { display: grid; grid-template-columns: 112px 1fr; gap: 10px; padding: 6px 0;
  border-bottom: 1px solid #E5EBE6; font-size: 14.5px; line-height: 1.4; }
.rf-qty { font-weight: 700; color: var(--rf-ink); font-variant-numeric: tabular-nums; }
.rf-note { color: var(--rf-muted); font-size: 13px; }
.rf-steps { margin: 0; padding-left: 22px; font-size: 14.5px; line-height: 1.55; }
.rf-steps li { margin-bottom: 7px; padding-left: 4px; }
.rf-steps li::marker { font-weight: 700; color: var(--rf-saffron-deep); }
.rf-nomethod { margin: 0; padding: 14px; border: 1.5px dashed var(--rf-line); border-radius: 10px;
  color: var(--rf-muted); font-size: 14.5px; line-height: 1.5; }
.rf-nomethod a { color: var(--rf-ink); font-weight: 700; }
.rf-desc { margin: 10px 0 0; color: var(--rf-muted); font-size: 15px; line-height: 1.55; max-width: 68ch; }

/* best match: photo and head side by side, then two equal columns, then nutrition */
.rf-feature { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px;
  border: 1px solid var(--rf-line); border-radius: 14px; padding: 20px; background: var(--rf-surface); }
.rf-feature-media, .rf-card-media { border-radius: 10px; overflow: hidden; background: var(--rf-paper); }
.rf-feature-media { aspect-ratio: 4 / 3; }
.rf-feature-media.rf-noimg { aspect-ratio: auto; min-height: 180px; align-self: start; }
.rf-feature-media img, .rf-card-media img { width: 100%; height: 100%; object-fit: cover; display: block; }
.rf-feature-head h3 { font-size: 32px !important; }
.rf-feature-cols { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px;
  border-top: 1px solid var(--rf-line); padding-top: 4px; }
.rf-feature-nutri { grid-column: 1 / -1; }
.rf-feature-nutri h4 { margin-top: 0 !important; }

/* the other four: equal columns */
.rf-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
.rf-card { display: flex; flex-direction: column; border: 1px solid var(--rf-line); border-radius: 14px;
  background: var(--rf-surface); overflow: hidden; }
.rf-card-media { aspect-ratio: 4 / 3; border-radius: 0; }
.rf-card-body { padding: 14px 16px 16px; display: flex; flex-direction: column; flex: 1; }
.rf-card h3 { font-size: 19px !important; }
.rf-card-meta { margin: 6px 0 0; font-size: 13.5px; color: var(--rf-muted); }
.rf-card details { margin-top: auto; padding-top: 12px; }
.rf-card summary { cursor: pointer; font-weight: 700; font-size: 14px; color: var(--rf-ink); padding: 8px 0 0;
  border-top: 1px solid #E5EBE6; }
.rf-card .rf-facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.rf-card .rf-nutri { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.rf-card .rf-ingredients li { grid-template-columns: 76px 1fr; font-size: 13.5px; }
.rf-card .rf-steps { font-size: 13.5px; }

/* model comparison: equal columns, the default model outlined */
.rf-cmp { display: grid; gap: 14px; margin-bottom: 14px; }
.rf-cmp-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.rf-cmp-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.rf-cmp-card { border: 1px solid var(--rf-line); border-radius: 12px; padding: 14px 16px; background: var(--rf-surface);
  display: flex; flex-direction: column; gap: 4px; }
.rf-cmp-card.is-default { border: 2px solid var(--rf-saffron); background: #FFFCF3; }
.rf-cmp-model { font-size: 13px; font-weight: 700; color: var(--rf-muted); }
.rf-cmp-dish { font-family: 'Young Serif', serif; font-weight: 400; font-size: 26px; line-height: 1.15; color: var(--rf-ink);
  text-transform: capitalize; }
.rf-cmp-stat { font-size: 13px; color: var(--rf-muted); font-variant-numeric: tabular-nums; }
.rf-cmp-card ol { margin: 8px 0 0; padding-left: 20px; font-size: 13.5px; line-height: 1.4; color: var(--rf-muted); }
.rf-cmp-card li { margin-bottom: 6px; }
.rf-cmp-card li span { display: inline; }
.rf-cmp-card li small { display: block; font-variant-numeric: tabular-nums; color: #8A978E; }
.rf-cmp-card li.is-shared span { color: var(--rf-ink); font-weight: 700; }

.rf-noimg { display: grid; place-content: center; text-align: center; gap: 4px;
  background: repeating-linear-gradient(135deg, #E9EFE9 0 12px, #F1F4EF 12px 24px); }
.rf-noimg span { font-family: 'Young Serif', serif; font-size: 26px; color: var(--rf-ink); text-transform: capitalize; }
.rf-noimg small { font-size: 12.5px; color: var(--rf-muted); }


@media (max-width: 1100px) {
  .rf-grid, .rf-cmp-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 760px) {
  #rf-head h1 { font-size: 34px; }
  .rf-dish { font-size: 40px !important; }
  .rf-bar { grid-template-columns: 110px 1fr 52px; }
  .rf-feature, .rf-feature-cols, .rf-grid, .rf-cmp-3, .rf-cmp-4 { grid-template-columns: 1fr; }
  .rf-facts { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .rf-nutri { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (prefers-reduced-motion: reduce) { .rf-bar-track span { transition: none; } }

/* ---- review round 1: visual fixes ---- */
/* equal-height panels must not stretch their contents (the button grew to 85 px) */
.rf-panel > .form, .rf-panel > .block, .rf-panel > button { flex-grow: 0 !important; }
#rf-btn-photo, #rf-btn-search { height: 46px !important; min-height: 46px !important; flex: none !important; }
/* the photo and the heat map fill their frames the same way, nothing cropped */
#rf-photo img, #rf-cam img { width: 100% !important; height: 100% !important; object-fit: contain !important;
  background: var(--rf-paper); }
/* the request box must look like an input, not a caption */
#rf-query textarea, #rf-query input { border: 1px solid var(--rf-line) !important; border-radius: 10px !important;
  font-size: 16px !important; padding: 12px 14px !important; background: #fff !important; }
#rf-query textarea:focus, #rf-query input:focus { border-color: var(--rf-saffron) !important;
  box-shadow: 0 0 0 3px rgba(226,162,27,.18) !important; }
/* best match: no frame inside the panel frame, photo no taller than the text beside it */
.rf-feature { grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); border: 0; padding: 0; background: transparent; }
.rf-feature-media { aspect-ratio: 3 / 2; }
/* section headings and steps readable in a compressed recording */
#rf-recipes h4 { font-family: 'Young Serif', Georgia, serif !important; font-weight: 400 !important;
  font-size: 19px !important; color: var(--rf-ink) !important; }
.rf-steps, .rf-ingredients li { font-size: 15.5px; }
/* the four cards keep their rows aligned: titles clamp to two lines */
.rf-card h3 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  min-height: 2.4em; }
/* compare: labelled groups, a 1 px default outline that does not shift content, readable scores */
.rf-cmp-group { font-family: 'Young Serif', Georgia, serif !important; font-weight: 400 !important;
  font-size: 18px !important; color: var(--rf-ink) !important; margin: 4px 0 10px !important; }
.rf-cmp-group span { font-family: 'Figtree', sans-serif; font-size: 13px; color: var(--rf-muted); margin-left: 6px; }
.rf-cmp-card.is-default { border: 1px solid var(--rf-saffron); box-shadow: inset 0 0 0 1px var(--rf-saffron); }
.rf-cmp-card li small { font-size: 12.5px; color: var(--rf-muted); }
.rf-cmp-card ol { padding-left: 1.2em; }

@media (max-width: 760px) {
  .gradio-container { padding: 16px 12px 32px !important; }
  .rf-panel { padding: 16px !important; }
  .rf-step-sub { margin-left: 0; }
  .rf-ingredients li { grid-template-columns: 84px 1fr; }
  .rf-best { white-space: nowrap; }
  .rf-score-track { width: 40px; }
  .rf-feature { grid-template-columns: 1fr; }
}
/* ---- review round 2 ---- */
.rf-more { margin: 32px 0 0 !important; padding-top: 24px; border-top: 1px solid var(--rf-line); }

/* ---- review round 1: UX fixes ---- */
button.primary:hover { background: #EDB53C !important; color: var(--rf-ink) !important; }
.rf-best { color: #8A5C00; }
.rf-card summary { padding: 12px 0; min-height: 44px; }
#rf-btn-photo { background: transparent !important; border: 1px solid var(--rf-line) !important;
  color: var(--rf-ink) !important; font-weight: 600 !important; height: 38px !important; min-height: 38px !important;
  align-self: flex-start; width: auto !important; padding: 0 16px !important; }
#rf-query label > span { font-size: 14px !important; color: var(--rf-muted) !important; font-weight: 600 !important; }
@media (max-width: 760px) {
  .rf-card-media.rf-noimg { aspect-ratio: auto; height: auto; padding: 10px 0; }
  .rf-panel .row > div { min-width: 0 !important; flex: 1 1 0 !important; }   /* photo + heat map side by side */
  .rf-noimg span { font-size: 18px; }
  .rf-ingredients li { grid-template-columns: 92px 1fr; }
}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Figtree"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#F1F4EF",
    body_text_color="#17301F",
    block_background_fill="#FFFFFF",
    block_border_color="#D3DCD4",
    block_label_text_color="#5B6A60",
    input_background_fill="#FFFFFF",
    input_border_color="#D3DCD4",
    input_border_color_focus="#E2A21B",
    checkbox_label_background_fill_selected="#17301F",
    checkbox_label_text_color_selected="#FFFFFF",
    checkbox_label_border_color_selected="#17301F",
)

FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"

with gr.Blocks(title="Photo to Recipe", theme=THEME, css=CSS, js=FORCE_LIGHT) as demo:
    gr.HTML(
        '<header id="rf-head"><h1>Photo to recipe</h1>'
        "<p>Upload a photo of a dish and say what you would like to cook, in English or Turkish. "
        "The app recognises the dish and ranks five recipes that fit your request.</p></header>"
    )
    dish_state = gr.State("")

    # two equal panels side by side, each in the same order: input, model, button, result
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=420, elem_classes="rf-panel"):
            gr.HTML('<div class="rf-step"><span class="n">1</span><h2>Photo</h2></div>'
                    '<p class="rf-step-sub">Recognises pizza, sushi, ramen, falafel, pancakes and five more dishes.</p>')
            with gr.Row():
                img_in = gr.Image(type="pil", label="Dish photo", height=220, elem_id="rf-photo",
                                  sources=["upload", "clipboard"])
                cam_out = gr.Image(label="Heat map", height=220, interactive=False, elem_id="rf-cam")
            photo_model = gr.Radio(PHOTO_MODELS, value=PHOTO_MODELS[0], label="Image model",
                                   elem_classes="rf-models")
            btn_photo = gr.Button("Run again", variant="secondary", size="sm", elem_id="rf-btn-photo",
                                  visible=False)      # shown once there is a photo to run on
            pred_out = gr.HTML(PHOTO_EMPTY, elem_id="rf-pred")

        with gr.Column(scale=1, min_width=420, elem_classes="rf-panel"):
            gr.HTML('<div class="rf-step"><span class="n">2</span><h2>Request</h2></div>'
                    '<p class="rf-step-sub">Time and diet limits are read from your text. '
                    "If a photo was recognised, only that dish's recipes are searched.</p>")
            text_in = gr.Textbox(
                label="What would you like to cook?", lines=1, max_lines=1, elem_id="rf-query",
                placeholder="e.g. vegetarian, under 30 minutes  (press Enter to search)",
            )
            text_model = gr.Radio(TEXT_MODELS, value=TEXT_MODELS[0], label="Search model",
                                  elem_classes="rf-models")
            btn_text = gr.Button("Find recipes", variant="primary", elem_id="rf-btn-search")
            summary_out = gr.HTML(SUMMARY_EMPTY, elem_id="rf-summary")

    # full-width results: the best recipe in full, then four equal cards
    with gr.Column(elem_classes="rf-panel rf-results"):
        gr.HTML('<div class="rf-step"><span class="n">3</span><h2>Recipes</h2></div>'
                '<p class="rf-step-sub">The best match is shown in full. Open any other card for its full recipe.</p>')
        recipes_out = gr.HTML(RECIPES_EMPTY, elem_id="rf-recipes")

    # every model side by side, for the same photo and the same request
    with gr.Column(elem_classes="rf-panel rf-results rf-compare"):
        gr.HTML('<div class="rf-step"><span class="n">4</span><h2>Compare models</h2></div>'
                '<p class="rf-step-sub">The same photo and the same request through every model in the homework. '
                "The app itself uses the two INT8 models.</p>")
        cmp_photo_out = gr.HTML(CMP_PHOTO_EMPTY, elem_id="rf-cmp-photo")
        cmp_search_out = gr.HTML(CMP_SEARCH_EMPTY, elem_id="rf-cmp-search")

    # --- events -------------------------------------------------------------
    search_io = ([text_in, dish_state, text_model], [summary_out, recipes_out])

    def recognise(event):
        """Recognise the dish, then refresh the recipes for it (or clear them) and move on to the request."""
        (event(ui_predict, [img_in, photo_model], [cam_out, dish_state, pred_out, btn_photo])
         .then(ui_refresh_recipes, *search_io)
         .then(_noop, None, None, js=FOCUS_REQUEST)
         .then(ui_compare_photo, [img_in], cmp_photo_out)
         .then(ui_compare_search, [text_in, dish_state], cmp_search_out))

    def find(event):
        """Search, bring the results into view, then fill in the comparison."""
        (event(ui_search, *search_io)
         .then(_noop, None, None, js=SCROLL_TO_RECIPES)
         .then(ui_compare_search, [text_in, dish_state], cmp_search_out))

    # a photo is recognised as soon as it arrives: upload, paste or the button
    recognise(img_in.upload)
    recognise(btn_photo.click)
    recognise(photo_model.change)          # switching the image model re-runs the photo
    (img_in.clear(ui_clear, None, [cam_out, dish_state, pred_out, btn_photo])
     .then(ui_reset_recipes, None, [summary_out, recipes_out])
     .then(lambda: (CMP_PHOTO_EMPTY, CMP_SEARCH_EMPTY), None, [cmp_photo_out, cmp_search_out]))

    # a search runs from the button or the Enter key
    find(btn_text.click)
    find(text_in.submit)

if __name__ == "__main__":
    print("loading models…")
    get_vgg_int8()
    get_vgg()
    get_cnn()
    # one warm-up pass per model so the first real photo is not slowed by CUDA / ONNX start-up
    _blank = Image.new("RGB", (256, 256), (200, 160, 120))
    for _m in PHOTO_MODELS:
        classify(_blank, _m)
    gradcam(_blank, 0)
    for name in TEXT_MODELS:
        get_retriever(name)
    print("ready: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, show_api=False)
