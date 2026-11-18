"""Gradio demo: photo -> dish + Grad-CAM, free text -> top-5 recipes.

Run:  python app.py
The default models are the quantized ones (INT8 VGG16 and INT8 E5 encoder);
the float models can be selected in the UI for comparison.
"""
from __future__ import annotations

import time
from pathlib import Path

import src.config  # noqa: F401  -- sets HF/TORCH cache dirs before huggingface_hub loads

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

PHOTO_MODELS = ["VGG16 INT8 (varsayılan)", "VGG16 fp32", "Küçük CNN (sıfırdan)"]
TEXT_MODELS = ["E5 INT8 (varsayılan)", "E5 fp32", "tf-idf + cosine", "word2vec (kendi eğitimimiz)"]

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
    return show_cam_on_image(rgb, heat, use_rgb=True)


def predict_photo(img: Image.Image, photo_model: str):
    if img is None:
        return None, "", "", "Bir fotoğraf yükleyin."
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
    top = int(prob.argmax())
    confidence = float(prob[top])
    text = "\n".join(
        f"{cls[i].replace('_', ' ')}: {float(prob[i]):.1%}"
        for i in prob.topk(5).indices.tolist()
    )
    ms = (time.perf_counter() - t0) * 1000
    cam_img = gradcam(img.convert("RGB"), top)
    warn = "" if confidence >= 0.30 else " · ⚠️ düşük güven: net bir yemek görünmüyor olabilir"
    return (cam_img, cls[top].replace("_", " "), text,
            f"{photo_model} · {ms:.0f} ms · güven {confidence:.1%}{warn}")


def _pool(dish_key: str, max_minutes: int | None, diet: list[str]) -> list:
    """Recipes obeying the constraints (and the predicted dish, when there is one)."""
    q = {"target_dish": dish_key, "max_minutes": max_minutes, "diet": diet}
    pool = [r for r in RECIPES if r["dish"] == dish_key] if dish_key else RECIPES
    return candidate_recipes(q, pool)


def build_pool(dish_key: str, c) -> tuple[list, str]:
    """Tightest pool that still has at least five recipes; reports what was relaxed."""
    steps = [
        (c.max_minutes, sorted(c.diet), ""),
        (None, sorted(c.diet), "süre filtresi gevşetildi"),
        (None, [], "süre ve diyet filtresi gevşetildi"),
    ]
    for max_minutes, diet, note in steps:
        pool = _pool(dish_key, max_minutes, diet)
        if len(pool) >= 5:
            return pool, note
    return RECIPES, "yeterli tarif yok; tüm tarifler arandı"


def search(text: str, photo_dish: str, text_model: str):
    if not text.strip() and not photo_dish:
        return [], "Sorgu yazın veya fotoğraf yükleyin."
    query = text.strip()
    if photo_dish and photo_dish.lower() not in query.lower():
        query = f"{query} {photo_dish}".strip()

    t0 = time.perf_counter()
    c = extract_constraints(query)
    dish_key = photo_dish.replace(" ", "_") if photo_dish else ""
    pool, note = build_pool(dish_key, c)
    r = get_retriever(text_model)
    hits = r.search(query, pool, k=5)
    ms = (time.perf_counter() - t0) * 1000

    rows = []
    for rec, score in hits:
        badges = []
        if rec["vegetarian"]:
            badges.append("vejetaryen")
        if rec["vegan"]:
            badges.append("vegan")
        rows.append([
            rec["name"],
            f"{rec['minutes']} dk",
            ", ".join(badges) or "—",
            f"{score:.3f}",
        ])
    info = f"{text_model} · {ms:.0f} ms · {len(pool)} aday tarif"
    if c.max_minutes or c.diet:
        info += f" · istenen: ≤{c.max_minutes or '∞'} dk, diyet: {', '.join(sorted(c.diet)) or 'yok'}"
    if note:
        info += f" · {note}"
    return rows, info


with gr.Blocks(title="Food Photo → Recipe Finder") as demo:
    gr.Markdown(
        "# 🍽️ Food Photo → Recipe Finder\n"
        "Bir yemek fotoğrafı yükleyin; model yemeği tahmin etsin ve Grad-CAM ile nereye baktığını gösterin. "
        "Sonra serbest metinle isteğinizi yazın (TR/EN): *\"vejetaryen, 30 dakikadan kısa\"* gibi."
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="pil", label="Yemek fotoğrafı")
            photo_model = gr.Dropdown(PHOTO_MODELS, value=PHOTO_MODELS[0], label="Görüntü modeli")
            btn_photo = gr.Button("Yemeği tahmin et", variant="primary")
            cam_out = gr.Image(label="Grad-CAM (VGG16 conv5_3)", height=260)
            dish_out = gr.Textbox(label="Tahmin", interactive=False)
            top5_out = gr.Textbox(label="En olası 5 sınıf", lines=5, interactive=False)
            photo_info = gr.Markdown()
        with gr.Column(scale=2):
            text_in = gr.Textbox(
                label="Ne pişirmek istiyorsunuz?",
                placeholder="ör. vejetaryen, 30 dakikadan kısa / vegan ramen under 30 minutes",
                lines=2,
            )
            text_model = gr.Dropdown(TEXT_MODELS, value=TEXT_MODELS[0], label="Arama modeli")
            btn_text = gr.Button("Tarif ara", variant="primary")
            table = gr.Dataframe(
                headers=["Tarif", "Süre", "Diyet", "Benzerlik"],
                label="En iyi 5 tarif", interactive=False, wrap=True,
            )
            search_info = gr.Markdown()

    btn_photo.click(predict_photo, [img_in, photo_model], [cam_out, dish_out, top5_out, photo_info])
    btn_text.click(search, [text_in, dish_out, text_model], [table, search_info])
    text_in.submit(search, [text_in, dish_out, text_model], [table, search_info])

if __name__ == "__main__":
    print("modeller yükleniyor…")
    get_vgg_int8()
    get_vgg()
    for name in TEXT_MODELS:
        get_retriever(name)
    print("hazır: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, show_api=False)
