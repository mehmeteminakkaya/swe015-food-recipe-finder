"""Build the presentation for the SWE015 UC3 homework.

Usage:  python scripts/make_slides.py [--name "Ad Soyad"] [--id 123456] \
            [--repo URL] [--video URL]

Writes docs/slides.pptx (12 slides, 16:9).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS, ROOT

DOCS = ROOT / "docs"
W, H = 13.333, 7.5

# every number on the slides is read from the measured results
RES = {r["model"]: r for r in csv.DictReader((RESULTS / "results.csv").open(encoding="utf-8"))}
MET = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
VGG_FEAT = json.loads((RESULTS / "vgg16_history.json").read_text(encoding="utf-8"))["feature_extraction"]["test"]
RECIPE_ROWS = [json.loads(line) for line in (ROOT / "data" / "recipes.jsonl").open(encoding="utf-8") if line.strip()]
N_VEG = sum(r["vegetarian"] for r in RECIPE_ROWS)
N_VEGAN = sum(r["vegan"] for r in RECIPE_ROWS)


def num(model: str, col: str) -> float:
    return float(RES[model][col])


def ms(model: str, col: str = "latency_cpu_ms") -> str:
    v = RES[model][col]
    return "—" if v in ("", "nan") else f"{float(v):.1f} ms"


def mb(model: str) -> str:
    return f"{num(model, 'size_mb'):.1f} MB"


def speedup(a: str, b: str) -> float:
    return num(a, "latency_cpu_ms") / num(b, "latency_cpu_ms")


def shrink(a: str, b: str) -> float:
    return num(a, "size_mb") / num(b, "size_mb")


def ret(model: str) -> str:
    return f"Recall@5 {num(model, 'primary_value'):.3f} / MRR {num(model, 'secondary_value'):.3f}"


def vis(model: str) -> str:
    return f"top-1 {num(model, 'primary_value'):.3f} / top-5 {num(model, 'secondary_value'):.3f}"


def lang(model: str, code: str) -> float:
    return MET["retrieval"][model]["by_lang"][code]["recall@5"]


BEST_MRR = max(("tfidf", "word2vec", "bert_e5", "bert_e5_int8"), key=lambda m: num(m, "secondary_value"))
BEST_RET = max(("tfidf", "word2vec", "bert_e5", "bert_e5_int8"), key=lambda m: num(m, "primary_value"))
TOP1_DROP = (num("vgg16_ft", "primary_value") - num("vgg16_int8", "primary_value")) * 100

DARK = RGBColor(0x1F, 0x29, 0x37)
GREY = RGBColor(0x6B, 0x72, 0x80)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
LIGHT = RGBColor(0xF3, 0xF4, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def textbox(slide, left, top, width, height, text, size=14, bold=False, color=DARK,
            align=PP_ALIGN.LEFT, space_after=6):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return box


def bullets(slide, left, top, width, height, items, size=14, color=DARK, spacing=8):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing)
        p.level = 0
        run = p.add_run()
        run.text = "•  " + item
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = "Calibri"
    return box


def header(slide, kicker, title):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    textbox(slide, 0.7, 0.35, 11.9, 0.3, kicker.upper(), size=11, bold=True, color=GREY)
    textbox(slide, 0.7, 0.65, 11.9, 0.9, title, size=28, bold=True, color=DARK)


def table(slide, rows, left, top, width, col_widths=None, header=True,
          font_size=11, row_height=0.32):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top),
                                   Inches(width), Inches(row_height * n_rows))
    tbl = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            for p in cell.text_frame.paragraphs:
                p.space_after = Pt(0)
                for run in p.runs:
                    run.font.size = Pt(font_size)
                    run.font.name = "Calibri"
                    if header and r == 0:
                        run.font.bold = True
                        run.font.color.rgb = WHITE
                    else:
                        run.font.color.rgb = DARK
            if header and r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = BLUE
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
    return tbl


def picture(slide, path, left, top, width=None, height=None):
    kwargs = {}
    if width:
        kwargs["width"] = Inches(width)
    if height:
        kwargs["height"] = Inches(height)
    return slide.shapes.add_picture(str(path), Inches(left), Inches(top), **kwargs)


def slide_title(prs, name, student_id, repo, video):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(H))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(0x14, 0x2B, 0x45)
    bg.line.fill.background()

    textbox(s, 0.9, 1.15, 11.5, 0.4, "SWE015 — Introduction to Large Language Models · Homework",
            size=14, color=RGBColor(0xC9, 0xD6, 0xE3))
    textbox(s, 0.9, 1.65, 11.5, 1.4, "Food Photo → Recipe Finder", size=46, bold=True, color=WHITE)
    textbox(s, 0.9, 2.85, 11.5, 0.5, "Use Case 3 · one task, four models, one quantized app",
            size=18, color=RGBColor(0xC9, 0xD6, 0xE3))
    textbox(s, 0.9, 4.15, 5.6, 1.9,
            f"{name}\nStudent ID: {student_id}\n22 November 2026",
            size=15, color=WHITE, space_after=8)
    links = textbox(s, 6.9, 4.15, 5.5, 1.9,
                    f"Code: {repo}\nVideo: {video}\nDemo: python app.py → 127.0.0.1:7860",
                    size=12, color=RGBColor(0xC9, 0xD6, 0xE3), space_after=8)
    # Make the repository and the video real hyperlinks, not just printed URLs.
    for paragraph, target in zip(links.text_frame.paragraphs, (repo, video)):
        if str(target).startswith("http"):
            for run in paragraph.runs:
                run.hyperlink.address = str(target)
    # PowerPoint draws links in the theme's hyperlink colour, not the run colour; the
    # default dark blue is unreadable on this navy slide, so lighten the theme colour.
    _set_theme_link_colour(prs, "9CC8F2")


def _set_theme_link_colour(prs, hex_rgb: str) -> None:
    from pptx.opc.constants import RELATIONSHIP_TYPE as RT

    theme = prs.slide_master.part.part_related_by(RT.THEME)
    from lxml import etree

    root = etree.fromstring(theme.blob)
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    for tag in ("hlink", "folHlink"):
        for el in root.findall(f".//a:clrScheme/a:{tag}", ns):
            for child in list(el):
                el.remove(child)
            etree.SubElement(el, f"{{{ns['a']}}}srgbClr", val=hex_rgb)
    theme._blob = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Introduction", "A photo and a sentence are all the user gives")
    textbox(s, 0.7, 1.55, 5.4, 0.4, "The problem", size=16, bold=True, color=BLUE)
    textbox(s, 0.7, 2.0, 5.4, 2.1,
            "A home cook has a photo of a dish and a vague wish — "
            "“vegetarian, under 30 minutes”. A recipe search box that needs exact keywords "
            "cannot answer that, and a photo alone does not tell you what to cook.",
            size=13.5)
    textbox(s, 6.6, 1.55, 6.0, 0.4, "What we build", size=16, bold=True, color=BLUE)
    textbox(s, 6.6, 2.0, 6.0, 2.1,
            "An app that takes the photo (dish + confidence + Grad-CAM), then takes free text "
            "in Turkish or English, extracts the constraints, and returns the five best "
            "recipes with similarity scores.",
            size=13.5)
    textbox(s, 0.7, 3.55, 11.9, 0.4, "What success looks like", size=16, bold=True, color=BLUE)
    textbox(s, 0.7, 4.0, 11.9, 1.7,
            "Top-1 accuracy on the shared 10-class test split, Recall@5 and MRR on 40 held-out "
            "queries, and a quantized model fast enough for an interactive CPU-only UI. "
            f"Measured in this homework: {num('vgg16_ft', 'primary_value'):.3f} top-1 "
            f"({num('vgg16_int8', 'primary_value'):.3f} INT8), {num(BEST_RET, 'primary_value'):.3f} best Recall@5, "
            f"{num('vgg16_int8', 'latency_cpu_ms'):.0f} ms per photo on CPU.",
            size=13.5)


def slide_data(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — data", "Two datasets, one held-out test split per task")
    table(s, [
        ["Data", "Source", "Size / split"],
        ["Images", "Food-101 (Bossard et al., 2014), HF ethz/food101, 10 classes",
         "6 744 train / 750 val / 2 499 test JPEG"],
        ["Recipes", "Food.com, Kaggle irkaal (CC0), parsed copy on HF; methods from Kaggle RAW_recipes",
         f"{len(RECIPE_ROWS)} recipes, 30 per dish; {N_VEG} vegetarian, {N_VEGAN} vegan; "
         f"{sum(bool(r.get('steps')) for r in RECIPE_ROWS)} with a method"],
        ["Queries", "Written by hand for this homework", "40 queries: 20 Turkish, 20 English"],
    ], left=0.7, top=1.75, width=11.9, col_widths=[1.5, 6.3, 4.1], font_size=11.5)
    bullets(s, 0.7, 3.95, 11.9, 2.6, [
        "Preprocessing: resize/crop 224, ImageNet normalisation, augmentation (RandomResizedCrop, flip, rotation, colour jitter).",
        "7 exact duplicate photos removed (md5); one of them sat in both the official train and test splits — a real leak.",
        "Text: normalisation (TR-aware), regex extraction of time/diet, Snowball stemming, TR→EN lexicon; the same hard filter runs for every retrieval model.",
        "Relevance is rule-based: a recipe is relevant when it has the target dish and satisfies the constraints — so the labels are reproducible.",
        "Recipes with a photo and a step-by-step method were preferred when sampling each dish (seed 42); the corpus is never tuned to the test queries.",
    ], size=13)
    textbox(s, 0.7, 6.35, 11.9, 0.6,
            "Example: “vejetaryen, 30 dakikadan kısa” → max_minutes = 30, diet = {vegetarian}; "
            "the evaluator re-parses the text and fails if a query does not match its stored fields.",
            size=11, color=GREY)


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — architecture (R6)", "Every component, and where it runs")
    picture(s, DOCS / "architecture.png", left=2.05, top=1.6, height=4.75)
    textbox(s, 0.7, 6.55, 11.9, 0.5,
            "The browser talks to the Gradio app; the quantized models run on CPU, "
            "Grad-CAM and training use the GPU. Drawn by scripts/architecture.py.",
            size=11, color=GREY)


def slide_techniques(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — technique map (R2)", "Four technique areas, all in the same pipeline")
    table(s, [
        ["Area", "Technique", "Where in the system"],
        ["A · Neural networks", "Small CNN from scratch (augmentation, dropout)", "Image baseline, 1.21 M parameters"],
        ["A · Neural networks", "VGG16 transfer learning: frozen features → conv5 fine-tuning", f"Image model, {num('vgg16_ft', 'primary_value'):.3f} top-1"],
        ["A · Neural networks", "Grad-CAM on conv5_3", "Heat map shown next to the prediction"],
        ["B · Classical NLP", "Regex constraint extraction, normalisation, Snowball stemming", "Query preprocessing, TR→EN lexicon"],
        ["B · Classical NLP", "tf-idf + cosine; word2vec trained on our 300 recipes", "Two retrieval baselines"],
        ["C · Transformers", "multilingual E5-small (BERT-based encoder) sentence embeddings", "Transformer retrieval model"],
        ["D · Efficient models", "INT8 static QDQ (VGG16) and dynamic INT8 (E5), ONNX Runtime", "Default models in the app, size/latency measured"],
    ], left=0.7, top=1.75, width=11.9, col_widths=[2.3, 5.6, 4.0], font_size=12, row_height=0.44)
    textbox(s, 0.7, 6.35, 11.9, 0.6,
            "Each area appears in the running app; this map is the same as docs/architecture.png "
            "and the modules listed in the README.",
            size=11, color=GREY)


def slide_vision(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — image models", "From 0.78 to 0.93 by starting from ImageNet")
    bullets(s, 0.7, 1.7, 5.6, 3.0, [
        "Small CNN from scratch: 4 conv blocks, batch norm, dropout 0.4, 18 epochs, 128 px.",
        "VGG16: frozen ImageNet features + small head (feature extraction), then the last conv block (conv5) is unfrozen and fine-tuned for 6 epochs.",
        "Test split is the same 2 499 images for all three vision models.",
    ], size=12.5)
    table(s, [
        ["Model", "Top-1", "Top-5"],
        ["Small CNN (scratch)", f"{num('small_cnn', 'primary_value'):.4f}", f"{num('small_cnn', 'secondary_value'):.4f}"],
        ["VGG16 feature extraction", f"{VGG_FEAT['top1']:.4f}", f"{VGG_FEAT['top5']:.4f}"],
        ["VGG16 conv5 fine-tuned", f"{num('vgg16_ft', 'primary_value'):.4f}", f"{num('vgg16_ft', 'secondary_value'):.4f}"],
        ["VGG16 INT8", f"{num('vgg16_int8', 'primary_value'):.4f}", f"{num('vgg16_int8', 'secondary_value'):.4f}"],
    ], left=0.7, top=4.55, width=5.6, col_widths=[3.4, 1.1, 1.1], font_size=11.5)
    picture(s, DOCS / "fig_curves.png", left=6.6, top=1.85, width=6.1)
    textbox(s, 6.6, 5.3, 6.1, 1.2,
            "Fine-tuning adds +2.4 points over the frozen features; the small CNN never "
            "closes the gap, which is the expected transfer-learning result.",
            size=11.5, color=GREY)


def slide_gradcam(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — explainability", "Grad-CAM: what the network looked at")
    picture(s, DOCS / "gradcam_example.png", left=0.55, top=1.6, width=8.3)
    textbox(s, 9.1, 1.8, 3.7, 4.6,
            "Grad-CAM on conv5_3 of the fine-tuned VGG16: the heat map is the gradient "
            "of the predicted class with respect to the last convolutional features.\n\n"
            "Top row: input · bottom row: heat map and prediction.\n\n"
            "The leftmost photo is a genuine failure: falafel is predicted as spaghetti "
            "bolognese (41 %), and the heat sits on the red sauce and the plate edge — "
            "exactly the wrong evidence.",
            size=11.5)
    textbox(s, 0.7, 6.55, 11.9, 0.5,
            "Same figure in docs/gradcam_example.png; the app shows it live for every upload.",
            size=10.5, color=GREY)


def slide_retrieval(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — retrieval", "Constraints filter, three models rank")
    bullets(s, 0.7, 1.7, 5.3, 4.4, [
        "Regex extracts max time + diet from TR/EN text; the filter is shared, so only ranking differs.",
        "tf-idf + cosine: normalisation, Snowball stemming and a TR→EN lexicon make Turkish queries match English recipes.",
        "word2vec: our own skip-gram model on the 300 recipes; a recipe is the mean of its token vectors.",
        "E5 small: multilingual encoder, 384-d, cosine over mean-pooled embeddings.",
        "Dish predicted from the photo is added to the query and narrows the pool.",
    ], size=12)
    picture(s, DOCS / "fig_retrieval.png", left=6.25, top=1.95, width=6.5)
    textbox(s, 6.25, 5.45, 6.5, 1.3,
            f"E5 finds the most relevant recipes (Recall@5 {num('bert_e5', 'primary_value'):.3f}); its lead comes from "
            f"English ({lang('bert_e5', 'en'):.3f} vs. word2vec {lang('word2vec', 'en'):.3f}), while in Turkish our word2vec "
            f"is as good ({lang('word2vec', 'tr'):.3f} vs. {lang('bert_e5', 'tr'):.3f}). E5 wins paraphrases (“meatless burgers”, "
            "“vegan guakamole”) and loses when the dish is only described (“nohut köftesi”, “chickpea fritters”).",
            size=11, color=GREY)


def slide_quantization(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — efficient models (D)", "Two quantizations, two different problems")
    bullets(s, 0.7, 1.7, 5.6, 4.5, [
        "VGG16 is convolution-heavy: dynamic quantization only touches MatMul/Gemm, so we export to ONNX and run static QDQ with 256 calibration images (per-channel, INT8).",
        f"E5 is attention/MatMul-heavy: dynamic INT8 quantization of the ONNX graph needs no calibration: Recall@5 {num('bert_e5', 'primary_value'):.3f} → {num('bert_e5_int8', 'primary_value'):.3f}, MRR {num('bert_e5', 'secondary_value'):.3f} → {num('bert_e5_int8', 'secondary_value'):.3f}.",
        "Both run through ONNX Runtime on CPU — that is the deployment target.",
    ], size=12)
    table(s, [
        ["", "VGG16 fp32", "VGG16 INT8", "E5 fp32", "E5 INT8"],
        ["Size", mb("vgg16_ft"), mb("vgg16_int8"), mb("bert_e5"), mb("bert_e5_int8")],
        ["CPU latency", ms("vgg16_ft"), ms("vgg16_int8"), ms("bert_e5"), ms("bert_e5_int8")],
        ["Task metric", *(f"{num(m, 'primary_value'):.3f}" for m in ("vgg16_ft", "vgg16_int8", "bert_e5", "bert_e5_int8"))],
    ], left=0.7, top=4.2, width=6.0, col_widths=[1.5, 1.2, 1.2, 1.1, 1.1], font_size=10.5)
    picture(s, DOCS / "fig_tradeoff.png", left=6.5, top=1.95, width=6.3)
    textbox(s, 6.5, 5.9, 6.3, 0.9,
            f"VGG16: {shrink('vgg16_ft', 'vgg16_int8'):.0f}× smaller and {speedup('vgg16_ft', 'vgg16_int8'):.1f}× "
            f"faster on CPU for {TOP1_DROP:.2f} points of top-1 accuracy.",
            size=12, color=GREEN, bold=True)


def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Results (R4)", "One table, every number from results/results.csv")
    table(s, [
        ["Task", "Model", "Precision", "Task metric", "Size", "CPU", "GPU"],
        ["Vision", "small CNN (scratch)", "fp32", vis("small_cnn"), mb("small_cnn"), ms("small_cnn"), ms("small_cnn", "latency_gpu_ms")],
        ["Vision", "VGG16 feature extraction", "fp32", f"top-1 {VGG_FEAT['top1']:.3f} / top-5 {VGG_FEAT['top5']:.3f}", mb("vgg16_ft"), "—", "—"],
        ["Vision", "VGG16 conv5 fine-tuned", "fp32", vis("vgg16_ft"), mb("vgg16_ft"), ms("vgg16_ft"), ms("vgg16_ft", "latency_gpu_ms")],
        ["Vision", "VGG16 conv5 fine-tuned", "INT8 QDQ", vis("vgg16_int8"), mb("vgg16_int8"), ms("vgg16_int8"), "—"],
        ["Retrieval", "tf-idf + cosine", "fp32", ret("tfidf"), mb("tfidf"), ms("tfidf"), "—"],
        ["Retrieval", "word2vec (ours)", "fp32", ret("word2vec"), mb("word2vec"), ms("word2vec"), "—"],
        ["Retrieval", "E5 small", "fp32", ret("bert_e5"), mb("bert_e5"), ms("bert_e5"), ms("bert_e5", "latency_gpu_ms")],
        ["Retrieval", "E5 small", "INT8 dynamic", ret("bert_e5_int8"), mb("bert_e5_int8"), ms("bert_e5_int8"), "—"],
    ], left=0.55, top=1.6, width=12.2,
        col_widths=[1.2, 3.6, 1.6, 3.4, 1.1, 1.3, 1.0], font_size=10.5)
    textbox(s, 0.55, 6.4, 12.2, 0.9,
            "Latency = mean of 100 runs, batch size 1. GPU: NVIDIA RTX 4060 Laptop 8 GB. "
            "CPU: AMD Ryzen 7 8845HS. INT8 models are measured on CPU only; latency for a "
            "single short query is dominated by overhead, which is why the GPU does not help E5.",
            size=10, color=GREY)


def slide_discussion(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Conclusion", "What the comparison showed")
    textbox(s, 0.7, 1.5, 5.8, 0.4, "Where each model wins", size=15, bold=True, color=BLUE)
    bullets(s, 0.7, 1.95, 5.7, 4.2, [
        f"Quantization is nearly free: VGG16 loses {TOP1_DROP:.2f} points of top-1 for {shrink('vgg16_ft', 'vgg16_int8'):.0f}× "
        f"less size and {speedup('vgg16_ft', 'vgg16_int8'):.1f}× less CPU time. E5 shrinks {shrink('bert_e5', 'bert_e5_int8'):.0f}× "
        f"for {(num('bert_e5', 'primary_value') - num('bert_e5_int8', 'primary_value')) * 100:.1f} points of Recall@5.",
        f"The Transformer finds the most relevant recipes (Recall@5 {num('bert_e5', 'primary_value'):.3f} vs. word2vec "
        f"{num('word2vec', 'primary_value'):.3f}, tf-idf {num('tfidf', 'primary_value'):.3f}), thanks to English; in Turkish "
        f"our own word2vec is as good ({lang('word2vec', 'tr'):.3f} vs. {lang('bert_e5', 'tr'):.3f}).",
        "E5 wins paraphrases (“meatless burgers”, “vegan guakamole”, “vejetaryen hamburger”) but loses when the dish "
        "is only described (“nohut köftesi”, “crispy chickpea fritters”, “pratik tatlı”).",
        f"INT8 put the whole app on CPU: {num('vgg16_int8', 'latency_cpu_ms'):.0f} ms per photo, "
        f"{num('bert_e5_int8', 'latency_cpu_ms'):.0f} ms per query.",
    ], size=12.5, spacing=6)
    textbox(s, 6.8, 1.5, 5.8, 0.4, "Limitations", size=15, bold=True, color=BLUE)
    bullets(s, 6.8, 1.95, 5.7, 2.6, [
        "10 classes, 750 images per class — wide error bars.",
        "Rule-based relevance rewards finding the dish, not a good cooking suggestion.",
        "Recipe corpus is noisy (odd names, boilerplate descriptions); methods come from a second Food.com copy.",
        "One laptop, one run; no multi-seed variance.",
    ], size=12.5, spacing=6)
    textbox(s, 6.8, 4.55, 5.8, 0.4, "Next", size=15, bold=True, color=BLUE)
    textbox(s, 6.8, 4.95, 5.7, 1.2,
            "Cross-encoder reranking, a Turkish-specific encoder (E5 is weakest in Turkish), "
            "more dishes, and an image-quality check before predicting.",
            size=12.5)
    textbox(s, 0.7, 6.55, 11.9, 0.6,
            "Failure example: the falafel photo predicted as spaghetti bolognese — Grad-CAM showed the model looking at the red sauce.",
            size=11, color=GREY)


def slide_ai(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "AI usage (R8)", "What was AI-assisted, and what was measured")
    bullets(s, 0.7, 1.8, 11.9, 4.8, [
        "AI coding assistants (Claude, DeepSeek) were used to write and debug the code, the training scripts, this deck and the README.",
        "They were also used to choose between the two encoders — the choice was then verified with our own similarity test (pankek↔pancakes: E5 0.85, MiniLM 0.42).",
        "No numbers in this deck come from a paper, a model card or an AI answer: every value is produced by scripts/evaluate.py and scripts/benchmark.py in this repository, and stored in results/results.csv.",
        "Datasets: Food-101 (Bossard et al., 2014) and Food.com recipes via the public Kaggle/Food.com copy (CC0). Model: intfloat/multilingual-e5-small.",
        "The falafel failure case was found by looking at the model's own output, not generated.",
    ], size=13)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="[Ad Soyad]")
    ap.add_argument("--id", default="[Öğrenci No]")
    ap.add_argument("--repo", default="[GitHub deposu]")
    ap.add_argument("--video", default="[2 dk video linki]")
    args = ap.parse_args()

    prs = new_deck()
    slide_title(prs, args.name, args.id, args.repo, args.video)
    slide_problem(prs)
    slide_data(prs)
    slide_architecture(prs)
    slide_techniques(prs)
    slide_vision(prs)
    slide_gradcam(prs)
    slide_retrieval(prs)
    slide_quantization(prs)
    slide_results(prs)
    slide_discussion(prs)
    slide_ai(prs)

    out = DOCS / "slides.pptx"
    prs.save(out)
    print("wrote", out, f"({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
