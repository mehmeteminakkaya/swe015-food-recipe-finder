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
    textbox(s, 6.9, 4.15, 5.5, 1.9,
            f"Code: {repo}\nVideo: {video}\nDemo: python app.py → 127.0.0.1:7860",
            size=12, color=RGBColor(0xC9, 0xD6, 0xE3), space_after=8)


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
            "Measured in this homework: 0.926 top-1 (0.924 INT8), 0.915 best Recall@5, "
            "17 ms per photo on CPU.",
            size=13.5)


def slide_data(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — data", "Two datasets, one held-out test split per task")
    table(s, [
        ["Data", "Source", "Size / split"],
        ["Images", "Food-101 (Bossard et al., 2014), HF ethz/food101, 10 classes",
         "6 744 train / 750 val / 2 499 test JPEG"],
        ["Recipes", "Food.com, Kaggle irkaal (CC0), parsed copy on HF",
         "300 recipes, 30 per dish; 171 vegetarian, 65 vegan"],
        ["Queries", "Written by hand for this homework", "40 queries: 20 Turkish, 20 English"],
    ], left=0.7, top=1.75, width=11.9, col_widths=[1.5, 6.3, 4.1], font_size=11.5)
    bullets(s, 0.7, 3.95, 11.9, 2.6, [
        "Preprocessing: resize/crop 224, ImageNet normalisation, augmentation (RandomResizedCrop, flip, rotation, colour jitter).",
        "7 exact duplicate photos removed (md5); one of them sat in both the official train and test splits — a real leak.",
        "Text: normalisation (TR-aware), regex extraction of time/diet, Snowball stemming, TR→EN lexicon; the same hard filter runs for every retrieval model.",
        "Relevance is rule-based: a recipe is relevant when it has the target dish and satisfies the constraints — so the labels are reproducible.",
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
        ["A · Neural networks", "VGG16 transfer learning: frozen features → conv5 fine-tuning", "Image model, 0.926 top-1"],
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
        ["Small CNN (scratch)", "0.7835", "0.9836"],
        ["VGG16 feature extraction", "0.9020", "0.9920"],
        ["VGG16 conv5 fine-tuned", "0.9260", "0.9952"],
        ["VGG16 INT8", "0.9236", "0.9952"],
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
            "The Transformer does not win here: with 300 recipes and dish-named queries the "
            "lexical models are hard to beat, while E5 wins the descriptive queries "
            "(chickpea fritters, “pratik tatlı”, sushi rolls).",
            size=11, color=GREY)


def slide_quantization(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Development — efficient models (D)", "Two quantizations, two different problems")
    bullets(s, 0.7, 1.7, 5.6, 4.5, [
        "VGG16 is convolution-heavy: dynamic quantization only touches MatMul/Gemm, so we export to ONNX and run static QDQ with 256 calibration images (per-channel, INT8).",
        "E5 is attention/MatMul-heavy: dynamic INT8 quantization of the ONNX graph needs no calibration and loses little quality.",
        "Both run through ONNX Runtime on CPU — that is the deployment target.",
    ], size=12)
    table(s, [
        ["", "VGG16 fp32", "VGG16 INT8", "E5 fp32", "E5 INT8"],
        ["Size", "110.3 MB", "27.7 MB", "471.1 MB", "119.2 MB"],
        ["CPU latency", "102.3 ms", "16.9 ms", "24.8 ms", "6.7 ms"],
        ["Task metric", "0.926", "0.924", "0.868", "0.852"],
    ], left=0.7, top=4.2, width=6.0, col_widths=[1.5, 1.2, 1.2, 1.1, 1.1], font_size=10.5)
    picture(s, DOCS / "fig_tradeoff.png", left=6.5, top=1.95, width=6.3)
    textbox(s, 6.5, 5.9, 6.3, 0.9,
            "4× smaller and 6× faster on CPU for 0.24 points of top-1 accuracy.",
            size=12, color=GREEN, bold=True)


def slide_results(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "Results (R4)", "One table, every number from results/results.csv")
    table(s, [
        ["Task", "Model", "Precision", "Task metric", "Size", "CPU", "GPU"],
        ["Vision", "small CNN (scratch)", "fp32", "top-1 0.784 / top-5 0.984", "4.9 MB", "12.8 ms", "2.8 ms"],
        ["Vision", "VGG16 feature extraction", "fp32", "top-1 0.902 / top-5 0.992", "110.3 MB", "—", "—"],
        ["Vision", "VGG16 conv5 fine-tuned", "fp32", "top-1 0.926 / top-5 0.995", "110.3 MB", "102.3 ms", "5.5 ms"],
        ["Vision", "VGG16 conv5 fine-tuned", "INT8 QDQ", "top-1 0.924 / top-5 0.995", "27.7 MB", "16.9 ms", "—"],
        ["Retrieval", "tf-idf + cosine", "fp32", "Recall@5 0.883 / MRR 0.946", "5.3 MB", "3.5 ms", "—"],
        ["Retrieval", "word2vec (ours)", "fp32", "Recall@5 0.915 / MRR 0.971", "1.0 MB", "0.6 ms", "—"],
        ["Retrieval", "E5 small", "fp32", "Recall@5 0.868 / MRR 0.907", "471.1 MB", "24.8 ms", "21.4 ms"],
        ["Retrieval", "E5 small", "INT8 dynamic", "Recall@5 0.852 / MRR 0.914", "119.2 MB", "6.7 ms", "—"],
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
        "Quantization is nearly free: VGG16 loses 0.24 points of top-1 for 4× less size and 6× less CPU time; E5 keeps its MRR while shrinking 4×.",
        "On 300 recipes, lexical retrieval is strong: word2vec 0.915 Recall@5, tf-idf 0.883.",
        "E5 wins descriptive queries (chickpea fritters, “pratik tatlı”, sushi rolls) but loses Turkish loanwords such as suşi, pankek.",
        "INT8 put the whole app on CPU: 17 ms per photo, 7 ms per query.",
    ], size=12.5, spacing=6)
    textbox(s, 6.8, 1.5, 5.8, 0.4, "Limitations", size=15, bold=True, color=BLUE)
    bullets(s, 6.8, 1.95, 5.7, 2.6, [
        "10 classes, 750 images per class — wide error bars.",
        "Rule-based relevance rewards finding the dish, not a good cooking suggestion.",
        "Recipe corpus is noisy (“Shih Tzu Sushi”) and has no steps column.",
        "One laptop, one run; no multi-seed variance.",
    ], size=12.5, spacing=6)
    textbox(s, 6.8, 4.55, 5.8, 0.4, "Next", size=15, bold=True, color=BLUE)
    textbox(s, 6.8, 4.95, 5.7, 1.2,
            "Cross-encoder reranking, a Turkish-specific encoder, recipe steps, "
            "and an image-quality check before predicting.",
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
