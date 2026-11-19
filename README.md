# Food Photo → Recipe Finder (SWE015 — Use Case 3)

[![tests](https://github.com/mehmeteminakkaya/swe015-food-recipe-finder/actions/workflows/tests.yml/badge.svg)](https://github.com/mehmeteminakkaya/swe015-food-recipe-finder/actions/workflows/tests.yml)

Photograph a dish, get the predicted dish with a Grad-CAM heat map, then type what
you want in English or Turkish ("vegetarian, under 30 minutes" / "vejetaryen, 30
dakikadan kısa") and get the five best-matching recipes: the best one in full (photo,
ingredients with quantities, step-by-step method, nutrition) and four more as cards.
A "Compare models" panel runs the same photo and request through every model of the
homework side by side.

The homework asks for one task solved with four models of increasing sophistication
on the same held-out test data, and for the quantized Transformer to run inside the
application. This repository contains that model ladder, the measurements, and the
Gradio app.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 1) data + weights from the release (~920 MB total, about 5 minutes)
python scripts/download_data.py --with-images

# 2) run the app
python app.py                     # http://127.0.0.1:7860
```

The weights are 200 MB (small CNN, VGG16 fp32 + INT8, tf-idf, word2vec, E5 INT8) and
the images are the 10-class subset; both live in the GitHub release
(<https://github.com/mehmeteminakkaya/swe015-food-recipe-finder/releases/latest>),
which keeps the repository free of large binary files. `download_data.py` also caches
the multilingual encoder so the first run works offline.

To rebuild everything from the original sources instead (about 1 h + 40 min of
training on an RTX 4060 Laptop):

```bash
python scripts/prepare_data.py      # Food-101 subset + recipe corpus
python scripts/enrich_recipes.py    # photo, quantities, method, nutrition for the cards
python scripts/dedupe_images.py     # remove the 7 exact duplicate photos
python scripts/build_indexes.py     # tf-idf, word2vec, E5 fp32, E5 INT8
python scripts/train_cnn.py         # small CNN from scratch
python scripts/train_vgg.py         # VGG16 feature extraction + conv5 fine-tuning
python scripts/quantize.py          # VGG16 -> ONNX -> INT8 QDQ
```

## Using the app

1. **Photo.** Upload or paste a photo, or click an example. It is recognised at once
   with the INT8 VGG16; the heat map shows where the model looked.
2. **Request.** Type what you want and press Enter (or click an example request).
   Time and diet limits are read from the text; the panel shows how the request was
   understood and how many recipes were left after the filters.
3. **Recipes.** The page scrolls to the results: the best match in full, four more
   below it. A new photo re-runs the request for the new dish.
4. **Compare models.** The same photo through all three image models and the same
   request through all four search models, side by side.

## What is in the box

| Component | Implementation |
|---|---|
| Image model (A) | Small CNN trained from scratch (`src/vision/small_cnn.py`) and VGG16 transfer learning: frozen feature extraction, then conv5 fine-tuning (`src/vision/vgg.py`) |
| Explainability (A) | Grad-CAM on VGG16 `conv5_3` (`app.py`, `scripts/gradcam_demo.py`) |
| Lexical baseline (B) | Normalisation, regex constraint extraction, Snowball stemming, TR→EN lexicon, tf-idf + cosine (`src/retrieval/text.py`, `tfidf.py`) |
| Small neural retrieval model (A/B) | word2vec trained on our 300 recipes, mean-pooled vectors, cosine (`src/retrieval/w2v.py`) |
| Transformer (C) | `intfloat/multilingual-e5-small`, E5 `query:`/`passage:` prefixes, cosine (`src/retrieval/bert.py`) |
| Efficient models (D) | VGG16 static INT8 QDQ quantization with ONNX Runtime (`scripts/quantize.py`); E5 dynamic INT8 quantization (`src/retrieval/quantize.py`) |

## Data

| Data | Source | Size / split |
|---|---|---|
| Images | Food-101 (Bossard et al., 2014), HF `ethz/food101`, 10 classes | 6 744 train / 750 val / 2 499 test. The official test split is the shared held-out set for every vision model. Seven exact duplicates were removed; one of them was also present in train, which would have leaked a test photo into training. |
| Recipes | Food.com Recipes (Kaggle `irkaal`, CC0), parsed copy HF `Karo8870/food.com-parsed-dataset`; step-by-step methods from Kaggle RAW_recipes (HF `KingName1/food.com`), matched on name + total time | 300 recipes, 30 per dish. 162 vegetarian, 58 vegan (rule-based from the ingredient list). 292 with a method, 239 with a photo |
| Test queries | Written by hand | 40 queries, 20 Turkish / 20 English, in `data/queries.jsonl` |

Within each dish, recipes that have a photo and a method are preferred, then a method
only, then a photo only; inside each tier the draw is random (seed 42). Names that
match the dish word but are another dish (pizza dip, sushi rice, Caesar dressing…)
are excluded, and no recipe belongs to two dishes. The corpus is never tuned to the
test queries.

Relevance is defined by rule, not by hand: a recipe is relevant for a query when it
belongs to the query's target dish **and** satisfies the query's constraints (total
time, vegetarian/vegan). The constraints are extracted from the query text with
regexes and applied as a hard filter for every retrieval model, so the models differ
only in ranking. `data/queries.jsonl` also stores the constraint fields explicitly,
and `scripts/evaluate.py` re-extracts them from the text, so a query whose wording
does not parse fails loudly instead of silently measuring something else.

## Results

`results/results.csv` is the single source of every number in the slides. Latency is
the mean of 100 runs, batch size 1, on an RTX 4060 Laptop 8 GB (GPU) and a Ryzen 7
8845HS (CPU). The INT8 ONNX models are measured on CPU only: quantization targets the
CPU deployment.

**Image classification** (2 499 test images, top-1 / top-5):

| Model | Precision | Top-1 | Top-5 | Size (MB) | CPU (ms) | GPU (ms) |
|---|---|---|---|---|---|---|
| Small CNN from scratch | fp32 | 0.7835 | 0.9836 | 4.9 | 7.1 | 1.2 |
| VGG16, feature extraction only | fp32 | 0.9020 | 0.9920 | 110.3 | — | — |
| VGG16, conv5 fine-tuned | fp32 | **0.9260** | **0.9952** | 110.3 | 83.6 | 5.2 |
| VGG16, conv5 fine-tuned | INT8 QDQ | 0.9236 | 0.9952 | **27.7** | **15.1** | — |

**Recipe retrieval** (40 queries, Recall@5 / MRR):

| Model | Precision | Recall@5 | MRR | TR Recall@5 | EN Recall@5 | Size (MB) | CPU (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + cosine | fp32 | 0.8262 | 0.8318 | 0.820 | 0.833 | 5.4 | 1.7 |
| word2vec (ours) | fp32 | 0.8800 | 0.9028 | **0.880** | 0.880 | **1.0** | **0.3** |
| E5 small | fp32 | **0.9062** | 0.9033 | 0.863 | **0.950** | 471.1 | 11.1 |
| E5 small | INT8 dynamic | 0.8900 | **0.9136** | 0.855 | 0.925 | 119.2 | 3.8 |

What the table says, in one paragraph: quantization is nearly free on this task —
VGG16 loses 0.24 points of top-1 accuracy for a 4× smaller and 5.5× faster CPU model,
and the E5 encoder becomes 4× smaller and 2.9× faster for 1.6 points of Recall@5
(its MRR even rises slightly). The multilingual Transformer finds the most relevant
recipes overall, and its lead comes from English (0.950 vs. 0.880); in Turkish our own
word2vec is as good (0.880 vs. 0.863). E5 wins paraphrases of the dish ("meatless
burgers", "vegan guakamole", "vejetaryen hamburger") and loses when the dish is only
described ("nohut köftesi", "crispy chickpea fritters", "pratik tatlı").

## Reproducing the results

```bash
python scripts/prepare_data.py      # Food-101 subset + recipe corpus
python scripts/enrich_recipes.py    # photo, quantities, method, nutrition (display only)
python scripts/dedupe_images.py     # remove the 7 exact duplicate photos
python scripts/build_indexes.py     # tf-idf, word2vec, E5 fp32, E5 INT8
python scripts/train_cnn.py         # small CNN from scratch      (~18 min)
python scripts/train_vgg.py         # VGG16 feature extraction + conv5 fine-tuning (~20 min)
python scripts/quantize.py          # VGG16 -> ONNX -> INT8 QDQ (256 calibration images)
python scripts/evaluate.py          # metrics -> results/metrics.json
python scripts/benchmark.py         # size/latency -> results/results.csv
python scripts/gradcam_demo.py      # docs/gradcam_example.png
python scripts/architecture.py      # docs/architecture.png (sizes/latency from results.csv)
python scripts/make_figures.py      # docs/fig_*.png
python scripts/make_slides.py       # docs/slides.pptx (every number read from results/)
python scripts/smoke_test.py        # exercises app.py without a browser
```

The demo video is recorded by a script too (needs `python app.py` running and
`pip install playwright==1.48.0 imageio-ffmpeg==0.5.1 && python -m playwright install ffmpeg`):

```bash
python scripts/record_demo.py                      # dist/video/demo_silent.mp4 + timeline.json
python scripts/make_storyboard.py                  # dist/video/storyboard.html
python scripts/record_demo.py --voice-dir <clips>  # add the narration (docs/VIDEO_NARRATION.md)
```

## Layout

```
app.py                     Gradio app (python app.py)
src/config.py              paths, classes, device, model caches inside the repo
src/data.py                recipes, queries, rule-based relevance, constraint filter
src/retrieval/text.py      normalisation, regex constraints, stemming, TR->EN lexicon
src/retrieval/tfidf.py     tf-idf + cosine baseline
src/retrieval/w2v.py       word2vec trained on our corpus
src/retrieval/bert.py      multilingual E5 encoder
src/retrieval/quantize.py  E5 -> ONNX -> INT8 (dynamic)
src/vision/…               small CNN, VGG16, data loaders, model loaders, evaluation
scripts/…                  data prep, training, quantization, evaluation, benchmark
data/queries.jsonl         the 40 test queries
results/                   metrics.json, results.csv, training histories
docs/                      PLAN.md, architecture.png, gradcam_example.png, slides, VIDEO_SCRIPT.md
```

## Tests

The text pipeline and the query set are covered by plain `unittest` tests that need
no torch, no weights and no GPU (the GitHub Actions workflow runs them on every push):

```bash
python -m unittest discover -s tests -v
```

They lock down the constraint parser (TR/EN time and diet phrasings, the
"quick → 30 minutes" rule, the "vegan implies vegetarian" rule), the Turkish
dotted/dotless I normalisation, and the fact that every query in `data/queries.jsonl`
still parses to the constraints stored next to it and has at least one relevant
recipe. A wording change that would silently alter the benchmark fails the build.

## Notes and limitations

- The image subset has only 10 classes and 750 images per class, so the error bars on
  accuracy are wide; the same split is used for every model.
- A real failure case, shown in `docs/gradcam_example.png`: a falafel photo is
  predicted as spaghetti bolognese (41 %). The heat map shows that the model reacts to
  the red sauce and the plate edge rather than to the falafel itself.
- `recipes.jsonl` comes from a noisy parsed copy of Food.com: a few names are still
  borderline (e.g. "Sushi Ginger"), and its `steps` column is broken. Retrieval
  therefore indexes name, description, category, tags and ingredients only; the method
  shown in the app comes from a second Food.com copy (292/300 matched on name + time),
  with nutrition-panel lines that some authors pasted into their method removed.
  Photos are Food.com links (239/300); recipes without one show a placeholder.
- Three recipes were labelled vegetarian although they contain fish (sardines,
  pollock, surimi); the meat/fish word list was extended and the labels re-derived
  (`python scripts/prepare_data.py --relabel`).
- Relevance is rule-based, so the retrieval benchmark rewards finding recipes of the
  target dish; it does not measure whether the recipe is a good cooking suggestion.
- The E5 encoder is a BERT-based multilingual encoder chosen over the MiniLM model
  named in the plan after a head-to-head on Turkish dish names (pankek↔pancakes:
  E5 0.85, MiniLM 0.42). Same technique area, same encoder size class.
- `app.py` patches one gradio 5.6.0 bug (its API-schema generator crashes on the
  boolean `additionalProperties` field of `gr.Image`, which makes every request 500).

## AI usage

AI coding assistants (Claude, DeepSeek) were used to write and debug the code, to
design the app, to script the demo recording, and to draft the slides and this README. Every number in the presentation comes from the runs in this repository
(`results/results.csv`); no results were copied from papers or model cards.
