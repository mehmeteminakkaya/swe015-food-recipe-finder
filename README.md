# Food Photo → Recipe Finder (SWE015 — Use Case 3)

Photograph a dish, get the predicted dish with a Grad-CAM heat map, then type what
you want in Turkish or English ("vejetaryen, 30 dakikadan kısa" / "vegetarian,
under 30 minutes") and get the five best-matching recipes with similarity scores.

The homework asks for one task solved with four models of increasing sophistication
on the same held-out test data, and for the quantized Transformer to run inside the
application. This repository contains that model ladder, the measurements, and the
Gradio app.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# 1) data: Food-101 10-class subset + recipe corpus (~1 h, 740 MB)
python scripts/prepare_data.py

# 2) models: either train them (~40 min on an RTX 4060 Laptop) ...
python scripts/build_indexes.py
python scripts/train_cnn.py
python scripts/train_vgg.py
python scripts/quantize.py

#    ... or download the published weights instead of step 2
set HF_REPO=<user>/swe015-food-recipe-finder
python scripts/download_data.py

# 3) run the app
python app.py                     # http://127.0.0.1:7860
```

The trained weights are not in git (see `.gitignore`). Either run the training
scripts below (about 40 minutes on an RTX 4060 Laptop), or fetch a published copy:

```bash
set HF_REPO=<user>/swe015-food-recipe-finder
python scripts/download_data.py
```

To publish the weights of a finished run to the Hub:

```bash
huggingface-cli login
python scripts/upload_models.py --repo <user>/swe015-food-recipe-finder
```

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
| Recipes | Food.com Recipes (Kaggle `irkaal`, CC0), parsed copy HF `Karo8870/food.com-parsed-dataset` | 300 recipes, 30 per dish. 171 vegetarian, 65 vegan (rule-based from the ingredient list) |
| Test queries | Written by hand | 40 queries, 20 Turkish / 20 English, in `data/queries.jsonl` |

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
| Small CNN from scratch | fp32 | 0.7835 | 0.9836 | 4.9 | 12.8 | 2.8 |
| VGG16, feature extraction only | fp32 | 0.9020 | 0.9920 | 110.3 | — | — |
| VGG16, conv5 fine-tuned | fp32 | **0.9260** | **0.9952** | 110.3 | 102.3 | 5.5 |
| VGG16, conv5 fine-tuned | INT8 QDQ | 0.9236 | 0.9952 | **27.7** | **16.9** | — |

**Recipe retrieval** (40 queries, Recall@5 / MRR):

| Model | Precision | Recall@5 | MRR | TR Recall@5 | EN Recall@5 | Size (MB) | CPU (ms) |
|---|---|---|---|---|---|---|---|
| tf-idf + cosine | fp32 | 0.8825 | 0.9458 | 0.885 | 0.880 | 5.3 | 3.5 |
| word2vec (ours) | fp32 | **0.9150** | **0.9708** | 0.960 | 0.870 | 1.0 | 0.6 |
| E5 small | fp32 | 0.8679 | 0.9073 | 0.852 | 0.883 | 471.1 | 24.8 |
| E5 small | INT8 dynamic | 0.8517 | 0.9135 | 0.810 | 0.893 | **119.2** | **6.7** |

What the table says, in one paragraph: quantization is nearly free on this task —
VGG16 loses 0.24 points of top-1 accuracy for a 4× smaller and 6× faster CPU model,
and the E5 encoder keeps MRR while getting 4× smaller and 3.7× faster. On the
retrieval side the honest finding is that the Transformer does **not** win: with 300
recipes and dish-named queries, lexical matches and our word2vec model are hard to
beat, while the dense encoder wins on the descriptive queries (q19, q20, q28, q29,
q30, q35, q39) and loses where the dish name is a Turkish loanword.

## Reproducing the results

```bash
python scripts/prepare_data.py      # Food-101 subset + recipe corpus
python scripts/dedupe_images.py     # remove the 7 exact duplicate photos
python scripts/build_indexes.py     # tf-idf, word2vec, E5 fp32, E5 INT8
python scripts/train_cnn.py         # small CNN from scratch      (~18 min)
python scripts/train_vgg.py         # VGG16 feature extraction + conv5 fine-tuning (~20 min)
python scripts/quantize.py          # VGG16 -> ONNX -> INT8 QDQ (256 calibration images)
python scripts/evaluate.py          # metrics -> results/metrics.json
python scripts/benchmark.py         # size/latency -> results/results.csv
python scripts/gradcam_demo.py      # docs/gradcam_example.png
python scripts/architecture.py      # docs/architecture.png
python scripts/smoke_test.py        # exercises app.py without a browser
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
docs/                      PLAN.md, architecture.png, gradcam_example.png, slides
```

## Notes and limitations

- The image subset has only 10 classes and 750 images per class, so the error bars on
  accuracy are wide; the same split is used for every model.
- A real failure case, shown in `docs/gradcam_example.png`: a falafel photo is
  predicted as spaghetti bolognese (41 %). The heat map shows that the model reacts to
  the red sauce and the plate edge rather than to the falafel itself.
- `recipes.jsonl` comes from a noisy parsed copy of Food.com: a few names are
  off-topic (e.g. "Shih Tzu Sushi"), `url` is only filled for 104/300 rows, and the
  `steps` column of the source is broken, so recipes are represented by name,
  description, category, tags and ingredients only.
- Relevance is rule-based, so the retrieval benchmark rewards finding recipes of the
  target dish; it does not measure whether the recipe is a good cooking suggestion.
- The E5 encoder is a BERT-based multilingual encoder chosen over the MiniLM model
  named in the plan after a head-to-head on Turkish dish names (pankek↔pancakes:
  E5 0.85, MiniLM 0.42). Same technique area, same encoder size class.
- `app.py` patches one gradio 5.6.0 bug (its API-schema generator crashes on the
  boolean `additionalProperties` field of `gr.Image`, which makes every request 500).

## AI usage

AI coding assistants were used to write and debug the code, to draft the slides and
this README. Every number in the presentation comes from the runs in this repository
(`results/results.csv`); no results were copied from papers or model cards.
