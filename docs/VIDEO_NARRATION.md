# Demo video: narration

The silent recording is `dist/video/demo_silent.mp4` (about 1:59, 1440x900; each scene is held until its narration clip ends). It is produced by `python scripts/record_demo.py` while `python app.py` is running.

## Adding the narration

Record or generate **one audio clip per scene** and name it after the scene (`intro.wav`, `app.wav`, …). Put all clips in one folder, then run:

```bash
python scripts/record_demo.py --voice-dir <folder>
```

The script places every clip at the second its scene starts and writes `dist/video/demo_final.mp4`. A clip longer than its scene is cut, and the script prints a warning. If that happens, shorten that text and regenerate only that clip.

## Timing

Scene start times come from `dist/video/timeline.json`. Word budgets assume about 2.4 words per second.

| File | Starts | Length | On screen |
|---|---|---|---|
| `intro` | 0:00 | 11 s | Title card |
| `app` | 0:11 | 12 s | App opens; clicking the pizza example recognises it |
| `query` | 0:24 | 14 s | A request is typed (one slip, corrected) + Enter; the page scrolls to the full best recipe |
| `falafel_upload` | 0:38 | 11 s | A falafel photo is uploaded; the model is wrong |
| `ramen_en` | 0:48 | 16 s | Ramen is recognised; a vegan request |
| `compare` | 1:05 | 9 s | Every model side by side for the same photo and request |
| `architecture` | 1:13 | 14 s | Architecture diagram |
| `results` | 1:27 | 14 s | Results table |
| `conclusion` | 1:42 | 13 s | Conclusion card |

---

### intro (13 s)

You have a photo of a dish and a wish like "vegetarian, under thirty minutes". This app recognises the dish in the photo and ranks five recipes that fit your wish.

### app (12 s)

First, an easy case: a pizza photo. One click is enough. The quantized VGG16 says pizza, and the Grad-CAM heat map shows where the model looked.

### query (17 s)

Now I type a request and press Enter. A regular expression reads the time and diet limits, E5 ranks the recipes, and the best one is shown in full.

### falafel_upload (11 s)

A hard case: a falafel photo uploaded from disk. The model gets it wrong and says spaghetti bolognese; the heat map sits on the red sauce.

### ramen_en (17 s)

Ramen is recognised with ninety-nine percent confidence. For "vegan ramen under thirty minutes", every result is vegan and takes thirty minutes or less.

### compare (9 s)

At the bottom, every model runs on the same photo and request: three image models, four search models.

### architecture (12 s)

Two paths: the photo goes through the INT8 VGG16, the text through regex filters and the INT8 E5 encoder. Both run on the CPU with ONNX Runtime.

### results (15 s)

Every model used the same test sets. After quantization, VGG16 is four times smaller and five and a half times faster on the CPU, and top-1 accuracy drops by only a quarter of a point.

### conclusion (14 s)

To conclude: quantization is almost free and runs the whole app on a CPU. The Transformer finds the best recipes, but in Turkish our own word2vec is just as good.

---

## TTS tips

- Any English TTS works, for example Coqui XTTS-v2 (`language="en"`), Kokoro or Parler-TTS on Hugging Face.
- Save every section as its own `.wav`.
- Numbers are already written as words so they are read correctly.
