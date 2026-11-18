# SWE015 Ödev – Use Case 3: Food Photo → Recipe Finder

Son teslim: **22 Kasım 2026 (Pazar)**
Donanım: RTX 4060 Laptop (8 GB) · 16 GB RAM · Python 3.11 · Windows 11

## 1. Kapsam

Kullanıcı bir yemek fotoğrafı yükler. Uygulama tahmin edilen yemeği güven skoru ve Grad-CAM ısı haritasıyla gösterir. Kullanıcı ardından serbest metinle istek yazar (TR veya EN, ör. "vejetaryen, 30 dakikadan kısa" / "vegetarian, under 30 minutes"). Uygulama benzerlik skorlarıyla birlikte en iyi 5 tarifi döner.

## 2. Veri

| Veri | Kaynak | Boyut / Bölme |
|---|---|---|
| Görseller | Food-101 (Bossard et al., 2014), HF `ethz/food101`, 10 sınıflık alt küme | Sınıf başına 750 train / 250 test. Train'in %10'u validation'a ayrılır. Test seti **tüm modeller için aynı**. |
| Tarifler | Food.com Recipes (Kaggle, irkaal, CC0). Ayrıştırılmış kopya: HF `Karo8870/food.com-parsed-dataset`. Kullanılan alanlar: ad, açıklama, kategori, etiketler, malzemeler, süre (hazırlık + pişirme). `steps` sütunu bozuk olduğu için kullanılmıyor. | 300 tarif (sınıf başına 30, seed=42 ile rastgele seçildi), 171'i vejetaryen. Vejetaryen/vegan etiketi malzeme listesinden kuralla çıkarılıyor. |
| Test sorguları | Kendimiz yazıyoruz | En az 40 sorgu (yarısı TR, yarısı EN) |

**10 sınıf** (tarif çeşitliliği ve vejetaryen/etli dengesi gözetilerek):
`pizza, hamburger, sushi, ramen, falafel, caesar_salad, pancakes, chocolate_cake, spaghetti_bolognese, guacamole`

**Sorgu alaka etiketi (relevance):** Uydurma veri olmaması ve sonuçların tekrarlanabilir olması için kural tabanlı tanımlanır. Bir tarif, bir sorgu için şu iki koşul birlikte sağlanıyorsa "relevant" sayılır:
1. Tarif sorgunun hedef yemeğine ait.
2. Tarif sorgudaki kısıtları (süre, diyet) gerçekten karşılıyor.

Etiketler `data/queries.jsonl` dosyasında tutulur ve elle gözden geçirilir.

## 3. Model merdiveni (R3)

İki alt görev var. Her birinin kendi merdiveni olacak ve her merdivenin tüm modelleri aynı test setinde değerlendirilecek.

### Görüntü sınıflandırma (test: 2500 Food-101 test görseli)

| # | Model | Alan |
|---|---|---|
| 1 | Sıfırdan eğitilen küçük CNN (augmentation + dropout) | A (baseline) |
| 2 | VGG16. Önce feature extraction, sonra son conv bloğu açılarak fine-tuning | A (transfer learning) |
| 3 | VGG16, INT8 quantized | D |
| + | Grad-CAM (VGG16'nın son conv katmanı üzerinde) | A |

### Tarif arama (test: en az 40 sorgu)

| # | Model | Alan |
|---|---|---|
| 1 | Normalizasyon + regex kısıt çıkarma + stemming (EN Snowball / TR snowball) + tf-idf + cosine | B (klasik baseline) |
| 2 | Kendi eğittiğimiz sinir ağı: tarifler üzerinde word2vec, sorgu ve tarif vektörlerinin ortalaması (merdivenin "NN" basamağı) | A/B |
| 3 | `intfloat/multilingual-e5-small` (BERT tabanlı çok dilli encoder, TR+EN) sentence embedding | C |
| 4 | Aynı model, INT8 quantized (ONNX Runtime dynamic quantization) | D |

Regex ile çıkarılan kısıtlar (süre ve diyet) **bütün** arama modellerinde aynı şekilde filtre olarak uygulanır. Böylece modeller arasındaki fark yalnızca sıralamadan gelir.

> **Uygulama notu (18 Kasım):** Planda MiniLM yazıyordu; Türkçe yemek adlarında
> (`pankek`↔`pancakes`) MiniLM 0.42, E5 0.85 benzerlik verdiği için aynı tekniğin
> (C: çok dilli BERT tabanlı encoder) E5-small varyantı kullanıldı.

Fotoğraftan tahmin edilen yemek adı sorguya eklenir (veya o yemeğin tarifleri öne çıkarılır). Bu birleştirmenin hangi yöntemle yapılacağı uygulama aşamasında netleşecek.

## 4. Metrikler ve sonuç tablosu (R4)

- **Görüntü:** top-1 ve top-5 accuracy. CNN quantization'dan önce ve sonra ayrı ayrı raporlanır.
- **Arama:** Recall@5 ve MRR.
- **Her model için:** boyut (MB) ve latency. Latency en az 100 çalıştırmanın ortalaması, CPU ve GPU ayrı belirtilir.
- Ölçümler `scripts/benchmark.py` tarafından üretilir ve `results/results.csv` dosyasına yazılır. Sunumdaki **her sayı bu dosyadan gelir.**

## 5. Teknik haritası (R2)

| Alan | Uygulamadaki karşılığı |
|---|---|
| A – NN & DL | Küçük CNN, VGG16 transfer learning + fine-tuning, augmentation, dropout, Grad-CAM |
| B – Klasik NLP | Regex ile kısıt çıkarma, normalizasyon, stemming, tf-idf + cosine, word2vec |
| C – Transformer | Multilingual MiniLM (BERT tabanlı) sentence embedding |
| D – Verimli modeller | INT8 quantization (VGG16 ve MiniLM), ONNX Runtime, boyut/latency/accuracy benchmark |

## 6. Uygulama (R1, R5)

Gradio arayüzü:
- Fotoğraf yükleme → tahmin edilen yemek, güven skoru, Grad-CAM görseli.
- Metin kutusu → top-5 tarif, benzerlik skorlarıyla.
- Model seçici: varsayılan olarak **quantized** modeller çalışır. Karşılaştırma için diğer modeller de seçilebilir.

## 7. Repo yapısı (R7)

```
app.py                  # Gradio uygulaması (tek çalıştırma komutu: python app.py)
src/
  data.py               # Food-101 alt kümesi, tarif seti hazırlama
  vision/               # small_cnn.py, vgg.py, gradcam.py, quantize.py
  retrieval/            # text.py (normalize/regex/stem), tfidf.py, w2v.py, bert.py, quantize.py
scripts/
  prepare_data.py
  train_cnn.py
  train_vgg.py
  build_indexes.py
  evaluate.py
  benchmark.py
data/queries.jsonl      # test sorguları + relevance etiketleri
results/                # results.csv, grafikler
docs/                   # PLAN.md, mimari diyagram
```

Model ağırlıkları repoya konmaz. Hugging Face Hub'a yüklenir ve `scripts/download_models.py` ile indirilir.

## 8. Takvim

| Tarih | İş |
|---|---|
| 8–9 Kas | Ortam kurulumu, veri hazırlama, tarif seti seçimi |
| 10–12 Kas | Küçük CNN ve VGG16 eğitimi, Grad-CAM |
| 13–15 Kas | Arama: tf-idf, word2vec, BERT. Test sorguları. |
| 16–17 Kas | Quantization, benchmark, sonuç tablosu |
| 18–19 Kas | Gradio uygulaması, README, HF Hub |
| 20–21 Kas | Sunum, mimari diyagram, video |
| 22 Kas | Teslim |

## 9. Açık konular

- Sunum için öğrenci adı ve numarası gerekecek.
- GitHub repo adı belirlenecek.
