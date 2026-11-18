# Teslim öncesi kontrol listesi — SWE015 UC3

Her satır ödevin bir şartını, onu karşılayan dosyayı ve doğrulama komutunu gösterir.
Son iki satır (video ve HF) dışındaki her şey bu repoda tamamlanmış durumda.

## Ödev şartları

| Şart | Durum | Kanıt |
|---|---|---|
| **R1** Çalışan uygulama (yeni girdi alır, sonuç döner) | ✅ | `python app.py` → http://127.0.0.1:7860 · fotoğraf yükle + serbest metin · tarayıcısız kontrol: `python scripts/smoke_test.py --image <jpg>` |
| **R2** Dört teknik alanın hepsi | ✅ | Slayt 5 (teknik haritası) · A: `src/vision/small_cnn.py`, `vgg.py`, Grad-CAM · B: `src/retrieval/text.py`, `tfidf.py`, `w2v.py` · C: `src/retrieval/bert.py` · D: `scripts/quantize.py`, `src/retrieval/quantize.py` |
| **R3** Dört model aynı test setinde | ✅ | Görüntü: 2 499 test görseli (tüm modeller) · Arama: 40 sorgu (`data/queries.jsonl`) · `results/metrics.json` |
| **R4** Sonuç tablosu: metrik + boyut + latency | ✅ | `results/results.csv` (100 koşu, CPU/GPU, donanım sütunu) · slayt 10 |
| **R5** Uygulama quantized Transformer'ı çalıştırır | ✅ | `app.py` varsayılan: VGG16 INT8 (QDQ) + E5 INT8 (dynamic); model seçiciyle fp32 karşılaştırması |
| **R6** Mimari diyagramı (kodla uyumlu) | ✅ | `docs/architecture.png` · üretici: `python scripts/architecture.py` |
| **R7** Kod deposu: README, requirements, ağırlık yok | ✅ | https://github.com/mehmeteminakkaya/swe015-food-recipe-finder · kurulum: `pip install -r requirements.txt` → `python scripts/download_data.py --with-images` → `python app.py` · ağırlıklar release'te (`v1.0-weights`), CI yeşil |
| **R8** AI kullanım slaytı | ✅ | Slayt 12 · README "AI usage" |
| Sunum 8–12 slayt, Intro/Development/Conclusion | ✅ | `docs/slides.pdf` (12 slayt), ilk slaytta isim/numara/repo/video alanı |
| 2 dakikalık video (sesli, canlı demo) | ❌ **çekilecek** | `docs/VIDEO_SCRIPT.md` senaryosu hazır |
| Ağırlıkların dış barındırılması (HF/Drive) | ⚠️ GitHub release ile karşılandı | HF'ye taşımak isterseniz: `huggingface-cli login` → `python scripts/upload_models.py --repo <kullanıcı>/swe015-food-recipe-finder` |

## Grading kapıları

**Gate 1 — 0 puan riskleri**
- Video yok veya uygulamanın canlı çalıştığını göstermiyor → **şu an tek risk bu.**
- Transformer yok / quantized model yok → yok, ikisi de var (E5 INT8 + VGG16 INT8).
- Sahte sonuç → tüm sayılar `results/results.csv`'den; komutlar README'de.

**Gate 2 — 50 puan riskleri**
- Dört alan / teknik haritası → slayt 5.
- Merdiven / aynı test seti → slayt 6, 8, 10 + `results/metrics.json`.
- Eksik sonuç tablosu → `results/results.csv` (metrik, boyut, CPU/GPU latency, donanım).
- Mimari diyagramı kodla uyuşmuyor → `scripts/architecture.py` diyagramı koddan üretiyor.
- Sunum yapısı → Introduction / Development / Conclusion.
- Kuantize model uygulamada çalışmıyor → varsayılan model INT8, canlı test edildi.
- Repo README'den kurulamıyor → temiz bir kopyada doğrulandı (2 komut, ~5 dk).
- AI slaytı yok → slayt 12.

## Videoyu çektikten sonra (3 komut)

1. Videoyu YouTube'a **unlisted** yükle (veya Drive "linki olan herkes"), gizli pencerede aç.
2. Slaytı ve PDF'i linkle yeniden üret:
   ```powershell
   python scripts/make_slides.py --name "Mehmet Emin Akkaya" --id 2309011036 `
       --repo "https://github.com/mehmeteminakkaya/swe015-food-recipe-finder" --video "<VIDEO_LINK>"
   ```
   Sonra `docs/slides.pptx` dosyasını PDF olarak dışa aktar (PowerPoint → Farklı Kaydet → PDF).
3. `docs/slides.pdf`'i ve `docs/slides.pptx`'i commit'le:
   ```powershell
   git add docs/slides.pdf docs/slides.pptx
   git commit -m "Slides: demo video link"
   git push
   ```

## Sunum anında uygulama

```powershell
.venv\Scripts\activate
python app.py          # http://127.0.0.1:7860
```

Model seçiciler: görüntüde `VGG16 INT8 (varsayılan)` / `VGG16 fp32` / `Küçük CNN`;
aramada `E5 INT8 (varsayılan)` / `E5 fp32` / `tf-idf` / `word2vec`.
Hazır test görselleri: `data/raw/food10/test/{pizza,falafel,...}/*.jpg`.
