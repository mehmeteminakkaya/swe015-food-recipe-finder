# 2 dakikalık video — çekim senaryosu (eski)

> **Güncel değil.** Video artık `scripts/record_demo.py` ile otomatik kaydediliyor ve
> anlatım İngilizce: güncel sahne listesi ve metinler `docs/VIDEO_NARRATION.md` içinde.
> Bu dosya yalnızca ilk elle çekim planı olarak duruyor.

Video, ödevin "uygulama gerçekten çalışıyor" kanıtı. Kurallar: ekran kaydı + sesli anlatım,
en fazla 2:00, YouTube'da **unlisted** veya Google Drive'da "linki olan herkes" olarak paylaş,
gizli pencerede linki kontrol et.

## Çekimden önce (5 dakika)

1. `python app.py` çalıştır, tarayıcıda http://127.0.0.1:7860 aç (modeller önceden yüklendiği için ilk tıklama hızlı olur).
2. Masaüstünde iki test fotoğrafı hazırla:
   - `data/raw/food10/test/pizza/validation_00500.jpg` (kolay örnek)
   - `data/raw/food10/test/falafel/validation_01766.jpg` (zor/hata örneği: VGG16 INT8 bunu %41 ile spaghetti bolognese sanıyor; slayt 7'deki Grad-CAM görseli de bu fotoğraf)
3. Tarayıcı penceresini 1080p'ye ayarla, yer imleri çubuğunu kapat, bildirimleri kapat.
4. Kayıt: Windows'ta `Win+G` veya OBS Studio, mikrofon açık, 1080p / 30 fps.

## Senaryo ve söylenecek cümleler

### 0:00 – 0:15 · Problem

> "Bir yemek uygulaması düşünün: kullanıcı elindeki yemeğin fotoğrafını çekiyor ve
> 'vejetaryen, 30 dakikadan kısa' gibi bir istek yazıyor. Amacımız fotoğraftan yemeği
> tanıyıp bu isteğe uyan beş tarifi getirmek. Bu bir SWE015 Use Case 3 projesi."

Ekranda: uygulamanın boş hâli, üstteki başlık görünsün.

### 0:15 – 1:15 · Canlı demo (en az iki farklı girdi, biri zor)

1. Kolay örnek: pizza fotoğrafını yükle → **Yemeği tahmin et** → tahmin, güven skoru ve
   Grad-CAM ısı haritası görünür.
   > "Model pizzayı yüzde 40 güvenle tanıdı; ısı haritası pizzanın kendisine bakıyor."
2. Metin: `vejetaryen, 30 dakikadan kısa` yaz → **Tarif ara** → 5 tarif, süre ve diyet
   etiketleri, benzerlik skorları.
   > "Kısıtlar metinden regex ile çıkarıldı, filtre tüm modellerde aynı; E5 INT8 modeli
   > 7 milisaniyede beş tarifi sıraladı."
3. Zor örnek (dürüstlük için önemli): falafel fotoğrafını yükle.
   > "Bu bir hata örneği: model falafeli spagetti bolonez olarak tahmin etti, çünkü
   > ısı haritasının gösterdiği gibi kırmızı sosa ve tabak kenarına bakıyor."
4. Model seçicide **VGG16 fp32** seçip aynı fotoğrafı tekrar çalıştır (isteğe bağlı, 5 sn).
5. İngilizce bir sorgu: `vegan ramen under 30 minutes`.
   > "Türkçe ve İngilizce aynı arayüzden çalışıyor."

### 1:15 – 1:45 · Mimari ve sonuçlar

Slayt 4 (mimari diyagram) → slayt 10 (sonuç tablosu).

> "Mimari dört teknik alanı kapsıyor: sıfırdan CNN ve VGG16 transfer learning, tf-idf ve
> kendi eğittiğimiz word2vec, çok dilli E5 encoder ve iki INT8 kuantizasyon.
> VGG16 INT8 110 megabayttan 27.7 megabayta indi, CPU'da 102 milisaniyeden 17 milisaniyeye
> düştü ve top-1 sadece 0.24 puan azaldı. Arama tarafında en iyi Recall@5 0.915 ile
> word2vec'te; E5 Türkçe ödünç kelimelerde zayıf kalıyor."

### 1:45 – 2:00 · Sonuç

> "Kuantizasyon bu görevde neredeyse bedava: uygulama tamamen CPU'da çalışıyor.
> Sınırlarım 10 sınıf ve 300 tarif; sonraki adım cross-encoder ile yeniden sıralama
> ve tarif adımlarını eklemek. Kod ve tüm sayılar GitHub deposunda."

## Kayıttan sonra

1. Videoyu YouTube'a **unlisted** yükle (veya Drive'a "linki olan herkes").
2. Gizli pencerede linki aç, gerçekten oynadığını doğrula.
3. Linki `docs/slides.pptx` 1. slaytındaki "Video:" alanına yaz ve PDF'i yeniden üret:
   `python scripts/make_slides.py --name "Mehmet Emin Akkaya" --id 2309011036 --repo "https://github.com/mehmeteminakkaya/swe015-food-recipe-finder" --video "<link>"`
   ardından PDF dışa aktarımı.
