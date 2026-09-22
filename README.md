# Twitter Sentiment Analysis 

Twitter metinlerini işleyerek duygu durumunu ve modelin güvenilirlik seviyesini analiz eden NLP projesidir. Çıktılar, tespit edilen duyguya uygun reaksiyon gösteren maymun görselleriyle desteklenir.

## Mimari ve Özellikler
* **Duygu Sınıflandırması:** Girdi metninin duygu polaritesini (pozitif, negatif, nötr) tespit eder.
* **Confidence Level (Güven Skoru):** Yapılan analiz tahmininin matematiksel doğruluk/güven oranını hesaplar.
* **Görsel Haritalama:** Algoritmanın tespit ettiği duygu durumunu (örn. mutlu, sinirli, şaşkın) önceden tanımlanmış görsellerle eşleştirerek UI/UX geri bildirimi sağlar.

## Kurulum ve Kullanım
Proje, TensorFlow/Keras altyapısını kullanmaktadır. Çalıştırmak için sanal ortamınızı aktif edip ana dizinden `arayuz.py` dosyasını tetiklemeniz yeterlidir:

```powershell
.\.venv\Scripts\activate
python src/arayuz.py