import os
import sys
import time

try:
    from veri_onisleme import veriyi_yukle_ve_hazirla
    from model_egitimi import modeli_egit
except ImportError as e:
    print(f"HATA: Modüller yüklenemedi. 'src' klasöründe olduğundan emin misin?\nDetay: {e}")
    sys.exit(1)

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/processed/tremo_cleaned.csv"))
    
    print("\n" + "="*50)
    print(" 🚀  DUYGU ANALİZİ PROJESİ - EĞİTİM MODÜLÜ")
    print("="*50)
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ HATA: Veri seti bulunamadı!")
        print(f"Aranan yol: {DATA_PATH}")
        print("Lütfen 'tremo_cleaned.csv' dosyasını 'data/processed' klasörüne koyduğundan emin ol.")
        return

    try:
        baslangic_zamani = time.time()

        print(f"\n[1/2] Veri Seti Yükleniyor ve İşleniyor...")
        df, label2id, id2label = veriyi_yukle_ve_hazirla(DATA_PATH)
        
        print(f"\n[2/2] Model Eğitimi Başlatılıyor (BERT-Base-Turkish)...")
        print("NOT: Bu işlem bilgisayar hızına göre zaman alabilir, lütfen bekleyin.\n")
        
        modeli_egit(df, label2id, id2label)
        
        gecen_sure = (time.time() - baslangic_zamani) / 60
        
        print("\n" + "="*50)
        print(f" ✅ EĞİTİM BAŞARIYLA TAMAMLANDI! ({gecen_sure:.1f} dakika sürdü)")
        print(f" 📂 Kaydedilen Model: models/bert_sentiment_model")
        print("="*50)
        print("\n💡 İPUCU: Modeli test etmek için 'python tahmin.py' komutunu çalıştırabilirsin.")
        
    except Exception as e:
        print(f"\n!!! BEKLENMEYEN BİR HATA OLUŞTU !!!\n{e}")

if __name__ == "__main__":
    main()