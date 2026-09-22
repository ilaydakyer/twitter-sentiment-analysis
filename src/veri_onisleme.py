import pandas as pd
import os

def veriyi_yukle_ve_hazirla(dosya_yolu):
    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"Veri seti bulunamadı: {dosya_yolu}")

    df = pd.read_csv(dosya_yolu)
    baslangic_sayisi = len(df)
    
    
    df.dropna(subset=['text', 'label'], inplace=True)
    
    df = df[df['text'].str.strip().astype(bool)]
    
    temizlenen_sayisi = baslangic_sayisi - len(df)
    
    if temizlenen_sayisi > 0:
        print(f"BİLGİ: {temizlenen_sayisi} adet boş veya hatalı satır silindi.")
    
    label_cevirisi = {
        "Happy": "Mutlu",
        "Fear": "Korku",
        "Sadness": "Üzgün",
        "Disgust": "Tiksinti",
        "Surprise": "Şaşkınlık",
        "Anger": "Öfke"
    }

    print("Etiketler Türkçeye çevriliyor...")
    df['label'] = df['label'].map(label_cevirisi)

    hatali_veri_sayisi = df['label'].isnull().sum()
    if hatali_veri_sayisi > 0:
        ornek_hatalar = df[df['label'].isnull()]
        raise ValueError(f"HATA: Çeviri sözlüğünde olmayan etiketler bulundu! "
                         f"{hatali_veri_sayisi} satır çevrilemedi.\n"
                         f"Örnekler:\n{ornek_hatalar.head()}")

    unique_labels = sorted(df['label'].unique())
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    
    df['label_id'] = df['label'].map(label2id)
    
    print("-" * 30)
    print(f"Veri Hazır.")
    print(f"Başlangıç Veri Sayısı: {baslangic_sayisi}")
    print(f"Temizlenmiş Veri Sayısı: {len(df)}")
    print(f"Sınıflar: {list(label2id.keys())}")
    print("-" * 30)
    
    return df, label2id, id2label