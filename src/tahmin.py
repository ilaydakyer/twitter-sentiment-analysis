import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def tahmin_et():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "bert_model_cikti")
    
    ESIK_DEGERI = 0.70  
    
    print(f"Model yükleniyor: {MODEL_PATH} ...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"HATA: Model yüklenemedi! Klasör yolunu kontrol et.\nDetay: {e}")
        return

    model.eval()
    id2label = model.config.id2label
    
    print("\n" + "="*50)
    print(" 🤖 DUYGU ANALİZİ ROBOTU (AKILLI MOD)")
    print(f" 🛡️ Güven Eşiği: %{ESIK_DEGERI*100}")
    print(" Çıkmak için 'q' veya 'exit' yaz.")
    print("="*50)

    while True:
        text = input("\n📝 Cümle Yaz: ")
        
        if text.lower() in ["q", "exit", "çıkış"]:
            print("Görüşmek üzere! 👋")
            break
            
        if len(text.strip()) < 3:
            print("⚠️ Lütfen daha uzun bir cümle kurun.")
            continue

        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            score, label_id = torch.max(probs, dim=-1)
            label_id = label_id.item()
            score = score.item()
            
            predicted_label = id2label[label_id]

            if score < ESIK_DEGERI:
                print(f"🤔 SONUÇ: Emin Değilim / Nötr (En yakın: {predicted_label}, Güven: %{score*100:.2f})")
            else:
                print(f"🔍 TAHMİN: {predicted_label} (Güven: %{score*100:.2f})")

if __name__ == "__main__":
    tahmin_et()