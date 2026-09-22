import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from torch.utils.data import Dataset

# Dataset Sınıfı
class DuyguAnaliziDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Metrik Hesaplama
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Ana Eğitim Fonksiyonu
def modeli_egit(df, label2id, id2label):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Sadece Çıktı Klasörü Tanımlıyoruz (Log klasörünü kaldırdık)
    OUTPUT_DIR = os.path.join(BASE_DIR, "bert_model_cikti")
    
    # Klasör yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model Kayıt Yeri : {OUTPUT_DIR}")
    
    # Cihaz ayarı
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Eğitim Cihazı    : {device.upper()} 🚀")
    
    MODEL_ADI = "dbmdz/bert-base-turkish-cased"
    
    # Veriyi ayır (%80 Eğitim, %20 Test)
    print("Veri seti hazırlanıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_id']
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
    
    train_dataset = DuyguAnaliziDataset(train_encodings, y_train)
    test_dataset = DuyguAnaliziDataset(test_encodings, y_test)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ADI, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Eğitim Parametreleri
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # logging_dir parametresini SİLDİK. Varsayılan yere (runs klasörüne) yazsın.
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",        
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Eğitim başlatılıyor... (GPU Devrede)")
    trainer.train()
    
    # Sonuçları Yazdır
    print("\n--- TEST SONUÇLARI ---")
    eval_results = trainer.evaluate()
    print(f"Accuracy : {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score : {eval_results['eval_f1']:.4f}")
    
    # Kaydet
    print(f"Model kaydediliyor: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)