import customtkinter as ctk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import threading
from PIL import Image

TWITTER_BG = "#15202b"
TWITTER_BLUE = "#1d9bf0"
TEXT_COLOR = "#ffffff"
SUBTEXT_COLOR = "#8b98a5"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TwitterCloneApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("twitter.com - Duygu Analizi")
        self.geometry("600x750")
        self.configure(fg_color=TWITTER_BG)
        self.resizable(False, False)

        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.ESIK_DEGERI = 0.80  
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.ASSETS_DIR = os.path.join(self.BASE_DIR, "assets")

        self.emoji_map = {
            "Mutlu": "Mutlu.jpg",
            "Üzgün": "Uzgun.jpg",
            "Korku": "Korku.jpg",
            "Öfke": "Ofke.jpg",
            "Şaşkınlık": "Saskinlik.jpg",
            "Tiksinti": "Tiksinti.jpg",
        }

        self.OFKE_KELIMELERI = [
            "allah belanı", "allah kahretsin", "lanet olsun", "bela", 
            "gerizekalı", "aptal", "şerefsiz", "haysiyetsiz", "rezalet", 
            "defol", "zıkkım", "geber", "nefret"
        ]

        self.frame_header = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_header.pack(pady=20, padx=20, fill="x", anchor="w")

        try:
            img_path = os.path.join(self.ASSETS_DIR, "profil.jpg")
            if os.path.exists(img_path):
                 profil_img = ctk.CTkImage(Image.open(img_path), size=(55, 55))
                 self.lbl_profil = ctk.CTkLabel(self.frame_header, text="", image=profil_img)
            else:
                 raise FileNotFoundError
        except:
            self.lbl_profil = ctk.CTkLabel(self.frame_header, text="", width=55, height=55, fg_color="#8b98a5", corner_radius=27)
        
        self.lbl_profil.pack(side="left")

        self.frame_text_container = ctk.CTkFrame(self.frame_header, fg_color="transparent")
        self.frame_text_container.pack(side="left", padx=12, anchor="center")

        self.lbl_isim = ctk.CTkLabel(self.frame_text_container, text="Anonim Kullanıcı", font=("Arial", 16, "bold"), text_color=TEXT_COLOR, anchor="w")
        self.lbl_isim.pack(fill="x")
        
        self.lbl_username = ctk.CTkLabel(self.frame_text_container, text="@ne-hissettigini-biliyorum", font=("Arial", 14), text_color=SUBTEXT_COLOR, anchor="w")
        self.lbl_username.pack(fill="x")

        self.txt_giris = ctk.CTkTextbox(
            self, 
            height=150, 
            font=("Arial", 20), 
            fg_color="transparent",
            text_color=TEXT_COLOR,
            border_width=0,
            wrap="word"
        )
        self.txt_giris.pack(fill="x", padx=20, pady=(10,0))
        self.txt_giris.insert("0.0", "Neler oluyor?")
        self.txt_giris.bind("<FocusIn>", self.placeholder_sil)

        self.separator = ctk.CTkFrame(self, height=1, fg_color="#38444d")
        self.separator.pack(fill="x", padx=20, pady=10)

        self.frame_action = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_action.pack(pady=5, padx=20, fill="x")

        self.lbl_durum = ctk.CTkLabel(self.frame_action, text="Model Yükleniyor...", text_color=TWITTER_BLUE, font=("Arial", 12))
        self.lbl_durum.pack(side="left")

        self.btn_gonder = ctk.CTkButton(
            self.frame_action, 
            text="Gönder", 
            fg_color=TWITTER_BLUE, 
            hover_color="#1a8cd8",
            font=("Arial", 15, "bold"),
            corner_radius=20,
            width=90,
            height=35,
            state="disabled",
            command=self.analiz_et
        )
        self.btn_gonder.pack(side="right")

        self.frame_sonuc = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_sonuc.pack(pady=30, padx=20, fill="both", expand=True)

        self.lbl_sonuc_img = ctk.CTkLabel(self.frame_sonuc, text="")
        self.lbl_sonuc_img.pack(pady=(0, 15))

        self.lbl_sonuc_text = ctk.CTkLabel(self.frame_sonuc, text="", font=("Arial", 22, "bold"), text_color=TEXT_COLOR)
        self.lbl_sonuc_text.pack()

        threading.Thread(target=self.modeli_yukle).start()

    def placeholder_sil(self, event):
        if self.txt_giris.get("0.0", "end").strip() == "Neler oluyor?":
            self.txt_giris.delete("0.0", "end")

    def modeli_yukle(self):
        try:
            MODEL_PATH = os.path.join(self.BASE_DIR, "bert_model_cikti")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            self.id2label = self.model.config.id2label
            self.model.eval()
            self.lbl_durum.configure(text="", text_color="green")
            self.btn_gonder.configure(state="normal")
        except Exception as e:
            self.lbl_durum.configure(text="Hata: Model bulunamadı!", text_color="red")

    def resim_getir(self, duygu_adi):
        dosya_adi = self.emoji_map.get(duygu_adi)
        if dosya_adi:
            dosya_yolu = os.path.join(self.ASSETS_DIR, dosya_adi)
            if os.path.exists(dosya_yolu):
                img = ctk.CTkImage(Image.open(dosya_yolu), size=(220, 220))
                self.lbl_sonuc_img.configure(image=img)
                return
        self.lbl_sonuc_img.configure(image=None)

    def analiz_et(self):
        text = self.txt_giris.get("0.0", "end").strip()
        text_lower = text.lower()
        if len(text) < 2: return

        self.lbl_durum.configure(text="Analiz ediliyor...", text_color=SUBTEXT_COLOR)
        self.lbl_sonuc_text.configure(text="")
        self.lbl_sonuc_img.configure(image=None)
        self.update()

        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score, label_id = torch.max(probs, dim=-1)
                
                guven = score.item()
                label = self.id2label[label_id.item()]
                for kelime in self.OFKE_KELIMELERI:
                    if kelime in text_lower:
                        if not (label in ["Mutlu", "Şaşkınlık"] and guven > 0.90):
                             label = "Öfke"
                             guven = 0.99 
                             break
                
                olumsuzluklar = ["değil", "yok", "hiç"]
                if label == "Mutlu":
                    for olumsuz in olumsuzluklar:
                        if olumsuz in text_lower:
                            label = "Üzgün"
                            guven = 0.85 
                            break
                if guven < self.ESIK_DEGERI:
                    label = "Nötr"
                    renk = "gray"
                    self.resim_getir("Nötr") 
                    self.lbl_sonuc_text.configure(
                        text="NÖTR / DURUM TESPİTİ\n(Duygu Bulunamadı)", 
                        text_color=renk
                    )
                else:
                    renk_map = {"Mutlu": "#00ba7c", "Öfke": "#f91880", "Korku": "#7856ff", "Üzgün": "#1d9bf0", "Nötr": "gray"}
                    renk = renk_map.get(label, TEXT_COLOR)
                    
                    self.resim_getir(label)
                    self.lbl_sonuc_text.configure(
                        text=f"{label.upper()}\n(Güven: %{guven*100:.1f})", 
                        text_color=renk
                    )
                
                self.lbl_durum.configure(text="")

        except Exception as e:
            print(f"Hata: {e}")
            self.lbl_durum.configure(text="Hata oluştu.", text_color="red")

if __name__ == "__main__":
    app = TwitterCloneApp()
    app.mainloop()