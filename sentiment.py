# distilbert_sentiment_improved.py

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import torch.nn as nn
import torch.optim as optim
import time

# === AYARLAR ===
MAX_LEN = 128
BATCH_SIZE = 32  # GPU belleğine göre 32 yapabilirsiniz
EPOCHS = 10  # Eğitim süresi artırıldı
SAMPLE_SIZE = 300000  # Daha fazla veri ile daha iyi genelleme
PATIENCE = 2  # Early stopping için


class DistilBertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTSentimentAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.class_weights = None
        self.early_stop_counter = 0
        self.best_f1 = 0

    def load_data(self, filepath, sample_size=SAMPLE_SIZE):
        try:
            cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
            df = pd.read_csv(filepath, encoding='latin-1', names=cols, header=None)
            df = df[['sentiment', 'text']]
            df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})
            df = df.dropna().reset_index(drop=True)
            df = df[df['text'].str.strip().astype(bool)].reset_index(drop=True)

            if sample_size and len(df) > sample_size:
                df = df.sample(sample_size, random_state=42).reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return self._create_sample_data()

    def _create_sample_data(self):
        texts = ["I love this product", "This is terrible", "I'm happy today", "Hate waiting", "Service was excellent",
                 "Movie is boring"]
        sentiments = [1, 0, 1, 0, 1, 0]
        return pd.DataFrame({'text': texts, 'sentiment': sentiments})

    def prepare_data(self, df, test_size=0.2):
        return train_test_split(
            df['text'],
            df['sentiment'],
            test_size=test_size,
            random_state=42,
            stratify=df['sentiment']
        )

    def compute_class_weights(self, y_train):
        class_counts = np.bincount(y_train)
        weights = 1. / class_counts
        return torch.FloatTensor(weights).to(self.device)

    def train(self, X_train, X_test, y_train, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
        train_dataset = DistilBertDataset(X_train, y_train, self.tokenizer)
        test_dataset = DistilBertDataset(X_test, y_test, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        self.class_weights = self.compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)  # Label smoothing eklendi

        best_f1 = 0
        self.early_stop_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Değerlendirme
            self.model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs.logits, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            f1 = f1_score(all_labels, all_preds)
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f} | F1: {f1:.4f}")

            # Early stopping ve model kaydetme
            if f1 > best_f1:
                best_f1 = f1
                self.best_f1 = f1
                torch.save(self.model.state_dict(), 'distilbert_sentiment.pt')
                self.early_stop_counter = 0
                print("Yeni en iyi model kaydedildi!")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= PATIENCE:
                    print("Early stopping aktif ediliyor...")
                    break

        print("\nFinal Sınıflandırma Raporu:")
        print(classification_report(all_labels, all_preds))
        return best_f1

    def predict(self, texts):
        self.model.load_state_dict(torch.load('distilbert_sentiment.pt'))  # En iyi modeli yükle
        self.model.eval()
        dataset = DistilBertDataset(pd.Series(texts), pd.Series([0] * len(texts)), self.tokenizer)
        loader = DataLoader(dataset, batch_size=8)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, 1)
                predictions.extend(preds.cpu().numpy())

        return ['Pozitif' if p == 1 else 'Negatif' for p in predictions]


def main():
    print("DistilBERT ile Twitter Duygu Analizi - İyileştirilmiş")
    print("-" * 50)

    analyzer = DistilBERTSentimentAnalyzer()

    csv_path = "training.1600000.processed.noemoticon.csv"
    if not os.path.exists(csv_path):
        print(f"Uyarı: Dosya bulunamadı: {csv_path}")
        print("Örnek veri ile devam ediliyor...")

    df = analyzer.load_data(csv_path)
    print(f"\nVeri seti yüklendi. Boyut: {len(df)} satır")
    print(f"Sınıf dağılımı:\n{df['sentiment'].value_counts()}")

    X_train, X_test, y_train, y_test = analyzer.prepare_data(df)
    print(f"Eğitim seti: {len(X_train)}, Test seti: {len(X_test)}")

    print("\nModel eğitimi başlatılıyor...")
    best_f1 = analyzer.train(X_train, X_test, y_train, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print(f"\nEn iyi F1 skoru: {best_f1:.4f}")

    test_texts = [
        "I absolutely love this new phone!",
        "The customer service was terrible and I'll never buy from them again.",
        "The weather is nice today.",
        "This movie was a complete waste of time and money."
    ]

    predictions = analyzer.predict(test_texts)
    print("\nÖrnek Tahminler:")
    for text, pred in zip(test_texts, predictions):
        print(f"{text[:50]}... → {pred}")


if __name__ == "__main__":
    main()