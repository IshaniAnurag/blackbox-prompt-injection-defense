import os
import time
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import pandas as pd
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = os.path.join("models", "prompt_injection_classifier")
LOG_FILE = os.path.join("logs", "training_log.txt")
TRAIN_CSV = os.path.join("data", "processed", "train.csv")
VAL_CSV = os.path.join("data", "processed", "val.csv")

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LEN = 64

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def train_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    log_entries = []

    print("Loading data...", flush=True)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = TextDataset(train_df["text"].values, train_df["label"].values, tokenizer, MAX_LEN)
    val_dataset = TextDataset(val_df["text"].values, val_df["label"].values, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    best_val_acc = 0.0

    header = f"=== Training Log — {MODEL_NAME} ===\nDevice: {device}\nBatch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}\n"
    print(header, flush=True)
    log_entries.append(header)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total

        log_str = f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} ({correct}/{total})"
        print(log_str, flush=True)
        log_entries.append(log_str + "\n")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            print(f"  --> Best model achieved! Saving model to {OUTPUT_DIR}...", flush=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    elapsed = time.time() - start_time
    summary_str = f"\nTraining completed in {elapsed:.2f} seconds. Best Val Accuracy: {best_val_acc:.4f}\n"
    print(summary_str, flush=True)
    log_entries.append(summary_str)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.writelines(log_entries)

    print(f"Training log written to {LOG_FILE}", flush=True)

if __name__ == "__main__":
    train_model()
