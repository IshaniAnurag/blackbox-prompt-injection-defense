import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = os.path.join("models", "prompt_injection_classifier")
TEST_CSV = os.path.join("data", "processed", "test.csv")
RESULTS_DIR = "results"
DOCS_DIR = "docs"

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    print("Loading test dataset from", TEST_CSV)
    df = pd.read_csv(TEST_CSV)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).values

    print(f"Test samples count: {len(df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading trained model from {MODEL_DIR} using device {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    y_preds = []
    y_probs = []

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_preds.extend(preds)
            y_probs.extend(probs[:, 1]) # probability of injection class

    acc = float(accuracy_score(y_true, y_preds))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_preds, average="binary")

    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "total_test_samples": int(len(df))
    }

    metrics_json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ EVALUATION RESULTS ================")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"                Predicted Benign (0)  Predicted Injection (1)")
    print(f"Actual Benign           {tn:^10}            {fp:^10}")
    print(f"Actual Injection        {fn:^10}            {tp:^10}")
    print("====================================================")

    # Write docs/results.md
    results_md = f"""# Empirical Evaluation Results — Black-Box Prompt Injection Classifier

## Performance Metrics (Held-Out Test Set: N={len(df)})

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `{acc:.4f}` | **{acc*100:.2f}%** |
| **Precision** | `{prec:.4f}` | **{prec*100:.2f}%** |
| **Recall** | `{rec:.4f}` | **{rec*100:.2f}%** |
| **F1 Score** | `{f1:.4f}` | **{f1*100:.2f}%** |

## Confusion Matrix Breakdown

```
                       Predicted Benign (0)   Predicted Injection (1)
Actual Benign (0)             {tn:^10}             {fp:^10}
Actual Injection (1)          {fn:^10}             {tp:^10}
```

- **True Negatives (TN)**: {tn} (Benign prompts correctly classified as Benign)
- **False Positives (FP)**: {fp} (Benign prompts incorrectly classified as Injection)
- **False Negatives (FN)**: {fn} (Prompt injections missed by classifier)
- **True Positives (TP)**: {tp} (Prompt injections correctly blocked)

All metrics were empirically derived from predictions on the held-out test dataset (`data/processed/test.csv`).
"""

    results_md_path = os.path.join(DOCS_DIR, "results.md")
    with open(results_md_path, "w", encoding="utf-8") as f:
        f.write(results_md)

    print(f"Saved evaluation metrics to {metrics_json_path} and {results_md_path}")

if __name__ == "__main__":
    evaluate()
