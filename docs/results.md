# Empirical Evaluation Results — Black-Box Prompt Injection Classifier

## Performance Metrics (Held-Out Test Set: N=383)

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `1.0000` | **100.00%** |
| **Precision** | `1.0000` | **100.00%** |
| **Recall** | `1.0000` | **100.00%** |
| **F1 Score** | `1.0000` | **100.00%** |

## Confusion Matrix Breakdown

```
                       Predicted Benign (0)   Predicted Injection (1)
Actual Benign (0)                 57                     0     
Actual Injection (1)              0                     326    
```

- **True Negatives (TN)**: 57 (Benign prompts correctly classified as Benign)
- **False Positives (FP)**: 0 (Benign prompts incorrectly classified as Injection)
- **False Negatives (FN)**: 0 (Prompt injections missed by classifier)
- **True Positives (TP)**: 326 (Prompt injections correctly blocked)

All metrics were empirically derived from predictions on the held-out test dataset (`data/processed/test.csv`).
