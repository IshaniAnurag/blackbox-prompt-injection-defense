# Empirical Evaluation Results — Black-Box Prompt Injection Classifier

## Performance Metrics (Held-Out Test Set: N=382)

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `0.9686` | **96.86%** |
| **Precision** | `0.9757` | **97.57%** |
| **Recall** | `0.9877` | **98.77%** |
| **F1 Score** | `0.9817` | **98.17%** |

## Confusion Matrix Breakdown

```
                       Predicted Benign (0)   Predicted Injection (1)
Actual Benign (0)                 49                     8     
Actual Injection (1)              4                     321    
```

- **True Negatives (TN)**: 49 (Benign prompts correctly classified as Benign)
- **False Positives (FP)**: 8 (Benign technical prompts with trigger words classified as Injection)
- **False Negatives (FN)**: 4 (Subtle indirect prompt injections missed by classifier)
- **True Positives (TP)**: 321 (Prompt injections correctly blocked)

All metrics were empirically derived from test dataset predictions (`data/processed/test.csv`).
