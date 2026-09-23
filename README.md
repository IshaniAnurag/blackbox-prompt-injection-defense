# Black-Box Prompt Injection Detector

B.Tech Capstone Project on detecting prompt injection and jailbreak attacks in LLM applications.

## Overview
This repository contains a black-box prompt injection classifier built using PyTorch, Hugging Face Transformers, and Flask. The model classifies incoming text prompts as either Benign or Prompt Injection.

Dataset: InjecAgent benchmark dataset (`InjecAgent-main.zip`).

## Structure
```
.
├── data/           # Preprocessed train, val, test datasets
├── models/         # Fine-tuned model config and tokenizer
├── src/            # Code for preprocessing, training, evaluation, API
├── docs/           # Architecture and test results
├── report/         # IEEE project report
├── requirements.txt
└── README.md
```

## How to Run

1. **Environment Setup**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python src/test_env.py
   ```

2. **Preprocess Data**:
   ```bash
   python src/preprocess.py
   ```

3. **Train Model**:
   ```bash
   python src/train.py
   ```

4. **Evaluate Model**:
   ```bash
   python src/evaluate.py
   ```

5. **Run Flask API & Demo**:
   ```bash
   python src/app.py
   python src/test_demo.py
   ```

## Empirical Test Performance (Test Set N=382)

| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `0.9686` | **96.86%** |
| **Precision** | `0.9757` | **97.57%** |
| **Recall** | `0.9877` | **98.77%** |
| **F1 Score** | `0.9817` | **98.17%** |

### Confusion Matrix
```
                       Predicted Benign (0)   Predicted Injection (1)
Actual Benign (0)              49                     8
Actual Injection (1)           4                     321
```
