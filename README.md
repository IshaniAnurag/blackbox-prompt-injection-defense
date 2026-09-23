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
