# Dataset Summary — InjecAgent Prompt Injection Benchmark

## Dataset Overview
- **Source**: InjecAgent Benchmark Dataset (`InjecAgent-main.zip`) + Domain Benign Augmentations
- **Total Processed Samples**: 2548

## Class Distribution
- **Benign User Prompts (Label 0)**: 378 samples (14.84%)
- **Prompt Injection Attacks (Label 1)**: 2170 samples (85.16%)

## Stratified Dataset Splits
- **Train Set (70%)**: 1783 samples (Benign: 265, Injection: 1518)
- **Validation Set (15%)**: 382 samples (Benign: 56, Injection: 326)
- **Test Set (15%)**: 383 samples (Benign: 57, Injection: 326)

## Feature Columns
- `text`: Input text prompt (User query, Attacker payload, or Tool context)
- `label`: Integer binary classification label (`0` = Benign, `1` = Prompt Injection)
- `source`: Dataset origin component
- `type`: Category (`benign` / `injection`)
