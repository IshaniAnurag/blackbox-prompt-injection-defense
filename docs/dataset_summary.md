# Dataset Summary — InjecAgent Prompt Injection Benchmark

## Dataset Overview
- **Source**: InjecAgent Benchmark Dataset (`InjecAgent-main.zip`) + Technical Edge Cases
- **Total Processed Samples**: 2546

## Class Distribution
- **Benign User Prompts (Label 0)**: 376 samples (14.77%)
- **Prompt Injection Attacks (Label 1)**: 2170 samples (85.23%)

## Stratified Dataset Splits
- **Train Set (70%)**: 1782 samples (Benign: 263, Injection: 1519)
- **Validation Set (15%)**: 382 samples (Benign: 56, Injection: 326)
- **Test Set (15%)**: 382 samples (Benign: 57, Injection: 325)

## Feature Columns
- `text`: Input text prompt (User query, Attacker payload, or Tool context)
- `label`: Integer binary classification label (`0` = Benign, `1` = Prompt Injection)
- `source`: Dataset origin component
- `type`: Category (`benign` / `injection`)
