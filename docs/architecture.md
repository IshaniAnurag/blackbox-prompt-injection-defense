# System Architecture — Black-Box Prompt Injection & Jailbreak Classifier

## Overview
The proposed system is a lightweight, high-performance sequence classifier designed to operate as a real-time defense layer between external user inputs/tool responses and the core Large Language Model (LLM) application runtime.

```
[ Input Text Prompt ]
         │
         ▼
[ WordPiece Tokenizer ] ──────► Token IDs & Attention Mask (Max Length: 256)
         │
         ▼
[ Transformer Backbone ] ────► DistilBERT (6 Encoder Layers, 768-dim Embeddings)
         │
         ▼
[ [CLS] Embedding Vector ] ──► 768-dimensional sequence representation
         │
         ▼
[ Classification Head ] ─────► Linear(768 -> 2) + Softmax Normalization
         │
         ▼
[ Output Probabilities ] ────► P(Benign) vs. P(Prompt Injection)
```

## Component Specification

1. **Input Preprocessing & Tokenization**:
   - Model: `distilbert-base-uncased` WordPiece Tokenizer.
   - Max Sequence Length: `256` tokens.
   - Special Tokens: `[CLS]` (sequence classification token at index 0) and `[SEP]` (sentence separator).

2. **Feature Extraction (Transformer Encoder Backbone)**:
   - Architecture: DistilBERT (Distilled version of BERT).
   - Layers: 6 Transformer Encoder blocks.
   - Hidden Dimension: 768.
   - Attention Heads: 12 self-attention heads per layer.
   - Parameters: ~66 Million parameters (lightweight, enabling sub-20ms CPU inference latency).

3. **Classification Head**:
   - Pooling: `[CLS]` token representation ($h_{[CLS]} \in \mathbb{R}^{768}$).
   - Dropout: $p = 0.2$ regularization.
   - Linear Layer: $\mathbf{W} \in \mathbb{R}^{2 \times 768}, \mathbf{b} \in \mathbb{R}^2$.
   - Output: Logits $\mathbf{z} = \mathbf{W} h_{[CLS]} + \mathbf{b}$.
   - Activation: Softmax $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$.

4. **Loss Function & Optimization**:
   - Loss Function: Cross-Entropy Loss ($\mathcal{L}_{CE} = - \sum_{i} y_i \log(\hat{y}_i)$).
   - Optimizer: AdamW ($\text{learning rate} = 3 \times 10^{-5}, \text{weight decay} = 0.01$).
   - Epochs: 4 epochs with validation monitoring.
