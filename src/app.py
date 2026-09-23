import os
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join("models", "prompt_injection_classifier")

app = Flask(__name__)

print("Loading trained classifier model into Flask app...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    MODEL_LOADED = True
    print("Model successfully loaded onto device:", device)
except Exception as e:
    print("Error loading model:", e)
    MODEL_LOADED = False

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "device": str(device)
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(force=True, silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field in JSON request body"}), 400

    prompt = str(data["prompt"])

    encoding = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))

    label_str = "Prompt Injection" if pred_class == 1 else "Benign"
    is_inj = (pred_class == 1)
    confidence = float(probs[pred_class])

    return jsonify({
        "prompt": prompt,
        "prediction": label_str,
        "is_injection": is_inj,
        "confidence": confidence,
        "probabilities": {
            "benign": float(probs[0]),
            "injection": float(probs[1])
        }
    }), 200

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
