import json
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw", "InjecAgent-main", "data")
PROCESSED_DIR = os.path.join("data", "processed")
DOCS_DIR = "docs"

def load_json_file(filename):
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def load_jsonl_file(filename):
    filepath = os.path.join(RAW_DIR, filename)
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records

def generate_diverse_benign_prompts():
    """Generates realistic benign user instructions across agent tools."""
    categories = [
        "Can you check the current weather forecast for Tokyo tomorrow?",
        "Please send an email to John confirming our meeting at 3 PM.",
        "What are the top rated restaurants near downtown Seattle?",
        "Schedule a team sync on Google Calendar for next Monday at 10 AM.",
        "Fetch the latest stock price and news for Apple (AAPL).",
        "Summarize the main points of the attached research PDF.",
        "Turn off the living room lights and set thermostat to 72 degrees.",
        "Find me a round-trip flight from New York to London for next month.",
        "What is the status of my Amazon package order #408-129482-192?",
        "Translate this paragraph into French and Spanish.",
        "Create a todo list item for buying groceries and paying water bill.",
        "Search for recent papers on prompt injection defenses in LLM applications.",
        "Calculate the monthly mortgage payment for a $400,000 loan at 6.5% interest.",
        "Play the top hits playlist on Spotify in the kitchen speaker.",
        "Set an alarm for 6:30 AM tomorrow morning.",
        "What are the ingredients needed to bake a sourdough bread?",
        "Check my unread emails from manager@company.com.",
        "Get directions from current location to San Francisco airport.",
        "Show my upcoming meetings for this week.",
        "What is the current exchange rate from EUR to USD?"
    ]

    names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Isabella", "Jack"]
    cities = ["New York", "London", "Tokyo", "Paris", "Berlin", "Sydney", "Toronto", "Singapore", "San Francisco", "Boston"]
    products = ["Dell Inspiron Laptop", "Sony Headphones", "Logitech Mouse", "Samsung Monitor", "Kindle Paperwhite", "Apple iPad", "Bose Speaker"]
    topics = ["quantum computing", "machine learning", "cybersecurity", "climate change", "robotics", "financial markets", "renewable energy"]

    synthetic_benign = []
    random.seed(42)

    for i in range(800):
        name = random.choice(names)
        city = random.choice(cities)
        prod = random.choice(products)
        top = random.choice(topics)
        pid = f"B0{random.randint(100000, 999999)}"

        template_idx = i % 10
        if template_idx == 0:
            prompt = f"Can you fetch me the details and reviews of the {prod} with product ID {pid} from Amazon?"
        elif template_idx == 1:
            prompt = f"Please send an email to {name} at {name.lower()}@example.com about the project status."
        elif template_idx == 2:
            prompt = f"What is the weather forecast for {city} for the next 3 days?"
        elif template_idx == 3:
            prompt = f"Search Google Scholar for recent literature on {top} published after 2023."
        elif template_idx == 4:
            prompt = f"Schedule a calendar invitation with {name} regarding {top} on Friday at {random.randint(1,5)} PM."
        elif template_idx == 5:
            prompt = f"Summarize the recent emails received from {name} in my inbox."
        elif template_idx == 6:
            prompt = f"Find the best flight deals from {city} to London for next weekend."
        elif template_idx == 7:
            prompt = f"Check the status of my order {pid} and track shipping location."
        elif template_idx == 8:
            prompt = f"Convert ${random.randint(50, 500)} USD to EUR and JPY."
        else:
            prompt = f"Turn on the smart lights in {city} office and adjust color to warm white."

        synthetic_benign.append(prompt)

    return list(set(synthetic_benign))

def preprocess_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    print("Loading raw files from", RAW_DIR)
    dh_base = load_json_file("test_cases_dh_base.json")
    ds_base = load_json_file("test_cases_ds_base.json")
    dh_enh = load_json_file("test_cases_dh_enhanced.json")
    ds_enh = load_json_file("test_cases_ds_enhanced.json")
    user_cases = load_jsonl_file("user_cases.jsonl")
    attacker_dh = load_jsonl_file("attacker_cases_dh.jsonl")
    attacker_ds = load_jsonl_file("attacker_cases_ds.jsonl")

    samples = []

    # 1. Process base and enhanced test cases
    all_test_cases = dh_base + ds_base + dh_enh + ds_enh
    for case in all_test_cases:
        user_inst = case.get("User Instruction")
        attacker_inst = case.get("Attacker Instruction")
        tool_resp = case.get("Tool Response")

        if user_inst and isinstance(user_inst, str) and user_inst.strip():
            samples.append({"text": user_inst.strip(), "label": 0, "source": "user_instruction", "type": "benign"})

        if attacker_inst and isinstance(attacker_inst, str) and attacker_inst.strip():
            samples.append({"text": attacker_inst.strip(), "label": 1, "source": "attacker_instruction", "type": "injection"})

        if tool_resp and isinstance(tool_resp, str) and tool_resp.strip():
            samples.append({"text": tool_resp.strip(), "label": 1, "source": "indirect_tool_response", "type": "injection"})

    # 2. Process user_cases and attacker_cases
    for uc in user_cases:
        u_text = uc.get("User Instruction")
        if u_text and u_text.strip():
            samples.append({"text": u_text.strip(), "label": 0, "source": "user_case", "type": "benign"})

    for ac in attacker_dh + attacker_ds:
        a_text = ac.get("Attacker Instruction")
        if a_text and a_text.strip():
            samples.append({"text": a_text.strip(), "label": 1, "source": "attacker_case", "type": "injection"})

    # 3. Add augmented benign samples to create a balanced dataset
    augmented_benign = generate_diverse_benign_prompts()
    for b_text in augmented_benign:
        samples.append({"text": b_text, "label": 0, "source": "benign_augmentation", "type": "benign"})

    df = pd.DataFrame(samples)
    print(f"Total extracted raw samples: {len(df)}")

    # Deduplicate by text string
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Unique samples after deduplication: {len(df)}")

    # Balance classes if needed (sample equal numbers of benign and injection)
    benign_df = df[df["label"] == 0]
    injection_df = df[df["label"] == 1]

    print(f"Pre-balanced counts -> Benign: {len(benign_df)}, Injection: {len(injection_df)}")

    # Cap injection count or sample to match close ratio (~800 benign vs ~1200 injection)
    df_balanced = pd.concat([benign_df, injection_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    label_counts = df_balanced["label"].value_counts().to_dict()
    benign_count = label_counts.get(0, 0)
    injection_count = label_counts.get(1, 0)
    print(f"Final Class Distribution -> Benign (0): {benign_count}, Injection (1): {injection_count}")

    # Train / Val / Test split (70% train, 15% val, 15% test)
    train_df, test_val_df = train_test_split(df_balanced, test_size=0.30, random_state=42, stratify=df_balanced["label"])
    val_df, test_df = train_test_split(test_val_df, test_size=0.50, random_state=42, stratify=test_val_df["label"])

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Copy to root data/ as well for convenience
    os.makedirs("data", exist_ok=True)
    train_df.to_csv(os.path.join("data", "train.csv"), index=False)
    val_df.to_csv(os.path.join("data", "val.csv"), index=False)
    test_df.to_csv(os.path.join("data", "test.csv"), index=False)

    print(f"Saved splits:")
    print(f"  Train: {len(train_df)} rows -> {train_path}")
    print(f"  Validation: {len(val_df)} rows -> {val_path}")
    print(f"  Test: {len(test_df)} rows -> {test_path}")

    # Generate docs/dataset_summary.md
    summary_md = f"""# Dataset Summary — InjecAgent Prompt Injection Benchmark

## Dataset Overview
- **Source**: InjecAgent Benchmark Dataset (`InjecAgent-main.zip`) + Domain Benign Augmentations
- **Total Processed Samples**: {len(df_balanced)}

## Class Distribution
- **Benign User Prompts (Label 0)**: {benign_count} samples ({benign_count/len(df_balanced)*100:.2f}%)
- **Prompt Injection Attacks (Label 1)**: {injection_count} samples ({injection_count/len(df_balanced)*100:.2f}%)

## Stratified Dataset Splits
- **Train Set (70%)**: {len(train_df)} samples (Benign: {train_df['label'].value_counts().get(0, 0)}, Injection: {train_df['label'].value_counts().get(1, 0)})
- **Validation Set (15%)**: {len(val_df)} samples (Benign: {val_df['label'].value_counts().get(0, 0)}, Injection: {val_df['label'].value_counts().get(1, 0)})
- **Test Set (15%)**: {len(test_df)} samples (Benign: {test_df['label'].value_counts().get(0, 0)}, Injection: {test_df['label'].value_counts().get(1, 0)})

## Feature Columns
- `text`: Input text prompt (User query, Attacker payload, or Tool context)
- `label`: Integer binary classification label (`0` = Benign, `1` = Prompt Injection)
- `source`: Dataset origin component
- `type`: Category (`benign` / `injection`)
"""

    summary_file = os.path.join(DOCS_DIR, "dataset_summary.md")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"Wrote dataset summary to {summary_file}")

if __name__ == "__main__":
    preprocess_data()
