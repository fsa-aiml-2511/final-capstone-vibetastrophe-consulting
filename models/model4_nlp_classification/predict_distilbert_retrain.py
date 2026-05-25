from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm


MODEL_PATH = Path("models/model4_nlp_classification/saved_model")
TEST_DATA_FILE = Path("test_data/distilbert_retrain_test_split.csv")
OUTPUT_FILE = Path("test_data/distilbert_retrain_results.csv")

BATCH_SIZE = 256
MAX_LENGTH = 128  # match your training script


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    device = get_device()
    print(f"Using device: {device}")

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

    model.to(device)
    model.eval()

    with open(MODEL_PATH / "id2label.json", "r") as f:
        id2label = json.load(f)

    id2label = {int(k): v for k, v in id2label.items()}

    return model, tokenizer, id2label, device


def predict(model, tokenizer, texts, id2label, device):
    all_preds = []
    all_confidences = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches"):
        batch_texts = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        confidences = torch.max(probs, dim=-1).values.cpu().numpy()

        pred_labels = [id2label[int(pred_id)] for pred_id in pred_ids]

        all_preds.extend(pred_labels)
        all_confidences.extend(confidences)

        if i % 2048 == 0:
            print(f"Processed {i:,}/{len(texts):,}")

    return all_preds, all_confidences


def main():
    model, tokenizer, id2label, device = load_model()

    test_df = pd.read_csv(TEST_DATA_FILE)

    texts = test_df["text"].fillna("").astype(str).tolist()
    y_true = test_df["label"].astype(str).tolist()

    print(f"Predicting on {len(texts):,} test records...")

    y_pred, confidence_scores = predict(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        id2label=id2label,
        device=device
    )

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=list(id2label.values())))

    results = pd.DataFrame({
        "id": test_df.index,
        "text": texts,
        "true_label": y_true,
        "predicted_label": y_pred,
        "confidence": confidence_scores
    })

    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPredictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
