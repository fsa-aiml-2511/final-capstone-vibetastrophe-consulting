#!/usr/bin/env python3
"""
Model 4: NLP Classification — Training Script
===============================================
Train a text classification model on your scenario's text data.

Approaches (pick one):
- TF-IDF + traditional classifier (simplest, often surprisingly good)
- LSTM / GRU neural network
- Fine-tuned transformer (BERT, DistilBERT)

IMPORTANT: Save your vectorizer/tokenizer alongside the model — you'll need
the same text preprocessing at prediction time.
"""
from pathlib import Path
import torch

 

PROCESSED_DATA = Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv")
SAVED_MODEL_DIR = Path("./models/model4_nlp_classification/saved_model")


def load_data():
    """Load text data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    from pipelines.data_pipeline import load_raw_data
    df = load_raw_data()
    return df
    


def preprocess_text(texts):
    """Clean and prepare text for modeling.

    Common steps:
    - Lowercase
    - Handle abbreviations and slang
    - Tokenize
    - Remove stopwords (optional — sometimes they help)

    IMPORTANT: Apply the SAME preprocessing at prediction time.
    """
    # TODO: Clean your text data
    from transformers import DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    texts = list(texts)
    tokenized_texts = tokenizer(texts, truncation = True, batched = True, max_length=128)
    return tokenized_texts

    


def vectorize_text(texts):
    """Convert text to numerical features.

    Option 1 — TF-IDF (simplest):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        # Save vectorizer! You need it at prediction time.
        joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")

    Option 2 — Embeddings (for neural network approaches):
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(texts)
    """
    # TODO: Vectorize your text

import torch
class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)



def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "weighted_f1": f1
    }


def train_model(X_train, y_train):
    """Train your text classifier.

    TF-IDF approach:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)

    Neural network approach:
        import tensorflow as tf
        model = tf.keras.Sequential([...])
    """
    # TODO: Train your model
    from transformers import Trainer, TrainingArguments, DistilBertTokenizerFast, DistilBertForSequenceClassification, DataCollatorWithPadding
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    
    train_dataset = ComplaintDataset(X_train, y_train)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=6)
    data_collator = DataCollatorWithPadding(tokenizer=DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased'))

    for param in model.distilbert.parameters():
        param.requires_grad = False
    for layer in model.distilbert.transformer.layer[-1:]:
        param.requires_grad = True 
    for param in model.pre_classifier.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    training_args = TrainingArguments(
        output_dir = SAVED_MODEL_DIR,
        num_train_epochs = 3,
        per_device_train_batch_size = 16,
        learning_rate = 2e-5,
        weight_decay = 0.01,
        save_strategy = "epoch",
        logging_strategy = "epoch",
    )

    trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    )
    
    trainer.train()

    return trainer

    


def evaluate_model(trainer, X_val, y_val):      #model, X_val, y_val):
    """Evaluate NLP model performance.

    Must include:
    - Classification report per category
    - Weighted F1 score
    - Confusion matrix
    - Example predictions with actual text
    """
    # TODO: Evaluate your model
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    import numpy as np

    val_dataset = ComplaintDataset(X_val, y_val)

    predictions = trainer.predict(val_dataset)

    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    print("Weighted F1:")
    print(f1_score(y_true, y_pred, average="weighted"))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    


def save_model(trainer):
    """Save model AND vectorizer/tokenizer.

    IMPORTANT: You must save both the model and the text preprocessor.

    Example:
        import joblib
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
        joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    """
    # TODO: Save your model and vectorizer
    trainer.save_model(SAVED_MODEL_DIR)
    trainer.tokenizer.save_pretrained(SAVED_MODEL_DIR)
    
import pandas as pd
def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:

  
    top_5 = ['Illegal Parking', 'HEAT/HOT WATER', 'Noise - Residential',
             'Snow or Ice', 'Blocked Driveway']
    df['complaint_category'] = df['complaint_type'].apply(
        lambda x: x if x in top_5 else 'Other'
        )

    print("Complaint category distribution:")
    print(df['complaint_category'].value_counts())

    coverage = df[df['complaint_category'] != 'Other'].shape[0] / len(df) * 100
    print(f"\nTop 5 categories cover {coverage:.1f}% of all complaints")
    print(f"Total classes: {df['complaint_category'].nunique()} (top 5 + Other)")

    return df


def main():
    # 1. Load data
    import pandas as pd
    from pathlib import Path
    
    df = pd.read_csv(Path("data/raw/smart_city_csvs/urbanpulse_311_complaints.csv"))

    df = create_complaint_categories(df)

    y = df['complaint_category']
    X = df['resolution_description']

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
    random_state=42, stratify=y)

    X_train_encoded = preprocess_text(X_train)
    X_val_encoded = preprocess_text(X_val)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)

    trainer = train_model(X_train_encoded, y_train_encoded)

    evaluate_model(trainer, X_val_encoded, y_val_encoded)

    save_model(trainer)

    # 2. Preprocess text
    # texts = preprocess_text(df["text_column"])

    # 3. Vectorize
    # X = vectorize_text(texts)

    # 4. Split (use stratified split for imbalanced classes)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)

    # 5. Train
    # model = train_model(X_train, y_train)

    # 6. Evaluate
    # evaluate_model(model, X_val, y_val)

    # 7. Save model + vectorizer
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
