#!/usr/bin/env python3
"""
Model 2: Deep Learning — Prediction Script
============================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model2_results.csv
"""
import pandas as pd
import platform
from pathlib import Path
import tensorflow as tf


# Paths
MODEL_PATH = Path("models/model2_deep_learning/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model2_results.csv"

def is_apple_silicon():
    return (
        platform.system() == "Darwin" and
        platform.machine() == "arm64"
    )

def disable_gpus():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Disable all GPUs by setting the visible device list to empty
            tf.config.set_visible_devices([], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def load_model():
    """Load your trained model from saved_model/.

    TensorFlow / Keras:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH / "model.keras")
    """
    model = tf.keras.models.load_model(MODEL_PATH / "model.keras")
    return model

def predict(model, test_data):
    """Generate predictions on test data.

    Should return a DataFrame with columns: id, prediction, probability, confidence
    """
    predictions = model.predict(test_data)
    return predictions

def main():
    if is_apple_silicon():
        print("Running on Apple Silicon — disabling GPU to avoid compatibility issues.")
        disable_gpus()

    # Load model
    model = load_model()

    # Load test data
    # TODO: Update this path to match your test data file
    test_df = pd.read_csv(TEST_DATA_DIR / "test_city_traffic_accidents.csv")

    # Generate predictions
    predictions = predict(model, test_df)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "prediction": predictions,
    #     "probability": raw_probabilities,
    #     "confidence": confidence_scores,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

    # print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
