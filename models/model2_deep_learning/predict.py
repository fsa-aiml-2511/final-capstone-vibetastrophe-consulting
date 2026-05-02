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

    features = [
        'Distance(mi)', 'Timezone', 
         'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 
        'wind_dir_deg', 'weather_cond_num', 'accident_dir',
        'hour','day_of_week','month','is_weekend',
        'is_morning_rush','is_evening_rush','is_rush_hour',
        'duration_min','wind_dir_deg','weather_cond_num','weather_data_available',
        'is_freezing','low_visibility','accident_dir','lat_bin',
        'n_road_features','has_traffic_control'
    ]
    X = test_data[features].copy()
    
    predictions = model.predict(X)
    return predictions

def main():
    if is_apple_silicon():
        print("Running on Apple Silicon — disabling GPU to avoid compatibility issues.")
        disable_gpus()

    # Load model
    model = load_model()

    # Load test data
    # TODO: Update this path to match your test data file
    test_df = pd.read_csv(TEST_DATA_DIR / "City_traffic_Test.csv")
    
    # Generate predictions
    predictions = predict(model, test_df)

    # Save results — MUST match output template exactly
    # The predictions are in a two dimensional array. I need to extract the results so they will fit in a column in the dataframe.
    results = pd.DataFrame({
        "id": test_df["ID"],
        "prediction": predictions.argmax(axis=1),  # Assuming a classification model with one-hot encoded outputs
        "probability": predictions.max(axis=1),  # The probability of the predicted class
        "confidence": predictions.max(axis=1) / predictions.sum(axis=1)  #
    })
    results.to_csv(OUTPUT_FILE, index=False)

    # "probability": raw_probabilities,
    # "confidence": confidence_scores,

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
