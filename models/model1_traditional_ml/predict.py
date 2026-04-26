#!/usr/bin/env python3
"""
Model 1: Traditional ML — Prediction Script
============================================
Loads the trained model and generates predictions on instructor test data.

Usage: python models/model1_traditional_ml/predict.py
Output: test_data/model1_results.csv
"""
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import clean_data, engineer_features

MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model1_results.csv"
TEST_DATA_FILE = TEST_DATA_DIR / "City_traffic_Test.csv"

DROP_COLS = [
    'Severity', 'Source', 'Description', 'Street', 'City', 'County',
    'Zipcode', 'Country', 'Airport_Code',
    'Start_Time', 'End_Time', 'Weather_Timestamp',
]

BOOL_COLS = [
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit',
    'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming',
    'Traffic_Signal', 'Turning_Loop',
]

CAT_COLS = [
    'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset',
    'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
    'State', 'Timezone',
]


def load_model():
    model        = joblib.load(MODEL_DIR / "model.joblib")
    scaler       = joblib.load(MODEL_DIR / "scaler.joblib")
    encoders     = joblib.load(MODEL_DIR / "encoders.joblib")
    medians      = joblib.load(MODEL_DIR / "medians.joblib")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, scaler, encoders, medians, feature_cols


def preprocess(df, encoders, scaler, medians, feature_cols):
    ids = df['ID'].copy()
    df = df.copy()

    df = clean_data(df)
    df = engineer_features(df)

    to_drop = [c for c in DROP_COLS + ['ID'] if c in df.columns]
    dt_cols = df.select_dtypes(include='datetime').columns.tolist()
    df = df.drop(columns=to_drop + dt_cols, errors='ignore')

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (
                df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )

    # Apply saved encoders; unseen categories get -1
    for col, le in encoders.items():
        if col in df.columns:
            known = set(le.classes_)
            df[col] = (
                df[col].fillna('Unknown').astype(str)
                .apply(lambda x: le.transform([x])[0] if x in known else -1)
            )

    # Align to training feature set (add missing cols, drop extras, reorder)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = medians.get(col, 0)
    df = df[feature_cols]

    # Fill remaining NaNs with training medians
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(medians.get(col, 0))

    X = scaler.transform(df)
    return ids, X


def predict(model, X):
    preds = model.predict(X)
    probas = model.predict_proba(X)

    top_probas = probas.max(axis=1)

    # confidence = top_prob / (top_prob + second_prob) — always >= raw probability
    sorted_p = np.sort(probas, axis=1)[:, ::-1]
    confidence = sorted_p[:, 0] / (sorted_p[:, 0] + sorted_p[:, 1])

    return preds, top_probas, confidence


def main():
    print("Loading model artifacts...")
    model, scaler, encoders, medians, feature_cols = load_model()

    print(f"Loading test data from {TEST_DATA_FILE}...")
    test_df = pd.read_csv(TEST_DATA_FILE)
    print(f"  {len(test_df):,} rows loaded")

    print("Preprocessing...")
    ids, X = preprocess(test_df, encoders, scaler, medians, feature_cols)

    print("Generating predictions...")
    preds, probabilities, confidence = predict(model, X)

    results = pd.DataFrame({
        'id':         ids.values,
        'prediction': preds,
        'probability': probabilities.round(4),
        'confidence':  confidence.round(4),
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
