#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
==========================================
Trains a RandomForestClassifier to predict accident severity (1-4)
from the Smart City traffic accidents dataset.

Run from the repo root after process_city_traffic.py has been executed:
    python pipelines/process_city_traffic.py
    python models/model1_traditional_ml/train.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import load_processed_data, split_data

SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"

# Columns to drop — text/ID/datetime strings (we use engineered temporal features instead)
DROP_COLS = [
    'ID', 'Source', 'Description', 'Street', 'City', 'County',
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


def load_data():
    return load_processed_data("processed_city_traffic_accidents.csv")


def preprocess_features(df):
    df = df.copy()

    y = df['Severity']
    df = df.drop(columns=['Severity'])

    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)

    # Drop any datetime-typed columns still in memory after CSV round-trip
    dt_cols = df.select_dtypes(include='datetime').columns.tolist()
    if dt_cols:
        df = df.drop(columns=dt_cols)

    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = (
                df[col].map({'True': 1, 'False': 0, True: 1, False: 0})
                .fillna(0)
                .astype(int)
            )

    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('Unknown').astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    num_cols = df.select_dtypes(include='number').columns
    medians = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    feature_cols = list(df.columns)
    X_train, X_val, y_train, y_val = split_data(df, y)

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_val_s = pd.DataFrame(
        scaler.transform(X_val), columns=feature_cols, index=X_val.index
    )

    return X_train_s, X_val_s, y_train, y_val, scaler, encoders, medians, feature_cols


def train_model(X_train, y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"  Class distribution: { {int(c): int(n) for c, n in zip(classes, counts)} }")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    print("\n=== Classification Report ===")
    print(classification_report(y_val, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_val, y_pred))

    wf1 = f1_score(y_val, y_pred, average='weighted')
    print(f"\nWeighted F1 Score: {wf1:.4f}")


def explain_model(model, feature_cols):
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]

    print("\n=== Top 15 Feature Importances ===")
    for i in top_idx:
        print(f"  {feature_cols[i]:<40} {importances[i]:.4f}")

    plt.figure(figsize=(11, 6))
    plt.title("Top 15 Feature Importances — Accident Severity (Model 1)")
    plt.bar(range(15), importances[top_idx], color='steelblue')
    plt.xticks(range(15), [feature_cols[i] for i in top_idx], rotation=45, ha='right')
    plt.ylabel("Importance")
    plt.tight_layout()

    chart_path = SAVED_MODEL_DIR / "feature_importance.png"
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Chart saved → {chart_path}")


def save_model(model, scaler, encoders, medians, feature_cols):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        SAVED_MODEL_DIR / "model.joblib")
    joblib.dump(scaler,       SAVED_MODEL_DIR / "scaler.joblib")
    joblib.dump(encoders,     SAVED_MODEL_DIR / "encoders.joblib")
    joblib.dump(medians,      SAVED_MODEL_DIR / "medians.joblib")
    joblib.dump(feature_cols, SAVED_MODEL_DIR / "feature_columns.joblib")
    print(f"  Artifacts saved → {SAVED_MODEL_DIR}")


def main():
    print("Step 1/5  Loading processed data...")
    df = load_data()
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("Step 2/5  Preprocessing features...")
    X_train, X_val, y_train, y_val, scaler, encoders, medians, feature_cols = preprocess_features(df)
    print(f"  {len(feature_cols)} features | train={len(X_train):,}  val={len(X_val):,}")

    print("Step 3/5  Training RandomForestClassifier...")
    model = train_model(X_train, y_train)

    print("Step 4/5  Evaluating...")
    evaluate_model(model, X_val, y_val)

    print("Step 5/5  Explaining & saving...")
    explain_model(model, feature_cols)
    save_model(model, scaler, encoders, medians, feature_cols)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
