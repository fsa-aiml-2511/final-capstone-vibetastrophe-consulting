"""
Model 5: Innovation — Road Deterioration Prediction
====================================================
Predicts road deterioration severity from 311 complaint data using XGBoost.

Usage: python models/model5_innovation/predict.py
Output: test_data/model5_results.csv
"""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT   = Path(__file__).resolve().parents[2]
MODEL_DIR      = Path(__file__).resolve().parent / "saved_model"
TEST_DATA_DIR  = PROJECT_ROOT / "test_data"
OUTPUT_FILE    = TEST_DATA_DIR / "model5_results.csv"
TEST_DATA_FILE = TEST_DATA_DIR / "urbanpulse_311_complaints.csv"

LEVEL_NAMES = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'}
MODEL_F1    = 0.82

ROAD_TYPES = {
    'Street Condition', 'Snow or Ice', 'Traffic Signal Condition',
    'Blocked Driveway', 'Street Light Condition', 'Highway Sign - Damaged',
    'Highway Sign - Missing', 'Sidewalk Condition', 'Curb Condition', 'Pothole',
}

SEVERITY_WEIGHT = {
    'Street Condition': 3, 'Pothole': 3, 'Sidewalk Condition': 2,
    'Curb Condition': 2, 'Snow or Ice': 2, 'Traffic Signal Condition': 2,
    'Blocked Driveway': 1, 'Street Light Condition': 1,
    'Highway Sign - Damaged': 2, 'Highway Sign - Missing': 1,
}

NUMERIC_COLS = [
    'hour_of_day', 'day_of_week', 'month', 'is_weekend',
    'severity_weight', 'resolution_hours',
]

DESCRIPTOR_RULES = [
    ('Noise',             ['LOUD', 'NOISE', 'BANGING', 'POUNDING', 'HORN', 'MUSIC', 'ALARM',
                           'BARKING', 'TELEVISION', 'PARTY', 'SHRIEKING', 'TALKING', 'CAR/TRUCK']),
    ('Heat_HotWater',     ['ENTIRE BUILDING', 'APARTMENT ONLY', 'RADIATOR', 'BOILER', 'NO HEAT',
                           'INADEQUATE', 'HEAT', 'HOT WATER', 'STEAM']),
    ('Plumbing_Water',    ['WATER SUPPLY', 'BASIN', 'SINK', 'BATHTUB', 'SHOWER', 'TOILET', 'LEAK',
                           'NO WATER', 'HYDRANT', 'SEWER', 'DIRTY WATER', 'SEWAGE', 'DRAIN', 'DAMP',
                           'SLOW LEAK', 'HEAVY FLOW', 'WATER METER', 'CATCH BASIN', 'WATER MAIN',
                           'PLUMBING']),
    ('Parking_Traffic',   ['PARKING', 'DOUBLE PARKED', 'LICENSE PLATE', 'BIKE LANE', 'BUS LAYOVER',
                           'DERELICT', 'DRIVEWAY', 'SIGNAL', 'CONE', 'CROSSWALK', 'BLOCKED HYDRANT',
                           'PEDESTRIAN', 'TRAFFIC', 'VEHICLE', 'WITH LICENSE']),
    ('Building_Structure',['CEILING', 'WALL', 'FLOOR', 'DOOR', 'WINDOW', 'CABINET', 'WIRING',
                           'OUTLET', 'INTERCOM', 'VENTILATION', 'ELECTRIC', 'REFRIGERATOR',
                           'COOKING GAS', 'SMOKE', 'CARBON MONOXIDE', 'LIGHTING', 'POWER', 'PAINT',
                           'PLASTER', 'MOLD', 'STRUCTURAL', 'BELL/BUZZER',
                           'GARBAGE/RECYCLING STORAGE', 'ILLEGAL CONVERSION']),
    ('Street_Sidewalk',   ['POTHOLE', 'SIDEWALK', 'ROADWAY', 'CAVE-IN', 'STREET LIGHT', 'SNOW',
                           'ICE', 'FAILED STREET', 'ROUGH', 'BROKEN SIDEWALK', 'ROAD',
                           'STREET REPAIR', 'PITTED']),
    ('Sanitation_Waste',  ['TRASH', 'GARBAGE', 'RECYCLING', 'GRAFFITI', 'DOG WASTE',
                           'ILLEGAL DUMP', 'DIRTY', 'LITTER', 'UNSANITARY', 'WASTE']),
    ('Pest_Animal',       ['RAT', 'PEST', 'MICE', 'MOUSE', 'ROACH', 'BED BUG', 'ANIMAL',
                           'INSECT', 'RODENT', 'MOSQUITO', 'BIRD', 'DOG']),
    ('Access_Elevator',   ['NO ACCESS', 'PARTIAL ACCESS', 'ACCESS', 'ELEVATOR']),
    ('Tree_Vegetation',   ['BRANCH', 'TREE', 'LIMB', 'FALLEN', 'TRUNK', 'ROOT']),
]


def load_model():
    model  = joblib.load(MODEL_DIR / "road_xgb_model.joblib")
    scaler = joblib.load(MODEL_DIR / "road_xgb_scaler.joblib")
    with open(MODEL_DIR / "road_xgb_features.json") as f:
        features = json.load(f)
    return model, scaler, features


def map_descriptor(desc):
    if not isinstance(desc, str):
        return 'Other'
    d = desc.upper()
    for category, keywords in DESCRIPTOR_RULES:
        if any(k in d for k in keywords):
            return category
    return 'Other'


def preprocess(df, features, scaler):
    # Clean: deduplicate and fill nulls
    df = df.drop_duplicates(subset='unique_key', keep='first').copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if len(mode) else 'Unknown')

    # Filter to road-related complaints only
    df['descriptor_cat'] = df['descriptor'].apply(map_descriptor)
    mask = df['complaint_type'].isin(ROAD_TYPES) | (df['descriptor_cat'] == 'Street_Sidewalk')
    df = df[mask].reset_index(drop=True)
    print(f"  Road complaints: {len(df):,} rows")

    # Engineer features
    created = pd.to_datetime(df['created_date'], errors='coerce')
    closed  = pd.to_datetime(df['closed_date'],  errors='coerce')
    df['hour_of_day']     = created.dt.hour.fillna(12).astype(int)
    df['day_of_week']     = created.dt.dayofweek.fillna(0).astype(int)
    df['month']           = created.dt.month.fillna(1).astype(int)
    df['is_weekend']      = (df['day_of_week'] >= 5).astype(int)
    df['severity_weight'] = df['complaint_type'].map(SEVERITY_WEIGHT).fillna(1)
    res_hrs = (closed - created).dt.total_seconds() / 3600
    res_hrs = res_hrs.where((res_hrs >= 0) & (res_hrs <= 720))
    df['resolution_hours'] = res_hrs.fillna(res_hrs.median() if res_hrs.notna().any() else 24.0)

    # One-hot encode and align to training feature set
    cat_cols = ['borough', 'open_data_channel_type', 'status', 'descriptor_cat']
    X = pd.get_dummies(df[NUMERIC_COLS + cat_cols], columns=cat_cols, drop_first=True)
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features].astype(float)
    X[NUMERIC_COLS] = scaler.transform(X[NUMERIC_COLS])

    return df['unique_key'], X


def predict(model, X):
    preds      = model.predict(X)
    proba      = model.predict_proba(X)
    confidence = proba.max(axis=1).round(4)
    levels     = [LEVEL_NAMES.get(int(p), 'Low') for p in preds]
    return levels, confidence


def main():
    print("Loading model artifacts...")
    model, scaler, features = load_model()

    print(f"Loading test data from {TEST_DATA_FILE}...")
    df = pd.read_csv(TEST_DATA_FILE)
    print(f"  {len(df):,} rows loaded")

    print("Preprocessing...")
    ids, X = preprocess(df, features, scaler)

    print("Generating predictions...")
    levels, confidence = predict(model, X)

    results = pd.DataFrame({
        'id':           ids.values,
        'prediction':   levels,
        'confidence':   confidence,
        'metric_name':  'weighted_f1',
        'metric_value': MODEL_F1,
    })

    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE} ({len(results):,} rows)")


if __name__ == "__main__":
    main()
