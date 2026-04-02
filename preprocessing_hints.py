"""
Smart City Data Preprocessing Hints
=====================================
These are HINTS, not complete solutions. Use them as a starting point
for your data pipeline. You'll need to adapt and expand these for your
specific models.

Datasets:
- city_traffic_accidents.csv (~500K accident records)
- pothole_images/ (~4,400 road surface images)
- urbanpulse_311_complaints.csv (~500K complaint records)
"""

# =============================================================================
# *** CLASS IMBALANCE WARNING ***
# =============================================================================
# EVERY dataset in this project has class imbalance. If you ignore it,
# your model will learn to predict the majority class and look "accurate"
# while being clinically useless.
#
# Techniques you MUST consider for every model:
#   1. class_weight='balanced' in sklearn models (easiest first step)
#   2. SMOTE (Synthetic Minority Oversampling) from imblearn
#   3. Stratified train/test splits (use stratify= in train_test_split)
#   4. Weighted loss functions in TensorFlow/Keras
#   5. Evaluation with weighted F1, precision, recall — NOT just accuracy
#
# A model that predicts the majority class for everything is WORTHLESS
# even if it gets 80%+ accuracy. Always check per-class metrics.
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# HINT 1: Loading the Accident Data
# =============================================================================

def load_accidents(filepath: str) -> pd.DataFrame:
    """
    Load the city traffic accidents dataset.

    Key gotchas:
    - Severity is 1-4 scale measuring TRAFFIC IMPACT, not injury severity
    - Severity 2 dominates (~80% of records) — MAJOR class imbalance
      A naive model that predicts Severity 2 for everything gets ~80% accuracy
      but is completely useless. You MUST use class weights or SMOTE.
      Weighted F1 is the real metric, not accuracy.
    - Datetime columns need parsing
    - Weather columns have significant missing values
    - Some lat/lng values are null (End_Lat, End_Lng especially)
    """
    df = pd.read_csv(filepath)

    # Parse datetime columns
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])

    return df


# =============================================================================
# HINT 2: Temporal Feature Engineering
# =============================================================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time patterns are among the strongest predictors of accident severity.

    Features to extract:
    - Hour of day (rush hour vs. off-peak)
    - Day of week (weekday vs. weekend)
    - Month (seasonal patterns — winter ice, summer heat)
    - Duration of traffic impact
    - Is it dark? (Sunrise_Sunset column helps, but you can derive from time too)
    """
    df['hour'] = df['Start_Time'].dt.hour
    df['day_of_week'] = df['Start_Time'].dt.dayofweek
    df['month'] = df['Start_Time'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Rush hour flags
    df['is_morning_rush'] = df['hour'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hour'].between(16, 19).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)

    # Duration of traffic impact (in minutes)
    if 'End_Time' in df.columns:
        df['duration_min'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
        # Cap extreme values
        df['duration_min'] = df['duration_min'].clip(0, 1440)  # Max 24 hours

    return df


# =============================================================================
# HINT 3: Weather Feature Processing
# =============================================================================

def process_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather is a major factor in accident severity.

    Missing values in weather columns are NOT random — they often mean:
    - Weather station was offline
    - Data wasn't available at the time of the accident
    - The weather API didn't return data for that location

    Strategy: Create a "weather_data_available" flag, then impute or drop.

    Key weather features:
    - Temperature(F): Freezing conditions are dangerous
    - Visibility(mi): Low visibility = more severe accidents
    - Precipitation(in): Rain/snow increases severity
    - Weather_Condition: Categorical (Clear, Rain, Snow, Fog, etc.)
    """
    weather_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    # Flag for whether weather data is available
    df['weather_data_available'] = df[weather_cols].notna().all(axis=1).astype(int)

    # Freezing conditions
    if 'Temperature(F)' in df.columns:
        df['is_freezing'] = (df['Temperature(F)'] <= 32).astype(int)

    # Low visibility
    if 'Visibility(mi)' in df.columns:
        df['low_visibility'] = (df['Visibility(mi)'] < 2).astype(int)

    # Group weather conditions
    if 'Weather_Condition' in df.columns:
        df['weather_group'] = df['Weather_Condition'].apply(categorize_weather)

    return df


def categorize_weather(condition) -> str:
    """Group detailed weather conditions into broader categories."""
    if pd.isna(condition):
        return 'unknown'

    condition = str(condition).lower()

    if any(w in condition for w in ['clear', 'fair']):
        return 'clear'
    elif any(w in condition for w in ['cloud', 'overcast']):
        return 'cloudy'
    elif any(w in condition for w in ['rain', 'drizzle', 'shower']):
        return 'rain'
    elif any(w in condition for w in ['snow', 'sleet', 'ice', 'wintry']):
        return 'snow_ice'
    elif any(w in condition for w in ['fog', 'mist', 'haze', 'smoke']):
        return 'low_visibility'
    elif any(w in condition for w in ['thunder', 'storm']):
        return 'storm'
    else:
        return 'other'


# =============================================================================
# HINT 4: Road Feature Processing
# =============================================================================

def process_road_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The dataset has 13 boolean road feature columns.

    These are already binary (True/False) and very useful for ML models.
    Consider creating aggregate features:
    - total_road_features: count of road features at the accident location
    - has_traffic_control: any of traffic signal, stop, give way, etc.
    """
    road_features = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
                     'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                     'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

    existing = [f for f in road_features if f in df.columns]

    # Total road features present
    df['n_road_features'] = df[existing].sum(axis=1)

    # Traffic control present
    control_features = ['Traffic_Signal', 'Stop', 'Give_Way', 'Traffic_Calming']
    existing_control = [f for f in control_features if f in df.columns]
    df['has_traffic_control'] = df[existing_control].any(axis=1).astype(int)

    return df


# =============================================================================
# HINT 5: Handling Severity Class Imbalance
# =============================================================================

def analyze_severity_distribution(df: pd.DataFrame):
    """
    Severity distribution is heavily imbalanced:
    - Severity 1: ~1-2% (very rare)
    - Severity 2: ~80% (dominant — this is your biggest challenge)
    - Severity 3: ~12-15%
    - Severity 4: ~5-8%

    This is a MAJOR challenge. If you just predict class 2 for everything,
    you'll get ~80% accuracy but your model is COMPLETELY USELESS.
    Weighted F1 is the real evaluation metric, not accuracy.

    Strategies:
    1. Class weights: Give higher weight to minority classes
       - sklearn: class_weight='balanced'
       - TensorFlow/Keras: class_weight parameter in model.fit()
    2. SMOTE or oversampling for minority classes
    3. Undersampling the majority class (Severity 2)
    4. Consider binary: "severe" (3-4) vs "not severe" (1-2)
    5. Focal loss — designed for class imbalance

    For evaluation: Use weighted F1, not just accuracy.
    Weighted F1 accounts for class imbalance by weighting each class by its support.
    """
    print("Severity Distribution:")
    print(df['Severity'].value_counts().sort_index())
    print(f"\nClass ratios:")
    print(df['Severity'].value_counts(normalize=True).sort_index().round(3))


# =============================================================================
# HINT 6: Pothole Image Preprocessing
# =============================================================================

def get_pothole_image_hints():
    """
    Tips for working with the pothole images:

    NOTE: Images are organized in positive/ (pothole) and negative/ (normal road)
    folders. This is a simple binary classification task.

    1. Images are VERY high resolution (2760x3680px) — RESIZE FIRST
       - Common sizes: 224x224 (ResNet), 128x128 (lighter), 299x299 (Inception)
       - If you try to load full-res images, you'll run out of memory fast

    2. Class distribution — 30/70 IMBALANCE:
       - Normal road (negative/): ~2,658 images (70%)
       - Pothole (positive/): ~1,119 images (30%)
       - This 30/70 imbalance means you need class weights or augmentation
       - A model predicting "normal" for everything gets 70% accuracy — useless

    3. Data augmentation is important:
       - Random horizontal/vertical flips (roads look the same flipped)
       - Random rotation (potholes don't have an "up")
       - Color jitter (different lighting conditions)
       - RandomResizedCrop (zoom in on pothole features)
       - DO NOT use augmentation on test set — only training

    4. Transfer learning recommended:
       - ResNet50, EfficientNet-B0, or MobileNet pre-trained on ImageNet
       - Fine-tune the last few layers on your pothole data
       - Freeze early layers (edge detection, texture features transfer well)

    5. Binary classification:
       - Label 0 = Normal road
       - Label 1 = Pothole
       - Use binary_crossentropy loss in TensorFlow/Keras

    6. Image loading with TensorFlow/Keras:
       from tensorflow.keras.preprocessing.image import ImageDataGenerator

       datagen = ImageDataGenerator(
           rescale=1./255,
           rotation_range=20,
           horizontal_flip=True,
           zoom_range=0.2,
           validation_split=0.2,
       )
       train_gen = datagen.flow_from_directory(
           'pothole_images/train',
           target_size=(224, 224),
           batch_size=32,
           class_mode='binary',
           subset='training',
       )
    """
    pass


# =============================================================================
# HINT 7: 311 Complaint Text Preprocessing
# =============================================================================

def preprocess_311_text(text: str) -> str:
    """
    311 complaint text is real citizen-submitted text — it's messy!

    Common issues:
    - Informal language, abbreviations
    - ALL CAPS (common in complaints)
    - Multiple languages (English, Spanish, Chinese)
    - Very short descriptions ("noise", "pothole")
    - Very long rants (500+ words)
    - Typos and misspellings
    - Address information mixed with complaint text

    For classification:
    1. Use complaint_type as your target label
    2. Use resolution_description as your input text
    3. Focus on the top 5 most common complaint types + an "Other" bucket
       (6 classes total). The top 5 typically cover ~50-60% of complaints.
    4. Map all remaining categories into "Other" — this is a practical
       real-world decision that city operations teams would actually make

    NLP approaches:
    1. TF-IDF + Logistic Regression/SVM (strong baseline!)
    2. Word embeddings + LSTM/GRU
    3. Pre-trained transformers (DistilBERT is fast, BERT is accurate)

    Tip: TF-IDF + Logistic Regression is a surprisingly strong baseline
    for text classification. Start there before going to deep learning.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    # Add your preprocessing steps here
    return text


def get_top_complaint_types(df: pd.DataFrame, n: int = 5) -> list:
    """
    Focus your NLP model on the top 5 complaint types + an "Other" bucket.

    With 186 categories, many have very few examples. Focusing on the top 5
    keeps your model tractable while covering the highest-impact complaint
    types. The "Other" bucket captures the long tail.

    The top 5 311 categories typically cover about 50-60% of complaints.

    The top 5 categories in this dataset are:
    - Illegal Parking
    - HEAT/HOT WATER
    - Noise - Residential
    - Snow or Ice
    - Blocked Driveway
    """
    top_types = df['complaint_type'].value_counts().head(n).index.tolist()
    coverage = df[df['complaint_type'].isin(top_types)].shape[0] / len(df) * 100
    print(f"Top {n} complaint types cover {coverage:.1f}% of all complaints")
    return top_types


def create_complaint_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map complaint types to the top 5 categories + "Other" (6 classes total).

    The top 5 categories are:
    - Illegal Parking
    - HEAT/HOT WATER
    - Noise - Residential
    - Snow or Ice
    - Blocked Driveway

    Everything else maps to "Other". This gives you 6 classes total —
    a much more manageable classification problem than 186 categories.
    """
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


# =============================================================================
# HINT 8: Sampling the Full Dataset
# =============================================================================

def sample_accidents_data(filepath: str, n_samples: int = 500000) -> pd.DataFrame:
    """
    The full traffic accidents dataset has 7.7M records — that's a LOT.

    For this project, we provide a sampled version (~500K records).
    But if you download the full dataset, here's how to sample:

    Important: Use STRATIFIED sampling to preserve severity distribution!
    Random sampling would give you mostly Severity 2.
    """
    df = pd.read_csv(filepath)

    # Stratified sampling preserving severity distribution
    sampled = df.groupby('Severity', group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples // 4), random_state=42)
    )

    return sampled.reset_index(drop=True)


# =============================================================================
# HINT 9: Innovation Model Ideas
# =============================================================================

def innovation_model_hints():
    """
    Your Innovation Model (Model 5) — Your Team's Choice

    This is your chance to surprise us. Identify a problem in the data
    that we DIDN'T ask you to solve, and build a model for it.

    Ideas from these datasets:
    1. Accident Hotspot Prediction
       - Cluster accidents by location to find high-risk intersections
       - Predict which intersections will see the most accidents next month
       - Huge value for proactive safety improvements

    2. Seasonal Complaint Pattern Forecasting
       - Use 311 time series data to predict complaint volumes
       - Help the city pre-position resources (e.g., more crews before pothole season)
       - Time series or regression approach

    3. Response Time Optimization
       - Predict how long 311 complaints take to resolve
       - Identify factors that slow resolution
       - Help dispatch prioritize urgent complaints

    4. Road Deterioration Prediction
       - Combine pothole data with weather/traffic patterns
       - Predict which roads will need maintenance next
       - Shift from reactive to predictive maintenance

    5. Multi-modal Accident Analysis
       - Combine accident text descriptions with structured features
       - NLP + tabular data fusion
       - Richer understanding of accident causes

    Requirements:
    - Clear urban value proposition for why this model matters
    - Defined success metric (you choose what to measure)
    - Cost-benefit estimate (how would this save taxpayer dollars?)
    - Output must match model5_results_template.csv format

    Output columns: id, prediction, confidence, metric_name, metric_value
    """
    pass


# =============================================================================
# HINT 10: Geographic Feature Engineering
# =============================================================================

def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Location matters for accident severity prediction.

    Feature ideas:
    1. State-level patterns (some states have more severe accidents)
    2. Urban vs. rural (can infer from city population or zip code)
    3. Latitude as a proxy for climate (northern = more ice/snow)
    4. Distance from nearest airport (proxy for traffic volume)
    5. Cluster analysis on lat/lng to find accident hotspots

    Warning: Don't use raw lat/lng as features — they're too specific
    and lead to overfitting. Instead, bin them or use for clustering.
    """
    # State-level average severity (target encoding — be careful of leakage!)
    # Only compute on training data, then apply to test

    # Latitude bins (rough climate proxy)
    if 'Start_Lat' in df.columns:
        df['lat_bin'] = pd.cut(df['Start_Lat'], bins=10, labels=False)

    return df
