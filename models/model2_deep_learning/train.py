#!/usr/bin/env python3
"""
Model 2: Deep Learning — Training Script
==========================================
Train a deep neural network on the same tabular data as Model 1.
Compare performance against your traditional ML model.

Framework: TensorFlow / Keras
"""
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import platform
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model2_deep_learning/saved_model/")

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

def load_data():
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    # TODO: Load your preprocessed dataset
    df = pd.read_csv(PROCESSED_DATA / "processed_city_traffic_accidents.csv")
    return df


def preprocess_features(df):
    """Prepare features for neural network training.

    DNN-specific considerations:
    - Scale all features to [0,1] or standardize (mean=0, std=1)
    - One-hot encode categoricals (or use embedding layers)
    - Convert to numpy arrays or tf.data.Dataset
    """
    target_col = "Severity_Binary"
    test_size = 0.2
    random_state = 42

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
    X = df[features].copy()
    y = df[target_col]

    # Encode any non-numeric columns so scaler/Keras can consume them
    obj_cols = X.select_dtypes(exclude=[np.number]).columns
    if len(obj_cols) > 0:
        X = pd.get_dummies(X, columns=obj_cols, dummy_na=True)

    # Ensure numeric dtypes and remove non-finite values
    X = X.astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Sanity check: report missing values before imputation
    missing_counts = X.isna().sum()
    if missing_counts.any():
        print("Top columns with NaN before fill:")
        print(missing_counts[missing_counts > 0].sort_values(ascending=False).head(20))

    # Impute remaining missing values
    X = X.fillna(0.0)

    # Sanity checks
    if X.isna().any().any():
        raise ValueError("X still contains NaN after fillna.")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("X contains non-finite values (inf or -inf) after cleanup.")

    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if not np.isfinite(X_train_scaled).all() or not np.isfinite(X_test_scaled).all():
        raise ValueError("Scaled feature matrix contains non-finite values.")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values


def build_model(input_dim, num_classes):
    """Define your neural network architecture.

    Example:
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='sigmoid'),
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    IMPORTANT: For class imbalance, use class_weight parameter in model.fit()
    or use a weighted loss function.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping.

    Example:
        from tensorflow.keras.callbacks import EarlyStopping

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            class_weight=class_weights,  # Handle imbalance!
        )
    """

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    print(f"Class weights: {class_weights}")

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weights,  # Handle imbalance!
    )
    return history

def evaluate_model(model, X_val, y_val):
    """Evaluate and compare against Model 1.

    Must include:
    - Classification report
    - Weighted F1 score
    - Training curves (loss and accuracy over epochs)
    - Comparison table: Model 1 vs Model 2 metrics
    """
    
    y_prob = model.predict(X_val).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def save_model(model):
    """Save the trained model to saved_model/.

    Example:
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_MODEL_DIR / "model.keras")
    """
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "model.keras")


def main():
    # 0. Disable GPUs if on a Apple Silicon based Mac
    if is_apple_silicon():
        print("Running on Apple Silicon Mac, disabling the GPU to avoid compatibility issues.")
        # disable_gpus()

    # 1. Load data
    df = load_data()

    # 2. Preprocess features
    X_train, X_val, y_train, y_val = preprocess_features(df)

    # 3. Build model
    model = build_model(input_dim=X_train.shape[1], num_classes=2)

    # 4. Train
    train_model(model, X_train, y_train, X_val, y_val)

    # 5. Evaluate and compare to Model 1
    evaluate_model(model, X_val, y_val)

    # 6. Save
    save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
