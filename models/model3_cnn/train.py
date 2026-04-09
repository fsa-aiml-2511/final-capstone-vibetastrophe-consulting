#!/usr/bin/env python3
"""
Model 3: CNN — Training Script
================================
Train a convolutional neural network for image classification.
Transfer learning is recommended (ResNet50, EfficientNet, DenseNet).

Framework: TensorFlow / Keras

IMPORTANT: Resize images before training! Raw images may be very high resolution
and will cause memory errors if loaded full-size.
"""
from gc import callbacks
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

RAW_IMAGES = Path("data/raw/images").resolve()

# RAW_IMAGES = Path("../../../../Spare31/pothole_images/")
SAVED_MODEL_DIR = Path("models/model3_cnn/saved_model/")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42


def load_images(image_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    """Load images and compute class weights for binary classification."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=validation_split,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split)

    CLASS_NAMES = ["negative", "positive"]  # Update with your actual class names based on directory structure

    train_gen = train_datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=RANDOM_SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=RANDOM_SEED,
    )

    class_counts = np.bincount(train_gen.classes)
    class_weight = {
        index: train_gen.samples / (len(class_counts) * count)
        for index, count in enumerate(class_counts)
        if count > 0
    }

    print(f"Class indices: {train_gen.class_indices}")
    print(f"Training samples per class: {class_counts.tolist()}")
    print(f"Class weights: {class_weight}")

    return train_gen, val_gen, class_weight

def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

def build_model():
    """Build an EfficientNetB0 binary classifier."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=IMAGE_SIZE + (3,)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(name="avg_pool"),
            tf.keras.layers.BatchNormalization(name="head_bn"),
            tf.keras.layers.Dropout(0.35, name="dropout_1"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
            tf.keras.layers.Dropout(0.25, name="dropout_2"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="prediction"),
        ],
        name="efficientnet_binary_classifier",
    )
    compile_model(model, learning_rate=3e-4)
    return model


def train_model(model, train_data, val_data, class_weight):
    """Train the CNN in two stages with checkpointing."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = SAVED_MODEL_DIR / "efficientnet_model.keras"

    stage1_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.summary()
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=8,
        callbacks=stage1_callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    base_model = model.get_layer("efficientnetb0")
    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, learning_rate=1e-5)

    stage2_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=stage2_callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    return tf.keras.models.load_model(checkpoint_path)


def evaluate_model(model, val_data):
    """Evaluate CNN performance and tune the decision threshold on validation data."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    val_data.reset()
    metrics = model.evaluate(val_data, verbose=0, return_dict=True)

    val_data.reset()
    y_prob = model.predict(val_data, verbose=0).ravel()
    y_true = val_data.classes

    thresholds = np.arange(0.35, 0.66, 0.05)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    y_pred = (y_prob >= best_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    weighted_f1 = classification_report(y_true, y_pred, output_dict=True, zero_division=0)["weighted avg"]["f1-score"]
    conf_matrix = confusion_matrix(y_true, y_pred)
    sample_predictions = y_prob[:5]

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {metrics['auc']:.4f}")
    print(f"Validation F1 Score: {weighted_f1:.4f}")
    print(f"Best validation threshold: {best_threshold:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sample Predictions:")
    print(sample_predictions)

    return {
        "accuracy": accuracy,
        "auc": metrics["auc"],
        "f1_score": weighted_f1,
        "threshold": best_threshold,
        "confusion_matrix": conf_matrix,
        "sample_predictions": sample_predictions,
    }


def save_model(model):
    """Save the trained model."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "efficientnet_model.keras")


def main():
    # 1. Load and preprocess images
    train_data, val_data, class_weight = load_images(
        RAW_IMAGES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )

    # 2. Build model
    model = build_model()
    
    # 3. Train
    trained_model = train_model(model, train_data, val_data, class_weight)
    
    # 4. Evaluate
    evaluate_model(trained_model, val_data)
    
    # 5. Save
    save_model(trained_model)

    print("Training complete!")


if __name__ == "__main__":
    main()
