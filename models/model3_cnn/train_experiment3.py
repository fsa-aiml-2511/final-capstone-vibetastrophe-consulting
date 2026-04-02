from gc import callbacks
import os
from pathlib import Path
from xml.parsers.expat import model
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_IMAGES = Path("/Volumes/Spare31/images").expanduser().resolve()

# RAW_IMAGES = Path("../../../../Spare31/pothole_images/")
SAVED_MODEL_DIR = Path("models/model3_cnn/saved_model/")

def load_images(image_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    """Load and preprocess images, returning (train_gen, val_gen).

    Expects image_dir to contain one subdirectory per class, e.g.:
        images/
            positive/   ← pothole images
            negative/   ← no-pothole images

    The train/val split is handled internally by ImageDataGenerator using a
    deterministic shuffle (seed=42), so both generators see the same split.

    Returns:
        train_gen: training data generator
        val_gen:   validation data generator
        class_weight: dict to pass to model.fit() to handle class imbalance
    """

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        # Add your augmentation here - it will only apply to the training subset
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training", # This tells Keras which part of the split to use
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation", # This tells Keras to use the remaining 20%
        seed=42
    )

    # Compute class weights to handle imbalance between positive/negative images
    total = train_gen.samples
    class_counts = np.bincount(train_gen.classes)
    class_weight = {
        i: total / (len(class_counts) * count)
        for i, count in enumerate(class_counts)
    }

    print(f"Class weights: {class_weight}")

    return train_gen, val_gen, class_weight


def build_model():
    """Build or fine-tune a CNN.

    Transfer learning example:
        import tensorflow as tf

        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze base layers initially

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
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


    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet' 
    )

    # Freeze the base to protect pre-trained patterns
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer= tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # ,
    # tf.keras.metrics.AUC(name='auc'),
    # tf.keras.metrics.Precision(name='precision'),
    # tf.keras.metrics.Recall(name='recall'),

    return model


def train_model(model, train_data, val_data, class_weight):
    """Train the CNN with callbacks.

    Use EarlyStopping and optionally ReduceLROnPlateau.
    Pass class_weight to model.fit() to handle imbalance.
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks_stage1 = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Stage 1: train head only
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=25,
        callbacks=callbacks_stage1,
        class_weight=class_weight
    )

    return model


def evaluate_model(model, val_data):
    """Evaluate CNN performance.

    Must include:
    - Accuracy and weighted F1
    - Confusion matrix
    - Sample predictions with images

    Bonus: Grad-CAM visualizations showing what the model "sees"
    """

    from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

    accuracy = model.evaluate(val_data)[1]  # Assuming 'accuracy' is the second metric
    f1_score = classification_report(val_data.classes, model.predict(val_data) > 0.5, output_dict=True)['weighted avg']['f1-score']
    conf_matrix = confusion_matrix(val_data.classes, model.predict(val_data) > 0.5)
    sample_predictions = model.predict(val_data)[:5]  # Get predictions for first 5 samples

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {f1_score:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sample Predictions:")
    print(sample_predictions)

    y_true = []
    y_prob = []

    # for images, labels in val_data:
    #     probs = model.predict(images, verbose=0).flatten()
    #     y_prob.extend(probs)
    #     y_true.extend(labels.numpy().flatten())

    # y_true = np.array(y_true)
    # y_prob = np.array(y_prob)

    # for threshold in [0.2, 0.3, 0.4, 0.5]:
    #     y_pred = (y_prob >= threshold).astype(int)
    #     print(
    #         f"threshold={threshold:.1f}",
    #         f"precision={precision_score(y_true, y_pred, zero_division=0):.4f}",
    #         f"recall={recall_score(y_true, y_pred, zero_division=0):.4f}",
    #         f"f1={f1_score(y_true, y_pred, zero_division=0):.4f}"
    #     )

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "confusion_matrix": conf_matrix,
        "sample_predictions": sample_predictions,
    }


def save_model(model):
    """Save the trained model.

    Example:
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_MODEL_DIR / "model.keras")
    """
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "efficientnet_model.keras")

def main():
    print("Loading and preprocessing images...")

    train_data, val_data, class_weight = load_images(RAW_IMAGES)

    model = build_model()

    trained_model = train_model(model, train_data, val_data, class_weight)

    evaluate_model(trained_model, val_data)

    save_model(trained_model)


if __name__ == "__main__":
    main()
