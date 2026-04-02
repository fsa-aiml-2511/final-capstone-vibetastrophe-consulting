import os
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_IMAGES = Path("/Volumes/Spare31/images").expanduser().resolve()
SAVED_MODEL_DIR = Path("models/model3_vit/saved_model/")

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 6
TRANSFORMER_UNITS = [128, 64]
MLP_HEAD_UNITS = [128, 64]

# ----------------------------
# Augmentation
# ----------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.02),
    tf.keras.layers.RandomZoom(0.1),
])

def load_images(image_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        rotation_range=20,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        seed=42
    )

    val_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        seed=42,
        shuffle=False
    )

    total = train_gen.samples
    class_counts = np.bincount(train_gen.classes)
    class_weight = {
        i: total / (len(class_counts) * count)
        for i, count in enumerate(class_counts)
    }

    print(f"Class weights: {class_weight}")
    return train_gen, val_gen, class_weight


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            print(e)

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # Add augmentation here
    augmented = data_augmentation(inputs)

    patches = Patches(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=PROJECTION_DIM,
            dropout=0.1,
        )(x1, x1)

        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, TRANSFORMER_UNITS, 0.1)

        encoded_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.3)(representation)

    features = mlp(representation, MLP_HEAD_UNITS, 0.3)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def train_model(model, train_data, val_data, class_weight):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

    callbacks_stage1 = [
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=SAVED_MODEL_DIR / "best_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=25,
        callbacks=callbacks_stage1,
        class_weight=class_weight
    )

    return model


def evaluate_model(model, val_data):
    from sklearn.metrics import classification_report, confusion_matrix

    val_data.reset()
    probs = model.predict(val_data, verbose=1).ravel()
    preds = (probs > 0.5).astype(int)
    y_true = val_data.classes

    results = model.evaluate(val_data, verbose=0)

    report = classification_report(y_true, preds, output_dict=True)
    conf_matrix = confusion_matrix(y_true, preds)

    print(f"Validation Accuracy: {report['accuracy']:.4f}")
    print(f"Validation F1 Score: {report['weighted avg']['f1-score']:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Sample Predictions:")
    print(probs[:5])

    return {
        "loss": results[0],
        "accuracy": report["accuracy"],
        "f1_score": report["weighted avg"]["f1-score"],
        "confusion_matrix": conf_matrix,
        "sample_predictions": probs[:5],
    }


def save_model(model):
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / "model.keras")


def main():
    print("Loading and preprocessing images...")

    train_data, val_data, class_weight = load_images(RAW_IMAGES)

    model = build_model()
    model.summary()

    trained_model = train_model(model, train_data, val_data, class_weight)

    evaluate_model(trained_model, val_data)
    save_model(trained_model)


if __name__ == "__main__":
    main()