#!/usr/bin/env python3
"""
Model 3: CNN — Prediction Script
==================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model3_results.csv
"""
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = Path("models/model3_cnn/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model3_results.csv"
THRESHOLD = 0.50  # Adjust this threshold based on your model's performance (e.g., 0.5 for balanced classes, or lower if you want to catch more positives)


def load_model():
    """Load your trained CNN model from saved_model/.

    TensorFlow / Keras:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH / "model.keras")
    """
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH / "efficientnet_model.keras")
    return model

def load_and_preprocess_images(image_dir):
    """Load images from the test_data/ image folder and apply transforms.

    Example using Keras:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        import numpy as np

        images, ids = [], []
        for img_path in sorted(Path(image_dir).glob("*.png")):
            img = load_img(img_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            ids.append(img_path.name)
        return np.array(images), ids
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np

    images, ids = [], []
    for img_path in sorted(Path(image_dir).glob("*.JPG")):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) # / 255.0
        images.append(img_array)
        ids.append(img_path.name)
    return np.array(images), ids


def predict(model, images, image_ids):
    """Generate predictions on image data.

    Should return a DataFrame with columns: image_id, predicted_class, confidence
    """
    # TODO: Run your model on the images
    y_prob = model.predict(images, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)



    return pd.DataFrame({
        "image_id": image_ids,  # Fill with image IDs
        "predicted_class": y_pred,  # Fill with predicted classes (0 or 1)
        "confidence": y_prob,  # Fill with confidence scores (0.0 to 1.0)
    })


def main():
    # Load model
    model = load_model()

    # Load test images from test_data/ image folder
    images, image_ids = load_and_preprocess_images(TEST_DATA_DIR / "images")

    # Generate predictions
    predictions = predict(model, images, image_ids)

    # Save results — MUST match output template exactly
    results = pd.DataFrame({
        "image_id": image_ids,
        "predicted_class": predictions["predicted_class"],
        "confidence": predictions["confidence"],
    })
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
