"""
Shared inference helpers for Model 3 CNN.

Keep web and batch inference on the same preprocessing path so the model sees
the same input distribution everywhere it is used.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.65


def preprocess_uploaded_image(image_source: Any) -> np.ndarray:
    """Convert an uploaded file or image source into a model-ready batch."""
    if hasattr(image_source, "seek"):
        image_source.seek(0)
    image = load_img(
        image_source,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        interpolation="nearest",
    )
    image_array = img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def predict_single_image(model: Any, image: Any, threshold: float = THRESHOLD) -> dict[str, Any]:
    """Run a single-image prediction and return a normalized result payload."""
    image_batch = preprocess_uploaded_image(image)
    probability = float(model.predict(image_batch, verbose=0).ravel()[0])
    predicted_class = int(probability >= threshold)
    label = "Positive" if predicted_class == 1 else "Negative"

    return {
        "predicted_class": predicted_class,
        "label": label,
        "confidence": probability,
        "threshold": threshold,
    }