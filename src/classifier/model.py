"""
model.py
Loads the pre-trained EfficientNetB0 + GRU shot classification model.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0

# 10 shot categories from the CricShotClassify / CrickShot10 dataset
SHOT_CLASSES = {
    "cover": 0,
    "defense": 1,
    "flick": 2,
    "hook": 3,
    "late_cut": 4,
    "lofted": 5,
    "pull": 6,
    "square_cut": 7,
    "straight": 8,
    "sweep": 9,
}

IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}


def build_model() -> tf.keras.Model:
    """
    Reconstruct the EfficientNetB0 + GRU architecture.
    Must match the architecture used when model_weights.h5 was saved.
    """
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.TimeDistributed(base_model, input_shape=(None, 224, 224, 3)),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.GRU(256, return_sequences=True),
            layers.GRU(128),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


def load_classifier(weights_path: str) -> tf.keras.Model:
    """Load the classifier with pre-trained weights."""
    model = build_model()
    model.load_weights(weights_path)
    return model


def predict_shot(
    model: tf.keras.Model, frames: np.ndarray
) -> tuple[str, float]:
    """
    Predict the shot class for a sequence of frames.

    Args:
        model:  Loaded classifier model.
        frames: np.ndarray of shape (n_frames, H, W, 3).

    Returns:
        (shot_name, confidence_percent)
    """
    batch = np.expand_dims(frames, axis=0)          # (1, n_frames, H, W, 3)
    predictions = model.predict(batch, verbose=0)   # (1, 10)
    idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][idx]) * 100.0
    return IDX_TO_CLASS[idx], confidence


def get_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Return a sub-model that outputs the Dense-1024 layer activations.
    Used for cosine similarity comparison between two shots.
    """
    # layers[-3] is the Dense(1024) layer (before Dropout and final softmax)
    return tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)


def cosine_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D feature vectors."""
    dot = np.dot(features1.flatten(), features2.flatten())
    norm = np.linalg.norm(features1) * np.linalg.norm(features2)
    if norm == 0:
        return 0.0
    return float(dot / norm)
