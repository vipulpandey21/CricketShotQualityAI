"""
Test the complete pipeline on a sample video
"""

import sys
import os
import numpy as np
from pathlib import Path

# Fix JAX import issue - must be before any TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['JAX_PLATFORMS'] = ''

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.video_utils import extract_frames, extract_raw_frames
from src.pose.estimator import run_pose_on_frames, pose_summary, compute_cricket_angles
from src.quality.scorer import score_shot

# Load model
print("Loading model...")
import tensorflow as tf
# Use keras from tensorflow (not standalone keras package)
keras = tf.keras
models = keras.models
layers = keras.layers
EfficientNetB0 = keras.applications.EfficientNetB0

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base.trainable = False
model = models.Sequential([
    layers.TimeDistributed(base, input_shape=(None, 224, 224, 3)),
    layers.TimeDistributed(layers.GlobalAveragePooling2D()),
    layers.GRU(256, return_sequences=True),
    layers.GRU(128),
    layers.Dense(1024, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])
model.load_weights("model_weights.h5")
print("✓ Model loaded")

# Test video
test_video = "data/pull/video1.avi"
print(f"\nTesting: {test_video}")

# 1. Extract frames
print("Extracting frames...")
frames = extract_frames(test_video, n_frames=30)
print(f"✓ Extracted {len(frames)} frames, shape={frames.shape}")

# 2. Predict shot
print("Predicting shot...")
batch = np.expand_dims(frames, axis=0)
preds = model.predict(batch, verbose=0)
idx = int(np.argmax(preds))
classes = ["cover", "defense", "flick", "hook", "late_cut", "lofted", "pull", "square_cut", "straight", "sweep"]
predicted_shot = classes[idx]
confidence = float(preds[0][idx]) * 100
print(f"✓ Predicted: {predicted_shot} ({confidence:.1f}%)")

# 3. Extract pose
print("Extracting pose...")
raw_frames = extract_raw_frames(test_video, max_frames=30)
frames_kp = run_pose_on_frames(raw_frames)
summary = pose_summary(frames_kp)
print(f"✓ Pose detected in {summary['detected_frames']}/{summary['total_frames']} frames")

# 4. Compute angles
print("Computing angles...")
angles = compute_cricket_angles(summary["avg_keypoints"])
print(f"✓ Angles computed:")
for name, value in angles.items():
    print(f"  - {name}: {value}°" if value else f"  - {name}: Not detected")

# 5. Score quality
print("Scoring quality...")
quality = score_shot(predicted_shot, angles)
print(f"✓ Quality Score: {quality.overall_score:.0f}/100 ({quality.grade})")
print(f"  Criteria breakdown:")
for c in quality.criteria:
    print(f"  - {c.name}: {c.score:.0f}/100 ({c.status})")

print("\n✅ Pipeline test successful!")
