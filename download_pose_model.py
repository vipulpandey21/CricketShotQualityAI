"""
Download MediaPipe Pose Landmarker model
"""
import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"

print(f"Downloading MediaPipe Pose model...")
print(f"URL: {MODEL_URL}")

urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print(f"✓ Model downloaded to {MODEL_PATH}")
print(f"File size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB")
