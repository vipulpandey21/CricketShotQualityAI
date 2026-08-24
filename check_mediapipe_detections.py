"""
Check if MediaPipe is detecting people at all in later frames
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.video_utils import extract_raw_frames

# Get model
model_path = 'pose_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.2,
    min_pose_presence_confidence=0.2,
    num_poses=5
)

video_path = "data/cover/video1.mp4"
frames = extract_raw_frames(video_path, max_frames=30)

print("=" * 80)
print("MEDIAPIPE DETECTION CHECK - All 30 Frames")
print("=" * 80)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    for frame_idx, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            print(f"Frame {frame_idx + 1:2d}: {len(result.pose_landmarks)} people detected")
            
            for person_idx, landmarks in enumerate(result.pose_landmarks):
                xs = [lm.x for lm in landmarks if lm.visibility > 0.5]
                ys = [lm.y for lm in landmarks if lm.visibility > 0.5]
                
                if len(xs) >= 5:
                    center_x = sum(xs) / len(xs)
                    center_y = sum(ys) / len(ys)
                    avg_vis = sum(lm.visibility for lm in landmarks) / len(landmarks)
                    print(f"         Person {person_idx + 1}: pos=({center_x:.3f}, {center_y:.3f}), vis={avg_vis:.2f}")
        else:
            print(f"Frame {frame_idx + 1:2d}: NO DETECTION")

print("=" * 80)
