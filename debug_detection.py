"""
Debug pose detection to understand what's being detected and filtered
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# Get model path
model_path = "pose_landmarker.task"

# Test on first cover video
video_file = Path("data/cover/video1.mp4")

print(f"Debugging detection on: {video_file}")
print("=" * 70)

# Open video and get first frame
cap = cv2.VideoCapture(str(video_file))
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error reading video")
    exit(1)

print(f"Frame size: {frame.shape}")

# Convert to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# Create PoseLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.3,  # Lower to detect more
    min_pose_presence_confidence=0.3,
    num_poses=5
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)
    
    print(f"\nNumber of people detected: {len(result.pose_landmarks) if result.pose_landmarks else 0}")
    
    if result.pose_landmarks:
        for pose_idx, landmarks in enumerate(result.pose_landmarks):
            print(f"\n--- Person {pose_idx + 1} ---")
            
            # Calculate metrics
            xs = [lm.x for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y for lm in landmarks if lm.visibility > 0.5]
            
            if len(xs) >= 5:
                bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
                avg_vis = sum(lm.visibility for lm in landmarks) / len(landmarks)
                
                print(f"  Bbox area: {bbox_area:.4f}")
                print(f"  Center: ({center_x:.3f}, {center_y:.3f})")
                print(f"  Avg visibility: {avg_vis:.3f}")
                print(f"  Visible landmarks: {len(xs)}/33")
                
                # Check filters
                pass_size = bbox_area > 0.03
                pass_position = center_y > 0.2
                pass_visibility = avg_vis > 0.4
                
                print(f"  Pass size filter (>0.03): {pass_size}")
                print(f"  Pass position filter (y>0.2): {pass_position}")
                print(f"  Pass visibility filter (>0.4): {pass_visibility}")
                print(f"  Overall: {'✓ PASS' if (pass_size and pass_position and pass_visibility) else '✗ FAIL'}")
            else:
                print(f"  Too few visible landmarks: {len(xs)}")
    else:
        print("No poses detected at all!")

print("\n" + "=" * 70)
