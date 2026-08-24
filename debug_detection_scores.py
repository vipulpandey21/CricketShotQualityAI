"""
Debug script to see what scores each detected person gets
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

def movement_variance(person_history, n=5):
    if len(person_history) < 2:
        return 0.0
    recent = person_history[-n:]
    if len(recent) < 2:
        return 0.0
    xs = [pos[0] for pos in recent]
    ys = [pos[1] for pos in recent]
    x_var = np.var(xs) if len(xs) > 1 else 0.0
    y_var = np.var(ys) if len(ys) > 1 else 0.0
    return float(x_var + y_var)

def calculate_shoulder_width(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        return abs(left_shoulder.x - right_shoulder.x)
    return 0.5

def check_bat_pose(landmarks):
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    if left_wrist.visibility < 0.4 or right_wrist.visibility < 0.4:
        return False
    return True

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

# Test on frame 10 (middle of action)
video_path = "data/cover/video1.mp4"
frames = extract_raw_frames(video_path, max_frames=30)
frame = frames[9]  # Frame 10

print("=" * 80)
print("DEBUGGING DETECTION SCORES - Frame 10")
print("=" * 80)

person_histories = {}

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        print(f"\nDetected {len(result.pose_landmarks)} people in frame")
        print("-" * 80)
        
        for person_idx, landmarks in enumerate(result.pose_landmarks):
            print(f"\nPERSON #{person_idx + 1}:")
            
            xs = [lm.x for lm in landmarks if lm.visibility > 0.5]
            ys = [lm.y for lm in landmarks if lm.visibility > 0.5]
            
            if len(xs) < 5:
                print("  SKIPPED: Too few visible landmarks")
                continue
            
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            print(f"  Position: x={center_x:.3f}, y={center_y:.3f}")
            
            # Hard rejects
            if center_y < 0.20:
                print("  ✗ HARD REJECT: y < 0.20 (wicketkeeper/background)")
                continue
            if center_y > 0.72:
                print("  ✗ HARD REJECT: y > 0.72 (umpire/bowler)")
                continue
            
            # Scoring
            score = 0
            breakdown = []
            
            # Zone scoring
            if 0.28 < center_y < 0.65 and 0.38 < center_x < 0.62:
                score += 50
                breakdown.append("Core batsman zone: +50")
            elif 0.28 < center_y < 0.65:
                score += 20
                breakdown.append("Edge of valid zone: +20")
            else:
                score += 5
                breakdown.append("Transition zone: +5")
            
            if 0.35 < center_x < 0.65:
                score += 15
                breakdown.append("Center pitch: +15")
            
            # Movement (simulate with single position)
            person_id = f"10_{person_idx}"
            if person_id not in person_histories:
                person_histories[person_id] = []
            person_histories[person_id].append((center_x, center_y))
            mv = movement_variance(person_histories[person_id], n=5)
            
            if mv > 0.03:
                score += 15
                breakdown.append(f"Moving (mv={mv:.4f}): +15")
            elif mv < 0.01:
                score -= 40
                breakdown.append(f"Static (mv={mv:.4f}): -40")
            else:
                breakdown.append(f"Moderate movement (mv={mv:.4f}): +0")
            
            # Bat pose
            if check_bat_pose(landmarks):
                score += 12
                breakdown.append("Bat pose: +12")
            
            # Wrists above shoulders
            left_wrist = landmarks[15]
            left_shoulder = landmarks[11]
            if (left_wrist.visibility > 0.5 and left_shoulder.visibility > 0.5 and
                left_wrist.y < left_shoulder.y):
                score -= 20
                breakdown.append("Arms raised: -20")
            
            # Crouching
            left_hip = landmarks[23]
            left_knee = landmarks[25]
            if (left_hip.visibility > 0.5 and left_knee.visibility > 0.5 and
                abs(left_hip.y - left_knee.y) < 0.05):
                score -= 15
                breakdown.append("Crouching: -15")
            
            # Side-on stance
            shoulder_width = calculate_shoulder_width(landmarks)
            if shoulder_width < 0.15:
                score += 10
                breakdown.append(f"Side-on stance (width={shoulder_width:.3f}): +10")
            
            # Visibility
            avg_vis = sum(lm.visibility for lm in landmarks) / len(landmarks)
            if avg_vis > 0.5:
                score += 5
                breakdown.append(f"Good visibility ({avg_vis:.2f}): +5")
            
            print(f"\n  SCORE BREAKDOWN:")
            for item in breakdown:
                print(f"    {item}")
            print(f"\n  TOTAL SCORE: {score}")
            print(f"  PASS THRESHOLD (40)? {'✓ YES' if score >= 40 else '✗ NO'}")
    else:
        print("No people detected in frame!")

print("\n" + "=" * 80)
