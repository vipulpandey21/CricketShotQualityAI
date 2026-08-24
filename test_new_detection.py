"""
Quick test script to verify new batsman detection logic
Tests on one video and shows detection stats
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames

# Test on one video
video_path = "data/cover/video1.mp4"
print(f"Testing new batsman detection on: {video_path}")
print("=" * 60)

# Extract frames
print("Extracting frames...")
frames = extract_raw_frames(video_path, max_frames=30)
print(f"Extracted {len(frames)} frames")

# Run new detection logic
print("\nRunning batsman detection with movement variance...")
keypoints_list = run_pose_on_frames(frames)

# Calculate detection stats
detected = sum(1 for kp in keypoints_list if kp is not None)
detection_rate = (detected / len(keypoints_list)) * 100

print("\n" + "=" * 60)
print("DETECTION RESULTS:")
print("=" * 60)
print(f"Total frames:     {len(keypoints_list)}")
print(f"Detected frames:  {detected}")
print(f"Detection rate:   {detection_rate:.1f}%")
print("=" * 60)

# Show frame-by-frame detection
print("\nFrame-by-frame detection:")
for i, kp in enumerate(keypoints_list):
    status = "✓ DETECTED" if kp is not None else "✗ NOT DETECTED"
    print(f"  Frame {i+1:2d}: {status}")

if detected > 0:
    print(f"\n✓ Detection working! Rate: {detection_rate:.1f}%")
    if detection_rate >= 80:
        print("  EXCELLENT: Ready to process all videos!")
    elif detection_rate >= 60:
        print("  GOOD: May need minor tuning")
    else:
        print("  WARNING: Detection rate lower than expected")
else:
    print("\n✗ NO DETECTION: Check the logic!")
