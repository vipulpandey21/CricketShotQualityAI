"""
Test skeleton overlay on a single video to verify batsman-only detection
"""

import cv2
from pathlib import Path
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames, draw_skeleton

# Test on first cover video
video_file = Path("data/cover/video1.mp4")
output_video = Path("data/cover/test_skeleton_overlay.avi")

print(f"Testing batsman detection on: {video_file}")
print("=" * 70)

# Open video
cap = cv2.VideoCapture(str(video_file))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read all frames
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Video info: {width}x{height} @ {fps}fps, {len(frames)} frames")
print(f"Running pose detection...")

# Run pose detection with improved batsman-only detection
keypoints_per_frame = run_pose_on_frames(frames)

# Count detections
detected_count = sum(1 for kp in keypoints_per_frame if kp is not None)
print(f"Batsman detected in {detected_count}/{len(frames)} frames ({detected_count/len(frames)*100:.1f}%)")

# Create output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

print(f"Generating overlay video...")

for frame_idx, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
    annotated = draw_skeleton(frame, keypoints)
    
    # Add info text
    status = "✓ BATSMAN DETECTED" if keypoints is not None else "✗ NO DETECTION"
    color = (0, 255, 0) if keypoints is not None else (0, 0, 255)
    
    cv2.putText(annotated, f"COVER - Frame {frame_idx+1}/{len(frames)}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, status, 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(annotated, f"Detection Rate: {detected_count/len(frames)*100:.1f}%", 
               (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    out.write(annotated)

out.release()

print(f"\n✅ Test complete!")
print(f"📁 Output saved to: {output_video}")
print(f"\nOpen the video to verify:")
print(f"  - Skeleton should only appear on the batsman")
print(f"  - No landmarks on umpire, fielders, or other people")
print(f"  - Consistent tracking across frames")
print("=" * 70)
