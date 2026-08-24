"""
Export skeleton data as MP4 video (with overlay) and JSON file
Usage: python export_skeleton.py data/pull/video1.mp4
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path

# Check if skeleton file exists
if len(sys.argv) < 2:
    print("Usage: python export_skeleton.py <video_path>")
    print("Example: python export_skeleton.py data/pull/video1.mp4")
    sys.exit(1)

video_path = sys.argv[1]
video_file = Path(video_path)

if not video_file.exists():
    print(f"❌ Video not found: {video_path}")
    sys.exit(1)

# Find corresponding skeleton file
skeleton_file = video_file.parent / f"skeleton_{video_file.stem}.npy"
if not skeleton_file.exists():
    print(f"❌ Skeleton file not found: {skeleton_file}")
    sys.exit(1)

print(f"✓ Found video: {video_path}")
print(f"✓ Found skeleton: {skeleton_file}")

# Load skeleton data
skeleton_data = np.load(skeleton_file)
print(f"✓ Skeleton shape: {skeleton_data.shape}")

# Export to JSON
landmark_names = [
    "nose", "left_shoulder", "right_shoulder", 
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

json_data = {
    "video": str(video_path),
    "frames": int(skeleton_data.shape[0]),
    "landmarks": int(skeleton_data.shape[1]),
    "coordinates": ["x", "y", "z"],
    "landmark_names": landmark_names,
    "data": []
}

for frame_idx in range(skeleton_data.shape[0]):
    frame_data = {
        "frame": frame_idx,
        "landmarks": {}
    }
    for lm_idx, lm_name in enumerate(landmark_names):
        x, y, z = skeleton_data[frame_idx, lm_idx]
        frame_data["landmarks"][lm_name] = {
            "x": float(x),
            "y": float(y),
            "z": float(z)
        }
    json_data["data"].append(frame_data)

# Save JSON
json_output = video_file.parent / f"skeleton_{video_file.stem}.json"
with open(json_output, 'w') as f:
    json.dump(json_data, f, indent=2)
print(f"✅ JSON saved: {json_output}")

# Create MP4 with skeleton overlay
print("\n🎬 Creating video with skeleton overlay...")

cap = cv2.VideoCapture(str(video_path))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
output_video = video_file.parent / f"skeleton_{video_file.stem}_overlay.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

# Skeleton connections (which landmarks to connect)
connections = [
    (0, 1), (0, 2),           # nose to shoulders
    (1, 2),                   # shoulders
    (1, 3), (3, 5),           # left arm
    (2, 4), (4, 6),           # right arm
    (1, 7), (2, 8),           # shoulders to hips
    (7, 8),                   # hips
    (7, 9), (9, 11),          # left leg
    (8, 10), (10, 12),        # right leg
]

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_idx < skeleton_data.shape[0]:
        # Draw skeleton
        h, w = frame.shape[:2]
        
        # Draw connections (lines)
        for connection in connections:
            start_idx, end_idx = connection
            x1, y1, _ = skeleton_data[frame_idx, start_idx]
            x2, y2, _ = skeleton_data[frame_idx, end_idx]
            
            # Convert normalized coords to pixel coords
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            
            # Draw line
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw landmarks (points)
        for lm_idx in range(skeleton_data.shape[1]):
            x, y, _ = skeleton_data[frame_idx, lm_idx]
            pt = (int(x * w), int(y * h))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"✅ Video saved: {output_video}")
print(f"\n📊 Summary:")
print(f"  - Original video: {video_path}")
print(f"  - Skeleton JSON: {json_output}")
print(f"  - Overlay video: {output_video}")
print(f"  - Frames processed: {frame_idx}")

