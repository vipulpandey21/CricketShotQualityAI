"""
Export all skeleton overlays as MP4 videos for all shot classes
"""

import cv2
import numpy as np
from pathlib import Path

# Shot classes
shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

# Skeleton connections
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

total_processed = 0
total_failed = 0

print("=" * 70)
print("EXPORTING ALL SKELETON OVERLAY VIDEOS")
print("=" * 70)

for shot_class in shot_classes:
    print(f"\n📂 Processing: {shot_class}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        print(f"  ⚠️  Folder not found: {data_dir}")
        continue
    
    # Process all MP4 videos in this folder
    video_files = list(data_dir.glob("video*.mp4"))
    
    for video_file in video_files:
        # Find corresponding skeleton file
        skeleton_file = data_dir / f"skeleton_{video_file.stem}.npy"
        
        if not skeleton_file.exists():
            print(f"  ⚠️  Skeleton not found for: {video_file.name}")
            total_failed += 1
            continue
        
        # Load skeleton data
        skeleton_data = np.load(skeleton_file)
        
        # Open video
        cap = cv2.VideoCapture(str(video_file))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video as AVI (guaranteed to work)
        output_video = data_dir / f"skeleton_{video_file.stem}_overlay.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
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
                
                # Add text info
                cv2.putText(frame, f"{shot_class.upper()} - Frame {frame_idx}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"  ✅ {video_file.name} → {output_video.name}")
        total_processed += 1

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Successfully processed: {total_processed} videos")
if total_failed > 0:
    print(f"⚠️  Failed/Skipped: {total_failed} videos")
print(f"\n📁 Skeleton overlay videos saved in each shot class folder")
print("=" * 70)

