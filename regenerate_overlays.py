"""
Regenerate skeleton overlay videos using improved batsman detection
Processes directly from video files with real-time pose detection visualization
"""

import cv2
import numpy as np
from pathlib import Path
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames, draw_skeleton

# Shot classes
shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

total_processed = 0
total_failed = 0

print("=" * 70)
print("REGENERATING SKELETON OVERLAY VIDEOS (BATSMAN-ONLY)")
print("=" * 70)

for shot_class in shot_classes:
    print(f"\n📂 Processing: {shot_class}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        print(f"  ⚠️  Folder not found: {data_dir}")
        continue
    
    # Process only original video files (not skeleton overlays)
    video_files = [f for f in data_dir.glob("video*.mp4") if not f.name.startswith("skeleton_")]
    video_files += [f for f in data_dir.glob("video*.avi") if not f.name.startswith("skeleton_")]
    
    for video_file in sorted(video_files):
        output_video_mp4 = data_dir / f"skeleton_{video_file.stem}_overlay.mp4"
        output_video_avi = data_dir / f"skeleton_{video_file.stem}_overlay.avi"
        
        print(f"  Processing {video_file.name}...")
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read all frames
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            
            # Run pose detection with improved batsman-only detection
            print(f"    Running pose detection on {len(frames)} frames...")
            keypoints_per_frame = run_pose_on_frames(frames)
            
            # Create output video (AVI format for reliability)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_video_avi), fourcc, fps, (width, height))
            
            # Draw skeleton on each frame
            detected_count = sum(1 for kp in keypoints_per_frame if kp is not None)
            print(f"    Batsman detected in {detected_count}/{len(frames)} frames")
            
            for frame_idx, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
                annotated = draw_skeleton(frame, keypoints)
                
                # Add info text
                status = "✓ BATSMAN DETECTED" if keypoints is not None else "✗ NO DETECTION"
                color = (0, 255, 0) if keypoints is not None else (0, 0, 255)
                cv2.putText(annotated, f"{shot_class.upper()} - Frame {frame_idx+1}/{len(frames)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, status, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                out.write(annotated)
            
            out.release()
            
            print(f"    ✅ Saved: {output_video_avi.name}")
            total_processed += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            total_failed += 1

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Successfully processed: {total_processed} videos")
if total_failed > 0:
    print(f"⚠️  Failed: {total_failed} videos")
print(f"\n📁 Skeleton overlay videos saved as AVI files in each folder")
print("=" * 70)
