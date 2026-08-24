"""
Generate skeleton overlay videos for all shot classes
Saves improved skeleton videos in each data folder
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
print("GENERATING SKELETON OVERLAY VIDEOS (BATSMAN-ONLY)")
print("=" * 70)

for shot_class in shot_classes:
    print(f"\n📂 {shot_class.upper()}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        print(f"  ⚠️  Folder not found")
        continue
    
    # Process only original video files (video1-5.mp4 or .avi)
    video_files = []
    for i in range(1, 6):
        mp4_file = data_dir / f"video{i}.mp4"
        avi_file = data_dir / f"video{i}.avi"
        if mp4_file.exists():
            video_files.append(mp4_file)
        elif avi_file.exists():
            video_files.append(avi_file)
    
    for video_file in sorted(video_files):
        output_video = data_dir / f"skeleton_{video_file.stem}_overlay.avi"
        
        print(f"  Processing {video_file.name}...", end=" ", flush=True)
        
        try:
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
            
            # Run pose detection with improved batsman-only detection
            keypoints_per_frame = run_pose_on_frames(frames)
            
            # Create output video (AVI format)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            # Draw skeleton on each frame
            detected_count = sum(1 for kp in keypoints_per_frame if kp is not None)
            
            for frame_idx, (frame, keypoints) in enumerate(zip(frames, keypoints_per_frame)):
                annotated = draw_skeleton(frame, keypoints)
                
                # Add info text
                status = "✓ BATSMAN" if keypoints is not None else "✗ NO DETECTION"
                color = (0, 255, 0) if keypoints is not None else (0, 0, 255)
                
                cv2.putText(annotated, f"{shot_class.upper()} - {video_file.stem}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, status, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                out.write(annotated)
            
            out.release()
            
            print(f"✅ ({detected_count}/{len(frames)} frames)")
            total_processed += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            total_failed += 1

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✅ Successfully generated: {total_processed} overlay videos")
if total_failed > 0:
    print(f"⚠️  Failed: {total_failed} videos")
print(f"\n📁 Skeleton overlay videos saved in each shot class folder as:")
print(f"   skeleton_video1_overlay.avi")
print(f"   skeleton_video2_overlay.avi")
print(f"   skeleton_video3_overlay.avi")
print(f"   skeleton_video4_overlay.avi")
print(f"   skeleton_video5_overlay.avi")
print("=" * 70)
