"""
Regenerate all skeleton overlay videos with improved batsman detection
Then convert them to MP4 format
"""

import cv2
import os
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames, draw_skeleton

# Shot classes
shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

print("=" * 80)
print("REGENERATING ALL SKELETON OVERLAY VIDEOS WITH ENHANCED DETECTION")
print("=" * 80)

# Step 1: Delete old overlay videos
print("\nStep 1: Deleting old skeleton overlay videos...")
deleted_count = 0
for shot_class in shot_classes:
    data_dir = Path(f"data/{shot_class}")
    if not data_dir.exists():
        continue
    
    # Delete both AVI and MP4 skeleton overlays
    for overlay_file in list(data_dir.glob("skeleton_*_overlay.avi")) + list(data_dir.glob("skeleton_*_overlay.mp4")):
        print(f"  Deleting: {overlay_file}")
        overlay_file.unlink()
        deleted_count += 1

print(f"✓ Deleted {deleted_count} old overlay videos\n")

# Step 2: Generate new overlays
print("=" * 80)
print("Step 2: Generating new skeleton overlay videos (AVI format)")
print("=" * 80)

total_processed = 0
total_failed = 0

for shot_class in shot_classes:
    print(f"\n📂 {shot_class.upper()}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        print(f"  ⚠️  Folder not found")
        continue
    
    # Process only original video files (video1-5.mp4)
    video_files = []
    for i in range(1, 6):
        mp4_file = data_dir / f"video{i}.mp4"
        if mp4_file.exists():
            video_files.append(mp4_file)
    
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
                
                # Add info text with enhanced detection label
                status = "✓ BATSMAN DETECTED" if keypoints is not None else "✗ NO DETECTION"
                color = (0, 255, 0) if keypoints is not None else (0, 0, 255)
                
                cv2.putText(annotated, f"{shot_class.upper()} - {video_file.stem}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, status, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(annotated, f"Frame {frame_idx+1}/{len(frames)}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                out.write(annotated)
            
            out.release()
            
            detection_rate = (detected_count / len(frames)) * 100
            print(f"✅ {detected_count}/{len(frames)} frames ({detection_rate:.1f}%)")
            total_processed += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            total_failed += 1

print("\n" + "=" * 80)
print("Step 3: Converting AVI to MP4...")
print("=" * 80)

converted_count = 0
for shot_class in shot_classes:
    data_dir = Path(f"data/{shot_class}")
    if not data_dir.exists():
        continue
    
    avi_files = list(data_dir.glob("skeleton_*_overlay.avi"))
    
    for avi_file in avi_files:
        mp4_file = avi_file.with_suffix('.mp4')
        print(f"Converting {avi_file.name}...", end=" ", flush=True)
        
        try:
            cap = cv2.VideoCapture(str(avi_file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(mp4_file), fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            
            # Delete AVI after successful conversion
            avi_file.unlink()
            converted_count += 1
            print("✅")
            
        except Exception as e:
            print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"✅ Generated: {total_processed} skeleton overlay videos")
print(f"✅ Converted: {converted_count} to MP4 format")
if total_failed > 0:
    print(f"⚠️  Failed: {total_failed} videos")
print("\n📁 All skeleton overlay videos are now in MP4 format!")
print("   Location: data/{shot_class}/skeleton_video{1-5}_overlay.mp4")
print("\n🎯 Enhanced detection features:")
print("   • Movement variance tracking (striker vs static players)")
print("   • Cricket-specific zone filtering")
print("   • Bat pose detection")
print("   • Side-on stance recognition")
print("=" * 80)
