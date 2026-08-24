"""
Regenerate all pipeline folders with enhanced batsman detection
Creates complete processing pipeline for each video
"""

import cv2
import json
import numpy as np
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames, draw_skeleton

shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

print("=" * 80)
print("REGENERATING ALL PIPELINE FOLDERS WITH ENHANCED DETECTION")
print("=" * 80)

# Step 1: Delete old pipeline folders
print("\nStep 1: Deleting old pipeline folders...")
deleted_count = 0
for shot_class in shot_classes:
    data_dir = Path(f"data/{shot_class}")
    if not data_dir.exists():
        continue
    
    for pipeline_folder in data_dir.glob("video*_pipeline"):
        if pipeline_folder.is_dir():
            print(f"  Deleting: {pipeline_folder}")
            shutil.rmtree(pipeline_folder)
            deleted_count += 1

print(f"✓ Deleted {deleted_count} old pipeline folders\n")

# Step 2: Generate new pipeline folders
print("=" * 80)
print("Step 2: Generating new pipeline folders with enhanced detection")
print("=" * 80)

total_processed = 0
total_failed = 0

for shot_class in shot_classes:
    print(f"\n{shot_class.upper()}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        print(f"  ⚠️  Folder not found")
        continue
    
    # Process video1-5.mp4
    for i in range(1, 6):
        video_file = data_dir / f"video{i}.mp4"
        if not video_file.exists():
            continue
        
        pipeline_folder = data_dir / f"video{i}_pipeline"
        print(f"  Processing {video_file.name}...", end=" ", flush=True)
        
        try:
            # Create pipeline folder structure
            pipeline_folder.mkdir(exist_ok=True)
            frames_dir = pipeline_folder / "01_extracted_frames"
            overlay_dir = pipeline_folder / "03_skeleton_overlay_frames"
            comparison_dir = pipeline_folder / "04_comparison_frames"
            
            frames_dir.mkdir(exist_ok=True)
            overlay_dir.mkdir(exist_ok=True)
            comparison_dir.mkdir(exist_ok=True)
            
            # Get video metadata
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            # Extract 30 frames from middle 60%
            extracted_frames = extract_raw_frames(str(video_file), max_frames=30)
            
            # Run enhanced pose detection
            keypoints_per_frame = run_pose_on_frames(extracted_frames)
            
            # Count detections
            detected_frames = sum(1 for kp in keypoints_per_frame if kp is not None)
            detection_rate = (detected_frames / len(keypoints_per_frame)) * 100
            
            # Save metadata
            metadata = {
                "video_file": video_file.name,
                "shot_class": shot_class,
                "total_frames": total_frames,
                "fps": round(fps, 2),
                "duration_seconds": round(duration, 2),
                "resolution": f"{width}x{height}",
                "extracted_frames": len(extracted_frames),
                "detection_method": "Enhanced batsman detection with movement variance",
                "detected_frames": detected_frames,
                "detection_rate": round(detection_rate, 1)
            }
            
            with open(pipeline_folder / "00_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Process each frame
            skeleton_data = []
            
            for frame_idx, (frame, keypoints) in enumerate(zip(extracted_frames, keypoints_per_frame)):
                # 1. Save original extracted frame
                frame_filename = f"frame_{frame_idx+1:03d}.jpg"
                cv2.imwrite(str(frames_dir / frame_filename), frame)
                
                # 2. Save skeleton keypoints to list
                if keypoints is not None:
                    frame_data = {
                        "frame_index": frame_idx,
                        "detected": True,
                        "landmarks": {}
                    }
                    
                    # Cricket landmarks (13 joints)
                    cricket_indices = {
                        0: "nose",
                        11: "left_shoulder", 12: "right_shoulder",
                        13: "left_elbow", 14: "right_elbow",
                        15: "left_wrist", 16: "right_wrist",
                        23: "left_hip", 24: "right_hip",
                        25: "left_knee", 26: "right_knee",
                        27: "left_ankle", 28: "right_ankle"
                    }
                    
                    for idx, name in cricket_indices.items():
                        if idx in keypoints:
                            x, y, z, vis = keypoints[idx]
                            frame_data["landmarks"][name] = {
                                "x": round(float(x), 4),
                                "y": round(float(y), 4),
                                "z": round(float(z), 4),
                                "visibility": round(float(vis), 4)
                            }
                    
                    skeleton_data.append(frame_data)
                else:
                    skeleton_data.append({
                        "frame_index": frame_idx,
                        "detected": False,
                        "landmarks": {}
                    })
                
                # 3. Save skeleton overlay frame (only if detected)
                if keypoints is not None:
                    overlay_frame = draw_skeleton(frame, keypoints)
                    cv2.imwrite(str(overlay_dir / frame_filename), overlay_frame)
                    
                    # 4. Save comparison frame (side-by-side)
                    comparison = np.hstack([frame, overlay_frame])
                    cv2.imwrite(str(comparison_dir / frame_filename), comparison)
            
            # Save skeleton keypoints JSON
            with open(pipeline_folder / "02_skeleton_keypoints.json", 'w') as f:
                json.dump({
                    "video": video_file.name,
                    "total_frames": len(skeleton_data),
                    "detected_frames": detected_frames,
                    "detection_rate": round(detection_rate, 1),
                    "frames": skeleton_data
                }, f, indent=2)
            
            # Create pipeline summary (ASCII only for Windows compatibility)
            status_text = "EXCELLENT" if detection_rate >= 70 else "GOOD" if detection_rate >= 50 else "MODERATE"
            
            summary = f"""CRICKET SHOT PROCESSING PIPELINE SUMMARY
{"=" * 80}

VIDEO INFORMATION:
  File: {video_file.name}
  Shot Class: {shot_class}
  Total Frames: {total_frames}
  FPS: {fps:.2f}
  Duration: {duration:.2f}s
  Resolution: {width}x{height}

ENHANCED DETECTION:
  Method: Cricket-specific batsman detection with movement variance
  Features:
    - Movement variance tracking (striker vs static players)
    - Zone filtering (reject keeper y<0.18, umpire y>0.75)
    - Bat pose detection (wrist visibility + elbow angles)
    - Side-on stance recognition (narrow shoulder width)
    - Temporal consistency (same person tracking)

EXTRACTION:
  Extracted Frames: {len(extracted_frames)} (from middle 60% of video)
  Frame Range: Frame {int(total_frames*0.2)} to {int(total_frames*0.8)}
  
DETECTION RESULTS:
  Detected Frames: {detected_frames}/{len(extracted_frames)}
  Detection Rate: {detection_rate:.1f}%
  Status: {status_text}

PIPELINE OUTPUTS:
  - 00_metadata.json                    Video and processing metadata
  - 01_extracted_frames/                {len(extracted_frames)} original frames (used for predictions)
  - 02_skeleton_keypoints.json          Detected landmarks (13 joints per frame)
  - 03_skeleton_overlay_frames/         {detected_frames} frames with skeleton visualization
  - 04_comparison_frames/               {detected_frames} side-by-side comparisons
  - PIPELINE_SUMMARY.txt                This summary file

LANDMARKS DETECTED (13 joints):
  - Nose (head position)
  - Left/Right Shoulder
  - Left/Right Elbow
  - Left/Right Wrist (bat grip)
  - Left/Right Hip
  - Left/Right Knee (footwork)
  - Left/Right Ankle (stance)

{"=" * 80}
Generated with Enhanced Batsman Detection
"""
            
            with open(pipeline_folder / "PIPELINE_SUMMARY.txt", 'w') as f:
                f.write(summary)
            
            print(f" OK {detection_rate:.1f}%")
            total_processed += 1
            
        except Exception as e:
            print(f"ERROR: {e}")
            total_failed += 1

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Generated: {total_processed} pipeline folders")
if total_failed > 0:
    print(f"Failed: {total_failed} pipelines")

print("\nEach pipeline folder contains:")
print("   - 00_metadata.json - Video info + detection stats")
print("   - 01_extracted_frames/ - 30 frames for predictions")
print("   - 02_skeleton_keypoints.json - Enhanced detection keypoints")
print("   - 03_skeleton_overlay_frames/ - Frames with skeleton")
print("   - 04_comparison_frames/ - Side-by-side comparisons")
print("   - PIPELINE_SUMMARY.txt - Complete summary")

print("\nAll pipelines now use enhanced batsman detection!")
print("=" * 80)
