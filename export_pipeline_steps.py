"""
Export complete pipeline steps for each video
Shows intermediate outputs at each processing stage
"""

import cv2
import numpy as np
import json
from pathlib import Path
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames, draw_skeleton

shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

print("=" * 70)
print("EXPORTING PIPELINE STEPS FOR ALL VIDEOS")
print("=" * 70)

for shot_class in shot_classes:
    print(f"\n📂 {shot_class.upper()}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        continue
    
    # Process each video (1-5)
    for video_num in range(1, 6):
        video_file = data_dir / f"video{video_num}.mp4"
        
        if not video_file.exists():
            continue
        
        print(f"  Processing video{video_num}.mp4...", end=" ", flush=True)
        
        # Create pipeline folder for this video
        pipeline_dir = data_dir / f"video{video_num}_pipeline"
        pipeline_dir.mkdir(exist_ok=True)
        
        # Step 1: Original video metadata
        cap = cv2.VideoCapture(str(video_file))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Step 2: Extract middle 60% frames (30 frames for model)
        frames = extract_raw_frames(str(video_file), max_frames=30)
        
        # Save metadata
        metadata = {
            "video_name": f"video{video_num}.mp4",
            "shot_class": shot_class,
            "original_fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames_in_video": total_frames_raw,
            "extracted_frames_count": len(frames),
            "extraction_method": "middle 60% of video (20%-80% range)",
            "frames_used_for_prediction": 30,
            "skeleton_landmarks": 13,
            "landmark_list": [
                "nose", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle"
            ]
        }
        
        with open(pipeline_dir / "00_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Step 3: Save extracted frames
        frames_dir = pipeline_dir / "01_extracted_frames"
        frames_dir.mkdir(exist_ok=True)
        
        for idx, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{idx:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        
        # Step 4: Run pose detection and save skeleton data
        keypoints_per_frame = run_pose_on_frames(frames)
        
        # Save skeleton keypoints as JSON
        skeleton_data = []
        detected_frames = []
        
        for idx, kp in enumerate(keypoints_per_frame):
            if kp is not None:
                detected_frames.append(idx)
                frame_data = {
                    "frame_index": idx,
                    "landmarks": {}
                }
                # Extract only cricket landmarks (13 joints)
                cricket_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                for lm_idx in cricket_indices:
                    if lm_idx in kp:
                        x, y, z, vis = kp[lm_idx]
                        frame_data["landmarks"][lm_idx] = {
                            "x": float(x),
                            "y": float(y),
                            "z": float(z),
                            "visibility": float(vis)
                        }
                skeleton_data.append(frame_data)
        
        with open(pipeline_dir / "02_skeleton_keypoints.json", "w") as f:
            json.dump({
                "total_frames": len(frames),
                "detected_frames_count": len(detected_frames),
                "detection_rate": f"{len(detected_frames)/len(frames)*100:.1f}%",
                "detected_frame_indices": detected_frames,
                "keypoints_data": skeleton_data
            }, f, indent=2)
        
        # Step 5: Save frames with skeleton overlay
        skeleton_frames_dir = pipeline_dir / "03_skeleton_overlay_frames"
        skeleton_frames_dir.mkdir(exist_ok=True)
        
        for idx, (frame, kp) in enumerate(zip(frames, keypoints_per_frame)):
            annotated = draw_skeleton(frame, kp)
            
            # Add text
            status = "DETECTED" if kp is not None else "NO_DETECTION"
            color = (0, 255, 0) if kp is not None else (0, 0, 255)
            cv2.putText(annotated, f"Frame {idx+1}/{len(frames)} - {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            frame_path = skeleton_frames_dir / f"skeleton_frame_{idx:03d}.jpg"
            cv2.imwrite(str(frame_path), annotated)
        
        # Step 6: Create summary visualization comparing original vs skeleton
        comparison_dir = pipeline_dir / "04_comparison_frames"
        comparison_dir.mkdir(exist_ok=True)
        
        for idx, (frame, kp) in enumerate(zip(frames, keypoints_per_frame)):
            if kp is not None:  # Only save frames where batsman was detected
                # Side-by-side comparison
                annotated = draw_skeleton(frame, kp)
                comparison = np.hstack([frame, annotated])
                
                cv2.putText(comparison, "ORIGINAL", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(comparison, "SKELETON OVERLAY", (width + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                frame_path = comparison_dir / f"comparison_frame_{idx:03d}.jpg"
                cv2.imwrite(str(frame_path), comparison)
        
        # Step 7: Create pipeline summary
        summary = f"""PIPELINE SUMMARY - {shot_class.upper()} - VIDEO {video_num}
{'=' * 60}

INPUT VIDEO
-----------
File: video{video_num}.mp4
Resolution: {width}x{height}
FPS: {fps}
Total Frames in Video: {total_frames_raw}

STEP 1: FRAME EXTRACTION
-------------------------
Method: Extract middle 60% of video (frames 20%-80%)
Frames Extracted: {len(frames)}
Output: 01_extracted_frames/ ({len(frames)} JPG files)

STEP 2: POSE DETECTION
-----------------------
Model: MediaPipe Pose Landmarker (batsman-only detection)
Landmarks Extracted: 13 key body points
Detection Algorithm:
  - Multi-person detection (up to 5 people)
  - Filters for batsman based on:
    * Size (larger person in frame)
    * Position (lower half of frame)
    * Visibility (high confidence)
    * Temporal consistency (same person across frames)
Frames with Batsman Detected: {len(detected_frames)}/{len(frames)}
Detection Rate: {len(detected_frames)/len(frames)*100:.1f}%
Output: 02_skeleton_keypoints.json

STEP 3: SKELETON VISUALIZATION
--------------------------------
Generated skeleton overlay for all frames
Green skeleton lines show body joints
Orange/Cyan dots mark key landmarks
Output: 03_skeleton_overlay_frames/ ({len(frames)} JPG files)

STEP 4: COMPARISON VISUALIZATION
---------------------------------
Side-by-side comparison of original vs skeleton
Only includes frames where batsman was detected
Output: 04_comparison_frames/ ({len(detected_frames)} JPG files)

FRAMES USED FOR PREDICTION
---------------------------
These {len(detected_frames)} frames will be used for:
  1. Shot classification (cover, defense, flick, etc.)
  2. Shot quality scoring
  3. Biomechanical analysis

Frame Indices Used: {detected_frames}

MODEL INPUT SHAPE
-----------------
RGB Frames: ({len(frames)}, {height}, {width}, 3)
Skeleton: ({len(frames)}, 13, 3) - 13 landmarks × (x,y,z) coordinates

{'=' * 60}
"""
        
        with open(pipeline_dir / "PIPELINE_SUMMARY.txt", "w") as f:
            f.write(summary)
        
        print(f"✅ ({len(detected_frames)}/{len(frames)} frames)")

print("\n" + "=" * 70)
print("PIPELINE EXPORT COMPLETE!")
print("=" * 70)
print("\nEach video now has a '_pipeline' folder containing:")
print("  📄 00_metadata.json - Video and extraction info")
print("  📁 01_extracted_frames/ - 30 frames used for prediction")
print("  📄 02_skeleton_keypoints.json - Detected landmarks data")
print("  📁 03_skeleton_overlay_frames/ - Frames with skeleton drawn")
print("  📁 04_comparison_frames/ - Side-by-side original vs skeleton")
print("  📄 PIPELINE_SUMMARY.txt - Complete processing summary")
print("=" * 70)
