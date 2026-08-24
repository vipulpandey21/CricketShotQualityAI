"""
skeleton_extractor.py
Batch extract and save skeleton keypoints from videos
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.video_utils import extract_raw_frames
from src.pose.estimator import run_pose_on_frames


def extract_skeleton_from_video(video_path, max_frames=30):
    """Extract skeleton keypoints from video and return as numpy array"""
    frames = extract_raw_frames(video_path, max_frames=max_frames)
    keypoints_list = run_pose_on_frames(frames)
    
    # Convert to numpy array (30, 13, 3) - only x,y,z, skip visibility
    skeleton_array = []
    for frame_kp in keypoints_list:
        if frame_kp is None:
            # If no pose detected, use zeros
            skeleton_array.append(np.zeros((13, 3)))
        else:
            # Extract only cricket landmarks (13 joints)
            cricket_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            frame_coords = []
            for idx in cricket_indices:
                if idx in frame_kp:
                    x, y, z, vis = frame_kp[idx]
                    frame_coords.append([x, y, z])
                else:
                    frame_coords.append([0, 0, 0])
            skeleton_array.append(frame_coords)
    
    return np.array(skeleton_array, dtype=np.float32)


def process_dataset(data_dir='data'):
    """Process all videos in data/ folder and save skeleton .npy files"""
    data_path = Path(data_dir)
    
    for class_folder in data_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        print(f"\nProcessing {class_folder.name}...")
        
        # Only look for original video files (not skeleton overlays)
        video_files = [f for f in class_folder.glob("*.mp4") if not f.name.startswith("skeleton_")]
        video_files += [f for f in class_folder.glob("*.avi") if not f.name.startswith("skeleton_")]
        
        if not video_files:
            print(f"  No videos found in {class_folder.name}")
            continue
        
        for video_file in sorted(video_files):
            skeleton_file = video_file.with_name(f"skeleton_{video_file.stem}.npy")
            
            if skeleton_file.exists():
                print(f"  Skipping {video_file.name} (already processed)")
                continue
            
            print(f"  Extracting {video_file.name}...")
            try:
                skeleton = extract_skeleton_from_video(str(video_file))
                np.save(skeleton_file, skeleton)
                print(f"    ✓ Saved {skeleton_file.name} shape={skeleton.shape}")
            except Exception as e:
                print(f"    ✗ Error processing {video_file.name}: {e}")


if __name__ == "__main__":
    process_dataset()
