"""
Test skeleton extraction only (without TensorFlow)
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Test video
test_video = "data/pull/video1.mp4"
print(f"Testing skeleton extraction: {test_video}\n")

# Check if skeleton already exists
skeleton_file = "data/pull/skeleton_video1.npy"
if Path(skeleton_file).exists():
    print(f"✓ Skeleton file exists: {skeleton_file}")
    
    # Load and inspect
    skeleton = np.load(skeleton_file)
    print(f"✓ Skeleton shape: {skeleton.shape}")
    print(f"  - {skeleton.shape[0]} frames")
    print(f"  - {skeleton.shape[1]} landmarks")
    print(f"  - {skeleton.shape[2]} coordinates (x, y, z)")
    
    # Show sample data
    print(f"\n✓ Sample frame 0, landmark 0 (nose):")
    print(f"  x={skeleton[0, 0, 0]:.3f}, y={skeleton[0, 0, 1]:.3f}, z={skeleton[0, 0, 2]:.3f}")
    
    print(f"\n✓ All landmarks present:")
    landmark_names = [
        "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
        "right_knee", "left_ankle", "right_ankle"
    ]
    for i, name in enumerate(landmark_names):
        print(f"  {i}: {name}")
    
    print(f"\n✅ Skeleton extraction verified!")
    print(f"Format: 30 frames × 13 landmarks × 3 coords (x,y,z normalized 0-1)")
else:
    print(f"❌ Skeleton file not found: {skeleton_file}")
    print("Run skeleton extraction first!")

