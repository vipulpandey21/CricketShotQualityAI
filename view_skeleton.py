"""
View skeleton .npy file structure
"""

import numpy as np
from pathlib import Path

# Load a skeleton file
skeleton_file = "data/pull/skeleton_video1.npy"
skeleton = np.load(skeleton_file)

print("=" * 60)
print("SKELETON FILE FORMAT")
print("=" * 60)
print(f"\nFile: {skeleton_file}")
print(f"Shape: {skeleton.shape}")
print(f"  - {skeleton.shape[0]} frames")
print(f"  - {skeleton.shape[1]} landmarks (joints)")
print(f"  - {skeleton.shape[2]} coordinates (x, y, z)")

print(f"\nData type: {skeleton.dtype}")
print(f"File size: {Path(skeleton_file).stat().st_size} bytes")

print("\n" + "=" * 60)
print("SAMPLE DATA (Frame 0)")
print("=" * 60)

landmark_names = [
    "nose", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

print("\nFrame 0 landmarks (x, y, z):")
for i in range(13):
    x, y, z = skeleton[0, i]
    print(f"  {i:2d}. {landmark_names[i]:15s}: x={x:.4f}, y={y:.4f}, z={z:.4f}")

print("\n" + "=" * 60)
print("FRAMES SUMMARY")
print("=" * 60)

# Check how many frames have non-zero data
valid_frames = 0
for frame_idx in range(skeleton.shape[0]):
    if np.any(skeleton[frame_idx] != 0):
        valid_frames += 1

print(f"Valid frames (with pose data): {valid_frames}/{skeleton.shape[0]}")

# Show frame-by-frame summary
print("\nFrame-by-frame status:")
for frame_idx in range(min(10, skeleton.shape[0])):
    has_data = np.any(skeleton[frame_idx] != 0)
    status = "✓ Has pose" if has_data else "✗ No pose"
    print(f"  Frame {frame_idx:2d}: {status}")

if skeleton.shape[0] > 10:
    print(f"  ... ({skeleton.shape[0] - 10} more frames)")

print("\n" + "=" * 60)
print("USAGE IN FUSION MODEL")
print("=" * 60)
print("""
This skeleton data will be used as:
1. Input shape: (batch_size, 30, 13, 3)
2. Reshaped to: (batch_size, 30, 39) [flatten 13×3]
3. Fed to Dense + LSTM layers
4. Combined with RGB features for final prediction

Each frame has 13 body joints × 3 coordinates = 39 values
""")
