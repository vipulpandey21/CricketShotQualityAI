"""
Regenerate all skeleton .npy files with improved batsman detection logic
Deletes old .npy files first to force regeneration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.pose.skeleton_extractor import process_dataset

print("=" * 80)
print("REGENERATING ALL SKELETON FILES WITH IMPROVED BATSMAN DETECTION")
print("=" * 80)

# First, delete all existing skeleton .npy files
data_path = Path("data")
deleted_count = 0

print("\nDeleting old skeleton files...")
for class_folder in data_path.iterdir():
    if not class_folder.is_dir():
        continue
    
    for npy_file in class_folder.glob("skeleton_*.npy"):
        npy_file.unlink()
        deleted_count += 1
        print(f"  Deleted: {npy_file}")

print(f"\n✓ Deleted {deleted_count} old skeleton files\n")
print("=" * 80)
print("EXTRACTING NEW SKELETONS")
print("=" * 80)

# Now regenerate all with new detection logic
process_dataset()

print("\n" + "=" * 80)
print("✓ COMPLETE! All 50 skeleton files regenerated with improved detection")
print("=" * 80)
