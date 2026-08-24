"""
Clean up folders - keep only MP4 videos (original + skeleton overlay)
Delete: AVI files, .npy files
Convert: skeleton_*_overlay.avi to MP4
"""

import cv2
from pathlib import Path

shot_classes = [
    "cover", "defense", "flick", "hook", "late_cut",
    "lofted", "pull", "square_cut", "straight", "sweep"
]

print("=" * 70)
print("CLEANING UP FOLDERS - KEEPING ONLY MP4 FILES")
print("=" * 70)

for shot_class in shot_classes:
    print(f"\n📂 {shot_class.upper()}")
    data_dir = Path(f"data/{shot_class}")
    
    if not data_dir.exists():
        continue
    
    # Step 1: Convert skeleton AVI overlay videos to MP4
    avi_overlays = list(data_dir.glob("skeleton_*_overlay.avi"))
    for avi_file in avi_overlays:
        mp4_file = avi_file.with_suffix('.mp4')
        
        if mp4_file.exists():
            print(f"  ✓ {mp4_file.name} already exists")
        else:
            print(f"  Converting {avi_file.name} to MP4...", end=" ")
            
            # Read AVI
            cap = cv2.VideoCapture(str(avi_file))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Write MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(mp4_file), fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            print("✅")
    
    # Step 2: Delete AVI files
    avi_files = list(data_dir.glob("*.avi"))
    for avi_file in avi_files:
        avi_file.unlink()
        print(f"  🗑️  Deleted {avi_file.name}")
    
    # Step 3: Delete .npy files
    npy_files = list(data_dir.glob("*.npy"))
    for npy_file in npy_files:
        npy_file.unlink()
        print(f"  🗑️  Deleted {npy_file.name}")
    
    # Step 4: Show remaining files
    remaining_files = sorted(data_dir.glob("*.mp4"))
    print(f"  📁 Final files ({len(remaining_files)} total):")
    for f in remaining_files:
        print(f"     - {f.name}")

print("\n" + "=" * 70)
print("CLEANUP COMPLETE!")
print("=" * 70)
print("Each folder now has:")
print("  • video1.mp4 to video5.mp4 (original dataset)")
print("  • skeleton_video1_overlay.mp4 to skeleton_video5_overlay.mp4 (playable)")
print("  Total: 10 MP4 files per folder")
print("=" * 70)
