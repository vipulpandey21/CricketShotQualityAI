"""
Convert AVI videos to MP4 format
"""

import cv2
from pathlib import Path

def convert_avi_to_mp4(avi_path, mp4_path):
    """Convert single AVI to MP4 with H.264 codec"""
    cap = cv2.VideoCapture(str(avi_path))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25  # default
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(str(mp4_path), fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return frame_count

def convert_all():
    """Convert all AVI files in data/ to MP4"""
    data_path = Path('data')
    
    for class_folder in data_path.iterdir():
        if not class_folder.is_dir():
            continue
        
        print(f"\nProcessing {class_folder.name}...")
        
        avi_files = list(class_folder.glob("*.avi"))
        
        for avi_file in avi_files:
            mp4_file = avi_file.with_suffix('.mp4')
            
            if mp4_file.exists():
                print(f"  Skipping {avi_file.name} (MP4 already exists)")
                continue
            
            print(f"  Converting {avi_file.name}...", end=" ")
            try:
                frames = convert_avi_to_mp4(avi_file, mp4_file)
                print(f"✓ ({frames} frames)")
            except Exception as e:
                print(f"✗ Error: {e}")

if __name__ == "__main__":
    convert_all()
    print("\n✅ Conversion complete!")
