"""
Download cricket shot dataset from HuggingFace using datasets library
"""

import os
import shutil
from pathlib import Path

def download_and_organize():
    """Download from HuggingFace and organize into data/ folders"""
    print("Downloading dataset from HuggingFace...")
    print("Dataset: rokmr/cricket-shot\n")
    
    try:
        from datasets import load_dataset
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset("rokmr/cricket-shot")
        
        print(f"✅ Dataset loaded!")
        print(f"Splits: {list(dataset.keys())}")
        
        # Use train split
        train_data = dataset['train']
        print(f"Train samples: {len(train_data)}")
        
        # Organize by class
        classes = ['cover', 'defense', 'flick', 'hook', 'late_cut', 
                   'lofted', 'pull', 'square_cut', 'straight', 'sweep']
        
        data_path = Path('data')
        sample_per_class = 5
        
        for class_name in classes:
            print(f"\nProcessing {class_name}...")
            
            # Filter samples for this class
            class_samples = [s for s in train_data if s['label'] == classes.index(class_name)]
            
            print(f"  Found {len(class_samples)} samples")
            
            # Take first 5
            for i, sample in enumerate(class_samples[:sample_per_class]):
                video = sample['video']  # This should be video data
                dest_file = data_path / class_name / f"video{i+1}.mp4"
                
                # Save video
                with open(dest_file, 'wb') as f:
                    f.write(video['bytes'])
                
                print(f"    Saved video{i+1}.mp4")
        
        print("\n✅ All done!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAlternative: Manual download")
        print("1. Go to: https://huggingface.co/datasets/rokmr/cricket-shot")
        print("2. Download dataset manually")
        print("3. Extract and place videos in data/ folders")
        return False

if __name__ == "__main__":
    download_and_organize()
