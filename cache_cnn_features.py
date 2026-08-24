"""
cache_cnn_features.py
Precompute the frozen EfficientNetB0 features for the whole dataset, once.

Why: in this architecture the EfficientNetB0 backbone is frozen
(`base_model.trainable = False` in src/classifier/model.py), so it produces
the same numbers every epoch. Running it repeatedly is the entire cost of
training on a CPU. Computing it once and caching lets the trainable part —
the GRU head — train in seconds instead of hours, so the head can actually
be tuned properly.

Output (features/ directory):
    <split>_X.npy      float32 (n_videos, 30, 1280)
    <split>_y.npy      int32   (n_videos,)
    <split>_paths.txt  one source video path per row, same order

Usage:
    python cache_cnn_features.py                  # train, val, test + demo
    python cache_cnn_features.py test demo        # only these
"""

import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.video_utils import extract_frames          # noqa: E402
from src.classifier.model import SHOT_CLASSES             # noqa: E402

CLASSES = list(SHOT_CLASSES)
DATASET = ROOT / "temp_hf_data" / "cricketshot"
OUT = ROOT / "features"
N_FRAMES = 30
BATCH = 32
EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos(folder: Path) -> list:
    """
    Video files in `folder`, excluding macOS AppleDouble sidecars.

    Those "._name.avi" files sit alongside every real video in this dataset
    and carry a video extension, but they are metadata: OpenCV opens them to
    zero frames. Left in, they double the apparent dataset size and feed the
    model 30 black frames per "video".
    """
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in EXTS and not p.name.startswith("._"))


def split_items(split: str) -> list:
    """Return [(video_path, class_index)] for a split name."""
    items = []
    if split == "demo":
        for ci, cls in enumerate(CLASSES):
            for i in range(1, 6):
                p = ROOT / "data" / cls / f"video{i}.mp4"
                if p.exists():
                    items.append((p, ci))
    else:
        for ci, cls in enumerate(CLASSES):
            for p in list_videos(DATASET / split / cls):
                items.append((p, ci))
    return items


def build_feature_extractor():
    """EfficientNetB0 -> GlobalAveragePooling, per frame. 1280-d output."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inp = tf.keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inp, x)


def main():
    splits = sys.argv[1:] or ["train", "val", "test", "demo"]
    OUT.mkdir(exist_ok=True)
    extractor = build_feature_extractor()

    for split in splits:
        items = split_items(split)
        if not items:
            print(f"{split}: no videos found, skipping")
            continue

        print(f"\n=== {split}: {len(items)} videos ===")
        X = np.zeros((len(items), N_FRAMES, 1280), dtype=np.float32)
        y = np.zeros(len(items), dtype=np.int32)
        t0 = time.time()

        for n, (path, ci) in enumerate(items):
            frames = extract_frames(str(path), n_frames=N_FRAMES)  # uint8 RGB
            feats = extractor.predict(frames, batch_size=BATCH, verbose=0)
            X[n] = feats
            y[n] = ci
            if (n + 1) % 25 == 0 or n + 1 == len(items):
                done = n + 1
                rate = done / (time.time() - t0)
                eta = (len(items) - done) / max(rate, 1e-6) / 60
                print(f"  {done}/{len(items)}  {rate:.2f} vid/s  eta {eta:.1f} min")

        np.save(OUT / f"{split}_X.npy", X)
        np.save(OUT / f"{split}_y.npy", y)
        (OUT / f"{split}_paths.txt").write_text(
            "\n".join(str(p) for p, _ in items), encoding="utf-8")
        print(f"  saved {split}_X.npy {X.shape}  ({time.time()-t0:.0f}s)")

    print(f"\nfeatures written to {OUT}")


if __name__ == "__main__":
    main()
