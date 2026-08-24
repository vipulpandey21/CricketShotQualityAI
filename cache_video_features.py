"""
cache_video_features.py
Cache features from a Kinetics-400 pretrained 3D CNN (r3d_18).

Why not EfficientNetB0: it is an ImageNet classifier, so its features
describe what a single frame LOOKS like. A cricket shot is defined by how
the body and bat MOVE, and every head tried on top of those per-frame
features plateaued at 58-60% test top-1 — including fusing in the striker's
skeleton, which changed top-1 by under 4 points. r3d_18 is trained on
Kinetics-400 video action recognition, so motion is what its features encode
in the first place. That is the right inductive bias for "sweep vs pull".

Frames are taken the same way the rest of the pipeline takes them —
consecutive from frame 0 — and fed as overlapping 16-frame windows, giving a
short sequence of 512-d descriptors per clip rather than a single vector, so
a head can still model the order of the shot.

Output (features/):
    <split>_vid<suffix>.npy         float32 (n_videos, n_windows, 512)
    <split>_vid<suffix>_paths.txt

Usage:
    python cache_video_features.py val demo --per-class 40 train
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.video import R3D_18_Weights, r3d_18

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cache_pose_features import split_items      # noqa: E402

FEAT = ROOT / "features"
N_FRAMES = 30
WINDOW = 16
STRIDE = 7
# Kinetics preprocessing, from R3D_18_Weights.KINETICS400_V1.transforms()
RESIZE = (171, 128)          # (w, h)
CROP = 112
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def read_clip(video_path, n=N_FRAMES):
    """n consecutive frames from frame 0, preprocessed for r3d_18."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, RESIZE, interpolation=cv2.INTER_LINEAR)
        y0 = (f.shape[0] - CROP) // 2
        x0 = (f.shape[1] - CROP) // 2
        f = f[y0:y0 + CROP, x0:x0 + CROP]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append((f - MEAN) / STD)
    cap.release()
    if not frames:
        return None
    while len(frames) < WINDOW:          # pad very short clips
        frames.append(frames[-1])
    return np.stack(frames)              # (T, H, W, 3)


def windows_of(clip):
    starts = list(range(0, max(1, len(clip) - WINDOW + 1), STRIDE))
    return [clip[s:s + WINDOW] for s in starts
            if len(clip[s:s + WINDOW]) == WINDOW] or [clip[:WINDOW]]


def main():
    per_class = None
    if "--per-class" in sys.argv:
        per_class = int(sys.argv[sys.argv.index("--per-class") + 1])
    splits = [a for a in sys.argv[1:]
              if not a.startswith("--") and not a.isdigit()] or ["val"]

    FEAT.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.eval()
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))

    for split in splits:
        items = split_items(split, per_class if split != "demo" else None)
        if not items:
            print(f"{split}: nothing found")
            continue
        suffix = f"_n{per_class}" if per_class and split != "demo" else ""
        print(f"\n=== video feats {split}{suffix}: {len(items)} videos ===")

        rows, t0 = [], time.time()
        for n, (path, _) in enumerate(items):
            clip = read_clip(path)
            if clip is None:
                rows.append(np.zeros((3, 512), dtype=np.float32))
            else:
                wins = windows_of(clip)
                # (n_win, 3, T, H, W)
                batch = torch.from_numpy(
                    np.stack(wins).transpose(0, 4, 1, 2, 3).copy())
                with torch.no_grad():
                    f = backbone(batch).squeeze(-1).squeeze(-1).squeeze(-1)
                rows.append(f.numpy().astype(np.float32))
            if (n + 1) % 25 == 0 or n + 1 == len(items):
                done = n + 1
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(items)}  {rate:.2f} vid/s  "
                      f"eta {(len(items)-done)/max(rate,1e-6)/60:.1f} min")

        width = max(r.shape[0] for r in rows)
        X = np.zeros((len(rows), width, 512), dtype=np.float32)
        for i, r in enumerate(rows):
            X[i, :r.shape[0]] = r
        np.save(FEAT / f"{split}_vid{suffix}.npy", X)
        (FEAT / f"{split}_vid{suffix}_paths.txt").write_text(
            "\n".join(str(p) for p, _ in items), encoding="utf-8")
        print(f"  saved {split}_vid{suffix}.npy {X.shape}")


if __name__ == "__main__":
    main()
