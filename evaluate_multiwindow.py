"""
evaluate_multiwindow.py
Does averaging predictions over several windows of a clip beat looking at
only the first 30 frames (what ShotPredictor does today)?

Median test clip is 63 frames — over 2x the 30-frame window the model
ever sees — so on a typical clip roughly half the footage, including
whatever happens after frame 30, is currently thrown away. This measures
whether reading it helps, by sampling windows at the start, middle and end
of each clip and averaging the resulting softmax, using the exact same
backbones/head ShotPredictor already loads (no retraining involved).

Usage: python evaluate_multiwindow.py [--limit N]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES                     # noqa: E402
from src.classifier.shot_predictor import (                       # noqa: E402
    ShotPredictor, N_FRAMES, SEQ_LEN, FRAME_PICKS, VID_RESIZE, VID_CROP,
    VID_MEAN, VID_STD, VID_WINDOW, VID_STRIDE, EFFNET_DIM, VID_DIM)

CLASSES = list(SHOT_CLASSES)


def read_window(video_path, start, n=N_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def effnet_features(predictor, frames):
    import tensorflow as tf
    wanted = set(FRAME_PICKS)
    picked, last = {}, None
    for i, frame in enumerate(frames):
        if i in wanted:
            f = tf.image.convert_image_dtype(frame, tf.uint8)
            f = tf.image.resize_with_pad(f, 224, 224).numpy()
            picked[i] = f[..., ::-1].astype(np.uint8)
            last = picked[i]
    if last is None:
        return np.zeros((SEQ_LEN, EFFNET_DIM), dtype=np.float32)
    batch = np.stack([picked.get(i, last) for i in FRAME_PICKS])
    return predictor._load_effnet().predict(batch, verbose=0).astype(np.float32)


def video_features(predictor, frames):
    import torch
    procd = []
    for f in frames:
        f = cv2.resize(f, VID_RESIZE, interpolation=cv2.INTER_LINEAR)
        y0 = (f.shape[0] - VID_CROP) // 2
        x0 = (f.shape[1] - VID_CROP) // 2
        f = f[y0:y0 + VID_CROP, x0:x0 + VID_CROP]
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        procd.append((f - VID_MEAN) / VID_STD)
    if not procd:
        return np.zeros((SEQ_LEN, VID_DIM), dtype=np.float32)
    while len(procd) < VID_WINDOW:
        procd.append(procd[-1])
    clip = np.stack(procd)
    starts = list(range(0, max(1, len(clip) - VID_WINDOW + 1), VID_STRIDE))
    wins = [clip[s:s + VID_WINDOW] for s in starts
           if len(clip[s:s + VID_WINDOW]) == VID_WINDOW] or [clip[:VID_WINDOW]]
    batch = torch.from_numpy(np.stack(wins).transpose(0, 4, 1, 2, 3).copy())
    with torch.no_grad():
        f = predictor._load_r3d()(batch).squeeze(-1).squeeze(-1).squeeze(-1)
    f = f.numpy().astype(np.float32)
    out = np.zeros((SEQ_LEN, VID_DIM), dtype=np.float32)
    out[:min(SEQ_LEN, len(f))] = f[:SEQ_LEN]
    return out


def predict_window(predictor, video_path, start):
    frames = read_window(video_path, start)
    eff = effnet_features(predictor, frames)
    vid = video_features(predictor, frames)
    x = np.concatenate([vid, eff], axis=-1)[None, ...]
    return predictor._load_head().predict(x, verbose=0)[0]


def top13(probs, y):
    order = np.argsort(probs)[::-1]
    return int(order[0] == y), int(y in order[:3])


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    paths = Path("features/test_paths.txt").read_text(encoding="utf-8").splitlines()
    labels = [CLASSES.index(Path(p).parent.name) for p in paths]
    if limit:
        paths, labels = paths[:limit], labels[:limit]

    predictor = ShotPredictor()
    predictor._load_effnet(); predictor._load_r3d(); predictor._load_head()

    s1 = s3 = m1 = m3 = 0
    n = len(paths)
    for i, (p, y) in enumerate(zip(paths, labels)):
        cap = cv2.VideoCapture(p)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        max_start = max(0, total - N_FRAMES)
        starts = sorted(set([0, max_start // 2, max_start]))

        window_probs = [predict_window(predictor, p, s) for s in starts]
        single = window_probs[0]
        multi = np.mean(window_probs, axis=0)

        c1, c3 = top13(single, y)
        s1 += c1; s3 += c3
        c1, c3 = top13(multi, y)
        m1 += c1; m3 += c3

        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}  single top-1 {s1/(i+1)*100:.1f}%  "
                 f"multi-window top-1 {m1/(i+1)*100:.1f}%")

    print(f"\nsingle-window (current)  top-1 {s1/n*100:.1f}%   top-3 {s3/n*100:.1f}%")
    print(f"multi-window (start/mid/end, avg)  top-1 {m1/n*100:.1f}%   top-3 {m3/n*100:.1f}%")
    print(f"change: {(m1-s1)/n*100:+.1f} points top-1")
    print(f"({n} test clips: 1 clip = {100/n:.1f} points — under ~3 points is noise)")


if __name__ == "__main__":
    main()
