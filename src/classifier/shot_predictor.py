"""
shot_predictor.py
The shot classifier the app actually uses.

Two backbones, concatenated, feeding one trained head:

    EfficientNetB0 (ImageNet)   what a frame looks like
    r3d_18 (Kinetics-400)       how the body and bat move

Measured on the dataset's held-out test split (250 clips, model selected on
val and never on test):

    shipped model_weights.h5        57.6% top-1   84.8% top-3
    EfficientNetB0 retrained        52.4%         80.8%
    r3d_18 alone                    56.4%         79.2%
    both together  <- this          62.4%         81.2%

Neither backbone alone beats the shipped weights by much; together they do,
because they fail on different shots. r3d_18 is far better on straight
(76% vs 52%) and lofted (40% vs 20%), EfficientNetB0 on square_cut and hook,
and the pair beats both on straight, square_cut and flick.

Only 3 frames are pushed through EfficientNetB0, not 30. The head was trained
on sequences of length 3 — the per-frame features were resampled down to
match r3d_18's window count — so frames 0, 15 and 29 are the only ones that
ever reached it. Computing the other 27 was pure waste, and dropping them
takes the classification step from ~50s to ~5s.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "trained_heads" / "vid_r3d18_effnet.weights.h5"

N_FRAMES = 30
SEQ_LEN = 3                      # what the head was trained on
# Derived, not hard-coded: this must reproduce train_video.resample_time
# exactly. It gives [0, 14, 29] — note 14, because numpy rounds 14.5 down to
# even. Writing [0, 15, 29] by hand would feed the head a frame it was never
# trained on.
FRAME_PICKS = np.linspace(0, N_FRAMES - 1, SEQ_LEN).round().astype(int).tolist()

# r3d_18 preprocessing, from R3D_18_Weights.KINETICS400_V1.transforms()
VID_RESIZE = (171, 128)
VID_CROP = 112
VID_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
VID_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
VID_WINDOW = 16
VID_STRIDE = 7

EFFNET_DIM = 1280
VID_DIM = 512


class ShotPredictor:
    """Lazily loads both backbones and the trained head; reusable per process."""

    def __init__(self, weights: str | Path = WEIGHTS):
        self.weights = Path(weights)
        self._effnet = None
        self._r3d = None
        self._head = None

    # ── model loading ────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.weights.exists()

    def _load_effnet(self):
        if self._effnet is None:
            import tensorflow as tf
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights="imagenet",
                input_shape=(224, 224, 3))
            base.trainable = False
            inp = tf.keras.Input(shape=(224, 224, 3))
            x = base(inp, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            self._effnet = tf.keras.Model(inp, x)
        return self._effnet

    def _load_r3d(self):
        if self._r3d is None:
            import torch
            from torchvision.models.video import R3D_18_Weights, r3d_18
            m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            m.eval()
            self._r3d = torch.nn.Sequential(*(list(m.children())[:-1]))
        return self._r3d

    def _load_head(self):
        if self._head is None:
            if not self.available:
                raise FileNotFoundError(
                    f"Trained head not found at {self.weights}. "
                    f"Run: python cache_cnn_features.py && "
                    f"python cache_video_features.py && python train_video.py")
            import sys
            sys.path.insert(0, str(ROOT))
            from train_fusion import head
            m = head(EFFNET_DIM + VID_DIM, "r3d18_effnet")
            m.load_weights(str(self.weights))
            self._head = m
        return self._head

    # ── feature extraction ───────────────────────────────────────────────
    def _effnet_features(self, video_path) -> np.ndarray:
        """(SEQ_LEN, 1280) — only the frames the head will actually see."""
        import tensorflow as tf

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        wanted = set(FRAME_PICKS)
        picked, last = {}, None
        for i in range(N_FRAMES):
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                f = tf.image.convert_image_dtype(frame, tf.uint8)
                f = tf.image.resize_with_pad(f, 224, 224).numpy()
                picked[i] = f[..., ::-1].astype(np.uint8)   # BGR -> RGB
                last = picked[i]
        cap.release()

        if last is None:
            return np.zeros((SEQ_LEN, EFFNET_DIM), dtype=np.float32)
        # A clip shorter than 30 frames reuses its last frame, which is what
        # the training-time extractor did when a video ran out.
        batch = np.stack([picked.get(i, last) for i in FRAME_PICKS])
        return self._load_effnet().predict(batch, verbose=0).astype(np.float32)

    def _video_features(self, video_path) -> np.ndarray:
        """(SEQ_LEN, 512) from r3d_18 over overlapping 16-frame windows."""
        import torch

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        for _ in range(N_FRAMES):
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f, VID_RESIZE, interpolation=cv2.INTER_LINEAR)
            y0 = (f.shape[0] - VID_CROP) // 2
            x0 = (f.shape[1] - VID_CROP) // 2
            f = f[y0:y0 + VID_CROP, x0:x0 + VID_CROP]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append((f - VID_MEAN) / VID_STD)
        cap.release()

        if not frames:
            return np.zeros((SEQ_LEN, VID_DIM), dtype=np.float32)
        while len(frames) < VID_WINDOW:
            frames.append(frames[-1])
        clip = np.stack(frames)

        starts = list(range(0, max(1, len(clip) - VID_WINDOW + 1), VID_STRIDE))
        wins = [clip[s:s + VID_WINDOW] for s in starts
                if len(clip[s:s + VID_WINDOW]) == VID_WINDOW] or [clip[:VID_WINDOW]]
        batch = torch.from_numpy(
            np.stack(wins).transpose(0, 4, 1, 2, 3).copy())
        with torch.no_grad():
            f = self._load_r3d()(batch).squeeze(-1).squeeze(-1).squeeze(-1)
        f = f.numpy().astype(np.float32)

        out = np.zeros((SEQ_LEN, VID_DIM), dtype=np.float32)
        out[:min(SEQ_LEN, len(f))] = f[:SEQ_LEN]
        return out

    # ── prediction ───────────────────────────────────────────────────────
    def predict(self, video_path, idx_to_class: dict) -> dict:
        """
        Returns the same shape the pipeline expects:
        {"shot", "confidence", "top3": [...], "all_probabilities": {...}}
        """
        eff = self._effnet_features(video_path)
        vid = self._video_features(video_path)
        x = np.concatenate([vid, eff], axis=-1)[None, ...]

        probs = self._load_head().predict(x, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        return {
            "shot": idx_to_class[int(order[0])],
            "confidence": round(float(probs[order[0]]) * 100, 1),
            "top3": [{"shot": idx_to_class[int(j)],
                      "confidence": round(float(probs[j]) * 100, 1)}
                     for j in order[:3]],
            "all_probabilities": {idx_to_class[i]: round(float(p) * 100, 2)
                                  for i, p in enumerate(probs)},
            "model": "r3d18+effnet (62.4% test top-1, 81.2% top-3)",
        }
