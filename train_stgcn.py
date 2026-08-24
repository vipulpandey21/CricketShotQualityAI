"""
train_stgcn.py
Direct skeleton-to-shot classifier: a small ST-GCN (spatial-temporal graph
convolutional network) over the 13 tracked cricket joints, instead of the
pooled 46-dim/frame summary the GRU/fusion heads use.

Why this is a genuinely different bet than the skeleton fusion already
tried (see the classifier-real-accuracy memory — four fusion forms, none
lifted top-1): those all fed the pooled per-frame vector into the same
sequence head used for CNN features, treating the skeleton as "just another
feature vector". A graph conv instead respects the actual skeleton
topology — a knee's movement is convolved with its hip and ankle
specifically, not blindly mixed with every other joint through a dense
layer. Whether that structural prior helps THIS dataset is exactly what
this script measures; it is not assumed.

Input: raw joint coordinates already sitting in the pose cache
(cache_pose_features.py's columns 0-25 = 13 joints (x,y) in box-relative
coordinates, columns 26-38 = visibility), reshaped to (T=30, V=13, C=3).
No new video processing needed for train/val — only the test split lacked a
pose cache and was extracted once, alongside this script, the same way
train/val already were.

Graph: the same 12 skeletal edges `estimator.draw_skeleton` already draws,
plus two edges connecting the nose to both shoulders (draw_skeleton omits
the nose from its connections since it only decorates the frame, but a
disconnected node is wasted in a graph conv).

Usage:
    python train_stgcn.py                          # train + val, test if cached
    python train_stgcn.py --train-suffix _n40 --val-suffix _n40 --test-suffix _n25
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES   # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
OUT = ROOT / "trained_heads"
SEED = 1337

# Must match cache_pose_features.py's JOINTS order exactly — that order is
# what columns 0-25 / 26-38 of the cache are laid out in.
JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
IDX = {j: i for i, j in enumerate(JOINTS)}
V = len(JOINTS)

# The 12 edges estimator.draw_skeleton connects, plus the nose tied to both
# shoulders so every node has at least one neighbour.
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
         (24, 26), (26, 28), (0, 11), (0, 12)]


def build_adjacency() -> np.ndarray:
    """Symmetric-normalised adjacency with self-loops (Kipf & Welling GCN)."""
    A = np.eye(V, dtype=np.float32)
    for a, b in EDGES:
        i, j = IDX[a], IDX[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    return (d_inv_sqrt @ A @ d_inv_sqrt).astype(np.float32)


class GraphConv(layers.Layer):
    """
    One spatial graph-conv step: aggregate each joint's features with its
    graph neighbours (fixed adjacency, not learned — 13 joints and ~500
    training clips is not enough data to also learn which joints matter to
    each other), then mix channels with a learned Dense.
    """
    def __init__(self, filters, adjacency, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self._adj_np = adjacency

    def build(self, input_shape):
        self.adj = tf.constant(self._adj_np, dtype=tf.float32)
        self.dense = layers.Dense(self.filters)
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, V, C) -> aggregate over V using the fixed adjacency
        x = tf.einsum("vw,btwc->btvc", self.adj, x)
        return self.dense(x)

    def get_config(self):
        cfg = super().get_config()
        cfg["filters"] = self.filters
        cfg["adjacency"] = self._adj_np.tolist()
        return cfg


def st_block(x, filters, adjacency, stride=1, dropout=0.3, l2=1e-4):
    """Graph conv (space) -> temporal conv (time) -> residual, ST-GCN's unit."""
    reg = tf.keras.regularizers.l2(l2)
    res = x
    y = GraphConv(filters, adjacency)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Conv2D(filters, (9, 1), strides=(stride, 1), padding="same",
                      kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)

    if res.shape[-1] != filters or stride != 1:
        res = layers.Conv2D(filters, (1, 1), strides=(stride, 1),
                            kernel_regularizer=reg)(res)
        res = layers.BatchNormalization()(res)
    y = layers.Add()([y, res])
    return layers.ReLU()(y)


def build_stgcn(adjacency, T=30, n_classes=10):
    """
    Kept small and shallow, no temporal downsampling — the first attempt
    (3 blocks, 32->64->96 channels, strided down to 8 frames, 149K params)
    memorised the 393-clip training set immediately (train accuracy climbed
    every epoch) while val accuracy never left chance (10%) and predictions
    collapsed onto one class. That is far more capacity than this amount of
    data supports, and losing 30 frames down to 8 throws away most of the
    already-short swing. This version is ~5x fewer parameters and keeps the
    full 30-frame resolution throughout.
    """
    inp = layers.Input(shape=(T, V, 3), name="skeleton")
    x = layers.BatchNormalization()(inp)
    x = st_block(x, 16, adjacency, stride=1)
    x = st_block(x, 24, adjacency, stride=1)
    x = layers.GlobalAveragePooling2D(name="embedding")(x)
    x = layers.Dropout(0.55)(x)
    out = layers.Dense(n_classes, activation="softmax",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return models.Model(inp, out, name="stgcn")


def fill_gaps(X: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate through frames where the striker wasn't found.

    cache_pose_features.py writes an all-zero row for a frame with no
    detection — a real coordinate value (0, 0, 0), not a missing-data
    marker. Left as-is, a graph conv reads that as "every joint sitting at
    the box's top-left corner", which is a fake pose, not an absence of
    one. ~20% of frames in this dataset are affected (checked by measuring
    the all-zero-frame fraction directly), so this is not a rare edge case.
    Matches the same fill-through-gaps approach striker_pose.py already
    uses for landmark dicts, just applied to the cached array.
    """
    X = X.copy()
    n, t, v, c = X.shape
    valid = np.abs(X).sum(axis=(2, 3)) > 0        # (N, T)
    idx = np.arange(t)
    for i in range(n):
        good = idx[valid[i]]
        if len(good) == t or len(good) == 0:
            continue
        for vtx in range(v):
            for ch in range(c):
                X[i, :, vtx, ch] = np.interp(idx, good, X[i, good, vtx, ch])
    return X


def to_skeleton(X: np.ndarray) -> np.ndarray:
    """(N, 30, 46) pose cache -> (N, 30, 13, 3) = (x, y, visibility), gap-filled."""
    n, t, _ = X.shape
    xy = X[:, :, :26].reshape(n, t, V, 2)
    vis = X[:, :, 26:39].reshape(n, t, V, 1)
    return fill_gaps(np.concatenate([xy, vis], axis=-1).astype(np.float32))


def load_pose(split: str, suffix: str):
    """(skeleton_array, labels) for one split, or (None, None) if uncached."""
    for name in (f"{split}_pose{suffix}.npy", f"{split}_pose.npy"):
        arr_path = FEAT / name
        if not arr_path.exists():
            continue
        stem = name[:-4]
        paths = (FEAT / f"{stem}_paths.txt").read_text(
            encoding="utf-8").splitlines()
        labels = []
        keep = []
        for i, p in enumerate(paths):
            cls = Path(p).parent.name
            if cls in CLASSES:
                labels.append(CLASSES.index(cls))
                keep.append(i)
        X = np.load(arr_path)[keep]
        y = np.array(labels, dtype=np.int32)
        # drop clips where the striker was never found (all-zero row)
        nonzero = np.abs(X).sum(axis=(1, 2)) > 0
        return to_skeleton(X[nonzero]), y[nonzero]
    return None, None


def evaluate(model, X, y):
    p = model.predict(X, batch_size=64, verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    per_class = {c: float((p[y == i].argmax(1) == i).mean())
                 for i, c in enumerate(CLASSES) if (y == i).any()}
    return top1, top3, per_class


def main():
    def arg(flag, default):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

    train_suffix = arg("--train-suffix", "_n40")
    val_suffix = arg("--val-suffix", "_n40")
    test_suffix = arg("--test-suffix", "_n25")
    OUT.mkdir(exist_ok=True)

    Xtr, ytr = load_pose("train", train_suffix)
    Xva, yva = load_pose("val", val_suffix)
    if Xtr is None or Xva is None:
        print("train/val pose cache missing — run cache_pose_features.py first")
        return
    Xte, yte = load_pose("test", test_suffix)

    print(f"train {len(ytr)} clips, val {len(yva)} clips"
         + (f", test {len(yte)} clips" if Xte is not None else " (no test cache yet)"))
    print(f"skeleton shape per clip: {Xtr.shape[1:]}")

    adjacency = build_adjacency()
    tf.keras.utils.set_random_seed(SEED)
    model = build_stgcn(adjacency, T=Xtr.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    print(f"params: {model.count_params():,}")

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=8, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200,
             batch_size=32, callbacks=cbs, verbose=2)

    top1, top3, per_class = evaluate(model, Xva, yva)
    print(f"\nval top-1 {top1*100:.1f}%   top-3 {top3*100:.1f}%")
    result = {"val": {"top1": top1, "top3": top3, "per_class": per_class}}

    if Xte is not None:
        t1, t3, tpc = evaluate(model, Xte, yte)
        print(f"test top-1 {t1*100:.1f}%   top-3 {t3*100:.1f}%   "
             f"(current shipped baseline: 62.4% / 81.2%)")
        result["test"] = {"top1": t1, "top3": t3, "per_class": tpc}

    print("\nper-class val top-1:")
    for c in CLASSES:
        print(f"  {c:<12} {per_class.get(c, 0)*100:5.1f}%")

    model.save_weights(str(OUT / "stgcn.weights.h5"))
    (OUT / "stgcn_results.json").write_text(json.dumps(result, indent=1))
    print(f"\nsaved trained_heads/stgcn.weights.h5, trained_heads/stgcn_results.json")


if __name__ == "__main__":
    main()
