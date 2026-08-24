"""
train_stgcn_fusion.py
Does the ST-GCN's learned representation help once it's fused with the
existing CNN backbone, even though it's weak standing alone (15.6% val
top-1 vs the 62.4% shipped r3d18+effnet)?

This is not a contradiction to test for: standalone accuracy and fusion
value are different questions. The classifier-real-accuracy memory records
that four earlier pooled-skeleton fusion attempts ranged from -4 to +3.6
points despite the pooled skeleton alone scoring only ~26% standalone — a
weak stream can still carry a little COMPLEMENTARY signal the CNN misses,
particularly on shots the CNN's appearance-only view struggles with. A
graph-structured embedding is a different representation of the same
skeleton than the pooled one already tried, so it is worth one honest check
before concluding skeleton work is exhausted for this dataset.

Method: load the trained ST-GCN, take its pooled embedding (the
`embedding` layer, right before the final classification Dense) per clip,
broadcast it across the 30 frames, and concatenate onto the existing
r3d18 + EfficientNetB0 per-frame features — same `head()` architecture
train_fusion.py/train_video.py already use, so this is the same experiment
family with one more feature stream added.

Usage: python train_stgcn_fusion.py
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_fusion import head   # noqa: E402
from train_stgcn import build_adjacency, build_stgcn, load_pose  # noqa: E402
from train_video import load as load_video, resample_time  # noqa: E402
from src.classifier.model import SHOT_CLASSES              # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
OUT = ROOT / "trained_heads"
SEED = 1337


def stgcn_embeddings(split, suffix):
    """Per-clip ST-GCN embedding, aligned to that split's pose-cache paths."""
    X, y = load_pose(split, suffix)
    adjacency = build_adjacency()
    model = build_stgcn(adjacency, T=X.shape[1])
    model.load_weights(str(OUT / "stgcn.weights.h5"))
    embedder = models.Model(model.input, model.get_layer("embedding").output)
    emb = embedder.predict(X, batch_size=64, verbose=0)   # (N, C)

    # load_pose already dropped label<0 and all-zero clips — rebuild the
    # matching path list the same way so embeddings can be joined by path.
    for name in (f"{split}_pose{suffix}.npy", f"{split}_pose.npy"):
        p = FEAT / name
        if p.exists():
            stem = name[:-4]
            paths = (FEAT / f"{stem}_paths.txt").read_text(
                encoding="utf-8").splitlines()
            break
    raw = np.load(p)
    keep = [i for i, pth in enumerate(paths)
           if Path(pth).parent.name in CLASSES]
    kept_paths = [paths[i] for i in keep]
    nonzero = np.abs(raw[keep]).sum(axis=(1, 2)) > 0
    kept_paths = [pp for pp, ok in zip(kept_paths, nonzero) if ok]
    assert len(kept_paths) == len(emb), \
        f"{split}: {len(kept_paths)} paths vs {len(emb)} embeddings"
    return dict(zip(kept_paths, emb))


def evaluate(model, X, y):
    p = model.predict(X, batch_size=64, verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    return top1, top3


def run(name, Xtr, ytr, Xva, yva):
    tf.keras.utils.set_random_seed(SEED)
    model = head(Xtr.shape[-1], name.replace("+", "_"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=150,
             batch_size=32, callbacks=cbs, verbose=0)
    top1, top3 = evaluate(model, Xva, yva)
    print(f"  {name:<20} val top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%")
    return top1, top3


def main():
    suffix = "_n40"

    # Reuse train_video.py's loader for the aligned vid+effnet+pose sets.
    tr, ytr, tr_paths = load_video("train", suffix)
    va, yva, va_paths = load_video("val", suffix)
    T = tr["vid"].shape[1]
    tr = {k: resample_time(v, T) for k, v in tr.items()}
    va = {k: resample_time(v, T) for k, v in va.items()}
    print(f"train {len(ytr)} clips, val {len(yva)} clips, T={T}")

    # ST-GCN embeddings, joined onto the same clips by path.
    emb_tr = stgcn_embeddings("train", suffix)
    emb_va = stgcn_embeddings("val", suffix)

    def attach_embedding(feat_dict, y, paths, emb_map):
        have = [i for i, p in enumerate(paths) if p in emb_map]
        Xv = feat_dict["vid"][have]
        Xe = feat_dict["effnet"][have]
        yy = y[have]
        emb = np.stack([emb_map[paths[i]] for i in have])          # (N, C)
        emb_bcast = np.repeat(emb[:, None, :], T, axis=1)           # (N, T, C)
        return Xv, Xe, emb_bcast, yy

    Xv_tr, Xe_tr, Eb_tr, ytr2 = attach_embedding(tr, ytr, tr_paths, emb_tr)
    Xv_va, Xe_va, Eb_va, yva2 = attach_embedding(va, yva, va_paths, emb_va)
    print(f"matched with ST-GCN embedding: train {len(ytr2)}, val {len(yva2)}")

    print("\nbaseline (no ST-GCN):")
    base_top1, base_top3 = run(
        "r3d18+effnet", np.concatenate([Xv_tr, Xe_tr], -1), ytr2,
        np.concatenate([Xv_va, Xe_va], -1), yva2)

    print("\nwith ST-GCN embedding fused in:")
    fus_top1, fus_top3 = run(
        "r3d18+effnet+stgcn",
        np.concatenate([Xv_tr, Xe_tr, Eb_tr], -1), ytr2,
        np.concatenate([Xv_va, Xe_va, Eb_va], -1), yva2)

    gain = (fus_top1 - base_top1) * 100
    print(f"\nST-GCN embedding changes val top-1 by {gain:+.1f} points "
         f"({base_top1*100:.1f}% -> {fus_top1*100:.1f}%)")
    print(f"({len(yva2)} val clips: 1 clip = {100/len(yva2):.1f} points — "
         f"under ~3 points is noise)")


if __name__ == "__main__":
    main()
