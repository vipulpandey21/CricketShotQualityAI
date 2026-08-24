"""
train_calibrated.py
Same winning architecture (r3d18+effnet, 62.4% test top-1) and the same
full 1250-clip training set — just with class-weighted loss, targeting a
known, specific problem: the shipped model over-predicts flick and lofted
(defense->flick 13/25, pull->flick 8, square_cut->flick 7 — see the
classifier-real-accuracy memory). High top-3 with a collapsed top-1 on
those classes means the head is miscalibrated, not that the features lack
the signal — inverse-frequency class weighting is the standard first fix
for exactly that pattern, and it costs nothing but a few minutes of
head-only retraining since the backbones stay frozen.

Saves under a new filename so the original 62.4% weights are untouched —
this is compared against, not assumed to replace it.

Usage: python train_calibrated.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES        # noqa: E402
from train_fusion import head                          # noqa: E402
from train_video import load, resample_time, evaluate  # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
OUT = ROOT / "trained_heads"
SEED = 1337


def main():
    # Empty suffix, deliberately: passing "_n40" here would match the
    # 400-clip subset cache (train_vid_n40.npy) that the pose experiments
    # use, silently training this on far less data than the 62.4% baseline
    # it's meant to be compared against — caught by the class weights
    # printing as a suspicious uniform 1.00 (that subset happens to be
    # perfectly balanced) and a result far below baseline for no honest
    # reason. Empty suffix falls through to the full train_vid.npy/train_X.npy.
    tr, ytr, _ = load("train", "", require=("effnet", "vid"))
    va, yva, _ = load("val", "", require=("effnet", "vid"))
    te, yte, _ = load("test", "", require=("effnet", "vid"))
    print(f"train {len(ytr)}, val {len(yva)}, test {len(yte)}")

    T = tr["vid"].shape[1]
    tr = {k: resample_time(v, T) for k, v in tr.items()}
    va = {k: resample_time(v, T) for k, v in va.items()}
    te = {k: resample_time(v, T) for k, v in te.items()}

    Xtr = np.concatenate([tr["vid"], tr["effnet"]], -1)
    Xva = np.concatenate([va["vid"], va["effnet"]], -1)
    Xte = np.concatenate([te["vid"], te["effnet"]], -1)

    counts = np.bincount(ytr, minlength=len(CLASSES))
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    class_weight = {i: float(w) for i, w in enumerate(weights)}
    print("class weights (inverse frequency):")
    for c, i in zip(CLASSES, range(len(CLASSES))):
        print(f"  {c:<12} n={counts[i]:4d}  weight={weights[i]:.2f}")

    tf.keras.utils.set_random_seed(SEED)
    model = head(Xtr.shape[-1], "r3d18_effnet_classweighted")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=150,
             batch_size=32, callbacks=cbs, verbose=0, class_weight=class_weight)

    v1, v3, vpc = evaluate(model, Xva, yva)
    t1, t3, tpc = evaluate(model, Xte, yte)
    print(f"\nval  top-1 {v1*100:.1f}%   top-3 {v3*100:.1f}%")
    print(f"test top-1 {t1*100:.1f}%   top-3 {t3*100:.1f}%   "
         f"(unweighted baseline: 62.4% / 81.2%)")

    print("\nper-class test top-1 (baseline vs class-weighted):")
    base = json.loads((OUT / "video_results.json").read_text())
    base_pc = base.get("r3d18+effnet", {}).get("test", {}).get("per_class", {})
    for c in CLASSES:
        b = base_pc.get(c, float("nan")) * 100
        w = tpc.get(c, 0) * 100
        print(f"  {c:<12} base {b:5.1f}%   weighted {w:5.1f}%   diff {w-b:+5.1f}")

    model.save_weights(str(OUT / "vid_r3d18_effnet_classweighted.weights.h5"))
    (OUT / "calibrated_results.json").write_text(json.dumps({
        "val": {"top1": v1, "top3": v3, "per_class": vpc},
        "test": {"top1": t1, "top3": t3, "per_class": tpc},
        "class_weight": class_weight,
    }, indent=1))
    print("\nsaved trained_heads/vid_r3d18_effnet_classweighted.weights.h5")


if __name__ == "__main__":
    main()
