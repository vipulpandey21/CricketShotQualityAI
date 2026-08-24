"""
train_fusion.py
Does adding the striker's skeleton to the pixel features actually improve
shot classification?

Three models are trained and compared:

    pixels_only   the cached EfficientNetB0 features (1280 per frame)
    pose_only     the striker skeleton features (46 per frame)
    fusion        both, concatenated (1326 per frame)

All three are trained on EXACTLY the same videos. Pose features exist for a
subset of the training split only, so pixels_only is trained on that same
subset rather than on all 1250 — otherwise the comparison would measure the
difference in training-set size, not the value of the skeleton, and pixels
would look worse for the wrong reason.

Selection is on the val split. Test is not touched here.

Usage:
    python train_fusion.py                    # auto-detects the pose files
    python train_fusion.py --pose-suffix _n40
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES     # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
OUT = ROOT / "trained_heads"
SEED = 1337


def read_paths(p: Path) -> list:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def load_aligned(split: str, pose_suffix: str):
    """
    CNN features, pose features and labels for the videos that have BOTH.

    The two caches were produced by separate runs over different video sets,
    so rows are matched by source path. Pairing by row index would silently
    attach one clip's skeleton to another clip's pixels.
    """
    Xc = np.load(FEAT / f"{split}_X.npy")
    y = np.load(FEAT / f"{split}_y.npy")
    cnn_paths = read_paths(FEAT / f"{split}_paths.txt")

    pose_file = FEAT / f"{split}_pose{pose_suffix}.npy"
    if not pose_file.exists():
        pose_file = FEAT / f"{split}_pose.npy"
        pose_paths = read_paths(FEAT / f"{split}_pose_paths.txt")
    else:
        pose_paths = read_paths(
            FEAT / f"{split}_pose{pose_suffix}_paths.txt")
    Xp = np.load(pose_file)

    cnn_idx = {p: i for i, p in enumerate(cnn_paths)}
    rows_c, rows_p = [], []
    for j, p in enumerate(pose_paths):
        i = cnn_idx.get(p)
        if i is not None:
            rows_c.append(i)
            rows_p.append(j)

    return (Xc[rows_c], Xp[rows_p], y[rows_c], len(pose_paths))


def head(n_feat, name):
    """
    One architecture for all three inputs, so the only thing that differs
    between the runs is what the model is allowed to see.
    """
    inp = layers.Input(shape=(None, n_feat))
    x = layers.Dropout(0.2)(inp)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True,
                                        dropout=0.2))(x)
    scores = layers.Dense(1, activation="tanh")(x)
    weights = layers.Softmax(axis=1)(scores)
    x = layers.Multiply()([x, weights])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out, name=name)


def run(name, Xtr, ytr, Xva, yva):
    tf.keras.utils.set_random_seed(SEED)
    model = head(Xtr.shape[-1], name)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=150,
              batch_size=32, callbacks=cbs, verbose=0)

    p = model.predict(Xva, batch_size=64, verbose=0)
    top1 = float((p.argmax(1) == yva).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([yva[i] in order[i] for i in range(len(yva))]))
    per_class = {c: float((p[yva == i].argmax(1) == i).mean())
                 for i, c in enumerate(CLASSES) if (yva == i).any()}
    print(f"  {name:<12} val top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%")
    model.save_weights(str(OUT / f"fusion_{name}.weights.h5"))
    return {"top1": top1, "top3": top3, "per_class": per_class,
            "n_features": int(Xtr.shape[-1])}


def main():
    suffix = "_n40"
    if "--pose-suffix" in sys.argv:
        suffix = sys.argv[sys.argv.index("--pose-suffix") + 1]
    OUT.mkdir(exist_ok=True)

    Ctr, Ptr, ytr, n_pose_tr = load_aligned("train", suffix)
    Cva, Pva, yva, n_pose_va = load_aligned("val", suffix)
    print(f"train: {len(ytr)} videos matched (of {n_pose_tr} with pose)")
    print(f"val  : {len(yva)} videos matched (of {n_pose_va} with pose)")
    print(f"cnn {Ctr.shape[-1]}-d, pose {Ptr.shape[-1]}-d per frame\n")

    # how often the striker was actually found — an all-zero row means not
    frac_tr = float((np.abs(Ptr).sum(axis=(1, 2)) > 0).mean())
    frac_va = float((np.abs(Pva).sum(axis=(1, 2)) > 0).mean())
    print(f"striker found in {frac_tr*100:.0f}% of train, "
          f"{frac_va*100:.0f}% of val\n")

    results = {}
    results["pixels_only"] = run("pixels_only", Ctr, ytr, Cva, yva)
    results["pose_only"] = run("pose_only", Ptr, ytr, Pva, yva)
    results["fusion"] = run("fusion",
                            np.concatenate([Ctr, Ptr], axis=-1), ytr,
                            np.concatenate([Cva, Pva], axis=-1), yva)

    print("\n" + "=" * 62)
    print(f"{'model':<14}{'val top-1':>12}{'val top-3':>12}{'features':>10}")
    for k, r in results.items():
        print(f"{k:<14}{r['top1']*100:>11.1f}%{r['top3']*100:>11.1f}%"
              f"{r['n_features']:>10}")

    base = results["pixels_only"]["top1"]
    gain = (results["fusion"]["top1"] - base) * 100
    print(f"\nskeleton changes top-1 by {gain:+.1f} points "
          f"({base*100:.1f}% -> {results['fusion']['top1']*100:.1f}%)")
    print(f"({len(yva)} val clips, so 1 clip = {100/len(yva):.1f} points — "
          f"treat anything under ~3 points as noise)")

    print("\nper-class val top-1:")
    print(f"{'class':<12}{'pixels':>9}{'pose':>9}{'fusion':>9}")
    for c in CLASSES:
        row = "".join(f"{results[k]['per_class'].get(c, 0)*100:>8.0f}%"
                      for k in ("pixels_only", "pose_only", "fusion"))
        print(f"{c:<12}{row}")

    (OUT / "fusion_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
