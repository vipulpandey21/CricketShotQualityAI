"""
train_fusion2.py
Better ways to combine pixels and skeleton than concatenating them.

Naive concatenation LOSES 4 points against pixels alone (39.6% -> 35.6% val
top-1 on the same 400 training clips). 1280 pixel dimensions against 46 pose
dimensions means the skeleton is a rounding error inside the input vector,
and with 400 clips there is not enough data for the model to learn to weight
it up. Yet the skeleton clearly carries signal — pose alone reaches 26.4%
against a 10% random baseline — and it beats pixels outright on some classes
(cover 36% vs 16%, lofted 36% vs 16%). So the information is there and the
combination method is what is wrong.

Three better combinations, all on the identical training clips:

    late_mean     average the two models' class probabilities
    late_weighted the same, sweeping the mix weight on val
    two_branch    separate encoders per input, concatenated as embeddings,
                  so pose gets its own capacity instead of competing for
                  space inside one wide input vector

Usage: python train_fusion2.py [--pose-suffix _n40]
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
from train_fusion import load_aligned, head       # noqa: E402

CLASSES = list(SHOT_CLASSES)
OUT = ROOT / "trained_heads"
SEED = 1337


def encoder(inp, units, dropout=0.2):
    """Shared encoder shape — a BiGRU with attention pooling over time."""
    x = layers.Dropout(dropout)(inp)
    x = layers.Bidirectional(layers.GRU(units, return_sequences=True,
                                        dropout=dropout))(x)
    scores = layers.Dense(1, activation="tanh")(x)
    weights = layers.Softmax(axis=1)(scores)
    x = layers.Multiply()([x, weights])
    return layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)


def two_branch(n_cnn, n_pose):
    """
    One encoder per input, then join. Pose gets its own 64-unit branch, so
    its 46 dimensions are not competing against 1280 inside a single vector.
    """
    ci = layers.Input(shape=(None, n_cnn), name="cnn")
    pi = layers.Input(shape=(None, n_pose), name="pose")
    c = encoder(ci, 128)
    p = encoder(pi, 64)
    c = layers.BatchNormalization()(c)
    p = layers.BatchNormalization()(p)
    x = layers.Concatenate()([c, p])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model([ci, pi], out, name="two_branch")


def fit(model, Xtr, ytr, Xva, yva, epochs=150):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=6, min_lr=1e-5),
    ]
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
              batch_size=32, callbacks=cbs, verbose=0)
    return model


def scores(p, y):
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    return top1, top3


def main():
    suffix = "_n40"
    if "--pose-suffix" in sys.argv:
        suffix = sys.argv[sys.argv.index("--pose-suffix") + 1]
    OUT.mkdir(exist_ok=True)

    Ctr, Ptr, ytr, _ = load_aligned("train", suffix)
    Cva, Pva, yva, _ = load_aligned("val", suffix)
    print(f"train {len(ytr)} clips, val {len(yva)} clips\n")

    results = {}

    # single-input baselines, retrained here so probabilities are available
    tf.keras.utils.set_random_seed(SEED)
    m_pix = fit(head(Ctr.shape[-1], "pix"), Ctr, ytr, Cva, yva)
    p_pix = m_pix.predict(Cva, batch_size=64, verbose=0)
    t1, t3 = scores(p_pix, yva)
    results["pixels_only"] = {"top1": t1, "top3": t3}
    print(f"  {'pixels_only':<14} top-1 {t1*100:5.1f}%  top-3 {t3*100:5.1f}%")

    tf.keras.utils.set_random_seed(SEED)
    m_pose = fit(head(Ptr.shape[-1], "pose"), Ptr, ytr, Pva, yva)
    p_pose = m_pose.predict(Pva, batch_size=64, verbose=0)
    t1, t3 = scores(p_pose, yva)
    results["pose_only"] = {"top1": t1, "top3": t3}
    print(f"  {'pose_only':<14} top-1 {t1*100:5.1f}%  top-3 {t3*100:5.1f}%")

    # late fusion — equal average
    t1, t3 = scores((p_pix + p_pose) / 2, yva)
    results["late_mean"] = {"top1": t1, "top3": t3}
    print(f"  {'late_mean':<14} top-1 {t1*100:5.1f}%  top-3 {t3*100:5.1f}%")

    # late fusion — weight swept on val. Reported honestly as val-selected:
    # the weight is a hyperparameter chosen on val, so this number is
    # optimistic until confirmed on test.
    best_w, best = None, -1
    for w in np.arange(0.0, 1.01, 0.05):
        t1, _ = scores(w * p_pix + (1 - w) * p_pose, yva)
        if t1 > best:
            best, best_w = t1, float(w)
    t1, t3 = scores(best_w * p_pix + (1 - best_w) * p_pose, yva)
    results["late_weighted"] = {"top1": t1, "top3": t3, "weight_pixels": best_w}
    print(f"  {'late_weighted':<14} top-1 {t1*100:5.1f}%  top-3 {t3*100:5.1f}%"
          f"   (pixels weight {best_w:.2f}, chosen on val)")

    # two-branch
    tf.keras.utils.set_random_seed(SEED)
    m_two = fit(two_branch(Ctr.shape[-1], Ptr.shape[-1]),
                [Ctr, Ptr], ytr, [Cva, Pva], yva)
    p_two = m_two.predict([Cva, Pva], batch_size=64, verbose=0)
    t1, t3 = scores(p_two, yva)
    results["two_branch"] = {"top1": t1, "top3": t3}
    print(f"  {'two_branch':<14} top-1 {t1*100:5.1f}%  top-3 {t3*100:5.1f}%")
    m_two.save_weights(str(OUT / "fusion_two_branch.weights.h5"))

    base = results["pixels_only"]["top1"]
    print("\n" + "=" * 64)
    print(f"{'model':<16}{'val top-1':>11}{'vs pixels':>12}")
    for k, r in results.items():
        print(f"{k:<16}{r['top1']*100:>10.1f}%"
              f"{(r['top1']-base)*100:>+11.1f}")
    print(f"\n{len(yva)} val clips: 1 clip = {100/len(yva):.1f} points. "
          f"Under ~3 points is noise.")

    (OUT / "fusion2_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
