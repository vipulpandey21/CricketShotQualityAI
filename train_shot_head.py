"""
train_shot_head.py
Retrain the shot classifier's trainable head on cached EfficientNetB0
features (see cache_cnn_features.py).

Why retrain at all: the shipped model_weights.h5 scores 57.6% top-1 on the
dataset's own held-out test split (84.8% top-3), not the 94% the README
claims, and its errors are lopsided rather than random — it answers "flick"
for a large share of every other class:

    defense    -> flick x13 of 25
    pull       -> flick x8
    square_cut -> flick x7
    cover      -> flick x6

A model that sees the right answer inside its top 3 for 85% of clips but
collapses onto a couple of favourite classes at top-1 is badly calibrated,
which is a head problem, and the head is the part we can retrain here.

Model selection is on the val split only. Test is reported once at the end
and is never used to choose anything.

Usage:
    python train_shot_head.py                 # try every variant
    python train_shot_head.py --variant gru   # just one
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


def load(split):
    X = np.load(FEAT / f"{split}_X.npy")
    y = np.load(FEAT / f"{split}_y.npy")
    return X, y


def head_original(n_feat):
    """The architecture the shipped weights use — the baseline to beat."""
    return models.Sequential([
        layers.Input(shape=(None, n_feat)),
        layers.GRU(256, return_sequences=True),
        layers.GRU(128),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ], name="original")


def head_gru(n_feat):
    """
    Smaller and more regularised. 1250 clips cannot support a 1024-wide
    dense layer on top of a 128-d recurrent state without memorising, which
    is the likely source of the collapse onto a few classes.
    """
    return models.Sequential([
        layers.Input(shape=(None, n_feat)),
        layers.Dropout(0.2),
        layers.GRU(128, return_sequences=True, dropout=0.2,
                   recurrent_dropout=0.0),
        layers.GRU(64, dropout=0.2),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ], name="gru")


def head_bigru_attn(n_feat):
    """
    Bidirectional, with attention pooling over time instead of taking only
    the last state. A cricket shot's identity is spread across the whole
    swing, so letting the head weight frames is a better fit than relying
    on wherever the sequence happens to end.
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
    return models.Model(inp, out, name="bigru_attn")


def head_pool_mlp(n_feat):
    """
    No recurrence at all — mean+max pooling over time then an MLP. Included
    as a control: if this matches the recurrent heads, the temporal ordering
    is not carrying much of the signal and that is worth knowing.
    """
    inp = layers.Input(shape=(None, n_feat))
    avg = layers.GlobalAveragePooling1D()(inp)
    mx = layers.GlobalMaxPooling1D()(inp)
    x = layers.Concatenate()([avg, mx])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out, name="pool_mlp")


VARIANTS = {
    "original": head_original,
    "gru": head_gru,
    "bigru_attn": head_bigru_attn,
    "pool_mlp": head_pool_mlp,
}


def evaluate(model, X, y, name):
    p = model.predict(X, batch_size=64, verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    per_class = {}
    for ci, cls in enumerate(CLASSES):
        m = y == ci
        per_class[cls] = float((p[m].argmax(1) == ci).mean()) if m.any() else None
    print(f"  {name:<6} top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%")
    return {"top1": top1, "top3": top3, "per_class": per_class,
            "probs": p}


def main():
    tf.keras.utils.set_random_seed(SEED)
    OUT.mkdir(exist_ok=True)

    want = None
    if "--variant" in sys.argv:
        want = sys.argv[sys.argv.index("--variant") + 1]

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    demo = None
    if (FEAT / "demo_X.npy").exists():
        demo = load("demo")
        print(f"demo  {demo[0].shape}")

    n_feat = Xtr.shape[-1]
    results = {}

    for name, builder in VARIANTS.items():
        if want and name != want:
            continue
        print(f"\n=== {name} ===")
        tf.keras.utils.set_random_seed(SEED)
        model = builder(n_feat)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        cbs = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=12,
                restore_best_weights=True, mode="max"),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
        ]
        model.fit(Xtr, ytr, validation_data=(Xva, yva),
                  epochs=120, batch_size=32, callbacks=cbs, verbose=0)

        val = evaluate(model, Xva, yva, "val")
        test = evaluate(model, Xte, yte, "test")
        entry = {"val": {k: v for k, v in val.items() if k != "probs"},
                 "test": {k: v for k, v in test.items() if k != "probs"}}
        if demo is not None:
            d = evaluate(model, demo[0], demo[1], "demo")
            entry["demo"] = {k: v for k, v in d.items() if k != "probs"}

        model.save_weights(str(OUT / f"head_{name}.weights.h5"))
        results[name] = entry

    print("\n" + "=" * 66)
    print(f"{'variant':<12}{'val top-1':>11}{'test top-1':>12}{'test top-3':>12}"
          f"{'demo top-1':>12}")
    for name, r in results.items():
        demo_s = f"{r['demo']['top1']*100:.1f}%" if "demo" in r else "-"
        print(f"{name:<12}{r['val']['top1']*100:>10.1f}%"
              f"{r['test']['top1']*100:>11.1f}%{r['test']['top3']*100:>11.1f}%"
              f"{demo_s:>12}")

    best = max(results, key=lambda k: results[k]["val"]["top1"])
    print(f"\nbest by VAL top-1: {best}")
    print("per-class test top-1 for that variant:")
    for cls, v in results[best]["test"]["per_class"].items():
        print(f"  {cls:<12}{v*100:5.0f}%")

    (OUT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
