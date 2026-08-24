"""
finetune_effnet.py
Partial backbone fine-tune: unfreeze the last few blocks of EfficientNetB0
and train it jointly with the head, instead of using it as a frozen
ImageNet feature extractor the way train_video.py/train_fusion.py do.
r3d_18 stays frozen (its cached features are reused as-is) — a timing
pilot showed r3d_18 costs ~139 min/epoch to fine-tune on this CPU vs
EfficientNetB0's ~10.5 min/epoch, so only the affordable half is touched
here. This is a bounded, partial experiment, not full backbone fine-tuning.

Because the backbone is now trainable, its features change every step, so
they can't be precomputed once like the frozen pipeline does — this script
reads raw frames from the source videos (only the 3 frames the head ever
sees: FRAME_PICKS, matching shot_predictor.py exactly) once into memory,
then runs the trainable backbone over them fresh each epoch.

Usage:
    python finetune_effnet.py --pilot     # 100 train clips, 3 epochs — sanity check
    python finetune_effnet.py             # full run
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES                    # noqa: E402
from src.classifier.shot_predictor import FRAME_PICKS, N_FRAMES  # noqa: E402
from train_video import load as load_video                       # noqa: E402

CLASSES = list(SHOT_CLASSES)
OUT = ROOT / "trained_heads"
SEED = 1337
SEQ_LEN = len(FRAME_PICKS)      # 3
UNFREEZE_LAST_N = 20            # last ~20 of EfficientNetB0's ~238 layers
LR = 1e-4
BATCH = 8


def raw_frames(video_path) -> np.ndarray:
    """(SEQ_LEN, 224, 224, 3) uint8 RGB — exactly what shot_predictor.py's
    _effnet_features reads, minus the backbone call."""
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
        return np.zeros((SEQ_LEN, 224, 224, 3), dtype=np.uint8)
    return np.stack([picked.get(i, last) for i in FRAME_PICKS])


def build_dataset(split, limit=None):
    """(raw_frames[N,SEQ_LEN,224,224,3], r3d_features[N,SEQ_LEN,512], y[N])."""
    feats, y, paths = load_video(split, "", require=("vid",))
    vid = feats["vid"]
    if limit:
        vid, y, paths = vid[:limit], y[:limit], paths[:limit]

    print(f"  {split}: extracting raw frames from {len(paths)} videos...")
    t0 = time.time()
    frames = np.zeros((len(paths), SEQ_LEN, 224, 224, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        frames[i] = raw_frames(p)
        if (i + 1) % 100 == 0 or i + 1 == len(paths):
            print(f"    {i+1}/{len(paths)}  ({time.time()-t0:.0f}s)")
    return frames, vid.astype(np.float32), y


def build_model(unfreeze_last_n=UNFREEZE_LAST_N):
    effnet = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    effnet.trainable = True
    for layer in effnet.layers[:-unfreeze_last_n]:
        layer.trainable = False
    n_trainable = sum(1 for l in effnet.layers if l.trainable)
    print(f"EfficientNetB0: {n_trainable}/{len(effnet.layers)} layers trainable")

    frames_in = layers.Input(shape=(SEQ_LEN, 224, 224, 3), name="frames")
    vid_in = layers.Input(shape=(SEQ_LEN, 512), name="r3d_features")

    x = layers.TimeDistributed(effnet)(frames_in)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)   # (T, 1280)
    merged = layers.Concatenate(axis=-1)([vid_in, x])                # (T, 1792)

    # Same pooling/head shape as train_fusion.head(), applied to `merged`
    # directly instead of a fresh Input, so this stays comparable to the
    # frozen-feature baseline.
    h = layers.Dropout(0.2)(merged)
    h = layers.Bidirectional(layers.GRU(128, return_sequences=True,
                                        dropout=0.2))(h)
    scores = layers.Dense(1, activation="tanh")(h)
    weights = layers.Softmax(axis=1)(scores)
    h = layers.Multiply()([h, weights])
    h = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(h)
    h = layers.BatchNormalization()(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dropout(0.4)(h)
    out = layers.Dense(10, activation="softmax")(h)

    return models.Model([frames_in, vid_in], out, name="effnet_finetune")


def evaluate(model, frames, vid, y):
    p = model.predict([frames, vid], batch_size=BATCH, verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    return top1, top3


def main():
    pilot = "--pilot" in sys.argv
    limit_tr = 100 if pilot else None
    limit_va = 50 if pilot else None
    epochs = 3 if pilot else 20

    print("=== building datasets (raw frame extraction, one-time cost) ===")
    Ftr, Vtr, ytr = build_dataset("train", limit_tr)
    Fva, Vva, yva = build_dataset("val", limit_va)
    print(f"train {len(ytr)}, val {len(yva)}")

    tf.keras.utils.set_random_seed(SEED)
    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6),
    ]

    print(f"\n=== training ({epochs} epochs max, batch {BATCH}) ===")
    t0 = time.time()
    model.fit([Ftr, Vtr], ytr, validation_data=([Fva, Vva], yva),
             epochs=epochs, batch_size=BATCH, callbacks=cbs, verbose=2)
    print(f"training took {(time.time()-t0)/60:.1f} min")

    v1, v3 = evaluate(model, Fva, Vva, yva)
    print(f"\nval top-1 {v1*100:.1f}%   top-3 {v3*100:.1f}%")

    if not pilot:
        print("\n=== test set ===")
        Fte, Vte, yte = build_dataset("test", None)
        t1, t3 = evaluate(model, Fte, Vte, yte)
        print(f"test top-1 {t1*100:.1f}%   top-3 {t3*100:.1f}%   "
             f"(frozen-backbone baseline: 62.4% / 81.2%)")
        model.save_weights(str(OUT / "effnet_finetuned.weights.h5"))
        print("saved trained_heads/effnet_finetuned.weights.h5")


if __name__ == "__main__":
    main()
