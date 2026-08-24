"""
train_seed2.py
Same r3d18+effnet architecture and full 1250-clip data as the shipped
62.4% model, different random seed — for a genuine ensemble diversity
test (evaluate_ensemble.py). The class-weighted run turned out numerically
identical to baseline (the training set is already perfectly balanced,
125 clips/class), so it can't supply ensemble diversity; a different seed
can.
"""
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_fusion import head
from train_video import load, resample_time, evaluate

tr, ytr, _ = load("train", "", require=("effnet", "vid"))
va, yva, _ = load("val", "", require=("effnet", "vid"))
te, yte, _ = load("test", "", require=("effnet", "vid"))
T = tr["vid"].shape[1]
tr = {k: resample_time(v, T) for k, v in tr.items()}
va = {k: resample_time(v, T) for k, v in va.items()}
te = {k: resample_time(v, T) for k, v in te.items()}
Xtr = np.concatenate([tr["vid"], tr["effnet"]], -1)
Xva = np.concatenate([va["vid"], va["effnet"]], -1)
Xte = np.concatenate([te["vid"], te["effnet"]], -1)

tf.keras.utils.set_random_seed(2024)
model = head(Xtr.shape[-1], "r3d18_effnet_seed2")
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
t1, t3, _ = evaluate(model, Xte, yte)
print(f"seed2 test top-1 {t1*100:.1f}%  top-3 {t3*100:.1f}%  (baseline 62.4%/81.2%)")
model.save_weights("trained_heads/vid_r3d18_effnet_seed2.weights.h5")
