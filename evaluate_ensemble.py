"""
evaluate_ensemble.py
Does averaging predictions from several already-trained heads beat the
single best one (r3d18+effnet, 62.4% test top-1)?

Only heads trained on the same r3d18+effnet feature combination are
ensembled with each other by default — mixing in heads trained on a
DIFFERENT feature set (e.g. pixels_only) would need re-deriving that
head's features for every clip too, which is a separate, bigger cost this
script does not pay. Diversity here comes from different architectures /
training runs (GRU, BiGRU+attention, pooled MLP, class-weighted) on the
same inputs, not from different inputs.

Usage: python evaluate_ensemble.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES        # noqa: E402
from train_fusion import head                          # noqa: E402
from train_video import load, resample_time            # noqa: E402

CLASSES = list(SHOT_CLASSES)
OUT = ROOT / "trained_heads"

# name -> (weights file, keras model-name used when it was built/trained)
# Class-weighted is deliberately excluded: it trained numerically identical
# to baseline (the full training set is already perfectly balanced, 125
# clips/class), so it supplies zero ensemble diversity. seed2 is the same
# architecture/data with a different random init — genuine diversity.
CANDIDATES = {
    "r3d18+effnet (baseline, 62.4%)": ("vid_r3d18_effnet.weights.h5", "r3d18_effnet"),
    "r3d18+effnet (seed2, 60.4%)": ("vid_r3d18_effnet_seed2.weights.h5",
                                    "r3d18_effnet_seed2"),
}


def evaluate(probs, y):
    top1 = float((probs.argmax(1) == y).mean())
    order = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    per_class = {c: float((probs[y == i].argmax(1) == i).mean())
                for i, c in enumerate(CLASSES) if (y == i).any()}
    return top1, top3, per_class


def main():
    # Empty suffix — see train_calibrated.py's comment: "_n40" matches the
    # 400-clip pose-experiment subset cache, not the full test set.
    te, yte, _ = load("test", "", require=("effnet", "vid"))
    T = te["vid"].shape[1]
    te = {k: resample_time(v, T) for k, v in te.items()}
    Xte = np.concatenate([te["vid"], te["effnet"]], -1)
    print(f"test {len(yte)} clips\n")

    probs = {}
    for name, (fname, model_name) in CANDIDATES.items():
        wpath = OUT / fname
        if not wpath.exists():
            print(f"  (skip) {name}: {fname} not found")
            continue
        m = head(Xte.shape[-1], model_name)
        m.load_weights(str(wpath))
        p = m.predict(Xte, batch_size=64, verbose=0)
        top1, top3, _ = evaluate(p, yte)
        print(f"  {name:<34} test top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%")
        probs[name] = p

    names = list(probs.keys())
    print("\nensemble combinations (simple average of softmax):")
    best = (None, -1.0)
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            avg = np.mean([probs[n] for n in combo], axis=0)
            top1, top3, _ = evaluate(avg, yte)
            label = " + ".join(c.split(" (")[0] for c in combo)
            print(f"  {label:<40} test top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%")
            if top1 > best[1]:
                best = (combo, top1)

    base_top1 = evaluate(probs[names[0]], yte)[0] if names else 0.0
    if best[0]:
        print(f"\nbest ensemble: {best[0]}  ({best[1]*100:.1f}% vs single-best "
             f"baseline {base_top1*100:.1f}%, {(best[1]-base_top1)*100:+.1f} points)")
        print(f"(250 test clips: 1 clip = 0.4 points — under ~3 points is noise)")


if __name__ == "__main__":
    main()
