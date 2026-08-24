"""
train_video.py
Compare feature backbones for shot classification on identical clips.

    efficientnet   ImageNet per-frame features (what the project ships)
    r3d18          Kinetics-400 video features — motion, not appearance
    r3d18+pose     r3d18 with the striker skeleton concatenated
    r3d18+effnet   both backbones concatenated

Every model uses the same head and the same training clips, so the only
variable is what the model gets to see.

Usage: python train_video.py [--pose-suffix _n40]
"""

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES        # noqa: E402
from train_fusion import head, read_paths            # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
OUT = ROOT / "trained_heads"
SEED = 1337


def load(split, suffix, require=("effnet", "vid")):
    """
    All available feature sets for one split, restricted to the videos that
    appear in every one of them and aligned by source path — the caches were
    written by separate runs over different video sets, so row order differs.
    """
    sets = {}
    for key, stems in (("effnet", ["X"]),
                       ("pose", [f"pose{suffix}", "pose"]),
                       ("vid", [f"vid{suffix}", "vid"])):
        for stem in stems:
            arr = FEAT / f"{split}_{stem}.npy"
            pth = FEAT / (f"{split}_paths.txt" if stem == "X"
                          else f"{split}_{stem}_paths.txt")
            if arr.exists() and pth.exists():
                sets[key] = (np.load(arr), read_paths(pth))
                break

    missing = [k for k in require if k not in sets]
    if missing:
        raise FileNotFoundError(
            f"{split}: missing feature cache(s) {missing}. "
            f"Run cache_cnn_features.py / cache_video_features.py first.")

    y_all = np.load(FEAT / f"{split}_y.npy")
    eff_paths = sets["effnet"][1]
    label_of = dict(zip(eff_paths, y_all))

    # Intersect only over the sets we are actually going to use — including
    # a partially-cached pose set here would silently shrink a full-data run
    # down to the pose subset.
    common = set(eff_paths)
    for key in require:
        common &= set(sets[key][1])
    common = [p for p in eff_paths if p in common]
    sets = {k: v for k, v in sets.items()
            if k in require or set(common) <= set(v[1])}

    out = {}
    for key, (arr, paths) in sets.items():
        index = {p: i for i, p in enumerate(paths)}
        out[key] = arr[[index[p] for p in common]]
    y = np.array([label_of[p] for p in common], dtype=np.int32)
    return out, y, common


def resample_time(X, T):
    """
    Match sequence lengths so feature sets can be concatenated. r3d_18 yields
    one vector per 16-frame window (3 for a 30-frame clip) while the per-frame
    sets yield 30, so the shorter one is repeated along time.
    """
    if X.shape[1] == T:
        return X
    idx = np.linspace(0, X.shape[1] - 1, T).round().astype(int)
    return X[:, idx]


def evaluate(model, X, y):
    p = model.predict(X, batch_size=64, verbose=0)
    top1 = float((p.argmax(1) == y).mean())
    order = np.argsort(p, axis=1)[:, ::-1][:, :3]
    top3 = float(np.mean([y[i] in order[i] for i in range(len(y))]))
    per_class = {c: float((p[y == i].argmax(1) == i).mean())
                 for i, c in enumerate(CLASSES) if (y == i).any()}
    return top1, top3, per_class


def run(name, Xtr, ytr, Xva, yva, test=None):
    tf.keras.utils.set_random_seed(SEED)
    model = head(Xtr.shape[-1], name.replace("+", "_"))
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
    top1, top3, per_class = evaluate(model, Xva, yva)
    line = f"  {name:<16} val top-1 {top1*100:5.1f}%   top-3 {top3*100:5.1f}%"
    out = {"top1": top1, "top3": top3, "per_class": per_class,
           "n_features": int(Xtr.shape[-1])}
    if test is not None:
        t1, t3, tpc = evaluate(model, *test)
        out["test"] = {"top1": t1, "top3": t3, "per_class": tpc}
        line += f"   |   test top-1 {t1*100:5.1f}%   top-3 {t3*100:5.1f}%"
    print(line)
    model.save_weights(str(OUT / f"vid_{name.replace('+','_')}.weights.h5"))
    return out


def main():
    suffix = "_n40"
    if "--pose-suffix" in sys.argv:
        suffix = sys.argv[sys.argv.index("--pose-suffix") + 1]
    OUT.mkdir(exist_ok=True)

    tr, ytr, ptr = load("train", suffix)
    va, yva, pva = load("val", suffix)
    try:
        te, yte, _ = load("test", suffix)
    except FileNotFoundError as exc:
        print(f"test split unavailable ({exc}); reporting val only")
        te, yte = None, None
    print(f"train {len(ytr)} clips, val {len(yva)} clips")
    print("feature sets: " + ", ".join(
        f"{k} {v.shape[1]}x{v.shape[2]}" for k, v in tr.items()) + "\n")

    T = tr["vid"].shape[1]      # align everything to the video model's length
    def prep(d):
        return {k: resample_time(v, T) for k, v in d.items()}
    tr, va = prep(tr), prep(va)
    te = prep(te) if te is not None else None

    results = {}
    results["efficientnet"] = run("efficientnet", tr["effnet"], ytr,
                                  va["effnet"], yva,
                                  test=(te["effnet"], yte) if te else None)
    results["r3d18"] = run("r3d18", tr["vid"], ytr, va["vid"], yva,
                       test=(te["vid"], yte) if te else None)
    # Pose is cached for a subset of train only, so on a full-data run it is
    # simply absent. Skipping is right: padding the missing clips with zeros
    # would train the model on a feature that is blank most of the time.
    if "pose" in tr and "pose" in va:
        results["r3d18+pose"] = run(
            "r3d18+pose", np.concatenate([tr["vid"], tr["pose"]], -1), ytr,
            np.concatenate([va["vid"], va["pose"]], -1), yva)
    else:
        print("  r3d18+pose      skipped — no pose cache for these clips")
    results["r3d18+effnet"] = run(
        "r3d18+effnet", np.concatenate([tr["vid"], tr["effnet"]], -1), ytr,
        np.concatenate([va["vid"], va["effnet"]], -1), yva,
        test=(np.concatenate([te["vid"], te["effnet"]], -1), yte) if te else None)

    base = results["efficientnet"]["top1"]
    print("\n" + "=" * 66)
    print(f"{'backbone':<16}{'val top-1':>11}{'val top-3':>11}{'vs effnet':>12}")
    for k, r in results.items():
        print(f"{k:<16}{r['top1']*100:>10.1f}%{r['top3']*100:>10.1f}%"
              f"{(r['top1']-base)*100:>+11.1f}")
    print(f"\n{len(yva)} val clips: 1 clip = {100/len(yva):.1f} points. "
          f"Under ~3 points is noise.")

    # Selection is on VAL. The test number for that same model is the honest
    # one to quote — picking whichever scored best on test would be selecting
    # on the set that is supposed to be untouched.
    best = max(results, key=lambda k: results[k]["top1"])
    print(f"\nbest by val: {best}")
    if "test" in results[best]:
        t = results[best]["test"]
        print(f"  -> its TEST top-1 {t['top1']*100:.1f}%  "
              f"top-3 {t['top3']*100:.1f}%   (shipped weights: 57.6% / 84.8%)")
    print("per-class val top-1:")
    print(f"{'class':<12}" + "".join(f"{k[:11]:>12}" for k in results))
    for c in CLASSES:
        print(f"{c:<12}" + "".join(
            f"{results[k]['per_class'].get(c, 0)*100:>11.0f}%" for k in results))

    (OUT / "video_results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
