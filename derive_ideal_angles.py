"""
derive_ideal_angles.py
Work out what professional batsmen actually do at impact, per shot type,
from the cached pose features — instead of guessing the ideal ranges.

Why: src/quality/scorer.py's ideal ranges are hand-written estimates, and
some are demonstrably wrong. For `hook` it wants "Arms Extension 70-120",
but a hook is played with the arms extended — data/hook/video1.mp4 measures
179 at impact — so a well-played hook is marked down. Four of the ten shots
(late_cut, square_cut, lofted, straight) have no rules at all and fall
through to a generic scorer.

Method: for every clip of a class, find the impact frame (peak wrist
movement, measured in the striker's own box coordinates so camera motion and
zoom drop out), read the joint angles there, and report the middle of the
distribution across clips. The interquartile range becomes the "ideal" band:
by construction, half of professional executions of that shot fall inside it.

Reads features/<split>_pose*.npy, which cache_pose_features.py writes as 30
frames x 46 dims per clip:
    0-25   13 joints as (x, y) inside the striker box
    26-38  per-joint visibility
    39-45  the 7 joint angles, divided by 180

Usage: python derive_ideal_angles.py [--splits train val]
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES        # noqa: E402
from src.pose.estimator import CRICKET_LANDMARKS, _first_prominent_peak  # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"

ANGLE_KEYS = ["front_knee_angle", "back_knee_angle", "front_elbow_angle",
              "back_elbow_angle", "shoulder_tilt_deg", "hip_tilt_deg",
              "trunk_lean_deg"]
JOINTS = list(CRICKET_LANDMARKS)
L_WRIST, R_WRIST = JOINTS.index(15), JOINTS.index(16)
ANGLE_OFF = 39
MIN_CLIPS = 8            # below this a per-class range is not worth quoting


def impact_index(clip: np.ndarray) -> int | None:
    """
    Frame of peak wrist movement.

    Positions are already relative to the striker's box, so this measures the
    batsman's own hand movement rather than the camera panning with him.

    Uses the same first-prominent-peak rule as estimator.impact_frame_index
    (which is why that function is imported rather than reimplemented here):
    taking the single globally-fastest frame put "impact" in the last 5 of 30
    frames for 38% of clips across this dataset, because standing up or
    setting off running after the shot often moves the hands faster than the
    swing did. Checked by eye on sweep clips specifically — the global-max
    frame regularly showed the batsman already upright with a ~160° front
    knee, which is not a sweep.
    """
    lw = clip[:, [L_WRIST * 2, L_WRIST * 2 + 1]]
    rw = clip[:, [R_WRIST * 2, R_WRIST * 2 + 1]]
    vis = clip[:, 26 + L_WRIST] + clip[:, 26 + R_WRIST]
    mid = (lw + rw) / 2

    speeds = []
    for i in range(1, len(mid)):
        if vis[i] <= 0 or vis[i - 1] <= 0:
            continue
        speeds.append((float(np.linalg.norm(mid[i] - mid[i - 1])), i))
    return _first_prominent_peak(speeds)


def load_split(split: str):
    """Return (pose_array, labels) for whichever pose cache exists."""
    for name in (f"{split}_pose.npy", f"{split}_pose_n40.npy"):
        arr = FEAT / name
        if not arr.exists():
            continue
        stem = name[:-4]
        paths = (FEAT / f"{stem}_paths.txt").read_text(
            encoding="utf-8").splitlines()
        labels = []
        for p in paths:
            cls = Path(p).parent.name
            labels.append(CLASSES.index(cls) if cls in CLASSES else -1)
        return np.load(arr), np.array(labels)
    return None, None


def main():
    splits = ["train", "val"]
    if "--splits" in sys.argv:
        splits = sys.argv[sys.argv.index("--splits") + 1:]

    per_class = {c: {k: [] for k in ANGLE_KEYS} for c in CLASSES}
    n_used = {c: 0 for c in CLASSES}

    for split in splits:
        X, y = load_split(split)
        if X is None:
            print(f"{split}: no pose cache, skipping")
            continue
        print(f"{split}: {len(X)} clips")
        for clip, label in zip(X, y):
            if label < 0 or not np.abs(clip).any():
                continue
            i = impact_index(clip)
            if i is None:
                continue
            angles = clip[i, ANGLE_OFF:ANGLE_OFF + len(ANGLE_KEYS)] * 180.0
            if not np.any(angles):
                continue
            cls = CLASSES[label]
            n_used[cls] += 1
            for k, a in zip(ANGLE_KEYS, angles):
                if a > 0:                       # 0 means the angle was missing
                    per_class[cls][k].append(float(a))

    out = {}
    print(f"\n{'class':<12}{'n':>4}  " +
          "".join(f"{k.replace('_angle','').replace('_deg',''):>18}"
                  for k in ANGLE_KEYS))
    for cls in CLASSES:
        row = {}
        cells = []
        for k in ANGLE_KEYS:
            vals = np.array(per_class[cls][k])
            if len(vals) < MIN_CLIPS:
                cells.append(f"{'-':>18}")
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            row[k] = {"low": round(float(q1), 1), "median": round(float(med), 1),
                      "high": round(float(q3), 1), "n": int(len(vals))}
            cells.append(f"{q1:5.0f}-{q3:<4.0f}(m{med:3.0f})".rjust(18))
        out[cls] = row
        print(f"{cls:<12}{n_used[cls]:>4}  " + "".join(cells))

    dest = ROOT / "ideal_angles.json"
    dest.write_text(json.dumps({
        "_note": "Interquartile range of each joint angle at the impact frame, "
                 "measured across professional clips of each shot type. "
                 "'low'/'high' are the 25th/75th percentiles, so half of real "
                 "executions of the shot land inside the band.",
        "_source_splits": splits,
        "_min_clips_per_range": MIN_CLIPS,
        "classes": out,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")

    print("\nCompare against the hand-written ranges in src/quality/scorer.py "
          "before replacing them — a range derived from few clips, or from a "
          "shot the pose pipeline handles poorly, is not automatically better.")


if __name__ == "__main__":
    main()
