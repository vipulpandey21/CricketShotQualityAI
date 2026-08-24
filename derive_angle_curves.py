"""
derive_angle_curves.py
Build the professional "movement" reference for the You-vs-Professionals
graph: instead of one snapshot at impact (derive_ideal_angles.py), this
reads each angle continuously from shot-start to impact, resamples every
clip onto the same normalized time axis, and reports the per-timestep
interquartile band across professional clips — so an uploaded video's own
curve can be laid directly over the band it should be inside at every
point in the swing, not just at the end of it.

Uses the EXACT SAME shot-start/impact rules as the live app
(estimator._first_prominent_peak / _nearest_quiet_minimum, imported rather
than reimplemented) so the professional reference and an uploaded video's
curve are measured the same way.

Reads features/<split>_pose*.npy — see derive_ideal_angles.py's docstring
for the cache layout (30 frames x 46 dims: 0-25 joints, 26-38 visibility,
39-45 angles/180).

Usage: python derive_angle_curves.py [--splits train val]
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.classifier.model import SHOT_CLASSES                         # noqa: E402
from src.pose.estimator import (CRICKET_LANDMARKS, _first_prominent_peak,  # noqa: E402
                                _nearest_quiet_minimum)
from src.pose.shot_curve import ANGLE_KEYS, N_CURVE_POINTS, resample_series  # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"

JOINTS = list(CRICKET_LANDMARKS)
L_WRIST, R_WRIST = JOINTS.index(15), JOINTS.index(16)
ANGLE_OFF = 39
MIN_CLIPS = 8            # below this a per-class band is not worth quoting


def wrist_speeds(clip: np.ndarray) -> list:
    """[(speed, frame_index), ...] from the box-relative wrist columns."""
    lw = clip[:, [L_WRIST * 2, L_WRIST * 2 + 1]]
    rw = clip[:, [R_WRIST * 2, R_WRIST * 2 + 1]]
    vis = clip[:, 26 + L_WRIST] + clip[:, 26 + R_WRIST]
    mid = (lw + rw) / 2

    speeds = []
    for i in range(1, len(mid)):
        if vis[i] <= 0 or vis[i - 1] <= 0:
            continue
        speeds.append((float(np.linalg.norm(mid[i] - mid[i - 1])), i))
    return speeds


def start_and_impact(clip: np.ndarray):
    """(start_index, impact_index) or (None, None) if no usable signal."""
    speeds = wrist_speeds(clip)
    if not speeds:
        return None, None
    impact = _first_prominent_peak(speeds)
    if impact is None:
        return None, None
    start = _nearest_quiet_minimum(speeds, impact)
    if start is None or start >= impact:
        return None, None
    return start, impact


def clip_curve(clip: np.ndarray, start: int, impact: int) -> dict:
    """{angle_key: [n resampled values]} for frames start..impact inclusive."""
    out = {}
    frames = list(range(start, impact + 1))
    for k, key in enumerate(ANGLE_KEYS):
        col = ANGLE_OFF + k
        xs, ys = [], []
        for i in frames:
            v = float(clip[i, col]) * 180.0
            xs.append(i)
            ys.append(v if v > 0 else None)   # 0 means the angle was missing
        out[key] = resample_series(xs, ys, N_CURVE_POINTS)
    return out


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

    # per_class[cls][angle] = list of [n_points] curves (one per clip)
    per_class = {c: {k: [] for k in ANGLE_KEYS} for c in CLASSES}
    n_used = {c: 0 for c in CLASSES}
    n_skipped_no_span = {c: 0 for c in CLASSES}
    gaps = {c: [] for c in CLASSES}

    for split in splits:
        X, y = load_split(split)
        if X is None:
            print(f"{split}: no pose cache, skipping")
            continue
        print(f"{split}: {len(X)} clips")
        for clip, label in zip(X, y):
            if label < 0 or not np.abs(clip).any():
                continue
            cls = CLASSES[label]
            start, impact = start_and_impact(clip)
            if start is None:
                n_skipped_no_span[cls] += 1
                continue
            curve = clip_curve(clip, start, impact)
            if all(v is None for v in curve[ANGLE_KEYS[0]]):
                n_skipped_no_span[cls] += 1
                continue
            n_used[cls] += 1
            gaps[cls].append(impact - start)
            for k in ANGLE_KEYS:
                per_class[cls][k].append(curve[k])

    out = {}
    print(f"\n{'class':<12}{'n':>4}{'skipped':>9}{'avg_gap':>9}")
    for cls in CLASSES:
        n = n_used[cls]
        avg_gap = (sum(gaps[cls]) / len(gaps[cls])) if gaps[cls] else 0.0
        print(f"{cls:<12}{n:>4}{n_skipped_no_span[cls]:>9}{avg_gap:>9.1f}")

        if n < MIN_CLIPS:
            continue
        cls_out = {}
        for k in ANGLE_KEYS:
            # [n_clips, N_CURVE_POINTS] — column j is every clip's value at
            # normalized timestep j (None where a clip had no data there).
            mat = per_class[cls][k]
            band = []
            for j in range(N_CURVE_POINTS):
                vals = [c[j] for c in mat if c[j] is not None]
                if len(vals) < MIN_CLIPS:
                    band.append(None)
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                band.append({"low": round(float(q1), 1),
                             "median": round(float(med), 1),
                             "high": round(float(q3), 1),
                             "n": len(vals)})
            cls_out[k] = band
        out[cls] = cls_out

    dest = ROOT / "ideal_angle_curves.json"
    dest.write_text(json.dumps({
        "_note": "Each angle resampled to n_points evenly-spaced steps from "
                 "shot-start (last pause before the downswing) to impact, "
                 "per professional clip, then the interquartile band taken "
                 "at each step across clips of the same shot type. Index 0 "
                 "= shot start, index n_points-1 = impact. Uses the same "
                 "shot-start/impact rules as the live app "
                 "(src/pose/estimator.py, src/pose/shot_curve.py).",
        "_source_splits": splits,
        "_min_clips_per_range": MIN_CLIPS,
        "n_points": N_CURVE_POINTS,
        "classes": out,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
