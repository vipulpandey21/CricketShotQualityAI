"""
shot_curve.py
Angle-value-over-time extraction, normalized so clips of different length
and frame rate can be compared frame-for-frame.

Where estimator.phase_angles() reads a shot at three fixed points (stance,
impact, follow-through), this reads it as a continuous curve from the last
moment of stillness before the downswing (estimator.shot_start_frame_index)
through impact, resampled onto a fixed number of evenly-spaced points. This
is what the "You vs Professionals" graph plots the pro band and the
uploaded clip's line against — a movement comparison instead of a single
snapshot.
"""

from __future__ import annotations

import numpy as np

from src.pose.estimator import (angles_at_frame, angles_from_world,
                                impact_frame_index, shot_start_frame_index)

ANGLE_KEYS = ["front_knee_angle", "back_knee_angle", "front_elbow_angle",
              "back_elbow_angle", "shoulder_tilt_deg", "hip_tilt_deg",
              "trunk_lean_deg"]

N_CURVE_POINTS = 25   # normalized time samples from shot-start to impact


def resample_series(xs: list, ys: list, n: int) -> list:
    """
    Resample (xs, ys) onto n evenly-spaced points spanning [xs[0], xs[-1]],
    via linear interpolation.

    `ys` entries may be None (angle not available that frame — occlusion,
    striker lost); those points are dropped before interpolating so a few
    bad frames don't break the curve. Returns [None]*n if fewer than two
    usable points remain.
    """
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 2:
        return [None] * n
    xp = [p[0] for p in pts]
    fp = [p[1] for p in pts]
    targets = np.linspace(xp[0], xp[-1], n)
    return [round(float(v), 2) for v in np.interp(targets, xp, fp)]


def raw_angle_curve(frames_kp: list, worlds: list | None, handedness: str,
                    start: int, end: int) -> dict:
    """{angle_key: [(frame_index, value_or_None), ...]} for start..end incl."""
    out = {k: [] for k in ANGLE_KEYS}
    for i in range(start, end + 1):
        if worlds is not None and i < len(worlds) and worlds[i]:
            angles = angles_from_world(worlds[i], handedness)
        elif i < len(frames_kp):
            angles = angles_at_frame(frames_kp[i], handedness)
        else:
            angles = {}
        for k in ANGLE_KEYS:
            out[k].append((i, angles.get(k)))
    return out


def normalized_shot_curve(frames_kp: list, worlds: list | None = None,
                          handedness: str = "right",
                          n: int = N_CURVE_POINTS) -> dict | None:
    """
    Full pipeline: find shot-start and impact, read the angle curve between
    them, resample each angle to n normalized time-steps
    (0 = shot start, n-1 = impact).

    Returns {"start_frame", "impact_frame", "n_points",
             "curves": {angle_key: [n floats-or-None]}}
    or None if there isn't a usable span (no pose data, or the swing is
    already underway at frame 0 with nothing earlier to anchor "start" to).
    """
    valid = [i for i, kp in enumerate(frames_kp) if kp is not None]
    if not valid:
        return None

    impact = impact_frame_index(frames_kp)
    if impact is None or frames_kp[impact] is None:
        impact = valid[len(valid) // 2]

    start = shot_start_frame_index(frames_kp, impact)
    if start is None or start >= impact:
        start = valid[0]
        if start >= impact:
            return None

    raw = raw_angle_curve(frames_kp, worlds, handedness, start, impact)
    curves = {}
    for k, pts in raw.items():
        xs = [i for i, _ in pts]
        ys = [v for _, v in pts]
        curves[k] = resample_series(xs, ys, n)

    return {"start_frame": start, "impact_frame": impact,
            "n_points": n, "curves": curves}
