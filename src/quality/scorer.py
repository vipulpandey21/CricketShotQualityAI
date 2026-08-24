"""
scorer.py
Shot-specific biomechanical quality scoring.

For each shot type, scores joint angles (measured at the impact frame, in
metric 3D from MediaPipe world landmarks — see estimator.angles_from_world)
against an ideal range and produces a 0-100 score per criterion.

Ideal ranges come from `ideal_angles.json`, the interquartile range (25th to
75th percentile) of each angle measured at impact across ~60-125 professional
clips per shot type, from derive_ideal_angles.py. By construction, half of
real executions of the shot fall inside that band, so it is a genuine "this
is what good players do" range rather than a guess.

Which angles matter for which shot — front knee for a cover drive, back
elbow for a hook — is cricket-coaching knowledge and is unchanged from the
original hand-written version. What changed is where the numeric bounds come
from. An earlier version of this file was suspected of being wrong (its hook
range asks for the elbow to stay at 70-120 degrees, which looked backwards
for a shot played with the arms extended) and was nearly rewritten on that
basis — but the range was measured at 93 degrees median and was correct; the
single clip that motivated the doubt was an outlier from a 2D-angle bug
elsewhere in the pipeline, not this file's ranges. This file's original
numbers were sound. The data-driven bounds below are not a correction of an
error so much as replacing estimates with measurements, and — new — they let
four shot types that had no rules at all (late_cut, square_cut, lofted,
straight — previously scored by _score_generic) get real ones.

If ideal_angles.json is missing (fresh clone, before derive_ideal_angles.py
has been run), every range falls back to the original hand-written numbers,
so scoring degrades gracefully rather than breaking.

Scoring formula per criterion:
  - angle within ideal range            → 100
  - angle within tolerance (±15°)       → 60-99 (linear decay)
  - angle outside tolerance             → 0-59  (further decay)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

IDEAL_ANGLES_PATH = Path(__file__).resolve().parents[2] / "ideal_angles.json"


@dataclass
class CriterionResult:
    name: str           # e.g. "Front Knee Bend"
    ideal: str          # e.g. "130°–150°"
    actual: str         # e.g. "118°"
    score: float        # 0–100
    status: str         # "✅ Good" / "⚠️ Fair" / "❌ Needs Work"


@dataclass
class ShotQualityResult:
    shot_type: str
    overall_score: float          # 0–100
    grade: str                    # Excellent / Good / Average / Needs Work
    criteria: list = field(default_factory=list)   # list of CriterionResult


def _load_derived_ranges() -> dict:
    if not IDEAL_ANGLES_PATH.exists():
        return {}
    try:
        data = json.loads(IDEAL_ANGLES_PATH.read_text(encoding="utf-8"))
        return data.get("classes", {})
    except (json.JSONDecodeError, OSError):
        return {}


_DERIVED = _load_derived_ranges()


def _range(shot_type: str, key: str, fallback: tuple) -> tuple:
    """
    (low, high) for one angle of one shot, preferring the measured
    interquartile range and falling back to the hand-written estimate when
    there isn't enough measured data (a range needs derive_ideal_angles.py's
    MIN_CLIPS_PER_RANGE, currently 8, professional clips to be trusted).
    """
    entry = _DERIVED.get(shot_type, {}).get(key)
    if entry and "low" in entry and "high" in entry:
        return entry["low"], entry["high"]
    return fallback


def _score_angle(actual: float, ideal_low: float, ideal_high: float,
                 tolerance: float = 15.0) -> float:
    """
    Score an angle against an ideal range [ideal_low, ideal_high].
    Returns 0-100.
    """
    if actual is None:
        return 50.0  # neutral if not detected

    # Inside ideal range → full marks
    if ideal_low <= actual <= ideal_high:
        return 100.0

    # How far outside?
    if actual < ideal_low:
        diff = ideal_low - actual
    else:
        diff = actual - ideal_high

    if diff <= tolerance:
        # Linear decay from 100 to 60 within tolerance zone
        return round(100.0 - (diff / tolerance) * 40.0, 1)
    elif diff <= tolerance * 3:
        # Steeper decay from 60 to 10
        extra = diff - tolerance
        return round(60.0 - (extra / (tolerance * 2)) * 50.0, 1)
    else:
        return max(0.0, round(10.0 - (diff - tolerance * 3) * 2, 1))


def _score_tilt(actual: float, ideal_max: float, tolerance: float = 8.0) -> float:
    """Score a tilt angle — lower is better (0° = perfectly level)."""
    if actual is None:
        return 50.0
    if actual <= ideal_max:
        return 100.0
    diff = actual - ideal_max
    if diff <= tolerance:
        return round(100.0 - (diff / tolerance) * 40.0, 1)
    return max(0.0, round(60.0 - (diff - tolerance) * 5, 1))


def _grade(score: float) -> str:
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Needs Work"


def _status(score: float) -> str:
    if score >= 80:
        return "✅ Good"
    elif score >= 55:
        return "⚠️ Fair"
    else:
        return "❌ Needs Work"


def _fmt(angles: dict, key: str) -> str:
    """Format an angle value for display — shows 'Not detected' if None."""
    v = angles.get(key)
    return f"{v}°" if v is not None else "Not detected"


def _range_label(low: float, high: float) -> str:
    return f"{low:.0f}°–{high:.0f}°"


def _angle_criterion(shot: str, angles: dict, key: str, label: str,
                     fallback: tuple) -> CriterionResult:
    """One angle-range criterion, ranges sourced via `_range`."""
    low, high = _range(shot, key, fallback)
    s = _score_angle(angles.get(key), low, high)
    return CriterionResult(label, _range_label(low, high), _fmt(angles, key),
                           s, _status(s))


def _score_cover(angles: dict) -> list:
    criteria = [
        _angle_criterion("cover", angles, "front_knee_angle",
                         "Front Knee Bend", (130, 155)),
        _angle_criterion("cover", angles, "front_elbow_angle",
                         "Lead Elbow Position", (100, 150)),
        _angle_criterion("cover", angles, "trunk_lean_deg",
                         "Body Lean (Forward)", (5, 25)),
    ]
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 10)
    criteria.insert(2, CriterionResult(
        "Shoulder Alignment", "<10° tilt", _fmt(angles, "shoulder_tilt_deg"),
        s, _status(s)))
    return criteria


def _score_pull(angles: dict) -> list:
    criteria = [
        _angle_criterion("pull", angles, "back_knee_angle",
                         "Back Knee Bend", (110, 140)),
        _angle_criterion("pull", angles, "back_elbow_angle",
                         "Arms Extension", (80, 130)),
    ]
    s = _score_tilt(angles.get("hip_tilt_deg"), 20, tolerance=10)
    criteria.append(CriterionResult("Hip Rotation", "<20°",
                                    _fmt(angles, "hip_tilt_deg"), s, _status(s)))
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 15)
    criteria.append(CriterionResult("Shoulder Alignment", "<15° tilt",
                                    _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))
    return criteria


def _score_sweep(angles: dict) -> list:
    return [
        _angle_criterion("sweep", angles, "front_knee_angle",
                         "Front Knee (Deep Bend)", (80, 110)),
        _angle_criterion("sweep", angles, "back_knee_angle",
                         "Back Knee Position", (90, 130)),
        _angle_criterion("sweep", angles, "front_elbow_angle",
                         "Bat Swing Plane", (90, 140)),
        _angle_criterion("sweep", angles, "trunk_lean_deg",
                         "Forward Body Lean", (20, 45)),
    ]


def _score_defense(angles: dict) -> list:
    criteria = [
        _angle_criterion("defense", angles, "front_knee_angle",
                         "Upright Stance (Knee)", (140, 175)),
        _angle_criterion("defense", angles, "front_elbow_angle",
                         "Soft Hands (Elbow)", (100, 150)),
    ]
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 8)
    criteria.append(CriterionResult("Shoulder Level", "<8° tilt",
                                    _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))
    criteria.append(_angle_criterion("defense", angles, "trunk_lean_deg",
                                     "Upright Body", (0, 15)))
    return criteria


def _score_hook(angles: dict) -> list:
    criteria = [
        _angle_criterion("hook", angles, "back_knee_angle",
                         "Back Knee Bend", (100, 135)),
        _angle_criterion("hook", angles, "back_elbow_angle",
                         "Arms Extension", (70, 120)),
    ]
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 20)
    criteria.append(CriterionResult("Shoulder Rotation", "<20°",
                                    _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))
    return criteria


def _score_flick(angles: dict) -> list:
    criteria = [
        _angle_criterion("flick", angles, "front_knee_angle",
                         "Front Knee Bend", (120, 150)),
        _angle_criterion("flick", angles, "front_elbow_angle",
                         "Wrist Flick (Elbow)", (90, 140)),
    ]
    s = _score_tilt(angles.get("hip_tilt_deg"), 15)
    criteria.append(CriterionResult("Hip Rotation", "<15°",
                                    _fmt(angles, "hip_tilt_deg"), s, _status(s)))
    return criteria


# ── Previously generic-only shots — now with real, measured rules ─────────
# The choice of WHICH angles matter for each shot is cricket-coaching logic
# (unchanged in spirit from the six shots above); only the numeric ranges are
# new, sourced the same way as everywhere else in this file.

def _score_late_cut(angles: dict) -> list:
    return [
        _angle_criterion("late_cut", angles, "front_elbow_angle",
                         "Controlled Elbow (close to body)", (90, 140)),
        _angle_criterion("late_cut", angles, "trunk_lean_deg",
                         "Lean Into the Cut", (10, 35)),
        _angle_criterion("late_cut", angles, "front_knee_angle",
                         "Base (Knee Bend)", (120, 165)),
    ]


def _score_square_cut(angles: dict) -> list:
    return [
        _angle_criterion("square_cut", angles, "back_elbow_angle",
                         "Horizontal Bat Swing", (80, 140)),
        _angle_criterion("square_cut", angles, "trunk_lean_deg",
                         "Lean Into the Cut", (10, 35)),
        _angle_criterion("square_cut", angles, "front_knee_angle",
                         "Base (Knee Bend)", (120, 165)),
    ]


def _score_lofted(angles: dict) -> list:
    criteria = [
        _angle_criterion("lofted", angles, "front_elbow_angle",
                         "Full Extension Through the Line", (90, 160)),
        _angle_criterion("lofted", angles, "trunk_lean_deg",
                         "Lean Back to Elevate", (15, 45)),
    ]
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 30)
    criteria.append(CriterionResult("Shoulder Rotation", "<30°",
                                    _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))
    return criteria


def _score_straight(angles: dict) -> list:
    return [
        _angle_criterion("straight", angles, "front_knee_angle",
                         "Full Extension Down the Line", (140, 175)),
        _angle_criterion("straight", angles, "front_elbow_angle",
                         "Straight Bat Path", (100, 160)),
        _angle_criterion("straight", angles, "trunk_lean_deg",
                         "Staying Tall", (0, 20)),
    ]


def _score_generic(angles: dict) -> list:
    """
    Last-resort fallback — reached only for a shot name outside the ten
    trained classes, which should not happen through the normal prediction
    path but is kept so an unexpected label cannot crash scoring.
    """
    criteria = []

    s = _score_angle(angles.get("front_knee_angle"), 120, 160)
    criteria.append(CriterionResult("Front Knee Position", "120°–160°", _fmt(angles, "front_knee_angle"), s, _status(s)))

    s = _score_tilt(angles.get("shoulder_tilt_deg"), 15)
    criteria.append(CriterionResult("Shoulder Alignment", "<15° tilt", _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))

    s = _score_angle(angles.get("trunk_lean_deg"), 0, 30)
    criteria.append(CriterionResult("Body Balance", "0°–30° lean", _fmt(angles, "trunk_lean_deg"), s, _status(s)))

    return criteria


# Shot → scoring function mapping
_SHOT_SCORERS = {
    "cover":      _score_cover,
    "pull":       _score_pull,
    "hook":       _score_hook,
    "sweep":      _score_sweep,
    "defense":    _score_defense,
    "flick":      _score_flick,
    "late_cut":   _score_late_cut,
    "lofted":     _score_lofted,
    "square_cut": _score_square_cut,
    "straight":   _score_straight,
}


def score_shot(shot_type: str, angles: dict) -> ShotQualityResult:
    """
    Compute quality score for a shot given computed joint angles.

    Args:
        shot_type: e.g. "cover", "pull", "sweep"
        angles:    dict from compute_cricket_angles() / phase_angles()["impact"]

    Returns:
        ShotQualityResult with overall score, grade, and per-criterion breakdown.
    """
    scorer_fn = _SHOT_SCORERS.get(shot_type, _score_generic)
    criteria = scorer_fn(angles)

    if criteria:
        overall = round(sum(c.score for c in criteria) / len(criteria), 1)
    else:
        overall = 50.0

    return ShotQualityResult(
        shot_type=shot_type,
        overall_score=overall,
        grade=_grade(overall),
        criteria=criteria,
    )
