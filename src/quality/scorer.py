"""
scorer.py
Shot-specific biomechanical quality scoring.

For each shot type, defines ideal angle ranges and scores each criterion 0-100.
Final score = weighted average of all criteria.

Scoring formula per criterion:
  - angle within ideal range            → 100
  - angle within tolerance (±15°)       → 60-99 (linear decay)
  - angle outside tolerance             → 0-59  (further decay)
"""

from dataclasses import dataclass, field


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


def _score_cover(angles: dict) -> list:
    criteria = []

    # Front knee: 130-155° (slight bend, not locked, not too deep)
    s = _score_angle(angles.get("front_knee_angle"), 130, 155)
    actual_fk = f"{angles['front_knee_angle']}°" if angles.get("front_knee_angle") is not None else "Not detected"
    criteria.append(CriterionResult("Front Knee Bend", "130°–155°", actual_fk, s, _status(s)))

    # Front elbow angle: 100-150° (elbow bent, bat coming through)
    s = _score_angle(angles.get("front_elbow_angle"), 100, 150)
    actual_fe = f"{angles['front_elbow_angle']}°" if angles.get("front_elbow_angle") is not None else "Not detected"
    criteria.append(CriterionResult("Lead Elbow Position", "100°–150°", actual_fe, s, _status(s)))

    # Shoulder tilt: keep low (<10°)
    s = _score_tilt(angles.get("shoulder_tilt_deg"), 10)
    actual_st = f"{angles['shoulder_tilt_deg']}°" if angles.get("shoulder_tilt_deg") is not None else "Not detected"
    criteria.append(CriterionResult("Shoulder Alignment", "<10° tilt", actual_st, s, _status(s)))

    # Trunk lean: slight forward lean 5-20°
    s = _score_angle(angles.get("trunk_lean_deg"), 5, 25)
    actual_tl = f"{angles['trunk_lean_deg']}°" if angles.get("trunk_lean_deg") is not None else "Not detected"
    criteria.append(CriterionResult("Body Lean (Forward)", "5°–25°", actual_tl, s, _status(s)))

    return criteria


def _score_pull(angles: dict) -> list:
    criteria = []

    s = _score_angle(angles.get("back_knee_angle"), 110, 140)
    criteria.append(CriterionResult("Back Knee Bend", "110°–140°", _fmt(angles, "back_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("back_elbow_angle"), 80, 130)
    criteria.append(CriterionResult("Arms Extension", "80°–130°", _fmt(angles, "back_elbow_angle"), s, _status(s)))

    s = _score_tilt(angles.get("hip_tilt_deg"), 20, tolerance=10)
    criteria.append(CriterionResult("Hip Rotation", "<20°", _fmt(angles, "hip_tilt_deg"), s, _status(s)))

    s = _score_tilt(angles.get("shoulder_tilt_deg"), 15)
    criteria.append(CriterionResult("Shoulder Alignment", "<15° tilt", _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))

    return criteria


def _score_sweep(angles: dict) -> list:
    criteria = []

    s = _score_angle(angles.get("front_knee_angle"), 80, 110)
    criteria.append(CriterionResult("Front Knee (Deep Bend)", "80°–110°", _fmt(angles, "front_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("back_knee_angle"), 90, 130)
    criteria.append(CriterionResult("Back Knee Position", "90°–130°", _fmt(angles, "back_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("front_elbow_angle"), 90, 140)
    criteria.append(CriterionResult("Bat Swing Plane", "90°–140°", _fmt(angles, "front_elbow_angle"), s, _status(s)))

    s = _score_angle(angles.get("trunk_lean_deg"), 20, 45)
    criteria.append(CriterionResult("Forward Body Lean", "20°–45°", _fmt(angles, "trunk_lean_deg"), s, _status(s)))

    return criteria


def _score_defense(angles: dict) -> list:
    criteria = []

    s = _score_angle(angles.get("front_knee_angle"), 140, 175)
    criteria.append(CriterionResult("Upright Stance (Knee)", "140°–175°", _fmt(angles, "front_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("front_elbow_angle"), 100, 150)
    criteria.append(CriterionResult("Soft Hands (Elbow)", "100°–150°", _fmt(angles, "front_elbow_angle"), s, _status(s)))

    s = _score_tilt(angles.get("shoulder_tilt_deg"), 8)
    criteria.append(CriterionResult("Shoulder Level", "<8° tilt", _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))

    s = _score_angle(angles.get("trunk_lean_deg"), 0, 15)
    criteria.append(CriterionResult("Upright Body", "0°–15° lean", _fmt(angles, "trunk_lean_deg"), s, _status(s)))

    return criteria


def _score_hook(angles: dict) -> list:
    criteria = []

    s = _score_angle(angles.get("back_knee_angle"), 100, 135)
    criteria.append(CriterionResult("Back Knee Bend", "100°–135°", _fmt(angles, "back_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("back_elbow_angle"), 70, 120)
    criteria.append(CriterionResult("Arms Extension", "70°–120°", _fmt(angles, "back_elbow_angle"), s, _status(s)))

    s = _score_tilt(angles.get("shoulder_tilt_deg"), 20)
    criteria.append(CriterionResult("Shoulder Rotation", "<20°", _fmt(angles, "shoulder_tilt_deg"), s, _status(s)))

    return criteria


def _score_flick(angles: dict) -> list:
    criteria = []

    s = _score_angle(angles.get("front_knee_angle"), 120, 150)
    criteria.append(CriterionResult("Front Knee Bend", "120°–150°", _fmt(angles, "front_knee_angle"), s, _status(s)))

    s = _score_angle(angles.get("front_elbow_angle"), 90, 140)
    criteria.append(CriterionResult("Wrist Flick (Elbow)", "90°–140°", _fmt(angles, "front_elbow_angle"), s, _status(s)))

    s = _score_tilt(angles.get("hip_tilt_deg"), 15)
    criteria.append(CriterionResult("Hip Rotation", "<15°", _fmt(angles, "hip_tilt_deg"), s, _status(s)))

    return criteria


def _score_generic(angles: dict) -> list:
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
    "late_cut":   _score_generic,
    "lofted":     _score_generic,
    "square_cut": _score_generic,
    "straight":   _score_generic,
}


def score_shot(shot_type: str, angles: dict) -> ShotQualityResult:
    """
    Compute quality score for a shot given computed joint angles.

    Args:
        shot_type: e.g. "cover", "pull", "sweep"
        angles:    dict from compute_cricket_angles()

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
