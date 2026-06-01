"""
scorer.py
Quantifies the correctness/quality of a cricket shot using two complementary signals:

1. Pose-based biomechanical scoring  — compares player keypoints against
   shot-specific ideal rules (angles, stances, weight transfer).
2. Feature-based similarity scoring — cosine similarity of EfficientNet
   feature vectors against a reference "gold standard" video.

Final quality score = weighted average of both signals (0–100).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShotFeedback:
    """Holds the quality score and per-criterion feedback for one shot."""
    shot_type: str
    pose_score: float           # 0–100
    similarity_score: float     # 0–100  (-1 if no reference available)
    final_score: float          # 0–100
    criteria: dict[str, float] = field(default_factory=dict)   # criterion → score
    feedback: list[str] = field(default_factory=list)          # human-readable tips

    @property
    def grade(self) -> str:
        if self.final_score >= 85:
            return "Excellent"
        elif self.final_score >= 70:
            return "Good"
        elif self.final_score >= 50:
            return "Average"
        else:
            return "Needs Work"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle (degrees) at point B formed by vectors BA and BC.
    All points are (x, y) or (x, y, z) arrays.
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _score_angle(actual: float, ideal: float, tolerance: float = 15.0) -> float:
    """
    Score an angle against an ideal value.
    Returns 100 if within tolerance, linearly decays to 0 at 2× tolerance.
    """
    diff = abs(actual - ideal)
    if diff <= tolerance:
        return 100.0
    elif diff <= 2 * tolerance:
        return 100.0 * (1 - (diff - tolerance) / tolerance)
    return 0.0


def _lateral_balance(left_ankle: np.ndarray, right_ankle: np.ndarray,
                      left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """
    Estimate weight distribution balance (0–100).
    Checks whether the hip centre is roughly over the foot midpoint.
    """
    foot_mid_x = (left_ankle[0] + right_ankle[0]) / 2
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    diff = abs(foot_mid_x - hip_mid_x)
    # diff is in normalised image coords (0–1); >0.1 is noticeably off-balance
    return max(0.0, 100.0 - diff * 800)


# ---------------------------------------------------------------------------
# Shot-specific biomechanical rule sets
# ---------------------------------------------------------------------------
# Each rule set is a function:
#   (avg_keypoints: dict[int, np.ndarray]) -> (criteria: dict, feedback: list, score: float)
#
# MediaPipe landmark indices used:
#   11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow
#   15=L_wrist,    16=R_wrist,    23=L_hip,   24=R_hip
#   25=L_knee,     26=R_knee,     27=L_ankle, 28=R_ankle

def _get(kp: dict, idx: int) -> np.ndarray | None:
    return kp.get(idx)


def _rule_cover_drive(kp: dict) -> tuple[dict, list, float]:
    criteria, feedback = {}, []

    # 1. Front knee bend — should be ~130–150° (slight flex, not locked)
    lhip, lknee, lankle = _get(kp, 23), _get(kp, 25), _get(kp, 27)
    if all(v is not None for v in [lhip, lknee, lankle]):
        angle = _angle_between(lhip, lknee, lankle)
        s = _score_angle(angle, ideal=140, tolerance=15)
        criteria["front_knee_bend"] = s
        if s < 60:
            feedback.append("Bend your front knee more — aim for ~140° for a solid cover drive base.")
    else:
        criteria["front_knee_bend"] = 50.0

    # 2. Elbow height — lead elbow should be above wrist level
    lelbow, lwrist = _get(kp, 13), _get(kp, 15)
    if lelbow is not None and lwrist is not None:
        # In image coords y increases downward, so elbow.y < wrist.y means elbow is higher
        elbow_up = lwrist[1] - lelbow[1]   # positive = elbow above wrist
        s = min(100.0, max(0.0, 50 + elbow_up * 400))
        criteria["lead_elbow_position"] = s
        if s < 60:
            feedback.append("Keep your lead elbow up through the cover drive swing.")
    else:
        criteria["lead_elbow_position"] = 50.0

    # 3. Shoulder alignment — shoulders should be roughly level (not drooping)
    lshoulder, rshoulder = _get(kp, 11), _get(kp, 12)
    if lshoulder is not None and rshoulder is not None:
        tilt = abs(lshoulder[1] - rshoulder[1])
        s = max(0.0, 100.0 - tilt * 600)
        criteria["shoulder_alignment"] = s
        if s < 60:
            feedback.append("Keep your shoulders level — avoid excessive tilt during the shot.")
    else:
        criteria["shoulder_alignment"] = 50.0

    score = float(np.mean(list(criteria.values())))
    return criteria, feedback, score


def _rule_pull_shot(kp: dict) -> tuple[dict, list, float]:
    criteria, feedback = {}, []

    # 1. Back foot pivot — back knee should be bent (~120°)
    rhip, rknee, rankle = _get(kp, 24), _get(kp, 26), _get(kp, 28)
    if all(v is not None for v in [rhip, rknee, rankle]):
        angle = _angle_between(rhip, rknee, rankle)
        s = _score_angle(angle, ideal=120, tolerance=20)
        criteria["back_knee_bend"] = s
        if s < 60:
            feedback.append("Bend your back knee more to get into a good pull-shot position.")
    else:
        criteria["back_knee_bend"] = 50.0

    # 2. Arm extension — wrists should be above shoulder level at contact
    lwrist, lshoulder = _get(kp, 15), _get(kp, 11)
    if lwrist is not None and lshoulder is not None:
        wrist_above = lshoulder[1] - lwrist[1]   # positive = wrist above shoulder
        s = min(100.0, max(0.0, 50 + wrist_above * 400))
        criteria["arm_extension"] = s
        if s < 60:
            feedback.append("Extend your arms fully and get your wrists above shoulder height for the pull.")
    else:
        criteria["arm_extension"] = 50.0

    # 3. Hip rotation — hips should rotate (left hip moves forward)
    lhip, rhip2 = _get(kp, 23), _get(kp, 24)
    if lhip is not None and rhip2 is not None:
        rotation = rhip2[0] - lhip[0]   # positive = hips open
        s = min(100.0, max(0.0, 50 + rotation * 300))
        criteria["hip_rotation"] = s
        if s < 60:
            feedback.append("Rotate your hips through the pull shot for more power.")
    else:
        criteria["hip_rotation"] = 50.0

    score = float(np.mean(list(criteria.values())))
    return criteria, feedback, score


def _rule_sweep(kp: dict) -> tuple[dict, list, float]:
    criteria, feedback = {}, []

    # 1. Front knee on ground — front knee should be very bent (~90°)
    lhip, lknee, lankle = _get(kp, 23), _get(kp, 25), _get(kp, 27)
    if all(v is not None for v in [lhip, lknee, lankle]):
        angle = _angle_between(lhip, lknee, lankle)
        s = _score_angle(angle, ideal=90, tolerance=20)
        criteria["front_knee_down"] = s
        if s < 60:
            feedback.append("Get your front knee lower — the sweep requires a deep knee bend (~90°).")
    else:
        criteria["front_knee_down"] = 50.0

    # 2. Head position — head should be low (nose y close to knee y)
    nose, lknee2 = _get(kp, 0), _get(kp, 25)
    if nose is not None and lknee2 is not None:
        head_low = abs(nose[1] - lknee2[1])
        s = max(0.0, 100.0 - head_low * 500)
        criteria["head_position"] = s
        if s < 60:
            feedback.append("Get your head down and eyes level with the ball for the sweep.")
    else:
        criteria["head_position"] = 50.0

    # 3. Bat swing — wrist should be low (below hip level)
    lwrist, lhip2 = _get(kp, 15), _get(kp, 23)
    if lwrist is not None and lhip2 is not None:
        wrist_low = lwrist[1] - lhip2[1]   # positive = wrist below hip
        s = min(100.0, max(0.0, 50 + wrist_low * 400))
        criteria["bat_swing_plane"] = s
        if s < 60:
            feedback.append("Keep the bat swing low and horizontal for an effective sweep.")
    else:
        criteria["bat_swing_plane"] = 50.0

    score = float(np.mean(list(criteria.values())))
    return criteria, feedback, score


def _rule_defense(kp: dict) -> tuple[dict, list, float]:
    criteria, feedback = {}, []

    # 1. Upright stance — back should be relatively straight
    lshoulder, lhip, lknee = _get(kp, 11), _get(kp, 23), _get(kp, 25)
    if all(v is not None for v in [lshoulder, lhip, lknee]):
        angle = _angle_between(lshoulder, lhip, lknee)
        s = _score_angle(angle, ideal=170, tolerance=15)
        criteria["upright_stance"] = s
        if s < 60:
            feedback.append("Stay more upright for the defensive shot — keep your back straight.")
    else:
        criteria["upright_stance"] = 50.0

    # 2. Soft hands — elbows should be close to body (not flared)
    lelbow, lshoulder2 = _get(kp, 13), _get(kp, 11)
    if lelbow is not None and lshoulder2 is not None:
        elbow_dist = abs(lelbow[0] - lshoulder2[0])
        s = max(0.0, 100.0 - elbow_dist * 500)
        criteria["elbow_tuck"] = s
        if s < 60:
            feedback.append("Tuck your elbows in for soft hands on the defensive shot.")
    else:
        criteria["elbow_tuck"] = 50.0

    # 3. Balance — lateral balance check
    lankle, rankle = _get(kp, 27), _get(kp, 28)
    lhip2, rhip2 = _get(kp, 23), _get(kp, 24)
    if all(v is not None for v in [lankle, rankle, lhip2, rhip2]):
        s = _lateral_balance(lankle, rankle, lhip2, rhip2)
        criteria["balance"] = s
        if s < 60:
            feedback.append("Distribute your weight evenly — stay balanced over both feet.")
    else:
        criteria["balance"] = 50.0

    score = float(np.mean(list(criteria.values())))
    return criteria, feedback, score


def _rule_generic(kp: dict) -> tuple[dict, list, float]:
    """Fallback rule set for shots without specific rules yet."""
    criteria, feedback = {}, []

    # Basic balance check
    lankle, rankle = _get(kp, 27), _get(kp, 28)
    lhip, rhip = _get(kp, 23), _get(kp, 24)
    if all(v is not None for v in [lankle, rankle, lhip, rhip]):
        s = _lateral_balance(lankle, rankle, lhip, rhip)
        criteria["balance"] = s
        if s < 60:
            feedback.append("Work on your balance — keep your weight centred.")
    else:
        criteria["balance"] = 50.0

    # Shoulder level
    lshoulder, rshoulder = _get(kp, 11), _get(kp, 12)
    if lshoulder is not None and rshoulder is not None:
        tilt = abs(lshoulder[1] - rshoulder[1])
        s = max(0.0, 100.0 - tilt * 600)
        criteria["shoulder_alignment"] = s
        if s < 60:
            feedback.append("Keep your shoulders level throughout the shot.")
    else:
        criteria["shoulder_alignment"] = 50.0

    score = float(np.mean(list(criteria.values())))
    return criteria, feedback, score


# Map shot names to their rule functions
_SHOT_RULES = {
    "cover":      _rule_cover_drive,
    "pull":       _rule_pull_shot,
    "hook":       _rule_pull_shot,       # hook and pull share similar mechanics
    "sweep":      _rule_sweep,
    "defense":    _rule_defense,
    "flick":      _rule_generic,
    "late_cut":   _rule_generic,
    "lofted":     _rule_generic,
    "square_cut": _rule_generic,
    "straight":   _rule_generic,
}


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class ShotQualityScorer:
    """
    Computes a quality score for a cricket shot.

    Args:
        pose_weight:       Weight of pose-based score in final score (0–1).
        similarity_weight: Weight of similarity-based score in final score (0–1).
                           pose_weight + similarity_weight should equal 1.
    """

    def __init__(self, pose_weight: float = 0.6, similarity_weight: float = 0.4):
        assert abs(pose_weight + similarity_weight - 1.0) < 1e-6, \
            "pose_weight + similarity_weight must equal 1.0"
        self.pose_weight = pose_weight
        self.similarity_weight = similarity_weight

    def score(
        self,
        shot_type: str,
        avg_keypoints: dict[int, np.ndarray],
        similarity: float = -1.0,
    ) -> ShotFeedback:
        """
        Compute quality score for a shot.

        Args:
            shot_type:      Predicted shot class name (e.g. "cover").
            avg_keypoints:  Dict from pose.estimator.aggregate_keypoints().
            similarity:     Cosine similarity vs reference video (0–1), or -1 if unavailable.

        Returns:
            ShotFeedback dataclass with scores and feedback tips.
        """
        rule_fn = _SHOT_RULES.get(shot_type, _rule_generic)
        criteria, feedback, pose_score = rule_fn(avg_keypoints)

        if similarity >= 0:
            sim_score = similarity * 100.0
            final = self.pose_weight * pose_score + self.similarity_weight * sim_score
        else:
            # No reference video — use pose score only
            final = pose_score
            sim_score = -1.0

        if not feedback:
            feedback.append("Good technique! Keep it up.")

        return ShotFeedback(
            shot_type=shot_type,
            pose_score=round(pose_score, 1),
            similarity_score=round(sim_score, 1),
            final_score=round(final, 1),
            criteria={k: round(v, 1) for k, v in criteria.items()},
            feedback=feedback,
        )
