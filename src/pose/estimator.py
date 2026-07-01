"""
estimator.py
MediaPipe Pose wrapper.

Extracts 33 body keypoints per frame from a list of BGR video frames.

Landmark indices used for cricket (MediaPipe numbering):
  0  = nose
  11 = left_shoulder   12 = right_shoulder
  13 = left_elbow      14 = right_elbow
  15 = left_wrist      16 = right_wrist
  23 = left_hip        24 = right_hip
  25 = left_knee       26 = right_knee
  27 = left_ankle      28 = right_ankle
"""

import cv2
import math
import numpy as np
import mediapipe as mp

# ── Landmark index → human-readable name ──────────────────────────────────────
CRICKET_LANDMARKS = {
    0:  "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


def run_pose_on_frames(bgr_frames: list) -> list:
    """
    Run MediaPipe Pose on a list of BGR numpy frames.

    Args:
        bgr_frames: list of (H, W, 3) BGR numpy arrays — raw video frames.

    Returns:
        List of dicts, one per frame.
        Each dict maps landmark_index (int) → (x, y, z, visibility) tuple.
        None is stored for frames where no person was detected.
    """
    mp_pose = mp.solutions.pose
    results_per_frame = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for frame in bgr_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                kp = {
                    idx: (lm.x, lm.y, lm.z, lm.visibility)
                    for idx, lm in enumerate(result.pose_landmarks.landmark)
                }
                results_per_frame.append(kp)
            else:
                results_per_frame.append(None)

    return results_per_frame


def draw_skeleton(bgr_frame: np.ndarray, keypoints: dict) -> np.ndarray:
    """
    Draw only the 13 cricket-relevant body joints and their connections.
    Skips all face landmarks (eyes, ears, mouth etc.)
    """
    if keypoints is None:
        return bgr_frame.copy()

    annotated = bgr_frame.copy()
    h, w = annotated.shape[:2]

    # Only draw connections between cricket-relevant joints
    CRICKET_CONNECTIONS = [
        (11, 12),  # left shoulder — right shoulder
        (11, 13),  # left shoulder — left elbow
        (13, 15),  # left elbow — left wrist
        (12, 14),  # right shoulder — right elbow
        (14, 16),  # right elbow — right wrist
        (11, 23),  # left shoulder — left hip
        (12, 24),  # right shoulder — right hip
        (23, 24),  # left hip — right hip
        (23, 25),  # left hip — left knee
        (25, 27),  # left knee — left ankle
        (24, 26),  # right hip — right knee
        (26, 28),  # right knee — right ankle
    ]

    # Colour scheme: upper body = orange, lower body = cyan, connections = white
    UPPER_BODY = {11, 12, 13, 14, 15, 16}
    LOWER_BODY = {23, 24, 25, 26, 27, 28}

    # Draw connections first (so dots appear on top)
    for (i, j) in CRICKET_CONNECTIONS:
        if i in keypoints and j in keypoints:
            xi, yi, _, vi = keypoints[i]
            xj, yj, _, vj = keypoints[j]
            if vi > 0.4 and vj > 0.4:
                pt1 = (int(xi * w), int(yi * h))
                pt2 = (int(xj * w), int(yj * h))
                cv2.line(annotated, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw joint dots
    for idx in CRICKET_LANDMARKS:
        if idx not in keypoints:
            continue
        x, y, _, vis = keypoints[idx]
        if vis < 0.4:
            continue
        px, py = int(x * w), int(y * h)
        color = (0, 165, 255) if idx in UPPER_BODY else (255, 200, 0)  # orange / cyan
        cv2.circle(annotated, (px, py), 6, color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)  # white border

    # Draw nose as a small dot to mark head position
    if 0 in keypoints:
        x, y, _, vis = keypoints[0]
        if vis > 0.4:
            cv2.circle(annotated, (int(x * w), int(y * h)), 4, (200, 200, 200), -1)

    return annotated




def aggregate_keypoints(frames_kp: list, vis_threshold: float = 0.3) -> dict:
    """
    Average keypoint positions across all valid frames.
    Uses vis_threshold=0.3 (permissive) so partially occluded joints
    like wrists and elbows are still included when detected.
    """
    accum: dict = {}
    for frame_kp in frames_kp:
        if frame_kp is None:
            continue
        for idx, (x, y, z, vis) in frame_kp.items():
            if vis > vis_threshold:
                accum.setdefault(idx, []).append([x, y, z])

    return {idx: np.mean(vals, axis=0) for idx, vals in accum.items()}


def pose_summary(frames_kp: list) -> dict:
    """
    Return summary statistics about pose detection quality.

    Returns dict with:
      - detected_frames: int   how many frames had a pose
      - total_frames: int
      - detection_rate: float  0.0 to 1.0
      - avg_keypoints: dict    from aggregate_keypoints()
      - cricket_joints: dict   only the 13 joints relevant for cricket,
                               mapped by name → (x, y) in 0-1 image coords
    """
    total = len(frames_kp)
    detected = sum(1 for f in frames_kp if f is not None)
    avg_kp = aggregate_keypoints(frames_kp)

    cricket_joints = {}
    for idx, name in CRICKET_LANDMARKS.items():
        if idx in avg_kp:
            x, y, z = avg_kp[idx]
            cricket_joints[name] = (round(float(x), 4), round(float(y), 4))

    return {
        "detected_frames": detected,
        "total_frames": total,
        "detection_rate": detected / total if total > 0 else 0.0,
        "avg_keypoints": avg_kp,
        "cricket_joints": cricket_joints,
    }


# ── Angle & Cricket Metrics ───────────────────────────────────────────────────

def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute angle in degrees at joint B formed by the vectors BA and BC.
    a, b, c are (x, y) or (x, y, z) arrays — normalised image coordinates.
    """
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_a = np.dot(ba, bc) / denom
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def compute_cricket_angles(avg_kp: dict) -> dict:
    """
    Compute biomechanically meaningful angles from averaged keypoints.

    Returns a dict:
      front_knee_angle    — hip→knee→ankle (left side, front foot for RHB)
      back_knee_angle     — right hip→knee→ankle (back foot for RHB)
      front_elbow_angle   — left shoulder→elbow→wrist
      back_elbow_angle    — right shoulder→elbow→wrist
      shoulder_tilt_deg   — how many degrees shoulders are tilted (0 = level)
      hip_tilt_deg        — how many degrees hips are tilted
      trunk_lean_deg      — angle of torso from vertical
    Any value is None if required joints were not detected.
    """
    def get(idx):
        return avg_kp.get(idx)

    result = {}

    # Front (left) knee angle — landmark 23 (L-hip), 25 (L-knee), 27 (L-ankle)
    lhip, lknee, lankle = get(23), get(25), get(27)
    if all(v is not None for v in [lhip, lknee, lankle]):
        result["front_knee_angle"] = round(angle_at_joint(lhip, lknee, lankle), 1)
    else:
        result["front_knee_angle"] = None

    # Back (right) knee angle — 24, 26, 28
    rhip, rknee, rankle = get(24), get(26), get(28)
    if all(v is not None for v in [rhip, rknee, rankle]):
        result["back_knee_angle"] = round(angle_at_joint(rhip, rknee, rankle), 1)
    else:
        result["back_knee_angle"] = None

    # Front (left) elbow angle — 11, 13, 15
    lshoulder, lelbow, lwrist = get(11), get(13), get(15)
    if all(v is not None for v in [lshoulder, lelbow, lwrist]):
        result["front_elbow_angle"] = round(angle_at_joint(lshoulder, lelbow, lwrist), 1)
    else:
        result["front_elbow_angle"] = None

    # Back (right) elbow angle — 12, 14, 16
    rshoulder, relbow, rwrist = get(12), get(14), get(16)
    if all(v is not None for v in [rshoulder, relbow, rwrist]):
        result["back_elbow_angle"] = round(angle_at_joint(rshoulder, relbow, rwrist), 1)
    else:
        result["back_elbow_angle"] = None

    # Shoulder tilt — difference in y between left and right shoulder
    if lshoulder is not None and rshoulder is not None:
        tilt_rad = math.atan2(abs(float(lshoulder[1]) - float(rshoulder[1])),
                              abs(float(lshoulder[0]) - float(rshoulder[0])) + 1e-8)
        result["shoulder_tilt_deg"] = round(math.degrees(tilt_rad), 1)
    else:
        result["shoulder_tilt_deg"] = None

    # Hip tilt
    if lhip is not None and rhip is not None:
        tilt_rad = math.atan2(abs(float(lhip[1]) - float(rhip[1])),
                              abs(float(lhip[0]) - float(rhip[0])) + 1e-8)
        result["hip_tilt_deg"] = round(math.degrees(tilt_rad), 1)
    else:
        result["hip_tilt_deg"] = None

    # Trunk lean — angle of torso from vertical
    # In image coords: Y increases downward. Trunk vector = shoulder_mid - hip_mid
    # points upward (negative Y). We measure angle from true vertical [0,-1].
    # A perfectly upright trunk = 0°. Leaning forward = positive angle.
    if all(v is not None for v in [lshoulder, rshoulder, lhip, rhip]):
        sh_mid = (np.array(lshoulder[:2]) + np.array(rshoulder[:2])) / 2.0
        hp_mid = (np.array(lhip[:2])     + np.array(rhip[:2]))     / 2.0
        trunk_vec = sh_mid - hp_mid   # points from hips toward shoulders

        trunk_len = np.linalg.norm(trunk_vec) + 1e-8
        trunk_unit = trunk_vec / trunk_len

        # Vertical "up" in image coords is (0, -1)
        vertical_up = np.array([0.0, -1.0])
        cos_a = float(np.clip(np.dot(trunk_unit, vertical_up), -1.0, 1.0))
        result["trunk_lean_deg"] = round(math.degrees(math.acos(cos_a)), 1)
    else:
        result["trunk_lean_deg"] = None

    return result
