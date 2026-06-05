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
    Draw MediaPipe pose skeleton on a single BGR frame.

    Args:
        bgr_frame:  Original BGR numpy frame.
        keypoints:  Dict from run_pose_on_frames (or None → returns frame unchanged).

    Returns:
        Annotated BGR frame (copy of original).
    """
    if keypoints is None:
        return bgr_frame.copy()

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    # Reconstruct a NormalizedLandmarkList so mp_drawing can use it
    from mediapipe.framework.formats import landmark_pb2
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for i in range(33):
        lm = landmark_list.landmark.add()
        if i in keypoints:
            lm.x, lm.y, lm.z, lm.visibility = keypoints[i]
        # else leave as 0,0,0,0

    annotated = bgr_frame.copy()
    mp_drawing.draw_landmarks(
        annotated,
        landmark_list,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )
    return annotated


def aggregate_keypoints(frames_kp: list) -> dict:
    """
    Average keypoint positions across all valid frames (visibility > 0.5).

    Args:
        frames_kp: Output of run_pose_on_frames.

    Returns:
        Dict mapping landmark_index → mean (x, y, z) as numpy array.
        Empty dict if no valid frames at all.
    """
    accum: dict = {}
    for frame_kp in frames_kp:
        if frame_kp is None:
            continue
        for idx, (x, y, z, vis) in frame_kp.items():
            if vis > 0.5:
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
