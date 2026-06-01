"""
estimator.py
MediaPipe Pose wrapper — extracts 33 body keypoints per frame from a video.
"""

from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# MediaPipe landmark indices we care about for cricket biomechanics
KEYPOINT_NAMES = {
    0: "nose",
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


class PoseEstimator:
    """
    Wraps MediaPipe Pose to extract keypoints from video frames.

    Usage:
        estimator = PoseEstimator()
        keypoints_per_frame = estimator.process_frames(bgr_frames)
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. Run: pip install mediapipe==0.10.14"
            )
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self._detection_conf = min_detection_confidence
        self._tracking_conf = min_tracking_confidence

    def process_frames(self, bgr_frames: list) -> list[dict | None]:
        """
        Run pose estimation on a list of BGR frames.

        Args:
            bgr_frames: List of BGR numpy arrays (raw video frames).

        Returns:
            List of dicts (one per frame). Each dict maps landmark_index → (x, y, z, visibility).
            None is stored for frames where no pose was detected.
        """
        results = []
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self._detection_conf,
            min_tracking_confidence=self._tracking_conf,
        ) as pose:
            for frame in bgr_frames:
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks:
                    kp = {}
                    for idx, lm in enumerate(result.pose_landmarks.landmark):
                        kp[idx] = (lm.x, lm.y, lm.z, lm.visibility)
                    results.append(kp)
                else:
                    results.append(None)
        return results

    def draw_landmarks(self, bgr_frame: np.ndarray, keypoints: dict | None) -> np.ndarray:
        """
        Draw pose landmarks on a frame for visualisation.

        Args:
            bgr_frame: Original BGR frame.
            keypoints: Dict from process_frames output (or None).

        Returns:
            Annotated BGR frame.
        """
        if keypoints is None or not _MP_AVAILABLE:
            return bgr_frame

        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # Reconstruct a NormalizedLandmarkList for drawing
        landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for i in range(33):
            lm = landmark_list.landmark.add()
            if i in keypoints:
                lm.x, lm.y, lm.z, lm.visibility = keypoints[i]
            else:
                lm.x = lm.y = lm.z = 0.0
                lm.visibility = 0.0

        annotated = bgr_frame.copy()
        mp_drawing.draw_landmarks(
            annotated,
            landmark_list,
            mp_pose.POSE_CONNECTIONS,
        )
        return annotated


def aggregate_keypoints(keypoints_per_frame: list[dict | None]) -> dict[int, np.ndarray]:
    """
    Average keypoint positions across all valid frames.

    Returns:
        Dict mapping landmark_index → mean (x, y, z) array.
    """
    accum: dict[int, list] = {}
    for frame_kp in keypoints_per_frame:
        if frame_kp is None:
            continue
        for idx, (x, y, z, vis) in frame_kp.items():
            if vis > 0.5:  # only use high-visibility keypoints
                accum.setdefault(idx, []).append([x, y, z])

    return {idx: np.mean(vals, axis=0) for idx, vals in accum.items()}
