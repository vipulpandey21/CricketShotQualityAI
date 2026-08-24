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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


def movement_variance(person_history: list, n: int = 5) -> float:
    """
    Calculate position variance over last N frames to detect movement.
    
    Args:
        person_history: List of (center_x, center_y) tuples from recent frames
        n: Number of frames to look back
    
    Returns:
        Variance metric (0.0 = static, >0.03 = moving)
    """
    if len(person_history) < 2:
        return 0.0
    
    recent = person_history[-n:]
    if len(recent) < 2:
        return 0.0
    
    xs = [pos[0] for pos in recent]
    ys = [pos[1] for pos in recent]
    
    # Calculate variance in both x and y
    x_var = np.var(xs) if len(xs) > 1 else 0.0
    y_var = np.var(ys) if len(ys) > 1 else 0.0
    
    # Return combined variance
    return float(x_var + y_var)


def calculate_shoulder_width(landmarks) -> float:
    """
    Calculate shoulder width ratio to detect side-on batting stance.
    Side-on stance has narrow shoulder width in image.
    
    Returns:
        Shoulder width as ratio of image width (0.0 to 1.0)
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    
    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        return abs(left_shoulder.x - right_shoulder.x)
    return 0.5  # Default moderate width if not visible


def check_bat_pose(landmarks) -> bool:
    """
    Check if person has bat-holding pose (both wrists visible, elbow angles).
    
    Returns:
        True if pose suggests holding bat
    """
    # Check if both wrists are visible
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    
    if left_wrist.visibility < 0.4 or right_wrist.visibility < 0.4:
        return False
    
    # Check elbow angles (batting grip typically has specific angles)
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]
    
    # Simple angle check: elbows should be bent (not straight, not too bent)
    if all(lm.visibility > 0.5 for lm in [left_shoulder, left_elbow, left_wrist]):
        # Elbow angle should be between 60-130 degrees for batting
        return True
    
    return False


def run_pose_on_frames(bgr_frames: list) -> list:
    """
    Striker-only pose extraction — delegates to `striker_pose`.

    Args:
        bgr_frames: list of (H, W, 3) BGR numpy arrays — raw video frames.

    Returns:
        List of dicts, one per frame; landmark_index → (x, y, z, visibility).
        None for frames where the striker is not present.

    The previous implementation is kept below as `run_pose_on_frames_legacy`
    for reference only. It ran MediaPipe on the whole broadcast frame and
    then scored the detected people to find the batsman, which cannot work
    at this camera distance: MediaPipe does not detect the far-end striker
    at all, so the striker was never a candidate and the winning score
    always went to the non-striker, the umpire or the keeper.
    """
    from src.pose.striker_pose import run_striker_pose_on_frames
    return run_striker_pose_on_frames(bgr_frames)


def run_pose_on_frames_legacy(bgr_frames: list) -> list:
    """
    Superseded by `run_pose_on_frames`. Kept only for comparison.

    DEFINITIVE BATSMAN DETECTION with cricket-specific logic:
    - Hard reject zones (keeper, umpire, bowler)
    - Movement variance tracking (striker moves, non-striker/umpire static)
    - Cricket-specific pose analysis (bat detection, side-on stance)
    - Temporal consistency across frames
    
    Args:
        bgr_frames: list of (H, W, 3) BGR numpy arrays — raw video frames.

    Returns:
        List of dicts, one per frame.
        Each dict maps landmark_index (int) → (x, y, z, visibility) tuple.
        None is stored for frames where no person was detected.
    """
    # Get the model path (same directory as this file's parent parent)
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(script_dir, 'pose_landmarker.task')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe pose model not found at {model_path}. "
            f"Run 'python download_pose_model.py' first."
        )
    
    # Create PoseLandmarker with new API - use IMAGE mode
    # Very low thresholds to detect everyone, then we'll pick the batsman
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.2,  # Ultra low - detect everyone
        min_pose_presence_confidence=0.2,
        num_poses=5  # Detect up to 5 people so we can filter for batsman
    )
    
    results_per_frame = []

    # v2 batsman-identification (replaces the old per-frame-only zone
    # heuristic). Two concrete bugs were found by inspecting real
    # misdetections and fixed here:
    #   1. Crouching (keeper) detection required hip AND knee visibility
    #      > 0.5 on one specific side; a crouching keeper's knee is very
    #      often occluded by pads, so the check silently never fired and
    #      the keeper could out-score the batsman. Replaced with a
    #      visible-landmarks bounding-box aspect ratio (height/width),
    #      which degrades gracefully under partial occlusion.
    #   2. The absolute hard-reject y-zones (<0.18 / >0.75) assumed one
    #      fixed camera framing and discarded a legitimately-placed
    #      batsman in more zoomed-out clips. Widened to <0.12 / >0.85 and
    #      paired with an explicit bowling-action rejection (wrist raised
    #      well above the head = delivery stride, not batting) so the
    #      bowler/keeper still can't be mistaken for the batsman even
    #      with the wider zone.
    # A track-lock also strongly prefers continuing the same physical
    # person once confidently identified, rather than re-scoring blind
    # every frame.
    locked_center = None
    person_histories = {}

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame_idx, frame in enumerate(bgr_frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                results_per_frame.append(None)
                continue

            candidates = []
            for person_idx, landmarks in enumerate(result.pose_landmarks):
                xs = [lm.x for lm in landmarks if lm.visibility > 0.5]
                ys = [lm.y for lm in landmarks if lm.visibility > 0.5]
                if len(xs) < 5:
                    continue
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                # Hard reject only extreme zones (widened from 0.18/0.75)
                if center_y < 0.12 or center_y > 0.85:
                    continue

                # Bowling-action rejection: wrist well above the nose is a
                # delivery stride / follow-through, never a batting stance.
                nose = landmarks[0]
                left_wrist, right_wrist = landmarks[15], landmarks[16]
                if nose.visibility > 0.5:
                    if (left_wrist.visibility > 0.5 and left_wrist.y < nose.y - 0.05) or \
                       (right_wrist.visibility > 0.5 and right_wrist.y < nose.y - 0.05):
                        continue

                score = 0

                # 1. ZONE SCORING (soft, within the widened valid range)
                if 0.20 < center_y < 0.70 and 0.35 < center_x < 0.65:
                    score += 50
                elif 0.15 < center_y < 0.80:
                    score += 20
                else:
                    score += 5

                if 0.30 < center_x < 0.70:
                    score += 15

                # 2. STANDING VS CROUCHING (robust bbox-aspect signal)
                all_ys = [lm.y for lm in landmarks if lm.visibility > 0.3]
                all_xs = [lm.x for lm in landmarks if lm.visibility > 0.3]
                aspect = None
                if len(all_ys) >= 4:
                    height = max(all_ys) - min(all_ys)
                    width = max(all_xs) - min(all_xs) + 1e-6
                    aspect = height / width
                if aspect is not None:
                    if aspect >= 1.6:
                        score += 25  # clearly standing (batsman stance)
                    elif aspect < 1.0:
                        score -= 35  # clearly crouched (keeper signature)

                # 3. MOVEMENT VARIANCE (batsman swings; others are static)
                person_id = f"{frame_idx}_{person_idx}"
                if locked_center is not None:
                    dist_to_lock = ((center_x - locked_center[0]) ** 2 +
                                     (center_y - locked_center[1]) ** 2) ** 0.5
                    if dist_to_lock < 0.15:
                        person_id = "LOCKED"
                person_histories.setdefault(person_id, []).append((center_x, center_y))
                mv = movement_variance(person_histories[person_id], n=6)
                if len(person_histories[person_id]) >= 3:
                    if mv > 0.03:
                        score += 15
                    elif mv < 0.01:
                        score -= 20

                # 4. POSE CHECKS (bat pose / side-on stance)
                if check_bat_pose(landmarks):
                    score += 12
                shoulder_width = calculate_shoulder_width(landmarks)
                if shoulder_width < 0.15:
                    score += 8

                # 5. TRACK-LOCK BONUS — strongly prefer the same physical
                # person once confidently identified, so one clear frame
                # anchors the rest of the clip instead of re-guessing.
                if locked_center is not None:
                    dist_to_lock = ((center_x - locked_center[0]) ** 2 +
                                     (center_y - locked_center[1]) ** 2) ** 0.5
                    if dist_to_lock < 0.15:
                        score += 60

                # 6. VISIBILITY
                avg_vis = sum(lm.visibility for lm in landmarks) / len(landmarks)
                if avg_vis > 0.5:
                    score += 5

                candidates.append({
                    'landmarks': landmarks, 'score': score,
                    'center_x': center_x, 'center_y': center_y,
                })

            if not candidates:
                results_per_frame.append(None)
                continue

            best = max(candidates, key=lambda c: c['score'])
            if best['score'] >= 30:
                kp = {idx: (lm.x, lm.y, lm.z, lm.visibility) for idx, lm in enumerate(best['landmarks'])}
                results_per_frame.append(kp)
                locked_center = (best['center_x'], best['center_y'])
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

def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                   dims: int = 2) -> float:
    """
    Compute angle in degrees at joint B formed by the vectors BA and BC.

    `dims=2` measures the angle as it appears in the image; `dims=3` measures
    the real anatomical angle and is only meaningful on world landmarks —
    see `angles_from_world`.
    """
    ba = a[:dims] - b[:dims]
    bc = c[:dims] - b[:dims]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_a = np.dot(ba, bc) / denom
    return math.degrees(math.acos(float(np.clip(cos_a, -1.0, 1.0))))


def _body_scale(frame_kp: dict) -> float:
    """
    Torso size in normalised units, used to make speeds comparable between a
    wide shot and a zoomed one.
    """
    pts = [frame_kp.get(i) for i in (11, 12, 23, 24)]
    pts = [p for p in pts if p is not None and p[3] > 0.3]
    if len(pts) < 2:
        return 0.25
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    return max(max(ys) - min(ys), max(xs) - min(xs), 0.05)


def _mid_wrist(frame_kp: dict):
    """Midpoint of the two wrists — the hands on the bat handle."""
    lw, rw = frame_kp.get(15), frame_kp.get(16)
    have = [w for w in (lw, rw) if w is not None and w[3] > 0.3]
    if not have:
        return None
    return (sum(w[0] for w in have) / len(have),
            sum(w[1] for w in have) / len(have))


IMPACT_PEAK_THRESHOLD = 0.5   # a peak must reach this fraction of the global
                              # max hand speed to count as "the swing"
SHOT_START_QUIET_THRESHOLD = 0.15  # a frame at/below this fraction of the
                                   # peak speed counts as "still" (backlift
                                   # held, not yet swinging). 0.12 was tried
                                   # first and was a hair too tight: on
                                   # data/pull/video1.mp4 it excluded the
                                   # correct dip (bowler's release stride,
                                   # batsman in backlift) by floating-point
                                   # rounding and fell back to a dip several
                                   # frames earlier where the bowler was
                                   # still mid run-up. 0.15 keeps both that
                                   # clip and data/sweep/video1.mp4 correct
                                   # — see shot_start_frame_index's docstring.


def _wrist_speeds(frames_kp: list) -> list:
    """
    [(speed, frame_index), ...] — frame-to-frame movement of the wrist
    midpoint, divided by torso size so the same shot scores the same
    whether the camera is wide or zoomed. Shared by `impact_frame_index`
    and `shot_start_frame_index` so both read the same motion signal.
    """
    speeds = []
    for i in range(1, len(frames_kp)):
        a, b = frames_kp[i - 1], frames_kp[i]
        if a is None or b is None:
            continue
        wa, wb = _mid_wrist(a), _mid_wrist(b)
        if wa is None or wb is None:
            continue
        scale = (_body_scale(a) + _body_scale(b)) / 2
        d = ((wb[0] - wa[0]) ** 2 + (wb[1] - wa[1]) ** 2) ** 0.5 / scale
        speeds.append((d, i))
    return speeds


def impact_frame_index(frames_kp: list) -> int | None:
    """
    Index of the frame closest to bat-ball contact.

    Uses hand speed: the hands are travelling fastest as the bat comes
    through the ball, so a peak in frame-to-frame movement of the wrist
    midpoint marks the moment of the shot. Speed is divided by torso size so
    the same shot scores the same whether the camera is wide or zoomed.

    Takes the FIRST prominent peak, not the single fastest frame in the
    clip. Across the training set, always taking the global fastest frame put
    "impact" in the last 5 of 30 frames for 38% of clips — because after
    playing the shot the batsman stands up or sets off running, and that
    recovery motion is frequently a bigger, faster hand movement than the
    swing itself. Checked by eye on data/sweep clips: frames selected this
    way showed the batsman already upright, and the front-knee angle read
    around 160° — impossible for a shot played from a crouch. The first peak
    that clears half the clip's fastest movement is reliably the swing,
    since nothing before the shot approaches that speed and the swing is
    always the first fast motion in these clips.

    Returns None if there are too few frames with visible wrists.
    """
    speeds = _wrist_speeds(frames_kp)

    if not speeds:
        # fall back to whichever frame has the most confident joints
        best, best_n = None, -1
        for i, kp in enumerate(frames_kp):
            if kp is None:
                continue
            n = sum(1 for idx in CRICKET_LANDMARKS
                    if idx in kp and kp[idx][3] > 0.5)
            if n > best_n:
                best, best_n = i, n
        return best

    return _first_prominent_peak(speeds)


def shot_start_frame_index(frames_kp: list, impact: int | None = None) -> int | None:
    """
    Index of the frame where the shot-playing motion begins — the last
    moment of stillness (top of the backlift) before the downswing that
    ends at the impact frame.

    Mirrors `_first_prominent_peak` in the opposite direction: instead of
    scanning forward for the first frame that clears a speed threshold,
    this scans backward from the impact frame for the LAST frame at or
    below a low fraction of the clip's peak speed
    (`SHOT_START_QUIET_THRESHOLD`). That's the batsman still holding the
    backswing, the moment immediately before the wrists start
    accelerating into the shot — needed as the other end of a
    stance-to-impact angle curve (see `src/pose/shot_curve.py`), since
    "impact" alone only gives one point in time.

    Finds the local minima of wrist speed before impact (a frame slower
    than both of its detected neighbours — a genuine dip, not just a low
    reading) and returns the one NEAREST to impact that is also quiet
    enough (`SHOT_START_QUIET_THRESHOLD` of the clip's peak speed) to count
    as a real pause rather than noise inside the swing itself.

    Two simpler rules were tried and rejected by checking actual frames
    against data/sweep/video1.mp4 and data/pull/video1.mp4:
      - "last frame under a loose threshold" (0.2) caught a shallow wobble
        one frame before the swing was already visibly underway — on the
        sweep clip that put "start" at a frame where the batsman was
        already down mid-sweep, four frames from impact.
      - "global minimum speed before impact" over-corrects the other way:
        on the pull clip the deepest dip was the bowler still mid run-up
        with the batsman just standing in his guard, nowhere near the shot.
        The nearest-to-impact QUALIFYING dip instead landed on the
        bowler's release stride with the batsman in his backlift, which is
        the actual start of the shot.

    Falls back to the first frame with a measurable wrist position if no
    quiet frame exists before impact (the swing is already underway from
    frame 0 — happens on a few tightly-cropped clips where the clip starts
    mid-swing).
    """
    speeds = _wrist_speeds(frames_kp)
    if not speeds:
        return 0 if frames_kp else None

    if impact is None:
        impact = _first_prominent_peak(speeds)
    if impact is None:
        return speeds[0][1]

    return _nearest_quiet_minimum(speeds, impact)


def _nearest_quiet_minimum(speeds: list, impact: int) -> int:
    """
    Core of `shot_start_frame_index`, factored out so the offline
    professional-clip derivation (`derive_angle_curves.py`) can reuse the
    exact same rule against its own cached speed arrays instead of
    reimplementing it — the same reason `_first_prominent_peak` is a
    standalone function rather than inlined into `impact_frame_index`.

    `speeds` is [(speed, frame_index), ...]; see `shot_start_frame_index`
    for the reasoning behind "nearest qualifying local minimum".
    """
    seq = sorted(((i, d) for d, i in speeds if i <= impact), key=lambda t: t[0])
    if len(seq) < 2:
        return seq[0][0] if seq else speeds[0][1]

    peak_speed = max(d for d, _ in speeds)
    cutoff = peak_speed * SHOT_START_QUIET_THRESHOLD

    minima = []
    for k in range(len(seq)):
        left = seq[k - 1][1] if k > 0 else seq[k][1]
        right = seq[k + 1][1] if k < len(seq) - 1 else seq[k][1]
        if seq[k][1] <= left and seq[k][1] <= right:
            minima.append(seq[k])

    for frame, speed in reversed(minima):
        if speed <= cutoff:
            return frame

    # Nothing dipped low enough to count as a real pause — the quietest
    # available moment is the best evidence there is.
    if minima:
        return min(minima, key=lambda t: t[1])[0]
    return seq[0][0]


def _first_prominent_peak(speeds: list, threshold: float = IMPACT_PEAK_THRESHOLD):
    """
    Earliest frame whose speed both (a) is a local maximum — faster than the
    frames immediately either side — and (b) clears `threshold` fraction of
    the clip's single fastest frame. Falls back to the global fastest frame
    if nothing qualifies as a local max (e.g. a clip that is only 2-3 frames
    long, or one continuous acceleration with no earlier bump).
    """
    if not speeds:
        return None
    speeds = sorted(speeds, key=lambda t: t[1])   # ensure frame order
    peak_speed = max(d for d, _ in speeds)
    cutoff = peak_speed * threshold

    for j in range(len(speeds)):
        d, i = speeds[j]
        if d < cutoff:
            continue
        d_prev = speeds[j - 1][0] if j > 0 else -1.0
        d_next = speeds[j + 1][0] if j < len(speeds) - 1 else -1.0
        if d >= d_prev and d >= d_next:
            return i

    return max(speeds)[1]


def angles_at_frame(frame_kp: dict, handedness: str = "right") -> dict:
    """
    Joint angles from ONE frame.

    `compute_cricket_angles(pose_summary(...)["avg_keypoints"])` averages every
    joint over the whole clip first, which is not a pose anybody ever held: a
    hook swings the body through a wide arc, so a knee that is bent early and
    straight late averages to straight. On data/hook/video1.mp4 that produced
    front knee 178 degrees, back knee 170 degrees and a back elbow of 16
    degrees — anatomically impossible — and the quality score came out 10/100
    for a shot played by an international batsman. Angles have to be read off
    a single real frame.
    """
    if not frame_kp:
        return compute_cricket_angles({}, handedness=handedness)
    single = {idx: np.array([v[0], v[1], v[2]])
              for idx, v in frame_kp.items() if v[3] > 0.3}
    return compute_cricket_angles(single, handedness=handedness)


def angles_from_world(world: dict, handedness: str = "right") -> dict:
    """
    Joint angles from MediaPipe world landmarks — metric 3D in metres,
    centred on the hips.

    These are the only coordinates these angles should be read from. The
    image landmarks are a 2D projection, and on the behind-the-bowler camera
    the batsman's legs point roughly along the view axis, so hip, knee and
    ankle project almost collinear and every shot measures a near-straight
    knee. Across the dataset the 2D front-knee angle sat between 155 and 177
    degrees for sweep, hook, pull and defense alike — no information at all.
    The image landmarks' z is only a relative depth hint, and using it as if
    it were metric is worse still: it puts the sweep front knee at 11
    degrees, which no knee can do.

    From world landmarks the same four clips separate the way they should:

        sweep    front knee 127   (crouched)
        defense  front knee  51   (deep forward defensive)
        cover    front knee 152
        hook     front knee 168   (played standing tall)
    """
    if not world:
        return compute_cricket_angles({})
    return compute_cricket_angles(
        {idx: np.array(xyz, dtype=float) for idx, xyz in world.items()},
        dims=3, handedness=handedness)


def phase_angles(frames_kp: list, worlds: list | None = None) -> dict:
    """
    Angles at the three phases of the shot, plus which frames they came from.

    `worlds` is the parallel list of MediaPipe world landmarks from
    `striker_pose`. When present the angles are measured from those, which is
    the only way to get a real knee angle off this camera — see
    `angles_from_world`. Without it the function still works, but the returned
    angles are image-plane projections and knees will read near-straight
    whatever the shot.

    Returns {"stance": {...}, "impact": {...}, "follow_through": {...},
             "frames": {...}, "source": "world" | "image"}
    """
    valid = [i for i, kp in enumerate(frames_kp) if kp is not None]
    if not valid:
        empty = compute_cricket_angles({})
        return {"stance": empty, "impact": empty, "follow_through": empty,
                "frames": {"stance": None, "impact": None,
                           "follow_through": None},
                "source": "none",
                "handedness": {"hand": "right", "confidence": 0.0,
                               "assumed": True}}

    impact = impact_frame_index(frames_kp)
    if impact is None or frames_kp[impact] is None:
        impact = valid[len(valid) // 2]

    stance = valid[0]
    after = [i for i in valid if i > impact]
    follow = after[min(len(after) - 1, max(0, len(after) // 2))] if after else impact

    # Detected once for the clip and reused, so every phase is measured
    # against the same front/back leg.
    hand_info = detect_handedness(frames_kp, worlds)
    hand = hand_info["hand"]

    def at(i):
        if worlds is not None and i < len(worlds) and worlds[i]:
            return angles_from_world(worlds[i], hand)
        return angles_at_frame(frames_kp[i], hand)

    have_world = worlds is not None and any(
        worlds[i] for i in (stance, impact, follow) if i < len(worlds))

    return {
        "stance": at(stance),
        "impact": at(impact),
        "follow_through": at(follow),
        "frames": {"stance": stance, "impact": impact,
                   "follow_through": follow},
        "source": "world" if have_world else "image",
        "handedness": hand_info,
    }


def detect_handedness(frames_kp: list, worlds: list | None = None) -> dict:
    """
    Is this a right- or left-handed batsman?

    Everything downstream depends on this. "Front leg" is the left leg for a
    right-hander and the RIGHT leg for a left-hander, so getting it wrong
    grades every criterion against the wrong limb — a left-hander's front-knee
    bend is scored using his back knee. It was previously hard-coded to
    right-handed, and it matters more here than the usual ~20% of players
    would suggest: this dataset contains horizontally mirrored clips (their
    scoreboards read backwards), which turn right-handers into left-handed
    presentations.

    Two independent cricket facts are used, and they vote:

      1. **Top hand.** A right-hander's top hand on the bat handle is the
         left; a left-hander's is the right. So the top hand is whichever
         wrist sits higher up the image.
      2. **Which side faces the bowler.** A right-hander stands with his left
         shoulder towards the bowler, and the camera is behind the bowler, so
         his left shoulder is nearer the camera. In MediaPipe world landmarks
         a smaller z is nearer the camera.

    Voting across every frame of the clip, rather than trusting one frame,
    keeps a moment of follow-through from flipping the answer.

    Returns {"hand": "right"|"left", "confidence": 0-1, "votes": {...}}
    """
    top_hand_r = top_hand_l = 0
    depth_r = depth_l = 0

    for i, kp in enumerate(frames_kp or []):
        if kp is None:
            continue
        lw, rw = kp.get(15), kp.get(16)
        if lw is not None and rw is not None and lw[3] > 0.4 and rw[3] > 0.4:
            # smaller y is higher up the image
            if lw[1] < rw[1]:
                top_hand_r += 1      # left hand on top -> right-hander
            elif rw[1] < lw[1]:
                top_hand_l += 1

        w = worlds[i] if worlds and i < len(worlds) and worlds[i] else None
        if w and 11 in w and 12 in w:
            if w[11][2] < w[12][2]:
                depth_r += 1         # left shoulder nearer camera -> RHB
            elif w[12][2] < w[11][2]:
                depth_l += 1

    right = top_hand_r + depth_r
    left = top_hand_l + depth_l
    total = right + left
    if total == 0:
        return {"hand": "right", "confidence": 0.0,
                "votes": {"right": 0, "left": 0}, "assumed": True}

    hand = "right" if right >= left else "left"
    return {
        "hand": hand,
        "confidence": round(max(right, left) / total, 2),
        "votes": {"right": right, "left": left,
                  "top_hand": [top_hand_r, top_hand_l],
                  "shoulder_depth": [depth_r, depth_l]},
        "assumed": False,
    }


def compute_cricket_angles(avg_kp: dict, dims: int = 2,
                           handedness: str = "right") -> dict:
    """
    Compute biomechanically meaningful angles from keypoints.

    Returns a dict:
      front_knee_angle    — hip→knee→ankle of the FRONT leg
      back_knee_angle     — hip→knee→ankle of the BACK leg
      front_elbow_angle   — shoulder→elbow→wrist of the top (front) arm
      back_elbow_angle    — shoulder→elbow→wrist of the bottom (back) arm
      shoulder_tilt_deg   — how many degrees shoulders are tilted (0 = level)
      hip_tilt_deg        — how many degrees hips are tilted
      trunk_lean_deg      — angle of torso from vertical
    Any value is None if required joints were not detected.

    `handedness` decides which physical side is "front": left for a
    right-handed batsman, right for a left-hander. See `detect_handedness`.
    """
    def get(idx):
        return avg_kp.get(idx)

    result = {}
    result["handedness"] = handedness

    # Front side is the left for a RHB, the right for a LHB.
    if handedness == "left":
        f_hip, f_knee, f_ankle = 24, 26, 28
        b_hip, b_knee, b_ankle = 23, 25, 27
        f_sh, f_el, f_wr = 12, 14, 16
        b_sh, b_el, b_wr = 11, 13, 15
    else:
        f_hip, f_knee, f_ankle = 23, 25, 27
        b_hip, b_knee, b_ankle = 24, 26, 28
        f_sh, f_el, f_wr = 11, 13, 15
        b_sh, b_el, b_wr = 12, 14, 16

    # Front knee angle
    fh, fk, fa = get(f_hip), get(f_knee), get(f_ankle)
    if all(v is not None for v in [fh, fk, fa]):
        result["front_knee_angle"] = round(angle_at_joint(fh, fk, fa, dims), 1)
    else:
        result["front_knee_angle"] = None

    # Back knee angle
    bh, bk, ba = get(b_hip), get(b_knee), get(b_ankle)
    if all(v is not None for v in [bh, bk, ba]):
        result["back_knee_angle"] = round(angle_at_joint(bh, bk, ba, dims), 1)
    else:
        result["back_knee_angle"] = None

    # Front elbow angle
    fs, fe, fw = get(f_sh), get(f_el), get(f_wr)
    if all(v is not None for v in [fs, fe, fw]):
        result["front_elbow_angle"] = round(angle_at_joint(fs, fe, fw, dims), 1)
    else:
        result["front_elbow_angle"] = None

    # Back elbow angle
    bs, be, bw = get(b_sh), get(b_el), get(b_wr)
    if all(v is not None for v in [bs, be, bw]):
        result["back_elbow_angle"] = round(angle_at_joint(bs, be, bw, dims), 1)
    else:
        result["back_elbow_angle"] = None

    # Tilt and lean are side-independent — they compare left against right —
    # so they need the anatomical landmarks, not the front/back mapping.
    lhip, rhip = get(23), get(24)
    lshoulder, rshoulder = get(11), get(12)

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
