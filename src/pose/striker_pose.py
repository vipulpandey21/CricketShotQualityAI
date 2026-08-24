"""
striker_pose.py
Striker-only pose extraction.

Why this exists
---------------
`estimator.py` ran MediaPipe Pose directly on the full 1280x720 broadcast
frame and then tried to pick the batsman out of the detected people with a
per-frame heuristic score. Two things made that unfixable by tuning:

  1. MediaPipe's pose detector does not reliably find the striker at all.
     The broadcast camera sits behind the bowler's arm, so the striker is
     only ~120px tall in a 720p frame. MediaPipe locks onto the big,
     close-to-camera people instead — the non-striker and the umpire.
     The striker never even entered the candidate list, so no amount of
     score tuning could select them.

  2. The decision was made per frame, so the choice flickered between
     people across a clip.

This module fixes both:

  * A dedicated person detector (YOLOv8 + BoT-SORT) finds *every* person
    including the small far-end striker, and gives stable track ids.
  * The striker is chosen once per clip at the *track* level, not per
    frame, so the choice cannot flicker.
  * Pose is then run on an upscaled crop around the striker only, which
    is what makes the 33 landmarks accurate on a small subject.

Striker identification rule (derived from measured track statistics, not
guessed — see the table in `select_striker_track`):

    Among tracks that are big enough and upright enough to be a standing
    player near the pitch, the striker is the *topmost* one in the frame.
    Everyone below them is the bowler, the umpire or the non-striker.
    The only person above/beside the striker is the wicketkeeper, who is
    separated by being markedly shorter (crouched behind the stumps and
    further from camera).
"""

from __future__ import annotations

import os

# Ultralytics will silently pip-install whatever it thinks is missing the
# first time it runs. That is how opencv-python 5.x once landed in this
# venv and broke numpy/tensorflow/mediapipe together. Pin it shut before
# ultralytics is ever imported.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from dataclasses import dataclass, field  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ── Tunables ─────────────────────────────────────────────────────────────────
# Every threshold below was read off measured per-track statistics across
# clips (cover / sweep / defense), not picked by feel.

YOLO_MODEL = "yolov8s.pt"
# Verified against yolov8m on all ten data/*/video1.mp4 clips: the same
# striker is selected in 10/10, with box centre and height agreeing to three
# decimals, at 14.5 s per clip instead of 33.6 s. yolov8x at imgsz 1280 was
# 3.9 s/frame — six times slower again, for no improvement in selection.
YOLO_IMGSZ = 960
YOLO_CONF = 0.15          # low: the far-end striker is small and low-contrast
MAX_DETECTIONS = 60       # frames YOLO runs on, evenly spaced across the clip.
                          # Bounds cost: at ~0.6 s/frame an unbounded pass over
                          # a 200-frame upload took >2 minutes on its own.

# Track admission. These are deliberately *relative* wherever possible: a
# fixed pixel/fraction threshold only works for one camera zoom level, and
# this has to hold on any broadcast clip, not just the ones it was built on.
MIN_TRACK_FRAMES = 8       # absolute; a real player persists, a glitch does not
                           # (callers that detect on only every Nth frame must
                           #  scale this down — see select_striker_track)
MIN_HEIGHT_RATIO = 0.45    # vs the tallest track in the clip — scale invariant
MIN_BOX_HEIGHT = 0.06      # absolute floor: below this it is crowd, not a player
MIN_ASPECT = 1.45          # h/w in PIXELS; crouched keeper ~1.1-1.3, striker ~1.8-2.9
CLIPPED_MIN_ASPECT = 0.85  # relaxed floor for edge-clipped boxes — see below
MAX_CX_OFFSET = 0.30       # striker stands near the pitch centre line
EDGE_TOL = 0.01            # a box within this of a frame edge counts as clipped

# Keeper test — see `_is_keeper_of`. Both conditions must hold.
KEEPER_HEIGHT_RATIO = 0.80   # keeper is clearly shorter (further from camera)
KEEPER_GAP_RATIO = 0.90      # ...and at the same end of the pitch

CROP_MARGIN = 0.35         # expand striker box by this fraction before pose
CROP_SIZE = 512            # upscale target — this is what makes small subjects work

MAX_LANDMARK_GAP = 3       # frames; longer dropouts are real occlusions, left empty
INTERPOLATED_VIS = 0.45    # just above draw_skeleton's 0.4 threshold

POSES_PER_CROP = 3         # the crop often catches a neighbour; see _pose_nearest_box
MAX_POSE_OFFSET = 0.50     # torso must land within half a box-height of the striker
BOX_TOLERANCE = 0.15       # ...and inside the box itself, allowing this margin

# Landmarks this project cares about (same set as estimator.CRICKET_LANDMARKS)
CRICKET_LANDMARKS = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow",    14: "right_elbow",
    15: "left_wrist",    16: "right_wrist",
    23: "left_hip",      24: "right_hip",
    25: "left_knee",     26: "right_knee",
    27: "left_ankle",    28: "right_ankle",
}


@dataclass
class TrackStats:
    """Clip-level summary of one tracked person."""
    tid: int
    coverage: float
    cy: float
    cx: float
    height: float
    aspect: float
    conf: float
    n_dets: int
    clipped: bool = False      # box runs off a frame edge, so its shape lies
    rejected: str | None = None
    boxes: dict = field(default_factory=dict)  # frame_idx -> (x1, y1, x2, y2) normalised


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_person_tracks(bgr_frames: list, max_detections: int = MAX_DETECTIONS
                         ) -> dict:
    """
    Run YOLO person detection + BoT-SORT tracking over the clip.

    Returns: {track_id: {frame_idx: (x1, y1, x2, y2)}} in normalised coords.

    Detection is capped at `max_detections` frames, evenly spaced. YOLO costs
    ~0.6 s per frame on CPU, so running it on every frame made cost scale with
    clip length without bound — an 8-second upload is ~200 frames and spent
    over two minutes here alone. The cap keeps it near-constant. Nothing is
    lost: the striker moves smoothly, so `_smooth_and_fill` interpolates the
    box for frames that were skipped, and pose still runs on every analysed
    frame.
    """
    from ultralytics import YOLO

    weights = os.path.join(_project_root(), YOLO_MODEL)
    if not os.path.exists(weights):
        weights = YOLO_MODEL  # let ultralytics fetch it
    model = YOLO(weights)

    n = len(bgr_frames)
    if n <= max_detections:
        indices = list(range(n))
    else:
        indices = sorted(set(
            np.linspace(0, n - 1, max_detections).round().astype(int).tolist()))

    tracks: dict = {}
    for idx in indices:
        frame = bgr_frames[idx]
        h, w = frame.shape[:2]
        res = model.track(
            frame, classes=[0], conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
            persist=True, tracker="botsort.yaml", verbose=False,
        )[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            if b.id is None:
                continue
            tid = int(b.id[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            tracks.setdefault(tid, {})[idx] = (x1 / w, y1 / h, x2 / w, y2 / h)
    return tracks


def summarise_tracks(tracks: dict, n_frames: int,
                     frame_shape: tuple | None = None) -> list:
    """
    Reduce each track to the clip-level statistics used for selection.

    `frame_shape` is (height, width) in pixels and should always be passed.
    Aspect ratio has to be measured in PIXELS: the boxes are stored in
    normalised coordinates, so on a frame that is not 16:9 a normalised h/w is
    not the person's real shape. Centre-cropping a 1280x720 clip to a 9:16
    phone short shrinks the width to 405px, which inflates every normalised
    width by 3.16x and collapses the apparent aspect ratio by the same factor
    — every upright player then failed MIN_ASPECT and the striker was either
    missed entirely or the bowler was chosen instead.
    """
    px_h, px_w = frame_shape if frame_shape else (1.0, 1.0)

    stats = []
    for tid, boxes in tracks.items():
        if not boxes:
            continue
        cys = [(b[1] + b[3]) / 2 for b in boxes.values()]
        cxs = [(b[0] + b[2]) / 2 for b in boxes.values()]
        hs = [b[3] - b[1] for b in boxes.values()]
        ws = [b[2] - b[0] for b in boxes.values()]
        med_h, med_w = float(np.median(hs)), float(np.median(ws))

        # Does this person run off an edge of the frame? If so their box is a
        # crop of them, not their outline, so height and aspect understate the
        # real figure and must not be used to reject them. This is what a
        # zoomed-in clip does to the batsman: at 1.6x the striker's head goes
        # above the top edge, his measured aspect drops below MIN_ASPECT, he is
        # thrown out, and the bowler wins by default.
        n_clipped = sum(
            1 for b in boxes.values()
            if b[0] <= EDGE_TOL or b[1] <= EDGE_TOL
            or b[2] >= 1 - EDGE_TOL or b[3] >= 1 - EDGE_TOL)

        stats.append(TrackStats(
            tid=tid,
            coverage=len(boxes) / max(n_frames, 1),
            cy=float(np.median(cys)),
            cx=float(np.median(cxs)),
            height=med_h,
            aspect=float((med_h * px_h) / max(med_w * px_w, 1e-6)),
            conf=1.0,
            n_dets=len(boxes),
            clipped=n_clipped > len(boxes) / 2,
            boxes=boxes,
        ))
    return stats


def select_striker_track(stats: list, min_frames: int = MIN_TRACK_FRAMES
                         ) -> TrackStats | None:
    """
    Pick the one track that is the striker, for the whole clip.

    Measured track statistics that this rule is built on:

        clip     person        cy     h      aspect
        cover1   STRIKER      0.20   0.26     2.0
        cover1   bowler       0.65   0.32     3.1
        cover1   umpire       0.73   0.33     2.7
        cover1   non-striker  0.60   0.34     2.6
        sweep1   STRIKER      0.25   0.27     2.8
        sweep1   keeper       0.26   0.14     1.1
        sweep1   non-striker  0.55   0.34     3.2
        defense1 STRIKER      0.19   0.24     1.9
        defense1 keeper       0.16   0.16     1.3

    Note the striker is always the highest *qualifying* person in frame,
    and the keeper — the only one who can sit above them — is always
    roughly half their height and much less upright.
    """
    if not stats:
        return None

    # "Tallest" must come from figures whose full outline is visible. Including
    # an edge-clipped giant (a zoomed clip's bowler, half out of frame) would
    # raise the bar and eliminate legitimate players.
    unclipped = [s.height for s in stats if not s.clipped]
    tallest = max(unclipped) if unclipped else max(s.height for s in stats)

    candidates = []
    for s in stats:
        if s.n_dets < min_frames:
            s.rejected = f"only {s.n_dets} frames < {min_frames}"
        elif s.height < MIN_BOX_HEIGHT:
            s.rejected = f"height {s.height:.2f} < {MIN_BOX_HEIGHT} (crowd)"
        elif s.height < tallest * MIN_HEIGHT_RATIO:
            # Applies to clipped tracks too. This is the guard against distant
            # fielders, and exempting clipped boxes from it let a fielder at the
            # very top of the frame — touching the edge, 0.08 of frame height —
            # be picked as the striker.
            s.rejected = (f"height {s.height:.2f} < {MIN_HEIGHT_RATIO:.2f}x tallest "
                          f"{tallest:.2f} (distant fielder)")
        elif s.aspect < MIN_ASPECT and not s.clipped:
            # The SHAPE test is relaxed, not fully waived, for clipped boxes:
            # a figure cut off by a frame edge has a truncated outline, so its
            # normal 1.45 aspect floor (which rejects a crouched keeper) does
            # not apply — a striker missing his legs to a tight zoom can
            # measure closer to square. But a person, even truncated, is never
            # WIDER than tall (that needs aspect < 1, which no partial human
            # silhouette produces). Waiving the check completely let a junk
            # detection with aspect 0.55 (wider than tall — not a person at
            # all) through as a "candidate" on data/square_cut/video5.mp4,
            # where it was small and near the top of frame and so out-ranked
            # the real, correctly-shaped batsman for the topmost-wins rule.
            s.rejected = f"aspect {s.aspect:.2f} < {MIN_ASPECT} (crouched — keeper)"
        elif s.aspect < CLIPPED_MIN_ASPECT and s.clipped:
            s.rejected = (f"aspect {s.aspect:.2f} < {CLIPPED_MIN_ASPECT} even "
                          f"clipped (not person-shaped)")
        elif abs(s.cx - 0.5) > MAX_CX_OFFSET:
            s.rejected = f"cx {s.cx:.2f} too far from pitch centre"
        else:
            candidates.append(s)

    if not candidates:
        return None

    # Striker = topmost qualifying person, after walking past any keeper.
    candidates.sort(key=lambda s: s.cy)
    best = candidates[0]
    for other in candidates[1:]:
        if _is_keeper_of(best, other):
            best = other      # `best` was the keeper; the striker is below them
        else:
            break
    return best


def _is_keeper_of(upper: TrackStats, lower: TrackStats) -> bool:
    """
    True if `upper` is the wicketkeeper standing behind `lower` (the striker).

    Both cricket invariants must hold:

      1. The keeper is clearly shorter on camera. They stand a few metres
         further from the camera than the striker and are crouched.

      2. They are at the *same end of the pitch* as the striker. Measured as
         the vertical gap between the two, divided by the striker's own
         on-screen height — a scale-free quantity, so it means the same
         thing at any zoom level. The non-striker is a full pitch length
         away and always lands above this ratio:

             defense   keeper  vs striker   0.19
             late_cut  keeper  vs striker   0.78
             cover     striker vs non-str   1.05
             sweep     striker vs non-str   1.11

    An earlier version tested horizontal overlap instead, on the assumption
    that the keeper stands directly behind the stumps. That is not reliable
    — in data/defense/video1.mp4 the keeper stands wide to the leg side, the
    overlap measured 0.14, and the keeper was picked as the striker for the
    whole clip.

    Heights are compared over the frames both tracks actually share, not
    each track's clip-wide median. On data/straight/video2.mp4 the camera
    zooms in partway through; the striker's track spans the whole clip
    (frames 0-49) so his median height is pulled down by the earlier, wider
    frames, while the keeper's track only starts at the zoom (frame 24) and
    is measured entirely at the larger, zoomed-in scale. Their whole-clip
    medians made the keeper measure taller than the striker and this test
    walked past the real striker onto the keeper. Two tracks' sizes are only
    comparable when measured at the same zoom level, i.e. the same frames.
    """
    # The whole test rests on comparing heights, so it must not run when the
    # upper figure's height is a measurement artefact. A zoomed clip cuts the
    # striker off at the top edge; his truncated box then looked "too short to
    # be the striker", the guard walked past him, and on data/sweep/video1.mp4
    # at 1.6x zoom it walked all the way down to the bowler.
    if upper.clipped:
        return False

    uh, lh, ucy, lcy = upper.height, lower.height, upper.cy, lower.cy
    common = set(upper.boxes) & set(lower.boxes)
    if len(common) >= 3:
        u_heights = [upper.boxes[f][3] - upper.boxes[f][1] for f in common]
        l_heights = [lower.boxes[f][3] - lower.boxes[f][1] for f in common]
        u_cys = [(upper.boxes[f][1] + upper.boxes[f][3]) / 2 for f in common]
        l_cys = [(lower.boxes[f][1] + lower.boxes[f][3]) / 2 for f in common]
        uh, lh = float(np.median(u_heights)), float(np.median(l_heights))
        ucy, lcy = float(np.median(u_cys)), float(np.median(l_cys))

    gap = abs(ucy - lcy) / max(lh, 1e-6)
    return uh < lh * KEEPER_HEIGHT_RATIO and gap < KEEPER_GAP_RATIO


def _smooth_and_fill(boxes: dict, n_frames: int) -> dict:
    """
    Interpolate frames where the striker track dropped out, then apply a
    small temporal smoothing so the crop window does not jitter.
    """
    idxs = sorted(boxes)
    if not idxs:
        return {}

    filled = {}
    for i in range(idxs[0], idxs[-1] + 1):
        if i in boxes:
            filled[i] = boxes[i]
            continue
        prev = max((k for k in idxs if k < i), default=None)
        nxt = min((k for k in idxs if k > i), default=None)
        if prev is None or nxt is None:
            continue
        t = (i - prev) / (nxt - prev)
        filled[i] = tuple(
            boxes[prev][j] * (1 - t) + boxes[nxt][j] * t for j in range(4)
        )

    keys = sorted(filled)
    smoothed = {}
    for i in keys:
        window = [filled[k] for k in keys if abs(k - i) <= 2]
        smoothed[i] = tuple(float(np.mean([w[j] for w in window])) for j in range(4))
    return smoothed


def _pose_nearest_box(poses, box, px1, py1, cw, ch, W, H,
                      return_index: bool = False):
    """
    Of the poses found in the crop, return the one whose body actually sits
    on the striker's detection box; None if none of them do.

    Matching is on the torso (shoulders and hips) rather than all 33
    landmarks, because outflung arms and the bat drag a whole-body centroid
    away from the person's actual position. Distance is expressed in units
    of the striker's own box height so the test means the same thing whether
    the shot is wide or zoomed right in.
    """
    if not poses:
        return None

    bx1, by1, bx2, by2 = box
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    bh = max(by2 - by1, 1e-6)

    bw = max(bx2 - bx1, 1e-6)
    # Tolerance band around the box. On a tight zoom the keeper's torso can
    # fall within half a box-height of the striker's centre, so distance
    # alone is not enough — the torso has to actually be on the box.
    tx, ty = bw * BOX_TOLERANCE, bh * BOX_TOLERANCE

    best, best_d, best_i = None, None, None
    for i, lms in enumerate(poses):
        torso = [lms[j] for j in (11, 12, 23, 24) if lms[j].visibility > 0.3]
        if not torso:
            continue
        cx = sum((px1 + p.x * cw) / W for p in torso) / len(torso)
        cy = sum((py1 + p.y * ch) / H for p in torso) / len(torso)
        if not (bx1 - tx <= cx <= bx2 + tx and by1 - ty <= cy <= by2 + ty):
            continue
        d = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5 / bh
        if best_d is None or d < best_d:
            best, best_d, best_i = lms, d, i

    # Half a box-height of slack: enough for a stretching batsman, far less
    # than the ~1.0+ that a different person standing alongside would score.
    if best_d is None or best_d > MAX_POSE_OFFSET:
        return None
    # The index is what lets the caller pull the matching world landmarks,
    # which live in a parallel list.
    return (best_i, best) if return_index else best


def pose_on_crop(landmarker, frame: np.ndarray, box: tuple):
    """
    (image_keypoints, world_landmarks) for the striker in this frame, or
    (None, None). See `_pose_on_crop_impl` for the detail.
    """
    out = _pose_on_crop(landmarker, frame, box)
    return out if out is not None else (None, None)


def _pose_on_crop(landmarker, frame: np.ndarray, box: tuple):
    """
    Run MediaPipe pose on an upscaled crop around the striker and map the
    landmarks back into full-frame normalised coordinates.
    """
    import mediapipe as mp

    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box

    # The crop must be square *in pixels*. Sizing it in normalised units
    # instead makes it a 16:9 rectangle, and resizing that to a square then
    # stretches the person ~1.8x vertically — MediaPipe is trained on
    # undistorted people and simply fails to find a pose in such a crop.
    cx_px, cy_px = (x1 + x2) / 2 * W, (y1 + y2) / 2 * H
    side = max((x2 - x1) * W, (y2 - y1) * H) * (1 + 2 * CROP_MARGIN)
    half = side / 2
    px1, py1 = int(round(cx_px - half)), int(round(cy_px - half))
    px2, py2 = int(round(cx_px + half)), int(round(cy_px + half))
    if px2 - px1 < 8 or py2 - py1 < 8:
        return None

    # Pad rather than clamp when the square runs off frame, so the crop
    # stays square and the coordinate mapping back stays exact.
    pad_l, pad_t = max(0, -px1), max(0, -py1)
    pad_r, pad_b = max(0, px2 - W), max(0, py2 - H)
    crop = frame[max(0, py1):min(H, py2), max(0, px1):min(W, px2)]
    if crop.size == 0:
        return None
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    ch, cw = crop.shape[:2]
    crop_rs = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
        return None

    # The crop is centred on the striker but the margin routinely pulls in a
    # neighbour — most often the keeper, who stands a metre away and is fully
    # visible while the striker is mid-stride. Asking for a single pose let
    # MediaPipe return whichever person it found most convincing, which in
    # data/defense/video1.mp4 was the keeper for the entire clip: the box was
    # on the striker but the skeleton was drawn on the man behind them.
    # So take every pose in the crop and keep the one actually standing where
    # the striker is.
    chosen = _pose_nearest_box(result.pose_landmarks, box, px1, py1, cw, ch,
                               W, H, return_index=True)
    if chosen is None:
        return None
    pose_i, lms = chosen

    # World landmarks: metric 3D, hip-centred, in metres. Joint angles must be
    # computed from these rather than from the image landmarks. In image
    # coordinates the leg points towards the camera on this broadcast angle,
    # so hip/knee/ankle project almost collinear and every shot measures a
    # near-straight knee — across the dataset the 2D front-knee angle sat at
    # 155-177 degrees for sweep, hook, pull and defense alike, which carries
    # no information. The image landmarks' z is only a relative depth hint and
    # is worse: using it naively gives an 11-degree knee. Measured from world
    # landmarks the same clips separate properly — sweep 127, defense 51,
    # hook 168.
    world = None
    if result.pose_world_landmarks and pose_i < len(result.pose_world_landmarks):
        wl = result.pose_world_landmarks[pose_i]
        world = {i: (float(l.x), float(l.y), float(l.z)) for i, l in enumerate(wl)}

    kp = {}
    for i, lm in enumerate(lms):
        # crop-normalised -> padded-crop pixels -> full-frame pixels -> normalised.
        # px1/py1 are the padded crop's origin, which may sit outside the
        # frame; that is exactly what makes this mapping correct.
        fx = (px1 + lm.x * cw) / W
        fy = (py1 + lm.y * ch) / H
        kp[i] = (float(fx), float(fy), float(lm.z), float(lm.visibility))
    # Returned separately rather than mixed into `kp`: every consumer of a
    # keypoint dict unpacks its values as 4-tuples, so a differently-shaped
    # entry in there would break them.
    return kp, world


def _fill_landmark_gaps(results: list, max_gap: int = MAX_LANDMARK_GAP,
                        vis_floor: float = 0.35) -> list:
    """
    Fill short per-joint dropouts along the clip.

    Because every frame here is the *same* tracked person, a joint that is
    confidently placed before and after a brief occlusion can be linearly
    interpolated across it. This is what stops legs/wrists from flickering
    out mid-shot when the bat or the stumps briefly hide them. Gaps longer
    than `max_gap` frames are left empty — those are real occlusions and
    should stay missing rather than be invented.
    """
    idxs = [i for i, r in enumerate(results) if r is not None]
    if len(idxs) < 2:
        return results

    joint_ids = sorted({j for i in idxs for j in results[i]})
    for j in joint_ids:
        good = [i for i in idxs if results[i].get(j, (0, 0, 0, 0))[3] >= vis_floor]
        for a, b in zip(good, good[1:]):
            if b - a <= 1 or b - a - 1 > max_gap:
                continue
            xa, ya, za, va = results[a][j]
            xb, yb, zb, vb = results[b][j]
            for i in range(a + 1, b):
                if results[i] is None:
                    continue
                t = (i - a) / (b - a)
                results[i][j] = (
                    xa + (xb - xa) * t, ya + (yb - ya) * t, za + (zb - za) * t,
                    # Sits just above the 0.4 draw threshold: enough to be
                    # drawn, but never mistaken for a real high-confidence
                    # detection by anything reading visibility downstream.
                    INTERPOLATED_VIS,
                )
    return results


def run_striker_pose_on_frames(bgr_frames: list, return_debug: bool = False):
    """
    Drop-in replacement for `estimator.run_pose_on_frames`.

    Returns a list (one entry per frame) of {landmark_idx: (x, y, z, vis)}
    in full-frame normalised coords, or None for frames where the striker
    is not present (e.g. crowd / wide replay shots).

    If return_debug is True, returns (results, debug_dict) where debug_dict
    carries the track statistics and the chosen track id.
    """
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    n = len(bgr_frames)
    tracks = detect_person_tracks(bgr_frames)
    stats = summarise_tracks(tracks, n, bgr_frames[0].shape[:2])
    striker = select_striker_track(stats)

    debug = {"stats": stats, "striker_tid": striker.tid if striker else None,
             "boxes": {}}

    if striker is None:
        results = [None] * n
        return (results, debug) if return_debug else results

    boxes = _smooth_and_fill(striker.boxes, n)
    debug["boxes"] = boxes

    model_path = os.path.join(_project_root(), "pose_landmarker.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"MediaPipe pose model not found at {model_path}. "
            f"Run 'python download_pose_model.py' first."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        # A low threshold cannot pull in a wrong person — `_pose_nearest_box`
        # decides which pose is kept — it only decides whether we get a
        # skeleton at all. Held at 0.5 this silently dropped 70% of the sweep
        # clip's frames (white kit on a sun-bleached pitch), even though the
        # striker was correctly boxed in every one of them.
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        num_poses=POSES_PER_CROP,
    )

    results, worlds = [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(bgr_frames):
            box = boxes.get(i)
            kp, world = pose_on_crop(landmarker, frame, box) if box else (None, None)
            results.append(kp)
            worlds.append(world)

    results = _fill_landmark_gaps(results)
    debug["worlds"] = worlds
    return (results, debug) if return_debug else results


def _read_all_frames(video_path: str) -> list:
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def select_striker_window(present: list, n_total: int, max_frames: int | None,
                          gap_tolerance: int = 2) -> list:
    """
    Given the frame indices where the striker is visible, return the frame
    indices to actually analyse.

    Takes the longest unbroken stretch of striker visibility — that is the
    batting camera shot — and samples evenly within it. Anything outside is
    a replay, a crowd cutaway or a wide establishing shot.

    `max_frames=None` keeps every frame of that stretch, which is what the
    overlay videos want; the classifier path passes a fixed count instead.
    """
    if not present:
        return []

    # `gap_tolerance` must reflect how far apart the DETECTED frames are. When
    # detection is strided (see MAX_DETECTIONS) consecutive detections are
    # several frames apart, and a fixed tolerance of 2 would read every single
    # detection as its own separate run — collapsing the window to one frame.
    runs, run = [], [present[0]]
    for a, b in zip(present, present[1:]):
        if b - a <= gap_tolerance:
            run.append(b)
        else:
            runs.append(run)
            run = [b]
    runs.append(run)

    best_run = max(runs, key=len)

    # Expand the run to every frame it spans. `present` only lists the frames
    # detection actually ran on, but `_smooth_and_fill` gives the caller a box
    # for every frame in between, and pose wants near-consecutive frames — the
    # impact detector and the landmark gap-filler both reason about motion
    # between adjacent frames.
    span = list(range(best_run[0], min(best_run[-1] + 1, n_total)))
    if max_frames is None or len(span) <= max_frames:
        return span
    step = len(span) / max_frames
    return [span[int(i * step)] for i in range(max_frames)]


def run_striker_pose_on_video(video_path: str, max_frames: int | None = 30,
                              return_debug: bool = False,
                              keep_all_frames: bool = False):
    """
    Full-clip entry point.

    `video_utils.extract_raw_frames` takes a fixed middle-60% slice of the
    clip on the assumption that the shot always lives there. On broadcast
    footage that assumption breaks — in data/lofted/video1.mp4 the middle
    60% is entirely a wide establishing shot and crowd, and the actual shot
    happens between 10% and 18% of the clip, so that clip yielded no pose
    data at all.

    This scans the whole clip for the striker first, then analyses only the
    stretch where they are actually on camera.

    Returns (frames, keypoints[, debug]) where frames are the analysed BGR
    frames and keypoints line up with them one-to-one.
    """
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    all_frames = _read_all_frames(video_path)
    if not all_frames:
        raise ValueError(f"no frames read from {video_path}")

    tracks = detect_person_tracks(all_frames)
    stats = summarise_tracks(tracks, len(all_frames),
                             all_frames[0].shape[:2])
    striker = select_striker_track(stats)

    debug = {"stats": stats, "striker_tid": striker.tid if striker else None,
             "n_total": len(all_frames), "window": [], "boxes": {}}

    if striker is None:
        return ([], [], debug) if return_debug else ([], [])

    # Detection may have been strided, so allow gaps of a couple of strides
    # before treating a break as a genuine cut to another camera.
    detected = sorted(striker.boxes)
    stride = max(1, int(np.ceil(len(all_frames) / MAX_DETECTIONS)))
    window = select_striker_window(detected, len(all_frames), max_frames,
                                   gap_tolerance=stride * 2 + 1)
    debug["window"] = window

    boxes_full = _smooth_and_fill(striker.boxes, len(all_frames))
    if keep_all_frames:
        # Whole clip, so the overlay video lines up frame-for-frame with the
        # original. Frames outside the striker's stretch simply get no box
        # and therefore no skeleton.
        frames = all_frames
        source = list(range(len(all_frames)))
        in_window = set(window)
        boxes = {j: boxes_full[i] for j, i in enumerate(source)
                 if i in in_window and i in boxes_full}
    else:
        source = window
        frames = [all_frames[i] for i in window]
        boxes = {j: boxes_full[i] for j, i in enumerate(source) if i in boxes_full}
    debug["boxes"] = boxes

    model_path = os.path.join(_project_root(), "pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        num_poses=POSES_PER_CROP,
    )

    results, worlds = [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for j, frame in enumerate(frames):
            box = boxes.get(j)
            kp, world = pose_on_crop(landmarker, frame, box) if box else (None, None)
            results.append(kp)
            worlds.append(world)

    results = _fill_landmark_gaps(results)
    debug["worlds"] = worlds
    return (frames, results, debug) if return_debug else (frames, results)
