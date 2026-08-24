"""
cache_pose_features.py
Precompute striker-skeleton features for the dataset, to be fused with the
CNN features when classifying the shot.

Feature design — the point is scale and position invariance. Joint positions
are expressed RELATIVE TO THE STRIKER'S OWN BOUNDING BOX, not to the frame.
A cover drive is the same shape whether the broadcast is zoomed in or wide
and whether the batsman stands left or right of centre, so features must not
encode where in the frame he happens to be. Raw frame coordinates would tie
the model to one camera framing, which is exactly what has to generalise to
newly uploaded clips.

Per frame (46 dims):
    26  13 cricket joints as (x, y) inside the striker box, 0-1
    13  per-joint visibility
     7  joint angles from compute_cricket_angles (front/back knee and elbow,
        shoulder tilt, hip tilt, trunk lean), scaled to 0-1

Speed: running the full striker pipeline on 1750 clips would take many
hours. Person detection runs on every Nth frame only and the box is
interpolated between — the striker moves smoothly, so the box does — while
MediaPipe still runs on every frame, since it is the cheap part.

Usage:
    python cache_pose_features.py --check          # 3 clips, verify quality
    python cache_pose_features.py demo test
    python cache_pose_features.py train val
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2                      # noqa: E402
import numpy as np              # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pose import striker_pose as sp                      # noqa: E402
from src.pose.estimator import (compute_cricket_angles,      # noqa: E402
                                angles_from_world, detect_handedness)
from src.classifier.model import SHOT_CLASSES                # noqa: E402

CLASSES = list(SHOT_CLASSES)
FEAT = ROOT / "features"
N_FRAMES = 30
POSE_DIM = 46

DETECT_EVERY = 3           # run YOLO on every 3rd frame, interpolate between
FAST_MODEL = "yolov8s.pt"  # lighter than the yolov8m used for the overlays
FAST_IMGSZ = 768

JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
ANGLE_KEYS = ["front_knee_angle", "back_knee_angle", "front_elbow_angle",
              "back_elbow_angle", "shoulder_tilt_deg", "hip_tilt_deg",
              "trunk_lean_deg"]


def first_frames(video_path, n=N_FRAMES):
    """The same 30 consecutive frames the classifier sees."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def detect_sparse(frames, model):
    """Person tracks, detecting on every DETECT_EVERY-th frame."""
    tracks = {}
    for i, frame in enumerate(frames):
        if i % DETECT_EVERY:
            continue
        h, w = frame.shape[:2]
        res = model.track(frame, classes=[0], conf=sp.YOLO_CONF,
                          imgsz=FAST_IMGSZ, persist=True,
                          tracker="botsort.yaml", verbose=False)[0]
        if res.boxes is None:
            continue
        for b in res.boxes:
            if b.id is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            tracks.setdefault(int(b.id[0]), {})[i] = (x1 / w, y1 / h,
                                                     x2 / w, y2 / h)
    return tracks


def pose_vector(kp, box, world=None, handedness="right"):
    """
    46-dim frame descriptor, or zeros when there is no pose.

    The angle slots (39-45) are measured from `world` (MediaPipe world
    landmarks, metric 3D) when available, and from `handedness`. Earlier this
    always used the 2D image landmarks with front/back hard-coded to a
    right-hander, which cannot see a bent knee at all on this camera angle —
    hip, knee and ankle project nearly collinear when the leg points toward
    the camera, so every shot read 155-177 degrees regardless of what the
    batsman actually did. World landmarks are real 3D; handedness picks the
    correct physical leg as "front" for a left-hander instead of grading him
    against his own back leg.
    """
    v = np.zeros(POSE_DIM, dtype=np.float32)
    if kp is None or box is None:
        return v

    bx1, by1, bx2, by2 = box
    bw = max(bx2 - bx1, 1e-6)
    bh = max(by2 - by1, 1e-6)

    for j, idx in enumerate(JOINTS):
        if idx not in kp:
            continue
        x, y, _, vis = kp[idx]
        # position within the striker's box, so zoom and placement drop out
        v[j * 2] = np.clip((x - bx1) / bw, -1.0, 2.0)
        v[j * 2 + 1] = np.clip((y - by1) / bh, -1.0, 2.0)
        v[26 + j] = vis

    if world:
        angles = angles_from_world(world, handedness)
    else:
        avg = {i: np.array([kp[i][0], kp[i][1], kp[i][2]])
               for i in kp if kp[i][3] > 0.3}
        angles = compute_cricket_angles(avg, handedness=handedness)
    for k, key in enumerate(ANGLE_KEYS):
        a = angles.get(key)
        if a is not None:
            v[39 + k] = a / 180.0
    return v


def features_for_video(video_path, model, landmarker):
    frames = first_frames(video_path)
    out = np.zeros((N_FRAMES, POSE_DIM), dtype=np.float32)
    if not frames:
        return out, 0

    tracks = detect_sparse(frames, model)
    n_scanned = max(1, (len(frames) + DETECT_EVERY - 1) // DETECT_EVERY)
    stats = sp.summarise_tracks(tracks, n_scanned, frames[0].shape[:2])
    # Detection ran on only every DETECT_EVERY-th frame, so a track can never
    # reach the module's default minimum frame count. Require a share of the
    # frames actually scanned instead.
    striker = sp.select_striker_track(
        stats, min_frames=max(3, int(n_scanned * 0.6)))
    if striker is None:
        return out, 0

    boxes = sp._smooth_and_fill(striker.boxes, len(frames))
    # carry the last known box across frames YOLO never saw
    last = None
    filled = {}
    for i in range(len(frames)):
        if i in boxes:
            last = boxes[i]
        if last is not None:
            filled[i] = last

    kps, worlds = [], []
    for i, frame in enumerate(frames):
        box = filled.get(i)
        kp, world = sp.pose_on_crop(landmarker, frame, box) if box else (None, None)
        kps.append(kp)
        worlds.append(world)
    kps = sp._fill_landmark_gaps(kps)

    # Detected once per clip so every frame is graded against the same
    # physical front/back side — see estimator.detect_handedness.
    hand = detect_handedness(kps, worlds)["hand"]

    n_ok = 0
    for i in range(min(N_FRAMES, len(frames))):
        out[i] = pose_vector(kps[i], filled.get(i), worlds[i], hand)
        n_ok += int(kps[i] is not None)
    return out, n_ok


def build_tools():
    from ultralytics import YOLO
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    weights = ROOT / FAST_MODEL
    model = YOLO(str(weights) if weights.exists() else FAST_MODEL)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(ROOT / "pose_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        num_poses=sp.POSES_PER_CROP,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return model, landmarker


def split_items(split, per_class=None):
    """
    [(video_path, class_index)] for a split.

    `per_class` caps how many videos are taken from each class, for the
    subset run that decides whether pose features are worth the full
    extraction cost. Taking the first N of a sorted listing keeps the
    selection identical between runs, and the classes stay balanced.
    """
    items = []
    if split == "demo":
        for ci, cls in enumerate(CLASSES):
            for i in range(1, 6):
                p = ROOT / "data" / cls / f"video{i}.mp4"
                if p.exists():
                    items.append((p, ci))
    else:
        base = ROOT / "temp_hf_data" / "cricketshot" / split
        for ci, cls in enumerate(CLASSES):
            folder = base / cls
            if not folder.exists():
                continue
            vids = [p for p in sorted(folder.iterdir())
                    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
                    and not p.name.startswith("._")]
            if per_class:
                vids = vids[:per_class]
            items.extend((p, ci) for p in vids)
    return items


def main():
    FEAT.mkdir(exist_ok=True)
    model, landmarker = build_tools()

    if "--check" in sys.argv:
        print("quality check on a few clips "
              f"({FAST_MODEL}, imgsz={FAST_IMGSZ}, detect every {DETECT_EVERY})\n")
        for cls in ["cover", "defense", "sweep"]:
            p = ROOT / "data" / cls / "video1.mp4"
            t0 = time.time()
            feats, n_ok = features_for_video(p, model, landmarker)
            print(f"  {cls:<10} pose on {n_ok}/{N_FRAMES} frames  "
                  f"nonzero dims {int((feats != 0).any(0).sum())}/{POSE_DIM}  "
                  f"{time.time()-t0:.1f}s")
        return

    per_class = None
    if "--per-class" in sys.argv:
        per_class = int(sys.argv[sys.argv.index("--per-class") + 1])

    splits = [a for a in sys.argv[1:]
              if not a.startswith("--") and not a.isdigit()]
    for split in (splits or ["demo"]):
        # demo is only 5 clips per class already; never cap it
        items = split_items(split, per_class if split != "demo" else None)
        if not items:
            print(f"{split}: nothing found")
            continue
        suffix = f"_n{per_class}" if per_class and split != "demo" else ""
        print(f"\n=== pose {split}{suffix}: {len(items)} videos ===")
        X = np.zeros((len(items), N_FRAMES, POSE_DIM), dtype=np.float32)
        found = 0
        t0 = time.time()
        for n, (path, _) in enumerate(items):
            X[n], n_ok = features_for_video(path, model, landmarker)
            found += int(n_ok > 0)
            if (n + 1) % 25 == 0 or n + 1 == len(items):
                done = n + 1
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(items)}  {rate:.2f} vid/s  "
                      f"eta {(len(items)-done)/max(rate,1e-6)/60:.1f} min  "
                      f"striker found in {found}/{done}")
        np.save(FEAT / f"{split}_pose{suffix}.npy", X)
        # The CNN features were cached over the FULL split, while a subset run
        # covers only part of it. Saving the paths lets the trainer line the
        # two feature sets up by video rather than by row index, which would
        # silently pair the wrong clips.
        (FEAT / f"{split}_pose{suffix}_paths.txt").write_text(
            "\n".join(str(p) for p, _ in items), encoding="utf-8")
        print(f"  saved {split}_pose{suffix}.npy {X.shape}")


if __name__ == "__main__":
    main()
