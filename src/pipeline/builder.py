"""
builder.py
Produce the full data folder for one clip — the same artefacts that
data/<class>/videoN_pipeline/ holds, for any video including uploads.

Layout produced:

    <out_dir>/
      00_metadata.json             clip info + detection stats
      01_extracted_frames/         the 30 frames the classifier is given
      02_skeleton_keypoints.json   all 33 landmarks per analysed frame
      03_skeleton_overlay_frames/  skeleton drawn on the striker
      04_comparison_frames/        original beside skeleton, side by side
      05_shot_analysis.json        prediction, top-3, joint angles, quality
      skeleton_<stem>.mp4          the overlay frames as a playable clip
      PIPELINE_SUMMARY.txt         human-readable digest

Everything the site shows for an upload comes from one call to
`build_pipeline`, so what the user downloads is exactly what they were
shown — there is no second code path that could disagree.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np

from src.pose.estimator import (CRICKET_LANDMARKS, compute_cricket_angles,
                                draw_skeleton, phase_angles, pose_summary)
from src.pose.shot_curve import normalized_shot_curve
from src.pose.striker_pose import run_striker_pose_on_video
from src.utils.video_utils import extract_frames

CRICKET_JOINTS = list(CRICKET_LANDMARKS)
DRAW_VIS = 0.4          # matches draw_skeleton's threshold
OVERLAY_FPS = 8

IDEAL_ANGLES_PATH = Path(__file__).resolve().parents[2] / "ideal_angles.json"
IDEAL_CURVES_PATH = Path(__file__).resolve().parents[2] / "ideal_angle_curves.json"
ANGLE_LABELS = {
    "front_knee_angle": "Front knee", "back_knee_angle": "Back knee",
    "front_elbow_angle": "Front elbow", "back_elbow_angle": "Back elbow",
    "shoulder_tilt_deg": "Shoulder tilt", "hip_tilt_deg": "Hip tilt",
    "trunk_lean_deg": "Trunk lean",
}


def _load_pro_ranges() -> dict:
    if not IDEAL_ANGLES_PATH.exists():
        return {}
    try:
        return json.loads(IDEAL_ANGLES_PATH.read_text(encoding="utf-8")).get("classes", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _load_pro_curves() -> dict:
    if not IDEAL_CURVES_PATH.exists():
        return {}
    try:
        return json.loads(IDEAL_CURVES_PATH.read_text(encoding="utf-8")).get("classes", {})
    except (json.JSONDecodeError, OSError):
        return {}


def pro_comparison(shot_type: str, angles: dict) -> list:
    """
    For each angle measured at impact, compare against the professional
    median for this shot type (from derive_ideal_angles.py's interquartile
    analysis of the dataset's professional clips).

    Returns a list of {label, actual, pro_median, diff, low, high, in_range}
    for every angle where both this clip's value and a professional range
    exist. Used for the "vs professional" panel — not for grading (scorer.py
    already does that); this is the plain-language "you vs the pros" view.
    """
    ranges = _load_pro_ranges().get(shot_type, {})
    if not ranges:
        return []

    out = []
    for key, label in ANGLE_LABELS.items():
        actual = angles.get(key)
        band = ranges.get(key)
        if actual is None or not band:
            continue
        median = band.get("median")
        low, high = band.get("low"), band.get("high")
        if median is None:
            continue
        out.append({
            "label": label,
            "actual": round(actual, 1),
            "pro_median": median,
            "diff": round(actual - median, 1),
            "low": low, "high": high,
            "in_range": low is not None and high is not None and low <= actual <= high,
            "n_pro_clips": band.get("n"),
        })
    return out


def curve_comparison(shot_type: str, curve: dict | None) -> dict:
    """
    Movement version of `pro_comparison`: the same angles read as a curve
    from shot-start to impact instead of one number at impact, laid over
    the professional band at every normalized time-step
    (`derive_angle_curves.py`'s output) — this is what makes the
    "You vs Professionals" comparison a shape-over-time match instead of a
    single point.

    Returns {angle_key: {label, user: [n], pro_low/median/high: [n or None],
    n_pro_clips, start_frame, impact_frame}} for every angle where both a
    curve and a professional band exist. Empty dict if there's no curve
    (striker not found / clip too short) or no professional data for this
    shot type.
    """
    if not curve:
        return {}
    pro = _load_pro_curves().get(shot_type, {})
    if not pro:
        return {}

    out = {}
    for key, label in ANGLE_LABELS.items():
        user_series = curve["curves"].get(key)
        band = pro.get(key)
        if not user_series or not band:
            continue
        if all(v is None for v in user_series):
            continue
        out[key] = {
            "label": label,
            "user": user_series,
            "pro_low": [b["low"] if b else None for b in band],
            "pro_median": [b["median"] if b else None for b in band],
            "pro_high": [b["high"] if b else None for b in band],
            "n_pro_clips": next((b["n"] for b in band if b), None),
            "start_frame": curve["start_frame"],
            "impact_frame": curve["impact_frame"],
        }
    return out

# H.264. Browsers — and therefore st.video — cannot decode OpenCV's default
# mp4v, so a file written with it appears as a blank player. Fall back to
# mp4v only if this build of OpenCV has no H.264 encoder, in which case the
# file is still valid for download even if it will not play inline.
VIDEO_CODECS = ("avc1", "h264", "mp4v")


def _open_writer(path, fps, size):
    """VideoWriter using the first codec this OpenCV build actually opens."""
    for codec in VIDEO_CODECS:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                 fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"no usable video codec from {VIDEO_CODECS}")


def _clip_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS)) or 0.0, 2),
        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
    }
    cap.release()
    info["duration_seconds"] = round(
        info["total_frames"] / info["fps"], 2) if info["fps"] else None
    return info


def _jsonable(obj):
    if is_dataclass(obj):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _side_by_side(original, annotated):
    h = original.shape[0]
    gap = np.full((h, 8, 3), 40, dtype=np.uint8)
    combined = np.hstack([original, gap, annotated])
    cv2.putText(combined, "ORIGINAL", (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(combined, "STRIKER SKELETON",
                (original.shape[1] + 22, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    return combined


def build_pipeline(video_path, out_dir, classifier=None, idx_to_class=None,
                   scorer=None, max_frames: int = 30,
                   write_frames: bool = True, predictor=None,
                   display_name: str | None = None) -> dict:
    """
    Run the whole analysis for one clip and write every artefact to out_dir.

    `predictor` is a ShotPredictor (r3d_18 + EfficientNetB0, 62.4% test
    top-1) and is preferred. `classifier` is the older single-backbone Keras
    model (57.6%) and is used only if no predictor is supplied.

    All of predictor/classifier/idx_to_class/scorer are optional: without
    them the pose half still runs, so the data folder is still produced when
    TensorFlow is unavailable.

    Returns a dict of everything the UI needs to display.
    """
    video_path = Path(video_path)
    # An upload arrives as a NamedTemporaryFile, so video_path.stem is
    # something like "tmp69gi2xsq". Naming the artefacts after that leaves the
    # user with skeleton_tmp69gi2xsq.mp4 inside their download.
    label = Path(display_name).stem if display_name else video_path.stem

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    result = {"video": display_name or video_path.name,
              "out_dir": str(out_dir), "timings": {}}
    result.update(_clip_info(video_path))

    # ── 1. classifier input frames ───────────────────────────────────────
    t0 = time.time()
    clf_frames = extract_frames(str(video_path), n_frames=max_frames)
    result["timings"]["extract_frames"] = round(time.time() - t0, 1)

    if write_frames:
        d = out_dir / "01_extracted_frames"
        d.mkdir()
        for i, f in enumerate(clf_frames):
            # stored RGB -> write as BGR so the files look right
            cv2.imwrite(str(d / f"frame_{i+1:03d}.jpg"), f[..., ::-1])
    result["extracted_frames"] = len(clf_frames)

    # ── 2. shot prediction ───────────────────────────────────────────────
    prediction = None
    if idx_to_class is not None and (predictor is not None or classifier is not None):
        t0 = time.time()
        if predictor is not None:
            prediction = predictor.predict(video_path, idx_to_class)
        else:
            probs = classifier.predict(np.expand_dims(clf_frames, 0), verbose=0)[0]
            order = np.argsort(probs)[::-1]
            prediction = {
                "shot": idx_to_class[int(order[0])],
                "confidence": round(float(probs[order[0]]) * 100, 1),
                "top3": [{"shot": idx_to_class[int(j)],
                          "confidence": round(float(probs[j]) * 100, 1)}
                         for j in order[:3]],
                "all_probabilities": {idx_to_class[i]: round(float(p) * 100, 2)
                                      for i, p in enumerate(probs)},
                "model": "EfficientNetB0+GRU (57.6% test top-1)",
            }
        result["timings"]["shot_prediction"] = round(time.time() - t0, 1)
    result["prediction"] = prediction

    # ── 3. striker pose ──────────────────────────────────────────────────
    t0 = time.time()
    pose_frames, keypoints, pose_dbg = run_striker_pose_on_video(
        str(video_path), max_frames=max_frames, return_debug=True)
    result["timings"]["striker_pose"] = round(time.time() - t0, 1)
    worlds = pose_dbg.get("worlds")
    boxes = pose_dbg.get("boxes") or {}
    result["striker_track_id"] = pose_dbg.get("striker_tid")

    summary = pose_summary(keypoints) if keypoints else {
        "detected_frames": 0, "total_frames": 0, "detection_rate": 0.0,
        "avg_keypoints": {}, "cricket_joints": {}}
    # Angles come from the impact frame, not from the clip-averaged pose.
    # See estimator.angles_at_frame for why the average is meaningless.
    phases = phase_angles(keypoints, worlds) if keypoints else None
    angles = phases["impact"] if phases else compute_cricket_angles({})
    handedness = phases["handedness"] if phases else {
        "hand": "right", "confidence": 0.0, "assumed": True}
    clip_avg_angles = compute_cricket_angles(
        summary["avg_keypoints"], handedness=handedness["hand"])

    joints_per_frame = [
        sum(1 for j in CRICKET_JOINTS
            if kp.get(j, (0, 0, 0, 0))[3] > DRAW_VIS)
        for kp in keypoints if kp is not None
    ]
    result["striker_found"] = bool(keypoints)
    result["detected_frames"] = summary["detected_frames"]
    result["analysed_frames"] = len(pose_frames)
    result["detection_rate"] = round(summary["detection_rate"] * 100, 1)
    result["avg_joints_per_frame"] = round(
        sum(joints_per_frame) / max(len(joints_per_frame), 1), 1)
    result["joint_angles"] = angles
    result["phases"] = phases
    result["handedness"] = handedness
    result["clip_avg_angles"] = clip_avg_angles
    result["impact_frame"] = phases["frames"]["impact"] if phases else None
    result["cricket_joints"] = summary["cricket_joints"]
    result["pro_comparison"] = (
        pro_comparison(prediction["shot"], angles) if prediction else [])

    # Movement curve — same angles, read continuously from shot-start to
    # impact instead of one snapshot. See src/pose/shot_curve.py.
    shot_curve = normalized_shot_curve(
        keypoints, worlds, handedness["hand"]) if keypoints else None
    result["shot_curve"] = shot_curve
    result["angle_curve"] = (
        curve_comparison(prediction["shot"], shot_curve) if prediction else {})

    # ── 4. keypoints, overlays, comparisons, video ───────────────────────
    kp_dump = {}
    for i, kp in enumerate(keypoints):
        if kp is None:
            kp_dump[f"frame_{i+1:03d}"] = None
            continue
        kp_dump[f"frame_{i+1:03d}"] = {
            CRICKET_LANDMARKS.get(idx, f"landmark_{idx}"): {
                "x": round(v[0], 5), "y": round(v[1], 5),
                "z": round(v[2], 5), "visibility": round(v[3], 4)}
            for idx, v in kp.items()
        }
    (out_dir / "02_skeleton_keypoints.json").write_text(
        json.dumps(_jsonable(kp_dump), indent=1), encoding="utf-8")

    if pose_frames:
        ov_dir = out_dir / "03_skeleton_overlay_frames"
        cmp_dir = out_dir / "04_comparison_frames"
        if write_frames:
            ov_dir.mkdir()
            cmp_dir.mkdir()

        h, w = pose_frames[0].shape[:2]
        vid_path = out_dir / f"skeleton_{label}.mp4"
        writer, codec = _open_writer(vid_path, OVERLAY_FPS, (w, h))
        result["video_codec"] = codec

        cmp_path = out_dir / f"comparison_{label}.mp4"
        cmp_w = w * 2 + 8      # matches _side_by_side's 8px divider
        cmp_writer, _ = _open_writer(cmp_path, OVERLAY_FPS, (cmp_w, h))

        for i, (frame, kp) in enumerate(zip(pose_frames, keypoints)):
            annotated = draw_skeleton(frame, kp) if kp else frame.copy()

            # Draw the striker's detection box too, so the video shows both
            # WHO was picked and WHAT was measured on them — the same look as
            # the skeleton_*_striker.mp4 files in data/.
            box = boxes.get(i)
            if box:
                cv2.rectangle(annotated,
                              (int(box[0] * w), int(box[1] * h)),
                              (int(box[2] * w), int(box[3] * h)),
                              (0, 255, 0), 2)
                cv2.putText(annotated, "STRIKER",
                            (int(box[0] * w), max(18, int(box[1] * h) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            label = "STRIKER TRACKED" if kp else (
                "striker found, no pose" if box else "no striker on camera")
            colour = (0, 255, 0) if kp else ((0, 200, 255) if box else (0, 0, 255))
            stamped = annotated.copy()
            cv2.putText(stamped, label, (14, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            writer.write(stamped)

            side = _side_by_side(frame, annotated)
            cmp_writer.write(side)

            if write_frames:
                cv2.imwrite(str(ov_dir / f"frame_{i+1:03d}.jpg"), stamped)
                cv2.imwrite(str(cmp_dir / f"frame_{i+1:03d}.jpg"), side)
        writer.release()
        cmp_writer.release()
        result["skeleton_video"] = str(vid_path)
        result["comparison_video"] = str(cmp_path)

    # ── 5. quality score ─────────────────────────────────────────────────
    quality, q = None, None
    if scorer is not None and prediction is not None:
        q = scorer(prediction["shot"], angles)
        quality = _jsonable(q)
    result["quality"] = quality

    # ── 6. metadata + analysis + summary ─────────────────────────────────
    (out_dir / "00_metadata.json").write_text(json.dumps(_jsonable({
        "video_file": result["video"],
        "total_frames": result["total_frames"],
        "fps": result["fps"],
        "duration_seconds": result["duration_seconds"],
        "resolution": result["resolution"],
        "extracted_frames": result["extracted_frames"],
        "detection_method": "striker-only (YOLO track + crop pose)",
        "striker_found": result["striker_found"],
        "analysed_frames": result["analysed_frames"],
        "detected_frames": result["detected_frames"],
        "detection_rate": result["detection_rate"],
        "avg_joints_per_frame": result["avg_joints_per_frame"],
        "timings_seconds": result["timings"],
    }), indent=2), encoding="utf-8")

    (out_dir / "05_shot_analysis.json").write_text(json.dumps(_jsonable({
        "prediction": prediction,
        "joint_angles_at_impact": angles,
        "impact_frame": result["impact_frame"],
        "angles_by_phase": phases,
        # kept for comparison only — averaging joints over the clip does not
        # describe any pose the batsman actually held
        "clip_averaged_angles_do_not_use": clip_avg_angles,
        "average_joint_positions": result["cricket_joints"],
        "quality": quality,
        "vs_professional": result["pro_comparison"],
        "vs_professional_movement": result["angle_curve"],
    }), indent=2), encoding="utf-8")

    (out_dir / "PIPELINE_SUMMARY.txt").write_text(
        _summary_text(result), encoding="utf-8")

    # In-memory handles for the caller, so the UI can render the same arrays
    # this function already computed rather than paying for a second pass.
    # Underscore-prefixed keys are never serialised — the JSON files above are
    # written from explicit dicts.
    result["_clf_frames"] = clf_frames
    result["_frames"] = pose_frames
    result["_keypoints"] = keypoints
    result["_summary"] = summary
    result["_quality"] = q if scorer is not None and prediction else None

    return result


def _summary_text(r: dict) -> str:
    L = []
    add = L.append
    add("CRICKET SHOT PIPELINE SUMMARY (striker-only detection)")
    add("=" * 78)
    add(f"Video           : {r['video']}")
    add(f"Resolution      : {r['resolution']}   fps {r['fps']}   "
        f"{r['total_frames']} frames ({r['duration_seconds']}s)")
    add("")
    add("SHOT PREDICTION")
    if r.get("prediction"):
        p = r["prediction"]
        add(f"  Predicted     : {p['shot']}  ({p['confidence']}% confidence)")
        add("  Top 3         : " + ", ".join(
            f"{t['shot']} {t['confidence']}%" for t in p["top3"]))
    else:
        add("  unavailable (classifier not loaded)")
    add("")
    add("STRIKER POSE")
    add(f"  Striker found : {'yes' if r['striker_found'] else 'NO'}")
    add(f"  Frames        : {r['detected_frames']}/{r['analysed_frames']} "
        f"({r['detection_rate']}%)")
    add(f"  Joints/frame  : {r['avg_joints_per_frame']} of 13")
    hnd = r.get("handedness") or {}
    if hnd.get("assumed"):
        add(f"  Handedness    : assumed right-handed (not enough data to detect)")
    else:
        add(f"  Handedness    : {hnd.get('hand','right')}-handed "
            f"(confidence {hnd.get('confidence',0)*100:.0f}%) — "
            f"'front' below means the {hnd.get('hand','right')}-hander's front side")
    add("")
    add(f"JOINT ANGLES AT IMPACT (frame {r.get('impact_frame')})")
    for k, v in (r.get("joint_angles") or {}).items():
        if k == "handedness":
            continue
        add(f"  {k:<20}{'-' if v is None else str(v) + ' deg'}")
    ph = r.get("phases")
    if ph:
        add("")
        add("  by phase" + f"  (stance f{ph['frames']['stance']}, "
            f"impact f{ph['frames']['impact']}, "
            f"follow-through f{ph['frames']['follow_through']})")
        keys = [k for k in ph["impact"] if k.endswith(("angle", "deg"))]
        add(f"  {'':<20}{'stance':>10}{'impact':>10}{'follow':>10}")
        for k in keys:
            row = "".join(
                f"{('-' if ph[p][k] is None else round(ph[p][k])):>10}"
                for p in ("stance", "impact", "follow_through"))
            add(f"  {k:<20}{row}")
    add("")
    add("SHOT QUALITY")
    q = r.get("quality")
    if q:
        add(f"  Overall       : {q['overall_score']}/100  ({q['grade']})")
        for c in q.get("criteria", []):
            add(f"  - {c['name']:<22} ideal {c['ideal']:<14} "
                f"actual {c['actual']:<10} {c['score']:.0f}/100  {c['status']}")
        add("")
        add("  NOTE: quality is graded against the PREDICTED shot type. If the")
        add("  prediction is wrong, these criteria are the wrong ones.")
    else:
        add("  unavailable")
    add("")

    vs_pro = r.get("pro_comparison")
    if vs_pro:
        add(f"VS PROFESSIONAL ({vs_pro[0].get('n_pro_clips','?')} clips of this shot type)")
        for c in vs_pro:
            arrow = "≈" if c["in_range"] else ("more than" if c["diff"] > 0 else "less than")
            add(f"  {c['label']:<14} you {c['actual']:>6.0f}°   "
                f"pro median {c['pro_median']:>6.0f}°   "
                f"({abs(c['diff']):.0f}° {arrow} typical)")
        add("")

    add("TIMINGS (seconds)")
    for k, v in r["timings"].items():
        add(f"  {k:<20}{v}")
    return "\n".join(L) + "\n"


def zip_pipeline(out_dir, zip_path=None) -> Path:
    """
    Zip a built pipeline folder so the site can offer it as one download.

    Reuses an existing archive that is already newer than everything in the
    folder. The previous version deleted and rebuilt the zip on every call,
    which on Windows raised

        PermissionError: [WinError 32] The process cannot access the file
        because it is being used by another process

    and crashed the whole app — Streamlit re-runs the script on every widget
    interaction, so this fired as soon as the user touched anything while the
    browser still held the last download open. Rebuilding a 13 MB archive on
    each interaction was wasteful even when it did succeed.
    """
    out_dir = Path(out_dir)
    zip_path = Path(zip_path) if zip_path else out_dir.with_suffix(".zip")

    if zip_path.exists():
        newest = max((p.stat().st_mtime for p in out_dir.rglob("*")
                      if p.is_file()), default=0)
        if zip_path.stat().st_mtime >= newest:
            return zip_path

    # Build under a distinct name, then move into place. Writing directly to
    # `zip_path` would hit the same lock we are trying to avoid.
    #
    # Take the built path from make_archive rather than reconstructing it.
    # Deriving it with `staging.with_suffix(".zip")` was wrong: with_suffix
    # REPLACES the last suffix, so a staging base of
    # "clip_pipeline.building" resolved to "clip_pipeline.zip" while the file
    # actually written was "clip_pipeline.building.zip" — the returned path did
    # not exist and the page died on read_bytes().
    staging_base = zip_path.with_name(zip_path.stem + "__building")
    built = Path(shutil.make_archive(str(staging_base), "zip", str(out_dir)))
    try:
        os.replace(built, zip_path)
    except OSError:
        # Target is locked by a reader. The staged archive is complete and
        # correct, so hand that back rather than failing the page.
        return built
    return zip_path
