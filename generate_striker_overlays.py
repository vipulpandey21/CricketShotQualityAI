"""
generate_striker_overlays.py
Write striker-only skeleton overlay videos next to the source clips.

    python generate_striker_overlays.py                 # all video1 clips
    python generate_striker_overlays.py --all           # all 5 per class
    python generate_striker_overlays.py cover sweep     # named classes only

For data/<class>/videoN.mp4 it writes:

    data/<class>/skeleton_videoN_striker.mp4

The existing skeleton_videoN_overlay.mp4 files are left untouched, so the
old multi-person output and the new striker-only output sit side by side in
the same folder for comparison.

The output video is the same length and frame rate as the source, so it can
be scrubbed against the original. A green box marks the person the model
identified as the striker; the skeleton is drawn only on them. Frames with
no box are stretches where the striker is not on camera at all — a replay,
a crowd cutaway or a wide establishing shot — and are deliberately left
clean rather than being annotated with a guess.
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pose.striker_pose import run_striker_pose_on_video  # noqa: E402
from src.pose.estimator import draw_skeleton                 # noqa: E402

SHOT_CLASSES = ["cover", "defense", "flick", "hook", "late_cut",
                "lofted", "pull", "square_cut", "straight", "sweep"]

CRICKET_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
ROOT = Path(__file__).resolve().parent


def source_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 1 else 25.0


def build_overlay(video_path: Path) -> tuple:
    frames, kps, dbg = run_striker_pose_on_video(
        str(video_path), max_frames=None, return_debug=True, keep_all_frames=True)

    if not frames:
        return None, dbg

    out_path = video_path.parent / f"skeleton_{video_path.stem}_striker.mp4"
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             source_fps(video_path), (w, h))

    drawn = 0
    for i, (frame, kp) in enumerate(zip(frames, kps)):
        img = draw_skeleton(frame, kp) if kp else frame.copy()
        box = dbg["boxes"].get(i)
        if box:
            cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)),
                          (int(box[2] * w), int(box[3] * h)), (0, 255, 0), 2)
            cv2.putText(img, "STRIKER", (int(box[0] * w),
                                         max(16, int(box[1] * h) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            label, colour = ("STRIKER TRACKED", (0, 255, 0)) if kp else \
                            ("striker found, no pose", (0, 200, 255))
        else:
            label, colour = ("no striker on camera", (0, 0, 255))
        if kp:
            drawn += 1
        cv2.putText(img, label, (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2)
        writer.write(img)
    writer.release()

    return out_path, dbg | {"drawn": drawn, "n_out": len(frames), "kps": kps}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_all = "--all" in sys.argv
    classes = args or SHOT_CLASSES
    indices = range(1, 6) if do_all else [1]

    jobs = []
    for cls in classes:
        for i in indices:
            p = ROOT / "data" / cls / f"video{i}.mp4"
            if p.exists():
                jobs.append(p)

    print(f"{len(jobs)} clip(s) to process\n")
    t_start = time.time()
    failures = []

    for n, video in enumerate(jobs, 1):
        print(f"[{n}/{len(jobs)}] {video.parent.name}/{video.name}",
              end=" ... ", flush=True)
        t0 = time.time()
        try:
            out_path, dbg = build_overlay(video)
        except Exception as exc:                       # noqa: BLE001
            print(f"ERROR: {exc}")
            failures.append((video, str(exc)))
            continue

        if out_path is None:
            print("NO STRIKER FOUND — nothing written")
            failures.append((video, "no striker track"))
            continue

        kps = dbg["kps"]
        counts = [sum(1 for j in CRICKET_JOINTS if k.get(j, (0, 0, 0, 0))[3] > 0.4)
                  for k in kps if k is not None]
        avg_joints = sum(counts) / max(len(counts), 1)
        win = dbg["window"]
        print(f"{dbg['drawn']}/{dbg['n_out']} frames  "
              f"joints {avg_joints:.1f}/13  "
              f"striker frames {win[0]}-{win[-1]}  ({time.time()-t0:.0f}s)")

    print(f"\ndone in {(time.time()-t_start)/60:.1f} min")
    if failures:
        print(f"\n{len(failures)} clip(s) need attention:")
        for v, why in failures:
            print(f"  {v.parent.name}/{v.name}: {why}")
    print("\nNew files are named  skeleton_<video>_striker.mp4")
    print("Old files  skeleton_<video>_overlay.mp4  were not modified.")


if __name__ == "__main__":
    main()
