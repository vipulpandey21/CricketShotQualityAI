"""
verify_striker.py
Point this at ANY cricket clip and see exactly who the model picked.

    python verify_striker.py data/cover/video1.mp4
    python verify_striker.py "C:\\path\\to\\my_new_clip.mp4"
    python verify_striker.py data/cover/video1.mp4 data/sweep/video1.mp4

For each clip it writes into verification/ :
    <clip>_sheet.jpg  every analysed frame in one grid — quickest to scan
    <clip>.mp4        the same frames as a playable video

A green box marks who the model decided is the striker; the skeleton is
drawn only on that person. If the box is ever on the umpire, the bowler,
the keeper or the non-striker, that clip is a failure — say so and it gets
fixed. Nothing here touches data/ or the trained model.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pose.striker_pose import run_striker_pose_on_video  # noqa: E402
from src.pose.estimator import draw_skeleton                 # noqa: E402

COLS = 6
TILE_W = 320
CRICKET_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def annotate(frame, kp, box, tid):
    img = draw_skeleton(frame, kp) if kp else frame.copy()
    h, w = img.shape[:2]
    if box:
        cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)),
                      (int(box[2] * w), int(box[3] * h)), (0, 255, 0), 2)
        cv2.putText(img, f"STRIKER (track {tid})",
                    (int(box[0] * w), max(16, int(box[1] * h) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if kp is None:
        cv2.putText(img, "no pose this frame", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return img


def process(video_path: Path, out_dir: Path):
    frames, kps, dbg = run_striker_pose_on_video(str(video_path), max_frames=30,
                                                 return_debug=True)
    name = f"{video_path.parent.name}_{video_path.stem}"

    if not frames:
        print(f"{name}: NO STRIKER FOUND in this clip — nothing drawn.")
        print("   (the model refuses to guess rather than skeleton the wrong person)")
        return

    detected = sum(1 for k in kps if k is not None)
    counts = [sum(1 for j in CRICKET_JOINTS if k.get(j, (0, 0, 0, 0))[3] > 0.4)
              for k in kps if k is not None]
    avg_joints = sum(counts) / max(len(counts), 1)
    win = dbg["window"]

    print(f"\n{name}")
    print(f"  clip length        {dbg['n_total']} frames")
    print(f"  analysed segment   frames {win[0]}-{win[-1]} "
          f"({win[0]/dbg['n_total']*100:.0f}%-{win[-1]/dbg['n_total']*100:.0f}% of clip)")
    print(f"  striker track id   {dbg['striker_tid']}")
    print(f"  skeleton found on  {detected}/{len(frames)} frames "
          f"({detected/len(frames)*100:.0f}%)")
    print(f"  joints per frame   {avg_joints:.1f} of 13")

    imgs = [annotate(f, k, dbg["boxes"].get(i), dbg["striker_tid"])
            for i, (f, k) in enumerate(zip(frames, kps))]

    tiles = []
    for i, img in enumerate(imgs):
        scale = TILE_W / img.shape[1]
        t = cv2.resize(img, (TILE_W, int(img.shape[0] * scale)))
        cv2.rectangle(t, (0, 0), (34, 20), (0, 0, 0), -1)
        cv2.putText(t, str(i), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        tiles.append(t)

    rows = []
    for r in range(0, len(tiles), COLS):
        row = tiles[r:r + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    header = np.zeros((40, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"{name}  -  green box = who the model picked as STRIKER",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"{name}_sheet.jpg"), np.vstack([header, sheet]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 8,
                         (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        vw.write(img)
    vw.release()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    out_dir = Path(__file__).resolve().parent / "verification"
    out_dir.mkdir(exist_ok=True)

    for arg in sys.argv[1:]:
        process(Path(arg), out_dir)

    print(f"\nOpen this folder to check the results:\n  {out_dir}")


if __name__ == "__main__":
    main()
