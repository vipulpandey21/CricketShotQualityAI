"""
build_results_folder.py
Runs the real pipeline (same build_pipeline() the app uses) on one sample
clip per shot class, and lays the output out as:

    Results/
      <clip_name>/
        00_metadata/metadata.json
        01_extracted_frames/frame_XXX.jpg ...
        02_skeleton_keypoints/skeleton_keypoints.json
        03_skeleton_overlay_frames/frame_XXX.jpg ...
        04_comparison_frames/frame_XXX.jpg ...
        05_shot_analysis/shot_analysis.json
        skeleton_video.mp4
        comparison_video.mp4
        PIPELINE_SUMMARY.txt

No PDF, no written report — every file here is a real, direct output of
the pipeline for that clip. Nothing summarised or fabricated.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.builder import build_pipeline
from src.classifier.shot_predictor import ShotPredictor
from src.quality.scorer import score_shot

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_ROOT = ROOT / "Results"
SHOT_CLASSES = {
    "cover": 0, "defense": 1, "flick": 2, "hook": 3, "late_cut": 4,
    "lofted": 5, "pull": 6, "square_cut": 7, "straight": 8, "sweep": 9,
}
IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}

predictor = ShotPredictor()
print(f"Predictor available: {predictor.available}")

if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
OUT_ROOT.mkdir()

for cls in sorted(SHOT_CLASSES):
    clip = DATA / cls / "video1.mp4"
    if not clip.exists():
        print(f"SKIP {cls}: video1.mp4 not found")
        continue

    clip_name = f"{cls}_video1"
    raw_dir = OUT_ROOT / f"_raw_{clip_name}"
    print(f"\n=== {clip_name} ===")
    result = build_pipeline(
        str(clip), raw_dir,
        predictor=predictor,
        classifier=None,
        idx_to_class=IDX_TO_CLASS,
        scorer=score_shot,
        max_frames=30,
        display_name=f"{clip_name}.mp4",
    )
    pred = result.get("prediction") or {}
    print(f"  predicted: {pred.get('shot')} ({pred.get('confidence')}%)")

    # ── reorganise into folder-per-step ─────────────────────────────────
    final_dir = OUT_ROOT / clip_name
    final_dir.mkdir()

    def move_as_folder(src_name, folder_name, inner_name=None):
        src = raw_dir / src_name
        if not src.exists():
            return
        dst_folder = final_dir / folder_name
        dst_folder.mkdir()
        if src.is_dir():
            for f in src.iterdir():
                shutil.move(str(f), str(dst_folder / f.name))
        else:
            shutil.move(str(src), str(dst_folder / (inner_name or src_name)))

    move_as_folder("00_metadata.json", "00_metadata", "metadata.json")
    move_as_folder("01_extracted_frames", "01_extracted_frames")
    move_as_folder("02_skeleton_keypoints.json", "02_skeleton_keypoints", "skeleton_keypoints.json")
    move_as_folder("03_skeleton_overlay_frames", "03_skeleton_overlay_frames")
    move_as_folder("04_comparison_frames", "04_comparison_frames")
    move_as_folder("05_shot_analysis.json", "05_shot_analysis", "shot_analysis.json")

    for extra in raw_dir.glob("*.mp4"):
        name = "skeleton_video.mp4" if extra.name.startswith("skeleton_") else "comparison_video.mp4"
        shutil.move(str(extra), str(final_dir / name))
    summary = raw_dir / "PIPELINE_SUMMARY.txt"
    if summary.exists():
        shutil.move(str(summary), str(final_dir / "PIPELINE_SUMMARY.txt"))

    shutil.rmtree(raw_dir, ignore_errors=True)

print(f"\nDone. Results written to: {OUT_ROOT}")
