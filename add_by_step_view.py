"""
add_by_step_view.py
Takes the already-built Results/<clip>/<step>/... folders and adds a
second, step-first view alongside them:

    Results/by_step/<step>/<clip>/...          (frame folders, per clip)
    Results/by_step/<step>/<clip>_<file>.json  (single JSON files)

The original Results/<clip>/<step>/... layout (by_clip) is left exactly
as it is — this only adds copies under by_step, nothing is moved or
deleted from the per-clip view.
"""

import shutil
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "Results"
BY_CLIP = RESULTS / "by_clip"
BY_STEP = RESULTS / "by_step"

STEPS = [
    "00_metadata", "01_extracted_frames", "02_skeleton_keypoints",
    "03_skeleton_overlay_frames", "04_comparison_frames", "05_shot_analysis",
]

# move the already-built per-clip folders under Results/by_clip/
BY_CLIP.mkdir(exist_ok=True)
clip_dirs = sorted(p for p in RESULTS.iterdir()
                    if p.is_dir() and p.name not in ("by_clip", "by_step")
                    and not p.name.startswith("_raw_"))
for c in clip_dirs:
    shutil.move(str(c), str(BY_CLIP / c.name))

# build the step-first view
if BY_STEP.exists():
    shutil.rmtree(BY_STEP)
BY_STEP.mkdir()

for step in STEPS:
    step_dir = BY_STEP / step
    step_dir.mkdir()

for clip_dir in sorted(BY_CLIP.iterdir()):
    clip_name = clip_dir.name
    for step in STEPS:
        src = clip_dir / step
        if not src.exists():
            continue
        files = list(src.iterdir())
        if len(files) == 1 and files[0].suffix == ".json":
            # single JSON file -> flatten with clip name prefix
            shutil.copy2(files[0], BY_STEP / step / f"{clip_name}.json")
        else:
            # frame folder -> copy as its own clip subfolder
            shutil.copytree(src, BY_STEP / step / clip_name)
    # also copy the top-level extras (summary, videos) into a matching spot
    extras_dir = BY_STEP / "extras" / clip_name
    extras_dir.mkdir(parents=True, exist_ok=True)
    for extra_name in ("PIPELINE_SUMMARY.txt", "skeleton_video.mp4", "comparison_video.mp4"):
        extra = clip_dir / extra_name
        if extra.exists():
            shutil.copy2(extra, extras_dir / extra_name)

print("Done.")
print(f"  {BY_CLIP}  (clip -> step)")
print(f"  {BY_STEP}  (step -> clip)")
