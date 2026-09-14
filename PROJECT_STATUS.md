# Project Status

Last updated: September 2026

## Where the project stands

The app is complete and working end to end. Upload a batting clip, filmed
from the standard broadcast angle, and it returns the predicted shot, a
skeleton overlay on the correct player, joint angles at impact, a
whole-shot movement curve compared against professionals, and a quality
score — all inside a Streamlit web page, no GPU required.

This document used to describe a planned RGB+Skeleton attention-fusion
architecture as the "next step." That plan was tried in several forms
(a pooled per-frame skeleton branch, then a proper ST-GCN skeleton-graph
branch) and none of them beat the plain video-only classifier. The
sections below describe what was actually built and measured, not what
was originally planned.

## Pipeline (as shipped)

1. **Person detection & tracking** — YOLOv8 detects every person in the
   frame; BoT-SORT tracks them across frames. One track is chosen as the
   striker per clip, using track length, box height, horizontal position,
   and aspect ratio, with an explicit rule to reject the wicketkeeper.
2. **Pose estimation** — MediaPipe Pose (BlazePose) runs on the striker's
   crop, producing 33 landmarks, reduced to 13 cricket-relevant joints.
3. **Handedness** — decided by voting across all frames (top-hand wrist
   height + shoulder depth), not assumed.
4. **Impact & shot-start frames** — found from wrist-speed: impact is the
   first prominent local speed peak reaching at least half the clip's
   overall peak; shot-start is the nearest quiet local minimum before it.
5. **Joint angles** — computed from MediaPipe's metric 3D world
   landmarks. 2D image-plane angles were tried first and collapsed to
   155-177 degrees for every shot on this camera angle, so all angle
   math uses the 3D landmarks instead.
6. **Shot classification** — R3D-18 (motion, pretrained on Kinetics-400)
   and EfficientNetB0 (appearance, pretrained on ImageNet), both frozen,
   fused through a BiGRU with attention pooling. 62.4% top-1 / 81.2%
   top-3 on a 250-clip held-out test split.
7. **Quality scoring** — at impact, each angle is compared to an
   interquartile range built from 61-65 professional clips per shot type
   and scored 100 / 60-99 / 0-59 depending on distance from the band.
   Across the whole shot, the same angles are resampled onto a 25-point
   timeline from shot-start to impact and plotted against a professional
   band built the same way.

## What was tried to push classification accuracy higher, and the result

The original plan was to fuse pose/skeleton information into the
classifier, expecting a meaningful accuracy gain. Seven different
approaches were tried and measured; every one of them was flat or
negative against the 62.4% baseline:

| Approach | Result vs. baseline |
|---|---|
| Pooled per-frame skeleton features, fused (4 variants) | -4.0 to +3.6 points |
| ST-GCN (skeleton graph convolution network), fused | -0.8 points |
| Class-weighted retraining | +0.0 (dataset was already perfectly balanced, 125 clips/class) |
| Ensembling two independently seeded models | +0.0 |
| Multi-window inference (averaging predictions over several clip windows) | -3.6 points (real loss) |
| Partial EfficientNetB0 backbone fine-tuning | -7.2 points (overfit: 83.2% train vs. 52-53% val) |

The consistent conclusion across all seven independent attempts: the
1250-clip training set is the binding constraint on accuracy, not the
model architecture. More data would very likely help; none of the
architectural changes tried did. Full experiment details, methodology,
and honest failure analysis for each are in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) (Section 11).

## Files referenced by old plans that are not part of the shipped pipeline

`download_dataset.py`, `convert_to_mp4.py`, `view_skeleton.py`,
`test_skeleton_only.py`, `test_pipeline.py`,
`src/pose/skeleton_extractor.py`, and `src/classifier/fusion_model.py`
are early exploratory scripts from before the striker-tracking and
3D-angle fixes landed. They still exist in the repo but are not called
by `app.py` or by any of the current training/derivation scripts. They
are kept for reference, not deleted, since they document earlier
approaches.

## What's next

No further architecture changes are planned given the evidence above.
The remaining honest limitations (shot-prediction accuracy as the system
ceiling, single camera angle, four shot types still on a generic quality
rule) are listed in [`README.md`](README.md) and detailed in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).
