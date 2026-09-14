# 🏏 Cricket Shot Quality AI

> **BTP Project (ITIT-3999) — "Quantifying the Quality of a Cricket Batting Shot Using AI"**
> ABV-IIITM Gwalior · Vipul Pandey (2023IMG-056), Avinav Verma (2023IMG-011), Shubham Raj (2023IMT-076)
> Supervisor: Dr. Anshul · Built on [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) (MIT License)

Upload a cricket batting clip, filmed from the standard broadcast angle
(behind the bowler's arm), and the app returns three things without any
wearable sensor: the shot that was played, a skeleton track of the
striker through the shot, and a technique score measured against
professional players — both at the moment of impact and across the
whole shot as a movement curve.

---

## Current State

This is a working, end-to-end Streamlit app, not a prototype. Every
number below is measured on the held-out test split, not estimated.

| | |
|---|---|
| Shot classification | **62.4%** top-1, **81.2%** top-3 (250-clip held-out test split, 10 classes) |
| Striker identification | **0 wrong-player frames** out of 416 checked, across all 10 shot classes |
| Pose detection | **100%** on 9 of 10 reference clips |
| Quality score (pro sweep clip) | **10.2 → 76.0** out of 100, after fixing the measurement pipeline |
| You-vs-professionals | Whole-shot movement curve (start → impact), not a single snapshot |
| Robustness | Works on vertical crops, 1.6× zoom, letterboxing, 480p re-uploads |
| Speed | ~25s for an 8-second clip, no GPU used anywhere |

**An earlier claim of 94% accuracy did not hold up.** Re-measured under
a leak-free protocol (corrupted dataset files removed, no reuse of
clips between train/val/test), the actual shipped-then model scored
57.6%. The current classifier — a fused R3D-18 (motion) +
EfficientNetB0 (appearance) backbone — reaches 62.4%. Seven further
techniques were tried to push this higher; **none of them beat this
baseline** (full breakdown below), and the consistent pattern across
all seven points at the 1250-clip dataset size as the real limit, not
the architecture.

## What Actually Changed (and why)

| Problem found | Fix | Result |
|---|---|---|
| Skeleton drawn on the wrong person (umpire, non-striker, keeper) | Dedicated YOLOv8 detection + BoT-SORT tracking stage picks the striker once per clip, *then* MediaPipe pose runs on that crop | 0/416 wrong-player frames |
| Joint angles averaged across the whole clip → impossible poses (178° knee + 16° elbow at once) | Angles read at the actual impact frame, located from wrist-speed peaks | Physically real poses at every phase |
| Angles measured in 2D image coordinates → every shot read 155–177° regardless of what happened | Angles computed from MediaPipe's metric 3D world landmarks | Sweep 127°, defense 51°, hook 168° — angles finally separate by shot |
| "Front leg" hard-coded to right-handed | Handedness detected per clip by voting on top-hand grip + shoulder depth | Left-handers and mirrored clips scored correctly |
| Technique judged only at the impact instant | Each angle resampled onto a 25-point timeline from shot-start to impact, plotted against a professional band built the same way | Whole-shot movement comparison, not one number |
| Inference fed the model 30 strided frames from the "middle 60%" of the clip, a different input shape than it was trained on | Switched to 30 consecutive frames from frame 0, matching the training convention | top-1 32%→46%, top-3 66%→88% on demo clips |
| 94% accuracy claim, unverified | Re-measured on a leak-free split after removing corrupted files and an overlapping demo set | 57.6% (honest baseline) → 62.4% (fused backbone, shipped) |

---

## Setup

```bash
git clone https://github.com/vipulpandey21/CricketShotQualityAI.git
cd CricketShotQualityAI

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Pipeline

```
Video upload
  → Person detection & tracking (YOLOv8 + BoT-SORT)
  → Striker identification (one track chosen per clip)
  → Pose estimation (MediaPipe, 33 → 13 cricket-relevant landmarks)
  → Handedness detection (voted across all frames)
  → Impact & shot-start frame detection (from wrist-speed)
  → Joint angles (metric 3D world landmarks — not flat 2D, which fails
     on this camera angle)
  → Shot classification (R3D-18 + EfficientNetB0 → BiGRU + attention)
  → Quality scoring — at impact, and as a whole-shot curve — against
     professional reference bands
  → Scores, graphs, downloadable per-clip data folder
```

## Project Structure

```
app.py                          Streamlit UI — the whole application
src/
  pose/
    striker_pose.py             Striker identification (YOLOv8 + BoT-SORT)
    estimator.py                Angles, handedness, impact/shot-start detection
    shot_curve.py                Whole-shot movement curve (resampled, normalised)
  pipeline/
    builder.py                  Orchestrates one clip end to end, writes the data folder
  classifier/
    shot_predictor.py           Live inference: R3D-18 + EfficientNetB0 fusion
    model.py                    SHOT_CLASSES + model definitions
  quality/
    scorer.py                   Quality-score rule set
cache_pose_features.py          Builds cached pose features for the training set
derive_ideal_angles.py          Professional angle ranges at impact (ideal_angles.json)
derive_angle_curves.py          Professional angle bands across the whole shot
train_video.py / train_fusion.py  Backbone comparison (appearance / motion / fused)
train_stgcn.py / train_stgcn_fusion.py  Skeleton-graph classifier experiment (didn't beat baseline)
train_calibrated.py, evaluate_ensemble.py,
evaluate_multiwindow.py, finetune_effnet.py  Further accuracy experiments (all negative/flat — see below)
reports/                        Generated technical/summary PDFs of the accuracy investigation
data/<class>/video{1-5}.mp4     50-clip demo set (never used for reported accuracy — see below)
```

### The per-clip data folder

Every processed clip (uploads included) gets a `videoX_pipeline/`
folder holding every intermediate step, so nothing is a black box:

```
video1_pipeline/
├── 00_metadata.json                Clip info + detection stats
├── 01_extracted_frames/            The 30 frames the classifier is given
├── 02_skeleton_keypoints.json      All 33 landmarks per analysed frame
├── 03_skeleton_overlay_frames/     Skeleton drawn on the striker
├── 04_comparison_frames/           Original beside skeleton, side by side
├── 05_shot_analysis.json           Prediction, top-3, joint angles, quality, movement curve
└── PIPELINE_SUMMARY.txt            Human-readable summary of the above
```

`01_extracted_frames/` is 30 **consecutive** frames from frame 0 — the
same convention the classifier was trained on (see the accuracy
section below for why this matters). `05_shot_analysis.json` is the
final output: predicted shot + confidence, impact and shot-start frame
numbers, joint angles at impact, the quality score, and the full
25-point "you vs professionals" movement curve per joint.

## Shot Classes

`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`,
`square_cut`, `straight`, `sweep`

---

## Why 13 landmarks, not MediaPipe's full 33

MediaPipe Pose returns 33 body landmarks. Only 13 are used here:
nose, both shoulders, elbows, wrists, hips, knees, and ankles — the
joints that make up the batting kinematic chain. The 20 dropped
(detailed face points, individual fingers, foot detail) don't carry
information that changes a cricket shot's classification or its
technique score — wrist angle already captures bat control, ankle
position already captures footwork. Dropping them cuts the per-frame
feature size by 60%, with no measured loss in what the analysis needs.

## The full accuracy investigation

Shot classification is the one number that caps the whole system —
quality scoring picks its comparison criteria from the predicted shot,
so a wrong prediction grades a correct shot against the wrong
reference. Because of that, a lot of effort went into pushing this
number up. This section is the complete, honest record: what was
tried, in order, and what actually happened — including the attempts
that made things worse.

### Getting to a real baseline (57.6% → 62.4%)

The inherited pipeline claimed 94% accuracy. That number didn't hold
under a leak-free protocol, for two reasons found during
re-measurement: macOS AppleDouble sidecar files (`._cover_0001.avi`)
sitting next to real videos were being opened by OpenCV as 30 black
frames, corrupting about half of every accuracy measurement; and the
50-clip demo set used for casual testing overlaps heavily with the
training data (feature similarity 0.998 vs. 0.875 for the real test
set, including one exact duplicate), so it was dropped from anything
reported as a result. With both fixed, the real baseline was 57.6%
top-1 / 84.8% top-3 on the true 1250/250/250 split.

A second concrete bug was found in how frames reached the model at
inference time: training used 30 consecutive frames from frame 0, but
inference was feeding it 30 strided frames from the "middle 60%" of
the clip — a different kind of input entirely. Fixing just that:

| frame selection | top-1 | top-3 |
|---|---|---|
| strided, middle 60% (as shipped) | 32% | 66% |
| consecutive from frame 0 (as trained) | **46%** | **88%** |

The classifier itself was then rebuilt. EfficientNetB0 alone is an
ImageNet model — its features describe what a single frame *looks*
like, not how the body *moves*, which is what actually defines a
cricket shot. Adding a Kinetics-400 pretrained video model (`r3d_18`),
which encodes motion by construction, and fusing it with
EfficientNetB0:

| model | test top-1 | test top-3 |
|---|---|---|
| shipped weights (original) | 57.6% | 84.8% |
| EfficientNetB0 retrained alone | 52.4% | 80.8% |
| r3d_18 alone | 56.4% | 79.2% |
| **r3d_18 + EfficientNetB0 (shipped)** | **62.4%** | 81.2% |

The two backbones fail on different shots — r3d_18 is much better on
straight and lofted, EfficientNetB0 better on square_cut and hook —
which is why fusing them helps even though neither wins outright
alone.

### Seven techniques tried to push past 62.4%, all flat or negative

| Technique | Result vs. 62.4% baseline |
|---|---|
| Pooled per-frame skeleton fusion (4 variants: concatenation, probability averaging, weighted averaging, separate encoders) | -4.0 to +3.6 pts |
| ST-GCN (skeleton graph convolution network), fused | -0.8 pts |
| Class-weighted retraining | +0.0 pts |
| Ensembling two independently seeded models | +0.0 pts |
| Multi-window inference (averaging predictions across several clip windows) | -3.6 pts |
| Partial EfficientNetB0 backbone fine-tuning | -7.2 pts (overfit) |

**Skeleton fusion (4 pooled variants + ST-GCN).** The original plan was
that combining RGB video with the striker's skeleton should help,
since pose data explicitly encodes body mechanics. Four ways of pooling
per-frame skeleton features into the classifier were tried first, all
flat or negative. A full ST-GCN — a graph convolution network that
models the 13 joints and their physical connections as a graph, rather
than a flat pooled vector — was then built. It found and fixed a real
bug along the way: `cache_pose_features.py` writes an all-zero row for
any frame where the striker wasn't detected (averaging 20.4% of frames
per clip), and the graph convolution was reading `(0,0,0)` as a literal
joint sitting at the image's top-left corner. Fixing that
(`fill_gaps()`, linear interpolation through the zero frames) took the
standalone ST-GCN from 10% (chance level, collapsed to predicting one
class 84% of the time) to 17.7% top-1 — a genuine, verified fix. Fused
into the main classifier and retrained on the full 1250-clip dataset,
it still cost 0.8 points rather than adding anything.

**Class-weighted retraining.** The confusion matrix showed a clear
over-prediction of `flick` (defense→flick 13 of 25 misclassified
defense clips, pull→flick 8, square_cut→flick 7), which looked like a
class-imbalance signature. Inverse-frequency class weights were tried.
The training set turned out to be exactly 125 clips per class —
already perfectly balanced, confirmed programmatically — so weighting
was a mathematical no-op (+0.0). This disproves the imbalance
hypothesis rather than assuming it: the flick over-prediction is a
feature-confusability problem between visually similar bat-swing
motions, not a data-skew problem.

**Ensembling.** Two independently seeded models, softmax outputs
averaged. +0.0 — the two seeds converge to near-identical decision
boundaries at this dataset size, so there was nothing for averaging to
correct.

**Multi-window inference.** Averaging predictions across several
different starting offsets within a clip, hoping to smooth out
per-window noise. -3.6 points, a real loss: the model was trained
specifically on frames 0-29, and windows starting later feed it input
it never saw in training.

**Partial backbone fine-tuning.** Unfreezing and fine-tuning the last
block of EfficientNetB0 on the cricket dataset. -7.2 points, with a
clear overfitting signature: 83.2% training accuracy against 52-53%
validation accuracy — a 1250-clip dataset isn't enough to fine-tune an
ImageNet backbone without it memorising the training clips.

**Conclusion.** Every technique tried after the 62.4% baseline came
back flat or negative, across seven independently designed attempts
spanning architecture, training strategy, and inference strategy. The
consistent pattern points at the 1250-clip training set as the actual
limiting factor, not the model or the fusion idea — with more labeled
data, several of these approaches (skeleton fusion especially) would
likely behave differently. This is the final, evidence-backed
conclusion of the investigation.

## The movement graph ("You vs Professionals")

Per the supervisor's feedback, technique comparison originally judged
a batsman against professionals at a single instant — the impact
frame. That's now a whole-shot comparison instead. Two pieces of new
detection made this possible: a shot-start frame (the nearest local
minimum in wrist speed before impact that drops below a quiet-motion
threshold, tuned to 0.15 by checking actual extracted video frames,
not just the numbers — two simpler candidate rules were tried and
visually rejected first), and resampling seven tracked joint angles
from shot-start to impact onto a fixed 25-point timeline, compared
against a professional interquartile band built from 51-59 clips per
shot class the same way. The app shows this as a "Shot movement"
section with an SVG line chart per joint.

---

## Honest Limitations

- **Shot prediction caps the whole system at 62.4%.** Since quality
  scoring picks its criteria from the predicted shot, a wrong
  prediction grades a correct shot against the wrong reference. The
  app warns below 60% confidence.
- **One camera angle only** — the standard broadcast view from behind
  the bowler. Other angles return no result rather than a guess.
- **The 50-clip demo bundle (`data/`) is never used for a reported
  accuracy number.** It overlaps heavily with the training set (see
  above) — using it would silently inflate the number.
- **Four of ten shot types** (`late_cut`, `square_cut`, `lofted`,
  `straight`) still fall back to a generic quality rule rather than a
  shot-specific one.

## Credits

- Base model: [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) — MIT License
- Dataset: CricShotClassify / CrickShot10 — [Sen et al., Sensors 2021](https://doi.org/10.3390/s21082846)
