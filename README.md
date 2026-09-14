# 🏏 Cricket Shot Quality AI

> **BTP Project (ITIT-3999) — "Quantifying the Quality of a Cricket Batting Shot Using AI"**
> ABV-IIITM Gwalior · Vipul Pandey (2023IMG-056), Avinav Verma (2023IMG-011), Shubham Raj (2023IMT-076)
> Supervisor: Dr. Anshul · Built on [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) (MIT License)

Upload a cricket batting clip, filmed from the standard broadcast angle
(behind the bowler's arm), and the app returns three things without any
wearable sensor: the shot that was played, a skeleton track of the batsman
through the shot, and a technique score measured against professional
players — both at the moment of impact and across the whole shot as a
movement curve.

## Current State

This is a working, end-to-end Streamlit app, not a prototype. Every number
below is measured on the held-out test split, not estimated.

| | |
|---|---|
| Shot classification | **62.4%** top-1, **81.2%** top-3 (250-clip held-out test split, 10 classes) |
| Striker identification | **0 wrong-player frames** out of 416 checked, across all 10 shot classes |
| Pose detection | **100%** on 9 of 10 reference clips |
| Quality score (pro sweep clip) | **10.2 → 76.0** out of 100, after fixing the measurement pipeline |
| You-vs-professionals | Whole-shot movement curve (start → impact), not a single snapshot |
| Robustness | Works on vertical crops, 1.6× zoom, letterboxing, 480p re-uploads |
| Speed | ~25s for an 8-second clip, no GPU used anywhere |

**An earlier claim of 94% accuracy did not hold up.** Re-measured under a
leak-free protocol (corrupted dataset files removed, no reuse of clips
between train/val/test), the actual shipped-then model scored 57.6%. The
current classifier — a fused R3D-18 (motion) + EfficientNetB0 (appearance)
backbone — reaches 62.4%. Seven further techniques were tried to push this
higher (a skeleton graph network, class-weighted retraining, ensembling,
multi-window inference, partial backbone fine-tuning); **none of them
beat this baseline**, and the consistent pattern across all seven points at
the 1250-clip dataset size as the real limit, not the architecture. See
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) §11 for the full, honest
breakdown of what was tried and why each one failed.

## What Actually Changed (and why)

| Problem found | Fix | Result |
|---|---|---|
| Skeleton drawn on the wrong person (umpire, non-striker, keeper) | Dedicated YOLOv8 detection + BoT-SORT tracking stage picks the striker once per clip, *then* MediaPipe pose runs on that crop | 0/416 wrong-player frames |
| Joint angles averaged across the whole clip → impossible poses (178° knee + 16° elbow at once) | Angles read at the actual impact frame, located from wrist-speed peaks | Physically real poses at every phase |
| Angles measured in 2D image coordinates → every shot read 155–177° regardless of what happened | Angles computed from MediaPipe's metric 3D world landmarks | Sweep 127°, defense 51°, hook 168° — angles finally separate by shot |
| "Front leg" hard-coded to right-handed | Handedness detected per clip by voting on top-hand grip + shoulder depth | Left-handers and mirrored clips scored correctly |
| Technique judged only at the impact instant | Each angle resampled onto a 25-point timeline from shot-start to impact, plotted against a professional band built the same way | Whole-shot movement comparison, not one number |
| 94% accuracy claim, unverified | Re-measured on a leak-free split after removing corrupted files and an overlapping demo set | 57.6% (honest baseline) → 62.4% (fused backbone, shipped) |

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
evaluate_multiwindow.py, finetune_effnet.py  Further accuracy experiments (all negative/flat — see PROGRESS_REPORT.md)
reports/                        Generated technical/summary PDFs of the accuracy investigation
data/<class>/video{1-5}.mp4     50-clip demo set (never used for reported accuracy — see below)
```

## Shot Classes

`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`,
`square_cut`, `straight`, `sweep`

## Honest Limitations

- **Shot prediction caps the whole system at 62.4%.** Since quality scoring
  picks its criteria from the predicted shot, a wrong prediction grades a
  correct shot against the wrong reference. The app warns below 60%
  confidence.
- **One camera angle only** — the standard broadcast view from behind the
  bowler. Other angles return no result rather than a guess.
- **The 50-clip demo bundle (`data/`) is never used for a reported
  accuracy number.** It overlaps heavily with the training set (feature
  similarity 0.998 vs. 0.875 to the real test split, including one exact
  duplicate) — using it would silently inflate the number.
- **Four of ten shot types** (`late_cut`, `square_cut`, `lofted`,
  `straight`) still fall back to a generic quality rule rather than a
  shot-specific one.

See [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) for the full write-up —
every defect found, the fix, the measurement, and the complete accuracy
investigation including what didn't work and why.

## Credits

- Base model: [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) — MIT License
- Dataset: CricShotClassify / CrickShot10 — [Sen et al., Sensors 2021](https://doi.org/10.3390/s21082846)
