# Quick Reference

Last updated: September 2026

A one-page cheat sheet for the project's real, current numbers and
methods. For full detail and proofs, see
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).

## Striker identification

Not size/position filtering on a single frame. YOLOv8 detects every
person in the clip; BoT-SORT tracks each one across all frames. The
striker's track is picked using: total frames the track persists, box
height, horizontal position relative to the crease area, and box aspect
ratio, with an explicit rule that rejects the wicketkeeper's track even
when it is tall and central. Verified on 416 sampled frames across all
10 shot classes: **0 wrong-player frames**.

## Accuracy numbers (all measured, not estimated)

| Metric | Value |
|---|---|
| Shot classification, top-1 | 62.4% (250-clip held-out test set) |
| Shot classification, top-3 | 81.2% |
| Original inherited claim | 94% — did not hold up under a leak-free re-measurement; real number was 57.6% |
| Striker identification | 0/416 wrong-player frames |
| Pose detection | 100% on 9 of 10 reference clips |
| Quality score, pro sweep clip | 10.2 → 76.0 / 100 after fixing the scoring pipeline |
| Inference speed | ~25s per 8-second clip, CPU only |

## Known confusion pattern (real, from the confusion matrix)

The classifier over-predicts `flick` for a specific set of shots:
`defense → flick` (13 of 25 misclassified defense clips), `pull → flick`
(8 clips), `square_cut → flick` (7 clips). Tested and ruled out as a
class-imbalance problem (the training set is exactly 125 clips per
class — perfectly balanced, confirmed programmatically; class-weighted
retraining changed accuracy by 0.0 points). This is a feature
confusability problem between visually similar bat-swing motions, not a
data problem, and none of the seven accuracy techniques tried fixed it.

## Joint angles

Computed from MediaPipe's **metric 3D world landmarks**, not flat 2D
image coordinates. 2D angles were tried first and read 155-177 degrees
for nearly every shot on this camera angle — the camera's viewing angle
flattens true joint geometry in 2D. 3D angles separate cleanly by shot:
sweep ≈ 127°, defense ≈ 51°, hook ≈ 168° at the front knee.

## Impact & shot-start detection

- **Impact frame**: the first local peak in wrist speed that reaches at
  least 50% of the clip's overall peak speed (not the global maximum,
  which is often follow-through/recovery motion after the actual shot).
- **Shot-start frame**: the nearest local minimum in wrist speed before
  impact that drops below a quiet-motion threshold (tuned to 0.15 by
  checking actual extracted video frames, not just the numbers).

## Movement graph ("You vs Professionals")

Each of 7 tracked joint angles is resampled onto a fixed 25-point
timeline running from shot-start to impact, and plotted against an
interquartile band built the same way from 51-59 professional clips per
shot class. This replaced the earlier single-snapshot-at-impact
comparison.

## Seven accuracy techniques tried, after the 62.4% baseline

All flat or negative — full detail in `PROGRESS_REPORT.md` §11.

| Technique | Result |
|---|---|
| Pooled skeleton fusion (4 variants) | -4.0 to +3.6 pts |
| ST-GCN skeleton-graph fusion | -0.8 pts |
| Class-weighted retrain | +0.0 pts |
| Ensembling (2 seeds) | +0.0 pts |
| Multi-window inference | -3.6 pts |
| Partial EfficientNetB0 fine-tune | -7.2 pts (overfit) |

**Conclusion**: the 1250-clip dataset is the binding constraint, not
architecture.

## Shot classes

`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`,
`square_cut`, `straight`, `sweep`

## Honest limitations

- Quality scoring depends on the predicted shot being correct — a wrong
  prediction grades against the wrong reference. Warning shown below 60%
  confidence.
- One camera angle only (standard broadcast, behind the bowler's arm).
- `late_cut`, `square_cut`, `lofted`, `straight` still use a generic
  quality rule, not a shot-specific one.
- The bundled 50-clip `data/` demo set is never used for a reported
  accuracy number — it overlaps too heavily with training data.
