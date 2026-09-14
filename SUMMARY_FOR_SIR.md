# Summary for Sir

Last updated: September 2026

## What the project does

Upload a cricket batting video, filmed from the standard broadcast angle
behind the bowler's arm. The app finds the striker, tracks their pose
through the shot, works out what shot they played, and scores their
technique against professional players — both at the moment of impact
and across the whole shot as a movement curve. Everything runs in a
Streamlit web app, no wearable sensors, no GPU.

## Where we started and what was wrong

We began from an existing open-source pipeline. It had three real
problems:

1. It sometimes drew the pose skeleton on the wrong person — the
   umpire, the non-striker, or the wicketkeeper — because it picked
   whoever's bounding box happened to be biggest or most central in a
   single frame.
2. Joint angles were averaged across the entire clip, producing
   physically impossible poses, like a 178-degree knee bend at the same
   time as a 16-degree elbow bend, because the "impact moment" was never
   actually located.
3. Angles were measured in flat 2D image coordinates. On this camera
   angle, that collapses almost every shot to 155-177 degrees regardless
   of what the batsman actually did, since the camera's viewing angle
   flattens the true joint geometry.

## What we changed, and why

- **Striker identification**: added a dedicated YOLOv8 detection +
  BoT-SORT tracking stage that follows every person across the clip and
  picks the striker's track using height, position, and how long each
  track persists, with an explicit rule to reject the keeper. Verified
  on 416 sampled frames across all 10 shot classes: 0 wrong-player
  frames.
- **Impact detection**: instead of averaging the whole clip, we locate
  the actual impact frame from the first prominent peak in wrist speed,
  and read angles only at that frame.
- **3D angles**: switched from 2D image-plane landmarks to MediaPipe's
  metric 3D world landmarks for every angle calculation. Angles now
  separate cleanly by shot type — sweep reads around 127 degrees,
  defense around 51, hook around 168 — instead of collapsing to the same
  range every time.
- **Handedness**: rather than hard-coding a right-handed batsman, we
  vote across all frames using top-hand grip height and shoulder depth,
  so left-handers and mirrored footage score correctly.
- **Movement graph ("you vs professionals")**: originally the app
  compared technique only at the single impact instant. Per your
  feedback, we now resample each angle onto a 25-point timeline running
  from shot-start to impact, and plot it against a band built from
  professional clips the same way, so the comparison shows the whole
  shot's movement, not one snapshot.
- **Accuracy claim correction**: the pipeline we inherited claimed 94%
  shot-classification accuracy. Re-measuring it under a leak-free
  protocol, after removing corrupted dataset files and an overlapping
  demo set, showed the real number was 57.6%. We rebuilt the classifier
  as a fusion of R3D-18 (motion) and EfficientNetB0 (appearance) and
  reached 62.4% top-1, 81.2% top-3, on a genuinely held-out 250-clip
  test split.

## What we tried after that, to push accuracy further

Once the 62.4% baseline was solid, we tested whether adding skeleton
information could raise it further, since that was the original plan.
We tried seven separate techniques:

1. Four different ways of pooling per-frame skeleton features into the
   classifier — results ranged from -4.0 to +3.6 points.
2. A proper skeleton-graph network (ST-GCN), which models the joints and
   their physical connections as a graph rather than a flat feature
   vector. Standalone, it reached 17.7% (a real improvement from a
   broken first attempt that scored 10%, caused by undetected frames
   being read as valid zero-value poses — traced and fixed). Fused with
   the main classifier, it cost 0.8 points rather than adding anything.
3. Class-weighted retraining — no change, because the training set
   turned out to already be perfectly balanced at 125 clips per class.
4. Ensembling two independently trained models — no change.
5. Averaging predictions across multiple time windows of each clip —
   this actually hurt accuracy by 3.6 points.
6. Partially unfreezing and fine-tuning the EfficientNetB0 backbone —
   this overfit badly (83.2% training accuracy against 52-53% validation
   accuracy) and lost 7.2 points.

Every one of the seven came back flat or negative. Taken together, they
point at the same conclusion: the training set, 1250 clips across 10
classes, is the actual limit on accuracy, not the model architecture.
Given more labeled clips, the fusion approach would likely help; on the
current dataset size, it does not.

## Honest state of the system today

- Shot classification: 62.4% top-1, 81.2% top-3.
- Since quality scoring depends on which shot was predicted, a wrong
  prediction grades the batsman against the wrong shot's reference. The
  app shows a low-confidence warning below 60%.
- Works reliably on vertical crops, zoomed footage, letterboxed video,
  and lower-resolution re-uploads — verified directly, not assumed.
- Runs in about 25 seconds per 8-second clip on CPU.
- Four of the ten shot types (late cut, square cut, lofted, straight)
  still use a generic quality rule rather than a shot-specific one,
  since we did not have professional reference clips for the finer
  distinctions there.

Full write-up, every defect found with before/after evidence, and the
complete accuracy investigation are in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).
