# Progress Report — Cricket Shot Quality Analyser

What the system did before this round of work, what was wrong with it, what
was changed, and what the measured result is. Every number here was measured,
not estimated; where something is still not good enough, that is stated.

---

## 1. Starting point

The project already had:

- A shot classifier (EfficientNetB0 + GRU) trained on the CrickShot10
  dataset, claimed at 94% accuracy in the README
- MediaPipe pose estimation producing a skeleton overlay
- Joint angle computation and a 0–100 quality score
- A Streamlit web app for uploading a clip
- 50 demo clips in `data/` (10 shot types × 5)

The intended flow was: upload a video → shot prediction → pose analysis →
quality score.

---

## 2. Problem 1 — the skeleton was on the wrong person

**What was found.** Inspecting the saved overlay frames showed the skeleton
was being drawn on the **non-striker** (the batsman standing at the bowler's
end), and in other clips on the umpire or wicketkeeper. The striker — the
batsman actually playing the shot — had no skeleton at all.

**Why the old approach could not work.** It ran MediaPipe on the full
1280×720 broadcast frame and then scored the detected people to guess which
was the batsman. But the broadcast camera sits behind the bowler's arm, so the
striker is only ~120 px tall. MediaPipe does not detect a person that small,
so the striker was never in the candidate list. No amount of adjusting the
scoring could pick someone who was never detected.

**What was built instead** (`src/pose/striker_pose.py`):

1. A dedicated person detector (YOLOv8 + BoT-SORT tracking) finds *every*
   person in the frame, including the small far-end striker, and gives each a
   stable track ID.
2. The striker is chosen **once for the whole clip** at the track level, not
   per frame — so the choice cannot flicker between people mid-clip.
3. Pose then runs on an **upscaled crop of that one person**, which is what
   makes the 33 landmarks accurate on a small subject.

**How the striker is identified.** Two cricket facts, both measured from real
track statistics rather than guessed:

- The striker is the **topmost** qualifying player in frame. Everyone below
  is the bowler, the umpire, or the non-striker.
- The only person who can appear above the striker is the **wicketkeeper**,
  who is separated by two things that must *both* hold: he is clearly shorter
  on camera (further away, crouched), **and** he is at the same end of the
  pitch — measured as vertical gap ÷ striker height < 0.9. The non-striker is
  a full pitch length away and always exceeds that ratio:

  | pair | gap ÷ height |
  |---|---|
  | defense: keeper vs striker | 0.19 |
  | late_cut: keeper vs striker | 0.78 |
  | cover: striker vs non-striker | 1.05 |
  | sweep: striker vs non-striker | 1.11 |

**Four separate bugs were found and fixed along the way**, each caught by
looking at actual frames rather than by reasoning:

1. *MediaPipe cannot see the striker* — fixed by the detector + crop approach
   above.
2. *The keeper was selected as the striker* for the whole of
   `data/defense/video1.mp4`. An earlier version of the keeper test used
   horizontal overlap, assuming the keeper stands directly behind the stumps.
   That is not reliable — in this clip the keeper stood wide to the leg side,
   overlap measured 0.14, and the test never fired. Replaced with the
   height + same-end test above.
3. *The crop was square in normalised units, not pixels.* Because x and y are
   normalised by different pixel counts (1280 vs 720), a "square" crop was
   really a 16:9 rectangle, and resizing it to a square stretched the batsman
   ~1.8× vertically. MediaPipe is trained on undistorted people and simply
   failed. Fixed by sizing the crop in pixels.
4. *The skeleton landed on the keeper even when the box was correct.* The crop
   margin pulls in a neighbour, and asking MediaPipe for a single pose let it
   return whichever person it found most convincing. Fixed by taking three
   poses per crop and keeping the one whose **torso actually sits on the
   striker's box**.

**Frame selection also had to change.** The old code always analysed the
middle 60% of a clip. In `data/lofted/video1.mp4` the shot happens between 10%
and 18% and the middle is entirely crowd and a wide establishing shot — that
clip produced no pose data at all. The new code scans the whole clip and
analyses the stretch where the striker is actually on camera. That clip went
from 0% to 100%.

**Verification.** An automatic checker examined every analysed frame of all
ten `video1` clips for signs of the box being on the wrong person (a taller
person overlapping and at the same end):

```
cover 0/46  defense 0/44  flick 0/44  hook 0/39  late_cut 0/38
lofted 0/29 pull 0/57     square_cut 0/45  straight 0/34  sweep 0/56
```

**0 wrong-person frames out of 416.** Detection rate is 100% on 9 of 10
clips, with 10–12 of the 13 cricket joints found per frame.

---

## 3. Problem 2 — the analysis numbers were wrong

Two independent bugs, both of which made the quality score meaningless.

**Bug A — angles were averaged over the whole clip.** The code averaged every
joint position across all 30 frames and then measured angles on that average.
That is not a pose anyone ever held: a hook swings the body through a wide
arc, so a knee that is bent early and straight late averages to straight. On
`data/hook/video1.mp4` this produced a front knee of 178°, a back knee of
170°, and a **back elbow of 16°** — anatomically impossible — and scored an
international batsman 10/100.

Fixed by measuring at the **impact frame**, found as the frame of peak hand
speed (the hands travel fastest as the bat comes through the ball), with the
speed divided by torso size so it means the same thing at any zoom. Angles at
stance, impact, and follow-through are all reported.

**Bug B — angles were measured in 2D.** Even at the impact frame, every shot
reported a nearly straight knee. Measured across the dataset, the 2D
front-knee angle sat at 155–177° for sweep, hook, pull, and defense alike —
carrying no information at all. The reason is geometric: on this camera angle
the batsman's legs point roughly along the viewing direction, so hip, knee,
and ankle project almost collinear.

MediaPipe's image-landmark `z` is only a relative depth hint, and using it as
if it were metric was worse — it put the sweep front knee at 11°, which no
knee can do. The correct source is MediaPipe's **world landmarks**: metric 3D
in metres, hip-centred. With those, the same clips separate the way they
should:

| shot | front knee (2D, wrong) | front knee (world 3D) | is that right? |
|---|---|---|---|
| sweep | 145° | **84°** | yes — deeply crouched |
| defense | 150° | **154°** | yes — upright forward defence |
| hook | 149° | **132°** | yes |
| cover | 175° | **152°** | yes |

**An important correction.** Partway through, the quality rubric in
`src/quality/scorer.py` was suspected of being wrong, because for `hook` it
asks for elbows bent to 70–120° while one clip measured 179°. Deriving the
real distributions from ~63 professional clips per class showed the rubric's
ranges were **correct all along** (hook's back elbow median is 93°). The
single 179° reading was an outlier from the broken 2D measurement. The rubric
was fine; the measurement feeding it was broken.

Result: sweep's quality score went from **10.2 → 76.0**, defense to 66,
lofted to 77 — plausible numbers for professional shots.

---

## 4. Problem 3 — shot prediction accuracy

**First, it was measured properly.** The README's 94% does not hold anywhere:

| | top-1 | top-3 |
|---|---|---|
| dataset's own held-out test split | **57.6%** | 84.8% |
| the 50 demo clips | 46% | 88% |

Two measurement traps had to be dealt with first:

1. **Half of every dataset folder is junk.** macOS AppleDouble sidecars named
   `._cover_0001.avi` sit next to each real video and carry a video
   extension, but OpenCV opens them to zero frames — feeding the model 30
   black frames. A first measurement read about half the true value because
   of this. Real counts are 1250 train / 250 val / 250 test.
2. **The 50 demo clips leak into training.** They came from this dataset.
   Median feature similarity of demo→train is 0.998 versus 0.875 for
   demo→val/test, and one is an exact duplicate. A retrained model scores 98%
   on the demo clips and 60% on test — so demo-clip accuracy must never be
   quoted as a result.

**A concrete bug was found in how frames were fed to the model.** Training
used 30 **consecutive** frames from frame 0. Inference used 30 **strided**
frames from the middle 60% — a different kind of input than the model was
ever trained on. Fixing only that:

| frame selection | top-1 | top-3 |
|---|---|---|
| strided, middle 60% (as shipped) | 32% | 66% |
| consecutive from frame 0 (as trained) | **46%** | **88%** |

Other placements were tried and are worse (start at 20%: 38%; centred: 34%;
averaging several windows: 36%).

**What was tried and did not work** — recorded so it is not repeated:

- **Four head architectures** (the original, a smaller regularised GRU, a
  bidirectional GRU with attention pooling, and a no-recurrence
  pooling+MLP control). All landed at 58–60% test top-1.
- **Fusing in the striker's skeleton**, four ways: concatenating features
  −4.0 points, averaging the two models' probabilities +0.8, weighted
  averaging +3.6 (with the weight tuned on val, so optimistic), separate
  encoders per input −2.8. The skeleton does carry real signal — alone it
  reaches 26% against a 10% random baseline, and it beats pixels outright on
  cover (36% vs 16%) and lofted (36% vs 16%) — but it never lifted top-1.

**The diagnosis.** EfficientNetB0 is an **ImageNet** model: its features
describe what a single frame *looks* like. A cricket shot is defined by how
the body and bat *move*. That was the ceiling, and it is why neither the head
nor the skeleton moved it.

**What did work.** Adding a **Kinetics-400 pretrained video model** (`r3d_18`),
which encodes motion by construction, alongside the existing features:

| model | test top-1 | test top-3 |
|---|---|---|
| shipped weights | 57.6% | 84.8% |
| EfficientNetB0 retrained | 52.4% | 80.8% |
| r3d_18 alone | 56.4% | 79.2% |
| **r3d_18 + EfficientNetB0** | **62.4%** | 81.2% |

The two backbones fail on different shots, which is why combining them helps:
r3d_18 is far better on straight (76% vs 52%) and lofted (40% vs 20%),
EfficientNetB0 better on square_cut and hook, and the pair beats both on
straight, square_cut, and flick.

**A methodological point worth making.** All of the fusion experiments above
were run on a 400-clip training subset, where every model sat at 38–41%. On
that subset the backbone gain looked like noise (+2.4 points); on the full
1250 clips it was **+7.2**. Subset results at this data size are not
trustworthy — a lesson that applies to the skeleton-fusion results too, which
may deserve a retest on full data.

Model selection was done on the **validation** split only; test was read once
at the end. Val and test agreed exactly at 62.4%.

**A side benefit.** The head was trained on sequences of length 3, so only
frames 0, 14, and 29 ever reached it — computing EfficientNet features for the
other 27 was pure waste. Dropping them took classification from ~50 s to
~1.5 s.

---

## 5. Problem 4 — does it work on any similar video?

This was tested rather than assumed. Real clips were re-encoded into the
shapes an uploaded clip actually takes, and striker selection re-run and
checked by eye on the boxed frames.

| clip shape | works? |
|---|---|
| original 1280×720 broadcast | yes |
| **vertical 9:16 (phone / YouTube short)** | yes |
| **1.6× centre zoom (close clip)** | yes |
| letterboxed with black bars | yes |
| 854×480 low-quality re-upload | yes |

Vertical and zoomed clips **failed at first**. Two more bugs, both the same
mistake as the crop bug in section 2 — reasoning in the wrong coordinate
space:

1. **Aspect ratio was computed in normalised coordinates.** Cropping 1280×720
   to 9:16 shrinks the width to 405 px, which inflates every normalised width
   by 3.16× and collapses apparent aspect by the same factor. Every upright
   player then failed the shape test and the striker was missed entirely.
   Aspect must be measured in **pixels**.
2. **Edge-clipped boxes were trusted.** A zoom cuts the striker off at the top
   of the frame, so his box is a crop of him, not his outline. His height read
   short, which both failed the shape test and made the keeper test conclude
   he was "too short to be the striker" — so it walked down and picked the
   **bowler**. Tracks now carry a `clipped` flag; for those the shape tests
   are waived and the keeper test refuses to run. The *size* test still
   applies to them, because that is what excludes distant fielders — waiving
   it too caused a regression where a fielder at the top of the frame,
   0.08 of frame height, was chosen as the striker.

**No regression:** all ten `video1` clips select an identical striker box
(centre and height to three decimals) before and after these changes.

**Confirmed on a real upload.** A vertical 480×854, 13-second YouTube clip
(KL Rahul) now gives: striker found, 30/30 frames with a skeleton, 10.3 of 13
joints, both output videos produced, in ~28 s.

---

## 6. Problem 5 — the web app

Several engineering faults were found by actually running the app rather than
only reading it:

- **The skeleton video never played.** It was written with OpenCV's default
  `mp4v` codec, which browsers cannot decode, so the player appeared blank.
  Switched to H.264 (`avc1`).
- **The upload limit was 3 MB** (`maxUploadSize = 3` in
  `.streamlit/config.toml`), which rejected most real clips. Raised to 300 MB.
- **The app crashed on interaction.** The download archive was deleted and
  rebuilt on every Streamlit rerun; on Windows this raised
  `PermissionError: [WinError 32]` whenever the browser still held the file
  open. It now builds once and is reused, staged under a separate name and
  moved into place.
- **A second bug in that same fix** returned a path that did not exist,
  because `Path.with_suffix` *replaces* the last suffix — a staging base of
  `clip_pipeline.building` resolved to `clip_pipeline.zip` while the file
  written was `clip_pipeline.building.zip`. Fixed by using the path
  `shutil.make_archive` returns instead of reconstructing it.
- **An 8-second upload took over two minutes**, because YOLO ran on every
  frame (~200 frames × 0.6 s). Detection is now capped at 60 evenly spaced
  frames with the box interpolated between them, and the detector was changed
  from `yolov8m` to `yolov8s` after verifying it picks the **same striker in
  10/10 clips**, with box centre and height agreeing to three decimals, at
  2.3× the speed. Total upload time: **92 s → ~25 s**.
- **The skeleton video had no bounding box**, unlike the reference
  `skeleton_*_striker.mp4` files. It now draws the green box and label, plus a
  second **side-by-side comparison video** (original next to skeleton), which
  is the quickest way to confirm the right player was chosen.

---

## 7. What an upload produces now

```
<video>_pipeline/
  00_metadata.json             clip info, detection rate, timings
  01_extracted_frames/         the 30 frames given to the classifier
  02_skeleton_keypoints.json   all 33 landmarks for every analysed frame
  03_skeleton_overlay_frames/  skeleton drawn on the striker
  04_comparison_frames/        original beside skeleton
  05_shot_analysis.json        prediction, top-3, angles by phase, quality
  skeleton_<video>.mp4         box + skeleton, playable
  comparison_<video>.mp4       original | skeleton, side by side
  PIPELINE_SUMMARY.txt         everything in readable form
```

All of it is viewable in the app and downloadable as a single zip. The app
computes this once per clip and renders every panel from that one result, so
what is downloaded cannot disagree with what was displayed.

---

## 8. Summary of measured results

| | before | after |
|---|---|---|
| skeleton on correct player | no — non-striker / umpire / keeper | **0 wrong frames in 416** |
| pose detection rate | 60–85% | **100% on 9 of 10 clips** |
| joint angles | clip-averaged, 2D — impossible values | **impact frame, metric 3D** |
| quality score (sweep) | 10.2 | **76.0** |
| shot prediction (test top-1) | 57.6% | **62.4%** |
| shot prediction (test top-3) | 84.8% | 81.2% |
| works on vertical / zoomed clips | no | **yes** |
| time per upload | 92 s | **~25 s** |
| upload size limit | 3 MB | 300 MB |
| data folder for uploads | did not exist | **full folder + zip** |

---

## 9. Honest limitations

- **Shot prediction is 62.4%, not perfect.** Roughly 6 of 10 clips get the
  right shot name; the right answer is in the top 3 about 8 times in 10. Going
  much higher would mean fine-tuning the backbones themselves, which is not
  practical on CPU. The skeleton, box, angles, quality score, and data folder
  are all correct and reliable — it is specifically the *shot name* that is at
  62.4%.
- **Because the quality criteria are chosen by the predicted shot type, a
  wrong prediction means the shot is graded against the wrong technique.** The
  app warns when confidence is below 60%.
- **Camera angle matters.** This is built for the standard broadcast angle
  filmed from behind the bowler's arm. Square-leg, stump-cam, or side-on
  phone footage is out of scope, and the model returns nothing rather than
  guessing a person.
- **Four of ten shots** (`late_cut`, `square_cut`, `lofted`, `straight`) still
  fall through to a generic quality scorer and have no shot-specific rules.
- **The skeleton-fusion experiments used only 400 training clips**, where
  results proved unreliable. They deserve a retest on the full 1250 before
  concluding the skeleton cannot help prediction.
