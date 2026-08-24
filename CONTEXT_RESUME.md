# Context Resume — Cricket Shot Quality AI (BTP Project)

Purpose of this file: a new chat session should be able to read this and
continue the work with **zero loss of context**. Everything below is either
verified against the actual files on disk (checked while writing this doc)
or carried over from the working session log — nothing here is guessed.
Where a number/path could go stale, it says how to re-check it.

Written: 2026-08-17. Project root:
`C:\Users\vipul\Desktop\Don't Delete\BTP-Project\CricketShotQualityAI`

---

## 1. The goal (unchanged, from day one)

Upload a cricket batting video → the app predicts the shot type → runs full
pose/quality analysis → produces a downloadable data folder for that video.
Delivered as a Streamlit web app (`app.py`). Scope is limited to the
**standard broadcast camera angle** (behind the bowler's arm) — not every
possible camera angle.

## 2. How the user works — read this before doing anything

- Communicates in Hinglish, casual, short messages. Reply in the same style
  in chat (this doc is English because it's a technical record).
- **Ask before implementing.** Confirm before starting a change; validate on
  a few clips before ever running something over the whole dataset.
- **Explicit threshold set this session**: only ask first for tasks estimated
  at **2-3+ hours**. Anything shorter: just do it, no need to check in.
- Wants "elite level" — correctness and polish both matter, but don't gold-
  plate; the user has said "not much is needed, just make it good" for UI
  work — clean and correct over feature-heavy.
- Corrects file/detail confusion patiently (e.g. asked what `._hook_XXXX.avi`
  files were — these are macOS AppleDouble sidecar junk files, 163 bytes,
  unplayable; the real video is the same name without the `._` prefix).
- Full memory files live at
  `C:\Users\vipul\.claude\projects\C--Users-vipul-Desktop-Don-t-Delete-BTP-Project\memory\`,
  indexed by `MEMORY.md` there. Relevant ones: `btp-goal.md`,
  `ask-before-implementing.md`, `striker-not-batsman.md`,
  `reframing-robustness.md`, `classifier-real-accuracy.md`.

## 3. Pipeline architecture (what the app actually does, end to end)

1. **Person detection + tracking**: YOLOv8 (`yolov8s.pt` for feature caching,
   `yolov8m`-class for the live overlay path) + BoT-SORT tracker.
2. **Striker identification** (`src/pose/striker_pose.py`): out of all tracked
   people in a clip, picks the one who is the *striker* (the batsman on strike
   playing the shot) — not just any batsman, not the non-striker, not the
   keeper, not the umpire. Decided once per clip from track statistics
   (height, position, aspect ratio, clipped-edge handling, keeper
   disambiguation), not per-frame.
3. **Pose estimation** (MediaPipe Pose, via `pose_landmarker.task`): run on
   the cropped striker box each frame. Produces both `pose_landmarks` (2D
   image-plane + relative z) and `pose_world_landmarks` (metric 3D, hip-
   centred).
4. **Handedness detection** (`src/pose/estimator.py::detect_handedness`):
   votes across all frames of the clip using top-hand wrist height + world-
   landmark shoulder z-depth → returns left/right + confidence. Needed
   because "front leg/arm" physically flips between a right-handed and
   left-handed batsman.
5. **Joint angles** (`src/pose/estimator.py`): front/back knee angle,
   front/back elbow angle, shoulder tilt, hip tilt, trunk lean — computed
   from **world landmarks** (not 2D — on this camera angle the legs project
   nearly collinear in 2D, so 2D-only angle math cannot see a bent knee at
   all), using the detected handedness to map to the correct physical side.
6. **Impact-frame detection** (`src/pose/estimator.py::impact_frame_index`
   → `_first_prominent_peak`): the frame of bat-ball contact, found from
   wrist-speed — see §5 for the exact rule and why it changed.
7. **Shot classification**: fusion model over CNN (video) features +
   MediaPipe pose features, `SHOT_CLASSES` = cover, defense, flick, hook,
   late_cut, lofted, pull, square_cut, straight, sweep (10 classes).
   Current real (non-training-set) accuracy: **62.4%**, up from a starting
   57.6%, after moving to a Kinetics-pretrained video backbone (see memory
   `classifier-real-accuracy.md`). **This is the known weak point** — see
   §8/§9.
8. **Quality scoring + "You vs Professionals"**: joint angles at
   stance/impact/follow-through compared against per-shot-class IQR ranges
   derived from ~650 professional training clips (`ideal_angles.json`, see
   §6). This is a **single-point-in-time comparison at the impact frame
   only** — this is exactly what the newest pending request (§9) wants to
   upgrade to a movement/time-based comparison.
9. **Streamlit app** (`app.py`, 982 lines) renders all of the above —
   uploaded video, skeleton overlay, verdict, quality breakdown, angle
   comparison, downloadable per-video data folder (zipped).

## 4. Everything fixed / built, in order

### 4.1 Striker/tracking correctness (validated across all 50 demo clips)
- Fixed skeleton originally drawing on the wrong person (non-striker /
  keeper / umpire) — striker selection logic built out in
  `src/pose/striker_pose.py`.
- **Reframing robustness** (memory: `reframing-robustness.md`): striker
  selection tested to survive vertical 9:16 crops, 1.6x zoom, letterboxing,
  480p — required two coordinate-space fixes: (a) aspect ratio must be
  computed in pixel space via `frame_shape`, not normalized coords (else a
  vertical crop inflates normalized width and every upright player fails
  the aspect check); (b) edge-clipped boxes get their shape checks relaxed
  (a zoomed striker cut off at the frame edge reads short/wrong-aspect
  otherwise).
- **Two more bugs found validating the remaining 40 demo clips** (clips
  video2–video5 per class, never tested before):
  - `square_cut/video5.mp4`: the clipped-box exemption was a *full* waiver
    on the aspect-ratio check, letting a junk detection (aspect 0.55,
    wider-than-tall — not a person) through as a candidate. Fixed with
    `CLIPPED_MIN_ASPECT = 0.85` in `striker_pose.py` — a floor, not a
    waiver: still lenient for a legitimately truncated striker, but still
    rejects shapes no person can have even clipped.
  - `straight/video2.mp4`: keeper-vs-striker disambiguation
    (`_is_keeper_of`) compared whole-clip median height between tracks,
    which breaks when the camera zooms mid-clip (the striker's track spans
    pre- and post-zoom frames, diluting his median height; the keeper's
    track only exists post-zoom, all at larger scale) — made the keeper
    read taller than the striker and walk the algorithm onto the wrong
    person. Fixed by comparing height only over the frame indices the two
    tracks actually share.
  - Verified with an automated per-frame wrong-person checker across all 50
    demo clips post-fix: 0 regressions, both bugs confirmed fixed.

### 4.2 Handedness (new capability this session)
- `compute_cricket_angles` previously **hardcoded** "left side = front,"
  silently misgrading every left-handed batsman and any mirrored clip.
- Added `detect_handedness(frames_kp, worlds=None)` in
  `src/pose/estimator.py`: votes on top-hand wrist height + world-landmark
  shoulder z-depth across the whole clip; returns
  `{"hand", "confidence", "votes", "assumed"}`.
- Threaded a `handedness` parameter through `compute_cricket_angles`,
  `angles_from_world`, `angles_at_frame`, `phase_angles`, and
  `cache_pose_features.py::pose_vector` — detected **once per clip** so
  every frame is graded against the same physical side.

### 4.3 Impact-frame detection fix (systematic bias found and fixed)
- Old rule: "frame of globally-fastest wrist speed." Measured to pick a
  post-shot recovery-movement frame (standing up, running) instead of the
  actual swing in **38%** of clips, because that motion is sometimes faster
  than the swing itself. Found by inspecting sweep clips where the "impact"
  frame showed the batsman already upright (front knee 160–167°, impossible
  mid-sweep).
- New rule, `_first_prominent_peak()` in `estimator.py`: take the **first**
  local speed-maximum that clears 50% of the clip's global peak speed
  (`IMPACT_PEAK_THRESHOLD = 0.5`), not the single fastest frame overall.
  Falls back to global max if nothing qualifies.
- Reduced the "impact landed in the last 5 of 30 frames" rate from 38% to
  15%.
- Applied identically in both the live pipeline (`estimator.py`) and the
  offline range-derivation script (`derive_ideal_angles.py` imports
  `_first_prominent_peak` directly rather than reimplementing it — confirmed
  still wired this way as of this doc).

### 4.4 `ideal_angles.json` regenerated
- Source: `derive_ideal_angles.py`, run over `train`+`val` pose caches.
- **Confirmed current state** (checked while writing this doc):
  `_source_splits: ["train", "val"]`, all 10 classes present (cover,
  defense, flick, hook, late_cut, lofted, pull, square_cut, straight,
  sweep), `_min_clips_per_range: 8`. Each class has per-angle
  `{low, median, high, n}` from the impact-frame IQR (25th/50th/75th
  percentile) across professional clips. n is roughly 61–65 per class,
  i.e. **~650 clips used out of 1250(train)+250(val)=1500 available** — the
  user was told this and explicitly deferred expanding it ("abhi nahi baad
  me" — not urgent, do later).
- Angle keys: `front_knee_angle`, `back_knee_angle`, `front_elbow_angle`,
  `back_elbow_angle`, `shoulder_tilt_deg`, `hip_tilt_deg`, `trunk_lean_deg`.

### 4.5 Pose feature re-caching
- `cache_pose_features.py::pose_vector` now uses `angles_from_world` when
  world landmarks are available (falls back to 2D `compute_cricket_angles`
  otherwise), with handedness passed through.
- Re-run this session to pick up the handedness + world-landmark fixes.
- **Confirmed current files on disk** (`features/`):
  `train_pose_n40.npy` (400, 30, 46), `val_pose_n40.npy` (250, 30, 46), plus
  matching `_paths.txt` files, `demo_pose.npy`, and the separate CNN/video
  feature caches (`train_X.npy`, `train_vid.npy`, `train_vid_n40.npy`, etc.)
  used by the fusion classifier training scripts (`train_fusion.py`,
  `train_fusion2.py`, `train_video.py`, `train_shot_head.py`). Trained model
  weights live in `trained_heads/` (multiple head architectures were tried:
  GRU, BiGRU+attention, pooled MLP, R3D-18, EfficientNet, two/three-way
  fusion — `fusion_results.json` / `fusion2_results.json` / `video_results.json`
  / `results.json` hold the comparison numbers from those experiments).

### 4.6 Full UI redesign (`app.py`, `.streamlit/config.toml`)
- Added a proper dark theme in `.streamlit/config.toml`
  (`base="dark"`, cyan primary `#22D3EE`, dark backgrounds, Inter/Space
  Grotesk-styled via CSS in `app.py`) — **confirmed still in place**.
- `app.py` rewritten (982 lines) with: a CSS design-system block (custom
  properties, hero header, section headers, stat cards, score/pro-comparison
  bars, verdict banner, native Streamlit element overrides for the file
  uploader, video sizing, tabs, buttons, dataframe), hand-written SVG icon
  set (`ICONS` dict + `icon()` helper) replacing all emoji, and a full
  page-structure pass: hero → upload cards → video → pipeline run → verdict
  banner → at-a-glance stat row → videos section (comparison + skeleton-only,
  with download buttons and a cached `pipeline_zip()`) → quality breakdown +
  joint angles (two columns) → "You vs Professionals" (redesigned bars,
  see 4.7) → optional reference-video comparison → downloadable pipeline
  data folder (Frames/Files/Raw output tabs via `st.segmented_control`).
- All emoji removed; replaced with Streamlit's `:material/...:` icon
  shorthand or the hand-written SVGs.
- Bug found + fixed mid-redesign: an early CSS rule forced Inter font onto
  every element including Streamlit's own icon-font spans
  (`[data-testid="stIconMaterial"]`, a ligature font), which broke icon
  rendering into literal text. Fixed by narrowing the font rule to
  `html, body, .stApp` only, no `!important`, letting Streamlit's more
  specific icon rule win.

### 4.7 UI feedback fixes (from user's screenshots)
User's 4 asks: (1) fix wrong predictions outside the 50-training demo clips
— **acknowledged as separate/bigger scope, not fixed here** (this is
exactly what §9's ST-GCN/accuracy request is about); (2) fix horizontal
scroll on the skeleton video panel for portrait/tall videos; (3) remove
excess emoji, make it professional; (4) make "You vs Professionals" more
readable.
- (3) done as part of 4.6.
- (4) done: "You vs Professionals" redesigned with a proper range-band bar
  per angle — `.csq-pro-band` (typical range), `.csq-pro-median` (floating
  "Pros X°" label), `.csq-pro-mark` (circular dot for the user's own value,
  floating "You Y°" label), low/high numbers under the bar, a 4-item legend.
  Verified end-to-end via a live sample-clip run in the browser.
- (2) done but **not empirically screenshot-verified** (the browser preview
  pane stopped compositing partway through verification and there's no way
  to drive a real file upload through the browser tool). Root cause: an
  earlier fix targeted `div[data-testid="stVideo"]`, which doesn't exist in
  this Streamlit version (confirmed via DOM query — actual container is
  `stElementContainer`); also Streamlit sets an inline `height` style on the
  `<video>` tag via JS from a measured aspect ratio, which can be wrong for
  portrait content in a narrow column. Fixed by targeting the bare `video`
  tag with `!important` on width/height/max-height/object-fit — this is
  guaranteed to win by CSS cascade rules regardless of any inline style
  Streamlit sets, but the user was told honestly this specific claim is
  unverified by screenshot and asked to confirm on a real portrait upload.
  **If revisiting: check with the user whether they confirmed this actually
  looks right now.**

## 5. Known limitations / explicitly open problems

- **Shot-prediction accuracy is 62.4% on real (non-training) video**, the
  single biggest quality gap. This is what §9 is about.
- `ideal_angles.json` uses ~650 of 1500 available professional clips
  (deferred, not urgent, by user's own choice).
- No skeleton-overlay videos exist yet for the 650 training clips
  themselves (only for the 50 demo clips + whatever a user runs live) — user
  was offered this, dismissed without choosing, said wait for next
  instruction. **Not started.**
- The portrait-video horizontal-scroll CSS fix is unverified live (see 4.7).
- Ball tracking was discussed (would give a precise, physically grounded
  impact-frame signal from trajectory-direction reversal at bat contact) but
  **not implemented** — it's hard (tiny/blurred/occluded ball, no labeled
  data) and would improve the *analysis* pipeline (angles/scoring), not
  shot-classification accuracy directly. Not currently planned unless the
  user asks again.
- Other previously-listed "next plan" items, not yet started, not currently
  in active request: full-1250-clip skeleton-fusion retraining, bat
  detection, a better pose model (RTMPose), 5-point shot phases (adding
  backlift/stride to the current stance/impact/follow-through), a dedicated
  left-handed-batsman test set, GPU fine-tuning on Colab.

## 6. THE CURRENT / ACTIVE REQUEST — not started yet

The user relayed feedback from their supervisor ("sir") and gave two
concrete asks, both still **unimplemented as of this doc**:

1. **Time/movement-based graph for "You vs Professionals"**, replacing (or
   augmenting) the current single-impact-frame comparison. Idea: starting
   from the frame where the shot-playing motion begins, plot an angle (or
   angles) over time for the professional reference data, and plot the same
   for the uploaded video, so the two can be compared directly as curves —
   intended to make the comparison "ekdum perfectly sharp" (much more
   precise than one static number).
   - Open technical questions to resolve before building: which angle(s) to
     plot; how to detect "shot start" consistently (an analogous problem to
     impact-frame detection — needs a data-driven signal, not a fixed frame
     index, since clip lengths vary — see `phase_angles()` in
     `src/pose/estimator.py` for the existing pattern of signal-based, not
     index-based, phase detection); how to normalize/align time axes across
     clips of different length/frame-rate (e.g. resample to a fixed number
     of samples from shot-start to impact); what the "professional" curve
     is built from — a single representative clip, or a band (like the
     current IQR) evaluated at each normalized time step.
2. **ST-GCN integration** (Spatial-Temporal Graph Convolutional Network) as
   a direct skeleton-to-shot-class classifier, intended to push prediction
   accuracy toward the user's stated goal of shot type being predicted
   correctly essentially always ("ekdum 100% sahi predict ho").
   - This was previously estimated (in an earlier "next plans" discussion,
     not fully detailed in this doc) at roughly a day of work — well past
     the user's 2-3-hour ask-first threshold, so **must be scoped and
     confirmed with the user before starting**, per the standing rule in
     §2.
   - Would need: a graph structure over the 13 cricket joints already
     tracked, a training pipeline reusing the existing pose feature caches
     (`features/train_pose_n40.npy` etc., or a re-extraction with all
     frames/joints if ST-GCN needs more than the current 46-dim-per-frame
     summary), and a decision on how it fuses with (or replaces) the
     current CNN+pose fusion classifier.
3. **Overarching instruction from the user, covering both of the above**:
   integrate everything properly so the shot prediction is fully correct
   and every displayed analysis detail is fully accurate — not just adding
   features in isolation.

**Correct next step, per the user's own standing rules**: since both pieces
are substantial, the right move is to lay out a concrete plan for each
(data needed, what changes technically, a realistic time estimate, how it
integrates with the existing `ideal_angles.json` / `phase_angles()` /
pro-comparison rendering / classifier fusion code) and confirm with the user
which to start with — not to start coding unprompted.

## 7. Quick file map (for a fresh session)

| File | Role |
|---|---|
| `app.py` | The Streamlit app — all UI, 982 lines |
| `.streamlit/config.toml` | Dark theme + upload size config |
| `src/pose/estimator.py` | Angles, handedness detection, impact-frame detection |
| `src/pose/striker_pose.py` | Striker track selection (YOLO+BoT-SORT → best track) |
| `src/pipeline/builder.py` | Orchestrates the full per-video pipeline (large file, read directly if needed) |
| `src/quality/scorer.py` | Hand-written quality-score rule set (pre-dates `ideal_angles.json`; the two should be compared, not assumed redundant) |
| `src/classifier/model.py` | `SHOT_CLASSES` list + classifier model definitions |
| `derive_ideal_angles.py` | Regenerates `ideal_angles.json` from cached pose features |
| `cache_pose_features.py` | Builds the 46-dim/frame pose feature `.npy` caches |
| `ideal_angles.json` | Professional angle ranges per shot class (impact frame only, IQR) |
| `features/*.npy` + `*_paths.txt` | Cached CNN + pose features for train/val/test/demo splits |
| `trained_heads/*.weights.h5`, `*_results.json` | Trained classifier heads + their accuracy comparisons |
| `data/<class>/video{1-5}.mp4` | The 50 demo clips used throughout validation |
| `temp_hf_data/cricketshot/{train,val,test}/<class>/` | The full ~1500-clip professional dataset (real files `<class>_NNNN.avi`; ignore `._<class>_NNNN.avi` — macOS junk, 163 bytes each) |
| Memory dir (see §2) | Cross-session facts: goal, working style, robustness findings, accuracy history |

---
*This file is a manually-maintained resume point, not auto-generated by the
app. If it drifts from reality, trust the code and update this file — don't
trust this file over the code.*
