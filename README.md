# 🏏 Cricket Shot Quality Analyser

> **BTP Project — "Quantifying the Quality/Correctness of a Cricket Shot Using AI"**

Built on top of [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) (MIT License).

---

## What This Project Does

This system goes beyond simple shot classification — it **quantifies how well** a cricket shot was executed using two complementary AI signals:

| Signal | Method | Weight |
|---|---|---|
| Biomechanical correctness | MediaPipe Pose → joint angle rules | 60% |
| Visual similarity to ideal | EfficientNet feature cosine similarity | 40% |

The final output is a **0–100 quality score** with a grade (Excellent / Good / Average / Needs Work) and actionable coaching feedback per criterion.

---

## Features

- **Shot Classification** — EfficientNetB0 + GRU classifies 10 batting shots from video (94% F1)
- **Pose Estimation** — MediaPipe extracts 33 body keypoints per frame
- **Biomechanical Scoring** — Shot-specific joint angle and stance rules for Cover Drive, Pull, Sweep, Defense (generic rules for remaining shots)
- **Similarity Scoring** — Cosine distance between player and reference video feature vectors
- **Coaching Feedback** — Human-readable tips per failing criterion
- **Streamlit UI** — Upload player video + optional reference video, get instant analysis

---

## Project Structure

```
CricketShotQualityAI/
├── app.py                        # Main Streamlit application
├── model_weights.h5              # Pre-trained EfficientNetB0+GRU weights
├── requirements.txt
├── src/
│   ├── classifier/
│   │   └── model.py              # Model architecture, load, predict, feature extract
│   ├── pose/
│   │   └── estimator.py          # MediaPipe pose wrapper + keypoint aggregation
│   ├── quality/
│   │   └── scorer.py             # Biomechanical rules + quality scoring engine
│   └── utils/
│       └── video_utils.py        # Frame extraction utilities
├── data/
│   └── reference_shots/          # Store reference/ideal shot videos here
├── Notebooks/                    # Original training notebooks (EfNetB0/B4/V2B0)
└── notebooks_new/                # New experiment notebooks (pose, quality, etc.)
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/CricketShotQualityAI.git
cd CricketShotQualityAI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## Shot Classes Supported

| Shot | Biomechanical Rules | Status |
|---|---|---|
| Cover Drive | Knee bend, elbow height, shoulder alignment | ✅ Full rules |
| Pull | Back knee bend, arm extension, hip rotation | ✅ Full rules |
| Hook | Same as Pull | ✅ Full rules |
| Sweep | Front knee down, head position, bat plane | ✅ Full rules |
| Defense | Upright stance, elbow tuck, balance | ✅ Full rules |
| Flick, Late Cut, Lofted, Square Cut, Straight | Balance + shoulder alignment | 🔄 Generic rules (expanding) |

---

## Roadmap

- [ ] Add full biomechanical rules for all 10 shot types
- [ ] Curate reference shot library (professional batsmen per shot type)
- [ ] Train a quality regression model on annotated data
- [ ] Add temporal quality tracking (score across frames, not just average)
- [ ] Export PDF coaching report

---

## Credits

- Original classification model: [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) — MIT License
- Dataset: CricShotClassify / CrickShot10 — [Sen et al., Sensors 2021](https://doi.org/10.3390/s21082846)
- Pose estimation: [MediaPipe](https://mediapipe.dev/) by Google
- CNN backbone: [EfficientNet](http://proceedings.mlr.press/v97/tan19a.html) — Tan & Le, ICML 2019
