# 🏏 Cricket Shot Quality Analyser

> **BTP Project — "Quantifying the Quality/Correctness of a Cricket Shot Using AI"**
> Timeline: 3 months | Built on [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) (MIT License)

---

## Current State (Week 1)

Working Streamlit app that:
- Classifies a cricket shot from a video (10 shot types, 94% accuracy)
- Compares two videos using cosine similarity of EfficientNet features

## Setup

```bash
git clone https://github.com/vipulpandey21/CricketShotQualityAI.git
cd CricketShotQualityAI

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
app.py                  ← Streamlit UI
model_weights.h5        ← Pre-trained EfficientNetB0+GRU weights (from Ritik's work)
src/
  utils/
    video_utils.py      ← Frame extraction from video
Notebooks/              ← Original training notebooks (reference only)
```

## Shot Classes

`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`, `square_cut`, `straight`, `sweep`

## Roadmap

- [x] Week 1 — Shot classification + similarity working app
- [ ] Week 2-3 — Add MediaPipe pose estimation, extract joint keypoints
- [ ] Week 4-5 — Build biomechanical scoring rules per shot type
- [ ] Week 6-8 — Quality score (0-100) + coaching feedback UI
- [ ] Week 9-12 — Regression model, reference shot library, final report

## Credits

- Base model: [RITIK-12/CricketShotClassification](https://github.com/RITIK-12/CricketShotClassification) — MIT License
- Dataset: CricShotClassify / CrickShot10 — [Sen et al., Sensors 2021](https://doi.org/10.3390/s21082846)
