"""
app.py  —  Cricket Shot Quality Analyser
Features:
  - Shot classification (EfficientNetB0 + GRU, 10 shot types)
  - MediaPipe pose estimation → skeleton overlay
  - Joint angle computation (knee, elbow, shoulder, trunk)
  - Shot quality score 0-100 with grade
  - Per-criterion score breakdown
  - Optional second video for similarity comparison
"""

import os
import sys
import tempfile
import shutil

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB0

sys.path.insert(0, os.path.dirname(__file__))
from src.utils.video_utils import extract_frames, extract_raw_frames
from src.pose.estimator import (
    run_pose_on_frames, draw_skeleton, pose_summary,
    compute_cricket_angles, CRICKET_LANDMARKS,
)
from src.quality.scorer import score_shot

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cricket Shot Analyser", page_icon="🏏", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
SHOT_CLASSES = {
    "cover": 0, "defense": 1, "flick": 2, "hook": 3, "late_cut": 4,
    "lofted": 5, "pull": 6, "square_cut": 7, "straight": 8, "sweep": 9,
}
IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}

GRADE_COLOR = {
    "Excellent": "#2e7d32",
    "Good":      "#1565c0",
    "Average":   "#e65100",
    "Needs Work":"#b71c1c",
}


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    model = models.Sequential([
        layers.TimeDistributed(base, input_shape=(None, 224, 224, 3)),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.GRU(256, return_sequences=True),
        layers.GRU(128),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.load_weights("model_weights.h5")
    return model


def predict(model, frames):
    batch = np.expand_dims(frames, axis=0)
    preds = model.predict(batch, verbose=0)
    idx = int(np.argmax(preds))
    # also return top-3 for display
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3 = [(IDX_TO_CLASS[i], round(float(preds[0][i]) * 100, 1)) for i in top3_idx]
    return IDX_TO_CLASS[idx], float(preds[0][idx]) * 100, top3


def get_features(model, frames):
    batch = np.expand_dims(frames, axis=0)
    _ = model(batch, training=False)
    feat_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    return feat_model.predict(batch, verbose=0)


def cosine_sim(f1, f2):
    dot = np.dot(f1.flatten(), f2.flatten())
    return float(dot / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)) * 100


def save_upload(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        return tmp.name


def score_bar(label: str, score: float, ideal: str, actual: str, status: str):
    """Render a single criterion as a labelled progress bar."""
    bar_color = "#2e7d32" if score >= 80 else ("#e65100" if score >= 55 else "#b71c1c")
    st.markdown(
        f"""
        <div style='margin-bottom:10px'>
          <div style='display:flex; justify-content:space-between; font-size:0.85rem'>
            <span><b>{label}</b> &nbsp; <span style='color:#888'>ideal: {ideal}</span></span>
            <span>{status} &nbsp; <b>{score:.0f}/100</b></span>
          </div>
          <div style='background:#333; border-radius:6px; height:10px; margin-top:3px'>
            <div style='width:{score}%; background:{bar_color}; height:10px; border-radius:6px'></div>
          </div>
          <div style='font-size:0.78rem; color:#aaa; margin-top:2px'>Measured: {actual}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏏 Cricket Shot Quality Analyser")
st.caption("Upload a batting video — get shot classification, pose analysis, joint angles, and quality score.")

model = load_model()

col_up1, col_up2 = st.columns(2)
with col_up1:
    st.subheader("Player Video")
    v1 = st.file_uploader("Upload batting video", type=["mp4", "avi", "mov"], key="v1")
with col_up2:
    st.subheader("Reference Video (optional)")
    v2 = st.file_uploader("Upload reference / ideal shot for comparison", type=["mp4", "avi", "mov"], key="v2")

if not v1:
    st.info("👆 Upload a video to start the analysis.")
    st.stop()

# ── Save and display ──────────────────────────────────────────────────────────
p1 = save_upload(v1)
with col_up1:
    st.video(v1)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Analysing video…"):
    # 1. Classification
    frames1 = extract_frames(p1, n_frames=30)
    shot1, conf1, top3 = predict(model, frames1)

    # 2. Pose
    raw_frames = extract_raw_frames(p1, max_frames=30)
    frames_kp  = run_pose_on_frames(raw_frames)
    summary    = pose_summary(frames_kp)
    angles     = compute_cricket_angles(summary["avg_keypoints"])

    # 3. Quality score
    quality    = score_shot(shot1, angles)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHOT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Shot Classification")

cls_col1, cls_col2 = st.columns([2, 1])

with cls_col1:
    st.markdown(
        f"<h2 style='margin:0'>{shot1.replace('_',' ').title()}</h2>"
        f"<p style='color:#aaa; margin:0'>Confidence: {conf1:.1f}%</p>",
        unsafe_allow_html=True,
    )

with cls_col2:
    st.markdown("**Top 3 predictions:**")
    for name, prob in top3:
        bar_w = int(prob)
        st.markdown(
            f"<div style='font-size:0.85rem; margin-bottom:4px'>"
            f"{name.replace('_',' ').title()} — {prob:.1f}%"
            f"<div style='background:#1565c0; height:6px; width:{bar_w}%; border-radius:3px'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Shot Quality Score")

grade_color = GRADE_COLOR.get(quality.grade, "#555")

qs_col1, qs_col2 = st.columns([1, 2])

with qs_col1:
    st.markdown(
        f"""
        <div style='text-align:center; padding:24px; border-radius:12px; background:#1e1e1e'>
          <div style='font-size:4rem; font-weight:900; color:{grade_color}'>{quality.overall_score:.0f}</div>
          <div style='font-size:0.8rem; color:#aaa'>out of 100</div>
          <div style='font-size:1.4rem; font-weight:700; color:{grade_color}; margin-top:6px'>{quality.grade}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with qs_col2:
    st.markdown(f"**Criterion Breakdown — {shot1.replace('_',' ').title()}**")
    for c in quality.criteria:
        score_bar(c.name, c.score, c.ideal, c.actual, c.status)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — POSE & JOINT ANGLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🦴 Pose & Joint Angles")

pose_col1, pose_col2 = st.columns([1, 1])

with pose_col1:
    # Best frame
    best_idx, best_score = 0, -1
    for i, kp in enumerate(frames_kp):
        if kp is not None:
            sc = sum(1 for v in kp.values() if v[3] > 0.5)
            if sc > best_score:
                best_score, best_idx = sc, i

    if frames_kp[best_idx] is not None:
        annotated = draw_skeleton(raw_frames[best_idx], frames_kp[best_idx])
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption="Skeleton overlay (best detected frame)", use_column_width=True)
    else:
        st.warning("No pose detected. Try a closer shot of the batsman.")

    # Detection rate
    det = summary["detection_rate"] * 100
    color = "green" if det > 70 else ("orange" if det > 40 else "red")
    st.markdown(
        f"<span style='color:{color}'>Pose detected in {summary['detected_frames']}/{summary['total_frames']} frames ({det:.0f}%)</span>",
        unsafe_allow_html=True,
    )

with pose_col2:
    st.markdown("**Computed Joint Angles**")

    angle_rows = [
        ("Front Knee Angle",   angles.get("front_knee_angle"),  "~130–155° (cover drive)"),
        ("Back Knee Angle",    angles.get("back_knee_angle"),   "~110–140° (pull/hook)"),
        ("Front Elbow Angle",  angles.get("front_elbow_angle"), "~100–150°"),
        ("Back Elbow Angle",   angles.get("back_elbow_angle"),  "~80–140°"),
        ("Shoulder Tilt",      angles.get("shoulder_tilt_deg"), "<10° (level)"),
        ("Hip Tilt",           angles.get("hip_tilt_deg"),      "<15° (balanced)"),
        ("Trunk Lean",         angles.get("trunk_lean_deg"),    "5–25° (forward lean)"),
    ]

    table_rows = ""
    for label, val, note in angle_rows:
        display = f"{val}°" if val is not None else "—"
        table_rows += f"| **{label}** | {display} | {note} |\n"

    st.markdown(
        "| Angle | Measured | Ideal Range |\n"
        "|---|---|---|\n" + table_rows
    )

    st.markdown("")
    st.markdown("**Key Joint Positions** *(x, y normalised 0–1)*")
    joints = summary["cricket_joints"]
    if joints:
        rows = "\n".join(f"| `{n}` | {x:.3f} | {y:.3f} |" for n, (x, y) in joints.items())
        st.markdown("| Joint | X | Y |\n|---|---|---|\n" + rows)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REFERENCE COMPARISON (optional)
# ══════════════════════════════════════════════════════════════════════════════
if v2:
    p2 = save_upload(v2)
    with col_up2:
        st.video(v2)

    with st.spinner("Analysing reference video…"):
        frames2 = extract_frames(p2, n_frames=30)
        shot2, conf2, _ = predict(model, frames2)
        f1 = get_features(model, frames1)
        f2 = get_features(model, frames2)
        sim = cosine_sim(f1, f2)

    st.markdown("## 🔁 Reference Comparison")
    col_up2.success(f"**{shot2.replace('_',' ').title()}** — {conf2:.1f}%")

    if shot1 == shot2:
        st.success(
            f"Both videos are **{shot1.replace('_',' ').title()}** — "
            f"Visual similarity: **{sim:.1f}%**"
        )
    else:
        st.warning(
            f"Player: **{shot1}** vs Reference: **{shot2}**. "
            f"Similarity: {sim:.1f}% (different shots — comparison limited)."
        )

    try:
        os.unlink(p2)
    except Exception:
        pass

# ── Cleanup ───────────────────────────────────────────────────────────────────
try:
    os.unlink(p1)
except Exception:
    pass
