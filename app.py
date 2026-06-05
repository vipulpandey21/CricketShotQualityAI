"""
app.py
Cricket Shot Classifier — Week 2
- Upload a video → classify shot type + confidence
- MediaPipe pose estimation → skeleton overlay + joint coordinates
- Upload two videos → compare visual similarity
Built on top of RITIK-12/CricketShotClassification (MIT License)
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
from src.pose.estimator import run_pose_on_frames, draw_skeleton, pose_summary, CRICKET_LANDMARKS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cricket Shot Classifier", page_icon="🏏", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
SHOT_CLASSES = {
    "cover": 0, "defense": 1, "flick": 2, "hook": 3, "late_cut": 4,
    "lofted": 5, "pull": 6, "square_cut": 7, "straight": 8, "sweep": 9,
}
IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classification model…")
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


def get_features(model, frames):
    """Extract 1024-d feature vector for similarity comparison."""
    batch = np.expand_dims(frames, axis=0)
    _ = model(batch, training=False)   # builds the graph
    feat_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    return feat_model.predict(batch, verbose=0)


def predict(model, frames):
    """Returns (shot_name, confidence_percent)"""
    batch = np.expand_dims(frames, axis=0)
    preds = model.predict(batch, verbose=0)
    idx = int(np.argmax(preds))
    return IDX_TO_CLASS[idx], float(preds[0][idx]) * 100


def cosine_sim(f1, f2):
    dot = np.dot(f1.flatten(), f2.flatten())
    return float(dot / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)) * 100


def save_upload(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        return tmp.name


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏏 Cricket Shot Classifier + Pose Analysis")
st.caption(
    "Upload a batting video to classify the shot and analyse body joint positions. "
    "Optionally upload a second video to compare similarity."
)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Video 1")
    v1 = st.file_uploader("Upload first video", type=["mp4", "avi", "mov"], key="v1")

with col2:
    st.subheader("Video 2 (optional — for similarity)")
    v2 = st.file_uploader("Upload second video", type=["mp4", "avi", "mov"], key="v2")

# ── Process Video 1 ───────────────────────────────────────────────────────────
if v1:
    p1 = save_upload(v1)

    with col1:
        st.video(v1)

    # ── Step 1: Shot Classification ───────────────────────────────────────────
    with st.spinner("Classifying shot…"):
        frames1 = extract_frames(p1, n_frames=30)
        shot1, conf1 = predict(model, frames1)

    st.markdown("---")
    st.subheader(f"Shot: **{shot1.replace('_', ' ').title()}** — {conf1:.1f}% confident")

    # ── Step 2: Pose Estimation ───────────────────────────────────────────────
    st.markdown("### 🦴 Pose Analysis")

    with st.spinner("Running MediaPipe pose estimation…"):
        raw_frames = extract_raw_frames(p1, max_frames=30)
        frames_kp  = run_pose_on_frames(raw_frames)
        summary    = pose_summary(frames_kp)

    pose_col1, pose_col2 = st.columns([1, 1])

    with pose_col1:
        # Pick best frame: the one with the most high-visibility landmarks
        best_idx = 0
        best_score = -1
        for i, kp in enumerate(frames_kp):
            if kp is not None:
                score = sum(1 for v in kp.values() if v[3] > 0.5)
                if score > best_score:
                    best_score = score
                    best_idx = i

        if frames_kp[best_idx] is not None:
            annotated = draw_skeleton(raw_frames[best_idx], frames_kp[best_idx])
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Skeleton overlay (best frame)", use_column_width=True)
        else:
            st.warning("No pose detected in any frame.")

    with pose_col2:
        # Detection quality
        det_rate = summary["detection_rate"] * 100
        color = "green" if det_rate > 70 else ("orange" if det_rate > 40 else "red")
        st.markdown(
            f"**Pose detection rate:** "
            f"<span style='color:{color}'>{summary['detected_frames']}/{summary['total_frames']} frames ({det_rate:.0f}%)</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        # Cricket joint coordinates
        joints = summary["cricket_joints"]
        if joints:
            st.markdown("**Key joint positions** *(normalised 0–1, top-left = 0,0)*")
            rows = []
            for name, (x, y) in joints.items():
                rows.append(f"| `{name}` | x={x:.3f} | y={y:.3f} |")

            st.markdown(
                "| Joint | X | Y |\n"
                "|---|---|---|\n" +
                "\n".join(rows)
            )
            st.caption(
                "These coordinates are the foundation for quality scoring (Week 4-5). "
                "Knee angle, elbow height, and hip rotation will be computed from these numbers."
            )
        else:
            st.warning("Could not extract joint positions — try a video with a clearer view of the batsman.")

    # ── Step 3: Video 2 similarity ────────────────────────────────────────────
    if v2:
        p2 = save_upload(v2)
        with col2:
            st.video(v2)

        with st.spinner("Classifying and comparing video 2…"):
            frames2 = extract_frames(p2, n_frames=30)
            shot2, conf2 = predict(model, frames2)
            f1 = get_features(model, frames1)
            f2 = get_features(model, frames2)
            sim = cosine_sim(f1, f2)

        col2.success(f"**{shot2.replace('_', ' ').title()}** — {conf2:.1f}% confident")

        st.markdown("---")
        if shot1 == shot2:
            st.success(
                f"Both videos are **{shot1.replace('_', ' ').title()}** — "
                f"Visual similarity: **{sim:.1f}%**"
            )
        else:
            st.warning(
                f"Different shots: **{shot1}** vs **{shot2}**. "
                f"Similarity score: {sim:.1f}% (cross-shot comparison is not meaningful)."
            )

        try:
            os.unlink(p2)
        except Exception:
            pass

    try:
        os.unlink(p1)
    except Exception:
        pass

else:
    st.info("👆 Upload a video above to get started.")
