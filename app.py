"""
app.py
Cricket Shot Classifier — Week 1
- Upload a video → get shot name + confidence
- Upload two videos → compare similarity
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
from src.utils.video_utils import extract_frames

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cricket Shot Classifier", page_icon="🏏", layout="wide")

# ── Constants ─────────────────────────────────────────────────────────────────
SHOT_CLASSES = {
    "cover": 0, "defense": 1, "flick": 2, "hook": 3, "late_cut": 4,
    "lofted": 5, "pull": 6, "square_cut": 7, "straight": 8, "sweep": 9,
}
IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}


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
    """Returns (shot_name, confidence_percent)"""
    batch = np.expand_dims(frames, axis=0)
    preds = model.predict(batch, verbose=0)
    idx = int(np.argmax(preds))
    return IDX_TO_CLASS[idx], float(preds[0][idx]) * 100


def get_features(model, frames):
    """Extract 1024-d feature vector from Dense layer (for similarity)"""
    feat_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    return feat_model.predict(np.expand_dims(frames, axis=0), verbose=0)


def cosine_sim(f1, f2):
    dot = np.dot(f1.flatten(), f2.flatten())
    return float(dot / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8)) * 100


def save_upload(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        return tmp.name


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏏 Cricket Shot Classifier")
st.caption("Upload a batting video to identify the shot type. Optionally compare two videos.")

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Video 1")
    v1 = st.file_uploader("Upload first video", type=["mp4", "avi", "mov"], key="v1")

with col2:
    st.subheader("Video 2 (optional)")
    v2 = st.file_uploader("Upload second video", type=["mp4", "avi", "mov"], key="v2")

# ── Results ───────────────────────────────────────────────────────────────────
if v1:
    p1 = save_upload(v1)
    with col1:
        st.video(v1)
    with st.spinner("Classifying video 1…"):
        frames1 = extract_frames(p1, n_frames=30)
        shot1, conf1 = predict(model, frames1)
    col1.success(f"**{shot1.replace('_', ' ').title()}** — {conf1:.1f}% confident")

if v2:
    p2 = save_upload(v2)
    with col2:
        st.video(v2)
    with st.spinner("Classifying video 2…"):
        frames2 = extract_frames(p2, n_frames=30)
        shot2, conf2 = predict(model, frames2)
    col2.success(f"**{shot2.replace('_', ' ').title()}** — {conf2:.1f}% confident")

if v1 and v2:
    st.markdown("---")
    with st.spinner("Comparing videos…"):
        f1 = get_features(model, frames1)
        f2 = get_features(model, frames2)
        sim = cosine_sim(f1, f2)

    if shot1 == shot2:
        st.success(f"Both videos are **{shot1.replace('_',' ').title()}** — Similarity: **{sim:.1f}%**")
    else:
        st.warning(
            f"Different shots detected (**{shot1}** vs **{shot2}**). "
            f"Similarity score: {sim:.1f}% (may not be meaningful)"
        )

    # cleanup
    try:
        os.unlink(p1)
        os.unlink(p2)
    except Exception:
        pass

elif not v1:
    st.info("👆 Upload a video above to get started.")
