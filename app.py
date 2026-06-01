"""
app.py
Streamlit app — Cricket Shot Quality Analyser
Extends the original CricketShotClassification repo with:
  • Pose estimation (MediaPipe)
  • Biomechanical quality scoring
  • Feature-based similarity scoring vs a reference video
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import tempfile
import shutil

import cv2
import numpy as np
import streamlit as st

from src.utils.video_utils import extract_frames, extract_raw_frames
from src.classifier.model import load_classifier, predict_shot, get_feature_extractor, cosine_similarity
from src.quality.scorer import ShotQualityScorer

# ── MediaPipe is optional — gracefully degrade if not installed ──────────────
try:
    from src.pose.estimator import PoseEstimator, aggregate_keypoints
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Shot Quality Analyser",
    page_icon="🏏",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH = "model_weights.h5"
N_FRAMES = 30
SCORER = ShotQualityScorer(pose_weight=0.6, similarity_weight=0.4)

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_upload(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp path and return the path."""
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(uploaded_file, tmp)
        return tmp.name


@st.cache_resource(show_spinner="Loading classification model…")
def get_model():
    return load_classifier(WEIGHTS_PATH)


def render_score_bar(label: str, score: float, color: str = "#4CAF50"):
    """Render a labelled progress bar for a score."""
    st.markdown(
        f"""
        <div style="margin-bottom:6px">
            <span style="font-size:0.85rem;font-weight:600">{label}</span>
            <span style="float:right;font-size:0.85rem">{score:.1f}/100</span>
        </div>
        <div style="background:#e0e0e0;border-radius:6px;height:12px;margin-bottom:12px">
            <div style="width:{score}%;background:{color};height:12px;border-radius:6px"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grade_color(grade: str) -> str:
    return {
        "Excellent": "#2e7d32",
        "Good":      "#1565c0",
        "Average":   "#f57f17",
        "Needs Work":"#b71c1c",
    }.get(grade, "#555")


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🏏 Cricket Shot Quality Analyser")
st.caption(
    "Upload a batting video to classify the shot, analyse biomechanical quality "
    "using pose estimation, and optionally compare against a reference video."
)

if not POSE_AVAILABLE:
    st.warning(
        "⚠️ **MediaPipe not installed** — pose-based scoring is disabled. "
        "Install it with `pip install mediapipe==0.10.14` to enable full quality analysis."
    )

model = get_model()
feature_extractor = get_feature_extractor(model)

# ── Two-column layout ─────────────────────────────────────────────────────────
col_player, col_ref = st.columns(2)

with col_player:
    st.subheader("Player Video")
    player_upload = st.file_uploader(
        "Upload the player's shot video", type=["mp4", "avi", "mov"], key="player"
    )

with col_ref:
    st.subheader("Reference Video (optional)")
    ref_upload = st.file_uploader(
        "Upload a reference / ideal shot video for comparison",
        type=["mp4", "avi", "mov"],
        key="reference",
    )

# ── Analysis ──────────────────────────────────────────────────────────────────
if player_upload:
    player_path = save_upload(player_upload)

    with col_player:
        st.video(player_upload)

    with st.spinner("Classifying shot…"):
        player_frames = extract_frames(player_path, N_FRAMES)
        shot_name, confidence = predict_shot(model, player_frames)

    st.markdown("---")
    st.subheader(f"Shot Detected: **{shot_name.replace('_', ' ').title()}**  —  confidence {confidence:.1f}%")

    # ── Pose analysis ─────────────────────────────────────────────────────────
    avg_kp = {}
    pose_annotated_frame = None

    if POSE_AVAILABLE:
        with st.spinner("Running pose estimation…"):
            estimator = PoseEstimator()
            raw_frames = extract_raw_frames(player_path, max_frames=60)
            kp_per_frame = estimator.process_frames(raw_frames)
            avg_kp = aggregate_keypoints(kp_per_frame)

            # Annotate the middle frame for display
            mid = len(raw_frames) // 2
            if kp_per_frame[mid] is not None:
                pose_annotated_frame = estimator.draw_landmarks(
                    raw_frames[mid], kp_per_frame[mid]
                )

    # ── Similarity vs reference ───────────────────────────────────────────────
    similarity = -1.0
    ref_shot_name = None

    if ref_upload:
        ref_path = save_upload(ref_upload)

        with col_ref:
            st.video(ref_upload)

        with st.spinner("Analysing reference video…"):
            ref_frames = extract_frames(ref_path, N_FRAMES)
            ref_shot_name, ref_conf = predict_shot(model, ref_frames)

            p_feat = feature_extractor.predict(
                np.expand_dims(player_frames, axis=0), verbose=0
            )
            r_feat = feature_extractor.predict(
                np.expand_dims(ref_frames, axis=0), verbose=0
            )
            similarity = cosine_similarity(p_feat, r_feat)

    # ── Score ─────────────────────────────────────────────────────────────────
    feedback = SCORER.score(shot_name, avg_kp, similarity)

    # ── Results layout ────────────────────────────────────────────────────────
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        grade_col = grade_color(feedback.grade)
        st.markdown(
            f"""
            <div style="text-align:center;padding:20px;border-radius:12px;
                        background:#f5f5f5;margin-bottom:16px">
                <div style="font-size:3rem;font-weight:800;color:{grade_col}">
                    {feedback.final_score}
                </div>
                <div style="font-size:1.1rem;color:{grade_col};font-weight:600">
                    {feedback.grade}
                </div>
                <div style="font-size:0.8rem;color:#777;margin-top:4px">Overall Quality Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_score_bar("Biomechanical Score", feedback.pose_score, "#1565c0")
        if feedback.similarity_score >= 0:
            render_score_bar("Similarity to Reference", feedback.similarity_score, "#6a1b9a")
        render_score_bar("Final Score", feedback.final_score, grade_col)

    with res_col2:
        st.markdown("#### Criterion Breakdown")
        for criterion, score in feedback.criteria.items():
            color = "#4CAF50" if score >= 70 else ("#FF9800" if score >= 50 else "#f44336")
            render_score_bar(criterion.replace("_", " ").title(), score, color)

        if pose_annotated_frame is not None:
            st.markdown("#### Pose Overlay")
            rgb_annotated = cv2.cvtColor(pose_annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_annotated, caption="Pose keypoints on mid-frame", use_container_width=True)

    # ── Feedback tips ─────────────────────────────────────────────────────────
    st.markdown("#### 💡 Coaching Feedback")
    for tip in feedback.feedback:
        st.info(tip)

    if ref_upload and ref_shot_name:
        if ref_shot_name == shot_name:
            st.success(
                f"Reference video is also a **{ref_shot_name.replace('_',' ').title()}** "
                f"— similarity score: **{feedback.similarity_score:.1f}%**"
            )
        else:
            st.warning(
                f"⚠️ Shot type mismatch — player played **{shot_name}**, "
                f"reference is **{ref_shot_name}**. Similarity score may not be meaningful."
            )

    # ── Cleanup temp files ────────────────────────────────────────────────────
    try:
        os.unlink(player_path)
        if ref_upload:
            os.unlink(ref_path)
    except Exception:
        pass

else:
    st.info("👆 Upload a player video above to get started.")
