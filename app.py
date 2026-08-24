"""
app.py  —  Cricket Shot Quality Analyser
Features:
  - Shot classification (r3d_18 + EfficientNetB0 fusion, 10 shot types)
  - Striker-only pose estimation → skeleton overlay
  - Joint angle computation at stance / impact / follow-through (metric 3D)
  - Shot quality score 0-100 with grade, derived from professional clips
  - You-vs-professionals comparison
  - Full pipeline data folder — frames, keypoints, videos — viewable + downloadable
  - Optional second video for similarity comparison
"""

import sys
import tempfile
import shutil
import os
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Try importing TensorFlow, if fails show error
TF_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    keras = tf.keras
    models = keras.models
    layers = keras.layers
    EfficientNetB0 = keras.applications.EfficientNetB0
    TF_AVAILABLE = True
except Exception as e:
    st.error(f"""
    **TensorFlow import failed**

    `{e}`

    This is caused by Windows Application Control blocking a DLL. Try adding
    the `venv` folder to Windows Defender exclusions, or run as Administrator.
    The skeleton and pose pipeline below still work without it.
    """, icon=":material/error:")

sys.path.insert(0, os.path.dirname(__file__))
from src.utils.video_utils import extract_frames, extract_raw_frames
from src.pose.estimator import (
    run_pose_on_frames, draw_skeleton, pose_summary,
    compute_cricket_angles, CRICKET_LANDMARKS,
)
from src.pose.striker_pose import run_striker_pose_on_video
from src.pipeline.builder import build_pipeline, zip_pipeline
from src.classifier.shot_predictor import ShotPredictor
from src.quality.scorer import score_shot

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Cricket Shot Analyser", page_icon=":material/sports_cricket:", layout="wide")

# Minimal line-icon set (Feather-style), used instead of emoji so the UI reads
# as one consistent visual language rather than mismatched platform emoji.
ICONS = {
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/></svg>',
    "trophy": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 4h10v5a5 5 0 0 1-10 0V4z"/><path d="M7 5H4a3 3 0 0 0 3 5"/><path d="M17 5h3a3 3 0 0 1-3 5"/><path d="M12 14v3"/><path d="M9 21h6"/><path d="M10 21v-2.5a2 2 0 0 1 4 0V21"/></svg>',
    "pose": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="5" r="2"/><line x1="12" y1="7" x2="12" y2="14"/><line x1="12" y1="9" x2="8" y2="12"/><line x1="12" y1="9" x2="16" y2="12"/><line x1="12" y1="14" x2="8" y2="20"/><line x1="12" y1="14" x2="16" y2="20"/></svg>',
    "list": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>',
    "film": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2.5" y="3.5" width="19" height="17" rx="2"/><line x1="7" y1="3.5" x2="7" y2="20.5"/><line x1="17" y1="3.5" x2="17" y2="20.5"/><line x1="2.5" y1="8.5" x2="7" y2="8.5"/><line x1="2.5" y1="15.5" x2="7" y2="15.5"/><line x1="17" y1="8.5" x2="21.5" y2="8.5"/><line x1="17" y1="15.5" x2="21.5" y2="15.5"/></svg>',
    "award": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M8.5 13.5 7 22l5-3 5 3-1.5-8.5"/></svg>',
    "folder": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6.5a2 2 0 0 1 2-2h4.5l2 2.5H19a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6.5z"/></svg>',
    "upload": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/><polyline points="7 9 12 4 17 9"/><line x1="12" y1="4" x2="12" y2="15"/></svg>',
    "users": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "sparkles": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M3 12h3M18 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/><circle cx="12" cy="12" r="3"/></svg>',
    "activity": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
}


def icon(name: str, size: int = 18) -> str:
    return (ICONS[name]
            .replace("<svg ", f"<svg width='{size}' height='{size}' "))


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');

:root {{
  --bg: #0A0E14;
  --bg-elev: #121821;
  --bg-elev-2: #19212C;
  --border: rgba(148,163,184,.14);
  --border-strong: rgba(148,163,184,.28);
  --text: #F1F5F9;
  --text-dim: #94A3B8;
  --text-faint: #5B6B82;
  --brand: #22D3EE;
  --brand-2: #A78BFA;
  --success: #34D399;
  --info: #60A5FA;
  --warning: #FBBF24;
  --danger: #F87171;
  --radius: 16px;
  --radius-sm: 10px;
}}

/* Font applied at body level (no !important) so it cascades normally —
   Streamlit's own icon elements (e.g. [data-testid="stIconMaterial"]) set a
   ligature icon-font directly on themselves with higher selector specificity,
   which must keep winning. An earlier !important blanket rule on span/div/
   button clobbered that and every icon rendered as literal text ("upload"
   instead of the glyph) instead of the icon. */
html, body, .stApp {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
h1, h2, h3, .csq-num, .csq-hero-title, .csq-section-title {{
  font-family: 'Space Grotesk', 'Inter', sans-serif;
}}

.stApp {{ background: radial-gradient(1200px 600px at 15% -10%, rgba(34,211,238,.08), transparent),
                       radial-gradient(1000px 500px at 100% 0%, rgba(167,139,250,.06), transparent),
                       var(--bg); }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
hr {{ border: none; height: 1px;
      background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
      margin: 2rem 0; }}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 8px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--brand); }}

/* ── Hero ─────────────────────────────────────────────────────────────── */
.csq-hero {{
  padding: 2.1rem 2.4rem; border-radius: 22px; margin-bottom: 1.6rem;
  background: linear-gradient(135deg, rgba(34,211,238,.10), rgba(167,139,250,.07) 60%, transparent);
  border: 1px solid var(--border);
  position: relative; overflow: hidden;
}}
.csq-hero::after {{
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(500px 220px at 90% -20%, rgba(34,211,238,.14), transparent 70%);
}}
.csq-hero-title {{
  font-size: 2.35rem; font-weight: 800; letter-spacing: -0.03em; margin: 0;
  background: linear-gradient(90deg, #F1F5F9 30%, var(--brand) 90%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  display: flex; align-items: center; gap: .55rem;
}}
.csq-hero-sub {{ color: var(--text-dim); font-size: .98rem; margin-top: .5rem; max-width: 640px; }}
.csq-trust-row {{ display: flex; gap: .5rem; margin-top: 1.1rem; flex-wrap: wrap; }}
.csq-trust {{
  display: inline-flex; align-items: center; gap: .4rem; font-size: .76rem; font-weight: 600;
  padding: .3rem .7rem; border-radius: 999px; color: var(--text-dim);
  background: var(--bg-elev); border: 1px solid var(--border);
}}
.csq-trust svg {{ color: var(--brand); flex-shrink: 0; }}

/* ── Section headers ──────────────────────────────────────────────────── */
.csq-section {{
  display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .3rem 0;
}}
.csq-section-icon {{
  width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center;
  justify-content: center; background: linear-gradient(135deg, rgba(34,211,238,.16), rgba(167,139,250,.12));
  color: var(--brand); flex-shrink: 0; border: 1px solid var(--border);
}}
.csq-section-title {{ font-size: 1.28rem; font-weight: 700; letter-spacing: -0.015em; color: var(--text); }}
.csq-section-sub {{ color: var(--text-dim); font-size: .87rem; margin: .15rem 0 1rem 44px; }}

/* ── Bordered containers (st.container(border=True)) as premium cards ──── */
[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: var(--radius) !important;
}}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {{
  transition: border-color .18s ease, transform .18s ease;
}}
[class*="st-key-stat_"] [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-upl_"] [data-testid="stVerticalBlockBorderWrapper"],
[class*="st-key-vid_"] [data-testid="stVerticalBlockBorderWrapper"] {{
  background: linear-gradient(180deg, var(--bg-elev), rgba(18,24,33,.7)) !important;
  border: 1px solid var(--border) !important;
}}
[class*="st-key-stat_"] [data-testid="stVerticalBlockBorderWrapper"]:hover {{
  border-color: var(--border-strong) !important;
}}

/* ── Stat cards ───────────────────────────────────────────────────────── */
.csq-stat-icon {{
  width: 30px; height: 30px; border-radius: 9px; display: flex; align-items: center;
  justify-content: center; margin-bottom: .55rem; border: 1px solid var(--border);
}}
.csq-stat-label {{
  font-size: .7rem; text-transform: uppercase; letter-spacing: .09em;
  color: var(--text-faint); font-weight: 700; margin-bottom: .2rem;
}}
.csq-num {{ font-size: 2rem; font-weight: 800; line-height: 1.1; color: var(--text); }}
.csq-num-sm {{ font-size: 1.05rem; }}
.csq-stat-sub {{ font-size: .78rem; color: var(--text-dim); margin-top: .3rem; }}

/* ── Pills / badges ───────────────────────────────────────────────────── */
.csq-pill {{
  display: inline-flex; align-items: center; gap: .35rem; padding: .28rem .7rem;
  border-radius: 999px; font-size: .76rem; font-weight: 600; border: 1px solid currentColor;
  background: color-mix(in srgb, currentColor 12%, transparent);
}}

/* ── Ranked bar list (top-3) ──────────────────────────────────────────── */
.csq-rank-row {{ margin-bottom: 9px; }}
.csq-rank-top {{ display: flex; justify-content: space-between; font-size: .82rem; margin-bottom: 3px; }}
.csq-rank-name {{ font-weight: 600; color: var(--text); }}
.csq-rank-pct {{ color: var(--text-dim); font-variant-numeric: tabular-nums; }}
.csq-rank-track {{ background: var(--bg-elev-2); height: 6px; border-radius: 4px; overflow: hidden; }}
.csq-rank-fill {{ height: 6px; border-radius: 4px; background: linear-gradient(90deg, var(--brand), var(--brand-2));
                  transition: width .5s cubic-bezier(.22,1,.36,1); }}

/* ── Score bars (quality criteria) ────────────────────────────────────── */
.csq-score-row {{ margin-bottom: 14px; }}
.csq-score-top {{ display: flex; justify-content: space-between; align-items: baseline; font-size: .87rem; }}
.csq-score-name {{ font-weight: 600; color: var(--text); }}
.csq-score-ideal {{ color: var(--text-faint); font-size: .78rem; margin-left: .4rem; }}
.csq-score-track {{ background: var(--bg-elev-2); border-radius: 7px; height: 10px; margin-top: 5px; overflow: hidden; }}
.csq-score-fill {{ height: 10px; border-radius: 7px; transition: width .6s cubic-bezier(.22,1,.36,1); }}
.csq-score-actual {{ font-size: .78rem; color: var(--text-dim); margin-top: 3px; }}

/* ── Pro-comparison bars ──────────────────────────────────────────────── */
.csq-pro-row {{
  margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid var(--border);
}}
.csq-pro-row:last-child {{ margin-bottom: 0; padding-bottom: 0; border-bottom: none; }}
.csq-pro-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .6rem; }}
.csq-pro-name {{ font-size: .95rem; font-weight: 700; color: var(--text); }}
.csq-pro-status {{
  font-size: .74rem; font-weight: 700; padding: .22rem .6rem; border-radius: 999px;
  white-space: nowrap;
}}
/* Track sits with extra top margin so the floating value bubble above the
   "you" marker has somewhere to go without clipping or overlapping the
   heading — the previous version crammed everything onto one text line
   ("you: X pros: Y (diff)"), which read as a wall of small grey numbers. */
.csq-pro-track {{
  position: relative; background: var(--bg-elev-2); border-radius: 9px; height: 12px;
  margin-top: 30px;
}}
.csq-pro-band {{ position: absolute; height: 12px; border-radius: 9px; background: rgba(52,211,153,.20); }}
.csq-pro-median {{ position: absolute; top: -4px; width: 2px; height: 20px; background: var(--text-faint); }}
.csq-pro-median-lbl {{
  position: absolute; top: -26px; transform: translateX(-50%); font-size: .68rem;
  color: var(--text-faint); white-space: nowrap; font-weight: 600;
}}
.csq-pro-mark {{
  position: absolute; top: -4px; width: 12px; height: 12px; margin-left: -6px;
  border-radius: 50%; border: 3px solid var(--bg-elev); box-shadow: 0 0 0 1px rgba(0,0,0,.3);
}}
.csq-pro-mark-lbl {{
  position: absolute; top: -26px; transform: translateX(-50%); font-size: .74rem;
  font-weight: 800; white-space: nowrap;
}}
.csq-pro-ends {{ display: flex; justify-content: space-between; margin-top: 6px; }}
.csq-pro-end {{ font-size: .7rem; color: var(--text-faint); font-variant-numeric: tabular-nums; }}
.csq-pro-legend {{ display: flex; gap: 1.4rem; margin-top: 1rem; flex-wrap: wrap; }}
.csq-pro-legend-item {{ display: flex; align-items: center; gap: .4rem; font-size: .78rem; color: var(--text-dim); }}
.csq-pro-legend-swatch {{ width: 12px; height: 12px; border-radius: 4px; flex-shrink: 0; }}
.csq-pro-legend-line {{ width: 12px; height: 2px; background: var(--text-faint); flex-shrink: 0; }}
.csq-pro-legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

/* ── Movement curves (shot-start → impact) ────────────────────────────── */
.csq-curve-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 14px;
}}
.csq-curve-card {{
  background: var(--bg-elev-2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: .9rem 1rem 1rem;
}}
.csq-curve-name {{ font-size: .88rem; font-weight: 700; color: var(--text); margin-bottom: .3rem; }}
.csq-curve-chart {{ width: 100%; display: block; }}

/* ── Verdict banner ───────────────────────────────────────────────────── */
.csq-verdict {{
  border-radius: var(--radius); padding: 1.3rem 1.6rem; margin-bottom: 1.2rem;
  border: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 1rem;
  background: linear-gradient(90deg, rgba(34,211,238,.08), transparent 60%);
}}

/* ── Streamlit native element restyling ───────────────────────────────── */
section[data-testid="stFileUploaderDropzone"] {{
  border-radius: var(--radius-sm); border: 1.5px dashed var(--border-strong) !important;
  background: var(--bg-elev) !important; transition: border-color .18s ease;
}}
section[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--brand) !important; }}

/* Constrain to the column width regardless of the source video's own pixel
   size. Streamlit measures each video's own aspect ratio in JS and writes
   the result as an INLINE height style on the <video> tag — for a portrait
   upload (e.g. a 720x1280 phone clip) sitting in a narrow half-width column,
   that produced a box far taller than the column, which read as the video
   rendering zoomed-in/cropped with a horizontal scrollbar to see the rest.
   An inline style beats an ordinary stylesheet rule, so this needs
   !important to actually win — and it targets the bare <video> tag (not a
   data-testid wrapper) because the wrapper's testid is not stable across
   Streamlit versions, while `st.video` is the only place this app puts a
   raw <video> element. */
video {{
  width: 100% !important; height: auto !important; max-height: 65vh !important;
  object-fit: contain !important; display: block; margin: 0 auto; background: #000;
  border-radius: var(--radius-sm); border: 1px solid var(--border);
}}

[data-testid="stTabs"] button[role="tab"] {{
  font-weight: 600; font-size: .87rem; border-radius: 8px 8px 0 0;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{ color: var(--brand) !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background-color: var(--brand) !important; }}

.stButton button, .stDownloadButton button {{
  border-radius: 10px !important; font-weight: 600 !important; border: 1px solid var(--border) !important;
  transition: transform .12s ease, border-color .12s ease;
}}
.stButton button:hover, .stDownloadButton button:hover {{
  border-color: var(--brand) !important; transform: translateY(-1px);
}}
.stDownloadButton button[kind="primary"] {{
  background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important;
  color: #05121A !important; border: none !important;
}}

div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
div[data-testid="stDataFrame"] {{ border-radius: var(--radius-sm); overflow: hidden; border: 1px solid var(--border); }}

.csq-caption {{ color: var(--text-dim); font-size: .85rem; line-height: 1.55; }}
.csq-mono {{ font-family: 'Space Grotesk', monospace; }}
</style>
""", unsafe_allow_html=True)


def section(name: str, ic: str, title: str, sub: str = ""):
    """Render a consistent icon + title section header, in the given container."""
    with name:
        st.markdown(
            f"<div class='csq-section'><div class='csq-section-icon'>{icon(ic, 18)}</div>"
            f"<div class='csq-section-title'>{title}</div></div>"
            + (f"<div class='csq-section-sub'>{sub}</div>" if sub else ""),
            unsafe_allow_html=True,
        )


def stat_card(container, ic: str, label: str, value: str, sub: str = "",
              color: str = "var(--text)", key: str = ""):
    with container:
        with st.container(border=True, key=key or None):
            st.markdown(
                f"<div class='csq-stat-icon' style='color:{color}'>{icon(ic, 16)}</div>"
                f"<div class='csq-stat-label'>{label}</div>"
                f"<div class='csq-num' style='color:{color}'>{value}</div>"
                + (f"<div class='csq-stat-sub'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True,
            )


# ── Constants ─────────────────────────────────────────────────────────────────
SHOT_CLASSES = {
    "cover": 0, "defense": 1, "flick": 2, "hook": 3, "late_cut": 4,
    "lofted": 5, "pull": 6, "square_cut": 7, "straight": 8, "sweep": 9,
}
IDX_TO_CLASS = {v: k for k, v in SHOT_CLASSES.items()}

GRADE_COLOR = {
    "Excellent": "#34D399",
    "Good":      "#60A5FA",
    "Average":   "#FBBF24",
    "Needs Work":"#F87171",
}
GRADE_BADGE = {
    "Excellent": "green", "Good": "blue", "Average": "orange", "Needs Work": "red",
}


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not TF_AVAILABLE:
        return None
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
    """Render a single quality criterion as a labelled, coloured progress bar."""
    color = "var(--success)" if score >= 80 else ("var(--warning)" if score >= 55 else "var(--danger)")
    st.markdown(
        f"""
        <div class='csq-score-row'>
          <div class='csq-score-top'>
            <span><span class='csq-score-name'>{label}</span>
              <span class='csq-score-ideal'>ideal {ideal}</span></span>
            <span style='color:{color}'>{status} <b>{score:.0f}</b>/100</span>
          </div>
          <div class='csq-score-track'>
            <div class='csq-score-fill' style='width:{score}%; background:{color}'></div>
          </div>
          <div class='csq-score-actual'>Measured: {actual}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def curve_chart_svg(user: list, pro_low: list, pro_median: list, pro_high: list,
                    width: int = 480, height: int = 160) -> str:
    """
    One angle plotted from shot-start (left) to impact (right): the
    professional IQR as a shaded band, the professional median as a dashed
    line, this clip as a solid line — the movement version of the single-
    point bar above it. Hand-rolled SVG rather than a charting library so it
    matches the rest of the app's design system exactly (same CSS vars,
    same dark theme, no extra dependency).

    Any of the three professional series may contain None at a given
    time-step (fewer than MIN_CLIPS professional clips had data there); the
    band/median are drawn as separate path segments so a gap doesn't get
    bridged with a straight line across missing data.
    """
    n = len(user)
    pad_l, pad_r, pad_t, pad_b = 6, 6, 22, 20
    plot_w = max(width - pad_l - pad_r, 1)
    plot_h = max(height - pad_t - pad_b, 1)

    all_vals = [v for series in (user, pro_low, pro_median, pro_high)
               for v in series if v is not None]
    if not all_vals or n < 2:
        return ""
    y_lo, y_hi = min(all_vals), max(all_vals)
    span = max(y_hi - y_lo, 1.0) * 1.24
    mid = (y_hi + y_lo) / 2
    y_lo, y_hi = mid - span / 2, mid + span / 2

    def xpix(i):
        return pad_l + (i / (n - 1)) * plot_w

    def ypix(v):
        return pad_t + (1 - (v - y_lo) / span) * plot_h

    def segments(vals):
        segs, cur = [], []
        for i, v in enumerate(vals):
            if v is None:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
                continue
            cur.append((xpix(i), ypix(v)))
        if len(cur) > 1:
            segs.append(cur)
        return segs

    def path_d(pts):
        return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    band_top = [(xpix(i), ypix(pro_high[i])) for i in range(n)
               if pro_low[i] is not None and pro_high[i] is not None]
    band_bot = [(xpix(i), ypix(pro_low[i])) for i in range(n)
               if pro_low[i] is not None and pro_high[i] is not None]

    parts = [f"<svg class='csq-curve-chart' viewBox='0 0 {width} {height}' "
            f"preserveAspectRatio='none' height='{height}'>"]

    if band_top:
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in band_top + band_bot[::-1]) + " Z"
        parts.append(f"<path d='{d}' fill='rgba(52,211,153,.16)' stroke='none'/>")

    for seg in segments(pro_median):
        parts.append(f"<path d='{path_d(seg)}' fill='none' stroke='var(--text-faint)' "
                     f"stroke-width='1.4' stroke-dasharray='3,3'/>")

    for seg in segments(user):
        parts.append(f"<path d='{path_d(seg)}' fill='none' stroke='var(--brand)' "
                     f"stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>")

    if user[0] is not None:
        x0, y0 = xpix(0), ypix(user[0])
        parts.append(f"<circle cx='{x0:.1f}' cy='{y0:.1f}' r='3.5' fill='var(--bg-elev-2)' "
                     f"stroke='var(--brand)' stroke-width='2'/>")
        parts.append(f"<text x='{x0:.1f}' y='{max(y0 - 9, 11):.1f}' font-size='11' "
                     f"font-weight='700' fill='var(--brand)' text-anchor='start'>"
                     f"{user[0]:.0f}°</text>")
    if user[-1] is not None:
        x1, y1 = xpix(n - 1), ypix(user[-1])
        parts.append(f"<circle cx='{x1:.1f}' cy='{y1:.1f}' r='4.5' fill='var(--brand)' "
                     f"stroke='var(--bg-elev-2)' stroke-width='2'/>")
        parts.append(f"<text x='{x1:.1f}' y='{max(y1 - 10, 11):.1f}' font-size='11' "
                     f"font-weight='700' fill='var(--brand)' text-anchor='end'>"
                     f"{user[-1]:.0f}°</text>")

    parts.append(f"<text x='{pad_l}' y='{height - 5}' font-size='9.5' "
                 f"fill='var(--text-faint)' text-anchor='start'>Shot start</text>")
    parts.append(f"<text x='{width - pad_r}' y='{height - 5}' font-size='9.5' "
                 f"fill='var(--text-faint)' text-anchor='end'>Impact</text>")
    parts.append("</svg>")
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class='csq-hero'>
      <div class='csq-hero-title'>Cricket Shot Quality Analyser</div>
      <div class='csq-hero-sub'>Upload a batting clip — filmed from behind the
        bowler's arm — for striker-only pose tracking, shot classification,
        biomechanical scoring against professional benchmarks, and the full
        analysis folder to download.</div>
      <div class='csq-trust-row'>
        <span class='csq-trust'>{icon('target',13)} 62.4% top-1 · 81% top-3 shot accuracy</span>
        <span class='csq-trust'>{icon('pose',13)} Striker-only skeleton tracking</span>
        <span class='csq-trust'>{icon('award',13)} Scored vs professional clips</span>
        <span class='csq-trust'>{icon('folder',13)} Full pipeline export</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not TF_AVAILABLE:
    st.warning("TensorFlow not available — showing skeleton extraction only. "
               "Shot classification and quality scoring are disabled.", icon=":material/warning:")

model = load_model() if TF_AVAILABLE else None

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
section(st.container(), "upload", "Upload", "Player clip required; a reference clip is optional.")

col_up1, col_up2 = st.columns(2)
with col_up1:
    with st.container(border=True, key="upl_1"):
        st.markdown(f"**{icon('film',15)} Player video**", unsafe_allow_html=True)
        v1 = st.file_uploader("Upload batting video", type=["mp4", "avi", "mov"],
                              key="v1", label_visibility="collapsed")
with col_up2:
    with st.container(border=True, key="upl_2"):
        st.markdown(f"**{icon('users',15)} Reference video (optional)**", unsafe_allow_html=True)
        v2 = st.file_uploader("Upload reference / ideal shot for comparison",
                              type=["mp4", "avi", "mov"], key="v2", label_visibility="collapsed")

# Sample clips, so the whole pipeline can be exercised without hunting for a
# file — and so a known shot type is available to sanity-check against.
SAMPLE_ROOT = Path(__file__).resolve().parent / "data"
SAMPLES = {}
if SAMPLE_ROOT.exists():
    for cls_dir in sorted(p for p in SAMPLE_ROOT.iterdir() if p.is_dir()):
        for i in range(1, 6):
            f = cls_dir / f"video{i}.mp4"
            if f.exists():
                SAMPLES[f"{cls_dir.name} · video{i}"] = f

sample_choice = None
if SAMPLES and not v1:
    st.markdown("<div style='margin-top:.6rem'></div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        sample_choice = st.selectbox(
            "…or try a sample clip", ["—"] + list(SAMPLES))
    if sample_choice == "—":
        sample_choice = None
    else:
        with sc2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            st.caption(f"True shot type: **{sample_choice.split(' · ')[0]}** — "
                       "note these clips are in the classifier's training data, so "
                       "getting them right is not evidence of accuracy.")

if not v1 and not sample_choice:
    st.info("Upload a video, or pick a sample clip, to start the analysis.", icon=":material/movie:")
    st.stop()

# ── Save and display ──────────────────────────────────────────────────────────
if v1:
    p1 = save_upload(v1)
    source_name, source_size = v1.name, v1.size
else:
    p1 = str(SAMPLES[sample_choice])
    source_name = Path(p1).parent.name + "_" + Path(p1).stem + ".mp4"
    source_size = Path(p1).stat().st_size
with col_up1:
    st.video(p1)

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# One call does the whole analysis AND writes every artefact to a pipeline
# folder for this upload. Everything shown below is read back out of that
# single result, so what the user downloads cannot disagree with what they
# were shown, and the ~90s of work is never paid for twice.
PIPELINE_ROOT = Path(tempfile.gettempdir()) / "cricket_pipelines"


@st.cache_resource(show_spinner=False)
def load_predictor():
    """
    The r3d_18 + EfficientNetB0 classifier (62.4% test top-1 vs the shipped
    model's 57.6%). Returns None if its head has not been trained yet, in
    which case the app falls back to the shipped weights rather than failing.
    """
    p = ShotPredictor()
    return p if p.available else None


@st.cache_data(show_spinner=False)
def analyse(name: str, size: int, model_ready: bool, _video_path: str):
    """
    Cached on the upload's name and size. The temp file path changes on every
    rerun, so keying on it would miss the cache every time and re-run the full
    analysis whenever the user touched a widget.
    """
    out_dir = PIPELINE_ROOT / f"{Path(name).stem}_pipeline"
    predictor = load_predictor() if model_ready else None
    return build_pipeline(
        _video_path, out_dir,
        predictor=predictor,
        classifier=None if predictor else (load_model() if model_ready else None),
        idx_to_class=IDX_TO_CLASS if model_ready else None,
        scorer=score_shot if model_ready else None,
        max_frames=30,
        display_name=name,
    )


with st.spinner("Analysing video… (striker tracking ~40s, "
                "shot classification ~5s)"):
    pipe = analyse(source_name, source_size, model is not None, p1)

frames1 = pipe["_clf_frames"]
raw_frames = pipe["_frames"]
frames_kp = pipe["_keypoints"]
summary = pipe["_summary"]
angles = pipe["joint_angles"]
quality = pipe["_quality"]

if pipe["prediction"]:
    shot1 = pipe["prediction"]["shot"]
    conf1 = pipe["prediction"]["confidence"]
    top3 = [(t["shot"], t["confidence"]) for t in pipe["prediction"]["top3"]]
else:
    shot1, conf1, top3 = "Unknown (TF unavailable)", 0.0, []

pipe_dir = Path(pipe["out_dir"])

# ══════════════════════════════════════════════════════════════════════════════
# VERDICT BANNER
# ══════════════════════════════════════════════════════════════════════════════
if not pipe["striker_found"]:
    st.error(
        "**Striker not found in this clip.** Nothing was drawn — the model "
        "returns nothing rather than putting a skeleton on the umpire or the "
        "bowler. This works on the standard broadcast angle filmed from "
        "behind the bowler's arm, with the batsman on strike at the far stumps.",
        icon=":material/block:",
    )
else:
    gc = GRADE_COLOR.get(quality.grade, "var(--text-dim)") if quality else "var(--text-dim)"
    st.markdown(
        f"""
        <div class='csq-verdict'>
          <div>
            <div style='font-size:.78rem; color:var(--text-faint); text-transform:uppercase;
                        letter-spacing:.08em; font-weight:700; margin-bottom:.3rem'>Predicted shot</div>
            <div style='font-size:1.9rem; font-weight:800'>{shot1.replace('_',' ').title()}
              <span style='font-size:1rem; font-weight:600; color:var(--text-dim)'>{conf1:.0f}% confidence</span>
            </div>
          </div>
          <div style='text-align:right'>
            <div style='font-size:.78rem; color:var(--text-faint); text-transform:uppercase;
                        letter-spacing:.08em; font-weight:700; margin-bottom:.3rem'>Shot quality</div>
            <div style='font-size:1.9rem; font-weight:800; color:{gc}'>
              {f"{quality.overall_score:.0f}" if quality else "—"}<span style='font-size:1rem;color:var(--text-dim)'>/100</span>
              {f"<span class='csq-pill' style='color:{gc}; margin-left:.5rem; vertical-align:middle'>{quality.grade}</span>" if quality else ""}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if pipe.get("prediction", {}).get("model"):
    st.caption(f"Classifier: {pipe['prediction']['model']} — figures are top-1 "
               f"/ top-3 on the dataset's held-out test split, not on this clip.")
if conf1 and conf1 < 60:
    st.warning("Low confidence. The quality criteria below are chosen by the "
               "predicted shot type, so if the shot name is wrong the score is "
               "measuring against the wrong technique.", icon=":material/warning:")

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — STAT ROW
# ══════════════════════════════════════════════════════════════════════════════
section(st.container(), "target", "At a glance")

r1, r2, r3, r4 = st.columns(4)
stat_card(r1, "target", "Predicted shot", shot1.replace('_', ' ').title(),
          f"{conf1:.0f}% confidence", key="stat_1")

if quality is not None:
    gc = GRADE_COLOR.get(quality.grade, "var(--text-dim)")
    stat_card(r2, "trophy", "Shot quality", f"{quality.overall_score:.0f}/100",
              key="stat_2", color=gc)
    with r2:
        st.markdown(f"<div style='margin-top:-14px'><span class='csq-pill' "
                    f"style='color:{gc}'>{quality.grade}</span></div>",
                    unsafe_allow_html=True)
else:
    stat_card(r2, "trophy", "Shot quality", "—", key="stat_2")

det = pipe["detection_rate"]
dc = "var(--success)" if det > 80 else ("var(--warning)" if det > 50 else "var(--danger)")
stat_card(r3, "pose", "Skeleton coverage", f"{det:.0f}%",
          f"{pipe['detected_frames']}/{pipe['analysed_frames']} frames · "
          f"{pipe['avg_joints_per_frame']}/13 joints", color=dc, key="stat_3")

with r4:
    with st.container(border=True, key="stat_4"):
        st.markdown(
            f"<div class='csq-stat-icon'>{icon('list',16)}</div>"
            f"<div class='csq-stat-label'>Top 3 predictions</div>",
            unsafe_allow_html=True)
        for name, prob in top3:
            st.markdown(
                f"""
                <div class='csq-rank-row'>
                  <div class='csq-rank-top'>
                    <span class='csq-rank-name'>{name.replace('_',' ').title()}</span>
                    <span class='csq-rank-pct'>{prob:.0f}%</span>
                  </div>
                  <div class='csq-rank-track'>
                    <div class='csq-rank-fill' style='width:{prob}%'></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — VIDEOS
# ══════════════════════════════════════════════════════════════════════════════
section(st.container(), "film", "Videos",
       "Skeleton is drawn only on the identified striker.")

vc1, vc2 = st.columns(2)
with vc1:
    with st.container(border=True, key="vid_1"):
        st.markdown("**Side by side — original vs skeleton**")
        cmp_video = pipe.get("comparison_video")
        if cmp_video and Path(cmp_video).exists():
            st.video(cmp_video)
            st.caption("Easiest way to check the skeleton is on the right player.")
        else:
            st.info("Not available for this clip.")

with vc2:
    with st.container(border=True, key="vid_2"):
        st.markdown("**Skeleton only**")
        sk_video = pipe.get("skeleton_video")
        if sk_video and Path(sk_video).exists():
            st.video(sk_video)
            st.caption("Frames marked *no striker on camera* are replays, crowd "
                       "or wide shots — left clean deliberately.")
        else:
            st.info("Not available for this clip.")

dl1, dl2, dl3 = st.columns(3)
if sk_video and Path(sk_video).exists():
    with dl1, open(sk_video, "rb") as fh:
        st.download_button("⬇ Skeleton video (.mp4)", fh.read(),
                           file_name=Path(sk_video).name, mime="video/mp4",
                           use_container_width=True)
if cmp_video and Path(cmp_video).exists():
    with dl2, open(cmp_video, "rb") as fh:
        st.download_button("⬇ Comparison video (.mp4)", fh.read(),
                           file_name=Path(cmp_video).name, mime="video/mp4",
                           use_container_width=True)


@st.cache_data(show_spinner=False)
def pipeline_zip(folder: str) -> tuple[str, bytes]:
    """
    Archive bytes for a built pipeline folder, built once per clip.

    Streamlit re-runs the whole script on every widget interaction, so calling
    zip_pipeline inline rebuilt a 13 MB archive each time the user moved a
    slider or switched a tab.
    """
    p = zip_pipeline(Path(folder))
    return p.name, p.read_bytes()

zip_name, zip_bytes = pipeline_zip(str(pipe_dir))
with dl3:
    st.download_button("⬇ All data (.zip)", zip_bytes, file_name=zip_name,
                       mime="application/zip", type="primary",
                       use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — QUALITY BREAKDOWN + JOINT ANGLES
# ══════════════════════════════════════════════════════════════════════════════
q_col, a_col = st.columns([1, 1])

with q_col:
    section(st.container(), "trophy", f"Quality breakdown — {shot1.replace('_',' ').title()}")
    with st.container(border=True):
        if quality is not None and quality.criteria:
            for c in quality.criteria:
                score_bar(c.name, c.score, c.ideal, c.actual, c.status)
        else:
            st.info("No quality criteria available for this shot type.")

with a_col:
    phases = pipe.get("phases")
    impact_f = pipe.get("impact_frame")
    section(st.container(), "pose", "Joint angles")
    with st.container(border=True):
        hnd = pipe.get("handedness") or {}
        if hnd and not hnd.get("assumed"):
            st.badge(f"{hnd['hand'].title()}-handed batsman "
                     f"({hnd['confidence']*100:.0f}% confidence)",
                     color="blue")
            st.caption("'Front' below means the front leg/arm for this handedness "
                       "— front is the left side for a right-hander, the right "
                       "side for a left-hander.")
        elif hnd.get("assumed"):
            st.caption("Could not detect handedness from this clip — "
                       "front/back assumed right-handed.")

        st.caption(
            f"Measured at the impact frame (frame {impact_f}) — the moment of peak "
            f"hand speed — from MediaPipe world landmarks, which are metric 3D. "
            f"Reading them off the 2D image instead makes every knee look straight "
            f"on this camera angle."
            if phases and phases.get("source") == "world" else
            "Measured from image coordinates — knee angles are unreliable here."
        )

        if phases:
            labels = {
                "front_knee_angle": "Front knee", "back_knee_angle": "Back knee",
                "front_elbow_angle": "Front elbow", "back_elbow_angle": "Back elbow",
                "shoulder_tilt_deg": "Shoulder tilt", "hip_tilt_deg": "Hip tilt",
                "trunk_lean_deg": "Trunk lean",
            }
            rows = []
            for key, label in labels.items():
                def fmt(phase):
                    v = phases[phase].get(key)
                    return "—" if v is None else f"{v:.0f}°"
                rows.append({"Angle": label, "Stance": fmt("stance"),
                             "Impact": fmt("impact"),
                             "Follow-through": fmt("follow_through")})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No angles — the striker was not found.")

st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — YOU VS PROFESSIONALS
# ══════════════════════════════════════════════════════════════════════════════
vs_pro = pipe.get("pro_comparison") or []
if vs_pro:
    n_pro = vs_pro[0].get("n_pro_clips")
    section(st.container(), "award", f"You vs professionals — {shot1.replace('_',' ').title()}",
           f"Each angle at impact against the median of {n_pro} professional "
           f"clips of this shot in the training data. The shaded band is the "
           f"range half of those professionals fall inside.")
    with st.container(border=True):
        for c in vs_pro:
            low, high, med, actual = c["low"], c["high"], c["pro_median"], c["actual"]
            span_lo = min(low, actual, med) - 15
            span_hi = max(high, actual, med) + 15
            span = max(span_hi - span_lo, 1e-6)
            band_l = (low - span_lo) / span * 100
            band_w = (high - low) / span * 100
            actual_pct = (actual - span_lo) / span * 100
            med_pct = (med - span_lo) / span * 100
            colour = "var(--success)" if c["in_range"] else "var(--warning)"
            status = ("Within typical range" if c["in_range"] else
                      f"{abs(c['diff']):.0f}° {'more' if c['diff'] > 0 else 'less'} than typical")
            # Value labels float above the marker/median tick rather than being
            # crammed into one line of small text next to the bar — the eye
            # goes straight from a number to the exact point it describes.
            st.markdown(
                f"""
                <div class='csq-pro-row'>
                  <div class='csq-pro-head'>
                    <span class='csq-pro-name'>{c['label']}</span>
                    <span class='csq-pro-status' style='color:{colour}; background:color-mix(in srgb, {colour} 16%, transparent)'>{status}</span>
                  </div>
                  <div class='csq-pro-track'>
                    <div class='csq-pro-mark-lbl' style='left:{actual_pct:.1f}%; color:{colour}'>You {actual:.0f}°</div>
                    <div class='csq-pro-median-lbl' style='left:{med_pct:.1f}%'>Pros {med:.0f}°</div>
                    <div class='csq-pro-band' style='left:{band_l:.1f}%; width:{band_w:.1f}%'></div>
                    <div class='csq-pro-median' style='left:{med_pct:.1f}%'></div>
                    <div class='csq-pro-mark' style='left:{actual_pct:.1f}%; background:{colour}'></div>
                  </div>
                  <div class='csq-pro-ends'>
                    <span class='csq-pro-end'>{low:.0f}°</span>
                    <span class='csq-pro-end'>typical range</span>
                    <span class='csq-pro-end'>{high:.0f}°</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class='csq-pro-legend'>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-swatch'
                style='background:rgba(52,211,153,.35)'></span>Typical professional range</span>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-line'></span>Professional median</span>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-dot'
                style='background:var(--success)'></span>This clip, within range</span>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-dot'
                style='background:var(--warning)'></span>This clip, outside range</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — SHOT MOVEMENT VS PROFESSIONALS
# ══════════════════════════════════════════════════════════════════════════════
angle_curves = pipe.get("angle_curve") or {}
if angle_curves:
    any_curve = next(iter(angle_curves.values()))
    n_pro = any_curve.get("n_pro_clips")
    section(st.container(), "activity", f"Shot movement — {shot1.replace('_',' ').title()}",
           f"Each angle traced from the last still moment before the swing "
           f"through to impact, resampled onto the same timeline as "
           f"{n_pro} professional clips of this shot — so the whole "
           f"movement is compared, not just the impact instant above.")
    with st.container(border=True):
        cards = []
        for key, c in angle_curves.items():
            chart = curve_chart_svg(c["user"], c["pro_low"], c["pro_median"], c["pro_high"])
            if not chart:
                continue
            cards.append(
                f"<div class='csq-curve-card'>"
                f"<div class='csq-curve-name'>{c['label']}</div>{chart}</div>"
            )
        st.markdown(f"<div class='csq-curve-grid'>{''.join(cards)}</div>",
                   unsafe_allow_html=True)
        st.markdown(
            """
            <div class='csq-pro-legend' style='margin-top:1.1rem'>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-swatch'
                style='background:rgba(52,211,153,.35)'></span>Typical professional range</span>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-line'
                style='border-top:1.5px dashed var(--text-faint); background:none; height:0'></span>Professional median</span>
              <span class='csq-pro-legend-item'><span class='csq-pro-legend-dot'
                style='background:var(--brand)'></span>This clip</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — REFERENCE COMPARISON (optional)
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

    section(st.container(), "users", "Reference comparison")
    with st.container(border=True):
        col_up2.success(f"**{shot2.replace('_',' ').title()}** — {conf2:.1f}%")
        if shot1 == shot2:
            st.success(
                f"Both videos are **{shot1.replace('_',' ').title()}** — "
                f"Visual similarity: **{sim:.1f}%**", icon=":material/check_circle:"
            )
        else:
            st.warning(
                f"Player: **{shot1}** vs Reference: **{shot2}**. "
                f"Similarity: {sim:.1f}% (different shots — comparison limited).",
                icon=":material/warning:"
            )

    try:
        os.unlink(p2)
    except Exception:
        pass
    st.markdown("<hr/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — PIPELINE DATA FOLDER
# ══════════════════════════════════════════════════════════════════════════════
section(st.container(), "folder", "Pipeline data",
       f"Every input and output for this clip, in the same layout as "
       f"data/&lt;class&gt;/videoN_pipeline/. Analysed in "
       f"{sum(pipe['timings'].values()):.0f}s — "
       + " · ".join(f"{k.replace('_',' ')} {v}s" for k, v in pipe["timings"].items()))

tab_frames, tab_files, tab_json = st.tabs(["Frames", "Files", "Raw output"])

with tab_frames:
    which = st.segmented_control(
        "Frame set", ["Comparison (original | skeleton)",
                     "Skeleton overlay", "Extracted (model input)"],
        default="Comparison (original | skeleton)", label_visibility="collapsed")
    folder = {"Comparison (original | skeleton)": "04_comparison_frames",
              "Skeleton overlay": "03_skeleton_overlay_frames",
              "Extracted (model input)": "01_extracted_frames"}[which or "Comparison (original | skeleton)"]
    imgs = sorted((pipe_dir / folder).glob("*.jpg")) if (pipe_dir / folder).exists() else []
    if not imgs:
        st.info("No frames in this set.")
    else:
        idx = st.slider(f"Frame — {len(imgs)} total", 1, len(imgs), 1)
        st.image(str(imgs[idx - 1]), use_column_width=True,
                 caption=f"{folder}/{imgs[idx-1].name}")

with tab_files:
    rows = []
    for p in sorted(pipe_dir.rglob("*")):
        rel = p.relative_to(pipe_dir)
        rows.append({
            "File": f"{rel}/" if p.is_dir() else str(rel),
            "Type": "folder" if p.is_dir() else (p.suffix.lstrip(".") or "—"),
            "Size": (f"{len(list(p.glob('*')))} files" if p.is_dir()
                     else f"{p.stat().st_size/1024:,.0f} KB"),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=330)
    st.download_button("⬇ Download this folder (.zip)", zip_bytes,
                       file_name=zip_name, mime="application/zip",
                       key="zip_files_tab")

with tab_json:
    j1, j2 = st.columns(2)
    for col, fname, label in ((j1, "00_metadata.json", "Metadata"),
                              (j2, "05_shot_analysis.json", "Shot analysis")):
        with col:
            st.markdown(f"**{label}** — `{fname}`")
            text = (pipe_dir / fname).read_text(encoding="utf-8")
            st.json(json.loads(text))
            st.download_button(f"⬇ {fname}", text, file_name=fname,
                               mime="application/json", key=f"dl_{fname}")

    st.markdown("**Pipeline summary** — `PIPELINE_SUMMARY.txt`")
    summary_txt = (pipe_dir / "PIPELINE_SUMMARY.txt").read_text(encoding="utf-8")
    st.code(summary_txt)
    st.download_button("⬇ PIPELINE_SUMMARY.txt", summary_txt,
                       file_name="PIPELINE_SUMMARY.txt", key="dl_summary")

    kp_file = pipe_dir / "02_skeleton_keypoints.json"
    if kp_file.exists():
        st.markdown("**Skeleton keypoints** — `02_skeleton_keypoints.json` "
                    f"({kp_file.stat().st_size/1024:,.0f} KB, all 33 landmarks "
                    "per frame)")
        with open(kp_file, "rb") as fh:
            st.download_button("⬇ 02_skeleton_keypoints.json", fh.read(),
                               file_name=kp_file.name, mime="application/json",
                               key="dl_kp")

st.caption(f"Folder on disk: `{pipe_dir}`")

# ── Cleanup ───────────────────────────────────────────────────────────────────
# The pipeline folder is kept — it is the deliverable. Only the raw upload
# temp file goes. A sample clip is a real file in data/ and must survive.
if v1:
    try:
        os.unlink(p1)
    except Exception:
        pass
