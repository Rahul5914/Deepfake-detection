"""
Deepfake Detection System — Streamlit Web App
=============================================
Upload an image or video and get an instant REAL/FAKE prediction.
Model: MobileNetV2 (Transfer Learning) — loaded from .h5 file
"""

import os
import pickle
import tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ─────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
IMG_SIZE       = 224
MODEL_H5_PATH  = "deepfake_model.h5"
PICKLE_PATH    = "deepfake_model.pkl"
IMAGE_EXTS     = {"jpg", "jpeg", "png", "bmp"}
VIDEO_EXTS     = {"mp4", "avi", "mov", "mkv"}
MAX_FRAMES     = 30
FRAME_SKIP     = 10

# ─────────────────────────────────────────────────────────
#  CUSTOM CSS STYLING
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .result-fake {
        background: #ffe5e5;
        border: 2px solid #e53e3e;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #c53030;
    }
    .result-real {
        background: #e6ffed;
        border: 2px solid #38a169;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
        color: #276749;
    }
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    .metric-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  MODEL LOADING (CACHED — loads only once per session)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load the Keras model from .h5 file.
    Uses Streamlit's cache so model stays in memory across interactions.
    Falls back to pickle metadata if available.
    """
    if not os.path.exists(MODEL_H5_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_H5_PATH)

    # Load metadata from pickle if available
    metadata = None
    if os.path.exists(PICKLE_PATH):
        with open(PICKLE_PATH, "rb") as f:
            metadata = pickle.load(f)

    return model, metadata


# ─────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────
def preprocess_frame(frame_bgr):
    """Resize, convert to RGB, normalize to [0,1], add batch dimension."""
    rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    normed  = resized.astype(np.float32) / 255.0
    return np.expand_dims(normed, axis=0)


# ─────────────────────────────────────────────────────────
#  PREDICTION FUNCTIONS
# ─────────────────────────────────────────────────────────
def predict_image(model, pil_image):
    """
    Predict REAL or FAKE from a PIL Image.
    Returns: (label, confidence, raw_score)
    """
    img_array = np.array(pil_image.convert("RGB"))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    inp       = preprocess_frame(img_bgr)
    raw       = float(model.predict(inp, verbose=0)[0][0])
    label     = "FAKE" if raw >= 0.5 else "REAL"
    conf      = raw if label == "FAKE" else 1.0 - raw
    return label, conf, raw


def predict_video(model, video_path):
    """
    Predict REAL or FAKE from a video file using majority voting.
    Returns: (final_label, confidence, fake_count, real_count, frame_results, sample_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, 0, 0, [], []

    predictions, scores, sample_frames = [], [], []
    frame_count = 0

    progress = st.progress(0, text="Analyzing video frames...")

    while cap.isOpened() and len(predictions) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        inp   = preprocess_frame(frame)
        raw   = float(model.predict(inp, verbose=0)[0][0])
        label = "FAKE" if raw >= 0.5 else "REAL"
        predictions.append(label)
        scores.append(raw)

        if len(sample_frames) < 6:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sample_frames.append((rgb, label, raw))

        progress.progress(min(len(predictions) / MAX_FRAMES, 1.0),
                          text=f"Analyzing frame {len(predictions)}/{MAX_FRAMES}...")

    cap.release()
    progress.empty()

    if not predictions:
        return None, None, 0, 0, [], []

    fake_n = predictions.count("FAKE")
    real_n = predictions.count("REAL")
    avg    = float(np.mean(scores))
    final  = "FAKE" if fake_n > real_n else "REAL"
    conf   = avg if final == "FAKE" else 1.0 - avg

    return final, conf, fake_n, real_n, scores, sample_frames


def make_confidence_bar(confidence, label):
    """Create a matplotlib confidence bar chart as bytes."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    color = "#e53e3e" if label == "FAKE" else "#38a169"
    ax.barh(0, confidence * 100, color=color, height=0.5)
    ax.barh(0, 100, color="#e2e8f0", height=0.5, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Confidence %", fontsize=9)
    ax.set_title(f"{confidence:.1%}", fontsize=12, fontweight="bold", color=color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ About This App")
    st.markdown("""
    **Model:** MobileNetV2  
    **Type:** Binary Classifier  
    **Input:** Images & Videos  
    
    ---
    
    **How it works:**
    1. Upload image or video
    2. CNN extracts visual features
    3. Classifier outputs REAL/FAKE
    4. Videos use majority voting
    
    ---
    
    **Supported Formats:**  
    🖼️ Images: `.jpg`, `.png`, `.bmp`  
    🎬 Videos: `.mp4`, `.avi`, `.mov`
    
    ---
    
    **⚠️ Limitations:**  
    - Trained on synthetic data  
    - For real use, train on FaceForensics++  
    - No face-crop detection step
    """)

    st.markdown("---")
    st.markdown("**Video Settings**")
    max_frames  = st.slider("Max frames to analyze", 10, 50, MAX_FRAMES)
    frame_skip  = st.slider("Analyze every Nth frame", 5, 20, FRAME_SKIP)


# ─────────────────────────────────────────────────────────
#  MAIN UI
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 Deepfake Detection System</h1>
    <p>Upload an image or video — AI will tell you if it's REAL or FAKE</p>
</div>
""", unsafe_allow_html=True)

# Load model
model, pkl_meta = load_model()

if model is None:
    st.error("""
    ❌ **Model file not found!**
    
    Please make sure `deepfake_model.h5` is in the same folder as `app.py`.
    
    **Steps to fix:**
    1. Run the Colab notebook (Step 4) to train the model
    2. Download `deepfake_model.h5` from Colab (Step 9)
    3. Place it in this project folder
    """)
    st.stop()

# Show model info
if pkl_meta:
    col1, col2, col3 = st.columns(3)
    col1.metric("Val Accuracy", f"{pkl_meta.get('final_val_acc', 0):.2%}")
    col2.metric("Epochs Trained", pkl_meta.get('epochs_trained', 'N/A'))
    col3.metric("Image Size", f"{pkl_meta.get('img_size', 224)}×{pkl_meta.get('img_size', 224)}")
    st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload Image or Video",
    type=list(IMAGE_EXTS | VIDEO_EXTS),
    help="Supported: .jpg .png .bmp .mp4 .avi .mov .mkv"
)

if uploaded_file is not None:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    # ── IMAGE ──────────────────────────────────────────────
    if ext in IMAGE_EXTS:
        st.markdown("### 🖼️ Image Analysis")
        pil_img = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("🧠 Analyzing image..."):
                label, conf, raw = predict_image(model, pil_img)

            result_class = "result-fake" if label == "FAKE" else "result-real"
            icon = "🔴" if label == "FAKE" else "🟢"

            st.markdown(f"""
            <div class="{result_class}">
                {icon} {label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence Score")
            conf_bar = make_confidence_bar(conf, label)
            st.image(conf_bar, use_container_width=True)

            st.markdown(f"""
            <div class="info-box">
                <b>Raw Score:</b> {raw:.4f}<br>
                <b>Decision Threshold:</b> 0.5<br>
                <b>Interpretation:</b> Score > 0.5 → FAKE
            </div>
            """, unsafe_allow_html=True)

    # ── VIDEO ──────────────────────────────────────────────
    elif ext in VIDEO_EXTS:
        st.markdown("### 🎬 Video Analysis")
        st.video(uploaded_file)

        # Save to temp file for OpenCV processing
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("🧠 Processing video frames..."):
            final, conf, fake_n, real_n, scores, sample_frames = predict_video(
                model, tmp_path
            )
        os.unlink(tmp_path)

        if final is None:
            st.error("❌ Could not process video. Please try a different file.")
        else:
            total = fake_n + real_n
            result_class = "result-fake" if final == "FAKE" else "result-real"
            icon = "🔴" if final == "FAKE" else "🟢"

            st.markdown(f"""
            <div class="{result_class}">
                {icon} Video Prediction: {final}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Frame-by-Frame Analysis")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames Analyzed", total)
            col2.metric("FAKE Frames", fake_n)
            col3.metric("REAL Frames", real_n)
            col4.metric("Confidence", f"{conf:.1%}")

            # Score timeline
            if scores:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.plot(scores, color="#e53e3e", linewidth=1.5, alpha=0.8)
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
                ax.fill_between(range(len(scores)), scores, 0.5,
                                where=[s > 0.5 for s in scores],
                                alpha=0.3, color="#e53e3e", label="FAKE region")
                ax.fill_between(range(len(scores)), scores, 0.5,
                                where=[s <= 0.5 for s in scores],
                                alpha=0.3, color="#38a169", label="REAL region")
                ax.set_xlabel("Frame Sample Index")
                ax.set_ylabel("Fake Score")
                ax.set_ylim(0, 1)
                ax.set_title("Frame-level Deepfake Scores", fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            # Sample frames grid
            if sample_frames:
                st.markdown("#### Sample Frames")
                cols = st.columns(min(3, len(sample_frames)))
                for i, (frm, lbl, sc) in enumerate(sample_frames[:3]):
                    c = sc if lbl == "FAKE" else 1.0 - sc
                    cols[i].image(
                        cv2.resize(frm, (150, 150)),
                        caption=f"{lbl} ({c:.1%})",
                        use_container_width=True
                    )

else:
    # Landing state
    st.markdown("""
    <div class="info-box">
        👆 Upload an image or video above to get started.<br><br>
        <b>This app detects</b> whether an image/video has been AI-generated or face-swapped (deepfake).<br>
        Useful for verifying content on social media platforms like Twitter, Instagram, or YouTube.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🖼️ Image Detection
        - Upload `.jpg`, `.png`, `.bmp`
        - Single-shot CNN prediction
        - Shows confidence score
        """)
    with col2:
        st.markdown("""
        ### 🎬 Video Detection
        - Upload `.mp4`, `.avi`, `.mov`
        - Frame-by-frame analysis
        - Majority voting for final result
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#a0aec0; font-size:0.85em;'>"
    "Deepfake Detection System | MobileNetV2 | Built with Streamlit & TensorFlow"
    "</p>",
    unsafe_allow_html=True
)
