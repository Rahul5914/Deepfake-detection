# 🔍 Deepfake Detection System

A deep learning web application that detects whether an image or video is **REAL** or **AI-generated (FAKE / Deepfake)**.

Built with **MobileNetV2**, **TensorFlow**, and **Streamlit**.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 🚀 Live Demo

> Deploy on Streamlit Cloud: [streamlit.io/cloud](https://streamlit.io/cloud)

---

## 📁 Project Structure

```
deepfake-detection/
│
├── app.py                          # Main Streamlit application
├── deepfake_model.h5               # Trained Keras model weights ← YOU MUST ADD THIS
├── deepfake_model.pkl              # Model metadata (val accuracy, config)
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit theme & server config
├── Deepfake_Detection_With_Pickle.ipynb  # Google Colab training notebook
└── README.md                       # This file
```

---

## 🛠️ Setup Instructions

### Step 1: Train the Model (Google Colab)

1. Open `Deepfake_Detection_With_Pickle.ipynb` in Google Colab
2. Click **Runtime → Run all**
3. After training, run **Step 9** to download:
   - `deepfake_model.h5` (~14 MB)
   - `deepfake_model.pkl` (~1 KB)

### Step 2: Add Model Files to Repository

Place the downloaded files in the **root** of this project:

```
deepfake-detection/
├── deepfake_model.h5    ← place here
├── deepfake_model.pkl   ← place here
├── app.py
...
```

### Step 3: Deploy on Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your GitHub repo
5. Set **Main file path** to `app.py`
6. Click **Deploy!**

That's it — your app will be live in ~2 minutes!

---

## 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Open your browser at: `http://localhost:8501`

---

## 🧠 How It Works

```
User uploads image/video
        ↓
Preprocessing (resize 224×224, normalize)
        ↓
MobileNetV2 Feature Extraction (frozen, pretrained on ImageNet)
        ↓
Custom Classifier Head (Dense 128 → Dense 1 sigmoid)
        ↓
Score ≥ 0.5 → FAKE   |   Score < 0.5 → REAL
        ↓
For videos: majority vote across sampled frames
```

### Why MobileNetV2?
- Lightweight (~3.4M trainable params)
- Pretrained features detect textures and artifacts
- Fast inference — no GPU required for Streamlit deployment

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | MobileNetV2 + Custom Head |
| Input Size | 224 × 224 × 3 |
| Training Epochs | 5 (with EarlyStopping) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Binary Crossentropy |

---

## 📋 Supported Formats

| Type | Formats |
|------|---------|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp` |
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv` |

---

## ⚠️ Limitations

1. **Synthetic training data** — for production accuracy, retrain on [FaceForensics++](https://github.com/ondyari/FaceForensics)
2. **No face detection** — analyzes entire image, not just face region
3. **Video file size** — Streamlit Cloud allows up to 200 MB uploads
4. **GAN Note** — GANs *generate* deepfakes; this CNN *detects* them

---

## 🔮 Future Improvements

- [ ] Add MTCNN face detection preprocessing
- [ ] Fine-tune last layers of MobileNetV2
- [ ] Add Grad-CAM heatmap visualization
- [ ] Train on FaceForensics++ or DFDC dataset
- [ ] Add EfficientNetB4 option for higher accuracy
- [ ] Batch image analysis

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using TensorFlow, OpenCV, and Streamlit*
