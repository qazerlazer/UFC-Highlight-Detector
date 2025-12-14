# UFC Highlight Detection

Detects UFC fight highlights using **audio-based machine learning**. Users can upload an audio or video clip, and the app predicts whether it contains a highlight based on crowd reaction and commentary energy.

---

## Description

This project uses a **Convolutional Neural Network (CNN)** trained on **mel spectrograms** extracted from UFC fight clips to classify moments as **highlight** or **non-highlight**. The pipeline converts audio to spectrograms, resizes them to a fixed input size, and feeds them to the CNN model. The model is deployed via **Streamlit**, allowing real-time predictions for uploaded clips. The CNN architecture and audio processing pipeline are designed to handle clips of **any duration**, with **~85% accuracy** on test data.

---

## Getting Started

### Dependencies

- **Python 3.10+**
- **TensorFlow / Keras**
- **Librosa**
- **NumPy**
- **MoviePy**
- **Streamlit**
- **Windows 10 / MacOS / Linux**

### Installing

1. **Clone this repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ufc-highlight-detector.git
cd ufc_crowd_classifier
```

2. **Install dependencies:**
```bash
pip install tensorflow librosa numpy moviepy streamlit
```

3. **Ensure folder structure matches:**
ufc_crowd_classifier/
├── data/                 # ignored in Git
├── clips_raw/            # ignored in Git
├── notebooks/train_model.ipynb
├── streamlit_app/app.py
└── README.md

---

## Executing Program

**Run the Streamlit app:**
```bash
cd streamlit_app
streamlit run app.py
```

### Step-by-step:
1. Upload a UFC clip (audio or video).
2. Audio is extracted (if video), converted to a **Mel spectrogram**, and resized to **128×128**.
3. Spectrogram is fed into the **CNN model**.
4. Prediction displayed: **Highlight (🔥)** or **Non-Highlight (😐)**.

---

## How It Works: Audio Pipeline
Audio File (any duration)
↓
Load with librosa
↓
Convert to Mel Spectrogram (128 × variable width)
↓
Resize to 128 × 128 (TensorFlow)
↓
Feed to CNN Model
↓
Prediction: Highlight or Non-Highlight

---

## CNN Architecture

**Input:** 128×128×1 mel spectrogram
Conv2D(32, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
↓
Conv2D(64, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
↓
Conv2D(128, 3×3) → ReLU → MaxPool(2×2) → BatchNorm
↓
Flatten → Dense(128) → ReLU → Dropout(0.5)
↓
Dense(64) → ReLU → Dropout(0.3)
↓
Dense(1) → Sigmoid → Output (0 or 1)

**Output:** 0 → Non-Highlight, 1 → Highlight

---


- If the app predicts everything as non-highlight, try using clips **~20 seconds long**.
- Close other programs using the clip files to avoid **file-locking errors** in Windows.
