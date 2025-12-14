import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import os

MODEL_PATH = "ufc_highlight_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def audio_to_melspectrogram(uploaded_file):
    

    # If video, extract audio
    if uploaded_file.type.startswith("video"):
        # Save temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        audio_clip = AudioFileClip(tmp_path)
        audio_path = tmp_path.replace(".mp4", ".wav")
        audio_clip.write_audiofile(audio_path, verbose=False, logger=None)
        uploaded_file = audio_path  # replace with audio file for librosa

    # Load audio with librosa
    y, sr = librosa.load(uploaded_file, sr=None)  
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Resize to 128x128 for CNN
    S_resized = tf.image.resize(S_dB[..., np.newaxis], (128, 128))
    return S_resized.numpy()

# ------------------ STREAMLIT APP ------------------
st.title("UFC Highlight Detector")
st.write(
    "Upload a UFC clip of any length (20 seconds recommended). "
    "The model will classify it based on **crowd reaction + audio energy**."
)

uploaded = st.file_uploader("Upload audio/video", type=["wav", "mp3", "mp4", "m4a"])

if uploaded:
    st.info("Processing clip...")
    try:
        spect = audio_to_melspectrogram(uploaded)
        spect = np.expand_dims(spect, axis=0)

        pred = model.predict(spect)[0][0]

        st.subheader("Prediction:")
        if pred > 0.5:
            st.success("**HIGHLIGHT DETECTED**")
        else:
            st.warning("Normal moment — highlight not detected")

        # Optional: Play uploaded audio
        if uploaded.type.startswith("video"):
            # play extracted audio
            audio_path = uploaded_file  # temporary extracted audio
        else:
            audio_path = uploaded

        st.audio(audio_path)

    except Exception as e:
        st.error(f"Error processing file: {e}")