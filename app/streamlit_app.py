#!/usr/bin/env python3
"""Deepfake Audio Detector — Streamlit Web App"""
import sys, os, json, io, tempfile, torch, numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa, librosa.display
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.resnet_spec import ResNetSpec
from src.preprocess import load_audio, preprocess_audio, extract_mel_spectrogram

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'best_model.pth')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'model_config.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.0562

@st.cache_resource
def load_model():
    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH))
        global THRESHOLD
        THRESHOLD = cfg.get('threshold', 0.0562)
    model = ResNetSpec(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

def predict(model, waveform):
    mel = extract_mel_spectrogram(waveform)
    x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores = torch.softmax(model(x), dim=-1).cpu().numpy()[0]
    deepfake_score = float(scores[1])
    prediction = 'Deepfake' if deepfake_score >= THRESHOLD else 'Genuine'
    confidence = deepfake_score if prediction == 'Deepfake' else (1 - deepfake_score)
    return prediction, confidence, deepfake_score

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")
st.title("🎙️ Deepfake Audio Detector")
st.markdown("Detect whether speech audio is genuine (human) or AI-generated (deepfake).")

uploaded = st.file_uploader("Upload a `.wav` or `.mp3` audio file", type=['wav', 'mp3'])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(uploaded, format='audio/wav')

    try:
        waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
    except:
        st.error("Failed to load audio. Ensure it's a valid WAV/MP3 file.")
        st.stop()

    waveform = preprocess_audio(waveform)

    model = load_model()
    prediction, confidence, raw_score = predict(model, waveform)

    st.markdown("---")
    col1, col2 = st.columns(2)

    if prediction == 'Genuine':
        col1.markdown("### ✅ Genuine (Human)")
        col1.success(f"Confidence: {confidence*100:.1f}%")
    else:
        col1.markdown("### ⚠️ Deepfake (AI-Generated)")
        col1.error(f"Confidence: {confidence*100:.1f}%")

    col2.progress(min(confidence, 1.0))

    st.markdown(f"**Raw deepfake score:** {raw_score:.4f} (threshold: {THRESHOLD:.4f})")

    # Mel-spectrogram
    fig, ax = plt.subplots(figsize=(10, 3))
    mel = extract_mel_spectrogram(waveform)
    librosa.display.specshow(mel, sr=16000, hop_length=160, x_axis='time', y_axis='mel',
                             ax=ax, cmap='magma')
    ax.set_title("Mel-Spectrogram")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    os.unlink(tmp_path)

st.markdown("---")
st.caption("Model: ResNet34 on Mel-Spectrograms | Dataset: FoR-LA | EER: 11.7% | Acc: 88.3%")
