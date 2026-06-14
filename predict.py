#!/usr/bin/env python3
"""Deepfake Audio Detection — CLI Prediction Tool
Usage: python predict.py --audio path/to/audio.wav
"""
import sys, os, argparse, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.resnet_spec import ResNetSpec
from src.preprocess import load_audio, preprocess_audio
from src.preprocess import extract_mel_spectrogram

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'best_model.pth')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'trained_models', 'model_config.json')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

THRESHOLD = None  # loaded from config

def load_model():
    global THRESHOLD
    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": "Model not found. Train first or run from project root."}))
        sys.exit(1)

    model = ResNetSpec(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH))
        THRESHOLD = cfg.get('threshold', 0.5)

    return model


def preprocess(filepath: str) -> np.ndarray:
    audio = load_audio(filepath)
    audio = preprocess_audio(audio)
    mel = extract_mel_spectrogram(audio)
    return mel


def predict(model, features: np.ndarray):
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        scores = torch.softmax(model(x), dim=-1).cpu().numpy()[0]
    # class 0 = Genuine, class 1 = Deepfake
    deepfake_score = float(scores[1])
    threshold = THRESHOLD if THRESHOLD is not None else 0.5
    prediction = 'Deepfake' if deepfake_score >= threshold else 'Genuine'
    confidence = deepfake_score if prediction == 'Deepfake' else (1 - deepfake_score)
    return prediction, confidence, deepfake_score


def main():
    parser = argparse.ArgumentParser(description='Deepfake Audio Detector')
    parser.add_argument('--audio', required=True, help='Path to .wav file')
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(json.dumps({"error": f"File not found: {args.audio}"}))
        sys.exit(1)

    model = load_model()
    features = preprocess(args.audio)
    prediction, confidence, raw_score = predict(model, features)

    print(json.dumps({
        "prediction": prediction,
        "confidence": round(confidence * 100, 1),
        "deepfake_score": round(raw_score, 4),
        "threshold": round(THRESHOLD, 4) if THRESHOLD else 0.5,
    }, indent=2))


if __name__ == '__main__':
    main()
