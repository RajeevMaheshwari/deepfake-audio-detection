# Deepfake Audio Detection

> **Genuine (Human) vs Deepfake (AI-Generated) Speech Classification**
>
> MARS Open Projects 2026 — Part I (AIML) — Problem Statement 2

## Problem Statement

Advances in generative AI have enabled the creation of highly realistic synthetic speech (deepfake audio), which can be misused for impersonation, fraud, and misinformation. This project develops a machine learning system to classify speech recordings as either **Genuine (Human)** or **Deepfake (AI-Generated)**.

## Dataset

**The Fake-or-Real (FoR) Dataset** ([kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset))

| Split | Genuine | Deepfake | Total |
|-------|---------|----------|-------|
| Training | 26,941 | 26,927 | 53,868 |
| Validation | 5,400 | 5,398 | 10,798 |
| Testing | 2,264 | 2,370 | 4,634 |

- **Format**: 16kHz mono WAV (for-norm variant)
- **Duration**: 1–4 seconds, padded/truncated to 4 seconds
- **Classes**: Balanced (~50/50 genuine/deepfake in all splits)

## Methodology

```
Audio Input (.wav/.mp3)
    │
    ▼
Preprocessing (16kHz resample → silence trim → peak normalize → 4s pad)
    │
    ▼
Feature Extraction (LFCC: 60 static + 60 delta + 60 delta-delta = 180-dim)
    │
    ▼
LCNN Model (5 conv blocks → BiLSTM → FC → Softmax)
    │
    ▼
Prediction: Genuine (Human) or Deepfake (AI-Generated) + Confidence Score
```

### Preprocessing
1. Resample to 16 kHz mono
2. Trim silence (30 dB threshold)
3. Peak normalize to ±1.0
4. Pad/truncate to 4 seconds (64,000 samples)
5. Extract LFCC features (Linear Frequency Cepstral Coefficients) with deltas and delta-deltas

### Feature Extraction: LFCC
LFCC (Linear Frequency Cepstral Coefficients) are chosen over MFCC because they better capture high-frequency artifacts produced by neural vocoders and TTS systems. The linear filterbank preserves energy at all frequencies equally, unlike the mel-scale which compresses high frequencies.

### Model Architecture: LCNN (Light CNN)
- 5 convolutional blocks with Max-Feature-Map (MFM) activation
- Filter progression: 32 → 64 → 128 → 64 → 64
- Bidirectional LSTM (128 hidden) for temporal context
- Dropout (0.5) for regularization
- Total parameters: ~606K

### Training
- **Loss**: Weighted Cross-Entropy (weights inverse to class frequency)
- **Optimizer**: Adam (lr=3e-4, weight_decay=1e-4)
- **Schedule**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch size**: 64
- **Early stopping**: Patience 10 on validation EER
- **Augmentation**: White noise (30%), time stretch (20%), pitch shift (10%)

## Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Overall Accuracy** | — | ≥ 80% | — |
| **Equal Error Rate (EER)** | — | ≤ 12% | — |
| **F1 Score** | — | ≥ 80% | — |
| **Genuine Accuracy** | — | ≥ 75% | — |
| **Deepfake Accuracy** | — | ≥ 75% | — |

*(Results populated after Kaggle GPU training completes)*

### Experiments Summary

| Experiment | Model | Features | Accuracy | EER | F1 |
|------------|-------|----------|----------|-----|-----|
| 1 | LCNN | LFCC (180-dim) | — | — | — |
| 2 | RawNet2 | Raw Waveform | — | — | — |
| 3 | ResNet34 | Mel-Spectrogram | — | — | — |
| 4 | Ensemble | Combined | — | — | — |

## Project Structure

```
mars/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── src/
│   ├── preprocess.py                  # Audio preprocessing & feature extraction
│   ├── dataset.py                     # PyTorch Dataset classes
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Evaluation utilities
│   ├── utils.py                       # Metrics (EER, F1), plotting
│   └── models/
│       ├── lcnn.py                    # Light CNN architecture
│       ├── rawnet2.py                 # RawNet2 (SincNet + ResBlocks)
│       ├── resnet_spec.py             # ResNet34 on spectrograms
│       └── ensemble.py               # Voting ensemble
├── experiments/                       # Experiment notebooks (1-4)
├── trained_models/
│   ├── best_model.pth                 # Best model weights
│   └── model_config.json             # Architecture config
├── reports/
│   └── performance_report.md          # Full evaluation report
├── app/
│   └── streamlit_app.py               # Streamlit web application
├── predict.py                         # CLI inference script
├── final_pipeline.ipynb               # Consolidated notebook
└── demo_script.md                     # Demo video script
```

## Setup

```bash
# Create virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API token)
python -c "
import kagglehub, os
os.environ['KAGGLE_KEY'] = 'YOUR_KAGGLE_TOKEN'
path = kagglehub.dataset_download('mohammedabdeldayem/the-fake-or-real-dataset')
"
```

## Usage

### CLI Prediction
```bash
python predict.py --audio path/to/audio.wav
# Output: {"prediction": "Genuine", "confidence": 0.9623}
```

### Streamlit Web App
```bash
streamlit run app/streamlit_app.py
```
Opens a web interface at http://localhost:8501 where you can upload audio files and get real-time predictions.

### Training
```bash
# Run individual experiments via Kaggle GPU (kernel pushed with kaggle CLI)
kaggle kernels push -p kaggle_kernels/deepfake_experiments
```

## Model Weights

Download trained model from Kaggle kernel output or run training locally.

## Demo Video

[Link to demo video] — ~2 minute walkthrough demonstrating the web app with genuine and deepfake audio samples.

## Streamlit App

[Deployed Streamlit app URL]

## License

MIT
