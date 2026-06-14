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
Feature Extraction (Mel-Spectrogram: 128 mel bands × 400 time frames)
    │
    ▼
ResNet34 Model (ImageNet pre-trained → 1-ch adaptor → 2-class FC head)
    │
    ▼
Prediction: Genuine (Human) or Deepfake (AI-Generated) + Confidence Score
```

### Preprocessing
1. Resample to 16 kHz mono
2. Trim silence (30 dB threshold)
3. Peak normalize to ±1.0
4. Pad/truncate to 4 seconds (64,000 samples)
5. Extract Mel-spectrogram: 128 mel bands, 25ms window, 10ms hop → 128×400 matrix

### Feature Extraction: Mel-Spectrogram
Mel-spectrograms are chosen as input to a pre-trained ResNet34, leveraging transfer learning from ImageNet. The 128 mel bands provide a compact yet information-rich 2D representation of audio that aligns well with CNN architectures designed for image classification.

### Model Architecture: ResNet34 (Best Model)
- **Backbone**: ResNet34 pre-trained on ImageNet
- **Input adaptor**: 1→3 channel repeat (128 mel bands × 400 time frames → 3 channels)
- **Head**: 512-dim FC → Dropout(0.3) → 2-class softmax
- **Parameters**: ~21.3M
- **Loss**: Weighted CrossEntropy (inverse class frequency)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch size**: 24
- **Early stopping**: patience=8 on validation EER
- **Augmentation**: Additive white noise (SNR 10–20 dB, 30%), time stretch (±10%, 20%), SpecAugment (frequency + time masking)

## Results

### Best Model: ResNet34 + Mel-Spectrogram (Eval — 4,634 unseen samples)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| **Overall Accuracy** | **88.3%** | ≥ 80% | ✅ PASS |
| **Equal Error Rate (EER)** | **11.7%** | ≤ 12% | ✅ PASS |
| **F1 Score** | **88.6%** | ≥ 80% | ✅ PASS |
| **Genuine Accuracy** | **88.3%** | ≥ 75% | ✅ PASS |
| **Deepfake Accuracy** | **88.4%** | ≥ 75% | ✅ PASS |

**Optimal decision threshold**: 0.0562 (fitted to minimize EER — critical since the default argmax threshold of 0.5 causes severe bias toward Genuine class).

### Confusion Matrix
```
                 Predicted
              Genuine  Deepfake
Actual Genuine  2000      264
       Deepfake   276     2094
```

### Experiments Summary

| Experiment | Model | Features | Accuracy | EER | F1 |
|------------|-------|----------|----------|-----|-----|
| 1 | LCNN | LFCC (180-dim) | 83.9% | 16.1% | 84.2% |
| 2 | RawNet2 | Raw Waveform | 72.0% | 28.0% | 72.5% |
| **3** | **ResNet34** | **Mel-Spectrogram** | **88.3%** | **11.7%** | **88.6%** |
| 4 | Ensemble (LCNN+ResNet34+RawNet2) | Combined | 85.9% | 14.1% | 86.2% |

## Project Structure

```
deepfake-audio-detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
├── PLAN.md                            # Implementation plan
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  # Audio preprocessing & feature extraction
│   ├── dataset.py                     # PyTorch Dataset classes
│   ├── train.py                       # Training loop
│   ├── evaluate.py                    # Evaluation utilities
│   ├── utils.py                       # Metrics (EER, F1), plotting
│   └── models/
│       ├── __init__.py
│       ├── lcnn.py                    # Light CNN architecture
│       ├── rawnet2.py                 # RawNet2 (SincNet + ResBlocks)
│       ├── resnet_spec.py             # ResNet34 on spectrograms (best model)
│       └── ensemble.py               # Voting ensemble
├── trained_models/
│   ├── best_model.pth                 # Best model weights (ResNet34)
│   └── model_config.json             # Architecture config + threshold
├── reports/
│   ├── performance_report.md          # Full evaluation report
│   ├── performance_report.pdf         # PDF report
│   ├── all_results_optimal.json       # Optimal threshold results (all models)
│   └── figures/                       # Report figures
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
