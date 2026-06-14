import os
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf
from .preprocess import (
    load_audio, preprocess_audio, extract_lfcc,
    extract_mel_spectrogram, augment_audio, SAMPLE_RATE,
    FIXED_LENGTH, FIXED_LENGTH_SEC
)


class AudioDataset(Dataset):
    def __init__(self, root_dir: str, feature_type: str = 'lfcc',
                 augment: bool = False, fixed_length: int = FIXED_LENGTH):
        self.root_dir = root_dir
        self.feature_type = feature_type
        self.augment = augment
        self.fixed_length = fixed_length
        self.samples = []

        for cls, label in [('real', 0), ('fake', 1)]:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

        self.genuine_count = sum(1 for _, l in self.samples if l == 0)
        self.fake_count = len(self.samples) - self.genuine_count
        self.class_weights = torch.tensor([
            len(self.samples) / (2 * self.genuine_count) if self.genuine_count > 0 else 1.0,
            len(self.samples) / (2 * self.fake_count) if self.fake_count > 0 else 1.0
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def _load_waveform(self, path: str) -> np.ndarray:
        audio = load_audio(path)
        audio = preprocess_audio(audio)
        if self.augment:
            audio = augment_audio(audio)
        audio = np.pad(audio, (0, max(0, self.fixed_length - len(audio))))[:self.fixed_length]
        return audio

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform = self._load_waveform(path)

        if self.feature_type == 'lfcc':
            features = extract_lfcc(waveform)
            return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        elif self.feature_type == 'melspec':
            features = extract_mel_spectrogram(waveform)
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        elif self.feature_type == 'raw':
            return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")


class InferenceDataset(Dataset):
    def __init__(self, audio_path: str, feature_type: str = 'lfcc',
                 fixed_length: int = FIXED_LENGTH):
        self.audio_path = audio_path
        self.feature_type = feature_type
        self.fixed_length = fixed_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        waveform = load_audio(self.audio_path)
        waveform = preprocess_audio(waveform)
        waveform = np.pad(waveform, (0, max(0, self.fixed_length - len(waveform))))[:self.fixed_length]

        if self.feature_type == 'lfcc':
            features = extract_lfcc(waveform)
            return torch.tensor(features, dtype=torch.float32)
        elif self.feature_type == 'melspec':
            features = extract_mel_spectrogram(waveform)
            return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif self.feature_type == 'raw':
            return torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
