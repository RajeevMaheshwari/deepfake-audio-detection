import numpy as np
import librosa
import scipy.signal
from typing import Tuple, Optional

SAMPLE_RATE = 16000
FIXED_LENGTH_SEC = 4
FIXED_LENGTH = SAMPLE_RATE * FIXED_LENGTH_SEC


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if len(trimmed) == 0:
        return audio
    return trimmed


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    if len(audio) == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int = FIXED_LENGTH) -> np.ndarray:
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), mode='constant')
    return audio


def preprocess_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = trim_silence(audio)
    audio = peak_normalize(audio)
    audio = pad_or_truncate(audio)
    return audio


def extract_lfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_lfcc: int = 60,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_frames: int = 401
) -> np.ndarray:
    if len(audio) < win_length:
        audio = np.pad(audio, (0, win_length - len(audio)))
    D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    linear_filterbank = np.zeros((n_lfcc, n_fft // 2 + 1))
    for i in range(n_lfcc):
        linear_filterbank[i] = freqs ** (i + 1)
    linear_filterbank /= (linear_filterbank.sum(axis=1, keepdims=True) + 1e-10)
    lfcc_spectrum = np.dot(linear_filterbank, D)
    lfcc_static = librosa.amplitude_to_db(lfcc_spectrum, ref=np.max)
    lfcc_delta = librosa.feature.delta(lfcc_static, width=3)
    lfcc_delta2 = librosa.feature.delta(lfcc_static, width=3, order=2)
    lfcc = np.vstack([lfcc_static, lfcc_delta, lfcc_delta2])
    if lfcc.shape[1] > n_frames:
        lfcc = lfcc[:, :n_frames]
    elif lfcc.shape[1] < n_frames:
        lfcc = np.pad(lfcc, ((0, 0), (0, n_frames - lfcc.shape[1])))
    return lfcc


def extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_frames: int = 400
) -> np.ndarray:
    if len(audio) < win_length:
        audio = np.pad(audio, (0, win_length - len(audio)))
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] > n_frames:
        mel_spec_db = mel_spec_db[:, :n_frames]
    elif mel_spec_db.shape[1] < n_frames:
        pad_width = n_frames - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    return mel_spec_db


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    r = np.random.random()
    if r < 0.3:
        snr_db = np.random.uniform(10, 20)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(audio))
        audio = audio + noise
    if r < 0.5:
        rate = np.random.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
    if r < 0.6:
        pitch_steps = np.random.uniform(-2, 2)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=pitch_steps)
    return audio
