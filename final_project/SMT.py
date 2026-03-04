from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np


def extract_mel_and_mfcc(
    audio_path: str | Path,
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, np.ndarray | int]:
    """
    Load one audio file and extract mel-spectrogram + MFCC using file-native sample rate.

    Returns:
        dict with keys:
            - sr: sampling rate (int)
            - win_length: analysis window length in samples
            - hop_length: hop length in samples
            - mel_spectrogram: power mel spectrogram [n_mels, n_frames]
            - log_mel_db: log-mel spectrogram in dB [n_mels, n_frames]
            - mfcc: MFCC features [n_mfcc, n_frames]
    """
    y, sr = librosa.load(str(audio_path), sr=None)

    win_length = max(1, int(round(sr * (window_ms / 1000.0))))
    hop_length = max(1, int(round(sr * (hop_ms / 1000.0))))
    n_fft = win_length

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )

    log_mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    mfcc = librosa.feature.mfcc(
        S=log_mel_db,
        n_mfcc=n_mfcc,
    )

    return {
        "sr": sr,
        "win_length": win_length,
        "hop_length": hop_length,
        "mel_spectrogram": mel_spectrogram,
        "log_mel_db": log_mel_db,
        "mfcc": mfcc,
    }


def extract_features_from_files(
    audio_paths: Iterable[str | Path],
    n_mels: int = 40,
    n_mfcc: int = 20,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> dict[str, dict[str, np.ndarray | int]]:
    """Extract mel-spectrogram and MFCC for multiple audio files."""
    results: dict[str, dict[str, np.ndarray | int]] = {}
    for path in audio_paths:
        path_str = str(path)
        results[path_str] = extract_mel_and_mfcc(
            audio_path=path,
            n_mels=n_mels,
            n_mfcc=n_mfcc,
            window_ms=window_ms,
            hop_ms=hop_ms,
        )
    return results


def extract_time_patches(
    features: np.ndarray,
    sr: int,
    hop_length: int,
    patch_ms: float = 300.0,
    patch_hop_frames: int = 1,
) -> np.ndarray:
    """
    Extract all time patches from a feature matrix.

    Args:
        features: 2D array [n_features, n_frames] (e.g., MFCC or mel-spectrogram).
        sr: Sampling rate used to generate the features.
        hop_length: Hop length in samples used to generate the features.
        patch_ms: Patch duration in milliseconds.
        patch_hop_frames: Step size between consecutive patches in frames.

    Returns:
        3D array [n_patches, n_features, patch_frames].
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D array with shape [n_features, n_frames].")
    if hop_length <= 0:
        raise ValueError("hop_length must be > 0.")
    if patch_hop_frames <= 0:
        raise ValueError("patch_hop_frames must be > 0.")

    n_features, n_frames = features.shape
    patch_frames = max(1, int(round((patch_ms / 1000.0) * sr / hop_length)))

    if n_frames < patch_frames:
        return np.empty((0, n_features, patch_frames), dtype=features.dtype)

    starts = range(0, n_frames - patch_frames + 1, patch_hop_frames)
    patches = [features[:, s : s + patch_frames] for s in starts]
    return np.stack(patches, axis=0)


def extract_300ms_patches_from_feature_dict(
    feature_dict: dict[str, np.ndarray | int],
    feature_key: str = "mfcc",
    patch_hop_frames: int = 1,
) -> np.ndarray:
    """
    Extract 300 ms patches from the selected feature in extract_mel_and_mfcc output.

    Args:
        feature_dict: Output dictionary from extract_mel_and_mfcc.
        feature_key: One of {"mfcc", "mel_spectrogram", "log_mel_db"}.
        patch_hop_frames: Step size between consecutive patches in frames.

    Returns:
        3D array [n_patches, n_features, patch_frames].
    """
    if "sr" not in feature_dict:
        raise KeyError("feature_dict is missing 'sr'.")
    if "hop_length" not in feature_dict:
        raise KeyError("feature_dict is missing 'hop_length'.")
    if feature_key not in feature_dict:
        raise KeyError(f"feature_dict is missing '{feature_key}'.")

    return extract_time_patches(
        features=np.asarray(feature_dict[feature_key]),
        sr=int(feature_dict["sr"]),
        hop_length=int(feature_dict["hop_length"]),
        patch_ms=300.0,
        patch_hop_frames=patch_hop_frames,
    )
