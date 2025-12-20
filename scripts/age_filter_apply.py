# scripts/age_filter_apply.py
"""
Apply learned age spectral filters safely.
NO echo
NO rain noise
NO muffling
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


SR = 24000
N_MELS = 80

FILTER_DIR = Path("learning/age_filters")


def apply_age_filter(
    wav_path: str,
    out_path: str,
    target_age: int,
):
    """
    Apply mel-spectral age correction.
    """

    # -------- Load audio --------
    y, sr = librosa.load(wav_path, sr=SR)

    # -------- Mel spectrogram --------
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmin=80,
        fmax=7600,
    )
    log_mel = np.log(mel + 1e-6)

    # -------- Load profiles --------
    adult_profile = np.load(FILTER_DIR / "adult_profile.npy")

    if target_age < 13:
        delta = np.load(FILTER_DIR / "child_delta.npy")
        strength = min((13 - target_age) / 8.0, 1.0)
    elif target_age > 60:
        # reuse inverse child delta for elderly (safe)
        delta = -np.load(FILTER_DIR / "child_delta.npy")
        strength = min((target_age - 60) / 25.0, 1.0)
    else:
        delta = np.zeros_like(adult_profile)
        strength = 0.0

    # -------- Apply filter (SAFE) --------
    adjusted = log_mel + strength * delta[:, None]

    # -------- Reconstruct audio --------
    mel_out = np.exp(adjusted)
    y_out = librosa.feature.inverse.mel_to_audio(
        mel_out,
        sr=sr,
        n_iter=32,
        hop_length=512,
    )

    # -------- Loudness normalize --------
    y_out = librosa.util.normalize(y_out)

    sf.write(out_path, y_out, sr)