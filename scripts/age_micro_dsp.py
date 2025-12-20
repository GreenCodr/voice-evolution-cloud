# scripts/age_micro_dsp.py

import numpy as np
import librosa
import soundfile as sf


def apply_micro_instability(
    wav_path: str,
    out_path: str,
    jitter: float = 0.0,
    shimmer: float = 0.0,
    tremor_rate: float = 0.0,
):
    """
    Adds human-like micro-instability:
    - jitter: random pitch noise
    - shimmer: amplitude fluctuation
    - tremor_rate: slow sinusoidal pitch wobble
    """

    y, sr = librosa.load(wav_path, sr=None)

    # ---------------- Pitch Jitter ----------------
    if jitter > 0:
        noise = np.random.normal(0, jitter, size=len(y))
        y = y + noise * 0.003

    # ---------------- Shimmer ----------------
    if shimmer > 0:
        amp = 1.0 + np.random.normal(0, shimmer, size=len(y))
        y = y * amp

    # ---------------- Tremor ----------------
    if tremor_rate > 0:
        t = np.linspace(0, len(y) / sr, num=len(y))
        tremor = 0.003 * np.sin(2 * np.pi * tremor_rate * t)
        y = y + tremor

    # Normalize
    y = y / np.max(np.abs(y) + 1e-6)

    sf.write(out_path, y, sr)