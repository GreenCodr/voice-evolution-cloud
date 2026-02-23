
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import soundfile as sf
import librosa

PathLike = Union[str, Path]

def load_audio(path: PathLike, sr: int | None = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio with soundfile (fast, stable) then resample with librosa if sr is requested.
    Returns (wav_float32, sr).
    """
    path = Path(path)
    wav, orig_sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:  # [T, C]
        if mono:
            wav = wav.mean(axis=1)
        else:
            wav = wav.T  # optional, but we keep simple: mono pipeline
    wav = wav.astype(np.float32)

    if sr is not None and sr != orig_sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr).astype(np.float32)
        return wav, sr

    return wav, orig_sr

def write_wav(path: PathLike, wav: np.ndarray, sr: int) -> Path:
    """
    Write float32 wav safely. Clips to [-1,1] to avoid overflow distortion.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim != 1:
        raise ValueError(f"write_wav expects mono 1D audio, got shape={wav.shape}")

    mx = float(np.max(np.abs(wav)) + 1e-9)
    if mx > 1.0:
        wav = wav / mx

    sf.write(str(path), wav, sr)
    return path

def duration_seconds(wav: np.ndarray, sr: int) -> float:
    return float(len(wav)) / float(sr)
