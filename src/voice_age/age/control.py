import numpy as np
import librosa
from scipy.signal import butter, filtfilt

from voice_age.config import DEFAULT_SR, BASE_AGE, MIN_AGE, MAX_AGE


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def normalize_audio(wav: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    wav = wav.astype(np.float32)
    mx = float(np.max(np.abs(wav)) + eps)
    if mx > 1.0:
        wav = wav / mx
    return wav


def _butter_filter(wav: np.ndarray, sr: int, cutoff_hz: float, btype: str, order: int = 4):
    sr = int(sr)
    nyq = 0.5 * sr
    cutoff = float(cutoff_hz)
    cutoff = max(20.0, min(cutoff, nyq - 50.0))
    b, a = butter(order, cutoff, btype=btype, fs=sr)  # cutoff in Hz
    return filtfilt(b, a, wav).astype(np.float32)


def _tilt_bright(wav: np.ndarray, amount: float) -> np.ndarray:
    """
    Brighter tilt using pre-emphasis.
    amount in [0..1]
    """
    a = 0.65 + 0.30 * _clamp(amount, 0.0, 1.0)  # 0.65..0.95
    y = np.empty_like(wav, dtype=np.float32)
    y[0] = wav[0]
    y[1:] = wav[1:] - a * wav[:-1]
    return y


def _tilt_dark(wav: np.ndarray, amount: float) -> np.ndarray:
    """
    Darker tilt using smoothing (de-emphasis-ish).
    amount in [0..1]
    """
    a = 0.35 + 0.35 * _clamp(amount, 0.0, 1.0)  # 0.35..0.70
    y = np.empty_like(wav, dtype=np.float32)
    y[0] = wav[0]
    for i in range(1, len(wav)):
        y[i] = a * y[i - 1] + (1.0 - a) * wav[i]
    return y.astype(np.float32)


def _pitch_shift_semitones(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Prefer Praat pitch shift (cleaner for speech). Fallback to librosa.
    """
    if abs(semitones) < 1e-4:
        return wav.astype(np.float32)

    wav = wav.astype(np.float32)

    try:
        import parselmouth
        from parselmouth.praat import call

        snd = parselmouth.Sound(wav, sampling_frequency=sr)
        f0min, f0max = 75, 500

        manipulation = call(snd, "To Manipulation", 0.01, f0min, f0max)
        pitch_tier = call(manipulation, "Extract pitch tier")

        factor = 2 ** (semitones / 12.0)
        call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, factor)

        call([manipulation, pitch_tier], "Replace pitch tier")
        resynth = call(manipulation, "Get resynthesis (overlap-add)")
        return resynth.values[0].astype(np.float32)

    except Exception:
        y = librosa.effects.pitch_shift(wav, sr=sr, n_steps=semitones)
        return y.astype(np.float32)



def _vtln_warp(wav: np.ndarray, sr: int, alpha: float) -> np.ndarray:
    """
    Simple VTLN-like frequency-axis warp.
    alpha > 1.0  => formants shift UP   (younger)
    alpha < 1.0  => formants shift DOWN (older)
    """
    alpha = float(_clamp(alpha, 0.85, 1.20))
    if abs(alpha - 1.0) < 1e-4:
        return wav.astype(np.float32)

    wav = wav.astype(np.float32)

    n_fft = 1024 if sr <= 16000 else 2048
    hop = n_fft // 4
    win = n_fft

    S = librosa.stft(wav, n_fft=n_fft, hop_length=hop, win_length=win)  # complex [F,T]
    F, T = S.shape

    freqs = np.arange(F, dtype=np.float32)
    query = freqs / alpha
    query = np.clip(query, 0.0, F - 1.0)

    out = np.empty_like(S)
    for t in range(T):
        re = np.interp(freqs, query, S[:, t].real).astype(np.float32)
        im = np.interp(freqs, query, S[:, t].imag).astype(np.float32)
        out[:, t] = re + 1j * im

    y = librosa.istft(out, hop_length=hop, win_length=win, length=len(wav))
    return y.astype(np.float32)


def apply_age_control(wav: np.ndarray, sr: int, target_age: float) -> np.ndarray:
    """
    Continuous (NO sudden jump at BASE_AGE):
    - Smoothly transitions from young -> neutral -> old
    - Stronger, audible changes from 25..60
    - Extra aging after 55 so 55->70 keeps changing
    """
    sr = int(sr or DEFAULT_SR)
    a = _clamp(float(target_age), MIN_AGE, MAX_AGE)

    y0 = normalize_audio(wav.astype(np.float32))

    def smoothstep(t: float) -> float:
        t = _clamp(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # Strength for "younger" and "older" sides (both are 0 at BASE_AGE)
    if a <= BASE_AGE:
        u_y = (BASE_AGE - a) / max(1e-6, (BASE_AGE - MIN_AGE))  # 0..1
        young = smoothstep(u_y)
        old = 0.0
    else:
        u_o = (a - BASE_AGE) / max(1e-6, (MAX_AGE - BASE_AGE))  # 0..1
        old = smoothstep(u_o)
        young = 0.0

    # Extra aging after 55 for 55->70 separation
    u2 = (a - 55.0) / max(1e-6, (MAX_AGE - 55.0))
    extra = smoothstep(u2)

    # --- KEY FIX: Avoid "direction flip shock" near BASE_AGE ---
    # We keep ALL effects proportional to (young or old).
    # No constant offsets. So 20->25 won't "snap" old.

    # 1) Pitch curve (stronger mid-age changes)
    # - max younger: +1.7 semitones
    # - max older:   -3.2 semitones, plus extra -1.2 after 55
    semis = (+1.7 * young) - (3.2 * old + 1.2 * extra)
    y = _pitch_shift_semitones(y0, sr, semis)

    # 2) Spectral tilt (brighter when young, darker when old) — scaled only
    y_bright = _tilt_bright(y, amount=0.65 * young)   # 0..0.65
    y_dark = _tilt_dark(y, amount=0.75 * old + 0.55 * extra)  # 0..1.30 (clamped inside)

    # Blend tilt results smoothly (no branch jump)
    # If young>0 -> lean bright; if old>0 -> lean dark; else unchanged
    y = y
    if young > 0:
        y = (1.0 - young) * y + young * y_bright
    if old > 0 or extra > 0:
        s_old = _clamp(old + 0.35 * extra, 0.0, 1.0)
        y = (1.0 - s_old) * y + s_old * y_dark

    y = y.astype(np.float32)

    # 3) Filters (also scaled, not “always on”)
    # Young: tiny rumble removal, but only when young strength > 0
    if young > 0:
        hp = _butter_filter(y, sr, 90.0, "highpass", order=2)
        y = (1.0 - young) * y + young * hp

    # Old: loss of brilliance with age (low-pass), but gradual and stronger 25..60
    if old > 0 or extra > 0:
        # Cutoff goes from ~12000 down to ~4200 by old age
        cutoff = 12000.0 - 6000.0 * old - 4500.0 * extra
        cutoff = max(3500.0, cutoff)
        lp = _butter_filter(y, sr, cutoff, "lowpass", order=2)
        s_lp = _clamp(0.90 * old + 0.55 * extra, 0.0, 1.0)
        y = (1.0 - s_lp) * y + s_lp * lp

        # Remove mud slightly, but scale it (no sudden change)
        hp2 = _butter_filter(y, sr, 70.0, "highpass", order=2)
        s_hp2 = _clamp(0.55 * old + 0.25 * extra, 0.0, 1.0)
        y = (1.0 - s_hp2) * y + s_hp2 * hp2

    return normalize_audio(y).astype(np.float32)