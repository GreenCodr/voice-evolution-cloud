
import argparse
import subprocess
from pathlib import Path

from voice_age.config import DEFAULT_SR, OUTPUTS_DIR
from voice_age.io.audio import load_audio, write_wav, duration_seconds
from voice_age.age.control import apply_age_control, normalize_audio


def main():
    p = argparse.ArgumentParser(prog="voice-age")
    p.add_argument("--in", dest="inp", required=True, help="Input audio (wav/mp3 etc.)")
    p.add_argument("--age", type=float, required=True, help="Target age (0-70)")
    p.add_argument("--out", dest="out", default="", help="Output wav path (default: data/outputs/age_<age>.wav)")
    p.add_argument("--play", action="store_true", help="Play output with afplay (macOS)")
    args = p.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"Missing input: {inp}")

    wav, sr = load_audio(inp, sr=DEFAULT_SR, mono=True)
    wav = normalize_audio(wav)

    y = apply_age_control(wav, sr, target_age=float(args.age))

    if args.out:
        out = Path(args.out)
    else:
        out = OUTPUTS_DIR / f"age_{int(round(args.age))}.wav"

    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, y, sr)

    print("OK ✅")
    print("in :", inp, "sr=", sr, "dur=", round(duration_seconds(wav, sr), 2), "sec")
    print("age:", args.age)
    print("out:", out, "size=", out.stat().st_size)

    if args.play:
        subprocess.run(["afplay", str(out)], check=False)


if __name__ == "__main__":
    main()
def load_wav(path, target_sr: int = None):
    """
    Load audio from path and return (wav_float32_mono, sr).

    - Uses soundfile for WAV/FLAC/OGG etc.
    - Falls back to librosa for formats like mp3/m4a if needed.
    - Resamples to target_sr if provided.
    """
    from pathlib import Path
    import numpy as np

    path = str(Path(path))

    # 1) Try soundfile first (fast + accurate for WAV)
    try:
        import soundfile as sf
        wav, sr = sf.read(path, always_2d=False)
        if isinstance(wav, np.ndarray) and wav.ndim == 2:
            wav = wav.mean(axis=1)  # to mono
        wav = wav.astype(np.float32)
    except Exception:
        # 2) Fallback: librosa can load mp3/m4a depending on backend
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=True)
        wav = wav.astype(np.float32)

    # Resample if needed
    if target_sr is not None and int(sr) != int(target_sr):
        import librosa
        wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(target_sr)).astype(np.float32)
        sr = int(target_sr)

    return wav, int(sr)
