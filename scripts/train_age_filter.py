# scripts/train_age_filter.py
"""
Learn age-related spectral filters from real speech.
This does NOT learn speaker identity.
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ------------------ PATHS ------------------

META = Path("datasets/common_voice/age_audio/all_age_metadata.csv")
OUT = Path("learning/age_filters")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------ AUDIO PARAMS ------------------

SR = 16000
N_MELS = 80


def mel_profile(wav_path: str) -> np.ndarray:
    """
    Compute average log-mel spectral profile of an audio file.
    """
    y, sr = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmin=80,
        fmax=7600,
    )
    return np.mean(np.log(mel + 1e-6), axis=1)


def build_profile(df: pd.DataFrame, max_samples=150) -> np.ndarray:
    feats = []
    for p in tqdm(df.audio_path.sample(min(len(df), max_samples))):
        try:
            feats.append(mel_profile(p))
        except Exception:
            continue
    return np.mean(feats, axis=0)


def main():
    df = pd.read_csv(META)

    print("Samples:")
    print(df.age_group.value_counts())

    adult_df = df[df.age_group == "adult"]
    child_df = df[df.age_group == "children"]

    print("\n🔍 Learning adult spectral profile")
    adult_profile = build_profile(adult_df)

    print("\n🔍 Learning child spectral profile")
    child_profile = build_profile(child_df)

    # Save base profiles
    np.save(OUT / "adult_profile.npy", adult_profile)
    np.save(OUT / "child_profile.npy", child_profile)

    # Save delta (THIS IS THE MAGIC)
    child_delta = child_profile - adult_profile
    np.save(OUT / "child_delta.npy", child_delta)

    print("\n✅ Age spectral filters learned")
    print("Saved to:", OUT.resolve())


if __name__ == "__main__":
    main()