# scripts/process_new_voice.py

from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import shutil  # ✅ NEW

from scripts.audio_preprocess import normalize_audio   # 🔑 CRITICAL
from scripts.audio_quality import audio_quality_gate
from scripts.embed_ecapa import extract_embedding
from scripts.speaker_verification import speaker_verification_gate
from scripts.device_fingerprint import extract_device_fingerprint, device_match_score
from scripts.confidence_engine import compute_confidence
from scripts.version_decision import decide_voice_version
from scripts.user_registry import UserRegistry
from scripts.audio_utils import get_audio_duration


# ------------------ CONSTANTS ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_DURATION_SEC = 10.0

# ECAPA identity-grade threshold (REALISTIC)
STRICT_SPEAKER_THRESHOLD = 0.75


# ------------------ HELPERS ------------------

def _safe_round(x, ndigits: int):
    """Round safely: returns None if x is None or not a number."""
    if x is None:
        return None
    try:
        return round(float(x), ndigits)
    except (TypeError, ValueError):
        return None


# ------------------ MAIN ENTRY ------------------

def process_new_voice(user_id: str, audio_path: str) -> dict:
    """
    Phase-2 backend: ECAPA-based identity verification
    Real-world safe (raw MP3/WAV supported)
    """

    audio_path = Path(audio_path)
    if not audio_path.exists():
        return {"accepted": False, "reason": "Audio file not found"}

    user = UserRegistry(user_id)

    # ====================================================
    # 🔊 AUDIO NORMALIZATION (ABSOLUTELY REQUIRED)
    # ====================================================
    try:
        clean_audio = normalize_audio(audio_path)
    except Exception as e:
        return {
            "accepted": False,
            "reason": "Audio preprocessing failed",
            "error": str(e),
        }

    # ====================================================
    # ✅ CRITICAL CLOUD FIX:
    # Persist cleaned audio into a stable project folder
    # (Streamlit Cloud temp uploads can be deleted or moved)
    # ====================================================
    stable_audio_dir = PROJECT_ROOT / "versions" / "audio"
    stable_audio_dir.mkdir(parents=True, exist_ok=True)

    stable_id = str(int(datetime.now(timezone.utc).timestamp()))
    stable_audio_path = stable_audio_dir / f"{user_id}_{stable_id}.wav"

    try:
        shutil.copyfile(str(clean_audio), str(stable_audio_path))

        # ✅ NEW: delete temp cleaned file to avoid disk bloat on Streamlit Cloud
        try:
            Path(str(clean_audio)).unlink(missing_ok=True)
        except Exception:
            pass

    except Exception:
        # If copy fails for any reason, fall back to the cleaned audio path
        stable_audio_path = Path(str(clean_audio))

    stable_audio_path = str(stable_audio_path)

    # ---------------- Duration ----------------
    duration = get_audio_duration(str(stable_audio_path))
    if duration < MIN_DURATION_SEC:
        return {
            "accepted": False,
            "reason": "Audio too short",
            "duration_sec": round(duration, 2),
        }

    # ---------------- Audio Quality (SOFT) ----------------
    quality = audio_quality_gate(str(stable_audio_path), dev_mode=True)
    soft_quality_fail = not quality["accepted"]

    # ---------------- ECAPA Embedding ----------------
    embedding = extract_embedding(stable_audio_path)
    embedding = embedding / np.linalg.norm(embedding)

    history_versions = user.get_versions()

    # ====================================================
    # 🧱 BASELINE BOOTSTRAP (FIRST VOICE ONLY)
    # ====================================================
    if not history_versions:
        version_id = str(int(datetime.now(timezone.utc).timestamp()))

        emb_dir = PROJECT_ROOT / "versions" / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        emb_path = emb_dir / f"{user_id}_{version_id}.npy"
        np.save(emb_path, embedding)

        user.add_voice_version(
            version_id=version_id,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(stable_audio_path),  # ✅ store STABLE CLEANED audio
            confidence=1.0,
            voice_type="RECORDED",
        )

        return {
            "accepted": True,
            "change_detected": False,
            "decision": {
                "action": "CREATE_BASELINE",
                "reason": "First voice stored",
            },
            "confidence": 1.0,
            "similarity": 1.0,
        }

    # ====================================================
    # 🔍 SPEAKER VERIFICATION (ECAPA)
    # ====================================================
    reference_embs = []
    for v in history_versions:
        p = v.get("embedding_path")
        if p:
            full = PROJECT_ROOT / p
            if full.exists():
                e = np.load(full).astype("float32")
                e = e / np.linalg.norm(e)
                reference_embs.append(e)

    speaker = speaker_verification_gate(
        new_emb=embedding,
        reference_embs=reference_embs,
        threshold=STRICT_SPEAKER_THRESHOLD,
    )

    # Pull similarity once and keep it consistent
    speaker_similarity = speaker.get("best_similarity", None)

    if not speaker["accepted"]:
        return {
            "accepted": False,
            "reason": "Different speaker detected",
            "similarity": _safe_round(speaker_similarity, 4),
        }

    # ---------------- Device Fingerprint (SOFT) ----------------
    device_score = 1.0
    try:
        latest = user.get_latest_version()
        if latest and latest.get("audio_path"):
            fp_ref = extract_device_fingerprint(latest["audio_path"])     # ✅ stable from registry now
            fp_new = extract_device_fingerprint(str(stable_audio_path))   # ✅ stable new audio
            device_score = device_match_score(fp_new, fp_ref)
    except Exception:
        pass

    # ---------------- Confidence (ADVISORY ONLY) ----------------
    confidence = compute_confidence(
        duration_s=quality.get("duration", duration),
        snr_db=quality.get("snr_db", None),
        speaker_similarity=speaker_similarity,
        device_match=device_score,
        history_count=len(reference_embs),
    )

    if soft_quality_fail:
        confidence *= 0.6

    # ---------------- Decision ----------------
    decision = decide_voice_version(
        similarity=speaker_similarity,
        confidence=confidence,
        speaker_ok=True,
        device_match=device_score,
        embedding_path="N/A",
        audio_path=str(stable_audio_path),  # ✅ stable audio path
        user_dob=user.data.get("date_of_birth"),
    )

    # ---------------- Persist ----------------
    if decision.get("action") == "CREATE_VERSION":
        version_id = str(int(datetime.now(timezone.utc).timestamp()))

        emb_dir = PROJECT_ROOT / "versions" / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)

        emb_path = emb_dir / f"{user_id}_{version_id}.npy"
        np.save(emb_path, embedding)

        user.add_voice_version(
            version_id=version_id,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(stable_audio_path),  # ✅ store STABLE CLEANED audio
            confidence=confidence,
            voice_type="RECORDED",
        )

    return {
        "accepted": True,
        "change_detected": False,
        "decision": decision,
        "confidence": round(confidence, 3),
        "similarity": _safe_round(speaker_similarity, 4),
        "audio_quality_soft_fail": soft_quality_fail,
    }