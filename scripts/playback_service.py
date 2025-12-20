# scripts/playback_service.py

from pathlib import Path
import uuid

from scripts.hybrid_playback_decider import decide_playback_mode
from scripts.synthesize_from_embedding import synthesize_from_embedding
from scripts.age_text_shaper import shape_text_for_age

# --------------------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def play_voice(user_id: str, target_age: int, text: str) -> dict:
    """
    Production-safe playback:
    - NO waveform DSP
    - ONLY embedding aging + neural TTS
    """

    decision = decide_playback_mode(user_id, target_age)
    mode = decision.get("mode")

    # ==================================================
    # RECORDED
    # ==================================================
    if mode == "RECORDED":
        version = decision.get("version") or decision.get("base_version")

        if not version or "audio_path" not in version:
            return {
                "mode": "ERROR",
                "reason": "Recorded version missing"
            }

        return {
            "mode": "RECORDED",
            "audio_path": version["audio_path"],
            "reason": "Using real recorded voice"
        }

    # ==================================================
    # AGED (NEURAL ONLY – SAFE)
    # ==================================================
    if mode == "AGED":
        base_version = decision.get("base_version")

        if not base_version or "audio_path" not in base_version:
            return {
                "mode": "ERROR",
                "reason": "Base version missing for aging"
            }

        uid = uuid.uuid4().hex[:8]
        out_path = OUTPUT_DIR / f"{user_id}_aged_{target_age}.wav"

        # ✔ text shaping (safe)
        shaped_text = shape_text_for_age(text, target_age)

        # ✔ neural synthesis only
        synthesize_from_embedding(
            text=shaped_text,
            out_path=str(out_path),
            speaker_embedding=decision["embedding"],
            reference_wav=base_version["audio_path"],
        )

        return {
            "mode": "AGED",
            "audio_path": str(out_path),
            "alpha": decision.get("alpha"),
            "relation": decision.get("relation"),
            "reason": "Age-evolved voice (neural, artifact-free)"
        }

    # ==================================================
    # FALLBACK
    # ==================================================
    return {
        "mode": "NONE",
        "reason": decision.get("reason", "No voice available")
    }