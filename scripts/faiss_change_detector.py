import sys
import csv
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import faiss

# ------------------ PATH SETUP ------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ------------------ IMPORTS ------------------

from audio_quality import audio_quality_gate
from speaker_verification import speaker_verification_gate
from device_fingerprint import extract_device_fingerprint, device_match_score
from scripts.confidence_engine import compute_confidence
from scripts.version_decision import decide_voice_version
from user_registry import UserRegistry

from scripts.audio_preprocess import normalize_audio  # ✅ NEW (same as Phase-2)

try:
    from register_version import register_version
except Exception:
    register_version = None

# ------------------ CONSTANTS ------------------

EMB_DIR = PROJECT_ROOT / "embeddings" / "ecapa"
VERSIONS_DIR = PROJECT_ROOT / "versions"
VERSIONS_AUDIO_DIR = VERSIONS_DIR / "audio"
MANIFEST_EMB = PROJECT_ROOT / "data" / "librispeech_manifest_small_emb.csv"

THRESHOLD = 0.75

# ✅ NEW: make speaker verification consistent with Phase-2
SPEAKER_THRESHOLD = 0.75

# ------------------ HELPERS ------------------

def load_embedding(p: Path) -> np.ndarray:
    return np.load(p).astype("float32")


def normalize(e: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(e)
    return e if n == 0 else e / n


def build_index(embs: list) -> Optional[faiss.Index]:
    if not embs:
        return None
    dim = embs[0].shape[0]
    index = faiss.IndexFlatIP(dim)
    index.add(np.vstack(embs))
    return index


# ------------------ 🔥 REQUIRED FUNCTION ------------------

def detect_change(user_id: str, new_embedding: np.ndarray) -> Tuple[bool, float]:
    """
    Core FAISS change detector used by process_new_voice()

    Returns:
        (change_detected, best_similarity)
    """

    user = UserRegistry(user_id)
    versions = user.get_versions()

    # First recording
    if not versions:
        return True, 0.0

    embeddings = []
    for v in versions:
        emb_rel = v.get("embedding_path")
        if not emb_rel:
            continue   # sparse data case

        emb_path = PROJECT_ROOT / emb_rel
        if emb_path.exists():
            embeddings.append(normalize(load_embedding(emb_path)))

    if not embeddings:
        return True, 0.0

    index = build_index(embeddings)
    new_emb = normalize(new_embedding)

    D, _ = index.search(np.expand_dims(new_emb, 0), 1)
    best_sim = float(D[0][0])

    change_detected = best_sim < THRESHOLD
    return change_detected, best_sim


# ------------------ CLI / BATCH MODE ------------------

def read_manifest():
    rows = {}
    if not MANIFEST_EMB.exists():
        return rows

    with open(MANIFEST_EMB, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[Path(r["emb_path"]).name] = r

    return rows


def main(threshold: float = THRESHOLD) -> int:
    print("FAISS change detector running | threshold =", threshold)

    user = UserRegistry("user_001")
    manifest = read_manifest()

    candidates = list(EMB_DIR.glob("*.npy"))
    if not candidates:
        print("❌ No embeddings found")
        return 1

    # Load existing versions
    version_embs = []
    for v in user.get_voice_versions():
        p = PROJECT_ROOT / v["embedding_path"]
        if p.exists():
            version_embs.append(normalize(load_embedding(p)))

    index = build_index(version_embs)

    for emb_path in candidates:
        emb = normalize(load_embedding(emb_path))

        best_sim = -1.0
        if index is not None:
            D, _ = index.search(np.expand_dims(emb, 0), 1)
            best_sim = float(D[0][0])

        row = manifest.get(emb_path.name)
        if not row:
            continue

        audio_rel = row.get("file_path") or row.get("preproc_path")
        audio_path = PROJECT_ROOT / audio_rel
        if not audio_path.exists():
            continue

        # ✅ NEW: normalize audio like Phase-2 (reduces Cloud drift)
        try:
            clean_audio = normalize_audio(audio_path)
        except Exception:
            continue

        # ---- Quality Gate ----
        quality = audio_quality_gate(str(clean_audio), dev_mode=True)
        if not quality["accepted"]:
            continue

        # ---- Speaker Gate ----
        speaker = speaker_verification_gate(emb, version_embs, threshold=SPEAKER_THRESHOLD)
        if not speaker["accepted"]:
            continue

        # ---- Device Gate ----
        device_score = 1.0
        try:
            ref = next(VERSIONS_AUDIO_DIR.glob("*.wav"))
            device_score = device_match_score(
                extract_device_fingerprint(str(clean_audio)),
                extract_device_fingerprint(str(ref)),
            )
        except StopIteration:
            pass
        except Exception:
            pass

        confidence = compute_confidence(
            duration_s=quality["duration"],
            snr_db=quality["snr_db"],
            speaker_similarity=speaker["best_similarity"],
            device_match=device_score,
            history_count=len(version_embs),
        )

        decision = decide_voice_version(
            similarity=best_sim,
            confidence=confidence,
            speaker_ok=True,
            device_match=device_score,
            embedding_path=str(emb_path.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            user_dob=row.get("dob"),
        )

        if decision["action"] != "CREATE_VERSION":
            continue

        VERSIONS_DIR.mkdir(exist_ok=True)
        final_emb = VERSIONS_DIR / emb_path.name
        shutil.move(str(emb_path), str(final_emb))

        user.add_voice_version(
            version_id=emb_path.stem,
            embedding_path=str(final_emb.relative_to(PROJECT_ROOT)),
            audio_path=str(audio_path.relative_to(PROJECT_ROOT)),
            confidence=confidence,
            voice_type="RECORDED",
        )

        print("✅ Version created:", emb_path.name)

    print("FAISS detection complete")
    return 0


# ------------------ ENTRY ------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    sys.exit(main(args.threshold))