# frontend/app.py

import sys
import os
import json
import tempfile
import time
import io
import zipfile
import subprocess
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

# ------------------ PATH FIX ------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

USERS_DIR = PROJECT_ROOT / "users"


# ------------------ HELPERS ------------------

def ffmpeg_to_wav_16k_mono(in_path: Path, out_path: Path, sr: int = 16000):
    """
    Convert any audio to SR Hz mono WAV using ffmpeg.
    This makes Phase-2 accept mp3/m4a/etc reliably.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "")[-2000:]
        raise RuntimeError(err if err else "ffmpeg failed")


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def make_zip_bytes(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.glob("*.wav")):
            z.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


def _safe_user_id(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw)
    raw = raw.strip("_")
    return raw


def _validate_dob(dob: str) -> str:
    dob = (dob or "").strip()
    # expected YYYY-MM-DD
    datetime.strptime(dob, "%Y-%m-%d")
    return dob


def _create_user_file(user_id: str, dob: str) -> Path:
    USERS_DIR.mkdir(parents=True, exist_ok=True)
    user_path = USERS_DIR / f"{user_id}.json"
    if user_path.exists():
        raise ValueError("User already exists")

    dob = _validate_dob(dob)

    user_obj = {
        "user_id": user_id,
        "date_of_birth": dob,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "voice_versions": []
    }
    user_path.write_text(json.dumps(user_obj, indent=2))
    return user_path


# ------------------ APP ------------------

def run_app():
    st.set_page_config(page_title="Voice Evolution System", layout="centered")

    # ==============================================================
    # HEADER
    # ==============================================================
    st.title("🎙️ Voice Evolution System")
    st.caption("Automatic voice change detection & age-based playback")
    st.divider()

    # ==============================================================
    # CREATE NEW USER
    # ==============================================================
    st.header("➕ Create New User")

    with st.form("create_user_form", clear_on_submit=False):
        new_user_id_raw = st.text_input("Choose a User ID (letters/numbers/_/-)")
        new_dob = st.text_input("Date of Birth (YYYY-MM-DD)")
        create_btn = st.form_submit_button("Create User")

    if create_btn:
        new_user_id = _safe_user_id(new_user_id_raw)
        if not new_user_id:
            st.error("User ID cannot be empty.")
        else:
            try:
                p = _create_user_file(new_user_id, new_dob)
                st.success(f"✅ Created user: {new_user_id}")
                st.caption(f"Saved at: {p}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

    st.divider()

    # ==============================================================
    # USER SELECTION
    # ==============================================================
    st.header("👤 User Dashboard")

    USERS_DIR.mkdir(parents=True, exist_ok=True)
    user_files = sorted(USERS_DIR.glob("*.json"))
    if not user_files:
        st.error("No users found. Create a user above.")
        st.stop()

    user_ids = [f.stem for f in user_files]
    selected_user = st.selectbox("Select User", user_ids)

    user_path = USERS_DIR / f"{selected_user}.json"
    user = json.loads(user_path.read_text())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("User ID", user.get("user_id", selected_user))
        st.metric("Date of Birth", user.get("date_of_birth", "Unknown"))
    with col2:
        st.metric("Total Voice Versions", len(user.get("voice_versions", [])))
        st.metric("Account Created", user.get("created_utc", "")[:10])

    st.divider()

    # ==============================================================
    # PHASE 1 — VOICE INGESTION
    # ==============================================================
    st.header("🎙️ Upload Voice Sample")

    uploaded = st.file_uploader(
        "Upload voice sample (WAV / MP3, minimum 10 seconds)",
        type=["wav", "mp3"],
        key="phase1_upload",
    )

    if uploaded:
        st.audio(uploaded)
        st.success("Voice file received ✔️")

        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Analyzing voice sample..."):
            from scripts.process_new_voice import process_new_voice

            result = process_new_voice(
                user_id=selected_user,
                audio_path=tmp_path,
            )

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        if not result.get("accepted", False):
            st.error(f"❌ {result.get('reason', 'Rejected')}")
        else:
            decision = result.get("decision", {})

            st.success("Voice analyzed successfully")
            st.write(f"• Change detected: `{result.get('change_detected')}`")
            st.write(f"• Decision: `{decision.get('action')}`")

            if "reason" in decision:
                st.write(f"• Reason: {decision['reason']}")

            st.metric("Confidence", result.get("confidence", 0.0))
            st.metric("Similarity", result.get("similarity", 0.0))

            if result.get("audio_quality_soft_fail"):
                st.warning("⚠️ Audio quality was suboptimal (soft penalty applied)")
    else:
        st.info("Waiting for voice input...")

    st.divider()

    # ==============================================================
    # PHASE 2 — AGE-BASED PLAYBACK (DSP, 5–70)
    # ==============================================================
    st.header("🎧 Age-Based Voice Playback (5–70)")
    st.caption("Upload → pick age → generate voice. Also generate a sample pack ZIP.")

    from voice_age.config import MIN_AGE, MAX_AGE, DEFAULT_SR
    from voice_age.age.control import apply_age_control

    import soundfile as sf
    import librosa

    uploaded2 = st.file_uploader(
        "Upload voice audio for age playback (wav/mp3/m4a/aac/flac/ogg)",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        key="phase2_upload",
    )

    age = st.slider(
        "Target age",
        min_value=int(MIN_AGE),
        max_value=int(MAX_AGE),
        value=25,
        step=1,
        key="phase2_age",
    )

    step = st.selectbox(
        "Sample pack step (ZIP)",
        [1, 2, 5, 10],
        index=2,
        key="phase2_step",
    )

    colA, colB = st.columns(2)
    btn_single = colA.button("Generate for this age", key="phase2_btn_single")
    btn_pack = colB.button("Generate sample pack", key="phase2_btn_pack")

    st.divider()

    if uploaded2 is None:
        st.info("Upload an audio file to generate age-based voice.")
        return

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        suffix = Path(uploaded2.name).suffix.lower()
        raw_path = td / f"input{suffix}"
        raw_path.write_bytes(uploaded2.getbuffer())

        # Standardize to DEFAULT_SR mono WAV
        wav_path = td / "input_std.wav"
        try:
            ffmpeg_to_wav_16k_mono(raw_path, wav_path, sr=int(DEFAULT_SR))
        except Exception as e:
            st.error(
                "Audio conversion failed. ffmpeg is probably missing.\n\n"
                "For Streamlit Cloud: add `ffmpeg` to `packages.txt`."
            )
            st.code(str(e))
            return

        # Load standardized audio
        wav, sr = librosa.load(str(wav_path), sr=int(DEFAULT_SR), mono=True)
        wav = wav.astype("float32")

        st.subheader("Input preview (standardized)")
        st.audio(read_bytes(wav_path), format="audio/wav")
        st.write(f"Processing sample rate: {sr} Hz")

        # ---------------- Single age output ----------------
        if btn_single:
            outdir = PROJECT_ROOT / "data" / "outputs" / "ui_single"
            outdir.mkdir(parents=True, exist_ok=True)

            out_path = outdir / f"{selected_user}_age_{age}.wav"
            y = apply_age_control(wav, sr, float(age))
            sf.write(str(out_path), y, sr)

            st.success(f"Generated: {out_path}")
            st.audio(read_bytes(out_path), format="audio/wav")
            st.download_button(
                "Download age WAV",
                data=read_bytes(out_path),
                file_name=out_path.name,
                mime="audio/wav",
            )

        # ---------------- Pack output ----------------
        if btn_pack:
            ts = time.strftime("%Y%m%d_%H%M%S")
            outdir = PROJECT_ROOT / "data" / "outputs" / "ui_packs" / f"pack_{selected_user}_{ts}_step{step}"
            outdir.mkdir(parents=True, exist_ok=True)

            prog = st.progress(0)
            ages = list(range(int(MIN_AGE), int(MAX_AGE) + 1, int(step)))

            for i, a in enumerate(ages, start=1):
                y = apply_age_control(wav, sr, float(a))
                out_path = outdir / f"age_{a}.wav"
                sf.write(str(out_path), y, sr)
                prog.progress(int(i * 100 / len(ages)))

            zip_bytes = make_zip_bytes(outdir)

            st.success(f"Sample pack ready: {outdir}  ({len(ages)} files)")
            st.download_button(
                "Download ZIP (all samples)",
                data=zip_bytes,
                file_name=f"{selected_user}_voice_samples_{ts}_step{step}.zip",
                mime="application/zip",
            )

            st.write("Listen in order:")
            for a in ages:
                p = outdir / f"age_{a}.wav"
                st.markdown(f"**Age {a}**")
                st.audio(read_bytes(p), format="audio/wav")