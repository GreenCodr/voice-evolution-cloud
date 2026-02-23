from __future__ import annotations

from pathlib import Path
import numpy as np

def run_post_engine(
    wav: np.ndarray,
    sr: int,
    engine: str = "none",
    *,
    rvc_model_path: str | None = None,
    pitch_shift: int = 0,
) -> np.ndarray:
    """
    Post-processing stage AFTER DSP.
    - engine="none": returns wav unchanged
    - engine="rvc": (placeholder for next step) will run RVC conversion
    """
    engine = (engine or "none").lower().strip()

    if engine in ("none", "off", "dsp"):
        return wav.astype(np.float32)

    if engine == "rvc":
        # Placeholder: next step we’ll implement the real call.
        # For now, keep behavior safe (no quality regression).
        if not rvc_model_path:
            raise ValueError("engine=rvc requires --rvc-model path")
        p = Path(rvc_model_path)
        if not p.exists():
            raise FileNotFoundError(f"RVC model not found: {p}")
        return wav.astype(np.float32)

    raise ValueError(f"Unknown post engine: {engine}")
