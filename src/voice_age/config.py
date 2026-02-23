from __future__ import annotations
from pathlib import Path

# repo root (…/voice project)
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SR = 16000

# Age range (UI constraints)
BASE_AGE = 23
MIN_AGE = 5
MAX_AGE = 70
