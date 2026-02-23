from pathlib import Path
import argparse

from voice_age.cli import load_wav, write_wav
from voice_age.age.control import apply_age_control
from voice_age.config import DEFAULT_SR, MIN_AGE, MAX_AGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input wav path")
    ap.add_argument("--outdir", default="data/outputs/all_samples", help="Output directory")
    ap.add_argument("--step", type=int, default=5, help="Age step (default 5)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wav, sr = load_wav(inp, target_sr=DEFAULT_SR)

    for age in range(MIN_AGE, MAX_AGE + 1, args.step):
        y = apply_age_control(wav, sr, float(age))
        out = outdir / f"age_{age}.wav"
        write_wav(out, y, sr)
        print("WROTE:", out)

    print("DONE ✅ ->", outdir)


if __name__ == "__main__":
    main()