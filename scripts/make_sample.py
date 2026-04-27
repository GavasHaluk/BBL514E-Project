"""Stratified ~500-row sample from a CICIDS2017 MachineLearningCVE file for demos."""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("/Users/sisyphos/Documents/claude/project/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "samples" / "tiny_sample.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    df = pd.read_csv(args.source)
    df.columns = df.columns.str.strip()

    half = args.rows // 2
    benign = df[df["Label"].str.upper() == "BENIGN"].sample(n=half, random_state=7)
    attack = df[df["Label"].str.upper() != "BENIGN"].sample(n=args.rows - half, random_state=7)
    sample = pd.concat([benign, attack]).sample(frac=1.0, random_state=7).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"Wrote {len(sample)} rows to {args.out}")
    print(sample["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
