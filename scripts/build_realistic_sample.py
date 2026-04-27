"""Build a CICIDS2017 test CSV in one of two modes.

  realistic: 80/20 benign/malicious, mixed easy + hard attacks.
  stress:    50/50, hard attacks only (no DDoS/PortScan/Hulk/Patator).
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/sisyphos/Documents/claude/project/MachineLearningCVE")
SEED = 7

# Labels matched by regex so an encoding swap (latin-1 -> utf-8) doesn't
# silently zero out the Web Attack rows. The dataset's Web Attack labels
# include a Unicode replacement char (\xef\xbf\xbd) under latin-1; matching
# on the surrounding tokens avoids hardcoding that.
REALISTIC_QUOTA = [
    (r"^DoS Hulk$",                  200),
    (r"^PortScan$",                  150),
    (r"^DDoS$",                      150),
    (r"^FTP-Patator$",                80),
    (r"^SSH-Patator$",                80),
    (r"^DoS GoldenEye$",              60),
    (r"^DoS slowloris$",              60),
    (r"^DoS Slowhttptest$",           60),
    (r"^Bot$",                        80),
    (r"^Web Attack.*Brute Force$",    30),
    (r"^Web Attack.*XSS$",            30),
    (r"^Web Attack.*Sql Injection$",  10),
    (r"^Infiltration$",               10),
    (r"^Heartbleed$",                 11),
]

STRESS_QUOTA = [
    (r"^DoS slowloris$",              250),
    (r"^DoS Slowhttptest$",           250),
    (r"^Bot$",                        600),
    (r"^Web Attack.*Brute Force$",    800),
    (r"^Web Attack.*XSS$",            562),
    (r"^Web Attack.*Sql Injection$",   21),
    (r"^Infiltration$",                36),
    (r"^Heartbleed$",                  11),
]

MODES = {
    "realistic": {"quota": REALISTIC_QUOTA, "attack_frac": 0.20, "out": "realistic_sample.csv"},
    "stress":    {"quota": STRESS_QUOTA,    "attack_frac": 0.50, "out": "stress_sample.csv"},
}


def load_full():
    print("Loading all 8 CSVs...")
    frames = []
    for f in sorted(glob.glob(str(DATA_DIR / "*.csv"))):
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} total rows")
    return df


def stratified_pull(df, mask, label_pattern, n, rng):
    pool = df[mask]
    if len(pool) == 0:
        print(f"  WARNING: 0 rows match '{label_pattern}'. Available labels: "
              f"{sorted(df['Label'].unique())[:8]}...")
        return pool
    take = min(n, len(pool))
    if take < n:
        print(f"  '{label_pattern}': only {len(pool)} available, taking all")
    idx = rng.choice(len(pool), size=take, replace=False)
    return pool.iloc[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODES.keys()), default="realistic")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = MODES[args.mode]
    out_path = args.out if args.out else (REPO / "samples" / cfg["out"])

    rng = np.random.default_rng(SEED)
    df = load_full()
    label_series = df["Label"].astype(str)

    print(f"\nMode: {args.mode}")

    n_attack_target = int(args.rows * cfg["attack_frac"])
    quota_total = sum(c for _, c in cfg["quota"])
    scale = n_attack_target / quota_total
    print(f"Target: {args.rows} rows = {args.rows - n_attack_target} benign + {n_attack_target} malicious")
    print(f"Scaling per-attack quotas by {scale:.3f}\n")

    pieces = []
    used = pd.Series(False, index=df.index)
    print("Sampling attacks:")
    for pattern, count in cfg["quota"]:
        scaled = max(1, int(round(count * scale)))
        mask = label_series.str.contains(pattern, regex=True, na=False) & ~used
        sample = stratified_pull(df, mask, pattern, scaled, rng)
        if len(sample) > 0:
            print(f"  {pattern:50s} -> {len(sample)}")
            pieces.append(sample)
            used.loc[sample.index] = True

    n_benign = args.rows - sum(len(p) for p in pieces)
    print(f"\nSampling Benign: {n_benign}")
    benign_mask = (label_series.str.upper() == "BENIGN") & ~used
    benign = stratified_pull(df, benign_mask, "BENIGN", n_benign, rng)
    pieces.append(benign)

    out_df = pd.concat(pieces, ignore_index=True)
    out_df = out_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(out_df):,} rows)")
    print("\nFinal label distribution:")
    print(out_df["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
