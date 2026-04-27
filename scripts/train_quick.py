"""Quick single-RF run for end-to-end smoke testing. No correlation filter,
no CV, no grid search. Use scripts/train_full.py for the real run."""
import argparse
import glob
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import schema
from app.predictor import BENIGN, MALICIOUS

DEFAULT_DATA = Path("/Users/sisyphos/Documents/claude/project/MachineLearningCVE")
DEFAULT_OUT = REPO / "models" / "model.pkl"


def load_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        sys.exit(f"No CSVs in {data_dir}")
    print(f"Reading {len(files)} files from {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df = schema.normalize_columns(df)
        frames.append(df)
        print(f"  {Path(f).name:60s} {len(df):>10,} rows")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df):,} rows")
    return df


def prepare(df, subsample):
    missing = schema.missing_columns(df)
    if missing:
        sys.exit(f"Missing columns in dataset: {missing[:5]}...")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.where(df[schema.LABEL_COLUMN].astype(str).str.upper() == "BENIGN", BENIGN, MALICIOUS)
    X = df[schema.EXPECTED_FEATURES].astype(np.float32)

    if subsample and subsample < len(X):
        idx = pd.Series(range(len(X))).groupby(y).sample(
            frac=subsample / len(X), random_state=7
        ).values
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]
        print(f"Subsampled to {len(X):,} rows (stratified)")

    n_mal = int((y == MALICIOUS).sum())
    print(f"Class balance: {len(y) - n_mal:,} benign / {n_mal:,} malicious "
          f"({100 * n_mal / len(y):.1f}% malicious)")
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--subsample", type=int, default=None,
                   help="Use a stratified subsample of N rows (default: full dataset)")
    p.add_argument("--trees", type=int, default=200)
    args = p.parse_args()

    df = load_dataset(args.data)
    X, y = prepare(df, args.subsample)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=7)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=7)
    print(f"Split: train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")

    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=7)),
        ("clf", RandomForestClassifier(
            n_estimators=args.trees,
            n_jobs=-1,
            random_state=7,
        )),
    ])

    print(f"Fitting RF (n_estimators={args.trees})...")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"Fit done in {time.time() - t0:.1f}s")

    mal_idx = list(pipe.classes_).index(MALICIOUS)
    proba_te = pipe.predict_proba(X_te)[:, mal_idx]
    pred_te = pipe.predict(X_te)
    print("\n--- test set ---")
    print(classification_report(y_te, pred_te, digits=4))
    auc = roc_auc_score((y_te == MALICIOUS).astype(int), proba_te)
    print(f"ROC-AUC: {auc:.4f}")

    pipe.version = "rf-v1-quick"
    pipe.trained_at = str(date.today())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
