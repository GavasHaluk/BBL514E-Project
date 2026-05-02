"""Refit the prep step on partner's pipelines so they're usable.

Partner's joblibs have a fitted clf inside an unfit ColumnTransformer
(they wrapped the prep around a pre-fit model instead of pipe.fit-ing
end to end). We rebuild prep, fit it on our train split, splice it back.
Output: models/models_a/<name>_fixed.joblib.

Run inside the deployment image (sklearn version has to match the pickle):
  docker run --rm -v $(pwd)/models:/app/models:rw \\
    -v /path/to/MachineLearningCVE:/data:ro \\
    bbl514e-ids:latest python scripts/splice_partner_pipelines.py --data /data
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_full import load_dataset, prepare, split  # noqa: E402

PARTNER_DIR = REPO / "models" / "models_a"


def rebuild_prep(unfit_prep, X_fit):
    """Structurally clone an unfitted ColumnTransformer and fit it on X_fit."""
    new_specs = [(name, clone(trans), cols) for name, trans, cols in unfit_prep.transformers]
    kw = {"remainder": getattr(unfit_prep, "remainder", "drop")}
    if hasattr(unfit_prep, "sparse_threshold"):
        kw["sparse_threshold"] = unfit_prep.sparse_threshold
    new_prep = ColumnTransformer(new_specs, **kw)
    new_prep.fit(X_fit)
    return new_prep


def splice_one(path, X_fit):
    print(f"\n--- {path.name} ---")
    m = joblib.load(path)
    if not hasattr(m, "named_steps") or "prep" not in m.named_steps:
        print("  not a Pipeline with 'prep' step; skip")
        return None
    prep = m.named_steps["prep"]
    print(f"  prep transformers: {[(n, type(t).__name__) for n, t, _ in prep.transformers]}")

    feats = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else list(X_fit.columns)
    missing = [c for c in feats if c not in X_fit.columns]
    if missing:
        print(f"  skip: pipeline wants {len(missing)} cols we don't have, e.g. {missing[:3]}")
        return None
    X_sub = X_fit[feats]

    new_prep = rebuild_prep(prep, X_sub)
    m.steps[0] = ("prep", new_prep)

    try:
        _ = m.predict_proba(X_sub.iloc[:5]) if hasattr(m, "predict_proba") else m.predict(X_sub.iloc[:5])
    except Exception as e:
        print(f"  smoke test failed: {type(e).__name__}: {e}")
        return None

    out = path.parent / f"{path.stem}_fixed.joblib"
    joblib.dump(m, out)
    print(f"  -> {out.name}")
    return out


def score_one(path, X_test, y_bin):
    m = joblib.load(path)
    feats = list(m.feature_names_in_) if hasattr(m, "feature_names_in_") else list(X_test.columns)
    X = X_test[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if hasattr(m, "predict_proba"):
        try:
            proba = m.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            proba = None
    else:
        proba = None
    if proba is None:
        score = m.decision_function(X)
        proba = (score - score.min()) / (score.max() - score.min() + 1e-9)
    auc = roc_auc_score(y_bin, proba)
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_bin == 1)).sum())
    fp = int(((pred == 1) & (y_bin == 0)).sum())
    fn = int(((pred == 0) & (y_bin == 1)).sum())
    acc = float((pred == y_bin).mean())
    return auc, acc, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/data"),
                    help="MachineLearningCVE directory (read-only inside container)")
    ap.add_argument("--sample", type=Path, default=REPO / "samples" / "realistic_sample.csv")
    args = ap.parse_args()

    print(f"Loading dataset from {args.data}...")
    df = load_dataset(args.data)
    X, y = prepare(df)
    X_tr, _, _, y_tr, _, _ = split(X, y)
    print(f"Train rows for refit: {len(X_tr):,}")

    inputs = sorted(p for p in PARTNER_DIR.glob("*.joblib") if not p.stem.endswith("_fixed"))
    print(f"Partner pipelines to splice: {len(inputs)}")
    fixed = []
    for p in inputs:
        try:
            out = splice_one(p, X_tr)
            if out is not None:
                fixed.append(out)
        except Exception as e:
            print(f"  SPLICE FAILED: {type(e).__name__}: {e}")

    if not fixed:
        print("\nNo fixed artifacts produced.")
        return
    if not args.sample.exists():
        print(f"\nNo sample at {args.sample}; skipping scoring")
        return

    print(f"\n=== Scoring on {args.sample.name} ===")
    sample = pd.read_csv(args.sample)
    sample.columns = sample.columns.str.strip()
    y_s = sample.pop("Label")
    y_bin = (y_s != "BENIGN").astype(int).to_numpy()
    print(f"{len(sample)} rows, {y_bin.sum()} mal ({100*y_bin.mean():.1f}%)\n")
    print(f"{'model':<42} {'AUC':>7} {'acc@0.5':>9} {'tp':>6} {'fp':>6} {'fn':>6}")
    print("-" * 82)
    for f in fixed:
        try:
            auc, acc, tp, fp, fn = score_one(f, sample, y_bin)
            print(f"{f.name:<42} {auc:>7.4f} {acc:>9.4f} {tp:>6} {fp:>6} {fn:>6}")
        except Exception as e:
            print(f"{f.name:<42}  SCORE FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
