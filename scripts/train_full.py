"""Train 7 classifiers on CICIDS2017 with grid-search CV.

Models: Naive Bayes, Decision Tree, Random Forest, MLP, SVM-RBF, KNN, plus
an RF+MLP soft-voting ensemble. 70/15/15 split, 5-fold stratified CV.

Resumable: a model is skipped if both <name>.pkl and <name>_metrics.json
exist under models/_runs/. Pass --rerun <name> to redo one (or 'all').

Run scripts/aggregate_runs.py afterwards to build the results table.
"""
import argparse
import glob
import json
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import metrics, schema
from app.predictor import BENIGN, MALICIOUS
from app.preprocessing import CorrelationFilter

DEFAULT_DATA = Path("/Users/sisyphos/Documents/claude/project/MachineLearningCVE")
RUNS_DIR = REPO / "models" / "_runs"
SEED = 7


# Each entry: (name, classifier, param_grid, train_subsample_n_or_None, gridsearch_n_jobs)
def model_specs():
    return [
        ("naive_bayes", GaussianNB(), {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
        }, None, -1),
        ("decision_tree", DecisionTreeClassifier(random_state=SEED), {
            "clf__max_depth": [None, 20, 40],
            "clf__min_samples_leaf": [1, 5],
        }, None, 2),
        ("random_forest", RandomForestClassifier(n_jobs=-1, random_state=SEED), {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 30],
            "clf__min_samples_leaf": [1, 5],
        }, None, 1),
        ("mlp", MLPClassifier(max_iter=80, early_stopping=False,
                              n_iter_no_change=8, random_state=SEED), {
            "clf__hidden_layer_sizes": [(64,), (128, 64)],
            "clf__alpha": [1e-4, 1e-3],
        }, None, 2),
        ("svm_rbf", SVC(kernel="rbf", probability=True, cache_size=1000,
                        random_state=SEED), {
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale", 0.01],
        }, 100_000, 5),
        ("knn", KNeighborsClassifier(n_jobs=-1), {
            "clf__n_neighbors": [5, 11, 25],
            "clf__weights": ["uniform", "distance"],
        }, 200_000, 1),
    ]


def make_pipeline(clf):
    return ImbPipeline([
        ("scaler", StandardScaler()),
        ("corr_filter", CorrelationFilter(threshold=0.95)),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", clf),
    ])


def load_dataset(data_dir):
    files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        sys.exit(f"No CSVs in {data_dir}")
    print(f"Reading {len(files)} files from {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        frames.append(schema.normalize_columns(df))
        print(f"  {Path(f).name:60s} {len(frames[-1]):>10,} rows")
    df = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(df):,} rows")
    return df


def prepare(df):
    missing = schema.missing_columns(df)
    if missing:
        sys.exit(f"Missing columns: {missing[:5]}...")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.where(df[schema.LABEL_COLUMN].astype(str).str.upper() == "BENIGN", BENIGN, MALICIOUS)
    X = df[schema.EXPECTED_FEATURES].astype(np.float32)
    return X, y


def split(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def stratified_subsample(X, y, n):
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return X_sub, y_sub


# Match CV scoring to the test-time metric (binary F1 on MALICIOUS).
F1_BINARY = make_scorer(f1_score, pos_label=MALICIOUS)


def train_one(name, clf, grid, subsample_n, gs_n_jobs, X_tr, y_tr, X_te, y_te):
    print(f"\n=== {name} ===", flush=True)
    if subsample_n:
        X_use, y_use = stratified_subsample(X_tr, y_tr, subsample_n)
        print(f"Train subsample: {len(X_use):,} rows (full train was {len(X_tr):,})", flush=True)
    else:
        X_use, y_use = X_tr, y_tr

    pipe = make_pipeline(clf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring=F1_BINARY, n_jobs=gs_n_jobs, refit=True, verbose=2)

    t0 = time.time()
    gs.fit(X_use, y_use)
    fit_seconds = time.time() - t0
    print(f"Fit done in {fit_seconds/60:.1f} min  best CV f1={gs.best_score_:.4f}")
    print(f"Best params: {gs.best_params_}")

    best = gs.best_estimator_
    mal_idx = list(best.classes_).index(MALICIOUS)
    proba_te = best.predict_proba(X_te)[:, mal_idx]
    pred_te = best.predict(X_te)

    m = metrics.compute(y_te, pred_te, proba_te)
    cm = m["confusion_matrix"]
    fpr = cm["fp"] / max(cm["fp"] + cm["tn"], 1)
    tpr = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    print(f"Test acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
          f"tpr={tpr:.4f}  fpr={fpr:.4f}")

    return best, {
        "name": name,
        "best_params": {k: (v if not callable(v) else str(v)) for k, v in gs.best_params_.items()},
        "best_cv_score": float(gs.best_score_),
        "fit_seconds": fit_seconds,
        "n_train_used": int(len(X_use)),
        "n_test": int(len(X_te)),
        "test_metrics": {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
            "tpr": tpr,
            "fpr": fpr,
            "confusion_matrix": cm,
        },
    }


def train_ensemble(X_tr, y_tr, X_te, y_te):
    rf_path = RUNS_DIR / "random_forest.pkl"
    mlp_path = RUNS_DIR / "mlp.pkl"
    if not (rf_path.exists() and mlp_path.exists()):
        print("Skipping ensemble: random_forest.pkl and mlp.pkl must both exist first.")
        return None, None
    print("\n=== ensemble_rf_mlp ===")

    rf_pipe = joblib.load(rf_path)
    mlp_pipe = joblib.load(mlp_path)

    voter = VotingClassifier(
        estimators=[("rf", rf_pipe), ("mlp", mlp_pipe)],
        voting="soft",
    )
    X_seed, _ = stratified_subsample(X_tr, y_tr, 2000)
    t0 = time.time()
    voter.estimators_ = [rf_pipe, mlp_pipe]
    voter.le_ = None
    voter.classes_ = np.array(sorted(set(y_tr)))
    voter.named_estimators_ = {"rf": rf_pipe, "mlp": mlp_pipe}
    _ = voter.predict_proba(X_seed[:10])
    fit_seconds = time.time() - t0

    mal_idx = list(voter.classes_).index(MALICIOUS)
    proba_te = voter.predict_proba(X_te)[:, mal_idx]
    pred_te = np.where(proba_te >= 0.5, MALICIOUS, BENIGN)
    m = metrics.compute(y_te, pred_te, proba_te)
    cm = m["confusion_matrix"]
    fpr = cm["fp"] / max(cm["fp"] + cm["tn"], 1)
    tpr = cm["tp"] / max(cm["tp"] + cm["fn"], 1)
    print(f"Test acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
          f"tpr={tpr:.4f}  fpr={fpr:.4f}")

    return voter, {
        "name": "ensemble_rf_mlp",
        "best_params": {"voting": "soft", "components": ["random_forest", "mlp"]},
        "best_cv_score": None,
        "fit_seconds": fit_seconds,
        "n_train_used": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_metrics": {
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc": m["auc"],
            "tpr": tpr,
            "fpr": fpr,
            "confusion_matrix": cm,
        },
    }


def already_done(name):
    return (RUNS_DIR / f"{name}.pkl").exists() and (RUNS_DIR / f"{name}_metrics.json").exists()


def save(name, model, meta):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, RUNS_DIR / f"{name}.pkl", compress=3)
    meta["saved_at"] = str(date.today())
    with open(RUNS_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    size_mb = (RUNS_DIR / f"{name}.pkl").stat().st_size / 1024 / 1024
    print(f"Saved {name}.pkl ({size_mb:.1f} MB) + {name}_metrics.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--rerun", default="",
                   help="Comma-separated model names to retrain even if cached. Use 'all' to redo everything.")
    p.add_argument("--only", default="",
                   help="Comma-separated model names to train (skip the rest).")
    args = p.parse_args()

    rerun = set(args.rerun.split(",")) if args.rerun else set()
    only = set(args.only.split(",")) if args.only else None

    df = load_dataset(args.data)
    X, y = prepare(df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    n_mal = int((y_tr == MALICIOUS).sum())
    print(f"Split: train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}")
    print(f"Train class balance: {len(y_tr)-n_mal:,} benign / {n_mal:,} malicious "
          f"({100*n_mal/len(y_tr):.1f}% malicious)")

    started_at = time.time()
    for name, clf, grid, subsample_n, gs_n_jobs in model_specs():
        if only and name not in only:
            continue
        if "all" not in rerun and name not in rerun and already_done(name):
            print(f"\n[skip] {name}: artifact exists. Pass --rerun {name} to redo.", flush=True)
            continue
        model, meta = train_one(name, clf, grid, subsample_n, gs_n_jobs, X_tr, y_tr, X_te, y_te)
        save(name, model, meta)

    if (not only or "ensemble_rf_mlp" in only) and (
        "all" in rerun or "ensemble_rf_mlp" in rerun or not already_done("ensemble_rf_mlp")
    ):
        voter, meta = train_ensemble(X_tr, y_tr, X_te, y_te)
        if voter is not None:
            save("ensemble_rf_mlp", voter, meta)

    elapsed = (time.time() - started_at) / 60
    print(f"\nAll done. Total wall time this session: {elapsed:.1f} min.")
    print("Run scripts/aggregate_runs.py to build the results table and pick a winner.")


if __name__ == "__main__":
    main()
