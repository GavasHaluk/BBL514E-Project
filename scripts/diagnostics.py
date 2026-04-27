"""Post-training RF diagnostics: feature-importance plot and learning curve."""
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import schema
from scripts.train_full import load_dataset, prepare, split, make_pipeline, DEFAULT_DATA

RUNS_DIR = REPO / "models" / "_runs"
FIGS_DIR = REPO / "models" / "figures"
SEED = 7

LC_MAX_TRAIN_ROWS = 500_000
LC_N_ESTIMATORS = 100
LC_CV_FOLDS = 3
LC_TRAIN_SIZES = np.array([0.1, 0.25, 0.5, 0.75, 1.0])


def feature_importance(rf_pipe):
    rf = rf_pipe.named_steps["clf"]
    corr_filter = rf_pipe.named_steps["corr_filter"]

    surviving_names = np.array(schema.EXPECTED_FEATURES)[corr_filter.keep_]
    importances = rf.feature_importances_
    assert len(surviving_names) == len(importances), \
        f"name/importance mismatch: {len(surviving_names)} vs {len(importances)}"

    order = np.argsort(importances)[::-1][:15]
    top_names = surviving_names[order][::-1]
    top_vals = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_names, top_vals, color="#3b6fb6")
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Random Forest top 15 of {len(surviving_names)} features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "rf_feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out}")

    return [(n, float(v)) for n, v in zip(top_names[::-1], top_vals[::-1])]


def learning_curve_plot(X_tr, y_tr):
    if len(X_tr) > LC_MAX_TRAIN_ROWS:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), LC_MAX_TRAIN_ROWS, replace=False)
        X_use = X_tr.iloc[idx] if hasattr(X_tr, "iloc") else X_tr[idx]
        y_use = y_tr[idx]
        print(f"  Subsampled to {LC_MAX_TRAIN_ROWS:,} rows")
    else:
        X_use, y_use = X_tr, y_tr

    rf = RandomForestClassifier(
        n_estimators=LC_N_ESTIMATORS,
        n_jobs=-1,
        random_state=SEED,
    )
    pipe = make_pipeline(rf)

    cv = StratifiedKFold(n_splits=LC_CV_FOLDS, shuffle=True, random_state=SEED)
    print(f"  Running learning_curve: {len(LC_TRAIN_SIZES)} sizes x {LC_CV_FOLDS} folds...")

    t0 = time.time()
    sizes, train_scores, val_scores = learning_curve(
        pipe, X_use, y_use,
        train_sizes=LC_TRAIN_SIZES,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,
        random_state=SEED,
        shuffle=True,
        verbose=1,
    )
    print(f"  learning_curve done in {(time.time()-t0)/60:.1f} min")

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, train_mean, "o-", color="#d6604d", label="Training F1")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    color="#d6604d", alpha=0.2)
    ax.plot(sizes, val_mean, "o-", color="#3b6fb6", label="Cross-val F1")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    color="#3b6fb6", alpha=0.2)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 (macro)")
    ax.set_title(f"Random Forest learning curve (n_estimators={LC_N_ESTIMATORS}, cv={LC_CV_FOLDS})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS_DIR / "rf_learning_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out}")

    return {
        "train_sizes": sizes.tolist(),
        "train_f1_mean": train_mean.tolist(),
        "val_f1_mean": val_mean.tolist(),
        "final_gap": float(train_mean[-1] - val_mean[-1]),
    }


def write_summary(top_features, lc):
    lines = [
        "# Random Forest diagnostics\n",
        "## Top 15 features by Gini importance\n",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for name, val in top_features:
        lines.append(f"| {name} | {val:.4f} |")
    lines += [
        "\n## Learning curve\n",
        f"- Final training F1: {lc['train_f1_mean'][-1]:.4f}",
        f"- Final cross-val F1: {lc['val_f1_mean'][-1]:.4f}",
        f"- Train/val gap: {lc['final_gap']:.4f}",
        "",
        "A small gap between training and cross-val F1 indicates the model "
        "is not overfitting. Cross-val F1 flattening at full data suggests "
        "additional samples would yield marginal gains.",
    ]
    out = FIGS_DIR / "diagnostics_summary.md"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    rf_path = RUNS_DIR / "random_forest.pkl"
    if not rf_path.exists():
        sys.exit(f"Missing {rf_path}. Run train_full.py first.")

    print("Loading random_forest.pkl...")
    rf_pipe = joblib.load(rf_path)

    print("\n[1/2] Feature importance")
    top_features = feature_importance(rf_pipe)
    for name, val in top_features[:5]:
        print(f"    {name:40s}  {val:.4f}")

    print("\n[2/2] Learning curve")
    df = load_dataset(DEFAULT_DATA)
    X, y = prepare(df)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split(X, y)
    lc = learning_curve_plot(X_tr, y_tr)

    write_summary(top_features, lc)
    print(f"\nDone. Figures in {FIGS_DIR}/")


if __name__ == "__main__":
    main()
