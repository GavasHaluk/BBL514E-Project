import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from app.predictor import BENIGN, MALICIOUS

ROC_MAX_POINTS = 120


def normalize_truth(y_raw: np.ndarray) -> np.ndarray:
    y = np.asarray(y_raw).astype(str)
    return np.where(np.char.upper(y) == "BENIGN", BENIGN, MALICIOUS)


def compute(y_true: np.ndarray, y_pred: np.ndarray, proba_mal: np.ndarray) -> dict:
    labels = [BENIGN, MALICIOUS]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel().tolist()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="binary", pos_label=MALICIOUS, zero_division=0
    )

    y_bin = (y_true == MALICIOUS).astype(int)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc": None,
        "auc": None,
    }

    # roc_auc_score throws if there's only one class present
    if len(np.unique(y_bin)) == 2:
        result["auc"] = float(roc_auc_score(y_bin, proba_mal))
        fpr, tpr, thr = roc_curve(y_bin, proba_mal)
        if len(fpr) > ROC_MAX_POINTS:
            idx = np.linspace(0, len(fpr) - 1, ROC_MAX_POINTS).astype(int)
            fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
        # roc_curve emits +inf as the leading threshold; clip so the JSON is sane.
        thr = np.where(np.isfinite(thr), thr, 1.0)
        result["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thr.tolist()}

    return result


def per_attack_recall(y_raw, y_pred):
    """For each non-benign attack name in y_raw, report support and recall.

    Sorted by support descending so the headline classes (DoS Hulk, PortScan)
    show up first. Single benign label collapses everything in the report --
    omit it.
    """
    y_raw = np.asarray(y_raw)
    y_pred = np.asarray(y_pred)
    out = []
    for name in pd.unique(y_raw):
        if str(name).upper() == "BENIGN":
            continue
        mask = y_raw == name
        n = int(mask.sum())
        tp = int((y_pred[mask] == MALICIOUS).sum())
        out.append({
            "label": str(name),
            "support": n,
            "tp": tp,
            "fn": n - tp,
            "recall": tp / n if n else 0.0,
        })
    out.sort(key=lambda r: -r["support"])
    return out
