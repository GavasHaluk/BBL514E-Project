import numpy as np
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

    if y_bin.min() != y_bin.max():
        auc = float(roc_auc_score(y_bin, proba_mal))
        fpr, tpr, _ = roc_curve(y_bin, proba_mal)
        if len(fpr) > ROC_MAX_POINTS:
            idx = np.linspace(0, len(fpr) - 1, ROC_MAX_POINTS).astype(int)
            fpr, tpr = fpr[idx], tpr[idx]
        result["auc"] = auc
        result["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    return result
