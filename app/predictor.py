import logging
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BENIGN = "Benign"
MALICIOUS = "Malicious"
DECISION_THRESHOLD = 0.5


class MockPredictor:
    name = "mock-rule-v1"
    version = "1.0"
    trained_at = str(date.today())

    def predict(self, X: pd.DataFrame):
        port = pd.to_numeric(X["Destination Port"], errors="coerce").fillna(-1).to_numpy()
        fwd_bytes = pd.to_numeric(X["Total Length of Fwd Packets"], errors="coerce").fillna(0).to_numpy()

        web_port = (port == 80) | (port == 443)
        small_fwd = fwd_bytes < 50

        proba = 0.1 + 0.4 * web_port.astype(float) + 0.4 * small_fwd.astype(float)
        proba = np.clip(proba, 0.05, 0.95)

        labels = np.where(proba >= DECISION_THRESHOLD, MALICIOUS, BENIGN)
        return labels, proba


class SklearnPredictor:
    name = "sklearn"

    def __init__(self, model_path: str):
        import joblib
        self.model = joblib.load(model_path)
        self.version = getattr(self.model, "version", "unknown")
        self.trained_at = getattr(self.model, "trained_at", None)

    def predict(self, X: pd.DataFrame):
        classes = list(self.model.classes_)
        mal_idx = classes.index(MALICIOUS)
        proba = self.model.predict_proba(X)[:, mal_idx]
        labels = np.where(proba >= DECISION_THRESHOLD, MALICIOUS, BENIGN)
        return labels, proba


def get_predictor():
    path = os.getenv("MODEL_PATH")
    if not path:
        log.warning("MODEL_PATH not set; falling back to MockPredictor.")
        return MockPredictor()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"MODEL_PATH={path} does not exist. Set MODEL_PATH to a valid pickle "
            "or unset it to use the mock predictor."
        )
    return SklearnPredictor(path)
