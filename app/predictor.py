import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BENIGN = "Benign"
MALICIOUS = "Malicious"
DECISION_THRESHOLD = 0.5


class SklearnPredictor:
    name = "sklearn"

    def __init__(self, model_path):
        import joblib
        self.model = joblib.load(model_path)
        self.version = getattr(self.model, "version", "unknown")
        self.trained_at = getattr(self.model, "trained_at", None)

        # subset/reorder X if the model only knows a column subset
        feats = getattr(self.model, "feature_names_in_", None)
        self.feature_names = list(feats) if feats is not None else None

        # partner's joblibs use 0/1, ours use Benign/Malicious
        classes = list(self.model.classes_)
        if MALICIOUS in classes:
            self._mal_idx = classes.index(MALICIOUS)
        elif 1 in classes:
            self._mal_idx = classes.index(1)
        else:
            raise ValueError(f"unrecognised classes_: {classes}")

    def predict(self, X: pd.DataFrame):
        if self.feature_names is not None:
            X = X[self.feature_names]
        proba = self.model.predict_proba(X)[:, self._mal_idx]
        labels = np.where(proba >= DECISION_THRESHOLD, MALICIOUS, BENIGN)
        return labels, proba
