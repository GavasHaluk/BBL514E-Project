"""Custom transformers used inside the trained pipeline. Imported by the
container at startup when unpickling models/model.pkl."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops one of each pair of features with |Pearson r| > threshold.

    Greedy left-to-right: keeps the first occurrence and drops any later
    column that correlates too strongly with a kept one.
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)  # constant columns produce NaN
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0.0)

        keep = np.ones(self.n_features_in_, dtype=bool)
        for i in range(self.n_features_in_):
            if not keep[i]:
                continue
            redundant = (abs_corr[i] > self.threshold) & keep
            redundant[: i + 1] = False
            keep[redundant] = False

        self.keep_ = keep
        self.n_dropped_ = int((~keep).sum())
        return self

    def transform(self, X):
        X = np.asarray(X.values if hasattr(X, "values") else X)
        return X[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = np.array([f"x{i}" for i in range(self.n_features_in_)])
        else:
            input_features = np.asarray(input_features)
        return input_features[self.keep_]
