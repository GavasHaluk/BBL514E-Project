"""Hits /api/predict with the realistic sample against every model in the
registry and asserts AUC stays above a per-model floor.

The whole point of this test is catching the kind of regression we ate
during the partner-pipeline salvage: a joblib that loads cleanly but ships
with the wrong `classes_`, the wrong feature subset, or a refit-on-bad-data
scaler -- the API still returns 200, the AUC just collapses.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "app" / "bundled" / "realistic_sample.csv"

# These two are intrinsically weaker on flow features. Naive Bayes is also
# badly miscalibrated at threshold 0.5 -- AUC is the only honest signal.
PER_MODEL_AUC_FLOOR = {
    "naive_bayes": 0.60,
    "svm_linear":  0.80,
}
DEFAULT_AUC_FLOOR = 0.95

client = TestClient(app)


def _floor_for(model_id: str) -> float:
    mid = model_id.lower()
    for needle, val in PER_MODEL_AUC_FLOOR.items():
        if needle in mid:
            return val
    return DEFAULT_AUC_FLOOR


def _registry_ids():
    r = client.get("/api/models")
    r.raise_for_status()
    return [m["id"] for m in r.json()["models"]]


REGISTRY_IDS = _registry_ids()


@pytest.mark.skipif(not SAMPLE.exists(), reason=f"missing bundled sample at {SAMPLE}")
@pytest.mark.skipif(not REGISTRY_IDS, reason="no models in registry")
@pytest.mark.parametrize("model_id", REGISTRY_IDS)
def test_predict_realistic_sample(model_id):
    with SAMPLE.open("rb") as fh:
        r = client.post(
            "/api/predict",
            files={"file": (SAMPLE.name, fh, "text/csv")},
            data={"model_id": model_id},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    metrics = body.get("metrics")
    assert metrics is not None, "Label column should produce a metrics block"

    auc = metrics["auc"]
    assert auc is not None, "binary AUC should be populated for realistic_sample"
    floor = _floor_for(model_id)
    assert auc >= floor, f"{model_id}: AUC {auc:.4f} below floor {floor:.2f}"

    # If the predicted column is the wrong class label, this catches it.
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fn"] > 0, "no positives detected -- classes_ mapping is probably wrong"


def test_compare_endpoint():
    if not REGISTRY_IDS or not SAMPLE.exists():
        pytest.skip("registry empty or sample missing")
    with SAMPLE.open("rb") as fh:
        r = client.post(
            "/api/predict_compare",
            files={"file": (SAMPLE.name, fh, "text/csv")},
            data={"model_ids": ",".join(REGISTRY_IDS)},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == len(REGISTRY_IDS)
    assert all("error" not in row for row in body["results"]), body["results"]
