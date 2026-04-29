import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import metrics, schema
from app.predictor import MALICIOUS, get_predictor

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ROW_PREVIEW_CAP = 10_000
DATASET_MALICIOUS_BASELINE_PCT = 17.0  # CICIDS2017 class prior
ROW_EXTRA_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
]

STATIC_DIR = Path(__file__).parent / "static"
SAMPLE_CSV = Path(__file__).parent / "bundled" / "demo_sample.csv"

app = FastAPI(title="CICIDS2017 Intrusion Detection", version="0.2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = get_predictor()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "predictor": {
            "name": predictor.name,
            "version": getattr(predictor, "version", "unknown"),
            "trained_at": getattr(predictor, "trained_at", None),
        },
        "baseline_malicious_pct": DATASET_MALICIOUS_BASELINE_PCT,
    }


@app.get("/api/schema")
def get_schema():
    return {"features": schema.EXPECTED_FEATURES, "label_column": schema.LABEL_COLUMN}


def _parse_csv(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large: {len(raw)} bytes (limit {MAX_UPLOAD_BYTES})")
    if not raw:
        raise HTTPException(400, "empty upload")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"bad csv: {e}")
    df = schema.normalize_columns(df)
    missing = schema.missing_columns(df)
    if missing:
        raise HTTPException(422, {"message": "missing feature columns", "missing": missing})
    return df


def _score(df: pd.DataFrame):
    y_true_raw = df.pop(schema.LABEL_COLUMN).to_numpy() if schema.LABEL_COLUMN in df.columns else None
    extras = {c: df[c].to_numpy() for c in ROW_EXTRA_FEATURES if c in df.columns}
    X = df[schema.EXPECTED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labels, proba_mal = predictor.predict(X)
    return labels, proba_mal, y_true_raw, extras


def _build_payload(labels, proba_mal, y_true_raw, extras) -> dict:
    n = int(len(labels))
    n_mal = int((labels == MALICIOUS).sum())
    result_metrics = None
    if y_true_raw is not None:
        result_metrics = metrics.compute(metrics.normalize_truth(y_true_raw), labels, proba_mal)

    order = np.lexsort((-proba_mal, labels != MALICIOUS))
    preview = order[:ROW_PREVIEW_CAP]

    rows = []
    for i in preview:
        row = {
            "row": int(i),
            "predicted": str(labels[i]),
            "proba_malicious": round(float(proba_mal[i]), 4),
            "true": str(y_true_raw[i]) if y_true_raw is not None else None,
        }
        for col, arr in extras.items():
            val = arr[i]
            if isinstance(val, (np.integer, np.floating)):
                val = val.item()
            row[col] = val
        rows.append(row)

    return {
        "predictor": {"name": predictor.name, "version": getattr(predictor, "version", "unknown")},
        "n_rows": n,
        "summary": {
            "benign": n - n_mal,
            "malicious": n_mal,
            "malicious_pct": round(100.0 * n_mal / n, 2) if n else 0.0,
            "baseline_malicious_pct": DATASET_MALICIOUS_BASELINE_PCT,
        },
        "metrics": result_metrics,
        "rows": rows,
        "rows_truncated": n > ROW_PREVIEW_CAP,
        "row_extra_features": [c for c in ROW_EXTRA_FEATURES if c in extras],
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    df = _parse_csv(await file.read())
    labels, proba_mal, y_true_raw, extras = _score(df)
    return JSONResponse(_build_payload(labels, proba_mal, y_true_raw, extras))


@app.post("/api/predict.csv")
async def predict_csv(file: UploadFile = File(...)):
    df = _parse_csv(await file.read())
    labels, proba_mal, y_true_raw, _ = _score(df)

    def stream():
        buf = io.StringIO()
        w = csv.writer(buf)
        header = ["row", "predicted", "proba_malicious"]
        if y_true_raw is not None:
            header.append("true")
        w.writerow(header)
        yield buf.getvalue()
        buf.seek(0); buf.truncate()
        for i in range(len(labels)):
            row = [i, labels[i], round(float(proba_mal[i]), 4)]
            if y_true_raw is not None:
                row.append(y_true_raw[i])
            w.writerow(row)
            if i % 500 == 0:
                yield buf.getvalue()
                buf.seek(0); buf.truncate()
        yield buf.getvalue()

    return StreamingResponse(
        stream(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


@app.get("/api/sample")
def sample():
    if not SAMPLE_CSV.exists():
        raise HTTPException(503, "No bundled sample available in this image.")
    df = _parse_csv(SAMPLE_CSV.read_bytes())
    labels, proba_mal, y_true_raw, extras = _score(df)
    payload = _build_payload(labels, proba_mal, y_true_raw, extras)
    payload["source"] = SAMPLE_CSV.name
    return JSONResponse(payload)
