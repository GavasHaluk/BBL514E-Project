import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import metrics, registry, schema
from app.predictor import MALICIOUS

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ROW_PREVIEW_CAP = 10_000
DATASET_MALICIOUS_BASELINE_PCT = 17.0
ROW_EXTRA_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "SYN Flag Count",
    "FIN Flag Count",
    "RST Flag Count",
]

STATIC_DIR = Path(__file__).parent / "static"
BUNDLED_DIR = Path(__file__).parent / "bundled"

# Hand-curated set of demo CSVs shipped inside the image. Order matters --
# the first entry that exists on disk is the one the "Try sample" button
# defaults to.
BUNDLED_SAMPLES = [
    {"id": "realistic", "label": "Realistic mix (5k rows, 80/20)", "filename": "realistic_sample.csv"},
    {"id": "stress",    "label": "Stress (5k rows, 50/50, hard attacks)", "filename": "stress_sample.csv"},
    {"id": "tiny",      "label": "Tiny (500 rows, BENIGN + DDoS)", "filename": "tiny_sample.csv"},
    {"id": "demo",      "label": "Demo (~150 rows)", "filename": "demo_sample.csv"},
]


def _sample_path(sample_id):
    entry = next((s for s in BUNDLED_SAMPLES if s["id"] == sample_id), None)
    if entry is None:
        return None, None
    p = BUNDLED_DIR / entry["filename"]
    return entry, p if p.exists() else None

app = FastAPI(title="CICIDS2017 Intrusion Detection", version="0.3.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _resolve(model_id):
    try:
        return registry.get(model_id)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to load model {model_id!r}: {e}")


@app.get("/api/health")
def health():
    items = registry.listing()
    if not items:
        return {
            "status": "no_models",
            "n_models": 0,
            "models_dir": str(registry.MODELS_DIR),
            "hint": f"drop a .pkl or .joblib into {registry.MODELS_DIR} and reload",
        }
    pred = _resolve(None)
    return {
        "status": "ok",
        "predictor": {
            "name": pred.name,
            "version": getattr(pred, "version", "unknown"),
            "trained_at": getattr(pred, "trained_at", None),
        },
        "default_model_id": registry.default_id(),
        "n_models": len(items),
        "baseline_malicious_pct": DATASET_MALICIOUS_BASELINE_PCT,
    }


@app.get("/api/models")
def list_models(refresh: bool = False):
    items = [{"id": it["id"], "label": it["label"]} for it in registry.listing(force=refresh)]
    if not items:
        return {"models": [], "default_id": None}
    return {"models": items, "default_id": registry.default_id()}


@app.get("/api/models/{model_id}")
def describe_model(model_id: str):
    if not any(it["id"] == model_id for it in registry.listing()):
        raise HTTPException(404, f"unknown model_id: {model_id!r}")
    try:
        meta = registry.describe(model_id)
    except Exception as e:
        raise HTTPException(500, f"failed to describe {model_id!r}: {e}")
    return {"id": model_id, **meta}


@app.get("/api/schema")
def get_schema():
    return {"features": schema.EXPECTED_FEATURES, "label_column": schema.LABEL_COLUMN}


def _parse_csv(raw):
    if not raw:
        raise HTTPException(400, "empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large: {len(raw)} bytes (limit {MAX_UPLOAD_BYTES})")
    try:
        # low_memory=False avoids the mixed-dtype DtypeWarning chatter on
        # CICIDS columns that occasionally hold "Infinity" string literals.
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"bad csv: {e}")
    df = schema.normalize_columns(df)
    missing = schema.missing_columns(df)
    if missing:
        raise HTTPException(422, {"message": "missing feature columns", "missing": missing})
    return df


def _split_frame(df):
    """Pop Label (if present), pull row-extras, and return the X matrix."""
    y_true = df.pop(schema.LABEL_COLUMN).to_numpy() if schema.LABEL_COLUMN in df.columns else None
    extras = {c: df[c].to_numpy() for c in ROW_EXTRA_FEATURES if c in df.columns}
    X = df[schema.EXPECTED_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y_true, extras


def _score(predictor, df):
    X, y_true, extras = _split_frame(df)
    labels, proba_mal = predictor.predict(X)
    return labels, proba_mal, y_true, extras


def _build_payload(predictor, model_id, labels, proba_mal, y_true_raw, extras):
    n = int(len(labels))
    n_mal = int((labels == MALICIOUS).sum())
    result_metrics = None
    per_attack = None
    if y_true_raw is not None:
        result_metrics = metrics.compute(metrics.normalize_truth(y_true_raw), labels, proba_mal)
        per_attack = metrics.per_attack_recall(y_true_raw, labels)
        if per_attack:
            result_metrics["per_attack"] = per_attack

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
        "predictor": {
            "name": predictor.name,
            "version": getattr(predictor, "version", "unknown"),
            "model_id": model_id,
        },
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
async def predict(file: UploadFile = File(...), model_id: str | None = Form(None)):
    pred = _resolve(model_id)
    df = _parse_csv(await file.read())
    labels, proba_mal, y_true_raw, extras = _score(pred, df)
    return JSONResponse(_build_payload(pred, model_id or registry.default_id(),
                                       labels, proba_mal, y_true_raw, extras))


@app.post("/api/predict.csv")
async def predict_csv(file: UploadFile = File(...), model_id: str | None = Form(None)):
    pred = _resolve(model_id)
    df = _parse_csv(await file.read())
    labels, proba_mal, y_true_raw, _ = _score(pred, df)

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


@app.post("/api/predict_compare")
async def predict_compare(
    file: UploadFile | None = File(None),
    model_ids: str = Form(...),
    sample_id: str | None = Form(None),
):
    ids = [s.strip() for s in model_ids.split(",") if s.strip()]
    if not ids:
        raise HTTPException(400, "model_ids is empty")

    if file is not None and file.filename:
        df = _parse_csv(await file.read())
        source = file.filename
    elif sample_id:
        entry, path = _sample_path(sample_id)
        if path is None:
            raise HTTPException(404, f"unknown or missing sample_id: {sample_id!r}")
        df = _parse_csv(path.read_bytes())
        source = entry["filename"]
    else:
        raise HTTPException(400, "supply either an upload or sample_id")

    X, y_true_raw, _ = _split_frame(df)
    truth_bin = metrics.normalize_truth(y_true_raw) if y_true_raw is not None else None

    rows = []
    for mid in ids:
        row = {"model_id": mid}
        try:
            pred = registry.get(mid)
        except (FileNotFoundError, ValueError) as e:
            row["error"] = str(e)
            rows.append(row)
            continue

        try:
            labels, proba_mal = pred.predict(X)
        except Exception as e:
            row["error"] = f"predict failed: {type(e).__name__}: {e}"
            rows.append(row)
            continue

        n_mal = int((labels == MALICIOUS).sum())
        row.update({
            "predictor": {"name": pred.name, "version": getattr(pred, "version", "unknown")},
            "n_rows": int(len(labels)),
            "n_malicious": n_mal,
            "malicious_pct": round(100.0 * n_mal / len(labels), 2) if len(labels) else 0.0,
        })
        if truth_bin is not None:
            row["metrics"] = metrics.compute(truth_bin, labels, proba_mal)
        rows.append(row)

    return {"source": source, "results": rows}


@app.get("/api/samples")
def list_samples():
    out = []
    for s in BUNDLED_SAMPLES:
        p = BUNDLED_DIR / s["filename"]
        if p.exists():
            out.append({"id": s["id"], "label": s["label"], "filename": s["filename"],
                        "size_bytes": p.stat().st_size})
    return {"samples": out, "default_id": out[0]["id"] if out else None}


@app.get("/api/sample")
def sample(model_id: str | None = None, sample_id: str | None = None):
    pred = _resolve(model_id)
    if sample_id is None:
        sample_id = next((s["id"] for s in BUNDLED_SAMPLES if (BUNDLED_DIR / s["filename"]).exists()), None)
        if sample_id is None:
            raise HTTPException(503, "No bundled samples available in this image.")
    entry, path = _sample_path(sample_id)
    if entry is None:
        raise HTTPException(404, f"unknown sample_id: {sample_id!r}")
    if path is None:
        raise HTTPException(503, f"sample {entry['filename']} not present in image")
    df = _parse_csv(path.read_bytes())
    labels, proba_mal, y_true_raw, extras = _score(pred, df)
    payload = _build_payload(pred, model_id or registry.default_id(),
                             labels, proba_mal, y_true_raw, extras)
    payload["source"] = entry["filename"]
    payload["sample_id"] = entry["id"]
    return JSONResponse(payload)
