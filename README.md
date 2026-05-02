# BBL514E-Project

Network intrusion detection on CICIDS2017. Binary classification (Benign vs
Malicious) served from a FastAPI backend with a vanilla HTML/JS frontend, all
in a single Docker container. Term project for BBL514E Pattern Recognition.

The container ships with six trained pipelines (Random Forest, Decision Tree,
Naive Bayes, MLP, Linear SVM, and an RF+MLP soft-voting ensemble). Pick one
from the dropdown, upload a CICIDS2017-style flow CSV, and the backend returns
predictions plus full metrics (accuracy / precision / recall / F1 / AUC,
confusion matrix, ROC curve, per-attack recall) when a `Label` column is
present.

## Run

```bash
docker compose up --build
# http://localhost:8000
```

The dropdown on the page lists every `.pkl` / `.joblib` file under `./models/`.
Drop a new artifact in and reload the page — the registry rescans on every
`/api/models` call when the directory mtime changes, no restart needed.

`SklearnPredictor` accepts any joblib-pickled estimator with `predict_proba`.
If the model declares `feature_names_in_` we subset the input frame to that.
Both `0`/`1` and `Benign`/`Malicious` class labels are accepted.

## API

| Method | Path | Notes |
|---|---|---|
| `GET`  | `/api/health` | predictor name / version / trained_at |
| `GET`  | `/api/models` | dropdown options + per-model `n_features_in_` and `classes_` |
| `GET`  | `/api/schema` | the 78 expected feature columns |
| `GET`  | `/api/samples` | bundled sample CSVs available to the dropdown |
| `GET`  | `/api/sample?sample_id=...` | scores one of the bundled samples |
| `POST` | `/api/predict` | multipart `file=<csv>`, max 25 MB, JSON |
| `POST` | `/api/predict.csv` | same input, streams a CSV of all predictions |
| `POST` | `/api/predict_compare` | scores one CSV against multiple `model_ids` |

```bash
curl -F "file=@app/bundled/realistic_sample.csv" http://localhost:8000/api/predict
curl -F "file=@app/bundled/stress_sample.csv"    http://localhost:8000/api/predict.csv -o predictions.csv
```

If the uploaded CSV contains a `Label` column, the response also includes
accuracy, precision, recall, F1, AUC, a confusion matrix and a per-attack
recall breakdown (DoS Hulk, PortScan, Heartbleed, etc.).

## Layout

```
app/
  main.py            FastAPI app + routes
  registry.py        Model discovery / lazy load
  predictor.py       SklearnPredictor wrapper
  schema.py          78 expected feature columns
  metrics.py         Confusion matrix / ROC / per-attack recall
  static/            index.html, app.js, styles.css
  bundled/           Demo + realistic + stress samples shipped in the image
models/              Trained .joblib artifacts (bind-mounted at runtime)
tests/               Smoke test that scores every model in the registry
Dockerfile
docker-compose.yml
requirements.txt
```

## Smoke test

```bash
pip install -r requirements.txt pytest httpx
pytest tests/
```

The smoke test boots the FastAPI app in-process, posts each bundled sample
against every model in the registry, and asserts AUC stays above the per-model
floor. Catches regressions when a new artifact ships with the wrong
`classes_` or feature subset.
