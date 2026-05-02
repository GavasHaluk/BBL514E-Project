# BBL514E-Project

Network intrusion detection on CICIDS2017. Binary classification (Benign vs
Malicious) served from a FastAPI backend with a vanilla HTML/JS frontend, all
in a single Docker container. Term project for BBL514E Pattern Recognition.

The shipped model is a Random Forest trained on the full ~2.83M-flow dataset
(70/15/15 stratified split, StandardScaler + correlation filter + SMOTE in an
imblearn pipeline, 5-fold CV grid search). See `models/results_table.md` for
the full 7-model comparison.

## Run

```bash
docker compose up --build
# http://localhost:8000
```

The container loads `models/model.pkl` at startup. If the file is missing it
falls back to a rule-based mock predictor so the UI still works.

The dropdown in the UI lists every `.pkl`/`.joblib` file in `./models/`, plus
the mock. Drop a file in, restart the container, refresh the page.

`SklearnPredictor` accepts any joblib-pickled estimator with `predict_proba`.
If the model declares `feature_names_in_` we subset the input frame to that.
Both `0`/`1` and `Benign`/`Malicious` class labels are accepted.

## API

| Method | Path | Notes |
|---|---|---|
| `GET`  | `/api/health` | predictor name / version / trained_at |
| `GET`  | `/api/models` | dropdown options |
| `GET`  | `/api/schema` | the 78 expected feature columns |
| `GET`  | `/api/sample` | scores the bundled demo CSV |
| `POST` | `/api/predict` | multipart `file=<csv>`, max 25 MB, JSON |
| `POST` | `/api/predict.csv` | same input, streams a CSV of all predictions |

```bash
curl -F "file=@samples/realistic_sample.csv" http://localhost:8000/api/predict
curl -F "file=@samples/stress_sample.csv"    http://localhost:8000/api/predict.csv -o predictions.csv
```

If the uploaded CSV contains a `Label` column, the response also includes
accuracy, precision, recall, F1, AUC and a confusion matrix.

## Layout

```
app/
  main.py
  registry.py
  schema.py
  predictor.py
  preprocessing.py
  metrics.py
  static/
  bundled/
scripts/
  train_full.py
  aggregate_runs.py
  diagnostics.py
  build_realistic_sample.py
  splice_partner_pipelines.py
  train_quick.py
  make_sample.py
samples/   # gitignored
models/    # gitignored except results_table.md
data/      # gitignored
Dockerfile
docker-compose.yml
requirements.txt
```

## Training

The dataset is not in this repo. Drop the eight CICIDS2017
`MachineLearningCVE/*.csv` files into `data/` (or pass `--data`) and run:

```bash
python scripts/train_full.py            # ~14h on M4, ~2h on a 16C/32T box
python scripts/aggregate_runs.py        # writes results_table.md, picks winner
python scripts/diagnostics.py           # RF importance + learning curve
docker compose restart                  # backend reloads models/model.pkl
```

See `HANDOFF.md` for the latest run's results.
