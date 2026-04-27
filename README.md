# BBL514E-Project

Network intrusion detection on CICIDS2017. Binary classification (Benign vs
Malicious) served from a FastAPI backend with a vanilla HTML/JS frontend, all
in a single Docker container. Term project for BBL514E Pattern Recognition.

The shipped model is a Random Forest trained on the full ~2.83M-flow dataset
with a 70/15/15 stratified split, StandardScaler + correlation filter + SMOTE
inside an `imblearn` pipeline, and 5-fold CV grid search. See
`models/results_table.md` for the full 7-model comparison.

## Run

```bash
docker compose up --build
# http://localhost:8000
```

The container loads `models/model.pkl` at startup. If the file is missing it
falls back to a deterministic rule-based mock predictor so the UI still works.

## API

| Method | Path | Notes |
|---|---|---|
| `GET`  | `/api/health` | predictor name / version / trained_at |
| `GET`  | `/api/schema` | the 78 expected feature columns |
| `GET`  | `/api/sample` | scores the bundled demo CSV |
| `POST` | `/api/predict` | multipart `file=<csv>`, ≤25 MB, JSON |
| `POST` | `/api/predict.csv` | same input, streams a CSV of all predictions |

```bash
curl -F "file=@samples/realistic_sample.csv" http://localhost:8000/api/predict
curl -F "file=@samples/stress_sample.csv"    http://localhost:8000/api/predict.csv -o predictions.csv
```

If the uploaded CSV contains a `Label` column, the response includes
accuracy / precision / recall / F1 / AUC and a confusion matrix.

## Layout

```
app/
  main.py              FastAPI routes
  schema.py            78 expected feature columns + validators
  predictor.py         MockPredictor, SklearnPredictor, factory
  preprocessing.py     CorrelationFilter (pickled inside the pipeline)
  metrics.py           accuracy / precision / recall / F1 / ROC / AUC
  static/              index.html, app.js, styles.css
  bundled/             demo_sample.csv (ships with the image)
scripts/
  train_full.py        7 classifiers + grid search, resumable
  aggregate_runs.py    builds results_table.md, copies winner to model.pkl
  diagnostics.py       RF feature importance + learning curve plots
  build_realistic_sample.py   builds samples/{realistic,stress}_sample.csv
  train_quick.py       single-RF smoke test
  make_sample.py       small 500-row sample for quick checks
samples/               gitignored; demo inputs
models/                gitignored except results_table.md; pickles + figures
data/                  gitignored; symlink/copy of MachineLearningCVE/
HANDOFF.md             notes from the overnight training run
Dockerfile
docker-compose.yml
requirements.txt
```

## Training

The dataset is not in this repo. Drop the eight CICIDS2017
`MachineLearningCVE/*.csv` files into `data/` (or pass `--data`) and run:

```bash
python scripts/train_full.py            # ~14 h on M4, ~2 h on a 16C/32T box
python scripts/aggregate_runs.py        # writes models/results_table.md, picks winner
python scripts/diagnostics.py           # RF importance + learning curve PNGs
docker compose restart                  # backend reloads models/model.pkl
```

See `HANDOFF.md` for the latest run's results and notes.
