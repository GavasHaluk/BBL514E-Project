#!/usr/bin/env bash
# Symlinks demo-grade models into models/ top level so the registry picks
# them up. Run once from the repo root: ./scripts/setup_demo_models.sh
#
# Symlinks point at siblings inside models/_runs and models/models_a, so
# they resolve correctly through the docker-compose ./models bind mount.
set -euo pipefail

cd "$(dirname "$0")/.."
cd models

# (visible_name, target_relative_to_models)
LINKS=(
  "random_forest.pkl|_runs/random_forest.pkl"
  "ensemble_rf_mlp.pkl|_runs/ensemble_rf_mlp.pkl"
  "decision_tree.pkl|_runs/decision_tree.pkl"
  "mlp.pkl|_runs/mlp.pkl"
  "knn.pkl|_runs/knn.pkl"
  "svm_rbf.pkl|_runs/svm_rbf.pkl"
  "naive_bayes.pkl|_runs/naive_bayes.pkl"
  "partner_random_forest.joblib|models_a/best_rf_pipeline_fixed.joblib"
  "partner_mlp.joblib|models_a/fast_mlp_pipeline_fixed.joblib"
  "partner_ensemble.joblib|models_a/final_ensemble_pipeline_fixed.joblib"
)

linked=0
skipped=0
for entry in "${LINKS[@]}"; do
  name="${entry%%|*}"
  target="${entry##*|}"
  if [[ ! -e "$target" ]]; then
    echo "  skip $name (target missing: $target)"
    skipped=$((skipped+1))
    continue
  fi
  ln -sfn "$target" "$name"
  echo "  link $name -> $target"
  linked=$((linked+1))
done

echo ""
echo "Linked $linked, skipped $skipped."
echo "Restart the container so the registry rescans:"
echo "  docker compose restart"
