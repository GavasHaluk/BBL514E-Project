"""Aggregate per-model metrics into a results table and copy the winner to models/model.pkl."""
import json
import sys
from datetime import date
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNS_DIR = REPO / "models" / "_runs"
OUT_TABLE = REPO / "models" / "results_table.md"
WINNER = REPO / "models" / "model.pkl"


def load_runs():
    runs = []
    for f in sorted(RUNS_DIR.glob("*_metrics.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def fmt_pct(x):
    return f"{x*100:.2f}" if x is not None else "n/a"


def fmt_num(x, digits=4):
    return f"{x:.{digits}f}" if x is not None else "n/a"


def build_table(runs):
    header = "| Model | Acc | Prec | Recall | F1 | TPR | FPR | AUC | TN | FP | FN | TP | Fit (min) |"
    sep = "|" + "|".join(["---"] * 13) + "|"
    rows = [header, sep]
    for r in runs:
        m = r["test_metrics"]
        cm = m["confusion_matrix"]
        rows.append(
            f"| {r['name']} | {fmt_pct(m['accuracy'])} | {fmt_pct(m['precision'])} | "
            f"{fmt_pct(m['recall'])} | {fmt_pct(m['f1'])} | {fmt_pct(m['tpr'])} | "
            f"{fmt_pct(m['fpr'])} | {fmt_num(m['auc'], 4)} | {cm['tn']} | {cm['fp']} | "
            f"{cm['fn']} | {cm['tp']} | {r['fit_seconds']/60:.1f} |"
        )
    return "\n".join(rows)


def proposal_check(runs):
    lines = ["## Success criteria (proposal §6C)\n"]
    for r in runs:
        m = r["test_metrics"]
        acc_ok = m["accuracy"] > 0.90
        fpr_ok = m["fpr"] < 0.05
        if acc_ok and fpr_ok:
            tag = "PASS"
        elif acc_ok or fpr_ok:
            tag = "PARTIAL"
        else:
            tag = "FAIL"
        lines.append(
            f"- [{tag}] **{r['name']}** "
            f"acc {fmt_pct(m['accuracy'])}% (need >90), "
            f"FPR {fmt_pct(m['fpr'])}% (need <5)"
        )

    singles = [r for r in runs if r["name"] != "ensemble_rf_mlp"]
    ensemble = next((r for r in runs if r["name"] == "ensemble_rf_mlp"), None)
    if ensemble and singles:
        best_single_f1 = max(r["test_metrics"]["f1"] for r in singles)
        ens_f1 = ensemble["test_metrics"]["f1"]
        verb = "beats" if ens_f1 >= best_single_f1 else "loses to"
        lines.append(
            f"\n- Ensemble vs singles: ensemble F1 = {fmt_pct(ens_f1)}%, "
            f"best single F1 = {fmt_pct(best_single_f1)}% ({verb} best single)"
        )
    return "\n".join(lines)


def main():
    runs = load_runs()
    if not runs:
        sys.exit(f"No runs found in {RUNS_DIR}. Train something first with scripts/train_full.py.")

    table = build_table(runs)
    proposal = proposal_check(runs)

    body = (
        "# Results: BBL514E intrusion detection\n\n"
        f"Generated from {len(runs)} model runs in `models/_runs/`.\n\n"
        f"## Test set metrics\n\n{table}\n\n"
        f"{proposal}\n"
    )
    OUT_TABLE.write_text(body)
    print(body)
    print(f"\nWrote {OUT_TABLE}")

    # Tiebreak F1 ties on AUC so DT/RF ties don't fall to alphabetical order.
    winner = max(runs, key=lambda r: (r["test_metrics"]["f1"], r["test_metrics"]["auc"] or 0.0))
    src = RUNS_DIR / f"{winner['name']}.pkl"
    print(f"\nWinner by test F1: {winner['name']} (F1 = {fmt_pct(winner['test_metrics']['f1'])}%)")

    if not src.exists():
        sys.exit(f"Winner pickle missing: {src}")

    pipe = joblib.load(src)
    pipe.version = f"{winner['name']}-v1"
    pipe.trained_at = winner.get("saved_at", str(date.today()))
    joblib.dump(pipe, WINNER, compress=3)
    size_mb = WINNER.stat().st_size / 1024 / 1024
    print(f"Copied {src.name} -> {WINNER} ({size_mb:.1f} MB)")
    print("Restart the container: docker compose restart")


if __name__ == "__main__":
    main()
