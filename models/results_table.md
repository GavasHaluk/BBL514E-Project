# Results: BBL514E intrusion detection

Generated from 7 model runs in `models/_runs/`.

## Test set metrics

| Model | Acc | Prec | Recall | F1 | TPR | FPR | AUC | TN | FP | FN | TP | Fit (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision_tree | 99.91 | 99.62 | 99.92 | 99.77 | 99.92 | 0.09 | 0.9994 | 340648 | 317 | 66 | 83581 | 25.0 |
| ensemble_rf_mlp | 99.81 | 99.21 | 99.85 | 99.53 | 99.85 | 0.19 | 0.9999 | 340301 | 664 | 126 | 83521 | 0.0 |
| knn | 98.98 | 95.96 | 98.99 | 97.45 | 98.99 | 1.02 | 0.9958 | 337479 | 3486 | 842 | 82805 | 1.5 |
| mlp | 99.46 | 97.67 | 99.63 | 98.64 | 99.63 | 0.58 | 0.9998 | 338977 | 1988 | 307 | 83340 | 56.6 |
| naive_bayes | 49.22 | 27.71 | 98.11 | 43.22 | 98.11 | 62.78 | 0.7358 | 126921 | 214044 | 1580 | 82067 | 7.7 |
| random_forest | 99.91 | 99.62 | 99.92 | 99.77 | 99.92 | 0.09 | 1.0000 | 340643 | 322 | 65 | 83582 | 67.0 |
| svm_rbf | 96.53 | 86.22 | 98.07 | 91.76 | 98.07 | 3.84 | 0.9958 | 327856 | 13109 | 1615 | 82032 | 104.8 |

## Success criteria (proposal §6C)

- [PASS] **decision_tree** acc 99.91% (need >90), FPR 0.09% (need <5)
- [PASS] **ensemble_rf_mlp** acc 99.81% (need >90), FPR 0.19% (need <5)
- [PASS] **knn** acc 98.98% (need >90), FPR 1.02% (need <5)
- [PASS] **mlp** acc 99.46% (need >90), FPR 0.58% (need <5)
- [FAIL] **naive_bayes** acc 49.22% (need >90), FPR 62.78% (need <5)
- [PASS] **random_forest** acc 99.91% (need >90), FPR 0.09% (need <5)
- [PASS] **svm_rbf** acc 96.53% (need >90), FPR 3.84% (need <5)

- Ensemble vs singles: ensemble F1 = 99.53%, best single F1 = 99.77% (loses to best single)
