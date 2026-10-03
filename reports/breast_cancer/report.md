# breast_cancer: run 20261003T195048Z-352b7995

- **Data:** `sklearn:breast_cancer` (569 rows, positive rate 0.373), sha256 `n/a (bundled dataset)`
- **Split:** 455 train / 114 test, stratified, seed 7
- **Code:** `dc9ca8aa4b5cc0aa732921446b76e3f6a616e094`; config sha256 `352b79955cb0`; sklearn 1.9.1, Python 3.12.13

## Model comparison: nested CV on the training rows (5 outer × 3 inner folds)

Mean ± standard deviation across outer folds. The folds share training data, so the spread is a rough guide, not a confidence interval.

| Model | ROC AUC | Accuracy | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| baseline_majority | 0.500 ± 0.000 | 0.626 ± 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.374 |
| logistic_regression **(selected)** | 0.994 ± 0.012 | 0.976 ± 0.031 | 0.968 ± 0.041 | 0.971 | 0.965 | 0.022 |
| random_forest | 0.992 ± 0.013 | 0.969 ± 0.024 | 0.959 ± 0.031 | 0.960 | 0.959 | 0.026 |
| hist_gradient_boosting | 0.992 ± 0.015 | 0.965 ± 0.028 | 0.953 ± 0.037 | 0.955 | 0.953 | 0.026 |

The model with the highest mean `roc_auc` was selected: **logistic_regression** with `{'C': 10.0}`.

## Held-out test set (n=114, scored once, after selection)

Threshold 0.5. The intervals are 95% percentile bootstrap over the test rows.

| Metric | Value | 95% CI |
|---|---|---|
| accuracy | 0.974 | [0.939, 1.000] |
| precision | 1.000 | [1.000, 1.000] |
| recall | 0.929 | [0.841, 1.000] |
| f1 | 0.963 | [0.914, 1.000] |
| roc_auc | 0.982 | [0.952, 1.000] |
| brier | 0.029 | [0.008, 0.060] |
| log_loss | 0.142 | [0.035, 0.301] |

Confusion matrix: TP 39 · FP 0 · FN 3 · TN 72

### Calibration (quantile bins)

| Bin | n | Mean predicted | Observed rate |
|---|---|---|---|
| (0.000, 0.000] | 12 | 0.000 | 0.000 |
| (0.000, 0.000] | 11 | 0.000 | 0.000 |
| (0.000, 0.000] | 11 | 0.000 | 0.091 |
| (0.000, 0.002] | 12 | 0.001 | 0.000 |
| (0.002, 0.008] | 11 | 0.004 | 0.000 |
| (0.008, 0.113] | 11 | 0.036 | 0.091 |
| (0.113, 0.914] | 12 | 0.434 | 0.500 |
| (0.914, 1.000] | 11 | 0.991 | 1.000 |
| (1.000, 1.000] | 11 | 1.000 | 1.000 |
| (1.000, 1.000] | 12 | 1.000 | 1.000 |
