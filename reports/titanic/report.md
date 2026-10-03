# titanic: run 20261003T194536Z-07809b04

- **Data:** `data/titanic/train.csv` (891 rows, positive rate 0.384), sha256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`
- **Split:** 712 train / 179 test, stratified, seed 42
- **Code:** `dc9ca8aa4b5cc0aa732921446b76e3f6a616e094`; config sha256 `07809b04b2a4`; sklearn 1.9.1, Python 3.12.13

## Model comparison: nested CV on the training rows (5 outer × 5 inner folds)

Mean ± standard deviation across outer folds. The folds share training data, so the spread is a rough guide, not a confidence interval.

| Model | ROC AUC | Accuracy | F1 | Precision | Recall | Brier |
|---|---|---|---|---|---|---|
| baseline_majority | 0.500 ± 0.000 | 0.617 ± 0.003 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.383 |
| logistic_regression | 0.863 ± 0.030 | 0.824 ± 0.029 | 0.764 ± 0.042 | 0.786 | 0.744 | 0.136 |
| svc_rbf | 0.851 ± 0.040 | 0.819 ± 0.027 | 0.749 ± 0.040 | 0.797 | 0.707 | 0.139 |
| random_forest | 0.875 ± 0.025 | 0.830 ± 0.017 | 0.766 ± 0.031 | 0.810 | 0.729 | 0.123 |
| hist_gradient_boosting **(selected)** | 0.884 ± 0.029 | 0.827 ± 0.031 | 0.762 ± 0.043 | 0.808 | 0.722 | 0.122 |

The model with the highest mean `roc_auc` was selected: **hist_gradient_boosting** with `{'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_depth': 3, 'max_iter': 200}`.

## Held-out test set (n=179, scored once, after selection)

Threshold 0.5. The intervals are 95% percentile bootstrap over the test rows.

| Metric | Value | 95% CI |
|---|---|---|
| accuracy | 0.804 | [0.737, 0.860] |
| precision | 0.793 | [0.678, 0.893] |
| recall | 0.667 | [0.547, 0.776] |
| f1 | 0.724 | [0.626, 0.809] |
| roc_auc | 0.845 | [0.771, 0.909] |
| brier | 0.141 | [0.109, 0.177] |
| log_loss | 0.471 | [0.369, 0.589] |

Confusion matrix: TP 46 · FP 12 · FN 23 · TN 98

### Calibration (quantile bins)

| Bin | n | Mean predicted | Observed rate |
|---|---|---|---|
| (0.008, 0.102] | 21 | 0.072 | 0.238 |
| (0.102, 0.104] | 17 | 0.104 | 0.059 |
| (0.104, 0.127] | 19 | 0.117 | 0.000 |
| (0.127, 0.152] | 15 | 0.141 | 0.133 |
| (0.152, 0.271] | 18 | 0.187 | 0.056 |
| (0.271, 0.404] | 17 | 0.355 | 0.412 |
| (0.404, 0.555] | 18 | 0.456 | 0.556 |
| (0.555, 0.775] | 18 | 0.683 | 0.556 |
| (0.775, 0.875] | 18 | 0.839 | 0.889 |
| (0.875, 0.972] | 18 | 0.938 | 0.944 |

### Slices

These are descriptive only. Slices with fewer than 30 rows are marked *small*.

**Sex**

| Value | n | Observed + rate | Predicted + rate | Accuracy | Recall | Precision |
|---|---|---|---|---|---|---|
| female | 61 | 0.738 | 0.754 | 0.820 | 0.889 | 0.870 |
| male | 118 | 0.203 | 0.102 | 0.797 | 0.250 | 0.500 |

**Pclass**

| Value | n | Observed + rate | Predicted + rate | Accuracy | Recall | Precision |
|---|---|---|---|---|---|---|
| 1 | 45 | 0.556 | 0.556 | 0.689 | 0.720 | 0.720 |
| 2 | 34 | 0.588 | 0.559 | 0.912 | 0.900 | 0.947 |
| 3 | 100 | 0.240 | 0.140 | 0.820 | 0.417 | 0.714 |

**Age**

| Value | n | Observed + rate | Predicted + rate | Accuracy | Recall | Precision |
|---|---|---|---|---|---|---|
| [0.0, 13.0) *small* | 14 | 0.500 | 0.429 | 0.786 | 0.714 | 0.833 |
| [13.0, 18.0) *small* | 14 | 0.500 | 0.500 | 1.000 | 1.000 | 1.000 |
| [18.0, 40.0) | 81 | 0.420 | 0.309 | 0.790 | 0.618 | 0.840 |
| [40.0, 60.0) *small* | 22 | 0.227 | 0.364 | 0.773 | 0.800 | 0.500 |
| [60.0, 120.0) *small* | 8 | 0.500 | 0.250 | 0.750 | 0.500 | 1.000 |
| missing | 40 | 0.300 | 0.250 | 0.800 | 0.583 | 0.700 |

