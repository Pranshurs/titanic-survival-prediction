# tabml: a reproducible tabular ML pipeline (Titanic worked example)

[![tests](https://github.com/Pranshurs/titanic-survival-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranshurs/titanic-survival-prediction/actions/workflows/ci.yml)

**Origin.** This repository began in October 2025 as an early learning project: a
single-script logistic regression on Kaggle's Titanic data. In October 2026 it was rebuilt
into a reference implementation for tabular binary classification. The history is kept;
the original code is in the commits before `3acf2db`.

The aim is a pipeline that another engineer can trust and reuse:
- **Configured, not hard-coded.** One YAML file defines a dataset, its features, the
  candidate models and the evaluation.
- **Leak-free evaluation.** Models are compared fairly, and the held-out test set is
  scored exactly once.
- **Reproducible runs.** Every run records what was trained, on which data, with which
  code.

Titanic is only the worked example. `configs/breast_cancer.yaml` runs the same code on a
different dataset, and CI runs it too.

## How a run works

```
data (CSV or sklearn bundle) ── schema checks, sha256 ──► stratified train/test split (seeded)
                                                              │ test rows set aside
train rows ──► for each model: nested CV
                 inner K-fold grid search (tuning) inside each outer fold
                 outer K-fold scores (comparison)
           ──► pick best mean outer ROC AUC ──► re-tune on all train rows, refit
           ──► score the test rows once ──► model.joblib + metadata.json + report.md
```

**Pipeline.** One sklearn `Pipeline` is persisted and later used for prediction, so
serving can't drift from what was evaluated. It has three stages:
1. Engineered features: named, stateless, row-wise transforms.
2. A `ColumnTransformer`:
   - numeric columns get median imputation with missing-value indicators, then scaling;
   - categorical columns get one-hot encoding, with an explicit "missing" category and
     unknown categories ignored.
3. The model.

**Candidates.**
- majority-class baseline
- logistic regression
- RBF SVM
- random forest
- histogram gradient boosting

**Metrics.**
- **Reported for every run:** accuracy, precision, recall, F1, ROC AUC, Brier score and
  log loss, each with a 95% bootstrap CI on the test set.
- **Also produced:** the confusion matrix, a calibration table, and slices by configured
  columns.

**Recorded per run.** Each run writes a metadata file with:
- the config hash and code commit
- the data sha256
- hashes of the train and test indices
- library versions
- the hyperparameters chosen in every outer fold

The runs are deterministic. Re-running gives identical numbers (tested), and the Docker
image (Linux) reproduced the macOS run's selected model and test metrics exactly.

## Results on Titanic

These come from Kaggle `train.csv` (891 rows, sha256 `7d118fef…`). There are 712 training
rows and 179 test rows, split with seed 42. The full report is in
[`reports/titanic/report.md`](reports/titanic/report.md), with
[`metadata.json`](reports/titanic/metadata.json) alongside it.

**Model comparison:** nested CV on the 712 training rows (5 outer × 5 inner folds). Values
are mean ± standard deviation across the outer folds.

| Model | ROC AUC | Accuracy | Brier |
|---|---|---|---|
| majority baseline | 0.500 ± 0.000 | 0.617 ± 0.003 | 0.383 |
| logistic regression | 0.863 ± 0.030 | 0.824 ± 0.029 | 0.136 |
| RBF SVM | 0.851 ± 0.040 | 0.819 ± 0.027 | 0.139 |
| random forest | 0.875 ± 0.025 | 0.830 ± 0.017 | 0.123 |
| **hist gradient boosting** (selected) | **0.884 ± 0.029** | 0.827 ± 0.031 | 0.122 |

The differences between the four real models are within about one standard deviation.
On this little data, "gradient boosting won" is a weak statement.

**Held-out test set:** 179 rows, scored once after selection. CIs are 95% bootstrap.

| Accuracy | Precision | Recall | F1 | ROC AUC | Brier |
|---|---|---|---|---|---|
| 0.804 [0.737, 0.860] | 0.793 [0.678, 0.893] | 0.667 [0.547, 0.776] | 0.724 [0.626, 0.809] | 0.845 [0.771, 0.909] | 0.141 [0.109, 0.177] |

The confusion matrix is TP 46 · FP 12 · FN 23 · TN 98.

The test ROC AUC (0.845) is below the CV estimate (0.884), but within the bootstrap
interval. The earlier version of this project reported 0.799 accuracy from a single split
with no tuning. This run isn't directly comparable to that: it uses a different split and
a different procedure.

**Slices** (report only):
- **By sex:** the model predicts survival for 75% of women and 10% of men. Its recall for
  men who survived is 0.25.
- **By class:** recall for third class is 0.42.

These numbers describe a model of a 1912 evacuation ("women and children first"). They
reflect the historical outcome the data records. They aren't evidence about fairness or
about any present-day decision system, and with slices this small (several under 30 rows)
they shouldn't be over-read.

## Use it

```bash
pip install -e ".[dev]"
# Titanic: download train.csv from https://www.kaggle.com/c/titanic/data into data/titanic/
tabml train -c configs/titanic.yaml          # writes artifacts/titanic/<run_id>/ and artifacts/titanic/LATEST
tabml predict -m artifacts/titanic -i data/titanic/test.csv -o predictions.csv
tabml train -c configs/breast_cancer.yaml    # no download needed
pytest -q                                    # synthetic data, no Kaggle files needed
pytest -m titanic                            # the real Titanic run, if the CSV is present
```

To run it with Docker:

```bash
docker build -t tabml .
docker run --rm --network none -v "$PWD/data/titanic:/app/data/titanic:ro" -v "$PWD/out:/out" tabml train -c configs/titanic.yaml --out /out
```

`requirements.txt` pins the exact versions behind the committed reports. `pyproject.toml`
sets the minimum versions.

## Use it on your own table

1. Copy `configs/titanic.yaml`, then set:
   - `data`: the path, the target, `positive_label` and an optional `id_column`;
   - the `numeric` and `categorical` feature columns;
   - `models`, with their grids;
   - `evaluation.slices`.
2. If you need a derived feature, add a stateless function to `src/tabml/features.py` with
   `@register("ColumnItNeeds", ...)`, and list its name under `features.engineered`.
   The function must add a column named after itself.
3. Run `tabml train -c your.yaml`.

The tests guard what makes results trustworthy:
- **Leakage test.** It corrupts every feature of the test rows and requires model
  selection and tuning to be unchanged. Two deliberately leaky variants of the training
  code fail it.
- **Determinism.**
- **Input validation.**
- **Prediction round trip.**

## Limitations

- Binary classification only. There's no time-aware splitting, so don't use it for
  temporal data as-is.
- The 0.5 threshold is fixed in the config; there's no threshold optimisation.
- Probabilities aren't recalibrated. The calibration table shows how far off they are.
- Titanic is small: 179 test rows give wide intervals.
- The Kaggle data isn't redistributed here.
