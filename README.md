# Titanic Survival Prediction

An early learning project (October 2025): a small scikit-learn pipeline for the Kaggle
Titanic competition. It is kept as a record of early ML work, not as a current example.

## What it does

- `titanic_survival/src/data/load_data.py` loads `train.csv` and makes an 80/20 split
  (`random_state=42`).
- `titanic_survival/src/features/preprocess.py` drops `Name`, `Ticket` and `Cabin`, encodes
  `Sex` and `Embarked`, and mean-fills missing numbers.
- `titanic_survival/src/models/train.py` trains logistic regression (or a random forest via
  `config.yaml`), reports holdout accuracy, and saves `models/model.pkl`.
- `titanic_survival/src/models/predict.py` writes `models/submission.csv` for Kaggle.

## Run it

1. Download `train.csv`, `test.csv` and `gender_submission.csv` from the
   [Kaggle Titanic competition](https://www.kaggle.com/c/titanic/data) into
   `titanic_survival/data/`. The data isn't redistributed here.
2. Install and run:

   ```bash
   pip install -r requirements.txt
   python -m titanic_survival.src.models.train     # prints holdout accuracy, saves the model
   python -m titanic_survival.src.models.predict   # writes the submission CSV
   ```

Docker runs the same training step: `docker build -t titanic . && docker run --rm -v "$(pwd)":/app titanic`.

## Result

Logistic regression reaches a holdout accuracy of **0.7989** on the 20% split (179
passengers). That's a single split with no cross-validation, measured on 2026-10-04.
No Kaggle leaderboard score is claimed.

## Known limitations

- Mean imputation is fitted separately on each frame, including the test set, rather than
  on the training data alone.
- There are no tests, cross-validation or feature engineering beyond the basics.
- Features are unscaled, so logistic regression stops at `max_iter=500` with a convergence warning.
