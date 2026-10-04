"""A synthetic, Titanic-shaped table so tests never need the Kaggle data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabml.config import Config
from tabml.data import Dataset


def synthetic_titanic(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.2, 0.55])
    titles = np.where(sex == "female", rng.choice(["Mrs", "Miss", "Mlle", "Lady"], n, p=[0.5, 0.4, 0.05, 0.05]),
                      rng.choice(["Mr", "Master", "Dr", "Rev"], n, p=[0.85, 0.1, 0.03, 0.02]))
    age = rng.normal(30, 13, n).clip(1, 80)
    age[rng.random(n) < 0.2] = np.nan
    sibsp, parch = rng.poisson(0.5, n), rng.poisson(0.4, n)
    fare = rng.gamma(2, 15, n) * (4 - pclass)
    logit = 2.2 * (sex == "female") - 0.9 * (pclass - 2) - 0.02 * np.nan_to_num(age, nan=30) + 0.3
    survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "PassengerId": np.arange(1, n + 1), "Survived": survived, "Pclass": pclass,
        "Name": [f"Person{i}, {t}. Given" for i, t in enumerate(titles)], "Sex": sex, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Ticket": "X", "Fare": fare,
        "Cabin": np.where(rng.random(n) < 0.25, "C85", None),
        "Embarked": rng.choice(["S", "C", "Q", None], n, p=[0.7, 0.18, 0.1, 0.02]),
    })


def small_config(**over) -> Config:
    base = {
        "name": "synthetic",
        "data": {"source": "csv", "path": "unused.csv", "target": "Survived", "id_column": "PassengerId"},
        "features": {"engineered": ["title", "family_size", "is_alone", "cabin_known", "fare_per_person"],
                     "numeric": ["Age", "Fare", "family_size", "fare_per_person"],
                     "categorical": ["Pclass", "Sex", "Embarked", "title", "is_alone", "cabin_known"]},
        "split": {"test_size": 0.25, "seed": 3},
        "cv": {"outer_folds": 3, "inner_folds": 3, "seed": 3},
        "models": {"baseline_majority": {}, "logistic_regression": {"grid": {"C": [0.1, 1.0]}},
                   "hist_gradient_boosting": {"grid": {"max_depth": [2, 3], "max_iter": [50]}}},
        "evaluation": {"bootstrap": 200, "slices": {"Sex": None, "Age": [0, 18, 60, 120]}},
    }
    base.update(over)
    return Config.model_validate(base)


def as_dataset(df: pd.DataFrame, cfg: Config) -> Dataset:
    y = (df[cfg.data.target] == cfg.data.positive_label).astype(int)
    return Dataset(X=df.drop(columns=[cfg.data.target, cfg.data.id_column]), y=y,
                   ids=df[cfg.data.id_column], source="synthetic", sha256=None)


@pytest.fixture()
def titanic_like() -> pd.DataFrame:
    return synthetic_titanic()
