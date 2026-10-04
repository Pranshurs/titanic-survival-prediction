"""Candidate models, keyed by the names used in configs.

Every entry is built with the run's seed. A most-frequent-class baseline is included so
that every comparison shows what "no model" scores.
"""

from __future__ import annotations

from typing import Callable

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

REGISTRY: dict[str, Callable[[int], ClassifierMixin]] = {
    "baseline_majority": lambda seed: DummyClassifier(strategy="most_frequent"),
    "logistic_regression": lambda seed: LogisticRegression(max_iter=2000, random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
    "hist_gradient_boosting": lambda seed: HistGradientBoostingClassifier(random_state=seed),
    # probability=True fits Platt scaling with an internal CV, seeded for determinism.
    "svc_rbf": lambda seed: SVC(kernel="rbf", probability=True, random_state=seed),
}


def build(name: str, seed: int) -> ClassifierMixin:
    return REGISTRY[name](seed)
