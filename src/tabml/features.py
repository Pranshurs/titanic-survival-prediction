"""Engineered features: stateless, row-wise transforms registered by name.

Each transform reads raw columns and adds new ones. They learn nothing from the data
(no means, no vocabularies), so they can't leak information across a train/test split.
Anything that *does* learn (imputation, scaling, encoding) lives in the sklearn
ColumnTransformer in ``tabml.pipeline``, where it is fitted on training folds only.

To support a new dataset, add a function here and list its name in the config.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

Transform = Callable[[pd.DataFrame], pd.DataFrame]
REGISTRY: dict[str, tuple[Transform, tuple[str, ...]]] = {}


def register(*requires: str) -> Callable[[Transform], Transform]:
    def wrap(fn: Transform) -> Transform:
        REGISTRY[fn.__name__] = (fn, requires)
        return fn

    return wrap


_TITLE = re.compile(r",\s*([^.]+)\.")
_TITLE_GROUPS = {
    "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master",
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
}


@register("Name")
def title(df: pd.DataFrame) -> pd.DataFrame:
    """Honorific from "Surname, Title. Given": Mr / Mrs / Miss / Master, everything else "Rare"."""
    raw = df["Name"].astype(str).str.extract(_TITLE, expand=False).str.strip()
    return df.assign(title=raw.map(_TITLE_GROUPS).fillna("Rare"))


@register("SibSp", "Parch")
def family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Passenger plus siblings/spouses plus parents/children aboard."""
    return df.assign(family_size=df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1)


@register("SibSp", "Parch")
def is_alone(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(is_alone=((df["SibSp"].fillna(0) + df["Parch"].fillna(0)) == 0).astype(int))


@register("Cabin")
def cabin_known(df: pd.DataFrame) -> pd.DataFrame:
    """Whether a cabin was recorded at all (mostly first class)."""
    return df.assign(cabin_known=df["Cabin"].notna().astype(int))


@register("Fare", "SibSp", "Parch")
def fare_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """Fare divided by family size (tickets were often shared). Missing fare stays missing."""
    size = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
    return df.assign(fare_per_person=df["Fare"] / size)


@register("Fare")
def log_fare(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(log_fare=np.log1p(df["Fare"].clip(lower=0)))


def apply(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    for name in names:
        fn, requires = REGISTRY[name]
        missing = [c for c in requires if c not in df.columns]
        if missing:
            raise KeyError(f"feature {name!r} needs columns {missing}")
        df = fn(df)
    return df
