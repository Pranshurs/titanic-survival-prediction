from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabml import features


def test_title_groups_rare_and_french_titles():
    df = pd.DataFrame({"Name": ["A, Mr. B", "A, Mlle. B", "A, Mme. B", "A, Dr. B", "A, the Countess. B", "no title"]})
    assert features.title(df)["title"].tolist() == ["Mr", "Miss", "Mrs", "Rare", "Rare", "Rare"]


def test_family_features_and_missing_values():
    df = pd.DataFrame({"SibSp": [0, 1, np.nan], "Parch": [0, 2, 1], "Fare": [10.0, 40.0, np.nan]})
    out = features.apply(df, ["family_size", "is_alone", "fare_per_person"])
    assert out["family_size"].tolist() == [1, 4, 2]
    assert out["is_alone"].tolist() == [1, 0, 0]
    assert out["fare_per_person"].iloc[1] == 10.0 and np.isnan(out["fare_per_person"].iloc[2])


def test_transforms_are_row_wise_so_cannot_leak():
    df = pd.DataFrame({"Name": ["A, Mr. B", "C, Mrs. D"], "SibSp": [0, 1], "Parch": [0, 0],
                       "Fare": [5.0, 7.0], "Cabin": [None, "B2"]})
    names = list(features.REGISTRY)
    whole = features.apply(df, names)
    parts = pd.concat([features.apply(df.iloc[[0]], names), features.apply(df.iloc[[1]], names)])
    pd.testing.assert_frame_equal(whole, parts)


def test_missing_input_column_is_named():
    with pytest.raises(KeyError, match="Cabin"):
        features.apply(pd.DataFrame({"x": [1]}), ["cabin_known"])
