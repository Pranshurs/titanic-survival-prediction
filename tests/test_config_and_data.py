from __future__ import annotations

import pytest
from pydantic import ValidationError

from tabml import data
from tabml.config import load_config
from tests.conftest import small_config


def test_unknown_model_and_feature_are_rejected():
    with pytest.raises(ValidationError, match="unknown models"):
        small_config(models={"xgboost": {}})
    with pytest.raises(ValidationError, match="unknown engineered"):
        small_config(features={"engineered": ["horoscope"], "numeric": ["Age"]})


def test_a_column_cannot_be_numeric_and_categorical():
    with pytest.raises(ValidationError, match="both numeric and categorical"):
        small_config(features={"numeric": ["Age"], "categorical": ["Age"]})


def test_bundled_configs_validate():
    for name in ("titanic", "breast_cancer"):
        cfg, root = load_config(f"configs/{name}.yaml")
        assert cfg.name == name and root.name


def test_csv_checks(tmp_path, titanic_like):
    cfg = small_config(data={"source": "csv", "path": "t.csv", "target": "Survived", "id_column": "PassengerId",
                             "sha256": "0" * 64})
    titanic_like.to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(data.DataError, match="sha256"):
        data.load(cfg, tmp_path)
    cfg = small_config(data={"source": "csv", "path": "t.csv", "target": "Survived", "id_column": "PassengerId"})
    ds = data.load(cfg, tmp_path)
    assert len(ds.y) == len(titanic_like) and "Survived" not in ds.X and "PassengerId" not in ds.X
    titanic_like.drop(columns=["Name"]).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(data.DataError, match="Name"):
        data.load(cfg, tmp_path)
    titanic_like.assign(Survived=2).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(data.DataError, match="binary"):
        data.load(cfg, tmp_path)


def test_positive_label_maps_to_one():
    cfg, root = load_config("configs/breast_cancer.yaml")
    ds = data.load(cfg, root)
    assert ds.y.sum() == 212  # malignant cases in sklearn's copy of WDBC


def test_split_is_stratified_and_deterministic(titanic_like):
    from tests.conftest import as_dataset

    cfg = small_config()
    ds = as_dataset(titanic_like, cfg)
    a, b = data.split(cfg, ds), data.split(cfg, ds)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert set(a[0]).isdisjoint(a[1]) and len(a[0]) + len(a[1]) == len(ds.y)
    assert abs(ds.y.iloc[a[1]].mean() - ds.y.mean()) < 0.03
