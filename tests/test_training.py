"""Model selection never sees the test rows; runs are reproducible; the saved model predicts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabml import data, predict, train
from tests.conftest import as_dataset, small_config, synthetic_titanic


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cfg = small_config()
    df = synthetic_titanic()
    ds = as_dataset(df, cfg)
    out = tmp_path_factory.mktemp("runs")
    first = train.run(cfg, ds, out, out)
    second = train.run(cfg, ds, out, out)

    # Corrupt every feature of every test row. (Labels stay: the split is stratified on
    # them, so changing labels would legitimately change which rows are test rows.)
    _, test_idx = data.split(cfg, ds)
    bad = df.copy()
    bad.loc[test_idx, ["Age", "Fare", "SibSp", "Parch"]] = 999.0
    bad.loc[test_idx, "Name"] = "X, Rev. Y"
    bad.loc[test_idx, "Sex"] = "unknown"
    corrupted = train.run(cfg, as_dataset(bad, cfg), out, out)
    return cfg, df, first, second, corrupted


def test_reruns_are_identical(runs):
    _, _, a, b, _ = runs
    assert a["comparison"].keys() == b["comparison"].keys()
    for name in a["comparison"]:
        assert a["comparison"][name]["folds"] == b["comparison"][name]["folds"]
    assert a["test"]["metrics"] == b["test"]["metrics"]
    assert a["split"] == b["split"]


def test_test_rows_never_influence_selection_or_tuning(runs):
    _, _, clean, _, corrupted = runs
    assert clean["split"] == corrupted["split"]
    for name in clean["comparison"]:
        assert clean["comparison"][name]["folds"] == corrupted["comparison"][name]["folds"]
    assert (clean["selected_model"], clean["selected_params"]) == (corrupted["selected_model"], corrupted["selected_params"])
    assert clean["test"]["metrics"] != corrupted["test"]["metrics"]  # the corruption was real


def test_baseline_is_compared_and_metadata_is_complete(runs):
    _, _, m, _, _ = runs
    assert m["comparison"]["baseline_majority"]["mean"]["roc_auc"] == 0.5
    for key in ("config_sha256", "code_commit", "data", "split", "environment", "input_columns", "selected_params"):
        assert key in m
    assert set(m["test"]["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"}
    lo, hi = m["test"]["ci95_bootstrap"]["accuracy"]
    assert lo <= m["test"]["metrics"]["accuracy"] <= hi
    c = m["test"]["confusion"]
    assert sum(c.values()) == m["test"]["n"]
    assert {r["value"] for r in m["test"]["slices"]["Sex"]} == {"female", "male"}


def test_saved_model_reproduces_and_validates_input(runs):
    cfg, df, m, _, _ = runs
    model = predict.load(m["dir"])
    rows = df.drop(columns=["Survived"]).head(20)
    out = model.predict(rows)
    assert list(out.columns) == ["PassengerId", "probability", "prediction"]
    assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()
    assert (out["prediction"] == (out["probability"] >= 0.5).astype(int)).all()
    with pytest.raises(predict.InputError, match="Name"):
        model.predict(rows.drop(columns=["Name"]))
    # Unseen category values are tolerated (one-hot ignores them) rather than crashing.
    model.predict(rows.assign(Embarked="Z"))
    assert np.isfinite(model.predict(pd.DataFrame(rows.iloc[[0]]))["probability"]).all()


def test_latest_pointer_resolves(runs):
    _, _, m, _, _ = runs
    from pathlib import Path

    latest = predict.resolve_run(Path(m["dir"]).parent)
    assert latest.name.startswith("20")
