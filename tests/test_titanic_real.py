"""Runs only with the Kaggle CSV present: pytest -m titanic."""

from __future__ import annotations

from pathlib import Path

import pytest

from tabml import data, train
from tabml.config import load_config

pytestmark = pytest.mark.titanic


def test_full_titanic_run(tmp_path):
    cfg, root = load_config("configs/titanic.yaml")
    if not (root / cfg.data.path).exists():
        pytest.skip("data/titanic/train.csv not present")
    m = train.run(cfg, data.load(cfg, root), root, Path(tmp_path))
    assert m["data"]["sha256"] == cfg.data.sha256
    assert m["test"]["metrics"]["roc_auc"] > m["comparison"]["baseline_majority"]["mean"]["roc_auc"]
