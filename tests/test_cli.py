from __future__ import annotations

import json

import pytest

from tabml.cli import main
from tests.conftest import synthetic_titanic


@pytest.fixture()
def project(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "data").mkdir()
    synthetic_titanic().to_csv(tmp_path / "data" / "t.csv", index=False)
    synthetic_titanic(seed=9).drop(columns=["Survived"]).head(15).to_csv(tmp_path / "data" / "new.csv", index=False)
    (tmp_path / "configs" / "t.yaml").write_text(json.dumps({
        "name": "t", "data": {"path": "data/t.csv", "target": "Survived", "id_column": "PassengerId"},
        "features": {"engineered": ["title"], "numeric": ["Age", "Fare"], "categorical": ["Sex", "Pclass", "title"]},
        "cv": {"outer_folds": 2, "inner_folds": 2},
        "models": {"baseline_majority": {}, "logistic_regression": {"grid": {"C": [1.0]}}},
        "evaluation": {"bootstrap": 100},
    }))
    return tmp_path


def test_train_then_predict(project, capsys):
    assert main(["train", "-c", str(project / "configs/t.yaml"), "--out", str(project / "artifacts")]) == 0
    assert "selected" in capsys.readouterr().out
    out = project / "pred.csv"
    assert main(["predict", "-m", str(project / "artifacts/t"), "-i", str(project / "data/new.csv"), "-o", str(out)]) == 0
    assert len(out.read_text().strip().splitlines()) == 16


def test_missing_data_file_is_a_clean_error(project, capsys):
    (project / "data" / "t.csv").unlink()
    assert main(["train", "-c", str(project / "configs/t.yaml"), "--out", str(project / "a")]) == 2
    assert "not found" in capsys.readouterr().err
