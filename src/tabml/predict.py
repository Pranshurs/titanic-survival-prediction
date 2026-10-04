"""Load a trained run and score new rows with exactly the pipeline that was evaluated."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class InputError(ValueError):
    pass


@dataclass
class TrainedModel:
    pipeline: Any
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        return float(self.metadata["test"]["threshold"])

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.metadata["input_columns"] if c not in rows.columns]
        if missing:
            raise InputError(f"input is missing required columns: {missing}")
        if rows.empty:
            raise InputError("input has no rows")
        p = self.pipeline.predict_proba(rows)[:, 1]
        out = pd.DataFrame({"probability": p, "prediction": (p >= self.threshold).astype(int)},
                           index=rows.index)
        id_col = self.metadata.get("id_column")
        if id_col and id_col in rows.columns:
            out.insert(0, id_col, rows[id_col].to_numpy())
        return out


def resolve_run(path: str | Path) -> Path:
    """Accept a run directory, or a dataset directory containing a LATEST pointer."""
    p = Path(path)
    if (p / "metadata.json").exists():
        return p
    if (p / "LATEST").exists():
        return p / (p / "LATEST").read_text().strip()
    raise FileNotFoundError(f"no trained run at {p}")


def load(path: str | Path) -> TrainedModel:
    run = resolve_run(path)
    return TrainedModel(joblib.load(run / "model.joblib"),
                        json.loads((run / "metadata.json").read_text(encoding="utf-8")))
