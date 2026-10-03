"""Experiment configuration: everything that defines a run, in one validated YAML file.

Swapping Titanic for another binary-classification table means writing a new config
(columns, target, which engineered features to apply, model grids). The code doesn't
change unless the new dataset needs a feature transform that isn't registered yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    source: Literal["csv", "sklearn"] = "csv"
    path: Optional[str] = None          # csv: file path, relative to the config's directory's parent
    dataset: Optional[str] = None       # sklearn: a bundled loader name, e.g. "breast_cancer"
    sha256: Optional[str] = None        # if set, the file must match or the run stops
    target: str
    positive_label: Any = 1             # value of `target` that counts as the positive class
    id_column: Optional[str] = None     # carried through to predictions, never a feature

    @model_validator(mode="after")
    def _source_fields(self) -> "DataConfig":
        if self.source == "csv" and not self.path:
            raise ValueError("data.path is required when source is csv")
        if self.source == "sklearn" and not self.dataset:
            raise ValueError("data.dataset is required when source is sklearn")
        return self


class FeatureConfig(_Strict):
    engineered: list[str] = Field(default_factory=list)  # names in tabml.features.REGISTRY, applied in order
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _disjoint(self) -> "FeatureConfig":
        both = set(self.numeric) & set(self.categorical)
        if both:
            raise ValueError(f"columns listed as both numeric and categorical: {sorted(both)}")
        if not self.numeric and not self.categorical:
            raise ValueError("at least one numeric or categorical feature is required")
        return self


class SplitConfig(_Strict):
    test_size: float = Field(0.2, gt=0, lt=1)
    seed: int = 42


class CVConfig(_Strict):
    outer_folds: int = Field(5, ge=2)   # model comparison (nested CV)
    inner_folds: int = Field(5, ge=2)   # hyperparameter search inside each outer fold
    seed: int = 42


class ModelConfig(_Strict):
    grid: dict[str, list[Any]] = Field(default_factory=dict)  # estimator params, without the "model__" prefix


class EvalConfig(_Strict):
    threshold: float = Field(0.5, gt=0, lt=1)
    bootstrap: int = Field(2000, ge=100)
    calibration_bins: int = Field(10, ge=2)
    # column -> null for categorical slices, or a list of bin edges for numeric ones
    slices: dict[str, Optional[list[float]]] = Field(default_factory=dict)
    min_slice_n: int = 30               # smaller slices are reported but flagged


class Config(_Strict):
    name: str
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig = SplitConfig()
    cv: CVConfig = CVConfig()
    selection_metric: Literal["roc_auc", "accuracy", "f1", "neg_brier_score", "neg_log_loss"] = "roc_auc"
    models: dict[str, ModelConfig]
    evaluation: EvalConfig = EvalConfig()
    n_jobs: int = 1                     # >1 is faster but floating-point results may differ slightly

    @field_validator("models")
    @classmethod
    def _known_models(cls, v: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        from .models import REGISTRY

        unknown = sorted(set(v) - set(REGISTRY))
        if unknown:
            raise ValueError(f"unknown models {unknown}; available: {sorted(REGISTRY)}")
        if not v:
            raise ValueError("at least one model is required")
        return v

    @field_validator("features")
    @classmethod
    def _known_features(cls, v: FeatureConfig) -> FeatureConfig:
        from .features import REGISTRY

        unknown = [f for f in v.engineered if f not in REGISTRY]
        if unknown:
            raise ValueError(f"unknown engineered features {unknown}; available: {sorted(REGISTRY)}")
        return v

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path) -> tuple[Config, Path]:
    """Return the config and the directory relative data paths resolve against (the repo root)."""
    path = Path(path).resolve()
    cfg = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return cfg, path.parent.parent
