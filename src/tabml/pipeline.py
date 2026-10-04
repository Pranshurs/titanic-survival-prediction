"""The sklearn Pipeline: engineered features -> column-wise preprocessing -> model.

Everything a prediction needs is inside the one fitted Pipeline that gets persisted, so
training and serving can't drift apart.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import features as feats
from .config import Config
from .models import build


def _to_str(x):
    # Categorical columns may be numeric codes (Pclass) or strings; encode both as strings.
    return x.astype("object").where(x.notna(), None).astype(str).replace("None", "missing")


def preprocessor(cfg: Config) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("as_str", FunctionTransformer(_to_str, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    parts = []
    if cfg.features.numeric:
        parts.append(("num", numeric, cfg.features.numeric))
    if cfg.features.categorical:
        parts.append(("cat", categorical, cfg.features.categorical))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=True)


def make_pipeline(cfg: Config, model_name: str) -> Pipeline:
    engineered = list(cfg.features.engineered)
    return Pipeline([
        ("engineer", FunctionTransformer(feats.apply, kw_args={"names": engineered}, validate=False)),
        ("prep", preprocessor(cfg)),
        ("model", build(model_name, cfg.cv.seed)),
    ])


def param_grid(cfg: Config, model_name: str) -> dict[str, list]:
    return {f"model__{k}": v for k, v in cfg.models[model_name].grid.items()}
