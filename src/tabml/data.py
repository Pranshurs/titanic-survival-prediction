"""Loading, schema checks and the one train/test split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .features import REGISTRY as FEATURES


class DataError(ValueError):
    pass


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series            # 1 = positive class
    ids: pd.Series | None
    source: str
    sha256: str | None


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_raw_columns(cfg: Config) -> list[str]:
    """Raw input columns a prediction request must provide."""
    produced = {name for name in cfg.features.engineered}
    needed = {c for name in cfg.features.engineered for c in FEATURES[name][1]}
    needed |= {c for c in cfg.features.numeric + cfg.features.categorical if c not in produced}
    return sorted(needed)


def load(cfg: Config, root: Path) -> Dataset:
    d = cfg.data
    if d.source == "csv":
        path = (root / d.path).resolve()
        if not path.exists():
            raise DataError(f"{path} not found (see the README for where to get the data)")
        digest = file_sha256(path)
        if d.sha256 and digest != d.sha256:
            raise DataError(f"{path} has sha256 {digest}, config expects {d.sha256}")
        df, source = pd.read_csv(path), str(d.path)
    else:
        import sklearn.datasets as skd

        loader = getattr(skd, f"load_{d.dataset}", None)
        if loader is None:
            raise DataError(f"sklearn has no load_{d.dataset}")
        df, source, digest = loader(as_frame=True).frame, f"sklearn:{d.dataset}", None

    if d.target not in df.columns:
        raise DataError(f"target column {d.target!r} not in data")
    labels = set(df[d.target].dropna().unique())
    if len(labels) != 2 or d.positive_label not in labels:
        raise DataError(f"target must be binary and contain positive_label {d.positive_label!r}; found {sorted(labels)}")
    if df[d.target].isna().any():
        raise DataError("target has missing values")
    missing = [c for c in required_raw_columns(cfg) if c not in df.columns]
    if missing:
        raise DataError(f"columns required by the config are missing: {missing}")

    y = (df[d.target] == d.positive_label).astype(int)
    ids = df[d.id_column] if d.id_column else None
    X = df.drop(columns=[c for c in (d.target, d.id_column) if c])
    return Dataset(X=X, y=y, ids=ids, source=source, sha256=digest)


def split(cfg: Config, ds: Dataset):
    """Stratified hold-out split. The test part is touched exactly once, after model selection."""
    idx = np.arange(len(ds.y))
    train_idx, test_idx = train_test_split(idx, test_size=cfg.split.test_size, random_state=cfg.split.seed,
                                           stratify=ds.y)
    return np.sort(train_idx), np.sort(test_idx)


def indices_digest(idx: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()
