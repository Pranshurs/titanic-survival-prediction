"""Metrics for a binary classifier's held-out predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

POINT_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss")


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
    }
    out["roc_auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return {k: float(out[k]) for k in POINT_METRICS}


def bootstrap_ci(y: np.ndarray, p: np.ndarray, threshold: float, n: int, seed: int) -> dict[str, list[float]]:
    """Percentile 95% intervals from resampling the test rows (the test set is fixed, not re-drawn)."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in POINT_METRICS}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for k, v in point_metrics(y[i], p[i], threshold).items():
            draws[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int) -> list[dict[str, float]]:
    """Reliability by quantile bin: mean predicted probability vs observed positive rate."""
    q = pd.qcut(p, q=bins, duplicates="drop")
    frame = pd.DataFrame({"y": y, "p": p, "bin": q})
    rows = []
    for interval, g in frame.groupby("bin", observed=True):
        rows.append({"bin": str(interval), "n": int(len(g)), "mean_predicted": float(g["p"].mean()),
                     "observed_rate": float(g["y"].mean())})
    return rows


def slices(X: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float,
           spec: dict[str, Any], min_n: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    pred = (p >= threshold).astype(int)
    for col, edges in spec.items():
        if col not in X.columns:
            continue
        key = pd.cut(X[col], bins=edges, right=False) if edges else X[col]
        key = key.astype("object").where(key.notna(), "missing").astype(str)
        rows = []
        for value in sorted(key.unique(), key=str):
            m = (key == value).to_numpy()
            yy, pp, dd = y[m], p[m], pred[m]
            rows.append({
                "value": value, "n": int(m.sum()), "small_sample": bool(m.sum() < min_n),
                "observed_positive_rate": float(yy.mean()), "predicted_positive_rate": float(dd.mean()),
                "mean_probability": float(pp.mean()),
                "accuracy": float((dd == yy).mean()),
                "recall": float(dd[yy == 1].mean()) if (yy == 1).any() else None,
                "precision": float(yy[dd == 1].mean()) if (dd == 1).any() else None,
            })
        out[col] = rows
    return out
