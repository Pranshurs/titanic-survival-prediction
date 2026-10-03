"""Training: nested-CV model comparison, selection, one final held-out evaluation.

1. Split once into train and test (stratified, seeded). The test rows are put aside.
2. For each candidate model, run nested cross-validation on the *training rows only*:
   an inner stratified K-fold grid search tunes hyperparameters, and the outer K-fold
   scores the tuned model on folds the search never saw. This estimates how each model
   *with its tuning procedure* generalises.
3. Pick the model with the best mean outer-fold ``selection_metric``.
4. Re-run its grid search on all training rows, refit, and score the test rows once.

Only the selected model is scored on the test set. Scoring every candidate there and then
choosing would turn the test set into another validation set.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from . import __version__, evaluate
from .config import Config
from .data import Dataset, indices_digest, required_raw_columns, split
from .pipeline import make_pipeline, param_grid

CV_SCORING = {
    # zero_division=0: the majority-class baseline never predicts positive, so its precision is 0, not undefined.
    "accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0), "recall": "recall", "f1": "f1",
    "roc_auc": "roc_auc", "neg_brier_score": "neg_brier_score", "neg_log_loss": "neg_log_loss",
}


def _search(cfg: Config, name: str) -> GridSearchCV:
    inner = StratifiedKFold(cfg.cv.inner_folds, shuffle=True, random_state=cfg.cv.seed)
    return GridSearchCV(make_pipeline(cfg, name), param_grid(cfg, name) or [{}],
                        scoring=cfg.selection_metric, cv=inner, n_jobs=cfg.n_jobs, refit=True,
                        error_score="raise")


def compare_models(cfg: Config, X, y) -> dict[str, Any]:
    outer = StratifiedKFold(cfg.cv.outer_folds, shuffle=True, random_state=cfg.cv.seed + 1)
    results = {}
    for name in cfg.models:
        t0 = time.perf_counter()
        cv = cross_validate(_search(cfg, name), X, y, cv=outer, scoring=CV_SCORING,
                            return_estimator=True, n_jobs=1, error_score="raise")
        folds = {k.removeprefix("test_"): [float(v) for v in cv[k]] for k in cv if k.startswith("test_")}
        results[name] = {
            "folds": folds,
            "mean": {k: float(np.mean(v)) for k, v in folds.items()},
            "std": {k: float(np.std(v, ddof=1)) for k, v in folds.items()},
            "chosen_params_per_fold": [_jsonable(e.best_params_) for e in cv["estimator"]],
            "seconds": round(time.perf_counter() - t0, 2),
        }
    return results


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("model__"): (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in params.items()}


def select(cfg: Config, comparison: dict[str, Any]) -> str:
    # Ties go to the earlier entry in the config, so list simpler models first.
    return max(cfg.models, key=lambda n: comparison[n]["mean"][cfg.selection_metric])


def _git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def run(cfg: Config, ds: Dataset, root: Path, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    train_idx, test_idx = split(cfg, ds)
    X_train, y_train = ds.X.iloc[train_idx], ds.y.iloc[train_idx].to_numpy()
    X_test, y_test = ds.X.iloc[test_idx], ds.y.iloc[test_idx].to_numpy()

    comparison = compare_models(cfg, X_train, y_train)
    best = select(cfg, comparison)

    search = _search(cfg, best).fit(X_train, y_train)
    final = search.best_estimator_
    p = final.predict_proba(X_test)[:, 1]
    ev = cfg.evaluation
    test = {
        "n": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "threshold": ev.threshold,
        "metrics": evaluate.point_metrics(y_test, p, ev.threshold),
        "ci95_bootstrap": evaluate.bootstrap_ci(y_test, p, ev.threshold, ev.bootstrap, cfg.split.seed),
        "confusion": evaluate.confusion(y_test, p, ev.threshold),
        "calibration": evaluate.calibration_table(y_test, p, ev.calibration_bins),
        "slices": evaluate.slices(X_test, y_test, p, ev.threshold, ev.slices, ev.min_slice_n),
    }

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started)) + "-" + cfg.digest()[:8]
    dest = out_dir / cfg.name / run_id
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, dest / "model.joblib")
    meta = {
        "run_id": run_id,
        "tabml_version": __version__,
        "config": cfg.model_dump(mode="json"),
        "config_sha256": cfg.digest(),
        "code_commit": _git_commit(root),
        "data": {"source": ds.source, "sha256": ds.sha256, "rows": int(len(ds.y)),
                 "positive_rate": float(ds.y.mean())},
        "split": {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                  "train_indices_sha256": indices_digest(train_idx),
                  "test_indices_sha256": indices_digest(test_idx)},
        "input_columns": required_raw_columns(cfg),
        "id_column": cfg.data.id_column,
        "comparison": comparison,
        "selected_model": best,
        "selected_params": _jsonable(search.best_params_),
        "test": test,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": __import__("pandas").__version__,
                        "platform": platform.platform()},
        "seconds": round(time.time() - started, 1),
    }
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    from .report import render

    (dest / "report.md").write_text(render(meta), encoding="utf-8")
    (out_dir / cfg.name / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    meta["dir"] = str(dest)
    return meta
