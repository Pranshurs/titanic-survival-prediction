"""Command line: ``tabml train`` and ``tabml predict``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data, predict, train
from .config import load_config


def _train(args: argparse.Namespace) -> int:
    cfg, root = load_config(args.config)
    try:
        ds = data.load(cfg, root)
    except data.DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    meta = train.run(cfg, ds, root, Path(args.out))
    t = meta["test"]["metrics"]
    print(f"selected {meta['selected_model']} {meta['selected_params']}")
    print("test: " + ", ".join(f"{k} {v:.3f}" for k, v in t.items()))
    print(f"run written to {meta['dir']}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    model = predict.load(args.model)
    rows = pd.read_csv(args.input)
    try:
        out = model.predict(rows)
    except predict.InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out)} predictions to {args.output}")
    else:
        out.to_csv(sys.stdout, index=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tabml", description="Reproducible tabular binary classification.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="compare models with nested CV, select, evaluate once on held-out data")
    t.add_argument("-c", "--config", required=True)
    t.add_argument("--out", default="artifacts")
    t.set_defaults(fn=_train)
    p = sub.add_parser("predict", help="score a CSV with a trained run")
    p.add_argument("-m", "--model", required=True, help="run directory, or artifacts/<name> to use LATEST")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output")
    p.set_defaults(fn=_predict)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
