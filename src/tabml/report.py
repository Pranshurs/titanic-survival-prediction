"""Markdown report from a run's metadata."""

from __future__ import annotations

from typing import Any


def _f(x: Any, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def render(m: dict[str, Any]) -> str:
    cfg, t = m["config"], m["test"]
    metric = cfg["selection_metric"]
    lines = [
        f"# {cfg['name']}: run {m['run_id']}",
        "",
        f"- **Data:** `{m['data']['source']}` ({m['data']['rows']} rows, positive rate "
        f"{m['data']['positive_rate']:.3f}), sha256 `{m['data']['sha256'] or 'n/a (bundled dataset)'}`",
        f"- **Split:** {m['split']['train_rows']} train / {m['split']['test_rows']} test, stratified, "
        f"seed {cfg['split']['seed']}",
        f"- **Code:** `{m['code_commit']}`; config sha256 `{m['config_sha256'][:12]}`; "
        f"sklearn {m['environment']['sklearn']}, Python {m['environment']['python']}",
        "",
        f"## Model comparison: nested CV on the training rows "
        f"({cfg['cv']['outer_folds']} outer × {cfg['cv']['inner_folds']} inner folds)",
        "",
        "Mean ± standard deviation across outer folds. The folds share training data, so the "
        "spread is a rough guide, not a confidence interval.",
        "",
        "| Model | ROC AUC | Accuracy | F1 | Precision | Recall | Brier |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in m["comparison"].items():
        mean, sd = r["mean"], r["std"]
        mark = " **(selected)**" if name == m["selected_model"] else ""
        lines.append(
            f"| {name}{mark} | {_f(mean['roc_auc'])} ± {_f(sd['roc_auc'])} | {_f(mean['accuracy'])} ± {_f(sd['accuracy'])} "
            f"| {_f(mean['f1'])} ± {_f(sd['f1'])} | {_f(mean['precision'])} | {_f(mean['recall'])} "
            f"| {_f(-mean['neg_brier_score'])} |")
    ci = t["ci95_bootstrap"]
    lines += [
        "",
        f"The model with the highest mean `{metric}` was selected: **{m['selected_model']}** "
        f"with `{m['selected_params']}`.",
        "",
        f"## Held-out test set (n={t['n']}, scored once, after selection)",
        "",
        "Threshold " + str(t["threshold"]) + ". The intervals are 95% percentile bootstrap over the test rows.",
        "",
        "| Metric | Value | 95% CI |",
        "|---|---|---|",
    ]
    for k, v in t["metrics"].items():
        lines.append(f"| {k} | {_f(v)} | [{_f(ci[k][0])}, {_f(ci[k][1])}] |")
    c = t["confusion"]
    lines += [
        "",
        f"Confusion matrix: TP {c['tp']} · FP {c['fp']} · FN {c['fn']} · TN {c['tn']}",
        "",
        "### Calibration (quantile bins)",
        "",
        "| Bin | n | Mean predicted | Observed rate |",
        "|---|---|---|---|",
    ]
    for r in t["calibration"]:
        lines.append(f"| {r['bin']} | {r['n']} | {_f(r['mean_predicted'])} | {_f(r['observed_rate'])} |")
    if t["slices"]:
        lines += ["", "### Slices", "",
                  "These are descriptive only. Slices with fewer than "
                  f"{cfg['evaluation']['min_slice_n']} rows are marked *small*.", ""]
        for col, rows in t["slices"].items():
            lines += [f"**{col}**", "", "| Value | n | Observed + rate | Predicted + rate | Accuracy | Recall | Precision |",
                      "|---|---|---|---|---|---|---|"]
            for r in rows:
                small = " *small*" if r["small_sample"] else ""
                lines.append(f"| {r['value']}{small} | {r['n']} | {_f(r['observed_positive_rate'])} | "
                             f"{_f(r['predicted_positive_rate'])} | {_f(r['accuracy'])} | {_f(r['recall'])} | "
                             f"{_f(r['precision'])} |")
            lines.append("")
    return "\n".join(lines) + "\n"
