"""Export the frozen v2 models for the static web demo into web/data/models_v2.json.

Reads the trained v2 pickles (scripts/train_eval_v2.py) and re-serializes them with exact
floats; nothing is retrained or evaluated. Staging tasks: gradient boosting trees (same
encoding as web/data/models.json, via scripts/export_web.py) plus the HMM parameters used by
somnibeat/smoothing.py. Apnea: gradient boosting trees plus the logistic regression pipeline
(median imputation -> standardization -> logistic), averaged and thresholded.

Usage:
    python scripts/export_web_v2.py                                   # results/v2/models
    python scripts/export_web_v2.py --models-dir results/v2/dry_run/models   # development

Layout (the format the browser reads; web/js/model_v2.js):
{
  "schema": 2, "version": "v2",
  "source": {"models_dir": "...", "dry_run": bool, "pickles": ["<task>.pkl", ...]},
  (no pickle hashes: pickle bytes include the thread count sklearn used when fitting, so they differ
  between machines although every parameter exported here is identical)
  "config": {"rr": {...}, "features": {...}, "epoch_s": 30, "apnea_epoch_s": 60},
  "spec": {...copy of results/v2/v2_final_spec.json...},
  "tasks": {
    "<staging task>": {"labels": [...], "columns": [...], "ensemble": "mean",
                       "models": [<hgb>], "decision": {"type": "hmm_viterbi"},
                       "hmm": {"method": "hmm_viterbi", "trans": [[...]], "init": [...], "prior": [...]}},
    "apnea": {"labels": ["normal", "apnea"], "columns": [...], "ensemble": "mean",
              "models": [<hgb>, <logistic>],
              "decision": {"type": "threshold", "positive": "apnea", "threshold": 0.49}}
  }
}
<hgb>      = {"kind": "hgb", "labels", "columns", "baseline", "trees"}   (web/js/model.js format)
<logistic> = {"kind": "logistic", "columns", "impute", "mean", "scale", "coef", "intercept"}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_web import dump_compact, fmt_float, fmt_float_list, tree_json  # noqa: E402

SCHEMA = 2
STAGE_TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]
APNEA_LABELS = ["normal", "apnea"]
SMOOTHING_METHODS = ("hmm_viterbi",)


def matrix_json(a) -> str:
    a = np.asarray(a, dtype=np.float64)
    return "[" + ",".join(fmt_float_list(row) for row in a) + "]"


def hgb_json(name: str, model, cols: list[str], labels: list[str]) -> str:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not isinstance(model, HistGradientBoostingClassifier):
        raise SystemExit(f"{name}: expected HistGradientBoostingClassifier, found {type(model).__name__}.")
    if list(getattr(model, "feature_names_in_", cols)) != list(cols):
        raise SystemExit(f"{name}: column list differs from the model's feature_names_in_.")
    if list(model.classes_) != list(range(len(labels))):
        raise SystemExit(f"{name}: unexpected classes {model.classes_}.")
    k = model.n_trees_per_iteration_
    if k != (1 if len(labels) == 2 else len(labels)):
        raise SystemExit(f"{name}: unexpected {k} trees per iteration for {len(labels)} labels.")
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).ravel()
    if len(baseline) != k:
        raise SystemExit(f"{name}: baseline length {len(baseline)} != {k}.")
    head = dump_compact({"kind": "hgb", "labels": list(labels), "columns": list(cols)})[:-1]
    trees = "[" + ",".join("[" + ",".join(tree_json(p.nodes) for p in it) + "]" for it in model._predictors) + "]"
    return head + ',"baseline":' + fmt_float_list(baseline) + ',"trees":' + trees + "}"


def logistic_json(name: str, pipe, cols: list[str]) -> str:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not isinstance(pipe, Pipeline) or len(pipe.steps) != 3:
        raise SystemExit(f"{name}: expected a 3-step Pipeline, found {pipe!r}.")
    imp, sc, lr = (s for _, s in pipe.steps)
    if not (isinstance(imp, SimpleImputer) and isinstance(sc, StandardScaler) and isinstance(lr, LogisticRegression)):
        raise SystemExit(f"{name}: expected SimpleImputer -> StandardScaler -> LogisticRegression.")
    if list(getattr(pipe, "feature_names_in_", cols)) != list(cols):
        raise SystemExit(f"{name}: column list differs from the pipeline's feature_names_in_.")
    n = len(cols)
    stats = np.asarray(imp.statistics_, dtype=np.float64)
    if imp.strategy != "median" or imp.add_indicator or not (np.isnan(imp.missing_values)):
        raise SystemExit(f"{name}: unsupported imputer settings.")
    if len(stats) != n or np.isnan(stats).any():
        # sklearn drops all-missing columns (statistics_ NaN); the web format does not model that.
        raise SystemExit(f"{name}: imputer has missing statistics (an all-NaN training column).")
    if not (sc.with_mean and sc.with_std) or len(sc.mean_) != n or len(sc.scale_) != n:
        raise SystemExit(f"{name}: unsupported scaler settings.")
    coef = np.asarray(lr.coef_, dtype=np.float64)
    if list(lr.classes_) != [0, 1] or coef.shape != (1, n) or len(lr.intercept_) != 1:
        raise SystemExit(f"{name}: expected a binary logistic regression with {n} coefficients.")
    head = dump_compact({"kind": "logistic", "columns": list(cols)})[:-1]
    return (head + ',"impute":' + fmt_float_list(stats) + ',"mean":' + fmt_float_list(sc.mean_)
            + ',"scale":' + fmt_float_list(sc.scale_) + ',"coef":' + fmt_float_list(coef[0])
            + ',"intercept":' + fmt_float(lr.intercept_[0]) + "}")


def union_columns(models) -> list[str]:
    out: list[str] = []
    for _, cols in models:
        for c in cols:
            if c not in out:
                out.append(c)
    return out


def staging_json(task: str, b: dict) -> str:
    labels = list(b["labels"])
    k = len(labels)
    if b["smoothing"] not in SMOOTHING_METHODS:
        raise SystemExit(f"{task}: unsupported smoothing {b['smoothing']!r}.")
    trans, init, prior = (np.asarray(b[x], dtype=np.float64) for x in ("trans", "init", "prior"))
    if trans.shape != (k, k) or init.shape != (k,) or prior.shape != (k,):
        raise SystemExit(f"{task}: HMM parameter shapes do not match {k} labels.")
    if (trans <= 0).any() or (init <= 0).any() or (prior <= 0).any():
        raise SystemExit(f"{task}: HMM parameters must be strictly positive (log space).")
    models = []
    for i, (m, cols) in enumerate(b["models"]):
        models.append(hgb_json(f"{task}[{i}]", m, list(cols), labels))
    head = dump_compact({"labels": labels, "columns": union_columns(b["models"]), "ensemble": "mean",
                         "decision": {"type": b["smoothing"]}})[:-1]
    hmm = ('{"method":' + json.dumps(b["smoothing"]) + ',"trans":' + matrix_json(trans)
           + ',"init":' + fmt_float_list(init) + ',"prior":' + fmt_float_list(prior) + "}")
    return head + ',"models":[' + ",".join(models) + '],"hmm":' + hmm + "}"


def apnea_json(b: dict) -> str:
    from sklearn.pipeline import Pipeline

    thr = float(b["threshold"])
    models = []
    for i, (m, cols) in enumerate(b["models"]):
        if isinstance(m, Pipeline):
            models.append(logistic_json(f"apnea[{i}]", m, list(cols)))
        else:
            models.append(hgb_json(f"apnea[{i}]", m, list(cols), APNEA_LABELS))
    head = dump_compact({"labels": APNEA_LABELS, "columns": union_columns(b["models"]), "ensemble": "mean",
                         "decision": {"type": "threshold", "positive": "apnea", "threshold": thr}})[:-1]
    return head + ',"models":[' + ",".join(models) + "]}"


def export(models_dir: Path, config: Path, spec_path: Path, out: Path) -> dict:
    cfg = tomllib.loads(config.read_text())
    spec = json.loads(spec_path.read_text())
    parts, names = [], []
    for task in STAGE_TASKS + ["apnea"]:
        p = models_dir / f"{task}.pkl"
        with open(p, "rb") as f:
            b = pickle.load(f)
        names.append(p.name)
        parts.append(json.dumps(task) + ":" + (apnea_json(b) if task == "apnea" else staging_json(task, b)))
    try:
        rel = models_dir.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel = models_dir.name
    source = {"models_dir": rel, "dry_run": "dry_run" in models_dir.resolve().parts, "pickles": names}
    config_block = {"rr": cfg["rr"], "features": cfg["features"], "epoch_s": int(cfg["epochs"]["length_s"]),
                    "apnea_epoch_s": 60}
    text = ('{"schema":' + str(SCHEMA) + ',"version":"v2","source":' + dump_compact(source)
            + ',"config":' + dump_compact(config_block) + ',"spec":' + dump_compact(spec)
            + ',"tasks":{' + ",".join(parts) + "}}")
    parsed = json.loads(text)  # must be valid JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return parsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", default=str(ROOT / "results" / "v2" / "models"),
                    help="directory with {sleep_wake,wake_nrem_rem,four_class,apnea}.pkl "
                         "(results/v2/dry_run/models for development)")
    ap.add_argument("--config", default=str(ROOT / "config" / "v2.toml"))
    ap.add_argument("--spec", default=str(ROOT / "results" / "v2" / "v2_final_spec.json"))
    ap.add_argument("--out", default=str(ROOT / "web" / "data" / "models_v2.json"))
    a = ap.parse_args(argv)
    models_dir = Path(a.models_dir)
    if not models_dir.is_absolute():
        models_dir = (Path.cwd() / models_dir)
    if not models_dir.is_dir():
        raise SystemExit(f"{models_dir} does not exist (the final v2 run may not have finished; "
                         "use --models-dir results/v2/dry_run/models for development).")
    out = Path(a.out)
    m = export(models_dir, Path(a.config), Path(a.spec), out)
    print(f"wrote {out} ({out.stat().st_size} bytes) from {m['source']['models_dir']}"
          + (" [DRY RUN models]" if m["source"]["dry_run"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
