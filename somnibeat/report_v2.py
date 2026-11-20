"""v2 night report from an RR series: the reference for web/js/report_v2.js.

Same analysis grid, RR cleaning, data-quality block, night metrics, error codes and
report shape as somnibeat/report.py (docs/web_contract.md), plus "model": "v2". Only
features and predictions change (frozen v2, results/v2/v2_final_spec.json):

Features
    v1 somnibeat.features.epoch_features + somnibeat.features_v2.long_features on the
    same grid (30 s stage epochs; 60 s apnea minutes), full RR series.
Staging (sleep_wake, wake_nrem_rem, four_class)
    p = mean of the task's gradient-boosting probabilities (pure-Python traversal of the
    exported trees, somnibeat.report.predict_proba), smoothed over the whole night with
    somnibeat.smoothing.smooth_record (HMM Viterbi; trans/init/prior from models_v2.json),
    label = argmax of the one-hot path. Every grid epoch is one consecutive sequence.
Apnea (per minute)
    p(apnea) = mean of gradient boosting and logistic regression
    (NaN -> imputer median, (x - mean) / scale, dot(coef) + intercept, 1 / (1 + exp(-z))),
    apnea if p >= threshold.

models_v2_json is web/data/models_v2.json (scripts/export_web_v2.py). The arithmetic
mirrors web/js/model_v2.js term by term; tests/test_parity_report_v2.py checks it against
web/js/report_v2.js and against the trained pickles (sklearn + somnibeat.smoothing).
Low-quality epochs are predicted like every other epoch, as in evaluation.
"""

from __future__ import annotations

import math

import numpy as np

from somnibeat import evaluate as E
from somnibeat import features as F
from somnibeat import features_v2 as F2
from somnibeat import rr as R
from somnibeat import smoothing as SM
from somnibeat.report import (INVALID_WARN_FRACTION, MIN_HOURS, OUTSIDE_WARN_S, ReportError, _given,
                              _round_half_up, epoch_grid, predict_proba)

STAGE_TASKS = ("sleep_wake", "wake_nrem_rem", "four_class")


# ---------------------------------------------------------------- features

def night_features(t, rr, valid, starts, epoch_len: float, cfg: dict) -> dict[str, np.ndarray]:
    """v1 epoch features followed by the v2 long-context columns (no name may repeat)."""
    out = dict(F.epoch_features(t, rr, valid, starts, epoch_len, cfg["rr"], cfg["features"]))
    extra = F2.long_features(t, rr, valid, starts, epoch_len, cfg["rr"])
    clash = set(out) & set(extra)
    if clash:
        raise ValueError(f"v1 and v2 feature names overlap: {sorted(clash)}")
    out.update(extra)
    return out


# ---------------------------------------------------------------- model inference (mirror of model_v2.js)

def _column(columns: dict, name: str, n_rows: int) -> np.ndarray:
    if name not in columns:
        raise KeyError(f"Missing feature column: {name}")
    return np.asarray(columns[name], dtype=np.float64)[:n_rows]


def logistic_proba(model: dict, columns: dict, n_rows: int) -> np.ndarray:
    """p(positive) of the exported SimpleImputer -> StandardScaler -> LogisticRegression pipeline.

    Accumulated column by column in model order, exactly as web/js/model_v2.js logisticProba.
    """
    names = model["columns"]
    if not (len(model["impute"]) == len(model["mean"]) == len(model["scale"]) == len(model["coef"]) == len(names)):
        raise ValueError("Logistic model parameters do not match its column list")
    acc = np.zeros(n_rows, dtype=np.float64)
    for j, name in enumerate(names):
        x = _column(columns, name, n_rows).copy()
        x[np.isnan(x)] = float(model["impute"][j])
        x = (x - float(model["mean"][j])) / float(model["scale"][j])
        acc += x * float(model["coef"][j])
    z = acc + float(model["intercept"])
    return 1.0 / (1.0 + np.exp(-z))


def model_proba(model: dict, columns: dict, n_rows: int, k: int) -> np.ndarray:
    """(n_rows, k) class probabilities of one exported model entry."""
    if model["kind"] == "hgb":
        return predict_proba(model, columns, n_rows)
    if model["kind"] == "logistic":
        if k != 2:
            raise ValueError("Logistic models are binary only")
        p1 = logistic_proba(model, columns, n_rows)
        return np.stack([1.0 - p1, p1], axis=1)
    raise ValueError(f"Unknown model kind: {model['kind']}")


def ensemble_proba(task: dict, columns: dict, n_rows: int) -> np.ndarray:
    """Mean of the models' probabilities: summed in model order, then divided."""
    if task["ensemble"] != "mean":
        raise ValueError(f"Unknown ensemble rule: {task['ensemble']}")
    k = len(task["labels"])
    parts = [model_proba(m, columns, n_rows, k) for m in task["models"]]
    if len(parts) == 1:
        return parts[0]
    acc = np.zeros((n_rows, k), dtype=np.float64)
    for p in parts:
        acc += p
    return acc / len(parts)


def predict_staging(task: dict, columns: dict, n_rows: int, epoch_idx=None) -> tuple[np.ndarray, np.ndarray]:
    """(model probabilities, label index per epoch) for one night."""
    idx = np.arange(n_rows) if epoch_idx is None else np.asarray(epoch_idx, dtype=np.int64)
    h = task.get("hmm")
    if not h or task["decision"]["type"] != h["method"]:
        raise ValueError("Staging task needs matching hmm parameters")
    p = ensemble_proba(task, columns, n_rows)
    s = SM.smooth_record(p, idx, np.asarray(h["trans"], dtype=np.float64), np.asarray(h["prior"], dtype=np.float64),
                         np.asarray(h["init"], dtype=np.float64), h["method"])
    return p, np.argmax(s, axis=1).astype(np.int64)


def predict_apnea(task: dict, columns: dict, n_rows: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """(ensemble p(apnea), p(apnea) per model, label index per minute)."""
    d = task["decision"]
    labels = task["labels"]
    if d["type"] != "threshold" or len(labels) != 2 or d["positive"] not in labels:
        raise ValueError(f"Bad apnea decision for labels {labels}")
    if task["ensemble"] != "mean":
        raise ValueError(f"Unknown ensemble rule: {task['ensemble']}")
    pos = labels.index(d["positive"])
    parts = [model_proba(m, columns, n_rows, 2)[:, pos].copy() for m in task["models"]]
    p = np.zeros(n_rows, dtype=np.float64)
    for q in parts:
        p += q
    if len(parts) > 1:
        p = p / len(parts)
    lab = np.where(p >= d["threshold"], pos, 1 - pos).astype(np.int64)
    return p, parts, lab


# ---------------------------------------------------------------- report

def build_report_v2(t, rr, models_v2_json: dict, lights_off_s=None, lights_on_s=None, file_name: str = "",
                    input_warnings=(), include_proba: bool = False) -> dict:
    """Report object of docs/web_contract.md with v2 predictions (same keys as web/js/report_v2.js).

    t: closing-beat times (s), rr: intervals (ms), both after CSV parsing.
    input_warnings: warnings from parsing, placed first in quality.warnings.
    Raises somnibeat.report.ReportError for an unusable grid (codes TOO_SHORT, BAD_LIGHTS).
    """
    cfg = models_v2_json["config"]
    epoch_s = float(cfg.get("epoch_s", 30))
    minute_s = float(cfg.get("apnea_epoch_s", 60))
    t = np.asarray(t, dtype=np.float64)
    rr = np.asarray(rr, dtype=np.float64)
    n = len(rr)
    if n < 2 or len(t) != n:
        raise ReportError("TOO_SHORT", "The file has fewer than 2 usable rows.", {"rows": n, "min_hours": MIN_HOURS})

    for name, v in (("lights_off_s", lights_off_s), ("lights_on_s", lights_on_s)):
        if _given(v) and not (math.isfinite(float(v)) and float(v) >= 0):
            raise ReportError("BAD_LIGHTS", "The lights-off or lights-on time is not valid.", {name: v})
    start_s = float(lights_off_s) if _given(lights_off_s) else 0.0
    end_s = float(lights_on_s) if _given(lights_on_s) else float(t[-1])
    if _given(lights_off_s) and _given(lights_on_s) and not end_s > start_s:
        raise ReportError("BAD_LIGHTS", "Lights on must be after lights off.",
                          {"lights_off_s": start_s, "lights_on_s": end_s})

    rec_start = float(t[0] - rr[0] / 1000.0)
    rec_end = float(t[-1])
    min_s = MIN_HOURS * 3600.0
    stage_starts = epoch_grid(start_s, end_s, epoch_s)
    n_epochs = len(stage_starts)
    if n_epochs * epoch_s < min_s:
        raise ReportError("TOO_SHORT", f"The time between lights off and lights on is shorter than {MIN_HOURS} hours.",
                          {"hours": n_epochs * epoch_s / 3600.0, "min_hours": MIN_HOURS})
    overlap = min(end_s, rec_end) - max(start_s, rec_start)
    if overlap < min_s:
        raise ReportError("TOO_SHORT",
                          f"The recording has fewer than {MIN_HOURS} hours of beats between lights off and lights on.",
                          {"hours": max(overlap, 0.0) / 3600.0, "min_hours": MIN_HOURS})

    valid = R.clean_rr(t, rr, cfg["rr"])
    minute_starts = epoch_grid(start_s, end_s, minute_s)
    n_minutes = len(minute_starts)
    feats30 = night_features(t, rr, valid, stage_starts, epoch_s, cfg)
    feats60 = night_features(t, rr, valid, minute_starts, minute_s, cfg)

    tasks = models_v2_json["tasks"]
    stages, proba_out = {}, {}
    for name in STAGE_TASKS:
        task = tasks[name]
        p, lab = predict_staging(task, feats30, n_epochs)
        stages[name] = [task["labels"][i] for i in lab]
        proba_out[name] = p
    task = tasks["apnea"]
    p_apnea, _, lab = predict_apnea(task, feats60, n_minutes)
    pos = task["labels"].index(task["decision"]["positive"])
    apnea_labels = [int(i == pos) for i in lab]
    proba_out["apnea"] = p_apnea

    is_wake = np.array([s == "wake" for s in stages["sleep_wake"]], dtype=bool)
    nm = E.night_metrics(is_wake)
    metrics = {"tib_min": float(nm["tib_min"]), "tst_min": float(nm["tst_min"]), "se_pct": float(nm["se_pct"]),
               "awakenings": int(nm["awakenings"]), "apnea_min": float(sum(apnea_labels)) * (minute_s / 60.0)}

    n_valid = int(valid.sum())
    invalid = n - n_valid
    invalid_pct = 100.0 * invalid / n
    warnings = [str(w) for w in input_warnings]
    if invalid / n > INVALID_WARN_FRACTION:
        warnings.append(f"{_round_half_up(invalid_pct)}% of beats were missing or unusable.")
    if start_s < rec_start - OUTSIDE_WARN_S or end_s > rec_end + OUTSIDE_WARN_S:
        warnings.append("Part of the analysed time has no recorded beats (before the first or after the last beat); "
                        "those epochs are marked as not enough usable data.")

    report = {
        "model": "v2",
        "fileName": file_name,
        "grid": {"start_s": start_s, "end_s": end_s, "epoch_s": epoch_s, "n_epochs": n_epochs},
        "stages": stages,
        "lowQuality": [int(v) for v in feats30["low_quality"]],
        "apnea": {"minute_s": minute_s, "labels": apnea_labels},
        "metrics": metrics,
        "quality": {"valid_fraction": n_valid / n, "invalid_pct": invalid_pct, "n_beats": n,
                    "duration_h": (rec_end - rec_start) / 3600.0, "warnings": warnings},
    }
    if include_proba:
        report["proba"] = {k: v.tolist() for k, v in proba_out.items()}
    return report
