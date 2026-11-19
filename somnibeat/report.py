"""Night report from an RR series: the reference for web/js/report.js.

Follows docs/web_contract.md (analysis grid, predictions, night metrics) and
is parity-tested against the JavaScript port in tests/test_parity_report.py.

Steps:
1. Clean the RR series (somnibeat.rr.clean_rr).
2. Analysis grid: start_s = lights-off offset (else 0), end_s = lights-on
   offset (else the time of the last beat); 30 s stage epochs and 60 s apnea
   minutes, start_s + L k for k = 0 .. floor((end_s - start_s) / L) - 1.
3. Features with somnibeat.features.epoch_features on the full RR series.
4. Predictions for every task in models.json with a pure-Python traversal of
   the exported HistGradientBoosting trees (same arithmetic as web/js/model.js)
   and the task decision rule.
5. Night metrics with somnibeat.evaluate.night_metrics on the sleep/wake
   predictions, apnea minutes, and a data-quality block.

Low-quality epochs are predicted like every other epoch, as in evaluation.
"""

from __future__ import annotations

import math

import numpy as np

from somnibeat import evaluate as E
from somnibeat import features as F
from somnibeat import rr as R

MIN_HOURS = 2  # same as web/js/rr.js
INVALID_WARN_FRACTION = 0.10  # docs/web_contract.md: warn above 10 % invalid intervals
OUTSIDE_WARN_S = 60.0  # warn when the grid extends this far beyond the recorded beats


class ReportError(ValueError):
    """Input that cannot produce a report. code matches web/js/report.js."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------- model inference

def _tree_leaf_values(tree: dict, X: np.ndarray) -> np.ndarray:
    """Leaf value per row (rows of X), traversal as in docs/web_contract.md."""
    f = np.asarray(tree["f"], dtype=np.int64)
    t = np.asarray(tree["t"], dtype=np.float64)
    m = np.asarray(tree["m"], dtype=bool)
    left = np.asarray(tree["l"], dtype=np.int64)
    right = np.asarray(tree["r"], dtype=np.int64)
    v = np.asarray(tree["v"], dtype=np.float64)
    leaf = np.asarray(tree["leaf"], dtype=bool)
    n = X.shape[0]
    node = np.zeros(n, dtype=np.int64)
    rows = np.arange(n)
    active = ~leaf[node]
    while active.any():
        r = rows[active]
        nd = node[r]
        x = X[r, f[nd]]
        isnan = np.isnan(x)
        with np.errstate(invalid="ignore"):
            go_left = np.where(isnan, m[nd], x <= t[nd])
        node[r] = np.where(go_left, left[nd], right[nd])
        active = ~leaf[node]
    return v[node]


def predict_proba(task: dict, columns: dict[str, np.ndarray], n_rows: int) -> np.ndarray:
    """Class probabilities (n_rows x n_classes) for one models.json task.

    raw_k = baseline_k + leaf values summed in iteration order; binary:
    p(class 1) = 1 / (1 + exp(-raw_0)), p(class 0) = 1 - p(class 1);
    multiclass: softmax with the maximum subtracted, summed in class order.
    """
    X = np.empty((n_rows, len(task["columns"])), dtype=np.float64)
    for j, name in enumerate(task["columns"]):
        if name not in columns:
            raise KeyError(f"Missing feature column: {name}")
        X[:, j] = np.asarray(columns[name], dtype=np.float64)[:n_rows]
    k = len(task["baseline"])
    n_classes = len(task["labels"])
    raw = np.empty((n_rows, k), dtype=np.float64)
    for c in range(k):
        raw[:, c] = float(task["baseline"][c])
    for iteration in task["trees"]:
        for c in range(k):
            raw[:, c] += _tree_leaf_values(iteration[c], X)
    out = np.empty((n_rows, n_classes), dtype=np.float64)
    if k == 1:
        p1 = 1.0 / (1.0 + np.exp(-raw[:, 0]))
        out[:, 0] = 1.0 - p1
        out[:, 1] = p1
    else:
        mx = raw.max(axis=1)
        e = np.exp(raw - mx[:, None])
        s = np.zeros(n_rows, dtype=np.float64)
        for c in range(k):
            s += e[:, c]
        out[:] = e / s[:, None]
    return out


def decide(task: dict, proba: np.ndarray) -> np.ndarray:
    """Label index per row: threshold on p(positive) (>=), or the first argmax."""
    d = task["decision"]
    if d["type"] == "threshold":
        pos = task["labels"].index(d["positive"])
        if len(task["labels"]) != 2:
            raise ValueError(f"Bad threshold decision for labels {task['labels']}")
        return np.where(proba[:, pos] >= d["threshold"], pos, 1 - pos).astype(np.int64)
    if d["type"] == "argmax":
        return np.argmax(proba, axis=1).astype(np.int64)
    raise ValueError(f"Unknown decision type: {d['type']}")


# ---------------------------------------------------------------- report

def epoch_grid(start_s: float, end_s: float, epoch_len: float) -> np.ndarray:
    n = max(0, int(math.floor((end_s - start_s) / epoch_len)))
    return start_s + epoch_len * np.arange(n, dtype=np.float64)


def _given(x) -> bool:
    return x is not None


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def build_report(t, rr, models_json: dict, cfg: dict | None = None, lights_off_s=None, lights_on_s=None,
                 file_name: str = "", input_warnings=(), include_proba: bool = False) -> dict:
    """Report object of docs/web_contract.md (same keys and values as web/js/report.js).

    t: closing-beat times (s), rr: intervals (ms), both after CSV parsing.
    cfg: config with "rr" and "features" blocks (default: models_json["config"]).
    input_warnings: warnings from parsing, placed first in quality.warnings.
    Raises ReportError for an unusable grid (codes TOO_SHORT, BAD_LIGHTS).
    """
    cfg = models_json["config"] if cfg is None else cfg
    mcfg = models_json["config"]
    epoch_s = float(mcfg.get("epoch_s", 30))
    minute_s = float(mcfg.get("apnea_epoch_s", 60))
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

    # Same span rule as parseRRCsv: from the start of the first interval to the last beat.
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
    feats30 = F.epoch_features(t, rr, valid, stage_starts, epoch_s, cfg["rr"], cfg["features"])
    feats60 = F.epoch_features(t, rr, valid, minute_starts, minute_s, cfg["rr"], cfg["features"])

    tasks = models_json["tasks"]
    stages, proba_out = {}, {}
    for name in ("sleep_wake", "wake_nrem_rem", "four_class"):
        task = tasks[name]
        p = predict_proba(task, feats30, n_epochs)
        stages[name] = [task["labels"][i] for i in decide(task, p)]
        proba_out[name] = p
    task = tasks["apnea"]
    p = predict_proba(task, feats60, len(minute_starts))
    pos = task["labels"].index(task["decision"]["positive"])
    apnea_labels = [int(i == pos) for i in decide(task, p)]
    proba_out["apnea"] = p

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
