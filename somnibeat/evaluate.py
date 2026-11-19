"""Night-level and epoch-level evaluation.

Definitions (also in docs/evaluation.md):
- Only epochs with a reference label are evaluated (artifact/unscored epochs
  are dropped from both reference and prediction).
- Time in bed (TIB) = number of evaluated epochs x 30 s. The datasets have no
  lights-off/lights-on markers, so the scored span is used as TIB.
- Total sleep time (TST) = sleep epochs x 30 s.
- Sleep efficiency (SE) = TST / TIB x 100.
- Awakenings = runs of >= 2 consecutive wake epochs (>= 1 min) that start
  after the first sleep epoch and end before the last sleep epoch.
- Apnea minutes = minutes labeled apnea (60 s grid).
- Errors are prediction - reference (positive = over-estimate).
- Record-level bootstrap: resample records with replacement; every metric is
  recomputed on the resampled set; percentile interval.
"""

from __future__ import annotations

import math

import numpy as np

EPOCH_MIN = 0.5


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)
    k = len(labels)
    lab = np.asarray(labels)
    order = np.argsort(lab)
    ti = order[np.searchsorted(lab[order], y_true)]
    pi = order[np.searchsorted(lab[order], y_pred)]
    if not (np.array_equal(lab[ti], y_true) and np.array_equal(lab[pi], y_pred)):
        raise ValueError("labels outside the declared label set")
    cm = np.bincount(ti * k + pi, minlength=k * k).reshape(k, k).astype(float)
    n = cm.sum()
    if n == 0:
        return math.nan
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / (n * n)
    # Undefined when both raters use a single class (pe == 1): NaN, skipped by nanmean.
    return float((po - pe) / (1 - pe)) if pe < 1 else math.nan


def awakenings(is_wake: np.ndarray, epoch_idx: np.ndarray | None = None, min_epochs: int = 2) -> int:
    """Wake runs of >= min_epochs between the first and last sleep epoch.

    epoch_idx gives each evaluated epoch's position on the original grid. A
    gap (dropped, unscored epochs) ends the current run without counting it,
    so wake on both sides of a gap is never merged into one awakening.
    """
    w = np.asarray(is_wake, bool)
    idx = np.arange(len(w)) if epoch_idx is None else np.asarray(epoch_idx)
    sleep_pos = np.where(~w)[0]
    if len(sleep_pos) == 0:
        return 0
    count, run = 0, 0
    for i in range(sleep_pos[0], sleep_pos[-1] + 1):
        if i > sleep_pos[0] and idx[i] - idx[i - 1] > 1:
            run = 0
        if w[i]:
            run += 1
        else:
            if run >= min_epochs:
                count += 1
            run = 0
    return count


def night_metrics(is_wake: np.ndarray, epoch_idx: np.ndarray | None = None) -> dict:
    w = np.asarray(is_wake, bool)
    n = len(w)
    tst = float((~w).sum()) * EPOCH_MIN
    tib = n * EPOCH_MIN
    return {"tib_min": tib, "tst_min": tst, "se_pct": 100.0 * tst / tib if tib else math.nan,
            "awakenings": awakenings(w, epoch_idx)}


def record_table(ref: dict[str, np.ndarray], pred: dict[str, np.ndarray],
                 epoch_idx: dict[str, np.ndarray] | None = None) -> list[dict]:
    rows = []
    for rid in sorted(ref):
        ix = None if epoch_idx is None else epoch_idx[rid]
        r, p = night_metrics(ref[rid], ix), night_metrics(pred[rid], ix)
        rows.append({"record": rid, **{f"ref_{k}": v for k, v in r.items()},
                     **{f"pred_{k}": v for k, v in p.items()},
                     "kappa": cohen_kappa(ref[rid].astype(int), pred[rid].astype(int), labels=[0, 1])})
    return rows


def summarize_errors(rows: list[dict], key: str) -> dict:
    e = np.array([r[f"pred_{key}"] - r[f"ref_{key}"] for r in rows], float)
    sd = float(e.std(ddof=1)) if len(e) > 1 else math.nan
    bias = float(e.mean())
    return {"bias": bias, "mae": float(np.abs(e).mean()), "sd": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "per_record_error": e.tolist()}


def bootstrap(records: list[str], stat_fn, n_resamples: int, seed: int, ci: float = 0.95,
              groups: dict[str, str] | None = None) -> dict:
    """stat_fn(list_of_record_ids) -> dict[str, float]. Returns point, lo, hi per stat.

    With groups (record -> subject), subjects are resampled and every record
    of a drawn subject is included, so repeated nights of one person are not
    treated as independent.
    """
    rng = np.random.default_rng(seed)
    point = stat_fn(records)
    samples = {k: [] for k in point}
    if groups is None:
        groups = {r: r for r in records}
    units = sorted(set(groups[r] for r in records))
    members = {u: [r for r in records if groups[r] == u] for u in units}
    for _ in range(n_resamples):
        drawn = rng.choice(np.array(units), size=len(units), replace=True)
        s = stat_fn([r for u in drawn for r in members[u]])
        for k, v in s.items():
            samples[k].append(v)
    a = (1 - ci) / 2
    return {k: {"point": float(point[k]),
                "ci_low": float(np.nanpercentile(samples[k], 100 * a)),
                "ci_high": float(np.nanpercentile(samples[k], 100 * (1 - a)))} for k in point}


def binary_suite(ref: dict[str, np.ndarray], pred: dict[str, np.ndarray],
                 n_resamples: int, seed: int, ci: float,
                 epoch_idx: dict[str, np.ndarray] | None = None,
                 groups: dict[str, str] | None = None) -> dict:
    """Full sleep/wake evaluation. Arrays are 1 = wake, 0 = sleep."""
    rows = record_table(ref, pred, epoch_idx)
    by = {r["record"]: r for r in rows}

    def stats(ids):
        rr = [by[i] for i in ids]
        yt = np.concatenate([ref[i] for i in ids]).astype(int)
        yp = np.concatenate([pred[i] for i in ids]).astype(int)
        e_tst = np.array([r["pred_tst_min"] - r["ref_tst_min"] for r in rr])
        e_se = np.array([r["pred_se_pct"] - r["ref_se_pct"] for r in rr])
        e_aw = np.array([r["pred_awakenings"] - r["ref_awakenings"] for r in rr])
        return {"kappa_pooled": cohen_kappa(yt, yp, labels=[0, 1]),
                "kappa_mean_per_record": float(np.nanmean([r["kappa"] for r in rr])),
                "accuracy": float((yt == yp).mean()),
                "wake_sensitivity": float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                "sleep_sensitivity": float(((yp == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1)),
                "tst_bias_min": float(e_tst.mean()), "tst_mae_min": float(np.abs(e_tst).mean()),
                "se_bias_pctpt": float(e_se.mean()), "se_mae_pctpt": float(np.abs(e_se).mean()),
                "awakenings_bias": float(e_aw.mean()), "awakenings_mae": float(np.abs(e_aw).mean())}

    return {"n_records": len(rows), "n_epochs": int(sum(len(v) for v in ref.values())),
            "metrics": bootstrap(sorted(ref), stats, n_resamples, seed, ci, groups),
            "bland_altman": {"tst_min": summarize_errors(rows, "tst_min"),
                             "se_pct": summarize_errors(rows, "se_pct"),
                             "awakenings": summarize_errors(rows, "awakenings")},
            "per_record": rows}


def multiclass_suite(ref: dict[str, np.ndarray], pred: dict[str, np.ndarray], labels: list[str],
                     n_resamples: int, seed: int, ci: float,
                     groups: dict[str, str] | None = None) -> dict:
    def stats(ids):
        yt = np.concatenate([ref[i] for i in ids])
        yp = np.concatenate([pred[i] for i in ids])
        out = {"kappa_pooled": cohen_kappa(yt, yp, labels=labels),
               "accuracy": float((yt == yp).mean())}
        for l in labels:
            tp = float(((yt == l) & (yp == l)).sum())
            out[f"recall_{l}"] = tp / max(float((yt == l).sum()), 1.0)
            out[f"precision_{l}"] = tp / max(float((yp == l).sum()), 1.0)
            e = [float((pred[i] == l).sum() - (ref[i] == l).sum()) * EPOCH_MIN for i in ids]
            out[f"minutes_bias_{l}"] = float(np.mean(e))
            out[f"minutes_mae_{l}"] = float(np.mean(np.abs(e)))
        return out

    yt = np.concatenate([ref[i] for i in sorted(ref)])
    yp = np.concatenate([pred[i] for i in sorted(ref)])
    cm = [[int(((yt == a) & (yp == b)).sum()) for b in labels] for a in labels]
    return {"n_records": len(ref), "n_epochs": int(len(yt)), "labels": labels,
            "confusion_matrix_rows_ref_cols_pred": cm,
            "metrics": bootstrap(sorted(ref), stats, n_resamples, seed, ci, groups)}


def count_suite(ref: dict[str, np.ndarray], pred: dict[str, np.ndarray], unit_min: float,
                n_resamples: int, seed: int, ci: float, groups: dict[str, str] | None = None) -> dict:
    """Apnea minutes: per-record count error (prediction - reference) plus minute-level agreement."""
    rows = []
    for rid in sorted(ref):
        rows.append({"record": rid, "ref_apnea_min": float(ref[rid].sum()) * unit_min,
                     "pred_apnea_min": float(pred[rid].sum()) * unit_min,
                     "total_min": float(len(ref[rid])) * unit_min,
                     "kappa": cohen_kappa(ref[rid], pred[rid], labels=[0, 1])})
    by = {r["record"]: r for r in rows}

    def stats(ids):
        e = np.array([by[i]["pred_apnea_min"] - by[i]["ref_apnea_min"] for i in ids])
        yt = np.concatenate([ref[i] for i in ids])
        yp = np.concatenate([pred[i] for i in ids])
        return {"apnea_min_bias": float(e.mean()), "apnea_min_mae": float(np.abs(e).mean()),
                "kappa_pooled": cohen_kappa(yt, yp, labels=[0, 1]),
                "accuracy": float((yt == yp).mean()),
                "sensitivity": float(((yp == 1) & (yt == 1)).sum() / max((yt == 1).sum(), 1)),
                "specificity": float(((yp == 0) & (yt == 0)).sum() / max((yt == 0).sum(), 1))}

    e = np.array([r["pred_apnea_min"] - r["ref_apnea_min"] for r in rows])
    sd = float(e.std(ddof=1)) if len(e) > 1 else math.nan
    return {"n_records": len(rows), "n_minutes": int(sum(len(v) for v in ref.values())),
            "metrics": bootstrap(sorted(ref), stats, n_resamples, seed, ci, groups),
            "bland_altman": {"apnea_min": {"bias": float(e.mean()), "mae": float(np.abs(e).mean()), "sd": sd,
                                           "loa_low": float(e.mean()) - 1.96 * sd,
                                           "loa_high": float(e.mean()) + 1.96 * sd,
                                           "per_record_error": e.tolist()}},
            "per_record": rows}
