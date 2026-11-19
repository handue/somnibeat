"""Train models and validate externally on UCDDB.

Protocol (fixed before any tuning; see docs/evaluation.md):
- Staging: train on CAP + MIT-BIH slpdb. Apnea: train on Apnea-ECG.
  External test for both: UCDDB, predicted exactly once per model.
- Splits and CV folds are by subject (slpdb a/b segments and Apnea-ECG
  c05/c06 share a subject). Bootstrap CIs on training data resample subjects.
- Models are fitted on epochs with adequate RR coverage. Out-of-fold
  predictions are made for every labeled training epoch, including
  low-quality ones, so internal CV and threshold selection see the same
  population that the external test sees.
- Binary decision thresholds are chosen on those out-of-fold predictions only.
- Baselines: (a) "asleep from lights-off" (every epoch sleep / no apnea);
  (b) majority class of the training labels.
- The run stops if the feature cache does not hold exactly the configured
  records of every dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from somnibeat import datasets  # noqa: E402
from somnibeat import evaluate as ev  # noqa: E402

META = ["dataset", "record", "subject", "epoch", "minute", "epoch_start_s", "stage", "apnea"]
TASKS = {
    "sleep_wake": {"W": "wake", "R": "sleep", "S1": "sleep", "S2": "sleep", "S3": "sleep", "S4": "sleep"},
    "wake_nrem_rem": {"W": "wake", "R": "rem", "S1": "nrem", "S2": "nrem", "S3": "nrem", "S4": "nrem"},
    "four_class": {"W": "wake", "R": "rem", "S1": "light", "S2": "light", "S3": "deep", "S4": "deep"},
}
LABELS = {"sleep_wake": ["sleep", "wake"], "wake_nrem_rem": ["wake", "nrem", "rem"],
          "four_class": ["wake", "light", "deep", "rem"], "apnea": ["normal", "apnea"]}
SPLITS = {"stage": (["cap", "slpdb"], "ucddb"), "apnea": (["apnea-ecg"], "ucddb")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def make_model(name: str, cfg: dict, seed: int):
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **cfg["models"]["gb"])
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(random_state=seed, **cfg["models"]["lr"]))


MODELS = ["gradient_boosting", "logistic_regression"]
ABSOLUTE_PREFIXES = ("mean_rr_", "sdnn_", "rmssd_", "pnn50_", "cv_", "log_", "hf_peak", "dfa_")


def lr_columns(cols: list[str]) -> list[str]:
    """Logistic regression uses per-night normalized features only: absolute
    heart-rate values differ too much between people for a linear model."""
    return [c for c in cols if not c.startswith(ABSOLUTE_PREFIXES)]


def check_cache(df: pd.DataFrame, datasets_needed: list[str], cfg: dict) -> dict:
    expected = datasets.list_records(cfg["paths"]["raw"], cfg)
    found = {}
    for d in datasets_needed:
        have = set(df.loc[df.dataset == d, "record"].unique())
        want = set(expected[d])
        if have != want:
            raise SystemExit(f"Feature cache mismatch for {d}: missing {sorted(want - have)}, "
                             f"unexpected {sorted(have - want)}. Re-run `make features`.")
        found[d] = sorted(have)
    return found


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [ev.cohen_kappa(y, (p >= t).astype(int), labels=[0, 1]) for t in grid]
    return float(grid[int(np.nanargmax(scores))])


def by_record(df: pd.DataFrame, values: np.ndarray) -> dict[str, np.ndarray]:
    return {rid: values[idx] for rid, idx in df.groupby("record").indices.items()}


def oof_predict(name: str, train: pd.DataFrame, cols: list[str], y: np.ndarray, cfg: dict,
                n_cls: int) -> np.ndarray:
    """Subject-grouped CV. Each fold fits on good-coverage rows of the other
    subjects and predicts every row (including low-quality) of its own subjects."""
    groups = (train["dataset"] + "/" + train["subject"]).to_numpy()
    good = train["low_quality"].to_numpy() == 0
    oof = np.full((len(train), n_cls), np.nan)
    gkf = GroupKFold(n_splits=int(cfg["evaluation"]["inner_folds"]))
    for k, (tr, va) in enumerate(gkf.split(train, y, groups)):
        tr = tr[good[tr]]
        m = make_model(name, cfg, int(cfg["seed"]) + k)
        m.fit(train.iloc[tr][cols], y[tr])
        oof[va] = m.predict_proba(train.iloc[va][cols])
    assert not np.isnan(oof).any()
    return oof


def data_block(train: pd.DataFrame, test: pd.DataFrame, train_sets: list[str], test_set: str,
               y_train: np.ndarray, y_test: np.ndarray, labels: list[str], unit: str) -> dict:
    return {
        "train": {d: {"records": int(train[train.dataset == d].record.nunique()),
                      "subjects": int(train[train.dataset == d].subject.nunique()),
                      f"{unit}_labeled": int((train.dataset == d).sum()),
                      f"{unit}_used_for_fitting": int(((train.dataset == d) & (train.low_quality == 0)).sum())}
                  for d in train_sets},
        "test": {test_set: {"records": int(test.record.nunique()), "subjects": int(test.subject.nunique()),
                            f"{unit}_labeled": int(len(test)),
                            f"{unit}_low_quality": int(test.low_quality.sum())}},
        "class_counts_train": {labels[i]: int((y_train == i).sum()) for i in range(len(labels))},
        "class_counts_test": {labels[i]: int((y_test == i).sum()) for i in range(len(labels))},
    }


def run_stage_task(task: str, df_all: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    seed = int(cfg["seed"])
    labels = LABELS[task]
    n_cls = len(labels)
    train_sets, test_set = SPLITS["stage"]
    df = df_all[df_all["stage"].isin(TASKS[task].keys())].copy()
    df["y"] = df["stage"].map(TASKS[task]).map({l: i for i, l in enumerate(labels)}).astype(int)
    feat = [c for c in df.columns if c not in META + ["y"]]
    train = df[df["dataset"].isin(train_sets)].reset_index(drop=True)
    test = df[df["dataset"] == test_set].reset_index(drop=True)
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()
    good = train["low_quality"].to_numpy() == 0
    groups_tr = {r: f"{d}/{s}" for r, d, s in train[["record", "dataset", "subject"]].drop_duplicates().itertuples(index=False)}
    ev_args = dict(n_resamples=int(cfg["evaluation"]["bootstrap_resamples"]), seed=seed,
                   ci=float(cfg["evaluation"]["ci"]))
    idx_tr, idx_te = by_record(train, train["epoch"].to_numpy()), by_record(test, test["epoch"].to_numpy())
    ref_tr, ref_te = by_record(train, y_tr), by_record(test, y_te)
    lab = np.array(labels)
    result = {"task": task, "labels": labels, "models": {}, "baselines": {}}

    for name in MODELS:
        cols = feat if name == "gradient_boosting" else lr_columns(feat)
        oof = oof_predict(name, train, cols, y_tr, cfg, n_cls)
        model = make_model(name, cfg, seed)
        model.fit(train.loc[good, cols], y_tr[good])
        p_te = model.predict_proba(test[cols])
        entry = {"n_features": len(cols)}
        if n_cls == 2:
            thr = best_threshold(y_tr, oof[:, 1])
            entry["threshold_p_wake"] = thr
            yo, yt = (oof[:, 1] >= thr).astype(int), (p_te[:, 1] >= thr).astype(int)
            entry["internal_cv"] = ev.binary_suite(ref_tr, by_record(train, yo), **ev_args,
                                                   epoch_idx=idx_tr, groups=groups_tr)
            entry["external_ucddb"] = ev.binary_suite(ref_te, by_record(test, yt), **ev_args, epoch_idx=idx_te)
        else:
            yo, yt = oof.argmax(1), p_te.argmax(1)
            entry["internal_cv"] = ev.multiclass_suite(
                {k: lab[v] for k, v in ref_tr.items()}, {k: lab[v] for k, v in by_record(train, yo).items()},
                labels, **ev_args, groups=groups_tr)
            entry["external_ucddb"] = ev.multiclass_suite(
                {k: lab[v] for k, v in ref_te.items()}, {k: lab[v] for k, v in by_record(test, yt).items()},
                labels, **ev_args)
            w = labels.index("wake")
            sw = ev.binary_suite({k: (v == w).astype(int) for k, v in ref_te.items()},
                                 {k: (v == w).astype(int) for k, v in by_record(test, yt).items()},
                                 **ev_args, epoch_idx=idx_te)
            sw.pop("per_record")
            entry["external_ucddb_sleep_wake"] = sw
        entry["internal_cv"].pop("per_record", None)
        result["models"][name] = entry
        save_model(out_dir, task, name, model, cols, labels, entry.get("threshold_p_wake"), p_te, test)

    majority = int(np.bincount(y_tr[good]).argmax())
    sleep_like = 0 if task == "sleep_wake" else labels.index("light" if task == "four_class" else "nrem")
    for bname, cls in (("asleep_from_lights_off", sleep_like), ("majority_class", majority)):
        pred = {k: np.full(len(v), cls) for k, v in ref_te.items()}
        if n_cls == 2:
            b = ev.binary_suite(ref_te, pred, **ev_args, epoch_idx=idx_te)
            b.pop("per_record")
        else:
            b = ev.multiclass_suite({k: lab[v] for k, v in ref_te.items()},
                                    {k: lab[v] for k, v in pred.items()}, labels, **ev_args)
        b["predicted_class"] = labels[cls]
        result["baselines"][bname] = b
    if task != "sleep_wake":
        result["baselines"]["asleep_from_lights_off"]["note"] = (
            f"Every epoch predicted as '{labels[sleep_like]}', the most common sleep stage.")
    if majority == sleep_like:
        result["baselines"]["majority_class"]["note"] = (
            "Identical to the lights-off baseline: the training majority class is the same class.")
    result["data"] = data_block(train, test, train_sets, test_set, y_tr, y_te, labels, "epochs")
    return result


def run_apnea_task(df_all: pd.DataFrame, cfg: dict, out_dir: Path) -> dict:
    seed = int(cfg["seed"])
    labels = LABELS["apnea"]
    train_sets, test_set = SPLITS["apnea"]
    df = df_all[df_all["apnea"] >= 0].copy()
    df["y"] = df["apnea"].astype(int)
    df["epoch"] = df["minute"]
    feat = [c for c in df.columns if c not in META + ["y"]]
    train = df[df["dataset"].isin(train_sets)].reset_index(drop=True)
    test = df[df["dataset"] == test_set].reset_index(drop=True)
    y_tr, y_te = train["y"].to_numpy(), test["y"].to_numpy()
    good = train["low_quality"].to_numpy() == 0
    groups_tr = {r: f"{d}/{s}" for r, d, s in train[["record", "dataset", "subject"]].drop_duplicates().itertuples(index=False)}
    ev_args = dict(n_resamples=int(cfg["evaluation"]["bootstrap_resamples"]), seed=seed,
                   ci=float(cfg["evaluation"]["ci"]))
    ref_tr, ref_te = by_record(train, y_tr), by_record(test, y_te)
    result = {"task": "apnea", "labels": labels, "unit": "minute", "models": {}, "baselines": {}}
    for name in MODELS:
        cols = feat if name == "gradient_boosting" else lr_columns(feat)
        oof = oof_predict(name, train, cols, y_tr, cfg, 2)
        model = make_model(name, cfg, seed)
        model.fit(train.loc[good, cols], y_tr[good])
        p_te = model.predict_proba(test[cols])
        thr = best_threshold(y_tr, oof[:, 1])
        yo, yt = (oof[:, 1] >= thr).astype(int), (p_te[:, 1] >= thr).astype(int)
        entry = {"n_features": len(cols), "threshold_p_apnea": thr,
                 "internal_cv": ev.count_suite(ref_tr, by_record(train, yo), 1.0, **ev_args, groups=groups_tr),
                 "external_ucddb": ev.count_suite(ref_te, by_record(test, yt), 1.0, **ev_args)}
        entry["internal_cv"].pop("per_record")
        result["models"][name] = entry
        save_model(out_dir, "apnea", name, model, cols, labels, thr, p_te, test)
    majority = int(np.bincount(y_tr[good]).argmax())
    for bname, cls in (("no_apnea", 0), ("majority_class", majority)):
        b = ev.count_suite(ref_te, {k: np.full(len(v), cls) for k, v in ref_te.items()}, 1.0, **ev_args)
        b.pop("per_record")
        b["predicted_class"] = labels[cls]
        result["baselines"][bname] = b
    result["data"] = data_block(train, test, train_sets, test_set, y_tr, y_te, labels, "minutes")
    return result


def save_model(out_dir: Path, task: str, name: str, model, cols, labels, thr, p_te, test) -> None:
    mdir = out_dir / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / f"{task}_{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "columns": cols, "labels": labels, "threshold": thr}, f)
    keep = [c for c in ("record", "epoch", "minute") if c in test.columns]
    out = test[keep].copy()
    for i, l in enumerate(labels):
        out[f"p_{l}"] = p_te[:, i]
    out.to_csv(mdir / f"{task}_{name}_ucddb_predictions.csv.gz", index=False,
               compression={"method": "gzip", "mtime": 0})


def plot_bland_altman(result: dict, figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs.mkdir(parents=True, exist_ok=True)
    task = result["task"]
    keys = (("apnea_min", "min"),) if task == "apnea" else (("tst_min", "min"), ("se_pct", "percentage points"))
    for name, entry in result["models"].items():
        ext = entry.get("external_ucddb_sleep_wake", entry["external_ucddb"]) if task != "apnea" else entry["external_ucddb"]
        rows = entry["external_ucddb"].get("per_record")
        if rows is None or "bland_altman" not in ext:
            continue
        for key, unit in keys:
            if f"ref_{key}" not in rows[0]:
                continue
            ref = np.array([r[f"ref_{key}"] for r in rows])
            pred = np.array([r[f"pred_{key}"] for r in rows])
            ba = ext["bland_altman"][key]
            fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
            ax.scatter((ref + pred) / 2, pred - ref, s=18, color="#2b6cb0")
            for v, ls in ((ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")):
                ax.axhline(v, color="#444", ls=ls, lw=1)
            ax.set_xlabel(f"Mean of model and PSG ({unit})")
            ax.set_ylabel(f"Model - PSG ({unit})")
            ax.set_title(f"{key} | {name.replace('_', ' ')} | UCDDB n={len(rows)}")
            fig.tight_layout()
            fig.savefig(figs / f"bland_altman_{task}_{key}_{name}.png")
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--tasks", nargs="*", default=["sleep_wake"])
    a = ap.parse_args()
    raw_cfg = Path(a.config).read_bytes()
    cfg = tomllib.loads(raw_cfg.decode())
    out_dir = Path(cfg["paths"]["results"])
    cache = Path(cfg["paths"]["cache"])
    stage_path, apnea_path = cache / "stage_features.csv.gz", cache / "apnea_features.csv.gz"
    stage_df = apnea_df = None
    for task in a.tasks:
        if task == "apnea":
            if apnea_df is None:
                apnea_df = pd.read_csv(apnea_path)
            records = check_cache(apnea_df, ["apnea-ecg", "ucddb"], cfg)
            res = run_apnea_task(apnea_df, cfg, out_dir)
            res["feature_cache_sha256"] = sha256_file(apnea_path)
        else:
            if stage_df is None:
                stage_df = pd.read_csv(stage_path, dtype={"stage": str}, keep_default_na=False,
                                       na_values=["", "nan", "NaN"])
                stage_df["stage"] = stage_df["stage"].fillna("")
            records = check_cache(stage_df, ["cap", "slpdb", "ucddb"], cfg)
            res = run_stage_task(task, stage_df, cfg, out_dir)
            res["feature_cache_sha256"] = sha256_file(stage_path)
        res["records"] = records
        res["config_sha256"] = hashlib.sha256(raw_cfg).hexdigest()
        res["seed"] = int(cfg["seed"])
        plot_bland_altman(res, out_dir / "figures")
        (out_dir / f"{task}.json").write_text(json.dumps(res, indent=2))
        print(f"wrote {out_dir / f'{task}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
