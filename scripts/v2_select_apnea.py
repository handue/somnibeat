"""v2 apnea: slpdb minute labels, slpdb minute features and candidate selection (docs/v2_protocol.md).

Usage:
  python scripts/v2_select_apnea.py check      # facts of the slpdb .st annotations -> results/v2/slpdb_apnea_check.json
  python scripts/v2_select_apnea.py features   # slpdb 60 s minute features -> data/cache/v2/apnea_features_slpdb.csv.gz
                                               # (+ v2 long-context columns -> apnea_features_extra_slpdb.csv.gz)
  python scripts/v2_select_apnea.py select     # candidates -> results/v2/selection_apnea.json
  python scripts/v2_select_apnea.py all

Only training databases (Apnea-ECG learning set, MIT-BIH slpdb) are read. UCDDB and HMC rows are
filtered out while reading and every frame is checked with assert_training_only.

Selection design
- Cross-dataset CV (primary): fold A fits on apnea-ecg and evaluates slpdb minutes; fold B fits on
  slpdb and evaluates apnea-ecg minutes. The score is the mean of the two folds' pooled minute
  kappa; apnea-minute MAE (per record) is the guard (not worse than the v1 reference by > 10 %).
  These folds measure model / class weight / feature set; they are identical for both values of
  `train_on`, because each fold's training side is one database by construction.
- Subject-grouped 5-fold CV on the union apnea-ecg + slpdb (secondary; it is also the only place
  where `train_on` = apnea-ecg vs apnea-ecg+slpdb can be compared on the same held-out minutes:
  the apnea-ecg-only variant fits on the apnea-ecg subjects of each training fold).
- Models fit on low_quality == 0 rows of the training side and predict every labeled row of the
  evaluation side. The decision threshold is chosen on out-of-fold predictions of a
  GroupKFold(5) inside the training side only (same grid and kappa rule as v1).
- v1 reference: gradient boosting, v1 features, no class weight, config/default.toml
  hyperparameters, fit on apnea-ecg. Its fold A is v1's procedure step for step (final model seed,
  inner seeds seed + k), so its fold A threshold must equal v1's 0.36; `v1_protocol_check`
  re-runs v1's internal CV and compares it with results/apnea.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import apnea_labels as al  # noqa: E402
from somnibeat import datasets, features, features_v2  # noqa: E402
from somnibeat import evaluate as ev  # noqa: E402
from somnibeat import rr as rrm  # noqa: E402
from somnibeat import v2_models as vm  # noqa: E402
from somnibeat.provenance import drop_timing  # noqa: E402

GZIP = {"method": "gzip", "mtime": 0}
RAW_SLPDB = ROOT / "data" / "raw" / "slpdb"
V1_APNEA = ROOT / "data" / "cache" / "apnea_features.csv.gz"
V2_CACHE = ROOT / "data" / "cache" / "v2"
SLPDB_FEATURES = V2_CACHE / "apnea_features_slpdb.csv.gz"
EXTRA_MAIN = V2_CACHE / "apnea_features_extra.csv.gz"            # written by build_features_v2.py
EXTRA_SLPDB = V2_CACHE / "apnea_features_extra_slpdb.csv.gz"     # written here (same code)
RESULTS = ROOT / "results" / "v2"
CHECK_OUT = RESULTS / "slpdb_apnea_check.json"
SELECT_OUT = RESULTS / "selection_apnea.json"
V1_RESULTS = ROOT / "results" / "apnea.json"
APNEA_SETS = {"apnea-ecg", "slpdb"}
META = ["dataset", "record", "subject", "minute", "apnea", "y"]
TRAIN_ON = ("apnea-ecg", "apnea-ecg+slpdb")


def assert_training_only(df: pd.DataFrame) -> None:
    vm.assert_training_only(df)
    assert set(df["dataset"].unique()) <= APNEA_SETS, sorted(df["dataset"].unique())


def slpdb_records() -> list[str]:
    return (RAW_SLPDB / "RECORDS").read_text().split()


# ----------------------------------------------------------------------------- check

def cmd_check() -> dict:
    recs = slpdb_records()
    facts = {r: al.record_facts(RAW_SLPDB, r) for r in recs}
    total = {}
    for f in facts.values():
        for k, v in f["event_token_counts"].items():
            total[k] = total.get(k, 0) + v
    out = {
        "source": "data/raw/slpdb/<record>.st (wfdb aux_note), token meanings from data/raw/slpdb/README.st",
        "findings": {
            "one_annotation_per_30s_epoch": all(f["max_offset_from_30s_grid_s"] < 0.1 for f in facts.values()),
            "max_offset_from_30s_grid_s": max(f["max_offset_from_30s_grid_s"] for f in facts.values()),
            "aux_format": "stage token first (W 1 2 3 4 R MT M), then zero or more event tokens",
            "tokens_have_durations": False,
            "tokens_are_per_epoch_markers": True,
            "note_durations": "Event tokens are bare labels in an epoch's aux string; no onset or "
                              "duration field exists, so only presence per 30 s epoch is known.",
            "tokens_seen_beyond_readme": sorted(set(total) - {"H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "MT"}),
            "records_without_any_event_token": [r for r, f in facts.items() if not f["event_token_counts"]],
            "records_without_respiratory_token": [r for r, f in facts.items() if f["epochs_with_respiratory_token"] == 0],
            "records_scoring_starts_after_0s": {r: f["first_annotation_s"] for r, f in facts.items()
                                                if f["first_annotation_s"] > 1.0},
        },
        "minute_rule": {
            "grid": "minute m = [60 m, 60 m + 60) s = 30 s grid epochs 2m and 2m+1; "
                    "floor(duration / 60) minutes (as Apnea-ECG)",
            "apnea_tokens": sorted(al.RESP_TOKENS),
            "ignored_tokens": sorted(al.OTHER_TOKENS),
            "label": "1 if an apnea token is in either epoch; 0 if both epochs annotated without one; "
                     "-1 otherwise",
        },
        "event_token_counts_total": dict(sorted(total.items())),
        "minutes_total": {k: int(sum(f[k] for f in facts.values()))
                          for k in ("minutes_total", "minutes_apnea", "minutes_normal", "minutes_unlabeled")},
        "records": facts,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECK_OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {CHECK_OUT}")
    return out


# ----------------------------------------------------------------------------- features

def _slpdb_minutes(rid: str) -> dict:
    v1cfg = tomllib.loads((ROOT / "config" / "default.toml").read_text())
    v2cfg = tomllib.loads((ROOT / "config" / "v2.toml").read_text())
    rr_df = pd.read_csv(ROOT / "data" / "cache" / "rr" / f"slpdb_{rid}.csv")
    t = rr_df["time_s"].to_numpy(np.float64)
    rr = rr_df["rr_ms"].to_numpy(np.float64)
    lab = al.slpdb_minute_labels(RAW_SLPDB, rid)
    starts = np.arange(len(lab)) * 60.0
    valid = rrm.clean_rr(t, rr, v1cfg["rr"])
    f = features.epoch_features(t, rr, valid, starts, 60.0, v1cfg["rr"], v1cfg["features"])
    df = pd.DataFrame(f)
    df.insert(0, "apnea", lab)
    df.insert(0, "minute", np.arange(len(df)))
    for c, v in (("subject", datasets.SLPDB_SUBJECT.get(rid, rid)), ("record", rid), ("dataset", "slpdb")):
        df.insert(0, c, v)
    valid2 = rrm.clean_rr(t, rr, v2cfg["rr"])
    ex = pd.DataFrame(features_v2.long_features(t, rr, valid2, starts, 60.0, v2cfg["rr"]))
    ex.insert(0, "minute", np.arange(len(ex)))
    ex.insert(0, "record", rid)
    ex.insert(0, "dataset", "slpdb")
    return {"base": df, "extra": ex}


def cmd_features(workers: int) -> None:
    recs = slpdb_records()
    with ProcessPoolExecutor(workers) as exe:
        res = list(exe.map(_slpdb_minutes, recs))
    base = pd.concat([r["base"] for r in res], ignore_index=True)
    extra = pd.concat([r["extra"] for r in res], ignore_index=True)
    v1_cols = list(pd.read_csv(V1_APNEA, nrows=0).columns)
    assert list(base.columns) == v1_cols, "column mismatch with the v1 apnea cache"
    V2_CACHE.mkdir(parents=True, exist_ok=True)
    base.to_csv(SLPDB_FEATURES, index=False, compression=GZIP)
    extra.to_csv(EXTRA_SLPDB, index=False, compression=GZIP)
    print(f"wrote {SLPDB_FEATURES} ({len(base)} minutes, {base.record.nunique()} records, "
          f"labeled {int((base.apnea >= 0).sum())}, apnea {int((base.apnea == 1).sum())}) and {EXTRA_SLPDB}")


# ----------------------------------------------------------------------------- data for selection

def load_data() -> tuple[pd.DataFrame, list[str], list[str]]:
    a = vm.read_training_csv(V1_APNEA, {"apnea-ecg"})
    s = pd.read_csv(SLPDB_FEATURES)
    assert_training_only(s)
    assert list(a.columns) == list(s.columns)
    df = pd.concat([a, s], ignore_index=True)
    v1_cols = [c for c in df.columns if c not in META]
    extra_cols: list[str] = []
    if EXTRA_MAIN.exists() and EXTRA_SLPDB.exists():
        ex = pd.concat([vm.read_training_csv(EXTRA_MAIN, {"apnea-ecg"}), pd.read_csv(EXTRA_SLPDB)],
                       ignore_index=True)
        assert_training_only(ex)
        keys = ["dataset", "record", "minute"]
        extra_cols = [c for c in ex.columns if c not in keys and c not in v1_cols]
        assert not ex.duplicated(keys).any()
        n = len(df)
        df = df.merge(ex[keys + extra_cols], on=keys, how="left", validate="one_to_one", indicator=True)
        assert len(df) == n
        assert (df["_merge"] == "both").all(), "extra features missing for some minutes"
        df = df.drop(columns="_merge")
    df = df[df["apnea"] >= 0].reset_index(drop=True)
    df["y"] = df["apnea"].astype(int)
    assert_training_only(df)
    return df, v1_cols, extra_cols


# ----------------------------------------------------------------------------- harness

class ApneaHarness:
    def __init__(self, df: pd.DataFrame, v1_cols: list[str], extra_cols: list[str], hp: dict,
                 n_resamples: int):
        assert_training_only(df)
        self.df, self.v1_cols, self.extra_cols, self.hp = df, v1_cols, extra_cols, hp
        self.seed, self.n_inner, self.n_resamples = hp["seed"], hp["inner_folds"], n_resamples
        self.y = df["y"].to_numpy()
        self.good = df["low_quality"].to_numpy() == 0
        self.groups = (df["dataset"] + "/" + df["subject"]).to_numpy()
        self.ds = df["dataset"].to_numpy()
        self.folds = {"A": (np.where(self.ds == "apnea-ecg")[0], np.where(self.ds == "slpdb")[0]),
                      "B": (np.where(self.ds == "slpdb")[0], np.where(self.ds == "apnea-ecg")[0])}
        for k, (tr, va) in enumerate(GroupKFold(self.n_inner).split(df, self.y, self.groups)):
            self.folds[f"S{k}"] = (tr, va)
        self.cache: dict = {}
        self.n_fits, self.fit_seconds = 0, 0.0

    def columns(self, base: str, fs: str) -> list[str]:
        cols = self.v1_cols + (self.extra_cols if fs == "v1+extra" else [])
        return cols if base == "gb" else vm.lr_columns(cols)

    def raw(self, base, cw, fs, fit_idx, pred_idx, seed, key) -> np.ndarray:
        ck = (base, cw, fs) + key
        if ck not in self.cache:
            fit = fit_idx[self.good[fit_idx]]
            assert set(self.ds[fit]) <= APNEA_SETS
            cols = self.columns(base, fs)
            t0 = time.time()
            m = vm.make_model(base, self.hp, seed, cw)
            m.fit(self.df.iloc[fit][cols], self.y[fit])
            self.cache[ck] = m.predict_proba(self.df.iloc[pred_idx][cols])[:, 1]
            self.n_fits += 1
            self.fit_seconds += time.time() - t0
        return self.cache[ck]

    def proba(self, spec, fit_idx, pred_idx, seed, key) -> np.ndarray:
        ps = [self.raw(b, spec["class_weight"], spec["feature_set"], fit_idx, pred_idx, seed, key)
              for b in vm.base_models(spec["model"])]
        return ps[0] if len(ps) == 1 else np.mean(ps, axis=0)

    def threshold(self, spec, tr, key) -> float:
        oof = np.full(len(tr), np.nan)
        for k, (a, b) in enumerate(GroupKFold(self.n_inner).split(tr, self.y[tr], self.groups[tr])):
            oof[b] = self.proba(spec, tr[a], tr[b], self.seed + k, key + ("inner", k))
        assert not np.isnan(oof).any()
        return vm.best_threshold(self.y[tr], oof)

    def train_side(self, spec, tr) -> np.ndarray:
        if spec["train_on"] == "apnea-ecg":
            return tr[self.ds[tr] == "apnea-ecg"]
        return tr

    def fold(self, spec, fold) -> tuple[np.ndarray, np.ndarray, float]:
        tr, va = self.folds[fold]
        key = (fold,)
        if fold.startswith("S"):
            tr = self.train_side(spec, tr)
            key = (fold, spec["train_on"])
            seed = self.seed + int(fold[1:])
        else:
            seed = self.seed
        p = self.proba(spec, tr, va, seed, key)
        thr = self.threshold(spec, tr, key)
        return va, (p >= thr).astype(int), thr

    def evaluate(self, idx, pred, per_record=False) -> dict:
        fr = self.df.iloc[idx]
        y = self.y[idx]
        recs = fr.groupby("record").indices
        groups = {r: f"{d}/{s}" for r, d, s in fr[["record", "dataset", "subject"]].drop_duplicates().itertuples(index=False)}
        out = ev.count_suite({r: y[i] for r, i in recs.items()}, {r: pred[i] for r, i in recs.items()}, 1.0,
                             n_resamples=self.n_resamples, seed=self.seed, ci=0.95, groups=groups)
        out["kappa_mean_per_record"] = float(np.nanmean([r["kappa"] for r in out["per_record"]]))
        if not per_record:
            out.pop("per_record")
            out["bland_altman"]["apnea_min"].pop("per_record_error")
        return out

    def run(self, spec) -> dict:
        t0 = time.time()
        res = {"cross_dataset": {}, "thresholds": {}}
        for fold, name in (("A", "train_apnea-ecg_eval_slpdb"), ("B", "train_slpdb_eval_apnea-ecg")):
            va, pred, thr = self.fold(spec, fold)
            res["cross_dataset"][name] = self.evaluate(va, pred, per_record=True)
            res["thresholds"][fold] = thr
        a, b = (res["cross_dataset"][k]["metrics"] for k in ("train_apnea-ecg_eval_slpdb", "train_slpdb_eval_apnea-ecg"))
        res["cross_dataset"]["mean"] = {k: (a[k]["point"] + b[k]["point"]) / 2 for k in a}
        pred_all = np.full(len(self.y), -1)
        for k in range(self.n_inner):
            va, pred, thr = self.fold(spec, f"S{k}")
            pred_all[va] = pred
            res["thresholds"][f"S{k}"] = thr
        assert (pred_all >= 0).all()
        res["subject_cv_union"] = {"all": self.evaluate(np.arange(len(self.y)), pred_all)}
        for d in sorted(APNEA_SETS):
            ix = np.where(self.ds == d)[0]
            res["subject_cv_union"][d] = self.evaluate(ix, pred_all[ix])
        res["seconds"] = round(time.time() - t0, 1)
        return res

    def v1_protocol_check(self) -> dict:
        """v1's internal CV on apnea-ecg only (pooled OOF threshold), compared with results/apnea.json."""
        ix = np.where(self.ds == "apnea-ecg")[0]
        spec = {"model": "gb", "class_weight": None, "feature_set": "v1"}
        oof = np.full(len(ix), np.nan)
        for k, (a, b) in enumerate(GroupKFold(self.n_inner).split(ix, self.y[ix], self.groups[ix])):
            oof[b] = self.proba(spec, ix[a], ix[b], self.seed + k, ("A", "inner", k))
        thr = vm.best_threshold(self.y[ix], oof)
        e = self.evaluate(ix, (oof >= thr).astype(int))
        v1 = json.loads(V1_RESULTS.read_text())["models"]["gradient_boosting"]
        mine = {"threshold": thr, "kappa_pooled": e["metrics"]["kappa_pooled"]["point"],
                "apnea_min_mae": e["metrics"]["apnea_min_mae"]["point"],
                "apnea_min_bias": e["metrics"]["apnea_min_bias"]["point"], "n_minutes": e["n_minutes"]}
        ref = {"threshold": v1["threshold_p_apnea"], "kappa_pooled": v1["internal_cv"]["metrics"]["kappa_pooled"]["point"],
               "apnea_min_mae": v1["internal_cv"]["metrics"]["apnea_min_mae"]["point"],
               "apnea_min_bias": v1["internal_cv"]["metrics"]["apnea_min_bias"]["point"],
               "n_minutes": v1["internal_cv"]["n_minutes"]}
        same = all(abs(mine[k] - ref[k]) < 1e-9 for k in mine)
        return {"reproduced": mine, "results_apnea_json": ref, "identical": same}


# ----------------------------------------------------------------------------- selection

V1_REFERENCE = {"name": "v1_reference", "model": "gb", "class_weight": None, "feature_set": "v1",
                "train_on": "apnea-ecg"}


def default_grid(extra_ok: bool) -> list[dict]:
    specs = [dict(V1_REFERENCE)]
    for m in vm.MODELS:
        for cw in vm.CLASS_WEIGHTS:
            for fs in vm.FEATURE_SETS:
                for to in TRAIN_ON:
                    s = {"model": m, "class_weight": cw, "feature_set": fs, "train_on": to}
                    if s == {k: V1_REFERENCE[k] for k in s} or (fs == "v1+extra" and not extra_ok):
                        continue
                    s["name"] = f"{m}|cw={cw}|{fs}|train={to}"
                    specs.append(s)
    return specs


def complexity(spec) -> int:
    return (spec["model"] == "ensemble_mean") + (spec["class_weight"] is not None) + \
           (spec["feature_set"] != "v1") + (spec["train_on"] != "apnea-ecg")


def rank(cands: list[dict]) -> dict:
    """Protocol rule on cross-dataset means; train_on is compared on union subject CV."""
    ref = next(c for c in cands if c["spec"]["name"] == "v1_reference")
    ref_mae = ref["result"]["cross_dataset"]["mean"]["apnea_min_mae"]
    rows = []
    for c in cands:
        m = c["result"]["cross_dataset"]["mean"]
        rows.append({"name": c["spec"]["name"], "xds_kappa": m["kappa_pooled"], "xds_mae": m["apnea_min_mae"],
                     "guard_ok": m["apnea_min_mae"] <= 1.10 * ref_mae, "complexity": complexity(c["spec"]),
                     "union_cv_kappa": c["result"]["subject_cv_union"]["all"]["metrics"]["kappa_pooled"]["point"],
                     "union_cv_mae": c["result"]["subject_cv_union"]["all"]["metrics"]["apnea_min_mae"]["point"]})
    ok = [r for r in rows if r["guard_ok"]]
    best = max(r["xds_kappa"] for r in ok)
    tied = [r for r in ok if best - r["xds_kappa"] < 0.01]
    pick = min(tied, key=lambda r: (r["complexity"], -r["xds_kappa"]))
    return {"v1_reference_xds_mae": ref_mae, "guard_mae_limit": 1.10 * ref_mae,
            "table": sorted(rows, key=lambda r: -r["xds_kappa"]),
            "pick_by_cross_dataset_rule": pick["name"],
            "note": "Cross-dataset folds do not depend on train_on (each side is one database), so candidates "
                    "differing only in train_on tie there; train_on is read from subject_cv_union."}


def cmd_select(bootstrap: int, v1_config: str) -> None:
    t0 = time.time()
    df, v1_cols, extra_cols = load_data()
    assert_training_only(df)
    print(f"loaded {len(df)} labeled minutes {df.groupby('dataset').size().to_dict()}, "
          f"{len(v1_cols)} v1 + {len(extra_cols)} extra features", flush=True)
    hp = vm.load_v1_hyperparameters(ROOT / v1_config)
    h = ApneaHarness(df, v1_cols, extra_cols, hp, bootstrap)
    check = h.v1_protocol_check()
    print(f"v1 protocol check identical={check['identical']}: {check['reproduced']}", flush=True)
    specs = default_grid(bool(extra_cols))
    cands = []
    for s in specs:
        r = h.run(s)
        cands.append({"spec": s, "result": r})
        m = r["cross_dataset"]["mean"]
        u = r["subject_cv_union"]["all"]["metrics"]
        print(f"{s['name']:<48} xds kappa {m['kappa_pooled']:.3f} MAE {m['apnea_min_mae']:.1f} | "
              f"union CV kappa {u['kappa_pooled']['point']:.3f} MAE {u['apnea_min_mae']['point']:.1f} "
              f"({r['seconds']}s, fits {h.n_fits})", flush=True)
    paths = [V1_APNEA, SLPDB_FEATURES] + ([EXTRA_MAIN, EXTRA_SLPDB] if extra_cols else [])
    out = {
        "protocol": "docs/v2_protocol.md (apnea)",
        "selection_metric": "cross_dataset.mean.kappa_pooled (minute kappa)",
        "guard": "cross_dataset.mean.apnea_min_mae not worse than v1_reference by > 10 %",
        "label_rule_slpdb": "somnibeat/apnea_labels.py (see results/v2/slpdb_apnea_check.json)",
        "datasets_loaded": sorted(df.dataset.unique()),
        "records": {d: sorted(df.loc[df.dataset == d, "record"].unique()) for d in sorted(df.dataset.unique())},
        "labeled_minutes": {d: {"n": int((df.dataset == d).sum()), "apnea": int(((df.dataset == d) & (df.y == 1)).sum())}
                            for d in sorted(df.dataset.unique())},
        "inputs_sha256": {str(p.relative_to(ROOT)): vm.sha256_file(p) for p in paths},
        "v1_config_sha256": hashlib.sha256((ROOT / v1_config).read_bytes()).hexdigest(),
        "hyperparameters": hp, "n_v1_features": len(v1_cols), "n_extra_features": len(extra_cols),
        "bootstrap_resamples": bootstrap,
        "v1_protocol_check": check,
        "ranking": rank(cands),
        "candidates": cands,
        "n_model_fits": h.n_fits, "fit_seconds": round(h.fit_seconds, 1),
        "total_seconds": round(time.time() - t0, 1),
    }
    SELECT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SELECT_OUT.write_text(json.dumps(drop_timing(out), indent=2))
    print(f"wrote {SELECT_OUT} ({out['total_seconds']}s); pick: {out['ranking']['pick_by_cross_dataset_rule']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "features", "select", "all"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--v1-config", default="config/default.toml")
    a = ap.parse_args()
    if a.command in ("check", "all"):
        cmd_check()
    if a.command in ("features", "all"):
        cmd_features(a.workers)
    if a.command in ("select", "all"):
        cmd_select(a.bootstrap, a.v1_config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
