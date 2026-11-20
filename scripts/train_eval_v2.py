"""Train the frozen v2 models and evaluate them once on the external test sets.

Spec: results/v2/v2_final_spec.json (chosen on training data only, see
docs/v2_protocol.md, docs/v2_selection.md, docs/v2_final_spec.md).

Test sets:
  HMC   (151 recordings, never used before)     primary v2 external test, staging only
  UCDDB (25 recordings, already used by v1)     second look, staging and apnea

Every test night is predicted the way the demo predicts it: every epoch on the
night's grid gets a probability, the whole night is smoothed with the HMM
(transitions and initial distribution from the training labels), and only
then are the epochs with an expert label scored. v1 is scored on HMC at the
same time, for a side-by-side comparison.

Data-quality rule (fixed in docs/v2_protocol.md before any HMC prediction):
primary analysis excludes records with < 50 % adequate-coverage epochs;
the analysis over all records is reported as secondary.

Modes:
  --final     predict the test sets. Refuses to run unless HEAD carries the
              tag v2-frozen and the working tree is clean.
  --dry-run   rehearsal on training data only: train on CAP, treat MIT-BIH
              slpdb as if it were a test set. Writes results/v2/dry_run/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import evaluate as ev  # noqa: E402
from somnibeat import smoothing as sm  # noqa: E402
from somnibeat import v2_models as vm  # noqa: E402

CACHE, V2 = ROOT / "data/cache", ROOT / "data/cache/v2"
STAGE_TRAIN = V2 / "stage_features_cap106_slpdb.csv.gz"
STAGE_TRAIN_EXTRA = V2 / "stage_features_extra_cap106.csv.gz"
STAGE_TEST = {"ucddb": (CACHE / "stage_features.csv.gz", V2 / "stage_features_extra_test.csv.gz"),
              "hmc": (V2 / "hmc_stage_features.csv.gz", V2 / "stage_features_extra_test.csv.gz")}
APNEA_FEAT, APNEA_EXTRA = CACHE / "apnea_features.csv.gz", V2 / "apnea_features_extra.csv.gz"
TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]
MIN_GOOD_FRACTION = 0.5


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path, keep: set[str]) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(path, dtype={"stage": str, "record": str, "subject": str}, keep_default_na=False,
                          na_values=["", "nan", "NaN"], chunksize=50000):
        parts.append(ch[ch["dataset"].isin(keep)])
    df = pd.concat(parts, ignore_index=True)
    if "stage" in df:
        df["stage"] = df["stage"].fillna("")
    return df


def join_extra(df: pd.DataFrame, extra: Path, keys: list[str], v1_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    ex = read(extra, set(df["dataset"].unique()))
    cols = [c for c in ex.columns if c not in vm.META and c not in v1_cols and c != "low_quality" and c not in keys]
    n = len(df)
    df = df.merge(ex[keys + cols], on=keys, how="left", validate="one_to_one")
    assert len(df) == n and not df[cols].isna().all(axis=1).any(), "extra features missing for some rows"
    return df, cols


# ----------------------------------------------------------------------------- staging

def train_staging(train: pd.DataFrame, v1_cols, extra_cols, spec, hp) -> dict:
    cols = v1_cols + (extra_cols if spec["feature_set"] == "v1+extra" else [])
    out = {}
    for task in TASKS:
        fr = vm.task_frame(train, task)
        y = fr["y"].to_numpy()
        good = fr["low_quality"].to_numpy() == 0
        K = len(vm.LABELS[task])
        models = []
        for base in vm.base_models(spec["model"]):
            m = vm.make_model(base, hp, hp["seed"], spec["class_weight"])
            c = cols if base == "gb" else vm.lr_columns(cols)
            m.fit(fr.loc[good, c], y[good])
            models.append((m, c))
        init = np.bincount(y[good], minlength=K) / good.sum()
        prior = init if spec["class_weight"] is None else np.full(K, 1.0 / K)
        out[task] = {"models": models, "labels": vm.LABELS[task], "trans": vm.transitions_for(fr, y, K),
                     "init": init, "prior": prior, "smoothing": spec["smoothing"]}
    return out


def predict_night(task_model: dict, night: pd.DataFrame) -> np.ndarray:
    night = night.sort_values("epoch")
    p = np.mean([m.predict_proba(night[c]) for m, c in task_model["models"]], axis=0)
    s = sm.smooth_record(p, night["epoch"].to_numpy(), task_model["trans"], task_model["prior"],
                         task_model["init"], task_model["smoothing"])
    return s.argmax(1)  # Viterbi output is one-hot; no threshold


def v1_predict_night(pk: dict, night: pd.DataFrame) -> np.ndarray:
    night = night.sort_values("epoch")
    p = pk["model"].predict_proba(night[pk["columns"]])
    if len(pk["labels"]) == 2:
        return (p[:, 1] >= pk["threshold"]).astype(int)
    return p.argmax(1)


def score_staging(test: pd.DataFrame, preds: dict[str, np.ndarray], task: str, records: list[str], ev_args) -> dict:
    """preds: record -> label index per epoch (sorted by epoch). Scores labeled epochs only."""
    labels = vm.LABELS[task]
    mapping = vm.TASKS[task]
    ref, pred, idx = {}, {}, {}
    for rid in records:
        night = test[test.record == rid].sort_values("epoch")
        lab = night["stage"].map(mapping)
        keep = lab.notna().to_numpy()
        ref[rid] = lab[keep].map({l: i for i, l in enumerate(labels)}).to_numpy().astype(int)
        pred[rid] = preds[rid][keep]
        idx[rid] = night["epoch"].to_numpy()[keep]
    if task == "sleep_wake":
        return ev.binary_suite(ref, pred, **ev_args, epoch_idx=idx)
    lab = np.array(labels)
    out = ev.multiclass_suite({k: lab[v] for k, v in ref.items()}, {k: lab[v] for k, v in pred.items()},
                              labels, **ev_args)
    w = labels.index("wake")
    sw = ev.binary_suite({k: (v == w).astype(int) for k, v in ref.items()},
                         {k: (v == w).astype(int) for k, v in pred.items()}, **ev_args, epoch_idx=idx)
    sw.pop("per_record")
    out["as_sleep_wake"] = sw
    return out


def baseline_preds(test: pd.DataFrame, task: str, cls: int, records) -> dict:
    return {r: np.full(int((test.record == r).sum()), cls) for r in records}


# ----------------------------------------------------------------------------- apnea

def apnea_model(train: pd.DataFrame, v1_cols, extra_cols, spec, hp) -> dict:
    cols = v1_cols + (extra_cols if spec["feature_set"] == "v1+extra" else [])
    y = train["apnea"].to_numpy().astype(int)
    good = train["low_quality"].to_numpy() == 0
    groups = (train["dataset"] + "/" + train["subject"]).to_numpy()

    def fit_predict(fit_idx, pred_idx, seed):
        fit_idx = fit_idx[good[fit_idx]]
        ps = []
        for base in vm.base_models(spec["model"]):
            c = cols if base == "gb" else vm.lr_columns(cols)
            m = vm.make_model(base, hp, seed, spec["class_weight"])
            m.fit(train.iloc[fit_idx][c], y[fit_idx])
            ps.append(m.predict_proba(train.iloc[pred_idx][c])[:, 1])
        return np.mean(ps, axis=0)

    oof = np.full(len(y), np.nan)
    for k, (a, b) in enumerate(GroupKFold(hp["inner_folds"]).split(train, y, groups)):
        oof[b] = fit_predict(a, b, hp["seed"] + 100 + k)
    thr = vm.best_threshold(y, oof)
    models = []
    for base in vm.base_models(spec["model"]):
        c = cols if base == "gb" else vm.lr_columns(cols)
        m = vm.make_model(base, hp, hp["seed"], spec["class_weight"])
        m.fit(train.loc[good, c], y[good])
        models.append((m, c))
    return {"models": models, "threshold": thr}


# ----------------------------------------------------------------------------- main

def check_frozen() -> str:
    tags = subprocess.run(["git", "tag", "--points-at", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.split()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=ROOT).stdout.strip()
    if "v2-frozen" not in tags or dirty:
        raise SystemExit("--final requires HEAD tagged v2-frozen and no uncommitted changes to tracked files.")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()


def good_fraction(test: pd.DataFrame) -> dict[str, float]:
    return {r: float((g["low_quality"] == 0).mean()) for r, g in test.groupby("record")}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=None)
    a = ap.parse_args()

    spec_all = json.loads((ROOT / "results/v2/v2_final_spec.json").read_text())
    spec, aspec = spec_all["staging"], spec_all["apnea"]
    hp = vm.load_v1_hyperparameters(ROOT / "config/default.toml")
    cfg = tomllib.loads((ROOT / "config/default.toml").read_text())
    n_boot = a.bootstrap or int(cfg["evaluation"]["bootstrap_resamples"])
    ev_args = dict(n_resamples=n_boot, seed=hp["seed"], ci=float(cfg["evaluation"]["ci"]))
    commit = check_frozen() if a.final else "dry-run"
    out_dir = ROOT / ("results/v2" if a.final else "results/v2/dry_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training data (never contains a test row)
    train_sets = {"cap", "slpdb"} if a.final else {"cap"}
    train = read(STAGE_TRAIN, train_sets)
    vm.assert_training_only(train)
    v1_cols = [c for c in train.columns if c not in vm.META]
    train, extra_cols = join_extra(train, STAGE_TRAIN_EXTRA, ["dataset", "record", "epoch"], v1_cols)
    staged = train_staging(train, v1_cols, extra_cols, spec, hp)

    if a.final:
        tests = {}
        for ds, (fpath, epath) in STAGE_TEST.items():
            t = read(fpath, {ds})
            t, _ = join_extra(t, epath, ["dataset", "record", "epoch"], v1_cols)
            tests[ds] = t
    else:
        t = read(STAGE_TRAIN, {"slpdb"})
        tests = {"slpdb_as_test": join_extra(t, STAGE_TRAIN_EXTRA, ["dataset", "record", "epoch"], v1_cols)[0]}

    v1_pk = {task: pickle.load(open(ROOT / f"results/models/{task}_gradient_boosting.pkl", "rb")) for task in TASKS}
    summary = {"commit": commit, "spec": spec_all, "test_sets": {}, "inputs": {
        "train_features_sha256": sha(STAGE_TRAIN), "train_extra_sha256": sha(STAGE_TRAIN_EXTRA)}}
    for ds, test in tests.items():
        gf = good_fraction(test)
        all_recs = sorted(gf)
        primary = [r for r in all_recs if gf[r] >= MIN_GOOD_FRACTION]
        excluded = [r for r in all_recs if gf[r] < MIN_GOOD_FRACTION]
        block = {"records_all": len(all_recs), "records_primary": len(primary),
                 "excluded_low_coverage": {r: round(gf[r], 3) for r in excluded}, "tasks": {}}
        for task in TASKS:
            v2p, v1p = {}, {}
            for rid in all_recs:
                night = test[test.record == rid]
                v2p[rid] = predict_night(staged[task], night)
                v1p[rid] = v1_predict_night(v1_pk[task], night)
            labels = vm.LABELS[task]
            sleep_like = 0 if task == "sleep_wake" else labels.index("light" if task == "four_class" else "nrem")
            tb = {}
            for name, recs in (("primary", primary), ("all_records", all_recs)):
                tb[name] = {"v2": score_staging(test, v2p, task, recs, ev_args),
                            "v1": score_staging(test, v1p, task, recs, ev_args),
                            "asleep_from_lights_off": score_staging(
                                test, baseline_preds(test, task, sleep_like, recs), task, recs, ev_args)}
                for k in ("v1", "asleep_from_lights_off"):
                    tb[name][k].pop("per_record", None)
            block["tasks"][task] = tb
            print(ds, task, "v2 kappa", round(tb["primary"]["v2"]["metrics"]["kappa_pooled"]["point"], 3),
                  "v1 kappa", round(tb["primary"]["v1"]["metrics"]["kappa_pooled"]["point"], 3), flush=True)
        summary["test_sets"][ds] = block

    # Apnea (UCDDB only; HMC has no respiratory events). Dry run: train on Apnea-ECG a*, test on b*/c*.
    if True:
        atr = read(APNEA_FEAT, {"apnea-ecg"})
        atr = atr[atr["apnea"] >= 0].reset_index(drop=True)
        if not a.final:
            ate = atr[~atr.record.str.startswith("a")].reset_index(drop=True)
            atr = atr[atr.record.str.startswith("a")].reset_index(drop=True)
        av1 = [c for c in atr.columns if c not in vm.META]
        atr, aextra = join_extra(atr, APNEA_EXTRA, ["dataset", "record", "minute"], av1)
        am = apnea_model(atr, av1, aextra, aspec, hp)
        if a.final:
            ate = read(APNEA_FEAT, {"ucddb"})
        ate, _ = join_extra(ate, APNEA_EXTRA, ["dataset", "record", "minute"], av1)
        gf = good_fraction(ate)
        v1a = pickle.load(open(ROOT / "results/models/apnea_gradient_boosting.pkl", "rb"))
        res = {}
        for name, recs in (("primary", [r for r in sorted(gf) if gf[r] >= MIN_GOOD_FRACTION]), ("all_records", sorted(gf))):
            ref, p2, p1 = {}, {}, {}
            for rid in recs:
                m = ate[(ate.record == rid)].sort_values("minute")
                keep = (m["apnea"] >= 0).to_numpy()
                ref[rid] = m["apnea"].to_numpy()[keep].astype(int)
                pr = np.mean([mm.predict_proba(m[c])[:, 1] for mm, c in am["models"]], axis=0)
                p2[rid] = (pr >= am["threshold"]).astype(int)[keep]
                p1[rid] = (v1a["model"].predict_proba(m[v1a["columns"]])[:, 1] >= v1a["threshold"]).astype(int)[keep]
            res[name] = {"v2": ev.count_suite(ref, p2, 1.0, **ev_args), "v1": ev.count_suite(ref, p1, 1.0, **ev_args),
                         "no_apnea": ev.count_suite(ref, {k: np.zeros_like(v) for k, v in ref.items()}, 1.0, **ev_args)}
            for k in ("v1", "no_apnea"):
                res[name][k].pop("per_record", None)
        summary["apnea_ucddb" if a.final else "apnea_dry_run"] = {"threshold_p_apnea": am["threshold"], **res}
        print("apnea v2 kappa", round(res["primary"]["v2"]["metrics"]["kappa_pooled"]["point"], 3))
        mdir = out_dir / "models"
        mdir.mkdir(exist_ok=True)
        with open(mdir / "apnea.pkl", "wb") as f:
            pickle.dump(am, f)

    mdir = out_dir / "models"
    mdir.mkdir(exist_ok=True)
    for task, tm in staged.items():
        with open(mdir / f"{task}.pkl", "wb") as f:
            pickle.dump(tm, f)
    summary["note"] = ("UCDDB is a second look: v1 was already evaluated on it. HMC is the untouched v2 test set."
                       if a.final else "Dry run on training data: trained on CAP only, slpdb treated as a test set.")
    (out_dir / "test_results.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out_dir / "test_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
