"""v2 candidate models and the selection harness (docs/v2_protocol.md).

Only training databases are ever loaded here. UCDDB and HMC rows are dropped
while the CSV is read and `assert_training_only` stops the run if any row of
another dataset is present.

A candidate spec is a dict:
    {"name": str,
     "model": "gb" | "lr" | "ensemble_mean",       # ensemble_mean = mean of gb and lr probabilities
     "class_weight": None | "balanced",
     "smoothing": "none" | "hmm_viterbi" | "hmm_forward_backward",
     "feature_set": "v1" | "v1+extra"}
The v1 reference is {"model": "gb", "class_weight": None, "smoothing": "none",
"feature_set": "v1"} with hyperparameters from config/default.toml.

Folds
- cross-dataset A: fit on cap, evaluate on slpdb; B: fit on slpdb, evaluate on cap.
- subject CV: GroupKFold(5) over cap + slpdb by dataset/subject (as v1),
  out-of-fold prediction for every labeled epoch.
Models are always fitted on low_quality == 0 rows of the training side and
predict every labeled row of the evaluation side.

Binary decision threshold (when the output is a probability): chosen on
out-of-fold predictions of a GroupKFold(5) inside the training side only
(smoothed the same way, with transitions/prior from the inner training side).
Viterbi output is a hard path and needs no threshold. Multi-class: argmax.
"""

from __future__ import annotations

import hashlib
import time
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

from somnibeat import evaluate as ev
from somnibeat import smoothing as sm

ALLOWED_DATASETS = frozenset({"cap", "slpdb", "apnea-ecg"})
FORBIDDEN_DATASETS = frozenset({"ucddb", "hmc"})
KEYS = ["dataset", "record", "epoch"]
META = ["dataset", "record", "subject", "epoch", "minute", "epoch_start_s", "stage", "apnea", "y"]
TASKS = {
    "sleep_wake": {"W": "wake", "R": "sleep", "S1": "sleep", "S2": "sleep", "S3": "sleep", "S4": "sleep"},
    "wake_nrem_rem": {"W": "wake", "R": "rem", "S1": "nrem", "S2": "nrem", "S3": "nrem", "S4": "nrem"},
    "four_class": {"W": "wake", "R": "rem", "S1": "light", "S2": "light", "S3": "deep", "S4": "deep"},
}
LABELS = {"sleep_wake": ["sleep", "wake"], "wake_nrem_rem": ["wake", "nrem", "rem"],
          "four_class": ["wake", "light", "deep", "rem"]}
# Same list as scripts/train_eval.py (v1): logistic regression sees per-night normalized features only.
ABSOLUTE_PREFIXES = ("mean_rr_", "sdnn_", "rmssd_", "pnn50_", "cv_", "log_", "hf_peak", "dfa_")
MODELS = ("gb", "lr", "ensemble_mean")
CLASS_WEIGHTS = (None, "balanced")
SMOOTHING = ("none", "hmm_viterbi", "hmm_forward_backward")
FEATURE_SETS = ("v1", "v1+extra")
V1_REFERENCE = {"name": "v1_reference", "model": "gb", "class_weight": None, "smoothing": "none",
                "feature_set": "v1"}


# ----------------------------------------------------------------------------- data

def assert_training_only(df: pd.DataFrame) -> None:
    present = set(df["dataset"].unique())
    bad = present - ALLOWED_DATASETS
    assert not bad, f"non-training datasets present: {sorted(bad)}"
    assert not (present & FORBIDDEN_DATASETS), "test-set rows loaded"


def read_training_csv(path: Path, datasets: set[str], chunksize: int = 50000) -> pd.DataFrame:
    """Read a feature CSV keeping only rows of the given training datasets (filtered per chunk)."""
    datasets = set(datasets)
    assert datasets <= ALLOWED_DATASETS, f"not training datasets: {sorted(datasets - ALLOWED_DATASETS)}"
    parts = []
    for chunk in pd.read_csv(path, dtype={"stage": str}, keep_default_na=False,
                             na_values=["", "nan", "NaN"], chunksize=chunksize):
        parts.append(chunk[chunk["dataset"].isin(datasets)])
    df = pd.concat(parts, ignore_index=True)
    if "stage" in df.columns:
        df["stage"] = df["stage"].fillna("")
    assert_training_only(df)
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def load_stage_data(features_path: Path, extra_path: Path | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Returns (df, v1 feature columns, extra feature columns). Row order follows the v1 cache."""
    df = read_training_csv(features_path, {"cap", "slpdb"})
    v1_cols = [c for c in df.columns if c not in META]
    extra_cols: list[str] = []
    if extra_path is not None and Path(extra_path).exists():
        ex = read_training_csv(Path(extra_path), {"cap", "slpdb"})
        extra_cols = [c for c in ex.columns if c not in META and c not in v1_cols and c != "low_quality"]
        ex = ex[KEYS + extra_cols]
        assert not ex.duplicated(KEYS).any(), "duplicate keys in extra features"
        n = len(df)
        df = df.merge(ex, on=KEYS, how="left", validate="one_to_one")
        assert len(df) == n
    assert_training_only(df)
    return df, v1_cols, extra_cols


def task_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    labels = LABELS[task]
    out = df[df["stage"].isin(TASKS[task].keys())].copy()
    out["y"] = out["stage"].map(TASKS[task]).map({l: i for i, l in enumerate(labels)}).astype(int)
    return out.reset_index(drop=True)


def lr_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if not c.startswith(ABSOLUTE_PREFIXES)]


# ----------------------------------------------------------------------------- models

def load_v1_hyperparameters(path: str | Path = "config/default.toml") -> dict:
    cfg = tomllib.loads(Path(path).read_text())
    return {"seed": int(cfg["seed"]), "gb": dict(cfg["models"]["gb"]), "lr": dict(cfg["models"]["lr"]),
            "inner_folds": int(cfg["evaluation"]["inner_folds"])}


def make_model(base: str, hp: dict, seed: int, class_weight):
    """Identical to scripts/train_eval.make_model when class_weight is None."""
    if base == "gb":
        kw = dict(hp["gb"])
        if class_weight is not None:
            kw["class_weight"] = class_weight
        return HistGradientBoostingClassifier(random_state=seed, **kw)
    if base == "lr":
        kw = dict(hp["lr"])
        if class_weight is not None:
            kw["class_weight"] = class_weight
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(random_state=seed, **kw))
    raise ValueError(base)


def base_models(model: str) -> list[str]:
    return ["gb", "lr"] if model == "ensemble_mean" else [model]


def validate_spec(spec: dict, extra_available: bool) -> dict:
    s = {"class_weight": None, "smoothing": "none", "feature_set": "v1", **spec}
    assert s["model"] in MODELS, s
    assert s["class_weight"] in CLASS_WEIGHTS, s
    assert s["smoothing"] in SMOOTHING, s
    assert s["feature_set"] in FEATURE_SETS, s
    if "name" not in s:
        s["name"] = "{model}|cw={class_weight}|{smoothing}|{feature_set}".format(**s)
    s["skipped"] = s["feature_set"] == "v1+extra" and not extra_available
    return s


def default_grid(extra_available: bool) -> list[dict]:
    specs = [dict(V1_REFERENCE)]
    for m in MODELS:
        for cw in CLASS_WEIGHTS:
            for smo in SMOOTHING:
                for fs in FEATURE_SETS:
                    s = {"model": m, "class_weight": cw, "smoothing": smo, "feature_set": fs}
                    if s == {k: V1_REFERENCE[k] for k in s} or (fs == "v1+extra" and not extra_available):
                        continue
                    specs.append(s)
    return specs


# ----------------------------------------------------------------------------- folds

class Harness:
    """Caches raw probabilities per (task, base model, class weight, feature set, fold) so
    candidates that differ only in ensembling or smoothing reuse the same fits."""

    def __init__(self, df: pd.DataFrame, v1_cols: list[str], extra_cols: list[str], hp: dict,
                 n_resamples: int = 200, ci: float = 0.95, log=print):
        assert_training_only(df)
        self.df, self.v1_cols, self.extra_cols, self.hp = df, v1_cols, extra_cols, hp
        self.n_resamples, self.ci, self.log = n_resamples, ci, log
        self.seed = hp["seed"]
        self.n_inner = hp["inner_folds"]
        self.tasks: dict[str, dict] = {}
        self.cache: dict = {}
        self.n_fits = 0
        self.fit_seconds = 0.0

    # -- task setup
    def task(self, task: str) -> dict:
        if task in self.tasks:
            return self.tasks[task]
        t = task_frame(self.df, task)
        assert_training_only(t)
        groups = (t["dataset"] + "/" + t["subject"]).to_numpy()
        outer = {"A": (np.where(t.dataset == "cap")[0], np.where(t.dataset == "slpdb")[0]),
                 "B": (np.where(t.dataset == "slpdb")[0], np.where(t.dataset == "cap")[0])}
        gkf = GroupKFold(n_splits=self.n_inner)
        for k, (tr, va) in enumerate(gkf.split(t, t["y"].to_numpy(), groups)):
            outer[f"S{k}"] = (tr, va)
        self.tasks[task] = {"frame": t, "y": t["y"].to_numpy(), "groups": groups,
                            "good": t["low_quality"].to_numpy() == 0, "folds": outer,
                            "labels": LABELS[task]}
        return self.tasks[task]

    def columns(self, base: str, feature_set: str) -> list[str]:
        cols = self.v1_cols + (self.extra_cols if feature_set == "v1+extra" else [])
        return cols if base == "gb" else lr_columns(cols)

    def inner_splits(self, task: str, tr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        T = self.task(task)
        gkf = GroupKFold(n_splits=self.n_inner)
        return [(tr[a], tr[b]) for a, b in gkf.split(tr, T["y"][tr], T["groups"][tr])]

    # -- raw probabilities
    def raw_proba(self, task: str, base: str, cw, fs: str, fit_idx: np.ndarray, pred_idx: np.ndarray,
                  seed: int, key: tuple) -> np.ndarray:
        ck = (task, base, cw, fs) + key
        if ck in self.cache:
            return self.cache[ck]
        T = self.task(task)
        fit_idx = fit_idx[T["good"][fit_idx]]
        cols = self.columns(base, fs)
        X = T["frame"]
        t0 = time.time()
        m = make_model(base, self.hp, seed, cw)
        m.fit(X.iloc[fit_idx][cols], T["y"][fit_idx])
        p = m.predict_proba(X.iloc[pred_idx][cols])
        assert p.shape[1] == len(T["labels"])
        self.n_fits += 1
        self.fit_seconds += time.time() - t0
        self.cache[ck] = p
        return p

    def model_proba(self, task, spec, fit_idx, pred_idx, seed, key) -> np.ndarray:
        ps = [self.raw_proba(task, b, spec["class_weight"], spec["feature_set"], fit_idx, pred_idx, seed, key)
              for b in base_models(spec["model"])]
        return ps[0] if len(ps) == 1 else np.mean(ps, axis=0)

    # -- smoothing
    def smooth(self, task: str, spec: dict, proba: np.ndarray, fit_idx: np.ndarray,
               pred_idx: np.ndarray) -> np.ndarray:
        if spec["smoothing"] == "none":
            return proba
        T = self.task(task)
        K = len(T["labels"])
        fr = T["frame"]
        trans = transitions_for(fr.iloc[fit_idx], T["y"][fit_idx], K)
        fit_good = fit_idx[T["good"][fit_idx]]
        init = np.bincount(T["y"][fit_good], minlength=K) / len(fit_good)
        # The classifier's implicit prior: training frequency, or uniform under balanced weights.
        prior = init if spec["class_weight"] is None else np.full(K, 1.0 / K)
        out = np.zeros_like(proba)
        sub = fr.iloc[pred_idx]
        for _, ix in sub.groupby("record", sort=False).indices.items():
            ix = np.asarray(ix)
            order = np.argsort(sub["epoch"].to_numpy()[ix], kind="stable")
            ix = ix[order]
            out[ix] = sm.smooth_record(proba[ix], sub["epoch"].to_numpy()[ix], trans, prior, init,
                                       spec["smoothing"])
        return out

    def scores(self, task, spec, fit_idx, pred_idx, seed, key) -> np.ndarray:
        p = self.model_proba(task, spec, fit_idx, pred_idx, seed, key)
        return self.smooth(task, spec, p, fit_idx, pred_idx)

    def nested_threshold(self, task, spec, tr: np.ndarray, key) -> float:
        """Threshold on out-of-fold, smoothed predictions inside the training side only."""
        T = self.task(task)
        oof = np.full(len(tr), np.nan)
        pos = {int(r): i for i, r in enumerate(tr)}
        for j, (itr, iva) in enumerate(self.inner_splits(task, tr)):
            s = self.scores(task, spec, itr, iva, self.seed + 100 + j, key + ("inner", j))
            oof[[pos[int(r)] for r in iva]] = s[:, 1]
        assert not np.isnan(oof).any()
        return best_threshold(T["y"][tr], oof)

    # -- predictions for one outer fold
    def fold_predictions(self, task: str, spec: dict, fold: str) -> tuple[np.ndarray, np.ndarray, float | None]:
        T = self.task(task)
        tr, va = T["folds"][fold]
        seed = self.seed if fold in ("A", "B") else self.seed + int(fold[1:])
        s = self.scores(task, spec, tr, va, seed, (fold,))
        thr = None
        if len(T["labels"]) == 2:
            if spec["smoothing"] == "hmm_viterbi":
                pred = s.argmax(1)
            else:
                thr = self.nested_threshold(task, spec, tr, (fold,))
                pred = (s[:, 1] >= thr).astype(int)
        else:
            pred = s.argmax(1)
        return va, pred, thr

    # -- metrics
    def evaluate(self, task: str, idx: np.ndarray, pred: np.ndarray) -> dict:
        T = self.task(task)
        fr = T["frame"].iloc[idx]
        y = T["y"][idx]
        labels = T["labels"]
        lab = np.array(labels)
        recs = fr.groupby("record").indices
        ref = {r: y[i] for r, i in recs.items()}
        prd = {r: pred[i] for r, i in recs.items()}
        eix = {r: fr["epoch"].to_numpy()[i] for r, i in recs.items()}
        groups = {r: f"{d}/{s}" for r, d, s in fr[["record", "dataset", "subject"]].drop_duplicates().itertuples(index=False)}
        for r, i in recs.items():
            assert np.all(np.diff(eix[r]) > 0), "epochs out of order"
        args = dict(n_resamples=self.n_resamples, seed=self.seed, ci=self.ci)
        w = labels.index("wake")
        sw = ev.binary_suite({r: (v == w).astype(int) for r, v in ref.items()},
                             {r: (v == w).astype(int) for r, v in prd.items()},
                             **args, epoch_idx=eix, groups=groups)
        out = {"n_records": sw["n_records"], "n_epochs": sw["n_epochs"], "sleep_wake": sw["metrics"]}
        if len(labels) > 2:
            mc = ev.multiclass_suite({r: lab[v] for r, v in ref.items()}, {r: lab[v] for r, v in prd.items()},
                                     labels, **args, groups=groups)
            out["multiclass"] = mc["metrics"]
            out["confusion_matrix_rows_ref_cols_pred"] = mc["confusion_matrix_rows_ref_cols_pred"]
        return out

    def run_candidate(self, task: str, spec: dict, subject_cv: bool = True) -> dict:
        T = self.task(task)
        t0 = time.time()
        res = {"cross_dataset": {}, "thresholds": {}}
        for fold, name in (("A", "train_cap_eval_slpdb"), ("B", "train_slpdb_eval_cap")):
            va, pred, thr = self.fold_predictions(task, spec, fold)
            res["cross_dataset"][name] = self.evaluate(task, va, pred)
            res["thresholds"][fold] = thr
        res["cross_dataset"]["mean"] = mean_of_folds(res["cross_dataset"]["train_cap_eval_slpdb"],
                                                     res["cross_dataset"]["train_slpdb_eval_cap"])
        if subject_cv:
            pred_all = np.full(len(T["y"]), -1)
            for k in range(self.n_inner):
                va, pred, thr = self.fold_predictions(task, spec, f"S{k}")
                pred_all[va] = pred
                res["thresholds"][f"S{k}"] = thr
            assert (pred_all >= 0).all()
            res["subject_cv"] = self.evaluate(task, np.arange(len(T["y"])), pred_all)
            if spec.get("name") == V1_REFERENCE["name"] and len(T["labels"]) == 2:
                res["v1_protocol_check"] = self.v1_protocol_check(task, spec)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    def v1_protocol_check(self, task: str, spec: dict) -> dict:
        """v1's internal CV (threshold picked on the pooled out-of-fold predictions). Only used to
        confirm that the v1 reference is reproduced exactly; not a selection number."""
        T = self.task(task)
        oof = np.full(len(T["y"]), np.nan)
        for k in range(self.n_inner):
            tr, va = T["folds"][f"S{k}"]
            oof[va] = self.model_proba(task, spec, tr, va, self.seed + k, (f"S{k}",))[:, 1]
        thr = best_threshold(T["y"], oof)
        pred = (oof >= thr).astype(int)
        e = self.evaluate(task, np.arange(len(T["y"])), pred)
        return {"threshold_p_wake": thr,
                "kappa_pooled": e["sleep_wake"]["kappa_pooled"]["point"],
                "tst_mae_min": e["sleep_wake"]["tst_mae_min"]["point"]}


def transitions_for(frame: pd.DataFrame, y: np.ndarray, n_classes: int) -> np.ndarray:
    seqs, idxs = [], []
    ep = frame["epoch"].to_numpy()
    for _, ix in frame.groupby("record", sort=False).indices.items():
        ix = np.asarray(ix)
        ix = ix[np.argsort(ep[ix], kind="stable")]
        seqs.append(y[ix])
        idxs.append(ep[ix])
    return sm.estimate_transitions(seqs, idxs, n_classes, alpha=1.0)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Same grid and rule as scripts/train_eval.best_threshold (v1)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [ev.cohen_kappa(y, (p >= t).astype(int), labels=[0, 1]) for t in grid]
    return float(grid[int(np.nanargmax(scores))])


SUMMARY_KEYS = {"sleep_wake": ["kappa_pooled", "kappa_mean_per_record", "tst_mae_min", "tst_bias_min",
                               "se_mae_pctpt", "awakenings_mae", "wake_sensitivity", "sleep_sensitivity"],
                "multiclass": ["kappa_pooled", "accuracy"]}


def mean_of_folds(a: dict, b: dict) -> dict:
    out = {}
    for block, keys in SUMMARY_KEYS.items():
        if block in a:
            out[block] = {k: (a[block][k]["point"] + b[block][k]["point"]) / 2 for k in keys}
    return out
