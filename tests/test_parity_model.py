"""Parity: web/js/model.js (run in Node) vs sklearn predict_proba on cached features.

Uses ~2000 rows: UCDDB staging rows (including rows with missing features)
for the three staging tasks and Apnea-ECG + UCDDB minute rows for apnea.
Requires web/data/models.json (scripts/export_web.py), the feature cache and
the trained pickles; skipped when the local caches are absent.
"""

from __future__ import annotations

import json
import math
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS_JSON = ROOT / "web" / "data" / "models.json"
CLI = ROOT / "tests" / "parity" / "model_cli.mjs"
STAGE_TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]
N_STAGE, N_APNEA, N_NAN = 1500, 500, 150
TOL = 1e-9

needed = [MODELS_JSON, ROOT / "data/cache/stage_features.csv.gz", ROOT / "data/cache/apnea_features.csv.gz",
          *[ROOT / f"results/models/{t}_gradient_boosting.pkl" for t in STAGE_TASKS + ["apnea"]]]
pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed"),
    pytest.mark.skipif(not all(p.exists() for p in needed), reason="local caches or models.json missing"),
]


def pick_rows(df: pd.DataFrame, cols: list[str], n: int, n_nan: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    has_nan = df[cols].isna().any(axis=1).to_numpy()
    nan_idx = rng.choice(np.flatnonzero(has_nan), size=min(n_nan, int(has_nan.sum())), replace=False)
    rest = rng.choice(np.flatnonzero(~has_nan), size=n - len(nan_idx), replace=False)
    return df.iloc[np.sort(np.concatenate([nan_idx, rest]))]


def load_bundle(task: str) -> dict:
    with open(ROOT / "results" / "models" / f"{task}_gradient_boosting.pkl", "rb") as f:
        return pickle.load(f)


def python_decision(bundle: dict, proba: np.ndarray) -> np.ndarray:
    if len(bundle["labels"]) == 2:
        return (proba[:, 1] >= bundle["threshold"]).astype(int)
    return np.argmax(proba, axis=1)


def as_json_column(values: np.ndarray) -> list:
    return [None if math.isnan(v) else float(v) for v in values]


@pytest.fixture(scope="module")
def parity(tmp_path_factory):
    bundles = {t: load_bundle(t) for t in STAGE_TASKS + ["apnea"]}
    cols = bundles["sleep_wake"]["columns"]
    stage = pd.read_csv(ROOT / "data/cache/stage_features.csv.gz")
    stage = pick_rows(stage[stage.dataset == "ucddb"].reset_index(drop=True), cols, N_STAGE, N_NAN, 1)
    apnea = pick_rows(pd.read_csv(ROOT / "data/cache/apnea_features.csv.gz"), bundles["apnea"]["columns"],
                      N_APNEA, N_NAN // 3, 2)
    frames = {**{t: stage for t in STAGE_TASKS}, "apnea": apnea}

    payload, expected = {}, {}
    for task, b in bundles.items():
        X = frames[task][b["columns"]]
        proba = b["model"].predict_proba(X)
        expected[task] = (proba, python_decision(b, proba), int(X.isna().any(axis=1).sum()))
        payload[task] = {"n": len(X), "columns": {c: as_json_column(X[c].to_numpy(np.float64)) for c in b["columns"]}}

    tmp = tmp_path_factory.mktemp("parity_model")
    (tmp / "in.json").write_text(json.dumps(payload))
    subprocess.run(["node", str(CLI), str(MODELS_JSON), str(tmp / "in.json"), str(tmp / "out.json")],
                   check=True, cwd=ROOT, timeout=600)
    got = json.loads((tmp / "out.json").read_text())
    return expected, got


@pytest.mark.parametrize("task", STAGE_TASKS + ["apnea"])
def test_probabilities_and_decisions_match(parity, task):
    expected, got = parity
    proba, labels, n_nan = expected[task]
    assert n_nan > 0, "the sample must include rows with missing features"
    js = np.asarray(got[task]["proba"], dtype=np.float64).reshape(proba.shape)
    diff = float(np.max(np.abs(js - proba)))
    print(f"{task}: rows={len(labels)} nan_rows={n_nan} max_abs_diff={diff:.3e}")
    assert diff < TOL, f"{task}: max |p_js - p_py| = {diff:.3e}"
    js_labels = np.asarray(got[task]["labels"])
    assert np.array_equal(js_labels, labels), f"{task}: {int((js_labels != labels).sum())} decisions differ"


def test_models_json_matches_pickles():
    models = json.loads(MODELS_JSON.read_text())
    for task in STAGE_TASKS + ["apnea"]:
        b, entry = load_bundle(task), models["tasks"][task]
        assert entry["columns"] == list(b["columns"])
        assert entry["labels"] == list(b["labels"])
        assert entry["baseline"] == [float(v) for v in np.ravel(b["model"]._baseline_prediction)]
        assert len(entry["trees"]) == len(b["model"]._predictors)
        if len(b["labels"]) == 2:
            assert entry["decision"] == {"type": "threshold", "positive": b["labels"][1], "threshold": b["threshold"]}
        else:
            assert entry["decision"] == {"type": "argmax"}
