"""Parity: web/js/model_v2.js and web/js/smoothing.js (run in Node) vs the v2 Python models.

Training datasets only (CAP, MIT-BIH slpdb, Apnea-ECG); no test-set row is read.
- ~1500 staging rows from the v2 training cache (v1 + extra features, CAP and slpdb),
  including rows with missing features: gradient boosting probabilities within 1e-9.
- ~500 Apnea-ECG minute rows (v1 + extra apnea features), including rows with missing
  features: gradient boosting and logistic probabilities within 1e-9, identical decisions.
- Every whole slpdb night: Viterbi paths of the full JS pipeline identical to
  somnibeat.smoothing.smooth_record on the Python probabilities, for the three staging tasks;
  JS smoothRecord fed the Python probabilities returns exactly the Python output.
- The tiny HMM examples of tests/test_v2_select.py (plus ties and gaps) through smoothing.js,
  and the Node unit tests in tests/parity/smoothing_unit.mjs.

Models: results/v2/models when all four pickles exist, else results/v2/dry_run/models
(override with SOMNIBEAT_V2_MODELS). models_v2.json is exported into a temporary directory
from the same pickles, so the test does not depend on the state of web/data.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from somnibeat import smoothing as sm  # noqa: E402

CLI = ROOT / "tests" / "parity" / "model_v2_cli.mjs"
UNIT = ROOT / "tests" / "parity" / "smoothing_unit.mjs"
WEB_JSON = ROOT / "web" / "data" / "models_v2.json"
STAGE_TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]
ALL_TASKS = STAGE_TASKS + ["apnea"]
N_STAGE, N_APNEA, N_NAN = 1500, 500, 150
TOL = 1e-9

V2 = ROOT / "data" / "cache" / "v2"
STAGE_FEAT, STAGE_EXTRA = V2 / "stage_features_cap106_slpdb.csv.gz", V2 / "stage_features_extra_cap106.csv.gz"
APNEA_FEAT, APNEA_EXTRA = ROOT / "data" / "cache" / "apnea_features.csv.gz", V2 / "apnea_features_extra.csv.gz"
TRAIN_SETS = {"cap", "slpdb"}


def models_dir() -> Path:
    env = os.environ.get("SOMNIBEAT_V2_MODELS")
    if env:
        return Path(env) if Path(env).is_absolute() else ROOT / env
    final = ROOT / "results" / "v2" / "models"
    if all((final / f"{t}.pkl").exists() for t in ALL_TASKS):
        return final
    return ROOT / "results" / "v2" / "dry_run" / "models"


MODELS = models_dir()
node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
have_models = pytest.mark.skipif(not all((MODELS / f"{t}.pkl").exists() for t in ALL_TASKS),
                                 reason="v2 pickles missing")
have_data = pytest.mark.skipif(not all(p.exists() for p in (STAGE_FEAT, STAGE_EXTRA, APNEA_FEAT, APNEA_EXTRA)),
                               reason="v2 feature caches missing")


def load(task: str) -> dict:
    with open(MODELS / f"{task}.pkl", "rb") as f:
        return pickle.load(f)


def train_script():
    """scripts/train_eval_v2.py as a module (read / join_extra only; main() is not run)."""
    spec = importlib.util.spec_from_file_location("train_eval_v2_for_parity", ROOT / "scripts" / "train_eval_v2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def union(models) -> list[str]:
    out: list[str] = []
    for _, cols in models:
        out += [c for c in cols if c not in out]
    return out


def pick_rows(df, cols, n, n_nan, seed):
    rng = np.random.default_rng(seed)
    has_nan = df[cols].isna().any(axis=1).to_numpy()
    nan_idx = rng.choice(np.flatnonzero(has_nan), size=min(n_nan, int(has_nan.sum())), replace=False)
    rest = rng.choice(np.flatnonzero(~has_nan), size=n - len(nan_idx), replace=False)
    return df.iloc[np.sort(np.concatenate([nan_idx, rest]))]


def col_json(values) -> list:
    return [None if math.isnan(v) else float(v) for v in np.asarray(values, dtype=np.float64)]


def run_cli(models_json: Path, payload: dict, tmp: Path) -> dict:
    (tmp / "in.json").write_text(json.dumps(payload))
    subprocess.run(["node", str(CLI), str(models_json), str(tmp / "in.json"), str(tmp / "out.json")],
                   check=True, cwd=ROOT, timeout=900)
    return json.loads((tmp / "out.json").read_text())


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    import export_web_v2

    out = tmp_path_factory.mktemp("models_v2") / "models_v2.json"
    export_web_v2.export(MODELS, ROOT / "config" / "v2.toml", ROOT / "results" / "v2" / "v2_final_spec.json", out)
    return out


@pytest.fixture(scope="module")
def parity(exported, tmp_path_factory):
    te = train_script()
    bundles = {t: load(t) for t in ALL_TASKS}

    stage = te.read(STAGE_FEAT, TRAIN_SETS)
    assert set(stage["dataset"].unique()) <= TRAIN_SETS
    v1_cols = [c for c in stage.columns if c not in te.vm.META]
    stage, _ = te.join_extra(stage, STAGE_EXTRA, ["dataset", "record", "epoch"], v1_cols)
    scols = union(bundles["sleep_wake"]["models"])
    for t in STAGE_TASKS:
        assert union(bundles[t]["models"]) == scols
    rows = pick_rows(stage.reset_index(drop=True), scols, N_STAGE, N_NAN, 1)

    apnea = te.read(APNEA_FEAT, {"apnea-ecg"})
    apnea = apnea[apnea["apnea"] >= 0].reset_index(drop=True)
    av1 = [c for c in apnea.columns if c not in te.vm.META]
    apnea, _ = te.join_extra(apnea, APNEA_EXTRA, ["dataset", "record", "minute"], av1)
    acols = union(bundles["apnea"]["models"])
    arows = pick_rows(apnea, acols, N_APNEA, N_NAN // 3, 2)

    expected = {"rows": {}, "nights": {t: [] for t in STAGE_TASKS}, "smooth": []}
    payload = {"rows": {}, "nights": {"tasks": STAGE_TASKS, "list": []}, "smooth": []}
    for t in STAGE_TASKS:
        parts = [m.predict_proba(rows[c]) for m, c in bundles[t]["models"]]
        expected["rows"][t] = {"parts": parts, "proba": np.mean(parts, axis=0),
                               "n_nan": int(rows[scols].isna().any(axis=1).sum())}
        payload["rows"][t] = {"n": len(rows), "columns": {c: col_json(rows[c]) for c in scols}}
    am = bundles["apnea"]
    aparts = [m.predict_proba(arows[c])[:, 1] for m, c in am["models"]]
    ap = np.mean(aparts, axis=0)
    expected["rows"]["apnea"] = {"parts": aparts, "p": ap, "labels": (ap >= am["threshold"]).astype(int),
                                 "n_nan": int(arows[acols].isna().any(axis=1).sum()),
                                 "n_nan_lr": int(arows[am["models"][-1][1]].isna().any(axis=1).sum())}
    payload["rows"]["apnea"] = {"n": len(arows), "columns": {c: col_json(arows[c]) for c in acols}}

    # Whole slpdb nights, as predict_night in scripts/train_eval_v2.py.
    slp = stage[stage.dataset == "slpdb"]
    records = sorted(slp.record.unique())
    for rid in records:
        night = slp[slp.record == rid].sort_values("epoch")
        ix = night["epoch"].to_numpy()
        payload["nights"]["list"].append({"n": len(night), "epoch": [int(v) for v in ix],
                                          "columns": {c: col_json(night[c]) for c in scols}})
        for t in STAGE_TASKS:
            b = bundles[t]
            p = np.mean([m.predict_proba(night[c]) for m, c in b["models"]], axis=0)
            s = sm.smooth_record(p, ix, b["trans"], b["prior"], b["init"], b["smoothing"])
            expected["nights"][t].append({"proba": p, "labels": s.argmax(1), "gaps": len(sm.segments(ix)) - 1})
            # Smoothing alone, fed the Python probabilities.
            payload["smooth"].append({"proba": p.ravel().tolist(), "epoch": [int(v) for v in ix],
                                      "trans": b["trans"].tolist(), "prior": b["prior"].tolist(),
                                      "init": b["init"].tolist(), "method": b["smoothing"]})
            expected["smooth"].append(s)

    # Tiny examples (tests/test_v2_select.py), exact ties and gaps.
    rng = np.random.default_rng(7)
    tiny = [
        (np.array([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]]), np.arange(3), np.array([[0.9, 0.1], [0.1, 0.9]]),
         np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]]), np.arange(3), np.full((2, 2), 0.5),
         np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.array([[0.8, 0.2], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]]), np.array([0, 1, 2, 10]),
         np.array([[0.99, 0.01], [0.01, 0.99]]), np.array([0.5, 0.5]), np.array([0.5, 0.5])),
        (np.full((5, 3), 1 / 3), np.arange(5), np.full((3, 3), 1 / 3), np.full(3, 1 / 3), np.full(3, 1 / 3)),
        (np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0], [1e-12, 0.3, 0.7]]), np.array([3, 4, 6]),
         np.full((3, 3), 1 / 3), np.full(3, 1 / 3), np.array([0.2, 0.3, 0.5])),
    ]
    for _ in range(20):
        k, n = 3, 8
        tiny.append((rng.dirichlet(np.ones(k), size=n), np.cumsum(rng.integers(1, 3, n)),
                     rng.dirichlet(np.ones(k), size=k), rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))))
    for p, ix, trans, prior, init in tiny:
        for method in ("hmm_viterbi", "hmm_forward_backward", "none"):
            payload["smooth"].append({"proba": p.ravel().tolist(), "epoch": [int(v) for v in ix],
                                      "trans": trans.tolist(), "prior": prior.tolist(), "init": init.tolist(),
                                      "method": method})
            expected["smooth"].append(sm.smooth_record(p, ix, trans, prior, init, method))
            expected.setdefault("smooth_method", []).append(method)

    got = run_cli(exported, payload, tmp_path_factory.mktemp("parity_model_v2"))
    return expected, got, records


@node
@have_models
@have_data
@pytest.mark.parametrize("task", STAGE_TASKS)
def test_staging_probabilities_match(parity, task):
    expected, got, _ = parity
    e = expected["rows"][task]
    assert e["n_nan"] > 0, "the sample must include rows with missing features"
    for i, part in enumerate(e["parts"]):
        js = np.asarray(got["rows"][task]["parts"][i]).reshape(part.shape)
        diff = float(np.max(np.abs(js - part)))
        print(f"{task}[{i}]: rows={len(part)} nan_rows={e['n_nan']} max_abs_diff={diff:.3e}")
        assert diff < TOL
    js = np.asarray(got["rows"][task]["proba"]).reshape(e["proba"].shape)
    assert float(np.max(np.abs(js - e["proba"]))) < TOL


@node
@have_models
@have_data
def test_apnea_probabilities_and_decisions_match(parity):
    expected, got, _ = parity
    e, g = expected["rows"]["apnea"], got["rows"]["apnea"]
    assert e["n_nan"] > 0 and e["n_nan_lr"] > 0, "the sample must include rows with missing features"
    for i, part in enumerate(e["parts"]):
        diff = float(np.max(np.abs(np.asarray(g["parts"][i]) - part)))
        print(f"apnea[{i}]: rows={len(part)} nan_rows={e['n_nan']} max_abs_diff={diff:.3e}")
        assert diff < TOL
    assert float(np.max(np.abs(np.asarray(g["p"]) - e["p"]))) < TOL
    labels = np.asarray(g["labels"])
    assert np.array_equal(labels, e["labels"]), f"{int((labels != e['labels']).sum())} apnea decisions differ"


@node
@have_models
@have_data
@pytest.mark.parametrize("task", STAGE_TASKS)
def test_viterbi_paths_on_whole_slpdb_nights(parity, task):
    expected, got, records = parity
    assert len(records) >= 10
    n_epochs = n_gaps = 0
    for rid, e, g in zip(records, expected["nights"][task], got["nights"][task]):
        js_p = np.asarray(g["proba"]).reshape(e["proba"].shape)
        assert float(np.max(np.abs(js_p - e["proba"]))) < TOL, rid
        labels = np.asarray(g["labels"])
        assert np.array_equal(labels, e["labels"]), f"{task} {rid}: {int((labels != e['labels']).sum())} epochs differ"
        n_epochs += len(labels)
        n_gaps += e["gaps"]
    print(f"{task}: {len(records)} slpdb nights, {n_epochs} epochs, {n_gaps} gaps: identical Viterbi paths")


@node
@have_models
@have_data
def test_smoothing_exact_on_python_probabilities(parity):
    expected, got, _ = parity
    methods = expected["smooth_method"]
    n_tiny = len(methods)
    n_night = len(expected["smooth"]) - n_tiny
    for i, (e, g) in enumerate(zip(expected["smooth"], got["smooth"])):
        js = np.asarray(g).reshape(e.shape)
        method = "hmm_viterbi" if i < n_night else methods[i - n_night]
        if method == "hmm_forward_backward":
            assert float(np.max(np.abs(js - e))) < 1e-12, i
        else:
            assert np.array_equal(js, e), f"case {i} ({method}) differs"


@node
def test_smoothing_js_unit_tests():
    r = subprocess.run(["node", "--test", str(UNIT)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def check_against_pickles(m: dict, mdir: Path) -> None:

    for task in ALL_TASKS:
        with open(mdir / f"{task}.pkl", "rb") as f:
            b = pickle.load(f)
        entry = m["tasks"][task]
        assert f"{task}.pkl" in m["source"]["pickles"]
        assert entry["ensemble"] == "mean" and len(entry["models"]) == len(b["models"])
        for (est, cols), me in zip(b["models"], entry["models"]):
            assert me["columns"] == list(cols)
            if me["kind"] == "hgb":
                assert me["baseline"] == [float(v) for v in np.ravel(est._baseline_prediction)]
                assert len(me["trees"]) == len(est._predictors)
            else:
                imp, sc, lr = (s for _, s in est.steps)
                assert me["impute"] == [float(v) for v in imp.statistics_]
                assert me["mean"] == [float(v) for v in sc.mean_]
                assert me["scale"] == [float(v) for v in sc.scale_]
                assert me["coef"] == [float(v) for v in lr.coef_[0]]
                assert me["intercept"] == float(lr.intercept_[0])
        if task == "apnea":
            assert entry["labels"] == ["normal", "apnea"]
            assert entry["decision"] == {"type": "threshold", "positive": "apnea", "threshold": b["threshold"]}
        else:
            assert entry["labels"] == list(b["labels"])
            assert entry["decision"] == {"type": b["smoothing"]}
            h = entry["hmm"]
            assert h["method"] == b["smoothing"]
            assert h["trans"] == b["trans"].tolist()
            assert h["init"] == b["init"].tolist()
            assert h["prior"] == b["prior"].tolist()
    spec = json.loads((ROOT / "results" / "v2" / "v2_final_spec.json").read_text())
    assert m["spec"] == spec


@have_models
def test_export_is_exact_and_deterministic(exported, tmp_path):
    import export_web_v2

    m = json.loads(exported.read_text())
    check_against_pickles(m, MODELS)
    again = tmp_path / "again.json"
    export_web_v2.export(MODELS, ROOT / "config" / "v2.toml", ROOT / "results" / "v2" / "v2_final_spec.json", again)
    assert again.read_bytes() == exported.read_bytes()


@pytest.mark.skipif(not WEB_JSON.exists(), reason="web/data/models_v2.json not exported yet")
def test_web_models_v2_json_matches_its_pickles():
    m = json.loads(WEB_JSON.read_text())
    mdir = ROOT / m["source"]["models_dir"]
    final = ROOT / "results" / "v2" / "models"
    if m["source"]["dry_run"] and all((final / f"{t}.pkl").exists() for t in ALL_TASKS):
        pytest.fail("web/data/models_v2.json was exported from the dry-run models but the final models exist: "
                    "run scripts/export_web_v2.py")
    if not all((mdir / f"{t}.pkl").exists() for t in ALL_TASKS):
        # A fresh clone has the committed web model but not the (git-ignored) trained pickles;
        # `make reproduce` runs this test again after `make v2` has rebuilt them.
        pytest.skip(f"trained v2 models not present in {mdir.relative_to(ROOT)} yet (run `make v2`)")
    check_against_pickles(m, mdir)
