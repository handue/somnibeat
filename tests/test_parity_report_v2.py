"""Python/JavaScript parity for the v2 night report (somnibeat/report_v2.py vs web/js/report_v2.js).

For each sample CSV in web/data/samples/ (and once with lights-off/on offsets) the v2 report
is built in Python and in Node (tests/parity/report_v2_cli.mjs) and compared: identical grid,
per-epoch labels for every staging task, low-quality flags, apnea labels and warnings; night
metrics and quality numbers within 1e-9; model probabilities within 1e-9, except rows whose
gradient-boosting traversal passes a split threshold within 1e-9 (last-bit feature differences may
send them either way; same allowance as tests/test_parity_report.py).

Also checks:
- the pure-Python v2 inference (exported JSON) against the trained pickles the JSON was
  exported from (sklearn predict_proba + somnibeat.smoothing.smooth_record): identical labels;
- error codes for unusable grids; the worker protocol (v2 default, explicit v1, fallback to v1
  with a warning when models_v2.json is unavailable, lazy fetch and caching);
- the Node run time of a full v2 report for the largest sample.

Only predictions are compared between implementations; no reference labels are read.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from somnibeat import report as REP
from somnibeat import report_v2 as REP2
from somnibeat import smoothing as SM

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tests" / "parity" / "report_v2_cli.mjs"
MODELS_V2_JSON = ROOT / "web" / "data" / "models_v2.json"
SAMPLES = sorted((ROOT / "web" / "data" / "samples").glob("*.csv"))
STAGE_TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]
TOL = 1e-9
PICKLE_TOL = 1e-12
MAX_NODE_S = 20.0
NODE = shutil.which("node")
LIGHTS_SAMPLE = "ucddb018"

pytestmark = [
    pytest.mark.skipif(NODE is None, reason="node is not installed"),
    pytest.mark.skipif(not MODELS_V2_JSON.exists() or len(SAMPLES) < 3,
                       reason="web/data is not exported (scripts/export_web_v2.py)"),
]


@pytest.fixture(scope="module")
def models() -> dict:
    return json.loads(MODELS_V2_JSON.read_text())


def read_rr_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # csv + float() rounds decimal strings exactly like JavaScript Number().
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["time_s"]) for r in rows]), np.array([float(r["rr_ms"]) for r in rows])


def lights_case(t: np.ndarray) -> tuple[float, float]:
    return 1234.5, float(t[-1]) - 1800.25


_PY_CACHE: dict = {}


def py_report(path: Path, models: dict, lights_off=None, lights_on=None) -> tuple[dict, dict]:
    """(report with probabilities, {epoch length: feature columns}); cached per input."""
    key = (path.name, lights_off, lights_on)
    if key not in _PY_CACHE:
        t, rr = read_rr_csv(path)
        feats = {}
        orig = REP2.night_features

        def recording(t_, rr_, valid, starts, epoch_len, cfg):
            out = orig(t_, rr_, valid, starts, epoch_len, cfg)
            feats[float(epoch_len)] = out
            return out

        REP2.night_features = recording
        try:
            rep = REP2.build_report_v2(t, rr, models, lights_off_s=lights_off, lights_on_s=lights_on,
                                       file_name=path.name, include_proba=True)
        finally:
            REP2.night_features = orig
        _PY_CACHE[key] = (rep, feats)
    return _PY_CACHE[key]


def run_node(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    res = subprocess.run([NODE, str(CLI), *args], capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    return res


def js_report(path: Path, out: Path, lights_off=None, lights_on=None) -> dict:
    args = [str(path), str(out)]
    if lights_off is not None:
        args += ["--lights-off", repr(float(lights_off))]
    if lights_on is not None:
        args += ["--lights-on", repr(float(lights_on))]
    run_node(*args)
    return json.loads(out.read_text())


def close(a, b, tol: float = TOL) -> bool:
    a = math.nan if a is None else float(a)  # JSON writes NaN as null
    b = math.nan if b is None else float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol * max(abs(a), abs(b), 1.0)


def split_tie_rows(task: dict, cols_by_name: dict, rows) -> set:
    """Rows whose traversal of any gradient-boosting model of `task` passes a split with
    |x - threshold| <= TOL * max(|x|, 1) (same rule as tests/test_parity_report.py).

    A feature that sits on a split threshold (possible when the sample was part of the training
    data, whose values define the bin thresholds; e.g. the Apnea-ECG sample for the apnea model)
    can go either way under last-bit differences between numpy/libm and V8, so that row's
    probability may legitimately differ by one leaf value. Every other row must agree within TOL.
    """
    ties = set()
    for model in task["models"]:
        if model["kind"] != "hgb":
            continue
        cols = [np.asarray(cols_by_name[c], dtype=float) for c in model["columns"]]
        for row in rows:
            for iteration in model["trees"]:
                for tree in iteration:
                    i = 0
                    while not tree["leaf"][i]:
                        x, th = cols[tree["f"][i]][row], tree["t"][i]
                        if math.isnan(x):
                            i = tree["l"][i] if tree["m"][i] else tree["r"][i]
                            continue
                        if abs(x - th) <= TOL * max(abs(x), 1.0):
                            ties.add(row)
                        i = tree["l"][i] if x <= th else tree["r"][i]
    return ties


def compare_reports(py: dict, js: dict, models: dict | None = None, feats: dict | None = None) -> float:
    """Asserts equality of everything but probabilities; returns the largest probability difference
    outside rows that sit on a split threshold (see split_tie_rows; needs models and the Python
    feature columns {epoch length: columns})."""
    assert py["model"] == js["model"] == "v2"
    assert set(py) == set(js)
    assert py["fileName"] == js["fileName"]
    for k in ("start_s", "end_s", "epoch_s", "n_epochs"):
        assert py["grid"][k] == js["grid"][k], f"grid {k}: {py['grid'][k]} vs {js['grid'][k]}"
    for task in STAGE_TASKS:
        p, j = py["stages"][task], js["stages"][task]
        assert len(p) == len(j) == py["grid"]["n_epochs"]
        diff = [i for i, (a, b) in enumerate(zip(p, j)) if a != b]
        assert not diff, f"{task}: {len(diff)} labels differ, first at {diff[:5]}"
    assert py["lowQuality"] == js["lowQuality"]
    assert py["apnea"]["minute_s"] == js["apnea"]["minute_s"]
    assert py["apnea"]["labels"] == js["apnea"]["labels"]
    assert set(py["metrics"]) == set(js["metrics"]) == {"tib_min", "tst_min", "se_pct", "awakenings", "apnea_min"}
    for k, v in py["metrics"].items():
        assert close(v, js["metrics"][k]), f"metric {k}: {v} vs {js['metrics'][k]}"
    assert py["metrics"]["awakenings"] == js["metrics"]["awakenings"]
    q, jq = py["quality"], js["quality"]
    assert set(q) == set(jq)
    for k in ("valid_fraction", "invalid_pct", "n_beats", "duration_h"):
        assert close(q[k], jq[k]), f"quality {k}: {q[k]} vs {jq[k]}"
    assert q["warnings"] == jq["warnings"]
    worst = 0.0
    for task in STAGE_TASKS + ["apnea"]:
        p = np.asarray(py["proba"][task], dtype=float)
        j = np.asarray(js["proba"][task], dtype=float).reshape(p.shape)
        d = np.abs(p - j).reshape(len(p), -1).max(axis=1) if p.size else np.zeros(0)
        off = np.flatnonzero(d > TOL)
        if len(off):
            assert models is not None and feats is not None, f"{task}: max |p_py - p_js| = {d.max():.3e}"
            L = float(models["config"]["apnea_epoch_s" if task == "apnea" else "epoch_s"])
            unexplained = set(off.tolist()) - split_tie_rows(models["tasks"][task], feats[L], off.tolist())
            assert not unexplained, f"{task}: rows {sorted(unexplained)[:5]} differ by up to {d.max():.3e}"
            print(f"\n{task}: {len(off)} rows on a split threshold, max |dp| {d.max():.2e}")
        rest = np.delete(d, off)
        worst = max(worst, float(rest.max()) if rest.size else 0.0)
    return worst


# ---------------------------------------------------------------- Python vs JavaScript

@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_report_v2_parity(path, models, tmp_path):
    py, feats = py_report(path, models)
    out = js_report(path, tmp_path / "js.json")
    assert "error" not in out, out
    worst = compare_reports(py, out["report"], models, feats)
    assert out["progress"] == [[s, st] for s in ("beats", "features", "report") for st in ("running", "done")]
    print(f"\n{path.stem}: {py['grid']['n_epochs']} epochs, max |dp| {worst:.2e}, "
          f"JS {out['timing_ms']['total']:.0f} ms")


def test_report_v2_parity_lights_off_on(models, tmp_path):
    path = next(p for p in SAMPLES if p.stem == LIGHTS_SAMPLE)
    t, _ = read_rr_csv(path)
    off, on = lights_case(t)
    py, feats = py_report(path, models, off, on)
    assert py["grid"]["start_s"] == off and py["grid"]["end_s"] == on
    out = js_report(path, tmp_path / "js.json", off, on)
    assert "error" not in out, out
    compare_reports(py, out["report"], models, feats)


@pytest.mark.parametrize("off,on,code", [
    (0.0, 3600.0, "TOO_SHORT"),           # window shorter than MIN_HOURS
    (3600.0, 3000.0, "BAD_LIGHTS"),       # lights on before lights off
    (80000.0, None, "TOO_SHORT"),         # lights off after the last beat: empty grid
    (-5.0, None, "BAD_LIGHTS"),           # negative offset
])
def test_grid_errors_match(off, on, code, models, tmp_path):
    path = next(p for p in SAMPLES if p.stem == "ucddb025")
    t, rr = read_rr_csv(path)
    out = js_report(path, tmp_path / "js.json", off, on)
    with pytest.raises(REP.ReportError) as exc:
        REP2.build_report_v2(t, rr, models, lights_off_s=off, lights_on_s=on)
    assert exc.value.code == code == out["error"]["code"]
    assert exc.value.message == out["error"]["message"]


# ---------------------------------------------------------------- JSON inference vs trained pickles

def source_pickles(models: dict) -> dict:
    src = models["source"]
    mdir = ROOT / src["models_dir"]
    paths = {name: mdir / name for name in src["pickles"]}
    if not all(p.exists() for p in paths.values()):
        pytest.skip(f"pickles of {src['models_dir']} are not available")
    out = {}
    for name, p in paths.items():
        with open(p, "rb") as f:
            out[name[:-len(".pkl")]] = pickle.load(f)
    return out


@pytest.mark.parametrize("case", [s.stem for s in SAMPLES] + [f"{LIGHTS_SAMPLE}_lights"])
def test_report_v2_matches_pickles(case, models):
    """Labels of build_report_v2 equal sklearn predict_proba + somnibeat.smoothing on the pickles."""
    pk = source_pickles(models)
    stem = case.removesuffix("_lights")
    path = next(p for p in SAMPLES if p.stem == stem)
    if case.endswith("_lights"):
        t, _ = read_rr_csv(path)
        rep, feats = py_report(path, models, *lights_case(t))
    else:
        rep, feats = py_report(path, models)
    cfg = models["config"]
    f30 = pd.DataFrame(feats[float(cfg["epoch_s"])])
    f60 = pd.DataFrame(feats[float(cfg["apnea_epoch_s"])])
    n = len(f30)
    assert n == rep["grid"]["n_epochs"]
    for task in STAGE_TASKS:
        bundle = pk[task]
        p = np.mean([m.predict_proba(f30[c]) for m, c in bundle["models"]], axis=0)
        dp = float(np.abs(p - np.asarray(rep["proba"][task])).max())
        assert dp <= PICKLE_TOL, f"{task}: max |p_json - p_sklearn| {dp:.2e}"
        s = SM.smooth_record(p, np.arange(n), bundle["trans"], bundle["prior"], bundle["init"], bundle["smoothing"])
        ref = [bundle["labels"][i] for i in s.argmax(1)]
        diff = [i for i, (a, b) in enumerate(zip(ref, rep["stages"][task])) if a != b]
        assert not diff, f"{task}: {len(diff)} labels differ from the pickle pipeline, first at {diff[:5]}"
    bundle = pk["apnea"]
    for m, _ in bundle["models"]:
        assert list(m.classes_) == [0, 1]
    pa = np.mean([m.predict_proba(f60[c])[:, 1] for m, c in bundle["models"]], axis=0)
    dp = float(np.abs(pa - np.asarray(rep["proba"]["apnea"])).max())
    assert dp <= PICKLE_TOL, f"apnea: max |p_json - p_sklearn| {dp:.2e}"
    ref = (pa >= bundle["threshold"]).astype(int).tolist()
    diff = [i for i, (a, b) in enumerate(zip(ref, rep["apnea"]["labels"])) if a != b]
    near = [i for i in diff if abs(pa[i] - bundle["threshold"]) <= PICKLE_TOL]
    assert diff == near, f"apnea: minutes {[i for i in diff if i not in near][:5]} differ from the pickle pipeline"
    print(f"\n{case}: {n} epochs, {len(f60)} minutes; labels identical to the pickles"
          + (f" except {len(near)} minutes on the threshold" if near else ""))


# ---------------------------------------------------------------- worker and timing

def test_worker_protocol(tmp_path):
    sample = next(p for p in SAMPLES if p.stem == "ucddb025")
    steps_ok = [(s, st) for s in ("read", "beats", "features", "report") for st in ("running", "done")]
    runs = [
        {"csvPath": str(sample), "fileName": sample.name},                       # 0 default -> v2
        {"csvPath": str(sample), "fileName": sample.name, "model": "v1"},        # 1 explicit v1
        {"csvText": "time_s,heart_rate\n1,60\n2,61\n", "fileName": "a.csv"},      # 2 parse error
        {"csvPath": str(sample), "fileName": sample.name, "model": "v2"},        # 3 explicit v2, cached
        {"csvPath": str(sample), "fileName": sample.name, "lightsOffS": 0, "lightsOnS": 600},  # 4 TOO_SHORT
        {"csvPath": str(sample), "fileName": sample.name, "model": "v3"},        # 5 unknown model
    ]
    (tmp_path / "in.json").write_text(json.dumps({"runs": runs}))
    run_node("--worker", str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    out = json.loads((tmp_path / "out.json").read_text())
    msgs = out["runs"]
    assert out["fetched"] == ["models_v2.json", "models.json"], "each models file is fetched once, lazily"
    for i in (0, 1, 3):
        assert [(m["step"], m["status"]) for m in msgs[i] if m["type"] == "progress"] == steps_ok
        assert msgs[i][-1]["type"] == "result" and msgs[i][-1]["report"]["fileName"] == sample.name
    assert msgs[0][-1]["report"]["model"] == msgs[3][-1]["report"]["model"] == "v2"
    assert msgs[1][-1]["report"]["model"] == "v1"
    assert msgs[0][-1]["report"] == msgs[3][-1]["report"]
    assert msgs[2][-1]["type"] == "error" and msgs[2][-1]["code"] == "NO_RR_COLUMN"
    last = msgs[4][-1]
    assert last["type"] == "error" and last["code"] == "TOO_SHORT" and last["details"]["min_hours"] == REP.MIN_HOURS
    assert msgs[5][-1]["type"] == "error" and msgs[5][-1]["code"] == "INTERNAL"


def test_worker_falls_back_to_v1(tmp_path):
    sample = next(p for p in SAMPLES if p.stem == "ucddb025")
    runs = [{"csvPath": str(sample), "fileName": sample.name},            # 0 v2 unavailable -> v1 + warning
            {"csvPath": str(sample), "fileName": sample.name}]            # 1 v2 retried and available
    failing = {"runs": runs, "failRuns": {"models_v2.json": [0, 0]}}
    (tmp_path / "in.json").write_text(json.dumps(failing))
    run_node("--worker", str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["fetched"] == ["models_v2.json", "models.json", "models_v2.json"]
    first, second = out["runs"][0][-1], out["runs"][1][-1]
    assert first["type"] == "result" and first["report"]["model"] == "v1"
    w = first["report"]["quality"]["warnings"]
    assert w and "v2 prediction models could not be loaded" in w[0] and "v1 models" in w[0]
    assert second["type"] == "result" and second["report"]["model"] == "v2"
    assert not any("could not be loaded" in x for x in second["report"]["quality"]["warnings"])
    # Neither models file available: the error protocol is unchanged.
    both = {"runs": runs[:1], "failRuns": {"models_v2.json": [0, 0], "models.json": [0, 0]}}
    (tmp_path / "in2.json").write_text(json.dumps(both))
    run_node("--worker", str(tmp_path / "in2.json"), str(tmp_path / "out2.json"))
    last = json.loads((tmp_path / "out2.json").read_text())["runs"][0][-1]
    assert last["type"] == "error" and last["code"] == "MODELS_UNAVAILABLE" and last["message"]


def test_node_timing_largest_sample(tmp_path):
    largest = max(SAMPLES, key=lambda p: p.stat().st_size)
    out = js_report(largest, tmp_path / "js.json")
    assert "error" not in out, out
    ms = out["timing_ms"]["total"]
    print(f"\n{largest.name}: full v2 JS report in {ms:.0f} ms")
    assert ms < MAX_NODE_S * 1000
