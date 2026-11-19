"""Python/JavaScript parity for the night report (somnibeat/report.py vs web/js/report.js).

For each sample CSV in web/data/samples/ (and once with lights-off/on offsets) the
report is built in Python and in Node (tests/parity/report_cli.mjs) and compared:
identical grid, per-epoch labels for every task, low-quality flags, apnea labels and
warnings; night metrics and quality numbers within 1e-9; probabilities within 1e-9.
Also checks the pure-Python tree traversal against the sklearn pickles, night metrics
against somnibeat.evaluate, the worker message protocol and the Node run time.
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

from somnibeat import evaluate as E
from somnibeat import features as F
from somnibeat import report as REP
from somnibeat import rr as R

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tests" / "parity" / "report_cli.mjs"
MODELS_JSON = ROOT / "web" / "data" / "models.json"
SAMPLES = sorted((ROOT / "web" / "data" / "samples").glob("*.csv"))
TASKS = ["sleep_wake", "wake_nrem_rem", "four_class", "apnea"]
TOL = 1e-9
MAX_NODE_S = 15.0
NODE = shutil.which("node")

pytestmark = [
    pytest.mark.skipif(NODE is None, reason="node is not installed"),
    pytest.mark.skipif(not MODELS_JSON.exists() or len(SAMPLES) < 3,
                       reason="web/data is not exported (scripts/export_web.py)"),
]


@pytest.fixture(scope="module")
def models() -> dict:
    return json.loads(MODELS_JSON.read_text())


def read_rr_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # csv + float() rounds decimal strings exactly like JavaScript Number().
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["time_s"]) for r in rows]), np.array([float(r["rr_ms"]) for r in rows])


def run_node(*args: str) -> subprocess.CompletedProcess:
    res = subprocess.run([NODE, str(CLI), *args], capture_output=True, text=True, timeout=300, cwd=ROOT)
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


def close(a, b) -> bool:
    a = math.nan if a is None else float(a)  # JSON writes NaN as null
    b = math.nan if b is None else float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= TOL * max(abs(a), abs(b), 1.0)


def feature_frames(t, rr, models, start_s, end_s) -> dict:
    cfg = models["config"]
    valid = R.clean_rr(t, rr, cfg["rr"])
    out = {}
    for L in (cfg["epoch_s"], cfg["apnea_epoch_s"]):
        starts = REP.epoch_grid(start_s, end_s, L)
        out[L] = F.epoch_features(t, rr, valid, starts, L, cfg["rr"], cfg["features"])
    return out


def split_tie_rows(task: dict, feats: dict, rows) -> set:
    """Rows whose traversal passes a split with |x - threshold| <= TOL * max(|x|, 1).

    A feature that sits on a split threshold (possible when the sample was part of the
    training data, whose values define the bin thresholds) can go either way under the
    last-bit differences between numpy/libm and V8, so its probability may legitimately
    differ. Every other row must agree within TOL.
    """
    cols = [np.asarray(feats[c], dtype=float) for c in task["columns"]]
    ties = set()
    for row in rows:
        for iteration in task["trees"]:
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


def compare_reports(py: dict, js: dict, models: dict | None = None, feats: dict | None = None) -> dict:
    assert py["fileName"] == js["fileName"]
    for k in ("start_s", "end_s", "epoch_s", "n_epochs"):
        assert py["grid"][k] == js["grid"][k], f"grid {k}: {py['grid'][k]} vs {js['grid'][k]}"
    for task in ("sleep_wake", "wake_nrem_rem", "four_class"):
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
    worst = {}
    for task in TASKS:
        p = np.asarray(py["proba"][task], dtype=float)
        j = np.asarray(js["proba"][task], dtype=float).reshape(p.shape)
        d = np.abs(p - j).max(axis=1) if p.size else np.zeros(0)
        off = np.flatnonzero(d > TOL)
        if len(off):
            assert models is not None and feats is not None, f"{task}: max |p_py - p_js| = {d.max():.3e}"
            L = models["config"]["apnea_epoch_s" if task == "apnea" else "epoch_s"]
            unexplained = set(off.tolist()) - split_tie_rows(models["tasks"][task], feats[L], off.tolist())
            assert not unexplained, f"{task}: rows {sorted(unexplained)[:5]} differ by up to {d.max():.3e}"
            print(f"\n{task}: {len(off)} rows on a split threshold, max |dp| {d.max():.2e}")
        worst[task] = float(d[np.setdiff1d(np.arange(len(d)), off)].max()) if len(d) > len(off) else 0.0
    return worst


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_report_parity(path, models, tmp_path):
    t, rr = read_rr_csv(path)
    py = REP.build_report(t, rr, models, file_name=path.name, include_proba=True)
    out = js_report(path, tmp_path / "js.json")
    assert "error" not in out, out
    worst = compare_reports(py, out["report"], models, feature_frames(t, rr, models, 0.0, float(t[-1])))
    assert out["progress"] == [[s, st] for s in ("beats", "features", "report") for st in ("running", "done")]
    print(f"\n{path.stem}: {py['grid']['n_epochs']} epochs, max |dp| {max(worst.values()):.2e}, "
          f"metrics {py['metrics']}, JS {out['timing_ms']['total']:.0f} ms")


def test_report_parity_lights_off_on(models, tmp_path):
    path = next(p for p in SAMPLES if p.stem == "ucddb018")
    t, rr = read_rr_csv(path)
    off, on = 1234.5, float(t[-1]) - 1800.25
    py = REP.build_report(t, rr, models, lights_off_s=off, lights_on_s=on, file_name=path.name, include_proba=True)
    assert py["grid"]["start_s"] == off and py["grid"]["end_s"] == on
    out = js_report(path, tmp_path / "js.json", off, on)
    assert "error" not in out, out
    compare_reports(py, out["report"])
    # Only lights off given: the grid runs to the last beat.
    py2 = REP.build_report(t, rr, models, lights_off_s=off, file_name=path.name, include_proba=True)
    out2 = js_report(path, tmp_path / "js2.json", off, None)
    compare_reports(py2, out2["report"])
    assert py2["grid"]["end_s"] == float(t[-1])


@pytest.mark.parametrize("off,on,code", [
    (0.0, 3600.0, "TOO_SHORT"),           # window shorter than MIN_HOURS
    (3600.0, 3000.0, "BAD_LIGHTS"),       # lights on before lights off
    (80000.0, None, "TOO_SHORT"),         # lights off after the last beat: empty grid
    (0.0, 86000.0, None),                 # window longer than the recording: report with a warning
])
def test_grid_errors_match(off, on, code, models, tmp_path):
    path = next(p for p in SAMPLES if p.stem == "ucddb025")
    t, rr = read_rr_csv(path)
    out = js_report(path, tmp_path / "js.json", off, on)
    if code is None:
        py = REP.build_report(t, rr, models, lights_off_s=off, lights_on_s=on, file_name=path.name, include_proba=True)
        compare_reports(py, out["report"])
        assert any("no recorded beats" in w for w in py["quality"]["warnings"])
        return
    with pytest.raises(REP.ReportError) as exc:
        REP.build_report(t, rr, models, lights_off_s=off, lights_on_s=on)
    assert exc.value.code == code == out["error"]["code"]
    assert exc.value.message == out["error"]["message"]


def test_night_metrics_match_evaluate(tmp_path):
    rng = np.random.default_rng(7)
    seqs = [[], [1, 1, 1], [0, 0], [0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 1]]
    for _ in range(40):
        n = int(rng.integers(1, 400))
        seqs.append((rng.random(n) < rng.uniform(0.05, 0.6)).astype(int).tolist())
    (tmp_path / "in.json").write_text(json.dumps(seqs))
    run_node("--metrics", str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    got = json.loads((tmp_path / "out.json").read_text())
    for w, js in zip(seqs, got):
        py = E.night_metrics(np.array(w, dtype=bool))
        for k, v in py.items():
            assert close(v, js[k]), f"{k} for {w}: {v} vs {js[k]}"


def test_traversal_matches_sklearn(models):
    """Pure-Python traversal (somnibeat.report.predict_proba) equals the sklearn pickles."""
    pkl = {t: ROOT / "results" / "models" / f"{t}_gradient_boosting.pkl" for t in TASKS}
    if not all(p.exists() for p in pkl.values()):
        pytest.skip("trained models are not available")
    cfg = models["config"]
    rows = {30: [], 60: []}
    for path in SAMPLES:
        t, rr = read_rr_csv(path)
        valid = R.clean_rr(t, rr, cfg["rr"])
        for L in rows:
            starts = REP.epoch_grid(0.0, float(t[-1]), L)
            rows[L].append(pd.DataFrame(F.epoch_features(t, rr, valid, starts, L, cfg["rr"], cfg["features"])))
    frames = {L: pd.concat(v, ignore_index=True) for L, v in rows.items()}
    for task in TASKS:
        with open(pkl[task], "rb") as f:
            bundle = pickle.load(f)
        entry = models["tasks"][task]
        X = frames[60 if task == "apnea" else 30][entry["columns"]]
        assert X.isna().any(axis=1).any(), "the samples must include rows with missing features"
        ref = bundle["model"].predict_proba(X)
        got = REP.predict_proba(entry, {c: X[c].to_numpy(np.float64) for c in X.columns}, len(X))
        diff = float(np.max(np.abs(got - ref)))
        print(f"\n{task}: {len(X)} rows, max |p_py - p_sklearn| {diff:.2e}")
        assert diff <= 1e-12
        if len(bundle["labels"]) == 2:
            ref_lab = (ref[:, 1] >= bundle["threshold"]).astype(int)
        else:
            ref_lab = np.argmax(ref, axis=1)
        assert np.array_equal(REP.decide(entry, got), ref_lab)


def test_worker_protocol(tmp_path):
    sample = next(p for p in SAMPLES if p.stem == "ucddb025")
    lines = sample.read_text().splitlines()
    header, body = lines[0], lines[1:]
    seconds = [header] + [f"{a},{float(b) / 1000:.5f}" for a, b in (r.split(",") for r in body)]
    shuffled = [header] + body[::-1]
    runs = [
        {"csvPath": str(sample), "fileName": sample.name},
        {"csvText": "time_s,heart_rate\n1,60\n2,61\n", "fileName": "a.csv"},
        {"csvText": "\n".join([header] + body[:500]), "fileName": "b.csv"},
        {"csvText": "\n".join(shuffled), "fileName": "c.csv"},
        {"csvText": "\n".join(seconds), "fileName": "d.csv"},
        {"csvPath": str(sample), "fileName": sample.name, "lightsOffS": 0, "lightsOnS": 600},
        {"csvPath": str(sample), "fileName": sample.name, "lightsOffS": 1800, "lightsOnS": 30000},
    ]
    (tmp_path / "in.json").write_text(json.dumps({"runs": runs}))
    run_node("--worker", str(tmp_path / "in.json"), str(tmp_path / "out.json"))
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["fetches"] == 1, "models.json must be fetched once and cached"
    msgs = out["runs"]
    steps = [(m["step"], m["status"]) for m in msgs[0] if m["type"] == "progress"]
    assert steps == [(s, st) for s in ("read", "beats", "features", "report") for st in ("running", "done")]
    assert msgs[0][-1]["type"] == "result" and msgs[0][-1]["report"]["fileName"] == sample.name
    expected = ["NO_RR_COLUMN", "TOO_SHORT", "NOT_INCREASING", "BAD_UNIT", "TOO_SHORT"]
    for m, code in zip(msgs[1:6], expected):
        last = m[-1]
        assert last["type"] == "error" and last["code"] == code, last
        assert isinstance(last["message"], str) and last["message"]
        if code == "TOO_SHORT":
            assert last["details"]["min_hours"] == REP.MIN_HOURS
    rep = msgs[6][-1]["report"]
    assert rep["grid"]["start_s"] == 1800 and rep["grid"]["end_s"] == 30000


def test_node_timing_largest_sample(tmp_path):
    largest = max(SAMPLES, key=lambda p: p.stat().st_size)
    out = js_report(largest, tmp_path / "js.json")
    ms = out["timing_ms"]["total"]
    print(f"\n{largest.name}: full JS report in {ms:.0f} ms")
    assert ms < MAX_NODE_S * 1000
