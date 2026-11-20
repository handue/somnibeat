"""Python/JavaScript parity for the v2 long-context features (somnibeat/features_v2.py).

Runs web/js/rr.js + web/js/features_v2.js in Node (tests/parity/features_v2_cli.mjs) and compares
against somnibeat.rr + somnibeat.features_v2.long_features on the 30 s and 60 s analysis grids
(docs/web_contract.md): identical column names and order, identical NaN patterns, every value
within |a - b| <= 1e-9 * max(|a|, |b|, 1).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

from somnibeat import features_v2 as V
from somnibeat import rr as R

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tests" / "parity" / "features_v2_cli.mjs"
RR_DIR = ROOT / "data" / "cache" / "rr"
RECORDS = ["ucddb_ucddb018.csv", "ucddb_ucddb025.csv", "apnea-ecg_c04.csv"]
TOL = 1e-9
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def cfg_rr() -> dict:
    with open(ROOT / "config" / "v2.toml", "rb") as f:
        return tomllib.load(f)["rr"]


@pytest.fixture(scope="module")
def cfg_json(cfg_rr, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("cfg") / "config.json"
    p.write_text(json.dumps({"rr": cfg_rr}))
    return p


def read_rr_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # Same parsing as tests/test_parity_features.py (float() == JavaScript Number()).
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["time_s"]) for r in rows]), np.array([float(r["rr_ms"]) for r in rows]))


def grid(start_s: float, end_s: float, epoch_len: float) -> np.ndarray:
    n = max(0, int(math.floor((end_s - start_s) / epoch_len)))
    return start_s + epoch_len * np.arange(n)


def run_node(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([NODE, *args], capture_output=True, text=True, timeout=600, cwd=ROOT)


def to_js(a) -> list:
    return [None if not math.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def from_js(a) -> np.ndarray:
    return np.array([math.nan if v is None else v for v in a], dtype=float)


def assert_close(name: str, py, js: np.ndarray) -> float:
    """Max error |a - b| / max(|a|, |b|, 1); fails above TOL or on any NaN-pattern mismatch."""
    py = np.asarray(py, dtype=float)
    assert py.shape == js.shape, f"{name}: shape {py.shape} vs {js.shape}"
    pn, jn = ~np.isfinite(py), ~np.isfinite(js)
    bad = np.flatnonzero(pn != jn)
    assert not len(bad), f"{name}: NaN mismatch at {bad[:5]} py={py[bad[:5]]} js={js[bad[:5]]}"
    a, b = py[~pn], js[~pn]
    if not len(a):
        return 0.0
    err = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    worst = int(np.argmax(err))
    assert err[worst] <= TOL, f"{name}: err {err[worst]:.3e} (py={a[worst]!r} js={b[worst]!r})"
    return float(err[worst])


def compare(py: dict, js_cols: dict) -> float:
    assert list(js_cols) == list(py), f"column names/order differ: {list(py)} vs {list(js_cols)}"
    return max(assert_close(k, v, from_js(js_cols[k])) for k, v in py.items())


@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize("epoch_len", [30, 60])
def test_record_parity(record, epoch_len, cfg_rr, cfg_json, tmp_path):
    path = RR_DIR / record
    if not path.exists():
        pytest.skip(f"{path} not cached")
    t, rr = read_rr_csv(path)
    valid = R.clean_rr(t, rr, cfg_rr)
    start_s, end_s = 0.0, float(t[-1])
    starts = grid(start_s, end_s, epoch_len)
    py = V.long_features(t, rr, valid, starts, float(epoch_len), cfg_rr)
    out = tmp_path / "js.json"
    end_arg = "auto" if epoch_len == 30 else repr(end_s)
    res = run_node(str(CLI), str(path), str(epoch_len), repr(start_s), end_arg, str(out), str(cfg_json))
    assert res.returncode == 0, res.stderr
    js = json.loads(out.read_text())
    assert js["valid"] == valid.astype(int).tolist()
    assert np.array_equal(np.array(js["grid"]), starts)
    worst = compare(py, js["columns"])
    ms = js["timing_ms"]["features"]
    print(f"\n{record} L={epoch_len}: {len(starts)} epochs, max err {worst:.2e}, JS long features {ms:.0f} ms")


def test_lights_off_on_grid(cfg_rr, cfg_json, tmp_path):
    """Non-zero start_s / end_s: beats outside the grid still feed the 600 s edge windows."""
    path = RR_DIR / RECORDS[0]
    if not path.exists():
        pytest.skip("record not cached")
    t, rr = read_rr_csv(path)
    valid = R.clean_rr(t, rr, cfg_rr)
    start_s, end_s = 1234.5, float(t[-1]) - 987.25
    for epoch_len in (30, 60):
        starts = grid(start_s, end_s, epoch_len)
        py = V.long_features(t, rr, valid, starts, float(epoch_len), cfg_rr)
        out = tmp_path / f"js{epoch_len}.json"
        res = run_node(str(CLI), str(path), str(epoch_len), repr(start_s), repr(end_s), str(out), str(cfg_json))
        assert res.returncode == 0, res.stderr
        compare(py, json.loads(out.read_text())["columns"])


# ------------------------------------------------------------------ synthetic edge cases


def run_raw(payload: dict, tmp_path: Path) -> dict:
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps(payload, allow_nan=False))
    res = run_node(str(CLI), "--raw", str(inp), str(out))
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text())


def synthetic_parity(t, rr, starts, epoch_len, cfg_rr, tmp_path, valid=None, extra=None):
    t, rr, starts = np.asarray(t, float), np.asarray(rr, float), np.asarray(starts, float)
    v = R.clean_rr(t, rr, cfg_rr) if valid is None else np.asarray(valid, bool)
    py = V.long_features(t, rr, v, starts, float(epoch_len), cfg_rr)
    payload = {"t": to_js(t), "rr": to_js(rr), "epoch_starts": to_js(starts), "epoch_len": epoch_len,
               "cfg_rr": cfg_rr, "baseline": True, **(extra or {})}
    if valid is not None:
        payload["valid"] = v.astype(int).tolist()
    js = run_raw(payload, tmp_path)
    assert js["valid"] == v.astype(int).tolist()
    compare(py, js["columns"])
    if len(t):
        assert_close("hr_baseline", V.hr_baseline(t, rr, v), from_js(js["baseline"]))
        assert_close("onsets", V.detect_accelerations(t, rr, v), np.array(js["onsets"], dtype=float))
    if "lomb_check" in js:  # the trig-cached Lomb-Scargle must equal features.js lombPower bit for bit
        assert js["lomb_check"]["mismatches"] == 0, js["lomb_check"]
    return py, js


def beats_from_rr(rr_ms: np.ndarray, t0: float = 0.0) -> np.ndarray:
    return np.round(t0 + np.cumsum(rr_ms) / 1000.0, 4)


def test_cvhr_freqs(cfg_rr, tmp_path):
    js = run_raw({"t": [], "rr": [], "epoch_starts": [], "epoch_len": 30, "cfg_rr": cfg_rr}, tmp_path)
    assert js["freqs"] == V.CVHR_FREQS.tolist()


def test_very_short_series(cfg_rr, tmp_path):
    rng = np.random.default_rng(11)
    rr = np.round(900 + 40 * rng.standard_normal(40), 2)
    t = beats_from_rr(rr)
    for n_epochs in (0, 1, 2, 3, 4):
        synthetic_parity(t, rr, 30.0 * np.arange(n_epochs), 30, cfg_rr, tmp_path)
    synthetic_parity(t[:2], rr[:2], [0.0], 30, cfg_rr, tmp_path)
    synthetic_parity(t[:1], rr[:1], [0.0, 30.0], 30, cfg_rr, tmp_path)
    # Exactly MIN_BEATS - 1 / MIN_BEATS valid beats in the only window.
    for nb in (31, 32, 33):
        r = np.full(nb, 1000.0) + np.arange(nb) % 3
        synthetic_parity(beats_from_rr(r), r, [0.0], 30, cfg_rr, tmp_path)


def test_accelerations_and_baseline(cfg_rr, tmp_path):
    """Periodic accelerations (counted), too-long events (dropped), gaps (detector reset)."""
    rr, s = [], 0.0
    rng = np.random.default_rng(12)
    while s < 3 * 3600:
        ph = s % 150.0
        if s > 5000 and s < 6000:
            r = 650.0 if ph < 90.0 else 1000.0                     # 90 s events: too long
        else:
            r = 720.0 if 60.0 <= ph < 72.0 else 1000.0            # 12 s events: counted
        r = round(r + 8 * rng.standard_normal(), 2)
        rr.append(r)
        s += r / 1000.0
    rr = np.array(rr)
    t = beats_from_rr(rr)
    t[4000:] += 5.0                                                 # a 5 s gap (> MAX_GAP_S)
    starts30 = 30.0 * np.arange(int(t[-1] // 30))
    py, _ = synthetic_parity(t, rr, starts30, 30, cfg_rr, tmp_path, extra={"lomb_check": True})
    assert np.nanmax(py["cvhr_index"]) > 0
    synthetic_parity(t, rr, 60.0 * np.arange(int(t[-1] // 60)), 60, cfg_rr, tmp_path)


def test_all_invalid_windows(cfg_rr, tmp_path):
    rng = np.random.default_rng(13)
    rr = np.round(850 + 50 * rng.standard_normal(9000), 2)
    rr[1500:2400] = 2500.0          # > 10 min out of range -> all-invalid 600 s windows
    rr[4000:4300] = 150.0
    rr[6000:6005] *= 2.0
    t = beats_from_rr(rr)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    py, js = synthetic_parity(t, rr, starts, 30, cfg_rr, tmp_path, extra={"lomb_check": True})
    assert np.isnan(py["mean_rr_w20"]).any() and np.isnan(py["log_cvhr_lo"]).any()
    assert js["lomb_check"]["windows"] > 0 and js["lomb_check"]["mismatches"] == 0


def test_dense_invalid_beats_trig_cache(cfg_rr, tmp_path):
    """Thousands of short invalid intervals inside one 600 s window: the index span of a window's
    valid beats exceeds the trig cache's minimum size (regression: slots were sized by the
    valid-beat count only, so beats i and i + slots shared a slot and corrupted the spectrum)."""
    rng = np.random.default_rng(16)
    rr = np.round(900 + 40 * rng.standard_normal(9000), 2)
    rr[2000:5800] = 20.0            # 3800 intervals in 76 s, all out of range: valid beats on both
                                    # sides of the run differ by 4096 indices inside one window
    t = beats_from_rr(rr)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    py, js = synthetic_parity(t, rr, starts, 30, cfg_rr, tmp_path, extra={"lomb_check": True})
    assert js["lomb_check"]["windows"] > 0 and js["lomb_check"]["mismatches"] == 0
    assert np.isfinite(py["log_cvhr_lo"]).any()


def test_constant_rr(cfg_rr, tmp_path):
    rr = np.full(9000, 1000.0)      # zero spectrum (total 0 -> ratios NaN), zero slope, degenerate scales
    t = beats_from_rr(rr, 0.5)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    py, _ = synthetic_parity(t, rr, starts, 30, cfg_rr, tmp_path)
    assert np.isnan(py["cvhr_lo_ratio"]).all()
    rr2 = np.round(1000 + 10 * np.sin(np.arange(9000) / 7.0), 2)
    synthetic_parity(beats_from_rr(rr2), rr2, starts, 60, cfg_rr, tmp_path)


def test_nan_heavy_and_random_mask(cfg_rr, tmp_path):
    rng = np.random.default_rng(14)
    rr = np.round(700 + 120 * rng.standard_normal(12000), 2)
    bad = rng.random(12000) < 0.45
    rr[bad] = rng.choice([100.0, 2600.0, math.nan], bad.sum())
    t = np.round(np.cumsum(np.where(np.isfinite(rr), rr, 800.0)) / 1000.0, 4)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    synthetic_parity(t, rr, starts, 30, cfg_rr, tmp_path)
    synthetic_parity(t, rr, 7.5 + 60.0 * np.arange(int(t[-1] // 60)), 60, cfg_rr, tmp_path)
    v = rng.random(12000) < 0.6
    rr_f = np.where(np.isfinite(rr), rr, 800.0)
    synthetic_parity(t, rr_f, starts, 30, cfg_rr, tmp_path, valid=v, extra={"lomb_check": True})


def test_percentile_rank_ties(cfg_rr, tmp_path):
    x = np.array([1.0, 2.0, 2.0, 3.0, math.nan, 0.5, 4.0, 2.0])
    for ref in (x, np.array([2.0, 2.0, math.nan]), np.array([math.nan]), np.array([])):
        js = run_raw({"t": [], "rr": [], "epoch_starts": [], "epoch_len": 30, "cfg_rr": cfg_rr,
                      "pct": {"x": to_js(x), "ref": to_js(ref)}}, tmp_path)
        assert_close("pct", V.percentile_rank(x, ref), from_js(js["pct"]))
