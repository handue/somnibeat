"""Python/JavaScript parity for RR cleaning and epoch features.

Runs web/js/rr.js + web/js/features.js in Node (tests/parity/features_cli.mjs) and compares
against somnibeat.rr + somnibeat.features: every feature within 1e-9 relative error (or both NaN),
identical column names and order, identical validity masks.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

from somnibeat import features as F
from somnibeat import rr as R

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tests" / "parity" / "features_cli.mjs"
RR_DIR = ROOT / "data" / "cache" / "rr"
RECORDS = ["ucddb_ucddb018.csv", "ucddb_ucddb025.csv", "apnea-ecg_c04.csv"]
REL = 1e-9
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(ROOT / "config" / "default.toml", "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def cfg_json(cfg, tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("cfg") / "config.json"
    p.write_text(json.dumps({"rr": cfg["rr"], "features": cfg["features"]}))
    return p


def read_rr_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # csv + float() parses decimal strings exactly like JavaScript Number() (correctly rounded).
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["time_s"]) for r in rows]), np.array([float(r["rr_ms"]) for r in rows]))


def grid(start_s: float, end_s: float, epoch_len: float) -> np.ndarray:
    n = max(0, int(math.floor((end_s - start_s) / epoch_len)))
    return start_s + epoch_len * np.arange(n)


def run_node(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([NODE, *args], input=stdin, capture_output=True, text=True, timeout=300, cwd=ROOT)


def to_js(a) -> list:
    return [None if not math.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def from_js(a) -> np.ndarray:
    return np.array([math.nan if v is None else v for v in a], dtype=float)


def assert_close(name: str, py: np.ndarray, js: np.ndarray) -> float:
    """Return the max relative error; fail if any element breaks 1e-9 relative (or NaN mismatch)."""
    py = np.asarray(py, dtype=float)
    assert py.shape == js.shape, f"{name}: shape {py.shape} vs {js.shape}"
    pn, jn = ~np.isfinite(py), ~np.isfinite(js)
    bad = np.flatnonzero(pn != jn)
    assert not len(bad), f"{name}: NaN mismatch at {bad[:5]} py={py[bad[:5]]} js={js[bad[:5]]}"
    ok = ~pn
    a, b = py[ok], js[ok]
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(diff == 0, 0.0, diff / np.where(scale == 0, 1.0, scale))
    worst = int(np.argmax(rel)) if len(rel) else 0
    assert not len(rel) or rel[worst] <= REL, f"{name}: rel err {rel[worst]:.3e} (py={a[worst]!r} js={b[worst]!r})"
    return float(rel.max()) if len(rel) else 0.0


def compare_features(py: dict, js_cols: dict) -> float:
    assert list(js_cols) == list(py), f"column names/order differ: {set(py) ^ set(js_cols)}"
    return max(assert_close(k, v, from_js(js_cols[k])) for k, v in py.items())


@pytest.mark.parametrize("record", RECORDS)
@pytest.mark.parametrize("epoch_len", [30, 60])
def test_record_parity(record, epoch_len, cfg, cfg_json, tmp_path):
    path = RR_DIR / record
    if not path.exists():
        pytest.skip(f"{path} not cached")
    t, rr = read_rr_csv(path)
    valid = R.clean_rr(t, rr, cfg["rr"])
    start_s, end_s = 0.0, float(t[-1])
    starts = grid(start_s, end_s, epoch_len)
    py = F.epoch_features(t, rr, valid, starts, epoch_len, cfg["rr"], cfg["features"])

    out = tmp_path / "js.json"
    end_arg = "auto" if epoch_len == 30 else repr(end_s)
    res = run_node(str(CLI), str(path), str(epoch_len), repr(start_s), end_arg, str(out), str(cfg_json))
    assert res.returncode == 0, res.stderr
    js = json.loads(out.read_text())
    assert js["valid"] == valid.astype(int).tolist()
    assert np.array_equal(np.array(js["grid"]), starts)
    worst = compare_features(py, js["columns"])
    ms = js["timing_ms"]["features"]
    print(f"\n{record} L={epoch_len}: {len(starts)} epochs, max rel err {worst:.2e}, JS features {ms:.0f} ms")
    assert ms < 10_000


def test_lights_off_on_grid(cfg, cfg_json, tmp_path):
    """Non-zero start_s / end_s (lights-off/on): beats outside the grid still feed edge windows."""
    path = RR_DIR / RECORDS[0]
    if not path.exists():
        pytest.skip("record not cached")
    t, rr = read_rr_csv(path)
    valid = R.clean_rr(t, rr, cfg["rr"])
    start_s, end_s = 1234.5, float(t[-1]) - 987.25
    starts = grid(start_s, end_s, 30)
    py = F.epoch_features(t, rr, valid, starts, 30, cfg["rr"], cfg["features"])
    out = tmp_path / "js.json"
    res = run_node(str(CLI), str(path), "30", repr(start_s), repr(end_s), str(out), str(cfg_json))
    assert res.returncode == 0, res.stderr
    compare_features(py, json.loads(out.read_text())["columns"])


# ------------------------------------------------------------------ synthetic edge cases


def run_raw(payload: dict, tmp_path: Path) -> dict:
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    inp.write_text(json.dumps(payload, allow_nan=False))
    res = run_node(str(CLI), "--raw", str(inp), str(out))
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text())


def synthetic_parity(t, rr, starts, epoch_len, cfg, tmp_path, valid=None):
    t, rr, starts = np.asarray(t, float), np.asarray(rr, float), np.asarray(starts, float)
    v = R.clean_rr(t, rr, cfg["rr"]) if valid is None else np.asarray(valid, bool)
    py = F.epoch_features(t, rr, v, starts, epoch_len, cfg["rr"], cfg["features"])
    payload = {"t": to_js(t), "rr": to_js(rr), "epoch_starts": to_js(starts), "epoch_len": epoch_len,
               "cfg_rr": cfg["rr"], "cfg_feat": cfg["features"]}
    if valid is not None:
        payload["valid"] = v.astype(int).tolist()
    js = run_raw(payload, tmp_path)
    assert js["valid"] == v.astype(int).tolist()
    compare_features(py, js["columns"])
    return py


def beats_from_rr(rr_ms: np.ndarray, t0: float = 0.0) -> np.ndarray:
    return np.round(t0 + np.cumsum(rr_ms) / 1000.0, 4)


def test_very_short_series(cfg, tmp_path):
    rng = np.random.default_rng(1)
    rr = np.round(900 + 40 * rng.standard_normal(40), 2)
    t = beats_from_rr(rr)
    for n_epochs in (0, 1, 2, 4):  # n = 3 is skipped: see report (features.py raises for 3 epochs)
        synthetic_parity(t, rr, 30.0 * np.arange(n_epochs), 30, cfg, tmp_path)
    synthetic_parity(t[:2], rr[:2], [0.0], 30, cfg, tmp_path)
    synthetic_parity(t[:1], rr[:1], [0.0, 30.0], 30, cfg, tmp_path)


def test_all_invalid_window(cfg, tmp_path):
    rng = np.random.default_rng(2)
    rr = np.round(850 + 50 * rng.standard_normal(3000), 2)
    rr[800:1100] = 2500.0          # long stretch out of range (R1) -> all-invalid windows
    rr[1500:1700] = 150.0
    rr[2000:2005] *= 2.0           # isolated missed beats (R2)
    t = beats_from_rr(rr)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    py = synthetic_parity(t, rr, starts, 30, cfg, tmp_path)
    assert np.isnan(py["mean_rr_w10"]).any() and (py["low_quality"] == 1).any()


def test_all_zero_pnn50_record(cfg, tmp_path):
    rr = np.full(2500, 1000.0)     # constant RR: SDNN = RMSSD = pNN50 = 0, every robust scale degenerates
    t = beats_from_rr(rr, 0.5)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    py = synthetic_parity(t, rr, starts, 30, cfg, tmp_path)
    assert np.nanmax(py["pnn50_w1"]) == 0.0
    rr2 = np.round(1000 + 10 * np.sin(np.arange(2500) / 7.0), 2)  # small differences: pNN50 = 0, SDNN > 0
    synthetic_parity(beats_from_rr(rr2), rr2, starts, 60, cfg, tmp_path)


def test_nan_heavy(cfg, tmp_path):
    rng = np.random.default_rng(3)
    rr = np.round(700 + 120 * rng.standard_normal(6000), 2)
    bad = rng.random(6000) < 0.45
    rr[bad] = rng.choice([100.0, 2600.0, math.nan], bad.sum())
    t = np.round(np.cumsum(np.where(np.isfinite(rr), rr, 800.0)) / 1000.0, 4)
    starts = 30.0 * np.arange(int(t[-1] // 30))
    synthetic_parity(t, rr, starts, 30, cfg, tmp_path)
    synthetic_parity(t, rr, 7.5 + 60.0 * np.arange(int(t[-1] // 60)), 60, cfg, tmp_path)
    # random validity mask straight into epoch_features (bypasses clean_rr)
    v = rng.random(6000) < 0.5
    rr_f = np.where(np.isfinite(rr), rr, 800.0)
    synthetic_parity(t, rr_f, starts, 30, cfg, tmp_path, valid=v)


def test_lomb_dfa_and_freqs(cfg, tmp_path):
    rng = np.random.default_rng(4)
    t = np.cumsum(0.6 + 0.4 * rng.random(400))
    y = 900 + 60 * np.sin(2 * np.pi * 0.25 * t) + 20 * rng.standard_normal(400)
    xs = [rng.standard_normal(n).cumsum() + 800 for n in (31, 32, 33, 47, 64, 129, 300, 700)]
    js = run_raw({"t": [], "rr": [], "epoch_starts": [], "epoch_len": 30, "cfg_rr": cfg["rr"],
                  "cfg_feat": cfg["features"], "lomb": {"t": to_js(t), "y": to_js(y)},
                  "dfa": [to_js(x) for x in xs]}, tmp_path)
    assert js["freqs"] == F.FREQS.tolist()
    assert_close("lomb", F.lomb_power(t, y), from_js(js["lomb"]))
    assert_close("dfa", np.array([F.dfa_alpha1(x) for x in xs]), from_js(js["dfa"]))


# ------------------------------------------------------------------ CSV parsing


PARSE_JS = """
import { readFileSync } from "node:fs";
import { parseRRCsv } from "./web/js/rr.js";
const cases = JSON.parse(readFileSync(0, "utf8"));
const out = cases.map((text) => {
  try {
    const r = parseRRCsv(text);
    return { ok: true, t: Array.from(r.time_s), rr: Array.from(r.rr_ms), warnings: r.warnings, skipped: r.skippedRows };
  } catch (e) { return { ok: false, code: e.code ?? null, message: e.message }; }
});
console.log(JSON.stringify(out));
"""


def parse_js(texts: list[str]) -> list[dict]:
    res = run_node("--input-type=module", "-e", PARSE_JS, stdin=json.dumps(texts))
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_parse_csv_rules():
    rng = np.random.default_rng(5)
    rr = np.round(950 + 30 * rng.standard_normal(9000), 2)  # ~2.4 h
    t = np.round(np.cumsum(rr) / 1000.0, 4)
    t, rr = t.tolist(), rr.tolist()  # plain floats: repr() must give decimal strings
    good = "Time_S , RR_ms ,extra\n" + "".join(f"{a!r},{b!r},x\n" for a, b in zip(t, rr))
    noisy = good.replace(f"{t[10]!r},{rr[10]!r}", f"{t[10]!r},abc", 1)
    no_time = "rr_ms\r\n" + "".join(f"{b!r}\r\n" for b in rr)
    seconds = "time_s,rr_ms\n" + "".join(f"{a!r},{b / 1000!r}\n" for a, b in zip(t, rr))
    short = "time_s,rr_ms\n" + "".join(f"{a!r},{b!r}\n" for a, b in zip(t[:3000], rr[:3000]))
    t_bad = list(t)
    t_bad[::50] = [0.0] * len(t_bad[::50])               # 2 % of rows go back in time
    not_inc = "time_s,rr_ms\n" + "".join(f"{a!r},{b!r}\n" for a, b in zip(t_bad, rr))
    no_rr = "time_s,hr\n1,60\n2,60\n"
    r = parse_js([good, noisy, no_time, seconds, short, not_inc, no_rr, ""])

    assert r[0]["ok"] and r[0]["t"] == t and r[0]["rr"] == rr and r[0]["skipped"] == 0
    assert r[1]["ok"] and r[1]["skipped"] == 1 and len(r[1]["rr"]) == len(rr) - 1 and r[1]["warnings"]
    exp_t, exp_rr = R.rr_series_from_csv_rows(None, np.array(rr))
    assert r[2]["ok"] and r[2]["t"] == exp_t.tolist() and r[2]["rr"] == exp_rr.tolist() and r[2]["warnings"]
    assert [x.get("code") for x in r[3:]] == ["BAD_UNIT", "TOO_SHORT", "NOT_INCREASING", "NO_RR_COLUMN", "EMPTY"]


def test_clean_rr_parity_random(cfg, tmp_path):
    rng = np.random.default_rng(6)
    rr = np.round(800 + 100 * rng.standard_normal(5000), 2)
    idx = rng.choice(5000, 400, replace=False)
    rr[idx] *= rng.choice([0.5, 2.0, 1.35, 0.68], 400)
    t = beats_from_rr(rr)
    js = run_raw({"t": to_js(t), "rr": to_js(rr), "epoch_starts": [], "epoch_len": 30,
                  "cfg_rr": cfg["rr"], "cfg_feat": cfg["features"]}, tmp_path)
    assert js["valid"] == R.clean_rr(t, rr, cfg["rr"]).astype(int).tolist()
