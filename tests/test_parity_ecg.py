"""Browser ECG converter (web/js/ecg.js) against the Python pipeline (scripts/ecg_to_rr.py).

The RR series the browser makes from an EDF must equal the one the evaluation used, so
the reported accuracy applies to an ECG dropped on the page.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
import ecg_to_rr  # noqa: E402
from test_ecg_to_rr import synthetic_ecg, write_edf  # noqa: E402

CLI = ROOT / "tests/parity/ecg_cli.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def run_js(path: Path, tmp: Path) -> dict:
    out = tmp / "js.json"
    subprocess.run(["node", str(CLI), str(path), str(out)], check=True, timeout=600)
    return json.loads(out.read_text())


def assert_same_rr(js: dict, csv: Path) -> None:
    assert "error" not in js, js
    ref = pd.read_csv(csv)
    assert len(js["time_s"]) == len(ref)
    assert np.array_equal(np.array(js["time_s"]), ref.time_s.to_numpy())
    assert np.array_equal(np.array(js["rr_ms"]), ref.rr_ms.to_numpy())


@pytest.mark.parametrize("fs", [100, 128, 200, 250, 256, 500, 512, 1000])
def test_filter_sections_match_scipy(tmp_path, fs):
    out = tmp_path / "sos.json"
    subprocess.run(["node", str(CLI), "--sos", str(fs), str(out)], check=True)
    js = np.array(json.loads(out.read_text())["sos"])
    py = butter(2, [5.0, min(25.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    assert js.shape == py.shape
    assert np.max(np.abs(js - py)) < 1e-14


def lead_off(x: np.ndarray, fs: int) -> np.ndarray:
    x = x.copy()
    x[int(100 * fs):int(130 * fs)] = 0.0  # 30 s flat line (electrode off)
    return x


@pytest.mark.parametrize("fs, transform", [(256, None), (128, None), (500, None), (256, "inverted"), (256, "lead_off")])
def test_synthetic_edf_matches_python(tmp_path, fs, transform):
    x, _ = synthetic_ecg()
    if fs != 256:  # resample the synthetic signal by regenerating on a new grid
        t_old = np.arange(len(x)) / 256.0
        t_new = np.arange(int(len(x) / 256.0 * fs)) / fs
        x = np.interp(t_new, t_old, x)
    if transform == "inverted":
        x = -x
    elif transform == "lead_off":
        x = lead_off(x, fs)
    edf = tmp_path / "night.edf"
    write_edf(edf, x, fs)
    ref = tmp_path / "py.csv"
    ecg_to_rr.convert(edf, ref)
    js = run_js(edf, tmp_path)
    assert_same_rr(js, ref)
    assert js["start_clock"] == "22:30"


@pytest.mark.parametrize("header, scale", [("Phone timestamp;RR-interval [ms]", 1.0), ("time,RR", 0.001)])
def test_rr_export_matches_python(tmp_path, header, scale):
    rr = np.array([812, 845, 860, 830, 801] * 50, dtype=float)
    sep = ";" if ";" in header else ","
    src = tmp_path / "strap.csv"
    src.write_text(header + "\n" + "\n".join(f"2026-01-01 23:00:{i % 60:02d}{sep}{v * scale:g}" for i, v in enumerate(rr)) + "\n")
    ref = tmp_path / "py.csv"
    ecg_to_rr.convert(src, ref)
    assert_same_rr(run_js(src, tmp_path), ref)


def test_errors(tmp_path):
    x, _ = synthetic_ecg(seconds=60)
    edf = tmp_path / "eeg.edf"
    write_edf(edf, x, 256, label="EEG C3")
    err = run_js(edf, tmp_path)["error"]
    assert err["code"] == "NO_ECG_CHANNEL" and err["details"]["labels"] == ["EEG C3"]
    write_edf(edf, x[::4], 64)
    assert run_js(edf, tmp_path)["error"]["code"] == "LOW_FS"
    junk = tmp_path / "junk.edf"
    junk.write_bytes(b"not an edf file" * 40)
    assert run_js(junk, tmp_path)["error"]["code"] == "NOT_EDF"


@pytest.mark.parametrize("raw, cached", [("ucddb/ucddb015.rec", "ucddb_ucddb015.csv"),
                                         ("capslpdb/n1.edf", "cap_n1.csv")])
def test_real_recording_matches_the_evaluated_rr_series(tmp_path, raw, cached):
    src, ref = ROOT / "data/raw" / raw, ROOT / "data/cache/rr" / cached
    if not (src.exists() and ref.exists()):
        pytest.skip("raw data or RR cache not present (run make download features)")
    assert_same_rr(run_js(src, tmp_path), ref)
