"""scripts/ecg_to_rr.py: ECG recording -> the RR CSV the demo reads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ecg_to_rr  # noqa: E402

FS = 256.0


def synthetic_ecg(seconds: float = 300.0, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """ECG-like signal: a narrow QRS spike (plus a small T wave) at known beat times, with noise."""
    rng = np.random.default_rng(seed)
    rr = 0.85 + 0.08 * np.sin(np.arange(1000) / 6.0) + rng.normal(0, 0.01, 1000)  # 0.75-0.95 s
    beats = 1.0 + np.cumsum(rr)
    beats = beats[beats < seconds - 1.0]
    t = np.arange(int(seconds * FS)) / FS
    x = rng.normal(0, 0.01, t.size)
    for b in beats:
        x += 1.0 * np.exp(-0.5 * ((t - b) / 0.012) ** 2)        # R wave
        x += 0.25 * np.exp(-0.5 * ((t - b - 0.25) / 0.05) ** 2)  # T wave
    return x, beats


def write_edf(path: Path, x: np.ndarray, fs: int, label: str = "ECG") -> None:
    """Minimal one-channel EDF (16-bit) as written by sleep-lab software."""
    rec_s = 1
    n_rec = len(x) // (fs * rec_s)
    x = x[: n_rec * fs]
    pmin, pmax, dmin, dmax = -5.0, 5.0, -32768, 32767
    dig = np.round((x - pmin) / (pmax - pmin) * (dmax - dmin) + dmin).astype("<i2")

    def f(s, w):
        return str(s).ljust(w)[:w]
    head = (f("0", 8) + f("X", 80) + f("Test", 80) + "02.01.20" + "22.30.00" + f(512, 8) + f("", 44)
            + f(n_rec, 8) + f(rec_s, 8) + f(1, 4))
    sig = (f(label, 16) + f("", 80) + f("mV", 8) + f(pmin, 8) + f(pmax, 8) + f(dmin, 8) + f(dmax, 8)
           + f("", 80) + f(fs * rec_s, 8) + f("", 32))
    path.write_bytes((head + sig).encode("ascii") + dig.tobytes())


def rr_close(out: Path, beats: np.ndarray) -> None:
    df = pd.read_csv(out)
    assert list(df.columns) == ["time_s", "rr_ms"]
    want_t, want_rr = beats[1:], np.diff(beats) * 1000
    assert len(df) == len(want_t)
    assert np.max(np.abs(df.time_s - want_t)) < 0.004
    assert np.max(np.abs(df.rr_ms - want_rr)) < 6.0


def test_csv_single_column(tmp_path):
    x, beats = synthetic_ecg()
    src = tmp_path / "ecg.csv"
    np.savetxt(src, x, fmt="%.5f")
    out = tmp_path / "rr.csv"
    r = ecg_to_rr.convert(src, out, fs=FS)
    rr_close(out, beats)
    assert r["beats"] == len(beats) and r["usable_fraction"] > 0.99 and r["start_clock"] is None


def test_csv_named_column(tmp_path):
    x, beats = synthetic_ecg()
    src = tmp_path / "ecg.csv"
    pd.DataFrame({"t": np.arange(len(x)) / FS, "ecg": x}).to_csv(src, index=False)
    out = tmp_path / "rr.csv"
    ecg_to_rr.convert(src, out, fs=FS, column="ecg")
    rr_close(out, beats)


def test_edf_matches_csv_and_reads_start_clock(tmp_path):
    x, beats = synthetic_ecg()
    edf = tmp_path / "night.edf"
    write_edf(edf, x, int(FS))
    assert ecg_to_rr.main([str(edf)]) == 0
    out = tmp_path / "night_rr.csv"  # default output name
    rr_close(out, beats)
    r = ecg_to_rr.convert(edf, tmp_path / "again.csv")
    assert r["start_clock"] == "22:30" and r["source"] == "EDF channel 'ECG'"


def test_errors(tmp_path):
    x, _ = synthetic_ecg(seconds=30)
    src = tmp_path / "ecg.csv"
    np.savetxt(src, x)
    with pytest.raises(SystemExit, match="--fs"):
        ecg_to_rr.convert(src, tmp_path / "o.csv")
    with pytest.raises(SystemExit, match="below"):
        ecg_to_rr.convert(src, tmp_path / "o.csv", fs=50)
    edf = tmp_path / "n.edf"
    write_edf(edf, x, int(FS), label="EEG C3")
    with pytest.raises(SystemExit, match="no channel matching .*--channel"):
        ecg_to_rr.convert(edf, tmp_path / "o.csv")
    with pytest.raises(SystemExit, match="unsupported"):
        ecg_to_rr.convert(tmp_path / "x.wav", tmp_path / "o.csv")


@pytest.mark.parametrize("raw, cached", [("ucddb/ucddb015.rec", "ucddb_ucddb015.csv"),
                                         ("slpdb/slp01a.hea", "slpdb_slp01a.csv")])
def test_real_recording_matches_the_evaluated_rr_series(tmp_path, raw, cached):
    src, ref = ROOT / "data/raw" / raw, ROOT / "data/cache/rr" / cached
    if not (src.exists() and ref.exists()):
        pytest.skip("raw data or RR cache not present (run make download features)")
    out = tmp_path / "rr.csv"
    ecg_to_rr.convert(src, out)
    assert out.read_bytes() == ref.read_bytes()


@pytest.mark.parametrize("header, scale", [("Phone timestamp,RR-interval [ms]", 1.0), ("time;RR", 0.001)])
def test_rr_export(tmp_path, header, scale):
    rr = np.array([812, 845, 860, 830, 801] * 50, dtype=float)
    sep = ";" if ";" in header else ","
    rows = [f"2026-01-01 23:00:{i % 60:02d}.000{sep}{v * scale:g}" for i, v in enumerate(rr)]
    src = tmp_path / "strap.csv"
    src.write_text(header + "\n" + "\n".join(rows) + "\n")
    out = tmp_path / "rr.csv"
    r = ecg_to_rr.convert(src, out)
    df = pd.read_csv(out)
    assert np.allclose(df.rr_ms, rr) and np.allclose(df.time_s, np.cumsum(rr) / 1000)
    assert r["beats"] == len(rr) and r["source"].startswith("RR column")
