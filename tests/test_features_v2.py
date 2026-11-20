"""Synthetic checks for somnibeat/features_v2.py (long-context v2 features)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from somnibeat import features_v2 as V
from somnibeat import rr as R

CFG_RR = {"min_rr_ms": 300, "max_rr_ms": 2000, "max_relative_jump": 0.30,
          "local_median_beats": 11, "min_epoch_coverage": 0.70}


def beats_from_rr(rr_fn, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an RR function of time into beat times; returns (t_close, rr_ms)."""
    t, ts, rrs = 0.0, [], []
    while t < duration_s:
        r = float(rr_fn(t))
        t += r / 1000.0
        ts.append(t)
        rrs.append(r)
    return np.array(ts), np.array(rrs)


def run(t, rr, length=30.0, starts=None):
    valid = R.clean_rr(t, rr, CFG_RR)
    if starts is None:
        starts = np.arange(0.0, t[-1] - length, length)
    return V.long_features(t, rr, valid, starts, length, CFG_RR), starts


def test_cvhr_frequency_grid():
    assert len(V.CVHR_FREQS) == 157
    assert V.CVHR_FREQS[0] == pytest.approx(0.01) and V.CVHR_FREQS[-1] == pytest.approx(0.40)
    lo = V.CVHR_FREQS[V.LO_IDX[0]:V.LO_IDX[1]]
    hi = V.CVHR_FREQS[V.HI_IDX[0]:V.HI_IDX[1]]
    assert lo.min() >= 0.01 - 1e-12 and lo.max() < 0.04
    assert hi.min() >= 0.04 - 1e-12 and hi.max() < 0.10


@pytest.mark.parametrize("f_hz, band", [(0.025, "lo"), (0.07, "hi")])
def test_periodic_signal_lands_in_band(f_hz, band):
    t, rr = beats_from_rr(lambda s: 900.0 + 60.0 * math.sin(2 * math.pi * f_hz * s), 3600.0)
    f, _ = run(t, rr)
    mid = slice(30, 90)  # epochs whose 600 s window is fully inside the recording
    other = "hi" if band == "lo" else "lo"
    assert np.all(f[f"cvhr_{band}_ratio"][mid] > 0.8)
    assert np.all(f[f"log_cvhr_{band}"][mid] > f[f"log_cvhr_{other}"][mid] + 2.0)
    assert np.all(np.isfinite(f[f"z_log_cvhr_{band}"]) | np.isnan(f[f"log_cvhr_{band}"]))


def test_cvhr_index_counts_accelerations():
    # 60 bpm baseline (1000 ms), a 15 s acceleration to ~80 bpm (750 ms) every 120 s
    # -> 5 accelerations per 10 min.
    def rr_fn(s):
        return 750.0 if (s % 120.0) >= 60.0 and (s % 120.0) < 75.0 else 1000.0

    t, rr = beats_from_rr(rr_fn, 3600.0)
    valid = R.clean_rr(t, rr, {**CFG_RR, "max_relative_jump": 0.5})
    on = V.detect_accelerations(t, rr, valid)
    assert len(on) == pytest.approx(30, abs=1)
    assert np.all(np.diff(on) > 100.0)
    f = V.long_features(t, rr, valid, np.arange(600.0, 3000.0, 30.0), 30.0, CFG_RR)
    assert np.allclose(f["cvhr_index"], 5.0, atol=0.6)
    # a flat series has none
    t2, rr2 = beats_from_rr(lambda s: 1000.0, 1800.0)
    f2, _ = run(t2, rr2)
    assert len(V.detect_accelerations(t2, rr2, np.ones(len(rr2), bool))) == 0
    assert np.all(f2["cvhr_index"][~np.isnan(f2["cvhr_index"])] == 0.0)


def test_sustained_rise_is_not_counted():
    # step up by 20 bpm and stay there: no return within 60 s -> not an acceleration
    t, rr = beats_from_rr(lambda s: 1000.0 if s < 900 else 750.0, 1800.0)
    on = V.detect_accelerations(t, rr, np.ones(len(rr), bool))
    assert len(on) == 0


def test_hr_slope():
    # HR rises linearly from 60 to 90 bpm over 60 min -> 0.5 bpm/min
    t, rr = beats_from_rr(lambda s: 60000.0 / (60.0 + 0.5 * s / 60.0), 3600.0)
    f, _ = run(t, rr)
    assert np.allclose(f["hr_slope_w20"][20:100], 0.5, atol=0.01)


def test_percentile_rank():
    ref = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    x = np.array([0.0, 1.0, 2.5, 4.0, 5.0, np.nan])
    p = V.percentile_rank(x, ref)
    assert np.allclose(p[:5], [0.0, 0.125, 0.5, 0.875, 1.0])
    assert math.isnan(p[5])
    assert np.all(np.isnan(V.percentile_rank(x, np.array([np.nan]))))


def test_percentile_rank_in_night_and_hours_to_end():
    t, rr = beats_from_rr(lambda s: 800.0 + 200.0 * s / 3600.0, 3600.0)  # RR increases all night
    f, starts = run(t, rr)
    p = f["pct_mean_rr_w1"]
    assert np.all(np.diff(p) > 0) and p[0] < 0.02 and p[-1] > 0.98
    assert f["hours_to_end"][-1] == 0.0
    assert f["hours_to_end"][0] == pytest.approx((starts[-1] - starts[0]) / 3600.0)


def test_nan_handling_gap():
    # 1 h of beats with a 20 min dropout (no beats) in the middle
    t, rr = beats_from_rr(lambda s: 900.0 + 30.0 * math.sin(2 * math.pi * 0.02 * s), 3600.0)
    keep = (t < 1200.0) | (t > 2400.0)
    t, rr = t[keep], rr[keep]
    rr = rr.copy()
    first_after = int(np.argmax(t > 2400.0))
    rr[first_after] = (t[first_after] - t[first_after - 1]) * 1000.0  # long interval, R1-invalid
    starts = np.arange(0.0, 3570.0, 30.0)
    f, _ = run(t, rr, starts=starts)
    k = int(np.searchsorted(starts, 1800.0))  # window [1515, 2115) has no beats
    for name in ("mean_rr_w20", "log_cvhr_lo", "cvhr_lo_ratio", "cvhr_index", "hr_slope_w20",
                 "pct_mean_rr_w1", "z_log_cvhr_lo"):
        assert math.isnan(f[name][k]), name
    assert f["coverage_w20"][k] == 0.0
    assert np.isfinite(f["log_cvhr_lo"][30]) and np.isfinite(f["z_log_cvhr_lo"][30])
    for name, x in f.items():
        if name.startswith("z_"):
            assert np.all(np.abs(x[np.isfinite(x)]) <= 10.0), name


def test_columns_and_empty_grid():
    t, rr = beats_from_rr(lambda s: 1000.0, 900.0)
    valid = R.clean_rr(t, rr, CFG_RR)
    f = V.long_features(t, rr, valid, np.zeros(0), 30.0, CFG_RR)
    assert all(len(v) == 0 for v in f.values())
    expected = V.NAMES + [f"z_{n}" for n in V.NAMES if n not in V.Z_EXCLUDE]
    assert list(f) == expected
