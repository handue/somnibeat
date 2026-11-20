"""Long-context RR features for v2 (extra columns next to the v1 feature tables).

Computed from the same cleaned RR series as v1 (somnibeat.rr.clean_rr with config [rr]) and the
same epoch grids. Every operation is an explicit loop or a plain numpy reduction so that a
JavaScript port (web/js) can reproduce it with the same semantics; nothing here calls an opaque
library routine. Reuses lomb_power and the robust-scale helpers of somnibeat/features.py.

For each epoch k starting at s_k (length L), with center c = s_k + L/2:

    w20 = [c - 300, c + 300)   (600 s; 20 staging epochs)

An interval belongs to a window if its closing beat time is inside it (as in v1).

Features per epoch
    mean_rr_w20, sdnn_w20, rmssd_w20, cv_w20, coverage_w20
        v1 base statistics (features._window_stats) on the 600 s window.
    log_cvhr_lo, log_cvhr_hi, cvhr_lo_ratio, cvhr_hi_ratio
        Lomb-Scargle power (features.lomb_power) of the valid RR in w20 on the grid
        CVHR_FREQS = k * 0.0025 Hz, k = 4..160 (0.01-0.40 Hz). lo = sum over k 4..15
        (0.01 <= f < 0.04 Hz), hi = sum over k 16..39 (0.04 <= f < 0.10 Hz), total = sum over
        the whole grid. log_* = log(power + 1e-6); *_ratio = band / total (NaN if total <= 0).
        Needs >= 32 valid intervals and >= 50 % coverage of the window, else NaN.
    cvhr_index
        Heart-rate accelerations per 10 min (see detect_accelerations): onsets inside w20,
        divided by (valid RR seconds in w20 / 600). NaN below 50 % coverage.
    hr_slope_w20
        Least-squares slope of HR = 60000 / rr (bpm) on time (minutes) over the valid beats of
        w20, in bpm per minute. Same validity rule as the spectral features.
    pct_mean_rr_w1, pct_rmssd_w1
        Per-night percentile rank of the v1 epoch features mean_rr_w1 / rmssd_w1 among the
        night's good epochs (coverage_w1 >= min_epoch_coverage):
        (#ref < x + 0.5 * #ref == x) / #ref, in [0, 1]; NaN if x is NaN or no reference.
    hours_to_end
        (start of the last grid epoch - s_k) / 3600. Uses only the grid, which the demo knows
        (lights-on, or the end of the uploaded file).
    z_<feature>
        Per-night robust z (v1 rule: (x - median) / scale over good epochs, scale = IQR, else
        1.4826 MAD, else SD, else 1; clipped to +-10) for every feature above except the
        coverage, the percentile ranks (already per-night normalised) and hours_to_end.
"""

from __future__ import annotations

import math

import numpy as np

from . import features as F

HALF_W20 = 300.0
LONG_BASE = ["mean_rr", "sdnn", "rmssd", "cv", "coverage"]
CVHR_FREQS = np.arange(4, 161) * 0.0025  # 0.01 .. 0.40 Hz, k * 0.0025 (same doubles in JS)
LO_IDX = (0, 12)    # k 4..15  -> 0.0100 .. 0.0375 Hz
HI_IDX = (12, 36)   # k 16..39 -> 0.0400 .. 0.0975 Hz
MIN_BEATS = 32
MIN_COVERAGE_LONG = 0.5

# CVHR acceleration detector
BASELINE_HALF_S = 30.0     # 60 s centered moving median of HR
BASELINE_MIN_BEATS = 10    # fewer valid beats in the baseline window -> no baseline (gap)
ONSET_BPM = 10.0           # HR - baseline > 10 bpm starts an acceleration
RETURN_BPM = 5.0           # HR - baseline <= 5 bpm is the return
MAX_EVENT_S = 60.0         # return must follow the onset within 60 s, else not counted
MAX_GAP_S = 3.0            # a gap between consecutive valid beats resets the detector

Z_EXCLUDE = {"coverage_w20", "pct_mean_rr_w1", "pct_rmssd_w1", "hours_to_end"}
NAMES = ([f"{b}_w20" for b in LONG_BASE]
         + ["log_cvhr_lo", "log_cvhr_hi", "cvhr_lo_ratio", "cvhr_hi_ratio", "cvhr_index",
            "hr_slope_w20", "pct_mean_rr_w1", "pct_rmssd_w1", "hours_to_end"])


def _range(t: np.ndarray, a: float, b: float) -> tuple[int, int]:
    """Indices [i0, i1) of intervals whose closing beat lies in [a, b) (binary search)."""
    return int(np.searchsorted(t, a, side="left")), int(np.searchsorted(t, b, side="left"))


def hr_baseline(t: np.ndarray, rr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-interval moving median of valid HR (bpm) over [t_i - 30, t_i + 30); NaN if invalid
    or fewer than BASELINE_MIN_BEATS valid beats in the window. Median: sort, middle value, or
    mean of the two middle values (np.median semantics)."""
    hr = np.full(len(rr), math.nan)
    hr[valid] = 60000.0 / rr[valid]
    out = np.full(len(rr), math.nan)
    for i in range(len(rr)):
        if not valid[i]:
            continue
        i0, i1 = _range(t, t[i] - BASELINE_HALF_S, t[i] + BASELINE_HALF_S)
        w = hr[i0:i1][valid[i0:i1]]
        if len(w) >= BASELINE_MIN_BEATS:
            out[i] = float(np.median(w))
    return out


def detect_accelerations(t: np.ndarray, rr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Onset times (s) of heart-rate accelerations: excess = HR - baseline rises above
    ONSET_BPM and falls back to <= RETURN_BPM within MAX_EVENT_S of the onset.

    Scans valid beats with a baseline in time order. State machine:
      idle     -> armed-event at a beat with excess > ONSET_BPM (onset = that beat's time)
      in event -> counted, back to idle, at the first beat with excess <= RETURN_BPM if
                  t - onset <= MAX_EVENT_S; if t - onset > MAX_EVENT_S first, the event is
                  dropped and the detector waits for excess <= RETURN_BPM before re-arming.
    A gap > MAX_GAP_S between consecutive usable beats (or an unusable beat) resets to idle.
    """
    base = hr_baseline(t, rr, valid)
    onsets = []
    state = 0  # 0 idle, 1 in event, 2 waiting for return after a too-long event
    onset = 0.0
    last_t = -math.inf
    for i in range(len(rr)):
        if not (valid[i] and math.isfinite(base[i])):
            state = 0
            continue
        ti = float(t[i])
        if ti - last_t > MAX_GAP_S:
            state = 0
        last_t = ti
        ex = 60000.0 / rr[i] - base[i]
        if state == 0:
            if ex > ONSET_BPM:
                state, onset = 1, ti
        elif state == 1:
            if ti - onset > MAX_EVENT_S:
                state = 2 if ex > RETURN_BPM else 0
            elif ex <= RETURN_BPM:
                onsets.append(onset)
                state = 0
        elif ex <= RETURN_BPM:
            state = 0
    return np.array(onsets, dtype=np.float64)


def _long_window(t: np.ndarray, rr: np.ndarray, valid: np.ndarray, a: float, b: float,
                 onsets: np.ndarray) -> dict:
    out = {f"{k}_w20": v for k, v in F._window_stats(t, rr, valid, a, b).items() if k in LONG_BASE}
    for k in ("log_cvhr_lo", "log_cvhr_hi", "cvhr_lo_ratio", "cvhr_hi_ratio", "cvhr_index",
              "hr_slope_w20"):
        out[k] = math.nan
    cov = out["coverage_w20"]
    if cov >= MIN_COVERAGE_LONG:
        j0, j1 = _range(onsets, a, b)
        out["cvhr_index"] = (j1 - j0) / (cov * (b - a) / 600.0)
    i0, i1 = _range(t, a, b)
    v = np.arange(i0, i1)[valid[i0:i1]]
    if len(v) < MIN_BEATS or cov < MIN_COVERAGE_LONG:
        return out
    tv, xv = t[v], rr[v]
    p = F.lomb_power(tv, xv, CVHR_FREQS)
    lo = float(p[LO_IDX[0]:LO_IDX[1]].sum())
    hi = float(p[HI_IDX[0]:HI_IDX[1]].sum())
    tot = float(p.sum())
    out["log_cvhr_lo"] = math.log(lo + 1e-6)
    out["log_cvhr_hi"] = math.log(hi + 1e-6)
    if tot > 0:
        out["cvhr_lo_ratio"] = lo / tot
        out["cvhr_hi_ratio"] = hi / tot
    hr = 60000.0 / xv
    tm = tv / 60.0
    tmm = float(tm.mean())
    den = float(((tm - tmm) ** 2).sum())
    if den > 0:
        out["hr_slope_w20"] = float(((tm - tmm) * (hr - float(hr.mean()))).sum()) / den
    return out


def percentile_rank(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(#ref < x + 0.5 * #ref == x) / #ref over finite ref values; NaN for NaN x or empty ref."""
    r = np.sort(ref[np.isfinite(ref)])
    out = np.full(len(x), math.nan)
    n = len(r)
    if n == 0:
        return out
    for i in range(len(x)):
        if not math.isfinite(x[i]):
            continue
        lt = int(np.searchsorted(r, x[i], side="left"))
        le = int(np.searchsorted(r, x[i], side="right"))
        out[i] = (lt + 0.5 * (le - lt)) / n
    return out


def robust_z(x: np.ndarray, good: np.ndarray) -> np.ndarray:
    ref = x[good]
    med = F._nanmedian(ref)
    return np.clip((x - med) / F._robust_scale(ref, med), -F.Z_CLIP, F.Z_CLIP)


def long_features(t: np.ndarray, rr: np.ndarray, valid: np.ndarray, epoch_start_s: np.ndarray,
                  epoch_len_s: float, cfg_rr: dict) -> dict[str, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    rr = np.asarray(rr, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    starts = np.asarray(epoch_start_s, dtype=np.float64)
    n = len(starts)
    onsets = detect_accelerations(t, rr, valid)
    out = {k: np.full(n, math.nan) for k in NAMES}
    mean_w1 = np.full(n, math.nan)
    rmssd_w1 = np.full(n, math.nan)
    cov_w1 = np.zeros(n)
    for k in range(n):
        s = float(starts[k])
        c = s + epoch_len_s / 2
        w1 = F._window_stats(t, rr, valid, s, s + epoch_len_s)
        mean_w1[k], rmssd_w1[k], cov_w1[k] = w1["mean_rr"], w1["rmssd"], w1["coverage"]
        for name, val in _long_window(t, rr, valid, c - HALF_W20, c + HALF_W20, onsets).items():
            out[name][k] = val
    good = cov_w1 >= cfg_rr["min_epoch_coverage"]
    out["pct_mean_rr_w1"] = percentile_rank(mean_w1, mean_w1[good])
    out["pct_rmssd_w1"] = percentile_rank(rmssd_w1, rmssd_w1[good])
    t_last = float(starts[-1]) if n else 0.0
    out["hours_to_end"] = (t_last - starts) / 3600.0
    for name in NAMES:
        if name not in Z_EXCLUDE:
            out[f"z_{name}"] = robust_z(out[name], good)
    return out
