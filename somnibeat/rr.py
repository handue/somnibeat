"""RR-interval construction and artifact rules.

These rules are mirrored exactly in web/js/features.js and covered by the
Python/JavaScript parity test. See docs/rr_rules.md for the rationale.

Input: beat times in seconds (strictly increasing).
Output: arrays aligned per interval i (between beat i-1 and beat i):
    t_s    time of the closing beat (s)
    rr_ms  interval length (ms)
    valid  bool, after the rules below

Rules (applied in order):
R1 range:   rr < min_rr_ms or rr > max_rr_ms  -> invalid
            (300-2000 ms = 30-200 bpm; catches missed and doubled beats at extremes)
R2 local:   m = median of range-valid rr in a centered window of `local_median_beats`
            intervals; |rr - m| / m > max_relative_jump -> invalid
            (catches a single missed beat, ~2x, or a spurious extra beat, ~0.5x)
R3 no interpolation: invalid intervals are dropped, never filled in.
Successive differences (RMSSD, pNN50) only use pairs of adjacent intervals
that are both valid.
"""

from __future__ import annotations

import numpy as np


def rr_from_beats(beat_t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(beat_t_s, dtype=np.float64)
    return t[1:], np.diff(t) * 1000.0


def clean_rr(t_s: np.ndarray, rr_ms: np.ndarray, cfg: dict) -> np.ndarray:
    rr = np.asarray(rr_ms, dtype=np.float64)
    n = len(rr)
    valid = (rr >= cfg["min_rr_ms"]) & (rr <= cfg["max_rr_ms"]) & np.isfinite(rr)
    half = int(cfg["local_median_beats"]) // 2
    jump = float(cfg["max_relative_jump"])
    out = valid.copy()
    for i in range(n):
        if not valid[i]:
            continue
        a, b = max(0, i - half), min(n, i + half + 1)
        w = rr[a:b][valid[a:b]]
        m = float(np.median(w))
        if abs(rr[i] - m) / m > jump:
            out[i] = False
    return out


def rr_series_from_csv_rows(time_s: np.ndarray | None, rr_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Uploaded CSV: time_s is the time of the closing beat. If absent, use cumulative RR."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if time_s is None:
        t = np.cumsum(rr) / 1000.0
    else:
        t = np.asarray(time_s, dtype=np.float64)
    return t, rr
