"""R-peak detection (offline, Python only).

The demo never sees raw ECG: it receives RR intervals. This module turns ECG
into beat times for training and validation.

Algorithm (energy-envelope detector, Pan-Tompkins style):
1. Zero-phase Butterworth band-pass 5-25 Hz (upper edge capped at 0.45 fs).
2. Squared first difference, smoothed with a 120 ms moving average (centered,
   so no group delay).
3. Adaptive threshold: 0.3 x a running median (over 5 blocks) of the 98th
   percentile of the envelope in 2 s blocks, floored at 0.25 x the recording
   median of those block values. Blocks below 0.1 x the recording median are
   treated as no signal and yield no beats.
4. Candidate peaks on the envelope with a 250 ms refractory period.
5. Search-back: in any gap longer than 1.66 x the running median RR, accept
   the largest envelope peak above half the threshold; repeat (up to 5
   passes) until no gap gains a beat.
6. Each beat is moved to the extremum of the band-passed ECG within +-75 ms,
   using the dominant polarity of the recording, then refined with parabolic
   interpolation to sub-sample precision.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, sosfiltfilt


def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    hi = min(25.0, 0.45 * fs)
    sos = butter(2, [5.0, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def _envelope(f: np.ndarray, fs: float) -> np.ndarray:
    d = np.diff(f, prepend=f[0])
    e = d * d
    w = max(1, int(round(0.12 * fs)))
    return np.convolve(e, np.ones(w) / w, mode="same")


def _threshold(env: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-sample threshold, per-sample no-signal mask)."""
    blk = int(2 * fs)
    nb = max(1, len(env) // blk)
    p_raw = np.array([np.percentile(env[i * blk:(i + 1) * blk], 98) for i in range(nb)])
    rec = float(np.median(p_raw))
    p = np.maximum(median_filter(p_raw, size=5, mode="nearest"), 0.25 * rec)
    centers = (np.arange(nb) + 0.5) * blk
    thr = 0.3 * np.interp(np.arange(len(env)), centers, p)
    dead_blk = p_raw < 0.1 * rec  # lead-off / flat line: no beats are accepted here
    dead = np.repeat(dead_blk, blk)
    dead = np.concatenate([dead, np.full(len(env) - len(dead), dead_blk[-1])])
    return thr, dead


def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Return R-peak times in seconds (float, sub-sample precision)."""
    x = np.asarray(ecg, dtype=np.float64)
    x = np.nan_to_num(x - np.nanmedian(x))
    f = _bandpass(x, fs)
    env = _envelope(f, fs)
    thr, dead = _threshold(env, fs)
    refractory = int(0.25 * fs)
    cand, _ = find_peaks(env, distance=refractory)
    cand = cand[~dead[cand]]
    peaks = cand[env[cand] > thr[cand]]

    # Search-back for missed beats in long gaps; repeat until nothing new is added (max 5 passes).
    low = cand[env[cand] > 0.5 * thr[cand]]
    for _ in range(5):
        if len(peaks) <= 8:
            break
        rr = np.diff(peaks)
        med = median_filter(rr.astype(float), size=9, mode="nearest")
        extra = []
        for i in np.where(rr > 1.66 * med)[0]:
            a, b = peaks[i] + refractory, peaks[i + 1] - refractory
            inside = low[(low > a) & (low < b)]
            if len(inside):
                extra.append(inside[np.argmax(env[inside])])
        if not extra:
            break
        peaks = np.sort(np.concatenate([peaks, extra]))

    # Move to the ECG extremum with the recording's dominant polarity.
    half = int(0.075 * fs)
    n = len(f)
    lo = np.clip(peaks - half, 0, n - 1)
    hi = np.clip(peaks + half + 1, 1, n)
    amax = np.array([f[a:b].max() for a, b in zip(lo, hi)])
    amin = np.array([-f[a:b].min() for a, b in zip(lo, hi)])
    sign = 1.0 if np.median(amax) >= np.median(amin) else -1.0
    g = sign * f
    idx = np.array([a + int(np.argmax(g[a:b])) for a, b in zip(lo, hi)], dtype=int)
    idx = np.unique(idx)
    # Enforce refractory after relocation (keep the larger of two close peaks).
    keep = [idx[0]] if len(idx) else []
    for k in idx[1:]:
        if k - keep[-1] < refractory:
            if g[k] > g[keep[-1]]:
                keep[-1] = k
        else:
            keep.append(k)
    idx = np.array(keep, dtype=int)
    # Parabolic interpolation.
    t = idx.astype(float)
    ok = (idx > 0) & (idx < n - 1)
    y0, y1, y2 = g[idx[ok] - 1], g[idx[ok]], g[idx[ok] + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    t[ok] += np.clip(delta, -0.5, 0.5)
    return t / fs
