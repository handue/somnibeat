"""Epoch features from a cleaned RR series.

Mirrored exactly in web/js/features.js (parity-tested). Keep every operation
explicit and order-stable so both implementations agree to ~1e-9.

For each epoch k starting at s_k (length L, 30 s for staging, 60 s for
apnea minutes), base features are computed on three centered windows:
    w1 = [s_k, s_k + L)                      (the epoch itself)
    w5 = epoch center +- 75 s  (150 s)
    w10 = epoch center +- 150 s (300 s)
An interval belongs to a window if its closing beat time is inside it.

Base features per window: mean_rr, sdnn, rmssd, pnn50, cv, coverage.
Spectral (w10 only): Lomb-Scargle power on a fixed grid, log band powers
VLF 0.01-0.05 Hz, LF 0.05-0.15 Hz, HF 0.15-0.40 Hz, log LF/HF, HF peak frequency.
DFA alpha1 (w10 only): box sizes 4..16 beats.

Then, per record: robust z-score of every base feature
(value - median) / scale over epochs with coverage_w1 >= min_epoch_coverage,
scale = IQR, else 1.4826 MAD, else SD, else 1; clipped to +-10;
neighbor differences of key z-features; centered rolling means; position in
the night (hours since the first epoch of the grid).
"""

from __future__ import annotations

import math

import numpy as np

WINDOWS = {"w1": None, "w5": 75.0, "w10": 150.0}  # half-widths around the epoch center
BASE = ["mean_rr", "sdnn", "rmssd", "pnn50", "cv", "coverage"]
SPECTRAL = ["log_vlf", "log_lf", "log_hf", "log_lf_hf", "hf_peak_hz", "dfa_a1"]
KEY = ["mean_rr_w1", "sdnn_w1", "rmssd_w1", "mean_rr_w5", "log_hf", "dfa_a1", "log_lf_hf", "sdnn_w5"]
Z_CLIP = 10.0
FREQS = np.round(np.arange(0.01, 0.4001, 0.005), 6)  # 79 frequencies


def _window_stats(t: np.ndarray, rr: np.ndarray, valid: np.ndarray, a: float, b: float) -> dict:
    i0 = int(np.searchsorted(t, a, side="left"))
    i1 = int(np.searchsorted(t, b, side="left"))
    idx = np.arange(i0, i1)
    v = idx[valid[i0:i1]]
    out = dict.fromkeys(BASE, math.nan)
    cov = float(rr[v].sum()) / 1000.0 / (b - a) if len(v) else 0.0
    out["coverage"] = min(cov, 1.0)
    if len(v) < 3:
        return out
    x = rr[v]
    m = float(x.mean())
    sd = float(math.sqrt(((x - m) ** 2).sum() / (len(x) - 1)))
    # successive differences: both intervals valid and adjacent
    pairs = idx[1:][valid[i0 + 1:i1] & valid[i0:i1 - 1]] if i1 - i0 > 1 else np.zeros(0, int)
    if len(pairs) >= 2:
        d = rr[pairs] - rr[pairs - 1]
        rmssd = float(math.sqrt((d * d).mean()))
        pnn50 = float((np.abs(d) > 50.0).mean())
    else:
        rmssd = pnn50 = math.nan
    out.update(mean_rr=m, sdnn=sd, rmssd=rmssd, pnn50=pnn50, cv=sd / m)
    return out


def lomb_power(t: np.ndarray, y: np.ndarray, freqs: np.ndarray = FREQS) -> np.ndarray:
    """Classical Lomb-Scargle periodogram divided by N (units of variance)."""
    y = y - y.mean()
    p = np.empty(len(freqs))
    for j, f in enumerate(freqs):
        w = 2.0 * math.pi * f
        s2 = float(np.sin(2 * w * t).sum())
        c2 = float(np.cos(2 * w * t).sum())
        tau = math.atan2(s2, c2) / (2 * w)
        c = np.cos(w * (t - tau))
        s = np.sin(w * (t - tau))
        cc = float((c * c).sum())
        ss = float((s * s).sum())
        yc = float((y * c).sum())
        ys = float((y * s).sum())
        p[j] = 0.5 * ((yc * yc / cc if cc > 0 else 0.0) + (ys * ys / ss if ss > 0 else 0.0)) / len(t)
    return p


def dfa_alpha1(x: np.ndarray) -> float:
    n = len(x)
    if n < 32:
        return math.nan
    y = np.cumsum(x - x.mean())
    logn, logf = [], []
    for s in range(4, 17):
        nb = n // s
        if nb < 2:
            continue
        k = np.arange(s, dtype=np.float64)
        km = k.mean()
        kvar = float(((k - km) ** 2).sum())
        res = 0.0
        for b in range(nb):
            seg = y[b * s:(b + 1) * s]
            sm = float(seg.mean())
            slope = float(((k - km) * (seg - sm)).sum()) / kvar
            r = seg - (sm + slope * (k - km))
            res += float((r * r).sum())
        F = math.sqrt(res / (nb * s))
        if F > 0:
            logn.append(math.log(s))
            logf.append(math.log(F))
    if len(logn) < 3:
        return math.nan
    ln, lf = np.array(logn), np.array(logf)
    return float(((ln - ln.mean()) * (lf - lf.mean())).sum() / ((ln - ln.mean()) ** 2).sum())


def _spectral(t: np.ndarray, rr: np.ndarray, valid: np.ndarray, a: float, b: float) -> dict:
    i0 = int(np.searchsorted(t, a, side="left"))
    i1 = int(np.searchsorted(t, b, side="left"))
    v = np.arange(i0, i1)[valid[i0:i1]]
    out = dict.fromkeys(SPECTRAL, math.nan)
    if len(v) < 32 or float(rr[v].sum()) / 1000.0 < 0.5 * (b - a):
        return out
    p = lomb_power(t[v], rr[v])

    def band(lo, hi):
        m = (FREQS >= lo - 1e-9) & (FREQS < hi - 1e-9)
        return float(p[m].sum())

    vlf, lf, hf = band(0.01, 0.05), band(0.05, 0.15), band(0.15, 0.40 + 0.005)
    hfm = (FREQS >= 0.15 - 1e-9)
    out.update(log_vlf=math.log(vlf + 1e-6), log_lf=math.log(lf + 1e-6), log_hf=math.log(hf + 1e-6),
               log_lf_hf=math.log((lf + 1e-6) / (hf + 1e-6)),
               hf_peak_hz=float(FREQS[hfm][int(np.argmax(p[hfm]))]),
               dfa_a1=dfa_alpha1(rr[v]))
    return out


def base_features(t: np.ndarray, rr: np.ndarray, valid: np.ndarray,
                  epoch_start_s: np.ndarray, epoch_len_s: float) -> dict[str, np.ndarray]:
    names = [f"{b}_{w}" for w in WINDOWS for b in BASE] + SPECTRAL
    out = {k: np.full(len(epoch_start_s), math.nan) for k in names}
    for k, s in enumerate(epoch_start_s):
        c = s + epoch_len_s / 2
        for w, half in WINDOWS.items():
            a, b = (s, s + epoch_len_s) if half is None else (c - half, c + half)
            st = _window_stats(t, rr, valid, a, b)
            for name, val in st.items():
                out[f"{name}_{w}"][k] = val
        sp = _spectral(t, rr, valid, c - 150.0, c + 150.0)
        for name, val in sp.items():
            out[name][k] = val
    return out


def _nanmedian(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else math.nan


def _nanpct(x: np.ndarray, q: float) -> float:
    x = np.sort(x[np.isfinite(x)])
    if not len(x):
        return math.nan
    pos = (len(x) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return float(x[lo] + (x[hi] - x[lo]) * (pos - lo))


def _robust_scale(ref: np.ndarray, med: float) -> float:
    """IQR, else 1.4826 x MAD, else SD, else 1.0 (degenerate features such as an all-zero pNN50)."""
    iqr = _nanpct(ref, 0.75) - _nanpct(ref, 0.25)
    if math.isfinite(iqr) and iqr > 1e-9:
        return iqr
    mad = 1.4826 * _nanmedian(np.abs(ref - med))
    if math.isfinite(mad) and mad > 1e-9:
        return mad
    r = ref[np.isfinite(ref)]
    if len(r) > 1:
        sd = float(math.sqrt(((r - r.mean()) ** 2).sum() / (len(r) - 1)))
        if sd > 1e-9:
            return sd
    return 1.0


def _rolling_nanmean(x: np.ndarray, win: int) -> np.ndarray:
    half = win // 2
    out = np.full(len(x), math.nan)
    for i in range(len(x)):
        seg = x[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = float(seg.mean())
    return out


def epoch_features(t: np.ndarray, rr: np.ndarray, valid: np.ndarray, epoch_start_s: np.ndarray,
                   epoch_len_s: float, cfg_rr: dict, cfg_feat: dict) -> dict[str, np.ndarray]:
    base = base_features(t, rr, valid, epoch_start_s, epoch_len_s)
    good = base["coverage_w1"] >= cfg_rr["min_epoch_coverage"]
    feats = dict(base)
    for name, x in base.items():
        if name.startswith("coverage"):
            continue
        ref = x[good]
        med = _nanmedian(ref)
        feats[f"z_{name}"] = np.clip((x - med) / _robust_scale(ref, med), -Z_CLIP, Z_CLIP)
    n = len(epoch_start_s)
    for key in KEY:
        z = feats[f"z_{key}"]
        for o in cfg_feat["neighbor_offsets"]:
            sh = np.full(n, math.nan)
            if abs(o) < n:  # shorter grids: the neighbor is off the grid -> NaN
                if o > 0:
                    sh[:n - o] = z[o:]
                else:
                    sh[-o:] = z[:n + o]
            feats[f"d{o:+d}_{key}"] = z - sh
        for win in cfg_feat["rolling_windows_epochs"]:
            feats[f"roll{win}_{key}"] = _rolling_nanmean(z, win)
    t0 = epoch_start_s[0] if n else 0.0
    # Position in the night is measured from the first epoch of the grid (lights-off in the demo,
    # start of scoring in the training data). No feature depends on the total night length.
    feats["hours_since_start"] = (epoch_start_s - t0) / 3600.0
    feats["low_quality"] = (~good).astype(float)
    return feats
