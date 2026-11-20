// Long-context RR features for v2 (extra columns next to the v1 feature tables).
// Line-for-line mirror of somnibeat/features_v2.py (parity-tested to 1e-9 relative in
// tests/test_parity_features_v2.py). Reuses the numpy-exact primitives of web/js/features.js
// (pairwise npSum, searchsorted, np.median, window statistics, Lomb-Scargle, robust scale).
//
// For each epoch k starting at s_k (length L), with center c = s_k + L/2:
//     w20 = [c - 300, c + 300)   (600 s; 20 staging epochs)
// An interval belongs to a window if its closing beat time is inside it (as in v1).
// See somnibeat/features_v2.py for the definition of every feature.

import {
  npSum, searchsortedLeft, npMedian, windowStats, nanmedian, robustScale, clip, Z_CLIP,
} from "./features.js";

export const HALF_W20 = 300.0;
export const LONG_BASE = ["mean_rr", "sdnn", "rmssd", "cv", "coverage"];
// np.arange(4, 161) * 0.0025: integer k times 0.0025 (one correctly rounded product, same in JS).
export const CVHR_FREQS = (() => {
  const out = new Float64Array(157);
  for (let i = 0; i < 157; i++) out[i] = (i + 4) * 0.0025;
  return out;
})();
export const LO_IDX = [0, 12];   // k 4..15  -> 0.0100 .. 0.0375 Hz
export const HI_IDX = [12, 36];  // k 16..39 -> 0.0400 .. 0.0975 Hz
export const MIN_BEATS = 32;
export const MIN_COVERAGE_LONG = 0.5;

// CVHR acceleration detector
export const BASELINE_HALF_S = 30.0;   // 60 s centered moving median of HR
export const BASELINE_MIN_BEATS = 10;  // fewer valid beats in the baseline window -> no baseline (gap)
export const ONSET_BPM = 10.0;         // HR - baseline > 10 bpm starts an acceleration
export const RETURN_BPM = 5.0;         // HR - baseline <= 5 bpm is the return
export const MAX_EVENT_S = 60.0;       // return must follow the onset within 60 s, else not counted
export const MAX_GAP_S = 3.0;          // a gap between consecutive valid beats resets the detector

export const Z_EXCLUDE = new Set(["coverage_w20", "pct_mean_rr_w1", "pct_rmssd_w1", "hours_to_end"]);
export const NAMES = LONG_BASE.map((b) => `${b}_w20`).concat(
  ["log_cvhr_lo", "log_cvhr_hi", "cvhr_lo_ratio", "cvhr_hi_ratio", "cvhr_index",
    "hr_slope_w20", "pct_mean_rr_w1", "pct_rmssd_w1", "hours_to_end"]);

/** Output column names in the Python dict order: NAMES, then z_<name> for every non-excluded name. */
export function longFeatureNames() {
  return NAMES.concat(NAMES.filter((n) => !Z_EXCLUDE.has(n)).map((n) => `z_${n}`));
}

/** np.searchsorted(arr, key, side="right") for one scalar key (numpy order: NaN sorts last). */
export function searchsortedRight(arr, key) {
  let lo = 0;
  let hi = arr.length;
  const keyNaN = key !== key;
  while (lo < hi) {
    const mid = lo + ((hi - lo) >> 1);
    const v = arr[mid];
    // numpy float less-or-equal for side="right": !(key < v)
    if (!(key < v || (v !== v && !keyNaN))) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Indices [i0, i1) of intervals whose closing beat lies in [a, b) (binary search). */
function range(t, a, b) {
  return [searchsortedLeft(t, a), searchsortedLeft(t, b)];
}

/**
 * Per-interval moving median of valid HR (bpm) over [t_i - 30, t_i + 30); NaN if invalid or
 * fewer than BASELINE_MIN_BEATS valid beats in the window (np.median semantics).
 */
export function hrBaseline(t, rr, valid) {
  const n = rr.length;
  const hr = new Float64Array(n).fill(NaN);
  for (let i = 0; i < n; i++) if (valid[i]) hr[i] = 60000.0 / rr[i];
  const out = new Float64Array(n).fill(NaN);
  let w = new Float64Array(256);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const [i0, i1] = range(t, t[i] - BASELINE_HALF_S, t[i] + BASELINE_HALF_S);
    if (w.length < i1 - i0) w = new Float64Array(i1 - i0);
    let m = 0;
    for (let j = i0; j < i1; j++) if (valid[j]) w[m++] = hr[j];
    if (m >= BASELINE_MIN_BEATS) out[i] = npMedian(w.subarray(0, m));
  }
  return out;
}

/**
 * Onset times (s) of heart-rate accelerations: excess = HR - baseline rises above ONSET_BPM and
 * falls back to <= RETURN_BPM within MAX_EVENT_S of the onset (state machine of features_v2.py).
 */
export function detectAccelerations(t, rr, valid) {
  const base = hrBaseline(t, rr, valid);
  const onsets = [];
  let state = 0; // 0 idle, 1 in event, 2 waiting for return after a too-long event
  let onset = 0.0;
  let lastT = -Infinity;
  for (let i = 0; i < rr.length; i++) {
    if (!(valid[i] && Number.isFinite(base[i]))) {
      state = 0;
      continue;
    }
    const ti = t[i];
    if (ti - lastT > MAX_GAP_S) state = 0;
    lastT = ti;
    const ex = 60000.0 / rr[i] - base[i];
    if (state === 0) {
      if (ex > ONSET_BPM) { state = 1; onset = ti; }
    } else if (state === 1) {
      if (ti - onset > MAX_EVENT_S) {
        state = ex > RETURN_BPM ? 2 : 0;
      } else if (ex <= RETURN_BPM) {
        onsets.push(onset);
        state = 0;
      }
    } else if (ex <= RETURN_BPM) {
      state = 0;
    }
  }
  return Float64Array.from(onsets);
}

// ---------------------------------------------------------------- Lomb-Scargle with a trig cache

// The 600 s windows of neighbouring epochs overlap (20 staging epochs per window), so the
// per-beat terms sin(2 w t_i) and cos(2 w t_i) of features.lomb_power are recomputed many times
// per night. lombPowerCached evaluates exactly the same expressions (same arguments, same
// Math.sin/Math.cos calls, same pairwise sums in the same order) as web/js/features.js lombPower,
// but memoizes those two terms per global beat index in a direct-mapped cache. The result is
// bitwise identical to lombPower(t[idx], y[idx], CVHR_FREQS); only the work is shared.
const NF = CVHR_FREQS.length;
const W2 = Float64Array.from(CVHR_FREQS, (f) => 2 * (2.0 * Math.PI * f));
let TC = null; // {t, slots, tag: Int32Array, sin: Float64Array, cos: Float64Array}

// `span` is the index span of the requested beats (last - first + 1): with slots >= span no two
// beats of one window can share a slot (indices i and i + slots never both occur), so the
// direct-mapped cache never returns another beat's terms. Sizing by the number of valid beats
// alone is not enough: invalid beats inside a window widen the span.
function trigCache(t, span) {
  let slots = 4096;
  while (slots < span) slots *= 2;
  if (!TC || TC.t !== t || TC.slots < slots) {
    TC = { t, slots, tag: new Int32Array(slots).fill(-1),
      sin: new Float64Array(slots * NF), cos: new Float64Array(slots * NF) };
  }
  return TC;
}

let LS = { n: 0 };
function lombScratch(n) {
  if (LS.n < n) {
    const m = Math.max(n, 1024);
    LS = { n: m, t: new Float64Array(m), yy: new Float64Array(m), sinA: new Float64Array(m),
      cosA: new Float64Array(m), cA: new Float64Array(m), sA: new Float64Array(m) };
  }
  return LS;
}

/** lomb_power(t[idx], y[idx], CVHR_FREQS) for the beat indices idx[0 .. n) of the full series t. */
export function lombPowerCached(t, y, idx, n) {
  const C = trigCache(t, n ? idx[n - 1] - idx[0] + 1 : 0);
  const mask = C.slots - 1;
  for (let q = 0; q < n; q++) {
    const i = idx[q];
    const slot = i & mask;
    if (C.tag[slot] === i) continue;
    C.tag[slot] = i;
    const o = slot * NF;
    const ti = t[i];
    for (let j = 0; j < NF; j++) {
      const arg = W2[j] * ti;
      C.sin[o + j] = Math.sin(arg);
      C.cos[o + j] = Math.cos(arg);
    }
  }
  const S = lombScratch(n);
  const tv = S.t, yy = S.yy, sinA = S.sinA, cosA = S.cosA, cA = S.cA, sA = S.sA;
  for (let q = 0; q < n; q++) { tv[q] = t[idx[q]]; yy[q] = y[idx[q]]; }
  const ym = npSum(yy, 0, n) / n;
  for (let q = 0; q < n; q++) yy[q] = yy[q] - ym;
  const p = new Float64Array(NF);
  for (let j = 0; j < NF; j++) {
    const w = 2.0 * Math.PI * CVHR_FREQS[j];
    for (let q = 0; q < n; q++) {
      const o = (idx[q] & mask) * NF + j;
      sinA[q] = C.sin[o];
      cosA[q] = C.cos[o];
    }
    const s2 = npSum(sinA, 0, n);
    const c2 = npSum(cosA, 0, n);
    const tau = Math.atan2(s2, c2) / (2 * w);
    for (let q = 0; q < n; q++) {
      const arg = w * (tv[q] - tau);
      cA[q] = Math.cos(arg);
      sA[q] = Math.sin(arg);
    }
    for (let q = 0; q < n; q++) { const c = cA[q]; sinA[q] = c * c; cosA[q] = yy[q] * c; }
    const cc = npSum(sinA, 0, n);
    const yc = npSum(cosA, 0, n);
    for (let q = 0; q < n; q++) { const s = sA[q]; sinA[q] = s * s; cosA[q] = yy[q] * s; }
    const ss = npSum(sinA, 0, n);
    const ys = npSum(cosA, 0, n);
    p[j] = 0.5 * ((cc > 0 ? yc * yc / cc : 0.0) + (ss > 0 ? ys * ys / ss : 0.0)) / n;
  }
  return p;
}

const ST = {};

function longWindow(t, rr, valid, a, b, onsets, out) {
  windowStats(t, rr, valid, a, b, ST);
  for (const k of LONG_BASE) out[`${k}_w20`] = ST[k];
  out.log_cvhr_lo = out.log_cvhr_hi = out.cvhr_lo_ratio = out.cvhr_hi_ratio = NaN;
  out.cvhr_index = out.hr_slope_w20 = NaN;
  const cov = out.coverage_w20;
  if (cov >= MIN_COVERAGE_LONG) {
    const [j0, j1] = range(onsets, a, b);
    out.cvhr_index = (j1 - j0) / (cov * (b - a) / 600.0);
  }
  const [i0, i1] = range(t, a, b);
  let nv = 0;
  for (let i = i0; i < i1; i++) if (valid[i]) nv++;
  if (nv < MIN_BEATS || cov < MIN_COVERAGE_LONG) return out;
  const idx = new Int32Array(nv), tv = new Float64Array(nv), xv = new Float64Array(nv);
  let q = 0;
  for (let i = i0; i < i1; i++) if (valid[i]) { idx[q] = i; tv[q] = t[i]; xv[q] = rr[i]; q++; }
  const p = lombPowerCached(t, rr, idx, nv); // == features.lomb_power(tv, xv, CVHR_FREQS)
  const lo = npSum(p, LO_IDX[0], LO_IDX[1] - LO_IDX[0]);
  const hi = npSum(p, HI_IDX[0], HI_IDX[1] - HI_IDX[0]);
  const tot = npSum(p, 0, p.length);
  out.log_cvhr_lo = Math.log(lo + 1e-6);
  out.log_cvhr_hi = Math.log(hi + 1e-6);
  if (tot > 0) {
    out.cvhr_lo_ratio = lo / tot;
    out.cvhr_hi_ratio = hi / tot;
  }
  const hr = new Float64Array(nv), tm = new Float64Array(nv);
  for (let i = 0; i < nv; i++) { hr[i] = 60000.0 / xv[i]; tm[i] = tv[i] / 60.0; }
  const tmm = npSum(tm, 0, nv) / nv;
  const hrm = npSum(hr, 0, nv) / nv;
  const sq = new Float64Array(nv), cr = new Float64Array(nv);
  for (let i = 0; i < nv; i++) {
    const d = tm[i] - tmm;
    sq[i] = d * d;
    cr[i] = d * (hr[i] - hrm);
  }
  const den = npSum(sq, 0, nv);
  if (den > 0) out.hr_slope_w20 = npSum(cr, 0, nv) / den;
  return out;
}

/** (#ref < x + 0.5 * #ref == x) / #ref over finite ref values; NaN for NaN x or empty ref. */
export function percentileRank(x, ref) {
  const fin = [];
  for (let i = 0; i < ref.length; i++) if (Number.isFinite(ref[i])) fin.push(ref[i]);
  const r = Float64Array.from(fin).sort();
  const out = new Float64Array(x.length).fill(NaN);
  const n = r.length;
  if (n === 0) return out;
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i])) continue;
    const lt = searchsortedLeft(r, x[i]);
    const le = searchsortedRight(r, x[i]);
    out[i] = (lt + 0.5 * (le - lt)) / n;
  }
  return out;
}

/** Per-night robust z (v1 rule), clipped to +-Z_CLIP. */
export function robustZ(x, good) {
  const refList = [];
  for (let i = 0; i < x.length; i++) if (good[i]) refList.push(x[i]);
  const ref = Float64Array.from(refList);
  const med = nanmedian(ref);
  const scale = robustScale(ref, med);
  const z = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) z[i] = clip((x[i] - med) / scale, -Z_CLIP, Z_CLIP);
  return z;
}

/**
 * long_features(t, rr, valid, epoch_start_s, epoch_len_s, cfg_rr).
 * Returns {name: Float64Array} with exactly the Python names, in the same insertion order.
 */
export function longFeatures(t, rr, valid, epochStarts, epochLen, cfgRR) {
  t = Float64Array.from(t);
  rr = Float64Array.from(rr);
  valid = Uint8Array.from(valid, (v) => (v ? 1 : 0));
  const starts = Float64Array.from(epochStarts);
  const n = starts.length;
  const onsets = detectAccelerations(t, rr, valid);
  const out = {};
  for (const k of NAMES) out[k] = new Float64Array(n).fill(NaN);
  const meanW1 = new Float64Array(n).fill(NaN);
  const rmssdW1 = new Float64Array(n).fill(NaN);
  const covW1 = new Float64Array(n);
  const w1 = {}, lw = {};
  for (let k = 0; k < n; k++) {
    const s = starts[k];
    const c = s + epochLen / 2;
    windowStats(t, rr, valid, s, s + epochLen, w1);
    meanW1[k] = w1.mean_rr; rmssdW1[k] = w1.rmssd; covW1[k] = w1.coverage;
    longWindow(t, rr, valid, c - HALF_W20, c + HALF_W20, onsets, lw);
    for (const [name, val] of Object.entries(lw)) out[name][k] = val;
  }
  const good = new Uint8Array(n);
  for (let i = 0; i < n; i++) good[i] = covW1[i] >= cfgRR.min_epoch_coverage ? 1 : 0;
  const sel = (x) => { const o = []; for (let i = 0; i < n; i++) if (good[i]) o.push(x[i]); return Float64Array.from(o); };
  out.pct_mean_rr_w1 = percentileRank(meanW1, sel(meanW1));
  out.pct_rmssd_w1 = percentileRank(rmssdW1, sel(rmssdW1));
  const tLast = n ? starts[n - 1] : 0.0;
  const hte = new Float64Array(n);
  for (let i = 0; i < n; i++) hte[i] = (tLast - starts[i]) / 3600.0;
  out.hours_to_end = hte;
  for (const name of NAMES) if (!Z_EXCLUDE.has(name)) out[`z_${name}`] = robustZ(out[name], good);
  return out;
}
