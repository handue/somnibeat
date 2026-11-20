// Epoch features from a cleaned RR series.
// Line-for-line mirror of somnibeat/features.py (parity-tested to 1e-9 relative in
// tests/test_parity_features.py). Every float reduction reproduces numpy semantics:
// sums use numpy's pairwise summation (np.add.reduce with initial 0.0), mean = sum / n,
// np.median, np.searchsorted(side="left"), np.argmax and percentile interpolation.

export const WINDOWS = [["w1", null], ["w5", 75.0], ["w10", 150.0]]; // half-widths around the epoch center
export const BASE = ["mean_rr", "sdnn", "rmssd", "pnn50", "cv", "coverage"];
export const SPECTRAL = ["log_vlf", "log_lf", "log_hf", "log_lf_hf", "hf_peak_hz", "dfa_a1"];
export const KEY = ["mean_rr_w1", "sdnn_w1", "rmssd_w1", "mean_rr_w5", "log_hf", "dfa_a1", "log_lf_hf", "sdnn_w5"];
export const Z_CLIP = 10.0;

// FREQS = np.round(np.arange(0.01, 0.4001, 0.005), 6): numpy arange fills start + i * delta with
// delta = (start + step) - start; np.round(x, 6) = rint(x * 1e6) / 1e6.
export const FREQS = (() => {
  const start = 0.01, stop = 0.4001, step = 0.005;
  const len = Math.ceil((stop - start) / step);
  const out = new Float64Array(len);
  const delta = (start + step) - start;
  for (let i = 0; i < len; i++) {
    const v = i === 0 ? start : i === 1 ? start + step : start + i * delta;
    out[i] = roundHalfEven(v * 1e6) / 1e6;
  }
  return out;
})();

function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// ---------------------------------------------------------------- numpy primitives

const PW_BLOCKSIZE = 128;

function pairwise(a, off, n) {
  if (n < 8) {
    let res = 0.0;
    for (let i = 0; i < n; i++) res += a[off + i];
    return res;
  }
  if (n <= PW_BLOCKSIZE) {
    let r0 = a[off], r1 = a[off + 1], r2 = a[off + 2], r3 = a[off + 3];
    let r4 = a[off + 4], r5 = a[off + 5], r6 = a[off + 6], r7 = a[off + 7];
    const end = n - (n % 8);
    let i = 8;
    for (; i < end; i += 8) {
      const p = off + i;
      r0 += a[p]; r1 += a[p + 1]; r2 += a[p + 2]; r3 += a[p + 3];
      r4 += a[p + 4]; r5 += a[p + 5]; r6 += a[p + 6]; r7 += a[p + 7];
    }
    let res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
    for (; i < n; i++) res += a[off + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwise(a, off, n2) + pairwise(a, off + n2, n - n2);
}

/** np.sum of a float64 vector (a[off .. off+n)): 0.0 + pairwise_sum. */
export function npSum(a, off = 0, n = a.length - off) {
  return 0.0 + pairwise(a, off, n);
}

/** np.mean: sum / n. */
export function npMean(a, off = 0, n = a.length - off) {
  return npSum(a, off, n) / n;
}

/** np.searchsorted(arr, key, side="left") for one scalar key (numpy binary search, NaN sorts last). */
export function searchsortedLeft(arr, key) {
  let lo = 0;
  let hi = arr.length;
  const keyNaN = key !== key;
  while (lo < hi) {
    const mid = lo + ((hi - lo) >> 1);
    const v = arr[mid];
    // numpy float less-than: a < b || (b != b && a == a)
    if (v < key || (keyNaN && v === v)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** np.median of a finite vector (sorts a copy). */
export function npMedian(x) {
  const n = x.length;
  if (n === 0) return NaN;
  const s = Float64Array.from(x).sort();
  const h = n >> 1;
  if (n % 2 === 1) return s[h] / 1;
  return (0.0 + s[h - 1] + s[h]) / 2;
}

/** np.argmax: first maximum, or the first NaN if any. */
function npArgmax(a) {
  let best = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== a[i]) return i;
    if (a[i] > a[best]) best = i;
  }
  return best;
}

function finiteOnly(x) {
  let k = 0;
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i])) out[k++] = x[i];
  return out.subarray(0, k);
}

// ---------------------------------------------------------------- per-window features

// Scratch buffers grown on demand (avoid allocations in the hot loop).
let SCR = { n: 0, x: null, y: null, a: null, b: null, c: null, d: null, s: null };
function scratch(n) {
  if (SCR.n < n) {
    const m = Math.max(n, 1024);
    SCR = { n: m, x: new Float64Array(m), y: new Float64Array(m), a: new Float64Array(m), b: new Float64Array(m),
      c: new Float64Array(m), d: new Float64Array(m), s: new Float64Array(m) };
  }
  return SCR;
}

export function windowStats(t, rr, valid, a, b, out) {
  const i0 = searchsortedLeft(t, a);
  const i1 = searchsortedLeft(t, b);
  const S = scratch(Math.max(0, i1 - i0));
  const x = S.x;
  let nv = 0;
  for (let i = i0; i < i1; i++) if (valid[i]) x[nv++] = rr[i];
  for (const k of BASE) out[k] = NaN;
  const cov = nv ? npSum(x, 0, nv) / 1000.0 / (b - a) : 0.0;
  out.coverage = Math.min(cov, 1.0);
  if (nv < 3) return out;
  const m = npSum(x, 0, nv) / nv;
  const sq = S.a;
  for (let i = 0; i < nv; i++) { const e = x[i] - m; sq[i] = e * e; }
  const sd = Math.sqrt(npSum(sq, 0, nv) / (nv - 1));
  // successive differences: both intervals valid and adjacent
  const dd = S.b;
  let np = 0, big = 0;
  if (i1 - i0 > 1) {
    for (let j = i0 + 1; j < i1; j++) {
      if (valid[j] && valid[j - 1]) {
        const d = rr[j] - rr[j - 1];
        dd[np++] = d * d;
        if (Math.abs(d) > 50.0) big++;
      }
    }
  }
  let rmssd, pnn50;
  if (np >= 2) {
    rmssd = Math.sqrt(npSum(dd, 0, np) / np);
    pnn50 = (0.0 + big) / np;
  } else {
    rmssd = pnn50 = NaN;
  }
  out.mean_rr = m; out.sdnn = sd; out.rmssd = rmssd; out.pnn50 = pnn50; out.cv = sd / m;
  return out;
}

/** Classical Lomb-Scargle periodogram divided by N (units of variance). */
export function lombPower(t, y, freqs = FREQS) {
  const n = t.length;
  const S = scratch(n);
  const yy = S.y, sinA = S.a, cosA = S.b, cA = S.c, sA = S.d;
  const ym = npSum(y, 0, n) / n;
  for (let i = 0; i < n; i++) yy[i] = y[i] - ym;
  const p = new Float64Array(freqs.length);
  for (let j = 0; j < freqs.length; j++) {
    const w = 2.0 * Math.PI * freqs[j];
    const w2 = 2 * w;
    for (let i = 0; i < n; i++) {
      const arg = w2 * t[i];
      sinA[i] = Math.sin(arg);
      cosA[i] = Math.cos(arg);
    }
    const s2 = npSum(sinA, 0, n);
    const c2 = npSum(cosA, 0, n);
    const tau = Math.atan2(s2, c2) / (2 * w);
    for (let i = 0; i < n; i++) {
      const arg = w * (t[i] - tau);
      cA[i] = Math.cos(arg);
      sA[i] = Math.sin(arg);
    }
    for (let i = 0; i < n; i++) { const c = cA[i]; sinA[i] = c * c; cosA[i] = yy[i] * c; }
    const cc = npSum(sinA, 0, n);
    const yc = npSum(cosA, 0, n);
    for (let i = 0; i < n; i++) { const s = sA[i]; sinA[i] = s * s; cosA[i] = yy[i] * s; }
    const ss = npSum(sinA, 0, n);
    const ys = npSum(cosA, 0, n);
    p[j] = 0.5 * ((cc > 0 ? yc * yc / cc : 0.0) + (ss > 0 ? ys * ys / ss : 0.0)) / n;
  }
  return p;
}

/** DFA short-term scaling exponent, box sizes 4..16 beats. */
export function dfaAlpha1(x) {
  const n = x.length;
  if (n < 32) return NaN;
  const xm = npSum(x, 0, n) / n;
  const y = new Float64Array(n);
  let acc = 0.0;
  for (let i = 0; i < n; i++) { acc += x[i] - xm; y[i] = acc; } // np.cumsum (sequential)
  const logn = [], logf = [];
  const k = new Float64Array(16), kd = new Float64Array(16), tmp = new Float64Array(16), seg = new Float64Array(16);
  for (let s = 4; s < 17; s++) {
    const nb = Math.floor(n / s);
    if (nb < 2) continue;
    for (let i = 0; i < s; i++) k[i] = i;
    const km = npSum(k, 0, s) / s;
    for (let i = 0; i < s; i++) { kd[i] = k[i] - km; tmp[i] = kd[i] * kd[i]; }
    const kvar = npSum(tmp, 0, s);
    let res = 0.0;
    for (let b = 0; b < nb; b++) {
      const o = b * s;
      const sm = npSum(y, o, s) / s;
      for (let i = 0; i < s; i++) tmp[i] = kd[i] * (y[o + i] - sm);
      const slope = npSum(tmp, 0, s) / kvar;
      for (let i = 0; i < s; i++) { const r = y[o + i] - (sm + slope * kd[i]); seg[i] = r * r; }
      res += npSum(seg, 0, s);
    }
    const F = Math.sqrt(res / (nb * s));
    if (F > 0) {
      logn.push(Math.log(s));
      logf.push(Math.log(F));
    }
  }
  const m = logn.length;
  if (m < 3) return NaN;
  const ln = Float64Array.from(logn), lf = Float64Array.from(logf);
  const lnm = npSum(ln, 0, m) / m, lfm = npSum(lf, 0, m) / m;
  const num = new Float64Array(m), den = new Float64Array(m);
  for (let i = 0; i < m; i++) { const a = ln[i] - lnm; num[i] = a * (lf[i] - lfm); den[i] = a * a; }
  return npSum(num, 0, m) / npSum(den, 0, m);
}

function bandMask(lo, hi) {
  const out = [];
  for (let j = 0; j < FREQS.length; j++) if (FREQS[j] >= lo - 1e-9 && FREQS[j] < hi - 1e-9) out.push(j);
  return out;
}
const BAND_VLF = bandMask(0.01, 0.05);
const BAND_LF = bandMask(0.05, 0.15);
const BAND_HF = bandMask(0.15, 0.40 + 0.005);
const HF_PEAK = (() => { const o = []; for (let j = 0; j < FREQS.length; j++) if (FREQS[j] >= 0.15 - 1e-9) o.push(j); return o; })();

function bandSum(p, idx) {
  const v = new Float64Array(idx.length);
  for (let i = 0; i < idx.length; i++) v[i] = p[idx[i]];
  return npSum(v, 0, v.length);
}

function spectral(t, rr, valid, a, b, out) {
  const i0 = searchsortedLeft(t, a);
  const i1 = searchsortedLeft(t, b);
  for (const k of SPECTRAL) out[k] = NaN;
  let nv = 0;
  for (let i = i0; i < i1; i++) if (valid[i]) nv++;
  if (nv < 32) return out;
  const tv = new Float64Array(nv), xv = new Float64Array(nv);
  let q = 0;
  for (let i = i0; i < i1; i++) if (valid[i]) { tv[q] = t[i]; xv[q] = rr[i]; q++; }
  if (npSum(xv, 0, nv) / 1000.0 < 0.5 * (b - a)) return out;
  const p = lombPower(tv, xv);
  const vlf = bandSum(p, BAND_VLF), lf = bandSum(p, BAND_LF), hf = bandSum(p, BAND_HF);
  const ph = new Float64Array(HF_PEAK.length);
  for (let i = 0; i < HF_PEAK.length; i++) ph[i] = p[HF_PEAK[i]];
  out.log_vlf = Math.log(vlf + 1e-6);
  out.log_lf = Math.log(lf + 1e-6);
  out.log_hf = Math.log(hf + 1e-6);
  out.log_lf_hf = Math.log((lf + 1e-6) / (hf + 1e-6));
  out.hf_peak_hz = FREQS[HF_PEAK[npArgmax(ph)]];
  out.dfa_a1 = dfaAlpha1(xv);
  return out;
}

export function baseFeatureNames() {
  const names = [];
  for (const [w] of WINDOWS) for (const b of BASE) names.push(`${b}_${w}`);
  return names.concat(SPECTRAL);
}

export function baseFeatures(t, rr, valid, epochStarts, epochLen) {
  const ne = epochStarts.length;
  const names = baseFeatureNames();
  const out = {};
  for (const k of names) out[k] = new Float64Array(ne).fill(NaN);
  const st = {}, sp = {};
  for (let k = 0; k < ne; k++) {
    const s = epochStarts[k];
    const c = s + epochLen / 2;
    for (const [w, half] of WINDOWS) {
      const a = half === null ? s : c - half;
      const b = half === null ? s + epochLen : c + half;
      windowStats(t, rr, valid, a, b, st);
      for (const name of BASE) out[`${name}_${w}`][k] = st[name];
    }
    spectral(t, rr, valid, c - 150.0, c + 150.0, sp);
    for (const name of SPECTRAL) out[name][k] = sp[name];
  }
  return out;
}

// ---------------------------------------------------------------- per-record normalisation

export function nanmedian(x) {
  const f = finiteOnly(x);
  return f.length ? npMedian(f) : NaN;
}

function nanpct(x, q) {
  const s = Float64Array.from(finiteOnly(x)).sort();
  if (!s.length) return NaN;
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** IQR, else 1.4826 x MAD, else SD, else 1.0 (degenerate features such as an all-zero pNN50). */
export function robustScale(ref, med) {
  const iqr = nanpct(ref, 0.75) - nanpct(ref, 0.25);
  if (Number.isFinite(iqr) && iqr > 1e-9) return iqr;
  const dev = new Float64Array(ref.length);
  for (let i = 0; i < ref.length; i++) dev[i] = Math.abs(ref[i] - med);
  const mad = 1.4826 * nanmedian(dev);
  if (Number.isFinite(mad) && mad > 1e-9) return mad;
  const r = finiteOnly(ref);
  if (r.length > 1) {
    const m = npSum(r, 0, r.length) / r.length;
    const sq = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) { const e = r[i] - m; sq[i] = e * e; }
    const sd = Math.sqrt(npSum(sq, 0, sq.length) / (r.length - 1));
    if (sd > 1e-9) return sd;
  }
  return 1.0;
}

function rollingNanmean(x, win) {
  const half = Math.floor(win / 2);
  const n = x.length;
  const out = new Float64Array(n).fill(NaN);
  const seg = new Float64Array(2 * half + 1);
  for (let i = 0; i < n; i++) {
    let k = 0;
    const hi = Math.min(n, i + half + 1);
    for (let j = Math.max(0, i - half); j < hi; j++) if (Number.isFinite(x[j])) seg[k++] = x[j];
    if (k) out[i] = npSum(seg, 0, k) / k;
  }
  return out;
}

export function clip(v, lo, hi) {
  return v !== v ? v : Math.min(Math.max(v, lo), hi);
}

/**
 * epoch_features(t, rr, valid, epoch_start_s, epoch_len_s, cfg_rr, cfg_feat).
 * Returns {name: Float64Array} with exactly the Python names, in the same insertion order.
 */
export function epochFeatures(t, rr, valid, epochStarts, epochLen, cfgRR, cfgFeat) {
  t = Float64Array.from(t);
  rr = Float64Array.from(rr);
  epochStarts = Float64Array.from(epochStarts);
  const base = baseFeatures(t, rr, valid, epochStarts, epochLen);
  const n = epochStarts.length;
  const cov = base.coverage_w1;
  const good = new Uint8Array(n);
  for (let i = 0; i < n; i++) good[i] = cov[i] >= cfgRR.min_epoch_coverage ? 1 : 0;
  const feats = { ...base };
  for (const [name, x] of Object.entries(base)) {
    if (name.startsWith("coverage")) continue;
    const refList = [];
    for (let i = 0; i < n; i++) if (good[i]) refList.push(x[i]);
    const ref = Float64Array.from(refList);
    const med = nanmedian(ref);
    const scale = robustScale(ref, med);
    const z = new Float64Array(n);
    for (let i = 0; i < n; i++) z[i] = clip((x[i] - med) / scale, -Z_CLIP, Z_CLIP);
    feats[`z_${name}`] = z;
  }
  for (const key of KEY) {
    const z = feats[`z_${key}`];
    for (const o of cfgFeat.neighbor_offsets) {
      const d = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const j = i + o;
        d[i] = j >= 0 && j < n ? z[i] - z[j] : NaN; // z - shifted z (NaN outside the grid)
      }
      feats[`d${o >= 0 ? "+" : ""}${o}_${key}`] = d;
    }
    for (const win of cfgFeat.rolling_windows_epochs) feats[`roll${win}_${key}`] = rollingNanmean(z, win);
  }
  const t0 = n ? epochStarts[0] : 0.0;
  // Position in the night is measured from the first epoch of the grid (lights-off in the demo,
  // start of scoring in the training data). No feature depends on the total night length.
  const hrs = new Float64Array(n);
  for (let i = 0; i < n; i++) hrs[i] = (epochStarts[i] - t0) / 3600.0;
  feats.hours_since_start = hrs;
  const lq = new Float64Array(n);
  for (let i = 0; i < n; i++) lq[i] = good[i] ? 0.0 : 1.0;
  feats.low_quality = lq;
  return feats;
}

/** Analysis grid per docs/web_contract.md: start_s + L k for k = 0 .. floor((end_s - start_s) / L) - 1. */
export function epochGrid(startS, endS, epochLen) {
  const n = Math.max(0, Math.floor((endS - startS) / epochLen));
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) out[k] = startS + epochLen * k;
  return out;
}
