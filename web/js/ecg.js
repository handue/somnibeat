// ECG recording -> RR intervals, in the browser.
// Mirror of somnibeat/edf.py (single-channel EDF reader), somnibeat/rpeak.py (R-peak
// detector) and scripts/ecg_to_rr.py (rounding, RR-interval exports), so a recording
// dropped on the page gives the RR series the reported accuracy was measured on.
// Parity-tested against the Python code in tests/test_parity_ecg.py.
//
// Nothing here leaves the browser: the file is read in slices from the user's disk.

export const MIN_FS = 100; // the 5-25 Hz band-pass and +-75 ms refinement need at least this
export const DEFAULT_CHANNEL = /ECG|EKG/i;

export const ECG_ERROR_CODES = Object.freeze({
  NOT_EDF: "NOT_EDF",           // header is not a readable EDF header
  NO_ECG_CHANNEL: "NO_ECG_CHANNEL",
  LOW_FS: "LOW_FS",             // ECG sampled below MIN_FS
  NO_BEATS: "NO_BEATS",         // fewer than two heartbeats found
  NO_INTERVALS: "NO_INTERVALS", // RR export with fewer than two numeric intervals
  TRUNCATED: "TRUNCATED",       // file holds no complete data record
});

function ecgError(code, message, details = {}) {
  const e = new Error(message);
  e.code = code;
  e.details = details;
  return e;
}

// ---------------------------------------------------------------------------
// EDF (somnibeat/edf.py)

const ASCII = new TextDecoder("latin1");

function field(s, a, b) {
  return s.slice(a, b);
}

/** Parse the fixed-width EDF header. `bytes` must hold at least the first 256 bytes; returns
 * {ns, needBytes} when the signal headers are not yet included. */
export function parseEdfHeader(bytes, fileSize) {
  if (bytes.length < 256) throw ecgError(ECG_ERROR_CODES.NOT_EDF, "The file is too short to be an EDF recording.");
  const h = ASCII.decode(bytes.subarray(0, 256));
  const ns = parseInt(field(h, 252, 256), 10);
  if (!(ns > 0 && ns < 4096) || field(h, 0, 8).trim() !== "0") {
    throw ecgError(ECG_ERROR_CODES.NOT_EDF, "This file does not look like an EDF recording (the header is not readable).");
  }
  const headerBytes = 256 * (ns + 1);
  if (bytes.length < headerBytes) return { ns, needBytes: headerBytes };
  const sh = ASCII.decode(bytes.subarray(256, headerBytes));
  const widths = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32];
  const fields = [];
  let pos = 0;
  for (const w of widths) {
    const col = [];
    for (let i = 0; i < ns; i++) col.push(sh.slice(pos + i * w, pos + (i + 1) * w).trim());
    fields.push(col);
    pos += w * ns;
  }
  const [labels, , units, pmin, pmax, dmin, dmax, , spr] = fields;
  const num = (s) => {
    const v = Number(s);
    if (s === "" || !Number.isFinite(v)) throw ecgError(ECG_ERROR_CODES.NOT_EDF, "The EDF header has a non-numeric field.");
    return v;
  };
  const time = field(h, 176, 184).split(/[.:]/).map((x) => parseInt(x, 10));
  const recDur = num(field(h, 244, 252).trim());
  const sprA = spr.map((x) => Math.trunc(num(x)));
  const total = sprA.reduce((a, b) => a + b, 0);
  if (!(total > 0) || !(recDur > 0)) throw ecgError(ECG_ERROR_CODES.NOT_EDF, "The EDF header has no samples per record.");
  let nRec = parseInt(field(h, 236, 244), 10);
  const fromSize = Math.floor((fileSize - headerBytes) / (2 * total));
  if (!(nRec > 0) || nRec > fromSize) nRec = fromSize; // header count missing or larger than the file holds
  const pad = (v) => String(v).padStart(2, "0");
  const startClock = time.length >= 2 && time.every((v) => Number.isInteger(v)) && time[0] < 24 && time[1] < 60
    ? `${pad(time[0])}:${pad(time[1])}` : null;
  return {
    ns, labels, units,
    physMin: pmin.map(num), physMax: pmax.map(num), digMin: dmin.map(num), digMax: dmax.map(num),
    samplesPerRecord: sprA, nRecords: nRec, recordDurationS: recDur, headerBytes, startClock,
  };
}

/** Index of the first channel whose label matches `pattern` (re.search, case-insensitive), or -1. */
export function findChannel(hdr, pattern = DEFAULT_CHANNEL) {
  return hdr.labels.findIndex((l) => pattern.test(l));
}

async function sliceBytes(source, a, b) {
  if (source instanceof Uint8Array) return source.subarray(a, Math.min(b, source.length));
  return new Uint8Array(await source.slice(a, b).arrayBuffer());
}

function sourceSize(source) {
  return source instanceof Uint8Array ? source.length : source.size;
}

/** Read the EDF header of a Blob/File or Uint8Array. */
export async function readEdfHeader(source) {
  const size = sourceSize(source);
  const first = await sliceBytes(source, 0, 256);
  let hdr = parseEdfHeader(first, size);
  if (hdr.needBytes) hdr = parseEdfHeader(await sliceBytes(source, 0, hdr.needBytes), size);
  if (hdr.needBytes) {
    throw ecgError(ECG_ERROR_CODES.TRUNCATED, "The EDF header is incomplete; the file may be cut off (an unfinished download?).");
  }
  return hdr;
}

/** read_channel: {x (physical units), fs, hdr, label}. Reads the file in slices of data records. */
export async function readEdfChannel(source, pattern = DEFAULT_CHANNEL, onProgress = null) {
  const hdr = await readEdfHeader(source);
  const i = findChannel(hdr, pattern);
  if (i < 0) {
    throw ecgError(ECG_ERROR_CODES.NO_ECG_CHANNEL,
      "No ECG channel was found in this recording.", { labels: hdr.labels });
  }
  if (!(hdr.nRecords > 0)) throw ecgError(ECG_ERROR_CODES.TRUNCATED, "The EDF file holds no complete data record.");
  const total = hdr.samplesPerRecord.reduce((a, b) => a + b, 0);
  const recBytes = 2 * total;
  const off = 2 * hdr.samplesPerRecord.slice(0, i).reduce((a, b) => a + b, 0);
  const spr = hdr.samplesPerRecord[i];
  const x = new Float64Array(hdr.nRecords * spr);
  const spanD = hdr.digMax[i] - hdr.digMin[i];
  const gain = spanD ? (hdr.physMax[i] - hdr.physMin[i]) / spanD : 1.0;
  const dmin = hdr.digMin[i];
  const pmin = hdr.physMin[i];
  const perChunk = Math.max(1, Math.floor((8 << 20) / recBytes));
  let k = 0;
  for (let r0 = 0; r0 < hdr.nRecords; r0 += perChunk) {
    const r1 = Math.min(hdr.nRecords, r0 + perChunk);
    const buf = await sliceBytes(source, hdr.headerBytes + r0 * recBytes, hdr.headerBytes + r1 * recBytes);
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    for (let r = 0; r < r1 - r0; r++) {
      const base = r * recBytes + off;
      for (let s = 0; s < spr; s++) x[k++] = (view.getInt16(base + 2 * s, true) - dmin) * gain + pmin;
    }
    if (onProgress) onProgress(r1 / hdr.nRecords);
  }
  return { x, fs: spr / hdr.recordDurationS, hdr, label: hdr.labels[i] };
}

// ---------------------------------------------------------------------------
// numpy / scipy building blocks (same operation order as the Python originals)

function cmul(a, b) { return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]; }
function cdiv(a, b) {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
}
function csqrt(a) {
  const r = Math.hypot(a[0], a[1]);
  const re = Math.sqrt((r + a[0]) / 2);
  const im = Math.sqrt((r - a[0]) / 2);
  return [re, a[1] < 0 ? -im : im];
}

/** butter(2, [5, min(25, 0.45 fs)], btype="band", fs=fs, output="sos"), section layout as scipy. */
export function bandpassSos(fs) {
  const hi = Math.min(25.0, 0.45 * fs);
  // iirfilter: normalise to Nyquist, pre-warp with fs = 2.
  const wn = [2 * 5.0 / fs, 2 * hi / fs];
  const warped = wn.map((w) => 2 * 2 * Math.tan(Math.PI * w / 2));
  const bw = warped[1] - warped[0];
  const wo = Math.sqrt(warped[0] * warped[1]);
  // buttap(2): poles -exp(j pi m / 4), m = -1, 1
  const pLp = [-1, 1].map((m) => [-Math.cos(Math.PI * m / 4), -Math.sin(Math.PI * m / 4)]);
  // lp2bp_zpk
  const pScaled = pLp.map((p) => [p[0] * bw / 2, p[1] * bw / 2]);
  const pBp = [];
  for (const sign of [1, -1]) {
    for (const p of pScaled) {
      const s = csqrt([p[0] * p[0] - p[1] * p[1] - wo * wo, 2 * p[0] * p[1]]);
      pBp.push([p[0] + sign * s[0], p[1] + sign * s[1]]);
    }
  }
  const kBp = bw * bw;
  // bilinear_zpk, fs = 2: zeros at 0 -> +1 (twice), extra zeros at -1 (twice)
  const fs2 = 4.0;
  const pZ = pBp.map((p) => cdiv([fs2 + p[0], p[1]], [fs2 - p[0], -p[1]]));
  let den = [1, 0];
  for (const p of pBp) den = cmul(den, [fs2 - p[0], -p[1]]);
  const kZ = kBp * cdiv([fs2 * fs2, 0], den)[0];
  // Conjugate pairs (upper half-plane representatives), the pair nearest the unit circle last.
  const upper = pZ.filter((p) => p[1] > 0);
  upper.sort((a, b) => Math.hypot(b[0], b[1]) - Math.hypot(a[0], a[1]));
  const near = upper[0];
  const far = upper[1];
  const aOf = (p) => [1.0, -2 * p[0], p[0] * p[0] + p[1] * p[1]];
  // The pole pair nearest the unit circle takes the zero pair nearest to it.
  const nearZero = near[0] >= 0 ? 1 : -1;
  const bNear = [1.0, -2 * nearZero, 1.0];
  const bFar = [kZ, 2 * nearZero * kZ, kZ];
  return [[...bFar, ...aOf(far)], [...bNear, ...aOf(near)]];
}

function lfilterZi(b, a) {
  const sumA = a[0] + a[1] + a[2];
  const yInf = (b[0] + b[1] + b[2]) / sumA;
  const c = [b[0] - yInf * a[0], b[1] - yInf * a[1], b[2] - yInf * a[2]];
  return [c[2] + c[1], c[2]];
}

function sosfiltZi(sos) {
  let scale = 1.0;
  return sos.map((s) => {
    const z = lfilterZi(s.slice(0, 3), s.slice(3));
    const out = [scale * z[0], scale * z[1]];
    scale *= (s[0] + s[1] + s[2]) / (s[3] + s[4] + s[5]);
    return out;
  });
}

function sosfiltInPlace(sos, y, zi0) {
  const nS = sos.length;
  const zi = zi0.map((z) => [z[0], z[1]]);
  for (let n = 0; n < y.length; n++) {
    let xCur = y[n];
    for (let s = 0; s < nS; s++) {
      const c = sos[s];
      const z = zi[s];
      const xNew = c[0] * xCur + z[0];
      z[0] = c[1] * xCur - c[4] * xNew + z[1];
      z[1] = c[2] * xCur - c[5] * xNew;
      xCur = xNew;
    }
    y[n] = xCur;
  }
}

/** sosfiltfilt(sos, x) with scipy's default odd padding. */
export function sosfiltfilt(sos, x) {
  let ntaps = 2 * sos.length + 1;
  ntaps -= Math.min(sos.filter((s) => s[2] === 0).length, sos.filter((s) => s[5] === 0).length);
  const edge = 3 * ntaps;
  const n = x.length;
  if (n <= edge) throw ecgError(ECG_ERROR_CODES.NO_BEATS, "The ECG signal is too short to filter.");
  const ext = new Float64Array(n + 2 * edge);
  for (let i = 0; i < edge; i++) ext[i] = 2 * x[0] - x[edge - i];
  ext.set(x, edge);
  for (let i = 0; i < edge; i++) ext[edge + n + i] = 2 * x[n - 1] - x[n - 2 - i];
  const zi = sosfiltZi(sos);
  const x0 = ext[0];
  sosfiltInPlace(sos, ext, zi.map((z) => [z[0] * x0, z[1] * x0]));
  ext.reverse();
  const y0 = ext[0];
  sosfiltInPlace(sos, ext, zi.map((z) => [z[0] * y0, z[1] * y0]));
  ext.reverse();
  return ext.subarray(edge, edge + n);
}

/** np.median (copy sorted). */
export function median(values, presorted = false) {
  const n = values.length;
  if (n === 0) return NaN;
  const s = presorted ? values : Float64Array.from(values).sort();
  const h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}

/** np.percentile(v, q) with the default linear method. `sorted` must be ascending. */
function percentileSorted(sorted, q) {
  const n = sorted.length;
  const vi = (n - 1) * (q / 100);
  const prev = Math.floor(vi);
  const next = Math.min(prev + 1, n - 1);
  const g = vi - prev;
  const a = sorted[Math.min(prev, n - 1)];
  const b = sorted[next];
  const d = b - a;
  return g >= 0.5 ? b - d * (1 - g) : a + d * g;
}

/** scipy.ndimage.median_filter(v, size, mode="nearest") for odd size. */
function medianFilterNearest(v, size) {
  const n = v.length;
  const h = size >> 1;
  const out = new Float64Array(n);
  const w = new Float64Array(size);
  for (let i = 0; i < n; i++) {
    for (let j = -h; j <= h; j++) w[j + h] = v[Math.min(n - 1, Math.max(0, i + j))];
    w.sort();
    out[i] = w[h];
  }
  return out;
}

/** np.interp(arange(n), xp, fp). */
function interpIndex(n, xp, fp, out = new Float64Array(n)) {
  const m = xp.length;
  const slopes = new Float64Array(Math.max(0, m - 1));
  for (let j = 0; j < m - 1; j++) slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const x = i;
    if (x >= xp[m - 1]) out[i] = fp[m - 1];
    else if (x < xp[0]) out[i] = fp[0];
    else {
      while (j < m - 2 && xp[j + 1] <= x) j++;
      out[i] = x === xp[j] ? fp[j] : slopes[j] * (x - xp[j]) + fp[j];
    }
  }
  return out;
}

/** scipy.signal.find_peaks(x, distance=d)[0]. */
function findPeaksDistance(x, distance) {
  const n = x.length;
  const peaks = [];
  let i = 1;
  const iMax = n - 1;
  while (i < iMax) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < iMax && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) {
        peaks.push((i + ahead - 1) >> 1);
        i = ahead;
      }
    }
    i++;
  }
  const np = peaks.length;
  const keep = new Uint8Array(np).fill(1);
  const order = Array.from({ length: np }, (_, k) => k).sort((a, b) => x[peaks[a]] - x[peaks[b]] || a - b);
  const d = Math.ceil(distance);
  for (let o = np - 1; o >= 0; o--) {
    const j = order[o];
    if (!keep[j]) continue;
    for (let k = j - 1; k >= 0 && peaks[j] - peaks[k] < d; k--) keep[k] = 0;
    for (let k = j + 1; k < np && peaks[k] - peaks[j] < d; k++) keep[k] = 0;
  }
  return Int32Array.from(peaks.filter((_, k) => keep[k]));
}

// ---------------------------------------------------------------------------
// R-peak detector (somnibeat/rpeak.py)

/** _envelope: squared first difference, 120 ms centred moving average (np.convolve "same").
 * `work` (length n) is overwritten with the squared differences; returns a new array. */
function envelope(f, fs, work) {
  const n = f.length;
  const e = work;
  e[0] = 0;
  for (let i = 1; i < n; i++) {
    const d = f[i] - f[i - 1];
    e[i] = d * d;
  }
  const w = Math.max(1, roundHalfEven(0.12 * fs)); // int(round(0.12 * fs)); Python rounds half to even
  const inv = 1 / w;
  const c = (w - 1) >> 1;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let acc = 0;
    const lo = Math.max(0, k + c - (w - 1));
    const hi = Math.min(n - 1, k + c);
    for (let m = lo; m <= hi; m++) acc += e[m] * inv;
    out[k] = acc;
  }
  return out;
}

/** _threshold: per-sample threshold (written into `out`) and no-signal mask. */
function threshold(env, fs, out) {
  const n = env.length;
  const blk = Math.trunc(2 * fs);
  const nb = Math.max(1, Math.floor(n / blk));
  const pRaw = new Float64Array(nb);
  for (let i = 0; i < nb; i++) pRaw[i] = percentileSorted(env.slice(i * blk, Math.min(n, (i + 1) * blk)).sort(), 98);
  const rec = median(pRaw);
  const p = medianFilterNearest(pRaw, 5).map((v) => Math.max(v, 0.25 * rec));
  const centers = new Float64Array(nb);
  for (let i = 0; i < nb; i++) centers[i] = (i + 0.5) * blk;
  const thr = interpIndex(n, centers, p, out);
  for (let i = 0; i < n; i++) thr[i] = 0.3 * thr[i];
  const deadBlk = new Uint8Array(nb);
  for (let i = 0; i < nb; i++) deadBlk[i] = pRaw[i] < 0.1 * rec ? 1 : 0;
  const dead = (i) => deadBlk[Math.min(nb - 1, Math.floor(i / blk))] === 1;
  return { thr, dead };
}

/** R-peak times in seconds (sub-sample precision), as somnibeat.rpeak.detect_r_peaks.
 * The input array is used as scratch space and is overwritten. */
export function detectRPeaks(ecg, fs) {
  const n = ecg.length;
  const x = ecg;
  const sorted = Float64Array.from(x).sort(); // NaN sorts last
  let nFinite = n;
  while (nFinite > 0 && Number.isNaN(sorted[nFinite - 1])) nFinite--;
  const med = median(sorted.subarray(0, nFinite), true);
  for (let i = 0; i < n; i++) {
    const v = x[i] - med; // np.nan_to_num
    x[i] = Number.isNaN(v) ? 0 : v === Infinity ? Number.MAX_VALUE : v === -Infinity ? -Number.MAX_VALUE : v;
  }
  const f = sosfiltfilt(bandpassSos(fs), x);
  const env = envelope(f, fs, x);
  const { thr, dead } = threshold(env, fs, x);
  const refractory = Math.trunc(0.25 * fs);
  const cand = Array.from(findPeaksDistance(env, refractory)).filter((c) => !dead(c));
  let peaks = cand.filter((c) => env[c] > thr[c]);

  // Search-back for missed beats in long gaps; repeat until nothing new is added (max 5 passes).
  const low = cand.filter((c) => env[c] > 0.5 * thr[c]);
  for (let pass = 0; pass < 5; pass++) {
    if (peaks.length <= 8) break;
    const rr = new Float64Array(peaks.length - 1);
    for (let i = 0; i < rr.length; i++) rr[i] = peaks[i + 1] - peaks[i];
    const m = medianFilterNearest(rr, 9);
    const extra = [];
    let lo = 0;
    for (let i = 0; i < rr.length; i++) {
      if (!(rr[i] > 1.66 * m[i])) continue;
      const a = peaks[i] + refractory;
      const b = peaks[i + 1] - refractory;
      while (lo < low.length && low[lo] <= a) lo++;
      let best = -1;
      for (let k = lo; k < low.length && low[k] < b; k++) if (best < 0 || env[low[k]] > env[best]) best = low[k];
      if (best >= 0) extra.push(best);
    }
    if (!extra.length) break;
    peaks = peaks.concat(extra).sort((u, v) => u - v);
  }
  if (!peaks.length) return new Float64Array(0);

  // Move to the ECG extremum with the recording's dominant polarity.
  const half = Math.trunc(0.075 * fs);
  const lo = peaks.map((p) => Math.min(n - 1, Math.max(0, p - half)));
  const hi = peaks.map((p) => Math.min(n, Math.max(1, p + half + 1)));
  const amax = new Float64Array(peaks.length);
  const amin = new Float64Array(peaks.length);
  for (let k = 0; k < peaks.length; k++) {
    let mx = -Infinity;
    let mn = Infinity;
    for (let i = lo[k]; i < hi[k]; i++) {
      if (f[i] > mx) mx = f[i];
      if (f[i] < mn) mn = f[i];
    }
    amax[k] = mx;
    amin[k] = -mn;
  }
  const sign = median(amax) >= median(amin) ? 1.0 : -1.0;
  const g = f; // g = sign * f, in place (f is not needed any more)
  for (let i = 0; i < n; i++) g[i] = sign * g[i];
  let idx = [];
  for (let k = 0; k < peaks.length; k++) {
    let best = lo[k];
    for (let i = lo[k] + 1; i < hi[k]; i++) if (g[i] > g[best]) best = i;
    idx.push(best);
  }
  idx = Array.from(new Set(idx)).sort((u, v) => u - v);
  // Enforce refractory after relocation (keep the larger of two close peaks).
  const keep = [idx[0]];
  for (let j = 1; j < idx.length; j++) {
    const k = idx[j];
    if (k - keep[keep.length - 1] < refractory) {
      if (g[k] > g[keep[keep.length - 1]]) keep[keep.length - 1] = k;
    } else keep.push(k);
  }
  // Parabolic interpolation.
  const t = new Float64Array(keep.length);
  for (let j = 0; j < keep.length; j++) {
    const i = keep[j];
    let ti = i;
    if (i > 0 && i < n - 1) {
      const y0 = g[i - 1];
      const y1 = g[i];
      const y2 = g[i + 1];
      const den = y0 - 2 * y1 + y2;
      const delta = Math.abs(den) > 1e-12 ? 0.5 * (y0 - y2) / den : 0.0;
      ti += Math.min(0.5, Math.max(-0.5, delta));
    }
    t[j] = ti / fs;
  }
  return t;
}

// ---------------------------------------------------------------------------
// RR output (scripts/ecg_to_rr.py)

/** np.rint: round half to even. */
export function roundHalfEven(y) {
  const r = Math.round(y);
  return Math.abs(y % 1) === 0.5 ? 2 * Math.round(y / 2) : r;
}

/** np.round(x, decimals) for decimals > 0: rint(x * 10**d) / 10**d. */
export function npRound(x, decimals) {
  const f = 10 ** decimals;
  return roundHalfEven(x * f) / f;
}

/** Beat times -> {time_s, rr_ms} rounded as the evaluation pipeline writes them. */
export function rrFromBeatsRounded(beats) {
  const m = Math.max(0, beats.length - 1);
  const time_s = new Float64Array(m);
  const rr_ms = new Float64Array(m);
  for (let i = 1; i < beats.length; i++) {
    time_s[i - 1] = npRound(beats[i], 4);
    rr_ms[i - 1] = npRound((beats[i] - beats[i - 1]) * 1000.0, 2);
  }
  return { time_s, rr_ms };
}

/** CSV text in the format the page reads (header time_s,rr_ms). */
export function rrCsv(time_s, rr_ms) {
  const lines = ["time_s,rr_ms"];
  for (let i = 0; i < time_s.length; i++) lines.push(`${time_s[i]},${rr_ms[i]}`);
  return lines.join("\n") + "\n";
}

/** Full conversion of an EDF recording (Blob/File or Uint8Array). */
export async function edfToRR(source, { pattern = DEFAULT_CHANNEL, onProgress = null } = {}) {
  return channelToRR(await readEdfChannel(source, pattern, onProgress));
}

/** Detect beats on a channel read by readEdfChannel (its samples are overwritten). */
export function channelToRR({ x, fs, hdr, label }) {
  const nSamples = x.length;
  if (fs < MIN_FS) {
    throw ecgError(ECG_ERROR_CODES.LOW_FS,
      `The ECG channel is sampled at ${fs} Hz, below the ${MIN_FS} Hz needed to locate heartbeats reliably.`,
      { fs, label });
  }
  const beats = detectRPeaks(x, fs);
  if (beats.length < 2) {
    throw ecgError(ECG_ERROR_CODES.NO_BEATS,
      `Fewer than two heartbeats were found in channel "${label}". Is this the ECG channel?`, { label });
  }
  const { time_s, rr_ms } = rrFromBeatsRounded(beats);
  return {
    time_s, rr_ms, fs, label, beats: beats.length,
    durationH: nSamples / fs / 3600, startClock: hdr.startClock,
    source: `EDF channel "${label}" at ${fs} Hz`,
  };
}

// ---------------------------------------------------------------------------
// RR-interval exports (chest-strap apps): one column named like "RR".

const RR_NAME = /(^|[^a-z])rr([^a-z]|$)/i;

function splitLine(line, delim) {
  return line.split(delim).map((c) => {
    let s = c.trim();
    if (s.length >= 2 && s[0] === '"' && s[s.length - 1] === '"') s = s.slice(1, -1).trim();
    return s;
  });
}

function sniffDelimiter(header) {
  for (const d of [",", ";", "\t"]) if (header.includes(d)) return d;
  return ",";
}

/** Name and index of the single RR-like column of a CSV export, or null. */
export function rrExportColumn(text) {
  const head = (text.charCodeAt(0) === 0xfeff ? text.slice(1) : text).split(/\r\n|\n|\r/).find((l) => l.trim() !== "");
  if (!head) return null;
  const delim = sniffDelimiter(head);
  const cols = splitLine(head, delim);
  if (cols.map((c) => c.toLowerCase()).includes("rr_ms")) return null; // already the page's format
  const hits = cols.map((c, i) => [c, i]).filter(([c]) => RR_NAME.test(c));
  return hits.length === 1 ? { name: hits[0][0], index: hits[0][1], delim } : null;
}

/** convert_rr_export: RR column (ms, or s when the median is below 5) -> time_s = running sum. */
export function rrExportToRR(text, col) {
  const lines = (text.charCodeAt(0) === 0xfeff ? text.slice(1) : text).split(/\r\n|\n|\r/);
  let li = lines.findIndex((l) => l.trim() !== "");
  const vals = [];
  for (li++; li < lines.length; li++) {
    if (lines[li].trim() === "") continue;
    const s = splitLine(lines[li], col.delim)[col.index];
    const v = s === undefined || s === "" ? NaN : Number(s);
    if (Number.isFinite(v)) vals.push(v);
  }
  if (vals.length < 2) {
    throw ecgError(ECG_ERROR_CODES.NO_INTERVALS, `The column "${col.name}" has fewer than two numeric intervals.`);
  }
  const unit = median(vals) < 5 ? "s" : "ms";
  const rr = Float64Array.from(vals, (v) => (unit === "s" ? v * 1000.0 : v));
  const time_s = new Float64Array(rr.length);
  const rr_ms = new Float64Array(rr.length);
  let acc = 0;
  for (let i = 0; i < rr.length; i++) {
    acc += rr[i];
    time_s[i] = npRound(acc / 1000.0, 4);
    rr_ms[i] = npRound(rr[i], 2);
  }
  return {
    time_s, rr_ms, fs: null, label: col.name, beats: rr.length,
    durationH: acc / 1000 / 3600, startClock: null,
    source: `RR column "${col.name}" (${unit})`,
  };
}
