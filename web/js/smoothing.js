// Temporal smoothing of per-epoch class probabilities with a first-order HMM.
// Line-for-line port of somnibeat/smoothing.py (Python is the reference).
//
// - States: the task's classes (0..K-1).
// - Emissions: scaled likelihoods e_t[k] = clip(p(k | x_t), EPS, 1) / prior[k].
// - Initial distribution: init; transitions: trans[i][j] = P(class_t = j | class_{t-1} = i).
// - Sequences are split at gaps (epoch index difference != 1); each segment is decoded
//   independently, starting again from init.
// - Viterbi tie rule: the lowest class index wins (strict '>' comparison).
//
// Arrays of probabilities are row-major Float64Array of length T * K (as returned by
// web/js/model.js predictProba); trans is an array of K rows; init and prior have length K.

export const EPS = 1e-9;

// proba: row-major (T, K). Returns a new row-major Float64Array (T, K).
export function emissions(proba, nRows, k, prior) {
  const out = new Float64Array(nRows * k);
  for (let t = 0; t < nRows; t++) {
    for (let c = 0; c < k; c++) {
      let v = proba[t * k + c];
      if (v < EPS) v = EPS;
      if (v > 1.0) v = 1.0;
      out[t * k + c] = v / prior[c];
    }
  }
  return out;
}

// Most likely state path, computed in log space. emis: row-major (n, k), read from row
// `offset` on (so a segment of a longer array can be decoded without copying).
// Returns an Int32Array of length n.
export function viterbi(emis, n, k, trans, init, offset = 0) {
  const path = new Int32Array(n);
  if (n === 0) return path;
  const logA = [];
  for (let i = 0; i < k; i++) {
    const row = new Float64Array(k);
    for (let j = 0; j < k; j++) row[j] = Math.log(trans[i][j]);
    logA.push(row);
  }
  const e = (t, j) => emis[(offset + t) * k + j];
  let delta = new Float64Array(k);
  for (let j = 0; j < k; j++) delta[j] = Math.log(init[j]) + Math.log(e(0, j));
  const back = new Int32Array(n * k);
  for (let t = 1; t < n; t++) {
    const next = new Float64Array(k);
    for (let j = 0; j < k; j++) {
      let bestI = 0;
      let bestV = delta[0] + logA[0][j];
      for (let i = 1; i < k; i++) {
        const v = delta[i] + logA[i][j];
        if (v > bestV) {
          bestI = i;
          bestV = v;
        }
      }
      next[j] = bestV + Math.log(e(t, j));
      back[t * k + j] = bestI;
    }
    delta = next;
  }
  let bestJ = 0;
  let bestV = delta[0];
  for (let j = 1; j < k; j++) {
    if (delta[j] > bestV) {
      bestJ = j;
      bestV = delta[j];
    }
  }
  path[n - 1] = bestJ;
  for (let t = n - 1; t > 0; t--) path[t - 1] = back[t * k + path[t]];
  return path;
}

// Posterior P(state_t | all emissions) with per-step normalization (scaling).
// emis: row-major (n, k) from row `offset`. Returns a row-major Float64Array (n, k).
export function forwardBackward(emis, n, k, trans, init, offset = 0) {
  const post = new Float64Array(n * k);
  if (n === 0) return post;
  const e = (t, j) => emis[(offset + t) * k + j];
  const fwd = new Float64Array(n * k);
  let s = 0.0;
  for (let j = 0; j < k; j++) {
    fwd[j] = init[j] * e(0, j);
    s += fwd[j];
  }
  for (let j = 0; j < k; j++) fwd[j] /= s;
  for (let t = 1; t < n; t++) {
    s = 0.0;
    for (let j = 0; j < k; j++) {
      let acc = 0.0;
      for (let i = 0; i < k; i++) acc += fwd[(t - 1) * k + i] * trans[i][j];
      fwd[t * k + j] = acc * e(t, j);
      s += fwd[t * k + j];
    }
    for (let j = 0; j < k; j++) fwd[t * k + j] /= s;
  }
  const bwd = new Float64Array(k).fill(1.0);
  for (let t = n - 1; t >= 0; t--) {
    s = 0.0;
    for (let j = 0; j < k; j++) {
      post[t * k + j] = fwd[t * k + j] * bwd[j];
      s += post[t * k + j];
    }
    for (let j = 0; j < k; j++) post[t * k + j] /= s;
    if (t > 0) {
      const nb = new Float64Array(k);
      s = 0.0;
      for (let i = 0; i < k; i++) {
        let acc = 0.0;
        for (let j = 0; j < k; j++) acc += trans[i][j] * e(t, j) * bwd[j];
        nb[i] = acc;
        s += acc;
      }
      for (let i = 0; i < k; i++) bwd[i] = nb[i] / s;
    }
  }
  return post;
}

// [start, end) index ranges of runs of consecutive epochs.
export function segments(epochIdx) {
  const out = [];
  let start = 0;
  for (let t = 1; t < epochIdx.length; t++) {
    if (epochIdx[t] - epochIdx[t - 1] !== 1) {
      out.push([start, t]);
      start = t;
    }
  }
  if (epochIdx.length) out.push([start, epochIdx.length]);
  return out;
}

// Returns row-major (T, K): the probabilities unchanged ("none"), posteriors
// ("hmm_forward_backward") or the one-hot decoded path ("hmm_viterbi").
export function smoothRecord(proba, epochIdx, trans, prior, init, method) {
  const nRows = epochIdx.length;
  const k = prior.length;
  if (proba.length !== nRows * k) {
    throw new Error(`smoothRecord: ${proba.length} probabilities for ${nRows} epochs x ${k} classes`);
  }
  if (method === "none") return Float64Array.from(proba);
  const emis = emissions(proba, nRows, k, prior);
  const out = new Float64Array(nRows * k);
  for (const [a, b] of segments(epochIdx)) {
    if (method === "hmm_viterbi") {
      const p = viterbi(emis, b - a, k, trans, init, a);
      for (let t = 0; t < b - a; t++) out[(a + t) * k + p[t]] = 1.0;
    } else if (method === "hmm_forward_backward") {
      out.set(forwardBackward(emis, b - a, k, trans, init, a), a * k);
    } else {
      throw new Error(`unknown smoothing method '${method}'`);
    }
  }
  return out;
}
