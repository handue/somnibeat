// v2 model inference from web/data/models_v2.json (exported by scripts/export_web_v2.py).
// Mirrors scripts/train_eval_v2.py (Python is the reference):
//
// Staging (sleep_wake, wake_nrem_rem, four_class):
//   p = mean over the task's models of predict_proba (gradient boosting, web/js/model.js)
//   smoothed = smoothRecord(p, epoch indices of the whole night, trans, prior, init, "hmm_viterbi")
//   label = argmax(smoothed)  (Viterbi output is one-hot; no threshold)
// Apnea (per minute):
//   p = mean of p(apnea) of gradient boosting and logistic regression
//   logistic: x -> NaN replaced by the imputer median -> (x - mean) / scale -> dot(coef) + intercept
//             -> 1 / (1 + exp(-z))
//   label = apnea if p >= threshold else normal
import { predictProba } from "./model.js";
import { smoothRecord } from "./smoothing.js";

function getColumns(names, columnsObj, nRows) {
  return names.map((name) => {
    const col = columnsObj[name];
    if (!col) throw new Error(`Missing feature column: ${name}`);
    if (col.length < nRows) throw new Error(`Feature column ${name} has ${col.length} rows, expected ${nRows}`);
    return col;
  });
}

// p(positive class) of the logistic pipeline for every row (Float64Array of length nRows).
export function logisticProba(model, columnsObj, nRows) {
  const cols = getColumns(model.columns, columnsObj, nRows);
  const n = cols.length;
  if (model.impute.length !== n || model.mean.length !== n || model.scale.length !== n || model.coef.length !== n) {
    throw new Error("Logistic model parameters do not match its column list");
  }
  const out = new Float64Array(nRows);
  for (let row = 0; row < nRows; row++) {
    let acc = 0.0;
    for (let j = 0; j < n; j++) {
      let x = cols[j][row];
      if (Number.isNaN(x)) x = model.impute[j];
      x = (x - model.mean[j]) / model.scale[j];
      acc += x * model.coef[j];
    }
    const z = acc + model.intercept;
    out[row] = 1 / (1 + Math.exp(-z));
  }
  return out;
}

// Row-major (nRows, K) class probabilities of one model entry.
function modelProba(model, columnsObj, nRows, k) {
  if (model.kind === "hgb") return predictProba(model, columnsObj, nRows);
  if (model.kind === "logistic") {
    if (k !== 2) throw new Error("Logistic models are binary only");
    const p1 = logisticProba(model, columnsObj, nRows);
    const out = new Float64Array(nRows * 2);
    for (let row = 0; row < nRows; row++) {
      out[row * 2] = 1 - p1[row];
      out[row * 2 + 1] = p1[row];
    }
    return out;
  }
  throw new Error(`Unknown model kind: ${model.kind}`);
}

// Mean of the models' probabilities, summed in model order then divided (as numpy mean).
function ensembleProba(task, columnsObj, nRows) {
  if (task.ensemble !== "mean") throw new Error(`Unknown ensemble rule: ${task.ensemble}`);
  const k = task.labels.length;
  const parts = task.models.map((m) => modelProba(m, columnsObj, nRows, k));
  if (parts.length === 1) return { proba: parts[0], parts };
  const proba = new Float64Array(nRows * k);
  for (const p of parts) for (let i = 0; i < proba.length; i++) proba[i] += p[i];
  for (let i = 0; i < proba.length; i++) proba[i] /= parts.length;
  return { proba, parts };
}

// Staging for one night. columnsObj: feature name -> Float64Array; epochIdx: epoch index per
// row in increasing order (default 0..nRows-1, the demo grid). Returns
// {proba: model probabilities (nRows x K), smoothed: one-hot (nRows x K), labels: Int32Array}.
export function predictStaging(task, columnsObj, nRows, epochIdx = null) {
  const k = task.labels.length;
  const idx = epochIdx || Int32Array.from({ length: nRows }, (_, i) => i);
  if (idx.length !== nRows) throw new Error(`epochIdx has ${idx.length} entries, expected ${nRows}`);
  const { proba } = ensembleProba(task, columnsObj, nRows);
  const h = task.hmm;
  if (!h || task.decision.type !== h.method) throw new Error("Staging task needs matching hmm parameters");
  const smoothed = smoothRecord(proba, idx, h.trans, h.prior, h.init, h.method);
  const labels = new Int32Array(nRows);
  for (let row = 0; row < nRows; row++) {
    const o = row * k;
    let best = 0;
    for (let c = 1; c < k; c++) if (smoothed[o + c] > smoothed[o + best]) best = c;
    labels[row] = best;
  }
  return { proba, smoothed, labels };
}

// Apnea minutes. Returns {pApnea: Float64Array (ensemble mean), parts: [Float64Array per model],
// labels: Int32Array (index into task.labels)}.
export function predictApnea(task, columnsObj, nRows) {
  const d = task.decision;
  const pos = task.labels.indexOf(d.positive);
  if (d.type !== "threshold" || task.labels.length !== 2 || pos < 0) {
    throw new Error(`Bad apnea decision for labels ${task.labels}`);
  }
  if (task.ensemble !== "mean") throw new Error(`Unknown ensemble rule: ${task.ensemble}`);
  const parts = task.models.map((m) => {
    const p = modelProba(m, columnsObj, nRows, 2);
    const out = new Float64Array(nRows);
    for (let row = 0; row < nRows; row++) out[row] = p[row * 2 + pos];
    return out;
  });
  const pApnea = new Float64Array(nRows);
  for (const p of parts) for (let row = 0; row < nRows; row++) pApnea[row] += p[row];
  if (parts.length > 1) for (let row = 0; row < nRows; row++) pApnea[row] /= parts.length;
  const labels = new Int32Array(nRows);
  for (let row = 0; row < nRows; row++) labels[row] = pApnea[row] >= d.threshold ? pos : 1 - pos;
  return { pApnea, parts, labels };
}
