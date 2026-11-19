// HistGradientBoostingClassifier inference from web/data/models.json.
// Matches sklearn predict_proba for numeric features (see docs/web_contract.md):
//   raw_k = baseline_k + sum over iterations of the leaf value of tree k
//   traversal: leaf -> value; NaN -> left if missing_go_to_left else right;
//              otherwise left if x <= threshold else right
//   binary: p(positive) = 1 / (1 + exp(-raw_0)); multiclass: softmax(raw)

const compiled = new WeakMap();

// Converts the JSON trees of one task to typed arrays once.
function compile(task) {
  let c = compiled.get(task);
  if (c) return c;
  const trees = task.trees.map((iteration) =>
    iteration.map((t) => ({
      f: Int32Array.from(t.f),
      t: Float64Array.from(t.t),
      m: Uint8Array.from(t.m),
      l: Int32Array.from(t.l),
      r: Int32Array.from(t.r),
      v: Float64Array.from(t.v),
      leaf: Uint8Array.from(t.leaf),
    })),
  );
  c = { trees, k: task.baseline.length, nClasses: task.labels.length };
  compiled.set(task, c);
  return c;
}

function leafValue(tree, x) {
  let i = 0;
  while (!tree.leaf[i]) {
    const val = x[tree.f[i]];
    if (Number.isNaN(val)) {
      i = tree.m[i] ? tree.l[i] : tree.r[i];
    } else {
      i = val <= tree.t[i] ? tree.l[i] : tree.r[i];
    }
  }
  return tree.v[i];
}

// task: one entry of models.json "tasks"; columnsObj: feature name -> Float64Array (length nRows).
// Returns a row-major Float64Array of length nRows * nClasses with class probabilities.
export function predictProba(task, columnsObj, nRows) {
  const { trees, k, nClasses } = compile(task);
  const cols = task.columns.map((name) => {
    const col = columnsObj[name];
    if (!col) throw new Error(`Missing feature column: ${name}`);
    if (col.length < nRows) throw new Error(`Feature column ${name} has ${col.length} rows, expected ${nRows}`);
    return col;
  });
  const nFeat = cols.length;
  const x = new Float64Array(nFeat);
  const raw = new Float64Array(k);
  const out = new Float64Array(nRows * nClasses);
  for (let row = 0; row < nRows; row++) {
    for (let j = 0; j < nFeat; j++) x[j] = cols[j][row];
    for (let c = 0; c < k; c++) raw[c] = task.baseline[c];
    for (let it = 0; it < trees.length; it++) {
      const iteration = trees[it];
      for (let c = 0; c < k; c++) raw[c] += leafValue(iteration[c], x);
    }
    const o = row * nClasses;
    if (k === 1) {
      const p1 = 1 / (1 + Math.exp(-raw[0]));
      out[o] = 1 - p1;
      out[o + 1] = p1;
    } else {
      let max = -Infinity;
      for (let c = 0; c < k; c++) if (raw[c] > max) max = raw[c];
      let sum = 0;
      for (let c = 0; c < k; c++) {
        const e = Math.exp(raw[c] - max);
        out[o + c] = e;
        sum += e;
      }
      for (let c = 0; c < k; c++) out[o + c] /= sum;
    }
  }
  return out;
}

// Label index per row using the task decision rule:
// threshold: positive label if p(positive) >= threshold, else the other label;
// argmax: the first class with the highest probability.
export function decide(task, proba, nRows) {
  const nClasses = task.labels.length;
  const out = new Int32Array(nRows);
  const d = task.decision;
  if (d.type === "threshold") {
    const pos = task.labels.indexOf(d.positive);
    if (pos < 0 || nClasses !== 2) throw new Error(`Bad threshold decision for labels ${task.labels}`);
    const neg = 1 - pos;
    for (let row = 0; row < nRows; row++) out[row] = proba[row * 2 + pos] >= d.threshold ? pos : neg;
  } else if (d.type === "argmax") {
    for (let row = 0; row < nRows; row++) {
      const o = row * nClasses;
      let best = 0;
      for (let c = 1; c < nClasses; c++) if (proba[o + c] > proba[o + best]) best = c;
      out[row] = best;
    }
  } else {
    throw new Error(`Unknown decision type: ${d.type}`);
  }
  return out;
}
