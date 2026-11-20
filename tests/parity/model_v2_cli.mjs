// Parity helper: runs web/js/model_v2.js and web/js/smoothing.js on inputs written by
// tests/test_parity_model_v2.py.
// Usage: node tests/parity/model_v2_cli.mjs <models_v2.json> <input.json> <output.json>
// input (every section optional; null in a column means NaN):
//   {"rows":   {"<task>": {"n": nRows, "columns": {"<feature>": [...]}}},
//    "nights": {"tasks": ["<task>", ...], "list": [{"n": nRows, "epoch": [...], "columns": {...}}, ...]},
//    "smooth": [{"proba": [row-major], "epoch": [...], "trans": [[...]], "prior": [...],
//                "init": [...], "method": "hmm_viterbi" | "hmm_forward_backward" | "none"}, ...]}
// output:
//   {"rows":   {"<staging task>": {"parts": [[row-major proba per model]], "proba": [...]},
//               "apnea": {"parts": [[p(apnea) per model]], "p": [...], "labels": [...]}},
//    "nights": {"<task>": [{"proba": [...], "labels": [...]}, ...]},
//    "smooth": [[row-major output], ...]}
import { readFileSync, writeFileSync } from "node:fs";
import { predictProba } from "../../web/js/model.js";
import { logisticProba, predictApnea, predictStaging } from "../../web/js/model_v2.js";
import { smoothRecord } from "../../web/js/smoothing.js";

const [modelsPath, inputPath, outputPath] = process.argv.slice(2);
if (!outputPath) {
  console.error("usage: node model_v2_cli.mjs <models_v2.json> <input.json> <output.json>");
  process.exit(2);
}
const models = JSON.parse(readFileSync(modelsPath, "utf8"));
const input = JSON.parse(readFileSync(inputPath, "utf8"));
const nanArray = (values) => Float64Array.from(values, (v) => (v === null ? NaN : v));
const toColumns = (obj) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, nanArray(v)]));
const taskOf = (name) => {
  const t = models.tasks[name];
  if (!t) throw new Error(`Unknown task: ${name}`);
  return t;
};

const out = {};
if (input.rows) {
  out.rows = {};
  for (const [name, block] of Object.entries(input.rows)) {
    const task = taskOf(name);
    const cols = toColumns(block.columns);
    if (name === "apnea") {
      const r = predictApnea(task, cols, block.n);
      out.rows[name] = { parts: r.parts.map((p) => Array.from(p)), p: Array.from(r.pApnea), labels: Array.from(r.labels) };
      // Cross-check: logisticProba directly equals the logistic part.
      task.models.forEach((m, i) => {
        if (m.kind !== "logistic") return;
        const lp = logisticProba(m, cols, block.n);
        if (lp.some((v, j) => v !== r.parts[i][j])) throw new Error("logistic part mismatch");
      });
    } else {
      const parts = task.models.map((m) => Array.from(predictProba(m, cols, block.n)));
      const r = predictStaging(task, cols, block.n);
      out.rows[name] = { parts, proba: Array.from(r.proba) };
    }
  }
}
if (input.nights) {
  out.nights = {};
  const nights = input.nights.list.map((night) => ({ ...night, columns: toColumns(night.columns) }));
  for (const name of input.nights.tasks) {
    const task = taskOf(name);
    out.nights[name] = nights.map((night) => {
      const r = predictStaging(task, night.columns, night.n, Int32Array.from(night.epoch));
      return { proba: Array.from(r.proba), labels: Array.from(r.labels) };
    });
  }
}
if (input.smooth) {
  out.smooth = input.smooth.map((c) =>
    Array.from(smoothRecord(Float64Array.from(c.proba), Int32Array.from(c.epoch), c.trans, c.prior, c.init, c.method)),
  );
}
writeFileSync(outputPath, JSON.stringify(out));
