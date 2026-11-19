// Parity helper: runs web/js/model.js on feature rows written by tests/test_parity_model.py.
// Usage: node tests/parity/model_cli.mjs <models.json> <input.json> <output.json>
// input:  {"<task>": {"n": nRows, "columns": {"<feature>": [number | null (NaN), ...]}}}
// output: {"<task>": {"proba": [row-major probabilities], "labels": [label index per row]}}
import { readFileSync, writeFileSync } from "node:fs";
import { predictProba, decide } from "../../web/js/model.js";

const [modelsPath, inputPath, outputPath] = process.argv.slice(2);
if (!outputPath) {
  console.error("usage: node model_cli.mjs <models.json> <input.json> <output.json>");
  process.exit(2);
}
const models = JSON.parse(readFileSync(modelsPath, "utf8"));
const input = JSON.parse(readFileSync(inputPath, "utf8"));
const out = {};
for (const [name, block] of Object.entries(input)) {
  const task = models.tasks[name];
  if (!task) throw new Error(`Unknown task: ${name}`);
  const cols = {};
  for (const [f, values] of Object.entries(block.columns)) {
    cols[f] = Float64Array.from(values, (v) => (v === null ? NaN : v));
  }
  const proba = predictProba(task, cols, block.n);
  out[name] = { proba: Array.from(proba), labels: Array.from(decide(task, proba, block.n)) };
}
writeFileSync(outputPath, JSON.stringify(out));
