// Module Web Worker: parses the RR CSV and builds the night report off the main thread.
// In:  {type: "run", csvText, fileName, lightsOffS, lightsOnS}
// Out: {type: "progress", step, status}  steps: read, beats, features, report; status: running | done
//      {type: "result", report}
//      {type: "error", code, message, details}
// Error codes come from rr.js (ERROR_CODES) and report.js (REPORT_ERROR_CODES), plus
// MODELS_UNAVAILABLE (data/models.json could not be loaded) and INTERNAL (unexpected failure).

import { parseRRCsv, MIN_HOURS } from "./rr.js";
import { buildReport } from "./report.js";

const MODELS_URL = new URL("../data/models.json", import.meta.url);

let modelsPromise = null;

function loadModels() {
  if (!modelsPromise) {
    modelsPromise = fetch(MODELS_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .catch((err) => {
        modelsPromise = null; // allow a retry on the next run
        const e = new Error(`The prediction models could not be loaded (${err.message}).`);
        e.code = "MODELS_UNAVAILABLE";
        throw e;
      });
  }
  return modelsPromise;
}

function postError(err) {
  const code = err && err.code ? err.code : "INTERNAL";
  const message = code === "INTERNAL"
    ? `The analysis failed unexpectedly (${err && err.message ? err.message : String(err)}).`
    : err.message;
  const details = { ...(err && err.details && typeof err.details === "object" ? err.details : {}) };
  if (code === "TOO_SHORT" && details.min_hours === undefined) details.min_hours = MIN_HOURS;
  self.postMessage({ type: "error", code, message, details });
}

async function run(msg) {
  const progress = (step, status) => self.postMessage({ type: "progress", step, status });
  progress("read", "running");
  const models = loadModels();
  models.catch(() => {}); // handled when awaited below
  const parsed = parseRRCsv(msg.csvText);
  const loaded = await models;
  progress("read", "done");
  const report = buildReport(
    { time_s: parsed.time_s, rr_ms: parsed.rr_ms, warnings: parsed.warnings },
    loaded,
    { lightsOffS: msg.lightsOffS, lightsOnS: msg.lightsOnS, fileName: msg.fileName || "", onProgress: progress },
  );
  self.postMessage({ type: "result", report });
}

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type !== "run") return;
  run(msg).catch(postError);
};
