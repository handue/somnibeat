// Module Web Worker: parses the RR CSV and builds the night report off the main thread.
// In:  {type: "run", csvText | edf, fileName, lightsOffS, lightsOnS, model?}
//      edf: a File/Blob holding an EDF recording; its ECG channel is converted to RR intervals
//      here (ecg.js, same detector as the evaluation). A csvText that is an RR-interval export
//      (one column named like "RR", no rr_ms) is converted the same way as scripts/ecg_to_rr.py.
//      model: "v2" (default) or "v1". v2 uses data/models_v2.json (report_v2.js); if that file
//      cannot be loaded the run falls back to v1 (data/models.json, report.js) and the report
//      carries a warning. Each models file is fetched lazily on first use and cached.
// Out: {type: "progress", step, status}  steps: read, ecg (EDF only), beats, features, report;
//                                          status: running | done
//      {type: "converted", csvText, info}  the RR file made from an EDF or RR export
//      {type: "result", report}            report.model is "v2" or "v1" (the models actually used)
//      {type: "error", code, message, details}
// Error codes come from rr.js (ERROR_CODES), ecg.js (ECG_ERROR_CODES) and report.js (REPORT_ERROR_CODES), plus
// MODELS_UNAVAILABLE (no models file could be loaded) and INTERNAL (unexpected failure,
// including an unknown model value).

import { parseRRCsv, MIN_HOURS } from "./rr.js";
import { buildReport } from "./report.js";
import { buildReportV2 } from "./report_v2.js";
import { readEdfChannel, channelToRR, rrCsv, rrExportColumn, rrExportToRR } from "./ecg.js";

const MODELS_URL = {
  v1: new URL("../data/models.json", import.meta.url),
  v2: new URL("../data/models_v2.json", import.meta.url),
};
const MODEL_NAMES = { v1: "prediction models", v2: "v2 prediction models" };

const modelsPromise = { v1: null, v2: null };

function loadModels(version) {
  if (!modelsPromise[version]) {
    modelsPromise[version] = fetch(MODELS_URL[version])
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .catch((err) => {
        modelsPromise[version] = null; // allow a retry on the next run
        const e = new Error(`The ${MODEL_NAMES[version]} could not be loaded (${err.message}).`);
        e.code = "MODELS_UNAVAILABLE";
        throw e;
      });
  }
  return modelsPromise[version];
}

// Resolves to {version, models, warning}: v2 when asked for (or by default) and available,
// otherwise v1 with a warning saying why.
async function loadFor(requested) {
  if (requested === "v2") {
    try {
      const models = await loadModels("v2");
      // A development export from the dry-run pickles must never pass silently as the frozen v2.
      const dryRun = models && models.source && models.source.dry_run === true;
      const warning = dryRun ? "These v2 models are a development (dry-run) export, not the final v2 models." : null;
      return { version: "v2", models, warning };
    } catch (err) {
      const models = await loadModels("v1");
      return {
        version: "v1",
        models,
        warning: `${err.message} This report uses the v1 models instead.`,
      };
    }
  }
  return { version: "v1", models: await loadModels("v1"), warning: null };
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
  const requested = msg.model === undefined || msg.model === null ? "v2" : msg.model;
  if (requested !== "v1" && requested !== "v2") throw new Error(`unknown model "${requested}"`);
  const progress = (step, status) => self.postMessage({ type: "progress", step, status });
  progress("read", "running");
  const loading = loadFor(requested);
  loading.catch(() => {}); // handled when awaited below
  let csvText = msg.csvText;
  let conv = null;
  if (msg.edf) {
    const channel = await readEdfChannel(msg.edf);
    progress("read", "done");
    progress("ecg", "running");
    conv = channelToRR(channel);
    progress("ecg", "done");
  } else {
    const col = rrExportColumn(String(csvText ?? ""));
    if (col) conv = rrExportToRR(csvText, col);
  }
  if (conv) {
    csvText = rrCsv(conv.time_s, conv.rr_ms);
    const info = { source: conv.source, beats: conv.beats, fs: conv.fs, label: conv.label,
      durationH: conv.durationH, startClock: conv.startClock, fromEcg: Boolean(msg.edf) };
    self.postMessage({ type: "converted", csvText, info });
  }
  const parsed = parseRRCsv(csvText);
  const { version, models, warning } = await loading;
  progress("read", "done");
  const input = { time_s: parsed.time_s, rr_ms: parsed.rr_ms, warnings: parsed.warnings };
  const opts = { lightsOffS: msg.lightsOffS, lightsOnS: msg.lightsOnS, fileName: msg.fileName || "", onProgress: progress };
  const report = version === "v2" ? buildReportV2(input, models, opts) : buildReport(input, models, opts);
  report.model = version;
  if (warning) report.quality.warnings.unshift(warning);
  self.postMessage({ type: "result", report });
}

self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type !== "run") return;
  run(msg).catch(postError);
};
