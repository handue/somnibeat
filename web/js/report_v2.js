// v2 night report from a parsed RR series. Mirror of somnibeat/report_v2.py
// (parity-tested in tests/test_parity_report_v2.py).
//
// Same analysis grid, RR cleaning, quality block, night metrics, error codes, progress steps
// and report shape as web/js/report.js (docs/web_contract.md), plus "model": "v2".
// Features: v1 epochFeatures + v2 longFeatures on the same grid (30 s epochs, 60 s minutes).
// Staging: gradient boosting -> whole-night HMM Viterbi (model_v2.js predictStaging).
// Apnea: mean of gradient boosting and logistic regression, thresholded (model_v2.js predictApnea).

import { cleanRR, MIN_HOURS } from "./rr.js";
import { epochFeatures, epochGrid } from "./features.js";
import { longFeatures } from "./features_v2.js";
import { predictStaging, predictApnea } from "./model_v2.js";
import { nightMetrics, INVALID_WARN_FRACTION, OUTSIDE_WARN_S, REPORT_ERROR_CODES } from "./report.js";

export const STAGE_TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"];

function reportError(code, message, details = {}) {
  const e = new Error(message);
  e.code = code;
  e.details = details;
  return e;
}

const given = (x) => x !== null && x !== undefined;

/** v1 epoch features followed by the v2 long-context columns (no name may repeat). */
export function nightFeatures(t, rr, valid, starts, epochLen, cfg) {
  const out = { ...epochFeatures(t, rr, valid, starts, epochLen, cfg.rr, cfg.features) };
  const extra = longFeatures(t, rr, valid, starts, epochLen, cfg.rr);
  for (const [name, col] of Object.entries(extra)) {
    if (name in out) throw new Error(`v1 and v2 feature names overlap: ${name}`);
    out[name] = col;
  }
  return out;
}

/**
 * buildReportV2({time_s, rr_ms, warnings?}, modelsV2, opts) -> report (docs/web_contract.md, plus model: "v2").
 * modelsV2: parsed data/models_v2.json. opts: {lightsOffS, lightsOnS, fileName, onProgress(step, status), includeProba}.
 * Progress steps emitted here: "beats", "features", "report" (as report.js).
 * Throws Error with .code (REPORT_ERROR_CODES) and .details for an unusable grid.
 */
export function buildReportV2(input, modelsV2, opts = {}) {
  const { lightsOffS = null, lightsOnS = null, fileName = "", onProgress = null, includeProba = false } = opts;
  const progress = (step, status) => { if (onProgress) onProgress(step, status); };
  const cfg = modelsV2.config;
  const epochS = cfg.epoch_s ?? 30;
  const minuteS = cfg.apnea_epoch_s ?? 60;
  const t = Float64Array.from(input.time_s);
  const rr = Float64Array.from(input.rr_ms);
  const n = rr.length;
  if (n < 2 || t.length !== n) {
    throw reportError(REPORT_ERROR_CODES.TOO_SHORT, "The file has fewer than 2 usable rows.", { rows: n, min_hours: MIN_HOURS });
  }

  for (const [name, v] of [["lights_off_s", lightsOffS], ["lights_on_s", lightsOnS]]) {
    if (given(v) && !(Number.isFinite(Number(v)) && Number(v) >= 0)) {
      throw reportError(REPORT_ERROR_CODES.BAD_LIGHTS, "The lights-off or lights-on time is not valid.", { [name]: v });
    }
  }
  const startS = given(lightsOffS) ? Number(lightsOffS) : 0.0;
  const endS = given(lightsOnS) ? Number(lightsOnS) : t[n - 1];
  if (given(lightsOffS) && given(lightsOnS) && !(endS > startS)) {
    throw reportError(REPORT_ERROR_CODES.BAD_LIGHTS, "Lights on must be after lights off.",
      { lights_off_s: startS, lights_on_s: endS });
  }

  const recStart = t[0] - rr[0] / 1000.0;
  const recEnd = t[n - 1];
  const minS = MIN_HOURS * 3600.0;
  const stageStarts = epochGrid(startS, endS, epochS);
  const nEpochs = stageStarts.length;
  if (nEpochs * epochS < minS) {
    throw reportError(REPORT_ERROR_CODES.TOO_SHORT,
      `The time between lights off and lights on is shorter than ${MIN_HOURS} hours.`,
      { hours: (nEpochs * epochS) / 3600.0, min_hours: MIN_HOURS });
  }
  const overlap = Math.min(endS, recEnd) - Math.max(startS, recStart);
  if (overlap < minS) {
    throw reportError(REPORT_ERROR_CODES.TOO_SHORT,
      `The recording has fewer than ${MIN_HOURS} hours of beats between lights off and lights on.`,
      { hours: Math.max(overlap, 0.0) / 3600.0, min_hours: MIN_HOURS });
  }

  progress("beats", "running");
  const valid = cleanRR(t, rr, cfg.rr);
  progress("beats", "done");

  progress("features", "running");
  const minuteStarts = epochGrid(startS, endS, minuteS);
  const nMinutes = minuteStarts.length;
  const feats30 = nightFeatures(t, rr, valid, stageStarts, epochS, cfg);
  const feats60 = nightFeatures(t, rr, valid, minuteStarts, minuteS, cfg);
  progress("features", "done");

  progress("report", "running");
  const stages = {};
  const proba = {};
  for (const name of STAGE_TASKS) {
    const task = modelsV2.tasks[name];
    const res = predictStaging(task, feats30, nEpochs);
    stages[name] = Array.from(res.labels, (i) => task.labels[i]);
    proba[name] = res.proba;
  }
  const apneaTask = modelsV2.tasks.apnea;
  const pos = apneaTask.labels.indexOf(apneaTask.decision.positive);
  const ap = predictApnea(apneaTask, feats60, nMinutes);
  const apneaLabels = Array.from(ap.labels, (i) => (i === pos ? 1 : 0));
  proba.apnea = ap.pApnea;

  const isWake = stages.sleep_wake.map((s) => s === "wake");
  const nm = nightMetrics(isWake);
  let apneaCount = 0;
  for (const v of apneaLabels) apneaCount += v;
  const metrics = { ...nm, apnea_min: apneaCount * (minuteS / 60.0) };

  let nValid = 0;
  for (let i = 0; i < n; i++) if (valid[i]) nValid++;
  const invalid = n - nValid;
  const invalidPct = (100.0 * invalid) / n;
  const warnings = (input.warnings || []).map(String);
  if (invalid / n > INVALID_WARN_FRACTION) {
    warnings.push(`${Math.floor(invalidPct + 0.5)}% of beats were missing or unusable.`);
  }
  if (startS < recStart - OUTSIDE_WARN_S || endS > recEnd + OUTSIDE_WARN_S) {
    warnings.push("Part of the analysed time has no recorded beats (before the first or after the last beat); " +
      "those epochs are marked as not enough usable data.");
  }

  const report = {
    model: "v2",
    fileName,
    grid: { start_s: startS, end_s: endS, epoch_s: epochS, n_epochs: nEpochs },
    stages,
    lowQuality: Array.from(feats30.low_quality, (v) => (v === 1 ? 1 : 0)),
    apnea: { minute_s: minuteS, labels: apneaLabels },
    metrics,
    quality: {
      valid_fraction: nValid / n,
      invalid_pct: invalidPct,
      n_beats: n,
      duration_h: (recEnd - recStart) / 3600.0,
      warnings,
    },
  };
  if (includeProba) report.proba = Object.fromEntries(Object.entries(proba).map(([k, v]) => [k, Array.from(v)]));
  progress("report", "done");
  return report;
}
