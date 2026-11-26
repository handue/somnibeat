// Night report from a parsed RR series: grid, features, predictions, night metrics.
// Mirror of somnibeat/report.py (parity-tested in tests/test_parity_report.py).
// See docs/web_contract.md for the analysis grid, prediction and night-metric rules.

import { cleanRR, MIN_HOURS } from "./rr.js";
import { epochFeatures, epochGrid } from "./features.js";
import { predictProba, decide } from "./model.js";

export const INVALID_WARN_FRACTION = 0.10; // warn above 10 % invalid intervals
export const OUTSIDE_WARN_S = 60.0; // warn when the grid extends this far beyond the recorded beats
export const EPOCH_MIN = 0.5; // somnibeat/evaluate.py EPOCH_MIN

export const REPORT_ERROR_CODES = Object.freeze({
  TOO_SHORT: "TOO_SHORT", // grid or beats inside it shorter than MIN_HOURS
  BAD_LIGHTS: "BAD_LIGHTS", // lights-off/on not usable
});

function reportError(code, message, details = {}) {
  const e = new Error(message);
  e.code = code;
  e.details = details;
  return e;
}

const given = (x) => x !== null && x !== undefined;

/** somnibeat.evaluate.awakenings without epoch_idx: wake runs of >= minEpochs between first and last sleep epoch. */
export function awakenings(isWake, minEpochs = 2) {
  let first = -1;
  let last = -1;
  for (let i = 0; i < isWake.length; i++) {
    if (!isWake[i]) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return 0;
  let count = 0;
  let run = 0;
  for (let i = first; i <= last; i++) {
    if (isWake[i]) {
      run += 1;
    } else {
      if (run >= minEpochs) count += 1;
      run = 0;
    }
  }
  return count;
}

/** somnibeat.evaluate.night_metrics(is_wake) without epoch_idx. */
export function nightMetrics(isWake) {
  const n = isWake.length;
  let sleep = 0;
  for (let i = 0; i < n; i++) if (!isWake[i]) sleep++;
  const tst = sleep * EPOCH_MIN;
  const tib = n * EPOCH_MIN;
  return { tib_min: tib, tst_min: tst, se_pct: tib ? (100.0 * tst) / tib : NaN, awakenings: awakenings(isWake) };
}

/**
 * buildReport({time_s, rr_ms, warnings?}, models, opts) -> report (docs/web_contract.md).
 * models: parsed data/models.json. opts: {lightsOffS, lightsOnS, fileName, onProgress(step, status), includeProba}.
 * Progress steps emitted here: "beats", "features", "report" ("read" belongs to the caller that parses the file).
 * Throws Error with .code (REPORT_ERROR_CODES) and .details for an unusable grid.
 */
export function buildReport(input, models, opts = {}) {
  const { lightsOffS = null, lightsOnS = null, fileName = "", onProgress = null, includeProba = false } = opts;
  const progress = (step, status) => { if (onProgress) onProgress(step, status); };
  const cfg = models.config;
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

  // Same span rule as parseRRCsv: from the start of the first interval to the last beat.
  const recStart = t[0] - rr[0] / 1000.0;
  const recEnd = t[n - 1];
  const minS = MIN_HOURS * 3600.0;
  const stageStarts = epochGrid(startS, endS, epochS);
  const nEpochs = stageStarts.length;
  if (nEpochs * epochS < minS) {
    throw reportError(REPORT_ERROR_CODES.TOO_SHORT,
      (opts.lightsOffS != null || opts.lightsOnS != null)
        ? `The time between lights off and lights on is shorter than ${MIN_HOURS} hours.`
        : `The recording covers fewer than ${MIN_HOURS} hours of beats.`,
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
  const feats30 = epochFeatures(t, rr, valid, stageStarts, epochS, cfg.rr, cfg.features);
  const feats60 = epochFeatures(t, rr, valid, minuteStarts, minuteS, cfg.rr, cfg.features);
  progress("features", "done");

  progress("report", "running");
  const stages = {};
  const proba = {};
  for (const name of ["sleep_wake", "wake_nrem_rem", "four_class"]) {
    const task = models.tasks[name];
    const p = predictProba(task, feats30, nEpochs);
    stages[name] = Array.from(decide(task, p, nEpochs), (i) => task.labels[i]);
    proba[name] = p;
  }
  const apneaTask = models.tasks.apnea;
  const nMinutes = minuteStarts.length;
  const pA = predictProba(apneaTask, feats60, nMinutes);
  const pos = apneaTask.labels.indexOf(apneaTask.decision.positive);
  const apneaLabels = Array.from(decide(apneaTask, pA, nMinutes), (i) => (i === pos ? 1 : 0));
  proba.apnea = pA;

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
