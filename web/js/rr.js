// RR CSV parsing and RR-interval cleaning.
// Mirror of somnibeat/rr.py (parity-tested in tests/test_parity_features.py).
// See docs/web_contract.md for the CSV format and error rules.

export const MIN_HOURS = 2;
export const MAX_NON_INCREASING_FRACTION = 0.01;
export const MEDIAN_RR_RANGE_MS = [300, 2000];

export const ERROR_CODES = Object.freeze({
  EMPTY: "EMPTY",                  // no header row or no data rows
  NO_RR_COLUMN: "NO_RR_COLUMN",    // header has no rr_ms column
  TOO_SHORT: "TOO_SHORT",          // fewer than MIN_HOURS of beats
  NOT_INCREASING: "NOT_INCREASING", // time_s not increasing in more than 1 % of rows
  BAD_UNIT: "BAD_UNIT",            // median rr_ms outside 300-2000 ms
  BAD_TIME_UNIT: "BAD_TIME_UNIT",  // time_s steps do not match rr_ms (e.g. time_s in milliseconds)
  BAD_TIME_START: "BAD_TIME_START", // time_s starts more than 24 h after 0 (e.g. clock or Unix time)
  TIME_RESET: "TIME_RESET",        // time_s jumps back by more than TIME_RESET_S once
});

// time_s checks (docs/web_contract.md). They only reject files the analysis grid cannot use.
export const TIME_STEP_RATIO_RANGE = [0.5, 2]; // median step of time_s (ms) / median rr_ms
export const MAX_START_S = 86400;
export const TIME_RESET_S = 10;

function rrError(code, message, details = {}) {
  const e = new Error(message);
  e.code = code;
  e.details = details;
  return e;
}

const NUMERIC_RE = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

function parseCell(raw) {
  if (raw === undefined) return NaN;
  let s = raw.trim();
  if (s.length >= 2 && s[0] === '"' && s[s.length - 1] === '"') s = s.slice(1, -1).trim();
  if (!NUMERIC_RE.test(s)) return NaN;
  const v = Number(s);
  return Number.isFinite(v) ? v : NaN;
}

function detectDelimiter(headerLine) {
  if (headerLine.includes(",")) return ",";
  if (headerLine.includes(";")) return ";";
  if (headerLine.includes("\t")) return "\t";
  return ",";
}

function splitHeaderCell(c) {
  let s = c.trim();
  if (s.length >= 2 && s[0] === '"' && s[s.length - 1] === '"') s = s.slice(1, -1).trim();
  return s.toLowerCase();
}

/** np.median for a finite Float64Array / array (copy is sorted). */
export function median(values) {
  const n = values.length;
  if (n === 0) return NaN;
  const x = Float64Array.from(values).sort();
  const h = n >> 1;
  if (n % 2 === 1) return x[h] / 1;
  // np.mean of the two middle values: (0 + a + b) / 2
  return (0 + x[h - 1] + x[h]) / 2;
}

/**
 * Parse an uploaded RR CSV (header row required; columns time_s and rr_ms).
 * Returns {time_s, rr_ms, warnings, skippedRows}; throws Error with .code (ERROR_CODES).
 */
export function parseRRCsv(text) {
  if (typeof text !== "string") text = String(text ?? "");
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const lines = text.split(/\r\n|\n|\r/);
  let li = 0;
  while (li < lines.length && lines[li].trim() === "") li++;
  if (li >= lines.length) throw rrError(ERROR_CODES.EMPTY, "The file is empty.");
  const delim = detectDelimiter(lines[li]);
  const header = lines[li].split(delim).map(splitHeaderCell);
  li++;
  const iRR = header.indexOf("rr_ms");
  const iT = header.indexOf("time_s");
  if (iRR < 0) {
    throw rrError(ERROR_CODES.NO_RR_COLUMN, "The file has no rr_ms column.", { header });
  }
  const tList = [];
  const rrList = [];
  let skipped = 0;
  let decimalComma = false;
  for (; li < lines.length; li++) {
    const line = lines[li];
    if (line.trim() === "") continue;
    const cells = line.split(delim);
    const rr = parseCell(cells[iRR]);
    const t = iT >= 0 ? parseCell(cells[iT]) : 0;
    if (Number.isNaN(rr) || Number.isNaN(t)) {
      const commaNum = (c) => /^\s*"?[+-]?\d+,\d+"?\s*$/.test(c || "");
      if (delim === ";" && (commaNum(cells[iRR]) || (iT >= 0 && commaNum(cells[iT])))) decimalComma = true;
      skipped++;
      continue;
    }
    rrList.push(rr);
    if (iT >= 0) tList.push(t);
  }
  const warnings = [];
  if (skipped > 0) {
    warnings.push(`${skipped} row${skipped === 1 ? "" : "s"} with a non-numeric value ${skipped === 1 ? "was" : "were"} skipped.`);
  }
  const n = rrList.length;
  if (n < 2) {
    let msg = "The file has fewer than 2 usable rows.";
    if (skipped > 0) {
      msg += ` ${skipped} row${skipped === 1 ? " has" : "s have"} a value that is not a number.`;
      if (decimalComma) {
        msg += " Decimal commas (812,5) are not supported: use a dot (812.5).";
      }
    }
    throw rrError(ERROR_CODES.TOO_SHORT, msg, { rows: n, hours: 0, skipped_rows: skipped });
  }
  const rr_ms = Float64Array.from(rrList);
  let time_s;
  if (iT >= 0) {
    time_s = Float64Array.from(tList);
  } else {
    // rr_series_from_csv_rows: np.cumsum(rr) / 1000.0 (sequential cumulative sum)
    time_s = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc += rr_ms[i];
      time_s[i] = acc / 1000.0;
    }
    warnings.push("No time_s column: beat times were computed as the cumulative sum of rr_ms.");
  }
  const med = median(rr_ms);
  if (!(med >= MEDIAN_RR_RANGE_MS[0] && med <= MEDIAN_RR_RANGE_MS[1])) {
    throw rrError(ERROR_CODES.BAD_UNIT,
      "The median rr_ms is outside 300-2000 ms. Is the column in seconds instead of milliseconds?",
      { median_rr_ms: med });
  }
  let nonIncreasing = 0;
  for (let i = 1; i < n; i++) if (!(time_s[i] > time_s[i - 1])) nonIncreasing++;
  if (nonIncreasing > MAX_NON_INCREASING_FRACTION * (n - 1)) {
    throw rrError(ERROR_CODES.NOT_INCREASING, "time_s is not increasing in more than 1 % of rows.",
      { non_increasing_rows: nonIncreasing, rows: n });
  }
  if (iT >= 0) {
    for (let i = 1; i < n; i++) {
      if (time_s[i] < time_s[i - 1] - TIME_RESET_S) {
        throw rrError(ERROR_CODES.TIME_RESET,
          `time_s jumps back from ${time_s[i - 1].toFixed(1)} s to ${time_s[i].toFixed(1)} s at data row ${i + 1}. Is this two recordings in one file, or a clock that restarted? Upload one night per file.`,
          { row: i + 1, from_s: time_s[i - 1], to_s: time_s[i] });
      }
    }
    const steps = new Float64Array(n - 1);
    for (let i = 1; i < n; i++) steps[i - 1] = time_s[i] - time_s[i - 1];
    const ratio = (median(steps) * 1000.0) / med;
    if (!(ratio >= TIME_STEP_RATIO_RANGE[0] && ratio <= TIME_STEP_RATIO_RANGE[1])) {
      const hint = ratio > 100 ? " Is time_s in milliseconds instead of seconds?" : "";
      throw rrError(ERROR_CODES.BAD_TIME_UNIT,
        `The steps between time_s values do not match the rr_ms intervals (about ${fmtRatio(ratio)} times larger).${hint}`,
        { step_ratio: ratio });
    }
    const start = time_s[0] - rr_ms[0] / 1000.0;
    if (start > MAX_START_S) {
      throw rrError(ERROR_CODES.BAD_TIME_START,
        `time_s starts at ${Math.round(time_s[0])} s. It should count seconds since the recording started, not clock or calendar time.`,
        { first_time_s: time_s[0] });
    }
  }
  // Span covered by the beats: from the start of the first interval to the last beat.
  const hours = (time_s[n - 1] - (time_s[0] - rr_ms[0] / 1000.0)) / 3600.0;
  if (!(hours >= MIN_HOURS)) {
    throw rrError(ERROR_CODES.TOO_SHORT, `The recording covers fewer than ${MIN_HOURS} hours of beats.`,
      { hours, rows: n });
  }
  return { time_s, rr_ms, warnings, skippedRows: skipped };
}

function fmtRatio(r) {
  if (!Number.isFinite(r)) return "—";
  return r >= 10 ? String(Math.round(r)) : r.toFixed(2);
}

/** rr_from_beats: (t[1:], diff(t) * 1000). */
export function rrFromBeats(beatT) {
  const n = beatT.length;
  const m = Math.max(0, n - 1);
  const t_s = new Float64Array(m);
  const rr_ms = new Float64Array(m);
  for (let i = 1; i < n; i++) {
    t_s[i - 1] = beatT[i];
    rr_ms[i - 1] = (beatT[i] - beatT[i - 1]) * 1000.0;
  }
  return { t_s, rr_ms };
}

/** clean_rr: R1 range rule then R2 local-median rule. Returns Uint8Array (1 = valid). */
export function cleanRR(t, rr, cfgRR) {
  const n = rr.length;
  const lo = cfgRR.min_rr_ms;
  const hi = cfgRR.max_rr_ms;
  const valid = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const v = rr[i];
    valid[i] = v >= lo && v <= hi && Number.isFinite(v) ? 1 : 0;
  }
  const half = Math.trunc(cfgRR.local_median_beats) >> 1; // int(x) // 2 for x >= 0
  const jump = Number(cfgRR.max_relative_jump);
  const out = Uint8Array.from(valid);
  const w = new Float64Array(2 * half + 1);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const a = Math.max(0, i - half);
    const b = Math.min(n, i + half + 1);
    let k = 0;
    for (let j = a; j < b; j++) if (valid[j]) w[k++] = rr[j];
    const m = median(w.subarray(0, k));
    if (Math.abs(rr[i] - m) / m > jump) out[i] = 0;
  }
  return out;
}

/** Fraction of intervals that are invalid after cleaning (for the "% unusable" warning). */
export function invalidFraction(valid) {
  const n = valid.length;
  if (!n) return 1;
  let bad = 0;
  for (let i = 0; i < n; i++) if (!valid[i]) bad++;
  return bad / n;
}
