// Parity CLI: node features_v2_cli.mjs <rr_csv> <epoch_len> <start_s> <end_s|auto> <out_json> <config_json>
// Reads the CSV with parseRRCsv, cleans it with config [rr], builds the analysis grid
// (docs/web_contract.md) and computes the v2 long-context features (web/js/features_v2.js).
// Writes {columns: {name: [values, NaN as null]}, valid: [0/1...], grid: [...], timing_ms: {...}}.
// config_json: {"rr": {...config [rr]}} (written by the test from config/v2.toml).
// Extra mode: node features_v2_cli.mjs --raw <in_json> <out_json> runs on {t, rr, epoch_starts,
// epoch_len, cfg_rr, valid?} and optionally {baseline: true} (also returns hrBaseline and the
// acceleration onsets), {pct: {x, ref}} (percentileRank) and {lomb_check: true} (number of w20
// windows where lombPowerCached differs in any bit from features.js lombPower; must be 0).

import { readFileSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { parseRRCsv, cleanRR } from "../../web/js/rr.js";
import { epochGrid, lombPower, searchsortedLeft } from "../../web/js/features.js";
import {
  longFeatures, hrBaseline, detectAccelerations, percentileRank, lombPowerCached, CVHR_FREQS, HALF_W20,
} from "../../web/js/features_v2.js";

const toJson = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));
const fromJson = (a) => Float64Array.from(a, (v) => (v === null ? NaN : v));

function columnsOut(feats) {
  const cols = {};
  for (const [k, v] of Object.entries(feats)) cols[k] = toJson(v);
  return cols;
}

function lombCheck(t, rr, valid, starts, epochLen) {
  let windows = 0, mismatches = 0;
  for (const s of starts) {
    const c = s + epochLen / 2;
    const i0 = searchsortedLeft(t, c - HALF_W20), i1 = searchsortedLeft(t, c + HALF_W20);
    const idx = [];
    for (let i = i0; i < i1; i++) if (valid[i]) idx.push(i);
    if (idx.length < 2) continue;
    const a = lombPowerCached(t, rr, Int32Array.from(idx), idx.length);
    const b = lombPower(Float64Array.from(idx, (i) => t[i]), Float64Array.from(idx, (i) => rr[i]), CVHR_FREQS);
    windows++;
    if (a.some((v, j) => !Object.is(v, b[j]))) mismatches++;
  }
  return { windows, mismatches };
}

function main(argv) {
  if (argv[0] === "--raw") {
    const inp = JSON.parse(readFileSync(argv[1], "utf8"));
    const t = fromJson(inp.t), rr = fromJson(inp.rr);
    const valid = inp.valid ? Uint8Array.from(inp.valid) : cleanRR(t, rr, inp.cfg_rr);
    const starts = fromJson(inp.epoch_starts);
    const t0 = performance.now();
    const feats = longFeatures(t, rr, valid, starts, inp.epoch_len, inp.cfg_rr);
    const t1 = performance.now();
    const out = { columns: columnsOut(feats), valid: Array.from(valid), freqs: Array.from(CVHR_FREQS),
      timing_ms: { features: t1 - t0 } };
    if (inp.baseline) {
      out.baseline = toJson(hrBaseline(t, rr, valid));
      out.onsets = Array.from(detectAccelerations(t, rr, valid));
    }
    if (inp.lomb_check) out.lomb_check = lombCheck(t, rr, valid, starts, inp.epoch_len);
    if (inp.pct) out.pct = toJson(percentileRank(fromJson(inp.pct.x), fromJson(inp.pct.ref)));
    writeFileSync(argv[2], JSON.stringify(out));
    return;
  }
  const [csvPath, lenArg, startArg, endArg, outPath, cfgPath] = argv;
  if (!cfgPath) {
    console.error("usage: node features_v2_cli.mjs <rr_csv> <epoch_len> <start_s> <end_s|auto> <out_json> <config_json>");
    process.exit(2);
  }
  const cfg = JSON.parse(readFileSync(cfgPath, "utf8"));
  const epochLen = Number(lenArg);
  const parsed = parseRRCsv(readFileSync(csvPath, "utf8"));
  const valid = cleanRR(parsed.time_s, parsed.rr_ms, cfg.rr);
  const startS = Number(startArg);
  const endS = endArg === "auto" ? parsed.time_s[parsed.time_s.length - 1] : Number(endArg);
  const grid = epochGrid(startS, endS, epochLen);
  const t0 = performance.now();
  const feats = longFeatures(parsed.time_s, parsed.rr_ms, valid, grid, epochLen, cfg.rr);
  const t1 = performance.now();
  writeFileSync(outPath, JSON.stringify({
    columns: columnsOut(feats), valid: Array.from(valid), grid: Array.from(grid),
    timing_ms: { features: t1 - t0 },
  }));
}

main(process.argv.slice(2));
