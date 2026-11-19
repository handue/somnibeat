// Parity CLI: node features_cli.mjs <rr_csv> <epoch_len> <start_s> <end_s> <out_json> [config_json]
// Reads the CSV with parseRRCsv, cleans it, builds the analysis grid (docs/web_contract.md),
// computes epoch features and writes {columns: {name: [values, NaN as null]}, valid: [0/1...],
// grid: [...], timing_ms: {...}}. end_s may be "auto" (time of the last beat).
// Optional config_json (default: "config" of web/data/models.json): {"rr": {...config [rr]}, "features": {...config [features]}} (written by the test from config/default.toml).
// Extra mode: node features_cli.mjs --raw <in_json> <out_json> runs on {t, rr, epoch_starts,
// epoch_len, cfg_rr, cfg_feat} (used for synthetic edge cases and for lombPower / dfaAlpha1 checks).

import { readFileSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { parseRRCsv, cleanRR } from "../../web/js/rr.js";
import { epochFeatures, epochGrid, lombPower, dfaAlpha1, FREQS } from "../../web/js/features.js";

const toJson = (a) => Array.from(a, (v) => (Number.isFinite(v) ? v : null));
const fromJson = (a) => Float64Array.from(a, (v) => (v === null ? NaN : v));

function columnsOut(feats) {
  const cols = {};
  for (const [k, v] of Object.entries(feats)) cols[k] = toJson(v);
  return cols;
}

function loadConfig(cfgPath) {
  if (cfgPath) return JSON.parse(readFileSync(cfgPath, "utf8"));
  // Default: the config exported with the models (web/data/models.json, written by scripts/export_web.py).
  const models = new URL("../../web/data/models.json", import.meta.url);
  try {
    return JSON.parse(readFileSync(models, "utf8")).config;
  } catch {
    console.error("no config_json given and web/data/models.json is not available");
    process.exit(2);
  }
}

function main(argv) {
  if (argv[0] === "--raw") {
    const inp = JSON.parse(readFileSync(argv[1], "utf8"));
    const t = fromJson(inp.t), rr = fromJson(inp.rr);
    const valid = inp.valid ? Uint8Array.from(inp.valid) : cleanRR(t, rr, inp.cfg_rr);
    const starts = fromJson(inp.epoch_starts);
    const feats = epochFeatures(t, rr, valid, starts, inp.epoch_len, inp.cfg_rr, inp.cfg_feat);
    const out = { columns: columnsOut(feats), valid: Array.from(valid), freqs: Array.from(FREQS) };
    if (inp.lomb) out.lomb = toJson(lombPower(fromJson(inp.lomb.t), fromJson(inp.lomb.y)));
    if (inp.dfa) out.dfa = inp.dfa.map((x) => { const v = dfaAlpha1(fromJson(x)); return Number.isFinite(v) ? v : null; });
    writeFileSync(argv[2], JSON.stringify(out));
    return;
  }
  const [csvPath, lenArg, startArg, endArg, outPath, cfgPath] = argv;
  if (!outPath) {
    console.error("usage: node features_cli.mjs <rr_csv> <epoch_len> <start_s> <end_s|auto> <out_json> [config_json]");
    process.exit(2);
  }
  const cfg = loadConfig(cfgPath);
  const epochLen = Number(lenArg);
  const t0 = performance.now();
  const parsed = parseRRCsv(readFileSync(csvPath, "utf8"));
  const t1 = performance.now();
  const valid = cleanRR(parsed.time_s, parsed.rr_ms, cfg.rr);
  const t2 = performance.now();
  const startS = Number(startArg);
  const endS = endArg === "auto" ? parsed.time_s[parsed.time_s.length - 1] : Number(endArg);
  const grid = epochGrid(startS, endS, epochLen);
  const feats = epochFeatures(parsed.time_s, parsed.rr_ms, valid, grid, epochLen, cfg.rr, cfg.features);
  const t3 = performance.now();
  writeFileSync(outPath, JSON.stringify({
    columns: columnsOut(feats), valid: Array.from(valid), grid: Array.from(grid),
    warnings: parsed.warnings, skippedRows: parsed.skippedRows,
    timing_ms: { parse: t1 - t0, clean: t2 - t1, features: t3 - t2 },
  }));
}

main(process.argv.slice(2));
