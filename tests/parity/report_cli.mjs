// Parity CLI for web/js/report.js (run by tests/test_parity_report.py).
//
// Report mode:
//   node report_cli.mjs <rr_csv> <out_json> [--lights-off S] [--lights-on S] [--models PATH]
//   Parses the CSV with parseRRCsv, runs buildReport (with probabilities) and writes
//   {report, timing_ms: {parse, report, total}, progress: [[step, status], ...]}.
//   Errors are written as {error: {code, message, details}}.
//
// Metrics mode: node report_cli.mjs --metrics <in_json> <out_json> (night metrics of 0/1 wake arrays).
//
// Worker mode:
//   node report_cli.mjs --worker <in_json> <out_json>
//   Loads web/js/worker.js with a minimal self/fetch shim and sends each message of
//   in_json ({runs: [{csvPath, fileName, lightsOffS, lightsOnS}]}) in turn. Writes the
//   posted messages per run plus how often data/models.json was fetched.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { performance } from "node:perf_hooks";
import { parseRRCsv } from "../../web/js/rr.js";
import { buildReport, nightMetrics } from "../../web/js/report.js";

const DEFAULT_MODELS = fileURLToPath(new URL("../../web/data/models.json", import.meta.url));

function errorOut(err) {
  return { error: { code: err.code || "INTERNAL", message: err.message, details: err.details || {} } };
}

function reportMode(argv) {
  const [csvPath, outPath, ...rest] = argv;
  const opts = { lightsOffS: null, lightsOnS: null, models: DEFAULT_MODELS };
  for (let i = 0; i < rest.length; i += 2) {
    if (rest[i] === "--lights-off") opts.lightsOffS = Number(rest[i + 1]);
    else if (rest[i] === "--lights-on") opts.lightsOnS = Number(rest[i + 1]);
    else if (rest[i] === "--models") opts.models = rest[i + 1];
    else throw new Error(`unknown option ${rest[i]}`);
  }
  const models = JSON.parse(readFileSync(opts.models, "utf8"));
  const text = readFileSync(csvPath, "utf8");
  const progress = [];
  let out;
  const t0 = performance.now();
  try {
    const parsed = parseRRCsv(text);
    const t1 = performance.now();
    const report = buildReport(
      { time_s: parsed.time_s, rr_ms: parsed.rr_ms, warnings: parsed.warnings },
      models,
      {
        lightsOffS: opts.lightsOffS, lightsOnS: opts.lightsOnS, fileName: csvPath.split("/").pop(),
        includeProba: true, onProgress: (step, status) => progress.push([step, status]),
      },
    );
    const t2 = performance.now();
    out = { report, progress, timing_ms: { parse: t1 - t0, report: t2 - t1, total: t2 - t0 } };
  } catch (err) {
    out = errorOut(err);
  }
  writeFileSync(outPath, JSON.stringify(out, (k, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v)));
}

async function workerMode(inPath, outPath) {
  const input = JSON.parse(readFileSync(inPath, "utf8"));
  let fetches = 0;
  let posted = [];
  let resolveRun = null;
  globalThis.fetch = async (url) => {
    fetches++;
    const text = readFileSync(fileURLToPath(url), "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(text) };
  };
  globalThis.self = {
    postMessage(msg) {
      posted.push(structuredClone(msg));
      if ((msg.type === "result" || msg.type === "error") && resolveRun) resolveRun();
    },
  };
  await import("../../web/js/worker.js");
  const runs = [];
  for (const r of input.runs) {
    posted = [];
    const done = new Promise((res) => { resolveRun = res; });
    const csvText = r.csvText ?? readFileSync(r.csvPath, "utf8");
    globalThis.self.onmessage({ data: { type: "run", csvText, fileName: r.fileName, lightsOffS: r.lightsOffS ?? null, lightsOnS: r.lightsOnS ?? null } });
    await done;
    runs.push(posted);
  }
  writeFileSync(outPath, JSON.stringify({ runs, fetches }, (k, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v)));
}

function metricsMode(inPath, outPath) {
  const seqs = JSON.parse(readFileSync(inPath, "utf8"));
  const out = seqs.map((w) => nightMetrics(w.map((v) => v === 1)));
  writeFileSync(outPath, JSON.stringify(out, (k, v) => (typeof v === "number" && !Number.isFinite(v) ? null : v)));
}

const argv = process.argv.slice(2);
if (argv[0] === "--worker") {
  await workerMode(argv[1], argv[2]);
} else if (argv[0] === "--metrics") {
  metricsMode(argv[1], argv[2]);
} else if (argv.length >= 2) {
  reportMode(argv);
} else {
  console.error("usage: node report_cli.mjs <rr_csv> <out_json> [--lights-off S] [--lights-on S] [--models PATH]\n" +
    "       node report_cli.mjs --worker <in_json> <out_json>");
  process.exit(2);
}
