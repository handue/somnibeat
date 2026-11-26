// Parity CLI for web/js/ecg.js (run by tests/test_parity_ecg.py and scripts/check_ecg_parity.py).
//   node tests/parity/ecg_cli.mjs <recording.edf|.rec|export.csv> <out.json>
//     Runs the browser converter and writes {time_s, rr_ms, beats, fs, start_clock, source, seconds}
//     or {error: {code, message, details}}.
//   node tests/parity/ecg_cli.mjs --sos <fs> <out.json>
//     Writes the band-pass filter sections {sos}.
import { readFileSync, writeFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { edfToRR, rrExportColumn, rrExportToRR, bandpassSos } from "../../web/js/ecg.js";

const [, , input, a2, a3] = process.argv;
if (input === "--sos") {
  writeFileSync(a3, JSON.stringify({ sos: bandpassSos(Number(a2)) }));
  process.exit(0);
}
const out = a2;
const t0 = performance.now();
try {
  let r;
  if (/\.(csv|txt)$/i.test(input)) {
    const text = readFileSync(input, "utf8");
    const col = rrExportColumn(text);
    if (!col) throw Object.assign(new Error("no RR column"), { code: "NO_RR_COLUMN" });
    r = rrExportToRR(text, col);
  } else {
    r = await edfToRR(new Uint8Array(readFileSync(input)));
  }
  writeFileSync(out, JSON.stringify({
    time_s: Array.from(r.time_s), rr_ms: Array.from(r.rr_ms), beats: r.beats, fs: r.fs,
    start_clock: r.startClock, source: r.source, seconds: (performance.now() - t0) / 1000,
  }));
} catch (err) {
  writeFileSync(out, JSON.stringify({ error: { code: err.code || "INTERNAL", message: err.message, details: err.details || {} } }));
}
