"""Run the browser ECG converter (web/js/ecg.js, under Node) on every EDF recording with a
cached RR series and compare it to the RR series the evaluation used (data/cache/rr).

Where the RR series differ, both are run through the browser report (report_v2.js) to see
whether any number in the report changes.

Usage: python scripts/check_ecg_parity.py [--out results/ecg_parity.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_ = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT_))
from somnibeat.provenance import drop_timing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tests/parity/ecg_cli.mjs"
REPORT_CLI = ROOT / "tests/parity/report_v2_cli.mjs"
RAW = {"ucddb": ("ucddb", ".rec"), "cap": ("capslpdb", ".edf"), "hmc": ("hmc/recordings", ".edf")}
CACHES = [ROOT / "data/cache/v2/rr", ROOT / "data/cache/rr"]  # v2 first; same detector, more records


def report(csv: Path, out: Path) -> dict:
    subprocess.run(["node", str(REPORT_CLI), str(csv), str(out)], check=True, timeout=600)
    r = json.loads(out.read_text())["report"]
    r.pop("fileName", None)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results/ecg_parity.json")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cached_by_name = {}
        for d in CACHES:
            for c in sorted(d.glob("*.csv")):
                cached_by_name.setdefault(c.stem, c)
        for name in sorted(cached_by_name):
            cached = cached_by_name[name]
            ds, rec = cached.stem.split("_", 1)
            if ds not in RAW:
                continue  # WFDB-only datasets (slpdb, apnea-ecg) are not EDF
            folder, ext = RAW[ds]
            raw = ROOT / "data/raw" / folder / f"{rec}{ext}"
            if not raw.exists():
                continue
            subprocess.run(["node", str(CLI), str(raw), str(tmp / "js.json")], check=True, timeout=900)
            js = json.loads((tmp / "js.json").read_text())
            ref = pd.read_csv(cached)
            row = {"dataset": ds, "record": rec, "fs": js.get("fs"), "intervals": int(len(ref)),
                   "seconds": round(js.get("seconds", 0), 2)}
            if "error" in js:
                row.update(error=js["error"]["code"], identical=0)
            else:
                t, rr = np.array(js["time_s"]), np.array(js["rr_ms"])
                same_len = len(t) == len(ref)
                ident = int(((t == ref.time_s.to_numpy()) & (rr == ref.rr_ms.to_numpy())).sum()) if same_len else 0
                row.update(intervals_js=int(len(t)), identical=ident)
                if ident != len(ref):
                    csv = tmp / "js.csv"
                    pd.DataFrame({"time_s": t, "rr_ms": rr}).to_csv(csv, index=False)
                    row["report_identical"] = report(csv, tmp / "a.json") == report(cached, tmp / "b.json")
            rows.append(row)
            print(f"{ds}/{rec}: {row['identical']}/{row['intervals']} identical"
                  + ("" if row["identical"] == row["intervals"] else f", report identical: {row.get('report_identical')}"))
    total = sum(r["intervals"] for r in rows)
    ident = sum(r["identical"] for r in rows)
    diff = [r for r in rows if r["identical"] != r["intervals"]]
    summary = {
        "recordings": len(rows), "datasets": sorted({r["dataset"] for r in rows}),
        "intervals": total, "identical_intervals": ident,
        "recordings_identical": len(rows) - len(diff),
        "recordings_with_differences": [r["dataset"] + "/" + r["record"] for r in diff],
        "reports_identical_where_different": all(r.get("report_identical") for r in diff),
    }
    print(f"slowest conversion (Node, including reading the file): {max(r['seconds'] for r in rows):.1f} s")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(drop_timing({"summary": summary, "records": rows}), indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
