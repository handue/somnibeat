"""Validate the R-peak detector against PhysioNet beat annotations.

References:
- MIT-BIH slpdb `.ecg` beat annotations (all 18 records).
- Apnea-ECG `.qrs` annotations (35 learning-set records). These are
  machine-generated QRS annotations distributed with the database, so
  agreement with them is a consistency check, not ground truth.

A detection matches a reference beat if within +-150 ms (one-to-one).
Detections inside reference gaps (> 2x local median RR) are not counted.
Timing error is reported signed (the reference fiducial point may differ from
the R-peak maximum) and offset-corrected.
Writes results/rpeak_validation.json.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from somnibeat import datasets  # noqa: E402

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
TOL_S = 0.150


def match(ref: np.ndarray, det: np.ndarray, tol: float = TOL_S) -> tuple[int, list[float]]:
    i = j = tp = 0
    errs = []
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            tp += 1
            errs.append(d)
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return tp, errs


def main() -> int:
    cfg = tomllib.loads(Path("config/default.toml").read_text())
    raw = Path(cfg["paths"]["raw"])
    cache = Path(cfg["paths"]["cache"]) / "beats"
    out = {"tolerance_ms": TOL_S * 1000, "datasets": {}}
    for ds, ext, recs in (("slpdb", "ecg", (raw / "slpdb" / "RECORDS").read_text().split()),
                          ("apnea-ecg", "qrs", datasets.APNEA_ECG_LEARNING)):
        rows = []
        for rid in recs:
            det = np.load(cache / f"{ds}_{rid}.npy")
            ann = wfdb.rdann(str(raw / ds / rid), ext)
            fs = wfdb.rdheader(str(raw / ds / rid)).fs
            keep = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
            ref = ann.sample[keep] / fs
            # restrict detections to the annotated span, and drop reference gaps
            # (> 2x local median RR) where the reference simply has no annotations
            det = det[(det >= ref[0] - TOL_S) & (det <= ref[-1] + TOL_S)]
            rr = np.diff(ref)
            med = np.array([np.median(rr[max(0, i - 10):i + 11]) for i in range(len(rr))])
            gaps = [(ref[i] + TOL_S, ref[i + 1] - TOL_S) for i in np.where(rr > 2 * med)[0]]
            for a, b in gaps:
                det = det[(det < a) | (det > b)]
            tp, errs = match(ref, det)
            e = np.array(errs) * 1000
            rows.append({"record": rid, "ref_beats": int(len(ref)), "detected": int(len(det)),
                         "reference_gaps_excluded": len(gaps),
                         "sensitivity": tp / len(ref), "ppv": tp / len(det) if len(det) else float("nan"),
                         "median_signed_timing_error_ms": float(np.median(e)),
                         "median_abs_timing_error_offset_corrected_ms": float(np.median(np.abs(e - np.median(e))))})
        tot_ref = sum(r["ref_beats"] for r in rows)
        tot_det = sum(r["detected"] for r in rows)
        tot_tp = sum(round(r["sensitivity"] * r["ref_beats"]) for r in rows)
        out["datasets"][ds] = {
            "reference": "beat annotations (.ecg)" if ds == "slpdb" else "machine QRS annotations (.qrs)",
            "records": len(rows), "pooled_sensitivity": tot_tp / tot_ref, "pooled_ppv": tot_tp / tot_det,
            "min_record_sensitivity": min(r["sensitivity"] for r in rows),
            "min_record_ppv": min(r["ppv"] for r in rows), "per_record": rows}
        print(ds, {k: v for k, v in out["datasets"][ds].items() if k != "per_record"})
    Path(cfg["paths"]["results"], "rpeak_validation.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
