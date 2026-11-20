"""ECG -> R-peaks -> cleaned RR -> epoch (30 s) and minute (60 s) feature tables.

Outputs (data/cache):
    beats/<dataset>_<record>.npy     R-peak times (s)
    rr/<dataset>_<record>.csv        time_s, rr_ms (what the demo would receive)
    stage_features.csv.gz            one row per 30 s epoch (UCDDB, CAP, slpdb)
    apnea_features.csv.gz            one row per 60 s minute (Apnea-ECG, UCDDB)
    record_summary.json              per-record beat counts and RR validity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from somnibeat import datasets, features, rpeak, rr as rrm  # noqa: E402


# Fixed gzip header time so the cache (and the SHA-256 stored in results/*.json) is reproducible.
GZIP = {"method": "gzip", "mtime": 0}

DETECTOR_SHA = hashlib.sha256(Path(rpeak.__file__).read_bytes()).hexdigest()

REQUIRED = {"hmc": ("{r}.edf", "{r}_sleepscoring.txt"),  # v2 external test set only
            "ucddb": ("{r}.rec", "{r}_stage.txt", "{r}_respevt.txt"),
            "cap": ("{r}.edf", "{r}.txt"),
            "slpdb": ("{r}.hea", "{r}.dat", "{r}.st"),
            "apnea-ecg": ("{r}.hea", "{r}.dat", "{r}.apn")}
RAW_DIR = {"ucddb": "ucddb", "cap": "capslpdb", "slpdb": "slpdb", "apnea-ecg": "apnea-ecg", "hmc": "hmc/recordings"}


def missing_files(raw: Path, recs: dict[str, list[str]], names: list[str]) -> list[str]:
    out = []
    for d in names:
        for r in recs[d]:
            for pat in REQUIRED[d]:
                f = raw / RAW_DIR[d] / pat.format(r=r)
                if not (f.exists() and f.stat().st_size > 0):
                    out.append(str(f))
    return out


def process(job: tuple[str, str, dict]) -> dict:
    dataset, rid, cfg = job
    try:
        return _process(dataset, rid, cfg)
    except Exception as e:  # report which record failed instead of losing the whole run
        raise RuntimeError(f"{dataset}/{rid}: {e!r}") from e


def _process(dataset: str, rid: str, cfg: dict) -> dict:
    cache = Path(cfg["paths"]["cache"])
    rec = datasets.load(cfg["paths"]["raw"], dataset, rid, cfg)
    bpath = cache / "beats" / f"{dataset}_{rid}.npy"
    spath = bpath.with_suffix(".sha256")
    if bpath.exists() and spath.exists() and spath.read_text().strip() == DETECTOR_SHA:
        beats = np.load(bpath)
    else:  # missing, or detected with a different version of rpeak.py
        beats = rpeak.detect_r_peaks(rec.ecg, rec.fs)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(bpath, beats)
        spath.write_text(DETECTOR_SHA)
    t, rr = rrm.rr_from_beats(beats)
    valid = rrm.clean_rr(t, rr, cfg["rr"])
    rp = cache / "rr" / f"{dataset}_{rid}.csv"
    rp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": np.round(t, 4), "rr_ms": np.round(rr, 2)}).to_csv(rp, index=False)
    # Features are computed from the rounded CSV values, exactly what the browser sees.
    t = np.round(t, 4)
    rr = np.round(rr, 2)
    valid = rrm.clean_rr(t, rr, cfg["rr"])
    out = {"dataset": dataset, "record": rid, "subject": rec.subject_id,
           "n_beats": int(len(beats)), "valid_fraction": float(valid.mean()),
           "duration_h": float(len(rec.ecg) / rec.fs / 3600), "fs": rec.fs}
    if rec.stages:
        f = features.epoch_features(t, rr, valid, rec.epoch_start_s, 30.0, cfg["rr"], cfg["features"])
        df = pd.DataFrame(f)
        df.insert(0, "stage", [s if s is not None else "" for s in rec.stages])
        df.insert(0, "epoch_start_s", rec.epoch_start_s)
        df.insert(0, "epoch", np.arange(len(df)))
        for c, v in (("subject", rec.subject_id), ("record", rid), ("dataset", dataset)):
            df.insert(0, c, v)
        out["stage_df"] = df
    if rec.apnea_minutes is not None:
        starts = np.arange(len(rec.apnea_minutes)) * 60.0
        f = features.epoch_features(t, rr, valid, starts, 60.0, cfg["rr"], cfg["features"])
        df = pd.DataFrame(f)
        df.insert(0, "apnea", rec.apnea_minutes)
        df.insert(0, "minute", np.arange(len(df)))
        for c, v in (("subject", rec.subject_id), ("record", rid), ("dataset", dataset)):
            df.insert(0, c, v)
        out["apnea_df"] = df
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=["ucddb", "cap", "slpdb", "apnea-ecg"])
    ap.add_argument("--prefix", default="", help="output file prefix, e.g. 'hmc_' (v2 test-set build)")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    recs = datasets.list_records(cfg["paths"]["raw"], cfg, include_hmc="hmc" in a.datasets)
    missing = missing_files(Path(cfg["paths"]["raw"]), recs, a.datasets)
    if missing:
        print("Missing raw files (run `make download`, or remove the record from the config):",
              *missing, sep="\n  ")
        return 1
    jobs = [(d, r, cfg) for d in a.datasets for r in recs[d]]
    stage, apnea, summary = [], [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(process, jobs):
            print(f"{res['dataset']:9s} {res['record']:9s} beats={res['n_beats']:6d} "
                  f"valid={res['valid_fraction']:.3f}", flush=True)
            if "stage_df" in res:
                stage.append(res.pop("stage_df"))
            if "apnea_df" in res:
                apnea.append(res.pop("apnea_df"))
            summary.append(res)
    cache = Path(cfg["paths"]["cache"])
    if stage:
        pd.concat(stage).to_csv(cache / f"{a.prefix}stage_features.csv.gz", index=False, compression=GZIP)
    if apnea:
        pd.concat(apnea).to_csv(cache / f"{a.prefix}apnea_features.csv.gz", index=False, compression=GZIP)
    (cache / f"{a.prefix}record_summary.json").write_text(json.dumps(
        {"detector_sha256": DETECTOR_SHA, "datasets": a.datasets, "records": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
