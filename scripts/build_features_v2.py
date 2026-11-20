"""Long-context v2 feature columns (somnibeat/features_v2.py) on the v1 epoch grids.

Inputs
    RR:    data/cache/rr/<dataset>_<record>.csv (v1) if present, else data/cache/v2/rr/... (HMC,
           records new in v2)
    grids: key columns only (dataset, record, epoch, epoch_start_s / minute) of
           data/cache/stage_features.csv.gz and data/cache/apnea_features.csv.gz (v1). Labels
           are never read. Records missing from the v1 caches (HMC, new CAP records) take their
           grid from the key columns of data/cache/v2/stage_features.csv.gz /
           apnea_features.csv.gz when another builder has written them, else (HMC only) from
           somnibeat.datasets.load_hmc(...).epoch_start_s.
Outputs (data/cache/v2, gzip mtime 0)
    stage_features_extra.csv.gz   keys (dataset, record, epoch) + v2 columns, 30 s epochs
    apnea_features_extra.csv.gz   keys (dataset, record, minute) + v2 columns, 60 s minutes
Running with a subset of --datasets replaces only those datasets' rows in existing outputs.

This script computes features only (per record, unsupervised); it fits nothing and reads no labels.
"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import datasets, features_v2, rr as rrm  # noqa: E402

GZIP = {"method": "gzip", "mtime": 0}
V1_CACHE = ROOT / "data" / "cache"
V2_CACHE = ROOT / "data" / "cache" / "v2"
DEFAULT_DATASETS = ["cap", "slpdb", "apnea-ecg", "ucddb"]
STAGE_OUT = V2_CACHE / "stage_features_extra.csv.gz"
APNEA_OUT = V2_CACHE / "apnea_features_extra.csv.gz"


def _keys(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, usecols=cols)  # key columns only: labels are never loaded


def load_grids() -> tuple[dict, dict]:
    """{(dataset, record): (epoch_idx, start_s)} for staging and apnea grids; v1 first."""
    stage, apnea = {}, {}
    for cache in (V1_CACHE, V2_CACHE):
        s = _keys(cache / "stage_features.csv.gz", ["dataset", "record", "epoch", "epoch_start_s"])
        for (d, r), g in s.groupby(["dataset", "record"], sort=False):
            stage.setdefault((d, r), (g["epoch"].to_numpy(), g["epoch_start_s"].to_numpy(float)))
        a = _keys(cache / "apnea_features.csv.gz", ["dataset", "record", "minute"])
        for (d, r), g in a.groupby(["dataset", "record"], sort=False):
            m = g["minute"].to_numpy()
            apnea.setdefault((d, r), (m, m * 60.0))
    return stage, apnea


def rr_path(dataset: str, rid: str) -> Path | None:
    for base in (V1_CACHE / "rr", V2_CACHE / "rr"):
        p = base / f"{dataset}_{rid}.csv"
        if p.exists():
            return p
    return None


def hmc_records() -> list[str]:
    if not hasattr(datasets, "load_hmc"):
        return []
    return sorted(p.stem[len("hmc_"):] for p in (V2_CACHE / "rr").glob("hmc_*.csv"))


def hmc_grid(cfg: dict, rid: str) -> tuple[np.ndarray, np.ndarray]:
    rec = datasets.load_hmc(Path(cfg["paths"]["raw"]) / "hmc", rid)
    st = np.asarray(rec.epoch_start_s, dtype=np.float64)
    return np.arange(len(st)), st


def process(job: tuple) -> dict:
    dataset, rid, path, sgrid, agrid, cfg = job
    try:
        t0 = time.time()
        df = pd.read_csv(path)
        t = df["time_s"].to_numpy(np.float64)
        rr = df["rr_ms"].to_numpy(np.float64)
        valid = rrm.clean_rr(t, rr, cfg["rr"])
        out = {"dataset": dataset, "record": rid}
        if sgrid is None and dataset == "hmc":
            sgrid = hmc_grid(cfg, rid)
        for key, grid, length, idx in (("stage", sgrid, 30.0, "epoch"), ("apnea", agrid, 60.0, "minute")):
            if grid is None or len(grid[1]) == 0:
                continue
            f = features_v2.long_features(t, rr, valid, grid[1], length, cfg["rr"])
            fd = pd.DataFrame(f)
            fd.insert(0, idx, grid[0])
            fd.insert(0, "record", rid)
            fd.insert(0, "dataset", dataset)
            out[key] = fd
        out["seconds"] = time.time() - t0
        return out
    except Exception as e:
        raise RuntimeError(f"{dataset}/{rid}: {e!r}") from e


def write(path: Path, frames: list[pd.DataFrame], done: set[str], key: str) -> pd.DataFrame | None:
    if not frames and not path.exists():
        return None
    new = pd.concat(frames, ignore_index=True) if frames else None
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["dataset"].isin(done)]
        new = old if new is None else pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["dataset", "record", key], kind="stable").reset_index(drop=True)
    new.to_csv(path, index=False, compression=GZIP)
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "v2.toml"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS + ["hmc"],
                    help="any of cap slpdb apnea-ecg ucddb hmc (hmc is skipped if not available)")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    t_start = time.time()
    stage_g, apnea_g = load_grids()
    jobs, skipped = [], []
    for d in a.datasets:
        if d == "hmc":
            recs = hmc_records()
            if not recs:
                print("hmc: skipped (no datasets.load_hmc or no data/cache/v2/rr/hmc_*.csv yet)")
                continue
        else:
            recs = sorted({r for (dd, r) in list(stage_g) + list(apnea_g) if dd == d})
        for r in recs:
            p = rr_path(d, r)
            if p is None:
                skipped.append(f"{d}/{r}")
                continue
            jobs.append((d, r, p, stage_g.get((d, r)), apnea_g.get((d, r)), cfg))
    if skipped:
        print("No RR CSV, skipped:", *skipped)
    done = {j[0] for j in jobs}
    stage, apnea = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(process, jobs):
            ns = len(res["stage"]) if "stage" in res else 0
            na = len(res["apnea"]) if "apnea" in res else 0
            print(f"{res['dataset']:9s} {res['record']:9s} epochs={ns:5d} minutes={na:5d} "
                  f"{res['seconds']:.1f}s", flush=True)
            if ns:
                stage.append(res["stage"])
            if na:
                apnea.append(res["apnea"])
    V2_CACHE.mkdir(parents=True, exist_ok=True)
    s = write(STAGE_OUT, stage, done, "epoch")
    m = write(APNEA_OUT, apnea, done, "minute")
    for name, df in (("stage epochs", s), ("apnea minutes", m)):
        if df is not None:
            counts = df.groupby("dataset").agg(records=("record", "nunique"), rows=("record", "size"))
            print(f"\n{name}:\n{counts.to_string()}")
    print(f"\nruntime {time.time() - t_start:.1f} s, {len(jobs)} records, {a.workers} workers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
