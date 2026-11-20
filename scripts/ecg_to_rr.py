"""Convert an overnight ECG recording into the RR file Somnibeat reads (time_s, rr_ms).

Uses the same R-peak detector and RR rounding as the evaluation pipeline
(somnibeat/rpeak.py, scripts/build_features.py), so a recording converted here
gives exactly the RR series the reported accuracy was measured on.

Supported inputs:
  .edf / .rec   European Data Format (most sleep-lab and Holter exports); the ECG
                channel is the first label matching --channel (default ECG|EKG)
  .hea          WFDB record (PhysioNet); the first channel matching --channel
  .csv / .txt   raw ECG samples: one column of values, or a column named by --column;
                the sampling rate must be given with --fs
  .csv / .txt   an RR-interval export (e.g. from an ECG chest strap app): a header
                row with one column whose name contains "RR" (or --rr-column NAME);
                values in ms (or in s, detected from the median); time_s is the
                running sum of the intervals

Usage:
  python scripts/ecg_to_rr.py night.edf                  # writes night_rr.csv
  python scripts/ecg_to_rr.py night.edf -o my_rr.csv
  python scripts/ecg_to_rr.py ecg.csv --fs 250 --column ecg
  python scripts/ecg_to_rr.py strap_export.csv           # RR export -> time_s, rr_ms

The output has a header row and one row per interval: time_s is the time of the
beat ending the interval (seconds since the recording started), rr_ms the
interval in milliseconds. Nothing is uploaded anywhere; the file stays on disk.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import rpeak  # noqa: E402
from somnibeat import rr as rrm  # noqa: E402
from somnibeat.edf import read_channel  # noqa: E402

MIN_FS = 100.0  # the detector's band-pass (5-25 Hz) and +-75 ms peak refinement need at least this


def read_ecg(path: Path, channel: str, fs: float | None, column: str | None) -> tuple[np.ndarray, float, str, str | None]:
    """Return (ECG samples, sampling rate in Hz, channel description, recording start clock HH:MM or None)."""
    suffix = path.suffix.lower()
    if suffix in (".edf", ".rec"):
        try:
            x, f, hdr, label = read_channel(path, channel)
        except KeyError as e:
            raise SystemExit(f"{e.args[0]}; name the ECG channel with --channel") from None
        return x, f, f"EDF channel {label!r}", hdr.start.strftime("%H:%M")
    if suffix == ".hea":
        import wfdb
        rec = wfdb.rdrecord(str(path.with_suffix("")))
        idx = [i for i, n in enumerate(rec.sig_name) if re.search(channel, n, re.I)]
        if not idx:
            raise SystemExit(f"{path}: no channel matching {channel!r} in {rec.sig_name}")
        start = rec.base_time.strftime("%H:%M") if rec.base_time else None
        return rec.p_signal[:, idx[0]].astype(np.float64), float(rec.fs), f"WFDB channel {rec.sig_name[idx[0]]!r}", start
    if suffix in (".csv", ".txt"):
        if fs is None:
            raise SystemExit("raw ECG samples in a CSV need the sampling rate: add --fs <Hz>")
        df = pd.read_csv(path, header=0 if column else None)
        if column:
            if column not in df.columns:
                raise SystemExit(f"{path}: no column {column!r} (columns: {list(df.columns)})")
            x = df[column]
        elif df.shape[1] == 1:
            x = df.iloc[:, 0]
        else:
            raise SystemExit(f"{path}: {df.shape[1]} columns; name the ECG column with --column")
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(x[0]):  # a header row read as data
            x = x[1:]
        return x, float(fs), f"CSV column {column or 1}", None
    raise SystemExit(f"{path}: unsupported file type {suffix!r} (use .edf, .rec, .hea, .csv or .txt)")


def rr_column(path: Path, name: str | None) -> str | None:
    """Name of the RR column of a CSV export, or None when the file is not an RR export."""
    if path.suffix.lower() not in (".csv", ".txt"):
        return None
    head = pd.read_csv(path, nrows=0, sep=None, engine="python")
    if name:
        if name not in head.columns:
            raise SystemExit(f"{path}: no column {name!r} (columns: {list(head.columns)})")
        return name
    hits = [c for c in head.columns if re.search(r"(^|[^a-z])rr([^a-z]|$)", str(c), re.I)]
    return hits[0] if len(hits) == 1 else None


def convert_rr_export(path: Path, out: Path, col: str) -> dict:
    df = pd.read_csv(path, sep=None, engine="python")
    rr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    rr = rr[np.isfinite(rr)]
    if len(rr) < 2:
        raise SystemExit(f"{path}: column {col!r} has fewer than two numeric intervals")
    unit = "s" if np.median(rr) < 5 else "ms"
    if unit == "s":
        rr = rr * 1000.0
    t = np.cumsum(rr) / 1000.0
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": np.round(t, 4), "rr_ms": np.round(rr, 2)}).to_csv(out, index=False)
    cfg = tomllib.loads((ROOT / "config/default.toml").read_text())
    valid = rrm.clean_rr(np.round(t, 4), np.round(rr, 2), cfg["rr"])
    return {"input": str(path), "source": f"RR column {col!r} ({unit})", "fs": None, "duration_h": t[-1] / 3600,
            "beats": int(len(rr)), "usable_fraction": float(np.mean(valid)), "start_clock": None, "output": str(out)}


def convert(path: Path, out: Path, channel: str = r"ECG|EKG", fs: float | None = None,
            column: str | None = None, rr_col: str | None = None) -> dict:
    if fs is None and column is None:
        col = rr_column(path, rr_col)
        if col is not None:
            return convert_rr_export(path, out, col)
    x, f, what, start = read_ecg(path, channel, fs, column)
    if f < MIN_FS:
        raise SystemExit(f"{path}: sampling rate {f:g} Hz is below {MIN_FS:g} Hz; R peaks cannot be located reliably")
    beats = rpeak.detect_r_peaks(x, f)
    if len(beats) < 2:
        raise SystemExit(f"{path}: fewer than two heartbeats found in {what}; is this the ECG channel?")
    t, rr = rrm.rr_from_beats(beats)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same rounding and writer as the evaluation pipeline (scripts/build_features.py).
    pd.DataFrame({"time_s": np.round(t, 4), "rr_ms": np.round(rr, 2)}).to_csv(out, index=False)
    cfg = tomllib.loads((ROOT / "config/default.toml").read_text())
    valid = rrm.clean_rr(np.round(t, 4), np.round(rr, 2), cfg["rr"])
    return {"input": str(path), "source": what, "fs": f, "duration_h": len(x) / f / 3600,
            "beats": int(len(beats)), "usable_fraction": float(np.mean(valid)), "start_clock": start,
            "output": str(out)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path, help="ECG recording (.edf, .rec, .hea, .csv, .txt)")
    ap.add_argument("-o", "--out", type=Path, help="output CSV (default: <input name>_rr.csv next to the input)")
    ap.add_argument("--channel", default=r"ECG|EKG", help="regular expression for the ECG channel label (EDF, WFDB)")
    ap.add_argument("--fs", type=float, help="sampling rate in Hz (required for CSV/TXT input)")
    ap.add_argument("--column", help="name of the ECG column in a CSV with a header row")
    ap.add_argument("--rr-column", help="name of the RR column of an RR-interval export (default: the one column named like RR)")
    a = ap.parse_args(argv)
    if not a.input.exists():
        raise SystemExit(f"{a.input}: file not found")
    out = a.out or a.input.with_name(a.input.stem + "_rr.csv")
    r = convert(a.input, out, a.channel, a.fs, a.column, a.rr_column)
    print(f"read {r['source']}" + (f" at {r['fs']:g} Hz" if r["fs"] else "") + f", {r['duration_h']:.1f} h")
    print(f"{r['beats']:,} heartbeats; {100 * r['usable_fraction']:.0f}% of intervals pass the RR quality rules")
    if r["start_clock"]:
        print(f"recording started at {r['start_clock']} (enter it as 'Recording started at' to see clock times)")
    if r["usable_fraction"] < 0.8:
        print("note: many intervals fail the quality rules (noise, electrode contact); the report marks those stretches")
    print(f"wrote {r['output']} -> open https://handue.github.io/somnibeat/ and choose 'Upload your file'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
