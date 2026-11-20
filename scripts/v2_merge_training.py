"""Build the v2 staging training table: 106 CAP records (v2 cache) + MIT-BIH slpdb (v1 cache).

Inputs
    data/cache/v2/stage_features.csv.gz   build_features.py --config config/v2.toml --datasets cap
    data/cache/stage_features.csv.gz      build_features.py (v1); only its slpdb rows are used
Output
    data/cache/v2/stage_features_cap106_slpdb.csv.gz
        every row of the v2 CAP table, then the v1 slpdb rows, in file order, same header and
        column order (both inputs come from the same feature code), gzip mtime 0.

The merge is done on the CSV text, so no value is re-parsed or re-formatted: the output is byte
identical to the file the v2 selection and the frozen evaluation used (its SHA-256 is stored in
results/v2/selection_cap106.json -> features_sha256 and results/v2/test_results.json ->
inputs.train_features_sha256). The gzip header carries the output file name (as pandas writes it),
so write to a file with the same base name when comparing bytes.

Usage: python scripts/v2_merge_training.py [--cap ...] [--slpdb ...] [--out ...]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

from pandas.io.common import get_handle

GZIP = {"method": "gzip", "mtime": 0}
ROOT = Path(__file__).resolve().parents[1]
CAP_V2 = ROOT / "data/cache/v2/stage_features.csv.gz"
V1 = ROOT / "data/cache/stage_features.csv.gz"
OUT = ROOT / "data/cache/v2/stage_features_cap106_slpdb.csv.gz"


def merge(cap: Path, v1: Path, out: Path) -> dict:
    with gzip.open(cap, "rb") as f:
        cap_lines = f.read().splitlines(keepends=True)
    with gzip.open(v1, "rb") as f:
        v1_lines = f.read().splitlines(keepends=True)
    if cap_lines[0] != v1_lines[0]:
        raise SystemExit(f"column order differs between {cap} and {v1}")
    if not cap_lines[0].startswith(b"dataset,"):
        raise SystemExit("first column must be 'dataset'")
    cap_rows = cap_lines[1:]
    if any(not l.startswith(b"cap,") for l in cap_rows):
        raise SystemExit(f"{cap} must hold CAP rows only")
    slpdb_rows = [l for l in v1_lines[1:] if l.startswith(b"slpdb,")]
    if not slpdb_rows:
        raise SystemExit(f"no slpdb rows in {v1}")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written through pandas' own file handle, exactly as to_csv(compression={"method": "gzip",
    # "mtime": 0}) writes: zlib's block boundaries depend on how the text reaches the compressor
    # (pandas' text wrapper and GzipFile buffering), and the gzip header stores the file name
    # (the output name without .gz). A plain gzip.GzipFile write gives different (equally valid) bytes.
    text = b"".join([cap_lines[0], *cap_rows, *slpdb_rows]).decode("utf-8")
    with get_handle(str(out), "w", compression=GZIP) as h:
        h.handle.write(text)
    return {"cap_rows": len(cap_rows), "slpdb_rows": len(slpdb_rows),
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=Path, default=CAP_V2)
    ap.add_argument("--slpdb", type=Path, default=V1, help="v1 stage table; only its slpdb rows are used")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    r = merge(a.cap, a.slpdb, a.out)
    print(f"wrote {a.out}: {r['cap_rows']} cap + {r['slpdb_rows']} slpdb rows, sha256 {r['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
