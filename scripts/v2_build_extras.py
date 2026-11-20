"""Rebuild the v2 long-context feature caches in the order that produced the frozen ones.

The v2 caches were first built by hand: build_features_v2.py wrote stage_features_extra.csv.gz in
place, and two later copy runs were renamed (the original file restored after each). This script
runs the same three build_features_v2.py calls with explicit output paths instead of renames:

  1 base    --datasets cap slpdb apnea-ecg ucddb --v1-grids-only   (grids from the v1 caches only,
            so CAP = the 49 v1 records, as before the 106-record v2 cache existed)
            -> stage_features_extra.csv.gz         cap49 + slpdb + ucddb   (v2 selection input)
            -> apnea_features_extra.csv.gz         apnea-ecg + ucddb       (apnea selection, final)
  2 cap106  --datasets cap, starting from the step-1 stage table (its cap rows replaced)
            -> stage_features_extra_cap106.csv.gz  cap106 + slpdb + ucddb  (final training extras)
  3 test    --datasets hmc, starting from the step-1 stage table (hmc rows added)
            -> stage_features_extra_test.csv.gz    cap49 + hmc + slpdb + ucddb (final test extras)

Steps 2 and 3 store 'stage_features_extra.csv' in the gzip header (the name the renamed files were
written under), so the files keep the SHA-256 recorded in results/v2/selection_cap106.json and
results/v2/test_results.json. Nothing here fits a model or reads a label.

Prerequisites: v1 caches (make features), data/cache/v2/stage_features.csv.gz (CAP, 106 records)
and data/cache/v2/hmc_stage_features.csv.gz + data/cache/v2/rr/hmc_*.csv (make v2-features runs
build_features.py for both first).

Usage:
  python scripts/v2_build_extras.py [--out-dir data/cache/v2] [--steps base cap106 test]
                                    [--workers 8] [--base-datasets ...] [--records ...]
--base-datasets / --records narrow the runs for quick checks (outputs then are partial tables).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_features_v2.py"
STEPS = ("base", "cap106", "test")
BASE_DATASETS = ["cap", "slpdb", "apnea-ecg", "ucddb"]
RENAMED_FROM = "stage_features_extra.csv.gz"


def commands(out_dir: Path, steps=STEPS, workers: int = 8, base_datasets=None, records=None,
             config: str | None = None) -> list[list[str]]:
    stage_base = out_dir / "stage_features_extra.csv.gz"
    common = ["--workers", str(workers)] + (["--config", config] if config else []) + \
             (["--records", *records] if records else [])
    cmds = []
    if "base" in steps:
        cmds.append(["--datasets", *(base_datasets or BASE_DATASETS), "--v1-grids-only",
                     "--stage-base", "none", "--apnea-base", "none",
                     "--stage-out", str(stage_base), "--apnea-out", str(out_dir / "apnea_features_extra.csv.gz")])
    if "cap106" in steps:
        cmds.append(["--datasets", "cap", "--stage-base", str(stage_base), "--apnea-out", "none",
                     "--stage-out", str(out_dir / "stage_features_extra_cap106.csv.gz"),
                     "--stage-gzip-name", RENAMED_FROM])
    if "test" in steps:
        cmds.append(["--datasets", "hmc", "--stage-base", str(stage_base), "--apnea-out", "none",
                     "--stage-out", str(out_dir / "stage_features_extra_test.csv.gz"),
                     "--stage-gzip-name", RENAMED_FROM])
    return [[sys.executable, str(BUILD), *c, *common] for c in cmds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "cache" / "v2")
    ap.add_argument("--steps", nargs="*", default=list(STEPS), choices=STEPS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default=None, help="passed to build_features_v2.py (default: its own)")
    ap.add_argument("--base-datasets", nargs="*", default=None, help="step 1 datasets (default: all four)")
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    if ("cap106" in a.steps or "test" in a.steps) and "base" not in a.steps \
            and not (a.out_dir / "stage_features_extra.csv.gz").exists():
        raise SystemExit(f"step 'base' output missing in {a.out_dir}")
    for cmd in commands(a.out_dir, a.steps, a.workers, a.base_datasets, a.records, a.config):
        print("+", " ".join(cmd[1:]), flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"step done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
