"""Download the raw PhysioNet files used by Somnibeat and verify SHA256 checksums.

Usage:
    python scripts/download.py [--config config/default.toml] [--datasets ucddb slpdb apnea-ecg cap]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files"
VERSIONS = {"ucddb": "1.0.0", "capslpdb": "1.0.0", "slpdb": "1.0.0", "apnea-ecg": "1.0.0"}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "somnibeat-research-downloader"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def file_list(project: str, cfg: dict, root: Path) -> list[str]:
    if project == "ucddb":
        recs = [l[:-4] for l in (root / "RECORDS").read_text().split() if l.endswith(".rec")]
        return [f for r in recs for f in (f"{r}.rec", f"{r}_stage.txt", f"{r}_respevt.txt")]
    if project == "slpdb":
        recs = (root / "RECORDS").read_text().split()
        return ["README.st"] + [f"{r}.{e}" for r in recs for e in ("hea", "dat", "st")]
    if project == "apnea-ecg":
        recs = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] + \
               [f"c{i:02d}" for i in range(1, 11)]
        return ["additional-information.txt"] + [f"{r}.{e}" for r in recs for e in ("hea", "dat", "apn")]
    if project == "capslpdb":
        recs = cfg["datasets"]["cap"]["records"]
        return [f"{r}.{e}" for r in recs for e in ("edf", "txt")]
    raise ValueError(project)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--datasets", nargs="*", default=["ucddb", "slpdb", "apnea-ecg", "capslpdb"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    raw = Path(cfg["paths"]["raw"])
    bad = []
    for project in a.datasets:
        v = VERSIONS[project]
        root = raw / project
        for meta in ("RECORDS", "SHA256SUMS.txt"):
            if not (root / meta).exists():
                fetch(f"{BASE}/{project}/{v}/{meta}", root / meta)
        sums = {}
        for line in (root / "SHA256SUMS.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
        files = file_list(project, cfg, root)

        def one(name: str) -> str | None:
            dest = root / name
            if not (dest.exists() and dest.stat().st_size > 0):
                fetch(f"{BASE}/{project}/{v}/{name}", dest)
            expected = sums.get(name)
            if expected and sha256(dest) != expected:
                return name
            return None

        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(one, files))
        failed = [r for r in results if r]
        bad += [f"{project}/{r}" for r in failed]
        print(f"{project}: {len(files)} files, {len(failed)} checksum failures")
    if bad:
        print("Checksum failures:", *bad, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
