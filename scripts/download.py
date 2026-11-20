"""Download the raw PhysioNet files used by Somnibeat and verify SHA256 checksums.

Usage:
    python scripts/download.py [--config config/default.toml] [--datasets ucddb slpdb apnea-ecg capslpdb hmc]
    python scripts/download.py --datasets hmc --list           # dry listing: print the plan, fetch nothing
    python scripts/download.py --datasets hmc --verify-only    # verify files on disk, fetch nothing

Datasets (PhysioNet project, version, local directory under paths.raw):
    ucddb      ucddb 1.0.0                 data/raw/ucddb
    slpdb      slpdb 1.0.0                 data/raw/slpdb
    apnea-ecg  apnea-ecg 1.0.0             data/raw/apnea-ecg
    capslpdb   capslpdb 1.0.0              data/raw/capslpdb   (records from [datasets.cap] of the config)
    hmc        hmc-sleep-staging 1.1       data/raw/hmc        (v2 external test set: every record in
               RECORDS as recordings/SN###.edf + recordings/SN###_sleepscoring.txt, plus RECORDS,
               SHA256SUMS.txt, LICENSE.txt, Subjects_info_aggregated.txt)

Every file listed in the project's SHA256SUMS.txt is verified (paths there are relative to the
project root, e.g. 'recordings/SN001.edf' for HMC); files without a listed checksum are only
checked for existence.
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
VERSIONS = {"ucddb": "1.0.0", "capslpdb": "1.0.0", "slpdb": "1.0.0", "apnea-ecg": "1.0.0", "hmc": "1.1"}
# PhysioNet project slug when it differs from the dataset name (= local directory name).
SLUGS = {"hmc": "hmc-sleep-staging"}
META_FILES = ("RECORDS", "SHA256SUMS.txt")


def project_url(project: str) -> str:
    return f"{BASE}/{SLUGS.get(project, project)}/{VERSIONS[project]}"


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
        return ["README.st"] + [f"{r}.{e}" for r in recs for e in ("hea", "dat", "st", "ecg")]  # .ecg: beat annotations for detector validation
    if project == "apnea-ecg":
        recs = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] + \
               [f"c{i:02d}" for i in range(1, 11)]
        return ["additional-information.txt"] + [f"{r}.{e}" for r in recs for e in ("hea", "dat", "apn", "qrs")]  # .qrs: QRS annotations for detector validation
    if project == "capslpdb":
        recs = cfg["datasets"]["cap"]["records"]
        return [f"{r}.{e}" for r in recs for e in ("edf", "txt")]
    if project == "hmc":
        # RECORDS lists 'recordings/SN001.edf', ...; each record also has a scoring file.
        recs = [l[:-4] for l in (root / "RECORDS").read_text().split() if l.endswith(".edf")]
        return ["RECORDS", "SHA256SUMS.txt", "LICENSE.txt", "Subjects_info_aggregated.txt"] + \
               [f for r in recs for f in (f"{r}.edf", f"{r}_sleepscoring.txt")]
    raise ValueError(project)


def read_sums(root: Path) -> dict[str, str]:
    """{relative path as written in SHA256SUMS.txt: sha256}; empty if the file is missing."""
    sums = {}
    path = root / "SHA256SUMS.txt"
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                sums[parts[1]] = parts[0]
    return sums


def plan(project: str, cfg: dict, raw: Path) -> list[dict]:
    """Dry listing: one entry per file (name, url, dest, expected sha256 or None, present).

    Needs RECORDS on disk for ucddb, slpdb and hmc; without it only the metadata files are listed.
    """
    root = raw / project
    try:
        files = file_list(project, cfg, root)
    except FileNotFoundError:
        files = list(META_FILES)
    for meta in reversed(META_FILES):
        if meta not in files:
            files.insert(0, meta)
    sums = read_sums(root)
    return [{"name": f, "url": f"{project_url(project)}/{f}", "dest": str(root / f),
             "sha256": sums.get(f), "present": (root / f).exists() and (root / f).stat().st_size > 0}
            for f in files]


def verify(project: str, cfg: dict, raw: Path, workers: int = 4) -> dict:
    """Check files already on disk against SHA256SUMS.txt; downloads nothing."""
    entries = plan(project, cfg, raw)

    def one(e: dict) -> str:
        if not e["present"]:
            return "missing"
        if e["sha256"] is None:
            return "no_checksum"
        return "ok" if sha256(Path(e["dest"])) == e["sha256"] else "mismatch"

    with ThreadPoolExecutor(workers) as ex:
        status = list(ex.map(one, entries))
    out = {"files": len(entries), "ok": [], "no_checksum": [], "missing": [], "mismatch": []}
    for e, st in zip(entries, status):
        out[st].append(e["name"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("--datasets", nargs="*", default=["ucddb", "slpdb", "apnea-ecg", "capslpdb"],
                    choices=sorted(VERSIONS))
    ap.add_argument("--workers", type=int, default=4)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="print the download plan and exit (fetches nothing)")
    mode.add_argument("--verify-only", action="store_true",
                      help="verify files already on disk against SHA256SUMS.txt (fetches nothing)")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    raw = Path(cfg["paths"]["raw"])
    if a.list:
        for project in a.datasets:
            entries = plan(project, cfg, raw)
            for e in entries:
                print(f"{'present' if e['present'] else 'MISSING':7s} {e['sha256'] or '-':64s} {e['url']}")
            print(f"{project}: {len(entries)} files, {sum(not e['present'] for e in entries)} to fetch, "
                  f"{sum(e['sha256'] is not None for e in entries)} with a listed checksum")
        return 0
    if a.verify_only:
        failed = False
        for project in a.datasets:
            r = verify(project, cfg, raw, a.workers)
            print(f"{project}: {r['files']} files, {len(r['ok'])} checksum ok, {len(r['no_checksum'])} without "
                  f"checksum, {len(r['missing'])} missing, {len(r['mismatch'])} checksum failures")
            for k in ("missing", "mismatch"):
                for name in r[k]:
                    print(f"  {k}: {project}/{name}")
            failed |= bool(r["missing"] or r["mismatch"])
        return 1 if failed else 0
    bad = []
    for project in a.datasets:
        base = project_url(project)
        root = raw / project
        for meta in META_FILES:
            if not (root / meta).exists():
                fetch(f"{base}/{meta}", root / meta)
        sums = read_sums(root)
        files = file_list(project, cfg, root)

        def one(name: str) -> str | None:
            dest = root / name
            if not (dest.exists() and dest.stat().st_size > 0):
                fetch(f"{base}/{name}", dest)
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
