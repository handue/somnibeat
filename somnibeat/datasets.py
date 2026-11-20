"""Dataset loaders. Each returns a Record with ECG and reference labels.

Stage labels use a common vocabulary: W, R, S1, S2, S3, S4. Any epoch whose
source label is outside that vocabulary (artifact, indeterminate, movement
time, unscored, UCDDB code 8) is stored as None and dropped downstream.
Epoch k covers [epoch_start_s[k], epoch_start_s[k] + 30) seconds from the
start of the ECG signal.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wfdb

from .edf import read_channel

EPOCH_S = 30.0

UCDDB_CODES = {0: "W", 1: "R", 2: "S1", 3: "S2", 4: "S3", 5: "S4"}  # 6,7,8 -> None
CAP_EVENTS = {"SLEEP-S0": "W", "SLEEP-S1": "S1", "SLEEP-S2": "S2", "SLEEP-S3": "S3",
              "SLEEP-S4": "S4", "SLEEP-REM": "R"}  # SLEEP-UNSCORED, SLEEP-MT -> None
SLPDB_STAGES = {"W": "W", "1": "S1", "2": "S2", "3": "S3", "4": "S4", "R": "R"}  # MT, M -> None


@dataclass
class Record:
    dataset: str
    record_id: str
    subject_id: str
    ecg: np.ndarray
    fs: float
    epoch_start_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stages: list = field(default_factory=list)
    apnea_minutes: np.ndarray | None = None  # 1 = apnea/hypopnea minute, 0 = normal, -1 = unlabeled
    meta: dict = field(default_factory=dict)


def _tod_offset(clock: str, start: dt.datetime) -> float:
    h, m, s = (int(x) for x in re.split(r"[:.]", clock)[:3])
    off = (start.replace(hour=h, minute=m, second=s, microsecond=0) - start).total_seconds()
    if off < -6 * 3600:
        off += 86400
    return off


# ------------------------------------------------------------------ UCDDB
RESP_LINE = re.compile(r"^\s*(\d\d:\d\d:\d\d)\s+(\S+)\s+(?:(PB|CS)\s+)?(\d+)\b")


def ucddb_resp_events(path: Path, start: dt.datetime) -> list[dict]:
    out = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = RESP_LINE.match(line)
        if m:
            out.append({"onset_s": _tod_offset(m.group(1), start), "type": m.group(2),
                        "duration_s": float(m.group(4))})
    return out


def load_ucddb(root: Path, rid: str, apnea_cfg: dict | None = None) -> Record:
    root = Path(root)
    x, fs, hdr, label = read_channel(root / f"{rid}.rec", r"^ECG$")
    codes = [int(c) for c in (root / f"{rid}_stage.txt").read_text().split()]
    stages = [UCDDB_CODES.get(c) for c in codes]
    rec = Record("ucddb", rid, rid, x, fs, np.arange(len(codes)) * EPOCH_S, stages,
                 meta={"ecg_label": label, "raw_codes": codes})
    events = ucddb_resp_events(root / f"{rid}_respevt.txt", hdr.start)
    rec.meta["resp_events"] = events
    prefixes = tuple((apnea_cfg or {}).get("ucddb_event_prefixes", ["APNEA", "HYP"]))
    exclude = set((apnea_cfg or {}).get("ucddb_exclude_types", ["POSSIBLE"]))
    n_min = int(len(x) / fs // 60)
    lab = np.zeros(n_min, dtype=int)
    for e in events:
        if e["type"] in exclude or not e["type"].startswith(prefixes):
            continue
        start = max(e["onset_s"], 0.0)
        end = min(e["onset_s"] + e["duration_s"], n_min * 60.0)
        if end <= start:
            continue
        a = int(start // 60)
        b = int(np.ceil(end / 60.0)) - 1  # an event ending exactly on a boundary does not mark the next minute
        lab[a:b + 1] = 1
    rec.apnea_minutes = lab
    return rec


# ------------------------------------------------------------------ CAP
def parse_cap_txt(path: Path, start: dt.datetime) -> list[tuple[float, str | None]]:
    lines = Path(path).read_text(errors="replace").splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if l.startswith("Sleep Stage"))
    cols = lines[hdr_i].split("\t")
    ti = next(i for i, c in enumerate(cols) if c.startswith("Time"))
    ei = cols.index("Event")
    out = []
    for l in lines[hdr_i + 1:]:
        p = l.split("\t")
        if len(p) > max(ti, ei) and p[ei].startswith("SLEEP-"):
            out.append((_tod_offset(p[ti], start), CAP_EVENTS.get(p[ei])))
    return out


def load_cap(root: Path, rid: str) -> Record:
    root = Path(root)
    x, fs, hdr, label = read_channel(root / f"{rid}.edf", r"ECG|EKG")
    ev = parse_cap_txt(root / f"{rid}.txt", hdr.start)
    # Place events on a contiguous 30 s grid anchored at the first event; missing epochs -> None.
    t0 = ev[0][0]
    n = int(round((ev[-1][0] - t0) / EPOCH_S)) + 1
    stages = [None] * n
    for t, s in ev:
        k = int(round((t - t0) / EPOCH_S))
        stages[k] = s
    starts = t0 + np.arange(n) * EPOCH_S
    keep = (starts >= 0) & (starts + EPOCH_S <= len(x) / fs + 1e-6)  # epochs outside the ECG are dropped
    return Record("cap", rid, rid, x, fs, starts[keep], [s for s, k in zip(stages, keep) if k],
                  meta={"ecg_label": label})


# ------------------------------------------------------------------ MIT-BIH slpdb
SLPDB_SUBJECT = {"slp01a": "slp01", "slp01b": "slp01", "slp02a": "slp02", "slp02b": "slp02"}


def load_slpdb(root: Path, rid: str) -> Record:
    root = Path(root)
    sig, f = wfdb.rdsamp(str(root / rid), channel_names=["ECG"])
    ann = wfdb.rdann(str(root / rid), "st")
    fs = float(f["fs"])
    starts, stages = [], []
    for s, aux in zip(ann.sample, ann.aux_note):
        tok = aux.replace("\x00", "").split()
        starts.append(s / fs)
        stages.append(SLPDB_STAGES.get(tok[0]) if tok else None)
    return Record("slpdb", rid, SLPDB_SUBJECT.get(rid, rid), sig[:, 0], fs,
                  np.array(starts), stages, meta={"ecg_label": "ECG", "comments": f["comments"]})


# ------------------------------------------------------------------ Apnea-ECG
# c06 is the same recording as c05 shifted by 80 s (identical RR series), so both map to one subject.
APNEA_ECG_SUBJECT = {"c06": "c05"}
APNEA_ECG_LEARNING = [f"a{i:02d}" for i in range(1, 21)] + [f"b{i:02d}" for i in range(1, 6)] + \
                     [f"c{i:02d}" for i in range(1, 11)]


def load_apnea_ecg(root: Path, rid: str) -> Record:
    root = Path(root)
    sig, f = wfdb.rdsamp(str(root / rid), channel_names=["ECG"])
    ann = wfdb.rdann(str(root / rid), "apn")
    fs = float(f["fs"])
    n_min = int(len(sig) / fs // 60)
    lab = -np.ones(n_min, dtype=int)
    for s, sym in zip(ann.sample, ann.symbol):
        k = int(round(s / fs / 60))
        if k < n_min:
            lab[k] = 1 if sym == "A" else 0
    return Record("apnea-ecg", rid, APNEA_ECG_SUBJECT.get(rid, rid), sig[:, 0], fs, apnea_minutes=lab, meta={"ecg_label": "ECG"})


def list_records(raw: Path, cfg: dict, include_hmc: bool = False) -> dict[str, list[str]]:
    raw = Path(raw)
    ucddb = [l[:-4] for l in (raw / "ucddb" / "RECORDS").read_text().split() if l.endswith(".rec")]
    slp = (raw / "slpdb" / "RECORDS").read_text().split()
    out = {"ucddb": ucddb, "slpdb": slp, "cap": list(cfg["datasets"]["cap"]["records"]),
           "apnea-ecg": APNEA_ECG_LEARNING}
    if include_hmc:  # v2 external test set; never part of the v1 default list
        out["hmc"] = hmc_listed_records(raw / "hmc")
    return out


def load(raw: Path, dataset: str, rid: str, cfg: dict) -> Record:
    raw = Path(raw)
    if dataset == "ucddb":
        return load_ucddb(raw / "ucddb", rid, cfg.get("apnea"))
    if dataset == "cap":
        return load_cap(raw / "capslpdb", rid)
    if dataset == "slpdb":
        return load_slpdb(raw / "slpdb", rid)
    if dataset == "apnea-ecg":
        return load_apnea_ecg(raw / "apnea-ecg", rid)
    if dataset == "hmc":
        return load_hmc(raw / "hmc", rid)
    raise ValueError(dataset)


# ------------------------------------------------------------------ HMC (v2 external test set)
# HMC Sleep Staging Database 1.1 (PhysioNet hmc-sleep-staging, CC BY 4.0), AASM scoring.
# Test set only: nothing here may be used for fitting, normalization, selection or tuning.
HMC_STAGES = {"Sleep stage W": "W", "Sleep stage R": "R", "Sleep stage N1": "S1",
              "Sleep stage N2": "S2", "Sleep stage N3": "S3"}  # anything else -> None
HMC_SCORING_COLUMNS = ["Date", "Time", "Recording onset", "Duration", "Annotation", "Linked channel"]


def parse_hmc_scoring(path: Path) -> list[dict]:
    """Rows of an HMC SN###_sleepscoring.txt file (comma-separated, header row first).

    Date is dd.mm.yy and Time is hh.mm.ss[.mmm]; onset and duration are seconds
    from the EDF start.
    """
    lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    cols = [c.strip() for c in lines[0].split(",")]
    if cols != HMC_SCORING_COLUMNS:
        raise ValueError(f"{path}: unexpected scoring columns {cols}")
    out = []
    for l in lines[1:]:
        p = [c.strip() for c in l.split(",")]
        p += [""] * (len(cols) - len(p))
        out.append({"date": p[0], "time": p[1], "onset_s": float(p[2]), "duration_s": float(p[3]),
                    "annotation": p[4], "linked_channel": p[5]})
    return out


def hmc_clock(date: str, time: str) -> dt.datetime:
    d, mo, y = (int(x) for x in date.split("."))
    t = time.split(".")
    us = int(round(float("0." + t[3]) * 1e6)) if len(t) > 3 else 0
    return dt.datetime(2000 + y if y < 85 else 1900 + y, mo, d, int(t[0]), int(t[1]), int(t[2]), us)


def _hmc_dir(root: Path) -> Path:
    root = Path(root)
    return root / "recordings" if (root / "recordings").is_dir() else root


def load_hmc(root: Path, rid: str) -> Record:
    """Load one HMC recording. `root` is data/raw/hmc (or its recordings/ folder).

    Stages: W, R, N1 -> S1, N2 -> S2, N3 -> S3 (so light = N1 + N2 and deep = N3 via
    config [stages]); any other stage label -> None. Epochs start at the listed
    'Recording onset' (seconds from the EDF start); epochs that do not end
    inside the ECG are dropped. meta lights_off_s / lights_on_s are the first
    'Lights off' / last 'Lights on' onsets, or None when not annotated.
    """
    d = _hmc_dir(root)
    x, fs, hdr, label = read_channel(d / f"{rid}.edf", r"^ECG$|ECG|EKG")
    rows = parse_hmc_scoring(d / f"{rid}_sleepscoring.txt")
    stage_rows = [r for r in rows if r["annotation"].startswith("Sleep stage")]
    dur = len(x) / fs
    keep = [r for r in stage_rows if r["onset_s"] + EPOCH_S <= dur + 1e-6]
    # Lights offsets from the Date/Time column (ms precision); the onset column is rounded to ~6 digits.
    clk = lambda r: round((hmc_clock(r["date"], r["time"]) - hdr.start).total_seconds(), 3)
    off = [clk(r) for r in rows if r["annotation"] == "Lights off"]
    on = [clk(r) for r in rows if r["annotation"] == "Lights on"]
    other = sorted({r["annotation"] for r in rows
                    if not r["annotation"].startswith("Sleep stage") and r["annotation"] not in ("Lights off", "Lights on")})
    return Record("hmc", rid, rid, x, fs, np.array([r["onset_s"] for r in keep]),
                  [HMC_STAGES.get(r["annotation"]) for r in keep],
                  meta={"ecg_label": label, "raw_stages": [r["annotation"] for r in keep],
                        "lights_off_s": off[0] if off else None, "lights_on_s": on[-1] if on else None,
                        "other_annotations": other, "edf_start": hdr.start.isoformat()})


def hmc_sha256_table(raw_hmc: Path) -> dict[str, str]:
    """Map 'recordings/SN001.edf' -> expected SHA-256 from SHA256SUMS.txt."""
    out = {}
    for l in (Path(raw_hmc) / "SHA256SUMS.txt").read_text().splitlines():
        p = l.split()
        if len(p) == 2:
            out[p[1]] = p[0]
    return out


def hmc_listed_records(raw_hmc: Path) -> list[str]:
    """Record ids listed in the HMC RECORDS file (e.g. SN001)."""
    return [Path(l).stem for l in (Path(raw_hmc) / "RECORDS").read_text().split() if l.endswith(".edf")]


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hmc_record_status(raw_hmc: Path, rid: str, sums: dict[str, str] | None = None) -> str:
    """'ok' if the EDF and scoring txt exist, are non-empty and match SHA256SUMS.txt;
    otherwise 'missing' or 'sha256_mismatch' (e.g. a download still in progress)."""
    raw_hmc = Path(raw_hmc)
    sums = sums if sums is not None else hmc_sha256_table(raw_hmc)
    for rel in (f"recordings/{rid}.edf", f"recordings/{rid}_sleepscoring.txt"):
        f = raw_hmc / rel
        if not (f.exists() and f.stat().st_size > 0):
            return "missing"
        if rel not in sums or _sha256_file(f) != sums[rel]:
            return "sha256_mismatch"
    return "ok"
