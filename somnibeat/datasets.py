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
    keep = starts + EPOCH_S <= len(x) / fs + 1e-6
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


def list_records(raw: Path, cfg: dict) -> dict[str, list[str]]:
    raw = Path(raw)
    ucddb = [l[:-4] for l in (raw / "ucddb" / "RECORDS").read_text().split() if l.endswith(".rec")]
    slp = (raw / "slpdb" / "RECORDS").read_text().split()
    return {"ucddb": ucddb, "slpdb": slp, "cap": list(cfg["datasets"]["cap"]["records"]),
            "apnea-ecg": APNEA_ECG_LEARNING}


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
    raise ValueError(dataset)
