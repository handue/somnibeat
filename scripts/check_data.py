"""Inspect the raw dataset files and report facts read from them.

Run before any modeling. Every fact in the report is read from the files on
disk, not from documentation. Each planning assumption is checked explicitly;
the script exits with status 1 if any assumption fails.

Usage:
    python scripts/check_data.py [--data-dir data/raw] [--out results/data_check.json]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import wfdb

EPOCH_S = 30.0

UCDDB_STAGE_CODES = {0: "Wake", 1: "REM", 2: "S1", 3: "S2", 4: "S3", 5: "S4",
                     6: "Artifact", 7: "Indeterminate",  # 0-7 from the PhysioNet UCDDB page
                     8: "Undocumented_8"}  # seen in ucddb008/024; decision 2026-11-19: drop as unscored
APNEA_ECG_ALLOWED = {"A", "N"}

checks: list[dict] = []


def check(dataset: str, name: str, expected, observed, ok: bool, note: str = "") -> None:
    checks.append({"dataset": dataset, "check": name, "expected": expected,
                   "observed": observed, "pass": bool(ok), "note": note})


def tod_offset(clock: str, start: dt.datetime) -> float:
    """Seconds from the recording start to a time-of-day string, wrapping midnight."""
    h, m, s = (int(x) for x in clock.split(":"))
    t = start.replace(hour=h, minute=m, second=s, microsecond=0)
    off = (t - start).total_seconds()
    if off < -6 * 3600:  # event after midnight, recording started before midnight
        off += 86400
    return off


def read_edf_header(path: Path) -> dict:
    """Parse the fixed-width EDF header directly.

    A strict reader (pyedflib) rejects some CAP files because of out-of-spec
    digital min/max fields, so only the header bytes are read here.
    """
    with open(path, "rb") as fh:
        h = fh.read(256).decode("ascii", "replace")
        ns = int(h[252:256])
        sh = fh.read(256 * ns).decode("ascii", "replace")
    size = path.stat().st_size

    def field(offset: int, width: int) -> list[str]:
        base = offset * ns
        return [sh[base + i * width: base + (i + 1) * width].strip() for i in range(ns)]

    labels = field(0, 16)
    units = field(16 + 80, 8)
    spr = [int(x) for x in field(16 + 80 + 8 + 8 * 4 + 80, 8)]
    n_rec = int(h[236:244])
    rec_dur = float(h[244:252])
    d, mo, y = (int(x) for x in h[168:176].split("."))
    hh, mi, ss = (int(x) for x in h[176:184].split("."))
    start = dt.datetime(2000 + y if y < 85 else 1900 + y, mo, d, hh, mi, ss)
    hdr_bytes = 256 * (ns + 1)
    n_rec_from_size = (size - hdr_bytes) // (2 * sum(spr))
    return {"start": start, "labels": labels, "units": units,
            "fs": [s / rec_dur for s in spr], "n_records_header": n_rec,
            "n_records_from_size": n_rec_from_size, "record_duration_s": rec_dur}


def edf_ecg_info(path: Path) -> dict:
    h = read_edf_header(path)
    n_rec = h["n_records_header"] if h["n_records_header"] > 0 else h["n_records_from_size"]
    ecg = [i for i, lab in enumerate(h["labels"]) if re.search(r"ECG|EKG", lab, re.I)]
    return {
        "start": h["start"],
        "duration_s": n_rec * h["record_duration_s"],
        "header_vs_filesize_records": [h["n_records_header"], h["n_records_from_size"]],
        "all_channels": h["labels"],
        "ecg_channels": [{"label": h["labels"][i], "fs_hz": h["fs"][i], "unit": h["units"][i]}
                         for i in ecg],
    }


# ---------------------------------------------------------------- UCDDB
RESP_LINE = re.compile(
    r"^\s*(\d\d:\d\d:\d\d)\s+(\S+)\s+(?:(PB|CS)\s+)?(\d+)(?:\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?))?")


def parse_respevt(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        m = RESP_LINE.match(line)
        if m:
            events.append({"time": m.group(1), "type": m.group(2), "pb_cs": m.group(3),
                           "duration_s": int(m.group(4)),
                           "desat_low_pct": float(m.group(5)) if m.group(5) else None})
    return events


def spo2_alignment(rec: Path, events: list[dict], offsets: np.ndarray) -> dict | None:
    """Find the clock shift that best matches listed desaturation nadirs to the SpO2 channel.

    Each event with a listed 'Low' value is compared with the minimum of the
    recorded SpO2 signal in [onset, onset + duration + 45 s]. If the event
    times are aligned to the EDF start, the mean absolute error is smallest
    at a shift close to zero.
    """
    import pyedflib  # strict reader is fine for UCDDB files

    with pyedflib.EdfReader(str(rec)) as f:
        i = f.getSignalLabels().index("SpO2")
        fs = f.getSampleFrequency(i)
        x = f.readSignal(i)
    idx = [k for k, e in enumerate(events) if e["desat_low_pct"] is not None]
    if len(idx) < 5:
        return None
    mae = {}
    for shift in range(-900, 901, 10):
        errs = []
        for k in idx:
            a = int((offsets[k] + shift) * fs)
            b = int((offsets[k] + shift + events[k]["duration_s"] + 45) * fs)
            seg = x[max(a, 0):max(min(b, len(x)), 0)]
            seg = seg[(seg > 50) & (seg <= 100)]  # drop dropouts
            if len(seg):
                errs.append(abs(seg.min() - events[k]["desat_low_pct"]))
        if errs:
            mae[shift] = float(np.mean(errs))
    best = min(mae, key=mae.get)
    return {"events_with_listed_nadir": len(idx), "best_shift_s": best,
            "mae_at_best": round(mae[best], 3), "mae_at_zero": round(mae[0], 3)}


def check_ucddb(root: Path) -> dict:
    out = {"records": {}}
    recs = sorted(root.glob("ucddb*.rec"))
    check("UCDDB", "records present", ">=1", len(recs), len(recs) >= 1)
    type_counts = collections.Counter()
    for rec in recs:
        rid = rec.stem
        info = edf_ecg_info(rec)
        codes = [int(x) for x in (root / f"{rid}_stage.txt").read_text().split()]
        n_ep = len(codes)
        events = parse_respevt(root / f"{rid}_respevt.txt")
        type_counts.update(e["type"] for e in events)
        offsets = np.array([tod_offset(e["time"], info["start"]) for e in events])
        ends = offsets + np.array([e["duration_s"] for e in events])
        resp = [i for i, e in enumerate(events) if e["type"].startswith(("APNEA", "HYP"))]
        onset_ep = (offsets[resp] // EPOCH_S).astype(int) if resp else np.array([], int)
        valid = onset_ep[(onset_ep >= 0) & (onset_ep < n_ep)]
        in_wake = float(np.mean([codes[i] == 0 for i in valid])) if len(valid) else None

        # Alignment probe: fraction of event onsets falling in scored sleep (codes 1-5)
        # for a range of hypothetical clock shifts. Correct alignment should peak near 0.
        probe = {}
        for shift in range(-1800, 1801, 300):
            ep = ((offsets[resp] + shift) // EPOCH_S).astype(int) if resp else np.array([], int)
            ep = ep[(ep >= 0) & (ep < n_ep)]
            probe[shift] = round(float(np.mean([1 <= codes[i] <= 5 for i in ep])), 3) if len(ep) else None

        r = {
            "edf_start": info["start"].isoformat(),
            "edf_duration_s": info["duration_s"],
            "ecg_channels": info["ecg_channels"],
            "stage_epochs": n_ep,
            "edf_duration_over_30s": round(info["duration_s"] / EPOCH_S, 3),
            "stage_code_counts": {UCDDB_STAGE_CODES.get(k, f"UNKNOWN_{k}"): v
                                  for k, v in sorted(collections.Counter(codes).items())},
            "resp_events": len(events),
            "resp_event_types": dict(collections.Counter(e["type"] for e in events)),
            "resp_event_duration_s_min_max": [int(min(e["duration_s"] for e in events)),
                                              int(max(e["duration_s"] for e in events))] if events else None,
            "first_event_offset_s": float(offsets.min()) if len(offsets) else None,
            "last_event_end_offset_s": float(ends.max()) if len(ends) else None,
            "apnea_hypopnea_onsets_in_wake_fraction": in_wake,
            "alignment_probe_fraction_in_sleep_by_shift_s": probe,
        }
        out["records"][rid] = r
        fs = [c["fs_hz"] for c in info["ecg_channels"]]
        check("UCDDB", f"{rid} ECG fs", 128.0, fs, fs == [128.0])
        check("UCDDB", f"{rid} stage epochs match 30 s x EDF length",
              "floor(duration/30) within 1", [n_ep, r["edf_duration_over_30s"]],
              abs(n_ep - info["duration_s"] // EPOCH_S) <= 1,
              "stage file has no timestamps; epoch 0 assumed to start at EDF start")
        check("UCDDB", f"{rid} stage codes known", "subset of 0-8 (8 dropped)", sorted(set(codes)),
              set(codes) <= set(UCDDB_STAGE_CODES))
        if len(offsets):
            check("UCDDB", f"{rid} resp events inside recording", "0 <= t <= duration",
                  [r["first_event_offset_s"], r["last_event_end_offset_s"], info["duration_s"]],
                  offsets.min() >= 0 and ends.max() <= info["duration_s"] + EPOCH_S)
        sp = spo2_alignment(rec, events, offsets)
        r["spo2_alignment"] = sp
        if sp is not None:
            check("UCDDB", f"{rid} event clock aligned to EDF start (SpO2 nadir test)",
                  "best shift within +-30 s", sp, abs(sp["best_shift_s"]) <= 30,
                  "listed desaturation 'Low' vs SpO2 channel minimum")
    out["all_resp_event_types"] = dict(type_counts)
    return out


# ---------------------------------------------------------------- CAP
def parse_cap_txt(path: Path) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if l.startswith("Sleep Stage"))
    cols = lines[hdr_i].split("\t")
    ti = next(i for i, c in enumerate(cols) if c.startswith("Time"))
    ei = cols.index("Event")
    di = next(i for i, c in enumerate(cols) if c.startswith("Duration"))
    rows = []
    for l in lines[hdr_i + 1:]:
        p = l.split("\t")
        if len(p) > max(ti, ei, di) and p[ei].startswith("SLEEP-"):
            rows.append({"stage": p[0], "time": p[ti].replace(".", ":"),
                         "event": p[ei], "duration_s": float(p[di])})
    return rows, cols


def check_cap(root: Path, records: list[str] | None = None) -> dict:
    out = {"records": {}}
    edfs = sorted(root.glob("*.edf"))
    if records is not None:  # only the records the config uses
        edfs = [e for e in edfs if e.stem in set(records)]
        missing = sorted(set(records) - {e.stem for e in edfs})
        check("CAP", "configured records downloaded", [], missing, not missing)
    check("CAP", "records present", ">=1", len(edfs), len(edfs) >= 1)
    for edf in edfs:
        rid = edf.stem
        info = edf_ecg_info(edf)
        rows, cols = parse_cap_txt(root / f"{rid}.txt")
        offs = np.array([tod_offset(r["time"], info["start"]) for r in rows])
        durs = np.array([r["duration_s"] for r in rows])
        gaps = np.diff(offs) - durs[:-1]
        r = {
            "edf_start": info["start"].isoformat(),
            "edf_duration_s": info["duration_s"],
            "ecg_channels": info["ecg_channels"],
            "txt_columns": cols,
            "stage_events": len(rows),
            "stage_event_counts": dict(collections.Counter(x["event"] for x in rows)),
            "stage_durations_s": sorted(set(durs.tolist())),
            "first_stage_offset_s": float(offs[0]),
            "first_offset_mod_30_s": float(offs[0] % EPOCH_S),
            "last_stage_end_offset_s": float(offs[-1] + durs[-1]),
            "gaps_between_stage_events_s": dict(collections.Counter(gaps.round(1).tolist())),
        }
        out["records"][rid] = r
        check("CAP", f"{rid} ECG channel found", ">=1", [c["label"] for c in info["ecg_channels"]],
              len(info["ecg_channels"]) >= 1)
        check("CAP", f"{rid} stage epoch length", [30.0], r["stage_durations_s"],
              r["stage_durations_s"] == [30.0])
        missing = int(np.round(gaps[gaps > 0.5] / EPOCH_S).sum())
        r["missing_epochs_inside_scoring"] = missing
        r["unscored_tail_s"] = info["duration_s"] - r["last_stage_end_offset_s"]
        check("CAP", f"{rid} stages on a 30 s grid", "every gap a multiple of 30 s",
              r["gaps_between_stage_events_s"],
              bool(np.all(np.abs(gaps / EPOCH_S - np.round(gaps / EPOCH_S)) < 0.02)),
              f"{missing} epochs missing inside the scored span; treated as unscored and dropped")
        check("CAP", f"{rid} stage onsets on whole seconds from EDF start", "offset % 1 == 0",
              r["first_stage_offset_s"], float(r["first_stage_offset_s"]).is_integer())
        overrun = r["last_stage_end_offset_s"] - info["duration_s"]
        r["epochs_past_ecg_end_dropped"] = max(0, int(np.ceil(overrun / EPOCH_S)))
        check("CAP", f"{rid} stages inside EDF", "first >= 0; at most 3 epochs past ECG end (dropped)",
              [r["first_stage_offset_s"], r["last_stage_end_offset_s"], info["duration_s"]],
              offs[0] >= 0 and overrun <= 3 * EPOCH_S,
              f"{r['epochs_past_ecg_end_dropped']} trailing epochs extend past the ECG and are dropped")
    return out


# ---------------------------------------------------------------- MIT-BIH slpdb
SLPDB_STAGES = {"W", "1", "2", "3", "4", "R", "MT", "M"}  # "M" seen in slp45; treated as MT (discarded)


def check_slpdb(root: Path) -> dict:
    out = {"records": {}}
    heas = sorted(root.glob("*.hea"))
    check("slpdb", "records present", ">=1", len(heas), len(heas) >= 1)
    for hea in heas:
        rid = hea.stem
        h = wfdb.rdheader(str(root / rid))
        ann = wfdb.rdann(str(root / rid), "st")
        aux = [s.replace("\x00", "") for s in ann.aux_note]  # slp41/slp45 pad aux with NUL
        stages = [s.split()[0] if s.strip() else "" for s in aux]
        tokens = collections.Counter(t for s in aux for t in s.split()[1:])
        d = np.diff(ann.sample)
        ecg = [i for i, n in enumerate(h.sig_name) if "ECG" in n.upper()]
        r = {
            "fs_hz": h.fs, "signals": h.sig_name, "comments": h.comments,
            "duration_s": h.sig_len / h.fs,
            "ecg_channels": [h.sig_name[i] for i in ecg],
            "stage_annotations": len(ann.sample),
            "first_annotation_s": float(ann.sample[0] / h.fs),
            "annotation_spacing_s": dict(collections.Counter((d / h.fs).round(3).tolist())),
            "stage_counts": dict(collections.Counter(stages)),
            "non_stage_tokens": dict(tokens),
        }
        out["records"][rid] = r
        check("slpdb", f"{rid} ECG fs", 250, h.fs, h.fs == 250 and len(ecg) >= 1)
        check("slpdb", f"{rid} stage epoch length", "30 s (+-0.01)", r["annotation_spacing_s"],
              all(abs(s - EPOCH_S) <= 0.01 for s in r["annotation_spacing_s"]),
              f"first annotation at {r['first_annotation_s']} s")
        check("slpdb", f"{rid} stage labels known", sorted(SLPDB_STAGES), sorted(set(stages)),
              set(stages) <= SLPDB_STAGES)
        check("slpdb", f"{rid} annotations inside record", "<= duration",
              [float(ann.sample[-1] / h.fs + EPOCH_S), r["duration_s"]],
              ann.sample[-1] / h.fs + EPOCH_S <= r["duration_s"] + 1)
    return out


# ---------------------------------------------------------------- Apnea-ECG
def check_apnea_ecg(root: Path) -> dict:
    out = {"records": {}}
    apns = sorted(root.glob("*.apn"))
    apns = [a for a in apns if re.fullmatch(r"[abc]\d\d", a.stem)]
    check("Apnea-ECG", "learning-set records present", ">=1", len(apns), len(apns) >= 1)
    for apn in apns:
        rid = apn.stem
        h = wfdb.rdheader(str(root / rid))
        ann = wfdb.rdann(str(root / rid), "apn")
        d = np.diff(ann.sample)
        n_minutes = h.sig_len / h.fs / 60
        r = {
            "fs_hz": h.fs, "signals": h.sig_name, "duration_s": h.sig_len / h.fs,
            "labels": len(ann.symbol), "label_counts": dict(collections.Counter(ann.symbol)),
            "first_label_s": float(ann.sample[0] / h.fs),
            "label_spacing_s": dict(collections.Counter((d / h.fs).round(3).tolist())),
            "recording_minutes": round(n_minutes, 2),
        }
        out["records"][rid] = r
        check("Apnea-ECG", f"{rid} ECG fs", 100, h.fs, h.fs == 100 and "ECG" in h.sig_name)
        check("Apnea-ECG", f"{rid} one label per minute from t=0", "spacing 60 s, first 0 s",
              [r["first_label_s"], r["label_spacing_s"]],
              r["first_label_s"] == 0 and set(r["label_spacing_s"]) == {60.0})
        r["unlabeled_tail_minutes"] = round(n_minutes - len(ann.symbol), 2)
        check("Apnea-ECG", f"{rid} labels never extend past the recording",
              "labels <= minutes + 1", [len(ann.symbol), r["recording_minutes"]],
              len(ann.symbol) <= n_minutes + 1,
              "unlabeled tail minutes are excluded from evaluation")
        check("Apnea-ECG", f"{rid} label symbols", sorted(APNEA_ECG_ALLOWED),
              sorted(set(ann.symbol)), set(ann.symbol) <= APNEA_ECG_ALLOWED)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/data_check.json")
    ap.add_argument("--config", default="config/default.toml")
    a = ap.parse_args()
    base = Path(a.data_dir)
    import tomllib
    cap_records = tomllib.loads(Path(a.config).read_text())["datasets"]["cap"]["records"]
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ucddb": check_ucddb(base / "ucddb"),
        "cap": check_cap(base / "capslpdb", cap_records),
        "slpdb": check_slpdb(base / "slpdb"),
        "apnea_ecg": check_apnea_ecg(base / "apnea-ecg"),
    }
    report["checks"] = checks
    report["all_pass"] = all(c["pass"] for c in checks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))

    for c in checks:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"[{flag}] {c['dataset']:10s} {c['check']}: expected {c['expected']}, observed {c['observed']}")
    print(f"\n{sum(c['pass'] for c in checks)}/{len(checks)} checks passed -> {a.out}")
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
