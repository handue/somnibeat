"""Inspect the HMC Sleep Staging Database files (v2 external test set) and cache RR series.

Every fact in the report is read from the files on disk. Each planning
assumption is a pass/fail check; the script exits with status 1 if any check
fails. Only records whose EDF and scoring txt both exist, are non-empty and
match SHA256SUMS.txt are processed (the download may still be running).

With beat detection on (the default), R-peaks are detected with
somnibeat.rpeak.detect_r_peaks and written to
    data/cache/v2/beats/hmc_<rid>.npy   (+ .sha256 = SHA-256 of rpeak.py)
    data/cache/v2/rr/hmc_<rid>.csv      time_s (4 dp), rr_ms (2 dp)
HMC is a test set: this script computes no model prediction and no
performance number. RR validity is reported only as a signal-quality fact.

Usage:
    python scripts/check_hmc.py [--config config/v2.toml] [--workers 8]
                                [--no-beats] [--allow-partial]
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from somnibeat import datasets, rpeak, rr as rrm  # noqa: E402
from somnibeat.edf import read_header  # noqa: E402

DETECTOR_SHA = hashlib.sha256(Path(rpeak.__file__).read_bytes()).hexdigest()
EXPECTED_STAGES = {"Sleep stage W", "Sleep stage N1", "Sleep stage N2", "Sleep stage N3", "Sleep stage R"}
LIGHTS = {"Lights off", "Lights on"}
RESP_PATTERN = re.compile(r"apnea|apnoea|hypopnea|hypopnoea|resp|desat|arousal|RERA", re.I)
EPOCH_S = 30.0

checks: list[dict] = []


def check(name: str, expected, observed, ok: bool, note: str = "") -> None:
    checks.append({"dataset": "hmc", "check": name, "expected": expected,
                   "observed": observed, "pass": bool(ok), "note": note})


# ---------------------------------------------------------------- per record (worker)
def inspect_record(job: tuple[str, str, dict, bool]) -> dict:
    raw_hmc, rid, cfg, do_beats = job
    try:
        return _inspect(Path(raw_hmc), rid, cfg, do_beats)
    except Exception as e:  # keep the record id in the error
        return {"record": rid, "status": "error", "error": repr(e)}


def _inspect(raw_hmc: Path, rid: str, cfg: dict, do_beats: bool) -> dict:
    status = datasets.hmc_record_status(raw_hmc, rid)
    out: dict = {"record": rid, "status": status}
    if status != "ok":
        return out
    rec_dir = raw_hmc / "recordings"
    edf = rec_dir / f"{rid}.edf"
    hdr = read_header(edf)
    n_rec_header = int(open(edf, "rb").read(256)[236:244].decode("ascii").strip())
    n_rec_size = (edf.stat().st_size - hdr.header_bytes) // (2 * int(hdr.samples_per_record.sum()))
    ecg_idx = [i for i, l in enumerate(hdr.labels) if re.search(r"ECG|EKG", l, re.I)]
    out.update({
        "edf_start": hdr.start.isoformat(),
        "edf_duration_s": hdr.duration_s,
        "edf_records_header_vs_size": [n_rec_header, int(n_rec_size)],
        "channels": hdr.labels,
        "ecg_channels": [{"label": hdr.labels[i], "fs_hz": float(hdr.fs[i]), "unit": hdr.units[i]}
                         for i in ecg_idx],
    })

    txt = rec_dir / f"{rid}_sleepscoring.txt"
    header_cols = [c.strip() for c in txt.read_text(errors="replace").splitlines()[0].split(",")]
    out["scoring_columns"] = header_cols
    rows = datasets.parse_hmc_scoring(txt)
    stage = [r for r in rows if r["annotation"].startswith("Sleep stage")]
    lights = [r for r in rows if r["annotation"] in LIGHTS]
    other = [r for r in rows if not r["annotation"].startswith("Sleep stage") and r["annotation"] not in LIGHTS]
    onsets = np.array([r["onset_s"] for r in stage])
    steps = np.diff(onsets)
    # Clock alignment: Date + Time column vs EDF start + Recording onset.
    def clock_err(rs: list[dict]) -> float | None:
        e = [abs((datasets.hmc_clock(r["date"], r["time"]) - hdr.start).total_seconds() - r["onset_s"]) for r in rs]
        return float(max(e)) if e else None
    scored_end = float(onsets[-1] + stage[-1]["duration_s"]) if stage else None
    out.update({
        "n_rows": len(rows),
        "stage_counts": dict(collections.Counter(r["annotation"] for r in stage)),
        "stage_durations_s": sorted({r["duration_s"] for r in stage}),
        "first_stage_onset_s": float(onsets[0]) if len(onsets) else None,
        "stage_onset_steps_s": sorted({float(s) for s in steps}),
        "stage_onsets_multiple_of_30": bool(np.all(np.abs(onsets / EPOCH_S - np.round(onsets / EPOCH_S)) < 1e-6)),
        "scored_end_s": scored_end,
        "edf_minus_scored_end_s": (hdr.duration_s - scored_end) if scored_end is not None else None,
        "max_clock_misalignment_stage_s": clock_err(stage),
        "max_clock_misalignment_other_s": clock_err(lights + other),
        "lights": [{"annotation": r["annotation"], "onset_s": r["onset_s"], "duration_s": r["duration_s"],
                    "linked_channel": r["linked_channel"]} for r in lights],
        "other_annotations": dict(collections.Counter(r["annotation"] for r in other)),
        "linked_channels": dict(collections.Counter(r["linked_channel"] for r in rows if r["linked_channel"])),
    })

    if do_beats:
        out.update(_beats(raw_hmc, rid, cfg))
    return out


def _beats(raw_hmc: Path, rid: str, cfg: dict) -> dict:
    cache = Path(cfg["paths"]["cache"])
    rec = datasets.load_hmc(raw_hmc, rid)
    bpath = cache / "beats" / f"hmc_{rid}.npy"
    spath = bpath.with_suffix(".sha256")
    if bpath.exists() and spath.exists() and spath.read_text().strip() == DETECTOR_SHA:
        beats = np.load(bpath)
    else:
        beats = rpeak.detect_r_peaks(rec.ecg, rec.fs)
        bpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(bpath, beats)
        spath.write_text(DETECTOR_SHA)
    t, rr = rrm.rr_from_beats(beats)
    t, rr = np.round(t, 4), np.round(rr, 2)
    rp = cache / "rr" / f"hmc_{rid}.csv"
    rp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": t, "rr_ms": rr}).to_csv(rp, index=False)
    valid = rrm.clean_rr(t, rr, cfg["rr"])
    dur = len(rec.ecg) / rec.fs
    # ECG signal facts per 30 s block: flat (1st-99th percentile range < 1 uV) or clipped
    # (>= 10 % of samples at the channel's physical min/max).
    hdr = read_header(raw_hmc / "recordings" / f"{rid}.edf")
    i = hdr.labels.index(rec.meta["ecg_label"])
    n = int(dur // EPOCH_S)
    blk = rec.ecg[: int(n * EPOCH_S * rec.fs)].reshape(n, -1)
    p1, p99 = np.percentile(blk, [1, 99], axis=1)
    tol = 1e-6 * (hdr.phys_max[i] - hdr.phys_min[i])
    clip = ((blk <= hdr.phys_min[i] + tol) | (blk >= hdr.phys_max[i] - tol)).mean(axis=1)
    return {"ecg_flat_block_fraction": float(np.mean(p99 - p1 < 1.0)),
            "ecg_clipped_block_fraction": float(np.mean(clip >= 0.10)),
            "n_beats": int(len(beats)), "beats_per_s": float(len(beats) / dur),
            "median_rr_ms": float(np.median(rr)) if len(rr) else None,
            "rr_valid_fraction": float(valid.mean()) if len(valid) else 0.0,
            "n_epochs_loaded": len(rec.stages),
            "n_epochs_unmapped": int(sum(s is None for s in rec.stages))}


# ---------------------------------------------------------------- global checks
def parse_demographics(path: Path) -> dict:
    text = path.read_text(errors="replace")
    out: dict = {}
    m = re.search(r"Gender distribution:\s*(\d+)\s*M\s*/\s*(\d+)\s*F", text)
    if m:
        out["male"], out["female"] = int(m.group(1)), int(m.group(2))
    for name in ("Age", "TIB", "TST", "SE", "AHI_TST", "ArI_TST"):
        m = re.search(rf"^{name}\s+([\d.]+)\s+\(([\d.]+)\)", text, re.M)
        if m:
            out[name] = {"mean": float(m.group(1)), "std": float(m.group(2))}
    out["per_subject_rows_present"] = bool(re.search(r"^SN\d{3}", text, re.M))
    return out


def summarize(raw_hmc: Path, recs: list[dict], listed: list[str], sums: dict, allow_partial: bool,
              do_beats: bool) -> dict:
    ok = [r for r in recs if r["status"] == "ok"]
    not_ok = {r["record"]: r["status"] for r in recs if r["status"] != "ok"}
    errors = {r["record"]: r["error"] for r in recs if r["status"] == "error"}

    # Files
    sum_edf = sorted(Path(k).stem for k in sums if re.fullmatch(r"recordings/SN\d+\.edf", k))
    check("RECORDS lists the same recordings as SHA256SUMS.txt", "identical sets",
          {"RECORDS": len(listed), "SHA256SUMS_edf": len(sum_edf)}, sorted(listed) == sum_edf)
    check("every listed record has a verified EDF + scoring txt (SHA-256)", f"{len(listed)} verified",
          {"verified": len(ok), "missing": sum(v == "missing" for v in not_ok.values()),
           "sha256_mismatch": sorted(k for k, v in not_ok.items() if v == "sha256_mismatch")},
          len(ok) == len(listed) or allow_partial,
          "--allow-partial: incomplete download tolerated, only verified records checked" if allow_partial and
          len(ok) < len(listed) else "")
    check("no record raised an error while being read", "none", errors, not errors)
    if not ok:
        check("at least one verified record", ">= 1", 0, False)
        return {}

    # ECG channel
    n_ecg = {r["record"]: len(r["ecg_channels"]) for r in ok}
    check("exactly one ECG channel per record", 1, sorted(set(n_ecg.values())),
          all(v == 1 for v in n_ecg.values()), str({k: v for k, v in n_ecg.items() if v != 1}))
    ecg_labels = collections.Counter(c["label"] for r in ok for c in r["ecg_channels"])
    check("ECG channel label is 'ECG'", ["ECG"], dict(ecg_labels), set(ecg_labels) == {"ECG"})
    fs = collections.Counter(c["fs_hz"] for r in ok for c in r["ecg_channels"])
    check("ECG sampling rate is 256 Hz in every record", 256.0, dict(fs), set(fs) == {256.0})
    units = collections.Counter(c["unit"] for r in ok for c in r["ecg_channels"])
    check("ECG physical unit is a voltage (uV or mV)", ["uV", "mV"], dict(units),
          set(units) <= {"uV", "mV"})
    bad_n = {r["record"]: r["edf_records_header_vs_size"] for r in ok
             if r["edf_records_header_vs_size"][0] != r["edf_records_header_vs_size"][1]}
    check("EDF data-record count in header equals the count implied by file size", "equal", bad_n, not bad_n)

    # Scoring file format
    cols = collections.Counter(tuple(r["scoring_columns"]) for r in ok)
    check("scoring txt columns", datasets.HMC_SCORING_COLUMNS, [list(c) for c in cols],
          set(cols) == {tuple(datasets.HMC_SCORING_COLUMNS)})
    vocab = collections.Counter()
    for r in ok:
        vocab.update(r["stage_counts"])
    check("stage label vocabulary is Sleep stage W/N1/N2/N3/R", sorted(EXPECTED_STAGES), dict(vocab),
          set(vocab) <= EXPECTED_STAGES)
    durs = sorted({d for r in ok for d in r["stage_durations_s"]})
    check("every stage annotation lasts 30 s (epoch length)", [30.0], durs, durs == [30.0])
    first = {r["record"]: r["first_stage_onset_s"] for r in ok if r["first_stage_onset_s"] != 0.0}
    check("first stage epoch starts at the EDF start (onset 0)", 0.0, first, not first)
    gaps = {r["record"]: r["stage_onset_steps_s"] for r in ok if r["stage_onset_steps_s"] != [30.0]}
    check("stage epochs are contiguous (onset step exactly 30 s)", [30.0], gaps, not gaps)
    grid = [r["record"] for r in ok if not r["stage_onsets_multiple_of_30"]]
    check("stage onsets lie on a 30 s grid from the EDF start", "all", grid, not grid)
    mis = {r["record"]: r["max_clock_misalignment_stage_s"] for r in ok if r["max_clock_misalignment_stage_s"] > 0.0015}
    check("stage rows: Date/Time column equals EDF start + Recording onset (<= 1.5 ms)", "<= 0.0015 s",
          {"max_s": max(r["max_clock_misalignment_stage_s"] for r in ok), "records_over": mis}, not mis)
    oth = [r["max_clock_misalignment_other_s"] for r in ok if r["max_clock_misalignment_other_s"] is not None]
    mis = {r["record"]: r["max_clock_misalignment_other_s"] for r in ok
           if r["max_clock_misalignment_other_s"] is not None and r["max_clock_misalignment_other_s"] > 0.0501}
    check("lights rows: Date/Time column equals EDF start + Recording onset (<= 0.05 s)", "<= 0.05 s",
          {"max_s": max(oth) if oth else None, "records_over": mis}, not mis,
          "the onset column is printed with ~6 significant digits (e.g. 25618.7 for clock 07.06.28.740), "
          "so late lights annotations are rounded to 0.1 s; the clock column keeps ms")
    over = {r["record"]: r["edf_minus_scored_end_s"] for r in ok if r["edf_minus_scored_end_s"] < -1e-6}
    tail = [r["edf_minus_scored_end_s"] for r in ok]
    check("scored epochs end inside the EDF", ">= 0 s unscored tail",
          {"tail_s_min": min(tail), "tail_s_max": max(tail), "records_scored_past_end": over}, not over,
          "epochs scored past the EDF end are dropped by load_hmc")

    # Lights
    lights_n = collections.Counter((sum(l["annotation"] == "Lights off" for l in r["lights"]),
                                    sum(l["annotation"] == "Lights on" for l in r["lights"])) for r in ok)
    check("each record has at most one 'Lights off' and one 'Lights on'", "<= (1, 1)",
          {f"off={a},on={b}": n for (a, b), n in lights_n.items()}, all(a <= 1 and b <= 1 for a, b in lights_n))
    order_bad, off_s, on_tail = [], [], []
    for r in ok:
        off = [l["onset_s"] for l in r["lights"] if l["annotation"] == "Lights off"]
        on = [l["onset_s"] for l in r["lights"] if l["annotation"] == "Lights on"]
        if off and on and off[0] >= on[-1]:
            order_bad.append(r["record"])
        if off:
            off_s.append(off[0])
        if on:
            on_tail.append(r["scored_end_s"] - on[-1])
        if (off and not 0 <= off[0] <= r["edf_duration_s"]) or (on and not 0 <= on[-1] <= r["edf_duration_s"]):
            order_bad.append(r["record"] + " (outside EDF)")
    check("lights off precedes lights on, both inside the EDF", "all", order_bad, not order_bad)
    lights_facts = {"records_with_lights_off": len(off_s), "records_with_lights_on": len(on_tail),
                    "lights_off_onset_s": _range(off_s), "scored_end_minus_lights_on_s": _range(on_tail)}

    # Other annotations (respiratory events expected to be absent)
    other = collections.Counter()
    for r in ok:
        other.update(r["other_annotations"])
    resp = {k: v for k, v in other.items() if RESP_PATTERN.search(k)}
    check("no respiratory-event annotations in the scoring txt", {}, resp, not resp,
          "HMC scoring txt holds sleep stages and lights only; apnea is not evaluable on HMC")
    check("no annotation types other than sleep stages and lights", {}, dict(other), not other)

    # Demographics
    demo = parse_demographics(raw_hmc / "Subjects_info_aggregated.txt")
    n_demo = demo.get("male", 0) + demo.get("female", 0)
    check("demographics gender counts sum to the number of listed records", len(listed),
          {"male": demo.get("male"), "female": demo.get("female")}, n_demo == len(listed))

    # Beats (signal quality only; no model numbers)
    beat_facts = {}
    if do_beats:
        bps = {r["record"]: round(r["beats_per_s"], 3) for r in ok}
        odd = {k: v for k, v in bps.items() if not 0.5 <= v <= 2.5}
        check("detected beat rate is physiological (0.5-2.5 beats/s = 30-150 bpm)", "0.5-2.5",
              {k: {"beats_per_s": v, "ecg_flat_block_fraction": round(e["ecg_flat_block_fraction"], 3),
                   "ecg_clipped_block_fraction": round(e["ecg_clipped_block_fraction"], 3)}
               for k, v in odd.items() for e in ok if e["record"] == k}, not odd,
              "a failure here is a property of the recording (lead-off/flat/clipped ECG), not of the loader; "
              "per docs/rr_rules.md such epochs get low coverage and are still evaluated")
        bad_sig = {r["record"]: [round(r["ecg_flat_block_fraction"], 3), round(r["ecg_clipped_block_fraction"], 3)]
                   for r in ok if r["ecg_flat_block_fraction"] + r["ecg_clipped_block_fraction"] > 0.05}
        vf = [r["rr_valid_fraction"] for r in ok]
        low = {r["record"]: round(r["rr_valid_fraction"], 3) for r in ok if r["rr_valid_fraction"] < 0.8}
        unm = {r["record"]: r["n_epochs_unmapped"] for r in ok if r["n_epochs_unmapped"]}
        check("every loaded epoch maps to W/R/S1/S2/S3", 0, unm, not unm)
        beat_facts = {"records_ecg_flat_or_clipped_over_5pct_[flat,clipped]": bad_sig,
                      "rr_valid_fraction": _range(vf), "records_rr_valid_below_0.8": low,
                      "median_rr_ms": _range([r["median_rr_ms"] for r in ok]),
                      "epochs_loaded_total": int(sum(r["n_epochs_loaded"] for r in ok))}

    durs_h = [r["edf_duration_s"] / 3600 for r in ok]
    return {"n_verified": len(ok), "edf_duration_h": _range(durs_h), "lights": lights_facts,
            "stage_epochs_total": dict(vocab), "linked_channels_in_scoring": dict(sum(
                (collections.Counter(r["linked_channels"]) for r in ok), collections.Counter())),
            "demographics_aggregated": demo, "beats": beat_facts}


def _range(v: list[float]) -> dict | None:
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"n": int(len(a)), "min": round(float(a.min()), 4), "median": round(float(np.median(a)), 4),
            "max": round(float(a.max()), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/v2.toml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-beats", action="store_true", help="skip R-peak detection and RR caching")
    ap.add_argument("--allow-partial", action="store_true",
                    help="do not fail when some listed records are not downloaded/verified yet")
    ap.add_argument("--out", default=None, help="default: <results>/hmc_check.json")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text())
    raw_hmc = Path(cfg["paths"]["raw"]) / "hmc"
    out = Path(a.out or Path(cfg["paths"]["results"]) / "hmc_check.json")
    sums = datasets.hmc_sha256_table(raw_hmc)
    listed = datasets.hmc_listed_records(raw_hmc)
    jobs = [(str(raw_hmc), rid, cfg, not a.no_beats) for rid in listed]
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(inspect_record, jobs))
    for r in recs:
        if r["status"] == "ok":
            extra = f" beats={r['n_beats']:6d} rr_valid={r['rr_valid_fraction']:.3f}" if "n_beats" in r else ""
            print(f"hmc {r['record']} ok ecg={r['ecg_channels'][0]['label'] if r['ecg_channels'] else '-'}"
                  f" dur={r['edf_duration_s'] / 3600:.2f}h{extra}", flush=True)
    facts = summarize(raw_hmc, recs, listed, sums, a.allow_partial, not a.no_beats)
    n_fail = sum(not c["pass"] for c in checks)
    report = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "PhysioNet hmc-sleep-staging 1.1 (CC BY 4.0)",
        "note": "Test set. Facts read from files; no model prediction or performance number computed.",
        "detector_sha256": DETECTOR_SHA if not a.no_beats else None,
        "n_listed": len(listed), "summary": facts, "n_checks": len(checks), "n_failed": n_fail,
        "checks": checks,
        "records": [{k: v for k, v in r.items() if k not in ("channels", "linked_channels")} for r in recs],
        "channels_first_verified_record": next((r["channels"] for r in recs if r["status"] == "ok"), None),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: observed={json.dumps(c['observed'], default=str)[:200]}"
              + (f"  ({c['note']})" if c["note"] else ""))
    print(f"{len(checks) - n_fail}/{len(checks)} checks passed -> {out}")
    if n_fail:
        print("FAILED: see the [FAIL] lines above; each lists the records or values that broke the assumption.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
