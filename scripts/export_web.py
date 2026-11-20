"""Export everything the static web demo needs into web/data and web/assets.

Writes (see docs/web_contract.md):
    web/data/models.json     gradient boosting trees for every task, exact floats
    web/data/results.json    subset of results/*.json (no per-epoch data)
    web/data/samples.json    sample metadata + attribution
    web/data/samples/*.csv   three sample RR files copied from data/cache/rr
    web/data/samples/*_reference.json
                             expert reference labels (stages per 30 s epoch, apnea per
                             minute) for the UCDDB samples; public PhysioNet annotations,
                             never a model prediction
    web/assets/*.svg         brand assets copied unmodified (favicon = mark)

Nothing here retrains or re-evaluates a model: the pickles in results/models
and the numbers in results/*.json are only read and re-serialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["sleep_wake", "wake_nrem_rem", "four_class", "apnea"]
THRESHOLD_KEYS = {"sleep_wake": ("threshold_p_wake", "wake"), "apnea": ("threshold_p_apnea", "apnea")}
SCHEMA = 1

LICENSE = "Open Data Commons Attribution License v1.0"
LICENSE_SHORT = "ODC-By 1.0"
LICENSE_URL = "https://opendatacommons.org/licenses/by/1-0/"
PHYSIONET_CITATION = ("Pollard T, Moody BE, et al. \"PhysioNet as a global platform for biomedical research.\" "
                      "Nature Health (2026). https://doi.org/10.1038/s44360-026-00096-z")

DATASETS = {
    "ucddb": {
        "name": "St. Vincent's University Hospital / University College Dublin Sleep Apnea Database (UCDDB)",
        "short": "UCDDB",
        "url": "https://physionet.org/content/ucddb/1.0.0/",
        "doi": "10.13026/C26C7D",
        "citation": "No original publication is listed on the PhysioNet page; cite the dataset DOI "
                    "(doi:10.13026/C26C7D) and the PhysioNet platform paper.",
        "population": "25 adults (21 male, 4 female) referred with suspected sleep-disordered breathing.",
    },
    "cap": {
        "name": "CAP Sleep Database (capslpdb)",
        "short": "CAP",
        "url": "https://physionet.org/content/capslpdb/1.0.0/",
        "doi": None,
        "citation": "Terzano MG, et al. \"Atlas, rules, and recording techniques for the scoring of cyclic "
                    "alternating pattern (CAP) in human sleep.\" Sleep Medicine 2:537-553 (2001).",
        "population": "A subset of patient recordings (insomnia, narcolepsy, periodic leg movements, REM sleep "
                      "behavior disorder, sleep-disordered breathing, bruxism) and 15 healthy controls; "
                      "epilepsy (NFLE) records are not used.",
    },
    "slpdb": {
        "name": "MIT-BIH Polysomnographic Database (slpdb)",
        "short": "MIT-BIH slpdb",
        "url": "https://physionet.org/content/slpdb/1.0.0/",
        "doi": "10.13026/C23K5S",
        "citation": "Ichimaru Y, Moody GB. \"Development of the polysomnographic database on CD-ROM.\" "
                    "Psychiatry and Clinical Neurosciences 53:175-177 (1999).",
        "population": "18 records from 16 male subjects (mean age 40) recorded in a sleep laboratory.",
    },
    "apnea-ecg": {
        "name": "Apnea-ECG Database (apnea-ecg)",
        "short": "Apnea-ECG",
        "url": "https://physionet.org/content/apnea-ecg/1.0.0/",
        "doi": None,
        "citation": "Penzel T, Moody GB, Mark RG, Goldberger AL, Peter JH. \"The Apnea-ECG Database.\" "
                    "Computers in Cardiology 27:255-258 (2000).",
        "population": "35 learning-set records (a01-a20, b01-b05, c01-c10) with per-minute apnea labels.",
    },
}

ASSETS = [("assets/somnibeat-mark.svg", "somnibeat-mark.svg"),
          ("assets/somnibeat-lockup.svg", "somnibeat-lockup.svg"),
          ("assets/somnibeat-mark.svg", "favicon.svg")]


# ---------------------------------------------------------------- JSON helpers

def fmt_float(x: float) -> str:
    """Shortest repr that round-trips exactly. +-inf as 1e999 (valid JSON, parses to +-Infinity)."""
    x = float(x)
    if math.isnan(x):
        raise ValueError("NaN cannot be exported")
    if math.isinf(x):
        return "1e999" if x > 0 else "-1e999"
    r = repr(x)
    return r[:-2] if r.endswith(".0") else r


def fmt_int_list(a) -> str:
    return "[" + ",".join(str(int(v)) for v in a) + "]"


def fmt_float_list(a) -> str:
    return "[" + ",".join(fmt_float(v) for v in a) + "]"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump_compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def dump_pretty(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


# ---------------------------------------------------------------- models.json

def tree_json(nodes: np.ndarray) -> str:
    leaf = nodes["is_leaf"].astype(bool)
    if nodes["is_categorical"].any():
        raise SystemExit("Categorical splits are not supported by the web model format.")
    # Fields that are never read are zeroed to keep the file small:
    # value is read only at leaves; f, t, m, l, r only at split nodes.
    f = np.where(leaf, 0, nodes["feature_idx"])
    t = np.where(leaf, 0.0, nodes["num_threshold"])
    m = np.where(leaf, 0, nodes["missing_go_to_left"])
    lft = np.where(leaf, 0, nodes["left"])
    rgt = np.where(leaf, 0, nodes["right"])
    v = np.where(leaf, nodes["value"], 0.0)
    return ('{"f":' + fmt_int_list(f) + ',"t":' + fmt_float_list(t) + ',"m":' + fmt_int_list(m)
            + ',"l":' + fmt_int_list(lft) + ',"r":' + fmt_int_list(rgt) + ',"v":' + fmt_float_list(v)
            + ',"leaf":' + fmt_int_list(leaf.astype(int)) + "}")


def task_json(task: str, bundle: dict) -> str:
    model = bundle["model"]
    cols, labels, thr = list(bundle["columns"]), list(bundle["labels"]), bundle["threshold"]
    if list(getattr(model, "feature_names_in_", cols)) != cols:
        raise SystemExit(f"{task}: pickle columns differ from the model's feature_names_in_.")
    if list(model.classes_) != list(range(len(labels))):
        raise SystemExit(f"{task}: unexpected classes {model.classes_}.")
    k = model.n_trees_per_iteration_
    if len(labels) == 2:
        if k != 1 or thr is None:
            raise SystemExit(f"{task}: binary model needs one tree per iteration and a threshold.")
        decision = {"type": "threshold", "positive": labels[1], "threshold": float(thr)}
    else:
        if k != len(labels):
            raise SystemExit(f"{task}: expected {len(labels)} trees per iteration, found {k}.")
        decision = {"type": "argmax"}
    baseline = np.asarray(model._baseline_prediction, dtype=np.float64).ravel()
    assert len(baseline) == k
    head = dump_compact({"labels": labels, "columns": cols, "decision": decision, "kind": "hgb"})[:-1]
    trees = "[" + ",".join("[" + ",".join(tree_json(p.nodes) for p in it) + "]" for it in model._predictors) + "]"
    return head + ',"baseline":' + fmt_float_list(baseline) + ',"trees":' + trees + "}"


def export_models(cfg: dict, out: Path) -> dict:
    parts = []
    for task in TASKS:
        with open(ROOT / "results" / "models" / f"{task}_gradient_boosting.pkl", "rb") as f:
            bundle = pickle.load(f)
        parts.append(json.dumps(task) + ":" + task_json(task, bundle))
    config = {"rr": cfg["rr"], "features": cfg["features"], "epoch_s": int(cfg["epochs"]["length_s"]),
              "apnea_epoch_s": 60}
    text = ('{"schema":' + str(SCHEMA) + ',"primary_model":' + json.dumps(cfg["models"]["primary"])
            + ',"config":' + dump_compact(config) + ',"tasks":{' + ",".join(parts) + "}}")
    json.loads(text)  # must be valid JSON
    out.write_text(text, encoding="utf-8")
    return json.loads(text)


# ---------------------------------------------------------------- results.json

def strip(obj):
    """Drop per-record / per-epoch arrays; keep every aggregate number unchanged."""
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in ("per_record", "per_record_error")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def task_block(task: str, res: dict) -> dict:
    models = {}
    for name, entry in res["models"].items():
        e = strip(entry)
        key = THRESHOLD_KEYS.get(task, (None,))[0]
        e["threshold"] = entry.get(key) if key else None
        models[name] = e
    return {"labels": res["labels"], "unit": res.get("unit", "epoch"), "models": models,
            "baselines": strip(res["baselines"]), "data": res["data"]}


def export_results(cfg: dict, out: Path) -> dict:
    rdir = ROOT / "results"
    primary = cfg["models"]["primary"]
    files = [f"{t}.json" for t in TASKS] + ["rpeak_validation.json", "data_check.json"]
    raw = {f: json.loads((rdir / f).read_text()) for f in files}
    tasks = {t: task_block(t, raw[f"{t}.json"]) for t in TASKS}

    sw = raw["sleep_wake.json"]["models"][primary]["external_ucddb"]["per_record"]
    ap = raw["apnea.json"]["models"][primary]["external_ucddb"]["per_record"]
    per_record = {
        "sleep_wake": [{k: r[k] for k in ("record", "ref_tib_min", "ref_tst_min", "pred_tst_min", "ref_se_pct",
                                          "pred_se_pct", "ref_awakenings", "pred_awakenings", "kappa")} for r in sw],
        "apnea": [{k: r[k] for k in ("record", "ref_apnea_min", "pred_apnea_min", "total_min", "kappa")} for r in ap],
    }

    rp = raw["rpeak_validation.json"]
    rpeak = {"tolerance_ms": rp["tolerance_ms"],
             "datasets": {d: {k: v for k, v in b.items() if k != "per_record"} for d, b in rp["datasets"].items()}}

    dc = raw["data_check.json"]
    data_check = {"all_pass": dc["all_pass"], "n_checks": len(dc["checks"]),
                  "n_failed": sum(1 for c in dc["checks"] if not c["pass"])}

    stage_data, apnea_data = raw["sleep_wake.json"]["data"], raw["apnea.json"]["data"]
    blocks = {
        "ucddb": (stage_data["test"]["ucddb"], "External test set (sleep staging and apnea); never used for training"),
        "cap": (stage_data["train"]["cap"], "Training (sleep staging)"),
        "slpdb": (stage_data["train"]["slpdb"], "Training (sleep staging)"),
        "apnea-ecg": (apnea_data["train"]["apnea-ecg"], "Training (apnea minutes)"),
    }
    if apnea_data["test"]["ucddb"]["records"] != blocks["ucddb"][0]["records"]:
        raise SystemExit("UCDDB record counts differ between staging and apnea results.")
    datasets = []
    for key, (block, role) in blocks.items():
        meta = DATASETS[key]
        datasets.append({"id": key, "name": meta["name"], "short": meta["short"], "role": role,
                         "records": block["records"], "subjects": block["subjects"],
                         "population": meta["population"], "license": LICENSE, "license_url": LICENSE_URL,
                         "citation": meta["citation"], "doi": meta["doi"], "url": meta["url"]})

    n_ucddb = blocks["ucddb"][0]["records"]
    n_slpdb = blocks["slpdb"][0]["subjects"]
    limitations = [
        "Most recordings come from sleep clinics: people referred for suspected sleep disorders, not the general population.",
        f"The external test set (UCDDB) has only {n_ucddb} nights, so every number has a wide confidence interval.",
        f"The MIT-BIH training data come from {n_slpdb} subjects, all male.",
        "Accuracy on home wearables (watches, chest straps, rings) has not been validated; "
        "all heartbeats here were detected from clinical ECG.",
        "The 3-stage (wake / NREM / REM) and 4-stage (wake / light / deep / REM) results are experimental "
        "and agree with the reference less well than sleep/wake.",
        "Apnea definitions differ: Apnea-ECG comes with its own expert per-minute apnea labels, while "
        "UCDDB minutes are derived here from scored apnea and hypopnea events, so external apnea numbers mix "
        "model error with definition mismatch.",
        "The apnea output is a screening hint, not a diagnosis. Only a sleep study read by a clinician can "
        "diagnose sleep apnea.",
    ]
    result = {
        "schema": SCHEMA,
        "generated_from": {f: sha256_file(rdir / f) for f in files},
        "primary_model": primary,
        "config_sha256": raw["sleep_wake.json"].get("config_sha256"),
        "tasks": tasks,
        "per_record": per_record,
        "rpeak_validation": rpeak,
        "data_check": data_check,
        "datasets": datasets,
        "physionet_citation": PHYSIONET_CITATION,
        "limitations": limitations,
    }
    out.write_text(dump_pretty(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------- samples

def select_samples(cfg: dict) -> list[dict]:
    """Apply the selection rules and check they still pick the expected records."""
    primary = cfg["models"]["primary"]
    ap = json.loads((ROOT / "results" / "apnea.json").read_text())
    rows = ap["models"][primary]["external_ucddb"]["per_record"]
    fewest = min(rows, key=lambda r: r["ref_apnea_min"])
    most = max(rows, key=lambda r: r["ref_apnea_min"])
    summary = json.loads((ROOT / "data" / "cache" / "record_summary.json").read_text())
    ae = [r for r in summary["records"] if r["dataset"] == "apnea-ecg"]
    noisiest = min(ae, key=lambda r: r["valid_fraction"])
    expected = {"ordinary": "ucddb018", "apnea": "ucddb025", "noisy": "c04"}
    got = {"ordinary": fewest["record"], "apnea": most["record"], "noisy": noisiest["record"]}
    if got != expected:
        raise SystemExit(f"Sample selection rules now pick {got}, expected {expected}. Update export_web.py.")

    ucddb_src = {"dataset": DATASETS["ucddb"]["name"], "license": LICENSE_SHORT, "license_url": LICENSE_URL,
                 "url": DATASETS["ucddb"]["url"], "doi": DATASETS["ucddb"]["doi"],
                 "note": "Part of the external test set: the models never saw this night during training. "
                         "Reference sleep stages and apnea events exist for it."}
    ae_src = {"dataset": DATASETS["apnea-ecg"]["name"], "license": LICENSE_SHORT, "license_url": LICENSE_URL,
              "url": DATASETS["apnea-ecg"]["url"], "doi": None,
              "note": "Part of the apnea training set. It has no sleep-stage reference, so the hypnogram "
                      "cannot be compared with a reference for this night."}
    fmt_min = lambda x: f"{x:g}"  # noqa: E731
    return [
        {"id": "ordinary", "title": "An ordinary night",
         "summary": f"A UCDDB night with the fewest reference apnea minutes in the test set "
                    f"({fmt_min(fewest['ref_apnea_min'])} reference apnea minutes).",
         "file": "samples/ucddb018.csv", "cache_file": "data/cache/rr/ucddb_ucddb018.csv",
         "selection": "UCDDB record with the fewest reference apnea minutes.",
         "has_stage_reference": True,
         "source": {**ucddb_src, "record": "ucddb018"}},
        {"id": "apnea", "title": "Many suspected apnea minutes",
         "summary": f"A UCDDB night with the most reference apnea minutes in the test set "
                    f"({fmt_min(most['ref_apnea_min'])} reference apnea minutes).",
         "file": "samples/ucddb025.csv", "cache_file": "data/cache/rr/ucddb_ucddb025.csv",
         "selection": "UCDDB record with the most reference apnea minutes.",
         "has_stage_reference": True,
         "source": {**ucddb_src, "record": "ucddb025"}},
        {"id": "noisy", "title": "A noisy night",
         "summary": "An Apnea-ECG night with many missing or unusable beats, to show how the report "
                    "marks stretches without enough usable data.",
         "file": "samples/apnea-ecg-c04.csv", "cache_file": "data/cache/rr/apnea-ecg_c04.csv",
         "selection": "Apnea-ECG record with the lowest fraction of valid RR intervals after cleaning "
                      "(data/cache/record_summary.json).",
         "has_stage_reference": False,
         "source": {**ae_src, "record": "c04"}},
    ]


# ---------------------------------------------------------------- reference labels

REFERENCE_SCHEMA = 1
REFERENCE_MODES = ("sleep_wake", "wake_nrem_rem", "four_class")


def stage_key_map(cfg: dict) -> dict[str, str]:
    """Common stage vocabulary (W, R, S1..S4) -> four_class label, from config [stages]."""
    out = {}
    for key in ("wake", "rem", "light", "deep"):
        for code in cfg["stages"][key]:
            out[code] = key
    return out


def mode_label(four: str | None, mode: str) -> str | None:
    """four_class label -> label of a display mode (same vocabulary as the models)."""
    if four is None:
        return None
    if mode == "four_class":
        return four
    if mode == "wake_nrem_rem":
        return "nrem" if four in ("light", "deep") else four
    return "wake" if four == "wake" else "sleep"


def last_beat_s(csv_path: Path) -> float:
    lines = [l for l in csv_path.read_text().splitlines() if l.strip()]
    header = [h.strip().lower() for h in lines[0].split(",")]
    return float(lines[-1].split(",")[header.index("time_s")])


def reference_json(cfg: dict, rid: str, sample_csv: Path) -> dict:
    """Reference arrays on the demo grid that starts at 0 (no lights-off/on).

    n_epochs = floor(last beat / 30), n_minutes = floor(last beat / 60), exactly as the
    report grid of web/js/report.js without clock inputs. Epoch k covers [30 k, 30 k + 30)
    s, minute m covers [60 m, 60 m + 60) s. Unscored epochs and epochs or minutes beyond
    the annotation are null.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from somnibeat.datasets import load_ucddb

    rec = load_ucddb(ROOT / "data" / "raw" / "ucddb", rid, cfg.get("apnea"))
    end_s = last_beat_s(sample_csv)
    epoch_s = int(cfg["epochs"]["length_s"])
    n_epochs = int(math.floor(end_s / epoch_s))
    n_min = int(math.floor(end_s / 60))
    if not np.array_equal(rec.epoch_start_s, np.arange(len(rec.stages)) * float(epoch_s)):
        raise SystemExit(f"{rid}: reference epochs are not on the {epoch_s} s grid from 0.")
    keys = stage_key_map(cfg)
    four = [keys.get(s) if s is not None else None for s in rec.stages]
    four = (four + [None] * n_epochs)[:n_epochs]
    ap = [int(v) if int(v) in (0, 1) else None for v in rec.apnea_minutes]
    ap = (ap + [None] * n_min)[:n_min]
    return {
        "schema": REFERENCE_SCHEMA,
        "record": rid,
        "kind": "expert reference (polysomnography), not a model output",
        "source": {"dataset": DATASETS["ucddb"]["name"], "url": DATASETS["ucddb"]["url"],
                   "doi": DATASETS["ucddb"]["doi"], "license": LICENSE_SHORT, "license_url": LICENSE_URL,
                   "files": [f"{rid}_stage.txt", f"{rid}_respevt.txt"]},
        "grid": {"start_s": 0, "epoch_s": epoch_s, "n_epochs": n_epochs, "minute_s": 60, "n_minutes": n_min},
        "stage_mapping": {k: list(cfg["stages"][k]) for k in ("wake", "rem", "light", "deep")},
        "apnea_rule": {"event_prefixes": list(cfg["apnea"]["ucddb_event_prefixes"]),
                       "exclude_types": list(cfg["apnea"]["ucddb_exclude_types"])},
        "stages": {mode: [mode_label(v, mode) for v in four] for mode in REFERENCE_MODES},
        "apnea": ap,
    }


def export_references(cfg: dict, web: Path, samples: list[dict]) -> None:
    """Write <stem>_reference.json for samples with a stage reference; set sample['reference']."""
    for s in samples:
        s.pop("reference", None)
        if not s.get("has_stage_reference"):
            continue
        csv_path = web / "data" / s["file"]
        rel = s["file"].rsplit(".", 1)[0] + "_reference.json"
        ref = reference_json(cfg, s["source"]["record"], csv_path)
        (web / "data" / rel).write_text(dump_compact(ref) + "\n", encoding="utf-8")
        s["reference"] = rel


def export_samples(cfg: dict, web: Path, copy_csv: bool = True) -> list[dict]:
    samples = select_samples(cfg)
    (web / "data" / "samples").mkdir(parents=True, exist_ok=True)
    for s in samples:
        src = s.pop("cache_file")
        if copy_csv:
            shutil.copyfile(ROOT / src, web / "data" / s["file"])
    export_references(cfg, web, samples)
    (web / "data" / "samples.json").write_text(dump_pretty(samples), encoding="utf-8")
    return samples


def export_assets(web: Path) -> None:
    (web / "assets").mkdir(parents=True, exist_ok=True)
    for src, dst in ASSETS:
        shutil.copyfile(ROOT / src, web / "assets" / dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "config" / "default.toml"))
    ap.add_argument("--web", default=str(ROOT / "web"))
    ap.add_argument("--reference-only", action="store_true",
                    help="only write the sample reference JSON files and samples.json "
                         "(models.json, results.json, sample CSVs and assets stay untouched)")
    args = ap.parse_args(argv)
    cfg = tomllib.loads(Path(args.config).read_text())
    web = Path(args.web)
    (web / "data").mkdir(parents=True, exist_ok=True)
    if args.reference_only:
        samples = export_samples(cfg, web, copy_csv=False)
        for p in ["data/samples.json"] + [f"data/{s['reference']}" for s in samples if s.get("reference")]:
            print(f"wrote web/{p} ({(web / p).stat().st_size} bytes)")
        return 0
    export_models(cfg, web / "data" / "models.json")
    export_results(cfg, web / "data" / "results.json")
    export_samples(cfg, web)
    export_assets(web)
    for p in ["data/models.json", "data/results.json", "data/samples.json"]:  # + *_reference.json
        print(f"wrote web/{p} ({(web / p).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
