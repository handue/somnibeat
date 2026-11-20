"""v2 candidate selection on the training databases only (docs/v2_protocol.md).

Usage:
  python scripts/v2_select.py --candidates default --out results/v2/selection_staging.json
  python scripts/v2_select.py --candidates my_candidates.json --tasks sleep_wake

--candidates: 'default' (full grid incl. the v1 reference) or a JSON file holding a
list of candidate specs (see somnibeat/v2_models.py). No UCDDB or HMC row is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from somnibeat import v2_models as vm  # noqa: E402
from somnibeat.provenance import drop_timing  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="default")
    ap.add_argument("--out", default="results/v2/selection_staging.json")
    ap.add_argument("--tasks", nargs="*", default=["sleep_wake", "wake_nrem_rem", "four_class"])
    ap.add_argument("--features", default="data/cache/stage_features.csv.gz")
    ap.add_argument("--extra", default="data/cache/v2/stage_features_extra.csv.gz")
    ap.add_argument("--v1-config", default="config/default.toml")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--no-subject-cv", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    assert out.resolve().is_relative_to((ROOT / "results" / "v2").resolve()), "outputs go under results/v2/"
    t_start = time.time()
    feats, extra = Path(a.features), Path(a.extra)
    df, v1_cols, extra_cols = vm.load_stage_data(feats, extra)
    vm.assert_training_only(df)
    extra_ok = bool(extra_cols)
    print(f"loaded {len(df)} rows, datasets {sorted(df.dataset.unique())}, "
          f"{len(v1_cols)} v1 + {len(extra_cols)} extra features ({time.time() - t_start:.1f}s)", flush=True)

    if a.candidates == "default":
        specs = vm.default_grid(extra_ok)
    else:
        specs = json.loads(Path(a.candidates).read_text())
    specs = [vm.validate_spec(s, extra_ok) for s in specs]

    hp = vm.load_v1_hyperparameters(a.v1_config)
    h = vm.Harness(df, v1_cols, extra_cols, hp, n_resamples=a.bootstrap)
    result = {
        "protocol": "docs/v2_protocol.md",
        "selection_metric": "cross_dataset.mean.sleep_wake.kappa_pooled",
        "guard": "cross_dataset.mean.sleep_wake.tst_mae_min not worse than v1_reference by > 10 %",
        "datasets_loaded": sorted(df.dataset.unique()),
        "records": {d: sorted(df.loc[df.dataset == d, "record"].unique()) for d in sorted(df.dataset.unique())},
        "features_path": str(feats), "features_sha256": vm.sha256_file(feats),
        "extra_features_path": str(extra) if extra_ok else None,
        "extra_features_sha256": vm.sha256_file(extra) if extra_ok else None,
        "n_v1_features": len(v1_cols), "n_extra_features": len(extra_cols),
        "v1_config_sha256": hashlib.sha256(Path(a.v1_config).read_bytes()).hexdigest(),
        "hyperparameters": hp, "bootstrap_resamples": a.bootstrap,
        "candidates": [],
    }
    for spec in specs:
        entry = {"spec": {k: v for k, v in spec.items() if k != "skipped"}, "tasks": {}}
        if spec["skipped"]:
            entry["skipped"] = "extra feature file not found"
            result["candidates"].append(entry)
            print(f"skip {spec['name']}: no extra features", flush=True)
            continue
        for task in a.tasks:
            r = h.run_candidate(task, spec, subject_cv=not a.no_subject_cv)
            entry["tasks"][task] = r
            m = r["cross_dataset"]["mean"]["sleep_wake"]
            print(f"{spec['name']:<40} {task:<14} xds kappa {m['kappa_pooled']:.3f} "
                  f"TST MAE {m['tst_mae_min']:.1f}  ({r['seconds']}s, fits so far {h.n_fits})", flush=True)
        result["candidates"].append(entry)
    result["n_model_fits"] = h.n_fits
    result["fit_seconds"] = round(h.fit_seconds, 1)
    result["total_seconds"] = round(time.time() - t_start, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drop_timing(result), indent=2))
    print(f"wrote {out} ({result['total_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
