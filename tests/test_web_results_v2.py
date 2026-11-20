"""web/data/results.json v2 block and the 'typical' demo sample (scripts/export_web.py).

Checks that the committed web export matches a fresh export of results/*.json and
results/v2/*.json, that every v2 number is copied unchanged from results/v2/test_results.json,
that the v1 keys keep their shape, and that the typical sample follows its selection rule.
Nothing here trains or evaluates a model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RES = json.loads((WEB / "data" / "results.json").read_text())
TEST = json.loads((ROOT / "results" / "v2" / "test_results.json").read_text())
TASKS = ["sleep_wake", "wake_nrem_rem", "four_class"]


def load_export():
    spec = importlib.util.spec_from_file_location("export_web", ROOT / "scripts" / "export_web.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_results_json_matches_fresh_export(tmp_path):
    ex = load_export()
    cfg = tomllib.loads((ROOT / "config" / "default.toml").read_text())
    out = tmp_path / "results.json"
    ex.export_results(cfg, out, RES["v2"]["frozen"])
    assert out.read_bytes() == (WEB / "data" / "results.json").read_bytes(), \
        "web/data/results.json is stale: rerun scripts/export_web.py --skip-models"


def test_generated_from_hashes_are_current():
    for name, digest in RES["generated_from"].items():
        assert hashlib.sha256((ROOT / "results" / name).read_bytes()).hexdigest() == digest, name
    for name in ("v2/test_results.json", "v2/v2_final_spec.json", "v2/v2_selection.json"):
        assert name in RES["generated_from"]


def test_v1_keys_unchanged_in_shape():
    for key in ("schema", "primary_model", "tasks", "per_record", "rpeak_validation", "data_check", "datasets",
                "physionet_citation", "limitations"):
        assert key in RES
    assert set(RES["tasks"]) == {"sleep_wake", "wake_nrem_rem", "four_class", "apnea"}
    ids = [d["id"] for d in RES["datasets"]]
    assert ids[:4] == ["ucddb", "cap", "slpdb", "apnea-ecg"] and ids[4] == "hmc"
    for d in RES["datasets"]:
        assert set(d["roles"]) == {"v1", "v2"}


@pytest.mark.parametrize("ds", ["hmc", "ucddb"])
def test_v2_metrics_copied_unchanged(ds):
    src = TEST["test_sets"][ds]
    dst = RES["v2"]["test_sets"][ds]
    assert dst["records_all"] == src["records_all"] and dst["records_primary"] == src["records_primary"]
    assert {x["record"]: x["adequate_fraction"] for x in dst["excluded_low_coverage"]} == src["excluded_low_coverage"]
    for task in TASKS:
        for an in ("primary", "all_records"):
            for model, entry in src["tasks"][task][an].items():
                out = dst["tasks"][task][an][model]
                assert out["metrics"] == entry["metrics"]
                assert out["n_records"] == entry["n_records"] and out["n_epochs"] == entry["n_epochs"]
                assert "per_record" not in out
    rows = src["tasks"]["sleep_wake"]["primary"]["v2"]["per_record"]
    exported = RES["v2"]["per_record"][f"{ds}_primary_sleep_wake"]
    assert [r["record"] for r in exported] == [r["record"] for r in rows]
    for a, b in zip(exported, rows):
        assert all(a[k] == b[k] for k in a)
    ba = dst["tasks"]["sleep_wake"]["primary"]["v2"]["bland_altman"]["tst_min"]
    assert "per_record_error" not in ba
    assert ba["bias"] == src["tasks"]["sleep_wake"]["primary"]["v2"]["bland_altman"]["tst_min"]["bias"]


def test_v2_apnea_copied_unchanged():
    for an in ("primary", "all_records"):
        for model, entry in TEST["apnea_ucddb"][an].items():
            assert RES["v2"]["apnea_ucddb"][an][model]["metrics"] == entry["metrics"]
    assert RES["v2"]["apnea_ucddb"]["threshold_p_apnea"] == TEST["apnea_ucddb"]["threshold_p_apnea"]


def test_v2_block_has_no_per_epoch_arrays():
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                assert k not in ("per_record_error", "proba", "labels_per_epoch"), f"{path}/{k}"
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            assert len(o) <= 200, f"{path}: list of {len(o)} items looks like per-epoch data"
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(RES["v2"])


def test_v2_selection_and_frozen():
    sel = json.loads((ROOT / "results" / "v2" / "v2_selection.json").read_text())
    v2 = RES["v2"]
    assert v2["selection"]["staging"]["chosen"]["name"] == sel["staging"]["chosen"]["name"]
    assert v2["selection"]["staging"]["chosen"]["xds_sw_kappa"] == sel["staging"]["chosen"]["xds_sw_kappa"]
    assert v2["selection"]["staging"]["v1_reference"]["xds_sw_kappa"] == sel["staging"]["v1_reference"]["xds_sw_kappa"]
    assert v2["frozen"]["tag"] == "v2-frozen" and v2["frozen"]["commit"] == TEST["commit"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", v2["frozen"]["date"])
    assert set(v2["frozen"]) == {"tag", "date", "commit"}, "only tag, date and commit (no tagger identity)"


def test_hmc_dataset_row():
    hmc = next(d for d in RES["datasets"] if d["id"] == "hmc")
    check = json.loads((ROOT / "results" / "v2" / "hmc_check.json").read_text())["summary"]["demographics_aggregated"]
    assert hmc["records"] == TEST["test_sets"]["hmc"]["records_all"]
    assert hmc["records_primary"] == TEST["test_sets"]["hmc"]["records_primary"]
    assert (hmc["male"], hmc["female"]) == (check["male"], check["female"])
    assert hmc["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert hmc["roles"]["v2"] == "v2 external test (never used before)"


def test_typical_sample_rule():
    rows = TEST["test_sets"]["ucddb"]["tasks"]["sleep_wake"]["primary"]["v2"]["per_record"]
    errs = sorted((abs(r["pred_tst_min"] - r["ref_tst_min"]), r["record"]) for r in rows)
    n = len(errs)
    expect = errs[n // 2][1] if n % 2 else min(errs[n // 2 - 1][1], errs[n // 2][1])
    samples = json.loads((WEB / "data" / "samples.json").read_text())
    assert [s["id"] for s in samples] == ["ordinary", "apnea", "noisy", "typical"]
    typ = samples[-1]
    assert typ["source"]["record"] == expect
    assert typ["file"] == f"samples/{expect}.csv"
    assert (WEB / "data" / typ["file"]).read_bytes() == (ROOT / "data" / "cache" / "rr" / f"ucddb_{expect}.csv").read_bytes()
    assert typ["reference"] == f"samples/{expect}_reference.json"
