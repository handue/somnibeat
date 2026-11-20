"""Sleep lab reference export for the demo (web/data/samples/*_reference.json).

Checks that scripts/export_web.py writes the expert reference arrays deterministically,
that the committed files match a fresh export, that they hold only reference labels
(no model prediction), and that the browser-side alignment and agreement helpers in
web/js/charts.js behave as documented on synthetic data. No model is run here.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
RAW = ROOT / "data" / "raw" / "ucddb"
NODE = shutil.which("node")

FOUR = {"wake", "rem", "light", "deep"}
VOCAB = {"sleep_wake": {"wake", "sleep"}, "wake_nrem_rem": {"wake", "nrem", "rem"}, "four_class": FOUR}
TOP_KEYS = {"schema", "record", "kind", "source", "grid", "stage_mapping", "apnea_rule", "stages", "apnea"}
UCDDB_SAMPLES = ["ucddb018", "ucddb025", "ucddb015"]


def load_export():
    spec = importlib.util.spec_from_file_location("export_web", ROOT / "scripts" / "export_web.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cfg():
    return tomllib.loads((ROOT / "config" / "default.toml").read_text())


needs_raw = pytest.mark.skipif(not (RAW / "ucddb018_stage.txt").exists(), reason="UCDDB raw data missing")


@needs_raw
def test_export_is_deterministic_and_matches_committed(tmp_path):
    ex = load_export()
    outs = []
    for i in range(2):
        web = tmp_path / f"web{i}"
        ex.export_samples(cfg(), web)
        outs.append(web)
    committed = WEB / "data"
    names = ["samples.json"] + [f"samples/{rid}_reference.json" for rid in UCDDB_SAMPLES]
    for name in names:
        a = (outs[0] / "data" / name).read_bytes()
        b = (outs[1] / "data" / name).read_bytes()
        assert a == b, f"{name} differs between two exports"
        assert a == (committed / name).read_bytes(), f"committed web/data/{name} is stale: rerun export_web.py"
    assert not (outs[0] / "data" / "samples" / "apnea-ecg-c04_reference.json").exists()


def test_samples_json_references():
    samples = json.loads((WEB / "data" / "samples.json").read_text())
    by_record = {s["source"]["record"]: s for s in samples}
    assert by_record["ucddb018"]["reference"] == "samples/ucddb018_reference.json"
    assert by_record["ucddb025"]["reference"] == "samples/ucddb025_reference.json"
    assert by_record["ucddb015"]["reference"] == "samples/ucddb015_reference.json"
    assert "reference" not in by_record["c04"] and by_record["c04"]["has_stage_reference"] is False
    for s in samples:
        if "reference" in s:
            assert (WEB / "data" / s["reference"]).exists()
            assert Path(s["reference"]).stem == Path(s["file"]).stem + "_reference"


@pytest.mark.parametrize("rid", UCDDB_SAMPLES)
def test_reference_content(rid):
    ref = json.loads((WEB / "data" / "samples" / f"{rid}_reference.json").read_text())
    assert set(ref) == TOP_KEYS, "unexpected keys (the file must hold reference labels only)"
    text = json.dumps(ref).lower()
    for word in ("pred", "prob", "estimate", "model_", "threshold"):
        assert word not in text, f"reference file mentions {word!r}"
    g = ref["grid"]
    assert g["start_s"] == 0 and g["epoch_s"] == 30 and g["minute_s"] == 60
    lines = [l for l in (WEB / "data" / "samples" / f"{rid}.csv").read_text().splitlines() if l.strip()]
    last = float(lines[-1].split(",")[0])
    assert g["n_epochs"] == int(last // 30) and g["n_minutes"] == int(last // 60)
    for mode, vocab in VOCAB.items():
        labels = ref["stages"][mode]
        assert len(labels) == g["n_epochs"]
        assert set(labels) - {None} <= vocab
    four = ref["stages"]["four_class"]
    derived_sw = [None if v is None else ("wake" if v == "wake" else "sleep") for v in four]
    derived_3 = [None if v is None else ("nrem" if v in ("light", "deep") else v) for v in four]
    assert ref["stages"]["sleep_wake"] == derived_sw
    assert ref["stages"]["wake_nrem_rem"] == derived_3
    assert len(ref["apnea"]) == g["n_minutes"] and set(ref["apnea"]) - {None} <= {0, 1}


@needs_raw
@pytest.mark.parametrize("rid", UCDDB_SAMPLES)
def test_reference_matches_raw_annotation(rid):
    """Stage codes come straight from <rid>_stage.txt with the config mapping (8 etc. -> gap)."""
    ref = json.loads((WEB / "data" / "samples" / f"{rid}_reference.json").read_text())
    codes = [int(c) for c in (RAW / f"{rid}_stage.txt").read_text().split()]
    ucddb = {0: "W", 1: "R", 2: "S1", 3: "S2", 4: "S3", 5: "S4"}
    keys = {code: k for k, v in cfg()["stages"].items() for code in v}
    expect = [keys.get(ucddb.get(c)) for c in codes]
    n = ref["grid"]["n_epochs"]
    expect = (expect + [None] * n)[:n]
    assert ref["stages"]["four_class"] == expect


JS_CHECK = r"""
import { alignReference, compareNight, hypnogramSvg } from '%CHARTS%';
const reference = {
  grid: { start_s: 0, epoch_s: 30, n_epochs: 8, minute_s: 60, n_minutes: 4 },
  stages: {
    sleep_wake: ['wake', 'sleep', 'sleep', null, 'sleep', 'wake', 'sleep', 'sleep'],
    wake_nrem_rem: ['wake', 'nrem', 'rem', null, 'nrem', 'wake', 'nrem', 'nrem'],
    four_class: ['wake', 'light', 'rem', null, 'deep', 'wake', 'light', 'light'],
  },
  apnea: [0, 1, 1, 0],
};
const report = {
  grid: { start_s: 0, end_s: 240, epoch_s: 30, n_epochs: 8 },
  stages: {
    sleep_wake: ['wake', 'sleep', 'wake', 'sleep', 'sleep', 'wake', 'sleep', 'sleep'],
    wake_nrem_rem: ['wake', 'nrem', 'wake', 'nrem', 'nrem', 'wake', 'nrem', 'rem'],
    four_class: ['wake', 'light', 'wake', 'light', 'deep', 'wake', 'light', 'rem'],
  },
  lowQuality: [0, 0, 0, 0, 0, 0, 0, 0],
  apnea: { minute_s: 60, labels: [0, 1, 0, 0] },
};
const a = alignReference(reference, report);
const c = compareNight(report, a, 'sleep_wake');
const c4 = compareNight(report, a, 'four_class');
// Lights off 60 s later: shift by 2 epochs / 1 minute, pad the end with gaps.
const shifted = { ...report, grid: { ...report.grid, start_s: 60 } };
const s = alignReference(reference, shifted);
const odd = alignReference(reference, { ...report, grid: { ...report.grid, start_s: 45 } });
const svg = hypnogramSvg({ report: { ...report, stages: { sleep_wake: a.stages.sleep_wake }, lowQuality: [], apnea: { minute_s: 60, labels: a.apnea } }, subject: 'Lab' });
console.log(JSON.stringify({ c, c4, s, odd, svgHasGapNote: svg.includes('1 of 8 epochs not scored') }));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_browser_compare_helpers(tmp_path):
    script = tmp_path / "check.mjs"
    script.write_text(JS_CHECK.replace("%CHARTS%", (WEB / "js" / "charts.js").as_uri()))
    out = json.loads(subprocess.run([NODE, str(script)], capture_output=True, text=True, check=True).stdout)
    c, c4, s = out["c"], out["c4"], out["s"]
    assert c["nScored"] == 7 and abs(c["agreementPct"] - 100 * 6 / 7) < 1e-12
    assert abs(c4["agreementPct"] - 100 * 5 / 7) < 1e-12
    assert c["tstModelMin"] == 2.5 and c["tstLabMin"] == 2.5
    assert c["apneaModelMin"] == 1 and c["apneaLabMin"] == 2
    assert s["stages"]["sleep_wake"] == ["sleep", None, "sleep", "wake", "sleep", "sleep", None, None]
    assert s["apnea"] == [1, 1, 0, None]
    assert out["odd"] is None
    assert out["svgHasGapNote"]
