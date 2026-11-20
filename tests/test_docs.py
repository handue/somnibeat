import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _render_module():
    spec = importlib.util.spec_from_file_location("render_docs", ROOT / "scripts/render_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_docs_are_rendered_from_current_results():
    """README.md and docs/WRITEUP.md must equal a fresh render from the results JSON."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts/render_docs.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


def test_v2_results_are_addressable():
    mod = _render_module()
    results = mod.load_results()
    assert "sleep_wake" in results and "v2_test_results" in results
    path = "v2_test_results/test_sets/hmc/tasks/sleep_wake/primary/v2/metrics/kappa_pooled"
    point = json.loads((ROOT / "results/v2/test_results.json").read_text())[
        "test_sets"]["hmc"]["tasks"]["sleep_wake"]["primary"]["v2"]["metrics"]["kappa_pooled"]["point"]
    assert mod.render("{{p:%s:3}}" % path, results) == f"{point:.3f}"


def test_len_and_keys_placeholders():
    mod = _render_module()
    results = {"x": {"d": {"a": 1, "b": 2}, "l": [1, 2, 3]}}
    assert mod.render("{{len:x/l}} {{len:x/d}}", results) == "3 2"
    assert mod.render("{{keys:x/d}}", results) == "a, b"


def test_rendered_docs_show_v2_hmc_results_and_exclusions():
    tr = json.loads((ROOT / "results/v2/test_results.json").read_text())
    hmc = tr["test_sets"]["hmc"]
    readme = (ROOT / "README.md").read_text()
    writeup = (ROOT / "docs/WRITEUP.md").read_text()
    for rec in hmc["excluded_low_coverage"]:
        assert rec in readme and rec in writeup
    k = hmc["tasks"]["sleep_wake"]["primary"]["v2"]["metrics"]["kappa_pooled"]
    assert f"{k['point']:.2f} (95% CI {k['ci_low']:.2f} to {k['ci_high']:.2f})" in readme
    for heading in ("## 1. Problem and who it is for", "## 2. Method", "## 3. Evaluation",
                    "## 4. Who benefits, and who could be harmed", "## 5. Limitations",
                    "## 6. If we had another month"):
        assert heading in writeup
