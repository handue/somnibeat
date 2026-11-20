"""Result files must not change between runs: no wall-clock timing, timestamps or checkout-specific commits."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from somnibeat.provenance import TIMING_KEYS, drop_timing

ROOT = Path(__file__).resolve().parents[1]
VOLATILE = TIMING_KEYS | {"generated", "timestamp", "created"}


def keys_anywhere(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(keys_anywhere(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(keys_anywhere(v) for v in obj)) if obj else set()
    return set()


def test_drop_timing_removes_nested_keys_only():
    d = {"seconds": 1.0, "a": [{"fit_seconds": 2, "b": 3}], "c": {"total_seconds": 4, "kappa": 0.5}}
    assert drop_timing(d) == {"a": [{"b": 3}], "c": {"kappa": 0.5}}
    assert d["seconds"] == 1.0  # input not modified


def regenerated_results() -> list[Path]:
    # results/v2/dry_run and selection_staging_smoke.json are historical records that make does not rebuild.
    out = [p for p in sorted((ROOT / "results").rglob("*.json"))
           if "dry_run" not in p.parts and p.name != "selection_staging_smoke.json" and "models" not in p.parts]
    assert out
    return out


@pytest.mark.parametrize("path", regenerated_results(), ids=lambda p: str(p.relative_to(ROOT)))
def test_results_have_no_volatile_fields(path):
    assert not keys_anywhere(json.loads(path.read_text())) & VOLATILE


def test_test_results_record_the_frozen_commit():
    tag = subprocess.run(["git", "rev-parse", "v2-frozen^{commit}"], cwd=ROOT, capture_output=True, text=True)
    if tag.returncode != 0:
        pytest.skip("git tag v2-frozen not available")
    commit = json.loads((ROOT / "results/v2/test_results.json").read_text())["commit"]
    assert commit == tag.stdout.strip()
