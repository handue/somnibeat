"""HMC loader on one record (skipped when the HMC files are not downloaded).

Checks parsing and label mapping only; HMC is a test set, so no model is run.
"""

from pathlib import Path

import numpy as np
import pytest

from somnibeat import datasets

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "hmc"
RID = "SN001"


def _available() -> bool:
    try:
        return datasets.hmc_record_status(RAW, RID) == "ok"
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="HMC SN001 not downloaded/verified")


def test_scoring_parse_and_clock():
    rows = datasets.parse_hmc_scoring(RAW / "recordings" / f"{RID}_sleepscoring.txt")
    assert rows[0]["onset_s"] == 0.0 and rows[0]["annotation"].startswith("Sleep stage")
    assert datasets.hmc_clock("02.01.01", "00.00.03.430").microsecond == 430000


def test_load_hmc_record():
    rec = datasets.load_hmc(RAW, RID)
    assert rec.dataset == "hmc" and rec.subject_id == RID
    assert rec.fs == 256.0 and rec.meta["ecg_label"] == "ECG"
    assert len(rec.stages) == len(rec.epoch_start_s) > 600
    assert set(rec.stages) <= {"W", "R", "S1", "S2", "S3"}
    assert np.all(np.diff(rec.epoch_start_s) == 30.0) and rec.epoch_start_s[0] == 0.0
    assert rec.epoch_start_s[-1] + 30.0 <= len(rec.ecg) / rec.fs
    raw = rec.meta["raw_stages"]
    m = {"Sleep stage W": "W", "Sleep stage R": "R", "Sleep stage N1": "S1",
         "Sleep stage N2": "S2", "Sleep stage N3": "S3"}
    assert [m[r] for r in raw] == rec.stages
    assert 0 < rec.meta["lights_off_s"] < rec.meta["lights_on_s"] <= len(rec.ecg) / rec.fs
    assert rec.meta["other_annotations"] == []


def test_dispatcher_and_records():
    assert datasets.load(RAW.parent, "hmc", RID, {}).record_id == RID
    listed = datasets.hmc_listed_records(RAW)
    assert listed[0] == "SN001" and len(listed) == len(set(listed))
    cfg = {"datasets": {"cap": {"records": []}}}
    assert "hmc" not in datasets.list_records(RAW.parent, cfg)
    assert datasets.list_records(RAW.parent, cfg, include_hmc=True)["hmc"] == listed
