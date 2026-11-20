"""slpdb minute apnea label rule (somnibeat/apnea_labels.py)."""

from pathlib import Path

import numpy as np
import pytest

from somnibeat import apnea_labels as al

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "slpdb"


def ev(*epochs):
    """Annotations on the 30 s grid from 0: one aux string per epoch."""
    return al.epoch_events([30.0 * i for i in range(len(epochs))], list(epochs))


def test_parse_aux():
    assert al.parse_aux("2 H X\x00") == ("2", ["H", "X"])
    assert al.parse_aux("W") == ("W", [])
    assert al.parse_aux("MT") == ("MT", [])
    assert al.parse_aux("") == (None, [])


@pytest.mark.parametrize("tok", sorted(al.RESP_TOKENS))
def test_every_respiratory_token_marks_the_minute(tok):
    lab = al.minute_labels(ev("2", f"2 {tok}", "2", "2"), 2)
    assert lab.tolist() == [1, 0]


@pytest.mark.parametrize("tok", ["L", "LA", "LL", "LLL", "LLLL", "A"])
def test_leg_movement_and_arousal_only_do_not_count(tok):
    assert al.minute_labels(ev(f"2 {tok}", f"R {tok} A"), 1).tolist() == [0]


def test_movement_time_epoch_is_scored_not_apnea():
    assert al.minute_labels(ev("MT", "2"), 1).tolist() == [0]


def test_minute_is_epochs_2m_and_2m_plus_1():
    lab = al.minute_labels(ev("W", "W", "2 OA", "2", "2", "2 CAA"), 3)
    assert lab.tolist() == [0, 1, 1]


def test_missing_epoch_unlabeled_unless_apnea_seen():
    # scoring starts at 90 s (epoch 3): minute 0 has no epochs, minute 1 only epoch 3
    e = al.epoch_events([90.0, 120.0, 150.0], ["2 X", "2", "2"])
    assert al.minute_labels(e, 3).tolist() == [-1, 1, 0]
    e = al.epoch_events([90.0, 120.0, 150.0], ["2", "2", "2"])
    assert al.minute_labels(e, 3).tolist() == [-1, -1, 0]
    # minutes beyond the annotations are unlabeled
    assert al.minute_labels(ev("2", "2"), 2).tolist() == [0, -1]


def test_sample_rounding_snaps_to_grid_and_offgrid_raises():
    e = al.epoch_events([0.004, 29.996, 60.0], ["2", "2 H", "2"])
    assert sorted(e) == [0, 1, 2]
    with pytest.raises(ValueError):
        al.epoch_events([15.0], ["2"])


def test_unknown_token_raises():
    with pytest.raises(ValueError):
        ev("2 ZZ")


@pytest.mark.skipif(not (RAW / "slp01a.st").exists(), reason="slpdb not downloaded")
def test_real_record_consistent_with_tokens():
    t, aux, fs, n = al.read_slpdb_st(RAW, "slp01a")
    lab = al.slpdb_minute_labels(RAW, "slp01a")
    assert len(lab) == int(n / fs // 60)
    e = al.epoch_events(t, aux)
    expect = [int(any(x in al.RESP_TOKENS for g in (2 * m, 2 * m + 1) for x in e.get(g, [])))
              for m in range(len(lab))]
    assert lab.tolist() == expect  # slp01a is scored from 0 s to the end
    assert np.all(al.slpdb_minute_labels(RAW, "slp41") == 0)  # no respiratory tokens
