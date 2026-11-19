import numpy as np
import pytest

from somnibeat import evaluate as ev


def test_kappa_perfect_and_chance():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert ev.cohen_kappa(y, y, labels=[0, 1]) == pytest.approx(1.0)
    assert ev.cohen_kappa(y, np.zeros_like(y), labels=[0, 1]) == pytest.approx(0.0)


def test_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(0)
    a = rng.integers(0, 3, 500)
    b = np.where(rng.random(500) < 0.6, a, rng.integers(0, 3, 500))
    assert ev.cohen_kappa(a, b, labels=[0, 1, 2]) == pytest.approx(cohen_kappa_score(a, b))


def test_awakenings_rules():
    # leading/trailing wake are not awakenings; a single wake epoch is too short
    w = np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    assert ev.awakenings(w) == 2
    assert ev.awakenings(np.ones(5)) == 0


def test_night_metrics():
    m = ev.night_metrics(np.array([1, 0, 0, 0]))
    assert m["tib_min"] == 2.0 and m["tst_min"] == 1.5 and m["se_pct"] == 75.0


def test_bootstrap_ci_contains_point():
    ref = {f"r{i}": np.array([1, 0, 0, 0, 1] * 20) for i in range(10)}
    pred = {k: np.roll(v, i % 3) for i, (k, v) in enumerate(ref.items())}
    out = ev.binary_suite(ref, pred, n_resamples=200, seed=1, ci=0.95)
    for m in out["metrics"].values():
        assert m["ci_low"] - 1e-9 <= m["point"] <= m["ci_high"] + 1e-9
    assert out["n_records"] == 10


def test_awakenings_do_not_merge_across_dropped_epochs():
    # one wake epoch, a dropped epoch (index 3 missing), one wake epoch: not a 2-epoch awakening
    w = np.array([0, 0, 1, 1, 0])
    assert ev.awakenings(w) == 1
    assert ev.awakenings(w, epoch_idx=np.array([0, 1, 2, 4, 5])) == 0


def test_kappa_undefined_for_single_class():
    import math
    assert math.isnan(ev.cohen_kappa(np.zeros(10), np.zeros(10), labels=[0, 1]))


def test_group_bootstrap_keeps_subject_records_together():
    seen = []

    def stat(ids):
        seen.append(sorted(ids))
        return {"n": float(len(ids))}

    groups = {"a1": "a", "a2": "a", "b": "b"}
    ev.bootstrap(["a1", "a2", "b"], stat, n_resamples=50, seed=0, groups=groups)
    for ids in seen[1:]:
        assert ids.count("a1") == ids.count("a2")


def test_count_suite_errors():
    ref = {"r1": np.array([1, 1, 0, 0]), "r2": np.array([0, 0, 0, 1])}
    pred = {"r1": np.array([1, 0, 0, 0]), "r2": np.array([1, 1, 0, 1])}
    out = ev.count_suite(ref, pred, 1.0, n_resamples=20, seed=0, ci=0.95)
    assert out["bland_altman"]["apnea_min"]["per_record_error"] == [-1.0, 2.0]
