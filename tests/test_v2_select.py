import itertools
import math

import numpy as np
import pandas as pd
import pytest

from somnibeat import smoothing as sm
from somnibeat import v2_models as vm


def brute_force(emis, trans, init):
    """Joint score of every path: returns (best path, posterior marginals)."""
    n, k = emis.shape
    best, best_s = None, -1.0
    post = np.zeros((n, k))
    for path in itertools.product(range(k), repeat=n):
        s = init[path[0]] * emis[0, path[0]]
        for t in range(1, n):
            s *= trans[path[t - 1], path[t]] * emis[t, path[t]]
        if s > best_s:
            best, best_s = path, s
        for t in range(n):
            post[t, path[t]] += s
    return np.array(best), post / post.sum(1, keepdims=True)


def test_viterbi_hand_computed():
    # Two states, sticky transitions. Emissions favor state 1 only weakly at t=1.
    trans = np.array([[0.9, 0.1], [0.1, 0.9]])
    init = np.array([0.5, 0.5])
    emis = np.array([[0.9, 0.1], [0.4, 0.6], [0.9, 0.1]])
    # Path 000: .5*.9 * .9*.4 * .9*.9 = 0.13122
    # Path 010: .5*.9 * .1*.6 * .1*.9 = 0.00243
    # So smoothing keeps the middle epoch in state 0, against its own argmax.
    assert sm.viterbi(emis, trans, init).tolist() == [0, 0, 0]
    # Without stickiness the emission wins.
    flat = np.full((2, 2), 0.5)
    assert sm.viterbi(emis, flat, init).tolist() == [0, 1, 0]


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        k, n = 3, 5
        trans = rng.dirichlet(np.ones(k), size=k)
        init = rng.dirichlet(np.ones(k))
        emis = rng.random((n, k)) + 0.01
        best, _ = brute_force(emis, trans, init)
        assert sm.viterbi(emis, trans, init).tolist() == best.tolist()


def test_forward_backward_sums_to_one_and_matches_brute_force():
    rng = np.random.default_rng(2)
    for k, n in ((2, 6), (3, 5), (4, 4)):
        trans = rng.dirichlet(np.ones(k), size=k)
        init = rng.dirichlet(np.ones(k))
        emis = rng.random((n, k)) * 5 + 0.01
        post = sm.forward_backward(emis, trans, init)
        np.testing.assert_allclose(post.sum(1), 1.0, atol=1e-12)
        _, bf = brute_force(emis, trans, init)
        np.testing.assert_allclose(post, bf, atol=1e-10)


def test_transition_estimation_ignores_gaps():
    y = [np.array([0, 0, 1, 1, 0])]
    ix = [np.array([0, 1, 2, 5, 6])]  # gap between epoch 2 and 5
    a = sm.estimate_transitions(y, ix, 2, alpha=0.0)
    # Counted pairs: 0->0 (0,1), 0->1 (1,2), 1->0 (5,6). The 1->1 across the gap is ignored.
    np.testing.assert_allclose(a, [[0.5, 0.5], [1.0, 0.0]])
    a1 = sm.estimate_transitions(y, ix, 2, alpha=1.0)
    np.testing.assert_allclose(a1, [[2 / 4, 2 / 4], [2 / 3, 1 / 3]])


def test_smooth_record_segments_and_none():
    p = np.array([[0.8, 0.2], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
    trans = np.array([[0.99, 0.01], [0.01, 0.99]])
    prior = np.array([0.5, 0.5])
    np.testing.assert_allclose(sm.smooth_record(p, np.arange(4), trans, prior, prior, "none"), p)
    v = sm.smooth_record(p, np.array([0, 1, 2, 10]), trans, prior, prior, "hmm_viterbi")
    # Last epoch is its own segment, so it keeps its own argmax.
    assert v.argmax(1).tolist() == [0, 0, 0, 1]
    fb = sm.smooth_record(p, np.array([0, 1, 2, 10]), trans, prior, prior, "hmm_forward_backward")
    np.testing.assert_allclose(fb.sum(1), 1.0)
    np.testing.assert_allclose(fb[3], p[3])  # single-epoch segment: posterior = init * emission, normalized


def test_no_test_data_assertion(tmp_path):
    df = pd.DataFrame({"dataset": ["cap", "slpdb", "ucddb", "hmc"], "record": ["a", "b", "c", "d"],
                       "epoch": [0, 0, 0, 0], "stage": ["W", "W", "W", "W"], "x": [1, 2, 3, 4]})
    with pytest.raises(AssertionError):
        vm.assert_training_only(df)
    with pytest.raises(AssertionError):
        vm.assert_training_only(df[df.dataset == "hmc"])
    vm.assert_training_only(df[df.dataset.isin(["cap", "slpdb"])])
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    got = vm.read_training_csv(path, {"cap", "slpdb"}, chunksize=1)
    assert sorted(got.dataset) == ["cap", "slpdb"]
    with pytest.raises(AssertionError):
        vm.read_training_csv(path, {"cap", "ucddb"})
    with pytest.raises(AssertionError):
        vm.Harness(df, ["x"], [], {"seed": 0, "inner_folds": 5, "gb": {}, "lr": {}})


def test_v1_reference_is_in_default_grid():
    grid = vm.default_grid(False)
    assert grid[0] == vm.V1_REFERENCE
    keys = [(s["model"], s["class_weight"], s["smoothing"], s["feature_set"]) for s in grid]
    assert len(keys) == len(set(keys)) == 18
    assert all(s["feature_set"] == "v1" for s in grid)
