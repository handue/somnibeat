"""Temporal smoothing of per-epoch class probabilities with a first-order HMM.

Everything here is written as plain loops over numpy arrays so the browser
demo can reproduce it line by line in JavaScript (no library calls beyond
log/exp/max).

Model
- States: the task's classes (0..K-1).
- Transition matrix A[i, j] = P(class_t = j | class_{t-1} = i), estimated
  from training label sequences: only pairs of consecutive scored epochs
  (epoch index difference exactly 1) are counted, plus `alpha` (Laplace
  smoothing) in every cell, then each row is normalized.
- Emissions: scaled likelihoods e_t[k] = p(k | x_t) / prior[k], where prior is
  the class prior the classifier was trained under. Probabilities are
  clipped to [EPS, 1] first.
- Initial distribution: `init` (the training class frequency).
- Sequences are split at gaps (epoch index difference != 1); each segment is
  decoded independently, starting again from `init`.

Tie rule (Viterbi): the lowest class index wins (strict '>' comparison).
"""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def estimate_transitions(label_seqs, epoch_idx_seqs, n_classes: int, alpha: float = 1.0) -> np.ndarray:
    """label_seqs / epoch_idx_seqs: one array per record, in epoch order."""
    counts = np.zeros((n_classes, n_classes))
    for y, ix in zip(label_seqs, epoch_idx_seqs):
        y = np.asarray(y, int)
        ix = np.asarray(ix, int)
        for t in range(1, len(y)):
            if ix[t] - ix[t - 1] == 1:
                counts[y[t - 1], y[t]] += 1.0
    a = counts + alpha
    for i in range(n_classes):
        s = 0.0
        for j in range(n_classes):
            s += a[i, j]
        for j in range(n_classes):
            a[i, j] = a[i, j] / s
    return a


def emissions(proba: np.ndarray, prior: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, float)
    out = np.empty_like(p)
    for t in range(p.shape[0]):
        for k in range(p.shape[1]):
            v = p[t, k]
            if v < EPS:
                v = EPS
            if v > 1.0:
                v = 1.0
            out[t, k] = v / prior[k]
    return out


def viterbi(emis: np.ndarray, trans: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Most likely state path, computed in log space. emis: (T, K) scaled likelihoods."""
    n, k = emis.shape
    path = np.zeros(n, int)
    if n == 0:
        return path
    log_a = [[math.log(trans[i, j]) for j in range(k)] for i in range(k)]
    delta = [math.log(init[j]) + math.log(emis[0, j]) for j in range(k)]
    back = np.zeros((n, k), int)
    for t in range(1, n):
        new = [0.0] * k
        for j in range(k):
            best_i, best_v = 0, delta[0] + log_a[0][j]
            for i in range(1, k):
                v = delta[i] + log_a[i][j]
                if v > best_v:
                    best_i, best_v = i, v
            new[j] = best_v + math.log(emis[t, j])
            back[t, j] = best_i
        delta = new
    best_j, best_v = 0, delta[0]
    for j in range(1, k):
        if delta[j] > best_v:
            best_j, best_v = j, delta[j]
    path[n - 1] = best_j
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def forward_backward(emis: np.ndarray, trans: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Posterior P(state_t | all emissions) with per-step normalization (scaling)."""
    n, k = emis.shape
    post = np.zeros((n, k))
    if n == 0:
        return post
    fwd = np.zeros((n, k))
    s = 0.0
    for j in range(k):
        fwd[0, j] = init[j] * emis[0, j]
        s += fwd[0, j]
    for j in range(k):
        fwd[0, j] /= s
    for t in range(1, n):
        s = 0.0
        for j in range(k):
            acc = 0.0
            for i in range(k):
                acc += fwd[t - 1, i] * trans[i, j]
            fwd[t, j] = acc * emis[t, j]
            s += fwd[t, j]
        for j in range(k):
            fwd[t, j] /= s
    bwd = np.ones(k)
    for t in range(n - 1, -1, -1):
        s = 0.0
        for j in range(k):
            post[t, j] = fwd[t, j] * bwd[j]
            s += post[t, j]
        for j in range(k):
            post[t, j] /= s
        if t > 0:
            nb = [0.0] * k
            s = 0.0
            for i in range(k):
                acc = 0.0
                for j in range(k):
                    acc += trans[i, j] * emis[t, j] * bwd[j]
                nb[i] = acc
                s += acc
            for i in range(k):
                bwd[i] = nb[i] / s
    return post


def segments(epoch_idx: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index ranges of runs of consecutive epochs."""
    ix = np.asarray(epoch_idx, int)
    out, start = [], 0
    for t in range(1, len(ix)):
        if ix[t] - ix[t - 1] != 1:
            out.append((start, t))
            start = t
    if len(ix):
        out.append((start, len(ix)))
    return out


def smooth_record(proba: np.ndarray, epoch_idx: np.ndarray, trans: np.ndarray, prior: np.ndarray,
                  init: np.ndarray, method: str) -> np.ndarray:
    """Returns (T, K): posteriors (forward_backward) or one-hot decoded path (viterbi)."""
    proba = np.asarray(proba, float)
    if method == "none":
        return proba.copy()
    emis = emissions(proba, prior)
    out = np.zeros_like(proba)
    for a, b in segments(epoch_idx):
        if method == "hmm_viterbi":
            p = viterbi(emis[a:b], trans, init)
            for t in range(b - a):
                out[a + t, p[t]] = 1.0
        elif method == "hmm_forward_backward":
            out[a:b] = forward_backward(emis[a:b], trans, init)
        else:
            raise ValueError(f"unknown smoothing method {method!r}")
    return out
