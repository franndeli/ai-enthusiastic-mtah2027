"""Offline (non-causal) segmentation with switching costs.

Two stages, each an exact Viterbi:

1. alive vs dead over each graded block, from the dead-ball log-odds;
2. home vs away inside each alive run, from the team log-odds. Every alive run starts free, so the
   team after a restart is decided by the evidence, and a stoppage can never be used as a cheap
   short-cut between H and A.

Boundaries land where the accumulated evidence changes sign: no lag from windows or confirmation.
"""

from __future__ import annotations

import numpy as np

from . import H, A, D, X


def viterbi2(score: np.ndarray, cost: float, cost_up: float | None = None) -> np.ndarray:
    """Binary Viterbi. score[t] = log-odds of state 1 vs state 0 at frame t.

    Maximises sum of +score/2 (state 1) or -score/2 (state 0) minus `cost` per 0->1 switch and
    `cost_up` (defaults to `cost`) per 1->0 switch. Returns an int8 array of states.
    """
    n = len(score)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    c01 = float(cost)
    c10 = float(cost if cost_up is None else cost_up)
    s = np.asarray(score, dtype=np.float64).tolist()
    # With delta = v1 - v0 the recursion collapses to delta_t = clip(delta_{t-1}, -c01, c10) + s_t,
    # and the back-pointers depend only on delta_{t-1}.
    delta = [0.0] * n
    d = s[0]
    delta[0] = d
    for t in range(1, n):
        if d > c10:
            d = c10
        elif d < -c01:
            d = -c01
        d += s[t]
        delta[t] = d
    states = np.zeros(n, dtype=np.int8)
    state = 1 if delta[-1] > 0 else 0
    states[-1] = state
    for t in range(n - 1, 0, -1):
        prev = delta[t - 1]
        if state == 1:
            state = 1 if prev >= -c01 else 0
        else:
            state = 0 if prev <= c10 else 1
        states[t - 1] = state
    return states


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segment(dead_logit: np.ndarray, team_logit: np.ndarray, blocks, n: int,
            cost_dead: float = 20.0, cost_team: float = 20.0, min_dead_s: float = 0.0,
            fps: float = 25.0, step: int = 1) -> np.ndarray:
    """Labels (int8, H/A/D, X outside blocks). `step` > 1 decodes on frame bins (faster)."""
    labels = np.full(n, X, dtype=np.int8)
    for a, b in blocks:
        dl = dead_logit[a:b]
        tl = team_logit[a:b]
        dead = _decode_binned(dl, cost_dead, step)
        if min_dead_s > 0:
            for s, e in _runs(dead == 1):
                if e - s < min_dead_s * fps:
                    dead[s:e] = 0
        lab = np.full(b - a, D, dtype=np.int8)
        for s, e in _runs(dead == 0):
            home = _decode_binned(tl[s:e], cost_team, step)
            lab[s:e] = np.where(home == 1, H, A)
        labels[a:b] = lab
    return labels


def _decode_binned(score: np.ndarray, cost: float, step: int) -> np.ndarray:
    if step <= 1:
        return viterbi2(score, cost)
    n = len(score)
    m = (n + step - 1) // step
    pad = np.zeros(m * step)
    pad[:n] = score
    binned = pad.reshape(m, step).sum(axis=1)
    states = viterbi2(binned, cost)
    return np.repeat(states, step)[:n]
