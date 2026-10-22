"""Offline Viterbi decoding of H/A/D from per-frame log-evidence. Pure numpy.

decode(log_evidence, switch_cost) returns int8 codes (H=0, A=1, D=2). Each half is decoded
separately by the caller (structure.py); this module knows nothing about frames or halves.

Transition model: staying costs 0; moving between states costs switch_cost[from, to] (>= 0, in
log-probability units). A 3x3 matrix so that H<->A changes, entering D and leaving D can be priced
separately. Vectorised over states; the loop over frames is unavoidable for Viterbi but cheap
(3 states x 200k frames).
"""
from __future__ import annotations

import numpy as np

N_STATES = 3


def switch_matrix(team_switch: float, to_dead: float, from_dead: float) -> np.ndarray:
    """3x3 cost matrix from three scalars. Diagonal is 0."""
    cost = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    cost[0, 1] = cost[1, 0] = team_switch
    cost[0, 2] = cost[1, 2] = to_dead
    cost[2, 0] = cost[2, 1] = from_dead
    return cost


def decode(log_evidence: np.ndarray, switch_cost: np.ndarray,
           initial: np.ndarray | None = None) -> np.ndarray:
    """Most probable state path. log_evidence (n, 3); switch_cost (3, 3) non-negative."""
    emis = np.asarray(log_evidence, dtype=np.float64)
    if emis.ndim != 2 or emis.shape[1] != N_STATES:
        raise ValueError("log_evidence must be (n, 3), got {}".format(emis.shape))
    if np.isnan(emis).any():
        raise ValueError("log_evidence contains NaN")
    cost = np.asarray(switch_cost, dtype=np.float64)
    if cost.shape != (N_STATES, N_STATES) or (cost < 0).any():
        raise ValueError("switch_cost must be a non-negative 3x3 matrix")
    n = emis.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int8)

    trans = -cost  # log-score of moving from row to column
    back = np.zeros((n, N_STATES), dtype=np.int8)
    score = emis[0] + (np.zeros(N_STATES) if initial is None else np.asarray(initial))
    for t in range(1, n):
        cand = score[:, None] + trans  # (from, to)
        best_from = np.argmax(cand, axis=0)
        back[t] = best_from
        score = cand[best_from, np.arange(N_STATES)] + emis[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
