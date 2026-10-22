"""Deterministic H/A/D Viterbi with frame-dependent costs and optional duration states.

Costs at index t apply to the transition into frame t; index zero is unused.
"""
from __future__ import annotations

import numpy as np

N_STATES = 3


def switch_matrix(team_switch: float, to_dead: float, from_dead: float) -> np.ndarray:
    """Construct costs indexed [source, destination], with free self transitions."""
    cost = np.array([[0, team_switch, to_dead], [team_switch, 0, to_dead],
                     [from_dead, from_dead, 0]], dtype=np.float64)
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("switch costs must be finite and non-negative")
    return cost


def decode(log_evidence: np.ndarray, switch_cost: np.ndarray,
           initial: np.ndarray | None = None, *, cost_scale: np.ndarray | None = None,
           min_duration_frames: int = 1) -> np.ndarray:
    """Maximise evidence minus transition costs, retaining the legacy initial argument.

    cost_scale is (n,) or (n,3,3). Duration states age up to min_duration_frames,
    then have a constant tail. Only mature states can exit or finish a sequence.
    This is a decoding constraint, never a post-filter of inferred short spells.
    """
    emis = np.asarray(log_evidence, dtype=np.float64)
    if emis.ndim != 2 or emis.shape[1] != N_STATES:
        raise ValueError("log_evidence must be (n, 3)")
    if not np.isfinite(emis).all():
        raise ValueError("log_evidence must be finite")
    cost = np.asarray(switch_cost, dtype=np.float64)
    if cost.shape != (3, 3) or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("switch_cost must be a finite non-negative 3x3 matrix")
    if np.any(np.diag(cost) != 0):
        raise ValueError("switch_cost diagonal must be zero")
    prior = np.zeros(3) if initial is None else np.asarray(initial, dtype=float)
    if prior.shape != (3,) or not np.isfinite(prior).all():
        raise ValueError("initial must be a finite length-three array")
    n = len(emis)
    scale = None if cost_scale is None else np.asarray(cost_scale, dtype=float)
    if scale is not None:
        if scale.shape not in ((n,), (n, 3, 3)):
            raise ValueError("cost_scale must be (n,) or (n,3,3)")
        if not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("cost_scale must be finite and non-negative")
    if not isinstance(min_duration_frames, (int, np.integer)) or not 1 <= min_duration_frames <= 50:
        raise ValueError("min_duration_frames must be an integer from 1 to 50")
    if not n:
        return np.empty(0, dtype=np.int8)
    age_cap = min(min_duration_frames, n)
    if age_cap == 1:
        back = np.empty((n, 3), dtype=np.int8)
        score = emis[0] + prior
        destinations = np.arange(3)
        for t in range(1, n):
            local_cost = cost if scale is None else cost * scale[t]
            candidates = score[:, None] - local_cost
            parents = candidates.argmax(axis=0)
            back[t] = parents
            score = candidates[parents, destinations] + emis[t]
        state = int(score.argmax())
        path = np.empty(n, dtype=np.int8)
        path[-1] = state
        for t in range(n - 1, 0, -1):
            state = int(back[t, state])
            path[t - 1] = state
        return path
    # Sparse age transitions avoid a dense (3*cap)^2 transition matrix.
    back = np.empty((n, 3, age_cap), dtype=np.int16)
    score = np.full((3, age_cap), -np.inf)
    score[:, 0] = emis[0] + prior
    for t in range(1, n):
        local_cost = cost if scale is None else cost * scale[t]
        candidates = score[:, -1, None] - local_cost
        np.fill_diagonal(candidates, -np.inf)
        parents = candidates.argmax(axis=0)
        nxt = np.full_like(score, -np.inf)
        nxt[:, 0] = candidates[parents, np.arange(3)]
        back[t, :, 0] = parents * age_cap + age_cap - 1
        nxt[:, 1:] = score[:, :-1]
        back[t, :, 1:] = np.arange(3)[:, None] * age_cap + np.arange(age_cap - 1)
        stay = score[:, -1] > nxt[:, -1]
        nxt[stay, -1] = score[stay, -1]
        back[t, stay, -1] = np.flatnonzero(stay) * age_cap + age_cap - 1
        score = nxt + emis[t, :, None]
    state = int(score[:, -1].argmax()) * age_cap + age_cap - 1
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        label, age = divmod(state, age_cap)
        path[t] = label
        if t:
            state = int(back[t, label, age])
    return path
