"""Re-track a team's fragmented player slots into exactly `k` continuous tracks per block.

The ATD gives tracklets (46 slots that start, stop and swap identity; ~20.5 objects per graded
frame); PathCRF needs the same players on every frame of an episode, with no gaps. Per block
(half) and team, a nearest-neighbour tracker with a constant-velocity prediction assigns each
frame's detections to `k` tracks (Hungarian, gated); a track without a detection coasts, and the
coasting stretches are then replaced by linear interpolation between the observed ends (held flat
at the edges). Extra detections (duplicates) are dropped. Deterministic; uses no labels.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_M = 4.0          # a detection further than this from a track's prediction cannot extend it
REINIT_S = 1.0        # a track coasting longer than this may jump to any free detection


def retrack(players: np.ndarray, team: np.ndarray, side: int, blocks, fps: float,
            k: int = 11) -> np.ndarray:
    """(n, k, 2) float32 tracks for one team; NaN outside the blocks."""
    n = players.shape[0]
    pos = players[:, team == side].astype(np.float64)
    out = np.full((n, k, 2), np.nan)
    reinit = REINIT_S * fps
    for a, b in blocks:
        seg = pos[a:b]
        T = b - a
        obs = np.zeros((T, k), dtype=bool)
        tracks = np.full((T, k, 2), np.nan)
        p = np.full((k, 2), np.nan)
        v = np.zeros((k, 2))
        last = np.full(k, -10 ** 9)
        for t in range(T):
            dets = seg[t][~np.isnan(seg[t, :, 0])]
            pred = p + v
            live = np.flatnonzero(~np.isnan(pred[:, 0]))
            taken = np.zeros(len(dets), dtype=bool)
            if len(dets) and len(live):
                cost = np.linalg.norm(pred[live, None, :] - dets[None, :, :], axis=2)
                gate = np.where((t - last[live])[:, None] > reinit, np.inf, GATE_M)
                cost = np.where(cost > gate, 1e6, cost)
                rows, cols = linear_sum_assignment(cost)
                for i, j in zip(rows, cols):
                    if cost[i, j] >= 1e6:
                        continue
                    tk = live[i]
                    v[tk] = 0.7 * v[tk] + 0.3 * (dets[j] - p[tk]) if t - last[tk] == 1 else 0.0
                    p[tk] = dets[j]
                    last[tk] = t
                    obs[t, tk] = True
                    taken[j] = True
            free = np.flatnonzero(~taken)
            if len(free):
                empty = [i for i in range(k) if np.isnan(p[i, 0])]
                stale = sorted((i for i in range(k) if not np.isnan(p[i, 0]) and not obs[t, i]
                                and t - last[i] > reinit), key=lambda i: last[i])
                for j, i in zip(free, empty + stale):
                    p[i] = dets[j]
                    v[i] = 0.0
                    last[i] = t
                    obs[t, i] = True
            coast = ~obs[t] & ~np.isnan(p[:, 0])
            p[coast] += v[coast]
            v[coast] *= 0.9
            tracks[t] = p
        for i in range(k):
            seen = np.flatnonzero(obs[:, i])
            if not len(seen):
                tracks[:, i] = np.nan
                continue
            for d in (0, 1):
                tracks[:, i, d] = np.interp(np.arange(T), seen, tracks[seen, i, d])
        out[a:b] = tracks
    for a, b in blocks:                     # a track never observed: the team's mean position
        for i in range(k):
            if np.isnan(out[a:b, i, 0]).all():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mean = np.nanmean(pos[a:b], axis=1)
                    fill = np.nanmean(mean, axis=0)
                out[a:b, i] = np.where(np.isnan(mean), fill, mean)
    return out.astype(np.float32)


def retrack_all(players: np.ndarray, team: np.ndarray, blocks, fps: float, k: int = 11):
    """(players (n, 2k, 2), team (2k,)): home tracks first."""
    home = retrack(players, team, 0, blocks, fps, k)
    away = retrack(players, team, 1, blocks, fps, k)
    return np.concatenate([home, away], axis=1), np.array([0] * k + [1] * k, dtype=np.int8)
