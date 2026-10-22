"""Turn per-frame features into log-evidence for H, A and D. Pure numpy.

Contract: scores(features, params) -> (n_frames, 3) float32 of log-probabilities, columns
H, A, D (possession.H/A/D). Rows need not be normalised; the decoder adds them to transition
scores. Uninformative frames should return (near-)equal values, never NaN.

This file holds version 1, hand-written rules. A learned model implements the same signature.
"""
from __future__ import annotations

import numpy as np

DEFAULT_PARAMS = {
    # team evidence from the nearest-player margin (metres): p(H | alive) = sigmoid(margin / scale)
    "margin_scale_m": 2.0,
    # cap of team evidence strength so one frame never dominates the switch cost
    "team_confidence_max": 0.9,
    # dead-ball evidence: fraction of slow players above which D becomes likely
    "slow_frac_dead": 0.75,
    "dead_confidence_max": 0.8,
    # with the ball missing, team evidence is flat and D evidence only comes from stillness
    "missing_ball_team_weight": 0.0,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def scores(features: dict, params: dict | None = None) -> np.ndarray:
    p = dict(DEFAULT_PARAMS, **(params or {}))
    n = len(features["ball_present"])
    present = features["ball_present"] > 0

    margin = features["nearest_away_m"] - features["nearest_home_m"]  # >0 : home closer
    margin = np.where(np.isnan(margin), 0.0, margin)
    p_home = _sigmoid(margin / p["margin_scale_m"])
    cap = p["team_confidence_max"]
    p_home = np.clip(p_home, 1.0 - cap, cap)
    p_home = np.where(present, p_home, 0.5 + p["missing_ball_team_weight"] * (p_home - 0.5))

    slow = features["frac_players_slow"]
    slow = np.where(np.isnan(slow), 0.0, slow)
    p_dead = _sigmoid((slow - p["slow_frac_dead"]) * 10.0)
    p_dead = np.clip(p_dead, 1.0 - p["dead_confidence_max"], p["dead_confidence_max"])

    out = np.empty((n, 3), dtype=np.float32)
    out[:, 0] = np.log((1.0 - p_dead) * p_home + 1e-6)
    out[:, 1] = np.log((1.0 - p_dead) * (1.0 - p_home) + 1e-6)
    out[:, 2] = np.log(p_dead + 1e-6)
    return out
