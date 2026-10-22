"""Turn per-frame features into log-evidence for H, A and D. Pure numpy.

Contract: scores(features, params) -> (n_frames, 3) float32 of log-probabilities, columns
H, A, D (possession.H/A/D). Rows need not be normalised; the decoder adds them to transition
scores. Uninformative frames return equal H and A values, never NaN.

This file holds version 1, hand-written rules. A learned model implements the same signature.

Two independent log-odds are built from clues and then combined:
- team logit (home vs away), meaningful only if the ball is alive: sustained control, plus the
  last / next clear controller where control is unclear (gaps, loose or flying ball);
- dead logit (dead vs alive): players walking, ball beyond the lines, Smart Tagging set-piece window.
Every clue has a weight in nats; a weight of 0 switches it off. Total team strength per frame is
capped, because neighbouring frames are strongly correlated: the decoder, not the evidence, must
accumulate them. Tags only add capped evidence; they never decide a label.
"""
from __future__ import annotations

import numpy as np

from possession import FPS

DEFAULT_PARAMS = {
    # --- team evidence ---
    "w_control": 1.5,              # sustained proximity of a controllable ball (control_balance, -1..1)
    "w_gap": 1.0,                  # last / next clear controller where control is unclear
    "gap_decay_s": 5.0,            # their influence fades with distance in time (seconds)
    "gap_control_below": 0.3,      # "unclear" means |control_balance| below this
    "team_confidence_max": 0.75,   # |team logit| <= logit(this)
    # --- dead-ball evidence ---
    "dead_bias": -1.2,             # log-odds of dead on a frame with no clue
    "w_slow": 1.5,                 # players walking: low smoothed mean speed
    "slow_ref_m_s": 1.5,           # mean player speed (m/s) that is neutral
    "slow_scale_m_s": 0.5,
    "w_out": 1.0,                  # ball beyond the lines
    "w_tag_dead": 0.5,             # inside a tagged set-piece window
    "dead_confidence_max": 0.8,    # dead probability clipped to [dead_floor, this]
    "dead_floor": 0.05,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    return np.log(p / (1.0 - p))


def _zero_nan(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), 0.0, x)


def team_logit(features: dict, p: dict) -> np.ndarray:
    """Log-odds home vs away (>0 home). Flat (0) where nothing speaks."""
    balance = _zero_nan(features["control_balance"])
    z = p["w_control"] * balance

    tau = p["gap_decay_s"] * FPS
    memory = (features["last_team_sign"] * np.exp(-features["frames_since_team"] / tau)
              + features["next_team_sign"] * np.exp(-features["frames_until_team"] / tau))
    z = z + np.where(np.abs(balance) < p["gap_control_below"], p["w_gap"] * memory, 0.0)

    cap = _logit(p["team_confidence_max"])
    return np.clip(z, -cap, cap)


def dead_logit(features: dict, p: dict) -> np.ndarray:
    """Log-odds dead vs alive, from the dead-ball clues."""
    slow = (p["slow_ref_m_s"] - features["all_mean_speed_smooth_m_s"]) / p["slow_scale_m_s"]
    z = p["dead_bias"] + p["w_slow"] * np.clip(_zero_nan(slow), -2, 2)
    z = z + p["w_out"] * _zero_nan(features["ball_out_frac"])
    if "tag_set_piece" in features:
        z = z + p["w_tag_dead"] * features["tag_set_piece"]
    return z


def scores(features: dict, params: dict | None = None) -> np.ndarray:
    """(n, 3) log-evidence H, A, D. The team logit only splits the alive mass between H and A: the
    favoured team keeps log(1 - p_dead), the other pays |logit|, so team confidence never decides
    alive-versus-dead (that is the dead logit's job)."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    z = team_logit(features, p)
    p_dead = np.clip(_sigmoid(dead_logit(features, p)), p["dead_floor"], p["dead_confidence_max"])
    log_alive = np.log(1.0 - p_dead)

    out = np.empty((len(z), 3), dtype=np.float32)
    out[:, 0] = log_alive - np.maximum(0.0, -z)
    out[:, 1] = log_alive - np.maximum(0.0, z)
    out[:, 2] = np.log(p_dead)
    return out
