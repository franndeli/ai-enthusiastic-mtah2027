"""Experimental tracker-robust activity head; no production wiring or data I/O.
Half-local player-only observations; train externally with fixed rule-ball logit offset.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from possession import FPS, evidence
from possession.io import Match

SERIES = ("speed", "relative_speed", "slow_share", "fast_share", "coverage")
WINDOWS = ("center1", "center4", "center15", "past3", "next3")
FEATURE_NAMES = tuple(f"{s}_{w}" for s in SERIES for w in WINDOWS)
RECIPES = {"replace": 1.0, "half": 0.5}


def blocks(match: Match) -> list[tuple[int, int]]:
    """Validated half-open blocks; disallow overlap and ambiguous cross-period processing."""
    out, previous = [], -1
    for _, first, last in match.halves:
        if not 0 <= first <= last < match.n_frames or first <= previous:
            raise ValueError("invalid or overlapping half bounds")
        out.append((first, last + 1))
        previous = last
    if not out:
        raise ValueError("no graded halves")
    return out


def displacement_speed(xy_m: np.ndarray, half_span: int, fps: int = FPS) -> np.ndarray:
    """One block, (frames, slots, 2). Missing intermediate samples invalidate displacement."""
    if xy_m.ndim != 3 or xy_m.shape[2] != 2 or half_span < 1 or fps <= 0:
        raise ValueError("invalid displacement input")
    n = len(xy_m)
    result = np.full(xy_m.shape[:2], np.nan, dtype=np.float64)
    if n <= 2 * half_span:
        return result
    seen = np.isfinite(xy_m).all(axis=2)
    sums = np.concatenate([np.zeros((1, xy_m.shape[1]), dtype=np.int64),
                           np.cumsum(seen, axis=0)], axis=0)
    width = 2 * half_span + 1
    valid = sums[width:] - sums[:-width] == width
    speed = np.linalg.norm(xy_m[2 * half_span:] - xy_m[:-2 * half_span], axis=2)
    speed *= fps / (2 * half_span)
    result[half_span:n - half_span] = np.where(valid & (speed <= 12.), speed, np.nan)
    return result


def window_mean(values: np.ndarray, frames: int, side: str = "center") -> np.ndarray:
    """NaN-aware mean, clipped to this block; side windows strictly exclude the present."""
    if frames < 1 or side not in ("center", "past", "next"):
        raise ValueError("invalid window")
    n = len(values)
    idx = np.arange(n)
    if side == "center":
        lo, hi = np.maximum(0, idx - frames), np.minimum(n, idx + frames + 1)
    elif side == "past":
        lo, hi = np.maximum(0, idx - frames), idx
    else:
        lo, hi = idx + 1, np.minimum(n, idx + frames + 1)
    valid = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(valid, values, 0.), dtype=np.float64)]
    counts = np.r_[0, np.cumsum(valid)]
    count = counts[hi] - counts[lo]
    out = np.full(n, np.nan)
    np.divide(sums[hi] - sums[lo], count, out=out, where=count > 0)
    return out


def player_design(match: Match, fps: int = FPS) -> np.ndarray:
    """25 player-only features, NaN when unobserved; independent of teams, ball and labels."""
    if match.players_xy.shape != (match.n_frames, match.n_slots, 2):
        raise ValueError("invalid player array shape")
    result = np.full((match.n_frames, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    scale = np.asarray(match.pitch_m, dtype=np.float64)
    for a, b in blocks(match):
        speed = displacement_speed(match.players_xy[a:b].astype(np.float64) * scale,
                                   max(1, round(.5 * fps)), fps)
        valid = np.isfinite(speed)
        count = valid.sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(speed, axis=1)
        denom = max(.25, float(np.nanmedian(med))) if np.isfinite(med).any() else .25
        slow = np.where(count > 0, np.sum(valid & (speed < 1.5), axis=1) / np.maximum(1, count), np.nan)
        fast = np.where(count > 0, np.sum(valid & (speed > 4.), axis=1) / np.maximum(1, count), np.nan)
        columns = []
        for values in (med, med / denom, slow, fast, count / 22.):
            columns.extend(window_mean(values, round(seconds * fps)) for seconds in (1, 4, 15))
            columns.extend(window_mean(values, 3 * fps, side) for side in ("past", "next"))
        result[a:b] = np.column_stack(columns)
    return result


def activity_logit(feats: dict, params: dict) -> np.ndarray:
    """Exactly the bias + slow-player component of the frozen comparator."""
    p = dict(evidence.DEFAULT_PARAMS, **params)
    speed = feats["all_mean_speed_smooth_m_s"]
    if p["slow_use_lead"]:
        speed = feats.get("all_mean_speed_lead_m_s", speed)
    slow = (p["slow_ref_m_s"] - speed) / p["slow_scale_m_s"]
    return p["dead_bias"] + p["w_slow"] * np.clip(np.where(np.isnan(slow), 0., slow), -2., 2.)


def rule_offset(feats: dict, params: dict) -> np.ndarray:
    p = dict(evidence.DEFAULT_PARAMS, **params)
    return evidence.dead_logit(feats, p) - activity_logit(feats, p)


def compose_scores(feats: dict, params: dict, learned_activity: np.ndarray,
                   fraction: float) -> np.ndarray:
    """Change activity only, preserving H/A odds and the rule-ball offset, caps and dtype."""
    if not 0 <= fraction <= 1:
        raise ValueError("invalid blend fraction")
    original = evidence.scores(feats, params)
    if learned_activity.shape != (len(original),) or not np.isfinite(learned_activity).all():
        raise ValueError("invalid activity logits")
    if fraction == 0:
        return original
    p = dict(evidence.DEFAULT_PARAMS, **params)
    old = activity_logit(feats, p)
    z = rule_offset(feats, p) + (1 - fraction) * old + fraction * learned_activity
    dead = np.clip(expit(z), p["dead_floor"], p["dead_confidence_max"])
    team = evidence.team_logit(feats, p)
    alive = np.log1p(-dead)
    return np.column_stack([alive - np.maximum(0., -team),
                            alive - np.maximum(0., team), np.log(dead)]).astype(np.float32)


@dataclass(frozen=True)
class ActivityModel:
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray

    def design(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("feature shape mismatch")
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.mean, values)
        return np.column_stack([np.ones(len(values)), (filled - self.mean) / self.scale,
                                missing.astype(np.float64)])

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.design(values) @ self.beta


def fit_activity(values: np.ndarray, dead: np.ndarray, offset: np.ndarray,
                 weights: np.ndarray) -> ActivityModel:
    """Weighted offset logistic regression, train-only normalization and fixed L2=0.001."""
    x = np.asarray(values, dtype=np.float64)
    y, offset, w = (np.asarray(v, dtype=np.float64) for v in (dead, offset, weights))
    if x.ndim != 2 or len(x) == 0 or any(v.shape != (len(x),) for v in (y, offset, w)):
        raise ValueError("invalid training shapes")
    if not np.isfinite(offset).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("invalid training weights or offsets")
    if set(np.unique(y)) != {0., 1.}:
        raise ValueError("training requires both dead and alive labels")
    w = w / w.sum()
    seen = np.isfinite(x)
    denom = np.maximum((seen * w[:, None]).sum(axis=0), 1e-12)
    mean = (np.where(seen, x, 0.) * w[:, None]).sum(axis=0) / denom
    variance = (np.where(seen, x - mean, 0.) ** 2 * w[:, None]).sum(axis=0) / denom
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.
    model = ActivityModel(mean, scale, np.zeros(1 + 2 * x.shape[1]))
    design = model.design(x)
    penalty = np.r_[0., np.full(design.shape[1] - 1, .001)]

    def objective(beta):
        z = design @ beta + offset
        loss = np.sum(w * (np.logaddexp(0., z) - y * z)) + .5 * np.sum(penalty * beta ** 2)
        grad = design.T @ (w * (expit(z) - y)) + penalty * beta
        return loss, grad

    result = minimize(objective, model.beta, jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("activity fit did not converge")
    return ActivityModel(mean, scale, result.x)
