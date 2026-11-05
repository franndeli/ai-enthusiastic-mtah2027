"""Learned hand-over detector (W7, D017): where exactly does the ball pass to the other team? Pure.

The Viterbi decoder charges one fixed price for every change of team, so a real hand-over and a
spurious one cost the same, and the external judges show the consequence: with the true team the
pipeline scores 0.91-0.95, and most of the gap is hand-overs it never places. A gradient-boosted
classifier, trained on external matches only (scripts/train_v7.py), reads a local window around
each frame and estimates the log-odds that the ball passes to a given team exactly there. They are
used twice:

    switch cost at frame t = clip(a - b * logit(t), lo, hi)   (only H <-> A; see cost_scale)
    team logit            += w * memory of the last hand-over believed in (see memory)

Team-symmetric by construction: one model scores "to home" and "to away" on features signed towards
the destination team. Reads the rule and PathCRF evidence, ball-to-player geometry and ball motion.
Pure numpy; sklearn only to fit and apply the classifier. Applying a given model is deterministic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import A, H, pathcrf
from possession.features import attack_direction, ball_velocity_change
from possession.io import AWAY, HOME, Match

WINDOWS = (12, 25, 75)
TOLERANCE_FRAMES = 2      # a frame this close to a true hand-over is a positive
NEGATIVE_EVERY = 12       # one easy negative every this many frames
HARD_WINDOW_FRAMES = 50   # frames this close to any true transition are always kept
AGE_CAP_FRAMES = 2000.0
HGB_PARAMS = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=60,
                  l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0)


def win_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of x[t+lo .. t+hi] ignoring NaN; NaN if the window holds no value."""
    n = len(x)
    valid = ~np.isnan(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0), dtype=np.float64)))
    count = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    idx = np.arange(n)
    a, b = np.clip(idx + lo, 0, n), np.clip(idx + hi + 1, 0, n)
    num, den = total[b] - total[a], count[b] - count[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def win_count(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Number of True in x[t+lo .. t+hi]."""
    n = len(x)
    total = np.concatenate(([0], np.cumsum(x.astype(np.int64))))
    idx = np.arange(n)
    return (total[np.clip(idx + hi + 1, 0, n)] - total[np.clip(idx + lo, 0, n)]).astype(np.float32)


def win_min(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Minimum of x[t+lo .. t+hi] ignoring NaN; NaN if the window holds no value. Past the ends of
    the array the nearest edge value stands in."""
    size = hi - lo + 1
    pad = size + max(abs(lo), abs(hi))
    filtered = minimum_filter1d(np.pad(np.where(np.isnan(x), np.inf, x), pad, mode="edge"), size, mode="nearest")
    # the centred filter covers [t - size // 2, t + (size - 1) // 2]: read it where that is the window
    out = filtered[np.arange(len(x)) + pad + lo + size // 2]
    return np.where(np.isinf(out), np.nan, out).astype(np.float32)


def _players_within(match: Match, seen: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) tracked players within radius_m of the ball; NaN where the ball is unseen."""
    delta = (match.players_xy - match.ball_xy[:, None, :]) * np.asarray(match.pitch_m, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        near = np.sum(delta * delta, axis=-1) < radius_m ** 2
    home = np.sum(near[:, match.team_of_slot == HOME], axis=1).astype(np.float32)
    away = np.sum(near[:, match.team_of_slot == AWAY], axis=1).astype(np.float32)
    return np.where(seen, home, np.nan), np.where(seen, away, np.nan)


def feature_matrix(match: Match, feats: dict, sign: float, z_team: np.ndarray, z_dead: np.ndarray,
                   pcrf: dict, params: dict) -> tuple[np.ndarray, tuple]:
    """(n, k) float32 and the column names, for "the ball passes to the team with this sign" (+1
    home, -1 away). pcrf: pathcrf.outputs of this match. Never prints; NaN = not computable."""
    near_new = feats["nearest_home_m"] if sign > 0 else feats["nearest_away_m"]
    near_old = feats["nearest_away_m"] if sign > 0 else feats["nearest_home_m"]
    gap = near_old - near_new            # > 0: the destination team is nearer
    vote = np.nan_to_num(feats["touch_vote"]) * sign
    new_touch, old_touch = vote > 0, vote < 0
    speed = feats["ball_speed_m_s"]
    change = ball_velocity_change(match.ball_xy, match.pitch_m)
    z = np.asarray(z_team, dtype=np.float64) * sign
    shape = sign * sum(params["shape_" + k] * np.nan_to_num(feats["shape_" + k + "_z"])
                       for k in ("len", "width", "speed", "vel"))
    direction = attack_direction(match)
    seen = feats["ball_present"] > 0
    within = {r: _players_within(match, seen, r) for r in (3.0, 6.0)}
    cols: dict[str, np.ndarray] = {
        "near_new": near_new, "near_old": near_old, "gap": gap,
        "vote": vote,
        "owner": np.nan_to_num(feats["owner_sign"]) * sign,
        "owner_age": np.minimum(feats["owner_age"], AGE_CAP_FRAMES),
        "speed": speed, "velocity_change": change,
        "seen": feats["ball_present"],
        "since_ball": np.minimum(feats["frames_since_ball"], AGE_CAP_FRAMES),
        "until_ball": np.minimum(feats["frames_until_ball"], AGE_CAP_FRAMES),
        "control": np.nan_to_num(feats["control_balance"]) * sign,
        "z": z, "z_dead": z_dead, "shape": shape,
        "ball_attack": (feats["ball_x_norm"] - 0.5) * direction * sign,
        "ball_wide": np.abs(feats["ball_y_norm"] - 0.5),
        "players_speed": feats["all_mean_speed_smooth_m_s"],
    }
    for w in WINDOWS:
        cols[f"gap_past{w}"] = win_mean(gap, -w, -1)
        cols[f"gap_fut{w}"] = win_mean(gap, 0, w - 1)
        cols[f"new_touch_fut{w}"] = win_count(new_touch, 0, w - 1)
        cols[f"new_touch_past{w}"] = win_count(new_touch, -w, -1)
        cols[f"old_touch_fut{w}"] = win_count(old_touch, 0, w - 1)
        cols[f"old_touch_past{w}"] = win_count(old_touch, -w, -1)
        cols[f"z_past{w}"] = win_mean(z, -w, -1)
        cols[f"z_fut{w}"] = win_mean(z, 0, w - 1)
        cols[f"seen_around{w}"] = win_mean(feats["ball_present"], -w, w)
        cols[f"speed_past{w}"] = win_mean(speed, -w, -1)
        cols[f"speed_fut{w}"] = win_mean(speed, 0, w - 1)
    cols["near_new_min_fut12"] = win_min(near_new, 0, 11)
    cols["near_old_min_past12"] = win_min(near_old, -12, -1)
    cols["change_max5"] = -win_min(-change, -5, 5)
    cols["shape_past75"] = win_mean(shape, -75, -1)
    cols["shape_fut75"] = win_mean(shape, 0, 74)
    for radius, (home, away) in within.items():
        new, old = (home, away) if sign > 0 else (away, home)
        cols[f"new_within{radius:g}"], cols[f"old_within{radius:g}"] = new, old
    for span in (5, 12):
        for name, near in (("new", near_new), ("old", near_old)):
            before, after = np.full_like(near, np.nan), np.full_like(near, np.nan)
            before[span:], after[:-span] = near[:-span], near[span:]
            cols[f"{name}_closing{span}"] = before - near
            cols[f"{name}_leaving{span}"] = after - near
    last_team = pathcrf.logodds(pcrf["p_home"]) * sign
    next_team = pathcrf.logodds(pcrf["p_next_home"]) * sign
    src, dst = pcrf["src_team"] * sign, pcrf["dst_team"] * sign
    cols.update(pcrf=last_team, pcrf_next=next_team, pcrf_src=src, pcrf_dst=dst,
                pcrf_loop=pcrf["edge_loop"], pcrf_dst_out=pcrf["dst_out"],
                pcrf_prev_out=pcrf["prev_out"], pcrf_next_out=pcrf["next_out"])
    for w in WINDOWS:
        cols[f"pcrf_past{w}"] = win_mean(last_team, -w, -1)
        cols[f"pcrf_fut{w}"] = win_mean(last_team, 0, w - 1)
        cols[f"pcrf_next_past{w}"] = win_mean(next_team, -w, -1)
        cols[f"pcrf_next_fut{w}"] = win_mean(next_team, 0, w - 1)
        cols[f"src_past{w}"] = win_mean(src, -w, -1)
        cols[f"src_fut{w}"] = win_mean(src, 0, w - 1)
    names = tuple(cols)
    X = np.stack([np.asarray(cols[k], dtype=np.float32) for k in names], axis=1)
    X[np.isinf(X)] = np.nan
    return X, names


@dataclass
class HandoverModel:
    classifier: HistGradientBoostingClassifier
    columns: tuple


def logits(model: HandoverModel, match: Match, feats: dict, z_team: np.ndarray, z_dead: np.ndarray,
           pcrf: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(log-odds of a hand-over to home, log-odds of a hand-over to away), one per frame."""
    out = []
    for sign in (1.0, -1.0):
        X, names = feature_matrix(match, feats, sign, z_team, z_dead, pcrf, params)
        if names != tuple(model.columns):
            raise ValueError("hand-over model was trained on a different feature layout")
        out.append(model.classifier.decision_function(X))
    return out[0], out[1]


def memory(to_home: np.ndarray, to_away: np.ndarray, threshold: float, tau_frames: float) -> np.ndarray:
    """Signed memory (+1 home) of the last hand-over believed in, fading with its age: the frames
    after a hand-over belong to the new team even where the other clues have not caught up."""
    vote = np.where((to_home > threshold) & (to_home >= to_away), 1.0,
                    np.where((to_away > threshold) & (to_away > to_home), -1.0, 0.0))
    idx = np.arange(len(vote))
    last = np.maximum.accumulate(np.where(vote != 0, idx, -1))
    owner = np.where(last >= 0, vote[np.maximum(last, 0)], 0.0)
    return owner * np.exp(-np.where(last >= 0, idx - last, 0) / tau_frames)


def cost_scale(to_home: np.ndarray, to_away: np.ndarray, team_switch: float, a: float, b: float,
               lo: float, hi: float) -> np.ndarray:
    """(n, 3, 3) multipliers of the decoder's switch costs: only a team change is priced by the
    classifier, cost(t) = clip(a - b * logit(t), lo, hi); stoppage transitions keep their rule cost."""
    scale = np.ones((len(to_home), 3, 3), dtype=np.float64)
    scale[:, A, H] = np.clip(a - b * to_home, lo, hi) / team_switch
    scale[:, H, A] = np.clip(a - b * to_away, lo, hi) / team_switch
    return scale


# --- training side (external labels only) -------------------------------------------------

def _sampling_seed(key) -> int:
    return int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16) % (2 ** 31)


def training_table(match: Match, feats: dict, codes: np.ndarray, z_team: np.ndarray, z_dead: np.ndarray,
                   pcrf: dict, params: dict, key) -> tuple[np.ndarray, np.ndarray, tuple]:
    """(X, y, columns): alive frames of one labelled copy, once per destination team. Positives are
    frames within TOLERANCE_FRAMES of a true change to that team from the other one; frames near any
    transition are all kept, and a deterministic sample of the rest (key seeds the sampling)."""
    codes = np.asarray(codes)
    n = match.n_frames
    alive = match.graded_mask() & (codes <= 1)
    transitions = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    near_transition = np.zeros(n, dtype=bool)
    for t in transitions:
        near_transition[max(t - HARD_WINDOW_FRAMES, 0):t + HARD_WINDOW_FRAMES + 1] = True
    rng = np.random.RandomState(_sampling_seed(key))
    Xs, ys = [], []
    names = ()
    for sign, code in ((1.0, H), (-1.0, A)):
        X, names = feature_matrix(match, feats, sign, z_team, z_dead, pcrf, params)
        positive = np.zeros(n, dtype=bool)
        for t in transitions:
            if codes[t] == code and codes[t - 1] == 1 - code:
                positive[max(t - TOLERANCE_FRAMES, 0):t + TOLERANCE_FRAMES + 1] = True
        keep = alive & (positive | near_transition | (rng.rand(n) < 1.0 / NEGATIVE_EVERY))
        Xs.append(X[keep])
        ys.append(positive[keep].astype(np.int8))
    return np.concatenate(Xs), np.concatenate(ys), names


def fit(tables: list[tuple]) -> HandoverModel:
    """tables: [(group, X, y, columns)], group = (provider, match id). Each match weighs the same
    inside its provider, each provider the same, and the copies of a match share its weight."""
    providers = sorted({t[0][0] for t in tables})
    matches = {p: {t[0] for t in tables if t[0][0] == p} for p in providers}
    copies = {g: sum(1 for t in tables if t[0] == g) for p in providers for g in matches[p]}
    X = np.concatenate([t[1] for t in tables])
    y = np.concatenate([t[2] for t in tables])
    w = np.concatenate([np.full(len(t[2]), 1.0 / (len(providers) * len(matches[t[0][0]]) * copies[t[0]]
                                                  * max(len(t[2]), 1))) for t in tables])
    w = w * (len(w) / w.sum())
    clf = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y, sample_weight=w)
    return HandoverModel(clf, tuple(tables[0][3]))
