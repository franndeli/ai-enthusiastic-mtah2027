"""Learned, ball-free correction of the dead-ball log-odds (W6-L recipe X1). Pure numpy/sklearn.

A gradient-boosted classifier, trained on external matches only (Metrica sample + IDSSE, clean
and with target-matched noise; scripts/train_v6.py), reads the rule dead logit and player-only
context (team speeds, spreads, separation, slow and out-of-pitch players, 0.5/1.5/4 s past and
future means) and returns a residual in the units of the rule logit:

    dead_logit_final = dead_logit_v5 + clip(residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)

Nothing derived from the ball enters the model: a learned dead model that reads the ball learns
provider artefacts (IDSSE freezes the ball during stoppages, Metrica loses it) and did not
transfer across providers in W6; this player-only one did (leave-one-provider-out 19/20 cells).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import FPS
from possession.features import attack_direction
from possession.io import Match

RESIDUAL_CLIP = 1.0
WINDOWS = (("w05", FPS // 2), ("w15", 3 * FPS // 2), ("w40", 4 * FPS))
SUBSAMPLE = 10                     # training: every 10th graded frame of a row
HGB_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=400,
                  l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=0)
COLUMNS = ("team_sep_att", "speed_sum", "spread_sum", "z_dead_v5", "all_mean_speed_smooth_m_s",
           "all_mean_speed_lead_m_s", "frac_players_slow", "out_player_frac",
           "z_dead_past_w05", "z_dead_fut_w05", "speed_all_past_w05", "speed_all_fut_w05",
           "z_dead_past_w15", "z_dead_fut_w15", "speed_all_past_w15", "speed_all_fut_w15",
           "z_dead_past_w40", "z_dead_fut_w40", "speed_all_past_w40", "speed_all_fut_w40")


class TrainingRow(NamedTuple):
    """One external labelled copy of a match; `group` (provider, match id) is shared by all
    copies of a match, `seed` is None for the clean copy."""
    group: tuple
    match: Match
    feats: dict
    z_dead: np.ndarray
    codes: np.ndarray
    seed: int | None


def _phase(seed: int | None) -> int:
    """First subsampled frame: copies of a match sample different frames (clean 0, seeds 3, 6...)."""
    return 0 if seed is None else 3 * (seed + 1) % SUBSAMPLE


@dataclass
class DeadResidualModel:
    classifier: HistGradientBoostingClassifier   # on COLUMNS
    slope: float            # Platt map of the rule logit: logit P(dead) ~ slope * z + intercept
    intercept: float
    columns: tuple = COLUMNS


def _roll_mean(x: np.ndarray, w: int, future: bool) -> np.ndarray:
    """Mean over [t-w, t-1] (past) or [t+1, t+w] (future), NaN ignored, NaN if empty."""
    s = pd.Series(x[::-1] if future else x, dtype="float64")
    r = s.rolling(w, min_periods=1).mean().shift(1).to_numpy()
    return (r[::-1] if future else r).astype(np.float32)


def feature_matrix(match: Match, feats: dict, z_dead: np.ndarray) -> np.ndarray:
    """(n, len(COLUMNS)) float32; z_dead is the rule dead logit the residual corrects."""
    with np.errstate(invalid="ignore"):
        sep = (feats["home_centroid_x_norm"] - feats["away_centroid_x_norm"]) * attack_direction(match)
    cols = {
        "team_sep_att": sep,
        "speed_sum": feats["home_mean_speed_m_s"] + feats["away_mean_speed_m_s"],
        "spread_sum": feats["home_spread_m"] + feats["away_spread_m"],
        "z_dead_v5": z_dead,
    }
    for k in ("all_mean_speed_smooth_m_s", "all_mean_speed_lead_m_s", "frac_players_slow",
              "out_player_frac"):
        cols[k] = feats[k]
    for name, w in WINDOWS:
        for key, x in (("z_dead", z_dead), ("speed_all", feats["all_mean_speed_m_s"])):
            cols[f"{key}_past_{name}"] = _roll_mean(x, w, False)
            cols[f"{key}_fut_{name}"] = _roll_mean(x, w, True)
    X = np.stack([np.asarray(cols[c], dtype=np.float32) for c in COLUMNS], axis=1)
    X[np.isinf(X)] = np.nan
    return X


def residual(model: DeadResidualModel, match: Match, feats: dict, z_dead: np.ndarray) -> np.ndarray:
    """(n,) residual in rule-logit units, clipped to +-RESIDUAL_CLIP."""
    if tuple(model.columns) != COLUMNS:
        raise ValueError("dead residual model was trained on a different feature layout")
    g = model.classifier.decision_function(feature_matrix(match, feats, z_dead))
    r = (g - model.intercept) / model.slope - z_dead
    return np.clip(r, -RESIDUAL_CLIP, RESIDUAL_CLIP).astype(np.float32)


def fit(rows: list[TrainingRow], matrices: list[np.ndarray] | None = None) -> DeadResidualModel:
    """Every match weighs equally inside its provider and each provider weighs half; inside a
    match, its copies (clean / noise seeds) share the match's weight. matrices: optional
    feature_matrix of each row, for callers that refit many times on the same rows."""
    from sklearn.linear_model import LogisticRegression
    providers = sorted({r.group[0] for r in rows})
    n_groups = {pv: len({r.group for r in rows if r.group[0] == pv}) for pv in providers}
    n_copies = {g: sum(1 for r in rows if r.group == g) for g in {r.group for r in rows}}
    parts = []
    for k, r in enumerate(rows):
        idx = np.arange(_phase(r.seed), r.match.n_frames, SUBSAMPLE)
        idx = idx[r.match.graded_mask()[idx]]
        X = (feature_matrix(r.match, r.feats, r.z_dead) if matrices is None else matrices[k])[idx]
        c = np.asarray(r.codes)[idx]
        keep = c <= 2
        share = 1.0 / (len(providers) * n_groups[r.group[0]] * n_copies[r.group])
        parts.append((X[keep], (c[keep] == 2).astype(np.int8), np.asarray(r.z_dead)[idx][keep],
                      np.full(keep.sum(), share / max(keep.sum(), 1))))
    X, y, z, w = (np.concatenate(p) for p in zip(*parts))
    w = w * (len(w) / w.sum())
    # Every row is used twice, as in the evaluated W6 recipe (team-swap augmentation, which leaves
    # these team-symmetric columns unchanged). Not neutral for the trees: it halves the effective
    # min_samples_leaf; kept so the shipped model is the evaluated one.
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(np.concatenate([X, X]), np.concatenate([y, y]), sample_weight=np.concatenate([w, w]))
    lr = LogisticRegression(C=1e6, fit_intercept=True, max_iter=1000)
    lr.fit(z[:, None], y, sample_weight=w)
    return DeadResidualModel(clf, float(lr.coef_[0, 0]), float(lr.intercept_[0]))
