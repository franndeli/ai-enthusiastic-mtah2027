"""Learned per-frame evidence: a gradient-boosted classifier over the pipeline features, trained on
labelled external matches (IDSSE) with ball/track noise injected. Same contract as evidence.scores.

scores(features, params) -> (n, 3) float32 log-evidence H, A, D, never NaN. Pure numpy apart from
loading the model file. Tags are not used by the classifier (it must work without them); a blend
weight < 1 mixes in the rule evidence, which does use them.

Direction-agnostic by construction: only distances, speeds and ball-availability enter the design
matrix (no absolute x), the model is trained on every match plus its team-swapped copy, and
prediction averages the original and the swapped view, so swapping H/A in the input swaps the
output columns exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from possession import A, D, H, FPS, external, features as feature_module
from possession.evidence import scores as rule_scores  # bound now: evaluation may patch the module
from possession.io import work_dir

DEFAULT_PARAMS = {
    "weight": 1.0,                 # 1 = learned only, 0 = rule evidence only (blend in between)
    "team_confidence_max": 0.75,   # |team logit| <= logit(this), as in evidence.DEFAULT_PARAMS
    "dead_confidence_max": 0.8,    # dead probability clipped to [dead_floor, this]
    "dead_floor": 0.05,
    "model_path": None,            # default: work_dir()/learned/evidence_hgb.joblib
}

# Recipe (fixed; the model hash only reproduces if these and the library versions do).
RANDOM_STATE = 0
HGB_PARAMS = dict(max_iter=200, max_leaf_nodes=31, learning_rate=0.1, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
TRAIN_STRIDE = 5                   # every 5th frame (5 fps): neighbours are near-duplicates
TRAIN_NOISE_SEEDS = (1, 2)         # evaluation uses other seeds
TEMPERATURE_GRID = np.round(np.arange(0.5, 6.01, 0.25), 2)
GAP_CAP_FRAMES = 100 * FPS         # "inf" (never seen) and very long gaps look alike
NEAR_CAP_M = 20.0
WINDOWS_FRAMES = (2 * FPS, 10 * FPS, 30 * FPS)

# Features of features.compute that are used as they are (direction-agnostic), by name.
_NEUTRAL = ("ball_present", "ball_speed_m_s", "ball_out_frac", "frames_since_ball",
            "frames_until_ball", "frames_since_team", "frames_until_team",
            "all_mean_speed_m_s", "all_mean_speed_smooth_m_s", "frac_players_slow")
_TEAM_PAIRS = (("nearest_home_m", "nearest_away_m"), ("home_spread_m", "away_spread_m"),
               ("home_mean_speed_m_s", "away_mean_speed_m_s"),
               ("home_n_tracked", "away_n_tracked"))
_SIGNED = ("control_balance", "last_team_sign", "next_team_sign")   # flip sign when teams swap


@dataclass
class LearnedModel:
    classifier: HistGradientBoostingClassifier
    temperature: float
    log_prior: np.ndarray        # (3,) log class frequencies of the training rows (H, A, D)
    feature_names: tuple


# -- features ----------------------------------------------------------------------------------

def swap_teams(features: dict) -> dict:
    """The same situation seen with home and away exchanged (only the names the model reads)."""
    out = {name: features[name] for name in _NEUTRAL}
    for home, away in _TEAM_PAIRS:
        out[home], out[away] = features[away], features[home]
    for name in _SIGNED:
        out[name] = -features[name]
    return out


def design_matrix(features: dict) -> tuple[np.ndarray, tuple]:
    """(n, F) float32 model input and its column names. NaN = unknown (the classifier handles it)."""
    cols = {name: features[name] for name in _NEUTRAL}
    for name in ("frames_since_ball", "frames_until_ball", "frames_since_team", "frames_until_team"):
        cols[name] = np.minimum(cols[name], GAP_CAP_FRAMES)
    for pair in _TEAM_PAIRS:
        for name in pair:
            cols[name] = features[name]
    for name in _SIGNED:
        cols[name] = features[name]

    near_home, near_away = features["nearest_home_m"], features["nearest_away_m"]
    near_diff = np.clip(near_away, 0, NEAR_CAP_M) - np.clip(near_home, 0, NEAR_CAP_M)
    cols["near_min_m"] = np.minimum(near_home, near_away)
    for size in WINDOWS_FRAMES:
        cols["near_diff_{}f".format(size)] = feature_module._moving_mean(near_diff, size)
    present = features["ball_present"]
    cols["ball_present_frac_10s"] = feature_module._moving_mean(present, 10 * FPS)
    cols["ball_present_frac_60s"] = feature_module._moving_mean(present, 60 * FPS)

    names = tuple(cols)
    return np.stack([np.asarray(cols[name], dtype=np.float32) for name in names], axis=1), names


# -- training ----------------------------------------------------------------------------------

def swap_labels(codes: np.ndarray) -> np.ndarray:
    out = codes.copy()
    out[codes == H], out[codes == A] = A, H
    return out


def training_rows(features: dict, codes: np.ndarray, stride: int = TRAIN_STRIDE):
    """(X, y) from one match's features: labelled frames every `stride`, plus the team-swapped copy."""
    rows = np.flatnonzero(np.isin(codes, (H, A, D)))[::stride]
    x, _ = design_matrix(features)
    x_swapped, _ = design_matrix(swap_teams(features))
    y = codes[rows].astype(np.int8)
    return (np.concatenate([x[rows], x_swapped[rows]]),
            np.concatenate([y, swap_labels(y)]))


def external_rows(ext, seeds: tuple = TRAIN_NOISE_SEEDS):
    """(X, y) of one labelled external match, noise-injected once per seed (no tags)."""
    parts = []
    for seed in seeds:
        noisy = external.inject_noise(ext.match, seed, fps=ext.fps)
        parts.append(training_rows(feature_module.compute(noisy), ext.codes))
    return (np.concatenate([x for x, _ in parts]), np.concatenate([y for _, y in parts]))


def fit_classifier(x: np.ndarray, y: np.ndarray, **overrides) -> HistGradientBoostingClassifier:
    """Single-threaded, fixed-seed fit so the result is reproducible byte for byte."""
    with threadpool_limits(limits=1):
        return HistGradientBoostingClassifier(**dict(HGB_PARAMS, **overrides)).fit(x, y)


def _class_log_prior(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(np.float64)
    return np.log(counts / counts.sum())


def fit_model(rows: dict, **overrides) -> LearnedModel:
    """Model on the stacked rows of {match_id: (x, y)}; temperature 1 until calibrate() sets it."""
    ids = sorted(rows)
    x = np.concatenate([rows[mid][0] for mid in ids])
    y = np.concatenate([rows[mid][1] for mid in ids])
    return LearnedModel(fit_classifier(x, y, **overrides), 1.0, _class_log_prior(y),
                        design_matrix_names())


def design_matrix_names() -> tuple:
    return design_matrix(_probe_features())[1]


def _probe_features() -> dict:
    n = 2 * max(WINDOWS_FRAMES)
    names = set(_NEUTRAL) | {n for pair in _TEAM_PAIRS for n in pair} | set(_SIGNED)
    return {name: np.zeros(n, dtype=np.float32) for name in names}


def leave_one_out_models(rows: dict, **overrides) -> dict:
    """{held_out_id: model trained on the other matches}."""
    return {mid: fit_model({k: v for k, v in rows.items() if k != mid}, **overrides)
            for mid in sorted(rows)}


def calibrate(models: dict, rows: dict) -> float:
    """Temperature minimising the held-out log-loss of each fold model on its own match's rows."""
    best, best_loss = 1.0, np.inf
    logits = {mid: np.log(np.clip(models[mid].classifier.predict_proba(rows[mid][0]), 1e-6, 1.0))
              for mid in models}
    for temperature in TEMPERATURE_GRID:
        loss = 0.0
        for mid, log_p in logits.items():
            scaled = log_p / temperature
            scaled = scaled - np.logaddexp.reduce(scaled, axis=1, keepdims=True)
            loss += -scaled[np.arange(len(scaled)), rows[mid][1]].sum()
        loss /= sum(len(rows[mid][1]) for mid in rows)
        if loss < best_loss:
            best, best_loss = float(temperature), loss
    return best


def fit_with_calibration(rows: dict, **overrides) -> tuple[LearnedModel, dict]:
    """(final model on all matches, {held_out_id: fold model}); every model carries the one
    temperature chosen on the folds' held-out matches."""
    folds = leave_one_out_models(rows, **overrides)
    temperature = calibrate(folds, rows)
    final = fit_model(rows, **overrides)
    for model in (final, *folds.values()):
        model.temperature = temperature
    return final, folds


# -- inference ---------------------------------------------------------------------------------

def class_log_prob(model: LearnedModel, features: dict) -> np.ndarray:
    """(n, 3) calibrated log p(H, A, D | features), symmetrised over the team swap."""
    x, _ = design_matrix(features)
    x_swapped, _ = design_matrix(swap_teams(features))
    prob = model.classifier.predict_proba(x)
    prob_swapped = model.classifier.predict_proba(x_swapped)[:, [1, 0, 2]]
    log_p = np.log(np.clip(0.5 * (prob + prob_swapped), 1e-6, 1.0)) / model.temperature
    return log_p - np.logaddexp.reduce(log_p, axis=1, keepdims=True)


def learned_scores(model: LearnedModel, features: dict, p: dict) -> np.ndarray:
    """Likelihood-style evidence: log p(class|x) minus the log training prior, in the rules' shape.

    The team logit only splits the alive mass between H and A and is capped; the dead
    probability is clipped. Both follow evidence.scores.
    """
    log_q = class_log_prob(model, features)
    cap = np.log(p["team_confidence_max"] / (1.0 - p["team_confidence_max"]))
    z = np.clip(log_q[:, H] - log_q[:, A], -cap, cap)
    prior_dead_logit = model.log_prior[D] - np.log1p(-np.exp(model.log_prior[D]))
    dead_logit = log_q[:, D] - np.logaddexp(log_q[:, H], log_q[:, A]) - prior_dead_logit
    p_dead = np.clip(1.0 / (1.0 + np.exp(-dead_logit)), p["dead_floor"], p["dead_confidence_max"])
    log_alive = np.log(1.0 - p_dead)
    out = np.empty((len(z), 3), dtype=np.float32)
    out[:, H] = log_alive - np.maximum(0.0, -z)
    out[:, A] = log_alive - np.maximum(0.0, z)
    out[:, D] = np.log(p_dead)
    return out


def default_model_path() -> Path:
    return work_dir() / "learned" / "evidence_hgb.joblib"


def save_model(model: LearnedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=0)


@lru_cache(maxsize=2)
def load_model(path: Path) -> LearnedModel:
    model = joblib.load(path)
    if model.feature_names != design_matrix_names():
        raise ValueError("model feature set differs from possession.learned.design_matrix")
    return model


def scores(features: dict, params: dict | None = None, *, rule_params: dict | None = None,
           model: LearnedModel | None = None) -> np.ndarray:
    """(n, 3) float32 log-evidence H, A, D. `params` is the config 'learned' block; `rule_params`
    the 'evidence' block used when weight < 1; `model` overrides the file (evaluation)."""
    p = dict(DEFAULT_PARAMS, **(params or {}))
    weight = float(p["weight"])
    if not 0.0 <= weight <= 1.0:
        raise ValueError("learned.weight must be in [0, 1]")
    if model is None:
        model = load_model(Path(p["model_path"]) if p["model_path"] else default_model_path())
    out = learned_scores(model, features, p)
    if weight < 1.0:
        out = (weight * out + (1.0 - weight) * rule_scores(features, rule_params)
               ).astype(np.float32)
    return out
