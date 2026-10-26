"""Two small logistic regressions, fitted on external labelled games only.

* dead model: P(dead | dead-ball features), with intercept;
* team model: P(home | alive, team features), *without* intercept and with scale-only
  standardisation, so it stays antisymmetric (no home/away prior can be learned).

Fitted by Newton's method with L2. Deterministic; weights are stored as JSON with rounded values
so the pipeline only needs inference at run time.
"""

from __future__ import annotations

import json

import numpy as np

from .features import DEAD_FEATURES, TEAM_FEATURES, matrix


def _fit_logistic(Xm: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1.0,
                  intercept: bool = True, iters: int = 50) -> np.ndarray:
    n, k = Xm.shape
    Z = np.column_stack([np.ones(n), Xm]) if intercept else Xm
    beta = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    if intercept:
        reg[0] = 1e-6
    for _ in range(iters):
        eta = np.clip(Z @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = Z.T @ (w * (y - p)) - reg * beta
        W = w * p * (1 - p)
        hess = (Z * W[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


class Model:
    def __init__(self, params: dict):
        self.p = params

    # -- fitting -------------------------------------------------------------------------------
    @classmethod
    def fit(cls, feature_sets, label_sets, stride: int = 5, l2: float = 1.0,
            team_names=TEAM_FEATURES, dead_names=DEAD_FEATURES,
            l2_team: float | None = None, set_weights=None, team_only=None) -> "Model":
        """feature_sets: list of feature dicts; label_sets: matching label arrays (H/A/D/X).
        `l2` regularises the dead-ball model, `l2_team` (default: same) the team model.
        `set_weights` weights every row of a set; `team_only` marks sets that train only the team
        model (e.g. a dataset whose dead-ball convention differs)."""
        from . import H, A, D
        n_sets = len(label_sets)
        set_weights = list(set_weights) if set_weights is not None else [1.0] * n_sets
        team_only = list(team_only) if team_only is not None else [False] * n_sets
        Xd, yd, wd_set, Xt, yt, wt = [], [], [], [], [], []
        for feats, lab, w, t_only in zip(feature_sets, label_sets, set_weights, team_only):
            rows = np.arange(0, len(lab), stride)
            rows = rows[np.isin(lab[rows], (H, A, D))]
            if not t_only:
                Xd.append(matrix(feats, dead_names)[rows])
                yd.append((lab[rows] == D).astype(np.float64))
                wd_set.append(np.full(len(rows), float(w)))
            alive = rows[np.isin(lab[rows], (H, A))]
            Xt.append(matrix(feats, team_names)[alive])
            yt.append((lab[alive] == H).astype(np.float64))
            wt.append(np.full(len(alive), float(w)))
        Xd, yd, wd_set, Xt, yt, wt = map(np.concatenate, (Xd, yd, wd_set, Xt, yt, wt))
        d_mean = np.nanmean(Xd, axis=0)
        Xd = np.where(np.isnan(Xd), d_mean, Xd)
        d_std = Xd.std(axis=0) + 1e-9
        Xd = (Xd - d_mean) / d_std
        t_scale = np.sqrt(np.mean(Xt ** 2, axis=0)) + 1e-9
        Xt = Xt / t_scale
        # balance classes for the dead model
        p_dead = np.sum(wd_set * yd) / np.sum(wd_set)
        wd = wd_set * np.where(yd > 0, 0.5 / max(p_dead, 1e-9), 0.5 / max(1 - p_dead, 1e-9))
        bd = _fit_logistic(Xd, yd, wd / wd.mean(), l2=l2, intercept=True)
        bt = _fit_logistic(Xt, yt, wt / wt.mean(), l2=l2 if l2_team is None else l2_team,
                           intercept=False)
        params = {
            "dead": {"names": list(dead_names), "mean": d_mean.tolist(), "std": d_std.tolist(),
                     "beta": bd.tolist()},
            "team": {"names": list(team_names), "scale": t_scale.tolist(), "beta": bt.tolist()},
        }
        return cls(_round(params))

    # -- inference -----------------------------------------------------------------------------
    def dead_logit(self, feats: dict) -> np.ndarray:
        q = self.p["dead"]
        Xm = matrix(feats, q["names"])
        mean = np.asarray(q["mean"])
        Xm = np.where(np.isnan(Xm), mean, Xm)
        Xm = (Xm - mean) / np.asarray(q["std"])
        beta = np.asarray(q["beta"])
        return beta[0] + Xm @ beta[1:]

    def team_logit(self, feats: dict) -> np.ndarray:
        q = self.p["team"]
        Xm = np.nan_to_num(matrix(feats, q["names"])) / np.asarray(q["scale"])
        return Xm @ np.asarray(q["beta"])

    def to_json(self, path: str) -> None:
        # fixed encoding and line endings: the same bytes on Windows and Linux
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.p, handle, indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, path: str) -> "Model":
        with open(path, encoding="utf-8") as handle:
            return cls(json.load(handle))


def _round(obj, digits: int = 6):
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, digits) for v in obj]
    if isinstance(obj, float):
        return round(obj, digits)
    return obj
