"""Softmax regression posteriors + Viterbi decoding over H/A/D (ROUTE_A_PLAN.md decision 4).

numpy only and deterministic: Newton's method has no random init and no minibatches, and Viterbi
breaks ties by lowest state index. Hybrid-HMM decoding: emissions are posterior / prior, scaled by
`emission_scale` because neighbouring frames share feature windows and are not independent
evidence; the scale is the one knob that trades responsiveness against fragmentation.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from possession.frames import D, X, period_runs

N_STATES = 3  # H, A, D


@dataclass(frozen=True)
class Model:
    mean: np.ndarray       # (F,) feature standardisation
    scale: np.ndarray      # (F,)
    weights: np.ndarray    # (F + 1, N_STATES), last row is the bias, last column is fixed at 0
    log_prior: np.ndarray  # (N_STATES,) training class frequencies
    log_trans: np.ndarray  # (N_STATES, N_STATES) per-frame transition log-probabilities
    emission_scale: float

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, weights=self.weights,
                 log_prior=self.log_prior, log_trans=self.log_trans,
                 emission_scale=np.float64(self.emission_scale))

    @classmethod
    def load(cls, path: Path) -> "Model":
        with np.load(path) as z:
            return cls(z["mean"], z["scale"], z["weights"], z["log_prior"], z["log_trans"],
                       float(z["emission_scale"]))

    def with_emission_scale(self, emission_scale: float) -> "Model":
        return replace(self, emission_scale=emission_scale)


def _design(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    standard = (values.astype(np.float64) - mean) / scale
    return np.hstack([standard, np.ones((len(values), 1))])


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_softmax(values: np.ndarray, labels: np.ndarray, l2: float = 1.0,
                max_iter: int = 50, tol: float = 1e-8) -> tuple:
    """(mean, scale, weights) by Newton's method; the last class is the reference (weights 0).

    Labels are 0..N_STATES-1. L2 on the weights, not the bias.
    """
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    z = _design(values, mean, scale)
    n, p = z.shape
    k = N_STATES - 1
    onehot = np.eye(N_STATES)[labels][:, :k]
    penalty = np.full(p, l2)
    penalty[-1] = 0.0
    penalty = np.tile(penalty, k)  # parameters stacked class-major: [class0 (p), class1 (p)]

    w = np.zeros((p, k))

    def loss(w):
        logits = np.hstack([z @ w, np.zeros((n, 1))])
        logits -= logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(logits).sum(axis=1))
        picked = logits[np.arange(n), labels]
        return float((log_norm - picked).sum() + 0.5 * (penalty * w.T.ravel() ** 2).sum())

    current = loss(w)
    for _ in range(max_iter):
        prob = _softmax(np.hstack([z @ w, np.zeros((n, 1))]))[:, :k]
        grad = (z.T @ (prob - onehot)).T.ravel() + penalty * w.T.ravel()
        hess = np.empty((k * p, k * p))
        for c in range(k):
            for d in range(k):
                weight = prob[:, c] * ((c == d) - prob[:, d])
                hess[c * p:(c + 1) * p, d * p:(d + 1) * p] = (z * weight[:, None]).T @ z
        hess += np.diag(penalty)
        step = np.linalg.solve(hess + 1e-9 * np.eye(k * p), grad).reshape(k, p).T
        t = 1.0
        candidate = loss(w - step)
        while candidate > current and t > 1e-4:  # backtracking keeps Newton monotone
            t *= 0.5
            candidate = loss(w - t * step)
        if candidate > current:
            break
        w -= t * step
        improvement = current - candidate
        current = candidate
        if improvement < tol * max(1.0, abs(current)):
            break
    return mean, scale, np.hstack([w, np.zeros((p, 1))])


def transition_log_probs(label_runs: list, smoothing: float = 1.0) -> np.ndarray:
    """Per-frame H/A/D transition log-probabilities counted over label sequences (Laplace)."""
    counts = np.full((N_STATES, N_STATES), smoothing)
    for labels in label_runs:
        labels = labels[labels != X]
        np.add.at(counts, (labels[:-1], labels[1:]), 1)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def fit(values: np.ndarray, labels: np.ndarray, label_runs: list,
        emission_scale: float, l2: float = 1.0) -> Model:
    mean, scale, weights = fit_softmax(values, labels, l2)
    prior = np.bincount(labels, minlength=N_STATES) + 1.0
    return Model(mean, scale, weights, np.log(prior / prior.sum()),
                 transition_log_probs(label_runs), emission_scale)


def predict_log_proba(model: Model, values: np.ndarray) -> np.ndarray:
    logits = _design(values, model.mean, model.scale) @ model.weights
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def viterbi(log_emission: np.ndarray, log_trans: np.ndarray) -> np.ndarray:
    """Most likely state path; uniform start; ties go to the lowest state index."""
    n, c = log_emission.shape
    back = np.empty((n, c), dtype=np.int8)
    score = log_emission[0].copy()
    states = np.arange(c)
    for t in range(1, n):
        candidates = score[:, None] + log_trans
        best = candidates.argmax(axis=0)
        back[t] = best
        score = candidates[best, states] + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(model: Model, values: np.ndarray, period: np.ndarray,
           extra_log_evidence: np.ndarray | None = None) -> np.ndarray:
    """H/A/D codes per frame, each period decoded on its own; X outside play.

    extra_log_evidence (n, N_STATES) is added to the scaled emissions, for sources the model was
    not trained on (e.g. Smart Tagging); it carries its own weight.
    """
    emission = model.emission_scale * (predict_log_proba(model, values) - model.log_prior)
    if extra_log_evidence is not None:
        emission = emission + extra_log_evidence
    out = np.full(len(values), X, dtype=np.int8)
    for a, b in period_runs(period):
        out[a:b] = viterbi(emission[a:b], model.log_trans)
    return out


def argmax_decode(model: Model, values: np.ndarray, period: np.ndarray) -> np.ndarray:
    """Per-frame argmax, no temporal structure: the ablation Viterbi has to beat."""
    out = predict_log_proba(model, values).argmax(axis=1).astype(np.int8)
    out[period == 0] = X
    return out


assert D == N_STATES - 1  # the reference class of fit_softmax is D
