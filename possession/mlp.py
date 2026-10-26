"""Deterministic numpy MLP posteriors over H/A/D: a non-linear drop-in for model.Model (PLAN I3).

tanh hidden layers and a softmax output, mean cross-entropy + L2 on the weights, mini-batch Adam
with cosine-decayed steps for a fixed number of epochs. Init and shuffles come from one seeded
RandomState, so the same rows and seed give the same weights byte for byte on one machine.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from possession.model import N_STATES, transition_log_probs

SEED = 2027


@dataclass(frozen=True)
class MLPModel:
    mean: np.ndarray       # (F,) feature standardisation
    scale: np.ndarray      # (F,)
    weights: tuple         # per layer (fan_in, fan_out); the last layer outputs N_STATES logits
    biases: tuple          # per layer (fan_out,)
    log_prior: np.ndarray  # (N_STATES,) training class frequencies
    log_trans: np.ndarray  # (N_STATES, N_STATES) per-frame transition log-probabilities
    emission_scale: float

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        layers = {f"{kind}{i}": a for kind, arrays in (("w", self.weights), ("b", self.biases))
                  for i, a in enumerate(arrays)}
        np.savez(path, mean=self.mean, scale=self.scale, log_prior=self.log_prior,
                 log_trans=self.log_trans, emission_scale=np.float64(self.emission_scale),
                 **layers)

    @classmethod
    def load(cls, path: Path) -> "MLPModel":
        with np.load(path) as z:
            n_layers = sum(name.startswith("w") for name in z.files)
            return cls(z["mean"], z["scale"], tuple(z[f"w{i}"] for i in range(n_layers)),
                       tuple(z[f"b{i}"] for i in range(n_layers)), z["log_prior"],
                       z["log_trans"], float(z["emission_scale"]))

    def with_emission_scale(self, emission_scale: float) -> "MLPModel":
        return replace(self, emission_scale=emission_scale)

    def log_proba(self, values: np.ndarray) -> np.ndarray:
        """(n, N_STATES) log posteriors; the interface decode() needs from any classifier."""
        standard = (values.astype(np.float64) - self.mean) / self.scale
        return _log_softmax(_forward(standard, self.weights, self.biases)[-1])


def _forward(x: np.ndarray, weights, biases) -> list:
    """Hidden activations followed by the output logits."""
    out = [x]
    for w, b in zip(weights[:-1], biases[:-1]):
        out.append(np.tanh(out[-1] @ w + b))
    out.append(out[-1] @ weights[-1] + biases[-1])
    return out


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def fit_mlp(values: np.ndarray, labels: np.ndarray, hidden: tuple = (32,), l2: float = 1e-4,
            epochs: int = 10, batch: int = 512, lr: float = 3e-3, seed: int = SEED) -> tuple:
    """(mean, scale, weights, biases); labels are 0..N_STATES-1."""
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale == 0] = 1.0
    x = (values.astype(np.float64) - mean) / scale
    onehot = np.eye(N_STATES)[labels]
    rng = np.random.RandomState(seed)
    sizes = (x.shape[1], *hidden, N_STATES)
    weights = [rng.normal(0, 1 / np.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros(b) for b in sizes[1:]]
    params = weights + biases
    moment1 = [np.zeros_like(p) for p in params]
    moment2 = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(x)
    total_steps = epochs * -(-n // batch)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            rows = order[start:start + batch]
            acts = _forward(x[rows], weights, biases)
            logits = acts.pop()
            delta = (np.exp(_log_softmax(logits)) - onehot[rows]) / len(rows)
            grad_w, grad_b = [None] * len(weights), [None] * len(weights)
            for layer in reversed(range(len(weights))):
                grad_w[layer] = acts[layer].T @ delta + l2 * weights[layer]
                grad_b[layer] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ weights[layer].T) * (1 - acts[layer] ** 2)
            step += 1
            # Cosine decay to zero: the last epochs settle instead of jittering at full step size.
            rate = lr * 0.5 * (1 + np.cos(np.pi * (step - 1) / total_steps))
            for p, g, m, v in zip(params, grad_w + grad_b, moment1, moment2):
                m *= beta1
                m += (1 - beta1) * g
                v *= beta2
                v += (1 - beta2) * g * g
                p -= rate * (m / (1 - beta1 ** step)) / (np.sqrt(v / (1 - beta2 ** step)) + eps)
    return mean, scale, tuple(weights), tuple(biases)


def fit(values: np.ndarray, labels: np.ndarray, label_runs: list, emission_scale: float,
        **fit_params) -> MLPModel:
    mean, scale, weights, biases = fit_mlp(values, labels, **fit_params)
    prior = np.bincount(labels, minlength=N_STATES) + 1.0
    return MLPModel(mean, scale, weights, biases, np.log(prior / prior.sum()),
                    transition_log_probs(label_runs), emission_scale)


def train_mlp(prepared: list, emission_scale: float, **fit_params) -> MLPModel:
    """Same contract as possession.train.train, so lomo(train_fn=train_mlp) is a paired swap."""
    return fit(np.vstack([p["train_x"] for p in prepared]),
               np.concatenate([p["train_y"] for p in prepared]),
               [r for p in prepared for r in p["runs"]], emission_scale, **fit_params)
