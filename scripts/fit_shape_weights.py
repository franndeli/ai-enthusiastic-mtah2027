#!/usr/bin/env python3
"""Fit the four ball-free shape weights (evidence shape_len/width/speed/vel) on IDSSE only.

    scripts/py scripts/fit_shape_weights.py [--seed 0] [--matches J03WPY ...]

Logistic regression (lbfgs, C=1, no intercept, single thread) of the home-vs-away label on the
z-scored shape features of features.ball_free_shape_features, as the constants were originally
defined. Supervised: alive IDSSE frames with their H/A labels. Self-supervised (comparison):
frames of clear control (|control_balance| >= 0.6) of the same matches, labelled by the rule.
Clean tracking and noisy tracking (inject_noise, --seed) are fitted separately. Prints
coefficients only; nothing from the target match is read. Source of the data: IDSSE open dataset
(external, CC-BY 4.0), loaded through possession.external.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from possession import A, H, external, features  # noqa: E402

KEYS = ("len", "width", "speed", "vel")
CLEAR_CONTROL_FIT = 0.6


def design(match, codes: np.ndarray):
    """(z matrix over KEYS, supervised labels, clear-control labels); -1 marks excluded frames."""
    shape = features.ball_free_shape_features(match)
    z = np.stack([shape["shape_%s_z" % k] for k in KEYS], axis=1).astype(np.float64)
    finite = np.isfinite(z).all(axis=1)
    supervised = np.where(finite & (codes == H), 1, np.where(finite & (codes == A), 0, -1))
    balance = features.compute(match)["control_balance"]
    selfsup = np.where(finite & (balance >= CLEAR_CONTROL_FIT), 1,
                       np.where(finite & (balance <= -CLEAR_CONTROL_FIT), 0, -1))
    return z, supervised, selfsup


def fit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    keep = y >= 0
    model = LogisticRegression(C=1.0, fit_intercept=False, solver="lbfgs", max_iter=1000,
                               random_state=0)
    return model.fit(z[keep], y[keep]).coef_[0]


def report(title: str, per_match: dict, pooled: np.ndarray) -> None:
    print("\n{}".format(title))
    print("{:8} {}".format("match", " ".join("{:>8}".format(k) for k in KEYS)))
    for mid, coef in per_match.items():
        print("{:8} {}".format(mid, " ".join("{:8.3f}".format(c) for c in coef)))
    stack = np.array(list(per_match.values()))
    print("{:8} {}".format("POOLED", " ".join("{:8.3f}".format(c) for c in pooled)))
    print("{:8} {}".format("mean", " ".join("{:8.3f}".format(c) for c in stack.mean(0))))
    print("{:8} {}".format("std", " ".join("{:8.3f}".format(c) for c in stack.std(0))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument("--matches", nargs="*", default=None)
    args = parser.parse_args()
    ids = args.matches or external.available_matches()
    if not ids:
        print("no IDSSE match on disk; run scripts/fetch_idsse.py first")
        return 1
    data = {"clean": {}, "noisy": {}}
    for mid in ids:
        ext = external.load_external(mid)
        data["clean"][mid] = design(ext.match, ext.codes)
        data["noisy"][mid] = design(external.inject_noise(ext.match, args.seed, fps=ext.fps), ext.codes)
    for tracking, per in data.items():
        for label, column in (("supervised (IDSSE H/A labels)", 1), ("self-supervised (clear control)", 2)):
            coefs = {mid: fit(d[0], d[column]) for mid, d in per.items()}
            pooled = fit(np.concatenate([d[0] for d in per.values()]),
                         np.concatenate([d[column] for d in per.values()]))
            report("{} tracking, {}".format(tracking, label), coefs, pooled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
