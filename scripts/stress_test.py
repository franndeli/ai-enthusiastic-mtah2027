#!/usr/bin/env python3
"""Evaluate deterministic missing-ball stress masks; print aggregate metrics only."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from possession import FPS, X, io
from run import DEFAULT_CONFIG, label_codes_from_config, load_config, load_tags


def missing_runs(match: io.Match) -> list[tuple[int, int]]:
    """Missing finite-XY ball runs, split at graded half boundaries (stop exclusive)."""
    missing = ~np.isfinite(match.ball_xy).all(axis=1)
    runs = []
    for _, first, last in match.halves:
        edges = np.diff(np.r_[False, missing[first:last + 1], False].astype(np.int8))
        runs.extend(zip((np.flatnonzero(edges == 1) + first).tolist(),
                        (np.flatnonzero(edges == -1) + first).tolist()))
    return runs


def transplant_mask(match: io.Match, gt: np.ndarray, seed: int,
                    target_missing: float = 0.41) -> np.ndarray:
    """Transplant empirical gaps into public blocks; trim final draw to target coverage.

    Native missingness is retained. If it already exceeds the target, no ball is hidden.
    Gaps never cross public-block boundaries; RNG and final trimming are deterministic.
    """
    if not 0 <= target_missing <= 1:
        raise ValueError("target_missing must be between zero and one")
    public = gt != X
    visible = np.isfinite(match.ball_xy).all(axis=1)
    mask = np.zeros(match.n_frames, dtype=bool)
    needed = max(0, int(round(public.sum() * target_missing)) - int((public & ~visible).sum()))
    if not needed:
        return mask
    lengths = np.array([stop - start for start, stop in missing_runs(match)], dtype=int)
    if not len(lengths):
        raise ValueError("transplant requires at least one observed missing-ball run")
    blocks = pm.valid_runs(gt)
    frames = np.flatnonzero(public)
    rng = np.random.RandomState(seed)
    # Every accepted draw hides at least one new visible frame, guaranteeing progress.
    while needed:
        available = frames[visible[frames] & ~mask[frames]]
        if not len(available):
            raise ValueError("insufficient visible public frames for transplant target")
        start = int(available[rng.randint(len(available))])
        length = int(lengths[rng.randint(len(lengths))])
        stop_block = next(stop for first, stop in blocks if first <= start < stop)
        stop = min(start + length, stop_block)
        added = np.flatnonzero(visible[start:stop] & ~mask[start:stop]) + start
        if len(added) > needed:
            stop = int(added[needed - 1]) + 1
            added = added[:needed]
        mask[start:stop] = True
        needed -= len(added)
    return mask


def transition_mask(gt: np.ndarray, seconds: float) -> np.ndarray:
    """Union of centred gaps, clipped to public blocks, using public transitions only."""
    mask = np.zeros(len(gt), dtype=bool)
    width = int(round(seconds * FPS))
    if width <= 0:
        raise ValueError("transition gap duration must be positive")
    for first, stop in pm.valid_runs(gt):
        transitions = np.flatnonzero(gt[first + 1:stop] != gt[first:stop - 1]) + first + 1
        for frame in transitions:
            start = int(frame) - width // 2
            mask[max(first, start):min(stop, start + width)] = True
    return mask


def run_stress(config: dict, match: io.Match, gt: np.ndarray,
               seeds: int = 5, durations: tuple[float, ...] = (5, 10, 20),
               native_codes: np.ndarray | None = None) -> list[dict]:
    """Recompute the full pipeline on copied ball arrays; return aggregate rows only."""
    tags = load_tags(config)
    if seeds < 1:
        raise ValueError("seeds must be positive")
    masks = [("native", [np.zeros(match.n_frames, dtype=bool)])]
    masks.append(("transplant", [transplant_mask(match, gt, seed) for seed in range(seeds)]))
    masks.extend((f"transition-centred({seconds:g}s)", [transition_mask(gt, seconds)])
                 for seconds in durations)
    public = gt != X
    rows = []
    for name, variants in masks:
        results, missing = [], []
        for mask in variants:
            if name == "native" and native_codes is not None:
                codes = native_codes
            else:
                ball_xy, ball_speed = match.ball_xy.copy(), match.ball_speed.copy()
                ball_xy[mask], ball_speed[mask] = np.nan, np.nan
                masked = replace(match, ball_xy=ball_xy, ball_speed=ball_speed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    codes = label_codes_from_config(config, masked, tags)
            results.append(pm.evaluate(gt, codes, bootstrap_n=0))
            missing.append(float(np.mean((mask | ~np.isfinite(match.ball_xy).all(axis=1))[public])))
        spells = [result["spell_f1_mean"] for result in results]
        rows.append({"mask": name, "spell_mean": float(np.mean(spells)),
                     "spell_min": float(np.min(spells)),
                     "macro_f1": float(np.mean([r["macro_f1"] for r in results])),
                     "possessions": float(np.mean([r["n_pred_spells"] for r in results])),
                     "missing_share": float(np.mean(missing)), "seeds": len(variants)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    started = time.monotonic()
    rows = run_stress(load_config(args.config), io.load_match(), io.load_public_gt(), args.seeds)
    print("mask | spellF1 mean | spellF1 min | macroF1 mean | possessions mean | missing share | seeds")
    for row in rows:
        print("{mask} | {spell_mean:.4f} | {spell_min:.4f} | {macro_f1:.4f} | "
              "{possessions:.1f} | {missing_share:.4f} | {seeds}".format(**row))
    print(f"elapsed_s={time.monotonic() - started:.2f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"stress_test failed: {type(exc).__name__} (details suppressed for data safety)", file=sys.stderr)
        raise SystemExit(1)
