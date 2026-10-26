#!/usr/bin/env python3
"""How well do candidate dead-ball clues separate dead from alive frames on external data, and
what do they look like on the public window? Prints shares only.

Clue: share of a centred window in which the ball was seen moving faster than a threshold
("moving_w"), restricted to frames where the ball was seen at least `min_seen` of the window.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import external, features, io, metrica  # noqa: E402

WINDOWS_S = (1, 3, 5)
THRESHOLDS = (0.05, 0.15, 0.3)


def clue(feats: dict, window_s: int, moving_m_s: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    speed = feats["ball_speed_m_s"]
    with np.errstate(invalid="ignore"):
        moving = (speed > moving_m_s).astype(np.float64)
    seen = (~np.isnan(speed)).astype(np.float64)
    w = window_s * 25
    moving_w = uniform_filter1d(moving, w, mode="constant")
    seen_w = uniform_filter1d(seen, w, mode="constant")
    return moving_w, seen_w


def report(label: str, feats: dict, codes: np.ndarray, graded: np.ndarray) -> None:
    for window_s in WINDOWS_S:
        moving_w, seen_w = clue(feats, window_s)
        for name, mask in (("dead", (codes == 2) & graded), ("alive", (codes < 2) & graded)):
            sel = mask & (seen_w >= 0.5)
            if sel.sum() < 100:
                continue
            frac = moving_w[sel] / seen_w[sel]
            print("{:22} w={}s {:5} n={:6d} ({:3.0%} of {:5})  moving share: p25 {:.2f} p50 {:.2f} "
                  "p75 {:.2f}   below 0.05/0.15/0.30: {:4.0%} {:4.0%} {:4.0%}".format(
                      label, window_s, name, int(sel.sum()), sel.sum() / max(mask.sum(), 1), name,
                      *np.percentile(frac, (25, 50, 75)),
                      *[float(np.mean(frac < t)) for t in THRESHOLDS]))


def main() -> int:
    match, gt = io.load_match(), io.load_public_gt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report("BAR-ATH public", features.compute(match), gt, gt != 3)
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            report(game, features.compute(ext.match), ext.codes, ext.match.graded_mask())
    ext = external.load_external(external.available_matches()[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report(ext.match.source + " noisy", features.compute(external.inject_noise(ext.match, 0)),
               ext.codes, ext.match.graded_mask())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dead_clue_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
