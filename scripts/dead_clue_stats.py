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


def travel_m(feats: dict, span_frames: int) -> np.ndarray:
    """Distance (m) between the smoothed ball positions span_frames before and after each frame;
    NaN unless both are seen. Jitter-robust: positions are 0.4 s medians."""
    from scipy.ndimage import median_filter
    xy = np.stack([feats["ball_x_norm"], feats["ball_y_norm"]], axis=1).astype(np.float64)
    seen = ~np.isnan(xy[:, 0])
    filled = np.where(seen[:, None], xy, 0.0)
    smooth = np.stack([median_filter(filled[:, k], 10, mode="nearest") for k in range(2)], axis=1)
    smooth[~seen] = np.nan
    out = np.full(len(xy), np.nan)
    delta = (smooth[2 * span_frames:] - smooth[:-2 * span_frames]) * np.array([105.0, 68.0])
    out[span_frames:-span_frames] = np.sqrt(np.sum(delta * delta, axis=1))
    return out


def report(label: str, feats: dict, codes: np.ndarray, graded: np.ndarray) -> None:
    for span_s in (1, 2):
        travel = travel_m(feats, span_s * 25)
        for name, mask in (("dead", (codes == 2) & graded), ("alive", (codes < 2) & graded)):
            sel = mask & ~np.isnan(travel)
            if sel.sum() < 100:
                continue
            t = travel[sel]
            print("{:22} travel over {}s {:5} n={:6d} ({:3.0%})  p10 {:4.1f} p25 {:4.1f} p50 {:4.1f} m  "
                  "below 1/2/3 m: {:4.0%} {:4.0%} {:4.0%}".format(
                      label, 2 * span_s, name, int(sel.sum()), sel.sum() / max(mask.sum(), 1),
                      *np.percentile(t, (10, 25, 50)), *[float(np.mean(t < m)) for m in (1, 2, 3)]))
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
