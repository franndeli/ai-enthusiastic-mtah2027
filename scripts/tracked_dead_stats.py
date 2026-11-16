#!/usr/bin/env python3
"""What does a stoppage look like while the ball is still tracked? Percentiles of ball-seen
features on dead versus alive graded frames, on the public window and the external judges, plus
likelihood ratios of a few candidate clues. Aggregates only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import FPS, features, io, metrica, skillcorner  # noqa: E402


def window_mean(x: np.ndarray, seconds: float) -> np.ndarray:
    return uniform_filter1d(np.nan_to_num(x.astype(np.float64)), int(seconds * FPS), mode="constant")


def per_frame(match, feats: dict) -> dict:
    ball = match.ball_xy
    dist = features._dist_m(match.players_xy, ball[:, None, :], match.pitch_m)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        n3 = np.nansum(dist < 3.0, axis=1).astype(np.float32)
        n8 = np.nansum(dist < 8.0, axis=1).astype(np.float32)
        nearest = np.fmin(feats["nearest_home_m"], feats["nearest_away_m"])
        speed = feats["ball_speed_m_s"]
        slow_ball_3s = window_mean(speed < 1.5, 3)          # share of 3 s with a slow seen ball
        fast_ball_3s = window_mean(speed > 5.0, 3)
        seen_3s = window_mean(feats["ball_present"], 3)
    return {
        "nearest_m": nearest, "n_within_3m": n3, "n_within_8m": n8, "ball_speed": speed,
        "players_speed": feats["all_mean_speed_smooth_m_s"],
        "slow_ball_3s": np.where(seen_3s > 0, slow_ball_3s / np.maximum(seen_3s, 1e-9), np.nan),
        "fast_ball_3s": np.where(seen_3s > 0, fast_ball_3s / np.maximum(seen_3s, 1e-9), np.nan),
    }


CLUES = {
    "far&slow_ball&slow_players": lambda f: (f["nearest_m"] > 2.0) & (f["ball_speed"] < 2.0) & (f["players_speed"] < 1.6),
    "n3==0&slowball": lambda f: (f["n_within_3m"] == 0) & (f["ball_speed"] < 2.0),
    "n8<=2": lambda f: f["n_within_8m"] <= 2,
    "slow_ball_3s>0.8": lambda f: f["slow_ball_3s"] > 0.8,
    "slow_ball_3s>0.8&nofast": lambda f: (f["slow_ball_3s"] > 0.8) & (f["fast_ball_3s"] < 0.05),
    "slow_ball_3s>0.8&players<1.6": lambda f: (f["slow_ball_3s"] > 0.8) & (f["players_speed"] < 1.6),
}


def report(label: str, match, codes: np.ndarray, graded: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = features.compute(match)
    f = per_frame(match, feats)
    seen = feats["ball_present"] > 0
    dead, alive = (codes == 2) & graded & seen, (codes < 2) & graded & seen
    print("{}: ball seen on {:.0%} of dead and {:.0%} of alive frames".format(
        label, seen[(codes == 2) & graded].mean(), seen[(codes < 2) & graded].mean()))
    for name in ("nearest_m", "n_within_3m", "n_within_8m", "ball_speed", "players_speed", "slow_ball_3s"):
        d, a = f[name][dead], f[name][alive]
        d, a = d[~np.isnan(d)], a[~np.isnan(a)]
        print("   {:14} dead p25/50/75 {:5.1f}/{:5.1f}/{:5.1f}   alive {:5.1f}/{:5.1f}/{:5.1f}".format(
            name, *np.percentile(d, (25, 50, 75)), *np.percentile(a, (25, 50, 75))))
    for name, fn in CLUES.items():
        with np.errstate(invalid="ignore"):
            c = np.nan_to_num(fn(f).astype(np.float64))
        c = uniform_filter1d(c, FPS, mode="constant") > 0.5
        pd, pa = c[dead].mean(), c[alive].mean()
        print("   clue {:30} P(c|dead,seen) {:5.1%}  P(c|alive,seen) {:5.1%}  LR {:5.1f}".format(
            name, pd, pa, pd / max(pa, 1e-6)))


def main() -> int:
    match, gt = io.load_match(), io.load_public_gt()
    report("BAR-ATH public", match, gt, gt != 3)
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        report(game, ext.match, ext.codes, ext.match.graded_mask())
    for mid in skillcorner.available_matches()[:3]:      # real automatic tracking (W7)
        ext = skillcorner.load_match(mid)
        report("SkillCorner " + mid, ext.match, ext.codes, ext.match.graded_mask())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"tracked_dead_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
