#!/usr/bin/env python3
"""Aggregate statistics of the touch / ownership signal on the match and on external data.

Compares the raw owner_sign changes per graded minute, how often a change is reverted within
1 s / 2 s / 5 s (ping-pong between the teams), and the ball-visible share, so that differences
between the automatic tracking and the external corrected tracking are visible as numbers only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import external, features, io, metrica  # noqa: E402


def stats(match: io.Match, label: str, params: dict | None = None) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = features.compute(match, None, params)
    graded = match.graded_mask()
    minutes = graded.sum() / 25.0 / 60.0
    owner = feats["owner_sign"]
    idx = np.flatnonzero(graded)
    o = owner[idx]
    change = np.flatnonzero((o[1:] != o[:-1]) & (o[1:] != 0) & (o[:-1] != 0)) + 1
    gaps = np.diff(change) / 25.0
    reverted = {w: float(np.mean(gaps < w)) if len(gaps) else 0.0 for w in (1, 2, 5)}
    speed = feats["ball_speed_m_s"][idx]
    present = feats["ball_present"][idx] > 0
    vote = feats["touch_vote"][idx]
    runs = np.flatnonzero(np.concatenate(([True], vote[1:] != vote[:-1])))
    lengths = np.diff(np.concatenate((runs, [len(vote)])))
    touch_runs = lengths[vote[runs] != 0] / 25.0
    print("{:16} ball {:4.0%}  owner changes/min {:5.2f}  reverted<1s {:4.0%} <2s {:4.0%} <5s {:4.0%}"
          "  touch runs/min {:5.1f} median {:4.2f}s  ball speed p50/p90 {:4.1f}/{:4.1f}  "
          "speed>5 share {:3.0%}".format(
              label, present.mean(), len(change) / minutes, reverted[1], reverted[2], reverted[5],
              len(touch_runs) / minutes, float(np.median(touch_runs)) if len(touch_runs) else 0,
              float(np.nanpercentile(speed, 50)), float(np.nanpercentile(speed, 90)),
              float(np.nanmean(speed > 5))))


def main() -> int:
    stats(io.load_match(), "BAR-ATH (ATD)")
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        stats(ext.match, game)
        stats(external.inject_noise(ext.match, 0, external.NoiseParams(ball_missing_share=0.15)),
              game + " noisy")
    for mid in external.available_matches()[:2]:
        ext = external.load_external(mid)
        stats(ext.match, mid)
        stats(external.inject_noise(ext.match, 0), mid + " noisy")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"touch_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
