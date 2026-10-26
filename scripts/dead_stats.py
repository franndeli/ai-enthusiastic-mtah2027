#!/usr/bin/env python3
"""Aggregate ball-visibility and touch-vote statistics inside dead versus alive frames.

Target: public-window ground truth (true D) and the full match using the pipeline's own D
prediction as a proxy. External: true labels. Prints shares only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import external, features, io, metrica  # noqa: E402
from run import DEFAULT_CONFIG, label_codes_from_config, load_config, load_tags  # noqa: E402


def report(label: str, feats: dict, codes: np.ndarray, graded: np.ndarray) -> None:
    present = feats["ball_present"] > 0
    vote = feats["touch_vote"] != 0
    moving = feats["ball_moving_frac"] > 0.5
    slow = feats["all_mean_speed_smooth_m_s"]
    for name, mask in (("dead", (codes == 2) & graded), ("alive", (codes < 2) & graded)):
        if not mask.any():
            continue
        print("{:28} {:5} n={:7d} ball seen {:4.0%}  touch vote {:4.0%}  ball moving {:4.0%}  "
              "mean player speed {:4.2f}  speed<1.5 {:4.0%}".format(
                  label, name, int(mask.sum()), present[mask].mean(), vote[mask].mean(),
                  moving[mask].mean(), float(np.nanmean(slow[mask])),
                  float(np.nanmean(slow[mask] < 1.5))))


def main() -> int:
    config = load_config(DEFAULT_CONFIG)
    match, gt = io.load_match(), io.load_public_gt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = features.compute(match, load_tags(config))
        pred = label_codes_from_config(config, match, load_tags(config))
    report("BAR-ATH public (true D)", feats, gt, gt != 3)
    report("BAR-ATH full (predicted D)", feats, pred, match.graded_mask())
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = features.compute(ext.match)
        report(game, feats, ext.codes, ext.match.graded_mask())
    for mid in external.available_matches()[:2]:
        ext = external.load_external(mid)
        noisy = external.inject_noise(ext.match, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = features.compute(noisy)
        report(mid + " noisy", feats, ext.codes, ext.match.graded_mask())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dead_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
