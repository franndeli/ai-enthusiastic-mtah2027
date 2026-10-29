#!/usr/bin/env python3
"""Why are short possessions missed? Per true-possession-duration bucket on the external judges:
share with the ball visible, share containing at least one touch vote of the owning team, share
containing a touch vote of the wrong team, and the decoded recall. Aggregates only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))
import possession_metrics as pm  # noqa: E402
from possession import FPS, features, metrica  # noqa: E402
from idsse_eval import predict  # noqa: E402
from run import DEFAULT_CONFIG, load_config  # noqa: E402

BUCKETS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 1e9))


def spells(codes: np.ndarray, graded: np.ndarray):
    """[(team, first, last)] maximal runs of one team (H=0/A=1) inside graded frames."""
    out, start = [], None
    for i in range(len(codes) + 1):
        c = codes[i] if i < len(codes) and graded[i] else 3
        if start is not None and (c != codes[start]):
            out.append((int(codes[start]), start, i - 1))
            start = None
        if start is None and c in (0, 1):
            start = i
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    fp = dict(config.get("features", {}))
    if args.radius is not None:
        fp["touch_radius_m"] = args.radius
    if args.margin is not None:
        fp["touch_margin_m"] = args.margin
    if args.min_frames is not None:
        fp["touch_min_frames"] = args.min_frames
    config["features"] = fp
    totals = {b: np.zeros(5) for b in BUCKETS}
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        graded = ext.match.graded_mask()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = features.compute(ext.match, params=fp)
            pred = predict(ext.match, config)
        vote, present = feats["touch_vote"], feats["ball_present"] > 0
        pred_spells = spells(pred, graded)
        pred_iv = {t: [(a, b) for tt, a, b in pred_spells if tt == t] for t in (0, 1)}
        for team, a, b in spells(ext.codes, graded):
            dur = (b - a + 1) / FPS
            sign = 1.0 if team == 0 else -1.0
            seg = vote[a:b + 1]
            best = 0.0
            for pa, pb in pred_iv[team]:
                inter = min(b, pb) - max(a, pa) + 1
                if inter > 0:
                    best = max(best, inter / (max(b, pb) - min(a, pa) + 1))
            for lo, hi in BUCKETS:
                if lo <= dur < hi:
                    totals[(lo, hi)] += [1, present[a:b + 1].mean() >= 0.5, np.any(seg == sign),
                                         np.any(seg == -sign), best >= 0.5]
    print("true possessions on the 3 Metrica games (config {}, touch radius {} margin {} min_frames {})".format(
        args.config.name, fp.get("touch_radius_m"), fp.get("touch_margin_m"), fp.get("touch_min_frames")))
    print("{:>8} {:>5} {:>9} {:>10} {:>11} {:>10}".format("dur s", "n", "ball>=50%", "own touch", "wrong touch", "recall@.5"))
    for (lo, hi), t in totals.items():
        n = max(t[0], 1)
        print("{:>8} {:5.0f} {:9.0%} {:10.0%} {:11.0%} {:10.0%}".format(
            "{}-{}".format(lo, "" if hi > 1e8 else hi), t[0], t[1] / n, t[2] / n, t[3] / n, t[4] / n))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"short_spell_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
