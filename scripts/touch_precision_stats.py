#!/usr/bin/env python3
"""Precision of touch votes on the external judges, split by whether the ball's velocity changed
around the vote (a real touch bends or kicks the ball; a ball flying past a player does not).

For each vote onset (first frame of a run of non-zero touch_vote): is the voted team the owner in
the ground truth at that frame (or within the next second, since labels switch at the touch)?
Buckets by the ball velocity change |v(after) - v(before)| over +-span frames. Aggregates only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import FPS, features, metrica  # noqa: E402

BINS = (0.0, 1.0, 2.0, 4.0, 8.0, 1e9)


def velocity(ball_norm: np.ndarray, pitch_m, span: int) -> np.ndarray:
    """Mean velocity (m/s, 2-vector) over the `span` frames ending at t (NaN if any missing)."""
    xy = ball_norm * np.asarray(pitch_m, dtype=np.float32)
    v = np.full_like(xy, np.nan)
    v[span:] = (xy[span:] - xy[:-span]) * FPS / span
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--min-frames", type=int, default=5)
    args = parser.parse_args()
    span = args.span
    counts = {b: np.zeros(3) for b in zip(BINS[:-1], BINS[1:])}  # n, correct now, correct within 1 s
    n_votes = n_true_changes = 0
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        m, codes, graded = ext.match, ext.codes, ext.match.graded_mask()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            feats = features.compute(m, params={"touch_radius_m": args.radius, "touch_margin_m": args.margin,
                                                 "touch_min_frames": args.min_frames})
        vote = feats["touch_vote"]
        v_before = velocity(m.ball_xy, m.pitch_m, span)             # ends at t
        v_after = np.full_like(v_before, np.nan)
        v_after[:-span] = v_before[span:]                            # starts at t
        dv = np.sqrt(np.sum((v_after - v_before) ** 2, axis=1))
        onsets = np.flatnonzero((vote != 0) & np.concatenate(([True], vote[1:] != vote[:-1])))
        team_code = np.where(vote > 0, 0, 1)
        n_true_changes += int(np.sum((codes[1:] != codes[:-1]) & (codes[1:] < 2) & graded[1:]))
        for t in onsets:
            if not graded[t]:
                continue
            # the velocity change nearest the onset, within +-span frames
            lo, hi = max(t - span, 0), min(t + span + 1, len(dv))
            local = dv[lo:hi]
            change = np.nanmax(local) if np.any(~np.isnan(local)) else np.nan
            if np.isnan(change):
                continue
            n_votes += 1
            now = codes[t] == team_code[t]
            soon = np.any(codes[t:t + FPS] == team_code[t])
            for (a, b) in counts:
                if a <= change < b:
                    counts[(a, b)] += [1, now, soon]
    print("touch votes (radius {} margin {} min_frames {}) by ball velocity change over +-{} frames; "
          "{} votes, {} true team changes".format(args.radius, args.margin, args.min_frames, span,
                                                  n_votes, n_true_changes))
    print("{:>12} {:>6} {:>7} {:>12} {:>12}".format("|dv| m/s", "n", "share", "precision", "prec(1 s)"))
    for (a, b), c in counts.items():
        n = max(c[0], 1)
        print("{:>12} {:6.0f} {:7.0%} {:12.0%} {:12.0%}".format(
            "{}-{}".format(a, "" if b > 1e8 else b), c[0], c[0] / max(n_votes, 1), c[1] / n, c[2] / n))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"touch_precision_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
