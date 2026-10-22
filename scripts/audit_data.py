#!/usr/bin/env python3
"""Audit tracking and public-window hypotheses; print aggregates only, never rows."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from possession import A, D, FPS, H, X, io
from stress_test import missing_runs


def share(mask: np.ndarray, eligible: np.ndarray) -> float:
    """Conditional share, NaN when the denominator is empty."""
    return float(mask[eligible].mean()) if eligible.any() else float("nan")


def quantiles(values: np.ndarray, percentiles: tuple = (50, 90, 100)) -> list[float]:
    finite = np.asarray(values)[np.isfinite(values)]
    return np.percentile(finite, percentiles).tolist() if finite.size else [float("nan")] * len(percentiles)


def nearest_teams(match: io.Match) -> np.ndarray:
    """Closest player by pitch-scaled distance; unknown when ball or all players missing."""
    distance = np.sum(((match.players_xy - match.ball_xy[:, None, :])
                       * np.asarray(match.pitch_m)) ** 2, axis=2)
    distance[~np.isfinite(distance)] = np.inf
    slots = np.argmin(distance, axis=1)
    teams = match.team_of_slot[slots].astype(np.int8)
    teams[~np.isfinite(distance[np.arange(match.n_frames), slots])] = X
    return teams


def main() -> int:
    match, gt = io.load_match(), io.load_public_gt()
    visible = np.isfinite(match.ball_xy).all(axis=1)
    runs = missing_runs(match)
    long_gap = np.zeros(match.n_frames, dtype=bool)
    for first, stop in runs:
        if stop - first > 10 * FPS:
            long_gap[first:stop] = True
    objects = np.isfinite(match.players_xy).all(axis=2).sum(axis=1) + visible
    outside = ((match.ball_xy < 0) | (match.ball_xy > 1)).any(axis=1)
    graded = match.graded_mask()
    print("tracking scope=graded halves; objects=players+ball; speeds=observed player m/s")
    scopes = []
    for half, first, last in match.halves:
        mask = np.zeros(match.n_frames, dtype=bool)
        mask[first:last + 1] = True
        scopes.append((f"half{half}", mask))
    scopes.append(("overall", graded))
    for name, mask in scopes:
        lengths = np.array([stop - first for first, stop in runs if mask[first]], dtype=float) / FPS
        gap_q = quantiles(lengths)
        obj_q = quantiles(objects[mask], (5, 95))
        speed_q = quantiles(match.players_speed[mask], (5, 50, 90, 95, 99, 100))
        print(f"{name}: frames={mask.sum()} ball_visible={share(visible, mask):.6f} "
              f"missing_runs={len(lengths)} gap_s_median/p90/max={gap_q} "
              f"long_gap_frame_share={share(long_gap, mask):.6f} "
              f"objects_mean={objects[mask].mean():.4f} objects_p5/p95={obj_q} "
              f"outside_share_visible={share(outside, mask & visible):.6f} "
              f"speed_m_s_p5/p50/p90/p95/p99/max={speed_q}")

    tags = io.load_smart_tagging()
    set_pieces = [tag for tag in tags if tag["code"] == "set pieces"]
    print(f"smart_tagging_total={len(tags)} counts_per_code={dict(sorted(Counter(t['code'] for t in tags).items()))}")
    print(f"set_piece_counts_per_type={dict(sorted(Counter(t['type'] for t in set_pieces).items()))}")
    public_runs = pm.valid_runs(gt)
    dead = [(first, stop) for first, stop, code in pm.segments(gt, public_runs) if code == D]
    overlaps, offsets = 0, []
    for first, stop in dead:
        matching = [tag for tag in set_pieces
                    if tag["start_frame"] < stop and tag["end_frame"] >= first]
        if matching:
            overlaps += 1
            if stop < len(gt) and gt[stop] in (H, A):
                # One value per D stretch: choose the overlapping tag whose end is closest.
                tag = min(matching, key=lambda t: (abs(t["end_frame"] - stop), t["start_frame"]))
                offsets.append(tag["end_frame"] - stop)
    offset_median = float(np.median(offsets)) if offsets else float("nan")
    print(f"public_D_stretches={len(dead)} overlapping_set_piece={overlaps} "
          f"restart_offset_samples={len(offsets)} median_tag_end_minus_GT_restart_frames={offset_median:g}")

    nearest = nearest_teams(match)
    persistence = np.full(match.n_frames, X, dtype=np.int8)
    for _, first, last in match.halves:
        known = nearest[first:last + 1] != X
        indices = np.maximum.accumulate(np.where(known, np.arange(last - first + 1), -1))
        valid = indices >= 0
        block = persistence[first:last + 1]
        block[valid] = nearest[first:last + 1][indices[valid]]
    public = gt != X
    alive = (gt == H) | (gt == A)
    missing_public = public & ~visible
    transitions = pm.transitions(gt, public_runs)
    transition_missing = sum(not visible[frame] for frame, _, _ in transitions)
    print(f"public_frames={public.sum()} ball_visible_share={share(visible, public):.6f} "
          f"nearest_team_accuracy_visible_HA={share(nearest == gt, public & alive & visible):.6f} "
          f"nearest_team_support={(public & alive & visible).sum()} "
          f"last_seen_accuracy_missing_all={share(persistence == gt, missing_public):.6f} "
          f"last_seen_support={missing_public.sum()} "
          f"last_seen_accuracy_missing_HA={share(persistence == gt, missing_public & alive):.6f} "
          f"last_seen_HA_support={(missing_public & alive).sum()} "
          f"transitions={len(transitions)} transitions_in_missing={transition_missing} "
          f"transition_missing_share={transition_missing / len(transitions) if transitions else float('nan'):.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"audit_data failed: {type(exc).__name__} (details suppressed for data safety)", file=sys.stderr)
        raise SystemExit(1)
