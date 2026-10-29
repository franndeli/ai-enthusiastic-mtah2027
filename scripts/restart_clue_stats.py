#!/usr/bin/env python3
"""How well do candidate *restart* clues separate dead from alive frames? Aggregates only.

Clues (all per frame, smoothed over a 1 s centred window):
  out_player   a slow player (< 1 m/s) standing beyond the pitch lines by > margin (thrower,
               ball retriever) -- needs no ball
  ball_edge    ball seen, slow (< 2 m/s) and within edge_m of a touchline / goal line
  ball_rest    ball seen, nearly still (< 0.5 m/s) and nobody within 2 m
  slow_frac    share of tracked players below 0.5 m/s (existing feature)
Prints P(clue | dead) and P(clue | alive) per judge plus the implied likelihood ratio.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import FPS, external, features, io, metrica  # noqa: E402

OUT_MARGIN_M = 0.5
EDGE_M = 2.0


def clues(match, feats: dict) -> dict:
    length, width = match.pitch_m
    xy, sp = match.players_xy, match.players_speed
    with np.errstate(invalid="ignore"):
        beyond = np.maximum.reduce([
            (-xy[:, :, 0]) * length, (xy[:, :, 0] - 1.0) * length,
            (-xy[:, :, 1]) * width, (xy[:, :, 1] - 1.0) * width])
        out_player = np.any((beyond > OUT_MARGIN_M) & (sp < 1.0), axis=1)
        bx, by = feats["ball_x_norm"], feats["ball_y_norm"]
        edge = np.minimum(np.minimum(bx, 1 - bx) * length, np.minimum(by, 1 - by) * width)
        ball_slow = feats["ball_speed_m_s"] < 2.0
        ball_edge = (edge < EDGE_M) & ball_slow
        nearest = np.fmin(feats["nearest_home_m"], feats["nearest_away_m"])
        ball_rest = (feats["ball_speed_m_s"] < 0.5) & (nearest > 2.0)
        slow_far = (feats["ball_speed_m_s"] < 2.0) & (nearest > 2.5)
        slow_far3 = (feats["ball_speed_m_s"] < 3.0) & (nearest > 3.0)
    out = {}
    for name, raw in (("out_player", out_player), ("ball_edge", ball_edge), ("ball_rest", ball_rest),
                      ("slow_far", slow_far), ("slow_far3", slow_far3)):
        out[name] = uniform_filter1d(np.nan_to_num(raw.astype(np.float64)), FPS, mode="constant") > 0.5
    direction = features.attack_direction(match)
    side = np.where(match.team_of_slot[None, :] == 0, 1.0, -1.0) * direction[:, None]  # +1: attacks +x
    with np.errstate(invalid="ignore"):
        own_half = np.where(side > 0, xy[:, :, 0] < 0.5, xy[:, :, 0] > 0.5)
        tracked = ~np.isnan(xy[:, :, 0])
        share_own = np.where(tracked.sum(1) >= 14, (own_half & tracked).sum(1) / np.maximum(tracked.sum(1), 1), 0.0)
        out["all_own_half"] = (uniform_filter1d(share_own, FPS, mode="constant") >= 0.95) & (direction != 0)
        rest1s = uniform_filter1d(np.nan_to_num((feats["ball_speed_m_s"] < 0.5).astype(np.float64)), FPS, mode="constant")
        out["ball_rest_1s"] = rest1s >= 0.9
    out["slow_frac>0.5"] = np.nan_to_num(feats["frac_players_slow"]) > 0.5
    out["slow_frac>0.7"] = np.nan_to_num(feats["frac_players_slow"]) > 0.7
    return out


def report(label: str, match, codes: np.ndarray, graded: np.ndarray) -> None:
    feats = features.compute(match)
    dead, alive = (codes == 2) & graded, (codes < 2) & graded
    for name, clue in clues(match, feats).items():
        p_dead, p_alive = clue[dead].mean(), clue[alive].mean()
        print("{:18} {:14} P(c|dead) {:5.1%}  P(c|alive) {:5.1%}  LR {:5.1f}   dead share {:4.0%}".format(
            label, name, p_dead, p_alive, p_dead / max(p_alive, 1e-6), dead.mean() / max(graded.mean(), 1e-9)))


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        match, gt = io.load_match(), io.load_public_gt()
        report("BAR-ATH public", match, gt, gt != 3)
        for game in metrica.available_games():
            ext = metrica.load_game(game)
            report(game, ext.match, ext.codes, ext.match.graded_mask())
        for mid in external.available_matches()[:2]:
            ext = external.load_external(mid)
            report("idsse " + mid[-5:], ext.match, ext.codes, ext.match.graded_mask())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"restart_clue_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
