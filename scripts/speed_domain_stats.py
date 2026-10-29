#!/usr/bin/env python3
"""Does "players slow" mean the same thing on the target tracking and on the external judges?

Compares, on dead and alive graded frames, the distribution of per-frame mean player speed
computed three ways: the tracking's own speed field (target only), 1-frame central differences
of positions, and positions `span` frames apart (0.5 s). Percentiles only.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import FPS, external, io, metrica  # noqa: E402


def pos_speed(xy_norm: np.ndarray, pitch_m, span: int) -> np.ndarray:
    xy = xy_norm * np.asarray(pitch_m, dtype=np.float32)
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    delta = xy[2 * span:] - xy[:-2 * span]
    out[span:-span] = np.sqrt(np.sum(delta * delta, axis=-1)) * FPS / (2 * span)
    return out


def describe(label: str, name: str, per_player: np.ndarray, dead: np.ndarray, alive: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(per_player, axis=1)
        slow = np.nanmean(per_player < 0.5, axis=1)
    for state, mask in (("dead", dead), ("alive", alive)):
        m = mean[mask & ~np.isnan(mean)]
        s = slow[mask & ~np.isnan(slow)]
        print("{:16} {:12} {:5}  mean speed p10/p50/p90 {:4.2f}/{:4.2f}/{:4.2f}   "
              "share of players <0.5 m/s p50/p90 {:3.0%}/{:3.0%}".format(
                  label, name, state, *np.percentile(m, (10, 50, 90)), *np.percentile(s, (50, 90))))


def main() -> int:
    match, gt = io.load_match(), io.load_public_gt()
    dead, alive = gt == 2, gt < 2
    describe("BAR-ATH public", "tracker", match.players_speed, dead, alive)
    describe("BAR-ATH public", "pos 1-frame", pos_speed(match.players_xy, match.pitch_m, 1), dead, alive)
    describe("BAR-ATH public", "pos 0.5 s", pos_speed(match.players_xy, match.pitch_m, 6), dead, alive)
    for game in metrica.available_games()[:2]:
        ext = metrica.load_game(game)
        g = ext.match.graded_mask()
        d, a = (ext.codes == 2) & g, (ext.codes < 2) & g
        describe(game, "pos 1-frame", ext.match.players_speed, d, a)
        describe(game, "pos 0.5 s", pos_speed(ext.match.players_xy, ext.match.pitch_m, 6), d, a)
    ext = external.load_external(external.available_matches()[0])
    g = ext.match.graded_mask()
    d, a = (ext.codes == 2) & g, (ext.codes < 2) & g
    describe("idsse", "pos 1-frame", ext.match.players_speed, d, a)
    describe("idsse", "pos 0.5 s", pos_speed(ext.match.players_xy, ext.match.pitch_m, 6), d, a)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"speed_domain_stats failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
