#!/usr/bin/env python3
"""How ball gaps relate to dead ball: the pipeline's dead share by missing-ball run length, on
the competition match (predictions only, label-free) and on the external judges with native gaps
(SkillCorner, Metrica sample), where the true dead share is shown next to it. The dead call is
the submitted one up to the team-side stages: PathCRF and the verifier are left out (they act
on the team evidence and cost minutes per match).

    scripts/py scripts/gap_dead_stats.py [--skillcorner 10]

Prints aggregate shares only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from possession import FPS, features, io, metrica, skillcorner, stack  # noqa: E402
from run import (DEFAULT_CONFIG, decode_blocks, label_codes_from_config, load_config,  # noqa: E402
                 load_models, load_tags)

BINS = ((0, 0, "seen"), (1, FPS, "<1s"), (FPS, 3 * FPS, "1-3s"), (3 * FPS, 10 * FPS, "3-10s"),
        (10 * FPS, 30 * FPS, "10-30s"), (30 * FPS, 10 ** 9, ">=30s"))


def gap_length(missing: np.ndarray, halves) -> np.ndarray:
    """Per frame, the length of the missing-ball run it belongs to (0 where the ball is seen)."""
    out = np.zeros(len(missing), dtype=np.int64)
    for _, first, last in halves:
        m = missing[first:last + 1].astype(np.int8)
        edges = np.diff(np.concatenate(([0], m, [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            out[first + a:first + b] = b - a
    return out


def table(match, pred, truth=None) -> np.ndarray:
    """rows = BINS; columns: share of graded frames, predicted dead share, true dead share."""
    graded = match.graded_mask()
    gap = gap_length(np.isnan(match.ball_xy[:, 0]), match.halves)
    rows = []
    for lo, hi, _ in BINS:
        m = graded & ((gap == 0) if hi == 0 else (gap >= lo) & (gap < hi))
        n = max(int(m.sum()), 1)
        rows.append((m.sum() / graded.sum(), np.sum(pred[m] == 2) / n,
                     np.nan if truth is None else np.sum(truth[m] == 2) / n))
    rows.append((1.0, np.mean(pred[graded] == 2), np.nan if truth is None else np.mean(truth[graded] == 2)))
    return np.array(rows, dtype=np.float64)


def external_codes(config: dict, match, models: dict) -> np.ndarray:
    """The same stages on an external match, decoded on that match's own halves."""
    feats = features.compute(match, None, config.get("features"))
    evidence = stack.log_evidence(feats, match, config["evidence"], models.get("dead_residual"))
    return decode_blocks(evidence, [(a, b) for _, a, b in match.halves], config, feats)


def show(name: str, tables: list) -> None:
    t = np.mean(tables, axis=0)
    print(f"\n{name} ({len(tables)} match{'es' if len(tables) > 1 else ''})")
    print(f"{'ball run':8} {'frames':>7} {'pred D':>7} {'true D':>7}")
    for (_, _, label), row in zip(BINS + ((0, 0, "all"),), t):
        true = "      -" if np.isnan(row[2]) else f"{row[2]:7.3f}"
        print(f"{label:8} {row[0]:7.3f} {row[1]:7.3f} {true}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skillcorner", type=int, default=10, help="first N SkillCorner matches")
    parser.add_argument("--no-target", action="store_true")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    models = {k: v for k, v in load_models(config).items() if k == "dead_residual"}
    if "v6" in config:
        config["v6"]["w_pathcrf"] = 0.0
    if not args.no_target:
        match = io.load_match()
        show("competition match", [table(match, label_codes_from_config(config, match, load_tags(config), models))])
    tables = []
    for mid in skillcorner.available_matches()[:args.skillcorner]:
        ext = skillcorner.load_match(mid)
        tables.append(table(ext.match, external_codes(config, ext.match, models), ext.codes))
    if tables:
        show("SkillCorner (event labels)", tables)
    tables = []
    for game in metrica.available_games():
        ext = metrica.load_game(game)
        tables.append(table(ext.match, external_codes(config, ext.match, models), ext.codes))
    if tables:
        show("Metrica sample (native gaps; models trained on these matches)", tables)
    return 0


if __name__ == "__main__":
    sys.exit(main())
