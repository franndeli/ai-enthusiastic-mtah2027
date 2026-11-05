#!/usr/bin/env python3
"""Select every rule parameter of the possession pipeline on external public data only.

    scripts/py scripts/tune_rules.py [--config config/possession.json] [--passes 2] [--jobs 8]
                                     [--tracker-noise] [--noisy-only]

Data: the Metrica sample games and IDSSE matches, clean and with target-matched noise seeds 0/1
(the training rows of scripts/train_v6.py). Objective: mean spellF1 of the rule stage over the
four groups (Metrica clean / noisy, IDSSE clean / noisy). --tracker-noise uses the W7 judge's noise;
--noisy-only scores the two noisy groups only. Coordinate descent over fixed grids,
starting from the config's values; each parameter takes the grid value with the best objective
(ties: the value nearest the current one). Nothing about the competition match is read, so the
Smart Tagging clue (no external tags) is fixed at 0. Prints the trajectory and the selected values
as JSON; deterministic.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

import possession_metrics as pm  # noqa: E402
from possession import evidence, features  # noqa: E402
from run import DEFAULT_CONFIG, decode_blocks  # noqa: E402
from train_v6 import external_rows  # noqa: E402

GRIDS = {
    "evidence.w_control": [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "evidence.w_shape": [0.15, 0.3, 0.45, 0.6, 0.8, 1.0],
    "evidence.w_touch": [0.5, 1.0, 1.5, 2.0],
    "evidence.touch_decay_s": [10.0, 20.0, 40.0],
    "evidence.touch_unseen_decay_s": [2.5, 5.0, 10.0],
    "evidence.team_confidence_max": [0.65, 0.75, 0.85],
    "evidence.w_touch_now": [0.5, 1.0, 2.0, 3.0],
    "evidence.dead_bias": [-1.6, -1.2, -0.8],
    "evidence.w_slow": [1.0, 1.5, 2.0],
    "evidence.slow_ref_m_s": [1.25, 1.5, 1.75],
    "evidence.slow_scale_m_s": [0.35, 0.5, 0.75],
    "evidence.w_out": [0.25, 0.5, 1.0, 1.5],
    "evidence.w_exit": [0.25, 0.5, 1.0],
    "evidence.w_moving": [1.0, 1.5, 2.0],
    "evidence.w_out_player": [0.5, 1.0, 1.5, 2.0],
    "evidence.w_slow_far": [0.5, 1.0, 1.5],
    "evidence.w_fresh_exit": [1.0, 1.5, 2.0],
    "evidence.dead_confidence_max": [0.7, 0.8, 0.9],
    "evidence.dead_floor": [0.02, 0.05, 0.1],
    "decoder.switch_cost.team_switch": [12.0, 16.0, 20.0],
    "decoder.switch_cost.to_dead": [6.0, 8.0, 10.0],
    "decoder.switch_cost.from_dead": [6.0, 8.0, 10.0],
    "features.touch_radius_m": [1.25, 1.5, 1.75, 2.0, 2.5],
    "features.touch_margin_m": [0.25, 0.5, 0.75],
    "features.touch_min_frames": [2, 3, 4],
    "features.ball_moving_m_s": [4.0, 5.0, 6.0],
    "features.touch_passby_dv_m_s": [0.0, 1.5, 2.0, 2.5],
}
FIXED = {"evidence.w_tag_dead": 0.0}

ROWS: list = []          # [(group key, match, codes, feats)] shared with forked workers


def get(config, key):
    node = config
    for part in key.split("."):
        node = node[part]
    return node


def put(config, key, value):
    out = copy.deepcopy(config)
    node = out
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return out


def objective(config) -> float:
    groups: dict = {}
    for kind, match, codes, feats in ROWS:
        ev = evidence.scores(feats, config["evidence"])
        pred = decode_blocks(ev, [(a, b) for _, a, b in match.halves], config, feats)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            groups.setdefault(kind, []).append(pm.evaluate(codes, pred, bootstrap_n=0)["spell_f1_mean"])
    return float(np.mean([np.mean(v) for v in groups.values()]))


def load_rows(config):
    global ROWS
    ROWS = [(f"{provider}-{'clean' if seed is None else 'noisy'}", match, codes,
             features.compute(match, None, config.get("features")))
            for provider, _, seed, match, codes in SOURCES]


SOURCES: list = []       # the labelled copies, built once: only their features change with a config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--tracker-noise", action="store_true")
    parser.add_argument("--noisy-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key, value in FIXED.items():
        config = put(config, key, value)
    started = time.time()
    SOURCES[:] = [row for row in external_rows(args.tracker_noise) if row[2] is not None or not args.noisy_only]
    load_rows(config)
    best = objective(config)
    print(f"start objective {best:.4f} ({len(ROWS)} rows, {time.time() - started:.0f}s)", flush=True)
    ctx = mp.get_context("fork")
    for n in range(args.passes):
        changed = False
        for key, grid in GRIDS.items():
            current = get(config, key)
            candidates = [put(config, key, v) for v in grid]
            if key.startswith("features."):
                scores = []
                for cand in candidates:       # features change: recompute them, sequentially
                    load_rows(cand)
                    scores.append(objective(cand))
                load_rows(config)
            else:
                with ctx.Pool(args.jobs) as pool:
                    scores = pool.map(objective, candidates)
            order = sorted(range(len(grid)), key=lambda i: (-round(scores[i], 9), abs(grid[i] - current)))
            choice = grid[order[0]]
            if choice != current and scores[order[0]] > best + 1e-9:
                config, best, changed = candidates[order[0]], scores[order[0]], True
                if key.startswith("features."):
                    load_rows(config)
            print(f"pass {n + 1} {key:34} {current!s:>6} -> {get(config, key)!s:>6}  objective {best:.4f}  "
                  f"grid " + " ".join(f"{v}:{s:.4f}" for v, s in zip(grid, scores)), flush=True)
        if not changed:
            break
    selected = {key: get(config, key) for key in list(GRIDS) + list(FIXED)}
    print(json.dumps(selected, indent=1))
    print(f"final objective {best:.4f} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
