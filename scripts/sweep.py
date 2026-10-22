#!/usr/bin/env python3
"""Score transition-cost grids; cache evidence privately and print aggregates only."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession import evidence, features, io, structure
from run import decode_blocks, load_tags
from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks


def grid(text: str) -> list[float]:
    values = [float(value) for value in text.split(',')]
    if not values or not np.isfinite(values).all() or min(values) < 0:
        raise argparse.ArgumentTypeError('cost grid must contain finite non-negative values')
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/possession.json')
    parser.add_argument('--team-switch', type=grid, default=grid('3,6,12'))
    parser.add_argument('--to-dead', type=grid, default=grid('3,6,12'))
    parser.add_argument('--from-dead', type=grid, default=grid('3,6,12'))
    parser.add_argument('--duration', action='store_true', help='require two-frame states')
    parser.add_argument('--evidence-cache', help='cache filename within the private work directory')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    config = json.loads(args.config.read_text())
    cache = None
    if args.evidence_cache:
        name = Path(args.evidence_cache)
        if name.name != str(name) or name.suffix != '.npy':
            parser.error('--evidence-cache must be a plain .npy filename')
        cache = io.work_dir() / name
    started = time.perf_counter()
    feats = features.compute(io.load_match(), load_tags(config))
    if cache is not None and cache.exists():
        log_evidence = np.load(cache, allow_pickle=False)
        print('reused evidence cache; caller must ensure evidence config matches')
    else:
        log_evidence = evidence.scores(feats, config.get('evidence'))
        if cache is not None:
            np.save(cache, log_evidence, allow_pickle=False)
    print('evidence frames={} elapsed_s={:.3f}'.format(len(log_evidence), time.perf_counter()-started))
    gt = io.load_public_gt()
    full_runs = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    duration = 2 if args.duration else config['decoder'].get('min_duration_frames', 1)
    print('team_switch to_dead from_dead spellF1 macroF1 public_possessions full_possessions mean_duration_s decode_s duration')
    for team, to_dead, from_dead in itertools.product(args.team_switch, args.to_dead, args.from_dead):
        run_config = dict(config, decoder=dict(config['decoder']))
        run_config['decoder'].update(
            switch_cost=dict(team_switch=team, to_dead=to_dead, from_dead=from_dead),
            min_duration_frames=duration)
        started = time.perf_counter()

        def decode_block(block, start):
            block_feats = {name: value[start:start + len(block)] for name, value in feats.items()}
            return decode_blocks(block, [(0, len(block) - 1)], run_config, block_feats)

        codes = structure.decode_halves(log_evidence, decode_block)
        elapsed = time.perf_counter() - started
        result = pm.evaluate(gt, codes, bootstrap_n=args.bootstrap)
        stats = pm.possession_stats(codes, full_runs)
        print('{:g} {:g} {:g} {:.6f} {:.6f} {} {} {:.3f} {:.3f} {}'.format(
            team, to_dead, from_dead, result['spell_f1_mean'], result['macro_f1'],
            result['n_pred_spells'], stats['n_spells'], stats['mean_duration_s'], elapsed, duration), flush=True)
        print('  bootstrap={} spell_CI=[{:.6f},{:.6f}] macro_CI=[{:.6f},{:.6f}]'.format(
            args.bootstrap, result['spell_bootstrap_low'], result['spell_bootstrap_high'],
            result['bootstrap_low'], result['bootstrap_high']))
    print('Public sweeps are diagnostics only; parameter selection uses IDSSE.')
    print('Only 17 public possessions: differences below 0.05 cannot select variants reliably.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as error:
        # Data-boundary failures must not echo values or paths from an asset.
        print('sweep failed: {}'.format(type(error).__name__), file=sys.stderr)
        raise SystemExit(1)
