#!/usr/bin/env python3
"""Run the possession pipeline on the Metrica sample matches and score it against event labels.

    scripts/py scripts/metrica_eval.py [--config config/possession_v5.json] [--noise] [--seed 0]
                                       [--semantics] [--bootstrap 1000]

Same judges as scripts/idsse_eval.py (clean and ATD-like noisy tracking), on the provider whose
label semantics are closest to the competition ground truth. --semantics prints label statistics
(spell durations, dead share) to compare with the public window. Aggregates only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

import possession_metrics as pm  # noqa: E402
from possession import external, metrica  # noqa: E402
from idsse_eval import predict, print_table  # noqa: E402
from run import load_config  # noqa: E402


def semantics(ext) -> dict:
    runs = [(first, last + 1) for _, first, last in ext.match.halves]
    spells = pm.possession_spells(ext.codes, runs)
    durations = np.array([(b - a) / ext.fps for a, b, _ in spells])
    graded = ext.match.graded_mask()
    codes = ext.codes[graded]
    ball_missing = float(np.isnan(ext.match.ball_xy[graded, 0]).mean())
    transitions = pm.transitions(ext.codes, runs)
    swaps = sum(1 for _, a, b in transitions if a in (0, 1) and b in (0, 1))
    return {"n_spells": len(spells), "mean_s": float(durations.mean()),
            "median_s": float(np.median(durations)),
            "under_2s": float(np.mean(durations < 2)), "under_4s": float(np.mean(durations < 4)),
            "dead_share": float(np.mean(codes == 2)), "team_swap_share": swaps / max(len(transitions), 1),
            "graded_min": graded.sum() / ext.fps / 60, "ball_missing": ball_missing}


DURATION_EDGES = (0.0, 2.0, 4.0, 8.0, 16.0, 1e9)
LABELS = "HADX"


def diagnose(ext, codes: np.ndarray) -> str:
    """Spell recall by true duration (IoU 0.5), transition latency by type; aggregates only."""
    runs = [(first, last + 1) for _, first, last in ext.match.halves]
    pred = pm.fill_abstentions(codes, runs)
    gt_sp, pr_sp = pm.possession_spells(ext.codes, runs), pm.possession_spells(pred, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gt_sp, pr_sp), len(gt_sp), 0.5)
    durations = np.array([(b - a) / ext.fps for a, b, _ in gt_sp])
    parts = []
    for lo, hi in zip(DURATION_EDGES[:-1], DURATION_EDGES[1:]):
        sel = (durations >= lo) & (durations < hi)
        if sel.any():
            parts.append("{:g}-{:g}s n={} R={:.2f}".format(lo, min(hi, 99), int(sel.sum()),
                                                        float(flags[sel].mean())))
    lines = ["  recall@0.5 by true duration: " + "; ".join(parts)]
    res = pm.transition_scores(pm.transitions(ext.codes, runs), pm.transitions(pred, runs), 50,
                               typed="pair")
    by_type: dict = {}
    for g, p in res["pairs"]:
        by_type.setdefault((LABELS[g[1]], LABELS[g[2]]), []).append(p[0] - g[0])
    missed: dict = {}
    for g in res["missed"]:
        key = (LABELS[g[1]], LABELS[g[2]])
        missed[key] = missed.get(key, 0) + 1
    cells = []
    for key in sorted(set(by_type) | set(missed)):
        lat = np.array(by_type.get(key, []))
        cells.append("{}>{} n={} miss={} med={:+.0f} mean={:+.1f}".format(
            key[0], key[1], len(lat), missed.get(key, 0),
            np.median(lat) if len(lat) else 0, lat.mean() if len(lat) else 0))
    lines.append("  transitions (tol 50 frames, + = late): " + "; ".join(cells))
    lines.append("  spurious transitions {}".format(res["n_spurious"]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "config" / "possession_v5.json"))
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="re-parse the source files")
    parser.add_argument("--diagnose", action="store_true", help="error breakdown per game")
    parser.add_argument("--ball-missing", type=float, default=0.15,
                        help="extra ball dropout share for --noise (the clean files already lack"
                             " the ball on 32-41%% of frames)")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    games = args.games or metrica.available_games()
    if not games:
        print("no Metrica sample games found; run scripts/py scripts/fetch_metrica.py")
        return 1
    loaded = {}
    for game in games:
        started = time.time()
        loaded[game] = metrica.load_game(game, refresh=args.refresh)
        if args.semantics:
            s = semantics(loaded[game])
            print("{}: {:.0f} graded min, ball missing {:.1%}, {} spells, mean {:.1f} s, median {:.1f} s, "
                  "<2 s {:.0%}, <4 s {:.0%}, dead share {:.1%}, team swaps {:.0%} of transitions "
                  "({:.1f} s)".format(game, s["graded_min"], s["ball_missing"], s["n_spells"],
                                      s["mean_s"], s["median_s"], s["under_2s"], s["under_4s"],
                                      s["dead_share"], s["team_swap_share"], time.time() - started))
    clean = {}
    for game, ext in loaded.items():
        started = time.time()
        codes = predict(ext.match, config)
        result = pm.evaluate(ext.codes, codes, bootstrap_n=args.bootstrap)
        result["seconds"] = time.time() - started
        clean[game] = result
        if args.diagnose:
            print(game + " clean\n" + diagnose(ext, codes))
    print_table("clean tracking (config {})".format(Path(args.config).name), clean, args.bootstrap)
    if args.noise:
        noisy = {}
        for game, ext in loaded.items():
            started = time.time()
            params = external.NoiseParams(ball_missing_share=args.ball_missing)
            degraded = external.inject_noise(ext.match, seed=args.seed, params=params, fps=ext.fps)
            codes = predict(degraded, config)
            result = pm.evaluate(ext.codes, codes, bootstrap_n=args.bootstrap)
            result["seconds"] = time.time() - started
            noisy[game] = result
            if args.diagnose:
                print(game + " noisy\n" + diagnose(ext, codes))
        print_table("noisy tracking (seed {})".format(args.seed), noisy, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
