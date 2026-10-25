"""External validation on Metrica's public sample games (never the match): leave one game out.

For each held-out game, fit the coefficients and choose the switching costs on the other two
(noise seeds 0-1), then score the held-out game (seeds 10-12) with the official metric, next to the
official baseline (windows 50 and 125) on the same degraded input. Paired bootstrap over GT
possessions, pooled over games and seeds. Prints aggregates only.

usage: python scripts/validate_external.py <sample-data/data> [--variants v3,v1] [--ref b50]
                                           [--noise default] [--out results.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.external import degrade, load_sample_game  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import baseline_labels, paired_bootstrap, score  # noqa: E402

VARIANTS = {
    "v1": F.TEAM_FEATURES_V1,
    "v2_touch": F.TEAM_FEATURES_V1 + F.TEAM_TOUCH,
    "v2_shape": F.TEAM_FEATURES_V1 + F.TEAM_SHAPE,
    "v2_all": F.TEAM_FEATURES_V1 + F.TEAM_TOUCH + F.TEAM_SHAPE,
    "v3": F.TEAM_FEATURES_V1 + F.TEAM_TOUCH + F.TEAM_SHAPE + F.TEAM_SHAPE6,
}
# Test-time noise scenarios (training always uses the default degradation).
NOISE = {
    "default": {},
    "more_missing_in_play": {"p_miss_alive": 0.35, "p_miss_dead": 0.55},
    "ball_seen_in_stoppages": {"p_miss_dead": 0.45, "p_miss_alive": 0.30},
    "longer_gaps": {"gap_median_s": 15.0},
    "noisier_positions": {"ball_noise_m": 0.8, "player_noise_m": 0.4, "p_player_drop": 0.15},
}
TRAIN_SEEDS, TEST_SEEDS = (0, 1), (10, 11, 12)
COSTS_DEAD, COSTS_TEAM = (20.0, 40.0, 80.0), (3.0, 5.0, 8.0, 12.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder with Sample_Game_1..3 (metrica-sports/sample-data/data)")
    parser.add_argument("--variants", default="v3", help="comma-separated: %s" % ",".join(VARIANTS))
    parser.add_argument("--ref", default="b50", help="reference for the paired bootstrap: b50, "
                                                    "b125 or one of the variants")
    parser.add_argument("--noise", default="default", choices=sorted(NOISE))
    parser.add_argument("--out", default=None, help="write the aggregate results as JSON")
    args = parser.parse_args()
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.ref in VARIANTS and args.ref not in names:
        names.append(args.ref)

    t0 = time.time()
    games = {g: load_sample_game(g, args.root) for g in (1, 2, 3)}
    feats = {}
    for g, (trk, gt) in games.items():
        for seed in TRAIN_SEEDS + TEST_SEEDS:
            noise = None if seed in TRAIN_SEEDS else NOISE[args.noise]
            deg = degrade(trk, gt, noise=noise, seed=seed)
            feats[(g, seed)] = (deg, gt, F.compute(deg))
    print("prepared %d degraded instances in %.0fs (test noise: %s)"
          % (len(feats), time.time() - t0, args.noise), flush=True)

    preds, rows = {}, {}
    for test in (1, 2, 3):
        for seed in TEST_SEEDS:
            deg, gt, _ = feats[(test, seed)]
            preds[("b50", test, seed)] = baseline_labels(deg, 50)
            preds[("b125", test, seed)] = baseline_labels(deg, 125)
    for name in names:
        team_names = VARIANTS[name]
        for test in (1, 2, 3):
            keys = [(g, s) for g in (1, 2, 3) if g != test for s in TRAIN_SEEDS]
            model = Model.fit([feats[k][2] for k in keys], [feats[k][1] for k in keys],
                              team_names=team_names)
            logits = {k: (model.dead_logit(feats[k][2]), model.team_logit(feats[k][2]))
                      for k in keys + [(test, s) for s in TEST_SEEDS]}
            grid = {}
            for cd in COSTS_DEAD:
                for ct in COSTS_TEAM:
                    grid[(cd, ct)] = np.mean([
                        score(feats[k][1], segment(*logits[k], feats[k][0].blocks, feats[k][0].n,
                                                   cost_dead=cd, cost_team=ct))["spellF1"]
                        for k in keys])
            cd, ct = max(grid, key=grid.get)
            for s in TEST_SEEDS:
                deg = feats[(test, s)][0]
                preds[(name, test, s)] = segment(*logits[(test, s)], deg.blocks, deg.n,
                                                 cost_dead=cd, cost_team=ct)
            print("  %-9s held-out game %d: costs dead %g / team %g" % (name, test, cd, ct),
                  flush=True)

    print("\n%-9s %8s %8s %8s %8s   %s" % ("variant", "spellF1", "macroF1", "poss", "gt_poss",
                                           "paired delta vs %s [5%%, 95%%]" % args.ref))
    results = {"noise": args.noise, "ref": args.ref, "variants": {}}
    for name in ["b125", "b50"] + names:
        s = [score(feats[(g, sd)][1], preds[(name, g, sd)]) for g in (1, 2, 3) for sd in TEST_SEEDS]
        row = {k: float(np.mean([r[k] for r in s])) for k in ("spellF1", "macroF1", "poss",
                                                                 "gt_poss")}
        if name != args.ref:
            cases = [(feats[(g, sd)][1], preds[(name, g, sd)], preds[(args.ref, g, sd)])
                     for g in (1, 2, 3) for sd in TEST_SEEDS]
            row["delta"] = paired_bootstrap(cases, n_draws=1000)
        results["variants"][name] = row
        delta = row.get("delta")
        print("%-9s %8.4f %8.4f %8.0f %8.0f   %s" % (
            name, row["spellF1"], row["macroF1"], row["poss"], row["gt_poss"],
            "%+.4f [%+.4f, %+.4f]" % (delta["delta"], delta["low"], delta["high"])
            if delta else "(reference)"))
    print("total %.0fs" % (time.time() - t0))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
