"""External validation on Metrica's public sample games (never the match): leave one game out.

For each held-out game, fit the coefficients and choose the switching costs on the other two
(noise seeds 0-1, degraded as each variant prescribes), then score the held-out game (seeds
10-12) under one or more test-time noise scenarios with the official metric, next to the official
baseline (windows 50 and 125) on the same degraded input. Paired bootstrap over GT possessions,
pooled over games and seeds. Prints aggregates only.

usage: python scripts/validate_external.py <sample-data/data> [--variants v4,v3] [--ref b50]
                                           [--noise default|all|a,b] [--out results.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import D, X  # noqa: E402
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.external import degrade, load_sample_game, random_noise  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import baseline_labels, paired_bootstrap, score  # noqa: E402

VARIANTS = {
    # name: (team features, dead-ball features, degradation of the training copies)
    "v1": (F.TEAM_FEATURES_V1, F.DEAD_FEATURES, "default"),
    "v2_touch": (F.TEAM_FEATURES_V1 + F.TEAM_TOUCH, F.DEAD_FEATURES, "default"),
    "v2_shape": (F.TEAM_FEATURES_V1 + F.TEAM_SHAPE, F.DEAD_FEATURES, "default"),
    "v2_all": (F.TEAM_FEATURES_V1 + F.TEAM_TOUCH + F.TEAM_SHAPE, F.DEAD_FEATURES, "default"),
    "v3": (F.TEAM_FEATURES_V3, F.DEAD_FEATURES, "default"),
    "v4": (F.TEAM_FEATURES_V4, F.DEAD_FEATURES_V4, "random"),
}
# Test-time noise scenarios: plausible ways the challenge ATD may differ from our default guess.
NOISE = {
    "default": {},
    "more_missing_in_play": {"p_miss_alive": 0.35, "p_miss_dead": 0.55},
    "ball_seen_in_stoppages": {"p_miss_dead": 0.45, "p_miss_alive": 0.30},
    "longer_gaps": {"gap_median_s": 15.0},
    "noisier_positions": {"ball_noise_m": 0.8, "player_noise_m": 0.4, "p_player_drop": 0.15},
    "smooth_players": {"player_noise_m": 0.03},
    "drifting_players": {"player_noise_m": 0.4, "player_noise_corr_s": 2.0},
}
TRAIN_SEEDS, TEST_SEEDS = (0, 1), (10, 11, 12)
COSTS_DEAD, COSTS_TEAM = (20.0, 40.0, 80.0), (3.0, 5.0, 8.0, 12.0)


def d_share(labels, gt):
    graded = gt != X
    return float(np.mean(labels[graded] == D))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder with Sample_Game_1..3 (metrica-sports/sample-data/data)")
    parser.add_argument("--variants", default="v4,v3", help="comma-separated: %s" % ",".join(VARIANTS))
    parser.add_argument("--ref", default="b50", help="reference for the paired bootstrap: b50, "
                                                    "b125 or one of the variants")
    parser.add_argument("--noise", default="default",
                        help="test scenario(s): 'all' or comma-separated from %s" % ",".join(NOISE))
    parser.add_argument("--out", default=None, help="write the aggregate results as JSON")
    args = parser.parse_args()
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.ref in VARIANTS and args.ref not in names:
        names.append(args.ref)
    scenarios = list(NOISE) if args.noise == "all" else [s.strip() for s in args.noise.split(",")]
    unknown = [s for s in scenarios if s not in NOISE] + [v for v in names if v not in VARIANTS]
    if unknown:
        parser.error("unknown: %s" % ", ".join(unknown))

    t0 = time.time()
    games = {g: load_sample_game(g, args.root) for g in (1, 2, 3)}

    # 1. fit every variant on two games and choose its costs there (memory: training copies only)
    fitted = {}
    for mode in sorted({VARIANTS[v][2] for v in names}):
        train = {}
        for g, (trk, gt) in games.items():
            for seed in TRAIN_SEEDS:
                deg = degrade(trk, gt, noise=random_noise(seed) if mode == "random" else None,
                              seed=seed)
                train[(g, seed)] = (deg, gt, F.compute(deg))
        for name in [v for v in names if VARIANTS[v][2] == mode]:
            team_names, dead_names, _ = VARIANTS[name]
            for test in (1, 2, 3):
                keys = [(g, s) for g in (1, 2, 3) if g != test for s in TRAIN_SEEDS]
                model = Model.fit([train[k][2] for k in keys], [train[k][1] for k in keys],
                                  team_names=team_names, dead_names=dead_names)
                logits = {k: (model.dead_logit(train[k][2]), model.team_logit(train[k][2]))
                          for k in keys}
                grid = {}
                for cd in COSTS_DEAD:
                    for ct in COSTS_TEAM:
                        grid[(cd, ct)] = np.mean([
                            score(train[k][1], segment(*logits[k], train[k][0].blocks,
                                                       train[k][0].n, cost_dead=cd,
                                                       cost_team=ct))["spellF1"]
                            for k in keys])
                fitted[(name, test)] = (model, max(grid, key=grid.get))
                print("  %-9s held-out game %d: costs dead %g / team %g"
                      % ((name, test) + fitted[(name, test)][1]), flush=True)
        del train
    print("fitted in %.0fs" % (time.time() - t0), flush=True)

    # 2. predict the held-out games under each test scenario, one degraded copy at a time
    preds, gts = {}, {}
    for sc in scenarios:
        for test, (trk, gt) in games.items():
            gts[test] = gt
            for seed in TEST_SEEDS:
                deg = degrade(trk, gt, noise=NOISE[sc], seed=seed)
                f = F.compute(deg)
                preds[("b50", test, sc, seed)] = baseline_labels(deg, 50)
                preds[("b125", test, sc, seed)] = baseline_labels(deg, 125)
                for name in names:
                    model, (cd, ct) = fitted[(name, test)]
                    preds[(name, test, sc, seed)] = segment(model.dead_logit(f),
                                                            model.team_logit(f), deg.blocks,
                                                            deg.n, cost_dead=cd, cost_team=ct)
                del f
        print("  predicted test noise %s (%.0fs)" % (sc, time.time() - t0), flush=True)

    results = {"ref": args.ref, "scenarios": {}}
    for sc in scenarios:
        print("\ntest noise: %s" % sc)
        print("%-9s %8s %8s %8s %8s %8s   %s" % ("variant", "spellF1", "macroF1", "poss", "gt_poss",
                                                "D err pp", "paired delta vs %s [5%%, 95%%]"
                                                % args.ref))
        rows = {}
        for name in ["b125", "b50"] + names:
            cases = [(g, s) for g in (1, 2, 3) for s in TEST_SEEDS]
            s_ = [score(gts[g], preds[(name, g, sc, s)]) for g, s in cases]
            row = {k: float(np.mean([r[k] for r in s_])) for k in ("spellF1", "macroF1", "poss",
                                                                     "gt_poss")}
            row["d_share_error_pp"] = 100 * float(np.mean([
                d_share(preds[(name, g, sc, s)], gts[g]) - d_share(gts[g], gts[g])
                for g, s in cases]))
            if name != args.ref:
                row["delta"] = paired_bootstrap(
                    [(gts[g], preds[(name, g, sc, s)], preds[(args.ref, g, sc, s)])
                     for g, s in cases], n_draws=1000)
            rows[name] = row
            delta = row.get("delta")
            print("%-9s %8.4f %8.4f %8.0f %8.0f %+8.1f   %s" % (
                name, row["spellF1"], row["macroF1"], row["poss"], row["gt_poss"],
                row["d_share_error_pp"],
                "%+.4f [%+.4f, %+.4f]" % (delta["delta"], delta["low"], delta["high"])
                if delta else "(reference)"))
        results["scenarios"][sc] = rows
    if len(scenarios) > 1:
        print("\nspellF1 over scenarios: mean / worst")
        for name in ["b125", "b50"] + names:
            vals = [results["scenarios"][sc][name]["spellF1"] for sc in scenarios]
            print("  %-9s %.4f / %.4f" % (name, np.mean(vals), np.min(vals)))
    print("total %.0fs" % (time.time() - t0))
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
