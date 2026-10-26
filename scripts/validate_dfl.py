"""Validation on the Sportec Open DFL matches (public, CC-BY 4.0; other games only) and the case for
v5 (Metrica + DFL training). Prints aggregates only.

1. Shipped v3 / v4 and the official baseline on DFL copies degraded like the ATD (out of sample:
   neither model saw DFL), scored against DFL labels smoothed to "controlled possession".
2. v5 when each DFL match is held out in turn (trained on Metrica + the other six), against shipped
   v4, for cost_team 8 / 12 / 16 (12 is what config/possession_v5.json ships).
3. v5 vs v4 on each Metrica game held out in turn (v5 adds all seven DFL matches).
4. Optional: shipped v4 vs v5 on SkillCorner (real broadcast tracking).

usage: python scripts/validate_dfl.py <dfl dir> <metrica sample-data/data>
           [--skillcorner <opendata>/data/matches] [--sc-tracking DIR] [--sc-cache DIR]
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import D, X  # noqa: E402
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.dfl import MATCHES, controlled, load as load_dfl  # noqa: E402
from possession.external import degrade, load_sample_game, random_noise  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import baseline_labels, paired_bootstrap, score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = sorted(set(F.TEAM_FEATURES_V4) | set(F.DEAD_FEATURES_V4) | set(F.TEAM_FEATURES_V3)
               | set(F.DEAD_FEATURES))
SEEDS, TEST_SEEDS, COSTS = (0, 1, 2), (10, 11, 12), (8.0, 12.0, 16.0)


def slim(f):
    return {k: f[k].astype(np.float32) for k in NAMES}


def decode(model, cost_team, trk, f):
    return segment(model.dead_logit(f), model.team_logit(f), trk.blocks, trk.n, cost_dead=40.0,
                   cost_team=cost_team, fps=trk.fps)


def report(title, rows, pairs):
    print("\n" + title)
    for name, cases in rows.items():
        s = [score(lab, p) for lab, p in cases]
        derr = np.mean([np.mean(p[lab != X] == D) - np.mean(lab[lab != X] == D) for lab, p in cases])
        print("  %-9s spellF1 %.4f  macroF1 %.4f  possessions %.0f (truth %.0f)  D error %+.1f pp" % (
            name, np.mean([r["spellF1"] for r in s]), np.mean([r["macroF1"] for r in s]),
            np.mean([r["poss"] for r in s]), np.mean([r["gt_poss"] for r in s]), 100 * derr))
    for a, b in pairs:
        pb = paired_bootstrap([(lab, p, q) for (lab, p), (_, q) in zip(rows[a], rows[b])],
                              n_draws=1000)
        print("  paired %s - %s: %+.4f [%+.4f, %+.4f]" % (a, b, pb["delta"], pb["low"], pb["high"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dfl", help="folder with dfl_<match>.npz (scripts/fetch_dfl.py)")
    parser.add_argument("metrica", help="folder with Sample_Game_1..3")
    parser.add_argument("--skillcorner", default=None, help="SkillCorner opendata data/matches")
    parser.add_argument("--sc-tracking", default=None)
    parser.add_argument("--sc-cache", default=None)
    args = parser.parse_args()
    t0 = time.time()
    shipped = {v: Model.from_json(os.path.join(REPO, "config", "possession_%s.json" % v))
               for v in ("v3", "v4", "v5")}

    dfl = {}
    for m in MATCHES:
        trk, raw = load_dfl(os.path.join(args.dfl, "dfl_%s.npz" % m))
        dfl[m] = (trk, controlled(raw, trk.blocks, trk.fps))
    met = {g: load_sample_game(g, args.metrica) for g in (1, 2, 3)}
    train = {}
    for key, (trk, lab) in [(("dfl", m), v) for m, v in dfl.items()] + \
                           [(("met", g), v) for g, v in met.items()]:
        for s in SEEDS:
            deg = degrade(trk, lab, noise=random_noise(s), seed=s)
            train[key + (s,)] = (lab, slim(F.compute(deg)))
    print("training copies ready (%.0fs)" % (time.time() - t0), flush=True)

    def fit(keys):
        return Model.fit([train[k][1] for k in keys], [train[k][0] for k in keys],
                         team_names=F.TEAM_FEATURES_V4, dead_names=F.DEAD_FEATURES_V4)

    # 1 + 2: DFL matches held out
    rows = {"b50": [], "v3": [], "v4": []}
    rows.update({"v5 c%g" % c: [] for c in COSTS})
    for m, (trk, lab) in dfl.items():
        fold = fit([k for k in train if not (k[0] == "dfl" and k[1] == m)])
        for s in TEST_SEEDS:
            deg = degrade(trk, lab, seed=s)
            f = slim(F.compute(deg))
            rows["b50"].append((lab, baseline_labels(deg, 50)))
            for v in ("v3", "v4"):
                cfg = shipped[v].p["decode"]
                rows[v].append((lab, decode(shipped[v], cfg["cost_team"], deg, f)))
            for c in COSTS:
                rows["v5 c%g" % c].append((lab, decode(fold, c, deg, f)))
        print("  DFL %s done (%.0fs)" % (m, time.time() - t0), flush=True)
    report("DFL matches held out (7 x 3 degraded copies), labels = controlled possession", rows,
           [("v4", "v3"), ("v5 c12", "v4"), ("v5 c16", "v5 c12"), ("v5 c12", "v5 c8")])

    # 3: Metrica games held out
    rows = {"v4": [], "v5": []}
    for g, (trk, gt) in met.items():
        v4 = fit([k for k in train if k[0] == "met" and k[1] != g])
        v5 = fit([k for k in train if not (k[0] == "met" and k[1] == g)])
        for s in TEST_SEEDS:
            deg = degrade(trk, gt, seed=s)
            f = slim(F.compute(deg))
            rows["v4"].append((gt, decode(v4, 8.0, deg, f)))
            rows["v5"].append((gt, decode(v5, 12.0, deg, f)))
    report("Metrica games held out (event-derived labels, more fragmented than the challenge's)",
           rows, [("v5", "v4")])
    del train
    gc.collect()

    # 4: SkillCorner
    if args.skillcorner:
        from possession.skillcorner import load_match
        rows = {"v4": [], "v5": []}
        for mid in sorted(os.listdir(args.skillcorner)):
            folder = os.path.join(args.skillcorner, mid)
            tdir = args.sc_tracking or folder
            path = next((os.path.join(tdir, n) for n in ("%s_tracking_extrapolated.jsonl" % mid,
                                                         "%s_tracking.jsonl" % mid)
                         if os.path.isfile(os.path.join(tdir, n))), None)
            if not os.path.isdir(folder) or path is None:
                continue
            trk, lab = load_match(mid, folder, path, args.sc_cache)
            f = slim(F.compute(trk))
            for v in ("v4", "v5"):
                rows[v].append((lab, decode(shipped[v], shipped[v].p["decode"]["cost_team"], trk, f)))
        report("SkillCorner (%d matches, real broadcast tracking), shipped configs" % len(rows["v4"]),
               rows, [("v5", "v4")])
    print("total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
