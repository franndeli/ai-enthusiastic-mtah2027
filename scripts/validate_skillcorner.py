"""Out-of-domain check on SkillCorner open data (github.com/SkillCorner/opendata, MIT licence):
A-League 2024/25 matches of broadcast tracking, i.e. a real tracker with real noise rather than our
simulated degradation. Other games only; never the challenge match.

Scores the shipped configs exactly as `run.py` applies them (same features, coefficients and
switching costs; nothing is trained here) and the official baseline (window 50), against labels
derived from SkillCorner's possession and game-interruption data (possession/skillcorner.py).
Paired bootstrap over GT possessions. Prints aggregates only.

usage: python scripts/validate_skillcorner.py <opendata>/data/matches [--tracking DIR]
           [--models config/possession_v4.json,config/possession_v3.json] [--cache DIR]

The tracking files are stored with Git LFS: clone with LFS installed (or run `git lfs pull`), or
point --tracking to a folder holding <id>_tracking_extrapolated.jsonl files.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import D, X  # noqa: E402
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import baseline_labels, paired_bootstrap, score  # noqa: E402
from possession.skillcorner import load_match  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tracking_file(match_id: str, match_dir: str, tracking_dir: str | None) -> str | None:
    folders = [tracking_dir] if tracking_dir else [match_dir]
    for folder in folders:
        for name in ("%s_tracking_extrapolated.jsonl" % match_id, "%s_tracking.jsonl" % match_id):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return None


def is_lfs_pointer(path: str) -> bool:
    with open(path, "rb") as handle:
        return handle.read(40).startswith(b"version https://git-lfs")


def d_share(labels, gt):
    graded = gt != X
    return float(np.mean(labels[graded] == D))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matches", help="SkillCorner opendata data/matches folder")
    parser.add_argument("--tracking", default=None,
                        help="folder with <id>_tracking_extrapolated.jsonl (default: each match folder)")
    parser.add_argument("--models", default="config/possession_v4.json,config/possession_v3.json",
                        help="comma-separated configs; the first is compared with the others")
    parser.add_argument("--cache", default=None, help="optional folder to cache the parsed matches")
    args = parser.parse_args()
    paths = [p if os.path.isabs(p) else os.path.join(REPO, p) for p in args.models.split(",") if p]
    models = {os.path.splitext(os.path.basename(p))[0]: Model.from_json(p) for p in paths}
    names = list(models)

    t0 = time.time()
    ids = sorted(d for d in os.listdir(args.matches) if os.path.isdir(os.path.join(args.matches, d)))
    preds, gts, skipped = {}, {}, []
    for mid in ids:
        folder = os.path.join(args.matches, mid)
        path = tracking_file(mid, folder, args.tracking)
        if path is None or is_lfs_pointer(path):
            skipped.append(mid)
            continue
        trk, gt = load_match(mid, folder, path, args.cache)
        feats = F.compute(trk)
        gts[mid] = gt
        preds[("b50", mid)] = baseline_labels(trk, 50)
        for name, model in models.items():
            cfg = model.p.get("decode", {"cost_dead": 40.0, "cost_team": 8.0})
            preds[(name, mid)] = segment(model.dead_logit(feats), model.team_logit(feats),
                                         trk.blocks, trk.n, cost_dead=cfg["cost_dead"],
                                         cost_team=cfg["cost_team"], fps=trk.fps)
        print("  match %s done (%.0fs)" % (mid, time.time() - t0), flush=True)
    if skipped:
        print("skipped %d matches without tracking (Git LFS not pulled?): %s"
              % (len(skipped), ", ".join(skipped)))
    if not gts:
        print("no SkillCorner tracking found; see the usage notes at the top of this script")
        return 2

    print("\n%d SkillCorner matches (other games; real broadcast-tracking noise)" % len(gts))
    print("%-16s %8s %8s %8s %8s %9s" % ("model", "spellF1", "macroF1", "poss", "gt_poss",
                                         "D err pp"))
    for name in names + ["b50"]:
        s = [score(gts[m], preds[(name, m)]) for m in gts]
        err = np.mean([d_share(preds[(name, m)], gts[m]) - d_share(gts[m], gts[m]) for m in gts])
        print("%-16s %8.4f %8.4f %8.0f %8.0f %+9.1f" % (
            name, np.mean([r["spellF1"] for r in s]), np.mean([r["macroF1"] for r in s]),
            np.mean([r["poss"] for r in s]), np.mean([r["gt_poss"] for r in s]), 100 * err))
    first = names[0]
    for other in names[1:] + ["b50"]:
        pb = paired_bootstrap([(gts[m], preds[(first, m)], preds[(other, m)]) for m in gts],
                              n_draws=1000)
        print("paired %s - %s: %+.4f [%+.4f, %+.4f]" % (first, other, pb["delta"], pb["low"],
                                                        pb["high"]))
    print("total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
