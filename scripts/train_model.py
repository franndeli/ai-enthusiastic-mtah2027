"""Fit the shipped coefficients on Metrica's public sample games 1-3 (degraded, fixed seeds) and write
config/possession_<version>.json. Deterministic. No match data is involved.

usage: python scripts/train_model.py <metrica sample-data/data> [out.json] [--version v4]
       python scripts/train_model.py <metrica sample-data/data> --version v5 --dfl <dir with dfl_*.npz>

v3  team features V1 + touch + shape (2 s, 6 s); dead-ball features from 0.24 s player speeds and
    the ball; every training copy degraded with the same noise model (DEFAULT_NOISE).
v4  dead-ball features from 1 s player speeds relative to each half's median, plus ball evidence;
    team features of v3 + the last team in control carried forward; every training copy degraded
    with a different random noise model (domain randomisation, `random_noise(seed)`).
    See docs/research/v4-robust-dead-ball.md.
v5  v4's features and degradation, trained on Metrica's sample games plus the seven Sportec Open DFL
    matches (scripts/fetch_dfl.py; CC-BY 4.0, data (c) DFL; Bassek et al. 2025, Scientific Data),
    whose labels come from corrected tracking like the challenge's, smoothed to "controlled
    possession" (possession.dfl.controlled). cost_dead 40, cost_team 12: the best value when each
    DFL match is held out in turn (scripts/validate_dfl.py). See docs/research/v5-dfl.md.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.dfl import MATCHES, controlled, load as load_dfl  # noqa: E402
from possession.external import DEFAULT_NOISE, degrade, load_sample_game, random_noise  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import score  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS = {
    # version: (team features, dead-ball features, degradation of the training copies)
    "v3": (F.TEAM_FEATURES_V3, F.DEAD_FEATURES, "default"),
    "v4": (F.TEAM_FEATURES_V4, F.DEAD_FEATURES_V4, "random"),
    "v5": (F.TEAM_FEATURES_V4, F.DEAD_FEATURES_V4, "random"),
}
SEEDS = (0, 1, 2)
COSTS_DEAD, COSTS_TEAM = (20.0, 40.0, 80.0), (3.0, 5.0, 8.0, 12.0)
DATA = ("Metrica Sports public sample games 1-3 (github.com/metrica-sports/sample-data), "
        "labels derived from Metrica's event data")
DATA_DFL = ("Sportec Open DFL / IDSSE, 7 matches (data (c) Deutsche Fussball Liga, CC-BY 4.0; "
            "Bassek, Rein, Weber & Memmert 2025, Scientific Data 12, 195, doi:10.1038/s41597-025-04505-y)"
            ", labels from BallStatus and BallPossession, possession changes shorter than 2 s ignored")
COST_TEAM_V5 = 12.0      # leave-one-DFL-match-out: 8 -> 0.5967, 12 -> 0.5978, 16 -> 0.5963


def slim(f: dict, names) -> dict:
    return {k: f[k].astype(np.float32) for k in names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder with Sample_Game_1..3 (metrica-sports/sample-data/data)")
    parser.add_argument("out", nargs="?", default=None,
                        help="output JSON (default config/possession_<version>.json)")
    parser.add_argument("--version", default="v3", choices=sorted(VERSIONS))
    parser.add_argument("--dfl", default=None, help="folder with dfl_<match>.npz (v5 only)")
    args = parser.parse_args()
    if args.version == "v5":
        if not args.dfl:
            parser.error("v5 needs --dfl <folder with dfl_*.npz> (scripts/fetch_dfl.py)")
        return train_v5(args)
    team, dead, mode = VERSIONS[args.version]
    out = args.out or os.path.join(REPO, "config", "possession_%s.json" % args.version)
    # Optional: leave one sample game out, so a rehearsal bundle built from it is out of sample.
    exclude = int(os.environ.get("MTAH_EXCLUDE_GAME", "0"))

    inst = []
    for g in [g for g in (1, 2, 3) if g != exclude]:
        trk, gt = load_sample_game(g, args.root)
        for seed in SEEDS:
            noise = random_noise(seed) if mode == "random" else None
            deg = degrade(trk, gt, noise=noise, seed=seed)
            inst.append((deg, gt, F.compute(deg)))
    model = Model.fit([i[2] for i in inst], [i[1] for i in inst], team_names=team, dead_names=dead)
    logits = [(model.dead_logit(f), model.team_logit(f)) for _, _, f in inst]
    grid = {}
    for cd in COSTS_DEAD:
        for ct in COSTS_TEAM:
            grid[(cd, ct)] = float(np.mean([
                score(gt, segment(dl, tl, deg.blocks, deg.n, cost_dead=cd, cost_team=ct))["spellF1"]
                for (deg, gt, _), (dl, tl) in zip(inst, logits)]))
    (cd, ct), best = max(grid.items(), key=lambda kv: kv[1])
    print("grid (training games, in-sample):")
    for k, v in sorted(grid.items()):
        print("  cost_dead %4g cost_team %4g  spellF1 %.4f" % (k[0], k[1], v))
    model.p["decode"] = {"cost_dead": cd, "cost_team": ct}
    if mode == "default":   # v3: kept exactly as first shipped, so the file regenerates byte for byte
        model.p["training"] = {"data": DATA, "degradation": DEFAULT_NOISE, "seeds": list(SEEDS),
                               "team_features": list(team)}
    else:
        model.p["training"] = {
            "version": args.version, "data": DATA, "seeds": list(SEEDS),
            "degradation": "DEFAULT_NOISE updated with random_noise(seed), one per seed",
            "noise_by_seed": {str(s): {k: round(v, 6) for k, v in random_noise(s).items()}
                              for s in SEEDS},
            "team_features": list(team), "dead_features": list(dead)}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.to_json(out)
    print("chosen cost_dead %g cost_team %g (in-sample spellF1 %.4f) -> %s" % (cd, ct, best, out))
    return 0


def train_v5(args) -> int:
    team, dead, _ = VERSIONS["v5"]
    names = sorted(set(team) | set(dead))
    out = args.out or os.path.join(REPO, "config", "possession_v5.json")
    exclude = int(os.environ.get("MTAH_EXCLUDE_GAME", "0"))
    met, dfl = [], []
    for g in [g for g in (1, 2, 3) if g != exclude]:
        trk, gt = load_sample_game(g, args.root)
        for seed in SEEDS:
            deg = degrade(trk, gt, noise=random_noise(seed), seed=seed)
            met.append((deg, gt, slim(F.compute(deg), names)))
    for match in MATCHES:
        trk, raw = load_dfl(os.path.join(args.dfl, "dfl_%s.npz" % match))
        lab = controlled(raw, trk.blocks, trk.fps)
        for seed in SEEDS:
            deg = degrade(trk, lab, noise=random_noise(seed), seed=seed)
            dfl.append((deg, lab, slim(F.compute(deg), names)))
        print("  %s ready" % match, flush=True)
    inst = met + dfl
    model = Model.fit([i[2] for i in inst], [i[1] for i in inst], team_names=team, dead_names=dead)
    model.p["decode"] = {"cost_dead": 40.0, "cost_team": COST_TEAM_V5}
    model.p["training"] = {
        "version": "v5", "data": [DATA, DATA_DFL], "seeds": list(SEEDS),
        "degradation": "DEFAULT_NOISE updated with random_noise(seed), one per seed",
        "noise_by_seed": {str(s): {k: round(v, 6) for k, v in random_noise(s).items()}
                          for s in SEEDS},
        "features_dtype": "float32",
        "cost_selection": "cost_dead 40 (as v4); cost_team 12, best held-out value when each "
                          "DFL match is left out in turn (8 / 12 / 16)",
        "team_features": list(team), "dead_features": list(dead)}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.to_json(out)
    print("cost_dead 40 cost_team %g -> %s" % (COST_TEAM_V5, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
