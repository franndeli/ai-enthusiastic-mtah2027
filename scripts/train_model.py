"""Fit the shipped coefficients on Metrica's public sample games 1-3 (degraded, fixed seeds) and write
config/possession_v3.json. Deterministic. No match data is involved.

usage: python scripts/train_model.py <metrica sample-data/data> [config/possession_v3.json]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.external import DEFAULT_NOISE, degrade, load_sample_game  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import score  # noqa: E402

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "possession_v3.json")
SEEDS = (0, 1, 2)
TEAM = F.TEAM_FEATURES_V1 + F.TEAM_TOUCH + F.TEAM_SHAPE + F.TEAM_SHAPE6
# Optional: leave one sample game out, so a rehearsal bundle built from it is out of sample.
EXCLUDE = int(os.environ.get("MTAH_EXCLUDE_GAME", "0"))

inst = []
for g in [g for g in (1, 2, 3) if g != EXCLUDE]:
    trk, gt = load_sample_game(g, ROOT)
    for seed in SEEDS:
        deg = degrade(trk, gt, seed=seed)
        inst.append((deg, gt, F.compute(deg)))
model = Model.fit([i[2] for i in inst], [i[1] for i in inst], team_names=TEAM)
logits = [(model.dead_logit(f), model.team_logit(f)) for _, _, f in inst]
grid = {}
for cd in (20.0, 40.0, 80.0):
    for ct in (3.0, 5.0, 8.0, 12.0):
        grid[(cd, ct)] = float(np.mean([score(gt, segment(dl, tl, deg.blocks, deg.n, cost_dead=cd,
                                                          cost_team=ct))["spellF1"]
                                        for (deg, gt, _), (dl, tl) in zip(inst, logits)]))
(cd, ct), best = max(grid.items(), key=lambda kv: kv[1])
print("grid (training games, in-sample):")
for k, v in sorted(grid.items()):
    print("  cost_dead %4g cost_team %4g  spellF1 %.4f" % (k[0], k[1], v))
model.p["decode"] = {"cost_dead": cd, "cost_team": ct}
model.p["training"] = {
    "data": "Metrica Sports public sample games 1-3 (github.com/metrica-sports/sample-data), "
            "labels derived from Metrica's event data",
    "degradation": DEFAULT_NOISE, "seeds": list(SEEDS), "team_features": list(TEAM)}
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
model.to_json(OUT)
print("chosen cost_dead %g cost_team %g (in-sample spellF1 %.4f) -> %s" % (cd, ct, best, OUT))
