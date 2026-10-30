"""Reproduce the v6 validation in docs/research/v6-pathcrf.md ("Validation of that exact setting"):
v5 against v5 + PathCRF on public matches that neither model was trained on. Aggregates only.

- Test matches: Sportec Open DFL J03WR9 and J03WQQ (PathCRF's test and validation matches) and
  Metrica sample games 1-3.
- v5 "fold" models: v5's features and recipe (scripts/train_model.py --version v5) trained without
  the test match, cached as v5_fold_<test>.json in --work (about 10 minutes each the first time).
- Each test match is degraded like the ATD (dropouts and identity swaps, no position jitter: the
  match's tracking report shows smooth positions), seeds 10-12.
- PathCRF exactly as in v6 (possession/pathcrf.py, config/possession_v6.json) at weights 0
  (= v5), 0.5 and 1 (= v6); scored with the official scorer against the external labels (DFL:
  controlled possession, possession.dfl.controlled).

One JSON line per copy goes to <work>/v6_validation.jsonl; a rerun resumes from it. About 4
minutes per copy on two CPU cores. Needs requirements-v6.txt.

usage: python scripts/validate_v6.py <dfl dir> <metrica sample-data/data> [--work DIR]
           [--only J03WR9,met1] [--seeds 10,11,12] [--summary]
       <dfl dir>: dfl_<match>.npz from scripts/fetch_dfl.py; <metrica ...>: the data/ folder of
       github.com/metrica-sports/sample-data (Sample_Game_1..3).
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import features as F  # noqa: E402
from possession import pathcrf  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.dfl import MATCHES, controlled, load as load_dfl  # noqa: E402
from possession.external import degrade, load_sample_game, random_noise  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.reference import score  # noqa: E402

NAMES = sorted(set(F.TEAM_FEATURES_V4) | set(F.DEAD_FEATURES_V4))
TESTS = ("J03WR9", "J03WQQ", "met1", "met2", "met3")
WEIGHTS = (0.0, 0.5, 1.0)
TRAIN_SEEDS = (0, 1, 2)
V6 = os.path.join(REPO, "config", "possession_v6.json")


def slim(feats: dict) -> dict:
    return {k: feats[k].astype(np.float32) for k in NAMES}


def fold_model(path: str, keys, load) -> Model:
    """v5's recipe without the test match (same order and seeds as the documented run); the
    training matches `keys` are only loaded when the file is not cached yet."""
    if not os.path.isfile(path):
        print("  training %s ..." % os.path.basename(path), flush=True)
        feats, labs = [], []
        for trk, lab in map(load, keys):
            for seed in TRAIN_SEEDS:
                deg = degrade(trk, lab, noise=random_noise(seed), seed=seed)
                feats.append(slim(F.compute(deg)))
                labs.append(lab)
        Model.fit(feats, labs, team_names=F.TEAM_FEATURES_V4,
                  dead_names=F.DEAD_FEATURES_V4).to_json(path)
    return Model.from_json(path)


def summary(rows) -> None:
    def show(name, rs):
        if not rs:
            return
        print("%s (%d copies)" % (name, len(rs)))
        for w in WEIGHTS:
            a = np.array([r["w%.1f" % w] for r in rs], dtype=float)
            print("  weight %.1f%s: spellF1 %.4f  macroF1 %.4f  possessions %.0f (truth %.0f)"
                  % (w, " (v5)" if w == 0 else " (v6)" if w == 1 else "      ",
                     a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()))
        matches = sorted({r["match"] for r in rs})
        rng = np.random.RandomState(0)
        for w in WEIGHTS[1:]:
            key = "w%.1f" % w
            diff = {m: [r[key][0] - r["w0.0"][0] for r in rs if r["match"] == m] for m in matches}
            every = np.concatenate([diff[m] for m in matches])
            boots = []
            for _ in range(4000):          # cluster bootstrap: matches, then copies inside them
                pick = rng.randint(0, len(matches), len(matches))
                boots.append(np.mean(np.concatenate(
                    [rng.choice(diff[matches[i]], len(diff[matches[i]])) for i in pick])))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print("  weight %.1f - v5: %+.4f [%+.4f, %+.4f], better on %d of %d copies"
                  % (w, every.mean(), lo, hi, int((every > 0).sum()), len(every)))

    show("DFL J03WR9 + J03WQQ", [r for r in rows if r["set"] == "dfl"])
    show("Metrica sample games 1-3", [r for r in rows if r["set"] == "met"])
    show("all", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dfl", help="folder with dfl_<match>.npz (scripts/fetch_dfl.py)")
    parser.add_argument("metrica", help="folder with Sample_Game_1..3 (metrica-sports/sample-data)")
    parser.add_argument("--work", default=os.path.join(os.path.expanduser("~"), ".mtah2027", "work",
                                                       "v6_validation"))
    parser.add_argument("--only", default=",".join(TESTS), help="subset of %s" % ",".join(TESTS))
    parser.add_argument("--seeds", default="10,11,12")
    parser.add_argument("--summary", action="store_true", help="only summarise the saved rows")
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)
    out = os.path.join(args.work, "v6_validation.jsonl")
    rows = [json.loads(line) for line in open(out)] if os.path.isfile(out) else []
    if args.summary:
        summary(rows)
        return 0
    done = {(r["set"], r["match"], r["seed"]) for r in rows}
    v6 = json.load(open(V6, encoding="utf-8"))
    pcfg, cost = v6["pathcrf"], v6["decode"]

    @functools.lru_cache(maxsize=None)
    def dfl(match):
        trk, raw = load_dfl(os.path.join(args.dfl, "dfl_%s.npz" % match))
        return trk, controlled(raw, trk.blocks, trk.fps)

    @functools.lru_cache(maxsize=None)
    def met(game):
        return load_sample_game(game, args.metrica)

    def load(item):
        kind, key = item
        return dfl(key) if kind == "dfl" else met(key)

    for test in [t.strip() for t in args.only.split(",") if t.strip()]:
        if test not in TESTS:
            parser.error("unknown test %s" % test)
        kind, key = ("met", test[3:]) if test.startswith("met") else ("dfl", test)
        todo = [s for s in map(int, args.seeds.split(",")) if (kind, key, s) not in done]
        if not todo:
            continue
        if kind == "dfl":
            keys = [("dfl", m) for m in MATCHES if m != key] + [("met", g) for g in (1, 2, 3)]
            trk, lab = dfl(key)
        else:
            keys = [("dfl", m) for m in MATCHES] + [("met", g) for g in (1, 2, 3) if g != int(key)]
            trk, lab = met(int(key))
        model = fold_model(os.path.join(args.work, "v5_fold_%s.json" % test), keys, load)
        for seed in todo:
            started = time.time()
            deg = degrade(trk, lab, noise={"player_noise_m": 0.0}, seed=seed)
            feats = slim(F.compute(deg))
            dead, team = model.dead_logit(feats), model.team_logit(feats)
            decode = dict(cost_dead=cost["cost_dead"], cost_team=cost["cost_team"], fps=deg.fps)
            v5 = segment(dead, team, deg.blocks, deg.n, **decode)
            logodds, _ = pathcrf.team_evidence(deg, v5, pcfg)
            row = {"set": kind, "match": key, "seed": seed}
            for w in WEIGHTS:
                r = score(lab, segment(dead, team + w * logodds, deg.blocks, deg.n, **decode))
                row["w%.1f" % w] = [r["spellF1"], r["macroF1"], r["poss"], r["gt_poss"]]
            row["seconds"] = round(time.time() - started)
            with open(out, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            rows.append(row)
            print("  %s seed %d: spellF1 v5 %.4f -> v6 %.4f (%d s)"
                  % (test, seed, row["w0.0"][0], row["w1.0"][0], row["seconds"]), flush=True)
    summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
