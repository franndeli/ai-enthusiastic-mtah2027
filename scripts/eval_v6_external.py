#!/usr/bin/env python3
"""Leakage-free external evaluation of v6 against v5 with the production code (W6, D015).

    scripts/py scripts/eval_v6_external.py [--seeds 2 3] [--all-matches]

For each evaluated match A, every learned part is retrained without A: the dead residual of A
on the other matches, and the verifier on the other PathCRF-unseen matches' clean + seed 0/1
rows, whose own residuals come from models trained without A and without themselves. A is scored
clean and with target-matched noise of fresh seeds (default 2 and 3). By default only the five
matches PathCRF never trained on are scored; --all-matches drops PathCRF (w_pathcrf 0) and scores
all ten. Prints aggregate scores only. External public data; never reads the competition match.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

import possession_metrics as pm  # noqa: E402
from possession import dead_residual, evidence, external, features, metrica, stack, verifier  # noqa: E402
from run import RULES_CONFIG, V6_CONFIG, decode_blocks  # noqa: E402
from train_v6 import PATHCRF_UNSEEN, pathcrf_probability  # noqa: E402


def copies(provider, mid, ext, seeds, config):
    out = [(None, ext.match)]
    noise = external.MATCHED_NOISE[provider]
    out += [(s, external.inject_matched_noise(ext.match, s, noise["share"], ext.fps,
                                              noise["ball_err_m"])) for s in seeds]
    rows = []
    rule = dict(evidence.DEFAULT_PARAMS, **config["evidence"])
    rule["w_fresh_exit"] = 0.0
    for seed, match in out:
        feats = features.compute(match, None, config.get("features"))
        rows.append(dead_residual.TrainingRow((provider, mid), match, feats,
                                              evidence.dead_logit(feats, rule), ext.codes, seed))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--all-matches", action="store_true")
    args = parser.parse_args()
    config = json.loads(V6_CONFIG.read_text())
    v5 = json.loads(RULES_CONFIG.read_text())
    w_pathcrf = 0.0 if args.all_matches else config["v6"]["w_pathcrf"]
    sources = [("metrica", g, metrica.load_game(g)) for g in metrica.available_games()]
    sources += [("idsse", m, external.load_external(m)) for m in external.available_matches()]
    pool = [r for p, m, e in sources for r in copies(p, m, e, (0, 1), config)]
    verifier_groups = {r.group for r in pool} if args.all_matches else PATHCRF_UNSEEN
    viterbi = verifier.viterbi_decode(config["decoder"])
    folds: dict = {}

    def residual_model(excluded: frozenset):
        if excluded not in folds:
            folds[excluded] = dead_residual.fit([r for r in pool if r.group not in excluded])
        return folds[excluded]

    def log_ev(row, model):
        blocks = [(a, b) for _, a, b in row.match.halves]
        return stack.log_evidence(
            row.feats, row.match, config["evidence"], model, w_pathcrf,
            lambda e: decode_blocks(e, blocks, config, row.feats),
            lambda match, alive: pathcrf_probability(*row.group, row.seed, match, alive))

    table = {}
    for provider, mid, ext in sources:
        group = (provider, mid)
        if group not in verifier_groups:
            continue
        tables = []
        for r in pool:
            if r.group in verifier_groups and r.group != group:
                ev = log_ev(r, residual_model(frozenset({group, r.group})))
                for a, b in [(a, b) for _, a, b in r.match.halves]:
                    prod, edits, X = verifier.block_table(ev[a:b + 1], r.feats, a, viterbi)
                    tables.append((X, verifier.labels(np.asarray(r.codes[a:b + 1]), prod, edits)))
        model = verifier.fit(tables)
        for row in copies(provider, mid, ext, tuple(args.seeds), config):
            blocks = [(a, b) for _, a, b in row.match.halves]
            ev = log_ev(row, residual_model(frozenset({group})))
            cand = np.full(row.match.n_frames, 3, dtype=np.int8)
            for a, b in blocks:
                cand[a:b + 1] = verifier.decode_block(model, ev[a:b + 1], row.feats, a, viterbi)
            base_feats = features.compute(row.match, None, v5.get("features"))
            base = decode_blocks(evidence.scores(base_feats, v5["evidence"]), blocks, v5, base_feats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sb = pm.evaluate(row.codes, base, bootstrap_n=0)
                sc = pm.evaluate(row.codes, cand, bootstrap_n=0)
            kind = "clean" if row.seed is None else "noisy"
            cell = table.setdefault((provider, mid), {"clean": [], "noisy": []})
            cell[kind].append((sb["spell_f1_mean"], sc["spell_f1_mean"], sb["macro_f1"], sc["macro_f1"]))
        print(f"{provider} {mid}: " + "  ".join(
            f"{k} {np.mean([x[1] - x[0] for x in v]):+.4f}" for k, v in table[group].items()), flush=True)
    for provider in ("metrica", "idsse"):
        for kind in ("clean", "noisy"):
            vals = [x for (p, _), c in table.items() if p == provider for x in c[kind]]
            if vals:
                v = np.array(vals)
                print(f"{provider:8} {kind:6} v5 {v[:, 0].mean():.4f} v6 {v[:, 1].mean():.4f} "
                      f"spell {v[:, 1].mean() - v[:, 0].mean():+.4f} macro {v[:, 3].mean() - v[:, 2].mean():+.4f}")
    cells = [np.mean([x[1] - x[0] for x in c[k]]) for c in table.values() for k in c]
    print(f"positive match cells {sum(d > 0 for d in cells)}/{len(cells)}, min {min(cells):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
