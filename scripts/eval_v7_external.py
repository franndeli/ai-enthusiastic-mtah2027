#!/usr/bin/env python3
"""Leakage-free external evaluation of v7 against v6 with the production code (W7, D017).

    scripts/py scripts/eval_v7_external.py [--seeds 4 5]

For each evaluated match A among the five PathCRF never trained on, every learned part of both
versions is refit without A, at every level: the classifier and verifier that score A never saw it,
nor did the residuals and classifiers behind their training rows. v6 keeps its own rules
(config/possession_v6.json) and is refit on the same rows, so it is not handicapped by the noise it
was first trained on. A is scored clean and with the W7 judge's noise (external.inject_tracker_noise)
of seeds never used for selection (default 4, 5; 0/1 train and tune, 2/3 selected the recipe).
Prints aggregate scores only. External public data; never reads the competition match.
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
from possession import decoder, evidence, external, metrica, verifier  # noqa: E402
from run import DEFAULT_CONFIG, V6_CONFIG, v7_evidence  # noqa: E402
from train_v6 import external_rows  # noqa: E402
from train_v7 import Nested, build_row  # noqa: E402


def noisy_copy(provider: str, mid: str, seed: int):
    ext = metrica.load_game(mid) if provider == "metrica" else external.load_external(mid)
    return external.inject_tracker_noise(ext, provider, seed), ext.codes


def v6_verifier(nested: Nested, excluded: frozenset) -> verifier.VerifierModel:
    """v6's verifier without `excluded`: v6 evidence, constant costs (each row's residual also
    excludes its own match)."""
    decode = verifier.viterbi_decode(nested.config["decoder"])
    tables = []
    for r in nested.pool:
        if r.group in nested.unseen and r.group not in excluded:
            z_team, z_dead, _ = nested.logits(r, excluded | {r.group})
            ev = evidence.combine(z_team, z_dead, nested.params)
            for a, b in [(a, b) for _, a, b in r.match.halves]:
                prod, edits, X = verifier.block_table(ev[a:b + 1], r.feats, a, decode)
                tables.append((X, verifier.labels(np.asarray(r.codes[a:b + 1]), prod, edits)))
    return verifier.fit(tables)


def decode_row(row, ev, model, config, scale=None) -> np.ndarray:
    codes = np.full(row.match.n_frames, 3, dtype=np.int8)
    for _, a, b in row.match.halves:
        decode = verifier.viterbi_decode(config["decoder"], None if scale is None else scale[a:b + 1])
        codes[a:b + 1] = verifier.decode_block(model, ev[a:b + 1], row.feats, a, decode)
    return codes


def score(codes, pred) -> tuple[float, float, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pm.evaluate(np.asarray(codes), pred, bootstrap_n=0)
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config_v6 = json.loads(V6_CONFIG.read_text())
    sources = external_rows(tracker_noise=True)
    nested = Nested([build_row((p, m), s, match, codes, config) for p, m, s, match, codes in sources], config)
    nested_v6 = Nested([build_row((p, m), s, match, codes, config_v6) for p, m, s, match, codes in sources], config_v6)
    table: dict = {}
    for group in nested.unseen:
        excluded = frozenset({group})
        model_v6 = v6_verifier(nested_v6, excluded)
        model_v7 = nested.fit_verifier(excluded)
        classifier = nested.classifier(excluded)
        copies = [(None, *next((r.match, r.codes) for r in nested.pool if r.group == group and r.seed is None))]
        copies += [(s, *noisy_copy(*group, s)) for s in args.seeds]
        for seed, match, codes in copies:
            row_v6 = build_row(group, seed, match, codes, config_v6)
            z_team, z_dead, _ = nested_v6.logits(row_v6, excluded)
            base = decode_row(row_v6, evidence.combine(z_team, z_dead, nested_v6.params), model_v6, config_v6)
            row = build_row(group, seed, match, codes, config)
            z_team, z_dead, pcrf = nested.logits(row, excluded)
            ev, scale = v7_evidence(config, classifier, row.match, row.feats, z_team, z_dead, pcrf, nested.params)
            cand = decode_row(row, ev, model_v7, config, scale)
            kind = "clean" if row.seed is None else "noisy"
            table.setdefault(group, {"clean": [], "noisy": []})[kind].append(
                (*score(row.codes, base), *score(row.codes, cand)))
        print(f"{group[0]} {group[1]}: " + "  ".join(
            f"{k} {np.mean([x[3] - x[0] for x in v]):+.4f}" for k, v in table[group].items()), flush=True)
    for provider in ("metrica", "idsse"):
        for kind in ("clean", "noisy"):
            vals = [np.mean(c[kind], axis=0) for (p, _), c in table.items() if p == provider]
            if vals:
                v = np.array(vals)
                print(f"{provider:8} {kind:6} v6 {v[:, 0].mean():.4f} v7 {v[:, 3].mean():.4f} "
                      f"spell {v[:, 3].mean() - v[:, 0].mean():+.4f} macro {v[:, 4].mean() - v[:, 1].mean():+.4f} "
                      f"possessions {v[:, 2].mean():.0f} -> {v[:, 5].mean():.0f}")
    cells = [np.mean([x[3] - x[0] for x in c[k]]) for c in table.values() for k in c]
    print(f"positive match cells {sum(d > 0 for d in cells)}/{len(cells)}, mean {np.mean(cells):+.4f}, min {min(cells):+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
