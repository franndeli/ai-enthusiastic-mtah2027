#!/usr/bin/env python3
"""Train the learned v7 components on external public data only and write them to models/.

    scripts/py scripts/train_v7.py [--config config/possession.json]   (needs torch: PathCRF)

Data: the Metrica sample games and the IDSSE matches (scripts/fetch_metrica.py,
scripts/fetch_idsse.py + scripts/idsse_prepare.py), clean and with the W7 judge's noise, seeds 0/1
(external.inject_tracker_noise). Nothing about the competition match is read.

1. Dead residual (possession/dead_residual.py): on all 30 rows.
2. Hand-over classifier (possession/handover.py): on the rows of the five matches PathCRF never
   trained on, each row's evidence carrying a dead residual trained without that row's match.
3. Spell-edit verifier (possession/verifier.py), same rows and features as v6's but on the v7 decodes:
   each row's evidence comes from a residual and a classifier trained without that row's match.
Prints the SHA-256 of the three files for config/possession.json; writing them there selects these models.
PathCRF outputs are cached in the work directory. Run twice, the residual came out byte-identical and the
other two did not, yet their submission was the same: 0 of 158,857 graded frames (docs/PLAN.md W7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from possession import dead_residual, evidence, features, handover, io, stack, verifier  # noqa: E402
from run import DEFAULT_CONFIG, decode_blocks, v7_evidence  # noqa: E402
from train_v6 import EXPECTED_ROWS, PATHCRF_UNSEEN, external_rows, pathcrf_version  # noqa: E402


def pathcrf_outputs(row: dead_residual.TrainingRow, alive: np.ndarray) -> dict:
    """PathCRF outputs of one row inside its rule alive stretches, cached on disk."""
    from possession import pathcrf
    key = pathcrf_version() + "_" + hashlib.sha256(np.packbits(alive).tobytes()).hexdigest()[:16]
    path = io.work_dir() / "v7_train" / f"pathcrf_{row.group[0]}_{row.group[1]}_{row.seed}_{key}.npz"
    if path.is_file():
        with np.load(path) as saved:
            return {k: saved[k] for k in saved.files}
    out = pathcrf.outputs(row.match, alive)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def build_row(group: tuple, seed, match, codes, config: dict) -> dead_residual.TrainingRow:
    feats = features.compute(match, None, config.get("features"))
    rule = dict(evidence.DEFAULT_PARAMS, **config["evidence"])
    rule["w_fresh_exit"] = 0.0
    return dead_residual.TrainingRow(group, match, feats, evidence.dead_logit(feats, rule), codes, seed)


class Nested:
    """The stack's learned parts, each fitted without a given set of whole matches. Models are kept;
    nothing frame-level is. `pool` is every labelled training copy; only the PathCRF-unseen matches
    train the classifier and the verifier, since they read PathCRF."""

    def __init__(self, pool: list, config: dict):
        self.pool, self.config = pool, config
        self.params = dict(evidence.DEFAULT_PARAMS, **config["evidence"])
        self.matrices = [dead_residual.feature_matrix(r.match, r.feats, r.z_dead) for r in pool]
        self.unseen = sorted({r.group for r in pool if r.group in PATHCRF_UNSEEN})
        self.residuals: dict = {}
        self.classifiers: dict = {}

    def residual(self, excluded: frozenset):
        if excluded not in self.residuals:
            keep = [k for k, r in enumerate(self.pool) if r.group not in excluded]
            self.residuals[excluded] = dead_residual.fit([self.pool[k] for k in keep],
                                                         [self.matrices[k] for k in keep])
        return self.residuals[excluded]

    def logits(self, row, excluded: frozenset) -> tuple[np.ndarray, np.ndarray, dict]:
        """(z_team, z_dead, PathCRF outputs) of a row with a residual fitted without `excluded`."""
        pcrf: dict = {}

        def home_probability(match, alive):
            pcrf.update(pathcrf_outputs(row, alive))
            return pcrf["p_home"]

        blocks = [(a, b) for _, a, b in row.match.halves]
        z_team, z_dead = stack.logits(
            row.feats, row.match, self.config["evidence"], self.residual(excluded),
            self.config["v6"]["w_pathcrf"], lambda e: decode_blocks(e, blocks, self.config, row.feats),
            home_probability)
        return z_team, z_dead, pcrf

    def classifier(self, excluded: frozenset) -> handover.HandoverModel:
        """Hand-over classifier without `excluded`; each row's residual also excludes its own match."""
        if excluded not in self.classifiers:
            tables = []
            for r in self.pool:
                if r.group in PATHCRF_UNSEEN and r.group not in excluded:
                    z_team, z_dead, pcrf = self.logits(r, excluded | {r.group})
                    tables.append((r.group, *handover.training_table(
                        r.match, r.feats, r.codes, z_team, z_dead, pcrf, self.params, (*r.group, r.seed))))
            self.classifiers[excluded] = handover.fit(tables)
        return self.classifiers[excluded]

    def verifier_tables(self, row, excluded: frozenset) -> list:
        """[(X, y)] per half of a row, on the v7 decodes of models fitted without `excluded`."""
        z_team, z_dead, pcrf = self.logits(row, excluded)
        ev, scale = v7_evidence(self.config, self.classifier(excluded), row.match, row.feats, z_team, z_dead,
                                pcrf, self.params)
        out = []
        for a, b in [(a, b) for _, a, b in row.match.halves]:
            decode = verifier.viterbi_decode(self.config["decoder"], scale[a:b + 1])
            prod, edits, X = verifier.block_table(ev[a:b + 1], row.feats, a, decode)
            out.append((X, verifier.labels(np.asarray(row.codes[a:b + 1]), prod, edits)))
        return out

    def fit_verifier(self, excluded: frozenset = frozenset()) -> verifier.VerifierModel:
        """Verifier without `excluded`; each row's stack also excludes its own match."""
        tables = []
        for r in self.pool:
            if r.group in PATHCRF_UNSEEN and r.group not in excluded:
                tables += self.verifier_tables(r, excluded | {r.group})
        return verifier.fit(tables)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    started = time.time()
    pool = [build_row((provider, mid), seed, match, codes, config)
            for provider, mid, seed, match, codes in external_rows(tracker_noise=True)]
    if len(pool) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} external rows; fetch them with scripts/fetch_metrica.py "
                         "and scripts/fetch_idsse.py + scripts/idsse_prepare.py")
    print(f"{len(pool)} external rows in {time.time() - started:.0f}s", flush=True)
    nested = Nested(pool, config)
    residual = nested.residual(frozenset())
    classifier = nested.classifier(frozenset())
    print(f"classifier done ({time.time() - started:.0f}s)", flush=True)
    model = nested.fit_verifier()
    print(f"verifier done ({time.time() - started:.0f}s)", flush=True)
    for section, name, value in (("v6", "dead_residual", residual), ("v7", "handover", classifier),
                                 ("v6", "verifier", model)):
        path = ROOT / config[section][name]["path"]
        print(f"{name}: {path.relative_to(ROOT)} sha256 {io.save_model(path, value)}")
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
