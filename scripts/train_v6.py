#!/usr/bin/env python3
"""Train the two learned v6 components on external public data only and write them to models/.

    scripts/py scripts/train_v6.py [--config config/possession.json]   (needs torch: PathCRF)

Data: the Metrica sample games and the IDSSE matches (scripts/fetch_metrica.py,
scripts/fetch_idsse.py), each clean and with target-matched noise seeds 0 and 1
(possession.external.inject_matched_noise). Nothing about the competition match is read.

1. Dead residual (possession/dead_residual.py): fitted on all 30 rows.
2. Spell-edit verifier (possession/verifier.py): fitted on the edits of the rows of the five
   matches PathCRF never trained on (IDSSE J03WR9, J03WQQ and the three Metrica games), with the
   full v6 evidence: each row's dead residual from a model trained without that row's match, and
   PathCRF team evidence inside the v5 alive stretches.
Prints the SHA-256 of both files for config/possession.json. Deterministic (fixed seeds); PathCRF
probabilities are cached in the work directory.
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

from possession import dead_residual, evidence, external, features, io, metrica, stack, verifier  # noqa: E402
from run import DEFAULT_CONFIG, decode_blocks  # noqa: E402

SEEDS = (0, 1)
EXPECTED_ROWS = 10 * (1 + len(SEEDS))
PATHCRF_UNSEEN = {("idsse", "J03WR9"), ("idsse", "J03WQQ"), ("metrica", "Sample_Game_1"),
                  ("metrica", "Sample_Game_2"), ("metrica", "Sample_Game_3")}


def external_rows():
    """[(provider, match_id, seed or None, Match, codes)] clean + matched-noise copies."""
    rows = []
    sources = [("metrica", g, metrica.load_game(g)) for g in metrica.available_games()]
    sources += [("idsse", m, external.load_external(m)) for m in external.available_matches()]
    for provider, mid, ext in sources:
        rows.append((provider, mid, None, ext.match, ext.codes))
        noise = external.MATCHED_NOISE[provider]
        for seed in SEEDS:
            rows.append((provider, mid, seed, external.inject_matched_noise(
                ext.match, seed, noise["share"], ext.fps, noise["ball_err_m"]), ext.codes))
    return rows


def pathcrf_version() -> str:
    """Identity of the PathCRF stage (code, re-tracker, checkpoint) for the probability cache."""
    files = [ROOT / "possession" / "pathcrf.py", ROOT / "possession" / "retrack.py",
             ROOT / "third_party" / "pathcrf" / "saved" / "140" / "model" / "state_dict_best.pt"]
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:12]


def pathcrf_probability(provider, mid, seed, match, alive):
    from possession import pathcrf
    key = pathcrf_version() + "_" + hashlib.sha256(np.packbits(alive).tobytes()).hexdigest()[:16]
    path = io.work_dir() / "v6_train" / f"pathcrf_{provider}_{mid}_{seed}_{key}.npz"
    if path.is_file():
        return np.load(path)["p"]
    p = pathcrf.home_probability(match, alive)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, p=p)
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    params = dict(evidence.DEFAULT_PARAMS, **config["evidence"])
    rule = dict(params, w_fresh_exit=0.0)
    started = time.time()
    rows = []
    for provider, mid, seed, match, codes in external_rows():
        feats = features.compute(match, None, config.get("features"))
        rows.append(dead_residual.TrainingRow((provider, mid), match, feats,
                                              evidence.dead_logit(feats, rule), codes, seed))
    print(f"{len(rows)} external rows in {time.time() - started:.0f}s", flush=True)
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(f"expected {EXPECTED_ROWS} external rows (3 Metrica + 7 IDSSE matches x "
                         "clean + 2 noise seeds); fetch them with scripts/fetch_metrica.py and "
                         "scripts/fetch_idsse.py + scripts/idsse_prepare.py")

    models = {"dead_residual": dead_residual.fit(rows)}
    viterbi = verifier.viterbi_decode(config["decoder"])
    tables = []
    groups = list(dict.fromkeys(r.group for r in rows if r.group in PATHCRF_UNSEEN))
    for group in groups:
        fold = dead_residual.fit([r for r in rows if r.group != group])
        for r in (r for r in rows if r.group == group):
            blocks = [(first, last) for _, first, last in r.match.halves]
            ev = stack.log_evidence(
                r.feats, r.match, config["evidence"], fold, config["v6"]["w_pathcrf"],
                lambda e, r=r, blocks=blocks: decode_blocks(e, blocks, config, r.feats),
                lambda match, alive, r=r: pathcrf_probability(*r.group, r.seed, match, alive))
            for a, b in blocks:
                prod, edits, X = verifier.block_table(ev[a:b + 1], r.feats, a, viterbi)
                tables.append((X, verifier.labels(np.asarray(r.codes[a:b + 1]), prod, edits)))
        print(f"verifier tables: {group[0]} {group[1]} done ({time.time() - started:.0f}s)", flush=True)
    models["verifier"] = verifier.fit(tables)
    for name, model in models.items():
        path = ROOT / config["v6"][name]["path"]
        print(f"{name}: {path.relative_to(ROOT)} sha256 {io.save_model(path, model)}")
    print(f"done in {time.time() - started:.0f}s ({sum(len(t[1]) for t in tables)} verifier edits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
