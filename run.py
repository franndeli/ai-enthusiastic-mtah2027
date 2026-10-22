#!/usr/bin/env python3
"""Regenerate the submission from the provided data with one command.

    scripts/py run.py --out evaluation/results/<name>.csv [--config config/possession.json]

Deterministic: no randomness anywhere in the pipeline; running twice yields the identical file.
Prints aggregates only (counts, timings), never frame-level content.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from possession import LABELS, decoder, evidence, features, io, structure  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "possession.json"


def load_config(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def label_codes_from_config(config: dict, match: io.Match) -> np.ndarray:
    feats = features.compute(match)
    log_evidence = evidence.scores(feats, config.get("evidence"))
    cost = decoder.switch_matrix(**config["decoder"]["switch_cost"])
    return structure.decode_halves(log_evidence, lambda e: decoder.decode(e, cost))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="submission CSV to write")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    started = time.time()
    config = load_config(Path(args.config))
    match = io.load_match()
    print("loaded tracking: {} frames x {} slots in {:.1f}s".format(
        match.n_frames, match.n_slots, time.time() - started))

    codes = label_codes_from_config(config, match)
    io.write_submission(args.out, codes)
    counts = {LABELS[k]: int(np.sum(codes == k)) for k in range(4)}
    print("wrote {}  labels {}  in {:.1f}s".format(args.out, counts, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
