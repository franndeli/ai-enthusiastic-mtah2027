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


def load_tags(config: dict) -> list | None:
    """Smart Tagging intervals unless the config switches them off."""
    return io.load_smart_tagging() if config.get("use_smart_tagging", True) else None


def restart_cost_scale(restart_prior: np.ndarray | None, discount: float) -> np.ndarray | None:
    """(n,3,3) multipliers of the switch costs, or None when there is nothing to scale."""
    if restart_prior is None or discount <= 0.0:
        return None
    prior = np.nan_to_num(np.asarray(restart_prior, dtype=np.float64), nan=0.0)
    scale = np.ones((len(prior), 3, 3), dtype=np.float64)
    scale[:, 2, 0] = scale[:, 2, 1] = 1.0 - discount * np.clip(prior, 0.0, 1.0)
    return scale


def label_codes_from_config(config: dict, match: io.Match, tags: list | None = None) -> np.ndarray:
    """Codes for every frame. tags: Smart Tagging intervals (io.load_smart_tagging) or None."""
    feats = features.compute(match, tags)
    log_evidence = evidence.scores(feats, config.get("evidence"))
    decoder_config = dict(config["decoder"])
    cost = decoder.switch_matrix(**decoder_config.pop("switch_cost"))
    # Every other decoder key is passed through (e.g. min_duration_frames). A per-frame
    # 'restart_prior' feature (0..1, peaking where a tagged set piece ends) makes leaving D
    # cheaper there: cost(D->team) *= 1 - restart_discount * prior. Other transitions unchanged.
    restart_discount = float(decoder_config.pop("restart_discount", 0.0))
    scale_all = restart_cost_scale(feats.get("restart_prior"), restart_discount)

    def decode_block(block_evidence: np.ndarray, start: int) -> np.ndarray:
        scale = None if scale_all is None else scale_all[start:start + len(block_evidence)]
        return decoder.decode(block_evidence, cost, cost_scale=scale, **decoder_config)

    return structure.decode_halves(log_evidence, decode_block)


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

    codes = label_codes_from_config(config, match, load_tags(config))
    io.write_submission(args.out, codes)
    counts = {LABELS[k]: int(np.sum(codes == k)) for k in range(4)}
    print("wrote {}  labels {}  in {:.1f}s".format(args.out, counts, time.time() - started))
    return 0


if __name__ == "__main__":
    sys.exit(main())
