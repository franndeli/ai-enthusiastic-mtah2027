#!/usr/bin/env python3
"""Regenerate the submission from the provided data with one command.

    scripts/py run.py --out evaluation/results/<name>.csv [--config config/possession.json]

Deterministic: no randomness anywhere in the pipeline; running twice yields the identical file
(PathCRF runs on CPU with fixed threads and rounded probabilities; the same file was produced on
macOS arm64 and Linux x86_64, docs/TECHNICAL.md section 7).
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

from possession import (LABELS, X, decoder, evidence, features, handover, io, pathcrf, stack,  # noqa: E402
                        structure, verifier)

DEFAULT_CONFIG = ROOT / "config" / "possession.json"
V6_CONFIG = ROOT / "config" / "possession_v6.json"     # v6, reproducible: default of its trainer and evaluator
RULES_CONFIG = ROOT / "config" / "possession_v5.json"   # rule-only v5: default of the rule evaluators


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


def decode_blocks(log_evidence: np.ndarray, blocks: list[tuple[int, int]],
                  config: dict, feats: dict | None = None) -> np.ndarray:
    """Decode inclusive frame blocks independently; X elsewhere, with shared decoder options."""
    decoder_config = dict(config["decoder"])
    cost = decoder.switch_matrix(**decoder_config.pop("switch_cost"))
    discount = float(decoder_config.pop("restart_discount", 0.0))
    scale = restart_cost_scale((feats or {}).get("restart_prior"), discount)
    codes = np.full(len(log_evidence), X, dtype=np.int8)
    for first, last in blocks:
        if not 0 <= first <= last < len(log_evidence):
            raise ValueError("decode block bounds outside evidence shape")
        block_scale = None if scale is None else scale[first:last + 1]
        codes[first:last + 1] = decoder.decode(
            log_evidence[first:last + 1], cost, cost_scale=block_scale, **decoder_config)
    return codes


def label_codes_from_config(config: dict, match: io.Match, tags: list | None = None,
                            models: dict | None = None) -> np.ndarray:
    """Codes for every frame. tags: Smart Tagging intervals (io.load_smart_tagging) or None.
    models: the config's learned components (load_models(config)); loaded when None."""
    feats = features.compute(match, tags, config.get("features"))
    models = load_models(config) if models is None else models

    def rule_decode(log_evidence: np.ndarray) -> np.ndarray:
        def decode_block(block_evidence: np.ndarray, start: int) -> np.ndarray:
            block_feats = None
            if "restart_prior" in feats:
                block_feats = {"restart_prior": feats["restart_prior"][start:start + len(block_evidence)]}
            return decode_blocks(block_evidence, [(0, len(block_evidence) - 1)], config, block_feats)
        return structure.decode_halves(log_evidence, decode_block)

    if "handover" in models:
        return v7_codes(config, match, feats, models, rule_decode)
    log_evidence = stack.log_evidence(feats, match, config.get("evidence") or {},
                                      models.get("dead_residual"),
                                      (config.get("v6") or {}).get("w_pathcrf", 0.0), rule_decode)
    model = models.get("verifier")
    if model is None:
        return rule_decode(log_evidence)
    viterbi = verifier.viterbi_decode(config["decoder"])
    return structure.decode_halves(
        log_evidence, lambda block, start: verifier.decode_block(model, block, feats, start, viterbi))


def v7_codes(config: dict, match: io.Match, feats: dict, models: dict, rule_decode) -> np.ndarray:
    """The v6 stack with learned hand-over costs and memory, then the spell-edit verifier (D017)."""
    params = dict(evidence.DEFAULT_PARAMS, **(config.get("evidence") or {}))
    pcrf: dict = {}

    def home_probability(pcrf_match, alive):
        pcrf.update(pathcrf.outputs(pcrf_match, alive))
        return pcrf["p_home"]

    z_team, z_dead = stack.logits(feats, match, config.get("evidence") or {}, models["dead_residual"],
                                  config["v6"]["w_pathcrf"], rule_decode, home_probability)
    log_evidence, scale = v7_evidence(config, models["handover"], match, feats, z_team, z_dead, pcrf, params)
    return structure.decode_halves(log_evidence, lambda block, start: verifier.decode_block(
        models["verifier"], block, feats, start,
        verifier.viterbi_decode(config["decoder"], scale[start:start + len(block)])))


def v7_evidence(config: dict, model, match: io.Match, feats: dict, z_team: np.ndarray, z_dead: np.ndarray,
                pcrf: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(log evidence, per-frame switch-cost multipliers) from the stack's logits. Shared by serving,
    training (scripts/train_v7.py) and evaluation so the three cannot drift apart."""
    v7 = config["v7"]
    to_home, to_away = handover.logits(model, match, feats, z_team, z_dead, pcrf, params)
    z = z_team + v7["memory_w"] * handover.memory(to_home, to_away, v7["memory_threshold"], v7["memory_tau_frames"])
    scale = handover.cost_scale(to_home, to_away, config["decoder"]["switch_cost"]["team_switch"],
                                v7["cost_a"], v7["cost_b"], v7["cost_lo"], v7["cost_hi"])
    return evidence.combine(z, z_dead, params), scale


class MissingModel(FileNotFoundError):
    """A model file named in the config is absent (its message names repository paths only)."""


def load_models(config: dict) -> dict:
    """The learned components named in the config's "v6" and "v7" sections, hash-checked ({} for v5)."""
    sections = {**(config.get("v6") or {}), **(config.get("v7") or {})}
    models = {}
    for name in ("dead_residual", "verifier", "handover"):
        if name in sections:
            path = ROOT / sections[name]["path"]
            if not path.is_file():
                raise MissingModel(f"{sections[name]['path']} is missing; regenerate it with "
                                   "scripts/py scripts/train_v7.py")
            models[name] = io.load_model(path, sections[name]["sha256"])
    return models


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="submission CSV to write")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-cache", action="store_true",
                        help="re-parse the tracking instead of using the cached arrays")
    args = parser.parse_args()

    try:
        started = time.time()
        config = load_config(Path(args.config))
        models = load_models(config)
        match = io.load_match(cache=not args.no_cache)
        print("loaded tracking: {} frames x {} slots in {:.1f}s".format(
            match.n_frames, match.n_slots, time.time() - started))

        codes = label_codes_from_config(config, match, load_tags(config), models)
        io.write_submission(args.out, codes)
        counts = {LABELS[k]: int(np.sum(codes == k)) for k in range(4)}
        print("wrote {}  labels {}  in {:.1f}s".format(args.out, counts, time.time() - started))
    except MissingModel as error:
        print(f"pipeline failed: {error}", file=sys.stderr)   # names repository files only
        raise SystemExit(1) from None
    except Exception as error:
        print(f"pipeline failed: {type(error).__name__} "
              "(loading, decoding or writing failed; details suppressed for data safety)",
              file=sys.stderr)
        raise SystemExit(1) from None

    return 0


if __name__ == "__main__":
    sys.exit(main())
