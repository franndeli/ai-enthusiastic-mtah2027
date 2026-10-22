#!/usr/bin/env python3
"""Leave-one-match-out IDSSE and our-match comparison of rule, learned and blended evidence.

    scripts/py scripts/learned_eval.py [--matches ...] [--weight 0.5] [--bootstrap 0] [--no-ours]

1. IDSSE: each match is scored (clean and noisy, eval seed 0) with a model trained on the others.
2. Our match: the final model (trained on all IDSSE matches) on the public window and the stress
   masks of scripts/stress_test.py, for rules / learned / blend. The learned evidence is swapped in
   by patching evidence.scores for the duration of the call, so run.py stays untouched.
Prints aggregates only.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation import possession_metrics as pm  # noqa: E402
from possession import evidence, external, features, io, learned  # noqa: E402
from run import DEFAULT_CONFIG, decode_blocks, label_codes_from_config, load_config, load_tags  # noqa: E402
from stress_test import run_stress  # noqa: E402
from train_evidence import build_rows, cached_matches  # noqa: E402

EVAL_SEED = 0


def make_scorers(model: learned.LearnedModel, config: dict, weight: float) -> dict:
    """name -> scores(features, params) for rules, learned only, and the blend."""
    rule_params, rule_scores = config.get("evidence"), evidence.scores
    block = config.get("learned", {})

    def with_weight(w: float):
        return lambda feats, params=None: learned.scores(
            feats, dict(block, weight=w), rule_params=rule_params, model=model)

    return {"rules": lambda feats, params=None: rule_scores(feats, rule_params),
            "learned": with_weight(1.0), "blend": with_weight(weight)}


def scaled_config(config: dict, scale: float) -> dict:
    """Copy of config whose decoder switch costs are multiplied by `scale`."""
    out = dict(config, decoder=dict(config["decoder"]))
    out["decoder"]["switch_cost"] = {k: v * scale for k, v in config["decoder"]["switch_cost"].items()}
    return out


def idsse_codes(match, log_evidence: np.ndarray, config: dict,
                feats: dict | None = None) -> np.ndarray:
    return decode_blocks(log_evidence, [(first, last) for _, first, last in match.halves],
                         config, feats)


@contextmanager
def patched_evidence(scorer):
    original = evidence.scores
    evidence.scores = scorer
    try:
        yield
    finally:
        evidence.scores = original


def mean_row(rows: list[dict]) -> dict:
    return {key: float(np.mean([r[key] for r in rows])) for key in ("spell_f1_mean", "macro_f1")}


def idsse_table(ids: list[str], folds: dict, config: dict, weight: float, scales: list[float],
                bootstrap: int) -> None:
    """Rules at scale 1; learned and blend at every scale in `scales` (decoder switch-cost factor)."""
    results = {"clean": {}, "noisy": {}}
    for mid in ids:
        ext = external.load_external(mid)
        scorers = make_scorers(folds[mid], config, weight)
        for kind, match in (("clean", ext.match),
                            ("noisy", external.inject_noise(ext.match, EVAL_SEED, fps=ext.fps))):
            feats = features.compute(match)
            row = {}
            for name, scorer in scorers.items():
                log_evidence = scorer(feats)
                for scale in ([1.0] if name == "rules" else scales):
                    label = name if name == "rules" else "{} x{:g}".format(name, scale)
                    row[label] = pm.evaluate(ext.codes, idsse_codes(
                        match, log_evidence, scaled_config(config, scale), feats), bootstrap_n=bootstrap)
            results[kind][mid] = row
    for kind in results:
        print("\nIDSSE leave-one-match-out, {} tracking (eval seed {}): spellF1 / macroF1".format(
            kind, EVAL_SEED))
        names = list(next(iter(results[kind].values())))
        print("{:8}".format("match") + "".join("{:>16}".format(n) for n in names))
        for mid, by_name in results[kind].items():
            print("{:8}".format(mid) + "".join(
                "{:>9.3f}/{:.3f}".format(by_name[n]["spell_f1_mean"], by_name[n]["macro_f1"])
                for n in names))
        means = {n: mean_row([r[n] for r in results[kind].values()]) for n in names}
        print("{:8}".format("MEAN") + "".join(
            "{:>9.3f}/{:.3f}".format(means[n]["spell_f1_mean"], means[n]["macro_f1"])
            for n in names))
        print("{:8}".format("poss/gt") + "".join(
            "{:>16.2f}".format(np.mean([r[n]["n_pred_spells"] / r[n]["n_gt_spells"]
                                        for r in results[kind].values()])) for n in names))


def our_match_table(final: learned.LearnedModel, config: dict, weight: float,
                    scales: dict, bootstrap: int, seeds: int) -> None:
    match, gt, tags = io.load_match(), io.load_public_gt(), load_tags(config)
    print("\nour match: public window and stress masks (spellF1 mean / min, macroF1, possessions)")
    for name, scorer in make_scorers(final, config, weight).items():
        run_config = scaled_config(config, scales.get(name, 1.0))
        with patched_evidence(scorer), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            codes = label_codes_from_config(run_config, match, tags)
            result = pm.evaluate(gt, codes, bootstrap_n=bootstrap)
            half_runs = [(first, last + 1) for _, first, last in match.halves]
            full = pm.possession_stats(pm.fill_abstentions(codes, half_runs), half_runs)
            stress = run_stress(run_config, match, gt, seeds=seeds, native_codes=codes)
        interval = ("" if not bootstrap else " [{:.3f}, {:.3f}]".format(
            result["spell_bootstrap_low"], result["spell_bootstrap_high"]))
        print("{}: public spellF1 {:.3f}{} macroF1 {:.3f}; full match {} possessions, mean {:.1f}s".format(
            name, result["spell_f1_mean"], interval, result["macro_f1"],
            full["n_spells"], full["mean_duration_s"]))
        for row in stress:
            print("   {:26} {:.3f} / {:.3f}  macro {:.3f}  poss {:.1f}  missing {:.2f}".format(
                row["mask"], row["spell_mean"], row["spell_min"], row["macro_f1"],
                row["possessions"], row["missing_share"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--weight", type=float, default=0.5, help="blend weight of the learned part")
    parser.add_argument("--switch-scales", type=float, nargs="+", default=[1.0],
                        help="IDSSE: decoder switch-cost factors tried for learned/blend")
    parser.add_argument("--our-scale", type=float, default=1.0,
                        help="our match: switch-cost factor for learned")
    parser.add_argument("--blend-scale", type=float, default=1.0,
                        help="our match: switch-cost factor for the blend")
    parser.add_argument("--no-idsse", action="store_true",
                        help="skip leave-one-match-out; use the model file (--model or default)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5, help="transplant mask seeds")
    parser.add_argument("--no-ours", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ids = args.matches or cached_matches()
    if args.no_idsse:
        final = learned.load_model(args.model or learned.default_model_path())
    else:
        final, folds = learned.fit_with_calibration(build_rows(ids))
        print("temperature {}".format(final.temperature))
        idsse_table(ids, folds, config, args.weight, args.switch_scales, args.bootstrap)
    if not args.no_ours:
        our_match_table(final, config, args.weight,
                        {"learned": args.our_scale, "blend": args.blend_scale},
                        args.bootstrap, args.seeds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - the data guard: no values in tracebacks
        print(f"learned_eval failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1)
