#!/usr/bin/env python3
"""Run the possession pipeline on the IDSSE matches and score it against their labels.

    scripts/py scripts/idsse_eval.py --config config/possession.json [--noise] [--seed 0]
                                     [--matches J03WPY ...] [--bootstrap 1000]

features.compute -> evidence.scores -> decoder.decode per half, scored with
evaluation/possession_metrics.evaluate. Nothing is trained, so every match is held out. Prints the
clean table, and the noisy one with --noise (inject_noise, same seed for every match).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import evidence, features  # noqa: E402
from possession import external  # noqa: E402
from possession.io import Match  # noqa: E402
from run import decode_blocks  # noqa: E402


def predict(match: Match, config: dict) -> np.ndarray:
    """Pipeline codes for one match: each half decoded on its own, X outside the halves."""
    feats = features.compute(match)
    log_evidence = evidence.scores(feats, config.get("evidence"))
    return decode_blocks(log_evidence, [(first, last) for _, first, last in match.halves],
                         config, feats)


def score_match(match: Match, gt: np.ndarray, config: dict, bootstrap: int) -> dict:
    started = time.time()
    result = pm.evaluate(gt, predict(match, config), bootstrap_n=bootstrap)
    result["seconds"] = time.time() - started
    return result


def print_table(title: str, rows: dict[str, dict], bootstrap: int) -> None:
    print("\n{}".format(title))
    print("{:8} {:>8} {:>8} {:>6} {:>6} {:>6} {:>7} {:>13} {:>17} {:>6}{}".format(
        "match", "spellF1", "macroF1", "D P", "D R", "D F1", "aliveF1", "poss pred/gt",
        "mean dur pred/gt", "sec",
        "  spell 5-95%" if bootstrap else ""))
    for mid, r in rows.items():
        print("{:8} {:8.3f} {:8.3f} {:6.3f} {:6.3f} {:6.3f} {:7.3f} {:>13} {:>17} {:6.1f}{}".format(
            mid, r["spell_f1_mean"], r["macro_f1"], r["per_class"]["D"]["precision"],
            r["per_class"]["D"]["recall"], r["per_class"]["D"]["f1"], r["alive_dead_f1"],
            "{}/{}".format(r["n_pred_spells"], r["n_gt_spells"]),
            "{:.1f}/{:.1f}s".format(r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"]),
            r["seconds"],
            "  [{:.3f}, {:.3f}]".format(r["spell_bootstrap_low"], r["spell_bootstrap_high"])
            if bootstrap else ""))
    mean = lambda key: float(np.mean([r[key] for r in rows.values()]))  # noqa: E731
    mean_d = lambda key: float(np.mean([r["per_class"]["D"][key] for r in rows.values()]))  # noqa: E731
    print("{:8} {:8.3f} {:8.3f} {:6.3f} {:6.3f} {:6.3f} {:7.3f} {:>13} {:>17} {:6.1f}".format(
        "MEAN", mean("spell_f1_mean"), mean("macro_f1"), mean_d("precision"), mean_d("recall"),
        mean_d("f1"), mean("alive_dead_f1"),
        "{:.0f}/{:.0f}".format(mean("n_pred_spells"), mean("n_gt_spells")),
        "{:.1f}/{:.1f}s".format(mean("mean_spell_duration_s"), mean("gt_mean_spell_duration_s")),
        mean("seconds")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "config" / "possession.json"))
    parser.add_argument("--noise", action="store_true", help="also score the noisy versions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    ids = args.matches or external.available_matches()
    if not ids:
        print("no IDSSE match on disk; run scripts/fetch_idsse.py first")
        return 1
    clean, noisy = {}, {}
    for mid in ids:
        ext = external.load_external(mid)
        clean[mid] = score_match(ext.match, ext.codes, config, args.bootstrap)
        if args.noise:
            degraded = external.inject_noise(ext.match, args.seed, fps=ext.fps)
            noisy[mid] = score_match(degraded, ext.codes, config, args.bootstrap)
    print_table("clean tracking (config {})".format(Path(args.config).name), clean, args.bootstrap)
    if args.noise:
        print_table("noisy tracking (seed {})".format(args.seed), noisy, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
