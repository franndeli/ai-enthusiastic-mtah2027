#!/usr/bin/env python3
"""Regenerate, validate and score a candidate; print one aggregate table row."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from possession import io
from run import DEFAULT_CONFIG, label_codes_from_config, load_config, load_tags
from stress_test import run_stress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("name must contain only letters, digits, underscore, dot or hyphen")
    if args.name.startswith("baseline_"):
        parser.error("baseline_ names are reserved for upstream reference files")
    config, match, gt = load_config(args.config), io.load_match(), io.load_public_gt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        codes = label_codes_from_config(config, match, load_tags(config))
    path = ROOT / "evaluation" / "results" / f"{args.name}.csv"
    io.write_submission(str(path), codes)
    invalid = pm.validate_submission(pm.read_label_csv(str(path)), match.n_frames)
    if invalid is not None:
        raise ValueError("candidate failed submission validation")
    result = pm.evaluate(gt, codes, bootstrap_n=1000)
    half_runs = [(first, last + 1) for _, first, last in match.halves]
    full_stats = pm.possession_stats(pm.fill_abstentions(codes, half_runs), half_runs)
    stress = {row["mask"]: row for row in run_stress(config, match, gt, seeds=1,
                                                   native_codes=codes)}
    print(f"{args.name} | {result['spell_f1_mean']:.4f} "
          f"[{result['spell_bootstrap_low']:.4f}, {result['spell_bootstrap_high']:.4f}] | "
          f"{result['macro_f1']:.4f} | {full_stats['n_spells']} | "
          f"{full_stats['mean_duration_s']:.2f} | {stress['transplant']['spell_mean']:.4f} | "
          f"{stress['transition-centred(10s)']['spell_mean']:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"score failed: {type(exc).__name__} (details suppressed for data safety)", file=sys.stderr)
        raise SystemExit(1)
