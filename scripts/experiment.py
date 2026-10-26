#!/usr/bin/env python3
"""Compare config variants on the external judges (Metrica sample, IDSSE) in one process.

    scripts/py scripts/experiment.py --variant base --variant touch:evidence.w_touch=1.0 \
        [--judges metrica,idsse] [--noise] [--seeds 0 1] [--public]

A variant is name:key=value,key=value with dotted keys into the config (evidence.* / decoder.*
/ decoder.switch_cost.*). Prints one row per variant and judge: mean spellF1, macroF1, predicted
possessions, mean spell duration. The public window is a correctness check, not a selector.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

import possession_metrics as pm  # noqa: E402
from possession import external, metrica  # noqa: E402
from idsse_eval import predict  # noqa: E402
from run import DEFAULT_CONFIG, load_config  # noqa: E402


def parse_variant(text: str) -> tuple[str, dict]:
    name, _, rest = text.partition(":")
    overrides = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        overrides[key.strip()] = json.loads(value)
    return name, overrides


def apply(config: dict, overrides: dict) -> dict:
    out = copy.deepcopy(config)
    for key, value in overrides.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_judges(names: list[str], noise: bool, seeds: list[int]) -> dict:
    """{judge: [(label, match, codes, fps)]} with noisy copies when requested."""
    judges = {}
    for name in names:
        items = []
        if name == "metrica":
            for game in metrica.available_games():
                ext = metrica.load_game(game)
                items.append((game, ext.match, ext.codes, ext.fps, 0.15))
        elif name == "idsse":
            for mid in external.available_matches():
                ext = external.load_external(mid)
                items.append((mid, ext.match, ext.codes, ext.fps, 0.41))
        else:
            raise ValueError("unknown judge " + name)
        rows = [(label, m, codes) for label, m, codes, _, _ in items]
        if noise:
            for label, m, codes, fps, share in items:
                for seed in seeds:
                    params = external.NoiseParams(ball_missing_share=share)
                    rows.append((f"{label}~{seed}", external.inject_noise(m, seed, params, fps),
                                 codes))
        judges[name] = rows
    return judges


def score_rows(rows, config) -> dict:
    spell, macro, n_pred, dur, clean_spell, noisy_spell, per = [], [], [], [], [], [], []
    for label, match, codes in rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = pm.evaluate(codes, predict(match, config), bootstrap_n=0)
        spell.append(result["spell_f1_mean"])
        macro.append(result["macro_f1"])
        n_pred.append(result["n_pred_spells"])
        dur.append(result["mean_spell_duration_s"])
        (noisy_spell if "~" in label else clean_spell).append(result["spell_f1_mean"])
        per.append("{}={:.3f}/{:.3f}".format(label[-6:], result["spell_f1_mean"], result["macro_f1"]))
    return {"per": " ".join(per),"spell": float(np.mean(spell)), "macro": float(np.mean(macro)),
            "clean": float(np.mean(clean_spell)) if clean_spell else float("nan"),
            "noisy": float(np.mean(noisy_spell)) if noisy_spell else float("nan"),
            "n_pred": float(np.mean(n_pred)), "dur": float(np.mean(dur))}


def public_row(config) -> dict:
    from possession import io
    from run import label_codes_from_config, load_tags
    from stress_test import run_stress
    match, gt = io.load_match(), io.load_public_gt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        codes = label_codes_from_config(config, match, load_tags(config))
        result = pm.evaluate(gt, codes, bootstrap_n=0)
        half_runs = [(first, last + 1) for _, first, last in match.halves]
        full = pm.possession_stats(pm.fill_abstentions(codes, half_runs), half_runs)
        stress = {row["mask"]: row for row in run_stress(config, match, gt, seeds=3,
                                                       native_codes=codes)}
    return {"spell": result["spell_f1_mean"], "macro": result["macro_f1"],
            "n_pred": result["n_pred_spells"], "full_n": full["n_spells"],
            "full_dur": full["mean_duration_s"],
            "transplant": stress["transplant"]["spell_mean"],
            "tc10": stress["transition-centred(10s)"]["spell_mean"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", action="append", required=True)
    parser.add_argument("--judges", default="metrica,idsse")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--per-match", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config)
    judges = load_judges(args.judges.split(","), args.noise, args.seeds)
    print("{:24} {:8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6}".format(
        "variant", "judge", "spell", "clean", "noisy", "macro", "poss", "dur"))
    for text in args.variant:
        name, overrides = parse_variant(text)
        config = apply(base, overrides)
        for judge, rows in judges.items():
            started = time.time()
            r = score_rows(rows, config)
            print("{:24} {:8} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:6.0f} {:6.1f}  ({:.0f}s)".format(
                name, judge, r["spell"], r["clean"], r["noisy"], r["macro"], r["n_pred"],
                r["dur"], time.time() - started))
            if args.per_match:
                print("    " + r["per"])
        if args.public:
            r = public_row(config)
            print("{:24} {:8} {:7.4f} {:>7} {:>7} {:7.4f} {:6.0f} {:>6}  full {} spells "
                  "{:.1f}s  transplant {:.4f}  tc10 {:.4f}".format(
                      name, "public", r["spell"], "-", "-", r["macro"], r["n_pred"], "-",
                      r["full_n"], r["full_dur"], r["transplant"], r["tc10"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
