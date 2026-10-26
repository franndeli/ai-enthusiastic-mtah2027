#!/usr/bin/env python3
"""Per-possession diagnostics of a config on the public window. Aggregates only.

For each public ground-truth possession: duration, ball-visible share, best same-team IoU with a
predicted spell, and the start/end offsets (frames, + = prediction late) of that best match.
Then transition latency by type. Nothing frame-level is printed.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from possession import io  # noqa: E402
from run import DEFAULT_CONFIG, label_codes_from_config, load_config, load_tags  # noqa: E402

LABELS = "HADX"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config, match, gt = load_config(args.config), io.load_match(), io.load_public_gt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        codes = label_codes_from_config(config, match, load_tags(config))
    graded = np.flatnonzero(gt != 3)
    runs = [(int(graded[0]), int(graded[-1]) + 1)]
    pred = pm.fill_abstentions(codes, runs)
    # diagnostic only: how much of the error is dead-ball handling? Give the prediction the true
    # D frames (and X where it said D on an alive frame, which the scorer forward-fills).
    oracle = np.where(gt == 2, 2, np.where((pred == 2) & (gt < 2), 3, pred)).astype(np.int8)
    oracle = pm.fill_abstentions(oracle, runs)
    print("spellF1 {:.4f}  with oracle dead frames {:.4f}  macroF1 {:.4f}".format(
        pm.spell_scores(gt, pred, runs)["mean_f1"], pm.spell_scores(gt, oracle, runs)["mean_f1"],
        pm.evaluate(gt, pred, bootstrap_n=0)["macro_f1"]))
    gt_sp = pm.possession_spells(gt, runs)
    pr_sp = pm.possession_spells(pred, runs)
    present = ~np.isnan(match.ball_xy[:, 0])
    print("gt possessions {}  predicted {}".format(len(gt_sp), len(pr_sp)))
    print(" i lab   dur_s ball%  bestIoU  start_off  end_off")
    for i, (a, b, k) in enumerate(gt_sp):
        best, off = 0.0, (None, None)
        for (c, d, kk) in pr_sp:
            if kk != k or d <= a or b <= c:
                continue
            iou = (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
            if iou > best:
                best, off = iou, (c - a, d - b)
        print("{:2d} {}   {:6.2f} {:5.0f}  {:7.3f}  {:>9}  {:>7}".format(
            i, LABELS[k], (b - a) / 25.0, 100.0 * present[a:b].mean(), best,
            "-" if off[0] is None else off[0], "-" if off[1] is None else off[1]))
    print("predicted spells: duration, GT possession covering most of it (share, same team?),"
          " ball%")
    for j, (c, d, kk) in enumerate(pr_sp):
        best, cover, same = -1, 0.0, False
        for i, (a, b, k) in enumerate(gt_sp + [(s, e, 2) for s, e, l in pm.segments(gt, runs) if l == 2]):
            overlap = max(0, min(b, d) - max(a, c)) / float(d - c)
            if overlap > cover:
                best, cover, same = i, overlap, (k == kk)
        print("  p{:2d} {} {:6.2f}s  gt {:>3} {:4.0%} {}  ball {:3.0f}%".format(
            j, LABELS[kk], (d - c) / 25.0, best if best < len(gt_sp) else "D", cover,
            "same" if same else "DIFF", 100.0 * present[c:d].mean()))
    gt_tr, pr_tr = pm.transitions(gt, runs), pm.transitions(pred, runs)
    res = pm.transition_scores(gt_tr, pr_tr, 50, typed="pair")
    by_type = {}
    for g, p in res["pairs"]:
        by_type.setdefault((LABELS[g[1]], LABELS[g[2]]), []).append(p[0] - g[0])
    print("transition latency (tol 50, typed pair): matched {} of {} gt, {} spurious".format(
        res["n_matched"], res["n_gt"], res["n_spurious"]))
    for key in sorted(by_type):
        lat = np.array(by_type[key])
        print("  {}->{}: n {}  mean {:+.1f}  median {:+.1f}".format(
            key[0], key[1], len(lat), lat.mean(), np.median(lat)))
    missed = {}
    for g in res["missed"]:
        key = (LABELS[g[1]], LABELS[g[2]])
        missed[key] = missed.get(key, 0) + 1
    print("  missed by type:", {"{}->{}".format(*k): v for k, v in sorted(missed.items())})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"diagnose failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
