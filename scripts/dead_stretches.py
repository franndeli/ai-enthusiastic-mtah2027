#!/usr/bin/env python3
"""Per-stretch aggregates of the dead-ball evidence on the public window's true D stretches and
on the alive possessions next to them. Aggregates only (means over each stretch)."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm  # noqa: E402
from possession import evidence, features, io  # noqa: E402
from run import DEFAULT_CONFIG, label_codes_from_config, load_config, load_tags  # noqa: E402

LABELS = "HADX"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config, match, gt = load_config(args.config), io.load_match(), io.load_public_gt()
    tags = load_tags(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        feats = features.compute(match, tags, config.get("features"))
        p = dict(evidence.DEFAULT_PARAMS, **config.get("evidence", {}))
        dead_z = evidence.dead_logit(feats, p)
        pred = label_codes_from_config(config, match, tags)
    graded = np.flatnonzero(gt != 3)
    runs = [(int(graded[0]), int(graded[-1]) + 1)]
    print("seg lab  dur_s ball% dead_z slow_z  spd  out%  exit% tag% touch% move% gap_s  pred: D% nspells")
    for a, b, k in pm.segments(gt, runs):
        sl = slice(a, b)
        speed = feats["all_mean_speed_lead_m_s"][sl]
        slow = np.clip((p["slow_ref_m_s"] - speed) / p["slow_scale_m_s"], -2, 2)
        n_spells = len(pm.possession_spells(pred, [(a, b)]))
        gap = feats["frames_since_ball"][sl] + feats["frames_until_ball"][sl]
        print("{} {:>4.1f}s {:4.0f} {:+6.2f} {:+6.2f} {:4.2f} {:4.0f} {:5.0f} {:4.0f} {:5.0f} {:5.0f} {:5.1f}  {:4.0f} {:3d}".format(
            LABELS[k], (b - a) / 25.0, 100 * np.mean(feats["ball_present"][sl]),
            float(np.mean(dead_z)) if False else float(np.mean(dead_z[sl])),
            float(np.nanmean(slow)), float(np.nanmean(speed)),
            100 * float(np.nanmean(feats["ball_out_frac"][sl])),
            100 * float(np.nanmean(feats["ball_exited"][sl])),
            100 * float(np.mean(feats.get("tag_set_piece", np.zeros(len(gt)))[sl])),
            100 * float(np.mean(feats["touch_vote"][sl] != 0)),
            100 * float(np.mean(feats["ball_moving_frac"][sl])),
            float(np.median(np.where(np.isinf(gap), 0, gap))) / 25.0,
            100 * float(np.mean(pred[sl] == 2)), n_spells))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"dead_stretches failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
