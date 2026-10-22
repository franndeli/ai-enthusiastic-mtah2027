#!/usr/bin/env python3
"""Convert each downloaded IDSSE match to a cached possession.io.Match + labels and print a label report.

    scripts/py scripts/idsse_prepare.py [--matches J03WPY ...] [--refresh] [--seed 0]

Public data, aggregates only. Compare the semantics with our public window: 17 possessions in
5 min, mean 14.2 s, 6 spells < 4 s, 4 D stretches of 5.5-20 s, 12 of 20 transitions team swaps.
Also verifies the noise-injection targets on each match.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import A, D, H, LABELS  # noqa: E402
from possession import external  # noqa: E402


def label_report(codes: np.ndarray, fps: int) -> dict:
    """Possession-structure aggregates of a code array, using the scorer's own run logic."""
    runs = pm.valid_runs(codes)
    graded = codes[codes != pm.X]
    spells = np.array([b - a for a, b, _ in pm.possession_spells(codes, runs)]) / fps
    dead = np.array([b - a for a, b, k in pm.segments(codes, runs) if k == D]) / fps
    kinds = {"H>A": 0, "A>H": 0, "team>D": 0, "D>team": 0}
    for _, src, dst in pm.transitions(codes, runs):
        if {src, dst} == {H, A}:
            kinds["H>A" if src == H else "A>H"] += 1
        elif dst == D:
            kinds["team>D"] += 1
        else:
            kinds["D>team"] += 1
    return {
        "n_graded": int(len(graded)),
        "shares": {LABELS[k]: float(np.mean(graded == k)) for k in (H, A, D)},
        "n_spells": int(len(spells)),
        "spell_mean_s": float(spells.mean()), "spell_median_s": float(np.median(spells)),
        "spell_lt2s": float(np.mean(spells < 2)), "spell_lt4s": float(np.mean(spells < 4)),
        "n_dead": int(len(dead)), "dead_median_s": float(np.median(dead)) if len(dead) else 0.0,
        "transitions": kinds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--refresh", action="store_true", help="re-parse even if cached")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ids = args.matches or external.available_matches()
    if not ids:
        print("no IDSSE match on disk; run scripts/fetch_idsse.py first")
        return 1
    for mid in ids:
        started = time.time()
        ext = external.load_external(mid, refresh=args.refresh)
        load_s = time.time() - started
        m, rep = ext.match, label_report(ext.codes, ext.fps)
        inside = m.graded_mask()
        print("== {}  {} frames x {} slots, fps {}, pitch {:.1f}x{:.1f} m, halves {}, {:.1f}s".format(
            mid, m.n_frames, m.n_slots, ext.fps, *m.pitch_m, [(h, a, b) for h, a, b in m.halves],
            load_s))
        print("   raw (BallStatus, BallPossession): count  {}".format(
            {(int(s), int(p)): int(c) for s, p, c in ext.raw_ball_counts}))
        print("   graded {}  absent-in-halves {}  ball NaN {:.4f}  ball x range {:.3f}..{:.3f}"
              "  tracked/frame {:.1f}".format(
                  rep["n_graded"], int(np.sum(ext.codes[inside] == pm.X)),
                  float(np.isnan(m.ball_xy[inside, 0]).mean()),
                  float(np.nanmin(m.ball_xy[:, 0])), float(np.nanmax(m.ball_xy[:, 0])),
                  float(np.mean(np.sum(~np.isnan(m.players_xy[inside][..., 0]), axis=1)))))
        print("   shares {}  spells {} ({:.1f}/5min) mean {:.1f}s median {:.1f}s  <2s {:.2f} <4s {:.2f}".format(
            {k: round(v, 3) for k, v in rep["shares"].items()}, rep["n_spells"],
            rep["n_spells"] / (rep["n_graded"] / ext.fps / 300),
            rep["spell_mean_s"], rep["spell_median_s"], rep["spell_lt2s"], rep["spell_lt4s"]))
        t = rep["transitions"]
        print("   D stretches {} median {:.1f}s  transitions {}  team swaps {:.2f} of all".format(
            rep["n_dead"], rep["dead_median_s"], t, (t["H>A"] + t["A>H"]) / max(1, sum(t.values()))))
        noisy = external.inject_noise(m, args.seed, fps=ext.fps)
        print("   noise (seed {}): {}".format(args.seed, {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in external.noise_report(m, noisy, ext.fps).items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
