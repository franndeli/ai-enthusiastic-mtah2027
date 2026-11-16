#!/usr/bin/env python3
"""Aggregate audit of the SkillCorner judge (external public data): tracking coverage and the
two label variants per match. Builds the per-match cache on first use (~20 s per match).

    scripts/py scripts/skillcorner_audit.py [--jobs 4]

Prints aggregates only (shares, counts, durations).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from possession import FPS, skillcorner  # noqa: E402


def spell_lengths(codes: np.ndarray) -> np.ndarray:
    """Lengths (frames) of maximal H or A runs."""
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    st, en = np.concatenate(([0], br)), np.concatenate((br, [len(codes)]))
    return (en - st)[codes[st] < 2]


def audit(mid: str) -> dict:
    ext = {k: skillcorner.load_match(mid, k) for k in skillcorner.LABEL_KINDS}
    m = ext["event"].match
    g = m.graded_mask()
    out = {"match": mid, "frames": int(g.sum()),
           "ball_seen": float(np.mean(~np.isnan(m.ball_xy[g, 0]))),
           "players": float(np.mean(np.sum(~np.isnan(m.players_xy[g][:, :, 0]), axis=1)))}
    for k, e in ext.items():
        c = e.codes[g]
        sp = np.concatenate([spell_lengths(e.codes[a:b + 1]) for _, a, b in m.halves]) / FPS
        out[k] = {"D": float(np.mean(c == 2)), "H": float(np.mean(c == 0)), "n": int(len(sp)),
                  "median_s": float(np.median(sp)), "lt2": float(np.mean(sp < 2)),
                  "lt4": float(np.mean(sp < 4))}
        seen = ~np.isnan(m.ball_xy[g, 0])
        out[k]["seen_dead"] = float(seen[c == 2].mean())
        out[k]["seen_alive"] = float(seen[c < 2].mean())
    ce, cp = ext["event"].codes[g], ext["phase"].codes[g]
    both = (ce < 2) & (cp < 2)
    out["dead_agree"] = float(np.mean((ce == 2) == (cp == 2)))
    out["team_agree"] = float(np.mean(ce[both] == cp[both]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    matches = skillcorner.available_matches()
    if not matches:
        raise SystemExit("no SkillCorner matches; run scripts/py scripts/fetch_skillcorner.py")
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=args.jobs)(delayed(audit)(mid) for mid in matches)
    for r in rows:
        e, p = r["event"], r["phase"]
        print(f"{r['match']} frames {r['frames']:6d} ball seen {r['ball_seen']:.2f} players {r['players']:.1f} | "
              f"event D {e['D']:.2f} n {e['n']:3d} med {e['median_s']:.1f}s <2s {e['lt2']:.2f} <4s {e['lt4']:.2f} "
              f"seen dead/alive {e['seen_dead']:.2f}/{e['seen_alive']:.2f} | "
              f"phase D {p['D']:.2f} n {p['n']:3d} med {p['median_s']:.1f}s | "
              f"agree dead {r['dead_agree']:.2f} team {r['team_agree']:.2f}")
    mean = lambda f: float(np.mean([f(r) for r in rows]))  # noqa: E731
    print(f"mean over {len(rows)} matches: ball seen {mean(lambda r: r['ball_seen']):.3f} "
          f"players {mean(lambda r: r['players']):.1f} | event D {mean(lambda r: r['event']['D']):.3f} "
          f"possessions {mean(lambda r: r['event']['n']):.0f} | phase D {mean(lambda r: r['phase']['D']):.3f} "
          f"possessions {mean(lambda r: r['phase']['n']):.0f} | agree dead {mean(lambda r: r['dead_agree']):.3f} "
          f"team {mean(lambda r: r['team_agree']):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
