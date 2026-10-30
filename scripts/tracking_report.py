"""How smooth and how complete is the player tracking? LOCAL ONLY on the match; aggregates only.

PathCRF (ball-free possession from player trajectories) tolerates players that drop out or swap
identity, but not frame-to-frame jitter: it reads velocities and accelerations. This report
measures, on the graded frames, the things that decide whether the ATD needs smoothing before
PathCRF, so the same numbers can be compared with the public benchmarks (DFL, SkillCorner):

* objects per frame (all, per team) and the share of frames with at least 11 per team;
* jitter: second difference |p(t+1) - 2 p(t) + p(t-1)| and the residual from a 1 s
  Savitzky-Golay fit, in metres, per slot over consecutive frames;
* teleports: share of slot steps faster than 12 m/s;
* tracklets: continuous runs per slot (count, median length);
* duplicates: share of frames with two slots of the same team within 1 m.

usage: python scripts/tracking_report.py [--data DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _savgol_kernel(window: int = 25, order: int = 2) -> np.ndarray:
    """Centre-point Savitzky-Golay smoothing weights (numpy only; scipy is not a dependency)."""
    t = np.arange(window) - window // 2
    basis = np.vander(t, order + 1, increasing=True)
    return (basis @ np.linalg.pinv(basis))[window // 2]


def report(players: np.ndarray, team: np.ndarray, graded: np.ndarray, fps: float) -> dict:
    """Aggregates over the graded frames; players (n, S, 2) metres, NaN when absent."""
    kernel = _savgol_kernel()
    half = len(kernel) // 2
    p = players.astype(np.float64)
    seen = ~np.isnan(p[..., 0]) & graded[:, None]
    out = {}
    per = seen.sum(axis=1)[graded]
    out["objects_mean"] = float(per.mean())
    out["objects_p10_p50_p90"] = tuple(float(v) for v in np.percentile(per, [10, 50, 90]))
    for side, name in ((0, "home"), (1, "away")):
        out["objects_%s_mean" % name] = float(seen[:, team == side].sum(axis=1)[graded].mean())
    full = (seen[:, team == 0].sum(axis=1) >= 11) & (seen[:, team == 1].sum(axis=1) >= 11)
    out["frames_11_per_team"] = float(full[graded].mean())
    d2, resid, steps = [], [], []
    n_runs, run_len = 0, []
    for s in range(p.shape[1]):
        starts, stops = _runs(seen[:, s])
        for a, b in zip(starts, stops):
            n_runs += 1
            run_len.append((b - a) / fps)
            seg = p[a:b, s]
            if b - a >= 3:
                d2.append(np.linalg.norm(seg[2:] - 2 * seg[1:-1] + seg[:-2], axis=1))
                steps.append(np.linalg.norm(np.diff(seg, axis=0), axis=1) * fps)
            if b - a >= 51:
                fit = np.stack([np.convolve(seg[:, d], kernel[::-1], mode="valid") for d in (0, 1)], axis=1)
                resid.append(np.linalg.norm(seg[half:len(seg) - half] - fit, axis=1))
    d2 = np.concatenate(d2) if d2 else np.array([np.nan])
    resid = np.concatenate(resid) if resid else np.array([np.nan])
    steps = np.concatenate(steps) if steps else np.array([np.nan])
    out["jitter_d2_median_m"] = float(np.median(d2))
    out["jitter_d2_p90_m"] = float(np.percentile(d2, 90))
    out["jitter_resid_median_m"] = float(np.median(resid))
    out["teleport_share"] = float(np.mean(steps > 12.0))
    out["tracklets_per_graded_minute"] = float(n_runs / (graded.sum() / fps / 60.0))
    out["tracklet_median_s"] = float(np.median(run_len)) if run_len else float("nan")
    dup = np.zeros(p.shape[0], dtype=bool)
    for side in (0, 1):
        q = p[:, team == side]
        k = q.shape[1]
        for c in range(0, len(q), 20000):
            qc = q[c:c + 20000]
            d = np.linalg.norm(qc[:, :, None, :] - qc[:, None, :, :], axis=3)
            d[:, np.arange(k), np.arange(k)] = np.inf
            d[np.isnan(d)] = np.inf
            dup[c:c + 20000] |= d.reshape(len(qc), -1).min(axis=1) < 1.0
    out["duplicate_frames"] = float(dup[graded].mean())
    return out


def main(argv=None) -> int:
    import games
    from possession.tracking import read_epts
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR", os.path.join(
        os.path.expanduser("~"), ".mtah2027", "data")))
    ap.add_argument("--game", default="BAR-ATH")
    args = ap.parse_args(argv)
    g = games.GAMES[args.game]
    paths = games.asset_paths(args.game, data_dir=args.data)
    if not (os.path.isfile(paths["atd_meta"]) and os.path.isfile(paths["atd_raw"])):
        print("ATD not found under %s" % os.path.join(args.data, "tactical"))
        return 2
    trk = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=tuple(g["pitch"]),
                    n_frames=int(g["n_video_frames"]))
    graded = np.zeros(trk.n, dtype=bool)
    for _, a, b in games.half_video_blocks(args.game):
        graded[a:b + 1] = True
    for key, value in report(trk.players, trk.team, graded, float(g["fps"])).items():
        if isinstance(value, tuple):
            print("%-28s %s" % (key, " / ".join("%.1f" % v for v in value)))
        else:
            print("%-28s %.4f" % (key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
