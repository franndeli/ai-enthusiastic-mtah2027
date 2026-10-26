"""ATD vs DFL domain-gap audit, aggregates only (ROUTE_A_PLAN.md risk 1).

    scripts/py -m possession.domain_gap

Calibrates the ATD-like corruption applied to DFL in possession.train.
"""

import numpy as np

from possession import atd
from possession.features import MAX_PLAYER_SPEED_M_S, speed_m_s
from possession.frames import FrameTable, period_runs
from possession.train import load_tables

CROWD_M = 5.0
CROWD_BUCKETS = (0, 1, 2, 3)  # players within CROWD_M of the ball; last bucket is "3 or more"


def run_lengths(mask: np.ndarray, runs: list) -> np.ndarray:
    """Lengths of True runs inside each period run."""
    out = []
    for a, b in runs:
        edges = np.diff(np.concatenate(([0], mask[a:b].astype(np.int8), [0])))
        out.append(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    return np.concatenate(out) if out else np.array([], int)


def crowd(table: FrameTable) -> np.ndarray:
    dist = np.hypot(table.players_xy_m[..., 0] - table.ball_xy_m[:, None, 0],
                    table.players_xy_m[..., 1] - table.ball_xy_m[:, None, 1])
    with np.errstate(invalid="ignore"):
        return np.minimum((dist < CROWD_M).sum(axis=1), CROWD_BUCKETS[-1])


def gap_stats(table: FrameTable) -> dict:
    runs = period_runs(table.period)
    play = table.period > 0
    missing = np.isnan(table.ball_xy_m[:, 0]) & play
    gaps = run_lengths(missing, runs) / table.fps
    slot_speed = speed_m_s(table.players_xy_m, table.fps, runs)
    finite = slot_speed[np.isfinite(slot_speed)]
    stats = {
        "ball_missing_share": missing.sum() / play.sum(),
        "gap_mean_s": gaps.mean() if len(gaps) else 0.0,
        "gap_median_s": float(np.median(gaps)) if len(gaps) else 0.0,
        "gap_p90_s": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
        "players_seen": (~np.isnan(table.players_xy_m[play, :, 0])).sum(axis=1).mean(),
        "speed_p50_m_s": float(np.median(finite)),
        "speed_p90_m_s": float(np.percentile(finite, 90)),
        "teleport_share": float((finite > MAX_PLAYER_SPEED_M_S).mean()),
    }
    # Hazard of losing the ball next frame, by how crowded the ball is now.
    visible = ~missing & play
    lost_next = np.zeros_like(visible)
    lost_next[:-1] = visible[:-1] & missing[1:] & (table.period[:-1] == table.period[1:])
    c = crowd(table)
    for bucket in CROWD_BUCKETS:
        at = visible & (c == bucket)
        stats[f"share_crowd{bucket}"] = at.sum() / max(visible.sum(), 1)
        stats[f"hazard_crowd{bucket}"] = lost_next[at].sum() / max(at.sum(), 1)
    return stats


def main() -> None:
    columns = {"ATD (BAR-ATH)": gap_stats(atd.load_game())}
    dfl = [gap_stats(t) for _, t in load_tables()]
    columns["DFL (mean of 7)"] = {k: float(np.mean([s[k] for s in dfl])) for k in dfl[0]}
    names = list(columns)
    print(f"{'':>22}" + "".join(f"{n:>18}" for n in names))
    for key in columns[names[0]]:
        print(f"{key:>22}" + "".join(f"{columns[n][key]:>18.4f}" for n in names))


if __name__ == "__main__":
    main()
