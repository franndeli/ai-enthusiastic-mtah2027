"""Ball-control features for the team call: which team last controlled the ball, and which next.

Nearest player is a poor possession proxy: a pass flies near nobody or past opponents, and loose or
contested balls stay with the last team in control (CHALLENGE.md). Same contract as features.py:
identity-free, windows in seconds inside periods, rows outside play zero, every column symmetric or
antisymmetric under a team swap. Team signs follow diff_m: negative = home, positive = away.
Thresholds come from paired DFL LOMO runs (docs/PLAN.md, unit I2).
"""

import numpy as np

from possession.features import Features, last_seen, next_seen, speed_m_s
from possession.frames import HOME, FrameTable, period_runs

CONTROL_RADIUS_M = 1.5
CONTROL_REL_SPEED_M_S = 7.0  # ball moving with the player (dribble touches included), not past him
CONTROL_MIN_S = 0.1          # held this long, so a ball flying through the radius is no control
MAX_CARRY_S = 10.0           # cap on "seconds since/until a control"


def _sustained(mask: np.ndarray, half_frames: int, runs: list) -> np.ndarray:
    """(n, k) bool: mask held on every frame of the centred +/- half_frames window, inside runs."""
    out = np.zeros(mask.shape, dtype=bool)
    width = 2 * half_frames + 1
    for a, b in runs:
        if b - a < width:
            continue
        cum = np.concatenate([np.zeros((1, mask.shape[1]), np.int32),
                              np.cumsum(mask[a:b], axis=0, dtype=np.int32)])
        out[a + half_frames:b - half_frames] = (cum[width:] - cum[:-width]) == width
    return out


def control_sign(table: FrameTable) -> np.ndarray:
    """(n,) -1 home controls, +1 away controls, NaN otherwise (nobody, or contested by both)."""
    runs = period_runs(table.period)
    rel = table.ball_xy_m.astype(np.float64)[:, None, :] - table.players_xy_m
    dist = np.hypot(rel[..., 0], rel[..., 1])
    rel_speed = speed_m_s(rel, table.fps, runs)  # |v_ball - v_player|, NaN at a track gap
    with np.errstate(invalid="ignore"):
        close = (dist < CONTROL_RADIUS_M) & (rel_speed < CONTROL_REL_SPEED_M_S)
    control = _sustained(close, int(round(CONTROL_MIN_S * table.fps / 2)), runs)
    home_ctrl = control[:, table.slot_team == HOME].any(axis=1)
    away_ctrl = control[:, table.slot_team != HOME].any(axis=1)
    return np.where(home_ctrl & ~away_ctrl, -1.0, np.where(away_ctrl & ~home_ctrl, 1.0, np.nan))


def trajectory_features(table: FrameTable) -> Features:
    runs = period_runs(table.period)
    ctrl = control_sign(table)
    last_value, since = last_seen(ctrl, runs)
    next_value, until = next_seen(ctrl, runs)
    # Carried control sharpens the seen-ball call but hurt the call inside ATD-like ball gaps in
    # LOMO, where features.last_sign/next_sign (nearest team at the gap edges) do better.
    seen = ~np.isnan(table.ball_xy_m[:, 0])
    cols = [("last_ctrl", np.where(seen, last_value, 0.0), True),
            ("since_ctrl", np.minimum(since / table.fps, MAX_CARRY_S) / MAX_CARRY_S, False),
            ("next_ctrl", np.where(seen, next_value, 0.0), True),
            ("until_ctrl", np.minimum(until / table.fps, MAX_CARRY_S) / MAX_CARRY_S, False)]
    values = np.stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for _, v, _ in cols],
                      axis=1).astype(np.float32)
    values[table.period == 0] = 0.0
    return Features(values=values, names=tuple(name for name, _, _ in cols),
                    antisymmetric=np.array([anti for _, _, anti in cols]))
