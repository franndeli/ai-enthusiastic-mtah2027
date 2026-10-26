"""Ball-free formation features over a FrameTable: who attacks, read from where the teams stand.

The team in possession pushes up and spreads out while the defending team drops deep and compacts,
and that survives the long ATD ball gaps. Same contract as possession.features: identity-free
(positions and slot_team only), windows in seconds inside each period, every column symmetric or
antisymmetric under swapping the teams, rows outside play all zero.
"""

import numpy as np

from possession.features import (MAX_PLAYER_SPEED_M_S, Features, _frames, _nanmean_rows,
                                 rolling_mean, speed_m_s)
from possession.frames import HOME, FrameTable, period_runs

VELOCITY_HALF_WINDOW_S = 0.5
ROLL_WINDOWS_S = (2.0, 5.0)


def attack_direction(table: FrameTable) -> np.ndarray:
    """(n,) +1 where home attacks towards +x, -1 towards -x, 0 outside play or undecidable.

    Each team stands nearer its own goal on average (goalkeeper, back line), so in each period home
    attacks towards the side where the away players sit on average.
    """
    x = table.players_xy_m[..., 0].astype(np.float64)
    home = table.slot_team == HOME
    out = np.zeros(table.n_frames)
    for p in np.unique(table.period[table.period != 0]):
        rows = x[table.period == p]
        means = [np.nansum(rows[:, m]) / max((~np.isnan(rows[:, m])).sum(), 1) for m in (home, ~home)]
        out[table.period == p] = np.sign(means[1] - means[0])
    return out


def _team_stats(u: np.ndarray, y: np.ndarray) -> dict:
    """Per-frame shape of one team; u is x oriented so that + points at the opponent goal."""
    seen = ~np.isnan(u)
    centroid = _nanmean_rows(u)
    deepest = np.sort(u, axis=1)  # NaN sorts last
    with np.errstate(invalid="ignore"):
        return dict(
            centroid=centroid,
            # second deepest: the last defender in front of the goalkeeper
            line=deepest[:, 1] if u.shape[1] > 1 else np.full(len(u), np.nan),
            length=np.sqrt(np.maximum(_nanmean_rows(u ** 2) - centroid ** 2, 0)),
            width=np.sqrt(np.maximum(_nanmean_rows(y ** 2) - _nanmean_rows(y) ** 2, 0)),
            opp_half=np.where(seen.any(axis=1), (u > 0).sum(axis=1) / np.maximum(seen.sum(axis=1), 1),
                              np.nan))


def _x_velocity_m_s(x: np.ndarray, fps: float, runs: list) -> np.ndarray:
    """Signed per-slot x velocity, centred difference inside runs; identity jumps become NaN."""
    h = _frames(VELOCITY_HALF_WINDOW_S, fps)
    out = np.full(x.shape, np.nan)
    for a, b in runs:
        if b - a > 2 * h:
            out[a + h:b - h] = (x[a + 2 * h:b] - x[a:b - 2 * h]) * fps / (2 * h)
    with np.errstate(invalid="ignore"):
        out[np.abs(out) > MAX_PLAYER_SPEED_M_S] = np.nan
    return out


def tactical_features(table: FrameTable) -> Features:
    """Home-minus-away formation contrasts (antisymmetric), raw and rolled, plus one symmetric sum."""
    fps = table.fps
    runs = period_runs(table.period)
    x = table.players_xy_m[..., 0].astype(np.float64)
    y = table.players_xy_m[..., 1].astype(np.float64)
    home = table.slot_team == HOME
    direction = attack_direction(table)
    home_shape, away_shape = (_team_stats(x[:, m] * sign[:, None], y[:, m])
                              for m, sign in ((home, direction), (~home, -direction)))
    slot_speed = speed_m_s(table.players_xy_m, fps, runs)
    slot_speed[slot_speed > MAX_PLAYER_SPEED_M_S] = np.nan
    contrasts = {f"{key}_diff": home_shape[key] - away_shape[key]
                 for key in ("centroid", "line", "length", "width", "opp_half")}
    # Both teams move with the play: the whole field drifts towards the attacked goal.
    contrasts["push_m_s"] = direction * _nanmean_rows(_x_velocity_m_s(x, fps, runs))
    contrasts["speed_diff_m_s"] = (_nanmean_rows(slot_speed[:, home])
                                   - _nanmean_rows(slot_speed[:, ~home]))
    cols = [("tac_centroid_sum", home_shape["centroid"] + away_shape["centroid"], False)]
    for key, series in contrasts.items():
        cols.append((f"tac_{key}", series, True))
        cols += [(f"tac_{key}_roll{s}s", rolling_mean(series, _frames(s, fps), runs), True)
                 for s in ROLL_WINDOWS_S]

    values = np.stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for _, v, _ in cols],
                      axis=1).astype(np.float32)
    values[table.period == 0] = 0.0
    return Features(values=values, names=tuple(name for name, _, _ in cols),
                    antisymmetric=np.array([anti for _, _, anti in cols]))
