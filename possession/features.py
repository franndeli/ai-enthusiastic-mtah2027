"""Identity-free, frame-rate invariant features over a FrameTable (ROUTE_A_PLAN.md decisions 1-2).

Each feature is either symmetric under swapping the teams or antisymmetric (flips sign); the
`antisymmetric` mask is what lets training mirror every match (negate those columns, swap H/A).
Windows are in seconds and never cross a period boundary; rows outside play are all zero.
"""

from dataclasses import dataclass

import numpy as np

from possession.frames import HOME, FrameTable, period_runs

NEAR_M = 3.0              # "at the ball" radius
OOB_MARGIN_M = 1.0        # tracking noise allowance before calling the ball out of play
SPEED_HALF_WINDOW_S = 0.2  # centred difference; single-frame diffs are too noisy on the ATD
MAX_GAP_S = 10.0          # cap on "seconds since/until the ball was seen"
DIFF_CLIP_M = 15.0
MAX_PLAYER_SPEED_M_S = 12.0  # above any sprint: the slot jumped to another person


@dataclass(frozen=True)
class Features:
    values: np.ndarray         # (n, F) float32, finite
    names: tuple
    antisymmetric: np.ndarray  # (F,) bool


def _frames(seconds: float, fps: float) -> int:
    return max(1, int(round(seconds * fps)))


def rolling_mean(values: np.ndarray, half_frames: int, runs: list) -> np.ndarray:
    """Centred mean over +/- half_frames ignoring NaN, inside each run; NaN where nothing seen."""
    out = np.full(len(values), np.nan)
    for a, b in runs:
        v = values[a:b]
        seen = ~np.isnan(v)
        cum_sum = np.concatenate(([0.0], np.cumsum(np.where(seen, v, 0.0))))
        cum_n = np.concatenate(([0], np.cumsum(seen)))
        i = np.arange(b - a)
        lo, hi = np.maximum(0, i - half_frames), np.minimum(b - a, i + half_frames + 1)
        count = cum_n[hi] - cum_n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(count > 0, (cum_sum[hi] - cum_sum[lo]) / count, np.nan)
    return out


def last_seen(values: np.ndarray, runs: list) -> tuple:
    """(most recent non-NaN value at or before each frame, frames since it), within runs.

    NaN / inf where nothing has been seen yet in the run.
    """
    value = np.full(len(values), np.nan)
    age = np.full(len(values), np.inf)
    for a, b in runs:
        idx = np.arange(b - a)
        last = np.maximum.accumulate(np.where(~np.isnan(values[a:b]), idx, -1))
        ok = last >= 0
        value[a:b][ok] = values[a:b][last[ok]]
        age[a:b][ok] = (idx - last)[ok]
    return value, age


def next_seen(values: np.ndarray, runs: list) -> tuple:
    """Mirror of last_seen: the next non-NaN value at or after each frame, and frames until it."""
    n = len(values)
    value, age = last_seen(values[::-1], [(n - b, n - a) for a, b in runs])
    return value[::-1], age[::-1]


def speed_m_s(xy: np.ndarray, fps: float, runs: list) -> np.ndarray:
    """Speed from a centred difference over +/- SPEED_HALF_WINDOW_S; xy is (n, m, 2), out (n, m)."""
    h = _frames(SPEED_HALF_WINDOW_S, fps)
    out = np.full(xy.shape[:2], np.nan)
    for a, b in runs:
        if b - a <= 2 * h:
            continue
        delta = xy[a + 2 * h:b].astype(np.float64) - xy[a:b - 2 * h]
        out[a + h:b - h] = np.hypot(delta[..., 0], delta[..., 1]) * fps / (2 * h)
    return out


def _min_over(dist: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.full(len(dist), np.nan)
    return np.fmin.reduce(dist[:, mask], axis=1)  # fmin ignores NaN unless all are NaN


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    count = (~np.isnan(values)).sum(axis=1)
    total = np.nansum(values, axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def features(table: FrameTable) -> Features:
    fps = table.fps
    runs = period_runs(table.period)
    length, width = table.pitch_m
    ball = table.ball_xy_m.astype(np.float64)
    players = table.players_xy_m
    home = table.slot_team == HOME

    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d_home, d_away = _min_over(dist, home), _min_over(dist, ~home)
    diff = np.clip(d_home - d_away, -DIFF_CLIP_M, DIFF_CLIP_M)
    seen = (~np.isnan(ball[:, 0])).astype(np.float64)
    with np.errstate(invalid="ignore"):
        near = dist < NEAR_M
        oob = ((np.abs(ball[:, 0]) > length / 2 + OOB_MARGIN_M)
               | (np.abs(ball[:, 1]) > width / 2 + OOB_MARGIN_M)).astype(np.float64)
    oob[seen == 0] = np.nan
    near_diff = near[:, home].sum(axis=1) - near[:, ~home].sum(axis=1)
    near_diff = np.where(seen > 0, near_diff, np.nan)
    ball_speed = np.clip(speed_m_s(ball[:, None, :], fps, runs)[:, 0], 0, 40)
    slot_speed = speed_m_s(players, fps, runs)
    slot_speed[slot_speed > MAX_PLAYER_SPEED_M_S] = np.nan  # tracklet identity switch, not running
    player_speed = _nanmean_rows(slot_speed)
    players_seen = (~np.isnan(players[..., 0])).sum(axis=1) / 22.0

    last_value, since = last_seen(diff, runs)
    next_value, until = next_seen(diff, runs)
    cols = []

    def add(name, values, anti=False):
        cols.append((name, values, anti))

    add("ball_seen", seen)
    add("diff_m", diff, True)
    add("dmin_m", np.clip(np.fmin(d_home, d_away), 0, 30))
    add("near_diff", near_diff, True)
    add("ball_speed_m_s", ball_speed)
    add("ball_oob", oob)
    add("player_speed_m_s", player_speed)
    add("players_seen", players_seen)
    for s in (0.5, 2.0, 5.0):
        add(f"diff_m_roll{s}s", rolling_mean(diff, _frames(s, fps), runs), True)
    add("near_diff_roll1s", rolling_mean(near_diff, _frames(1.0, fps), runs), True)
    for s in (1.0, 5.0):
        add(f"ball_seen_roll{s}s", rolling_mean(seen, _frames(s, fps), runs))
    for s in (2.0, 5.0):
        add(f"player_speed_roll{s}s", rolling_mean(player_speed, _frames(s, fps), runs))
    add("ball_speed_roll1s", rolling_mean(ball_speed, _frames(1.0, fps), runs))
    add("ball_oob_roll2s", rolling_mean(oob, _frames(2.0, fps), runs))
    add("last_sign", np.sign(last_value), True)
    add("since_seen", np.minimum(since / fps, MAX_GAP_S) / MAX_GAP_S)
    add("next_sign", np.sign(next_value), True)
    add("until_seen", np.minimum(until / fps, MAX_GAP_S) / MAX_GAP_S)

    values = np.stack([np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for _, v, _ in cols],
                      axis=1).astype(np.float32)
    values[table.period == 0] = 0.0
    return Features(values=values, names=tuple(name for name, _, _ in cols),
                    antisymmetric=np.array([anti for _, _, anti in cols]))
