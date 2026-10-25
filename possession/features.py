"""Per-frame features from tracking. Pure numpy; every array has one row per frame.

Two feature sets:

* team features (H vs A), antisymmetric: swapping the teams negates them, so a model without an
  intercept cannot learn "the home side usually has the ball";
* dead-ball features (dead vs alive), from the ball's visibility and kinematics and from how the
  players move (players walk during stoppages), so they still work when the ball is missing.
"""

from __future__ import annotations

import numpy as np

from .tracking import HOME, AWAY, Tracking

MAX_BALL_SPEED = 40.0      # m/s; faster jumps between observations are tracker errors
INTERP_MAX_S = 0.4         # interpolate ball gaps up to this long
MAX_PLAYER_SPEED = 12.0    # m/s; faster is an identity switch, not running


def rolling_mean(x: np.ndarray, half: int) -> np.ndarray:
    """Centred mean over +/- half frames ignoring NaN; NaN where the window has no value."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    cv = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    cc = np.concatenate(([0.0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    s = cv[hi] - cv[lo]
    c = cc[hi] - cc[lo]
    out = np.full(len(x), np.nan)
    np.divide(s, c, out=out, where=c > 0)
    return out


def rolling_max(x: np.ndarray, half: int) -> np.ndarray:
    """Centred max of a boolean/0-1 array over +/- half frames."""
    x = np.asarray(x, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(x > 0)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    return (c[hi] - c[lo] > 0).astype(np.float64)


def clean_ball(ball: np.ndarray, fps: float) -> np.ndarray:
    """Drop isolated impossible jumps, then interpolate short gaps. Returns float64 (N, 2)."""
    b = ball.astype(np.float64).copy()
    for _ in range(3):
        idx = np.flatnonzero(~np.isnan(b[:, 0]))
        if len(idx) < 3:
            break
        step = np.linalg.norm(np.diff(b[idx], axis=0), axis=1) / (np.diff(idx) / fps)
        fast = step > MAX_BALL_SPEED
        # a point is an outlier if the jumps into and out of it are both impossible
        bad = np.zeros(len(idx), dtype=bool)
        bad[1:-1] = fast[:-1] & fast[1:]
        if not bad.any():
            break
        b[idx[bad]] = np.nan
    seen = ~np.isnan(b[:, 0])
    idx = np.flatnonzero(seen)
    if len(idx) >= 2:
        gaps = np.diff(idx)
        short = (gaps > 1) & (gaps <= int(round(INTERP_MAX_S * fps)) + 1)
        for k in np.flatnonzero(short):
            a, z = idx[k], idx[k + 1]
            w = (np.arange(a + 1, z) - a) / float(z - a)
            b[a + 1:z] = b[a] + (b[z] - b[a]) * w[:, None]
    return b


def ball_speed(b: np.ndarray, fps: float, half: int = 2) -> np.ndarray:
    """Centred-difference speed (m/s) over +/- half frames; NaN where either end is missing."""
    n = len(b)
    v = np.full(n, np.nan)
    if n > 2 * half:
        d = np.linalg.norm(b[2 * half:] - b[:-2 * half], axis=1) / (2.0 * half / fps)
        v[half:n - half] = d
    return v


def player_speeds(players: np.ndarray, fps: float, half: int = 3) -> np.ndarray:
    """(N, S) speed per slot from a centred difference; implausible values become NaN."""
    p = players.astype(np.float64)
    n = p.shape[0]
    v = np.full(p.shape[:2], np.nan)
    if n > 2 * half:
        d = np.linalg.norm(p[2 * half:] - p[:-2 * half], axis=2) / (2.0 * half / fps)
        v[half:n - half] = d
    v[v > MAX_PLAYER_SPEED] = np.nan
    return v


def _nanmedian_rows(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape[0], np.nan)
    ok = (~np.isnan(x)).any(axis=1)
    if ok.any():
        out[ok] = np.nanmedian(x[ok], axis=1)
    return out


def _time_since_seen(seen: np.ndarray, fps: float, cap_s: float = 30.0) -> np.ndarray:
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    t = np.where(last >= 0, (np.arange(n) - last) / fps, cap_s)
    return np.minimum(t, cap_s)


def _edge_distance(xy: np.ndarray, pitch) -> np.ndarray:
    length, width = pitch
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([np.abs(x), np.abs(length - x), np.abs(y), np.abs(width - y)])


def _carry_last(values: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """values at the last seen frame, carried forward (NaN before the first)."""
    n = len(seen)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.full(n, np.nan)
    ok = last >= 0
    out[ok] = values[last[ok]]
    return out


def compute(trk: Tracking) -> dict:
    """All per-frame features, as a dict of (N,) float64 arrays."""
    fps = trk.fps
    length, width = trk.pitch
    raw_seen = ~np.isnan(trk.ball[:, 0])
    b = clean_ball(trk.ball, fps)
    seen = ~np.isnan(b[:, 0])
    vb = ball_speed(b, fps)

    # distances ball -> players, nearest per team
    d = np.linalg.norm(trk.players.astype(np.float64) - b[:, None, :], axis=2)   # (N, S)
    home = trk.team == HOME
    away = trk.team == AWAY
    d_home = np.fmin.reduce(d[:, home], axis=1) if home.any() else np.full(trk.n, np.nan)
    d_away = np.fmin.reduce(d[:, away], axis=1) if away.any() else np.full(trk.n, np.nan)
    both = ~np.isnan(d_home) & ~np.isnan(d_away)
    d_min = np.fmin(d_home, d_away)

    inside = seen & (b[:, 0] > -0.5) & (b[:, 0] < length + 0.5) & (b[:, 1] > -0.5) & (b[:, 1] < width + 0.5)
    out_of_bounds = seen & ~inside

    f = {}
    # -- team (antisymmetric) ------------------------------------------------------------------
    close = np.where(both & inside, np.exp(-d_home / 1.5) - np.exp(-d_away / 1.5), np.nan)
    adv = np.where(both & inside, np.clip(d_away - d_home, -5.0, 5.0), np.nan)
    slow = np.where(np.isnan(vb), 1.0, (vb < 10.0).astype(np.float64))
    control = np.where(both & inside & (d_min < 1.5) & (slow > 0),
                       np.sign(d_away - d_home) * (np.abs(d_away - d_home) > 0.5), np.nan)
    f["t_close"] = np.nan_to_num(close)
    f["t_adv"] = np.nan_to_num(adv)
    f["t_control"] = np.nan_to_num(control)
    for half, tag in ((5, "02"), (25, "1"), (75, "3")):
        f["t_close_%s" % tag] = np.nan_to_num(rolling_mean(close, half))
        f["t_control_%s" % tag] = np.nan_to_num(rolling_mean(control, half))

    # -- dead ball -----------------------------------------------------------------------------
    vp = player_speeds(trk.players, fps)
    med = _nanmedian_rows(vp)
    slow_frac = np.where(np.isnan(vp), np.nan, (vp < 1.5).astype(np.float64))
    with np.errstate(invalid="ignore"):
        slow_share = np.nanmean(slow_frac, axis=1) if slow_frac.size else np.full(trk.n, np.nan)
    since = _time_since_seen(raw_seen, fps)
    until = _time_since_seen(raw_seen[::-1], fps)[::-1]
    gap_len = np.where(raw_seen, 0.0, since + until)
    raw = trk.ball.astype(np.float64)
    edge_last = _carry_last(_edge_distance(raw, trk.pitch), raw_seen)
    edge_next = _carry_last(_edge_distance(raw[::-1], trk.pitch), raw_seen[::-1])[::-1]
    centre = np.hypot(b[:, 0] - length / 2.0, b[:, 1] - width / 2.0)

    g = {}
    g["d_speed_1"] = rolling_mean(med, 25)
    g["d_speed_3"] = rolling_mean(med, 75)
    g["d_speed_10"] = rolling_mean(med, 250)
    g["d_slow_1"] = rolling_mean(slow_share, 25)
    g["d_seen_1"] = rolling_mean(raw_seen.astype(np.float64), 25)
    g["d_seen_5"] = rolling_mean(raw_seen.astype(np.float64), 125)
    g["d_out_1"] = rolling_max(out_of_bounds, 25)
    g["d_bspeed_1"] = np.log1p(rolling_mean(vb, 12))
    g["d_still_2"] = rolling_mean(np.where(np.isnan(vb), np.nan, (vb < 0.5).astype(np.float64)), 50)
    g["d_gap_log"] = np.log1p(gap_len)
    g["d_gap_edge"] = np.where(raw_seen, 0.0, np.exp(-np.fmin(edge_last, edge_next) / 3.0))
    g["d_gap_out"] = np.where(raw_seen, 0.0, (np.fmin(edge_last, edge_next) < 1.0).astype(np.float64))
    g["d_centre"] = np.where(seen, np.exp(-centre / 2.0), 0.0)
    g["d_nearest"] = np.log1p(np.where(np.isnan(d_min), 10.0, np.minimum(d_min, 10.0)))
    # fill remaining NaN with neutral values (column means are applied in the model)
    f.update(g)
    f.update(_team_touch_features(trk, b, d, d_home, d_away, inside, fps))
    f.update(_shape_features(trk, fps))
    return f


def _signed_team(trk: Tracking, nearest: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """+1 where the nearest slot is home, -1 away, 0 where not valid."""
    sign = np.where(trk.team[np.clip(nearest, 0, len(trk.team) - 1)] == HOME, 1.0, -1.0)
    return np.where(valid, sign, 0.0)


def _runs_of(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _team_touch_features(trk, b, d, d_home, d_away, inside, fps) -> dict:
    """Touch-level evidence: control episodes, dribbles, receptions and kicks (antisymmetric)."""
    n = trk.n
    dd = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(dd, axis=1)
    d_near = dd[np.arange(n), nearest]
    seen = np.isfinite(d_near) & inside
    contested = seen & (d_home < 1.5) & (d_away < 1.5)
    other = np.where(trk.team[nearest] == HOME, d_away, d_home)
    margin = np.where(np.isnan(other), 10.0, other) - d_near

    # ball and player velocity vectors (centred differences)
    vb = np.full((n, 2), np.nan)
    if n > 4:
        vb[2:-2] = (b[4:] - b[:-4]) / (4.0 / fps)
    p = trk.players.astype(np.float64)
    vp = np.full(p.shape, np.nan)
    if n > 6:
        vp[3:-3] = (p[6:] - p[:-6]) / (6.0 / fps)
    v_near = vp[np.arange(n), nearest]
    rel = np.linalg.norm(vb - v_near, axis=1)
    speed = np.linalg.norm(vb, axis=1)

    out = {}
    # control episodes: >= 5 consecutive frames within 1.2 m of the same team, not contested
    ctrl = seen & (d_near < 1.2) & (margin > 0.5) & ~contested
    team_sign = _signed_team(trk, nearest, ctrl)
    episode = np.zeros(n)
    for sign in (1.0, -1.0):
        for s, e in _runs_of(team_sign == sign):
            if e - s >= 5:
                episode[s:e] = sign
    out["t_episode"] = episode
    out["t_episode_1"] = np.nan_to_num(rolling_mean(np.where(seen, episode, np.nan), 25))
    # dribble: nearest player within 1.5 m moving with the ball
    drib = seen & (d_near < 1.5) & (rel < 2.5) & ~contested
    out["t_dribble_1"] = np.nan_to_num(rolling_mean(np.where(seen, _signed_team(trk, nearest, drib),
                                                             np.nan), 25))
    # receptions (ball slows sharply next to a player) and kicks (ball speeds up next to one)
    dv = np.full(n, np.nan)
    if n > 6:
        dv[3:-3] = speed[6:] - speed[:-6]
    near2 = seen & (d_near < 2.0)
    recv = np.where(near2 & (dv < -4.0), _signed_team(trk, nearest, near2), 0.0)
    kick = np.where(near2 & (dv > 4.0), _signed_team(trk, nearest, near2), 0.0)
    # a reception is evidence from that moment on; a kick is evidence for the moments before it
    k = int(round(0.6 * fps))
    recv_fwd = np.convolve(recv, np.r_[np.zeros(k), np.ones(k + 1)], mode="same")
    kick_bwd = np.convolve(kick, np.r_[np.ones(k + 1), np.zeros(k)], mode="same")
    out["t_recv"] = np.clip(recv_fwd, -3, 3)
    out["t_kick"] = np.clip(kick_bwd, -3, 3)
    out["t_contested_close"] = np.where(contested, 0.0, np.nan_to_num(
        np.where(seen, np.exp(-d_home / 1.5) - np.exp(-d_away / 1.5), 0.0)))
    return out


def _shape_features(trk: Tracking, fps: float) -> dict:
    """Ball-free, antisymmetric team-shape features, oriented by each half's attacking direction."""
    n = trk.n
    p = trk.players.astype(np.float64)
    home = trk.team == HOME
    away = trk.team == AWAY
    length, width = trk.pitch
    # +1 if home attacks towards x = length in that block (home's mean x is the smaller one)
    orient = np.zeros(n)
    with np.errstate(invalid="ignore"):
        hx = np.nanmean(p[:, home, 0], axis=1)
        ax = np.nanmean(p[:, away, 0], axis=1)
    for a, b in (trk.blocks or [(0, n)]):
        sign = 1.0 if np.nanmean(hx[a:b]) < np.nanmean(ax[a:b]) else -1.0
        orient[a:b] = sign
    with np.errstate(invalid="ignore"):
        allx = np.nanmean(p[:, :, 0], axis=1)
        vx = np.full(n, np.nan)
        vx[25:-25] = (allx[50:] - allx[:-50]) / 2.0
        h_wid = np.nanstd(p[:, home, 1], axis=1)
        a_wid = np.nanstd(p[:, away, 1], axis=1)
        h_len = np.nanstd(p[:, home, 0], axis=1)
        a_len = np.nanstd(p[:, away, 0], axis=1)
    out = {}
    block = orient * (allx - length / 2.0) / (length / 2.0)
    out["s_block_2"] = np.nan_to_num(rolling_mean(block, 50))
    out["s_vel_2"] = np.nan_to_num(rolling_mean(orient * vx, 50))
    out["s_width_2"] = np.nan_to_num(rolling_mean((h_wid - a_wid) / 10.0, 50))
    out["s_length_2"] = np.nan_to_num(rolling_mean((h_len - a_len) / 10.0, 50))
    out["s_block_6"] = np.nan_to_num(rolling_mean(block, 150))
    out["s_vel_6"] = np.nan_to_num(rolling_mean(orient * vx, 150))
    out["s_width_6"] = np.nan_to_num(rolling_mean((h_wid - a_wid) / 10.0, 150))
    out["s_length_6"] = np.nan_to_num(rolling_mean((h_len - a_len) / 10.0, 150))
    return out


TEAM_FEATURES_V1 = ("t_close", "t_adv", "t_control", "t_close_02", "t_control_02", "t_close_1",
                    "t_control_1", "t_close_3", "t_control_3")
TEAM_TOUCH = ("t_episode", "t_episode_1", "t_dribble_1", "t_recv", "t_kick", "t_contested_close")
TEAM_SHAPE = ("s_block_2", "s_vel_2", "s_width_2", "s_length_2")
TEAM_SHAPE6 = ("s_block_6", "s_vel_6", "s_width_6", "s_length_6")
TEAM_FEATURES = TEAM_FEATURES_V1
DEAD_FEATURES = ("d_speed_1", "d_speed_3", "d_speed_10", "d_slow_1", "d_seen_1", "d_seen_5",
                 "d_out_1", "d_bspeed_1", "d_still_2", "d_gap_log", "d_gap_edge", "d_gap_out",
                 "d_centre", "d_nearest")


def matrix(features: dict, names) -> np.ndarray:
    return np.column_stack([features[k] for k in names]).astype(np.float64)
