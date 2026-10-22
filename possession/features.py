"""Per-frame features from tracking arrays and Smart Tagging. Pure numpy; no file access, no printing.

Every feature is a (n_frames,) float32 array in a dict. NaN means "not computable on this frame"
(for example, ball features when the ball is missing). Names carry units: _m metres, _m_s speed,
_frac a fraction of frames in a centred window, _norm pitch-normalised 0-1 coordinates.

Windowed features are centred (the pipeline is offline, so look-ahead is allowed).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

from possession import FPS
from possession.io import AWAY, HOME, Match

CONTROL_RADIUS_M = 2.0           # a player this close to the ball can be controlling it
CONTROL_WINDOW_FRAMES = 3 * FPS // 2   # control is sustained proximity, a touch is not enough
CONTROL_MAX_SPEED_M_S = 8.0      # a faster ball is in flight or being passed: nobody controls it
CLEAR_CONTROL = 0.3              # |control_balance| from which a frame counts as clear control
BALL_SPEED_HALF_SPAN = 2         # ball speed is measured over positions this many frames either side
BALL_WINDOW_FRAMES = FPS         # out-of-bounds share is taken over 1 s
SHAPE_WINDOW_FRAMES = 2 * FPS    # team speed is smoothed over 2 s
PLAYER_SLOW_M_S = 0.5
SET_PIECE_CODE = "set pieces"
RESTART_SIGMA_FRAMES = FPS       # width of the restart bump around the end of a set-piece window


def _dist_m(a_xy: np.ndarray, b_xy: np.ndarray, pitch_m) -> np.ndarray:
    """Euclidean distance in metres between normalised points; broadcasts on leading axes."""
    scale = np.asarray(pitch_m, dtype=np.float32)
    delta = (a_xy - b_xy) * scale
    return np.sqrt(np.sum(delta * delta, axis=-1))


def _moving_mean(values: np.ndarray, size: int) -> np.ndarray:
    """Centred moving mean ignoring NaN; NaN where the window holds no valid sample."""
    valid = ~np.isnan(values)
    total = uniform_filter1d(np.where(valid, values, 0.0).astype(np.float64), size, mode="constant")
    count = uniform_filter1d(valid.astype(np.float64), size, mode="constant")
    with np.errstate(all="ignore"):
        return np.where(count > 1e-9, total / count, np.nan)


def nearest_by_team(match: Match, ball_xy: np.ndarray):
    """Distance in metres from `ball_xy` (n, 2) to the nearest tracked player of each team."""
    dist = _dist_m(match.players_xy, ball_xy[:, None, :], match.pitch_m)  # (n, slots)
    out = []
    for team in (HOME, AWAY):
        masked = np.where(match.team_of_slot == team, dist, np.nan)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # halftime rows are all NaN
            out.append(np.nanmin(masked, axis=1).astype(np.float32))
    return out[0], out[1]


def ball_speed_from_positions(ball_xy: np.ndarray, pitch_m, half_span: int = BALL_SPEED_HALF_SPAN):
    """Ball speed in m/s from positions `half_span` frames either side; NaN unless both are seen.

    The tracker's own ball speed is missing on ~95% of frames, so speed is always derived here.
    """
    delta = np.full_like(ball_xy, np.nan)
    delta[half_span:-half_span] = ball_xy[2 * half_span:] - ball_xy[:-2 * half_span]
    metres = delta * np.asarray(pitch_m, dtype=np.float32)
    return (np.sqrt(np.sum(metres * metres, axis=1)) * FPS / (2 * half_span)).astype(np.float32)


def ball_features(match: Match) -> dict:
    """Distances from the ball to the nearest player of each team, ball kinematics and bounds."""
    ball = match.ball_xy
    ball_present = ~np.isnan(ball[:, 0])
    nearest_home, nearest_away = nearest_by_team(match, ball)
    with np.errstate(invalid="ignore"):
        out_of_bounds = (ball[:, 0] < 0) | (ball[:, 0] > 1) | (ball[:, 1] < 0) | (ball[:, 1] > 1)
    return {
        "ball_present": ball_present.astype(np.float32),
        "nearest_home_m": nearest_home,
        "nearest_away_m": nearest_away,
        "ball_speed_m_s": ball_speed_from_positions(ball, match.pitch_m),
        "ball_x_norm": ball[:, 0].astype(np.float32),
        "ball_y_norm": ball[:, 1].astype(np.float32),
        "ball_out": np.where(ball_present, out_of_bounds, np.nan).astype(np.float32),
    }


def gap_features(ball_present: np.ndarray) -> dict:
    """Frames since the ball was last seen and until it is next seen (inf at the ends)."""
    n = len(ball_present)
    idx = np.arange(n)
    last = np.where(ball_present, idx, -1)
    last = np.maximum.accumulate(last)
    since = np.where(last >= 0, idx - last, np.inf)
    nxt = np.where(ball_present, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    until = np.where(nxt < n, nxt - idx, np.inf)
    return {"frames_since_ball": since.astype(np.float32),
            "frames_until_ball": until.astype(np.float32)}


def control_features(feats: dict) -> dict:
    """Who has the ball under control, from sustained proximity of a slow-enough ball.

    control_balance: windowed (home near - away near), in -1..1. Frames with the ball unseen, in
    flight, or far from everyone count as "nobody near", so absence dilutes rather than votes.
    """
    controllable = ~(feats["ball_speed_m_s"] > CONTROL_MAX_SPEED_M_S)  # unknown speed: allow
    near_home = (feats["nearest_home_m"] < CONTROL_RADIUS_M) & controllable
    near_away = (feats["nearest_away_m"] < CONTROL_RADIUS_M) & controllable
    balance = uniform_filter1d(near_home.astype(np.float64) - near_away.astype(np.float64),
                               CONTROL_WINDOW_FRAMES, mode="constant")
    return {"control_balance": balance.astype(np.float32)}


def team_memory_features(control_balance: np.ndarray) -> dict:
    """Which team last / next showed clear control, and how many frames away that was.

    Ball gaps last ~10 s (median), far too long to interpolate the ball, but the control seen at
    both ends of a gap says who had it before and who has it after: evidence can then place a
    hand-over at the middle of the gap instead of leaving it arbitrary. Sign +1 home, -1 away;
    frames are inf (sign 0) when no such frame exists before / after.
    """
    n = len(control_balance)
    idx = np.arange(n)
    clear = np.abs(control_balance) >= CLEAR_CONTROL
    last = np.maximum.accumulate(np.where(clear, idx, -1))
    nxt = np.minimum.accumulate(np.where(clear, idx, n)[::-1])[::-1]
    sign = np.sign(control_balance)
    return {
        "last_team_sign": np.where(last >= 0, sign[np.maximum(last, 0)], 0.0).astype(np.float32),
        "next_team_sign": np.where(nxt < n, sign[np.minimum(nxt, n - 1)], 0.0).astype(np.float32),
        "frames_since_team": np.where(last >= 0, idx - last, np.inf).astype(np.float32),
        "frames_until_team": np.where(nxt < n, nxt - idx, np.inf).astype(np.float32),
    }


def team_shape_features(match: Match) -> dict:
    """Collective features that need no ball: centroids, spread, activity per team."""
    out = {}
    speed = match.players_speed
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # halftime rows are all NaN
        for name, team in (("home", HOME), ("away", AWAY)):
            mask = match.team_of_slot == team
            xy = match.players_xy[:, mask, :]
            sp = speed[:, mask]
            out["{}_centroid_x".format(name)] = np.nanmean(xy[:, :, 0], axis=1).astype(np.float32)
            out["{}_centroid_y".format(name)] = np.nanmean(xy[:, :, 1], axis=1).astype(np.float32)
            out["{}_spread_m".format(name)] = (np.nanstd(xy[:, :, 0], axis=1)
                                               * match.pitch_m[0]).astype(np.float32)
            out["{}_mean_speed_m_s".format(name)] = np.nanmean(sp, axis=1).astype(np.float32)
            out["{}_n_tracked".format(name)] = np.sum(~np.isnan(xy[:, :, 0]), axis=1).astype(np.float32)
        out["all_mean_speed_m_s"] = np.nanmean(speed, axis=1).astype(np.float32)
        out["frac_players_slow"] = np.nanmean(
            np.where(np.isnan(speed), np.nan, speed < PLAYER_SLOW_M_S), axis=1).astype(np.float32)
    out["all_mean_speed_smooth_m_s"] = _moving_mean(
        out["all_mean_speed_m_s"], SHAPE_WINDOW_FRAMES).astype(np.float32)
    return out


def tagging_features(tags: list, n_frames: int) -> dict:
    """Soft priors from Smart Tagging instances ({code, start_frame, end_frame, team, type}).

    tag_set_piece: 1 inside a 'set pieces' window (any type), else 0.
    restart_prior: bump in 0..1 centred on the end of a set-piece window (the likely restart), for
        the decoder to make leaving D cheaper there.
    """
    set_piece = np.zeros(n_frames, dtype=np.float32)
    restart = np.zeros(n_frames, dtype=np.float32)
    idx = np.arange(n_frames)
    for tag in tags:
        if tag["code"] != SET_PIECE_CODE:
            continue
        first, last = max(int(tag["start_frame"]), 0), min(int(tag["end_frame"]), n_frames - 1)
        if last < first:
            continue
        set_piece[first:last + 1] = 1.0
        np.maximum(restart, np.exp(-0.5 * ((idx - last) / RESTART_SIGMA_FRAMES) ** 2), out=restart)
    return {"tag_set_piece": set_piece, "restart_prior": restart}


def compute(match: Match, tags: list | None = None) -> dict:
    """All features for every frame of the recording; Smart Tagging priors only when `tags` given."""
    feats = ball_features(match)
    feats.update(gap_features(feats["ball_present"] > 0))
    feats.update(control_features(feats))
    feats["ball_out_frac"] = _moving_mean(feats["ball_out"].astype(np.float64),
                                          BALL_WINDOW_FRAMES).astype(np.float32)
    feats.update(team_memory_features(feats["control_balance"]))
    feats.update(team_shape_features(match))
    if tags is not None:
        feats.update(tagging_features(tags, match.n_frames))
    return feats
