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
SPEED_LEAD_FRAMES = 3 * FPS      # dead-ball onset: players slow down after the ball dies, so look ahead
EXIT_VELOCITY_FRAMES = 5         # ball velocity for the exit projection: last 0.2 s of the track
EXIT_PROJECT_MAX_FRAMES = 2 * FPS  # a ball unseen for longer is not extrapolated any further
SHAPE_SMOOTH_FRAMES = 2 * FPS    # ball-free shape differences are smoothed over 2 s
SHAPE_VEL_HALF_SPAN = 4 * FPS    # block velocity compares positions 4 s either side
SHAPE_Z_CLIP = 3.0
SET_PIECE_CODE = "set pieces"
RESTART_SIGMA_FRAMES = FPS       # width of the restart bump around the end of a set-piece window
RESTART_TEAM_AFTER_FRAMES = 4 * FPS  # the team taking a set piece is the likely owner this long after it
FRESH_EXIT_FRAMES = 5 * FPS      # stoppage onset: the first seconds after the ball left the pitch
FRESH_EXIT_PERSIST_FRAMES = 2 * FPS  # ... counted only if the ball stays out this long (W6-S)


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


def ball_exit_features(ball_xy: np.ndarray) -> dict:
    """Has the ball left the pitch and not been seen back inside since?

    ball_exited: 1 while the last seen ball position is beyond the lines, or its last velocity
    carries it beyond them within EXIT_PROJECT_MAX_FRAMES (the tracker loses a ball that leaves
    through a crowd); 0 when the ball is seen inside the lines; NaN before the first sighting.
    Stays 1 through the whole gap, until the ball is seen inside again.
    """
    n = len(ball_xy)
    idx = np.arange(n)
    present = ~np.isnan(ball_xy[:, 0])
    last = np.maximum.accumulate(np.where(present, idx, -1))
    safe = np.maximum(last, 0)
    prev = np.maximum(safe - EXIT_VELOCITY_FRAMES, 0)
    steps = np.maximum(safe - prev, 1)
    velocity = np.where(present[prev][:, None], (ball_xy[safe] - ball_xy[prev]) / steps[:, None], 0.0)
    horizon = np.minimum(idx - safe, EXIT_PROJECT_MAX_FRAMES)[:, None]
    projected = ball_xy[safe] + velocity * horizon
    with np.errstate(invalid="ignore"):
        beyond = np.any((projected < 0) | (projected > 1), axis=1)
    return {"ball_exited": np.where(last >= 0, beyond, np.nan).astype(np.float32)}


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


TOUCH_RADIUS_M = 1.5             # a player this close to the ball is playing it
TOUCH_MARGIN_M = 0.5             # ... if no opponent is about as close
TOUCH_MIN_FRAMES = 5             # a ball flying past a player is nearer for fewer frames
PASSBY_SPAN_FRAMES = 5           # ball velocity before / after a touch is averaged over 0.2 s
PASSBY_MIN_SPEED_M_S = 3.0       # a ball that never slows below this while "touched" ...
PASSBY_MEDIAN_FRAMES = 5         # positions are median-filtered first (tracker jitter)
PASSBY_FLOOR_MULTIPLE = 2.0      # threshold rises with the match's own jitter floor (median change)


def ball_velocity_change(ball_xy: np.ndarray, pitch_m, span: int = PASSBY_SPAN_FRAMES) -> np.ndarray:
    """|v(after) - v(before)| in m/s at each frame, velocities averaged over `span` frames on
    either side of median-filtered positions; NaN where either side has an unseen ball. A kick or
    a controlling touch changes the ball's velocity; a ball flying past a player does not."""
    from scipy.ndimage import median_filter
    seen = ~np.isnan(ball_xy[:, 0])
    filled = np.where(seen[:, None], ball_xy, 0.0).astype(np.float32)
    xy = np.stack([median_filter(filled[:, k], PASSBY_MEDIAN_FRAMES, mode="nearest")
                   for k in range(2)], axis=1) * np.asarray(pitch_m, dtype=np.float32)
    xy[~seen] = np.nan
    before = np.full_like(xy, np.nan)
    before[span:] = (xy[span:] - xy[:-span]) * FPS / span
    after = np.full_like(xy, np.nan)
    after[:-span] = before[span:]
    return np.sqrt(np.sum((after - before) ** 2, axis=1)).astype(np.float32)


def _passby_runs(vote: np.ndarray, starts: np.ndarray, lengths: np.ndarray, ball_speed: np.ndarray,
                 velocity_change: np.ndarray, min_change_m_s: float) -> np.ndarray:
    """Per-run flag: the ball stayed fast through the whole run and its velocity never changed by
    min_change_m_s within PASSBY_SPAN_FRAMES of it -- it flew past the player untouched."""
    n = len(vote)
    flags = np.zeros(len(starts), dtype=bool)
    for k, (a, length) in enumerate(zip(starts, lengths)):
        if vote[a] == 0:
            continue
        b = a + length
        speed = ball_speed[a:b]
        change = velocity_change[max(a - PASSBY_SPAN_FRAMES, 0):min(b + PASSBY_SPAN_FRAMES, n)]
        if np.all(np.isnan(speed)) or np.all(np.isnan(change)):
            continue
        flags[k] = (np.nanmin(speed) > PASSBY_MIN_SPEED_M_S) and (np.nanmax(change) < min_change_m_s)
    return flags


def touch_features(feats: dict, radius_m: float = TOUCH_RADIUS_M, margin_m: float = TOUCH_MARGIN_M,
                   min_frames: int = TOUCH_MIN_FRAMES, passby_dv_m_s: float = 0.0,
                   ball_xy: np.ndarray | None = None, pitch_m=None) -> dict:
    """Who last played the ball: the ground truth keeps the owner through flights and loose balls
    and switches at the opponent's first touch, so ownership is a causal "last team near the
    ball" memory rather than a centred control window.

    touch_vote: +1 / -1 while the nearest home / away player is within radius_m of a visible ball
        and the other team is at least margin_m farther, sustained for min_frames; else 0.
        With passby_dv_m_s > 0 (needs ball_xy and pitch_m), a run during which the ball stays
        faster than PASSBY_MIN_SPEED_M_S and never changes velocity by passby_dv_m_s is a ball
        flying past, not a touch, and is dropped (on Metrica such runs are 11% of votes at 47%
        precision; the rest are at 84%).
    owner_sign: last non-zero vote (0 before the first); owner_age: frames since it (inf before).
    """
    home = np.where(np.isnan(feats["nearest_home_m"]), np.inf, feats["nearest_home_m"])
    away = np.where(np.isnan(feats["nearest_away_m"]), np.inf, feats["nearest_away_m"])
    with np.errstate(invalid="ignore"):
        vote = np.where((home < radius_m) & (away - home > margin_m), 1.0,
                        np.where((away < radius_m) & (home - away > margin_m), -1.0, 0.0))
    n = len(vote)
    if min_frames > 1 and n:
        starts = np.flatnonzero(np.concatenate(([True], vote[1:] != vote[:-1])))
        lengths = np.diff(np.concatenate((starts, [n])))
        short = np.repeat(lengths < min_frames, lengths)
        vote = np.where(short, 0.0, vote)
    if passby_dv_m_s > 0 and n:
        if ball_xy is None or pitch_m is None:
            raise ValueError("the pass-by filter needs ball_xy and pitch_m")
        starts = np.flatnonzero(np.concatenate(([True], vote[1:] != vote[:-1])))
        lengths = np.diff(np.concatenate((starts, [n])))
        change = ball_velocity_change(ball_xy, pitch_m)
        # the tracker's jitter floor, read off the match itself (label-free): a typical frame
        # has no touch, so the median change is noise, and a real change must stand above it
        floor = float(np.nanmedian(change)) if np.any(~np.isnan(change)) else 0.0
        passby = _passby_runs(vote, starts, lengths, feats["ball_speed_m_s"], change,
                              passby_dv_m_s + PASSBY_FLOOR_MULTIPLE * floor)
        vote = np.where(np.repeat(passby, lengths), 0.0, vote)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(vote != 0, idx, -1))
    owner = np.where(last >= 0, vote[np.maximum(last, 0)], 0.0)
    age = np.where(last >= 0, idx - last, np.inf)
    return {"touch_vote": vote.astype(np.float32), "owner_sign": owner.astype(np.float32),
            "owner_age": age.astype(np.float32)}


BALL_MOVING_M_S = 5.0            # a ball seen travelling this fast is in play (above the jitter)


def ball_motion_features(ball_speed: np.ndarray, window: int = BALL_WINDOW_FRAMES,
                         moving_m_s: float = BALL_MOVING_M_S) -> dict:
    """Share of a centred 1 s window in which the ball was seen travelling faster than
    moving_m_s (unseen frames count as not moving). An alive clue that needs a real ball: the
    players slow down only after the whistle and speed up only after the restart kick."""
    with np.errstate(invalid="ignore"):
        moving = (ball_speed > moving_m_s).astype(np.float64)
    return {"ball_moving_frac": uniform_filter1d(moving, window, mode="constant").astype(np.float32)}


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
            out["{}_centroid_x_norm".format(name)] = np.nanmean(xy[:, :, 0], axis=1).astype(np.float32)
            out["{}_centroid_y_norm".format(name)] = np.nanmean(xy[:, :, 1], axis=1).astype(np.float32)
            out["{}_spread_m".format(name)] = (np.nanstd(xy[:, :, 0], axis=1)
                                               * match.pitch_m[0]).astype(np.float32)
            out["{}_mean_speed_m_s".format(name)] = np.nanmean(sp, axis=1).astype(np.float32)
            out["{}_n_tracked".format(name)] = np.sum(~np.isnan(xy[:, :, 0]), axis=1).astype(np.float32)
        out["all_mean_speed_m_s"] = np.nanmean(speed, axis=1).astype(np.float32)
        out["frac_players_slow"] = np.nanmean(
            np.where(np.isnan(speed), np.nan, speed < PLAYER_SLOW_M_S), axis=1).astype(np.float32)
    smooth = _moving_mean(out["all_mean_speed_m_s"], SHAPE_WINDOW_FRAMES)
    out["all_mean_speed_smooth_m_s"] = smooth.astype(np.float32)
    # min(now, SPEED_LEAD ahead): moves the start of a slow stretch earlier without delaying its
    # end (when play restarts the speed only rises, so the minimum is still "now")
    ahead = np.full_like(smooth, np.nan)
    ahead[:-SPEED_LEAD_FRAMES] = smooth[SPEED_LEAD_FRAMES:]
    out["all_mean_speed_lead_m_s"] = np.fmin(smooth, ahead).astype(np.float32)
    return out


OUT_PLAYER_MARGIN_M = 0.5        # a player standing this far beyond a line is off the pitch
OUT_PLAYER_MAX_SPEED_M_S = 1.0   # ... and standing (a thrower, a ball retriever)
EDGE_M = 2.0                     # a slow ball this close to a line is at a restart spot
EDGE_MAX_SPEED_M_S = 2.0
FAR_BALL_M = 2.5                 # a slow ball nobody is within this distance of ...
FAR_BALL_MAX_SPEED_M_S = 2.0     # ... is waiting for a restart (loose balls get picked up)
QUIET_PLAYERS_M_S = 1.8          # a ball that travels under QUIET_TRAVEL_M in 2 s with nobody within
QUIET_TRAVEL_M = 8.0             # QUIET_FAR_M while the players walk: nobody is playing (a ball
QUIET_FAR_M = 2.0                # carried or drifting during a stoppage)


def ball_travel_m(ball_xy: np.ndarray, pitch_m, span: int) -> np.ndarray:
    """Distance in metres between the ball positions `span` frames before and after each frame,
    positions median-filtered over 0.4 s so that tracker jitter does not read as travel; NaN
    unless both are seen."""
    from scipy.ndimage import median_filter
    seen = ~np.isnan(ball_xy[:, 0])
    filled = np.where(seen[:, None], ball_xy, 0.0).astype(np.float64)
    smooth = np.stack([median_filter(filled[:, k], 10, mode="nearest") for k in range(2)], axis=1)
    smooth[~seen] = np.nan
    out = np.full(len(ball_xy), np.nan, dtype=np.float32)
    delta = (smooth[2 * span:] - smooth[:-2 * span]) * np.asarray(pitch_m, dtype=np.float64)
    out[span:-span] = np.sqrt(np.sum(delta * delta, axis=1))
    return out


def fresh_exit_features(ball_exited: np.ndarray) -> dict:
    """fresh_exit: 1 during the first FRESH_EXIT_FRAMES after the ball left the pitch (seen or
    projected, ball_exited switching on), while it has not been seen back inside, and only for
    exits that last at least FRESH_EXIT_PERSIST_FRAMES (a ball skimming a line is no stoppage).
    v5's dead clues stay flat for 3-5 s after a real exit; this clue is 20-150 times likelier on
    the first seconds of a stoppage than in play on both external judges (W6-S)."""
    n = len(ball_exited)
    idx = np.arange(n)
    exited = np.nan_to_num(ball_exited) > 0.5
    rise = exited & ~np.concatenate(([False], exited[:-1]))
    last_rise = np.maximum.accumulate(np.where(rise, idx, -1))
    next_off = np.minimum.accumulate(np.where(~exited, idx, n)[::-1])[::-1]
    lasting = (last_rise >= 0) & (next_off[np.maximum(last_rise, 0)] - last_rise >= FRESH_EXIT_PERSIST_FRAMES)
    fresh = exited & (last_rise >= 0) & (idx - last_rise <= FRESH_EXIT_FRAMES) & lasting
    return {"fresh_exit": fresh.astype(np.float32)}


def restart_features(match: Match, feats: dict) -> dict:
    """Where restarts happen: a standing player beyond the lines (throw-in, retrieving the ball;
    needs no ball), a slow ball at the edge of the pitch, a slow ball nobody is near, and the
    "quiet" conjunction of a slow far ball with walking players. All as shares of a centred 1 s
    window. On the external judges a standing out-of-pitch player is 6-50 times likelier on dead
    frames, and the quiet conjunction has an alive false-positive rate of 0.0-0.5 % on every judge
    (scripts/restart_clue_stats.py, scripts/tracked_dead_stats.py). Needs team_shape_features."""
    length, width = match.pitch_m
    xy, speed = match.players_xy, match.players_speed
    with np.errstate(invalid="ignore"):
        beyond = np.maximum.reduce([(-xy[:, :, 0]) * length, (xy[:, :, 0] - 1.0) * length,
                                    (-xy[:, :, 1]) * width, (xy[:, :, 1] - 1.0) * width])
        out_player = np.any((beyond > OUT_PLAYER_MARGIN_M) & (speed < OUT_PLAYER_MAX_SPEED_M_S), axis=1)
        bx, by = feats["ball_x_norm"], feats["ball_y_norm"]
        edge = np.minimum(np.minimum(bx, 1 - bx) * length, np.minimum(by, 1 - by) * width)
        edge_slow = (edge < EDGE_M) & (feats["ball_speed_m_s"] < EDGE_MAX_SPEED_M_S)
        nearest = np.fmin(feats["nearest_home_m"], feats["nearest_away_m"])
        slow_far = (feats["ball_speed_m_s"] < FAR_BALL_MAX_SPEED_M_S) & (nearest > FAR_BALL_M)
        # jitter-robust ball travel: distance between 0.4 s-median positions 1 s either side
        quiet = (ball_travel_m(match.ball_xy, match.pitch_m, FPS) < QUIET_TRAVEL_M) \
            & (nearest > QUIET_FAR_M) & (feats["all_mean_speed_smooth_m_s"] < QUIET_PLAYERS_M_S)
    return {
        "restart_quiet_frac": uniform_filter1d(np.nan_to_num(quiet.astype(np.float64)),
                                               2 * FPS, mode="constant").astype(np.float32),
        "ball_slow_far_frac": uniform_filter1d(np.nan_to_num(slow_far.astype(np.float64)),
                                               BALL_WINDOW_FRAMES, mode="constant").astype(np.float32),
        "out_player_frac": uniform_filter1d(out_player.astype(np.float64), BALL_WINDOW_FRAMES,
                                            mode="constant").astype(np.float32),
        "ball_edge_slow_frac": uniform_filter1d(np.nan_to_num(edge_slow.astype(np.float64)),
                                                BALL_WINDOW_FRAMES, mode="constant").astype(np.float32),
    }


def _row_percentiles(values: np.ndarray, quantiles) -> np.ndarray:
    """Per-row percentiles ignoring NaN, vectorised (np.nanpercentile loops in Python per row).
    Nearest-rank on the sorted valid values; NaN rows stay NaN. Returns (len(quantiles), n)."""
    ordered = np.sort(values, axis=1)               # NaN sorts last
    count = np.sum(~np.isnan(values), axis=1)
    rows = np.arange(values.shape[0])
    out = []
    for q in quantiles:
        rank = np.clip(np.round(q / 100.0 * (count - 1)).astype(int), 0, None)
        picked = ordered[rows, np.minimum(rank, values.shape[1] - 1)]
        out.append(np.where(count > 0, picked, np.nan))
    return np.stack(out)


def attack_direction(match: Match) -> np.ndarray:
    """+1 where HOME attacks towards +x, -1 towards -x, 0 outside the graded halves.

    Inferred per half from where each team keeps its goalkeeper: the side whose line-most
    player sits closest to the goal line (median over the half) is the side that team defends.
    Both teams are inferred and must agree, otherwise the half is left at 0 (no shape evidence).
    """
    direction = np.zeros(match.n_frames, dtype=np.float32)
    for _, first, last in match.halves:
        defends_left = {}
        for name, team in (("home", HOME), ("away", AWAY)):
            x = match.players_xy[first:last + 1, match.team_of_slot == team, 0]
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                left = np.nanmedian(np.nanmin(x, axis=1))
                right = 1.0 - np.nanmedian(np.nanmax(x, axis=1))
            defends_left[name] = bool(left < right)
        if defends_left["home"] != defends_left["away"]:
            direction[first:last + 1] = 1.0 if defends_left["home"] else -1.0
    return direction


def _robust_z(values: np.ndarray, graded: np.ndarray) -> np.ndarray:
    """Centre on the median and scale by the std over graded frames: removes a team's constant
    style (always wider, always longer) so that only the home-away difference in time remains."""
    ref = values[graded & ~np.isnan(values)]
    if len(ref) < 2 or ref.std() < 1e-6:
        return np.full(values.shape, np.nan, dtype=np.float32)
    z = np.clip((values - np.median(ref)) / ref.std(), -SHAPE_Z_CLIP, SHAPE_Z_CLIP)
    return z.astype(np.float32)


def ball_free_shape_features(match: Match) -> dict:
    """Home-minus-away differences that need no ball, as z-scores (positive favours home).

    Attackers stretch the pitch: the team in possession is longer (front line minus back line along
    its attack direction) and wider, the defending team runs more, and the whole block drifts
    towards the possessor's goal. Measured against clear-control frames this separates the teams
    well; it is the only team signal left while the ball is unseen.
    shape_len_z, shape_width_z, shape_speed_z, shape_vel_z; NaN where the direction is unknown.
    """
    direction = attack_direction(match)
    graded = match.graded_mask() & (direction != 0)
    xy, speed = match.players_xy, match.players_speed
    length, width, mean_speed = {}, {}, {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for team in (HOME, AWAY):
            side = 1.0 if team == HOME else -1.0
            mask = match.team_of_slot == team
            along = xy[:, mask, 0] * (direction * side)[:, None]
            lines = _row_percentiles(along, (10, 90))
            length[team] = lines[1] - lines[0]
            width[team] = np.nanstd(xy[:, mask, 1], axis=1) * match.pitch_m[1]
            mean_speed[team] = np.nanmean(speed[:, mask], axis=1)
        block_x = np.nanmean(xy[:, :, 0], axis=1)
    span = SHAPE_VEL_HALF_SPAN
    block_smooth = _moving_mean(block_x.astype(np.float64), FPS)
    velocity = np.full(match.n_frames, np.nan)
    velocity[span:-span] = (block_smooth[2 * span:] - block_smooth[:-2 * span]) / (2 * span / FPS)
    diffs = {"shape_len_z": length[HOME] - length[AWAY],
             "shape_width_z": width[HOME] - width[AWAY],
             "shape_speed_z": mean_speed[HOME] - mean_speed[AWAY],
             "shape_vel_z": velocity * direction}
    out = {}
    for name, diff in diffs.items():
        smooth = _moving_mean(np.where(direction != 0, diff, np.nan).astype(np.float64),
                              SHAPE_SMOOTH_FRAMES)
        out[name] = _robust_z(smooth, graded)
    return out


def tagging_features(tags: list, n_frames: int) -> dict:
    """Soft priors from Smart Tagging instances ({code, start_frame, end_frame, team, type}).

    tag_set_piece: 1 inside a 'set pieces' window (any type), else 0.
    restart_prior: bump in 0..1 centred on the end of a set-piece window (the likely restart), for
        the decoder to make leaving D cheaper there.
    restart_team: +1 / -1 from 1 s before to 4 s after the end of a set-piece window whose tag
        names HOME / AWAY (the taker owns the ball afterwards); 0 otherwise. Throw-ins carry no team.
    """
    set_piece = np.zeros(n_frames, dtype=np.float32)
    restart = np.zeros(n_frames, dtype=np.float32)
    restart_team = np.zeros(n_frames, dtype=np.float32)
    idx = np.arange(n_frames)
    for tag in tags:
        if tag["code"] != SET_PIECE_CODE:
            continue
        first, last = max(int(tag["start_frame"]), 0), min(int(tag["end_frame"]), n_frames - 1)
        if last < first:
            continue
        set_piece[first:last + 1] = 1.0
        taker = {HOME: 1.0, AWAY: -1.0}.get(tag["team"], 0.0)
        restart_team[max(first, last - FPS):last + RESTART_TEAM_AFTER_FRAMES + 1] = taker
        np.maximum(restart, np.exp(-0.5 * ((idx - last) / RESTART_SIGMA_FRAMES) ** 2), out=restart)
    return {"tag_set_piece": set_piece, "restart_prior": restart, "restart_team": restart_team}


def compute(match: Match, tags: list | None = None, params: dict | None = None) -> dict:
    """All features for every frame of the recording; Smart Tagging priors only when `tags` given.
    params: optional overrides of the touch rule (touch_radius_m, touch_margin_m, touch_min_frames)."""
    p = params or {}
    feats = ball_features(match)
    feats.update(gap_features(feats["ball_present"] > 0))
    feats.update(control_features(feats))
    feats.update(ball_exit_features(match.ball_xy))
    feats.update(fresh_exit_features(feats["ball_exited"]))
    feats.update(ball_motion_features(feats["ball_speed_m_s"],
                                      moving_m_s=p.get("ball_moving_m_s", BALL_MOVING_M_S)))
    feats["ball_out_frac"] = _moving_mean(feats["ball_out"].astype(np.float64),
                                          BALL_WINDOW_FRAMES).astype(np.float32)
    feats.update(team_memory_features(feats["control_balance"]))
    feats.update(touch_features(feats, p.get("touch_radius_m", TOUCH_RADIUS_M),
                                p.get("touch_margin_m", TOUCH_MARGIN_M),
                                int(p.get("touch_min_frames", TOUCH_MIN_FRAMES)),
                                float(p.get("touch_passby_dv_m_s", 0.0)),
                                match.ball_xy, match.pitch_m))
    feats.update(team_shape_features(match))
    feats.update(ball_free_shape_features(match))
    feats.update(restart_features(match, feats))
    if tags is not None:
        feats.update(tagging_features(tags, match.n_frames))
    return feats
