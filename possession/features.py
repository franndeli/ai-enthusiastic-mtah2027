"""Per-frame features from tracking arrays. Pure numpy; no file access, no printing.

Every feature is a (n_frames,) float32 array in a dict. NaN means "not computable on this frame"
(for example, ball features when the ball is missing). Names carry units: _m metres, _m_s speed.
"""
from __future__ import annotations

import numpy as np

from possession.io import AWAY, HOME, Match


def _dist_m(a_xy: np.ndarray, b_xy: np.ndarray, pitch_m) -> np.ndarray:
    """Euclidean distance in metres between normalised points; broadcasts on leading axes."""
    scale = np.asarray(pitch_m, dtype=np.float32)
    delta = (a_xy - b_xy) * scale
    return np.sqrt(np.sum(delta * delta, axis=-1))


def ball_features(match: Match) -> dict:
    """Distances from the ball to the nearest player of each team and ball kinematics."""
    pitch = match.pitch_m
    ball = match.ball_xy[:, None, :]
    dist = _dist_m(match.players_xy, ball, pitch)  # (n, slots), NaN where either is missing
    home = match.team_of_slot == HOME
    away = match.team_of_slot == AWAY
    with np.errstate(all="ignore"):
        nearest_home = np.nanmin(np.where(home, dist, np.nan), axis=1)
        nearest_away = np.nanmin(np.where(away, dist, np.nan), axis=1)
    ball_present = ~np.isnan(match.ball_xy[:, 0])
    return {
        "ball_present": ball_present.astype(np.float32),
        "nearest_home_m": nearest_home.astype(np.float32),
        "nearest_away_m": nearest_away.astype(np.float32),
        "ball_speed_m_s": match.ball_speed.astype(np.float32),
        "ball_x_norm": match.ball_xy[:, 0].astype(np.float32),
        "ball_y_norm": match.ball_xy[:, 1].astype(np.float32),
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


def team_shape_features(match: Match) -> dict:
    """Collective features that need no ball: centroids, spread, activity per team."""
    out = {}
    speed = match.players_speed
    for name, team in (("home", HOME), ("away", AWAY)):
        mask = match.team_of_slot == team
        xy = match.players_xy[:, mask, :]
        sp = speed[:, mask]
        with np.errstate(all="ignore"):
            out["{}_centroid_x".format(name)] = np.nanmean(xy[:, :, 0], axis=1).astype(np.float32)
            out["{}_centroid_y".format(name)] = np.nanmean(xy[:, :, 1], axis=1).astype(np.float32)
            out["{}_spread_m".format(name)] = (np.nanstd(xy[:, :, 0], axis=1)
                                               * match.pitch_m[0]).astype(np.float32)
            out["{}_mean_speed_m_s".format(name)] = np.nanmean(sp, axis=1).astype(np.float32)
            out["{}_n_tracked".format(name)] = np.sum(~np.isnan(xy[:, :, 0]), axis=1).astype(np.float32)
    with np.errstate(all="ignore"):
        out["all_mean_speed_m_s"] = np.nanmean(speed, axis=1).astype(np.float32)
        out["frac_players_slow"] = np.nanmean((speed < 0.5).astype(np.float32), axis=1).astype(np.float32)
    return out


def compute(match: Match) -> dict:
    """All features for every frame of the recording."""
    feats = ball_features(match)
    feats.update(gap_features(feats["ball_present"] > 0))
    feats.update(team_shape_features(match))
    return feats
