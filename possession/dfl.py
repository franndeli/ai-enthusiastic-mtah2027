"""Sportec Open DFL / IDSSE data (Bassek, Rein, Weber & Memmert 2025, Scientific Data 12, 195;
CC-BY 4.0): seven 2022/23 Bundesliga and 2. Bundesliga matches with 25 Hz positions and, on every
frame, `BallPossession` (home / away) and `BallStatus` (alive / dead) -- the two things our labels
encode. Other games only, used as external training and validation data.

`to_arrays` turns a kloppy TrackingDataset (loaded with coordinates="sportec") into compact arrays
(centimetres as int16, origin at a corner, y growing downwards like the rest of the pipeline);
`load` reads them back as (Tracking, labels). kloppy is only needed for the conversion.
"""

from __future__ import annotations

import numpy as np

from . import H, A, D, X
from .tracking import HOME, AWAY, Tracking

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
_NAN16 = np.int16(-32768)


def _to_cm(xy: np.ndarray) -> np.ndarray:
    out = np.full(xy.shape, _NAN16, dtype=np.int16)
    ok = ~np.isnan(xy)
    out[ok] = np.clip(np.round(xy[ok] * 100.0), -32767, 32767).astype(np.int16)
    return out


def _from_cm(a: np.ndarray) -> np.ndarray:
    out = a.astype(np.float32) / 100.0
    out[a == _NAN16] = np.nan
    return out


def to_arrays(dataset) -> dict:
    """Arrays for one match from a kloppy Sportec TrackingDataset in provider coordinates."""
    from kloppy.domain import BallState
    dims = dataset.metadata.pitch_dimensions
    length, width = float(dims.pitch_length), float(dims.pitch_width)
    home, away = dataset.metadata.teams[0], dataset.metadata.teams[1]
    players = list(home.players) + list(away.players)
    slot = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([HOME] * len(home.players) + [AWAY] * len(away.players), dtype=np.int8)
    periods = {}
    for frame in dataset.frames:
        periods.setdefault(frame.period.id, []).append(frame)
    balls, plays, labs, blocks, start = [], [], [], [], 0
    for pid in sorted(periods):
        frames = periods[pid]
        first = frames[0].frame_id
        n = frames[-1].frame_id - first + 1
        ball = np.full((n, 2), np.nan)
        pos = np.full((n, len(players), 2), np.nan)
        lab = np.full(n, X, dtype=np.int8)
        for frame in frames:
            i = frame.frame_id - first
            b = frame.ball_coordinates
            if b is not None and b.x is not None:
                ball[i] = (b.x, b.y)
            for player, data in frame.players_data.items():
                c = data.coordinates
                k = slot.get(player.player_id)
                if c is not None and k is not None:
                    pos[i, k] = (c.x, c.y)
            if frame.ball_state == BallState.DEAD:
                lab[i] = D
            elif frame.ball_owning_team is not None:
                lab[i] = H if frame.ball_owning_team.team_id == home.team_id else A
        balls.append(ball)
        plays.append(pos)
        labs.append(lab)
        blocks.append((start, start + n))
        start += n
    ball, pos, lab = np.concatenate(balls), np.concatenate(plays), np.concatenate(labs)
    # provider coordinates: metres from the centre spot, y up -> metres from a corner, y down
    for xy in (ball, pos):
        xy[..., 0] += length / 2.0
        xy[..., 1] = width / 2.0 - xy[..., 1]
    return {"ball_cm": _to_cm(ball), "players_cm": _to_cm(pos), "team": team, "labels": lab,
            "blocks": np.array(blocks, dtype=np.int64).reshape(-1, 2),
            "pitch": np.array([length, width]),
            "fps": np.array(float(dataset.metadata.frame_rate or 25.0))}


def controlled(labels: np.ndarray, blocks, fps: float, min_s: float = 2.0) -> np.ndarray:
    """DFL's BallPossession changes on every touch (370-511 possessions per match, a quarter to a
    third of them under 2 s). CHALLENGE.md counts a change only when the new team gains controlled
    possession ("a player keeps the ball for a few seconds"), so inside each alive stretch a run of
    the other team shorter than `min_s` stays with the team that had the ball."""
    out = labels.copy()
    for a, b in blocks:
        seg = out[a:b]
        alive = np.isin(seg, (H, A))
        cut = np.flatnonzero(np.diff(alive.astype(np.int8)) != 0) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(seg)]):
            if not alive[s]:
                continue
            run = seg[s:e]
            edges = np.flatnonzero(np.diff(run) != 0) + 1
            prev = None
            for rs, re_ in zip(np.r_[0, edges], np.r_[edges, len(run)]):
                team = int(run[rs])
                if prev is not None and team != prev and (re_ - rs) < min_s * fps:
                    run[rs:re_] = prev
                else:
                    prev = team
        out[a:b] = seg
    return out


def save(arrays: dict, path: str) -> None:
    np.savez_compressed(path, **arrays)


def load(path: str):
    """(Tracking, labels int8 H/A/D/X) for one converted match."""
    z = np.load(path)
    trk = Tracking(ball=_from_cm(z["ball_cm"]), players=_from_cm(z["players_cm"]), team=z["team"],
                   fps=float(z["fps"]), pitch=tuple(float(v) for v in z["pitch"]),
                   blocks=[(int(a), int(b)) for a, b in z["blocks"]])
    return trk, z["labels"]
