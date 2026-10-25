"""Tracking container and readers.

Two sources share one representation:

* the challenge ATD (FIFA/EPTS metadata XML + raw text), read by `read_epts`;
* Metrica's public sample games (CSV for games 1-2, EPTS for game 3), used only as external
  validation data.

Coordinates are converted from the normalised 0-1 convention (origin top-left, x along the
length) to metres, keeping the same origin. Missing observations are NaN.

Array index = provider frame number - 1. For the challenge ATD that is the video frame, i.e. the
submission row (DATA.md: video_frame = frame_id - 1).
"""

from __future__ import annotations

import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1


@dataclass
class Tracking:
    ball: np.ndarray            # (N, 2) float32, metres, NaN = not seen
    players: np.ndarray         # (N, S, 2) float32, metres, NaN = slot absent
    team: np.ndarray            # (S,) int8, 0 = home, 1 = away
    fps: float
    pitch: tuple                # (length, width) in metres
    blocks: list = field(default_factory=list)   # [(start, stop)] half-open, one per half
    extra: dict = field(default_factory=dict)    # further channels, e.g. {"ball_z": (N,)}

    @property
    def n(self) -> int:
        return int(self.ball.shape[0])


# -- EPTS (challenge ATD and Metrica Sample Game 3) ----------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(node, name):
    return [c for c in node if _local(c.tag) == name]


def parse_epts_metadata(meta_path: str) -> dict:
    """Teams, players, frame rate, pitch and the per-frame-range data format specifications."""
    root = ET.parse(meta_path).getroot()
    nodes = {}
    for node in root.iter():
        nodes.setdefault(_local(node.tag), []).append(node)

    fps = float(nodes["FrameRate"][0].text) if "FrameRate" in nodes else 25.0
    teams = [t.get("id") for t in nodes.get("Team", [])]
    home_id = teams[0] if teams else None
    for score in nodes.get("Score", []):          # explicit local team when present
        if score.get("idLocalTeam"):
            home_id = score.get("idLocalTeam")
    players = [(p.get("id"), p.get("teamId")) for p in nodes.get("Player", [])]
    pitch = None
    if "FieldSize" in nodes:
        fs = nodes["FieldSize"][0]
        width = _children(fs, "Width")
        height = _children(fs, "Height")
        if width and height:
            pitch = (float(width[0].text), float(height[0].text))

    channel_of = {}
    for pc in nodes.get("PlayerChannel", []):
        channel_of[pc.get("id")] = (pc.get("playerId"), pc.get("channelId"))

    specs = []
    for spec in nodes.get("DataFormatSpecification", []):
        groups, ball = [], []
        for reg in spec.iter():
            if _local(reg.tag) != "SplitRegister":
                continue
            refs = _children(reg, "PlayerChannelRef")
            if refs:
                group = []
                for ref in refs:
                    player_id, channel = channel_of.get(ref.get("playerChannelId"), (None, None))
                    group.append((player_id, channel))
                groups.append(group)
            balls = _children(reg, "BallChannelRef")
            if balls:
                ball = [b.get("channelId") for b in balls]
        specs.append({"start": int(spec.get("startFrame")), "end": int(spec.get("endFrame")),
                      "groups": groups, "ball": ball})
    return {"fps": fps, "teams": teams, "home_id": home_id, "players": players,
            "pitch": pitch, "specs": specs}


def read_epts(meta_path: str, raw_path: str, pitch: tuple | None = None,
              n_frames: int | None = None) -> Tracking:
    """Read an EPTS pair into a Tracking. Driven by the DataFormatSpecification(s) when present.

    Without a specification (or one with no player channels) it falls back to the convention the
    official baseline relies on: one ';'-separated group per <Player>, in metadata order, each
    group "x,y,speed", and the ball as "x,y[,...]".
    """
    meta = parse_epts_metadata(meta_path)
    pitch = pitch or meta["pitch"] or (105.0, 68.0)
    player_ids = [pid for pid, _ in meta["players"]]
    index_of = {pid: i for i, pid in enumerate(player_ids)}
    team = np.array([HOME if tid == meta["home_id"] else AWAY for _, tid in meta["players"]],
                    dtype=np.int8)

    with open(raw_path) as handle:
        lines = [line.rstrip("\n\r") for line in handle if line.strip()]
    frames = np.fromiter((int(line[:line.index(":")]) for line in lines), dtype=np.int64,
                         count=len(lines))
    n = int(n_frames or frames.max())
    S = len(player_ids)

    specs = [s for s in meta["specs"] if s["groups"]]
    fallback = [{"start": int(frames.min()), "end": int(frames.max()),
                 "groups": [[(pid, "x"), (pid, "y"), (pid, "speed")] for pid in player_ids],
                 "ball": None}]
    players, ball, extra = _fill(lines, frames, specs or fallback, index_of, n, S)
    if specs and np.isnan(players).all():
        # The specification names no x/y channels we recognise: use the baseline's convention.
        players, ball, extra = _fill(lines, frames, fallback, index_of, n, S)

    length, width = pitch
    players[..., 0] *= length
    players[..., 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return Tracking(ball=ball, players=players, team=team, fps=meta["fps"], pitch=tuple(pitch),
                    blocks=[], extra=extra)


def _fill(lines, frames, specs, index_of, n, S):
    """Parse the raw lines into (players (n, S, 2), ball (n, 2), extra) following `specs`."""
    players = np.full((n, S, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    extra = {}
    table = str.maketrans({":": ",", ";": ","})
    for spec in specs:
        rows = np.flatnonzero((frames >= spec["start"]) & (frames <= spec["end"]))
        if not len(rows):
            continue
        text = "\n".join(lines[i].translate(table) for i in rows)
        values = pd.read_csv(io.StringIO(text), header=None, na_values=["NaN", "nan", ""],
                             dtype=np.float64, engine="c").to_numpy()
        idx = frames[rows] - 1
        col = 1
        for group in spec["groups"]:
            for player_id, channel in group:
                slot = index_of.get(player_id)
                if slot is not None and channel in ("x", "y"):
                    axis = 0 if channel == "x" else 1
                    players[idx, slot, axis] = values[:, col]
                elif slot is not None:
                    key = "player_" + str(channel)
                    if key not in extra:
                        extra[key] = np.full((n, S), np.nan, dtype=np.float32)
                    extra[key][idx, slot] = values[:, col]
                col += 1
        ball_channels = spec["ball"] or ["x", "y"] + ["ball_%d" % k for k in
                                                      range(values.shape[1] - col - 2)]
        for k, channel in enumerate(ball_channels):
            if col + k >= values.shape[1]:
                break
            if channel in ("x", "y"):
                ball[idx, 0 if channel == "x" else 1] = values[:, col + k]
            else:
                key = "ball_" + str(channel)
                if key not in extra:
                    extra[key] = np.full(n, np.nan, dtype=np.float32)
                extra[key][idx] = values[:, col + k]
    return players, ball, extra


# -- Metrica sample games 1 and 2 (CSV) ----------------------------------------------------------

def _read_metrica_team_csv(path: str):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    columns = list(raw.columns)
    frame = raw["Frame"].to_numpy(dtype=np.int64)
    period = raw["Period"].to_numpy(dtype=np.int64)
    players, ball = {}, None
    for i, name in enumerate(columns):
        if name.startswith("Player"):
            players[name] = raw.iloc[:, [i, i + 1]].to_numpy(dtype=np.float64)
        elif name == "Ball":
            ball = raw.iloc[:, [i, i + 1]].to_numpy(dtype=np.float64)
    return frame, period, players, ball


def read_metrica_csv(home_csv: str, away_csv: str, pitch=(105.0, 68.0)) -> tuple:
    """(Tracking, period array) for Metrica sample games 1-2."""
    frame_h, period, home, ball = _read_metrica_team_csv(home_csv)
    frame_a, _, away, _ = _read_metrica_team_csv(away_csv)
    if not np.array_equal(frame_h, frame_a):
        raise ValueError("home and away files are not frame-aligned")
    n = int(frame_h.max())
    names = list(home) + list(away)
    team = np.array([HOME] * len(home) + [AWAY] * len(away), dtype=np.int8)
    players = np.full((n, len(names), 2), np.nan, dtype=np.float32)
    idx = frame_h - 1
    for s, name in enumerate(names):
        xy = home[name] if s < len(home) else away[name]
        players[idx, s] = xy
    b = np.full((n, 2), np.nan, dtype=np.float32)
    b[idx] = ball
    per = np.zeros(n, dtype=np.int64)
    per[idx] = period
    length, width = pitch
    players[..., 0] *= length
    players[..., 1] *= width
    b[:, 0] *= length
    b[:, 1] *= width
    return Tracking(ball=b, players=players, team=team, fps=25.0, pitch=tuple(pitch)), per


# -- cache ---------------------------------------------------------------------------------------

def save_npz(trk: Tracking, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"ball": trk.ball, "players": trk.players, "team": trk.team,
               "fps": np.array(trk.fps), "pitch": np.array(trk.pitch, dtype=np.float64),
               "blocks": np.array(trk.blocks, dtype=np.int64).reshape(-1, 2)}
    for key, value in trk.extra.items():
        payload["extra__" + key] = value
    np.savez(path, **payload)


def load_npz(path: str) -> Tracking:
    data = np.load(path)
    extra = {k[len("extra__"):]: data[k] for k in data.files if k.startswith("extra__")}
    return Tracking(ball=data["ball"], players=data["players"], team=data["team"],
                    fps=float(data["fps"]), pitch=tuple(float(v) for v in data["pitch"]),
                    blocks=[tuple(int(v) for v in row) for row in data["blocks"]], extra=extra)
