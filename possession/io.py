"""Load the match assets into numpy arrays and write submissions. The only module that touches files.

Never prints or logs data values: errors name fields, shapes and frame indices only.
"""
from __future__ import annotations

import hashlib
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks, require, source_frames
from possession import FPS, LABELS

log = logging.getLogger(__name__)

HOME, AWAY = 0, 1
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"


@dataclass
class Match:
    """Tracking for one recording, indexed by that recording's video frame (0-based).

    players_xy: (n_frames, n_slots, 2) normalised 0-1, NaN when the slot is not observed.
    players_speed: (n_frames, n_slots) m/s, NaN when missing.
    ball_xy: (n_frames, 2) normalised, NaN when the ball is not observed.
    ball_speed: (n_frames,) m/s, NaN when missing.
    team_of_slot: (n_slots,) HOME=0 / AWAY=1.
    halves: [(half, first_frame, last_frame)] graded blocks on this timeline (tactical only).
    pitch_m: (length, width) in metres, for converting normalised distances.
    """
    players_xy: np.ndarray
    players_speed: np.ndarray
    ball_xy: np.ndarray
    ball_speed: np.ndarray
    team_of_slot: np.ndarray
    halves: list
    pitch_m: tuple
    source: str = "tactical"

    @property
    def n_frames(self) -> int:
        return int(self.players_xy.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.players_xy.shape[1])

    def graded_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_frames, dtype=bool)
        for _, first, last in self.halves:
            mask[first:last + 1] = True
        return mask


def work_dir() -> Path:
    path = Path(os.environ.get("MTAH_WORK_DIR") or Path.home() / ".mtah2027" / "work")
    path.mkdir(parents=True, exist_ok=True)
    return path


def team_of_each_slot(meta_path: str) -> np.ndarray:
    """Slot -> HOME/AWAY from the EPTS metadata; the first Team listed is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in ATD metadata, found {}".format(len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("no Player entries in ATD metadata")
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def parse_atd(raw_path: str, n_frames: int, n_slots: int):
    """Stream the raw ATD text into arrays. Lines are '<frame_id>:<p>;<p>;...:<x,y,s>'."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    seen = 0
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            if not 0 <= frame < n_frames:
                raise ValueError("ATD frame_id out of range at video frame {}".format(frame))
            seen += 1
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot >= n_slots or not group or group[0] == MISSING:
                    continue
                values = group.split(",")
                players[frame, slot, 0] = float(values[0])
                players[frame, slot, 1] = float(values[1])
                if len(values) > 2 and values[2] and values[2][0] != MISSING:
                    players[frame, slot, 2] = float(values[2])
            group = parts[2]
            if group and group[0] != MISSING:
                values = group.split(",")
                ball[frame, 0] = float(values[0])
                ball[frame, 1] = float(values[1])
                if len(values) > 2 and values[2] and values[2][0] != MISSING:
                    ball[frame, 2] = float(values[2])
    if seen != n_frames:
        log.warning("ATD has %d records for %d video frames", seen, n_frames)
    return players, ball


def load_match(game: str = GAME, source: str = "tactical", cache: bool = True) -> Match:
    """The tracking of one recording, cached as .npz in MTAH_WORK_DIR keyed by file size."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    n_frames = source_frames(source, game)
    team_of_slot = team_of_each_slot(meta_path)
    halves = half_video_blocks(game) if source == "tactical" else []
    pitch_m = GAMES[game]["pitch"]

    key = hashlib.sha1("{}:{}:{}".format(source, os.path.getsize(raw_path),
                                         len(team_of_slot)).encode()).hexdigest()[:12]
    cache_path = work_dir() / "atd_{}_{}.npz".format(source, key)
    if cache and cache_path.exists():
        with np.load(cache_path) as blob:
            players, ball = blob["players"], blob["ball"]
    else:
        players, ball = parse_atd(raw_path, n_frames, len(team_of_slot))
        if cache:
            np.savez_compressed(cache_path, players=players, ball=ball)

    return Match(players_xy=players[:, :, :2], players_speed=players[:, :, 2],
                 ball_xy=ball[:, :2], ball_speed=ball[:, 2], team_of_slot=team_of_slot,
                 halves=halves, pitch_m=pitch_m, source=source)


def load_smart_tagging(game: str = GAME) -> list:
    """[{code, start_frame, end_frame, team, half, type}] on the tactical timeline.

    team is HOME/AWAY or None (36 instances, every throw-in, are tagged N/A).
    """
    from starter.load_smart_tagging import load as _load
    home_name, _ = team_names(game)
    out = []
    for event in _load(game):
        tag_team = (event["tags"].get("Team") or "").strip().lower()
        team = None
        if tag_team and tag_team != "n/a":
            team = HOME if tag_team == home_name.strip().lower() else AWAY
        out.append({
            "code": (event["code"] or "").strip().lower(),
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "team": team,
            "half": event["tags"].get("Half"),
            "type": (event["tags"].get("Type") or "").strip().lower(),
        })
    return out


def team_names(game: str = GAME):
    """(home, away) club names as spelled in the ATD metadata."""
    meta_path = require(["atd_meta"], game)[0]
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def write_submission(path: str, codes: np.ndarray) -> str:
    """frame,label CSV from an int8 code array (H=0, A=1, D=2, X=3)."""
    codes = np.asarray(codes)
    if codes.ndim != 1:
        raise ValueError("codes must be 1-D, got shape {}".format(codes.shape))
    if codes.min() < 0 or codes.max() > 3:
        raise ValueError("codes outside 0..3")
    letters = np.array(LABELS)[codes]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, letter) for i, letter in enumerate(letters))
    return path


def load_public_gt(game: str = GAME) -> np.ndarray:
    """Public ground truth as codes (X where ungraded); public file, safe to read."""
    import sys
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "evaluation"))
    import possession_metrics as pm  # noqa: E402
    path = root / "evaluation" / "gt" / "{}_possession_gt_public.csv".format(game)
    return pm.label_codes(pm.read_label_csv(str(path)))
