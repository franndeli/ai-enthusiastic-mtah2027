"""Load the match assets into numpy arrays, write submissions, load the learned models. The only
production module that touches files, apart from pathcrf.py reading its vendored checkpoint.

Never prints or logs data values: errors name fields, shapes and frame indices only.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from possession import LABELS

HOME, AWAY = 0, 1
FIELD_SEP, GROUP_SEP, MISSING = ":", ";", "N"
PARSER_VERSION = 2


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


def _atd_float(values: list[str], column: int, frame_idx: int,
               slot_idx: int | None, field: str) -> float:
    """Convert one field without including asset tokens in exception messages."""
    try:
        return float(values[column])
    except (ValueError, IndexError, OverflowError):
        location = f"video frame index {frame_idx}"
        if slot_idx is not None:
            location += f", slot index {slot_idx}"
        raise ValueError(f"ATD {location}: invalid {field}") from None


def parse_atd(raw_path: str, n_frames: int, n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Stream ATD into arrays, rejecting malformed, duplicate and missing frame records."""
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    seen = np.zeros(n_frames, dtype=bool)
    with open(raw_path) as handle:
        for record_idx, line in enumerate(handle):
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError(f"ATD record index {record_idx}: invalid field structure")
            try:
                frame_idx = int(parts[0]) - 1
            except (ValueError, OverflowError):
                raise ValueError(f"ATD record index {record_idx}: invalid frame_id") from None
            if not 0 <= frame_idx < n_frames:
                raise ValueError(f"ATD record index {record_idx}: frame_id out of range")
            if seen[frame_idx]:
                raise ValueError(f"ATD video frame index {frame_idx}: duplicate frame_id")
            seen[frame_idx] = True
            for slot_idx, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot_idx >= n_slots or not group or group[0] == MISSING:
                    continue
                values = group.split(",")
                for column, field in enumerate(("player_x", "player_y")):
                    players[frame_idx, slot_idx, column] = _atd_float(
                        values, column, frame_idx, slot_idx, field)
                if len(values) > 2 and values[2] and values[2][0] != MISSING:
                    players[frame_idx, slot_idx, 2] = _atd_float(
                        values, 2, frame_idx, slot_idx, "player_speed")
            group = parts[2]
            if group and group[0] != MISSING:
                values = group.split(",")
                for column, field in enumerate(("ball_x", "ball_y")):
                    ball[frame_idx, column] = _atd_float(values, column, frame_idx, None, field)
                if len(values) > 2 and values[2] and values[2][0] != MISSING:
                    ball[frame_idx, 2] = _atd_float(values, 2, frame_idx, None, "ball_speed")
    if not seen.all():
        raise ValueError(f"ATD missing frame_id records: count {int((~seen).sum())}")
    return players, ball


def load_match(game: str = GAME, source: str = "tactical", cache: bool = True) -> Match:
    """Load tracking; cache identity includes source, raw size/mtime_ns, metadata SHA1,
    parser version and slot count. Wrong cached array shapes trigger reparsing.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, source=source)
    n_frames = source_frames(source, game)
    team_of_slot = team_of_each_slot(meta_path)
    halves = half_video_blocks(game) if source == "tactical" else []
    pitch_m = GAMES[game]["pitch"]

    raw_stat = Path(raw_path).stat()
    metadata_sha1 = hashlib.sha1(Path(meta_path).read_bytes()).hexdigest()
    identity = (source, raw_stat.st_size, raw_stat.st_mtime_ns, metadata_sha1,
                PARSER_VERSION, len(team_of_slot))
    key = hashlib.sha1(repr(identity).encode()).hexdigest()
    cache_path = work_dir() / "atd_{}_{}.npz".format(source, key)
    players = ball = None
    if cache and cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as blob:
                cached_players, cached_ball = blob["players"], blob["ball"]
            if (cached_players.shape == (n_frames, len(team_of_slot), 3)
                    and cached_ball.shape == (n_frames, 3)):
                players, ball = cached_players, cached_ball
        except (ValueError, OSError, KeyError, EOFError):
            pass  # Invalid caches are disposable; source parsing remains authoritative.
    if players is None:
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


def load_model(path: str | Path, sha256: str):
    """A pickled model trained on external data (scripts/train_v6.py), checked against the
    SHA-256 recorded in the config before unpickling."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"model file {Path(path).name} does not match its recorded SHA-256")
    import pickle
    return pickle.loads(data)


def save_model(path: str | Path, model) -> str:
    """Pickle a model (protocol 5) and return its SHA-256."""
    import pickle
    data = pickle.dumps(model, protocol=5)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()
