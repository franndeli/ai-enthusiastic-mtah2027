"""MTAH ATD adapter: the Metrica FIFA/EPTS metadata XML + raw text pair -> FrameTable.

Rows are tactical video frames (raw frame_id f is row f-1, as in starter/baseline.py). Errors name
line indices and field counts only, never data values (docs/DATA_POLICY.md).
"""

import sys

import numpy as np

import games
from possession.frames import FrameTable
from starter.baseline import team_of_each_track

# Raw ATD line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>", NaN for a missing value.
FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","


def _xy(group: str) -> list:
    """First two values of one "x,y,speed" group; a bare "NaN" group counts as missing."""
    values = group.split(VALUE_SEP)
    return values[:2] if len(values) >= 2 else ["nan", "nan"]


def parse_raw(raw_path, n_frames: int, n_slots: int):
    """(ball_xy_norm (n, 2), players_xy_norm (n, k, 2)) float32, all-NaN for absent frames."""
    ball = np.full((n_frames, 2), np.nan, np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, np.float32)
    with open(raw_path) as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError(f"line {line_idx}: {len(parts)} ':' fields, expected 3")
            row = int(parts[0]) - 1  # frame_id counts from 1, rows from 0
            if not 0 <= row < n_frames:
                raise ValueError(f"line {line_idx}: frame_id outside 1..{n_frames}")
            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError(f"line {line_idx}: {len(groups)} player groups, expected {n_slots}")
            flat = parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP)
            if len(flat) == 3 * n_slots:  # fast path: every group is a full x,y,speed triple
                players[row] = np.array(flat, np.float32).reshape(n_slots, 3)[:, :2]
            else:
                players[row] = np.array([_xy(g) for g in groups], np.float32)
            ball[row] = np.array(_xy(parts[2]), np.float32)
    return ball, players


def load_atd(meta_path, raw_path, n_frames: int, pitch_m: tuple, half_blocks: list,
             fps: float = 25.0) -> FrameTable:
    """FrameTable in metres (origin at pitch centre); period from [(half, first, last)] video blocks."""
    slot_team = team_of_each_track(meta_path)
    ball, players = parse_raw(raw_path, n_frames, len(slot_team))
    length, width = pitch_m
    scale = np.array([length, width], np.float32)
    centre = scale / 2
    period = np.zeros(n_frames, np.int8)
    for half, first, last in half_blocks:
        period[max(first, 0):last + 1] = half
    return FrameTable(fps=float(fps), pitch_m=(float(length), float(width)), period=period,
                      ball_xy_m=ball * scale - centre, players_xy_m=players * scale - centre,
                      slot_team=slot_team)


def load_game(game: str = games.GAME, data_dir=None) -> FrameTable:
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    return load_atd(meta_path, raw_path, games.source_frames("tactical", game),
                    games.GAMES[game]["pitch"], games.half_video_blocks(game),
                    games.GAMES[game]["fps"])


if __name__ == "__main__":
    try:
        table = load_game()
    except (IOError, ValueError) as error:
        sys.exit(f"cannot load ATD: {error}")
    play = table.period > 0
    ball_seen = ~np.isnan(table.ball_xy_m[play, 0])
    tracked = (~np.isnan(table.players_xy_m[play, :, 0])).sum(axis=1)
    print(f"frames {table.n_frames}, slots {len(table.slot_team)}, play frames {int(play.sum())}")
    print(f"ball visible on play frames: {100 * ball_seen.mean():.1f}%")
    print(f"mean tracked players on play frames: {tracked.mean():.2f}")
