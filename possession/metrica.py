"""Metrica Sports sample matches (public, github.com/metrica-sports/sample-data) as Match + labels.

Same provider as the competition match: 25 fps, normalised 0-1 coordinates on 105x68 m, and
event data synchronised to the tracking. Labels are derived from the events with the "last team
to play the ball" semantics that the public ground truth shows (0.8 s possessions after a single
touch): owner = team of the latest ball-playing event (set piece, pass, carry, recovery, shot,
ball lost); dead from a ball-out / foul / goal / offside until the next set piece. These are
event-derived labels, not corrected-tracking truth: use them for relative comparisons.

Downloaded by scripts/fetch_metrica.py into the work dir; parsed once and cached as npz.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from possession import D, H, A, X
from possession.external import ExternalMatch, speed_m_s
from possession.io import AWAY, HOME, Match, work_dir

GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
PITCH_M = (105.0, 68.0)
FPS = 25
TOUCH_TYPES = {"SET PIECE", "PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST"}


def metrica_dir() -> Path:
    return work_dir() / "external" / "metrica"


def cache_dir() -> Path:
    path = work_dir() / "external" / "metrica_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- pure label derivation ---------------------------------------------------------------------

def labels_from_events(events: list[dict], periods: dict[int, tuple[int, int]]) -> np.ndarray:
    """Per-frame codes on a 0-based timeline from event records
    {team: HOME/AWAY/None, type, subtype (upper-case str), period, start, end (frames on the
    same 0-based timeline as the codes)}.

    periods: {period: (first_frame, last_frame)} inclusive. Frames outside periods are X.
    """
    n = max(last for _, last in periods.values()) + 1
    codes = np.full(n, X, dtype=np.int8)
    by_period: dict[int, list[dict]] = {}
    for event in events:
        by_period.setdefault(int(event["period"]), []).append(event)
    for period, (first, last) in periods.items():
        rows = sorted(by_period.get(period, []), key=lambda e: (e["start"], e["end"]))
        owner, dead = None, True  # dead until the kick-off set piece
        cursor = first
        for event in rows:
            kind, sub = event["type"], event["subtype"]
            start = int(min(max(event["start"], first), last))
            if kind == "SET PIECE":
                codes[cursor:start] = _code(owner, dead)
                owner, dead, cursor = event["team"], False, start
            elif kind in TOUCH_TYPES:
                codes[cursor:start] = _code(owner, dead)
                if event["team"] is not None:
                    owner = event["team"]
                cursor = start
                if kind == "SHOT" and "GOAL" in sub and "OWN" not in sub:
                    stop = int(min(max(event["end"], start), last))
                    codes[cursor:stop] = _code(owner, dead)
                    dead, cursor = True, stop
                elif kind == "BALL LOST" and ("OFFSIDE" in sub or "END HALF" in sub):
                    dead = True
            elif kind == "BALL OUT":
                stop = int(min(max(event["end"], start), last))
                codes[cursor:stop] = _code(owner, dead)
                dead, cursor = True, stop
            elif kind == "FAULT RECEIVED":
                codes[cursor:start] = _code(owner, dead)
                dead, cursor = True, start
        codes[cursor:last + 1] = _code(owner, dead)
    return codes


def _code(owner, dead: bool) -> int:
    if dead or owner is None:
        return D
    return H if owner == HOME else A


# -- parsing -----------------------------------------------------------------------------------

def _team_code(name: str):
    key = (name or "").strip().lower()
    if key in ("home", "team a"):
        return HOME
    if key in ("away", "team b"):
        return AWAY
    return None


def read_events_csv(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        out = []
        for row in reader:
            out.append({
                "team": _team_code(row["Team"]), "type": row["Type"].strip().upper(),
                "subtype": (row["Subtype"] or "").strip().upper(), "period": int(row["Period"]),
                "start": int(row["Start Frame"]), "end": int(row["End Frame"]),
            })
    return out


def read_events_json(path: Path) -> list[dict]:
    with open(path) as handle:
        data = json.load(handle)["data"]
    out = []
    for event in data:
        subs = event.get("subtypes")
        if isinstance(subs, list):
            sub = "-".join(s["name"] for s in subs)
        elif isinstance(subs, dict):
            sub = subs.get("name", "")
        else:
            sub = ""
        team = event.get("team") or {}
        out.append({
            "team": _team_code(team.get("name", "")), "type": event["type"]["name"].strip().upper(),
            "subtype": (sub or "").strip().upper(), "period": int(event["period"]),
            "start": int(event["start"]["frame"]), "end": int(event["end"]["frame"]),
        })
    return out


def read_tracking_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(period, frame_id, players_xy (m, k, 2), ball_xy (m, 2)) from one team's CSV
    (three header rows: team, jersey numbers, column names; last two columns are the ball)."""
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader), next(reader)
        names = next(reader)
        rows = [r for r in reader if r]
    values = np.array([[float(v) if v not in ("", "NaN", "nan") else np.nan for v in r]
                       for r in rows], dtype=np.float64)
    if values.shape[1] != len(names) or len(names) < 5 or (len(names) - 5) % 2:
        raise ValueError("unexpected tracking column layout: {} columns".format(len(names)))
    period = values[:, 0].astype(np.int8)
    frame_id = values[:, 1].astype(np.int64)
    k = (len(names) - 5) // 2
    players = values[:, 3:3 + 2 * k].reshape(len(values), k, 2).astype(np.float32)
    ball = values[:, -2:].astype(np.float32)
    return period, frame_id, players, ball


def load_csv_game(game: str) -> ExternalMatch:
    base = metrica_dir()
    ph, fh, home_xy, ball_h = read_tracking_csv(base / f"{game}_RawTrackingData_Home_Team.csv")
    pa, fa, away_xy, ball_a = read_tracking_csv(base / f"{game}_RawTrackingData_Away_Team.csv")
    if not np.array_equal(fh, fa) or not np.array_equal(ph, pa):
        raise ValueError("home and away tracking files disagree on frames")
    ball = np.where(np.isnan(ball_h), ball_a, ball_h)
    players = np.concatenate([home_xy, away_xy], axis=1)
    team_of_slot = np.array([HOME] * home_xy.shape[1] + [AWAY] * away_xy.shape[1], dtype=np.int8)
    events = read_events_csv(base / f"{game}_RawEventsData.csv")
    return _assemble(game, ph, fh, players, ball, team_of_slot, events)


def load_epts_game(game: str) -> ExternalMatch:
    from kloppy import metrica
    base = metrica_dir()
    dataset = metrica.load_tracking_epts(
        meta_data=str(base / f"{game}_metadata.xml"), raw_data=str(base / f"{game}_tracking.txt"),
        coordinates="metrica")
    meta = dataset.metadata
    home_team, away_team = meta.teams
    players = list(home_team.players) + list(away_team.players)
    slot_of = {p.player_id: i for i, p in enumerate(players)}
    team_of_slot = np.array([HOME] * len(home_team.players) + [AWAY] * len(away_team.players),
                            dtype=np.int8)
    frames = dataset.records
    m = len(frames)
    frame_id = np.empty(m, dtype=np.int64)
    period = np.empty(m, dtype=np.int8)
    ball = np.full((m, 2), np.nan, dtype=np.float32)
    xy = np.full((m, len(players), 2), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        frame_id[i] = frame.frame_id
        period[i] = frame.period.id
        if frame.ball_coordinates is not None:
            ball[i] = (frame.ball_coordinates.x, frame.ball_coordinates.y)
        for player, data in frame.players_data.items():
            if data.coordinates is not None:
                xy[i, slot_of[player.player_id]] = (data.coordinates.x, data.coordinates.y)
    events = read_events_json(base / f"{game}_events.json")
    return _assemble(game, period, frame_id, xy, ball, team_of_slot, events)


def _assemble(game: str, period: np.ndarray, frame_id: np.ndarray, players_xy: np.ndarray,
              ball_xy: np.ndarray, team_of_slot: np.ndarray, events: list[dict]) -> ExternalMatch:
    base = int(frame_id.min())
    idx = frame_id - base
    n = int(idx.max()) + 1
    scale = np.asarray(PITCH_M, dtype=np.float32)
    players = np.full((n, players_xy.shape[1], 2), np.nan, dtype=np.float32)
    players[idx] = players_xy
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[idx] = ball_xy
    periods = {int(p): (int(idx[period == p].min()), int(idx[period == p].max()))
               for p in np.unique(period)}
    shifted = [dict(e, start=e["start"] - base, end=e["end"] - base) for e in events]
    codes = labels_from_events(shifted, periods)
    halves = [(p, first, last) for p, (first, last) in sorted(periods.items())]
    match = Match(players_xy=players, players_speed=speed_m_s(players * scale, FPS),
                  ball_xy=ball, ball_speed=speed_m_s(ball * scale, FPS),
                  team_of_slot=team_of_slot, halves=halves, pitch_m=PITCH_M, source="metrica")
    return ExternalMatch(match, codes, FPS, np.zeros((0, 3), dtype=np.int64))


def available_games() -> list[str]:
    base = metrica_dir()
    out = []
    for game in GAMES:
        needed = ([f"{game}_events.json", f"{game}_metadata.xml", f"{game}_tracking.txt"]
                  if game == "Sample_Game_3" else
                  [f"{game}_RawEventsData.csv", f"{game}_RawTrackingData_Home_Team.csv",
                   f"{game}_RawTrackingData_Away_Team.csv"])
        if all((base / name).exists() for name in needed):
            out.append(game)
    return out


def load_game(game: str, refresh: bool = False) -> ExternalMatch:
    from possession.external import load_cache, save_cache
    cache = cache_dir() / f"{game}.npz"
    if cache.exists() and not refresh:
        return load_cache(cache)
    ext = load_epts_game(game) if game == "Sample_Game_3" else load_csv_game(game)
    save_cache(cache, ext)
    return ext
