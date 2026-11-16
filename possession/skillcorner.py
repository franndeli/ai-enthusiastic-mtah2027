"""SkillCorner open data as a third external judge (github.com/SkillCorner/opendata, MIT licence):
20 A-League 2024/25 matches of broadcast tracking at 10 fps with the provider's dynamic events
and phases of play. Fetch with scripts/fetch_skillcorner.py. External data only: nothing here
reads the competition match.

Why it matters: it is real automatic tracking with the ball undetected on ~40 % of the frames
and players out of the camera view (their positions are the provider's extrapolation), so it
needs no synthetic noise. Its labels are derived by us from provider-generated events, so they
are weaker than corrected tracking (docs/research/third_provider.md); two variants:

* "event" (first built by a teammate on branch feat/possession-pipeline-v5): H/A is the team of
  the last player possession, carried through passes and loose balls (CHALLENGE.md: a loose
  ball stays with the last team). D runs from a possession followed by a game interruption
  (the ball seen leaving the pitch within 4 s of its end, else its end for a foul, else 1 s
  later) to the restart before the next possession (the last moment the seen ball was nearly
  still within 5 s of the reception, else 1 s before it), and before a period's first possession.
  From the restart to that reception the ball belongs to the team taking the set piece (the
  receiver's team, or its opponent when the provider marks the restart as "against" it).
* "phase": H/A is the team in possession of the provider's phase of play, D everything else.

Tracking is resampled to 25 fps: positions are interpolated only between consecutive 10 fps
samples; the ball counts as seen only where the provider marks it detected.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from possession import A, D, H, X
from possession.io import AWAY, HOME, Match, work_dir

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
FILES = ("tracking.jsonl", "match.json", "dynamic_events.csv", "phases.csv")
FPS_IN, FPS_OUT = 10, 25
TICKS_IN, TICKS_OUT = 5, 2          # both rates on a common 50 Hz clock
HALF_GAP_FRAMES = 10 * FPS_OUT      # ungraded frames between the halves on the joined timeline
OUT_MARGIN_M = 0.3                  # a seen ball this far beyond a line has left the pitch
OUT_LOOK_FRAMES = 4 * FPS_IN
OUT_DEFAULT_FRAMES = FPS_IN         # ... else the stoppage starts 1 s after the possession ends
STILL_M_S = 1.0                     # restart: the seen ball nearly still ...
STILL_LOOK_FRAMES = 5 * FPS_IN      # ... within 5 s before the first reception
RESTART_DEFAULT_FRAMES = FPS_IN     # ... else 1 s before it
FOUL_ENDS = ("foul_suffered", "foul_committed")
LABEL_KINDS = ("event", "phase")
CACHE_VERSION = 1


def skillcorner_dir() -> Path:
    return work_dir() / "external" / "skillcorner" / COMMIT


def cache_dir() -> Path:
    path = work_dir() / "external" / "skillcorner_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def available_matches() -> list[str]:
    base = skillcorner_dir()
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if all((d / f).is_file() for f in FILES))


# -- pure conversion ---------------------------------------------------------------------------

def resample_positions(frames: np.ndarray, values: np.ndarray, target_ticks: np.ndarray) -> np.ndarray:
    """values at 10 fps source `frames` -> 50 Hz `target_ticks`: the exact sample where one
    exists, linear interpolation between two consecutive source frames, NaN otherwise."""
    ticks = np.asarray(frames, dtype=np.int64) * TICKS_IN
    right = np.searchsorted(ticks, target_ticks, side="left")
    out = np.full((len(target_ticks),) + values.shape[1:], np.nan, dtype=np.float32)
    exact = (right < len(ticks)) & (ticks[np.minimum(right, len(ticks) - 1)] == target_ticks)
    out[exact] = values[right[exact]]
    at = np.flatnonzero(~exact & (right > 0) & (right < len(ticks)))
    at = at[ticks[right[at]] - ticks[right[at] - 1] == TICKS_IN]
    r = right[at]
    w = ((target_ticks[at] - ticks[r - 1]) / float(TICKS_IN)).reshape((-1,) + (1,) * (values.ndim - 1))
    out[at] = values[r - 1] + w * (values[r] - values[r - 1])
    return out


def central_speed(xy_m: np.ndarray, fps: int = FPS_OUT) -> np.ndarray:
    """Speed (m/s) from positions one frame either side; NaN where a neighbour is missing."""
    speed = np.full(xy_m.shape[:-1], np.nan, dtype=np.float32)
    if len(xy_m) > 2:
        speed[1:-1] = np.sqrt(np.sum((xy_m[2:] - xy_m[:-2]) ** 2, axis=-1)) * (fps / 2.0)
    return speed


def event_labels(n: int, periods: list[tuple[int, int]], starts: np.ndarray, ends: np.ndarray,
                 teams: np.ndarray, before: np.ndarray, after: np.ndarray, foul_end: np.ndarray,
                 taker: np.ndarray, ball_m: np.ndarray, pitch_m: tuple) -> np.ndarray:
    """"event" labels on the 10 fps source frames 0..n-1 (X outside the periods).

    periods: half-open [first, stop) source-frame ranges. Per player possession, sorted by
    start: starts / ends (source frames), teams (H or A), before / after (a game interruption
    precedes / follows it), foul_end (it ended with a foul), taker (the team that took the
    restart preceding it; its own team when none). ball_m: (n, 2) seen ball in metres
    with the origin at a corner, NaN when not detected."""
    lab = np.full(n, X, dtype=np.int8)
    length, width = pitch_m
    seen = ~np.isnan(ball_m[:, 0])
    with np.errstate(invalid="ignore"):
        out = seen & ((ball_m[:, 0] < -OUT_MARGIN_M) | (ball_m[:, 0] > length + OUT_MARGIN_M)
                      | (ball_m[:, 1] < -OUT_MARGIN_M) | (ball_m[:, 1] > width + OUT_MARGIN_M))
        step = np.full(n, np.nan)
        step[1:] = np.linalg.norm(np.diff(ball_m, axis=0), axis=1) * FPS_IN
        still = seen & (step < STILL_M_S)
    for a, b in periods:
        sel = np.flatnonzero((starts >= a) & (starts < b))
        owner = np.searchsorted(starts[sel], np.arange(a, b), side="right") - 1
        team = np.where(owner >= 0, teams[sel][np.maximum(owner, 0)], -1)
        dead = team < 0                       # before the period's first possession
        for i, k in enumerate(sel):
            if not after[k]:
                continue
            end = int(min(max(ends[k], a), b))
            look = np.flatnonzero(out[end:min(b, end + OUT_LOOK_FRAMES)])
            d0 = end + int(look[0]) if len(look) else end if foul_end[k] else end + OUT_DEFAULT_FRAMES
            if i + 1 == len(sel):
                d1 = b                        # nothing restarts before the period ends
            else:
                later = sel[i + 1:]
                flagged = later[before[later]]
                first = flagged[0] if len(flagged) else later[0]
                r = int(starts[first])
                window = np.arange(max(d0, r - STILL_LOOK_FRAMES), r)
                quiet = window[still[window]] if len(window) else window
                d1 = int(quiet[-1]) + 1 if len(quiet) else max(d0, r - RESTART_DEFAULT_FRAMES)
                team[max(min(d1, r), a) - a:r - a] = taker[first]   # the restart in flight
            if d1 > d0:
                dead[max(d0, a) - a:min(d1, b) - a] = True
        lab[a:b] = np.where(dead, D, np.where(team == H, H, A))
    return lab


def phase_labels(n: int, periods: list[tuple[int, int]], phases: list[tuple[int, int, int]]) -> np.ndarray:
    """"phase" labels on the source frames: phases are (first, stop, H or A); D elsewhere inside
    the periods, X outside."""
    lab = np.full(n, X, dtype=np.int8)
    for a, b in periods:
        lab[a:b] = D
    inside = lab == D
    for first, stop, team in phases:
        span = slice(max(first, 0), min(stop, n))
        lab[span] = np.where(inside[span], team, lab[span])
    return lab


def restart_is_own(flag: str) -> bool:
    """Whether the team of the first possession after a restart took that restart, from the
    provider's game_interruption_before value ("throw_in_for", "corner_against", ...): its own
    set piece ("_for") or the kick-off after a goal against it; an "_against" restart was taken
    by the opponent and intercepted. Unflagged possessions count as their own."""
    if flag.startswith("goal_"):
        return flag == "goal_against"
    return not flag.endswith("_against")


# -- file reading ------------------------------------------------------------------------------

def _read_tracking(path: Path, slot: dict, length: float, width: float):
    frames, period, players, ball = [], [], [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("period") is None:
                continue
            xy = np.full((len(slot), 2), np.nan, dtype=np.float32)
            for p in row.get("player_data") or []:
                s = slot.get(p.get("player_id"))
                if s is not None and p.get("x") is not None and p.get("y") is not None:
                    xy[s] = (p["x"] + length / 2.0, width / 2.0 - p["y"])
            b = row.get("ball_data") or {}
            seen = b.get("is_detected") and b.get("x") is not None and b.get("y") is not None
            frames.append(int(row["frame"]))
            period.append(int(row["period"]))
            players.append(xy)
            ball.append((b["x"] + length / 2.0, width / 2.0 - b["y"]) if seen else (np.nan, np.nan))
    return (np.asarray(frames, dtype=np.int64), np.asarray(period, dtype=np.int64),
            np.stack(players), np.asarray(ball, dtype=np.float32))


def _read_possessions(path: Path, home_id: int):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["event_type"] != "player_possession":
                continue
            rows.append((int(float(r["frame_start"])), int(float(r["frame_end"])),
                         H if int(float(r["team_id"])) == home_id else A,
                         r["game_interruption_before"], bool(r["game_interruption_after"]),
                         r["end_type"] in FOUL_ENDS))
    rows.sort(key=lambda t: (t[0], t[1]))
    cols = list(zip(*rows))
    teams = np.asarray(cols[2], dtype=np.int8)
    own = np.array([restart_is_own(flag) for flag in cols[3]], dtype=bool)
    return (np.asarray(cols[0], dtype=np.int64), np.asarray(cols[1], dtype=np.int64), teams,
            np.asarray([bool(flag) for flag in cols[3]], dtype=bool),
            np.asarray(cols[4], dtype=bool), np.asarray(cols[5], dtype=bool),
            np.where(own, teams, 1 - teams).astype(np.int8))


def _read_phases(path: Path, home_id: int):
    with open(path, newline="", encoding="utf-8") as handle:
        return [(int(float(r["frame_start"])), int(float(r["frame_end"])),
                 H if int(float(r["team_in_possession_id"])) == home_id else A)
                for r in csv.DictReader(handle) if r["team_in_possession_id"]]


def load_from_files(folder: Path) -> tuple[Match, dict]:
    """(Match at 25 fps with both halves on one timeline, {label kind: codes})."""
    meta = json.loads((folder / "match.json").read_text(encoding="utf-8"))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = int(meta["home_team"]["id"])
    slot = {p["id"]: i for i, p in enumerate(meta["players"])}
    team_of_slot = np.array([HOME if int(p["team_id"]) == home_id else AWAY for p in meta["players"]],
                            dtype=np.int8)
    frames, period, players, ball = _read_tracking(folder / "tracking.jsonl", slot, length, width)
    n10 = int(frames.max()) + 1
    periods = [(int(frames[period == p].min()), int(frames[period == p].max()) + 1)
               for p in sorted(set(period.tolist()))]
    ball10 = np.full((n10, 2), np.nan, dtype=np.float32)
    ball10[frames] = ball
    source = {"event": event_labels(n10, periods, *_read_possessions(folder / "dynamic_events.csv", home_id),
                                    ball10.astype(np.float64), (length, width)),
              "phase": phase_labels(n10, periods, _read_phases(folder / "phases.csv", home_id))}
    scale = np.array([length, width], dtype=np.float32)
    parts, halves, offset = [], [], 0
    for half, (a, b) in enumerate(periods, start=1):
        rows = np.flatnonzero((frames >= a) & (frames < b))
        ticks = np.arange(a * TICKS_IN, (b - 1) * TICKS_IN + 1, TICKS_OUT, dtype=np.int64)
        xy = resample_positions(frames[rows], players[rows], ticks)
        bxy = resample_positions(frames[rows], ball[rows], ticks)
        nearest = np.minimum((ticks + TICKS_IN // 2) // TICKS_IN, n10 - 1)
        parts.append((xy / scale, central_speed(xy), bxy / scale, central_speed(bxy),
                      {k: v[nearest] for k, v in source.items()}))
        halves.append((half, offset, offset + len(ticks) - 1))
        offset += len(ticks) + HALF_GAP_FRAMES
    n = offset - HALF_GAP_FRAMES
    match = Match(players_xy=np.full((n, len(slot), 2), np.nan, dtype=np.float32),
                  players_speed=np.full((n, len(slot)), np.nan, dtype=np.float32),
                  ball_xy=np.full((n, 2), np.nan, dtype=np.float32),
                  ball_speed=np.full(n, np.nan, dtype=np.float32),
                  team_of_slot=team_of_slot, halves=halves, pitch_m=(length, width),
                  source="skillcorner")
    codes = {k: np.full(n, X, dtype=np.int8) for k in LABEL_KINDS}
    for (_, first, last), (xy, speed, bxy, bspeed, labels) in zip(halves, parts):
        match.players_xy[first:last + 1] = xy
        match.players_speed[first:last + 1] = speed
        match.ball_xy[first:last + 1] = bxy
        match.ball_speed[first:last + 1] = bspeed
        for k in LABEL_KINDS:
            codes[k][first:last + 1] = labels[k]
    return match, codes


def load_match(match_id: str, labels: str = "event", refresh: bool = False):
    """ExternalMatch (possession.external) of one SkillCorner match; parses the files (~20 s)
    only on a cache miss."""
    from possession.external import ExternalMatch
    if labels not in LABEL_KINDS:
        raise ValueError("labels must be one of {}".format(LABEL_KINDS))
    cache = cache_dir() / f"{match_id}_v{CACHE_VERSION}.npz"
    if cache.exists() and not refresh:
        z = np.load(cache)
        match = Match(players_xy=z["players_xy"], players_speed=z["players_speed"],
                      ball_xy=z["ball_xy"], ball_speed=z["ball_speed"],
                      team_of_slot=z["team_of_slot"], halves=[tuple(int(v) for v in h) for h in z["halves"]],
                      pitch_m=tuple(float(v) for v in z["pitch_m"]), source="skillcorner")
        codes = {k: z["codes_" + k] for k in LABEL_KINDS}
    else:
        match, codes = load_from_files(skillcorner_dir() / match_id)
        np.savez_compressed(cache, players_xy=match.players_xy, players_speed=match.players_speed,
                            ball_xy=match.ball_xy, ball_speed=match.ball_speed,
                            team_of_slot=match.team_of_slot, halves=np.asarray(match.halves),
                            pitch_m=np.asarray(match.pitch_m),
                            **{"codes_" + k: v for k, v in codes.items()})
    return ExternalMatch(match, codes[labels], FPS_OUT, np.empty((0, 3), dtype=np.int64))
