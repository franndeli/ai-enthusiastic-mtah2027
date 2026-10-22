"""IDSSE / Sportec Open DFL matches as possession.io.Match + H/A/D/X labels, and noise injection.

Public data (Bassek et al. 2025, CC-BY 4.0), downloaded by scripts/fetch_idsse.py. It gives the
same kind of labels as our ground truth (BallStatus, BallPossession), so it validates the pipeline
on held-out matches. inject_noise degrades a clean match towards the missingness and identity
loss of our own tracking (ball dropout, jitter, fragmented and duplicated tracks).

Coordinates: load_tracking(coordinates=None) returns kloppy's own system (kloppy 3.18, checked in
test_external against a synthetic file): normalised 0..1, origin top-left, y top-to-bottom, which is
already our Match convention; metres = norm * pitch_m, with the pitch from the match metadata (the
raw Sportec system is metres from the centre, y bottom-to-top). Teams swap ends at half-time, so absolute x is NOT direction-agnostic; the current features only
use distances between ball and players and player speeds, which are.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession import D, H, A, X
from possession.io import AWAY, HOME, Match, work_dir

IDSSE_MATCHES = ("J03WPY", "J03WN1", "J03WMX", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
Z_P90 = 1.2815515655446004  # standard-normal 90th percentile


@dataclass
class ExternalMatch:
    """One labelled match: tracking, per-frame codes (H/A/D, X outside the halves) and fps."""
    match: Match
    codes: np.ndarray
    fps: int
    raw_ball_counts: np.ndarray  # rows (BallStatus, BallPossession, count); -1 = attribute absent


def idsse_dir() -> Path:
    return work_dir() / "external" / "idsse"


def cache_dir() -> Path:
    path = work_dir() / "external" / "idsse_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- pure conversion ---------------------------------------------------------------------------

def speed_m_s(xy_m: np.ndarray, fps: int) -> np.ndarray:
    """Central-difference speed over axis 0 of (n, ..., 2) metres; NaN at the ends and gaps."""
    speed = np.full(xy_m.shape[:-1], np.nan, dtype=np.float32)
    delta = xy_m[2:] - xy_m[:-2]
    speed[1:-1] = np.sqrt(np.sum(delta * delta, axis=-1)) * (fps / 2.0)
    speed[np.isnan(xy_m[..., 0])] = np.nan  # no position, no speed
    return speed


def label_codes(alive: np.ndarray, home_owns: np.ndarray) -> np.ndarray:
    """Dead ball -> D; alive -> H if the home side owns the ball, else A."""
    return np.where(alive, np.where(home_owns, H, A), D).astype(np.int8)


def assemble(frame_id: np.ndarray, period: np.ndarray, alive: np.ndarray, home_owns: np.ndarray,
             ball_xy: np.ndarray, players_xy: np.ndarray, team_of_slot: np.ndarray,
             pitch_m: tuple, fps: int, raw_ball_counts: np.ndarray) -> ExternalMatch:
    """Place the observed frames (xy normalised 0..1) on the match's own timeline (frame_id - first frame_id).

    Frames absent from the source (the gap between halves) are NaN positions and label X. Halves
    are the first/last observed index of each period.
    """
    if len(frame_id) == 0:
        raise ValueError("no frames")
    idx = np.asarray(frame_id, dtype=np.int64) - int(frame_id.min())
    n = int(idx.max()) + 1
    scale = np.asarray(pitch_m, dtype=np.float32)

    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[idx] = ball_xy
    players = np.full((n, players_xy.shape[1], 2), np.nan, dtype=np.float32)
    players[idx] = players_xy
    codes = np.full(n, X, dtype=np.int8)
    codes[idx] = label_codes(alive, home_owns)

    halves = [(int(p), int(idx[period == p].min()), int(idx[period == p].max()))
              for p in np.unique(period)]
    match = Match(
        players_xy=players, players_speed=speed_m_s(players * scale, fps),
        ball_xy=ball, ball_speed=speed_m_s(ball * scale, fps),
        team_of_slot=np.asarray(team_of_slot, dtype=np.int8),
        halves=halves, pitch_m=tuple(float(v) for v in pitch_m), source="idsse")
    return ExternalMatch(match, codes, fps, raw_ball_counts)


# -- loading (kloppy) --------------------------------------------------------------------------

def raw_ball_value_counts(tracking_path: Path) -> np.ndarray:
    """Counts of (BallStatus, BallPossession) over the ball frames; kloppy hides the raw values.

    kloppy maps BallPossession == 1 to home and everything else (2, 0, missing) to away, so the
    raw distribution is the only way to know whether the label is clean.
    """
    from lxml import etree  # kloppy dependency
    counts: dict[tuple[int, int], int] = {}
    in_ball = False
    for event, elem in etree.iterparse(str(tracking_path), events=("start", "end"),
                                       huge_tree=True):
        if event == "start" and elem.tag == "FrameSet":
            in_ball = elem.get("TeamId") == "BALL"
        elif event == "start" and elem.tag == "Frame" and in_ball:
            key = (int(elem.get("BallStatus", -1)), int(elem.get("BallPossession", -1)))
            counts[key] = counts.get(key, 0) + 1
        elif event == "end" and elem.tag in ("Frame", "FrameSet"):
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    return np.array([(s, p, c) for (s, p), c in sorted(counts.items())], dtype=np.int64)


def load_from_files(meta_path: Path, tracking_path: Path) -> ExternalMatch:
    """Parse one match with kloppy (only_alive=False keeps the dead frames) and assemble it.

    One Python loop over frames while unpacking kloppy objects; it runs once per match and the
    result is cached by idsse_prepare, so it is not a hot path.
    """
    from kloppy import sportec
    from kloppy.domain import BallState

    dataset = sportec.load_tracking(meta_data=str(meta_path), raw_data=str(tracking_path),
                                    only_alive=False)
    meta = dataset.metadata
    home_team, away_team = meta.teams
    players = list(home_team.players) + list(away_team.players)
    slot_of = {p.player_id: i for i, p in enumerate(players)}
    team_of_slot = np.array([HOME] * len(home_team.players) + [AWAY] * len(away_team.players))

    frames = dataset.records
    m = len(frames)
    frame_id = np.empty(m, dtype=np.int64)
    period = np.empty(m, dtype=np.int8)
    alive = np.empty(m, dtype=bool)
    home_owns = np.empty(m, dtype=bool)
    ball_xy = np.empty((m, 2), dtype=np.float32)
    players_xy = np.full((m, len(players), 2), np.nan, dtype=np.float32)
    for i, frame in enumerate(frames):
        frame_id[i] = frame.frame_id
        period[i] = frame.period.id
        alive[i] = frame.ball_state == BallState.ALIVE
        home_owns[i] = frame.ball_owning_team is home_team
        ball_xy[i] = (frame.ball_coordinates.x, frame.ball_coordinates.y)
        for player, data in frame.players_data.items():
            players_xy[i, slot_of[player.player_id]] = (data.coordinates.x, data.coordinates.y)

    pitch_m = (meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width)
    return assemble(frame_id, period, alive, home_owns, ball_xy, players_xy, team_of_slot,
                    pitch_m, int(meta.frame_rate), raw_ball_value_counts(tracking_path))


def save_cache(path: Path, ext: ExternalMatch) -> None:
    m = ext.match
    np.savez_compressed(
        path, players_xy=m.players_xy, players_speed=m.players_speed, ball_xy=m.ball_xy,
        ball_speed=m.ball_speed, team_of_slot=m.team_of_slot, halves=np.array(m.halves),
        pitch_m=np.array(m.pitch_m), codes=ext.codes, fps=ext.fps,
        raw_ball_counts=ext.raw_ball_counts)


def load_cache(path: Path) -> ExternalMatch:
    with np.load(path) as z:
        match = Match(
            players_xy=z["players_xy"], players_speed=z["players_speed"], ball_xy=z["ball_xy"],
            ball_speed=z["ball_speed"], team_of_slot=z["team_of_slot"],
            halves=[tuple(int(v) for v in row) for row in z["halves"]],
            pitch_m=tuple(float(v) for v in z["pitch_m"]), source="idsse")
        return ExternalMatch(match, z["codes"], int(z["fps"]), z["raw_ball_counts"])


def available_matches() -> list[str]:
    """Match ids whose meta and tracking files are both on disk."""
    base = idsse_dir()
    return [mid for mid in IDSSE_MATCHES
            if (base / f"{mid}_meta.xml").exists() and (base / f"{mid}_tracking.xml").exists()]


def load_external(match_id: str, refresh: bool = False) -> ExternalMatch:
    """Cached match; parses the XML (minutes) only on a cache miss."""
    cache = cache_dir() / f"{match_id}.npz"
    if cache.exists() and not refresh:
        return load_cache(cache)
    base = idsse_dir()
    ext = load_from_files(base / f"{match_id}_meta.xml", base / f"{match_id}_tracking.xml")
    save_cache(cache, ext)
    return ext


# -- noise injection ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseParams:
    """Provisional targets for our own tracking; every value is a knob, none is established."""
    ball_missing_share: float = 0.41
    run_median_s: float = 10.7
    run_p90_s: float = 34.0
    ball_jitter_m: float = 0.3
    n_slots_per_team: int = 23
    segment_mean_s: float = 60.0
    duplicate_prob: float = 0.03
    duplicate_offset_m: float = 0.5


def dropout_mask(n: int, share: float, median_frames: float, p90_frames: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """True where the ball is dropped: lognormal outage runs (given median and p90) separated by
    exponential present-runs whose mean makes the expected missing share equal `share`."""
    sigma = np.log(p90_frames / median_frames) / Z_P90
    drop_mean = median_frames * np.exp(sigma ** 2 / 2.0)
    gap_mean = drop_mean * (1.0 - share) / share
    edges = np.empty(0)
    while len(edges) == 0 or edges[-1] < n:
        k = int(2 * n / (drop_mean + gap_mean)) + 8
        gaps = rng.exponential(gap_mean, k)
        drops = rng.lognormal(np.log(median_frames), sigma, k)
        lengths = np.maximum(1.0, np.round(np.stack([gaps, drops], axis=1).ravel()))
        edges = np.cumsum(lengths)
    return np.searchsorted(edges, np.arange(n), side="right") % 2 == 1


def _fragment(match: Match, params: NoiseParams, fps: int,
              rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut every track into segments, re-assign them to random free slots of the same team, and
    copy some segments to a second slot as a slightly offset duplicate."""
    n = match.n_frames
    per_team = params.n_slots_per_team
    xy = np.full((n, 2 * per_team, 2), np.nan, dtype=np.float32)
    speed = np.full((n, 2 * per_team), np.nan, dtype=np.float32)
    mean_frames = params.segment_mean_s * fps
    offset_norm = params.duplicate_offset_m / np.asarray(match.pitch_m, dtype=np.float32)
    for team in (HOME, AWAY):
        segments = []  # (start, stop, track, is_duplicate)
        for track in np.flatnonzero(match.team_of_slot == team):
            observed = ~np.isnan(match.players_xy[:, track, 0])
            if not observed.any():
                continue
            first, last = int(np.argmax(observed)), n - 1 - int(np.argmax(observed[::-1]))
            cuts = first + np.cumsum(rng.exponential(mean_frames, int(3 * (last - first) / mean_frames) + 3))
            bounds = np.concatenate(([first], cuts[cuts < last].astype(int), [last + 1]))
            for a, b in zip(bounds[:-1], bounds[1:]):
                if observed[a:b].any():
                    segments.append((int(a), int(b), int(track), False))
                    if rng.rand() < params.duplicate_prob:
                        segments.append((int(a), int(b), int(track), True))
        segments.sort()
        free_at = np.zeros(per_team, dtype=np.int64)
        for a, b, track, duplicate in segments:
            free = np.flatnonzero(free_at <= a)
            if len(free) == 0:
                raise ValueError("more than {} concurrent tracks for team {} at frame {}".format(
                    per_team, team, a))
            slot = int(free[rng.randint(len(free))])
            free_at[slot] = b
            col = team * per_team + slot
            shift = rng.normal(0.0, 1.0, 2).astype(np.float32) * offset_norm if duplicate else 0.0
            xy[a:b, col] = match.players_xy[a:b, track] + shift
            speed[a:b, col] = match.players_speed[a:b, track]
    team_of_slot = np.repeat(np.array([HOME, AWAY], dtype=np.int8), per_team)
    return xy, speed, team_of_slot


def inject_noise(match: Match, seed: int, params: NoiseParams = NoiseParams(),
                 fps: int = 25) -> Match:
    """Noisy copy: ball jitter (speed recomputed), ball dropout inside the halves, slot
    fragmentation. Deterministic for a given seed."""
    rng = np.random.RandomState(seed)
    scale = np.asarray(match.pitch_m, dtype=np.float32)
    ball = match.ball_xy + rng.normal(0.0, params.ball_jitter_m, match.ball_xy.shape).astype(
        np.float32) / scale
    ball_speed = speed_m_s(ball * scale, fps)

    graded = np.flatnonzero(match.graded_mask())
    dropped = graded[dropout_mask(len(graded), params.ball_missing_share,
                                  params.run_median_s * fps, params.run_p90_s * fps, rng)]
    ball[dropped] = np.nan
    ball_speed[dropped] = np.nan

    xy, speed, team_of_slot = _fragment(match, params, fps, rng)
    return Match(players_xy=xy, players_speed=speed, ball_xy=ball, ball_speed=ball_speed,
                 team_of_slot=team_of_slot, halves=list(match.halves), pitch_m=match.pitch_m,
                 source=match.source)


def missing_runs_s(missing: np.ndarray, halves: list, fps: int) -> np.ndarray:
    """Lengths in seconds of maximal True runs of `missing` inside each half."""
    out = []
    for _, first, last in halves:
        window = missing[first:last + 1].astype(np.int8)
        edges = np.diff(np.concatenate(([0], window, [0])))
        out.append(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1))
    return np.concatenate(out) / fps


def noise_report(clean: Match, noisy: Match, fps: int = 25) -> dict:
    """Aggregates showing the noise targets are met (graded frames only)."""
    graded = clean.graded_mask()
    missing = np.isnan(noisy.ball_xy[:, 0])
    runs = missing_runs_s(missing, noisy.halves, fps)

    def tracked(m: Match, team: int) -> float:
        cols = m.team_of_slot == team
        return float(np.mean(np.sum(~np.isnan(m.players_xy[graded][:, cols, 0]), axis=1)))

    return {
        "ball_missing_share": float(missing[graded].mean()),
        "ball_missing_clean_share": float(np.isnan(clean.ball_xy[graded, 0]).mean()),
        "missing_run_median_s": float(np.median(runs)) if len(runs) else 0.0,
        "missing_run_p90_s": float(np.percentile(runs, 90)) if len(runs) else 0.0,
        "n_missing_runs": int(len(runs)),
        "mean_tracked_home": (tracked(clean, HOME), tracked(noisy, HOME)),
        "mean_tracked_away": (tracked(clean, AWAY), tracked(noisy, AWAY)),
        "n_slots": (clean.n_slots, noisy.n_slots),
        "slots_ever_used": int(np.sum(~np.all(np.isnan(noisy.players_xy[..., 0]), axis=0))),
    }
