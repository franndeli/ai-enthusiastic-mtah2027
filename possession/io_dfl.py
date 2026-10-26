"""DFL/Sportec tracking (Bassek et al. 2025, CC BY) -> FrameTable with H/A/D labels from BallStatus/BallPossession.

Streaming stdlib iterparse: kloppy keeps a dict per object per frame, too heavy for 400 MB XML.
CLI: scripts/py -m possession.io_dfl converts every downloaded match and prints aggregates only.
"""

import logging
import os
import sys
import time
import xml.etree.ElementTree as ET
from array import array
from pathlib import Path

import numpy as np

from possession.frames import AWAY, HOME, A, D, FrameTable, H, X, period_runs

logger = logging.getLogger(__name__)

FPS = 25.0
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
BALL = "BALL"


def read_matchinfo(path: Path) -> tuple:
    """(match_id, pitch_m, {person_id: HOME/AWAY}) in lineup order, home team first."""
    root = ET.parse(path).getroot()
    info = root.find("MatchInformation")
    env = info.find("Environment")
    roles = {"home": HOME, "guest": AWAY}
    teams = sorted(info.find("Teams").findall("Team"), key=lambda t: roles[t.get("Role")])
    players = {p.get("PersonId"): roles[t.get("Role")] for t in teams for p in t.iter("Player")}
    return (info.find("General").get("MatchId"),
            (float(env.get("PitchX")), float(env.get("PitchY"))), players)


def parse_positions(path: Path) -> dict:
    """{(period, object_id): columns} with object_id BALL for the ball; columns are compact arrays."""
    tracks, current = {}, None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            if elem.tag == "FrameSet":
                period = SECTIONS.get(elem.get("GameSection"))
                obj = BALL if elem.get("TeamId") == BALL else elem.get("PersonId")
                # Extra time or unknown sections are not part of the frame table.
                current = None if period is None else tracks.setdefault(
                    (period, obj), {k: array("i" if k == "N" else "f") for k in
                                    ("N", "X", "Y", "BallStatus", "BallPossession")})
            continue
        if elem.tag == "Frame" and current is not None:
            a = elem.attrib
            current["N"].append(int(a["N"]))
            current["X"].append(float(a["X"]))
            current["Y"].append(float(a["Y"]))
            if "BallStatus" in a:
                current["BallStatus"].append(float(a["BallStatus"]))
                current["BallPossession"].append(float(a.get("BallPossession", "nan")))
            elem.clear()
        elif elem.tag == "FrameSet":
            elem.clear()  # drop the cleared Frame children too
    return tracks


def build_half(tracks: dict, period: int, slots: dict) -> tuple:
    """Arrays for one half on its own frame range [min N, max N] over every tracked object."""
    half = {obj: cols for (p, obj), cols in tracks.items() if p == period}
    if not half:
        raise ValueError(f"no frames for period {period}")
    first = min(min(c["N"]) for c in half.values())
    n = max(max(c["N"]) for c in half.values()) - first + 1
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    labels = np.full(n, D, np.int8)
    for obj, cols in half.items():
        idx = np.frombuffer(cols["N"], np.int32) - first
        xy = np.stack([np.frombuffer(cols["X"], np.float32), np.frombuffer(cols["Y"], np.float32)], 1)
        if obj == BALL:
            if len(cols["BallStatus"]) != len(idx):
                raise ValueError(f"period {period}: ball frames without BallStatus")
            ball[idx] = xy
            alive = np.frombuffer(cols["BallStatus"], np.float32) == 1
            owner = np.frombuffer(cols["BallPossession"], np.float32)
            # DFL BallPossession: 1 = home, 2 = guest. Alive frames with no owner stay D.
            labels[idx[alive & (owner == 1)]] = H
            labels[idx[alive & (owner == 2)]] = A
        elif obj in slots:
            players[idx, slots[obj]] = xy
    return np.full(n, period, np.int8), ball, players, labels


def load_dfl(matchinfo_path: Path, positions_path: Path) -> FrameTable:
    """Labelled FrameTable: first half, one separator row (period 0, label X), second half.

    Raw DFL coordinates are already metres centred on the pitch with x along the length; they are
    kept as is (no orientation by attacking direction). Frames without a ball record are labelled D.
    """
    _, pitch_m, players = read_matchinfo(matchinfo_path)
    tracks = parse_positions(positions_path)
    seen = {obj for _, obj in tracks}
    # One slot per player who was tracked; referees and the ball are not in the lineup map.
    slots = {pid: i for i, pid in enumerate(p for p in players if p in seen)}
    halves = [build_half(tracks, period, slots) for period in (1, 2)]
    sep = (np.zeros(1, np.int8), np.full((1, 2), np.nan, np.float32),
           np.full((1, len(slots), 2), np.nan, np.float32), np.full(1, X, np.int8))
    period, ball, xy, labels = (np.concatenate([halves[0][i], sep[i], halves[1][i]]) for i in range(4))
    return FrameTable(fps=FPS, pitch_m=pitch_m, period=period, ball_xy_m=ball, players_xy_m=xy,
                      slot_team=np.array([players[p] for p in slots], np.int8), labels=labels)


def count_spells(table: FrameTable) -> int:
    """Maximal runs of H or A within each half."""
    total = 0
    for start, stop in period_runs(table.period):
        lab = table.labels[start:stop]
        starts = np.concatenate(([True], lab[1:] != lab[:-1]))
        total += int(np.sum(starts & np.isin(lab, (H, A))))
    return total


def main() -> int:
    ext = Path(os.environ.get("MTAH_EXTERNAL_DIR", Path.home() / ".mtah2027" / "external")) / "dfl"
    out = Path(os.environ.get("MTAH_WORK_DIR", Path.home() / ".mtah2027" / "work")) / "dfl"
    infos = sorted(ext.glob("DFL_02_01_matchinformation_*.xml"))
    if not infos:
        print(f"no matchinformation files in {ext}; run scripts/fetch_external.py", file=sys.stderr)
        return 1
    print("match_id        n_frames  slots  ball_missing%   H%    A%    D%  spells  seconds")
    for info in infos:
        positions = ext / info.name.replace("02_01_matchinformation", "04_03_positions_raw_observed")
        tic = time.perf_counter()
        table = load_dfl(info, positions)
        match_id = read_matchinfo(info)[0]
        table.save(out / f"{match_id}.npz")
        play = table.period != 0
        missing = 100 * np.isnan(table.ball_xy_m[play, 0]).mean()
        share = [100 * np.mean(table.labels[play] == c) for c in (H, A, D)]
        print(f"{match_id:<15} {table.n_frames:>8} {len(table.slot_team):>6} {missing:>13.2f} "
              f"{share[0]:>5.1f} {share[1]:>5.1f} {share[2]:>5.1f} {count_spells(table):>7} "
              f"{time.perf_counter() - tic:>8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
