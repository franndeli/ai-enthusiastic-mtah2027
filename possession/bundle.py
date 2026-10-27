"""Dress-rehearsal bundles: a public, labelled match laid out exactly like the challenge's data.

For one external match (other teams, public data) this writes the files a pipeline reads
(DATA.md), under the names games.py expects, so any team pipeline runs on it unchanged:

* ``tactical/BAR-ATH_tactical_atd.txt``: FIFA/EPTS raw tracking, one record per video frame
  (frame_id 1..N), 46 slots of ``x,y,speed`` (0-1 coordinates, speed in m/s), ball ``x,y,speed``
  (speed NaN, as the tracker's own ball speed is mostly missing), ``NaN`` where unseen;
* ``tactical/BAR-ATH_tactical_atd.xml``: metadata, 23 slots per team, the home team first;
* ``tactical/BAR-ATH_tactical_smart_tagging.xml``: the two kick-offs only (no other tags exist
  for these matches);
* ``gt/<name>_possession_gt.csv``: the frame,label ground truth.

The match's two periods are placed on the challenge's graded halves (games.py): the first at the
start of half 1, the second at the second-half kick-off, so the 2,934 graded frames before it are
D, as in the challenge. Graded frames the external match does not reach are X in the ground truth
(never scored); a period longer than its half is cut at the end of the half.
"""

from __future__ import annotations

import os

import numpy as np

from . import D, LABELS, X

SLOTS_PER_TEAM = 23
SPEED_HALF_SPAN = 3          # frames either side of the centred difference (0.24 s at 25 fps)


def layout(game: str = "BAR-ATH") -> dict:
    """Frame count, half-open graded blocks and each half's first frame of data, from games.py."""
    import games
    g = games.GAMES[game]
    blocks = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    kick2 = games.anchor_video_frame(g["kickoff_2nd"]) + 1     # FAQ.md: 2,934 frames before it
    return {"n": int(g["n_video_frames"]), "fps": float(g["fps"]), "blocks": blocks,
            "starts": [blocks[0][0], kick2], "pitch": tuple(g["pitch"])}


def _slots(trk, team: int, rows) -> list:
    """Up to 23 source slots of one team, those seen on the most frames first, in source order."""
    idx = np.flatnonzero(trk.team == team)
    seen = np.array([(~np.isnan(trk.players[rows, s, 0])).sum() for s in idx])
    keep = idx[seen > 0]
    if len(keep) > SLOTS_PER_TEAM:
        order = np.argsort(-seen[seen > 0], kind="stable")[:SLOTS_PER_TEAM]
        keep = np.sort(keep[order])
    return [int(s) for s in keep]


def place(trk, labels: np.ndarray, lay: dict, seed: int = 0) -> dict:
    """Arrays of the bundle: players (n, 46, 2) and ball (n, 2) in metres (corner origin, y down,
    the external match's own pitch), slot_team (46,), ground truth (n,) int8."""
    n = lay["n"]
    rows = np.concatenate([np.arange(a, b) for a, b in trk.blocks[:2]])
    src = []
    for team in (0, 1):
        s = _slots(trk, team, rows)
        src += s + [-1] * (SLOTS_PER_TEAM - len(s))
    slot_team = np.array([0] * SLOTS_PER_TEAM + [1] * SLOTS_PER_TEAM, dtype=np.int8)
    players = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    gt = np.full(n, X, dtype=np.int8)
    used = [k for k, s in enumerate(src) if s >= 0]
    cols = [src[k] for k in used]
    rng = np.random.RandomState(seed)
    for (a, b), start, (sa, sb) in zip(lay["blocks"], lay["starts"], trk.blocks[:2]):
        length = min(b - start, sb - sa)
        dst = slice(start, start + length)
        players[dst, used] = trk.players[sa:sa + length][:, cols]
        ball[dst] = trk.ball[sa:sa + length]
        gt[dst] = labels[sa:sa + length]
        if start > a:
            # before the 2nd-half kick-off: graded, dead; players near their kick-off positions,
            # the ball on the centre spot for the last 20 s
            gt[a:start] = D
            players[a:start, used] = trk.players[sa][cols]
            players[a:start] += rng.normal(0.0, 0.05, players[a:start].shape).astype(np.float32)
            ball[max(a, start - 500):start] = (trk.pitch[0] / 2.0, trk.pitch[1] / 2.0)
    return {"players": players, "ball": ball, "slot_team": slot_team, "gt": gt,
            "pitch": tuple(trk.pitch)}


def speeds(players: np.ndarray, fps: float, blocks, half_span: int = SPEED_HALF_SPAN) -> np.ndarray:
    """(n, S) speed in m/s by a centred difference inside each block; NaN where an end is unseen."""
    out = np.full(players.shape[:2], np.nan, dtype=np.float32)
    h = half_span
    for a, b in blocks:
        if b - a > 2 * h:
            d = players[a + 2 * h:b] - players[a:b - 2 * h]
            out[a + h:b - h] = np.hypot(d[..., 0], d[..., 1]) * fps / (2 * h)
    return out


def write(out_dir: str, arrays: dict, lay: dict, name: str, game: str = "BAR-ATH") -> dict:
    """Write the bundle; returns the paths. Coordinates normalised by the match's own pitch."""
    import games
    stem = games.SOURCES["tactical"]["stem"].replace("BAR-ATH", game)
    tac = os.path.join(out_dir, "tactical")
    gt_dir = os.path.join(out_dir, "gt")
    os.makedirs(tac, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    paths = {"atd_raw": os.path.join(tac, stem + "_atd.txt"),
             "atd_meta": os.path.join(tac, stem + "_atd.xml"),
             "smart_tagging": os.path.join(tac, stem + "_smart_tagging.xml"),
             "gt": os.path.join(gt_dir, name + "_possession_gt.csv")}
    n, fps = lay["n"], lay["fps"]
    players, ball, slot_team = arrays["players"], arrays["ball"], arrays["slot_team"]
    S = players.shape[1]
    scale = np.array(arrays["pitch"], dtype=np.float64)
    speed = speeds(players, fps, lay["blocks"])
    speed[np.isnan(players[..., 0])] = np.nan
    vals = np.empty((n, S, 3), dtype=np.float64)
    vals[..., :2] = players / scale
    vals[..., 2] = speed
    bvals = np.full((n, 3), np.nan)
    bvals[:, :2] = ball / scale
    fmt = "%d:" + ";".join(["%.5f,%.5f,%.2f"] * S) + ":%.5f,%.5f,%.2f\n"
    flat = np.concatenate([vals.reshape(n, -1), bvals], axis=1)
    with open(paths["atd_raw"], "w", encoding="utf-8", newline="\n") as handle:
        for i in range(n):
            handle.write((fmt % ((i + 1,) + tuple(flat[i]))).replace("nan", "NaN"))
    with open(paths["atd_meta"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata><GlobalConfig>'
                     '<FrameRate>%d</FrameRate></GlobalConfig><Teams>'
                     '<Team id="T1"><Name>Home FC</Name></Team>'
                     '<Team id="T2"><Name>Away FC</Name></Team></Teams><Players>' % int(fps))
        for k in range(S):
            handle.write('<Player id="P%d" teamId="%s"><Name>Track %d</Name>'
                         '<ShirtNumber>%d</ShirtNumber></Player>'
                         % (k + 1, "T1" if slot_team[k] == 0 else "T2", k + 1, k + 1))
        handle.write("</Players></Metadata></main>\n")
    with open(paths["smart_tagging"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write("<file><ALL_INSTANCES>")
        for k, kick in enumerate(lay["starts"]):
            handle.write("<instance><ID>%d</ID><start>%.2f</start><end>%.2f</end>"
                         "<code>SET PIECES</code><label><group>Type</group>"
                         "<text>Kick Off (Start)</text></label><label><group>Half</group>"
                         "<text>%s Half</text></label></instance>"
                         % (k + 1, (kick - 90) / fps, (kick + 35) / fps, ("1st", "2nd")[k]))
        handle.write("</ALL_INSTANCES></file>\n")
    with open(paths["gt"], "w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (i, LABELS[c]) for i, c in enumerate(arrays["gt"]))
    return paths
