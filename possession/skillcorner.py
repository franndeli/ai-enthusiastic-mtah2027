"""SkillCorner open data (github.com/SkillCorner/opendata, MIT licence): 20 A-League 2024/25
matches of broadcast tracking at 10 fps with per-frame possession and dynamic events.

Used only as external training/validation data (other games). Converted to the pipeline's frame
rate (25 fps, linear interpolation) and coordinate convention (metres, origin at a corner). The
ball counts as seen only where SkillCorner marks it detected. Labels:

* H/A: the team SkillCorner marks in possession, carried forward through passes and loose balls
  until the other team has the ball (CHALLENGE.md: a loose ball stays with the last team);
* D: from a possession that ends in a game interruption (ball out, foul, goal) until the restart,
  located on the tracked ball where possible, and before each period's first possession.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import H, A, D, X
from .tracking import HOME, AWAY, Tracking

FPS_IN, FPS_OUT = 10.0, 25.0


def _upsample(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Linear interpolation from 10 to 25 fps along axis 0; NaN wherever a neighbour is NaN."""
    f = np.arange(n_out) * (FPS_IN / FPS_OUT)
    lo = np.minimum(np.floor(f).astype(np.int64), len(arr) - 1)
    hi = np.minimum(lo + 1, len(arr) - 1)
    w = (f - lo).reshape((-1,) + (1,) * (arr.ndim - 1))
    return (1.0 - w) * arr[lo] + w * arr[hi]


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def load_match(match_id: str, meta_dir: str, tracking_path: str, cache_dir: str | None = None):
    """(Tracking at 25 fps, labels int8 H/A/D/X) for one SkillCorner match."""
    if cache_dir:
        cached = os.path.join(cache_dir, "sc_%s.npz" % match_id)
        if os.path.isfile(cached):
            z = np.load(cached)
            trk = Tracking(ball=z["ball"], players=z["players"], team=z["team"], fps=FPS_OUT,
                           pitch=tuple(z["pitch"]), blocks=[tuple(b) for b in z["blocks"]])
            return trk, z["labels"]
    with open(os.path.join(meta_dir, "%s_match.json" % match_id), encoding="utf-8") as handle:
        meta = json.load(handle)
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    ids = [p["id"] for p in meta["players"]]
    slot = {pid: i for i, pid in enumerate(ids)}
    team = np.array([HOME if p["team_id"] == home_id else AWAY for p in meta["players"]],
                    dtype=np.int8)

    with open(tracking_path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    n10 = max(r["frame"] for r in rows) + 1
    players = np.full((n10, len(ids), 2), np.nan)
    ball = np.full((n10, 2), np.nan)
    period = np.zeros(n10, dtype=np.int64)
    poss = np.full(n10, -1, dtype=np.int64)
    for r in rows:
        t = r["frame"]
        period[t] = r["period"] or 0
        b = r.get("ball_data") or {}
        if b.get("is_detected") and b.get("x") is not None:
            ball[t] = (b["x"], b["y"])
        for p in r.get("player_data") or []:
            s = slot.get(p["player_id"])
            if s is not None and p.get("x") is not None:
                players[t, s] = (p["x"], p["y"])
        g = (r.get("possession") or {}).get("group")
        poss[t] = H if g == "home team" else (A if g == "away team" else -1)
    # centre-origin metres -> corner-origin metres (y flipped so that it grows downwards)
    players[..., 0] += length / 2.0
    players[..., 1] = width / 2.0 - players[..., 1]
    ball[:, 0] += length / 2.0
    ball[:, 1] = width / 2.0 - ball[:, 1]

    labels10 = _labels(match_id, meta_dir, n10, period, poss, ball, length, width)
    n25 = int(np.floor((n10 - 1) * FPS_OUT / FPS_IN)) + 1
    idx = np.minimum(np.round(np.arange(n25) * FPS_IN / FPS_OUT).astype(np.int64), n10 - 1)
    labels = labels10[idx]
    per25 = period[idx]
    blocks = [(int(a), int(b)) for p in sorted(set(per25[per25 > 0].tolist()))
              for a, b in _runs(per25 == p)]
    trk = Tracking(ball=_upsample(ball, n25).astype(np.float32),
                   players=_upsample(players, n25).astype(np.float32), team=team, fps=FPS_OUT,
                   pitch=(length, width), blocks=blocks)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(os.path.join(cache_dir, "sc_%s.npz" % match_id), ball=trk.ball,
                 players=trk.players, team=trk.team, pitch=np.array(trk.pitch),
                 blocks=np.array(blocks, dtype=np.int64).reshape(-1, 2), labels=labels)
    return trk, labels


def _labels(match_id, meta_dir, n10, period, poss, ball, length, width) -> np.ndarray:
    lab = np.full(n10, X, dtype=np.int8)
    ev = pd.read_csv(os.path.join(meta_dir, "%s_dynamic_events.csv" % match_id), low_memory=False)
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start", kind="stable")
    out = (ball[:, 0] < -0.3) | (ball[:, 0] > length + 0.3) | (ball[:, 1] < -0.3) | \
          (ball[:, 1] > width + 0.3)
    seen = ~np.isnan(ball[:, 0])
    step = np.full(n10, np.nan)
    step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS_IN
    for p in sorted(set(period[period > 0].tolist())):
        idx = np.flatnonzero(period == p)
        a, b = int(idx.min()), int(idx.max()) + 1
        team = poss[a:b].copy()
        last = -1
        for i in range(len(team)):          # carry the last team through passes and loose balls
            if team[i] >= 0:
                last = team[i]
            team[i] = last
        dead = team < 0                     # before the period's first possession
        ev_p = pp[(pp["frame_start"] >= a) & (pp["frame_start"] < b)]
        starts = ev_p["frame_start"].to_numpy(dtype=np.int64)
        before = ev_p["game_interruption_before"].notna().to_numpy()
        for row in ev_p[ev_p["game_interruption_after"].notna()].itertuples(index=False):
            end = int(row.frame_end)
            # the ball goes out (or the whistle blows) after the possession ends
            look = np.flatnonzero(out[end:min(b, end + 40)] & seen[end:min(b, end + 40)])
            if len(look):
                d0 = end + int(look[0])
            elif str(row.end_type) in ("foul_suffered", "foul_committed"):
                d0 = end
            else:
                d0 = end + 10
            nxt = np.flatnonzero((starts > end) & before)
            if not len(nxt):
                continue
            r = int(starts[nxt[0]])
            # the restart kick: last moment the tracked ball was nearly still before the reception
            win = np.arange(max(d0, r - 50), r)
            still = win[(step[win] < 1.0) & seen[win]] if len(win) else win
            d1 = int(still[-1]) + 1 if len(still) else max(d0, r - 10)
            if d1 > d0:
                dead[d0 - a:d1 - a] = True
        lab[a:b] = np.where(dead, D, np.where(team == H, H, A))
    return lab
