"""External validation data: Metrica's public sample games (github.com/metrica-sports/sample-data).

Not the challenge match. Used to compare versions of the pipeline on independent, labelled
football, as RULES.md allows for other games.

* `load_sample_game` reads tracking + events and derives frame labels H/A/D/X from the events.
* `degrade` makes the clean sample tracking look like the challenge ATD: the ball goes missing in
  long bursts (much more often during stoppages), players drop out, positions get noise.

Event-derived labels follow CHALLENGE.md as closely as events allow: possession changes when the
other team starts an on-ball action (recovery, pass, set piece, shot, carry); loose balls stay with
the last team; the ball is dead from the moment it goes out / a foul or stoppage is called until the
next restart. Their conventions are not the challenge ground truth's, so absolute scores do not
transfer; differences between versions are what we use.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import H, A, D, X
from .tracking import HOME, AWAY, Tracking, read_epts, read_metrica_csv

ON_BALL = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "CARRY", "BALL LOST", "BALL OUT"}
STOPPING_LOSS = {"FORCED", "OFFSIDE", "HAND BALL", "END HALF"}


def _events_csv(path: str) -> pd.DataFrame:
    e = pd.read_csv(path)
    out = pd.DataFrame({
        "team": np.where(e["Team"].str.lower() == "home", HOME, AWAY),
        "type": e["Type"].str.upper().str.strip(),
        "subtypes": e["Subtype"].fillna("").str.upper().apply(
            lambda s: frozenset(p.strip() for p in s.split("-") if p.strip())),
        "start": e["Start Frame"].astype(np.int64) - 1,
        "end": e["End Frame"].astype(np.int64) - 1,
        "period": e["Period"].astype(np.int64),
    })
    return out.reset_index(drop=True)


def _events_json(path: str, home_id: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)["data"]
    rows = []
    for ev in data:
        sub = ev.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        names = frozenset(s["name"].upper() for s in (sub or []) if s and s.get("name"))
        start = ev["start"]["frame"]
        end = ev["end"]["frame"] if ev.get("end") and ev["end"].get("frame") is not None else start
        rows.append({"team": HOME if ev["team"]["id"] == home_id else AWAY,
                     "type": ev["type"]["name"].upper().strip(), "subtypes": names,
                     "start": int(start) - 1, "end": int(end) - 1, "period": int(ev["period"])})
    return pd.DataFrame(rows)


def _dead_start(ev) -> int | None:
    """Frame where this event makes the ball dead, or None."""
    t, sub = ev.type, ev.subtypes
    if t == "BALL OUT":
        return max(ev.start, ev.end)
    if t == "SHOT" and ("OUT" in sub or "GOAL" in sub):
        return max(ev.start, ev.end)
    if t == "FAULT RECEIVED":
        return ev.start
    if t == "CHALLENGE" and "FAULT" in sub and "ADVANTAGE" not in sub:
        return ev.start
    if t == "BALL LOST" and sub & STOPPING_LOSS:
        return max(ev.start, ev.end) if "OFFSIDE" in sub else ev.start
    return None


def labels_from_events(events: pd.DataFrame, n: int, periods: np.ndarray) -> tuple:
    """(labels, blocks). Blocks are the periods; frames before each kick-off are D."""
    labels = np.full(n, X, dtype=np.int8)
    blocks = []
    for p in sorted(set(periods[periods > 0].tolist())):
        where = np.flatnonzero(periods == p)
        a, b = int(where.min()), int(where.max()) + 1
        ev = events[(events.period == p)].sort_values(["start"], kind="stable")
        if not len(ev):
            continue
        blocks.append((a, b))
        team = np.full(b - a, -1, dtype=np.int8)
        dead = np.zeros(b - a, dtype=bool)
        on_ball = ev[ev.type.isin(ON_BALL)]
        # possession: team of the most recent on-ball event (changes only when the other team acts)
        starts = np.clip(on_ball.start.to_numpy() - a, 0, b - a - 1)
        order = np.argsort(starts, kind="stable")
        marks = np.full(b - a, -1, dtype=np.int64)
        marks[starts[order]] = on_ball.team.to_numpy()[order]
        # forward fill
        last = -1
        for i in range(b - a):
            if marks[i] >= 0:
                last = marks[i]
            team[i] = last
        # dead intervals: before the kick-off, and from each stopping event to the next restart
        kick = ev[(ev.type == "SET PIECE") & ev.subtypes.apply(lambda s: "KICK OFF" in s)]
        first_kick = int(kick.start.min()) - a if len(kick) else 0
        dead[:max(0, first_kick)] = True
        restarts = np.sort(on_ball.start.to_numpy())
        for e in ev.itertuples(index=False):
            s = _dead_start(e)
            if s is None:
                continue
            nxt = restarts[restarts > s]
            stop = int(nxt[0]) if len(nxt) else b
            dead[max(0, s - a):max(0, min(b, stop) - a)] = True
        lab = np.where(dead, D, np.where(team == HOME, H, np.where(team == AWAY, A, D)))
        labels[a:b] = lab
    return labels, blocks


def load_sample_game(game: int, root: str) -> tuple:
    """(Tracking, gt labels) for Metrica sample game 1, 2 or 3."""
    folder = os.path.join(root, "Sample_Game_%d" % game)
    if game in (1, 2):
        trk, periods = read_metrica_csv(
            os.path.join(folder, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % game),
            os.path.join(folder, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % game))
        events = _events_csv(os.path.join(folder, "Sample_Game_%d_RawEventsData.csv" % game))
    else:
        meta = os.path.join(folder, "Sample_Game_3_metadata.xml")
        trk = read_epts(meta, os.path.join(folder, "Sample_Game_3_tracking.txt"))
        from .tracking import parse_epts_metadata
        info = parse_epts_metadata(meta)
        events = _events_json(os.path.join(folder, "Sample_Game_3_events.json"), info["home_id"])
        periods = np.zeros(trk.n, dtype=np.int64)
        # period boundaries from the provider parameters in the metadata
        import xml.etree.ElementTree as ET
        params = {}
        for node in ET.parse(meta).getroot().iter():
            if node.tag.endswith("ProviderParameter"):
                name = node.find("Name").text
                value = node.find("Value").text
                if value and value.strip().lstrip("-").isdigit():
                    params[name] = int(value)
        periods[params["first_half_start"] - 1:params["first_half_end"]] = 1
        periods[params["second_half_start"] - 1:params["second_half_end"]] = 2
    labels, blocks = labels_from_events(events, trk.n, periods)
    trk.blocks = blocks
    return trk, labels


# -- degradation to ATD-like noise ---------------------------------------------------------------

DEFAULT_NOISE = {
    # Ball visibility, calibrated to public aggregate facts about the challenge ATD: ~41% of graded
    # frames without ball, gaps of median ~10 s, most missing frames inside stoppages.
    "p_miss_dead": 0.70,        # target share of dead-ball frames without ball
    "p_miss_alive": 0.22,       # target share of live frames without ball
    "gap_median_s": 8.0,        # median gap length during live play
    "gap_sigma": 0.9,           # lognormal spread of gap lengths
    "ball_noise_m": 0.3,        # gaussian noise on the visible ball
    "player_noise_m": 0.15,
    "p_player_drop": 0.07,      # share of player observations removed (~20.5 of 22 tracked)
    "player_drop_len_s": 3.0,   # mean length of a player dropout
    # optional "player_noise_corr_s": 0 (default) = independent jitter per frame; > 0 = the error
    # drifts slowly over that many seconds (left out here so v3's recorded config is unchanged)
}


def random_noise(seed: int) -> dict:
    """A random degradation (domain randomisation): training on many plausible trackers keeps the
    models from depending on one guess of how the challenge ATD is noisy."""
    rng = np.random.RandomState(10000 + seed)
    return {
        "p_miss_dead": float(rng.uniform(0.45, 0.85)),
        "p_miss_alive": float(rng.uniform(0.15, 0.40)),
        "gap_median_s": float(rng.uniform(5.0, 15.0)),
        "ball_noise_m": float(rng.uniform(0.1, 0.6)),
        "player_noise_m": float(rng.uniform(0.0, 0.3)),
        "player_noise_corr_s": float(rng.choice([0.0, 0.5, 2.0])),
        "p_player_drop": float(rng.uniform(0.02, 0.12)),
    }


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def degrade(trk: Tracking, gt: np.ndarray, noise: dict | None = None, seed: int = 0) -> Tracking:
    """Return a noisy copy of `trk`. Uses the labels only to decide *where* the ball is or is not
    seen (stoppages vs play), mimicking the tracker; the pipeline never sees the labels.

    The clean sample data has no ball during stoppages, while the challenge ATD sees it in roughly
    a third of dead frames (being placed for the restart, or rolling off the pitch). So inside each
    dead stretch we add a stationary ball at the restart spot before the restart, and a short trail
    continuing the last motion where the stretch starts. During play the ball is hidden in bursts,
    preferably where players crowd around it.
    """
    cfg = dict(DEFAULT_NOISE)
    cfg.update(noise or {})
    rng = np.random.RandomState(seed)
    n = trk.n
    fps = trk.fps
    clean = trk.ball.astype(np.float64)
    ball = clean.copy()
    players = trk.players.astype(np.float64).copy()
    seen_clean = ~np.isnan(clean[:, 0])

    # -- dead stretches: visible ball before the restart, trail after the ball goes dead --------
    dead_runs = _runs(gt == D)
    for a, b in dead_runs:
        length = b - a
        ball[a:b] = np.nan
        nxt = np.flatnonzero(seen_clean[b:min(n, b + 25)])
        if len(nxt):
            spot = clean[b + nxt[0]] + rng.normal(0, 0.3, 2)
            k = int(round(length * rng.uniform(0.25, 0.65) * (1 - cfg["p_miss_dead"]) / 0.45))
            k = min(length, max(0, k))
            if k:
                ball[b - k:b] = spot + rng.normal(0, 0.05, (k, 2))
        prev = np.flatnonzero(seen_clean[max(0, a - 10):a])
        if len(prev) >= 2:
            p1 = clean[max(0, a - 10) + prev[-1]]
            p0 = clean[max(0, a - 10) + prev[0]]
            vel = (p1 - p0) / max(1, prev[-1] - prev[0])
            k = min(length, int(rng.uniform(0.2, 0.8) * fps))
            ball[a:a + k] = p1 + np.outer(np.arange(1, k + 1), vel * 0.9)

    # -- live play: hide in bursts, weighted towards crowded situations -------------------------
    alive = np.isin(gt, (H, A))
    hide = np.zeros(n, dtype=bool)
    d = np.linalg.norm(players - clean[:, None, :], axis=2)
    crowd = np.sum(d < 5.0, axis=1).astype(np.float64)
    weight = np.where(alive & seen_clean, 1.0 + crowd, 0.0)
    if weight.sum() > 0:
        prob = weight / weight.sum()
        target = cfg["p_miss_alive"] * alive.sum()
        guard = 0
        while (hide & alive).sum() < target and guard < 100000:
            guard += 1
            start = rng.choice(n, p=prob)
            length = int(rng.lognormal(np.log(cfg["gap_median_s"] * fps), cfg["gap_sigma"]))
            hide[start:start + max(1, length)] = True
    hide &= alive
    ball[hide] = np.nan
    visible = ~np.isnan(ball[:, 0])
    ball[visible] += rng.normal(0, cfg["ball_noise_m"], (visible.sum(), 2))
    # occasional false detections: short jumps of 5-20 m
    for t in np.flatnonzero(visible & (rng.rand(n) < 0.002)):
        angle = rng.uniform(0, 2 * np.pi)
        jump = rng.uniform(5, 20) * np.array([np.cos(angle), np.sin(angle)])
        ball[t:t + rng.randint(1, 4)] += jump

    # -- players: noise, dropouts and identity swaps inside a team ------------------------------
    noise = rng.normal(0, cfg["player_noise_m"], players.shape)
    corr = int(round(cfg.get("player_noise_corr_s", 0.0) * fps))
    if corr > 1:
        # moving average along time, rescaled to keep the same standard deviation
        c = np.cumsum(np.concatenate([np.zeros((1,) + noise.shape[1:]), noise]), axis=0)
        smooth = (c[corr:] - c[:-corr]) / corr
        noise = np.concatenate([smooth, np.repeat(smooth[-1:], corr - 1, axis=0)]) * np.sqrt(corr)
    players += noise
    S = players.shape[1]
    for s in range(S):
        present = np.flatnonzero(~np.isnan(players[:, s, 0]))
        if not len(present):
            continue
        n_drops = int(cfg["p_player_drop"] * len(present) / (cfg["player_drop_len_s"] * fps))
        for _ in range(n_drops):
            start = present[rng.randint(0, len(present))]
            length = int(rng.exponential(cfg["player_drop_len_s"] * fps)) + 1
            players[start:start + length, s] = np.nan
    for a, b in trk.blocks:
        for side in (HOME, AWAY):
            slots = np.flatnonzero(trk.team == side)
            n_swaps = int((b - a) / (fps * 120.0))
            for _ in range(n_swaps):
                t = rng.randint(a, b)
                i, j = rng.choice(slots, 2, replace=False)
                tmp = players[t:b, i].copy()
                players[t:b, i] = players[t:b, j]
                players[t:b, j] = tmp
    return Tracking(ball=ball.astype(np.float32), players=players.astype(np.float32),
                    team=trk.team.copy(), fps=trk.fps, pitch=trk.pitch, blocks=list(trk.blocks),
                    extra={})
