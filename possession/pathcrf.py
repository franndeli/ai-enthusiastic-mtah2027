"""Ball-free team evidence from PathCRF (Kim et al., KDD 2026), vendored unmodified in
third_party/pathcrf (MPL-2.0; see its NOTICE.md).

PathCRF infers, from player trajectories alone, which player has the ball or is passing it, so it
does not care whether the ATD sees the ball. Its pretrained model (trained on five Sportec Open DFL
matches) is run inside the stretches v5 already calls alive, on the ATD's tracklets re-tracked into
11 continuous tracks per team (`retrack.py`); its probability that the last possessor is the home
team becomes a log-odds added to v5's team logit before the final decode (config "pathcrf").

Deterministic on CPU: fixed thread count, deterministic algorithms, probabilities rounded so that
last-bit differences between machines do not reach the decoder. Uses no labels; prints nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import H, A
from .retrack import retrack_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD = os.path.join(ROOT, "third_party", "pathcrf")
PITCH = (105.0, 68.0)          # PathCRF's pitch (third_party/pathcrf/datatools/config.py)


def _modules():
    if THIRD not in sys.path:
        sys.path.insert(0, THIRD)
    import torch
    from datatools import utils as pcrf_utils
    from inference import inference
    from models.utils import build_model
    return torch, pcrf_utils, inference, build_model


def load(trial: int = 140, threads: int = 4):
    """(model, args) of a vendored PathCRF checkpoint, on CPU, in eval mode."""
    torch, _, _, build_model = _modules()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    folder = os.path.join(THIRD, "saved", str(trial))
    with open(os.path.join(folder, "args.json"), encoding="utf-8") as handle:
        args = json.load(handle)
    model = build_model(args, device=torch.device("cpu"))
    state = torch.load(os.path.join(folder, "model", "state_dict_best.pt"), map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, args


def frame_table(players: np.ndarray, team: np.ndarray, ball: np.ndarray, pitch, blocks,
                alive: np.ndarray, fps: float) -> pd.DataFrame:
    """PathCRF's tracking table: home_<k>_x/_y on a 105 x 68 m pitch, ball_state, period, time."""
    n = players.shape[0]
    sx, sy = PITCH[0] / pitch[0], PITCH[1] / pitch[1]
    period = np.zeros(n, dtype=int)
    stamp = np.zeros(n)
    for k, (a, b) in enumerate(blocks):
        period[a:b] = k + 1
        stamp[a:b] = np.arange(b - a) / fps
    cols = {"frame_id": np.arange(n), "period_id": period, "timestamp": stamp,
            "ball_state": np.where(alive, "alive", "dead"),
            "ball_owning_team_id": np.full(n, None, dtype=object)}
    for s in range(players.shape[1]):
        side = "home" if team[s] == 0 else "away"
        cols["%s_%d_x" % (side, s)] = players[:, s, 0].astype(np.float64) * sx
        cols["%s_%d_y" % (side, s)] = players[:, s, 1].astype(np.float64) * sy
    cols["ball_x"] = ball[:, 0].astype(np.float64) * sx
    cols["ball_y"] = ball[:, 1].astype(np.float64) * sy
    table = pd.DataFrame(cols)
    return table[period > 0].reset_index(drop=True)


def home_probability(trk, alive: np.ndarray, model, args, digits: int = 3) -> np.ndarray:
    """(n,) P(the last possessor is home), rounded; NaN where PathCRF gives nothing."""
    torch, pcrf_utils, inference, _ = _modules()
    tracks, tteam = retrack_all(trk.players, trk.team, trk.blocks, trk.fps)
    table = frame_table(tracks, tteam, trk.ball, trk.pitch, trk.blocks, alive, trk.fps)
    sink = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        warnings.simplefilter("ignore")
        table, _ = pcrf_utils.label_frames_and_episodes(table, fps=trk.fps)
        table, _ = pcrf_utils.label_phases(table)
        table = pcrf_utils.calculate_running_features(table, fps=trk.fps)
        with torch.no_grad():
            prev, _, _, _ = inference(model, table, ball_feature=False, use_crf=True,
                                      evaluate=False, window_seconds=args["window_seconds"],
                                      fps=trk.fps, sample_freq=int(trk.fps / args["target_fps"]))
    out = np.full(trk.n, np.nan)
    if prev is None:
        return out
    home = [c for c in prev.columns if c.startswith("home_")]
    away = [c for c in prev.columns if c.startswith("away_")]
    h = prev[home].sum(axis=1, min_count=1).to_numpy(dtype=np.float64)
    a = prev[away].sum(axis=1, min_count=1).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[prev.index.to_numpy(dtype=int)] = h / (h + a)
    return np.round(out, digits)


def team_evidence(trk, labels: np.ndarray, cfg: dict) -> tuple:
    """(log-odds (n,), info): PathCRF evidence inside the alive stretches of `labels`, 0 elsewhere."""
    started = time.time()
    model, args = load(int(cfg.get("trial", 140)), int(cfg.get("threads", 4)))
    alive = np.isin(labels, (H, A))
    p = home_probability(trk, alive, model, args, int(cfg.get("digits", 3)))
    eps = 10.0 ** -int(cfg.get("digits", 3))
    clip = float(cfg.get("clip", 4.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        logodds = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))
    logodds = np.nan_to_num(np.clip(logodds, -clip, clip))
    covered = ~np.isnan(p)
    info = {"coverage_of_alive": round(float(covered[alive].mean()) if alive.any() else 0.0, 4),
            "seconds": round(time.time() - started, 1)}
    return logodds, info
