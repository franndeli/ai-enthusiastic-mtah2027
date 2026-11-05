"""Ball-free team evidence from PathCRF (Kim et al., KDD 2026), vendored unmodified in
third_party/pathcrf (MPL-2.0, see its NOTICE.md; integration first built by a teammate on branch
feat/possession-pipeline-v5, E10).

PathCRF infers from the 22 players' movement alone which player has or is passing the ball. The
tracklet slots are re-tracked into 11 continuous tracks per team (`retrack.py`), the pretrained
Dynamic Sparse CRF (trial 140, trained on five Sportec Open DFL matches) runs inside the stretches
the rule pipeline decodes as alive, and P(the last possessor is home) per frame becomes team
log-odds added to the rule team logit (config v6.w_pathcrf, possession/stack.py).

Deterministic on CPU: fixed thread count, deterministic algorithms, probabilities rounded to
DIGITS decimals so last-bit differences between machines rarely reach the decoder. Needs torch and
torch_geometric (imported lazily). Uses no labels; prints nothing.
"""
from __future__ import annotations

import contextlib
import json
from functools import lru_cache
import sys
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from possession import FPS
from possession.io import Match
from possession.retrack import retrack_all

THIRD = Path(__file__).resolve().parent.parent / "third_party" / "pathcrf"
PITCH = (105.0, 68.0)    # PathCRF's pitch (third_party/pathcrf/datatools/config.py)
DIGITS = 3
THREADS = 4
TRIAL = 140
CLIP = 4.0


def _modules():
    if str(THIRD) not in sys.path:
        sys.path.insert(0, str(THIRD))
    import torch
    from datatools import utils as pcrf_utils
    from inference import inference
    from models.utils import build_model
    return torch, pcrf_utils, inference, build_model


@lru_cache(maxsize=1)
def load_model():
    """(model, args) of the vendored checkpoint; reads only third_party/pathcrf (repository files)."""
    torch, _, _, build_model = _modules()
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True, warn_only=True)
    folder = THIRD / "saved" / str(TRIAL)
    args = json.loads((folder / "args.json").read_text())
    model = build_model(args, device=torch.device("cpu"))
    state = torch.load(folder / "model" / "state_dict_best.pt", map_location="cpu",
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, args


def frame_table(players_m, team, ball_m, pitch, blocks, alive, fps) -> pd.DataFrame:
    """PathCRF's tracking table (105 x 68 m); blocks are half-open [a, b) periods."""
    n = players_m.shape[0]
    sx, sy = PITCH[0] / pitch[0], PITCH[1] / pitch[1]
    period = np.zeros(n, dtype=int)
    stamp = np.zeros(n)
    for k, (a, b) in enumerate(blocks):
        period[a:b] = k + 1
        stamp[a:b] = np.arange(b - a) / fps
    cols = {"frame_id": np.arange(n), "period_id": period, "timestamp": stamp,
            "ball_state": np.where(alive, "alive", "dead"),
            "ball_owning_team_id": np.full(n, None, dtype=object)}
    for s in range(players_m.shape[1]):
        side = "home" if team[s] == 0 else "away"
        cols["%s_%d_x" % (side, s)] = players_m[:, s, 0].astype(np.float64) * sx
        cols["%s_%d_y" % (side, s)] = players_m[:, s, 1].astype(np.float64) * sy
    cols["ball_x"] = ball_m[:, 0].astype(np.float64) * sx
    cols["ball_y"] = ball_m[:, 1].astype(np.float64) * sy
    table = pd.DataFrame(cols)
    return table[period > 0].reset_index(drop=True)


def _team_code(names) -> np.ndarray:
    """+1 home, -1 away, 0 for an out-of-pitch node, per PathCRF node name."""
    return np.array([1.0 if str(x).startswith("home_") else -1.0 if str(x).startswith("away_") else 0.0
                     for x in names], dtype=np.float32)


def _share(frame: pd.DataFrame, tag: str) -> np.ndarray:
    cols = [c for c in frame.columns if c.startswith(tag)]
    return frame[cols].sum(axis=1, min_count=1).to_numpy(dtype=np.float64)


def outputs(match: Match, alive: np.ndarray, fps: int = FPS) -> dict:
    """Everything the checkpoint infers per frame, rounded to DIGITS; NaN where it gives nothing.

    p_home, p_next_home: P(the last / next possessor is home). prev_out, next_out: the mass of the
    out-of-pitch nodes. src_team, dst_team, edge_loop, dst_out: the CRF-decoded possession path (+1
    home, -1 away, 0 out of pitch; a self-loop is a player on the ball, a different pair a pass in
    flight, an out-of-pitch destination the ball leaving)."""
    torch, pcrf_utils, inference, _ = _modules()
    model, args = load_model()
    scale = np.asarray(match.pitch_m, dtype=np.float64)
    players = match.players_xy.astype(np.float64) * scale
    ball = match.ball_xy.astype(np.float64) * scale
    blocks = [(first, last + 1) for _, first, last in match.halves]
    tracks, tteam = retrack_all(players, np.asarray(match.team_of_slot), blocks, fps)
    table = frame_table(tracks, tteam, ball, match.pitch_m, blocks, alive, fps)
    sink = StringIO()   # PathCRF's progress output could echo values: keep it off the console
    with warnings.catch_warnings(), contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        warnings.simplefilter("ignore")
        table, _ = pcrf_utils.label_frames_and_episodes(table, fps=fps)
        table, _ = pcrf_utils.label_phases(table)
        table = pcrf_utils.calculate_running_features(table, fps=fps)
        with torch.no_grad():
            prev, nxt, path, _ = inference(model, table, ball_feature=False, use_crf=True,
                                           evaluate=False, window_seconds=args["window_seconds"],
                                           fps=fps, sample_freq=int(fps / args["target_fps"]))
    n = match.n_frames
    out = {k: np.full(n, np.nan) for k in ("p_home", "p_next_home", "prev_out", "next_out", "src_team",
                                           "dst_team", "edge_loop", "dst_out")}
    if prev is None:
        return out
    index = prev.index.to_numpy(dtype=int)
    for name, frame, key in (("prev", prev, "p_home"), ("next", nxt, "p_next_home")):
        h, a = _share(frame, "home_"), _share(frame, "away_")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[key][index] = h / (h + a)
        out[f"{name}_out"][index] = _share(frame, "out_")
    valid = path["edge_src"].notna() & path["edge_dst"].notna()
    src, dst = path.loc[valid, "edge_src"].to_numpy(), path.loc[valid, "edge_dst"].to_numpy()
    frames = path.index[valid].to_numpy(dtype=int)
    out["src_team"][frames] = _team_code(src)
    out["dst_team"][frames] = _team_code(dst)
    out["edge_loop"][frames] = (src == dst).astype(np.float64)
    out["dst_out"][frames] = np.array([str(x).startswith("out_") for x in dst], dtype=np.float64)
    return {k: np.round(v, DIGITS) for k, v in out.items()}


def home_probability(match: Match, alive: np.ndarray, fps: int = FPS) -> np.ndarray:
    """(n,) P(last possessor is home), rounded to DIGITS; NaN where PathCRF gives nothing."""
    return outputs(match, alive, fps)["p_home"]


def logodds(p: np.ndarray, clip: float = CLIP) -> np.ndarray:
    """Home log-odds clipped to +-clip; 0 where PathCRF gave nothing."""
    eps = 10.0 ** -DIGITS
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))
    return np.nan_to_num(np.clip(lo, -clip, clip))
