"""Structural rules from the published frame table (DATA.md, FAQ.md, games.py). Pure numpy.

- Frames outside the graded halves are X (never scored).
- Each half is decoded on its own; nothing carries across halftime.
- Graded frames before a half's kick-off are D (the second half is graded from 110,808 but kicks
  off at 113,742). A leading X there would be back-filled by the scorer with our first label.
"""
from __future__ import annotations

import numpy as np

from games import GAME, GAMES, anchor_video_frame, half_video_blocks
from possession import D, X


def kickoff_frames(game: str = GAME) -> dict:
    """{half: first video frame of play}, following the explicit FAQ boundaries."""
    entry = GAMES[game]
    kick = {1: anchor_video_frame(entry["kickoff"]),
            2: anchor_video_frame(entry["kickoff_2nd"])}
    # FAQ "Which frames are actually graded?" overrides the anchor table:
    # its first-half anchor is one frame late, second-half one frame early.
    if game == "BAR-ATH":
        kick = {1: 11185, 2: 113742}
    return kick


def decode_halves(log_evidence: np.ndarray, decode_fn, game: str = GAME) -> np.ndarray:
    """Run decode_fn on each half's play block; X outside; D between graded start and kick-off.

    decode_fn(block_evidence) or decode_fn(block_evidence, start): the second form receives the
    block's first video frame so per-frame side inputs can be sliced to match.
    """
    n = log_evidence.shape[0]
    codes = np.full(n, X, dtype=np.int8)
    wants_start = _accepts_start(decode_fn)
    for half, first_graded, last_graded in half_video_blocks(game):
        first_play = max(first_graded, kickoff_frames(game)[half])
        codes[first_graded:first_play] = D
        block = log_evidence[first_play:last_graded + 1]
        codes[first_play:last_graded + 1] = (decode_fn(block, first_play) if wants_start
                                             else decode_fn(block))
    return codes


def _accepts_start(fn) -> bool:
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return len([p for p in params.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
