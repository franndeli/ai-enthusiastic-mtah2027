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
    """{half: first video frame of play}, from the published anchors."""
    entry = GAMES[game]
    return {1: anchor_video_frame(entry["kickoff"]), 2: anchor_video_frame(entry["kickoff_2nd"])}


def half_play_blocks(game: str = GAME) -> list:
    """[(half, first_play_frame, last_graded_frame)]: the part of each half worth decoding."""
    kick = kickoff_frames(game)
    return [(half, max(first, kick[half]), last) for half, first, last in half_video_blocks(game)]


def decode_halves(log_evidence: np.ndarray, decode_fn, game: str = GAME) -> np.ndarray:
    """Run decode_fn on each half's play block; X outside; D between graded start and kick-off."""
    n = log_evidence.shape[0]
    codes = np.full(n, X, dtype=np.int8)
    for half, first_graded, last_graded in half_video_blocks(game):
        first_play = max(first_graded, kickoff_frames(game)[half])
        codes[first_graded:first_play] = D
        codes[first_play:last_graded + 1] = decode_fn(log_evidence[first_play:last_graded + 1])
    return codes
