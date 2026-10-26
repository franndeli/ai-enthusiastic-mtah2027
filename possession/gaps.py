"""Where the ball is missing, and what a labelling does there. Aggregates only.

DATA.md: the ball is missing on 41% of graded frames (59.3% seen), disproportionately on the frames
that matter. This module splits graded frames into "ball seen" and gaps by length, so the label mix
and the team changes inside gaps can be compared between submissions, and with the external
benchmark where the truth is known.
"""

from __future__ import annotations

import numpy as np

from . import H, A, D, X

BUCKETS = ("ball seen", "gap < 1 s", "gap 1-10 s", "gap >= 10 s")


def _runs(mask: np.ndarray):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def buckets(ball_x: np.ndarray, blocks, fps: float) -> np.ndarray:
    """Per frame: -1 outside the graded blocks, 0 ball seen, 1/2/3 inside a ball gap of < 1 s,
    1-10 s or >= 10 s. Gaps are cut at block edges."""
    seen = ~np.isnan(np.asarray(ball_x, dtype=np.float64))
    out = np.full(len(seen), -1, dtype=np.int8)
    for a, b in blocks:
        out[a:b] = 0
        for s, e in _runs(~seen[a:b]):
            length = (e - s) / fps
            out[a + s:a + e] = 1 if length < 1.0 else (2 if length < 10.0 else 3)
    return out


def gap_summary(bucket: np.ndarray, window=None) -> dict:
    """Share of graded frames without ball, and how the missing frames split by gap length."""
    graded = bucket >= 0
    if window is not None:
        inside = np.zeros(len(bucket), dtype=bool)
        inside[window[0]:window[1]] = True
        graded &= inside
    missing = graded & (bucket > 0)
    out = {"graded_frames": int(graded.sum()),
           "missing_share": float(missing.sum() / max(graded.sum(), 1))}
    for k in (1, 2, 3):
        m = graded & (bucket == k)
        out["missing_in_%d" % k] = float(m.sum() / max(missing.sum(), 1))
        out["gaps_%d" % k] = len(_runs(m))
    return out


def label_profile(bucket: np.ndarray, labels: np.ndarray) -> dict:
    """Per bucket: share of frames per label and H<->A changes inside gaps (D in between ignored)."""
    out = {}
    for k in range(4):
        m = bucket == k
        n = int(m.sum())
        row = {"frames": n}
        for code, name in ((H, "H"), (A, "A"), (D, "D"), (X, "X")):
            row[name] = float(np.sum(labels[m] == code) / max(n, 1))
        if k > 0:
            changes, gaps = 0, 0
            for s, e in _runs(m):
                gaps += 1
                team = labels[s:e][np.isin(labels[s:e], (H, A))]
                changes += int(np.sum(team[1:] != team[:-1]))
            row["gaps"], row["team_changes"] = gaps, changes
        out[BUCKETS[k]] = row
    return out
