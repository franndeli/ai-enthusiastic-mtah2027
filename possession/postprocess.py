"""Hard rules applied after decoding (ROUTE_A_PLAN.md decision 6)."""

import numpy as np

from possession.frames import D, X


def apply_rules(codes: np.ndarray, period: np.ndarray, dead_spans: list) -> np.ndarray:
    """Copy of codes with known dead-ball spans forced to D and frames outside play set to X.

    dead_spans are half-open [start, stop) frame ranges, e.g. the second half's graded frames
    before its kickoff (FAQ.md: 2,934 frames, all D).
    """
    out = codes.copy()
    for start, stop in dead_spans:
        out[start:stop] = D
    out[period == 0] = X
    return out
