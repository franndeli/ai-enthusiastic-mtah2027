"""Common frame table: the one shape every tracking source (DFL, MTAH ATD, ...) is converted to.

Identity-free on purpose (ROUTE_A_PLAN.md, decision 1): positions and team membership only, so
features computed from it run unchanged on every source. Metres, origin at the pitch centre,
x along the length; NaN marks a missing observation. Row i is frame i of the source timeline.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOME, AWAY = 0, 1
# Label codes match evaluation/possession_metrics.py.
H, A, D, X = 0, 1, 2, 3


@dataclass(frozen=True)
class FrameTable:
    fps: float
    pitch_m: tuple                    # (length_m, width_m)
    period: np.ndarray                # (n,) int8: 1 or 2 = half, 0 = outside play (never graded)
    ball_xy_m: np.ndarray             # (n, 2) float32, NaN when the ball is not seen
    players_xy_m: np.ndarray          # (n, k, 2) float32, NaN when a slot is empty
    slot_team: np.ndarray             # (k,) int8, HOME or AWAY per slot
    labels: np.ndarray | None = None  # (n,) int8 H/A/D/X; only sources with ground truth

    def __post_init__(self):
        n = len(self.period)
        k = len(self.slot_team)
        if self.ball_xy_m.shape != (n, 2):
            raise ValueError(f"ball_xy_m shape {self.ball_xy_m.shape} != ({n}, 2)")
        if self.players_xy_m.shape != (n, k, 2):
            raise ValueError(f"players_xy_m shape {self.players_xy_m.shape} != ({n}, {k}, 2)")
        if not np.isin(self.slot_team, (HOME, AWAY)).all():
            raise ValueError("slot_team must be HOME (0) or AWAY (1)")
        if self.labels is not None and self.labels.shape != (n,):
            raise ValueError(f"labels shape {self.labels.shape} != ({n},)")

    @property
    def n_frames(self) -> int:
        return len(self.period)

    def save(self, path: Path) -> None:
        arrays = dict(fps=np.float64(self.fps), pitch_m=np.asarray(self.pitch_m, np.float64),
                      period=self.period, ball_xy_m=self.ball_xy_m,
                      players_xy_m=self.players_xy_m, slot_team=self.slot_team)
        if self.labels is not None:
            arrays["labels"] = self.labels
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "FrameTable":
        with np.load(path) as z:
            return cls(fps=float(z["fps"]), pitch_m=tuple(z["pitch_m"].tolist()),
                       period=z["period"], ball_xy_m=z["ball_xy_m"],
                       players_xy_m=z["players_xy_m"], slot_team=z["slot_team"],
                       labels=z["labels"] if "labels" in z.files else None)


def period_runs(period: np.ndarray) -> list:
    """Half-open [start, stop) spans of constant non-zero period: the halves, never merged."""
    breaks = np.flatnonzero(np.diff(period)) + 1
    bounds = np.concatenate(([0], breaks, [len(period)]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if period[a] != 0]
