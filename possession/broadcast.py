"""Formation-only constant-offset alignment; no labels, ball fitting, or file access."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
from scipy.spatial import cKDTree
from possession.io import Match


@dataclass
class Alignment:
    tactical_frame: np.ndarray  # broadcast indexed; -1 means rejected
    segments: int
    aligned_segments: int


def formation_descriptor(match: Match) -> np.ndarray:
    """Team-aware marginal position quantiles in metres, independent of slot identity."""
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for team in (0, 1):
            positions = match.players_xy[:, match.team_of_slot == team] * match.pitch_m
            values = np.nanquantile(positions, np.linspace(0, 1, 11), axis=1)
            parts.append(values.transpose(1, 0, 2).reshape(match.n_frames, -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def align(tactical: Match, broadcast: Match, valid: np.ndarray,
          tactical_descriptor: np.ndarray | None = None,
          broadcast_descriptor: np.ndarray | None = None,
          stride: int = 25, min_segment_frames: int = 75,
          max_residual_m: float = 8., ambiguity_ratio: float = .8) -> Alignment:
    """Match each usable broadcast run, rejecting ambiguous or non-forward sequences.

    Candidate clocks come from nearest formations. Scores compare whole sequences;
    the second-best clock must be at least two seconds away, since adjacent frames
    are necessarily similar. Ball positions are never used to estimate the clock.
    """
    td = formation_descriptor(tactical) if tactical_descriptor is None else tactical_descriptor
    bd = formation_descriptor(broadcast) if broadcast_descriptor is None else broadcast_descriptor
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != (broadcast.n_frames,):
        raise ValueError('broadcast valid mask has wrong shape')
    tvalid = tactical.graded_mask() & np.isfinite(td).all(axis=1)
    tvalid &= np.isfinite(tactical.players_xy).all(axis=2).sum(axis=1) >= 14
    usable = valid & np.isfinite(bd).all(axis=1)
    usable &= np.isfinite(broadcast.players_xy).all(axis=2).sum(axis=1) >= 14
    bounds = np.flatnonzero(np.diff(np.r_[False, usable, False]))
    mapping = np.full(broadcast.n_frames, -1, dtype=np.int32)
    references = np.flatnonzero(tvalid)[::stride]
    if not len(references):
        return Alignment(mapping, len(bounds)//2, 0)
    tree = cKDTree(td[references])
    accepted = 0
    for first, stop in bounds.reshape(-1, 2):
        if stop-first < min_segment_frames:
            continue
        samples = np.arange(first, stop, stride)
        _, neighbours = tree.query(bd[samples], k=min(4, len(references)), workers=1)
        neighbours = np.asarray(neighbours).reshape(len(samples), -1)
        offsets = references[neighbours] - samples[:, None]
        buckets, counts = np.unique(np.rint(offsets/stride).astype(int), return_counts=True)
        top = buckets[np.argsort(-counts, kind='stable')[:12]] * stride
        candidates = np.unique((top[:, None] + np.arange(-stride, stride+1)).ravel())
        scores = np.full(len(candidates), np.inf)
        reversed_scores = np.full(len(candidates), np.inf)
        for j, offset in enumerate(candidates):
            idx = samples + offset
            if idx[0] < 0 or idx[-1] >= tactical.n_frames or not tvalid[idx].all():
                continue
            scores[j] = np.sqrt(np.mean((td[idx]-bd[samples])**2))
            reversed_scores[j] = np.sqrt(np.mean((td[idx]-bd[samples[::-1]])**2))
        best = int(scores.argmin())
        offset = candidates[best]
        other = scores[np.abs(candidates-offset) > 2*stride]
        second = other.min() if len(other) else np.inf
        # A reverse replay can contain exact nearest formations but cannot explain
        # their forward evolution with one clock offset.
        if not (scores[best] < max_residual_m and scores[best] < ambiguity_ratio*second
                and scores[best] < .9*reversed_scores.min()):
            continue
        idx = np.arange(first, stop) + offset
        if idx[0] < 0 or idx[-1] >= tactical.n_frames or not tvalid[idx].all():
            continue
        mapping[first:stop] = idx
        accepted += 1
    return Alignment(mapping, len(bounds)//2, accepted)


def recover_mapped_ball(tactical: Match, broadcast: Match, alignment: Alignment) -> np.ndarray:
    """Recover in-pitch balls; deterministic first broadcast observation wins overlaps."""
    result = np.full((tactical.n_frames, 2), np.nan, dtype=np.float32)
    ball = broadcast.ball_xy
    good = (alignment.tactical_frame >= 0) & np.isfinite(ball).all(axis=1)
    good &= ((ball >= 0) & (ball <= 1)).all(axis=1)
    source = np.flatnonzero(good)
    destination = alignment.tactical_frame[source]
    unique, first = np.unique(destination, return_index=True)
    result[unique] = ball[source[first]]
    return result


def recovered_ball(match_tactical: Match, match_broadcast: Match, valid: np.ndarray,
                   **kwargs) -> np.ndarray:
    """Broadcast ball on the tactical clock, NaN wherever alignment is rejected."""
    alignment = align(match_tactical, match_broadcast, valid, **kwargs)
    return recover_mapped_ball(match_tactical, match_broadcast, alignment)


def chamfer_m(tactical: Match, broadcast: Match, tactical_idx: np.ndarray,
              broadcast_idx: np.ndarray) -> np.ndarray:
    """Symmetric team-aware nearest-player residual; sampled inputs bound memory."""
    distances = []
    for team in (0, 1):
        a = tactical.players_xy[tactical_idx][:, tactical.team_of_slot == team] * tactical.pitch_m
        b = broadcast.players_xy[broadcast_idx][:, broadcast.team_of_slot == team] * broadcast.pitch_m
        d = np.sqrt(np.sum((a[:, :, None]-b[:, None, :])**2, axis=-1))
        d[~np.isfinite(d)] = np.inf
        for minima in (d.min(axis=1), d.min(axis=2)):
            distances.append(minima[np.isfinite(minima)])
    return np.concatenate(distances)
