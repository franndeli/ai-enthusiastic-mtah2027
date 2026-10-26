"""Smart Tagging -> per-frame log-evidence over H/A/D for model.decode (I4). Aggregates only.

    scripts/py -m possession.smart_tagging      # per-code agreement table + calibration, public window

Metrica's tags are inferred, not truth, and phases are not possessions, so each tag only adds a
calibrated log-likelihood ratio: log P(state | tag) - log P(state), rates measured on the public
5 minutes (RULES.md) and shrunk toward "no information" by Beta pseudo-counts in intervals.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from possession.frames import A, D, H, X
from starter.load_smart_tagging import load as load_instances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402

TEAM_SIDE = {"barcelona": H, "atletic club": A}  # DATA.md; N/A (every throw-in) -> no team
# Possession phases: True = the Team tag owns the ball, False = it names the defending team.
# Each DEFENDING/DEFENSIVE code is an opposite-team twin of an attacking instance (140 of 142),
# so claims are unioned per side, never summed. Zone codes (BALL IN FINAL THIRD, BALL IN THE BOX,
# PLAYERS IN THE BOX: pooled public agreement 0.50) and GOALS give no evidence.
PHASE_OWNS = {
    "BUILD UP": True, "PROGRESSION": True, "CREATING CHANCES": True,
    "ATTACKING TRANSITION": True, "LONG BALL": True,
    "DEFENDING IN ATTACKING THIRD": False, "DEFENDING IN MIDDLE THIRD": False,
    "DEFENDING IN DEFENSIVE THIRD": False, "DEFENSIVE TRANSITION": False,
}
SET_PIECE = "SET PIECES"
# Set-piece windows are 35 s and the restart sits ~5 s after their start (model D->alive switch
# over all 59: median 5.3 s, IQR 3.4-6.7 s), so: dead ball around the start, the tagged team after.
RESTART_DEAD_S = (-4.0, 4.0)
RESTART_TEAM_S = (6.0, 16.0)
PSEUDO_INTERVALS = 2.0  # Beta prior strength toward "no information", in tagged intervals


@dataclass(frozen=True)
class Calibration:
    dead_share: float    # b: base D share of the calibration frames (the prior)
    phase_agree: float   # P(owner side | phase-covered alive frame), shrunk toward 0.5
    restart_dead: float  # P(D | frame near a set-piece start), shrunk toward dead_share
    restart_agree: float  # P(tagged team | alive frame after a set piece), shrunk toward 0.5


def _code(instance) -> str:
    return (instance["code"] or "").strip().upper()


def _side(instance):
    return TEAM_SIDE.get((instance["tags"].get("Team") or "").strip().lower())


def _spans(instances, n_frames: int, fps: float) -> dict:
    """Unique half-open frame spans per evidence kind, clipped to the timeline.

    phase / restart_team: {(start, stop, side)}; restart_dead: {(start, stop)}.
    """
    out = {"phase": set(), "restart_dead": set(), "restart_team": set()}

    def clip(start, stop):
        return min(max(start, 0), n_frames), min(max(stop, 0), n_frames)

    for inst in instances:
        code, side = _code(inst), _side(inst)
        start, stop = inst["start_frame"], inst["end_frame"]
        if code in PHASE_OWNS and side is not None:
            owner = side if PHASE_OWNS[code] else 1 - side
            out["phase"].add((*clip(start, stop), owner))
        elif code == SET_PIECE:
            dead = clip(start + round(RESTART_DEAD_S[0] * fps), start + round(RESTART_DEAD_S[1] * fps))
            out["restart_dead"].add(dead)
            if side is not None:
                team = clip(start + round(RESTART_TEAM_S[0] * fps),
                            start + round(RESTART_TEAM_S[1] * fps))
                out["restart_team"].add((*team, side))
    return {kind: {s for s in spans if s[1] > s[0]} for kind, spans in out.items()}


def _side_masks(spans, n_frames: int) -> np.ndarray:
    """(n, 2) bool: frame covered by at least one claim for H / A."""
    mask = np.zeros((n_frames, 2), bool)
    for start, stop, side in spans:
        mask[start:stop, side] = True
    return mask


def _mask(spans, n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, bool)
    for start, stop in spans:
        mask[start:stop] = True
    return mask


def _agreement(masks: np.ndarray, gt: np.ndarray, spans) -> tuple:
    """(agreeing frames, alive frames, intervals) over GT-alive frames claimed by exactly one side."""
    alive = (gt == H) | (gt == A)
    one = masks.sum(axis=1) == 1
    claimed = np.where(masks[:, A], A, H)
    use = alive & one
    n_intervals = sum(alive[a:b].any() for a, b, _ in spans)
    return int((gt[use] == claimed[use]).sum()), int(use.sum()), n_intervals


def _shrink(rate: float, n_intervals: int, target: float) -> float:
    return float((rate * n_intervals + target * PSEUDO_INTERVALS) / (n_intervals + PSEUDO_INTERVALS))


def calibrate(instances, gt: np.ndarray, fps: float) -> Calibration:
    """Rates on the graded frames of gt (H/A/D/X codes), shrunk by tagged intervals present."""
    n = len(gt)
    spans = _spans(instances, n, fps)
    graded = gt != X
    if not (gt == D).any() or not ((gt == H) | (gt == A)).any():
        raise ValueError("calibration labels need both alive (H/A) and dead (D) frames")
    dead_share = float((gt[graded] == D).mean())

    def team_rate(kind):
        agree, alive, n_int = _agreement(_side_masks(spans[kind], n), gt, spans[kind])
        return _shrink(agree / alive if alive else 0.5, n_int, 0.5)

    near = _mask(spans["restart_dead"], n) & graded
    n_dead = sum(graded[a:b].any() for a, b in spans["restart_dead"])
    raw_dead = float((gt[near] == D).mean()) if near.any() else dead_share
    return Calibration(dead_share, team_rate("phase"),
                       _shrink(raw_dead, n_dead, dead_share), team_rate("restart_team"))


def log_evidence(instances, n_frames: int, cal: Calibration, fps: float) -> np.ndarray:
    """(n_frames, 3) log P(state | tags) - log P(state) over H/A/D; 0 where nothing is tagged.

    Prior (1-b)/2, (1-b)/2, b. A team claim with agreement p leaves D at the prior:
    [log 2p, log 2(1-p), 0] for [claimed side, other side, D]. A dead-ball claim with rate d:
    [log((1-d)/(1-b))] * 2 and log(d/b). Claims of one kind are unioned per side; kinds add.
    """
    spans = _spans(instances, n_frames, fps)
    b = cal.dead_share
    out = np.zeros((n_frames, 3))
    for kind, p in (("phase", cal.phase_agree), ("restart_team", cal.restart_agree)):
        mask = _side_masks(spans[kind], n_frames).astype(np.float64)
        own, other = np.log(2 * p), np.log(2 * (1 - p))
        out[:, H] += mask[:, H] * own + mask[:, A] * other
        out[:, A] += mask[:, A] * own + mask[:, H] * other
    dead = _mask(spans["restart_dead"], n_frames)
    d = cal.restart_dead
    out[dead, H] += np.log((1 - d) / (1 - b))
    out[dead, A] += np.log((1 - d) / (1 - b))
    out[dead, D] += np.log(d / b)
    return out


def public_gt_codes(game: str) -> np.ndarray:
    return pm.label_codes(pm.read_label_csv(ROOT / "evaluation" / "gt" /
                                            f"{game}_possession_gt_public.csv"))


def game_log_evidence(game: str, n_frames: int, fps: float) -> np.ndarray:
    """Thin I/O shell: calibrate on the public window, evidence for the whole match."""
    instances = load_instances(game)
    return log_evidence(instances, n_frames, calibrate(instances, public_gt_codes(game), fps), fps)


def main() -> None:
    import games
    game = games.GAME
    fps = games.GAMES[game]["fps"]
    instances = load_instances(game)
    gt = public_gt_codes(game)
    graded = gt != X
    print(f"{len(instances)} instances; public graded frames {int(graded.sum())}")
    print(f"{'code':<30}{'n':>4}{'n_win':>6}{'frames':>7}{'d_rate':>7}{'alive':>6}{'agree':>6}")
    for code in sorted({_code(i) for i in instances}):
        group = [i for i in instances if _code(i) == code]
        frames = alive = agree = dead = in_window = 0
        for inst in group:
            labels = gt[max(inst["start_frame"], 0):max(inst["end_frame"], 0)]
            labels = labels[labels != X]
            in_window += len(labels) > 0
            frames += len(labels)
            dead += int((labels == D).sum())
            if _side(inst) is not None:
                alive += int(((labels == H) | (labels == A)).sum())
                agree += int((labels == _side(inst)).sum())
        d_rate = f"{dead / frames:.2f}" if frames else "-"
        rate = f"{agree / alive:.2f}" if alive else "-"
        print(f"{code:<30}{len(group):>4}{in_window:>6}{frames:>7}{d_rate:>7}{alive:>6}{rate:>6}")
    spans = _spans(instances, len(gt), fps)
    for kind in ("phase", "restart_team"):
        agree, alive, n_int = _agreement(_side_masks(spans[kind], len(gt)), gt, spans[kind])
        print(f"{kind}: {n_int} intervals in window, agreement {agree}/{alive}")
    print(calibrate(instances, gt, fps))


if __name__ == "__main__":
    main()
