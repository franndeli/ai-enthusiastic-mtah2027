"""Leave-one-match-out training and validation on DFL (ROUTE_A_PLAN.md T4). Aggregates only.

    scripts/py -m possession.train                       # LOMO table: baseline vs argmax vs Viterbi
    scripts/py -m possession.train --fit-all --emission-scale 0.1   # final model -> work dir

Model selection happens here, on external matches scored with the official spellF1, never on the
public 5 minutes (EVALUATION.md misranking trap). Every table gets the same seeded ball dropout,
because the target ATD misses the ball on ~43% of frames and DFL almost never does.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from possession.features import features
from possession.frames import AWAY, HOME, A, H, X, FrameTable, period_runs
from possession.model import Model, argmax_decode, decode, fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402
from starter.baseline import smooth  # noqa: E402

SEED = 2027
DROPOUT_RATE = 0.43        # ATD ball missing share, all frames (DATA.md)
DROPOUT_MEAN_BLOCK_S = 1.5  # occlusions come in blocks, not single frames
TRAIN_STEP_S = 0.2         # subsample training rows; neighbours are near-duplicates
EMISSION_SCALES = (0.02, 0.05, 0.1, 0.2, 0.5)
assert (HOME, AWAY) == (H, A)  # nearest_team_codes and smooth() treat a team code as its label


def work_dir() -> Path:
    return Path(os.environ.get("MTAH_WORK_DIR", Path.home() / ".mtah2027" / "work"))


def drop_ball(table: FrameTable, rate: float, mean_block_s: float,
              rng: np.random.RandomState) -> FrameTable:
    """Copy with the ball hidden in geometric-length blocks covering ~rate of the frames."""
    n = table.n_frames
    hidden_mean = max(1.0, mean_block_s * table.fps)
    shown_mean = hidden_mean * (1 - rate) / rate
    mask = np.zeros(n, dtype=bool)
    pos, hidden = 0, bool(rng.rand() < rate)
    while pos < n:
        length = rng.geometric(1.0 / (hidden_mean if hidden else shown_mean))
        mask[pos:pos + length] = hidden
        pos += length
        hidden = not hidden
    ball = table.ball_xy_m.copy()
    ball[mask] = np.nan
    return FrameTable(table.fps, table.pitch_m, table.period, ball, table.players_xy_m,
                      table.slot_team, table.labels)


def mirror(values: np.ndarray, labels: np.ndarray, antisymmetric: np.ndarray) -> tuple:
    """Team-swapped copy: antisymmetric columns negate, H and A trade places."""
    flipped = values.copy()
    flipped[:, antisymmetric] *= -1
    swapped = np.where(labels == H, A, np.where(labels == A, H, labels)).astype(labels.dtype)
    return flipped, swapped


def nearest_team_codes(table: FrameTable) -> np.ndarray:
    """starter/baseline.py step 1 on a FrameTable: team of the nearest player, -1 if no ball."""
    dist = np.hypot(table.players_xy_m[..., 0] - table.ball_xy_m[:, None, 0],
                    table.players_xy_m[..., 1] - table.ball_xy_m[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    codes = table.slot_team[nearest].astype(np.int8)
    codes[~np.isfinite(dist.min(axis=1))] = -1
    return codes


def baseline_codes(table: FrameTable, window: int = 50) -> np.ndarray:
    return pm.label_codes(smooth(nearest_team_codes(table), window))


def prepare(table: FrameTable, rng: np.random.RandomState) -> dict:
    """Dropout-augmented table, its features, and the subsampled + mirrored training rows."""
    dropped = drop_ball(table, DROPOUT_RATE, DROPOUT_MEAN_BLOCK_S, rng)
    feats = features(dropped)
    step = max(1, int(round(TRAIN_STEP_S * table.fps)))
    rows = np.flatnonzero((table.period > 0) & (table.labels != X))[::step]
    values, labels = feats.values[rows], table.labels[rows]
    flipped, swapped = mirror(values, labels, feats.antisymmetric)
    runs = [table.labels[a:b] for a, b in period_runs(table.period)]
    mirrored_runs = [np.where(r == H, A, np.where(r == A, H, r)) for r in runs]
    return dict(table=dropped, values=feats.values, train_x=np.vstack([values, flipped]),
                train_y=np.concatenate([labels, swapped]), runs=runs + mirrored_runs)


def train(prepared: list, emission_scale: float) -> Model:
    return fit(np.vstack([p["train_x"] for p in prepared]),
               np.concatenate([p["train_y"] for p in prepared]),
               [r for p in prepared for r in p["runs"]], emission_scale)


def score(labels: np.ndarray, pred: np.ndarray) -> tuple:
    result = pm.evaluate(labels.astype(np.int8), pred.astype(np.int8))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def load_tables() -> list:
    paths = sorted((work_dir() / "dfl").glob("*.npz"))
    if not paths:
        sys.exit("no DFL tables found; run scripts/fetch_external.py and possession.io_dfl first")
    return [(p.stem, FrameTable.load(p)) for p in paths]


def lomo(tables: list) -> None:
    rng = np.random.RandomState(SEED)
    prepared = [prepare(t, rng) for _, t in tables]
    header = ["match", "gt_spells", "base50"] + ["argmax"] + [f"vit{s}" for s in EMISSION_SCALES]
    print("spellF1 (macroF1) [predicted spells] per held-out match")
    print("  ".join(f"{h:>22}" for h in header))
    totals = np.zeros((len(header) - 2, 2))
    for i, (name, table) in enumerate(tables):
        start = time.time()
        model = train(prepared[:i] + prepared[i + 1:], EMISSION_SCALES[0])
        held = prepared[i]
        preds = [baseline_codes(held["table"]),
                 argmax_decode(model, held["values"], table.period)]
        preds += [decode(model.with_emission_scale(s), held["values"], table.period)
                  for s in EMISSION_SCALES]
        cells, n_gt = [], 0
        for j, pred in enumerate(preds):
            spell, macro, n_pred, n_gt = score(table.labels, pred)
            totals[j] += (spell, macro)
            cells.append(f"{spell:.4f} ({macro:.3f}) [{n_pred}]")
        print("  ".join(f"{c:>22}" for c in [name, str(n_gt)] + cells),
              f"  {time.time() - start:.0f}s")
    means = totals / len(tables)
    print("  ".join(f"{c:>22}" for c in ["MEAN", ""] + [f"{s:.4f} ({m:.3f})" for s, m in means]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-all", action="store_true", help="fit on every match and save")
    parser.add_argument("--emission-scale", type=float, default=0.1)
    args = parser.parse_args()
    tables = load_tables()
    print(f"{len(tables)} DFL matches, {sum(t.n_frames for _, t in tables)} frames")
    if not args.fit_all:
        lomo(tables)
        return
    rng = np.random.RandomState(SEED)
    model = train([prepare(t, rng) for _, t in tables], args.emission_scale)
    path = work_dir() / "model.npz"
    model.save(path)
    print(f"model saved to the work dir as {path.name} (emission_scale {args.emission_scale})")


if __name__ == "__main__":
    main()
