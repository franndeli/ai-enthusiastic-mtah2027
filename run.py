"""Regenerate the submission from the provided data with one command. Prints aggregates only.

    scripts/py run.py --out evaluation/results/route_a.csv
    scripts/py run.py --out evaluation/results/no_tags.csv --smart-tagging-weight 0   # ablation

Prerequisites (deterministic, see docs/PLAN.md Route A): scripts/fetch_external.py, then
`scripts/py -m possession.io_dfl`, then `scripts/py -m possession.train --fit-all`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import games
from possession import atd, smart_tagging
from possession.frames import period_runs
from possession.mlp import MLPModel
from possession.model import decode
from possession.postprocess import apply_rules
from possession.train import all_features, work_dir
from starter.baseline import write_submission

sys.path.insert(0, str(Path(__file__).resolve().parent / "evaluation"))
import possession_metrics as pm  # noqa: E402


def second_half_pre_kickoff(game: str) -> tuple:
    """Graded second-half frames before its kickoff; anchor + 1 gives FAQ.md's 2,934 frames."""
    _, start, _ = games.half_video_blocks(game)[1]
    return start, games.anchor_video_frame(games.GAMES[game]["kickoff_2nd"]) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="submission CSV to write")
    parser.add_argument("--game", default=games.GAME)
    parser.add_argument("--smart-tagging-weight", type=float, default=0.25, metavar="W",
                        help="weight of the Smart Tagging log-evidence added before Viterbi "
                             "(possession/smart_tagging.py, D009); 0 disables it")
    args = parser.parse_args()
    if args.smart_tagging_weight < 0:
        parser.error("--smart-tagging-weight must be >= 0")

    model_path = work_dir() / "model.npz"
    if not model_path.exists():
        sys.exit("no trained model in the work dir; run `scripts/py -m possession.train --fit-all`")
    try:
        table = atd.load_game(args.game)
    except IOError as error:
        sys.exit(f"cannot load ATD: {error}")

    extra = None
    if args.smart_tagging_weight > 0:
        try:
            extra = args.smart_tagging_weight * smart_tagging.game_log_evidence(
                args.game, table.n_frames, table.fps)
        except IOError as error:
            sys.exit(f"cannot load Smart Tagging: {error}")
    codes = decode(MLPModel.load(model_path), all_features(table).values, table.period, extra)
    codes = apply_rules(codes, table.period, [second_half_pre_kickoff(args.game)])
    letters = np.array(pm.LABELS)[codes]
    write_submission(args.out, letters)

    problem = pm.validate_submission(pm.read_label_csv(args.out),
                                     games.source_frames("tactical", args.game))
    if problem:
        sys.exit(f"invalid submission: {problem}")
    stats = pm.possession_stats(codes, period_runs(table.period))
    counts = {label: int(np.sum(letters == label)) for label in pm.LABELS}
    print(f"wrote a valid submission: {counts}; {stats['n_spells']} possessions, "
          f"mean {stats['mean_duration_s']:.1f} s, home share {stats['home_share']:.3f}")


if __name__ == "__main__":
    main()
