"""Aggregate-only report: how much the Smart Tagging could add to a submission, measured without any
labels. For each tag code: how much of the match it covers (with and without the ball), what the
submission says inside it, and how often the team the tag implies agrees with the submission,
overall and where the model is confident. Prints aggregates only: no rows, no frame numbers, no
times.

usage: python scripts/tag_report.py [--data DIR] [--model config/possession_v4.json]
                                    [--submission evaluation/results/v4.csv]

Agreement on confident frames (ball seen, |team log-odds| >= 2) estimates how reliable a tag is;
coverage and agreement inside ball gaps show how much it could add where the model is weakest.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import A, D, H, LABELS  # noqa: E402
from possession import features as F  # noqa: E402
from possession import smart_tagging  # noqa: E402
from possession.gaps import buckets  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.pipeline import DEFAULT_MODEL, PRIVATE_ROOT, game_constants  # noqa: E402
from possession.tags import IN_POSSESSION, OUT_OF_POSSESSION, interval, possession_sign  # noqa: E402
from possession.tracking import read_epts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(num, den):
    return "%5.1f%%" % (100.0 * num / den) if den else "   - "


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR",
                                                         os.path.join(PRIVATE_ROOT, "data")))
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--submission", default=os.path.join(REPO, "evaluation", "results",
                                                             "v4.csv"))
    args = parser.parse_args()

    games, n_frames, fps, pitch, blocks = game_constants(args.game)
    paths = games.asset_paths(args.game, data_dir=args.data)
    if not all(os.path.isfile(paths[k]) for k in ("atd_meta", "atd_raw", "smart_tagging")):
        print("match data not found; set MTAH_DATA_DIR to the folder that contains tactical/")
        return 2
    trk = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=pitch, n_frames=n_frames)
    trk.blocks = blocks
    events = smart_tagging.load(paths["smart_tagging"], fps)
    lab = pd.read_csv(args.submission, dtype={"label": "object"})["label"].str.strip()
    lab = lab.map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=np.int8)
    model = Model.from_json(args.model)
    logit = model.team_logit(F.compute(trk))
    bucket = buckets(trk.ball[:, 0], blocks, fps)
    graded = bucket >= 0
    seen, gap = bucket == 0, bucket > 0
    alive = graded & np.isin(lab, (H, A))
    sub_team = np.where(lab == H, 1, np.where(lab == A, -1, 0))
    confident = alive & seen & (np.abs(logit) >= 2.0)
    print("%d tag instances; graded frames %d, ball missing on %s of them; submission %s"
          % (len(events), graded.sum(), pct(gap.sum(), graded.sum()),
             os.path.basename(args.submission)))

    print("\nper code: instances (no team), mean length, coverage of graded frames and of ball-gap "
          "frames, submission D inside, agreement of the tag's team with the submission on alive "
          "frames: all / confident (ball seen, |log-odds| >= 2) / inside ball gaps")
    print("  %-30s %5s %5s %7s %8s %8s %7s %8s %10s %8s" % (
        "code", "n", "no-tm", "mean s", "cover", "cover gp", "D in", "agree", "agree conf",
        "agree gp"))
    union = np.zeros(trk.n, dtype=bool)
    net = np.zeros(trk.n, dtype=np.int32)
    for code in IN_POSSESSION + OUT_OF_POSSESSION:
        evs = [e for e in events if e["code"].strip().upper() == code]
        if not evs:
            continue
        cover = np.zeros(trk.n, dtype=bool)
        sign = np.zeros(trk.n, dtype=np.int32)
        for e in evs:
            a, b = interval(e, trk.n)
            cover[a:b] = True
            sign[a:b] += possession_sign(e)
        teamed = cover & (sign != 0)
        union |= teamed
        net += sign
        c, cg = graded & cover, gap & cover
        agree = lambda m: pct(np.sum(m & (np.sign(sign) == sub_team)), np.sum(m))  # noqa: E731
        print("  %-30s %5d %5d %7.1f %8s %8s %7s %8s %10s %8s" % (
            code.lower(), len(evs), sum(possession_sign(e) == 0 for e in evs),
            np.mean([(e["end"] - e["start"]) / fps for e in evs]), pct(c.sum(), graded.sum()),
            pct(cg.sum(), gap.sum()), pct(np.sum(c & (lab == D)), c.sum()),
            agree(teamed & alive), agree(teamed & confident), agree(teamed & alive & gap)))

    conflict = union & (net == 0)
    m = union & alive & ~conflict
    print("\nall phases with a team: cover %s of graded frames, %s of ball-gap frames, %s of the "
          "frames the submission calls D; conflicting tags on %s of covered frames"
          % (pct(np.sum(union & graded), graded.sum()), pct(np.sum(union & gap), gap.sum()),
             pct(np.sum(union & graded & (lab == D)), np.sum(graded & (lab == D))),
             pct(conflict.sum(), union.sum())))
    print("  agreement with the submission on alive frames: all %s, confident %s, ball gaps %s"
          % (pct(np.sum(m & (np.sign(net) == sub_team)), m.sum()),
             pct(np.sum(m & confident & (np.sign(net) == sub_team)), np.sum(m & confident)),
             pct(np.sum(m & gap & (np.sign(net) == sub_team)), np.sum(m & gap))))

    print("\nset pieces by type: submission D share 3 s before the tag, inside it, 3 s after it")
    k = int(round(3 * fps))
    types = sorted({e["tags"].get("Type", "?") for e in events
                    if e["code"].strip().upper() == "SET PIECES"})
    for t in types:
        evs = [e for e in events if e["code"].strip().upper() == "SET PIECES"
               and e["tags"].get("Type", "?") == t]
        parts = {"before": [], "inside": [], "after": []}
        for e in evs:
            a, b = interval(e, trk.n)
            for name, (s, f) in (("before", (a - k, a)), ("inside", (a, b)),
                                 ("after", (b, b + k))):
                s, f = max(0, s), min(trk.n, f)
                w = graded[s:f]
                if w.any():
                    parts[name].append(np.mean(lab[s:f][w] == D))
        print("  %-24s n %3d   D before %s  inside %s  after %s" % (
            t, len(evs), *("%5.1f%%" % (100 * np.mean(v)) if v else "    -" for v in
                           (parts["before"], parts["inside"], parts["after"]))))

    print("\ntransitions: does the submission hand the ball to the tagged side around the tag start?")
    for code in ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION"):
        evs = [e for e in events if e["code"].strip().upper() == code and possession_sign(e)]
        before, after, n_ok = 0, 0, 0
        for e in evs:
            a, _ = interval(e, trk.n)
            want = possession_sign(e)
            pre = sub_team[max(0, a - 3 * int(fps)):max(0, a - int(fps))]
            post = sub_team[a + int(fps):a + 3 * int(fps)]
            pre, post = pre[pre != 0], post[post != 0]
            if len(pre) and len(post):
                n_ok += 1
                before += int(np.mean(pre) * want < 0)
                after += int(np.mean(post) * want > 0)
        print("  %-22s n %3d  other side has it 1-3 s before: %s   tagged side has it 1-3 s after: %s"
              % (code.lower(), n_ok, pct(before, n_ok), pct(after, n_ok)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
