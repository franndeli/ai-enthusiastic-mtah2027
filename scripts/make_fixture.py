"""Build a full-size, ATD-format dress-rehearsal bundle from a PUBLIC Metrica sample game.

The output follows DATA.md exactly (199,319 records, frame_id 1..199319, 46 slots of "x,y,speed",
ball "x,y,z", EPTS-style metadata, Smart Tagging with the two kick-offs) so the whole local
workflow -- read, predict, write, official validation, official scoring -- can be rehearsed without
the challenge data. The sample game's two periods are placed on the challenge's graded halves and
a matching frame,label ground truth is written from its events.

usage: python scripts/make_fixture.py <metrica sample-data/data> <out dir> [game] [seed]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import D, LABELS, X  # noqa: E402
from possession.external import degrade, load_sample_game  # noqa: E402

ROOT, OUT = sys.argv[1], sys.argv[2]
GAME = int(sys.argv[3]) if len(sys.argv) > 3 else 1
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 7
N, SLOTS, FPS = 199319, 46, 25
HALVES = [(11185, 90652), (110808, 190198)]     # games.half_video_blocks, half-open
PRE_KICKOFF_2 = 2934                            # FAQ.md: 2nd half graded before its kick-off
PITCH = (105.3, 68.0)

trk, gt = load_sample_game(GAME, ROOT)
deg = degrade(trk, gt, seed=SEED)
S = deg.players.shape[1]
home = np.flatnonzero(deg.team == 0)
away = np.flatnonzero(deg.team == 1)
slot_team = np.array([0] * 23 + [1] * 23)
slot_src = list(home[:23]) + [-1] * (23 - len(home[:23])) + list(away[:23]) + [-1] * (23 - len(away[:23]))

players = np.full((N, SLOTS, 2), np.nan)
ball = np.full((N, 2), np.nan)
labels = np.full(N, X, dtype=np.int8)
for (a, b), (sa, sb), pre in zip(HALVES, deg.blocks, (0, PRE_KICKOFF_2)):
    length = min(b - a - pre, sb - sa)
    dst = slice(a + pre, a + pre + length)
    src = slice(sa, sa + length)
    for k, s in enumerate(slot_src):
        if s >= 0:
            players[dst, k] = deg.players[src, s]
    ball[dst] = deg.ball[src]
    labels[dst] = gt[src]
    labels[a:a + pre] = D
    # before the 2nd-half kick-off: players walking to their halves, ball on the centre spot
    if pre:
        for k, s in enumerate(slot_src):
            if s >= 0:
                players[a:a + pre, k] = deg.players[sa, s]
        players[a:a + pre] += np.random.RandomState(SEED).normal(0, 0.05, (pre, SLOTS, 2))
        ball[a + pre - 500:a + pre] = (PITCH[0] / 2, PITCH[1] / 2)
    labels[a + pre + length:b] = D

os.makedirs(os.path.join(OUT, "tactical"), exist_ok=True)
norm_p = players / np.array(PITCH)
norm_b = ball / np.array(PITCH)
with open(os.path.join(OUT, "tactical", "BAR-ATH_tactical_atd.txt"), "w") as handle:
    for i in range(N):
        groups = []
        for k in range(SLOTS):
            x, y = norm_p[i, k]
            groups.append("NaN,NaN,NaN" if np.isnan(x) else "%.5f,%.5f,%.2f" % (x, y, 0.0))
        bx, by = norm_b[i]
        bstr = "NaN,NaN,NaN" if np.isnan(bx) else "%.5f,%.5f,0.00" % (bx, by)
        handle.write("%d:%s:%s\n" % (i + 1, ";".join(groups), bstr))
with open(os.path.join(OUT, "tactical", "BAR-ATH_tactical_atd.xml"), "w") as handle:
    handle.write('<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata><GlobalConfig>'
                 '<FrameRate>25</FrameRate></GlobalConfig><Teams>'
                 '<Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team>'
                 '</Teams><Players>')
    for k in range(SLOTS):
        handle.write('<Player id="P%d" teamId="%s"><Name>Track %d</Name><ShirtNumber>%d</ShirtNumber>'
                     '</Player>' % (k + 1, "T1" if slot_team[k] == 0 else "T2", k + 1, k + 1))
    handle.write("</Players></Metadata></main>\n")
with open(os.path.join(OUT, "tactical", "BAR-ATH_tactical_smart_tagging.xml"), "w") as handle:
    handle.write("<file><ALL_INSTANCES>")
    for k, (a, b) in enumerate(HALVES):
        kick = a + (PRE_KICKOFF_2 if k else 0)
        handle.write("<instance><ID>%d</ID><start>%.2f</start><end>%.2f</end><code>SET PIECES</code>"
                     "<label><group>Type</group><text>Kick Off (Start)</text></label>"
                     "<label><group>Half</group><text>%s Half</text></label></instance>"
                     % (k + 1, (kick - 90) / 25.0, (kick + 35) / 25.0, "1st" if k == 0 else "2nd"))
    handle.write("</ALL_INSTANCES></file>\n")
gt_dir = os.path.join(OUT, "gt")
os.makedirs(gt_dir, exist_ok=True)
with open(os.path.join(gt_dir, "FIXTURE_possession_gt.csv"), "w") as handle:
    handle.write("frame,label\n")
    handle.writelines("%d,%s\n" % (i, LABELS[c]) for i, c in enumerate(labels))
print("fixture written to %s (game %d, seed %d)" % (OUT, GAME, SEED))
