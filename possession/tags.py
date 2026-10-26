"""Smart Tagging intervals as possession evidence (DATA.md, "Smart Tagging").

Every phase carries the team performing it. For attacking phases and set pieces that is the team in
possession; for the defending phases and the defensive transition it is the team without the ball.
Throw-ins and a few other instances are tagged `N/A` and carry no team. Tags are automatic, with
false positives and negatives, and intervals are phases, not possessions: this module only turns
them into per-frame evidence; how much weight it gets is decided elsewhere.
"""

from __future__ import annotations

import numpy as np

# The team performing the phase has the ball.
IN_POSSESSION = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD",
                 "BALL IN THE BOX", "PLAYERS IN THE BOX", "ATTACKING TRANSITION", "LONG BALL",
                 "SET PIECES", "GOALS")
# The team performing the phase is defending.
OUT_OF_POSSESSION = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                     "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")
# Tag text -> +1 home, -1 away. BAR-ATH: home is Barcelona (OVERVIEW.md); DATA.md spells the away
# side "Atletic Club".
TEAM_SIGN = {"barcelona": 1, "fc barcelona": 1, "atletic club": -1, "athletic club": -1,
             "athletic": -1}


def team_sign(text: str) -> int:
    return TEAM_SIGN.get((text or "").strip().lower(), 0)


def possession_sign(event: dict) -> int:
    """+1 if the tag says home has the ball, -1 away, 0 no team or not a phase we use."""
    code = event["code"].strip().upper()
    sign = team_sign(event["tags"].get("Team", ""))
    if code in IN_POSSESSION:
        return sign
    if code in OUT_OF_POSSESSION:
        return -sign
    return 0


def interval(event: dict, n: int):
    a = max(0, min(n, int(event["start"])))
    b = max(a, min(n, int(event["end"]) + 1))
    return a, b


def evidence(events: list, n: int, codes=None):
    """(net, home, away): per frame, how many tags say home / away has the ball, and their
    difference. `codes` restricts to a subset of codes."""
    home = np.zeros(n, dtype=np.int16)
    away = np.zeros(n, dtype=np.int16)
    for ev in events:
        if codes is not None and ev["code"].strip().upper() not in codes:
            continue
        sign = possession_sign(ev)
        if sign == 0:
            continue
        a, b = interval(ev, n)
        if sign > 0:
            home[a:b] += 1
        else:
            away[a:b] += 1
    return home.astype(np.int32) - away, home, away
