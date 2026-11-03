"""The v6 evidence stack (D015): rule logits plus the learned and PathCRF corrections. Pure.

One function builds the (n, 3) log-evidence for both serving (run.py) and training
(scripts/train_v6.py), so the two cannot drift apart:

    team logit = rule team logit (+ w_pathcrf * PathCRF home log-odds inside the rule alive stretches)
    dead logit = rule dead logit (+ learned ball-free residual) + w_fresh_exit * fresh_exit

With no dead model and w_pathcrf 0 this is exactly evidence.scores (v5 rules + fresh exit).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from possession import A, H, dead_residual, evidence
from possession.io import Match


def log_evidence(feats: dict, match: Match, params: dict, dead_model=None,
                 w_pathcrf: float = 0.0, rule_decode: Callable | None = None,
                 home_probability: Callable | None = None) -> np.ndarray:
    """rule_decode(log_evidence) -> codes decodes the rule evidence (needed only for PathCRF's
    alive mask); home_probability(match, alive) defaults to pathcrf.home_probability."""
    if dead_model is None and not w_pathcrf:
        return evidence.scores(feats, params)
    p = dict(evidence.DEFAULT_PARAMS, **params)
    z_team = evidence.team_logit(feats, p)
    z_dead_rule = evidence.dead_logit(feats, dict(p, w_fresh_exit=0.0))
    z_dead = z_dead_rule
    if dead_model is not None:
        # the residual reads the rule logit it was trained on: external matches have no Smart
        # Tagging, so the tag clue is left out of its input (it stays in z_dead itself)
        untagged = evidence.dead_logit(feats, dict(p, w_fresh_exit=0.0, w_tag_dead=0.0))
        z_dead = z_dead + dead_residual.residual(dead_model, match, feats, untagged)
    # added after the residual: the residual reads the rule logit, and this order is the one the
    # models were evaluated and trained with (floating-point sums are order-sensitive)
    z_dead = z_dead + p["w_fresh_exit"] * feats["fresh_exit"]
    if w_pathcrf:
        from possession import pathcrf
        alive = np.isin(rule_decode(evidence.combine(z_team, z_dead_rule, p)), (H, A))
        p_home = (home_probability or pathcrf.home_probability)(match, alive)
        z_team = z_team + w_pathcrf * pathcrf.logodds(p_home)
    return evidence.combine(z_team, z_dead, p)
