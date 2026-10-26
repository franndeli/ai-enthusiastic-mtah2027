"""Synthetic checks of the Metrica sample-data label derivation; never reads downloaded files."""
import unittest

import numpy as np

from possession import A, D, H, X
from possession import metrica as M
from possession.io import AWAY, HOME


def ev(team, kind, start, end=None, sub="", period=1):
    return {"team": team, "type": kind, "subtype": sub, "period": period, "start": start,
            "end": start if end is None else end}


class LabelTests(unittest.TestCase):
    periods = {1: (0, 99)}

    def test_dead_until_kickoff_then_owner_follows_last_touch(self):
        codes = M.labels_from_events([
            ev(HOME, "SET PIECE", 10, sub="KICK OFF"),
            ev(HOME, "PASS", 20, 30),
            ev(AWAY, "RECOVERY", 40, sub="INTERCEPTION"),
            ev(AWAY, "PASS", 50, 60),
        ], self.periods)
        self.assertTrue((codes[:10] == D).all())
        self.assertTrue((codes[10:40] == H).all())
        self.assertTrue((codes[40:] == A).all())

    def test_ball_lost_keeps_the_loser_until_the_recovery(self):
        codes = M.labels_from_events([
            ev(HOME, "SET PIECE", 0),
            ev(HOME, "BALL LOST", 20, 30, sub="INTERCEPTION"),
            ev(AWAY, "RECOVERY", 30),
        ], self.periods)
        self.assertTrue((codes[:30] == H).all())
        self.assertTrue((codes[30:] == A).all())

    def test_ball_out_is_dead_from_the_crossing_until_the_restart(self):
        codes = M.labels_from_events([
            ev(HOME, "SET PIECE", 0),
            ev(HOME, "BALL OUT", 20, 25),
            ev(AWAY, "SET PIECE", 60, sub="THROW IN"),
        ], self.periods)
        self.assertTrue((codes[:25] == H).all())
        self.assertTrue((codes[25:60] == D).all())
        self.assertTrue((codes[60:] == A).all())

    def test_foul_goal_and_offside_are_stoppages(self):
        codes = M.labels_from_events([
            ev(AWAY, "SET PIECE", 0),
            ev(AWAY, "FAULT RECEIVED", 10),
            ev(AWAY, "SET PIECE", 20, sub="FREE KICK"),
            ev(AWAY, "SHOT", 30, 35, sub="ON TARGET-GOAL"),
            ev(HOME, "SET PIECE", 50, sub="KICK OFF"),
            ev(HOME, "BALL LOST", 60, sub="OFFSIDE"),
            ev(AWAY, "SET PIECE", 70, sub="FREE KICK"),
        ], self.periods)
        self.assertTrue((codes[:10] == A).all())
        self.assertTrue((codes[10:20] == D).all())
        self.assertTrue((codes[20:35] == A).all())
        self.assertTrue((codes[35:50] == D).all())
        self.assertTrue((codes[50:60] == H).all())
        self.assertTrue((codes[60:70] == D).all())
        self.assertTrue((codes[70:] == A).all())

    def test_challenges_and_cards_do_not_change_the_owner(self):
        codes = M.labels_from_events([
            ev(HOME, "SET PIECE", 0),
            ev(AWAY, "CHALLENGE", 20, sub="TACKLE-WON"),
            ev(AWAY, "CARD", 30, sub="YELLOW"),
        ], self.periods)
        self.assertTrue((codes == H).all())

    def test_periods_are_independent_and_frames_outside_are_x(self):
        codes = M.labels_from_events([
            ev(HOME, "SET PIECE", 0, period=1),
            ev(AWAY, "SET PIECE", 60, period=2),
        ], {1: (0, 39), 2: (60, 99)})
        self.assertTrue((codes[:40] == H).all())
        self.assertTrue((codes[40:60] == X).all())
        self.assertTrue((codes[60:] == A).all())


class ParsingTests(unittest.TestCase):
    def test_team_names(self):
        self.assertEqual(M._team_code("Home"), HOME)
        self.assertEqual(M._team_code("Team B"), AWAY)
        self.assertIsNone(M._team_code(""))


if __name__ == "__main__":
    unittest.main()
