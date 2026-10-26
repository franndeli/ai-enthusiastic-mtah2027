import unittest

import numpy as np

from possession.frames import AWAY, HOME, FrameTable
from possession.trajectory import control_sign, trajectory_features

FPS = 25.0


def play(n_dribble=50, n_flight=25, n_after=100):
    """Home A dribbles at 2 m/s, passes at 15 m/s past a standing opponent, home B dribbles on.

    Slots: home A, away (0.5 m off the pass line at x = -8), home B. Returns (ball, players).
    """
    n = n_dribble + n_flight + n_after
    ball = np.zeros((n, 2))
    players = np.zeros((n, 3, 2))
    t = np.arange(n_dribble) / FPS
    players[:n_dribble, 0, 0] = -20 + 2 * t
    ball[:n_dribble, 0] = players[:n_dribble, 0, 0] + 0.5
    players[n_dribble:, 0, 0] = players[n_dribble - 1, 0, 0]
    start = ball[n_dribble - 1, 0]
    ball[n_dribble:n_dribble + n_flight, 0] = start + 15 * (np.arange(n_flight) + 1) / FPS
    players[:, 1] = [-8.0, 0.5]
    receive = ball[n_dribble + n_flight - 1, 0]
    players[:n_dribble + n_flight, 2, 0] = receive
    t = np.arange(n_after) / FPS
    players[n_dribble + n_flight:, 2, 0] = receive + 2 * t
    ball[n_dribble + n_flight:, 0] = players[n_dribble + n_flight:, 2, 0] + 0.5
    return ball, players


def table(ball, players, period=None, slot_team=(HOME, AWAY, HOME)):
    n = len(ball)
    return FrameTable(fps=FPS, pitch_m=(105.0, 68.0),
                      period=np.ones(n, np.int8) if period is None else period,
                      ball_xy_m=ball.astype(np.float32), players_xy_m=players.astype(np.float32),
                      slot_team=np.array(slot_team, np.int8))


def column(feats, name):
    return feats.values[:, feats.names.index(name)]


class ControlTest(unittest.TestCase):
    def test_dribble_is_control_and_pass_past_an_opponent_is_not(self):
        ctrl = control_sign(table(*play()))
        self.assertTrue((ctrl[10:40] == -1).all())   # home A dribbling
        self.assertTrue((ctrl[90:160] == -1).all())  # home B dribbling
        self.assertTrue(np.isnan(ctrl[55:72]).all())  # in flight, also past the away player
        self.assertFalse((ctrl == 1).any())

    def test_contested_ball_is_no_control(self):
        ball, players = play()
        players[20:40, 1] = ball[20:40] + [0.0, 1.0]  # opponent runs alongside the dribbler
        self.assertTrue(np.isnan(control_sign(table(ball, players))[27:33]).all())

    def test_single_frame_flicker_next_to_an_opponent_is_no_control(self):
        ball, players = play()
        ball[20] = [-8.0, -0.5]  # one-frame tracking glitch 1 m from the away player
        self.assertFalse((control_sign(table(ball, players)) == 1).any())

    def test_interception_changes_control(self):
        ball, players = play()
        ball[60:, 0] = ball[60, 0] + 2 * np.arange(len(ball) - 60) / FPS
        players[60:, 1] = ball[60:] - [0.5, 0.0]  # away player takes the ball and runs with it
        players[:, 2] = [40.0, 20.0]
        self.assertTrue((control_sign(table(ball, players))[80:160] == 1).all())


class TrajectoryFeaturesTest(unittest.TestCase):
    def test_last_team_in_control_carries_through_pass_and_contest(self):
        ball, players = play()
        players[20:40, 1] = ball[20:40] + [0.0, 1.0]
        feats = trajectory_features(table(ball, players))
        self.assertTrue((column(feats, "last_ctrl")[10:90] == -1).all())
        self.assertTrue((column(feats, "next_ctrl")[50:90] == -1).all())

    def test_next_control_announces_an_interception(self):
        ball, players = play()
        ball[60:, 0] = ball[60, 0] + 2 * np.arange(len(ball) - 60) / FPS
        players[60:, 1] = ball[60:] - [0.5, 0.0]
        players[:, 2] = [40.0, 20.0]
        feats = trajectory_features(table(ball, players))
        self.assertEqual(column(feats, "last_ctrl")[55], -1)
        self.assertEqual(column(feats, "next_ctrl")[55], 1)

    def test_control_carries_across_a_ball_gap_but_is_silent_inside_it(self):
        ball, players = play()
        ball[100:140] = np.nan
        feats = trajectory_features(table(ball, players))
        last, since = column(feats, "last_ctrl"), column(feats, "since_ctrl")
        self.assertTrue((last[100:140] == 0).all())  # gaps are left to features.last_sign
        self.assertEqual(last[142], -1)              # ball back, control from before the gap
        self.assertEqual(column(feats, "next_ctrl")[95], -1)
        self.assertGreater(since[142], since[139])
        self.assertGreater(since[139], since[101])
        self.assertTrue(np.isfinite(feats.values).all())

    def test_team_swap_flips_antisymmetric_and_keeps_symmetric(self):
        ball, players = play()
        a = trajectory_features(table(ball, players))
        b = trajectory_features(table(ball, players, slot_team=(AWAY, HOME, AWAY)))
        anti = a.antisymmetric
        self.assertTrue(anti.any() and (~anti).any())
        np.testing.assert_allclose(a.values[:, anti], -b.values[:, anti], atol=1e-6)
        np.testing.assert_allclose(a.values[:, ~anti], b.values[:, ~anti], atol=1e-6)

    def test_control_does_not_carry_over_the_half_boundary(self):
        period = np.array([1] * 60 + [0] + [2] * 114, np.int8)
        feats = trajectory_features(table(*play(), period=period))
        self.assertTrue((feats.values[60] == 0).all())
        last, nxt = column(feats, "last_ctrl"), column(feats, "next_ctrl")
        self.assertTrue((last[10:60] == -1).all())
        self.assertEqual(last[61], 0)  # first-half control is not carried into the second half
        self.assertEqual(nxt[59], 0)   # nor is second-half control seen from the first
        self.assertEqual(nxt[61], -1)

    def test_missing_ball_and_empty_slots_stay_finite(self):
        ball, players = play()
        players = np.concatenate([players, np.full((len(ball), 43, 2), np.nan)], axis=1)
        players[30:35, 0] = np.nan  # the dribbler drops out of tracking
        slots = (HOME, AWAY, HOME) + (HOME, AWAY) * 21 + (AWAY,)
        feats = trajectory_features(table(ball, players, slot_team=slots))
        self.assertTrue(np.isfinite(feats.values).all())
        self.assertEqual(column(feats, "last_ctrl")[32], -1)
        blind = table(np.full_like(ball, np.nan), players, slot_team=slots)
        self.assertTrue(np.isnan(control_sign(blind)).all())
        self.assertTrue(np.isfinite(trajectory_features(blind).values).all())

    def test_identity_switch_between_home_slots_keeps_the_controlling_team(self):
        ball, players = play()
        players[:, 2] = [30.0, -20.0]  # home B far away; then swaps identity with the dribbler
        players[25:50, [0, 2]] = players[25:50, [2, 0]]
        last = column(trajectory_features(table(ball, players)), "last_ctrl")
        self.assertTrue((last[10:50] == -1).all())


if __name__ == "__main__":
    unittest.main()
