"""Synthetic checks of the feature rules; never reads match assets."""
import unittest
import warnings

import numpy as np

from possession import FPS
from possession import features as F
from possession.io import AWAY, HOME, Match

PITCH = (100.0, 50.0)   # metres; 1 m = 0.01 in x, 0.02 in y


def make_match(n, ball_xy=None, home_xy=(0.5, 0.5), away_xy=(0.9, 0.9), speed=2.0):
    """One home and one away player per frame; ball NaN unless given as (n, 2)."""
    players = np.zeros((n, 2, 2), dtype=np.float32)
    players[:, 0] = home_xy
    players[:, 1] = away_xy
    ball = np.full((n, 2), np.nan, dtype=np.float32) if ball_xy is None else np.asarray(ball_xy, np.float32)
    return Match(players_xy=players, players_speed=np.full((n, 2), speed, dtype=np.float32),
                 ball_xy=ball, ball_speed=np.full(n, np.nan, dtype=np.float32),
                 team_of_slot=np.array([HOME, AWAY], dtype=np.int8), halves=[(1, 0, n - 1)],
                 pitch_m=PITCH)


def still_ball(n, x, y):
    return np.tile(np.array([x, y], dtype=np.float32), (n, 1))


class BallTests(unittest.TestCase):
    def test_speed_from_positions(self):
        # 0.4 m per frame in x at 25 fps = 10 m/s; edges and gaps are NaN
        ball = np.stack([np.arange(20) * 0.004, np.full(20, 0.5)], axis=1)
        ball[10] = np.nan
        speed = F.ball_speed_from_positions(ball.astype(np.float32), PITCH)
        np.testing.assert_allclose(speed[5], 10.0, rtol=1e-4)
        self.assertTrue(np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all())
        self.assertTrue(np.isnan(speed[[8, 12]]).all())  # one end of the window is the gap

    def test_missing_ball_is_nan_not_zero(self):
        feats = F.ball_features(make_match(30))
        self.assertTrue((feats["ball_present"] == 0).all())
        self.assertTrue(np.isnan(feats["nearest_home_m"]).all())
        self.assertTrue(np.isnan(feats["ball_out"]).all())

    def test_out_of_bounds_flag(self):
        ball = still_ball(4, 0.5, 0.5)
        ball[1] = (1.02, 0.5)
        ball[2] = (0.5, -0.01)
        ball[3] = (np.nan, np.nan)
        out = F.ball_features(make_match(4, ball))["ball_out"]
        np.testing.assert_array_equal(out[:3], [0, 1, 1])
        self.assertTrue(np.isnan(out[3]))

    def test_all_nan_rows_do_not_warn(self):
        match = make_match(200, still_ball(200, 0.5, 0.5))
        match.players_xy[:100] = np.nan  # halftime-like rows
        match.players_speed[:100] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            feats = F.compute(match)
        self.assertTrue(np.isnan(feats["nearest_home_m"][:100]).all())
        self.assertTrue(np.isnan(feats["all_mean_speed_smooth_m_s"][:50]).all())
        self.assertFalse(np.isnan(feats["all_mean_speed_smooth_m_s"][150]))

    def test_gap_features(self):
        present = np.array([1, 0, 0, 0, 1, 0], dtype=bool)
        gap = F.gap_features(present)
        np.testing.assert_array_equal(gap["frames_since_ball"], [0, 1, 2, 3, 0, 1])
        np.testing.assert_array_equal(gap["frames_until_ball"], [0, 3, 2, 1, 0, np.inf])


class ControlTests(unittest.TestCase):
    n = 100

    def balance(self, ball, home_xy=(0.5, 0.5), away_xy=(0.9, 0.9)):
        match = make_match(self.n, ball, home_xy=home_xy, away_xy=away_xy)
        return F.compute(match)["control_balance"]

    def test_sustained_proximity_is_control(self):
        bal = self.balance(still_ball(self.n, 0.505, 0.5))  # home player 0.5 m away
        self.assertGreater(bal[50], 0.9)

    def test_away_control_is_negative(self):
        bal = self.balance(still_ball(self.n, 0.905, 0.9))
        self.assertLess(bal[50], -0.9)

    def test_single_frame_touch_is_not_clear_control(self):
        ball = still_ball(self.n, 0.7, 0.5)  # far from everyone
        ball[50] = (0.505, 0.5)              # one frame next to the home player
        bal = self.balance(ball)
        self.assertLess(abs(bal[50]), F.CLEAR_CONTROL)

    def test_flying_ball_next_to_player_is_not_control(self):
        # ball crosses the home player at 20 m/s (0.8 m per frame): a pass, not control
        x = 0.30 + np.arange(self.n) * 0.008
        ball = np.stack([x, np.full(self.n, 0.5)], axis=1)
        bal = self.balance(ball, home_xy=(0.5, 0.5))
        self.assertEqual(float(np.abs(bal).max()), 0.0)

    def test_far_or_missing_ball_is_flat(self):
        self.assertEqual(float(np.abs(self.balance(still_ball(self.n, 0.7, 0.3))).max()), 0.0)
        self.assertEqual(float(np.abs(self.balance(None)).max()), 0.0)


class MemoryTests(unittest.TestCase):
    def test_last_and_next_clear_controller(self):
        bal = np.zeros(10, dtype=np.float32)
        bal[1] = 0.8    # home
        bal[7] = -0.6   # away
        mem = F.team_memory_features(bal)
        np.testing.assert_array_equal(mem["last_team_sign"], [0, 1, 1, 1, 1, 1, 1, -1, -1, -1])
        np.testing.assert_array_equal(mem["next_team_sign"], [1, 1, -1, -1, -1, -1, -1, -1, 0, 0])
        self.assertEqual(mem["frames_since_team"][0], np.inf)
        self.assertEqual(mem["frames_until_team"][9], np.inf)
        self.assertEqual(mem["frames_since_team"][5], 4)
        self.assertEqual(mem["frames_until_team"][5], 2)

    def test_weak_balance_is_not_clear(self):
        mem = F.team_memory_features(np.full(5, 0.1, dtype=np.float32))
        self.assertTrue((mem["last_team_sign"] == 0).all())
        self.assertTrue(np.isinf(mem["frames_since_team"]).all())


class StoppageAndTagTests(unittest.TestCase):
    def test_players_walking_gives_low_smoothed_speed(self):
        n = 200
        match = make_match(n, still_ball(n, 0.5, 0.5), speed=1.0)
        match.players_speed[100:] = 4.0
        feats = F.compute(match)
        self.assertLess(feats["all_mean_speed_smooth_m_s"][50], 1.1)
        self.assertGreater(feats["all_mean_speed_smooth_m_s"][150], 3.9)

    def test_tagging_set_piece_window_and_restart_bump(self):
        tags = [
            {"code": "set pieces", "start_frame": 100, "end_frame": 199, "team": None, "type": "throw-in"},
            {"code": "build up", "start_frame": 300, "end_frame": 350, "team": HOME, "type": ""},
            {"code": "set pieces", "start_frame": -20, "end_frame": -5, "team": AWAY, "type": "corner kick"},
        ]
        feats = F.tagging_features(tags, 400)
        self.assertEqual(feats["tag_set_piece"][99], 0)
        self.assertEqual(feats["tag_set_piece"][100], 1)
        self.assertEqual(feats["tag_set_piece"][199], 1)
        self.assertEqual(feats["tag_set_piece"][200], 0)
        self.assertEqual(feats["tag_set_piece"][300:351].sum(), 0)  # phase tags are not set pieces
        self.assertAlmostEqual(float(feats["restart_prior"][199]), 1.0, places=5)
        self.assertLess(float(feats["restart_prior"][199 + 4 * FPS]), 0.01)
        self.assertLess(float(feats["restart_prior"][300]), 0.01)

    def test_compute_has_tag_features_only_with_tags(self):
        match = make_match(30, still_ball(30, 0.5, 0.5))
        self.assertNotIn("restart_prior", F.compute(match))
        self.assertIn("restart_prior", F.compute(match, tags=[]))

    def test_compute_shapes_are_per_frame(self):
        n = 60
        feats = F.compute(make_match(n, still_ball(n, 0.5, 0.5)), tags=[])
        for name, values in feats.items():
            self.assertEqual(values.shape, (n,), name)
            self.assertEqual(values.dtype, np.float32, name)


if __name__ == "__main__":
    unittest.main()
