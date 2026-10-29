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


class TouchTests(unittest.TestCase):
    """Ownership = last team to play the ball; votes need sustained, unambiguous proximity."""

    def feats(self, home_m, away_m):
        n = len(home_m)
        return {"nearest_home_m": np.asarray(home_m, np.float32),
                "nearest_away_m": np.asarray(away_m, np.float32),
                "ball_present": (~np.isnan(np.asarray(home_m, np.float32))).astype(np.float32)}

    def test_owner_switches_at_opponents_touch_and_persists_through_flight(self):
        far = np.full(60, 20.0)
        home = far.copy()
        home[:10] = 0.5                  # home player plays it ...
        away = far.copy()
        away[40:50] = 0.5                # ... away player receives it 1.2 s later
        out = F.touch_features(self.feats(home, away))
        self.assertTrue((out["touch_vote"][:10] == 1).all())
        self.assertTrue((out["touch_vote"][10:40] == 0).all())
        self.assertTrue((out["owner_sign"][:40] == 1).all())     # flight: still home
        self.assertTrue((out["owner_sign"][40:] == -1).all())
        self.assertEqual(out["owner_age"][39], 30)
        self.assertEqual(out["owner_age"][45], 0)

    def test_contested_ball_and_flash_proximity_do_not_vote(self):
        home = np.full(30, 0.8)
        away = np.full(30, 1.0)          # both within reach, no margin
        out = F.touch_features(self.feats(home, away))
        self.assertTrue((out["touch_vote"] == 0).all())
        home = np.full(30, 20.0)
        home[10:12] = 0.5                # a ball flying past: 2 frames only
        out = F.touch_features(self.feats(home, np.full(30, 20.0)), min_frames=3)
        self.assertTrue((out["touch_vote"] == 0).all())
        self.assertTrue((out["owner_sign"] == 0).all())
        self.assertTrue(np.isinf(out["owner_age"]).all())

    def test_missing_ball_keeps_the_owner(self):
        home = np.full(20, np.nan)
        home[:5] = 0.3
        out = F.touch_features(self.feats(home, np.full(20, np.nan)))
        self.assertTrue((out["owner_sign"] == 1).all())
        self.assertEqual(out["owner_age"][19], 15)


class PassByTests(unittest.TestCase):
    """A ball flying past a player at constant velocity is not a touch; a kick changes velocity."""

    def test_velocity_change_is_zero_on_straight_flight_and_positive_at_a_kick(self):
        n = 60
        ball = np.stack([np.arange(n) * 0.004, np.full(n, 0.5)], axis=1).astype(np.float32)  # 10 m/s
        change = F.ball_velocity_change(ball, PITCH)
        self.assertLess(float(np.nanmax(change[10:50])), 1e-3)
        kicked = ball.copy()
        kicked[30:, 1] = 0.5 + np.arange(n - 30) * 0.004                # +10 m/s sideways from frame 30
        change = F.ball_velocity_change(kicked, PITCH)
        self.assertGreater(float(np.nanmax(change[25:36])), 4.9)
        self.assertLess(float(np.nanmax(change[5:20])), 1e-3)
        ball[20] = np.nan
        self.assertTrue(np.isnan(F.ball_velocity_change(ball, PITCH)[20]))

    def test_pass_by_run_is_dropped_but_a_stopping_touch_is_kept(self):
        n = 80
        home = np.full(n, 20.0, np.float32)
        home[30:38] = 0.5                                            # within reach for 8 frames
        away = np.full(n, 20.0, np.float32)
        fast = np.stack([np.arange(n) * 0.004, np.full(n, 0.5)], axis=1).astype(np.float32)
        feats = {"nearest_home_m": home, "nearest_away_m": away,
                 "ball_present": np.ones(n, np.float32),
                 "ball_speed_m_s": F.ball_speed_from_positions(fast, PITCH)}
        plain = F.touch_features(feats)
        self.assertTrue((plain["touch_vote"][30:38] == 1).all())
        filtered = F.touch_features(feats, passby_dv_m_s=1.5, ball_xy=fast, pitch_m=PITCH)
        self.assertTrue((filtered["touch_vote"] == 0).all())         # flew past untouched
        stopped = fast.copy()
        stopped[34:] = stopped[34]                                   # the player stops it
        feats["ball_speed_m_s"] = F.ball_speed_from_positions(stopped, PITCH)
        kept = F.touch_features(feats, passby_dv_m_s=1.5, ball_xy=stopped, pitch_m=PITCH)
        self.assertTrue((kept["touch_vote"][30:38] == 1).all())
        with self.assertRaises(ValueError):
            F.touch_features(feats, passby_dv_m_s=1.5)


class RestartFeatureTests(unittest.TestCase):
    def test_standing_player_beyond_the_line_is_a_restart_clue(self):
        n = 50
        match = make_match(n, still_ball(n, 0.5, 0.5), home_xy=(0.5, -0.02), speed=0.3)  # 1 m out
        out = F.restart_features(match, F.ball_features(match))
        self.assertGreater(float(out["out_player_frac"][25]), 0.9)
        match.players_speed[:] = 3.0                                 # running past the line: no
        out = F.restart_features(match, F.ball_features(match))
        self.assertEqual(float(out["out_player_frac"].max()), 0.0)
        match = make_match(n, still_ball(n, 0.5, 0.5), home_xy=(0.5, 0.004), speed=0.3)   # inside
        out = F.restart_features(match, F.ball_features(match))
        self.assertEqual(float(out["out_player_frac"].max()), 0.0)

    def test_slow_far_ball_needs_nobody_near_and_a_seen_slow_ball(self):
        n = 50
        far = make_match(n, still_ball(n, 0.3, 0.3))                 # players 20+ m away
        out = F.restart_features(far, F.ball_features(far))
        self.assertGreater(float(out["ball_slow_far_frac"][25]), 0.9)
        near = make_match(n, still_ball(n, 0.3, 0.3), home_xy=(0.31, 0.3))  # 1 m away
        out = F.restart_features(near, F.ball_features(near))
        self.assertEqual(float(out["ball_slow_far_frac"].max()), 0.0)
        unseen = make_match(n)
        out = F.restart_features(unseen, F.ball_features(unseen))
        self.assertEqual(float(out["ball_slow_far_frac"].max()), 0.0)
        self.assertEqual(float(out["ball_edge_slow_frac"].max()), 0.0)


class BallMotionTests(unittest.TestCase):
    def test_moving_share_counts_only_seen_fast_frames(self):
        speed = np.full(100, np.nan, np.float32)
        speed[40:60] = 8.0               # in flight for 0.8 s
        speed[60:80] = 1.0               # then slow
        out = F.ball_motion_features(speed)["ball_moving_frac"]
        self.assertAlmostEqual(float(out[50]), 20 / 25, places=5)
        self.assertEqual(float(out[0]), 0.0)
        self.assertLess(float(out[75]), 0.2)


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


def two_team_match(n=400, flip=False, home_len=40.0, away_len=25.0, halves=None):
    """11 v 11 with goalkeepers near their goal lines; HOME defends left unless flip.
    Outfield players span `*_len` metres along x (100 m pitch), width 30 m."""
    rng = np.random.RandomState(0)
    players = np.zeros((n, 22, 2), np.float32)
    for team, length in ((0, home_len), (1, away_len)):
        x = 0.05 + 0.9 * 0 + np.linspace(0.12, 0.12 + length / 100.0, 11)
        x[0] = 0.04                                    # goalkeeper
        y = np.linspace(0.2, 0.8, 11)
        xs = (1 - x) if team == 1 else x               # AWAY defends the right
        if flip:
            xs = 1 - xs
        players[:, team * 11:(team + 1) * 11, 0] = xs
        players[:, team * 11:(team + 1) * 11, 1] = y + rng.uniform(-0.01, 0.01, (n, 11))
    slots = np.array([HOME] * 11 + [AWAY] * 11, np.int8)
    return Match(players_xy=players, players_speed=np.full((n, 22), 2.0, np.float32),
                 ball_xy=np.full((n, 2), np.nan, np.float32),
                 ball_speed=np.full(n, np.nan, np.float32), team_of_slot=slots,
                 halves=halves or [(1, 0, n - 1)], pitch_m=PITCH)


class ShapeTests(unittest.TestCase):
    def test_attack_direction_from_goalkeepers(self):
        np.testing.assert_array_equal(np.unique(F.attack_direction(two_team_match())), [1.0])
        np.testing.assert_array_equal(np.unique(F.attack_direction(two_team_match(flip=True))), [-1.0])

    def test_attack_direction_flips_between_halves_and_is_zero_outside(self):
        match = two_team_match(halves=[(1, 0, 99), (2, 200, 299)])
        match.players_xy[200:, :, 0] = 1 - match.players_xy[200:, :, 0]  # sides swap at half-time
        direction = F.attack_direction(match)
        self.assertEqual(float(direction[50]), 1.0)
        self.assertEqual(float(direction[250]), -1.0)
        self.assertEqual(float(direction[150]), 0.0)

    def test_disagreeing_goalkeepers_give_no_direction(self):
        match = two_team_match()
        match.players_xy[:, 11:, 0] = match.players_xy[:, :11, 0]  # both teams "defend" left
        self.assertTrue((F.attack_direction(match) == 0).all())

    def test_longer_team_favours_itself_and_mirror_gives_same_z(self):
        # the home team is longer for the first half of the recording, the away team after
        n = 400
        a = two_team_match(n, home_len=45.0, away_len=25.0)
        b = two_team_match(n, home_len=25.0, away_len=45.0)
        a.players_xy[n // 2:] = b.players_xy[n // 2:]
        z = F.ball_free_shape_features(a)["shape_len_z"]
        self.assertGreater(float(z[n // 4]), 0.5)
        self.assertLess(float(z[3 * n // 4]), -0.5)
        mirror = two_team_match(n, flip=True, home_len=45.0, away_len=25.0)
        mirror.players_xy[n // 2:] = two_team_match(n, flip=True, home_len=25.0,
                                                    away_len=45.0).players_xy[n // 2:]
        zm = F.ball_free_shape_features(mirror)["shape_len_z"]
        np.testing.assert_allclose(z, zm, atol=1e-5)  # nothing depends on the attacking side

    def test_constant_style_is_removed(self):
        z = F.ball_free_shape_features(two_team_match())  # home always longer: no information
        for name in ("shape_len_z", "shape_speed_z"):  # exactly constant in this fixture
            values = z[name]
            self.assertTrue(np.isnan(values).all() or np.nanmax(np.abs(values)) < 1e-3, name)

    def test_row_percentiles_ignore_nan(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [np.nan, np.nan, np.nan, np.nan, np.nan],
                           [7.0, np.nan, 9.0, np.nan, np.nan]])
        low, high = F._row_percentiles(values, (0, 100))
        np.testing.assert_array_equal(low[[0, 2]], [1.0, 7.0])
        np.testing.assert_array_equal(high[[0, 2]], [5.0, 9.0])
        self.assertTrue(np.isnan(low[1]) and np.isnan(high[1]))


class RestartTeamTests(unittest.TestCase):
    def test_taker_owns_after_window_throw_in_has_no_team(self):
        tags = [
            {"code": "set pieces", "start_frame": 100, "end_frame": 199, "team": HOME, "type": "corner kick"},
            {"code": "set pieces", "start_frame": 500, "end_frame": 599, "team": None, "type": "throw-in"},
            {"code": "set pieces", "start_frame": 800, "end_frame": 899, "team": AWAY, "type": "goal kick"},
        ]
        t = F.tagging_features(tags, 1200)["restart_team"]
        self.assertEqual(float(t[150]), 0.0)                       # before the final second
        self.assertEqual(float(t[199]), 1.0)
        self.assertEqual(float(t[199 + 4 * FPS]), 1.0)
        self.assertEqual(float(t[199 + 4 * FPS + 1]), 0.0)
        self.assertEqual(float(np.abs(t[500:700]).max()), 0.0)
        self.assertEqual(float(t[900]), -1.0)


class DeadOnsetTests(unittest.TestCase):
    def test_speed_lead_moves_onset_earlier_not_the_end(self):
        n = 600
        match = make_match(n, still_ball(n, 0.5, 0.5), speed=4.0)
        match.players_speed[200:400] = 0.5          # a stoppage, 200..399
        feats = F.compute(match)
        lead, smooth = feats["all_mean_speed_lead_m_s"], feats["all_mean_speed_smooth_m_s"]
        self.assertGreater(float(smooth[190]), 2.5)
        self.assertLess(float(lead[190]), 1.0)                 # slow already visible ahead
        self.assertLess(float(lead[300]), 1.0)
        self.assertGreater(float(lead[430]), 3.0)              # no extension after the restart
        self.assertTrue(np.isnan(feats["all_mean_speed_lead_m_s"][-1]) or lead[-1] <= smooth[-1] + 1e-6)

    def test_exit_ball_seen_outside_stays_exited_through_the_gap(self):
        ball = np.full((200, 2), np.nan, np.float32)
        ball[:50] = (0.5, 0.5)
        ball[50:60] = (1.02, 0.5)                   # seen beyond the line
        ball[150:] = (0.6, 0.5)                     # back inside after a gap
        exited = F.ball_exit_features(ball)["ball_exited"]
        self.assertEqual(float(exited[10]), 0.0)
        self.assertEqual(float(exited[55]), 1.0)
        self.assertEqual(float(exited[100]), 1.0)   # unseen, last seen out
        self.assertEqual(float(exited[160]), 0.0)   # seen inside again

    def test_exit_projects_a_ball_lost_while_heading_out(self):
        ball = np.full((200, 2), np.nan, np.float32)
        ball[:50, 0] = np.linspace(0.80, 0.98, 50)   # 0.0037 per frame towards the right line
        ball[:50, 1] = 0.5
        ball[120:] = (0.5, 0.5)
        exited = F.ball_exit_features(ball)["ball_exited"]
        self.assertEqual(float(exited[40]), 0.0)     # still inside and seen
        self.assertEqual(float(exited[90]), 1.0)     # unseen; 40 frames of projection pass x = 1
        self.assertEqual(float(exited[130]), 0.0)

    def test_exit_ball_lost_while_still_is_not_exited(self):
        ball = np.full((200, 2), np.nan, np.float32)
        ball[:50] = (0.5, 0.5)
        exited = F.ball_exit_features(ball)["ball_exited"]
        self.assertEqual(float(exited[150]), 0.0)

    def test_exit_before_first_sighting_is_nan(self):
        ball = np.full((20, 2), np.nan, np.float32)
        ball[10:] = (0.5, 0.5)
        self.assertTrue(np.isnan(F.ball_exit_features(ball)["ball_exited"][:10]).all())


if __name__ == "__main__":
    unittest.main()
