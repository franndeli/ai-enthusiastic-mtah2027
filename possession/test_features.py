import unittest

import numpy as np

from possession.features import features, last_seen, next_seen, rolling_mean
from possession.frames import AWAY, HOME, FrameTable


def scene(n=100, fps=25.0, period=None):
    """Home player at the ball moving along x at 2 m/s, away player 10 m away. Ball always seen."""
    t = np.arange(n) / fps
    ball = np.stack([t * 2.0, np.zeros(n)], axis=1)
    players = np.stack([ball + [0.5, 0.0], ball + [0.0, 10.0]], axis=1)
    return FrameTable(fps=fps, pitch_m=(105.0, 68.0),
                      period=np.ones(n, np.int8) if period is None else period,
                      ball_xy_m=ball.astype(np.float32), players_xy_m=players.astype(np.float32),
                      slot_team=np.array([HOME, AWAY], np.int8))


def column(feats, name):
    return feats.values[:, feats.names.index(name)]


def replace(table, **changes):
    fields = dict(fps=table.fps, pitch_m=table.pitch_m, period=table.period,
                  ball_xy_m=table.ball_xy_m, players_xy_m=table.players_xy_m,
                  slot_team=table.slot_team, labels=table.labels)
    fields.update(changes)
    return FrameTable(**fields)


class HelperTest(unittest.TestCase):
    def test_rolling_mean_ignores_nan_and_stays_inside_runs(self):
        values = np.array([1.0, np.nan, 3.0, 100.0, 100.0])
        out = rolling_mean(values, 1, [(0, 3), (3, 5)])
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 100.0, 100.0])

    def test_rolling_mean_all_missing_is_nan(self):
        self.assertTrue(np.isnan(rolling_mean(np.full(4, np.nan), 2, [(0, 4)])).all())

    def test_last_and_next_seen(self):
        values = np.array([np.nan, 1.0, np.nan, np.nan, -2.0, np.nan])
        value, age = last_seen(values, [(0, 6)])
        np.testing.assert_array_equal(age, [np.inf, 0, 1, 2, 0, 1])
        np.testing.assert_array_equal(value[1:], [1, 1, 1, -2, -2])
        value, age = next_seen(values, [(0, 6)])
        np.testing.assert_array_equal(age, [1, 0, 2, 1, 0, np.inf])
        np.testing.assert_array_equal(value[:5], [1, 1, -2, -2, -2])

    def test_last_seen_does_not_cross_runs(self):
        _, age = last_seen(np.array([1.0, np.nan, np.nan]), [(0, 1), (1, 3)])
        self.assertTrue(np.isinf(age[1:]).all())


class FeaturesTest(unittest.TestCase):
    def test_finite_and_named(self):
        feats = features(scene())
        self.assertEqual(feats.values.shape, (100, len(feats.names)))
        self.assertTrue(np.isfinite(feats.values).all())

    def test_home_at_ball_gives_negative_distance_difference(self):
        diff = column(features(scene()), "diff_m")
        np.testing.assert_allclose(diff, -9.5, atol=1e-4)

    def test_team_swap_flips_antisymmetric_and_keeps_symmetric(self):
        table = scene()
        swapped = replace(table, slot_team=1 - table.slot_team)
        a, b = features(table), features(swapped)
        anti = a.antisymmetric
        np.testing.assert_allclose(a.values[:, anti], -b.values[:, anti], atol=1e-5)
        np.testing.assert_allclose(a.values[:, ~anti], b.values[:, ~anti], atol=1e-5)

    def test_identity_switch_between_slots_changes_nothing(self):
        table = scene()
        players = np.concatenate([table.players_xy_m, table.players_xy_m + [0, 20.0]], axis=1)
        slots = np.array([HOME, AWAY, HOME, AWAY], np.int8)
        before = replace(table, players_xy_m=players.copy(), slot_team=slots)
        swapped = players.copy()
        swapped[50:, [0, 2]] = swapped[50:, [2, 0]]  # two home tracklets swap identity mid-match
        after = replace(table, players_xy_m=swapped, slot_team=slots)
        np.testing.assert_allclose(features(before).values, features(after).values, atol=1e-5)

    def test_missing_ball_is_flagged_and_gap_timers_count(self):
        table = scene()
        ball = table.ball_xy_m.copy()
        ball[40:60] = np.nan
        feats = features(replace(table, ball_xy_m=ball))
        self.assertEqual(column(feats, "ball_seen")[40:60].sum(), 0)
        self.assertEqual(column(feats, "diff_m")[50], 0)
        self.assertEqual(column(feats, "last_sign")[50], -1)  # home still last seen at the ball
        self.assertGreater(column(feats, "since_seen")[59], column(feats, "since_seen")[41])
        self.assertTrue(np.isfinite(feats.values).all())

    def test_track_gaps_fall_back_to_remaining_players(self):
        table = scene()
        players = table.players_xy_m.copy()
        players[10:20, 0] = np.nan  # the home player drops out of tracking
        feats = features(replace(table, players_xy_m=players))
        self.assertEqual(column(feats, "diff_m")[15], 0)  # no home distance -> no difference
        self.assertTrue(np.isfinite(feats.values).all())

    def test_ball_out_of_play_flags_oob(self):
        table = scene()
        ball = table.ball_xy_m.copy()
        ball[30:40] = [0.0, 40.0]  # beyond the touchline
        oob = column(features(replace(table, ball_xy_m=ball)), "ball_oob")
        self.assertTrue((oob[30:40] == 1).all())
        self.assertEqual(oob[10], 0)

    def test_windows_do_not_cross_half_boundary(self):
        period = np.array([1] * 50 + [0] + [2] * 49, np.int8)
        table = scene(period=period)
        players = table.players_xy_m.copy()
        players[51:] = players[51:, ::-1]  # away at the ball in the second half
        feats = features(replace(table, players_xy_m=players))
        roll = column(feats, "diff_m_roll5.0s")
        self.assertTrue((roll[:50] < 0).all() and (roll[51:] > 0).all())
        self.assertTrue((feats.values[50] == 0).all())

    def test_single_frame_flicker_is_damped_by_windows(self):
        table = scene()
        players = table.players_xy_m.copy()
        players[50] = players[50, ::-1]
        feats = features(replace(table, players_xy_m=players))
        self.assertGreater(column(feats, "diff_m")[50], 0)
        self.assertLess(column(feats, "diff_m_roll2.0s")[50], 0)

    def test_speeds_are_frame_rate_invariant(self):
        fast = column(features(scene(n=250, fps=25.0)), "ball_speed_m_s")
        slow = column(features(scene(n=100, fps=10.0)), "ball_speed_m_s")
        self.assertAlmostEqual(float(fast[125]), 2.0, places=3)
        self.assertAlmostEqual(float(slow[50]), 2.0, places=3)


if __name__ == "__main__":
    unittest.main()
