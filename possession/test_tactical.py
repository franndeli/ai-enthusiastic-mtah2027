import unittest

import numpy as np

from possession.frames import AWAY, HOME, FrameTable
from possession.tactical import attack_direction, tactical_features

FPS = 25.0
# Home has the ball attacking +x: wide and pushed up; away compact and deep. Slot 0 of each is the GK.
HOME_ATTACKING = np.array([[-50, 0], [5, -25], [5, 25], [30, 0]], float)
AWAY_DEFENDING = np.array([[50, 0], [35, -5], [35, 5], [25, 0]], float)


def table(players, period, slot_team=None):
    n = len(period)
    team = np.array([HOME] * 4 + [AWAY] * 4, np.int8) if slot_team is None else slot_team
    return FrameTable(fps=FPS, pitch_m=(105.0, 68.0), period=np.asarray(period, np.int8),
                      ball_xy_m=np.full((n, 2), np.nan, np.float32),
                      players_xy_m=players.astype(np.float32), slot_team=team)


def formation(n, home=HOME_ATTACKING, away=AWAY_DEFENDING, drift_x_m_s=0.0):
    """Static home + away formation over n frames, all players drifting along x."""
    players = np.repeat(np.vstack([home, away])[None], n, axis=0)
    players[..., 0] += (np.arange(n) / FPS * drift_x_m_s)[:, None]
    return players


def mirrored(players):
    return players * [-1, 1]


def column(feats, name):
    return feats.values[:, feats.names.index(name)]


def noisy_table(seed=0, n=300):
    """Two halves with a period-0 gap, random motion and many missing slots."""
    rng = np.random.RandomState(seed)
    period = np.r_[np.ones(n // 2), np.zeros(10), np.full(n // 2, 2)]
    players = formation(len(period)) + rng.normal(0, 3, (len(period), 8, 2))
    players[len(period) // 2:] = mirrored(players[len(period) // 2:])
    players[rng.rand(len(period), 8) < 0.3] = np.nan
    players[50:60, :4] = np.nan  # the whole home team unseen
    players[:, 7] = np.nan       # a slot never seen
    return table(players, period)


class TacticalTest(unittest.TestCase):
    def test_finite_named_and_zero_outside_play(self):
        t = noisy_table()
        feats = tactical_features(t)
        self.assertEqual(feats.values.shape, (t.n_frames, len(feats.names)))
        self.assertEqual(len(feats.antisymmetric), len(feats.names))
        self.assertTrue(np.isfinite(feats.values).all())
        self.assertTrue((feats.values[t.period == 0] == 0).all())

    def test_team_swap_is_exactly_symmetric_or_antisymmetric(self):
        t = noisy_table()
        swapped = table(t.players_xy_m, t.period, slot_team=1 - t.slot_team)
        a, b = tactical_features(t), tactical_features(swapped)
        sign = np.where(a.antisymmetric, -1.0, 1.0).astype(np.float32)
        np.testing.assert_array_equal(b.values, a.values * sign)
        self.assertTrue(np.abs(a.values[:, a.antisymmetric]).sum(axis=0).all())  # not trivially 0

    def test_slot_order_does_not_matter(self):
        t = noisy_table()
        order = np.random.RandomState(1).permutation(8)
        shuffled = table(t.players_xy_m[:, order], t.period, slot_team=t.slot_team[order])
        np.testing.assert_allclose(tactical_features(shuffled).values, tactical_features(t).values,
                                   atol=1e-4)

    def test_direction_flips_across_halves_and_with_teams(self):
        players = np.vstack([formation(20), mirrored(formation(20))])
        t = table(players, [1] * 20 + [2] * 20)
        np.testing.assert_array_equal(attack_direction(t), [1] * 20 + [-1] * 20)
        swapped = table(players, t.period, slot_team=1 - t.slot_team)
        np.testing.assert_array_equal(attack_direction(swapped), [-1] * 20 + [1] * 20)
        self.assertEqual(attack_direction(table(players, [0] * 40)).tolist(), [0] * 40)

    def test_home_attacking_formation_reads_home_in_both_halves(self):
        n = 100
        players = np.vstack([formation(n, drift_x_m_s=2.0), mirrored(formation(n, drift_x_m_s=2.0))])
        feats = tactical_features(table(players, [1] * n + [2] * n))
        inner = np.r_[20:80, n + 20:n + 80]  # clear of the velocity window at the period edges
        for name in ("tac_centroid_diff", "tac_line_diff", "tac_length_diff", "tac_width_diff",
                     "tac_opp_half_diff", "tac_push_m_s", "tac_push_m_s_roll5.0s"):
            self.assertTrue((column(feats, name)[inner] > 0).all(), name)
        np.testing.assert_allclose(column(feats, "tac_push_m_s")[inner], 2.0, rtol=1e-4)
        # Orientation invariance: the mirrored second half reads the same as the first.
        np.testing.assert_allclose(feats.values[20:80], feats.values[n + 20:n + 80], atol=1e-4)

    def test_windows_do_not_cross_the_period_boundary(self):
        n = 100
        # Same sides in both halves, roles reversed: the jump at the boundary is not a movement.
        second = formation(n, home=AWAY_DEFENDING * [-1, 1], away=HOME_ATTACKING * [-1, 1])
        t = table(np.vstack([formation(n), second]), [1] * n + [2] * n)
        feats = tactical_features(t)
        self.assertTrue((column(feats, "tac_push_m_s") == 0).all())
        for s in (2.0, 5.0):
            np.testing.assert_allclose(column(feats, f"tac_centroid_diff_roll{s}s"),
                                       column(feats, "tac_centroid_diff"), atol=1e-4)
        self.assertTrue((column(feats, "tac_width_diff")[:n] > 0).all())
        self.assertTrue((column(feats, "tac_width_diff")[n:] < 0).all())


if __name__ == "__main__":
    unittest.main()
