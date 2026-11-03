"""Synthetic checks of the learned dead residual, the fresh-exit clue and the evidence stack."""
import unittest

import numpy as np

from possession import evidence, stack
from possession import dead_residual as DR
from possession import features as F
from possession.external import inject_matched_noise
from possession.io import Match


def synthetic_match(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)[:, None]
    base = rng.uniform(0.1, 0.9, (1, 22, 2))
    xy = (base + 0.02 * np.sin(t[:, :, None] / (20 + np.arange(22)[None, :, None]))).astype(np.float32)
    speed = np.abs(rng.normal(1.5, 0.5, (n, 22))).astype(np.float32)
    ball = (0.5 + 0.3 * np.stack([np.sin(t[:, 0] / 50), np.cos(t[:, 0] / 70)], 1)).astype(np.float32)
    return Match(players_xy=xy, players_speed=speed, ball_xy=ball, ball_speed=np.full(n, np.nan, np.float32),
                 team_of_slot=np.array([0] * 11 + [1] * 11, np.int8), halves=[(1, 0, n - 1)],
                 pitch_m=(105.0, 68.0))


class FreshExitTest(unittest.TestCase):
    def test_only_lasting_exits_and_only_their_first_seconds(self):
        exited = np.zeros(1000, np.float32)
        exited[100:130] = 1          # 1.2 s outside: skims the line, no stoppage
        exited[400:800] = 1          # 16 s outside: a stoppage
        fresh = F.fresh_exit_features(exited)["fresh_exit"]
        self.assertEqual(fresh[100:130].sum(), 0)
        self.assertTrue(np.all(fresh[400:400 + F.FRESH_EXIT_FRAMES + 1] == 1))
        self.assertEqual(fresh[400 + F.FRESH_EXIT_FRAMES + 1:].sum(), 0)

    def test_nan_before_first_sighting_is_not_an_exit(self):
        exited = np.full(300, np.nan, np.float32)
        self.assertEqual(F.fresh_exit_features(exited)["fresh_exit"].sum(), 0)

    def test_weight_zero_leaves_dead_logit_unchanged(self):
        feats = F.compute(synthetic_match())
        p = dict(evidence.DEFAULT_PARAMS)
        feats["fresh_exit"] = np.ones_like(feats["fresh_exit"])
        base = evidence.dead_logit(feats, dict(p, w_fresh_exit=0.0))
        np.testing.assert_allclose(evidence.dead_logit(feats, dict(p, w_fresh_exit=1.5)), base + 1.5)


class DeadResidualTest(unittest.TestCase):
    def test_fit_is_deterministic_and_residual_is_clipped(self):
        samples = []
        for k in range(4):
            m = synthetic_match(1200, k)
            feats = F.compute(m)
            dz = evidence.dead_logit(feats, evidence.DEFAULT_PARAMS)
            codes = np.where(np.arange(1200) % 300 < 100, 2, 0).astype(np.int8)
            samples.append(DR.TrainingRow(("synthetic" if k < 2 else "other", f"m{k}"), m, feats, dz, codes, None))
        a, b = DR.fit(samples), DR.fit(samples)
        m, feats, dz = samples[0][1], samples[0][2], samples[0][3]
        ra, rb = DR.residual(a, m, feats, dz), DR.residual(b, m, feats, dz)
        np.testing.assert_array_equal(ra, rb)
        self.assertLessEqual(float(np.abs(ra).max()), DR.RESIDUAL_CLIP)
        self.assertEqual(DR.feature_matrix(m, feats, dz).shape, (1200, len(DR.COLUMNS)))


class StackTest(unittest.TestCase):
    def test_without_models_and_weights_the_stack_is_v5(self):
        feats = F.compute(synthetic_match())
        np.testing.assert_array_equal(
            stack.log_evidence(feats, synthetic_match(), {}), evidence.scores(feats, {}))

    def test_fresh_exit_moves_only_the_dead_column(self):
        m = synthetic_match()
        feats = F.compute(m)
        feats["fresh_exit"] = np.ones_like(feats["fresh_exit"])
        base, ev = stack.log_evidence(feats, m, {}), stack.log_evidence(feats, m, {"w_fresh_exit": 1.5})
        np.testing.assert_allclose(ev[:, 0] - ev[:, 1], base[:, 0] - base[:, 1], atol=1e-5)
        self.assertTrue((ev[:, 2] >= base[:, 2]).all())


class MatchedNoiseTest(unittest.TestCase):
    def test_deterministic_smooth_ball_and_player_dropout(self):
        m = synthetic_match(3000)
        a, b = inject_matched_noise(m, 0, 0.4), inject_matched_noise(m, 0, 0.4)
        np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
        self.assertGreater(np.isnan(a.ball_xy[:, 0]).mean(), 0.2)
        seen = ~np.isnan(a.players_xy[:, :, 0])
        self.assertLess(seen.sum(axis=1).mean(), 22)
        ball = a.ball_xy * np.array(m.pitch_m)
        acc = np.linalg.norm(ball[2:] - 2 * ball[1:-1] + ball[:-2], axis=1)
        self.assertLess(np.nanmedian(acc), 0.05)   # smooth, unlike iid 0.3 m jitter


if __name__ == "__main__":
    unittest.main()
