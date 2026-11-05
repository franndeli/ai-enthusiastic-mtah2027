"""Synthetic tests of the learned hand-over stage; never read the gated data."""
from __future__ import annotations

import unittest

import numpy as np

from possession import A, D, H, decoder, evidence, features, handover, stack, verifier
from possession.io import AWAY, HOME, Match


def synthetic_match(n: int = 1200, seed: int = 0) -> Match:
    rng = np.random.RandomState(seed)
    players = rng.uniform(0.1, 0.9, (1, 22, 2)) + np.cumsum(rng.normal(0, 0.0005, (n, 22, 2)), axis=0)
    ball = np.clip(0.5 + np.cumsum(rng.normal(0, 0.002, (n, 2)), axis=0), 0.02, 0.98).astype(np.float32)
    ball[100:160] = np.nan
    team = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
    players = players.astype(np.float32)
    players[300:320, 3] = np.nan                     # a track gap
    scale = np.array([105.0, 68.0], dtype=np.float32)
    speed = np.full((n, 22), 2.0, dtype=np.float32)
    return Match(players_xy=players, players_speed=speed, ball_xy=ball,
                 ball_speed=np.full(n, 3.0, dtype=np.float32), team_of_slot=team,
                 halves=[(1, 0, n // 2 - 1), (2, n // 2, n - 1)], pitch_m=tuple(float(v) for v in scale))


def swapped(match: Match) -> Match:
    """The same match with the two teams exchanged (and mirrored, so attack directions swap too)."""
    return Match(players_xy=np.concatenate([match.players_xy[:, 11:], match.players_xy[:, :11]], axis=1),
                 players_speed=np.concatenate([match.players_speed[:, 11:], match.players_speed[:, :11]], axis=1),
                 ball_xy=match.ball_xy, ball_speed=match.ball_speed, team_of_slot=match.team_of_slot,
                 halves=match.halves, pitch_m=match.pitch_m)


def pathcrf_outputs(n: int, sign: float = 1.0) -> dict:
    t = np.linspace(0, 12, n)
    p = 0.5 + sign * 0.4 * np.sin(t)
    return {"p_home": p, "p_next_home": 1.0 - (1.0 - p) * 0.9 if sign > 0 else p * 0.9,
            "prev_out": np.zeros(n), "next_out": np.zeros(n),
            "src_team": sign * np.sign(np.sin(t)), "dst_team": sign * np.sign(np.sin(t + 0.1)),
            "edge_loop": np.ones(n), "dst_out": np.zeros(n)}


def inputs(match: Match, seed: int = 0):
    feats = features.compute(match, None, None)
    params = dict(evidence.DEFAULT_PARAMS)
    rng = np.random.RandomState(seed)
    return feats, params, rng.normal(0, 1, match.n_frames), rng.normal(0, 1, match.n_frames)


class Windows(unittest.TestCase):
    def setUp(self):
        self.x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, np.nan, 8.0])

    def test_win_mean_ignores_nan_and_clips_at_the_ends(self):
        out = handover.win_mean(self.x, 0, 1)
        self.assertEqual(out[0], 1.5)
        self.assertEqual(out[1], 2.0)
        self.assertEqual(out[7], 8.0)
        self.assertTrue(np.isnan(handover.win_mean(self.x, -1, -1)[0]))
        self.assertTrue(np.isnan(handover.win_mean(self.x, 0, 0)[2]))

    def test_win_count_counts_inside_the_window_only(self):
        mask = np.array([True, False, True, True])
        np.testing.assert_array_equal(handover.win_count(mask, 0, 1), [1, 1, 2, 1])
        np.testing.assert_array_equal(handover.win_count(mask, -2, -1), [0, 1, 1, 1])

    def test_win_min_reads_exactly_its_window(self):
        x = np.array([5.0, 1.0, 7.0, 3.0, 9.0, 2.0, 8.0, 6.0])
        for lo, hi in ((0, 2), (-2, -1), (-1, 1), (0, 3), (-3, -2)):
            out = handover.win_min(x, lo, hi)
            for t in range(len(x)):
                # past the ends the nearest edge value stands in, as the filter's mode says
                want = min(x[int(np.clip(t + k, 0, len(x) - 1))] for k in range(lo, hi + 1))
                self.assertEqual(out[t], want, (lo, hi, t))

    def test_win_min_nan(self):
        x = np.array([np.nan, np.nan, 3.0, np.nan])
        out = handover.win_min(x, 0, 1)
        self.assertTrue(np.isnan(out[0]))
        self.assertEqual(out[1], 3.0)


class Memory(unittest.TestCase):
    def test_last_believed_hand_over_wins_and_fades(self):
        n = 100
        to_home, to_away = np.full(n, -5.0), np.full(n, -5.0)
        to_home[10], to_away[50] = 1.0, 1.0
        mem = handover.memory(to_home, to_away, -1.0, 10.0)
        self.assertEqual(mem[5], 0.0)            # nothing believed yet
        self.assertEqual(mem[10], 1.0)
        self.assertAlmostEqual(mem[20], np.exp(-1.0))
        self.assertEqual(mem[50], -1.0)
        self.assertLess(abs(mem[99]), abs(mem[60]))

    def test_below_threshold_is_ignored(self):
        mem = handover.memory(np.full(10, -2.0), np.full(10, -3.0), -1.0, 10.0)
        np.testing.assert_array_equal(mem, np.zeros(10))

    def test_symmetric_under_team_swap(self):
        rng = np.random.RandomState(0)
        a, b = rng.normal(-1, 2, 200), rng.normal(-1, 2, 200)
        np.testing.assert_allclose(handover.memory(a, b, -1.0, 50.0), -handover.memory(b, a, -1.0, 50.0))


class CostScale(unittest.TestCase):
    def test_only_team_changes_are_priced_and_clipped(self):
        to_home, to_away = np.array([3.0, -20.0]), np.array([-20.0, 0.0])
        scale = handover.cost_scale(to_home, to_away, 16.0, 8.0, 8.0, 1.0, 40.0)
        self.assertEqual(scale.shape, (2, 3, 3))
        self.assertEqual(scale[0, A, H], 1.0 / 16.0)     # 8 - 8 * 3 clipped up to lo
        self.assertEqual(scale[1, A, H], 40.0 / 16.0)    # clipped down to hi
        self.assertEqual(scale[1, H, A], 8.0 / 16.0)     # logit 0 costs a
        for frm, to in ((H, D), (A, D), (D, H), (D, A), (H, H), (A, A), (D, D)):
            np.testing.assert_array_equal(scale[:, frm, to], [1.0, 1.0])

    def test_a_believed_hand_over_is_taken_and_an_unbelieved_one_is_not(self):
        n = 300
        emis = np.zeros((n, 3))
        emis[:, D] = -5.0
        emis[:, H] = 0.2                 # slight home preference everywhere ...
        emis[150:, A] = 0.3              # ... away better from frame 150 on, by 15 nats in all
        cost = decoder.switch_matrix(16.0, 8.0, 8.0)
        np.testing.assert_array_equal(decoder.decode(emis, cost), np.full(n, H))   # not worth 16 nats
        to_home, to_away = np.full(n, -20.0), np.full(n, -20.0)
        to_away[150] = 1.0
        scale = handover.cost_scale(to_home, to_away, 16.0, 8.0, 8.0, 1.0, 40.0)
        path = decoder.decode(emis, cost, cost_scale=scale)
        self.assertEqual(path[149], H)
        self.assertEqual(path[150], A)


class Features(unittest.TestCase):
    def test_shape_names_and_no_inf(self):
        match = synthetic_match()
        feats, params, z_team, z_dead = inputs(match)
        X, names = handover.feature_matrix(match, feats, 1.0, z_team, z_dead, pathcrf_outputs(match.n_frames), params)
        self.assertEqual(X.shape, (match.n_frames, len(names)))
        self.assertEqual(len(set(names)), len(names))
        self.assertFalse(np.isinf(X).any())
        self.assertEqual(X.dtype, np.float32)

    def test_destination_team_features_swap_with_the_teams(self):
        """"to home" on a match equals "to away" on the same match with the teams exchanged."""
        match = synthetic_match()
        feats, params, z_team, z_dead = inputs(match)
        p = pathcrf_outputs(match.n_frames)
        X, names = handover.feature_matrix(match, feats, 1.0, z_team, z_dead, p, params)
        other = swapped(match)
        feats2 = features.compute(other, None, None)
        p2 = dict(p, p_home=1.0 - p["p_home"], p_next_home=1.0 - p["p_next_home"],
                  src_team=-p["src_team"], dst_team=-p["dst_team"])
        X2, names2 = handover.feature_matrix(other, feats2, -1.0, -z_team, z_dead, p2, params)
        self.assertEqual(names, names2)
        for column in ("near_new", "near_old", "gap", "vote", "owner", "z", "pcrf", "pcrf_next", "pcrf_src",
                       "new_within3", "old_within6", "gap_past25", "z_fut75", "new_touch_fut12", "pcrf_past25"):
            k = names.index(column)
            np.testing.assert_allclose(X[:, k], X2[:, k], atol=1e-3, err_msg=column)


class Training(unittest.TestCase):
    def test_table_marks_only_changes_to_that_team(self):
        match = synthetic_match()
        feats, params, z_team, z_dead = inputs(match)
        n = match.n_frames
        codes = np.full(n, H, dtype=np.int8)
        codes[200:300] = A          # one hand-over to away (200), one to home (300)
        codes[400:450] = D          # a stoppage: entering or leaving it is not a hand-over
        X, y, names = handover.training_table(match, feats, codes, z_team, z_dead, pathcrf_outputs(n), params, ("t", 1))
        self.assertEqual(X.shape, (len(y), len(names)))
        self.assertEqual(int(y.sum()), 2 * (2 * handover.TOLERANCE_FRAMES + 1))

    def test_table_is_deterministic_and_keyed(self):
        match = synthetic_match()
        feats, params, z_team, z_dead = inputs(match)
        codes = np.full(match.n_frames, H, dtype=np.int8)
        codes[200:] = A
        p = pathcrf_outputs(match.n_frames)
        a = handover.training_table(match, feats, codes, z_team, z_dead, p, params, ("x", 1))
        b = handover.training_table(match, feats, codes, z_team, z_dead, p, params, ("x", 1))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_fit_logits_and_layout_guard(self):
        match = synthetic_match()
        feats, params, z_team, z_dead = inputs(match)
        n = match.n_frames
        codes = np.full(n, H, dtype=np.int8)
        for t in range(50, n - 50, 100):
            codes[t:t + 50] = A
        p = pathcrf_outputs(n)
        table = handover.training_table(match, feats, codes, z_team, z_dead, p, params, ("x", 1))
        model = handover.fit([(("x", "m"), *table)])
        to_home, to_away = handover.logits(model, match, feats, z_team, z_dead, p, params)
        self.assertEqual(to_home.shape, (n,))
        self.assertTrue(np.isfinite(to_home).all() and np.isfinite(to_away).all())
        broken = handover.HandoverModel(model.classifier, ("other",))
        with self.assertRaises(ValueError):
            handover.logits(broken, match, feats, z_team, z_dead, p, params)


class Wiring(unittest.TestCase):
    def test_logits_compose_to_the_v6_evidence(self):
        match = synthetic_match()
        feats = features.compute(match, None, None)
        params = {"w_fresh_exit": 1.5}
        z_team, z_dead = stack.logits(feats, match, params)
        np.testing.assert_array_equal(evidence.combine(z_team, z_dead, dict(evidence.DEFAULT_PARAMS, **params)),
                                      evidence.scores(feats, params))

    def test_verifier_decode_carries_the_frame_scale(self):
        n = 300
        emis = np.zeros((n, 3))
        emis[:, D] = -5.0
        emis[:, H] = 0.2
        emis[150:, A] = 0.3
        config = {"switch_cost": {"team_switch": 16.0, "to_dead": 8.0, "from_dead": 8.0}}
        to_away = np.full(n, -20.0)
        to_away[150] = 1.0
        scale = handover.cost_scale(np.full(n, -20.0), to_away, 16.0, 8.0, 8.0, 1.0, 40.0)
        np.testing.assert_array_equal(verifier.viterbi_decode(config)(emis, 1.0), np.full(n, H))
        path = verifier.viterbi_decode(config, scale)(emis, 1.0)
        self.assertEqual(path[150], A)


if __name__ == "__main__":
    unittest.main()
