"""Published boundaries and independent half initialisation, using synthetic evidence."""
import unittest
import numpy as np
from games import GAME, half_video_blocks
from possession import D, X
from possession.structure import decode_halves, kickoff_frames


class StructureTests(unittest.TestCase):
    def test_boundaries_and_independent_halves(self):
        evidence = np.zeros((199319, 3))
        calls = []

        def decode(block):
            calls.append(len(block))
            return np.full(len(block), len(calls) - 1, dtype=np.int8)

        codes = decode_halves(evidence, decode)
        graded = np.zeros(len(codes), dtype=bool)
        for half, first, last in half_video_blocks(GAME):
            kick = max(first, kickoff_frames()[half])
            graded[first:last + 1] = True
            self.assertTrue(np.all(codes[first:kick] == D))
            self.assertTrue(np.all(codes[kick:last + 1] == half - 1))
            self.assertEqual(calls[half - 1], last - kick + 1)
        self.assertTrue(np.all(codes[~graded] == X))
        self.assertTrue(np.all(codes[110808:113742] == D))
        self.assertEqual(len(calls), 2)
        self.assertEqual(kickoff_frames()[1], 11185)
        self.assertEqual(kickoff_frames()[2] - 110808, 2934)
        self.assertEqual(codes[11185], 0)

    def test_two_argument_callback_receives_absolute_block_start(self):
        evidence = np.zeros((199319, 3))
        calls = []

        def decode(block, start):
            calls.append((start, len(block)))
            return np.zeros(len(block), dtype=np.int8)

        decode_halves(evidence, decode)
        self.assertEqual(calls, [(11185, 90651 - 11185 + 1),
                                 (113742, 190197 - 113742 + 1)])


class RunnerDecodeTests(unittest.TestCase):
    def test_restart_scale_only_discounts_dead_to_team(self):
        from run import restart_cost_scale
        prior = np.array([0., .5, 1., np.nan, -1., 2.])
        scale = restart_cost_scale(prior, .4)
        self.assertEqual(scale.shape, (6, 3, 3))
        expected = np.ones_like(scale)
        expected[:, 2, 0] = expected[:, 2, 1] = [1., .8, .6, 1., 1., .6]
        np.testing.assert_allclose(scale, expected)
        self.assertIsNone(restart_cost_scale(None, .5))
        self.assertIsNone(restart_cost_scale(prior, 0.))
        self.assertIsNone(restart_cost_scale(prior, -.1))

    def test_shared_decode_honours_duration_and_absolute_restart_slice(self):
        from run import decode_blocks
        from possession.decoder import decode, switch_matrix
        evidence = np.tile([0., -2., -3.], (20, 1))
        evidence[12:14] = [-5., 0., -3.]
        prior = np.linspace(0., 1., 20)
        config = {'decoder': {'switch_cost': {'team_switch': 2., 'to_dead': 2.,
                                             'from_dead': 2.},
                              'min_duration_frames': 2, 'restart_discount': .5}}
        from run import restart_cost_scale
        expected = np.full(20, X, dtype=np.int8)
        scale = restart_cost_scale(prior, .5)
        for first, last in [(2, 7), (10, 18)]:
            expected[first:last + 1] = decode(evidence[first:last + 1],
                switch_matrix(2., 2., 2.), cost_scale=scale[first:last + 1],
                min_duration_frames=2)
        actual = decode_blocks(evidence, [(2, 7), (10, 18)], config,
                               {'restart_prior': prior})
        np.testing.assert_array_equal(actual, expected)
