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
