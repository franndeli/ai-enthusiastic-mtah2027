import unittest

import numpy as np

from possession.frames import A, D, H, X
from possession.postprocess import apply_rules


class ApplyRulesTest(unittest.TestCase):
    def test_dead_spans_and_outside_play(self):
        codes = np.array([H, H, A, A, H, A], np.int8)
        period = np.array([0, 1, 1, 2, 2, 2], np.int8)
        out = apply_rules(codes, period, [(3, 5)])
        np.testing.assert_array_equal(out, [X, H, A, D, D, A])
        np.testing.assert_array_equal(codes, [H, H, A, A, H, A])  # input untouched


if __name__ == "__main__":
    unittest.main()
