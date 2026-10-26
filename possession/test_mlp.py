import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession.frames import A, D, H, X
from possession.mlp import MLPModel, fit, train_mlp
from possession.model import decode, fit_softmax, predict_log_proba, Model

FAST = dict(hidden=(16,), epochs=150, batch=128, lr=1e-2)


def xor(n=2000, seed=0):
    """H where the two features share a sign, A where they differ, D near the axes: not linear."""
    rng = np.random.RandomState(seed)
    values = rng.uniform(-1, 1, (n, 2))
    labels = np.where(values[:, 0] * values[:, 1] > 0, H, A)
    labels[np.abs(values).min(axis=1) < 0.15] = D
    return values, labels.astype(np.int8)


class MLPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values, cls.labels = xor()
        cls.model = fit(cls.values, cls.labels, [cls.labels], emission_scale=0.5, **FAST)

    def test_learns_xor_that_softmax_cannot(self):
        test_values, test_labels = xor(seed=1)
        mlp_acc = (self.model.log_proba(test_values).argmax(axis=1) == test_labels).mean()
        mean, scale, weights = fit_softmax(self.values, self.labels)
        softmax = Model(mean, scale, weights, np.zeros(3), np.zeros((3, 3)), 1.0)
        softmax_acc = (predict_log_proba(softmax, test_values).argmax(axis=1) == test_labels).mean()
        self.assertGreater(mlp_acc, 0.9)
        self.assertLess(softmax_acc, 0.6)

    def test_posteriors_normalised(self):
        np.testing.assert_allclose(np.exp(self.model.log_proba(self.values)).sum(axis=1), 1.0)

    def test_deterministic(self):
        again = fit(self.values, self.labels, [self.labels], emission_scale=0.5, **FAST)
        for a, b in zip(self.model.weights + self.model.biases, again.weights + again.biases):
            np.testing.assert_array_equal(a, b)
        other_seed = fit(self.values, self.labels, [self.labels], emission_scale=0.5,
                         seed=1, **FAST)
        self.assertFalse(np.array_equal(self.model.weights[0], other_seed.weights[0]))

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.model.save(Path(tmp) / "m.npz")
            loaded = MLPModel.load(Path(tmp) / "m.npz")
        np.testing.assert_array_equal(loaded.log_proba(self.values), self.model.log_proba(self.values))
        np.testing.assert_array_equal(loaded.log_trans, self.model.log_trans)
        self.assertEqual(len(loaded.weights), 2)
        self.assertEqual(loaded.emission_scale, 0.5)

    def test_train_mlp_plugs_into_decode(self):
        half = len(self.values) // 2
        prepared = [dict(train_x=self.values[s], train_y=self.labels[s], runs=[self.labels[s]])
                    for s in (slice(None, half), slice(half, None))]
        model = train_mlp(prepared, emission_scale=1.0, **FAST)
        period = np.array([1] * 999 + [0] + [2] * 1000, np.int8)
        codes = decode(model.with_emission_scale(0.2), self.values, period)
        self.assertEqual(codes[999], X)
        self.assertTrue(np.isin(codes[period > 0], (H, A, D)).all())
        self.assertEqual(model.with_emission_scale(0.2).emission_scale, 0.2)


if __name__ == "__main__":
    unittest.main()
