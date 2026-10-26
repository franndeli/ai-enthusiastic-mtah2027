import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession.frames import A, D, H, X
from possession.model import (Model, argmax_decode, decode, fit, fit_softmax,
                              predict_log_proba, transition_log_probs, viterbi)


def blobs(n=600, seed=0):
    """Three separable classes along one feature, plus a noise feature and a constant one."""
    rng = np.random.RandomState(seed)
    labels = np.repeat([H, A, D], n // 3).astype(np.int8)
    centre = np.array([-2.0, 2.0, 0.0])[labels]
    values = np.stack([centre + rng.normal(0, 0.5, n), rng.normal(0, 1, n), np.ones(n)], axis=1)
    return values, labels


class SoftmaxTest(unittest.TestCase):
    def test_learns_separable_classes(self):
        values, labels = blobs()
        mean, scale, weights = fit_softmax(values, labels)
        model = Model(mean, scale, weights, np.zeros(3), np.zeros((3, 3)), 1.0)
        accuracy = (predict_log_proba(model, values).argmax(axis=1) == labels).mean()
        self.assertGreater(accuracy, 0.95)
        np.testing.assert_array_equal(weights[:, D], 0)  # reference class

    def test_deterministic(self):
        values, labels = blobs()
        np.testing.assert_array_equal(fit_softmax(values, labels)[2], fit_softmax(values, labels)[2])

    def test_posteriors_normalised(self):
        values, labels = blobs()
        model = fit(values, labels, [labels], emission_scale=1.0)
        np.testing.assert_allclose(np.exp(predict_log_proba(model, values)).sum(axis=1), 1.0)


class ViterbiTest(unittest.TestCase):
    def test_sticky_transitions_remove_single_frame_flicker(self):
        emission = np.log(np.full((20, 3), 0.05))
        emission[:, H] = np.log(0.9)
        emission[10] = np.log([0.05, 0.9, 0.05])  # one frame says A
        sticky = np.log(np.full((3, 3), 0.001) + np.eye(3) * 0.997)
        self.assertTrue((viterbi(emission, sticky) == H).all())
        loose = np.log(np.full((3, 3), 1 / 3))
        self.assertEqual(viterbi(emission, loose)[10], A)

    def test_follows_a_sustained_change(self):
        emission = np.log(np.full((40, 3), 0.05))
        emission[:20, H] = np.log(0.9)
        emission[20:, A] = np.log(0.9)
        sticky = np.log(np.full((3, 3), 0.001) + np.eye(3) * 0.997)
        path = viterbi(emission, sticky)
        self.assertTrue((path[:20] == H).all() and (path[20:] == A).all())

    def test_transition_counts_skip_x(self):
        log_trans = transition_log_probs([np.array([H, H, X, A, A], np.int8)], smoothing=0.0 + 1e-9)
        self.assertGreater(log_trans[H, A], np.log(0.4))  # H -> A across the dropped X

    def test_decode_marks_outside_play_and_splits_periods(self):
        values, labels = blobs(n=30)
        model = fit(values, labels, [labels], emission_scale=1.0)
        period = np.array([1] * 14 + [0] + [2] * 15, np.int8)
        for codes in (decode(model, values, period), argmax_decode(model, values, period)):
            self.assertEqual(codes[14], X)
            self.assertTrue(np.isin(codes[period > 0], (H, A, D)).all())

    def test_extra_evidence_can_override_the_model(self):
        values, labels = blobs(n=30)
        model = fit(values, labels, [labels], emission_scale=1.0)
        period = np.ones(30, np.int8)
        push_away = np.tile([0.0, 1e3, 0.0], (30, 1))
        self.assertTrue((decode(model, values, period, push_away) == A).all())

    def test_save_load_round_trip(self):
        values, labels = blobs()
        model = fit(values, labels, [labels], emission_scale=0.1)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp) / "m.npz")
            loaded = Model.load(Path(tmp) / "m.npz")
        np.testing.assert_array_equal(loaded.weights, model.weights)
        self.assertEqual(loaded.emission_scale, 0.1)


if __name__ == "__main__":
    unittest.main()
