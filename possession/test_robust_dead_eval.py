"""Synthetic leakage, pairing, selection and noise checks for the BR2 evaluator."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from possession.test_robust_dead import fixture
from scripts import robust_dead_eval as ev


def fake_rows(delta=.05, seeds=ev.DEV_SEEDS, recipe="half"):
    return [dict(match=mid, provider=ev.provider(mid), scenario=scenario, seed=seed, recipe=recipe,
                 baseline=dict(spell=.5, macro=.7, dead_f1=.7),
                 candidate=dict(spell=.5 + delta, macro=.7, dead_f1=.7))
            for mid in ev.DEV for scenario, seed in ev.conditions(seeds)]


class EvaluationBoundaryTest(unittest.TestCase):
    def test_fit_excludes_evaluation_before_loading(self):
        with patch.object(ev, "training_rows") as load:
            with self.assertRaises(ValueError):
                ev.fit_fold((ev.DEV[0],), (ev.DEV[0],), {}, Path("unused"))
            load.assert_not_called()

    def test_unknown_match_fails_before_io(self):
        with self.assertRaises(ValueError):
            ev.load_match("target:BAR-ATH")

    def test_duplicate_or_missing_pairs_rejected(self):
        rows = fake_rows()
        ev.validate_rows(rows, ev.DEV, ev.DEV_SEEDS, "half")
        for broken in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                ev.validate_rows(broken, ev.DEV, ev.DEV_SEEDS, "half")

    def test_seed_replication_does_not_shrink_interval(self):
        rows = fake_rows()
        for r in rows:
            r["candidate"]["spell"] += .01 * ev.DEV.index(r["match"])
        a = ev.summarize(rows)
        b = ev.summarize(rows * 9)
        np.testing.assert_allclose(a["match_ci95"], b["match_ci95"], atol=1e-14)

    def test_clean_loss_blocks_promotion(self):
        rows = fake_rows()
        for r in rows:
            if r["scenario"] == "clean":
                r["candidate"]["spell"] = .45
        self.assertFalse(ev.summarize(rows)["numerical_gate"])

    def test_one_provider_regression_blocks_promotion(self):
        rows = fake_rows(.15)
        for r in rows:
            if r["provider"] == "metrica":
                r["candidate"]["spell"] = .49
        self.assertFalse(ev.summarize(rows)["numerical_gate"])

    def test_tracker_only_gain_does_not_pass_standard_gate(self):
        rows = fake_rows(.10)
        for r in rows:
            if r["scenario"] == "standard":
                r["candidate"]["spell"] = .51
        self.assertFalse(ev.summarize(rows)["numerical_gate"])

    def test_selection_and_fixed_tie_break(self):
        good = ev.summarize(fake_rows())
        self.assertTrue(good["numerical_gate"])
        bad = ev.summarize(fake_rows(0.))
        self.assertIsNone(ev.select(dict(half=bad, replace=bad)))
        self.assertEqual(ev.select(dict(half=good, replace=good)), "half")
        self.assertEqual(ev.select(dict(half=bad, replace=good)), "replace")

    def test_evaluation_labels_never_affect_predictions(self):
        from possession import robust_dead as rd
        from run import load_config, DEFAULT_CONFIG
        m = fixture()
        model = rd.ActivityModel(np.zeros(25), np.ones(25), np.zeros(51))
        predictions = []

        def capture(gt, codes):
            predictions.append(codes.copy())
            return dict(spell=.5, macro=.5, dead_f1=.5, possessions=1)

        with patch.object(ev, "conditions", return_value=[("clean", None)]), \
                patch.object(ev, "metrics", side_effect=capture):
            for label in (0, 2):
                ext = SimpleNamespace(match=m, codes=np.full(m.n_frames, label, np.int8))
                with patch.object(ev, "load_match", return_value=ext):
                    ev.score_fold(ev.DEV[0], model, load_config(DEFAULT_CONFIG), (), ("half",))
        np.testing.assert_array_equal(predictions[0], predictions[2])
        np.testing.assert_array_equal(predictions[1], predictions[3])

    def test_rejection_stops_before_confirmation_or_target(self):
        rows = fake_rows(0.) + fake_rows(0., recipe="replace")
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(ev.os.environ, {"RUNPOD_POD_ID": "synthetic-test-only"}), \
                patch.object(ev, "work_dir", return_value=Path(tmp)), \
                patch.object(ev, "fingerprint", return_value="synthetic"), \
                patch.object(ev, "load_config", return_value={}), \
                patch.object(ev, "fit_fold", return_value=(None, {})) as fit, \
                patch.object(ev, "score_fold", side_effect=lambda mid, *args:
                             [r for r in rows if r["match"] == mid]) as score, \
                redirect_stdout(io.StringIO()):
            report = ev.run_experiment()
        self.assertIsNone(report["selected"])
        self.assertFalse(report["target_evaluated"])
        self.assertFalse(report["production_promotion"])
        self.assertEqual(score.call_count, len(ev.DEV))
        for call in fit.call_args_list:
            train, evaluated = call.args[:2]
            self.assertFalse(set(train) & set(evaluated))
            self.assertFalse(set(train + evaluated) & set(ev.CONFIRM))

    def test_noise_has_no_labels_and_preserves_missingness(self):
        m = fixture(1000)
        m.ball_xy[100:130] = np.nan
        before = m.players_xy.copy()
        # Isolate added player noise from the pre-existing fragmentation implementation.
        with patch.object(ev.external, "inject_noise", side_effect=lambda match, *args: copy.deepcopy(match)):
            for scenario in ("smooth", "jitter", "drift"):
                a = ev.perturb(m, "metrica", scenario, 11)
                b = ev.perturb(m, "metrica", scenario, 11)
                np.testing.assert_array_equal(a.players_xy, b.players_xy)
                self.assertTrue(np.isnan(a.ball_xy[100:130]).all())
                self.assertTrue(np.isnan(a.players_speed[[0, 499, 500, 999]]).all())
        np.testing.assert_array_equal(m.players_xy, before)


if __name__ == "__main__":
    unittest.main()
