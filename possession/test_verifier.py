"""Synthetic checks of the spell-edit verifier; never reads match assets."""
import sys
import unittest
from pathlib import Path

import numpy as np

from possession import verifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
import possession_metrics as pm  # noqa: E402


class VerifierTest(unittest.TestCase):
    def test_true_positives_match_the_official_scorer(self):
        rng = np.random.RandomState(1)
        for _ in range(100):
            n = 400
            gt = np.repeat(rng.randint(0, 3, 20), 20)[:n].astype(np.int8)
            pred = gt.copy()
            for _ in range(6):
                a = rng.randint(0, n - 30)
                pred[a:a + rng.randint(1, 30)] = rng.randint(0, 3)
            tp = verifier.true_positives(verifier._spells(gt), verifier._spells(pred))
            runs = [(0, n)]
            ref = pm.spell_scores(gt, pred, runs)
            self.assertEqual([ref["per_iou"][k]["n_matched"] for k in pm.SPELL_IOUS], tp.tolist())

    def test_no_edits_when_every_scale_decodes_the_same(self):
        prod, edits = verifier.propose(np.zeros((50, 3)), lambda e, s: np.zeros(len(e), np.int8))
        self.assertEqual(edits, [])
        self.assertTrue((prod == 0).all())

    def test_edits_rebuild_each_scaled_path_and_accept_never_overlaps(self):
        rng = np.random.RandomState(2)
        emis = rng.normal(0, 1, (600, 3))
        viterbi = verifier.viterbi_decode({"switch_cost": {"team_switch": 4.0, "to_dead": 3.0, "from_dead": 3.0}})
        prod, edits = verifier.propose(emis, viterbi)
        for s in verifier.SCALES:
            alt = viterbi(emis, s)
            path = prod.copy()
            for e in edits:
                if s in e["scales"]:
                    path[e["a"]:e["b"]] = e["new"]
            np.testing.assert_array_equal(path, alt)
        out = verifier.accept(prod, edits, np.arange(len(edits), 0, -1, dtype=float))
        spans = {}
        for e in edits:
            spans.setdefault((e["a"], e["b"]), []).append(e["new"])
        changed = verifier._runs(out != prod)
        self.assertTrue(changed)
        for a, b in changed:            # every change is exactly one whole proposed edit
            self.assertIn((a, b), spans)
            self.assertTrue(any(np.array_equal(out[a:b], new) for new in spans[(a, b)]))
        for (a1, b1), (a2, b2) in zip(changed, changed[1:]):
            self.assertGreater(a2, b1)  # applied edits never touch

    def test_features_are_team_antisymmetric(self):
        rng = np.random.RandomState(3)
        n = 400
        emis = rng.normal(0, 1, (n, 3))
        feats = {k: rng.normal(0, 1, n) for k in verifier.FEATURE_KEYS}
        feats["touch_vote"] = rng.choice([-1.0, 0.0, 1.0], n)
        prod, edits = verifier.propose(emis, verifier.viterbi_decode({"switch_cost": {"team_switch": 3.0, "to_dead": 2.0, "from_dead": 2.0}}))
        X = verifier.features(emis, feats, prod, edits)
        swap = {0: 1, 1: 0, 2: 2}
        emis_s = emis[:, [1, 0, 2]]
        feats_s = dict(feats, touch_vote=-feats["touch_vote"], owner_sign=-feats["owner_sign"],
                       control_balance=-feats["control_balance"],
                       nearest_home_m=feats["nearest_away_m"], nearest_away_m=feats["nearest_home_m"])
        prod_s = np.vectorize(swap.get)(prod).astype(np.int8)
        edits_s = [dict(e, new=np.vectorize(swap.get)(e["new"]).astype(np.int8)) for e in edits]
        np.testing.assert_allclose(verifier.features(emis_s, feats_s, prod_s, edits_s), X)


if __name__ == "__main__":
    unittest.main()
