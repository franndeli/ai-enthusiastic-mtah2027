import contextlib
import io
import unittest
from unittest import mock

from possession import pipeline
from possession.model import Model


class TestEntryPoint(unittest.TestCase):
    def test_default_model_is_v6(self):
        self.assertTrue(pipeline.DEFAULT_MODEL.endswith("possession_v6.json"))
        self.assertTrue(Model.from_json(pipeline.DEFAULT_MODEL).p.get("pathcrf"))

    def test_missing_packages(self):
        self.assertEqual(pipeline.missing_packages(["numpy", "no_such_package_mtah"]),
                         ["no_such_package_mtah"])

    def test_v6_without_its_packages_stops_before_reading_data(self):
        out = io.StringIO()
        with mock.patch.object(pipeline, "V6_PACKAGES", ("no_such_package_mtah",)), \
                contextlib.redirect_stdout(out):
            code = pipeline.main(["--out", "unused.csv", "--data", "/nonexistent"])
        self.assertEqual(code, 2)
        self.assertIn("no_such_package_mtah", out.getvalue())
        self.assertIn("requirements-v6.txt", out.getvalue())
        self.assertIn("--model config/possession_v5.json", out.getvalue())
        self.assertNotIn("match data not found", out.getvalue())


if __name__ == "__main__":
    unittest.main()
