import unittest
import tempfile
import sys
import os
from pathlib import Path
from unittest.mock import patch

from biohub_lab.doctor import check_environment


class DoctorTests(unittest.TestCase):
    def test_control_python_and_consent_are_reported_without_secret_values(self):
        report = check_environment(packages=())

        self.assertTrue(report["python_supported_by_control"])
        self.assertIn("live_submission_consent", report)
        self.assertNotIn("KAGGLE_API_TOKEN", str(report))

    def test_default_report_includes_optional_scientific_adapters(self):
        report = check_environment()

        names = {item["name"] for item in report["packages"]}
        self.assertTrue({"ultrack", "cellpose", "stardist"}.issubset(names))

    def test_reports_real_run_paths_without_reading_file_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            report = check_environment(
                packages=(),
                data_dir=root / "train",
                extra_paths={"test_dir": root / "test", "baseline_root": root},
            )

        self.assertTrue(report["paths"]["data_dir"]["exists"])
        self.assertFalse(report["paths"]["test_dir"]["exists"])
        self.assertTrue(report["paths"]["baseline_root"]["exists"])

    def test_probes_external_candidate_interpreter_without_importing_models(self):
        report = check_environment(
            packages=(),
            external_python=sys.executable,
            external_packages=("json", "module_that_does_not_exist",),
        )

        self.assertTrue(report["external_python"]["available"])
        packages = {item["name"]: item["available"] for item in report["external_python"]["packages"]}
        self.assertTrue(packages["json"])
        self.assertFalse(packages["module_that_does_not_exist"])

    def test_reports_kaggle_auth_presence_without_exposing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "kaggle.json").write_text('{"username":"user","key":"secret"}\n', encoding="utf-8")
            with patch.dict(os.environ, {"KAGGLE_CONFIG_DIR": str(config_dir)}, clear=False):
                report = check_environment(packages=())

        self.assertTrue(report["kaggle_auth_configured"])
        self.assertIn("kaggle.json", report["kaggle_auth_sources"])
        self.assertNotIn("secret", str(report))


if __name__ == "__main__":
    unittest.main()
