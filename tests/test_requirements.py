import tempfile
import unittest
from pathlib import Path

from biohub_lab.requirements import missing_requirements
from biohub_lab.runner import run_experiment


ROOT = Path(__file__).resolve().parents[1]


class RequirementsTests(unittest.TestCase):
    def test_reports_missing_env_path_and_import_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory)
            missing = missing_requirements(
                {
                    "requires": ["DATA_DIR", "package_that_does_not_exist", "missing-relative-path"]
                },
                env={"DATA_DIR": str(existing)},
                project_root=ROOT,
            )

        self.assertIn("package_that_does_not_exist", missing)
        self.assertIn("missing-relative-path", missing)
        self.assertNotIn("DATA_DIR", missing)

    def test_runner_records_blocked_candidate_without_starting_subprocess(self):
        spec = {
            "name": "blocked_requirements_test",
            "family": "external",
            "requires": ["REQUIRED_FOR_TEST"],
            "command": ["${PYTHON}", "-m", "biohub_lab.demo_candidate", "--run-dir", "${RUN_DIR}"]
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(spec, Path(directory), ROOT, env={})

        self.assertEqual(result["status"], "blocked")
        self.assertIn("REQUIRED_FOR_TEST", result["blocked_requirements"])


if __name__ == "__main__":
    unittest.main()
