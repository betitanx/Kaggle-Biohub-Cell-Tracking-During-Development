import tempfile
import unittest
from pathlib import Path

from biohub_lab.lablog import update_lab_proposals, update_lab_results, update_lab_submissions


class LabLogTests(unittest.TestCase):
    def test_generated_section_is_idempotent_and_contains_weakness_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n\n## 12. Registro automático\n\n", encoding="utf-8")
            records = [
                {
                    "run_id": "one",
                    "experiment": "candidate",
                    "status": "completed",
                    "local_score": 0.8,
                    "in_domain_score": 0.82,
                    "holdout_score": 0.78,
                    "artifact": "/tmp/candidate.csv",
                }
            ]

            update_lab_results(path, records)
            first = path.read_text(encoding="utf-8")
            update_lab_results(path, records)
            second = path.read_text(encoding="utf-8")

            self.assertEqual(first, second)
            self.assertIn("0.951219512195122", first)

    def test_submission_section_is_idempotent_and_records_public_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n", encoding="utf-8")
            first_results = [
                {
                    "experiment": "candidate",
                    "status": "submitted",
                    "reference": "123",
                    "public_score": 0.4567,
                    "artifact": "/tmp/candidate.csv",
                }
            ]
            second_results = [
                {
                    "experiment": "candidate-two",
                    "status": "pending",
                    "reference": "456",
                    "public_score": "—",
                    "artifact": "/tmp/candidate-two.csv",
                }
            ]

            update_lab_submissions(path, first_results)
            update_lab_submissions(path, second_results)
            first = path.read_text(encoding="utf-8")
            update_lab_submissions(path, second_results)
            second = path.read_text(encoding="utf-8")

            self.assertEqual(first, second)
            self.assertIn("0.4567", first)
            self.assertIn("123", first)
            self.assertIn("456", first)

    def test_results_do_not_present_missing_artifacts_as_usable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n", encoding="utf-8")
            update_lab_results(
                path,
                [{"run_id": "missing", "experiment": "candidate", "status": "invalid", "artifact": "runs/missing/submission.csv"}],
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("| invalid | `—` |", text)

    def test_results_show_synthetic_diagnostic_scope_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n", encoding="utf-8")
            update_lab_results(
                path,
                [
                    {
                        "run_id": "synthetic",
                        "experiment": "synthetic_quality",
                        "status": "completed",
                        "evaluation_scope": "synthetic_diagnostic_only",
                        "diagnostic_score": 0.7729100529,
                        "artifact": "/tmp/synthetic.csv",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("synthetic_diagnostic_only", text)
        self.assertIn("diag:0.7729", text)

    def test_subset_scope_is_not_logged_as_competition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n", encoding="utf-8")
            update_lab_results(
                path,
                [
                    {
                        "run_id": "subset",
                        "experiment": "classical_subset",
                        "status": "completed",
                        "data_scope": "subset",
                        "local_score": 0.2,
                        "holdout_score": 0.1,
                        "artifact": "/tmp/subset.csv",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("subset_diagnostic_only", text)

    def test_proposal_section_records_adaptive_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.md"
            path.write_text("# Lab\n", encoding="utf-8")
            update_lab_proposals(
                path,
                [
                    {
                        "name": "classical__adaptive__wide",
                        "family": "classical",
                        "proposal_reason": "promising_holdout_test_weaker_hypothesis",
                        "hypothesis": "test weaker",
                    }
                ],
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("promising_holdout_test_weaker_hypothesis", text)


if __name__ == "__main__":
    unittest.main()
