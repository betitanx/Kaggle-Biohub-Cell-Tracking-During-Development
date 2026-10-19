import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from biohub_lab.cli import main, wait_for_duration


class CliTests(unittest.TestCase):
    def test_production_cycle_keeps_daily_budget_and_two_submission_cap(self):
        config_path = Path(__file__).parents[1] / "config" / "experiments.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(config["daily_submission_limit"], 5)
        self.assertEqual(config["max_submissions_per_cycle"], 2)
        deep = next(item for item in config["experiments"] if item["name"] == "cellmot_unet_transformer")
        self.assertEqual(deep["notebook_expected_timepoints"], 100)
        self.assertIn("--pool-kernel-um", deep["command"])
        pool_index = deep["command"].index("--pool-kernel-um")
        self.assertEqual(deep["command"][pool_index + 1], "3.0")
        self.assertIn("--ilp-disappearance-weight", deep["command"])
        disappearance_index = deep["command"].index("--ilp-disappearance-weight")
        self.assertEqual(deep["command"][disappearance_index + 1], "1.4")

    def test_doctor_subcommand_reports_ultrack_label_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            labels.mkdir()
            output = StringIO()
            with redirect_stdout(output):
                result = main(
                    [
                        "doctor",
                        "--ultrack-labels-dir",
                        str(labels),
                        "--ultrack-validation-labels-dir",
                        str(root / "validation-labels"),
                        "--ultrack-holdout-labels-dir",
                        str(root / "holdout-labels"),
                    ]
                )
            report = json.loads(output.getvalue())

        self.assertEqual(result, 0)
        self.assertTrue(report["paths"]["ultrack_labels_dir"]["exists"])
        self.assertFalse(report["paths"]["ultrack_validation_labels_dir"]["exists"])
        self.assertFalse(report["paths"]["ultrack_holdout_labels_dir"]["exists"])

    def test_split_subcommand_creates_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train"
            train_split = root / "train_split"
            holdout = root / "holdout"
            source.mkdir()
            for name in ("a", "b"):
                (source / (name + ".zarr")).mkdir()
                (source / (name + ".geff")).write_text(name, encoding="utf-8")

            result = main(
                [
                    "split",
                    "--source-dir",
                    str(source),
                    "--train-dir",
                    str(train_split),
                    "--holdout-dir",
                    str(holdout),
                    "--fraction",
                    "0.5",
                    "--seed",
                    "3",
                ]
            )
            manifest = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
            train_split_exists = train_split.is_dir()

        self.assertEqual(result, 0)
        self.assertEqual(len(manifest["holdout_datasets"]), 1)
        self.assertTrue(train_split_exists)

    def test_rank_does_not_require_a_preexisting_lab_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            run_dir = runs / "one"
            run_dir.mkdir(parents=True)
            artifact = root / "submission.csv"
            artifact.write_text(
                (Path(__file__).parent / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (run_dir / "result.json").write_text(
                json.dumps(
                    {
                        "run_id": "one",
                        "experiment": "candidate",
                        "family": "classical",
                        "status": "completed",
                        "artifact": str(artifact),
                        "artifact_valid": True,
                        "local_score": 0.8,
                        "in_domain_score": 0.8,
                        "holdout_score": 0.75,
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                result = main(["rank", "--runs-dir", str(runs), "--output-dir", "outputs"])
                plan_exists = (root / "outputs" / "submission_plan.json").is_file()
            finally:
                os.chdir(previous)

        self.assertEqual(result, 0)
        self.assertTrue(plan_exists)

    def test_wait_for_duration_sleeps_only_until_the_deadline(self):
        slept = []

        remaining = wait_for_duration(
            started_at=100.0,
            duration_seconds=10.0,
            clock=lambda: 106.5,
            sleep=lambda seconds: slept.append(seconds),
        )

        self.assertEqual(remaining, 3.5)
        self.assertEqual(slept, [3.5])

    def test_live_run_waits_before_submitting_real_candidate(self):
        events = []

        def write_real_plan(*_args, **_kwargs):
            Path("outputs").mkdir(parents=True, exist_ok=True)
            (Path("outputs") / "submission_plan.json").write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "candidates": [
                            {"experiment": "real", "family": "classical", "artifact": "real.csv"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return {}

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with patch("biohub_lab.cli.load_config", return_value={"competition": "biohub-cell-tracking-during-development", "experiments": [], "max_submissions_per_cycle": 5}), patch(
                    "biohub_lab.cli.run_enabled_experiments"
                ), patch("biohub_lab.cli.collect_records", return_value=[]), patch(
                    "biohub_lab.cli.write_submission_plan", side_effect=write_real_plan
                ), patch("biohub_lab.cli.write_proposal_queue", return_value={"proposals": []}), patch(
                    "biohub_lab.cli.wait_for_duration", side_effect=lambda *_args, **_kwargs: events.append("wait")
                ), patch("biohub_lab.cli.submit_plan", side_effect=lambda *_args, **_kwargs: events.append("submit") or []):
                    result = main(
                        [
                            "run",
                            "--duration-seconds",
                            "10",
                            "--output-dir",
                            "runs",
                            "--live-submit",
                        ]
                    )
            finally:
                os.chdir(previous)

        self.assertEqual(result, 0)
        self.assertEqual(events, ["wait", "submit"])

    def test_live_run_does_not_wait_for_non_live_diagnostic_candidate(self):
        events = []

        def write_non_live_plan(*_args, **_kwargs):
            Path("outputs").mkdir(parents=True, exist_ok=True)
            (Path("outputs") / "submission_plan.json").write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "candidates": [
                            {
                                "experiment": "synthetic",
                                "family": "synthetic_quality",
                                "live_eligible": False,
                                "artifact": "synthetic.csv",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return {}

        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                with patch("biohub_lab.cli.load_config", return_value={"competition": "biohub-cell-tracking-during-development", "experiments": [], "max_submissions_per_cycle": 5}), patch(
                    "biohub_lab.cli.run_enabled_experiments"
                ), patch("biohub_lab.cli.collect_records", return_value=[]), patch(
                    "biohub_lab.cli.write_submission_plan", side_effect=write_non_live_plan
                ), patch("biohub_lab.cli.write_proposal_queue", return_value={"proposals": []}), patch(
                    "biohub_lab.cli.wait_for_duration", side_effect=lambda *_args, **_kwargs: events.append("wait")
                ), patch("biohub_lab.cli.submit_plan", side_effect=lambda *_args, **_kwargs: events.append("submit") or []):
                    result = main(
                        [
                            "run",
                            "--duration-seconds",
                            "10",
                            "--output-dir",
                            "runs",
                            "--live-submit",
                        ]
                    )
            finally:
                os.chdir(previous)

        self.assertEqual(result, 0)
        self.assertEqual(events, ["submit"])


if __name__ == "__main__":
    unittest.main()
