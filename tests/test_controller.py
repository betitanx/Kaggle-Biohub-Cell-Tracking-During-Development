import json
import tempfile
import unittest
from pathlib import Path

from biohub_lab.controller import collect_records, run_enabled_experiments, write_submission_plan
from biohub_lab.runner import load_config


ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    def test_short_run_creates_plan_with_at_most_five_candidates(self):
        config = load_config(ROOT / "config" / "experiments.json")
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            output = Path(directory) / "outputs"
            run_enabled_experiments(config, runs, ROOT, duration_seconds=10, max_experiments=2)
            records = [json.loads(path.read_text()) for path in runs.glob("*/result.json")]
            plan = write_submission_plan(
                records,
                output,
                limit=2,
                competition=config["competition"],
                expected_timepoints=config["expected_timepoints"],
                expected_datasets={"sample_1"},
                daily_submission_limit=5,
            )

            self.assertEqual(len(plan["candidates"]), 2)
            self.assertEqual(plan["max_candidates"], 2)
            self.assertEqual(plan["limit"], 5)
            self.assertEqual(plan["expected_timepoints"], 100)
            self.assertEqual(plan["expected_datasets"], ["sample_1"])
            self.assertTrue((output / "ranking.csv").is_file())
            self.assertTrue((output / "submission_plan.json").is_file())

    def test_adaptive_cycle_runs_next_proposal_after_initial_candidate(self):
        config = {
            "adaptive": {"enabled": True, "max_proposals_per_cycle": 1},
            "experiments": [
                {
                    "name": "adaptive_smoke",
                    "family": "smoke",
                    "enabled": True,
                    "command": [
                        "${PYTHON}",
                        "-m",
                        "biohub_lab.demo_candidate",
                        "--run-dir",
                        "${RUN_DIR}",
                        "--variant",
                        "weak_ensemble",
                    ],
                    "adaptive_variants": [
                        {"name": "weaker", "direction": "weaker", "args": []}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            results = run_enabled_experiments(config, runs, ROOT, duration_seconds=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["experiment"], "adaptive_smoke__adaptive__weaker")
        self.assertEqual(results[1]["proposal_reason"], "promising_holdout_test_weaker_hypothesis")

    def test_submission_plan_preserves_family_for_live_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            records = [
                {
                    "run_id": "smoke-run",
                    "experiment": "smoke_candidate",
                    "family": "smoke",
                    "status": "completed",
                    "artifact": str(artifact),
                    "artifact_valid": True,
                    "local_score": 0.8,
                    "in_domain_score": 0.8,
                    "holdout_score": 0.75,
                }
            ]

            plan = write_submission_plan(
                records,
                Path(directory) / "outputs",
                limit=5,
                cycle_id="cycle-test",
                window_started_at="2026-08-04T12:00:00+00:00",
                window_seconds=14400,
            )

        self.assertEqual(plan["candidates"][0]["family"], "smoke")
        self.assertEqual(plan["cycle_id"], "cycle-test")
        self.assertEqual(plan["window_seconds"], 14400.0)

    def test_submission_plan_can_select_a_recomputing_notebook_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            records = [
                {
                    "run_id": "runner-plan",
                    "experiment": "classical_runner",
                    "family": "classical",
                    "status": "completed",
                    "artifact": str(artifact),
                    "artifact_valid": True,
                    "local_score": 0.8,
                    "holdout_score": 0.8,
                }
            ]
            plan = write_submission_plan(
                records,
                Path(directory) / "outputs",
                limit=1,
                submission_mode="notebook",
                notebook_dataset="owner/candidates",
                notebook_payload_mode="runner",
                notebook_enable_gpu=True,
                notebook_machine_shape="NvidiaTeslaT4",
                notebook_runner_source="outputs/kaggle/runner.py",
            )

        self.assertEqual(plan["notebook_payload_mode"], "runner")
        self.assertTrue(plan["notebook_enable_gpu"])
        self.assertEqual(plan["notebook_machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(plan["notebook_runner_source"], "outputs/kaggle/runner.py")

    def test_submission_plan_excludes_artifact_missing_expected_timepoints(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            records = [
                {
                    "run_id": "coverage-run",
                    "experiment": "coverage_candidate",
                    "family": "classical",
                    "status": "completed",
                    "artifact": str(artifact),
                    "artifact_valid": True,
                    "local_score": 0.8,
                    "holdout_score": 0.8,
                }
            ]

            plan = write_submission_plan(
                records,
                Path(directory) / "outputs",
                limit=5,
                expected_timepoints=3,
            )

        self.assertEqual(plan["candidates"], [])

    def test_collect_records_can_isolate_a_submission_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "runs"
            for name, cycle_id in (("old", "cycle-old"), ("new", "cycle-new")):
                run_dir = runs / name
                run_dir.mkdir(parents=True)
                (run_dir / "result.json").write_text(
                    json.dumps({"run_id": name, "experiment": name, "cycle_id": cycle_id}),
                    encoding="utf-8",
                )

            current = collect_records(runs, cycle_id="cycle-new")

        self.assertEqual([record["experiment"] for record in current], ["new"])

    def test_run_result_keeps_cycle_id_for_current_plan(self):
        config = load_config(ROOT / "config" / "experiments.json")
        with tempfile.TemporaryDirectory() as directory:
            results = run_enabled_experiments(
                config,
                Path(directory) / "runs",
                ROOT,
                duration_seconds=5,
                max_experiments=1,
                cycle_id="cycle-test",
            )

        self.assertEqual(results[0]["cycle_id"], "cycle-test")


if __name__ == "__main__":
    unittest.main()
