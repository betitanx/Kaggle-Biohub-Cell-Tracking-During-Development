import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from biohub_lab.budget import SubmissionBudget
from biohub_lab.kaggle import KaggleClient, NotebookSubmissionClient, submit_plan


ROOT = Path(__file__).resolve().parents[1]


class FakeNotebookApi:
    def __init__(self):
        self.dataset_calls = []
        self.kernel_metadata = []
        self.kernel_sources = []
        self.submission_calls = []
        self.output_bytes = None

    def dataset_status(self, _dataset, format=None):
        if format == "json":
            return json.dumps({"status": "ready", "current_version_number": 2})
        return "ready"

    def dataset_create_version(self, folder, **kwargs):
        self.dataset_calls.append((Path(folder), kwargs))
        self.output_bytes = next(Path(folder).glob("*.csv")).read_bytes()
        return SimpleNamespace(version_number=2)

    def kernels_push(self, folder):
        metadata = json.loads((Path(folder) / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.kernel_metadata.append(metadata)
        notebook = json.loads((Path(folder) / metadata["code_file"]).read_text(encoding="utf-8"))
        self.kernel_sources.append("".join(notebook["cells"][0]["source"]))
        return SimpleNamespace(ref="/code/" + metadata["id"], version_number=3)

    def kernels_status(self, _kernel):
        return SimpleNamespace(status="KernelWorkerStatus.COMPLETE")

    def kernels_output(self, _kernel, folder, **_kwargs):
        (Path(folder) / "submission.csv").write_bytes(self.output_bytes)

    def competition_submit_code(self, *args, **kwargs):
        self.submission_calls.append((args, kwargs))
        return SimpleNamespace(ref=99123)


class StaleKernelResponseApi(FakeNotebookApi):
    """Simulate a concurrent push returning another kernel's valid ref."""

    def kernels_push(self, folder):
        response = super().kernels_push(folder)
        return SimpleNamespace(ref="owner/stale-kernel", version_number=response.version_number)


class HangingKernelOutputApi(FakeNotebookApi):
    def kernels_output(self, _kernel, _folder, **_kwargs):
        time.sleep(0.2)


class KaggleSafetyTests(unittest.TestCase):
    def test_notebook_client_times_out_when_kernel_output_download_hangs(self):
        api = HangingKernelOutputApi()
        client = NotebookSubmissionClient(
            "biohub-cell-tracking-during-development",
            "owner/candidates",
            api=api,
            output_timeout_seconds=0.01,
        )

        with self.assertRaisesRegex(TimeoutError, "timed out downloading"):
            client._verify_kernel_output("owner/kernel")

    def test_notebook_client_uses_requested_kernel_when_push_response_is_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = StaleKernelResponseApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                expected_datasets={"sample_1"},
                expected_timepoints=range(2),
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.dry_run_many(
                [{"experiment": "stale-response", "artifact": str(artifact)}],
                "cycle-test",
                root / "dry-run-output",
            )

        self.assertEqual(result[0]["status"], "dry_run_complete")
        self.assertEqual(result[0]["kernel"], api.kernel_metadata[0]["id"])

    def test_notebook_client_rejects_missing_expected_dataset_after_kernel_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                expected_datasets={"required_dataset"},
                expected_timepoints=range(2),
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "missing-dataset", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "failed")
        self.assertIn("missing expected datasets", result[0]["error"])
        self.assertEqual(api.submission_calls, [])

    def test_notebook_client_dry_run_persists_and_validates_kernel_output_without_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            runner = root / "runner.py"
            runner.write_text("print('recompute')\n", encoding="utf-8")
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                runner_source=runner,
                expected_datasets={"sample_1"},
                expected_timepoints=range(2),
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.dry_run_many(
                [{"experiment": "dry-run", "artifact": str(artifact)}],
                "cycle-test",
                root / "dry-run-output",
            )

            output = Path(result[0]["artifact"])
            self.assertTrue(output.is_file())

        self.assertEqual(result[0]["status"], "dry_run_complete")
        self.assertTrue(result[0]["validation"]["valid"])
        self.assertEqual(api.submission_calls, [])

    def test_notebook_client_rejects_incomplete_timepoint_coverage_before_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidate.csv"
            artifact.write_bytes((ROOT / "tests" / "fixtures" / "submission.csv").read_bytes())
            api = FakeNotebookApi()
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                expected_timepoints=range(3),
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "q995", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "failed")
        self.assertIn("missing node timepoints", result[0]["error"])
        self.assertEqual(api.dataset_calls, [])
        self.assertEqual(api.submission_calls, [])

    def test_notebook_client_publishes_dataset_pushes_fresh_kernel_and_submits_code(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidate.csv"
            artifact.write_bytes(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes().replace(b"\n", b"\r\n")
            )
            api = FakeNotebookApi()
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "q995", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(result[0]["reference"], "99123")
        self.assertEqual(result[0]["dataset_version"], 2)
        self.assertEqual(len(api.dataset_calls), 1)
        self.assertEqual(api.kernel_metadata[0]["dataset_sources"], ["owner/candidates/2"])
        self.assertLessEqual(len(api.kernel_metadata[0]["title"]), 50)
        self.assertEqual(api.submission_calls[0][1]["kernel_version"], 3)
        self.assertNotIn(b"\r\n", api.output_bytes)

    def test_notebook_client_can_preserve_crlf_for_controlled_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidate.csv"
            artifact.write_bytes((ROOT / "tests" / "fixtures" / "submission.csv").read_bytes().replace(b"\r\n", b"\n"))
            api = FakeNotebookApi()
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                line_ending="crlf",
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "q999-crlf", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertIn(b"\r\n", api.output_bytes)
        self.assertNotIn(b"\n", api.output_bytes.replace(b"\r\n", b""))

    def test_notebook_client_can_embed_artifact_without_dataset_input(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidate.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="embedded",
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "embedded-control", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(api.dataset_calls, [])
        self.assertEqual(api.output_bytes, expected)
        self.assertEqual(api.kernel_metadata[0]["dataset_sources"], [])

    def test_notebook_client_can_push_a_recomputing_runner_without_dataset_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            runner = root / "runner.py"
            runner.write_text("print('recompute from Kaggle competition input')\n", encoding="utf-8")
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                runner_source=runner,
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "runner-control", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(api.dataset_calls, [])
        self.assertEqual(api.kernel_metadata[0]["dataset_sources"], [])
        self.assertIn("recompute from Kaggle competition input", api.kernel_sources[0])

    def test_runner_can_attach_public_support_pack_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            runner = root / "runner.py"
            runner.write_text("print('recompute with support pack')\n", encoding="utf-8")
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                runner_source=runner,
                additional_dataset_sources=["pilkwang/biohub-tracking-support-pack-50ep-v1"],
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "support-pack-control", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertEqual(
            api.kernel_metadata[0]["dataset_sources"],
            ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        )

    def test_runner_can_build_h13_finetune_with_pretrain_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                additional_dataset_sources=[
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "daifanhao/biohub-synthetic-detector-pretrain-public-v1",
                ],
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{
                    "experiment": "h13-finetune-control",
                    "artifact": str(artifact),
                    "notebook_runner_kind": "h13_finetune",
                    "notebook_runner_options": {"epochs": 1, "max_iters": 1},
                }],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertIn("unet_pretrained.pth", api.kernel_sources[0])
        self.assertIn("h13_finetune", api.kernel_sources[0])
        self.assertEqual(
            api.kernel_metadata[0]["dataset_sources"],
            [
                "pilkwang/biohub-tracking-support-pack-50ep-v1",
                "daifanhao/biohub-synthetic-detector-pretrain-public-v1",
            ],
        )

    def test_notebook_client_can_request_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                enable_gpu=True,
                machine_shape="NvidiaTeslaT4",
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [{"experiment": "gpu-runner-control", "artifact": str(artifact)}],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertTrue(api.kernel_metadata[0]["enable_gpu"])
        self.assertEqual(api.kernel_metadata[0]["machine_shape"], "NvidiaTeslaT4")

    def test_runner_can_publish_the_metric_probe_as_an_explicit_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            expected = (ROOT / "tests" / "fixtures" / "submission.csv").read_bytes()
            artifact.write_bytes(expected)
            api = FakeNotebookApi()
            api.output_bytes = expected
            client = NotebookSubmissionClient(
                "biohub-cell-tracking-during-development",
                "owner/candidates",
                api=api,
                payload_mode="runner",
                additional_dataset_sources=["pilkwang/biohub-tracking-support-pack-50ep-v1"],
                poll_seconds=0,
                sleep=lambda _seconds: None,
            )

            result = client.submit_many(
                [
                    {
                        "experiment": "support-pack-metric-probe",
                        "artifact": str(artifact),
                        "notebook_runner_kind": "support_pack_metric_probe",
                        "notebook_runner_det_threshold": 0.9675,
                        "notebook_runner_options": {
                            "pool_kernel_um": 3.0,
                            "edge_min_threshold": 0.25,
                            "edge_topk_parents": 2,
                            "edge_topk_children": 2,
                            "edge_max_distance_um": 10.0,
                            "det_tta_mode": "dihedral8",
                            "feature_tta": True,
                            "feature_tta_blend": 0.75,
                            "metric_probe_max_components": 1200,
                            "metric_probe_forks": 5,
                            "gap_closing": True,
                            "gap_closing_max_total_distance_um": 10.0,
                        },
                    }
                ],
                "cycle-test",
            )

        self.assertEqual(result[0]["status"], "submitted")
        self.assertIn("metric_probe", api.kernel_sources[0])
        self.assertIn("1200", api.kernel_sources[0])
        self.assertIn("0.25", api.kernel_sources[0])
        self.assertIn("edge_topk_children", api.kernel_sources[0])
        self.assertIn("10.0", api.kernel_sources[0])
        self.assertIn("dihedral8", api.kernel_sources[0])
        self.assertIn("feature_tta", api.kernel_sources[0])
        self.assertIn("0.75", api.kernel_sources[0])
        self.assertIn("GapClosingConfig", api.kernel_sources[0])

    def test_submit_plan_routes_notebook_mode_without_file_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "submission_mode": "notebook",
                        "notebook_dataset": "owner/candidates",
                        "limit": 5,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [
                            {"experiment": "q995", "family": "classical", "artifact": str(artifact)}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(KaggleClient, "remaining_submission_limit", return_value=5), patch(
                "biohub_lab.kaggle._kaggle_api", return_value=object()
            ), patch.object(
                NotebookSubmissionClient,
                "submit_many",
                return_value=[{"experiment": "q995", "status": "submitted", "reference": "99123"}],
            ) as submit_many:
                result = submit_plan(
                    plan,
                    state,
                    live=True,
                    env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"},
                )

        self.assertEqual(result[0]["status"], "submitted")
        submit_many.assert_called_once()

    def test_client_reads_current_remote_submission_limit(self):
        completed = SimpleNamespace(returncode=0, stdout='{"numAllowedNow":5}', stderr="")
        with patch("biohub_lab.kaggle.shutil.which", return_value="/usr/bin/kaggle"), patch(
            "biohub_lab.kaggle.subprocess.run", return_value=completed
        ) as run:
            allowed = KaggleClient("biohub-cell-tracking-during-development").remaining_submission_limit(
                env={"KAGGLE_CONFIG_DIR": ".kaggle"}
            )

        self.assertEqual(allowed, 5)
        self.assertIn("submission-limits", run.call_args.args[0])

    def test_live_plan_blocks_when_remote_submission_already_uses_one_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "require_all_candidates": True,
                        "candidates": [
                            {"experiment": "a", "family": "classical", "artifact": "a.csv"},
                            {"experiment": "b", "family": "classical", "artifact": "b.csv"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            limit_response = SimpleNamespace(returncode=0, stdout='{"numAllowedNow":5}', stderr="")
            submissions_response = SimpleNamespace(
                returncode=0,
                stdout=(
                    "ref,fileName,date,description,status,publicScore,privateScore\n"
                    "123,submission.csv,2026-09-01,biohub-notebook:a,SubmissionStatus.PENDING,,\n"
                ),
                stderr="",
            )
            with patch("biohub_lab.kaggle.shutil.which", return_value="/usr/bin/kaggle"), patch(
                "biohub_lab.kaggle.subprocess.run", side_effect=[limit_response, submissions_response]
            ) as run, patch.object(NotebookSubmissionClient, "submit_many") as submit_many:
                result = submit_plan(
                    plan,
                    state,
                    live=True,
                    env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"},
                )

        self.assertEqual(result[0]["status"], "already_attempted")
        self.assertEqual(result[1]["status"], "blocked_remote_duplicate")
        self.assertFalse(state.exists())
        submit_many.assert_not_called()
        self.assertIn("submissions", run.call_args_list[1].args[0])

    def test_client_falls_back_to_cli_next_to_active_python(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            bin_dir.mkdir()
            python_path = bin_dir / "python"
            kaggle_path = bin_dir / "kaggle"
            python_path.touch()
            kaggle_path.touch()
            completed = SimpleNamespace(returncode=0, stdout='{"numAllowedNow":4}', stderr="")
            with patch("biohub_lab.kaggle.shutil.which", return_value=None), patch(
                "biohub_lab.kaggle.sys.executable", str(python_path)
            ), patch("biohub_lab.kaggle.subprocess.run", return_value=completed) as run:
                allowed = KaggleClient("biohub-cell-tracking-during-development").remaining_submission_limit()

        self.assertEqual(allowed, 4)
        self.assertEqual(run.call_args.args[0][0], str(kaggle_path))

    def test_client_falls_back_to_cli_when_active_python_is_a_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv_bin = root / "venv" / "bin"
            runtime_bin = root / "runtime" / "bin"
            venv_bin.mkdir(parents=True)
            runtime_bin.mkdir(parents=True)
            runtime_python = runtime_bin / "python"
            python_path = venv_bin / "python"
            kaggle_path = venv_bin / "kaggle"
            runtime_python.touch()
            python_path.symlink_to(runtime_python)
            kaggle_path.touch()
            completed = SimpleNamespace(returncode=0, stdout='{"numAllowedNow":3}', stderr="")
            with patch("biohub_lab.kaggle.shutil.which", return_value=None), patch(
                "biohub_lab.kaggle.sys.executable", str(python_path)
            ), patch("biohub_lab.kaggle.sys.prefix", str(root / "venv")), patch(
                "biohub_lab.kaggle.subprocess.run", return_value=completed
            ) as run:
                allowed = KaggleClient("biohub-cell-tracking-during-development").remaining_submission_limit()

        self.assertEqual(allowed, 3)
        self.assertEqual(run.call_args.args[0][0], str(kaggle_path))

    def test_client_requires_explicit_live_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "missing.csv"
            with self.assertRaises(PermissionError):
                KaggleClient("biohub-cell-tracking-during-development").submit(artifact, "test", live=False)

    def test_submit_plan_is_dry_run_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            state = Path(directory) / "state.json"
            plan.write_text('{"competition":"biohub-cell-tracking-during-development","limit":5,"candidates":[{"experiment":"a","artifact":"a.csv"}]}')

            result = submit_plan(plan, state, live=False)

            self.assertEqual(result[0]["status"], "dry_run")
            self.assertFalse(state.exists())

    def test_live_plan_refuses_before_reserving_without_environment_consent(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            state = Path(directory) / "state.json"
            plan.write_text('{"competition":"biohub-cell-tracking-during-development","limit":5,"candidates":[{"experiment":"a","artifact":"a.csv"}]}')

            with self.assertRaises(PermissionError):
                submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "0"})

            self.assertFalse(state.exists())

    def test_live_plan_refuses_before_persisted_four_hour_window_expires(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            state = root / "state.json"
            artifact = root / "submission.csv"
            artifact.write_text((ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "window_started_at": datetime.now(timezone.utc).isoformat(),
                        "window_seconds": 4 * 60 * 60,
                        "candidates": [{"experiment": "a", "family": "classical", "artifact": str(artifact)}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(PermissionError):
                submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

            self.assertFalse(state.exists())

    def test_live_plan_never_submits_smoke_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            state = Path(directory) / "state.json"
            plan.write_text('{"competition":"biohub-cell-tracking-during-development","limit":5,"candidates":[{"experiment":"smoke","family":"smoke","artifact":"smoke.csv"}]}')

            with patch.object(KaggleClient, "remaining_submission_limit", return_value=5):
                result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

            self.assertEqual(result[0]["status"], "blocked_smoke_candidate")
            self.assertFalse(state.exists())

    def test_live_plan_never_submits_explicitly_non_live_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            state = Path(directory) / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
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

            result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

        self.assertEqual(result[0]["status"], "blocked_non_live_candidate")
        self.assertFalse(state.exists())

    def test_live_plan_skips_an_experiment_already_attempted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            state = root / "state.json"
            artifact = root / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan.write_text(
                '{"competition":"biohub-cell-tracking-during-development","limit":5,'
                '"candidates":[{"experiment":"a","family":"classical",'
                '"artifact":"%s"}]}' % str(artifact)
            )
            SubmissionBudget(state, daily_limit=5).record(
                "a", datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc), status="submitted"
            )

            with patch.object(KaggleClient, "remaining_submission_limit", return_value=5):
                result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

        self.assertEqual(result[0]["status"], "already_attempted")

    def test_client_uses_installed_submit_contract_and_extracts_public_result_without_logging_token(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "submission.csv"
            artifact.write_text((ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
            completed = SimpleNamespace(
                returncode=0,
                stdout="Submission ref: 123\nPublic Score: 0.4567\nsecret-token",
                stderr="",
            )
            with patch("biohub_lab.kaggle.shutil.which", return_value="/usr/bin/kaggle"), patch(
                "biohub_lab.kaggle.subprocess.run", return_value=completed
            ) as run:
                result = KaggleClient("biohub-cell-tracking-during-development").submit(
                    artifact,
                    "candidate",
                    wait_seconds=77,
                    live=True,
                    env={
                        "BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1",
                        "KAGGLE_API_TOKEN": "secret-token",
                    },
                )

        command = run.call_args.args[0]
        self.assertNotIn("--wait", command)
        self.assertNotIn("77", command)
        self.assertEqual(result["reference"], "123")
        self.assertEqual(result["public_score"], 0.4567)
        self.assertNotIn("secret-token", result["output"])

    def test_live_plan_can_retry_explicitly_released_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text((ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [{"experiment": "candidate", "family": "classical", "artifact": str(artifact)}],
                    }
                ),
                encoding="utf-8",
            )
            SubmissionBudget(state, daily_limit=5).record("candidate", status="failed")

            with patch.object(KaggleClient, "remaining_submission_limit", return_value=5), patch.object(
                KaggleClient,
                "submit",
                return_value={"returncode": 0, "reference": "123", "output": "ok"},
            ):
                result = submit_plan(
                    plan,
                    state,
                    live=True,
                    retry_failed=True,
                    env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"},
                )

        self.assertEqual(result[0]["status"], "submitted")

    def test_live_plan_reserves_at_most_five_real_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text((ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [
                            {"experiment": "candidate-{}".format(index), "family": "classical", "artifact": str(artifact)}
                            for index in range(6)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                KaggleClient,
                "submit",
                return_value={"returncode": 0, "reference": "123", "public_score": 0.4, "output": "ok"},
            ) as submit:
                with patch.object(KaggleClient, "remaining_submission_limit", return_value=5):
                    result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

            payload = json.loads(state.read_text(encoding="utf-8"))

        self.assertEqual(submit.call_count, 5)
        self.assertEqual(sum(item["status"] == "submitted" for item in result), 5)
        self.assertEqual(result[-1]["status"], "budget_exhausted")
        self.assertEqual(len(payload["records"]), 5)

    def test_live_plan_uses_lower_remote_allowance_as_the_effective_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text((ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"), encoding="utf-8")
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [
                            {"experiment": "candidate-{}".format(index), "family": "classical", "artifact": str(artifact)}
                            for index in range(4)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(KaggleClient, "remaining_submission_limit", return_value=2), patch.object(
                KaggleClient,
                "submit",
                return_value={"returncode": 0, "reference": "123", "public_score": 0.4, "output": "ok"},
            ) as submit:
                result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

        self.assertEqual(submit.call_count, 2)
        self.assertEqual(sum(item["status"] == "submitted" for item in result), 2)
        self.assertEqual(result[-1]["status"], "budget_exhausted")

    def test_live_plan_can_require_all_candidates_before_reserving_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "require_all_candidates": True,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [
                            {
                                "experiment": "candidate-{}".format(index),
                                "family": "classical",
                                "artifact": str(artifact),
                            }
                            for index in range(2)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(KaggleClient, "remaining_submission_limit", return_value=1), patch.object(
                KaggleClient,
                "submit",
                return_value={"returncode": 0, "reference": "123", "public_score": 0.4, "output": "ok"},
            ) as submit:
                result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

        self.assertEqual(submit.call_count, 0)
        self.assertEqual([item["status"] for item in result], ["blocked_remote_budget", "blocked_remote_budget"])

    def test_live_plan_adds_remote_remaining_slots_to_existing_local_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "submission.csv"
            artifact.write_text(
                (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            state = root / "state.json"
            plan.write_text(
                json.dumps(
                    {
                        "competition": "biohub-cell-tracking-during-development",
                        "limit": 5,
                        "window_started_at": "2020-01-01T00:00:00+00:00",
                        "window_seconds": 1,
                        "candidates": [
                            {
                                "experiment": "candidate-{}".format(index),
                                "family": "classical",
                                "artifact": str(artifact),
                            }
                            for index in range(4)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            SubmissionBudget(state, daily_limit=5).record("already-submitted", status="submitted")
            with patch.object(KaggleClient, "remaining_submission_limit", return_value=4), patch.object(
                KaggleClient,
                "submit",
                return_value={"returncode": 0, "reference": "123", "public_score": 0.4, "output": "ok"},
            ) as submit:
                result = submit_plan(plan, state, live=True, env={"BIOHUB_ALLOW_LIVE_SUBMISSIONS": "1"})

        self.assertEqual(submit.call_count, 4)
        self.assertEqual(sum(item["status"] == "submitted" for item in result), 4)


if __name__ == "__main__":
    unittest.main()
