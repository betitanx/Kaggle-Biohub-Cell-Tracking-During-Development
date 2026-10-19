import unittest
from pathlib import Path

from biohub_lab.cloud import (
    RemoteRunConfig,
    is_transient_remote_error,
    normalise_gpu_request,
    remote_config_from_env,
    require_kaggle_api_token,
)


class CloudConfigTests(unittest.TestCase):
    def test_remote_layout_and_command_are_persistent_and_explicit(self):
        config = RemoteRunConfig(
            gpu="H100",
            duration_hours=4.0,
            remote_project_root=Path("/root/biohub"),
            volume_root=Path("/mnt/data"),
        )

        environment = config.environment()
        command = config.command()

        self.assertEqual(environment["TEST_DATA_DIR"], "/mnt/data/data/test")
        self.assertEqual(environment["HOLDOUT_DATA_DIR"], "/mnt/data/data/holdout")
        self.assertEqual(command[0:3], ["/usr/local/bin/python", "-m", "biohub_lab.cli"])
        self.assertIn("/mnt/data/runs", command)
        self.assertNotIn("--live-submit", command)

    def test_command_can_use_ephemeral_candidate_runs(self):
        config = RemoteRunConfig(volume_root=Path("/mnt/data"))

        command = config.command(output_dir=Path("/tmp/biohub-runs"))

        self.assertIn("/tmp/biohub-runs", command)
        self.assertNotIn("/mnt/data/runs", command)

    def test_command_can_use_a_targeted_remote_config(self):
        config = RemoteRunConfig(
            remote_project_root=Path("/root/biohub"),
            config_path=Path("config/canary.json"),
        )

        command = config.command()

        self.assertIn("/root/biohub/config/canary.json", command)

    def test_subset_cannot_enable_live_submission(self):
        with self.assertRaises(ValueError):
            RemoteRunConfig(data_scope="subset", live_submit=True).validate()

    def test_environment_constructor_never_needs_a_secret(self):
        config = remote_config_from_env(
            {
                "BIOHUB_MODAL_GPU": "L40S",
                "BIOHUB_DATA_SCOPE": "full",
                "BIOHUB_REMOTE_DURATION_HOURS": "2.5",
            }
        )

        self.assertEqual(config.gpu, "L40S")
        self.assertEqual(config.duration_hours, 2.5)
        self.assertNotIn("KAGGLE_API_TOKEN", config.environment())

    def test_kaggle_secret_must_contain_a_non_empty_api_token(self):
        with self.assertRaisesRegex(RuntimeError, "KAGGLE_API_TOKEN"):
            require_kaggle_api_token({})

        require_kaggle_api_token({"KAGGLE_API_TOKEN": "opaque-token"})

    def test_transient_remote_errors_include_rate_limit_and_server_errors(self):
        self.assertTrue(is_transient_remote_error("429 Too Many Requests"))
        self.assertTrue(is_transient_remote_error("HTTP 503 from API"))
        self.assertFalse(is_transient_remote_error("401 Authentication required"))

    def test_gpu_request_can_explicitly_select_cpu(self):
        self.assertIsNone(normalise_gpu_request("cpu"))
        self.assertIsNone(normalise_gpu_request("none"))
        self.assertEqual(normalise_gpu_request("L40S"), "L40S")


if __name__ == "__main__":
    unittest.main()
