exit=0
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_subset import _list_prefix, _run_kaggle, download_prefix


class DownloadSubsetTests(unittest.TestCase):
    def test_kaggle_command_retries_transient_rate_limit(self):
        transient = subprocess.CompletedProcess(
            args=["kaggle"], returncode=1, stdout="", stderr="429 Too Many Requests"
        )
        success = subprocess.CompletedProcess(
            args=["kaggle"], returncode=0, stdout="ok", stderr=""
        )

        with patch("scripts.download_subset.subprocess.run", side_effect=[transient, success]), patch(
            "scripts.download_subset.time.sleep"
        ) as sleep:
            completed = _run_kaggle(["kaggle", "competitions", "files"])

        self.assertEqual(completed.stdout, "ok")
        sleep.assert_called_once()

    def test_kaggle_command_retries_a_timeout(self):
        success = subprocess.CompletedProcess(
            args=["kaggle"], returncode=0, stdout="ok", stderr=""
        )

        with patch(
            "scripts.download_subset.subprocess.run",
            side_effect=[subprocess.TimeoutExpired(["kaggle"], 1), success],
        ) as run, patch("scripts.download_subset.time.sleep") as sleep:
            completed = _run_kaggle(["kaggle"], timeout_seconds=1)

        self.assertEqual(completed.stdout, "ok")
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 1)
        sleep.assert_called_once()

    def test_list_prefix_accepts_json_followed_by_pager_token(self):
        completed = subprocess.CompletedProcess(
            args=["kaggle"],
            returncode=0,
            stdout='[{"name":"test/embryo.zarr/zarr.json","size":12}]\nNext Page Token = opaque-token\n',
            stderr="",
        )

        with patch("scripts.download_subset.subprocess.run", return_value=completed):
            files = _list_prefix("competition", "test/embryo.zarr/", "kaggle")

        self.assertEqual(files, [{"name": "test/embryo.zarr/zarr.json", "size": 12}])

    def test_download_prefix_uses_cached_manifest_without_relisting(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"name": "test/a.zarr/zarr.json", "size": 2},
                            {"name": "test/a.zarr/0/c/0", "size": 1},
                            {"name": "test/aa.zarr/zarr.json", "size": 3},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "scripts.download_subset._list_prefix",
                side_effect=AssertionError("manifest should avoid relisting"),
            ):
                result = download_prefix(
                    "competition",
                    "test/a.zarr/",
                    Path(directory) / "data",
                    dry_run=True,
                    manifest_path=manifest,
                )

        self.assertEqual(result["files"], 2)
        self.assertEqual(result["bytes"], 3)

    def test_download_prefix_uses_bounded_parallel_file_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "entries": [
                            {"name": "test/a.zarr/one.bin", "size": 3},
                            {"name": "test/a.zarr/two.bin", "size": 4},
                            {"name": "test/a.zarr/zarr.json", "size": 5},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            def fake_download(command, **_kwargs):
                file_name = command[command.index("--file") + 1]
                temporary = Path(command[command.index("--path") + 1])
                payload = bytes([len(file_name)]) * int(
                    {"one.bin": 3, "two.bin": 4, "zarr.json": 5}[Path(file_name).name]
                )
                (temporary / Path(file_name).name).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("scripts.download_subset.subprocess.run", side_effect=fake_download):
                result = download_prefix(
                    "competition",
                    "test/a.zarr/",
                    Path(directory) / "data",
                    manifest_path=manifest,
                    workers=2,
                )

        self.assertEqual(result["files"], 3)
        self.assertEqual(result["bytes"], 12)
        self.assertEqual(result["downloaded"], 3)

    def test_download_prefix_refuses_non_positive_worker_count(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            download_prefix("competition", "test/a.zarr/", Path("data"), workers=0)

    def test_download_prefix_refuses_non_positive_timeout(self):
        with self.assertRaisesRegex(ValueError, "timeout_seconds"):
            download_prefix("competition", "test/a.zarr/", Path("data"), timeout_seconds=0)

    def test_list_prefix_follows_page_token_until_root_marker(self):
        first_page = subprocess.CompletedProcess(
            args=["kaggle"],
            returncode=0,
            stdout='[{"name":"test/embryo.zarr/0/c/0","size":4}]\nNext Page Token = next-page\n',
            stderr="",
        )
        second_page = subprocess.CompletedProcess(
            args=["kaggle"],
            returncode=0,
            stdout='[{"name":"test/embryo.zarr/zarr.json","size":12}]\n',
            stderr="",
        )

        with patch(
            "scripts.download_subset.subprocess.run",
            side_effect=[first_page, second_page],
        ):
            files = _list_prefix("competition", "test/embryo.zarr/", "kaggle")

        self.assertEqual([item["name"] for item in files], [
            "test/embryo.zarr/0/c/0",
            "test/embryo.zarr/zarr.json",
        ])


if __name__ == "__main__":
    unittest.main()
