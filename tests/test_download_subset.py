exit=0
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import Barrier, Lock
from unittest.mock import patch

import pytest

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
            directory = Path(directory).resolve()
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
            directory = Path(directory).resolve()
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


@pytest.fixture
def download_tree(tmp_path):
    # Use physical temporary parents: macOS /var is itself a symlink.
    return tmp_path.resolve()


def fake_file_download(command, **_kwargs):
    temporary = Path(command[command.index("--path") + 1])
    (temporary / "payload.bin").write_bytes(b"new")
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


@pytest.mark.parametrize("prefix", [
    "", "/", "/test/a.zarr/", "../escape/", "test/../a.zarr/",
    "test/./a.zarr/", "test//a.zarr/", "test/a.zarr//", "test\\a.zarr\\",
    "C:/test/a.zarr/", "C:test/a.zarr/", "test/C:escape/", "test/a.zarr:stream/",
    "test/a\x00.zarr/",
])
def test_invalid_prefix_is_rejected_before_listing_or_mutation(download_tree, prefix):
    output = download_tree / "not-created"
    with patch("scripts.download_subset._list_prefix") as listing, patch(
        "scripts.download_subset._run_kaggle"
    ) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", prefix, output)
    listing.assert_not_called()
    run.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("name", [
    "/test/a.zarr/escape.bin", "test/a.zarr/../../../escape.bin",
    "test/a.zarr/nested/../escape.bin", "test/a.zarr/./escape.bin",
    "test/a.zarr/..\\escape.bin", "test/a.zarr/C:escape.bin",
    "test/a.zarr/C:/escape.bin", "test/a.zarr/name:stream", "test/a.zarr//escape.bin", "test/a.zarr/",
    "test/a.zarr/bad\x00.bin",
])
def test_all_cached_names_are_validated_before_mutation(download_tree, name, dry_run):
    manifest = download_tree / "manifest.json"
    manifest.write_text(json.dumps({"entries": [
        {"name": "test/a.zarr/000-valid.bin", "size": 3},
        {"name": name, "size": 3},
    ]}), encoding="utf-8")
    output = download_tree / "not-created"
    with patch("scripts.download_subset._run_kaggle", side_effect=fake_file_download) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output,
                            manifest_path=manifest, workers=1, dry_run=dry_run)
    run.assert_not_called()
    assert not output.exists()
    assert not (download_tree / "escape.bin").exists()


@pytest.mark.parametrize("name", [
    "test/a.zarr/../../../escape.bin", "/test/a.zarr/escape.bin",
    "C:/escape.bin", "C:escape.bin", "\\\\server\\share\\escape.bin",
    "test/aa.zarr/escape.bin", "", None, 123,
])
def test_all_upstream_names_are_validated_before_mutation(download_tree, name):
    files = [{"name": "test/a.zarr/valid.bin", "size": 3}, {"name": name, "size": 3}]
    output = download_tree / "not-created"
    with patch("scripts.download_subset._list_prefix", return_value=files), patch(
        "scripts.download_subset._run_kaggle", side_effect=fake_file_download
    ) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output, workers=1)
    run.assert_not_called()
    assert not output.exists()


def test_real_listing_traversal_is_rejected_before_download(download_tree):
    page = subprocess.CompletedProcess([], 0, stdout=json.dumps([
        {"name": "test/a.zarr/../../../escape.bin", "size": 3},
        {"name": "test/a.zarr/zarr.json", "size": 3},
    ]), stderr="")

    def listing_or_download(command, **kwargs):
        if "--file" in command:
            return fake_file_download(command, **kwargs)
        return page

    output = download_tree / "not-created"
    with patch("scripts.download_subset._run_kaggle", side_effect=listing_or_download) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output)
    assert run.call_count == 1
    assert not output.exists()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("location", ["root", "ancestor", "parent", "destination"])
@pytest.mark.parametrize("dangling", [False, True])
def test_symlinks_are_rejected_before_any_download(download_tree, location, dangling, dry_run):
    outside = download_tree / "outside"
    if not dangling:
        outside.mkdir()
    output = download_tree / "output"
    if location == "root":
        link = output
    elif location == "ancestor":
        link = download_tree / "linked-parent"
        output = link / "new-output"
    elif location == "parent":
        link = output / "test" / "a.zarr"
    else:
        link = output / "test" / "a.zarr" / "zarr.json"
        if not dangling:
            outside.rmdir()
            outside.write_bytes(b"old")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    files = [{"name": "test/a.zarr/000-valid.bin", "size": 3},
             {"name": "test/a.zarr/zarr.json", "size": 3}]
    with patch("scripts.download_subset._list_prefix", return_value=files), patch(
        "scripts.download_subset._run_kaggle", side_effect=fake_file_download
    ) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output, workers=1, dry_run=dry_run)
    run.assert_not_called()
    assert link.is_symlink()
    if location == "destination" and not dangling:
        assert outside.read_bytes() == b"old"
    elif not dangling:
        assert list(outside.iterdir()) == []
    else:
        assert not outside.exists()
    if location == "destination":
        assert not link.with_name("000-valid.bin").exists()


@pytest.mark.parametrize("obstruction", ["directory", "parent-file"])
def test_invalid_destination_is_rejected_before_other_files_download(download_tree, obstruction):
    output = download_tree / "output"
    destination = output / "test/a.zarr/zarr.json"
    if obstruction == "directory":
        destination.mkdir(parents=True)
    else:
        destination.parent.parent.mkdir(parents=True)
        destination.parent.write_bytes(b"old")
    files = [{"name": "test/a.zarr/000-valid.bin", "size": 3},
             {"name": "test/a.zarr/zarr.json", "size": 3}]
    with patch("scripts.download_subset._list_prefix", return_value=files), patch(
        "scripts.download_subset._run_kaggle", side_effect=fake_file_download
    ) as run:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output, workers=1)
    run.assert_not_called()


@pytest.mark.parametrize("change", ["directory", "symlink", "dangling", "parent-symlink", "root-symlink"])
def test_destination_is_rechecked_before_final_move(download_tree, change):
    output = download_tree / "output"
    destination = output / "test/a.zarr/zarr.json"
    outside = download_tree / "outside"
    if change != "dangling":
        if change.endswith("-symlink"):
            outside.mkdir()
        else:
            outside.write_bytes(b"old")

    def change_destination(command, **kwargs):
        result = fake_file_download(command, **kwargs)
        if change == "directory":
            destination.mkdir()
        elif change == "parent-symlink":
            destination.parent.rmdir()
            destination.parent.symlink_to(outside)
        elif change == "root-symlink":
            output.rename(download_tree / "original-output")
            output.symlink_to(outside)
        else:
            destination.symlink_to(outside)
        return result

    with patch("scripts.download_subset._list_prefix", return_value=[
        {"name": "test/a.zarr/zarr.json", "size": 3}
    ]), patch("scripts.download_subset._run_kaggle", side_effect=change_destination), patch(
        "scripts.download_subset.shutil.move"
    ) as move:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output, workers=1)
    move.assert_not_called()
    if change == "symlink":
        assert outside.read_bytes() == b"old"
    elif change == "dangling":
        assert not outside.exists()
    elif change.endswith("-symlink"):
        assert list(outside.iterdir()) == []


@pytest.mark.parametrize("prefix", ["test/a.zarr", "test/a.zarr/"])
def test_dry_run_validates_without_creating_output(download_tree, prefix):
    output = download_tree / "absent-parent" / "output"
    with patch("scripts.download_subset._list_prefix", return_value=[
        {"name": "test/a.zarr/zarr.json", "size": 3}
    ]), patch("scripts.download_subset._run_kaggle") as run:
        result = download_prefix("competition", prefix, output, dry_run=True)
    assert result == {"prefix": prefix, "files": 1, "bytes": 3, "downloaded": 0}
    run.assert_not_called()
    assert not output.parent.exists()


@pytest.mark.parametrize("dangling", [False, True])
def test_downloaded_candidate_symlink_is_never_published(download_tree, dangling):
    outside = download_tree / "outside.bin"
    if not dangling:
        outside.write_bytes(b"old")
    output = download_tree / "output"

    def symlink_download(command, **_kwargs):
        temporary = Path(command[command.index("--path") + 1])
        (temporary / "payload.bin").symlink_to(outside)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with patch("scripts.download_subset._list_prefix", return_value=[
        {"name": "test/a.zarr/zarr.json", "size": 3}
    ]), patch("scripts.download_subset._run_kaggle", side_effect=symlink_download), patch(
        "scripts.download_subset.shutil.move", wraps=shutil.move
    ) as move:
        with pytest.raises(ValueError):
            download_prefix("competition", "test/a.zarr/", output)
    move.assert_not_called()
    assert not (output / "test/a.zarr/zarr.json").is_symlink()
    if dangling:
        assert not outside.exists()
    else:
        assert outside.read_bytes() == b"old"


def test_resume_skips_complete_files_and_replaces_incomplete_regular_files(download_tree):
    output = download_tree / "output"
    parent = output / "test/a.zarr"
    parent.mkdir(parents=True)
    (parent / "complete.bin").write_bytes(b"old")
    (parent / "unknown-size.bin").write_bytes(b"keep")
    (parent / "partial.bin").write_bytes(b"x")
    files = [{"name": "test/a.zarr/complete.bin", "size": 3},
             {"name": "test/a.zarr/unknown-size.bin", "size": 0},
             {"name": "test/a.zarr/partial.bin", "size": 3}]
    with patch("scripts.download_subset._list_prefix", return_value=files), patch(
        "scripts.download_subset._run_kaggle", side_effect=fake_file_download
    ) as run:
        result = download_prefix("competition", "test/a.zarr/", output)
    assert result["downloaded"] == 1
    assert run.call_count == 1
    assert (parent / "complete.bin").read_bytes() == b"old"
    assert (parent / "unknown-size.bin").read_bytes() == b"keep"
    assert (parent / "partial.bin").read_bytes() == b"new"


def test_parallel_downloads_respect_worker_limit(download_tree):
    barrier = Barrier(2, timeout=5)
    lock = Lock()
    active = peak = 0

    def simultaneous_download(command, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            return fake_file_download(command, **kwargs)
        finally:
            with lock:
                active -= 1

    files = [{"name": f"test/a.zarr/{index}.bin", "size": 3} for index in range(6)]
    with patch("scripts.download_subset._list_prefix", return_value=files), patch(
        "scripts.download_subset._run_kaggle", side_effect=simultaneous_download
    ):
        result = download_prefix("competition", "test/a.zarr/", download_tree / "output", workers=2)
    assert peak == 2
    assert result["downloaded"] == 6


if __name__ == "__main__":
    unittest.main()
