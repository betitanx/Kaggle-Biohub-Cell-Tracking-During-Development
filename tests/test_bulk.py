import tempfile
import unittest
import zipfile
from pathlib import Path

from biohub_lab.bulk import extract_competition_archive


class BulkArchiveTests(unittest.TestCase):
    def test_extracts_expected_files_and_skips_existing_matching_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.zip"
            output = root / "data"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("sample_submission.csv", "id\n")
                handle.writestr("test/dataset.zarr/zarr.json", "{}")
                handle.writestr("train/dataset.geff/nodes/zarr.json", "{}")

            expected = {
                "test/dataset.zarr/zarr.json": 2,
                "train/dataset.geff/nodes/zarr.json": 2,
            }
            first = extract_competition_archive(archive, output, expected)
            second = extract_competition_archive(archive, output, expected)

            self.assertEqual(first["extracted"], 2)
            self.assertEqual(first["skipped"], 0)
            self.assertEqual(second["extracted"], 0)
            self.assertEqual(second["skipped"], 2)
            self.assertEqual((output / "test/dataset.zarr/zarr.json").read_text(), "{}")

    def test_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("test/../outside.txt", "bad")

            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                extract_competition_archive(
                    archive,
                    root / "data",
                    {"test/../outside.txt": 3},
                )

    def test_rejects_missing_or_mismatched_manifest_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bundle.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("test/dataset.zarr/zarr.json", "{}")

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                extract_competition_archive(
                    archive,
                    root / "data",
                    {"test/dataset.zarr/zarr.json": 3},
                )

            with self.assertRaisesRegex(ValueError, "missing archive members"):
                extract_competition_archive(
                    archive,
                    root / "data-2",
                    {
                        "test/dataset.zarr/zarr.json": 2,
                        "train/missing.geff/zarr.json": 2,
                    },
                )


if __name__ == "__main__":
    unittest.main()
