import tempfile
import unittest
from pathlib import Path

from biohub_lab.validation_subset import prepare_validation_subset


class ValidationSubsetTests(unittest.TestCase):
    def test_subset_is_deterministic_and_pairs_zarr_with_geff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train"
            target = root / "subset"
            source.mkdir()
            for name in ("b", "a", "c"):
                (source / (name + ".zarr")).mkdir()
                (source / (name + ".geff")).write_text(name, encoding="utf-8")

            manifest = prepare_validation_subset(source, target, limit=2)
            self.assertEqual(manifest["dataset_names"], ["a", "b"])
            self.assertTrue((target / "a.zarr").is_symlink())
            self.assertTrue((target / "b.geff").is_symlink())
            self.assertEqual(manifest["ground_truth_count"], 2)
            (target / "a.geff").unlink()
            repaired = prepare_validation_subset(source, target, limit=2)
            self.assertTrue((target / "a.geff").is_symlink())
            self.assertEqual(repaired["ground_truth_count"], 2)


if __name__ == "__main__":
    unittest.main()
