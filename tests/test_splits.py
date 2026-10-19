import json
import tempfile
import unittest
from pathlib import Path

from biohub_lab.splits import create_group_holdout


class SplitTests(unittest.TestCase):
    def test_creates_deterministic_dataset_level_holdout_without_copying_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train"
            train_split = root / "train_split"
            holdout = root / "holdout"
            source.mkdir()
            for name in ("dataset_a", "dataset_b", "dataset_c", "dataset_d"):
                (source / (name + ".zarr")).mkdir()
                (source / (name + ".geff")).write_text(name, encoding="utf-8")

            manifest = create_group_holdout(source, holdout, fraction=0.25, seed=17, train_dir=train_split)
            payload = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest, payload)
            self.assertEqual(len(manifest["holdout_datasets"]), 1)
            self.assertTrue(set(manifest["holdout_datasets"]).issubset(manifest["source_datasets"]))
            self.assertEqual(
                set(manifest["train_datasets"]) | set(manifest["holdout_datasets"]),
                set(manifest["source_datasets"]),
            )
            chosen = manifest["holdout_datasets"][0]
            self.assertTrue((holdout / (chosen + ".zarr")).is_symlink())
            self.assertTrue((holdout / (chosen + ".geff")).is_symlink())
            self.assertTrue(all((train_split / (name + ".zarr")).is_symlink() for name in manifest["train_datasets"]))
            self.assertTrue(all((train_split / (name + ".geff")).is_symlink() for name in manifest["train_datasets"]))
            self.assertEqual((source / (chosen + ".geff")).read_text(encoding="utf-8"), chosen)

    def test_refuses_to_overwrite_nonempty_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train"
            holdout = root / "holdout"
            source.mkdir()
            holdout.mkdir()
            for name in ("a", "b"):
                (source / (name + ".zarr")).mkdir()
                (source / (name + ".geff")).write_text(name, encoding="utf-8")
            (holdout / "keep.txt").write_text("do not remove", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_group_holdout(source, holdout, fraction=0.5, seed=1)


if __name__ == "__main__":
    unittest.main()
