import json
import tempfile
import unittest
from pathlib import Path

from biohub_lab.local_cache import prepare_data_cache


class LocalCacheTests(unittest.TestCase):
    def test_copies_raw_data_and_recreates_split_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for split in ("test", "train", "train_split", "holdout"):
                (root / split).mkdir(parents=True)
            for name in ("a", "b"):
                (root / "train" / (name + ".zarr")).mkdir()
                (root / "train" / (name + ".zarr") / "0").write_text("image", encoding="utf-8")
                (root / "train" / (name + ".geff")).mkdir()
            (root / "test" / "sample.zarr").mkdir()
            (root / "train_split" / "a.zarr").symlink_to(Path("../train/a.zarr"), target_is_directory=True)
            (root / "train_split" / "a.geff").symlink_to(Path("../train/a.geff"), target_is_directory=True)
            (root / "holdout" / "b.zarr").symlink_to(Path("../train/b.zarr"), target_is_directory=True)
            (root / "holdout" / "b.geff").symlink_to(Path("../train/b.geff"), target_is_directory=True)
            (root / "holdout" / "manifest.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
            cache = prepare_data_cache(root, Path(directory) / "cache")

            self.assertEqual(set(cache), {"root", "test", "train_split", "holdout"})
            self.assertTrue((cache["test"] / "sample.zarr").is_dir())
            self.assertTrue((cache["train_split"] / "a.zarr").is_symlink())
            self.assertTrue((cache["train_split"] / "a.zarr").is_dir())
            self.assertTrue((cache["holdout"] / "b.geff").is_symlink())
            self.assertEqual(
                (cache["holdout"] / "manifest.json").read_text(encoding="utf-8"),
                json.dumps({"seed": 1}),
            )
