import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biohub_lab.local_cache import prepare_data_cache


class LocalCacheTests(unittest.TestCase):
    def test_refuses_existing_cache_without_deleting_its_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for split in ("test", "train", "train_split", "holdout"):
                (root / split).mkdir(parents=True)
            cache = Path(directory) / "cache"
            cache.mkdir()
            sentinel = cache / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with patch("biohub_lab.local_cache.shutil.rmtree", side_effect=AssertionError("unsafe deletion")):
                with self.assertRaises(FileExistsError):
                    prepare_data_cache(root, cache)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_refuses_cache_symlinks_including_dangling_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for split in ("test", "train", "train_split", "holdout"):
                (root / split).mkdir(parents=True)
            sentinel = root / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            for name, target in (("source-link", root), ("dangling-link", root / "missing")):
                with self.subTest(name=name):
                    link = Path(directory) / name
                    link.symlink_to(target, target_is_directory=True)
                    with patch("biohub_lab.local_cache.shutil.rmtree", side_effect=AssertionError("unsafe deletion")):
                        with self.assertRaises(ValueError):
                            prepare_data_cache(root, link)
                    self.assertTrue(link.is_symlink())
                    self.assertEqual(sentinel.read_text(), "keep")

    def test_refuses_overlapping_source_and_cache_in_both_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for split in ("test", "train", "train_split", "holdout"):
                (root / split).mkdir(parents=True)
            for cache in (root, root / "new-cache", root.parent):
                with self.subTest(cache=cache):
                    with patch("biohub_lab.local_cache.shutil.rmtree", side_effect=AssertionError("unsafe deletion")):
                        with self.assertRaises(ValueError):
                            prepare_data_cache(root, cache)
            self.assertFalse((root / "new-cache").exists())

    def test_validates_source_before_creating_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            (root / "test").mkdir(parents=True)
            cache = Path(directory) / "cache"
            with self.assertRaises(FileNotFoundError):
                prepare_data_cache(root, cache)
            self.assertFalse(cache.exists())

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
