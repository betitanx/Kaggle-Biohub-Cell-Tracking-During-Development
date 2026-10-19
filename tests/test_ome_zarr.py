import tempfile
import unittest
from pathlib import Path

from biohub_lab.ome_zarr import (
    DEFAULT_SCALE,
    discover_zarr_datasets,
    parse_ome_scale,
    resolve_zarr_path,
)


class OmeZarrTests(unittest.TestCase):
    def test_discovers_zarr_datasets_and_can_require_geff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_a.zarr").mkdir()
            (root / "train_a.geff").mkdir()
            (root / "test_b.zarr").mkdir()
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(
                discover_zarr_datasets(root, require_geff=False),
                [root / "test_b.zarr", root / "train_a.zarr"],
            )
            self.assertEqual(
                discover_zarr_datasets(root, require_geff=True),
                [root / "train_a.zarr"],
            )

    def test_resolves_dataset_stem_and_parses_ngff_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample"
            (root / "sample.zarr").mkdir()

            self.assertEqual(resolve_zarr_path(path), root / "sample.zarr")
            self.assertEqual(resolve_zarr_path(root / "sample.zarr"), root / "sample.zarr")

        attrs = {
            "multiscales": [
                {
                    "datasets": [
                        {
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1, 1.625, 0.40625, 0.40625]}
                            ]
                        }
                    ]
                }
            ]
        }
        self.assertEqual(parse_ome_scale(attrs), (1.625, 0.40625, 0.40625))
        self.assertEqual(parse_ome_scale({}), DEFAULT_SCALE)


if __name__ == "__main__":
    unittest.main()
