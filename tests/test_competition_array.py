import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from biohub_lab.competition_array import open_competition_array


class CompetitionArrayTests(unittest.TestCase):
    def test_reads_a_zarr_v3_frame_without_importing_zarr(self):
        try:
            import blosc2
        except ImportError as exc:  # pragma: no cover - the project ML env has blosc2.
            self.skipTest("blosc2 is required for the fallback reader: {}".format(exc))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sample.zarr"
            array_root = root / "0"
            (array_root / "c").mkdir(parents=True)
            (root / "zarr.json").write_text(
                json.dumps(
                    {
                        "zarr_format": 3,
                        "node_type": "group",
                        "attributes": {
                            "multiscales": [
                                {
                                    "datasets": [
                                        {
                                            "coordinateTransformations": [
                                                {"type": "scale", "scale": [1, 1, 1, 1]}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            (array_root / "zarr.json").write_text(
                json.dumps(
                    {
                        "zarr_format": 3,
                        "node_type": "array",
                        "shape": [2, 2, 3, 4],
                        "data_type": "uint16",
                        "chunk_grid": {"configuration": {"chunk_shape": [1, 2, 3, 4]}},
                        "chunk_key_encoding": {"configuration": {"separator": "/"}},
                        "fill_value": 0,
                        "codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "blosc", "configuration": {"typesize": 2}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            expected = np.arange(48, dtype=np.uint16).reshape(2, 2, 3, 4)
            for time_index in range(2):
                compressed = blosc2.compress(
                    expected[time_index : time_index + 1].tobytes(),
                    typesize=2,
                    codec=blosc2.Codec.ZSTD,
                    filter=blosc2.Filter.BITSHUFFLE,
                )
                chunk_path = array_root / "c" / str(time_index) / "0" / "0" / "0"
                chunk_path.parent.mkdir(parents=True, exist_ok=True)
                chunk_path.write_bytes(compressed)

            opened = open_competition_array(root)
            actual = opened.array[1].copy()
            attrs = opened.attrs

        self.assertEqual(opened.array.shape, (2, 2, 3, 4))
        np.testing.assert_array_equal(actual, expected[1])
        self.assertEqual(attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"], [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
