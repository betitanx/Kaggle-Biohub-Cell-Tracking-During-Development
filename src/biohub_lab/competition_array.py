"""Minimal Zarr v3 reader for Kaggle's compressed competition volumes.

Kaggle's base Notebook image does not always include the optional ``zarr``
package.  The competition volumes use one full ``(Z, Y, X)`` chunk per frame
with the standard bytes+Blosc codec pipeline, so a small NumPy/Blosc2 reader is
enough to keep the actual candidate executable without network installation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


@dataclass
class OpenedCompetitionArray:
    path: Path
    array: "ZarrV3Array"
    attrs: Dict[str, Any]


def _product(values: Tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


class ZarrV3Array:
    """Read integer time frames from a regular Zarr v3 chunk grid."""

    def __init__(self, path: Path, metadata: Mapping[str, Any]):
        try:
            import numpy as np  # type: ignore
            import blosc2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("fallback Zarr reader requires numpy and blosc2") from exc

        self.path = Path(path)
        self._np = np
        self._blosc2 = blosc2
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.chunk_shape = tuple(
            int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
        )
        self.dtype = np.dtype(str(metadata["data_type"]))
        self.fill_value = metadata.get("fill_value", 0)
        self.separator = str(
            metadata.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
        )
        self.codecs = tuple(metadata.get("codecs", ()))
        if len(self.shape) != 4 or len(self.chunk_shape) != 4:
            raise ValueError("Biohub competition arrays must be four-dimensional")
        if any(value <= 0 for value in self.shape + self.chunk_shape):
            raise ValueError("Zarr array and chunk dimensions must be positive")

    def _chunk_path(self, indices: Tuple[int, ...]) -> Path:
        path = self.path / "c"
        for index in indices:
            path = path / str(index)
        return path

    def _decode_chunk(self, raw: bytes, actual_shape: Tuple[int, ...]) -> Any:
        data = raw
        for codec in reversed(self.codecs):
            name = str(codec.get("name", "")).lower() if isinstance(codec, Mapping) else ""
            if name == "blosc":
                data = self._blosc2.decompress(data)
            elif name == "bytes":
                continue
            else:
                raise RuntimeError("unsupported Zarr v3 codec in competition volume: {}".format(name))
        values = self._np.frombuffer(data, dtype=self.dtype)
        full_size = _product(self.chunk_shape)
        actual_size = _product(actual_shape)
        if values.size == full_size:
            return values.reshape(self.chunk_shape)
        if values.size == actual_size:
            return values.reshape(actual_shape)
        raise ValueError(
            "decoded Zarr chunk has {} values; expected {} or {}".format(
                values.size, actual_size, full_size
            )
        )

    def __getitem__(self, index: int) -> Any:
        if not isinstance(index, int):
            raise TypeError("fallback competition reader supports integer time indices only")
        time_index = int(index)
        if time_index < 0:
            time_index += self.shape[0]
        if time_index < 0 or time_index >= self.shape[0]:
            raise IndexError("time index out of bounds: {}".format(index))

        frame_shape = self.shape[1:]
        frame = self._np.full(frame_shape, self.fill_value, dtype=self.dtype)
        time_chunk_index = time_index // self.chunk_shape[0]
        local_time_index = time_index % self.chunk_shape[0]
        chunk_counts = tuple(
            int(math.ceil(self.shape[axis] / float(self.chunk_shape[axis])))
            for axis in range(1, 4)
        )
        for z_chunk in range(chunk_counts[0]):
            for y_chunk in range(chunk_counts[1]):
                for x_chunk in range(chunk_counts[2]):
                    starts = (
                        time_chunk_index * self.chunk_shape[0],
                        z_chunk * self.chunk_shape[1],
                        y_chunk * self.chunk_shape[2],
                        x_chunk * self.chunk_shape[3],
                    )
                    actual_shape = tuple(
                        min(self.chunk_shape[axis], self.shape[axis] - starts[axis])
                        for axis in range(4)
                    )
                    chunk_path = self._chunk_path(
                        (time_chunk_index, z_chunk, y_chunk, x_chunk)
                    )
                    if not chunk_path.is_file():
                        continue
                    chunk = self._decode_chunk(chunk_path.read_bytes(), actual_shape)
                    z0, y0, x0 = starts[1:]
                    z1, y1, x1 = z0 + actual_shape[1], y0 + actual_shape[2], x0 + actual_shape[3]
                    frame[z0:z1, y0:y1, x0:x1] = chunk[
                        local_time_index,
                        : actual_shape[1],
                        : actual_shape[2],
                        : actual_shape[3],
                    ]
        return frame


def open_competition_array(path: Path, array_key: str = "0") -> OpenedCompetitionArray:
    """Open a competition OME-Zarr array without importing the zarr package."""

    root = Path(path)
    root_metadata_path = root / "zarr.json"
    array_root = root / array_key
    if not (array_root / "zarr.json").is_file():
        array_root = root
    array_metadata_path = array_root / "zarr.json"
    if not root_metadata_path.is_file() or not array_metadata_path.is_file():
        raise FileNotFoundError("Zarr v3 metadata not found under {}".format(root))
    root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
    array_metadata = json.loads(array_metadata_path.read_text(encoding="utf-8"))
    return OpenedCompetitionArray(
        path=root,
        array=ZarrV3Array(array_root, array_metadata),
        attrs=dict(root_metadata.get("attributes", {})),
    )
