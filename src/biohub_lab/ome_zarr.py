"""Small, lazy OME-Zarr helpers for candidate adapters.

The control plane does not import Zarr at startup.  A real-data run receives a
clear dependency error if the optional ``zarr`` package is missing, while
tests can still exercise dataset discovery and metadata parsing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


DEFAULT_SCALE = (1.625, 0.40625, 0.40625)


@dataclass
class OpenedOmeZarr:
    """An opened image array plus the metadata needed by a candidate."""

    path: Path
    array: Any
    attrs: Dict[str, Any]
    scale: Tuple[float, float, float]


def resolve_zarr_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix == ".zarr":
        resolved = path
    else:
        resolved = path.with_suffix(".zarr")
    if not resolved.exists():
        raise FileNotFoundError("OME-Zarr dataset not found: {}".format(resolved))
    return resolved


def discover_zarr_datasets(data_dir: Path, require_geff: bool = False) -> List[Path]:
    """Return sorted ``.zarr`` paths, optionally requiring sibling ``.geff``."""

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError("data directory not found: {}".format(data_dir))
    datasets: List[Path] = []
    for path in sorted(data_dir.glob("*.zarr")):
        if not require_geff or path.with_suffix(".geff").exists():
            datasets.append(path)
    return datasets


def parse_ome_scale(attrs: Mapping[str, Any]) -> Tuple[float, float, float]:
    """Extract the final three NGFF scale values as ``(Z, Y, X)``."""

    multiscales = attrs.get("multiscales")
    if not multiscales:
        return DEFAULT_SCALE
    try:
        transform = multiscales[0]["datasets"][0]["coordinateTransformations"][0]
        if transform.get("type") != "scale":
            raise ValueError("OME-NGFF coordinate transform must be type=scale")
        scale = tuple(float(value) for value in transform["scale"][-3:])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid OME-NGFF multiscales metadata") from exc
    if len(scale) != 3 or any(value <= 0 for value in scale):
        raise ValueError("OME-NGFF scale must contain three positive values")
    return scale  # type: ignore[return-value]


def open_ome_zarr(path: Path, array_key: str = "0") -> OpenedOmeZarr:
    """Open an OME-Zarr array without loading image data into memory."""

    zarr_path = resolve_zarr_path(path)
    try:
        import zarr  # type: ignore
    except ImportError:
        from .competition_array import open_competition_array

        fallback = open_competition_array(zarr_path, array_key=array_key)
        if len(tuple(fallback.array.shape)) != 4:
            raise ValueError(
                "Biohub images must have shape (T, Z, Y, X); got {}".format(
                    tuple(fallback.array.shape)
                )
            )
        return OpenedOmeZarr(
            path=zarr_path,
            array=fallback.array,
            attrs=fallback.attrs,
            scale=parse_ome_scale(fallback.attrs),
        )

    root = zarr.open(str(zarr_path), mode="r")
    attrs = dict(getattr(root, "attrs", {}))
    if hasattr(root, "shape"):
        array = root
    else:
        try:
            array = root[array_key]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "OME-Zarr group {} does not contain array key {!r}".format(zarr_path, array_key)
            ) from exc
        attrs = dict(getattr(root, "attrs", {}))
    if len(tuple(array.shape)) != 4:
        raise ValueError(
            "Biohub images must have shape (T, Z, Y, X); got {}".format(tuple(array.shape))
        )
    return OpenedOmeZarr(
        path=zarr_path,
        array=array,
        attrs=attrs,
        scale=parse_ome_scale(attrs),
    )
