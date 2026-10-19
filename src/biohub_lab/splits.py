"""Reproducible group-level train/holdout dataset splits."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def _dataset_names(source_dir: Path) -> List[str]:
    names = []
    for image in sorted(source_dir.glob("*.zarr")):
        if image.with_suffix(".geff").exists():
            names.append(image.stem)
    return names


def _link(source: Path, target: Path) -> None:
    os.symlink(str(source.resolve()), str(target), target_is_directory=source.is_dir())


def create_group_holdout(
    source_dir: Path,
    holdout_dir: Path,
    fraction: float = 0.2,
    seed: int = 0,
    train_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Create a dataset-level holdout using symlinks and write a manifest.

    The source remains untouched.  A non-empty target is rejected so a prior
    split cannot be silently replaced.  Splitting by dataset rather than by
    frame prevents temporal leakage from the same embryo/video.
    """

    source_dir = Path(source_dir).resolve()
    holdout_dir = Path(holdout_dir)
    train_dir = Path(train_dir) if train_dir is not None else None
    if not source_dir.is_dir():
        raise FileNotFoundError("source dataset directory not found: {}".format(source_dir))
    if holdout_dir.resolve() == source_dir:
        raise ValueError("holdout directory must differ from source directory")
    if train_dir is not None and train_dir.resolve() in {source_dir, holdout_dir.resolve()}:
        raise ValueError("train directory must differ from source and holdout directories")
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between 0 and 1")
    names = _dataset_names(source_dir)
    if len(names) < 2:
        raise ValueError("at least two datasets with .zarr and .geff are required")
    holdout_dir.mkdir(parents=True, exist_ok=True)
    if any(holdout_dir.iterdir()):
        raise FileExistsError("holdout directory is not empty: {}".format(holdout_dir))
    if train_dir is not None:
        train_dir.mkdir(parents=True, exist_ok=True)
        if any(train_dir.iterdir()):
            raise FileExistsError("train directory is not empty: {}".format(train_dir))

    holdout_count = max(1, min(len(names) - 1, int(round(len(names) * fraction))))
    shuffled = list(names)
    random.Random(int(seed)).shuffle(shuffled)
    holdout_names = sorted(shuffled[:holdout_count])
    train_names = sorted(set(names) - set(holdout_names))
    for name in holdout_names:
        _link(source_dir / (name + ".zarr"), holdout_dir / (name + ".zarr"))
        _link(source_dir / (name + ".geff"), holdout_dir / (name + ".geff"))
    if train_dir is not None:
        for name in train_names:
            _link(source_dir / (name + ".zarr"), train_dir / (name + ".zarr"))
            _link(source_dir / (name + ".geff"), train_dir / (name + ".geff"))

    manifest: Dict[str, Any] = {
        "source_dir": str(source_dir),
        "train_dir": str(train_dir.resolve()) if train_dir is not None else None,
        "holdout_dir": str(holdout_dir.resolve()),
        "source_datasets": names,
        "train_datasets": train_names,
        "holdout_datasets": holdout_names,
        "fraction": fraction,
        "seed": int(seed),
        "link_mode": "symlink",
    }
    with (holdout_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest
