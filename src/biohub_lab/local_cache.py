"""Prepare a local SSD view of the staged competition data for one cycle."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict


def _copy_tree(source: Path, destination: Path, workers: int = 1) -> None:
    if not source.is_dir():
        raise FileNotFoundError("staged data directory not found: {}".format(source))
    destination.mkdir(parents=True, exist_ok=True)
    entries = sorted(source.iterdir())

    def copy_entry(entry: Path) -> None:
        target = destination / entry.name
        if entry.is_dir() and not entry.is_symlink():
            shutil.copytree(entry, target, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target, follow_symlinks=False)

    worker_count = max(1, min(int(workers), len(entries))) if entries else 1
    if worker_count == 1:
        for entry in entries:
            copy_entry(entry)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            list(executor.map(copy_entry, entries))


def _link_split(source_root: Path, split_name: str, destination: Path) -> None:
    source_split = source_root / split_name
    if not source_split.is_dir():
        raise FileNotFoundError("staged split directory not found: {}".format(source_split))
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_split.iterdir()):
        if source.name == "manifest.json":
            shutil.copy2(source, destination / source.name)
            continue
        if source.suffix not in {".zarr", ".geff"}:
            continue
        link = destination / source.name
        if link.exists() or link.is_symlink():
            link.unlink()
        relative = os.path.relpath(source_root / "train" / source.name, destination)
        link.symlink_to(relative, target_is_directory=source.is_dir())


def prepare_data_cache(
    source_root: Path,
    cache_root: Path,
    workers: int = 4,
) -> Dict[str, Path]:
    """Copy data into a new cache without deleting any existing directory.

    The source Volume remains untouched.  The returned paths have the same
    layout expected by ``RemoteRunConfig`` and can be passed to candidates
    through environment variables.  The cache is deliberately scoped to an
    explicit new directory owned by the current worker. Choose a different
    path for each run; existing caches are never implicitly cleared.
    """

    source_root = Path(source_root).resolve()
    cache_root = Path(cache_root)
    if cache_root.is_symlink():
        raise ValueError("cache_root must not be a symlink")
    cache_root = cache_root.resolve()
    if cache_root == Path("/") or len(cache_root.parts) < 3:
        raise ValueError("cache_root must be a narrow temporary directory")
    if cache_root.is_relative_to(source_root) or source_root.is_relative_to(cache_root):
        raise ValueError("cache_root and source_root must not overlap")
    for split in ("test", "train", "train_split", "holdout"):
        if not (source_root / split).is_dir():
            raise FileNotFoundError("staged data directory not found: {}".format(source_root / split))
    if cache_root.exists():
        raise FileExistsError("cache_root already exists; choose a new cache directory")
    cache_root.mkdir(parents=True, exist_ok=False)
    _copy_tree(source_root / "test", cache_root / "test", workers=workers)
    _copy_tree(source_root / "train", cache_root / "train", workers=workers)
    _link_split(source_root, "train_split", cache_root / "train_split")
    _link_split(source_root, "holdout", cache_root / "holdout")
    return {
        "root": cache_root,
        "test": cache_root / "test",
        "train_split": cache_root / "train_split",
        "holdout": cache_root / "holdout",
    }
