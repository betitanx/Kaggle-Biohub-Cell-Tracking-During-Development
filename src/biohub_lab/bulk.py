"""Safe extraction helpers for the Kaggle competition bundle archive."""

import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable


def _member_destination(output_dir: Path, name: str) -> Path | None:
    """Resolve one archive member while rejecting traversal and symlinks."""

    normalized = str(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe archive member: {}".format(name))
    if not normalized.startswith(("test/", "train/")):
        return None
    root = Path(output_dir).resolve()
    destination = (root / Path(*path.parts)).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise ValueError("unsafe archive member: {}".format(name)) from error
    return destination


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (int(info.external_attr) >> 16) & 0o170000
    return stat.S_ISLNK(mode)


def _validate_manifest(expected_sizes: Dict[str, int]) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    for raw_name, raw_size in expected_sizes.items():
        name = str(raw_name).replace("\\", "/")
        if _member_destination(Path("/"), name) is None:
            raise ValueError("manifest member is outside test/train: {}".format(raw_name))
        size = int(raw_size)
        if size < 0:
            raise ValueError("manifest size must be non-negative: {}".format(raw_name))
        normalized[name] = size
    if not normalized:
        raise ValueError("bulk archive manifest is empty")
    return normalized


def _write_member(handle: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with handle.open(info, "r") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if partial.stat().st_size != int(info.file_size):
            raise ValueError("size mismatch after extracting {}".format(info.filename))
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def extract_competition_archive(
    archive_path: Path,
    output_dir: Path,
    expected_sizes: Dict[str, int],
    prefixes: Iterable[str] = ("test/", "train/"),
) -> Dict[str, int]:
    """Extract manifest-listed competition files with a safe path boundary.

    Existing files with the expected byte size are retained, so rerunning a
    failed extraction does not rewrite completed chunks. Files are written to
    a sibling partial file and renamed only after size validation.
    """

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    expected = _validate_manifest(expected_sizes)
    allowed_prefixes = tuple(str(prefix).rstrip("/") + "/" for prefix in prefixes)
    with zipfile.ZipFile(archive_path) as archive:
        members: Dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            destination = _member_destination(output_dir, info.filename)
            if destination is None or not info.filename.startswith(allowed_prefixes):
                continue
            if _is_symlink(info):
                raise ValueError("symlink archive member is not allowed: {}".format(info.filename))
            if info.is_dir():
                continue
            if info.filename in members:
                raise ValueError("duplicate archive member: {}".format(info.filename))
            members[info.filename] = info

        missing = sorted(set(expected) - set(members))
        if missing:
            raise ValueError("missing archive members: {}".format(", ".join(missing[:5])))

        extracted = 0
        skipped = 0
        for name in sorted(expected):
            info = members[name]
            expected_size = expected[name]
            if int(info.file_size) != expected_size:
                raise ValueError("size mismatch in archive member: {}".format(name))
            destination = _member_destination(output_dir, name)
            assert destination is not None
            if destination.is_file() and destination.stat().st_size == expected_size:
                skipped += 1
                continue
            if destination.exists() and not destination.is_file():
                raise ValueError("destination is not a regular file: {}".format(destination))
            _write_member(archive, info, destination)
            extracted += 1

    return {"files": len(expected), "bytes": sum(expected.values()), "extracted": extracted, "skipped": skipped}
