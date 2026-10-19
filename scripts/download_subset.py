#!/usr/bin/env python3
"""Download one competition OME-Zarr prefix without fetching the 81 GiB bundle."""

import argparse
import json
import re
import shutil
import stat
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from biohub_lab.cloud import is_transient_remote_error, retry_delay_seconds
from biohub_lab.cloud_data import entries_for_prefix, parse_file_listing_page


_NEXT_PAGE_RE = re.compile(r"Next Page Token = (.+)\n")


def _validate_remote_path(value: Any, *, prefix: bool = False) -> str:
    """Accept only unambiguous relative POSIX paths from untrusted listings."""

    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("unsafe prefix or file name: {!r}".format(value))
    path = value[:-1] if prefix and value.endswith("/") else value
    parts = path.split("/")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError("unsafe prefix or file name: {!r}".format(value))
    return path


def _check_local_path(path: Path, *, directory: bool) -> None:
    """Reject links (including dangling ones) before resolving any component."""

    for component in [*reversed(path.parents), path]:
        try:
            mode = component.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("symlink in output path: {}".format(component))
        is_directory = directory or component != path
        if not (stat.S_ISDIR(mode) if is_directory else stat.S_ISREG(mode)):
            raise ValueError("invalid output path type: {}".format(component))


def _checked_destination(output_dir: Path, name: str) -> Path:
    destination = output_dir / name
    _check_local_path(destination, directory=False)
    if not destination.resolve().is_relative_to(output_dir):
        raise ValueError("destination escapes output directory: {!r}".format(name))
    return destination


def _run_kaggle(
    command: List[str], attempts: int = 8, timeout_seconds: int = 300
) -> subprocess.CompletedProcess[str]:
    """Run a Kaggle CLI command with bounded backoff for transient API errors."""

    for attempt in range(attempts):
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            if attempt + 1 >= attempts:
                return subprocess.CompletedProcess(
                    command,
                    returncode=1,
                    stdout=str(error.stdout or ""),
                    stderr="Kaggle command timed out after {} seconds".format(timeout_seconds),
                )
            time.sleep(retry_delay_seconds(attempt))
            continue
        if completed.returncode == 0:
            return completed
        detail = "{}\n{}".format(completed.stdout, completed.stderr)
        if attempt + 1 >= attempts or not is_transient_remote_error(detail):
            return completed
        time.sleep(retry_delay_seconds(attempt))
    raise AssertionError("unreachable")


def _list_prefix(competition: str, prefix: str, executable: str, page_size: int = 200) -> List[Dict[str, Any]]:
    normalized = prefix.rstrip("/") + "/"
    root_marker = normalized.rstrip("/") + "/zarr.json"
    selected: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        command = [executable, "competitions", "files", competition, "--format", "json", "--page-size", str(page_size)]
        if token:
            command.extend(["--page-token", token])
        completed = _run_kaggle(command)
        if completed.returncode != 0:
            raise RuntimeError("Kaggle file listing failed: {}".format(completed.stderr.strip()[-500:]))
        start = completed.stdout.find("[")
        if start < 0:
            raise RuntimeError("Kaggle file listing returned no JSON page")
        page = parse_file_listing_page(completed.stdout)
        selected.extend(item for item in page if str(item.get("name", "")).startswith(normalized))
        if any(str(item.get("name", "")) == root_marker for item in page):
            break
        match = _NEXT_PAGE_RE.search(completed.stdout)
        token = match.group(1).strip() if match else None
        if not token:
            break
    if not selected:
        raise FileNotFoundError("no competition files matched prefix: {}".format(prefix))
    selected.sort(key=lambda item: str(item.get("name", "")))
    return selected


def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a JSON list")
    return [entry for entry in entries if isinstance(entry, dict)]


def download_prefix(
    competition: str,
    prefix: str,
    output_dir: Path,
    executable: str = "kaggle",
    dry_run: bool = False,
    manifest_path: Optional[Path] = None,
    workers: int = 4,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Download exact files under one OME-Zarr prefix and restore their paths.

    Files in a prefix are independent, so a bounded worker pool improves
    throughput while keeping each destination resumable and deterministic.
    Output paths require physical, non-symlinked ancestors (on macOS, use
    /private/var or /private/tmp instead of the /var or /tmp aliases).
    Dry runs validate without creating output directories.
    """

    if workers < 1:
        raise ValueError("workers must be at least one")
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    normalized = _validate_remote_path(prefix, prefix=True) + "/"
    if manifest_path is None:
        files = _list_prefix(competition, prefix, executable)
    else:
        files = entries_for_prefix(_load_manifest(Path(manifest_path)), prefix)
        if not files:
            raise FileNotFoundError("no manifest files matched prefix: {}".format(prefix))
    # Validate every selected name before mkdir, temporary files or workers.
    for item in files:
        name = _validate_remote_path(item.get("name"))
        if not name.startswith(normalized):
            raise ValueError("file name is outside requested prefix: {!r}".format(name))
    total_bytes = sum(int(item.get("size", 0)) for item in files)
    output_dir = Path(output_dir).absolute()
    _check_local_path(output_dir, directory=True)
    output_dir = output_dir.resolve()
    for item in files:
        _checked_destination(output_dir, item["name"])
    if dry_run:
        return {"prefix": prefix, "files": len(files), "bytes": total_bytes, "downloaded": 0}
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_one(item: Dict[str, Any]) -> bool:
        name = item["name"]
        destination = _checked_destination(output_dir, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _checked_destination(output_dir, name)
        expected_size = int(item.get("size", 0))
        if destination.is_file() and (expected_size <= 0 or destination.stat().st_size == expected_size):
            return False
        with tempfile.TemporaryDirectory(prefix=".kaggle-file-", dir=str(output_dir)) as temporary:
            command = [executable, "competitions", "download", competition, "--file", name, "--path", temporary, "--quiet"]
            completed = _run_kaggle(command, timeout_seconds=timeout_seconds)
            if completed.returncode != 0:
                raise RuntimeError("Kaggle file download failed for {}: {}".format(name, completed.stderr.strip()[-500:]))
            _checked_destination(output_dir, name)
            candidates = []
            for path in Path(temporary).iterdir():
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError("symlink in downloaded output: {}".format(path))
                if stat.S_ISREG(mode):
                    candidates.append(path)
            if len(candidates) != 1:
                raise RuntimeError("expected one downloaded file for {}, found {}".format(name, len(candidates)))
            # Best-effort recheck, not protection against hostile concurrent swaps.
            _checked_destination(output_dir, name)
            shutil.move(str(candidates[0]), str(destination))
        if expected_size > 0 and destination.stat().st_size != expected_size:
            raise RuntimeError("size mismatch after downloading {}".format(name))
        return True

    downloaded = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kaggle-download") as executor:
        futures = [executor.submit(download_one, item) for item in files]
        for future in as_completed(futures):
            downloaded += int(future.result())
    return {"prefix": prefix, "files": len(files), "bytes": total_bytes, "downloaded": downloaded}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="biohub-cell-tracking-during-development")
    parser.add_argument("--prefix", required=True, help="Exact dataset prefix, e.g. test/<stem>.zarr/")
    parser.add_argument("--output-dir", type=Path, default=Path("data/selected"))
    parser.add_argument("--executable", default="kaggle")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, help="cached JSON listing from the competition")
    parser.add_argument("--workers", type=int, default=4, help="bounded concurrent file downloads")
    parser.add_argument("--timeout-seconds", type=int, default=300, help="per-file Kaggle CLI timeout")
    args = parser.parse_args()
    result = download_prefix(
        args.competition,
        args.prefix,
        args.output_dir,
        args.executable,
        args.dry_run,
        args.manifest,
        args.workers,
        args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
