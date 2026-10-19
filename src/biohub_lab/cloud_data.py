"""Manifest parsing helpers for chunked Kaggle downloads."""

import json
from typing import Any, Iterable, List, Optional


def parse_file_listing_page(stdout: str) -> List[dict[str, Any]]:
    """Decode the JSON array from a Kaggle CLI page, ignoring pager text."""

    start = stdout.find("[")
    if start < 0:
        raise ValueError("Kaggle file listing returned no JSON page")
    try:
        payload, _end = json.JSONDecoder().raw_decode(stdout[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Kaggle file listing returned invalid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("Kaggle file listing page must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def dataset_prefix(name: str) -> Optional[str]:
    """Return the directory prefix for a root OME-Zarr/GEFF metadata file."""

    normalized = str(name).strip().lstrip("/")
    if not normalized.endswith("/zarr.json"):
        return None
    prefix = normalized[: -len("zarr.json")]
    if not prefix.endswith((".zarr/", ".geff/")):
        return None
    return prefix


def dataset_prefixes(
    entries: Iterable[dict[str, Any]],
    split: Optional[str] = None,
) -> List[str]:
    """Collect deterministic dataset prefixes from manifest entries."""

    result = set()
    for entry in entries:
        prefix = dataset_prefix(str(entry.get("name", "")))
        if prefix is None:
            continue
        if split is not None and not prefix.startswith(str(split).rstrip("/") + "/"):
            continue
        result.add(prefix)
    return sorted(result)


def entries_for_prefix(
    entries: Iterable[dict[str, Any]],
    prefix: str,
) -> List[dict[str, Any]]:
    """Select and sort manifest entries below one exact directory prefix."""

    normalized = str(prefix).strip().lstrip("/").rstrip("/") + "/"
    selected = [
        entry
        for entry in entries
        if str(entry.get("name", "")).lstrip("/").startswith(normalized)
    ]
    return sorted(selected, key=lambda entry: str(entry.get("name", "")))
