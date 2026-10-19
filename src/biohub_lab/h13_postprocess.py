"""Deterministic physical-link rescue for H13 prediction graphs."""

from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - exercised through the real Kaggle environment.
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
except ImportError:  # pragma: no cover - the dependency-light fallback is tested.
    _KDTree = None


DEFAULT_SCALE_UM = (1.625, 0.40625, 0.40625)


def _distance_um(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    scale_um: tuple[float, float, float],
) -> float:
    delta = [
        (float(left[axis]) - float(right[axis])) * scale
        for axis, scale in zip(("z", "y", "x"), scale_um)
    ]
    return sqrt(sum(value * value for value in delta))


def rescue_physical_links(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_distance_um: float = 7.0,
    scale_um: tuple[float, float, float] = DEFAULT_SCALE_UM,
    max_existing_edges: int | None = 0,
) -> list[dict[str, Any]]:
    """Add deterministic adjacent-frame links when a graph is under-linked.

    The rescue is deliberately conservative: candidates are restricted to
    ``t -> t+1`` and greedily matched by physical distance, so every source
    and target receives at most one rescued edge. Existing model edges are
    preserved. ``max_existing_edges=0`` makes this a fallback for an empty
    edge prediction, while ``None`` always fills unmatched nodes.
    """

    radius = float(max_distance_um)
    if radius < 0.0 or not isfinite(radius):
        raise ValueError("max_distance_um must be finite and non-negative")
    if len(scale_um) != 3 or any(float(value) <= 0.0 or not isfinite(float(value)) for value in scale_um):
        raise ValueError("scale_um must contain three finite positive values")
    if max_existing_edges is not None and int(max_existing_edges) < 0:
        raise ValueError("max_existing_edges must be non-negative or None")

    result = [dict(row) for row in rows]
    node_rows = [row for row in result if str(row.get("row_type")) == "node"]
    edge_rows = [row for row in result if str(row.get("row_type")) == "edge"]
    if max_existing_edges is not None and len(edge_rows) > int(max_existing_edges):
        for row_id, row in enumerate(result):
            row["id"] = row_id
        return result

    by_dataset_time: dict[tuple[str, int], list[dict[str, Any]]] = {}
    node_time: dict[tuple[str, int], int] = {}
    for row in node_rows:
        dataset = str(row["dataset"])
        node_id = int(row["node_id"])
        timepoint = int(row["t"])
        by_dataset_time.setdefault((dataset, timepoint), []).append(row)
        node_time[(dataset, node_id)] = timepoint

    used_sources: dict[tuple[str, int], set[int]] = {}
    used_targets: dict[tuple[str, int], set[int]] = {}
    for row in edge_rows:
        dataset = str(row["dataset"])
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        source_time = node_time.get((dataset, source_id))
        target_time = node_time.get((dataset, target_id))
        if source_time is None or target_time != source_time + 1:
            continue
        used_sources.setdefault((dataset, source_time), set()).add(source_id)
        used_targets.setdefault((dataset, source_time), set()).add(target_id)

    scale = tuple(float(value) for value in scale_um)
    rescued: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in node_rows}):
        times = sorted(time for name, time in by_dataset_time if name == dataset)
        for timepoint in times:
            left = sorted(by_dataset_time.get((dataset, timepoint), []), key=lambda row: int(row["node_id"]))
            right = sorted(by_dataset_time.get((dataset, timepoint + 1), []), key=lambda row: int(row["node_id"]))
            if not left or not right:
                continue

            left_coords = [
                [float(row[axis]) * scale[index] for index, axis in enumerate(("z", "y", "x"))]
                for row in left
            ]
            right_coords = [
                [float(row[axis]) * scale[index] for index, axis in enumerate(("z", "y", "x"))]
                for row in right
            ]
            pairs: list[tuple[float, int, int]] = []
            if _KDTree is not None:
                tree = _KDTree(right_coords)
                for left_index, coordinate in enumerate(left_coords):
                    for right_index in tree.query_ball_point(coordinate, radius):
                        distance = _distance_um(left[left_index], right[int(right_index)], scale)
                        if distance <= radius:
                            pairs.append((distance, left_index, int(right_index)))
            else:  # pragma: no cover - only used when SciPy is unavailable.
                for left_index, left_row in enumerate(left):
                    for right_index, right_row in enumerate(right):
                        distance = _distance_um(left_row, right_row, scale)
                        if distance <= radius:
                            pairs.append((distance, left_index, right_index))

            pairs.sort(key=lambda item: (item[0], int(left[item[1]]["node_id"]), int(right[item[2]]["node_id"])))
            source_used = used_sources.setdefault((dataset, timepoint), set())
            target_used = used_targets.setdefault((dataset, timepoint), set())
            for _distance, left_index, right_index in pairs:
                source_id = int(left[left_index]["node_id"])
                target_id = int(right[right_index]["node_id"])
                if source_id in source_used or target_id in target_used:
                    continue
                source_used.add(source_id)
                target_used.add(target_id)
                rescued.append(
                    {
                        "id": -1,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )

    result.extend(rescued)
    for row_id, row in enumerate(result):
        row["id"] = row_id
    return result
