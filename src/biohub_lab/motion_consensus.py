"""Conservative adjacent-frame motion consensus for candidate experiments.

The helper is intentionally independent of any external tracker.  It uses
the existing graph as one linker and compares a physical nearest-neighbour
choice with a constant-velocity extrapolation.  Only a target chosen by both
rules is eligible for addition, and topology limits are enforced before an
edge is appended.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


DEFAULT_SCALE_UM = (1.625, 0.40625, 0.40625)
_COORDINATE_KEYS = ("z", "y", "x")


def _validate_distance(value: float, name: str) -> float:
    distance = float(value)
    if distance < 0.0 or not math.isfinite(distance):
        raise ValueError("{} must be finite and non-negative".format(name))
    return distance


def _validate_scale(scale_um: Iterable[float]) -> tuple[float, float, float]:
    scale = tuple(float(value) for value in scale_um)
    if len(scale) != 3 or any(value <= 0.0 or not math.isfinite(value) for value in scale):
        raise ValueError("scale_um must contain three finite positive values")
    return scale  # type: ignore[return-value]


def _position(row: Mapping[str, Any], scale_um: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(row[key]) * scale_um[index] for index, key in enumerate(_COORDINATE_KEYS))  # type: ignore[return-value]


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _edge_row(dataset: str, source_id: int, target_id: int) -> dict[str, Any]:
    return {
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


def add_motion_consensus_links(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_distance_um: float = 7.0,
    max_prediction_error_um: float = 3.0,
    scale_um: Iterable[float] = DEFAULT_SCALE_UM,
    max_out_degree: int = 2,
    require_unique_parent: bool = True,
) -> list[dict[str, Any]]:
    """Add only deterministic consensus links between adjacent frames.

    For every source node with available capacity, two independent choices
    are made among target nodes at ``t + 1``: the closest target to the source
    and the closest target to a constant-velocity prediction based on one
    existing parent edge.  A link is appended only when both choices identify
    the same target and both physical-distance gates pass.  At most one new
    link is added per source in a call; existing edges are never replaced.

    By default, a source must have a unique existing parent so the velocity
    choice is genuinely independent of nearest-neighbour.  The optional
    ``require_unique_parent=False`` fallback uses zero velocity and is kept
    explicit for controlled ablations.
    """

    radius = _validate_distance(max_distance_um, "max_distance_um")
    prediction_error = _validate_distance(max_prediction_error_um, "max_prediction_error_um")
    scale = _validate_scale(scale_um)
    if int(max_out_degree) not in (1, 2):
        raise ValueError("max_out_degree must be 1 or 2")
    max_degree = int(max_out_degree)

    result = [dict(row) for row in rows]
    nodes: dict[tuple[str, int], dict[str, Any]] = {}
    nodes_by_dataset_time: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        if str(row.get("row_type")) != "node":
            continue
        dataset = str(row["dataset"])
        node_id = int(row["node_id"])
        key = (dataset, node_id)
        if key in nodes:
            raise ValueError("duplicate node id in dataset: {}".format(node_id))
        node = dict(row)
        nodes[key] = node
        nodes_by_dataset_time[(dataset, int(node["t"]))].append(node)

    outgoing: dict[tuple[str, int], list[int]] = defaultdict(list)
    incoming: dict[tuple[str, int], list[int]] = defaultdict(list)
    parents: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in result:
        if str(row.get("row_type")) != "edge":
            continue
        dataset = str(row["dataset"])
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        source = nodes.get((dataset, source_id))
        target = nodes.get((dataset, target_id))
        if source is None or target is None:
            raise ValueError("edge references a missing node")
        source_time = int(source["t"])
        target_time = int(target["t"])
        outgoing[(dataset, source_id)].append(target_id)
        incoming[(dataset, target_id)].append(source_id)
        # Legal gap-closing edges remain part of the topology budget, but
        # cannot provide a one-step velocity estimate.
        if target_time == source_time + 1:
            parents[(dataset, target_id)].append(source_id)

    candidates: list[tuple[float, str, int, int]] = []
    for (dataset, timepoint), source_rows in sorted(nodes_by_dataset_time.items()):
        target_rows = nodes_by_dataset_time.get((dataset, timepoint + 1), [])
        if not target_rows:
            continue
        target_rows = sorted(target_rows, key=lambda row: int(row["node_id"]))
        target_positions = {
            int(row["node_id"]): _position(row, scale) for row in target_rows
        }
        for source in sorted(source_rows, key=lambda row: int(row["node_id"])):
            source_id = int(source["node_id"])
            if len(outgoing[(dataset, source_id)]) >= max_degree:
                continue
            source_position = _position(source, scale)
            parent_ids = parents.get((dataset, source_id), [])
            if require_unique_parent and len(parent_ids) != 1:
                continue
            if len(parent_ids) == 1:
                parent = nodes[(dataset, parent_ids[0])]
                parent_position = _position(parent, scale)
                predicted_position = tuple(
                    2.0 * current - previous
                    for current, previous in zip(source_position, parent_position)
                )
            else:
                predicted_position = source_position

            available = [
                row for row in target_rows
                if not incoming[(dataset, int(row["node_id"]))]
            ]
            if not available:
                continue

            nearest = min(
                available,
                key=lambda row: (_distance(source_position, target_positions[int(row["node_id"])]), int(row["node_id"])),
            )
            predicted = min(
                available,
                key=lambda row: (_distance(predicted_position, target_positions[int(row["node_id"])]), int(row["node_id"])),
            )
            target_id = int(nearest["node_id"])
            if target_id != int(predicted["node_id"]):
                continue
            source_distance = _distance(source_position, target_positions[target_id])
            prediction_distance = _distance(predicted_position, target_positions[target_id])
            if source_distance > radius or prediction_distance > prediction_error:
                continue
            candidates.append((source_distance + prediction_distance, dataset, source_id, target_id))

    for _cost, dataset, source_id, target_id in sorted(candidates):
        if len(outgoing[(dataset, source_id)]) >= max_degree:
            continue
        if incoming[(dataset, target_id)]:
            continue
        result.append(_edge_row(dataset, source_id, target_id))
        outgoing[(dataset, source_id)].append(target_id)
        incoming[(dataset, target_id)].append(source_id)

    for row_id, row in enumerate(result):
        row["id"] = row_id
    return result
