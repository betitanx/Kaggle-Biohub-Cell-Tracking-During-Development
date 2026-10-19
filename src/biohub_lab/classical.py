"""Dependency-light 3D peak detection and temporal nearest-neighbour linking.

This module is deliberately small and deterministic.  It is a falsifiable
candidate (H1) for the Biohub challenge, not a claim that a classical method
will beat the public Cellmot baseline.  Keeping the core independent of
NumPy makes the graph logic unit-testable in the control environment; the
real-data adapter can pass NumPy/Zarr frames without changing the algorithm.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

try:  # Optional acceleration for the real-data adapter.
    import numpy as _np  # type: ignore
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
except ImportError:  # pragma: no cover - exercised in dependency-light installs.
    _np = None
    _KDTree = None


Number = Union[int, float]
Threshold = Union[Number, Sequence[Number]]


@dataclass(frozen=True)
class Detection:
    """A detected cell centre in voxel coordinates."""

    t: int
    z: int
    y: int
    x: int
    intensity: float
    node_id: int = -1


@dataclass
class TrackingGraph:
    """A submission-ready graph for one dataset."""

    dataset: str
    nodes: List[Detection]
    edges: List[Tuple[int, int]]
    threshold: Optional[float] = None


def _shape(frame: object) -> Tuple[int, int, int]:
    raw_shape = getattr(frame, "shape", None)
    if raw_shape is not None:
        shape = tuple(int(value) for value in raw_shape)
    else:
        try:
            shape = (len(frame), len(frame[0]), len(frame[0][0]))  # type: ignore[index]
        except (IndexError, TypeError, AttributeError) as exc:
            raise ValueError("frame must be a 3D array-like object") from exc
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("frame must have three non-empty dimensions (Z, Y, X)")
    return shape  # type: ignore[return-value]


def _value(frame: object, z: int, y: int, x: int) -> float:
    try:
        value = frame[z, y, x]  # type: ignore[index]
    except (IndexError, TypeError):
        value = frame[z][y][x]  # type: ignore[index]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame contains a non-numeric voxel value") from exc


def _neighbourhood(value: Sequence[int]) -> Tuple[int, int, int]:
    if len(value) != 3 or any(int(item) < 0 for item in value):
        raise ValueError("neighborhood must contain three non-negative integers")
    return tuple(int(item) for item in value)  # type: ignore[return-value]


def detect_local_maxima(
    frame: object,
    threshold: Number = 0.0,
    neighborhood: Sequence[int] = (1, 1, 1),
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Return deterministic 3D local maxima above ``threshold``.

    Equal-valued plateaus produce one maximum: the lexicographically first
    voxel wins.  Candidates are sorted by descending intensity and then by
    coordinates, so a cap produces reproducible results across runs.
    """

    if max_detections is not None and max_detections <= 0:
        return []
    threshold_value = float(threshold)
    rz, ry, rx = _neighbourhood(neighborhood)
    z_size, y_size, x_size = _shape(frame)
    candidates: List[Detection] = []

    for z in range(z_size):
        for y in range(y_size):
            for x in range(x_size):
                center = _value(frame, z, y, x)
                if not math.isfinite(center) or center < threshold_value:
                    continue
                is_maximum = True
                z_start, z_stop = max(0, z - rz), min(z_size, z + rz + 1)
                y_start, y_stop = max(0, y - ry), min(y_size, y + ry + 1)
                x_start, x_stop = max(0, x - rx), min(x_size, x + rx + 1)
                for other_z in range(z_start, z_stop):
                    for other_y in range(y_start, y_stop):
                        for other_x in range(x_start, x_stop):
                            if (other_z, other_y, other_x) == (z, y, x):
                                continue
                            other = _value(frame, other_z, other_y, other_x)
                            if other > center or (
                                other == center
                                and (other_z, other_y, other_x) < (z, y, x)
                            ):
                                is_maximum = False
                                break
                        if not is_maximum:
                            break
                    if not is_maximum:
                        break
                if is_maximum:
                    candidates.append(
                        Detection(t=0, z=z, y=y, x=x, intensity=center)
                    )

    candidates.sort(key=lambda item: (-item.intensity, item.z, item.y, item.x))
    if max_detections is not None:
        return candidates[:max_detections]
    return candidates


def detect_local_maxima_fast(
    frame: object,
    threshold: Number = 0.0,
    neighborhood: Sequence[int] = (1, 1, 1),
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Use SciPy's vectorised maximum filter when available.

    The pure-Python implementation remains the fallback, so this optimisation
    does not change the candidate's reproducibility or its minimal test
    surface.  Plateau tie-breaking is applied after the vectorised filter.
    """

    try:
        import numpy as np  # type: ignore
        from scipy.ndimage import maximum_filter  # type: ignore
    except ImportError:
        return detect_local_maxima(frame, threshold, neighborhood, max_detections)
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError("frame must have three non-empty dimensions (Z, Y, X)")
    rz, ry, rx = _neighbourhood(neighborhood)
    maximum = maximum_filter(array, size=(2 * rz + 1, 2 * ry + 1, 2 * rx + 1), mode="nearest")
    mask = np.isfinite(array) & (array >= float(threshold)) & (array == maximum)
    candidates: List[Detection] = []
    for z, y, x in np.argwhere(mask):
        z, y, x = int(z), int(y), int(x)
        center = float(array[z, y, x])
        z_start, z_stop = max(0, z - rz), min(array.shape[0], z + rz + 1)
        y_start, y_stop = max(0, y - ry), min(array.shape[1], y + ry + 1)
        x_start, x_stop = max(0, x - rx), min(array.shape[2], x + rx + 1)
        if any(
            float(array[other_z, other_y, other_x]) == center
            and (other_z, other_y, other_x) < (z, y, x)
            for other_z in range(z_start, z_stop)
            for other_y in range(y_start, y_stop)
            for other_x in range(x_start, x_stop)
            if (other_z, other_y, other_x) != (z, y, x)
        ):
            continue
        candidates.append(Detection(t=0, z=z, y=y, x=x, intensity=center))
    candidates.sort(key=lambda item: (-item.intensity, item.z, item.y, item.x))
    if max_detections is not None:
        return candidates[:max_detections]
    return candidates


def _validate_scale(scale_um: Sequence[Number]) -> Tuple[float, float, float]:
    if len(scale_um) != 3:
        raise ValueError("scale_um must contain (Z, Y, X)")
    scale = tuple(float(value) for value in scale_um)
    if any(value <= 0.0 or not math.isfinite(value) for value in scale):
        raise ValueError("scale_um values must be finite and positive")
    return scale  # type: ignore[return-value]


def _distance_um(
    left: Detection,
    right: Detection,
    scale_um: Tuple[float, float, float],
) -> float:
    dz = (left.z - right.z) * scale_um[0]
    dy = (left.y - right.y) * scale_um[1]
    dx = (left.x - right.x) * scale_um[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def link_detections(
    previous: Sequence[Detection],
    current: Sequence[Detection],
    scale_um: Sequence[Number] = (1.625, 0.40625, 0.40625),
    max_distance_um: Number = 7.0,
    use_spatial_index: bool = True,
) -> List[Tuple[int, int]]:
    """Greedily link adjacent-frame detections in physical space.

    Each source and target can participate in at most one link.  This is an
    intentional H1 limitation: divisions require a separate topology-aware
    hypothesis (for example the public transformer or Ultrack candidate).
    """

    scale = _validate_scale(scale_um)
    max_distance = float(max_distance_um)
    if max_distance < 0.0:
        return []
    if previous and current and any(item.t + 1 != current[0].t for item in previous):
        raise ValueError("link_detections expects detections from adjacent frames")
    pairs: List[Tuple[float, int, int]] = []
    # A frame can contain thousands of detections.  The old all-pairs loop is
    # useful as a dependency-free fallback but becomes quadratic in the real
    # competition data.  cKDTree limits distance calculations to plausible
    # neighbours while the final sort/greedy assignment remains exactly the
    # same deterministic rule.
    use_tree = bool(use_spatial_index and _np is not None and _KDTree is not None)
    use_tree = use_tree and math.isfinite(max_distance)
    if use_tree and previous and current:
        left_coordinates = _np.asarray(
            [[item.z * scale[0], item.y * scale[1], item.x * scale[2]] for item in previous],
            dtype=float,
        )
        right_coordinates = _np.asarray(
            [[item.z * scale[0], item.y * scale[1], item.x * scale[2]] for item in current],
            dtype=float,
        )
        tree = _KDTree(right_coordinates)
        for previous_index, left in enumerate(left_coordinates):
            for current_index in sorted(tree.query_ball_point(left, r=max_distance)):
                delta = left - right_coordinates[current_index]
                distance = float(_np.sqrt(_np.dot(delta, delta)))
                if distance <= max_distance:
                    pairs.append((distance, previous_index, int(current_index)))
    else:
        for previous_index, left in enumerate(previous):
            for current_index, right in enumerate(current):
                distance = _distance_um(left, right, scale)
                if distance <= max_distance:
                    pairs.append((distance, previous_index, current_index))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_previous = set()
    used_current = set()
    links: List[Tuple[int, int]] = []
    for _distance, previous_index, current_index in pairs:
        if previous_index in used_previous or current_index in used_current:
            continue
        used_previous.add(previous_index)
        used_current.add(current_index)
        links.append((previous_index, current_index))
    return sorted(links)


def infer_division_links(
    graph: TrackingGraph,
    scale_um: Sequence[Number] = (1.625, 0.40625, 0.40625),
    max_link_distance_um: Number = 7.0,
    sister_max_distance_um: Number = 5.0,
    midpoint_tolerance_um: Number = 3.0,
    max_parent_displacement_delta_um: Number = 3.0,
    min_sibling_intensity_ratio: Number = 0.9,
) -> TrackingGraph:
    """Add conservative second-daughter links after one-to-one matching.

    This is a bounded reproduction of the public notebook idea: retain the
    one-to-one assignment, then add at most one unmatched daughter when the
    parent, assigned child and candidate child form a compact midpoint
    geometry.  It is an explicit hypothesis, not part of H1's baseline.
    """

    scale = _validate_scale(scale_um)
    max_distance = float(max_link_distance_um)
    sister_distance_limit = float(sister_max_distance_um)
    midpoint_limit = float(midpoint_tolerance_um)
    displacement_delta_limit = float(max_parent_displacement_delta_um)
    intensity_ratio = float(min_sibling_intensity_ratio)
    if any(
        value < 0.0 or not math.isfinite(value)
        for value in (max_distance, sister_distance_limit, midpoint_limit, displacement_delta_limit)
    ):
        raise ValueError("division inference distances must be finite and non-negative")
    if not 0.0 <= intensity_ratio <= 1.0 or not math.isfinite(intensity_ratio):
        raise ValueError("min_sibling_intensity_ratio must be finite and between zero and one")
    _check_graph(graph)
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    nodes_by_time = {}
    for node in graph.nodes:
        nodes_by_time.setdefault(node.t, []).append(node)
    outgoing = {}
    incoming_targets = set()
    for source_id, target_id in graph.edges:
        outgoing.setdefault(source_id, []).append(target_id)
        incoming_targets.add(target_id)

    edges = list(graph.edges)
    for parent in sorted(graph.nodes, key=lambda item: item.node_id):
        children = outgoing.get(parent.node_id, [])
        if len(children) != 1:
            continue
        first_child = nodes_by_id[children[0]]
        first_parent_distance = _distance_um(parent, first_child, scale)
        candidates = []
        for candidate in nodes_by_time.get(parent.t + 1, []):
            if candidate.node_id in incoming_targets or candidate.node_id == first_child.node_id:
                continue
            parent_distance = _distance_um(parent, candidate, scale)
            if parent_distance > max_distance:
                continue
            if abs(parent_distance - first_parent_distance) > displacement_delta_limit:
                continue
            if first_child.intensity > 0.0 and candidate.intensity < first_child.intensity * intensity_ratio:
                continue
            sister_distance = _distance_um(first_child, candidate, scale)
            if sister_distance > sister_distance_limit:
                continue
            midpoint_z = (first_child.z + candidate.z) / 2.0
            midpoint_y = (first_child.y + candidate.y) / 2.0
            midpoint_x = (first_child.x + candidate.x) / 2.0
            midpoint_distance = math.sqrt(
                ((parent.z - midpoint_z) * scale[0]) ** 2
                + ((parent.y - midpoint_y) * scale[1]) ** 2
                + ((parent.x - midpoint_x) * scale[2]) ** 2
            )
            if midpoint_distance > midpoint_limit:
                continue
            candidates.append((midpoint_distance, sister_distance, parent_distance, candidate.node_id))
        if not candidates:
            continue
        _midpoint, _sister, _parent_distance, candidate_id = min(candidates)
        edges.append((parent.node_id, candidate_id))
        incoming_targets.add(candidate_id)
        outgoing.setdefault(parent.node_id, []).append(candidate_id)
    return TrackingGraph(
        dataset=graph.dataset,
        nodes=list(graph.nodes),
        edges=sorted(set(edges)),
        threshold=graph.threshold,
    )


def remove_isolated_nodes(graph: TrackingGraph) -> TrackingGraph:
    """Remove nodes with no temporal evidence and remap IDs deterministically.

    This is the smallest H8 post-processing ablation.  It is intentionally
    conservative about graph integrity but intentionally aggressive about
    single-frame detections; real use must compare it against holdout because
    a biologically real cell can also be visible for only one frame.
    """

    _check_graph(graph)
    connected = set()
    for source_id, target_id in graph.edges:
        connected.add(source_id)
        connected.add(target_id)
    kept = [node for node in graph.nodes if node.node_id in connected]
    remap = {node.node_id: index for index, node in enumerate(kept)}
    nodes = [
        Detection(
            t=node.t,
            z=node.z,
            y=node.y,
            x=node.x,
            intensity=node.intensity,
            node_id=remap[node.node_id],
        )
        for node in kept
    ]
    edges = sorted(
        (remap[source_id], remap[target_id])
        for source_id, target_id in graph.edges
        if source_id in remap and target_id in remap
    )
    return TrackingGraph(dataset=graph.dataset, nodes=nodes, edges=edges, threshold=graph.threshold)


def _threshold_for_frame(threshold: Threshold, time_index: int) -> float:
    if isinstance(threshold, (int, float)):
        return float(threshold)
    values = list(threshold)
    if time_index >= len(values):
        raise ValueError("threshold sequence is shorter than the frame sequence")
    return float(values[time_index])


def _fallback_detection(frame: object) -> Optional[Detection]:
    """Return the brightest finite voxel when thresholding finds no node."""

    if _np is not None:
        array = _np.asarray(frame)
        finite = _np.isfinite(array)
        if not finite.any():
            return None
        masked = _np.where(finite, array, -_np.inf)
        flat_index = int(masked.argmax())
        z, y, x = (int(value) for value in _np.unravel_index(flat_index, masked.shape))
        return Detection(t=0, z=z, y=y, x=x, intensity=float(masked.flat[flat_index]))

    shape = _shape(frame)
    best: Optional[Tuple[float, int, int, int]] = None
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                value = _value(frame, z, y, x)
                if not math.isfinite(value):
                    continue
                candidate = (value, -z, -y, -x)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return None
    value, neg_z, neg_y, neg_x = best
    return Detection(t=0, z=-neg_z, y=-neg_y, x=-neg_x, intensity=value)


def track_frames(
    frames: Iterable[object],
    dataset: str,
    threshold: Threshold = 0.0,
    neighborhood: Sequence[int] = (1, 1, 1),
    scale_um: Sequence[Number] = (1.625, 0.40625, 0.40625),
    max_link_distance_um: Number = 7.0,
    max_detections_per_frame: Optional[int] = None,
    fast: bool = True,
    ensure_timepoint_coverage: bool = False,
) -> TrackingGraph:
    """Detect and link a stream of ``(Z, Y, X)`` frames."""

    if not str(dataset):
        raise ValueError("dataset must be non-empty")
    nodes: List[Detection] = []
    edges: List[Tuple[int, int]] = []
    previous: List[Detection] = []
    for time_index, image in enumerate(frames):
        detector = detect_local_maxima_fast if fast else detect_local_maxima
        detected = detector(
            image,
            threshold=_threshold_for_frame(threshold, time_index),
            neighborhood=neighborhood,
            max_detections=max_detections_per_frame,
        )
        if not detected and ensure_timepoint_coverage and (
            max_detections_per_frame is None or max_detections_per_frame > 0
        ):
            fallback = _fallback_detection(image)
            if fallback is not None:
                detected = [fallback]
        current: List[Detection] = []
        for item in detected:
            node = Detection(
                t=time_index,
                z=item.z,
                y=item.y,
                x=item.x,
                intensity=item.intensity,
                node_id=len(nodes),
            )
            nodes.append(node)
            current.append(node)
        for previous_index, current_index in link_detections(
            previous,
            current,
            scale_um=scale_um,
            max_distance_um=max_link_distance_um,
        ):
            edges.append((previous[previous_index].node_id, current[current_index].node_id))
        previous = current
    return TrackingGraph(dataset=str(dataset), nodes=nodes, edges=edges)


def _check_graph(graph: TrackingGraph) -> None:
    node_times = {node.node_id: node.t for node in graph.nodes}
    if len(node_times) != len(graph.nodes):
        raise ValueError("node_id must be unique within each dataset")
    outgoing = {}
    for source_id, target_id in graph.edges:
        if source_id not in node_times or target_id not in node_times:
            raise ValueError("edge references an unknown node_id")
        if source_id == target_id:
            raise ValueError("self-edge is not allowed")
        if node_times[target_id] != node_times[source_id] + 1:
            raise ValueError("edges must connect adjacent timepoints")
        outgoing[source_id] = outgoing.get(source_id, 0) + 1
    if any(count > 2 for count in outgoing.values()):
        raise ValueError("a node cannot have more than two outgoing edges")


def write_submission_csv(graphs: Iterable[TrackingGraph], path: Path) -> Path:
    """Write one or more tracking graphs in the Kaggle CSV contract.

    Internal graph IDs are zero-based and optimized for algorithms. The Kaggle
    CSV contract uses positive node IDs, reset per dataset, so IDs are remapped
    only at this serialization boundary.
    """

    graphs = list(graphs)
    for graph in graphs:
        _check_graph(graph)
    if len({graph.dataset for graph in graphs}) != len(graphs):
        raise ValueError("dataset names must be unique in a submission")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "id",
        "dataset",
        "row_type",
        "node_id",
        "t",
        "z",
        "y",
        "x",
        "source_id",
        "target_id",
    ]
    row_id = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for graph in graphs:
            submission_ids = {
                node.node_id: index + 1 for index, node in enumerate(graph.nodes)
            }
            for node in graph.nodes:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": graph.dataset,
                        "row_type": "node",
                        "node_id": submission_ids[node.node_id],
                        "t": node.t,
                        "z": node.z,
                        "y": node.y,
                        "x": node.x,
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for source_id, target_id in graph.edges:
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": graph.dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": submission_ids[source_id],
                        "target_id": submission_ids[target_id],
                    }
                )
                row_id += 1
    return path
