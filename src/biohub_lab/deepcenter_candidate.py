"""Optional DeepCenter shape-prior adapter.

The public DeepCenterUNet3D artifact is a center-heatmap detector.  This
module deliberately stops at inference and deterministic candidate ranking:
it does not mutate a tracking graph, invent edges, or claim a competition
score.  A later experiment may decide whether a gated candidate should be
added to the primary detector/tracker.

PyTorch and the downloaded training script are loaded lazily so the control
plane and its unit tests remain usable without the optional ML stack.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class DeepCenterInferenceConfig:
    """Geometry and normalization settings saved with the public checkpoint."""

    pool_factor: int = 4
    base_channels: int = 24
    norm_lo_pct: float = 50.0
    norm_hi_pct: float = 99.5
    norm_clip_lo: float = -0.5
    norm_clip_hi: float = 6.0

    def __post_init__(self) -> None:
        if int(self.pool_factor) != self.pool_factor or self.pool_factor <= 0:
            raise ValueError("pool_factor must be a positive integer")
        if int(self.base_channels) != self.base_channels or self.base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")
        percentages = (self.norm_lo_pct, self.norm_hi_pct)
        if any(not math.isfinite(float(value)) for value in percentages):
            raise ValueError("normalization percentiles must be finite")
        if not 0.0 <= float(self.norm_lo_pct) <= 100.0:
            raise ValueError("norm_lo_pct must be between 0 and 100")
        if not 0.0 <= float(self.norm_hi_pct) <= 100.0:
            raise ValueError("norm_hi_pct must be between 0 and 100")
        if float(self.norm_hi_pct) < float(self.norm_lo_pct):
            raise ValueError("norm_hi_pct must be >= norm_lo_pct")
        clips = (self.norm_clip_lo, self.norm_clip_hi)
        if any(not math.isfinite(float(value)) for value in clips):
            raise ValueError("normalization clip bounds must be finite")
        if float(self.norm_clip_hi) < float(self.norm_clip_lo):
            raise ValueError("norm_clip_hi must be >= norm_clip_lo")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DeepCenterInferenceConfig":
        """Read only inference-relevant keys from a training config."""

        defaults = cls()
        values = {
            "pool_factor": mapping.get("pool_factor", defaults.pool_factor),
            "base_channels": mapping.get("base_channels", defaults.base_channels),
            "norm_lo_pct": mapping.get("norm_lo_pct", defaults.norm_lo_pct),
            "norm_hi_pct": mapping.get("norm_hi_pct", defaults.norm_hi_pct),
            "norm_clip_lo": mapping.get("norm_clip_lo", defaults.norm_clip_lo),
            "norm_clip_hi": mapping.get("norm_clip_hi", defaults.norm_clip_hi),
        }
        return cls(**values)


def _as_finite_volume(volume: Any, name: str = "volume") -> np.ndarray:
    array = np.asarray(volume, dtype=np.float32)
    if array.ndim != 3 or any(int(size) <= 0 for size in array.shape):
        raise ValueError("{} must be a non-empty 3D array".format(name))
    if not np.isfinite(array).all():
        raise ValueError("{} must contain only finite values".format(name))
    return array


def block_mean_xy(volume: Any, factor: int = 4) -> np.ndarray:
    """Average non-overlapping XY blocks while preserving the Z axis."""

    if int(factor) != factor or int(factor) <= 0:
        raise ValueError("factor must be a positive integer")
    array = _as_finite_volume(volume)
    factor = int(factor)
    if factor == 1:
        return array.astype(np.float32, copy=False)
    z, y, x = (int(value) for value in array.shape)
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    if y2 == 0 or x2 == 0:
        raise ValueError("factor is larger than the XY dimensions")
    cropped = array[:, :y2, :x2]
    pooled = cropped.reshape(
        z,
        y2 // factor,
        factor,
        x2 // factor,
        factor,
    ).mean(axis=(2, 4), dtype=np.float32)
    return np.asarray(pooled, dtype=np.float32)


def normalize_dynamic_range(
    volume: Any,
    lo_pct: float = 50.0,
    hi_pct: float = 99.5,
    clip_lo: float = -0.5,
    clip_hi: float = 6.0,
) -> np.ndarray:
    """Apply the public pack's percentile normalization and clipping."""

    if not all(
        math.isfinite(float(value))
        for value in (lo_pct, hi_pct, clip_lo, clip_hi)
    ):
        raise ValueError("normalization parameters must be finite")
    if not 0.0 <= float(lo_pct) <= 100.0 or not 0.0 <= float(hi_pct) <= 100.0:
        raise ValueError("percentiles must be between 0 and 100")
    if float(hi_pct) < float(lo_pct):
        raise ValueError("hi_pct must be >= lo_pct")
    if float(clip_hi) < float(clip_lo):
        raise ValueError("clip_hi must be >= clip_lo")
    array = _as_finite_volume(volume)
    lo, hi = np.percentile(array, [float(lo_pct), float(hi_pct)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    ratio = (array - float(lo)) / float(hi - lo)
    return np.clip(ratio, float(clip_lo), float(clip_hi)).astype(np.float32)


def preprocess_frame(
    frame: Any,
    config: DeepCenterInferenceConfig = DeepCenterInferenceConfig(),
) -> np.ndarray:
    """Convert an original-voxel frame into the model's pooled input."""

    pooled = block_mean_xy(frame, factor=config.pool_factor)
    return normalize_dynamic_range(
        pooled,
        lo_pct=config.norm_lo_pct,
        hi_pct=config.norm_hi_pct,
        clip_lo=config.norm_clip_lo,
        clip_hi=config.norm_clip_hi,
    )


def pooled_peak_to_original(
    peak_zyx: Sequence[float],
    pool_factor: int = 4,
) -> Tuple[float, float, float]:
    """Map a pooled heatmap index to original voxel coordinates."""

    if int(pool_factor) != pool_factor or int(pool_factor) <= 0:
        raise ValueError("pool_factor must be a positive integer")
    if len(peak_zyx) != 3:
        raise ValueError("peak_zyx must contain z, y, and x")
    values = tuple(float(value) for value in peak_zyx)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("peak coordinates must be finite")
    factor = int(pool_factor)
    offset = (factor - 1) / 2.0
    return (values[0], values[1] * factor + offset, values[2] * factor + offset)


def _maximum_filter_fallback(array: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return array.copy()
    padded = np.pad(
        array,
        ((radius, radius), (radius, radius), (radius, radius)),
        mode="constant",
        constant_values=-np.inf,
    )
    result = np.full_like(array, -np.inf, dtype=np.float32)
    for dz in range(2 * radius + 1):
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                result = np.maximum(
                    result,
                    padded[
                        dz : dz + array.shape[0],
                        dy : dy + array.shape[1],
                        dx : dx + array.shape[2],
                    ],
                )
    return result


def find_center_peaks(
    heatmap: Any,
    threshold: float = 0.4,
    min_distance: int = 1,
    max_peaks: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic local maxima above a center-heatmap threshold."""

    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if int(min_distance) != min_distance or int(min_distance) < 0:
        raise ValueError("min_distance must be a non-negative integer")
    if max_peaks is not None and (
        int(max_peaks) != max_peaks or int(max_peaks) < 0
    ):
        raise ValueError("max_peaks must be a non-negative integer or None")
    array = _as_finite_volume(heatmap, name="heatmap")
    radius = int(min_distance)
    try:
        from scipy.ndimage import maximum_filter  # type: ignore

        local = maximum_filter(
            array,
            size=2 * radius + 1,
            mode="constant",
            cval=-np.inf,
        )
    except ImportError:
        local = _maximum_filter_fallback(array, radius)
    mask = (array == local) & (array >= float(threshold))
    coords = np.argwhere(mask).astype(np.int64, copy=False)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.float32)
    scores = array[coords[:, 0], coords[:, 1], coords[:, 2]].astype(
        np.float32, copy=False
    )
    order = np.lexsort(
        (
            coords[:, 2],
            coords[:, 1],
            coords[:, 0],
            -scores.astype(np.float64),
        )
    )
    if max_peaks is not None:
        order = order[: int(max_peaks)]
    return coords[order], scores[order]


def load_deepcenter_model(
    checkpoint_path: Path,
    source_script: Path,
    device: str = "cpu",
) -> Tuple[Any, DeepCenterInferenceConfig, ModuleType]:
    """Load the public model source/checkpoint without importing torch eagerly."""

    checkpoint_path = Path(checkpoint_path)
    source_script = Path(source_script)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("DeepCenter checkpoint not found: {}".format(checkpoint_path))
    if not source_script.is_file():
        raise FileNotFoundError("DeepCenter source script not found: {}".format(source_script))
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("DeepCenter inference requires the optional torch dependency") from exc

    module_name = "biohub_deepcenter_source_{}".format(abs(hash(source_script)) & 0xFFFFFFFF)
    spec = importlib.util.spec_from_file_location(module_name, source_script)
    if spec is None or spec.loader is None:
        raise ImportError("could not load DeepCenter source script: {}".format(source_script))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "DeepCenterUNet3D"):
        raise AttributeError("DeepCenter source does not define DeepCenterUNet3D")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("DeepCenter checkpoint must contain a mapping")
    saved_config = checkpoint.get("config", {})
    if not isinstance(saved_config, Mapping):
        raise ValueError("DeepCenter checkpoint config must be a mapping")
    config = DeepCenterInferenceConfig.from_mapping(saved_config)
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("DeepCenter checkpoint does not contain model_state")
    model = module.DeepCenterUNet3D(
        in_channels=1,
        base_channels=int(config.base_channels),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, config, module


def predict_center_peaks(
    model: Any,
    frame: Any,
    config: DeepCenterInferenceConfig = DeepCenterInferenceConfig(),
    device: str = "cpu",
    threshold: float = 0.4,
    min_distance: int = 1,
    max_peaks: Optional[int] = None,
    time_index: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Infer heatmap peaks and map them to original voxel coordinates."""

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("DeepCenter inference requires the optional torch dependency") from exc
    prepared = preprocess_frame(frame, config=config)
    tensor = torch.from_numpy(prepared).to(device=device, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if getattr(logits, "ndim", None) != 5:
            raise ValueError("DeepCenter model output must have shape (N, C, Z, Y, X)")
        heatmap = torch.sigmoid(logits[0, 0]).detach().cpu().numpy().astype(np.float32)
    pooled_peaks, scores = find_center_peaks(
        heatmap,
        threshold=threshold,
        min_distance=min_distance,
        max_peaks=max_peaks,
    )
    result: list[dict[str, Any]] = []
    for pooled, score in zip(pooled_peaks, scores):
        original = pooled_peak_to_original(pooled, pool_factor=config.pool_factor)
        item: dict[str, Any] = {
            "z": float(original[0]),
            "y": float(original[1]),
            "x": float(original[2]),
            "pooled_z": int(pooled[0]),
            "pooled_y": int(pooled[1]),
            "pooled_x": int(pooled[2]),
            "score": float(score),
        }
        if time_index is not None:
            if int(time_index) != time_index or int(time_index) < 0:
                raise ValueError("time_index must be a non-negative integer")
            item["t"] = int(time_index)
        result.append(item)
    return result


def _position(item: Mapping[str, Any], require_time: bool = True) -> Tuple[Optional[int], float, float, float]:
    keys = ("z", "y", "x")
    try:
        coordinates = tuple(float(item[key]) for key in keys)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("positions must contain finite z, y, and x") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("positions must contain finite z, y, and x")
    if require_time:
        try:
            raw_t = item["t"]
            if int(raw_t) != raw_t or int(raw_t) < 0:
                raise ValueError("t must be a non-negative integer")
            time_index: Optional[int] = int(raw_t)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("positions must contain a non-negative integer t") from exc
    else:
        time_index = None
    return time_index, coordinates[0], coordinates[1], coordinates[2]


def _distance_um(
    first: Tuple[float, float, float],
    second: Tuple[float, float, float],
    scale_um: Sequence[float],
) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    scale = np.asarray(tuple(float(value) for value in scale_um), dtype=np.float64)
    return float(np.sqrt(np.sum((delta * scale) ** 2)))


def rank_center_rescue_candidates(
    peaks: Iterable[Mapping[str, Any]],
    endpoint_positions: Iterable[Mapping[str, Any]] = (),
    existing_nodes: Iterable[Mapping[str, Any]] = (),
    max_endpoint_distance_um: float = 7.0,
    min_existing_distance_um: float = 2.0,
    scale_um: Sequence[float] = VOXEL_SCALE_UM,
) -> list[dict[str, Any]]:
    """Rank center peaks near endpoints while rejecting duplicate nodes.

    The result is a reviewable proposal list only.  It intentionally does not
    add nodes or edges to a tracking graph.
    """

    limits = (max_endpoint_distance_um, min_existing_distance_um)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in limits):
        raise ValueError("distance gates must be finite and non-negative")
    scale = tuple(float(value) for value in scale_um)
    if len(scale) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in scale):
        raise ValueError("scale_um must contain three positive finite values")

    endpoints_by_time: dict[
        int, list[Tuple[Tuple[float, float, float], Mapping[str, Any]]]
    ] = {}
    for endpoint in endpoint_positions:
        time_index, z, y, x = _position(endpoint)
        assert time_index is not None
        endpoints_by_time.setdefault(time_index, []).append(
            ((z, y, x), endpoint)
        )
    existing_by_time: dict[int, list[Tuple[float, float, float]]] = {}
    for node in existing_nodes:
        time_index, z, y, x = _position(node)
        assert time_index is not None
        existing_by_time.setdefault(time_index, []).append((z, y, x))

    ranked: list[dict[str, Any]] = []
    for peak in peaks:
        if not isinstance(peak, Mapping):
            raise ValueError("peaks must be mappings")
        time_index, z, y, x = _position(peak)
        assert time_index is not None
        try:
            score = float(peak["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("peaks must contain a finite score") from exc
        if not math.isfinite(score):
            raise ValueError("peaks must contain a finite score")
        position = (z, y, x)
        endpoint_records = endpoints_by_time.get(time_index, [])
        endpoint_distances = [
            _distance_um(position, endpoint_position, scale)
            for endpoint_position, _endpoint in endpoint_records
        ]
        if not endpoint_distances:
            continue
        nearest_endpoint_index = min(
            range(len(endpoint_distances)),
            key=lambda index: (endpoint_distances[index], index),
        )
        endpoint_distance = endpoint_distances[nearest_endpoint_index]
        if endpoint_distance > float(max_endpoint_distance_um):
            continue
        existing_distances = [
            _distance_um(position, node, scale)
            for node in existing_by_time.get(time_index, [])
        ]
        nearest_existing = min(existing_distances) if existing_distances else None
        if nearest_existing is not None and nearest_existing < float(min_existing_distance_um):
            continue
        item = dict(peak)
        endpoint_metadata = endpoint_records[nearest_endpoint_index][1]
        for key, value in endpoint_metadata.items():
            if key not in {"t", "z", "y", "x"}:
                item.setdefault(key, value)
        item.update(
            {
                "t": time_index,
                "z": float(z),
                "y": float(y),
                "x": float(x),
                "score": score,
                "endpoint_distance_um": float(endpoint_distance),
                "nearest_existing_distance_um": (
                    float(nearest_existing) if nearest_existing is not None else None
                ),
            }
        )
        ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["endpoint_distance_um"]),
            int(item["t"]),
            float(item["z"]),
            float(item["y"]),
            float(item["x"]),
        )
    )
    return ranked


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
        "source_id": int(source_id),
        "target_id": int(target_id),
    }


def add_center_rescue_nodes(
    rows: Iterable[Mapping[str, Any]],
    peaks: Iterable[Mapping[str, Any]],
    *,
    frame_count: int,
    spatial_shape: Sequence[int],
    max_endpoint_distance_um: float = 7.0,
    min_existing_distance_um: float = 2.0,
    scale_um: Sequence[float] = VOXEL_SCALE_UM,
    max_out_degree: int = 2,
    require_unique_parent: bool = True,
) -> Tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add at most one gated DeepCenter rescue node per parent.

    Endpoints are extrapolated from an adjacent parent edge using constant
    velocity.  A heatmap peak is eligible only when it is close to that
    endpoint, separated from existing nodes in the same frame, and unique for
    its parent.  The new node receives exactly one adjacent parent edge.  The
    function is intentionally a post-processor with an explicit report; it is
    not a replacement for the primary detector or ILP solver.
    """

    if int(frame_count) != frame_count or int(frame_count) <= 0:
        raise ValueError("frame_count must be a positive integer")
    if len(spatial_shape) != 3 or any(int(size) <= 0 for size in spatial_shape):
        raise ValueError("spatial_shape must contain three positive dimensions")
    if int(max_out_degree) not in (1, 2):
        raise ValueError("max_out_degree must be 1 or 2")
    radius = float(max_endpoint_distance_um)
    duplicate_radius = float(min_existing_distance_um)
    if (
        not math.isfinite(radius)
        or radius < 0.0
        or not math.isfinite(duplicate_radius)
        or duplicate_radius < 0.0
    ):
        raise ValueError("distance gates must be finite and non-negative")
    scale = tuple(float(value) for value in scale_um)
    if len(scale) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in scale):
        raise ValueError("scale_um must contain three positive finite values")

    result = [dict(row) for row in rows]
    node_rows = [row for row in result if str(row.get("row_type")) == "node"]
    edge_rows = [row for row in result if str(row.get("row_type")) == "edge"]
    if not node_rows:
        return result, {
            "endpoint_count": 0,
            "ranked_candidate_count": 0,
            "accepted_candidate_count": 0,
            "added_nodes": 0,
            "added_edges": 0,
            "require_unique_parent": bool(require_unique_parent),
        }

    nodes: dict[int, dict[str, Any]] = {}
    nodes_by_time: dict[int, list[dict[str, Any]]] = {}
    dataset_names = {str(row["dataset"]) for row in node_rows}
    if len(dataset_names) != 1:
        raise ValueError("add_center_rescue_nodes expects rows from one dataset")
    dataset = next(iter(dataset_names))
    for row in node_rows:
        node_id = int(row["node_id"])
        if node_id in nodes:
            raise ValueError("duplicate node id: {}".format(node_id))
        node = dict(row)
        nodes[node_id] = node
        nodes_by_time.setdefault(int(node["t"]), []).append(node)

    outgoing: dict[int, list[int]] = {}
    adjacent_parents: dict[int, list[int]] = {}
    for row in edge_rows:
        source_id = int(row["source_id"])
        target_id = int(row["target_id"])
        if source_id not in nodes or target_id not in nodes:
            raise ValueError("edge references a missing node")
        outgoing.setdefault(source_id, []).append(target_id)
        source_time = int(nodes[source_id]["t"])
        target_time = int(nodes[target_id]["t"])
        if target_time == source_time + 1:
            adjacent_parents.setdefault(target_id, []).append(source_id)

    def voxel_position(row: Mapping[str, Any]) -> Tuple[float, float, float]:
        return tuple(float(row[key]) for key in ("z", "y", "x"))  # type: ignore[return-value]

    endpoints: list[dict[str, Any]] = []
    for source_id in sorted(nodes):
        source = nodes[source_id]
        source_time = int(source["t"])
        target_time = source_time + 1
        if target_time >= int(frame_count):
            continue
        if len(outgoing.get(source_id, [])) >= int(max_out_degree):
            continue
        parent_ids = adjacent_parents.get(source_id, [])
        if require_unique_parent and len(parent_ids) != 1:
            continue
        current = voxel_position(source)
        if len(parent_ids) == 1:
            previous = voxel_position(nodes[parent_ids[0]])
            predicted = tuple(2.0 * now - before for now, before in zip(current, previous))
        else:
            predicted = current
        nearby_existing = [
            voxel_position(target)
            for target in nodes_by_time.get(target_time, [])
            if _distance_um(predicted, voxel_position(target), scale) <= radius
        ]
        if nearby_existing:
            continue
        endpoints.append(
            {
                "t": target_time,
                "z": predicted[0],
                "y": predicted[1],
                "x": predicted[2],
                "parent_node_id": source_id,
            }
        )

    ranked = rank_center_rescue_candidates(
        peaks,
        endpoint_positions=endpoints,
        existing_nodes=node_rows,
        max_endpoint_distance_um=radius,
        min_existing_distance_um=duplicate_radius,
        scale_um=scale,
    )
    accepted: list[dict[str, Any]] = []
    used_parents: set[int] = set()
    accepted_positions: list[Tuple[int, Tuple[float, float, float]]] = []
    for candidate in ranked:
        parent_id = int(candidate.get("parent_node_id", -1))
        if parent_id in used_parents or parent_id not in nodes:
            continue
        candidate_time = int(candidate["t"])
        candidate_position = (
            float(candidate["z"]),
            float(candidate["y"]),
            float(candidate["x"]),
        )
        if any(
            timepoint == candidate_time
            and _distance_um(candidate_position, position, scale) < duplicate_radius
            for timepoint, position in accepted_positions
        ):
            continue
        used_parents.add(parent_id)
        accepted_positions.append((candidate_time, candidate_position))
        accepted.append(candidate)

    next_node_id = max(int(row["node_id"]) for row in node_rows) + 1
    added_nodes = 0
    added_edges = 0
    bounds = tuple(int(size) - 1 for size in spatial_shape)
    for candidate in accepted:
        parent_id = int(candidate["parent_node_id"])
        if len(outgoing.get(parent_id, [])) >= int(max_out_degree):
            continue
        node_id = next_node_id
        next_node_id += 1
        coordinates = [
            max(0, min(bounds[index], int(math.floor(float(candidate[key]) + 0.5))))
            for index, key in enumerate(("z", "y", "x"))
        ]
        node = {
            "id": -1,
            "dataset": dataset,
            "row_type": "node",
            "node_id": node_id,
            "t": int(candidate["t"]),
            "z": coordinates[0],
            "y": coordinates[1],
            "x": coordinates[2],
            "source_id": -1,
            "target_id": -1,
        }
        result.append(node)
        result.append(_edge_row(dataset, parent_id, node_id))
        outgoing.setdefault(parent_id, []).append(node_id)
        nodes[node_id] = node
        added_nodes += 1
        added_edges += 1

    for row_id, row in enumerate(result):
        row["id"] = row_id
    return result, {
        "endpoint_count": len(endpoints),
        "ranked_candidate_count": len(ranked),
        "accepted_candidate_count": len(accepted),
        "added_nodes": added_nodes,
        "added_edges": added_edges,
        "max_endpoint_distance_um": radius,
        "min_existing_distance_um": duplicate_radius,
        "max_out_degree": int(max_out_degree),
        "require_unique_parent": bool(require_unique_parent),
    }
