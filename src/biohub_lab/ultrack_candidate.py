"""Optional Ultrack bridge for label-based 3D tracking.

The control plane remains independent of Ultrack.  The adapter loads the
optional package only when a real candidate requests it, passes segmentation
labels to the documented ``Tracker.track`` API, and converts the exported
tracks layer into the project's deterministic CSV graph contract.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .classical import Detection, TrackingGraph, write_submission_csv
from .evaluation import convert_csv_to_geff, evaluate_geff_directory
from .ome_zarr import discover_zarr_datasets, open_ome_zarr
from .validation import validate_submission


_MISSING = object()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _integer(value: Any, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be an integer".format(field)) from exc
    if not math.isfinite(numeric) or int(numeric) != numeric:
        raise ValueError("{} must be a finite integer".format(field))
    return int(numeric)


def _coordinate(value: Any, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(field)) from exc
    if not math.isfinite(numeric):
        raise ValueError("{} must be finite".format(field))
    return int(math.floor(numeric + 0.5))


def _identifier(value: Any) -> Any:
    if _is_missing(value):
        raise ValueError("track node id cannot be missing")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric) and int(numeric) == numeric:
        return int(numeric)
    return str(value)


def _parent_values(value: Any) -> List[Any]:
    if value is _MISSING or _is_missing(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    parents: List[Any] = []
    for item in values:
        if _is_missing(item):
            continue
        parent = _identifier(item)
        if isinstance(parent, int) and parent < 0:
            continue
        parents.append(parent)
    return parents


def _tracks_rows(tracks: Any) -> List[Mapping[str, Any]]:
    if hasattr(tracks, "to_dict"):
        try:
            rows = tracks.to_dict(orient="records")
        except TypeError:
            rows = tracks.to_dict("records")
    else:
        rows = list(tracks)
    if not isinstance(rows, list):
        rows = list(rows)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Ultrack tracks export must contain mapping-like rows")
    return rows


def _lineage_parent(lineage: Optional[Mapping[Any, Any]], key: Any) -> Any:
    if not lineage:
        return _MISSING
    for candidate in (key, str(key)):
        try:
            if candidate in lineage:
                return lineage[candidate]
        except TypeError:
            continue
    return _MISSING


def graph_from_tracks_rows(
    dataset: str,
    tracks: Any,
    lineage: Optional[Mapping[Any, Any]] = None,
) -> TrackingGraph:
    """Convert Ultrack's tracks-layer rows into a submission graph.

    Ultrack exports ``t, z, y, x, id`` and ``parent_id`` when parent metadata
    is enabled.  Parent links are retained, including divisions; non-adjacent
    gaps are rejected because the competition CSV contract is frame-adjacent.
    """

    if not str(dataset):
        raise ValueError("dataset must be non-empty")
    prepared: List[Dict[str, Any]] = []
    seen: set = set()
    for index, row in enumerate(_tracks_rows(tracks)):
        raw_id = row.get("id", index)
        node_key = _identifier(raw_id)
        if node_key in seen:
            raise ValueError("Ultrack export contains duplicate node id: {}".format(node_key))
        seen.add(node_key)
        time_index = _integer(row.get("t"), "t")
        z = _coordinate(row.get("z"), "z")
        y = _coordinate(row.get("y"), "y")
        x = _coordinate(row.get("x"), "x")
        if min(time_index, z, y, x) < 0:
            raise ValueError("Ultrack coordinates and time must be non-negative")
        prepared.append(
            {
                "key": node_key,
                "t": time_index,
                "z": z,
                "y": y,
                "x": x,
                "intensity": float(row.get("quality", row.get("prob", 1.0)) or 1.0),
                "parent": row.get("parent_id", _MISSING),
            }
        )
    prepared.sort(key=lambda item: (item["t"], str(item["key"]), item["z"], item["y"], item["x"]))
    node_by_key = {item["key"]: index for index, item in enumerate(prepared)}
    nodes = [
        Detection(
            t=item["t"],
            z=item["z"],
            y=item["y"],
            x=item["x"],
            intensity=item["intensity"] if math.isfinite(item["intensity"]) else 1.0,
            node_id=index,
        )
        for index, item in enumerate(prepared)
    ]
    edges: List[Tuple[int, int]] = []
    for child_index, item in enumerate(prepared):
        parent_value = item["parent"]
        if parent_value is _MISSING:
            parent_value = _lineage_parent(lineage, item["key"])
        for parent_key in _parent_values(parent_value):
            if parent_key not in node_by_key:
                raise ValueError("Ultrack parent id does not reference a node: {}".format(parent_key))
            parent_index = node_by_key[parent_key]
            if nodes[parent_index].t + 1 != nodes[child_index].t:
                raise ValueError("Ultrack parent edge must connect adjacent timepoints")
            edges.append((parent_index, child_index))
    return TrackingGraph(dataset=str(dataset), nodes=nodes, edges=sorted(set(edges)))


def track_labels_with_ultrack(
    labels: Any,
    dataset: str,
    working_dir: Path,
    scale_um: Sequence[float] = (1.625, 0.40625, 0.40625),
    min_frontier: float = 0.0,
    max_link_distance_um: float = 15.0,
    ultrack_module: Optional[Any] = None,
) -> TrackingGraph:
    """Run the documented Ultrack label path and return a graph."""

    if ultrack_module is None:
        try:
            import ultrack as ultrack_module  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Ultrack adapter requires the selected external Python environment") from exc
    config = ultrack_module.MainConfig()
    data_config = getattr(config, "data_config", getattr(config, "data", None))
    segmentation_config = getattr(config, "segmentation_config", getattr(config, "segmentation", None))
    linking_config = getattr(config, "linking_config", getattr(config, "linking", None))
    if data_config is None or segmentation_config is None or linking_config is None:
        raise AttributeError("MainConfig does not expose data/segmentation/linking configuration")
    isolated_dir = Path(working_dir)
    isolated_dir.mkdir(parents=True, exist_ok=True)
    data_config.working_dir = isolated_dir
    data_config.database_file_name = "data.db"
    segmentation_config.min_frontier = float(min_frontier)
    linking_config.max_distance = float(max_link_distance_um)
    tracker = ultrack_module.Tracker(config=config)
    tracker.track(labels=labels, scale=tuple(float(value) for value in scale_um), overwrite="all")
    exported = tracker.to_tracks_layer()
    tracks = exported[0] if isinstance(exported, (tuple, list)) else exported
    lineage = exported[1] if isinstance(exported, (tuple, list)) and len(exported) > 1 else None
    return graph_from_tracks_rows(dataset, tracks, lineage=lineage)


def _load_labels_graphs(
    data_dir: Path,
    labels_dir: Path,
    run_dir: Path,
    scale_um: Optional[Sequence[float]] = None,
    max_frames: Optional[int] = None,
    min_frontier: float = 0.0,
    max_link_distance_um: float = 15.0,
) -> List[TrackingGraph]:
    data_paths = {path.stem: path for path in discover_zarr_datasets(Path(data_dir), require_geff=False)}
    label_paths = {path.stem: path for path in discover_zarr_datasets(Path(labels_dir), require_geff=False)}
    missing = sorted(set(data_paths) - set(label_paths))
    if missing:
        raise FileNotFoundError("labels are missing for datasets: {}".format(", ".join(missing)))
    graphs: List[TrackingGraph] = []
    for name in sorted(data_paths):
        opened = open_ome_zarr(label_paths[name])
        frame_count = int(opened.array.shape[0])
        if max_frames is not None:
            frame_count = min(frame_count, int(max_frames))
        if frame_count <= 0:
            raise ValueError("labels dataset has no frames: {}".format(label_paths[name]))
        if len(tuple(opened.array.shape)) != 4:
            raise ValueError("Ultrack labels must have shape (T, Z, Y, X)")
        labels = opened.array[:frame_count]
        physical_scale = tuple(float(value) for value in (scale_um or opened.scale))
        graphs.append(
            track_labels_with_ultrack(
                labels,
                dataset=name,
                working_dir=Path(run_dir) / "ultrack" / name,
                scale_um=physical_scale,
                min_frontier=min_frontier,
                max_link_distance_um=max_link_distance_um,
            )
        )
    if not graphs:
        raise FileNotFoundError("no OME-Zarr datasets found in {}".format(data_dir))
    return graphs


def summarise_graphs(graphs: Sequence[TrackingGraph]) -> Dict[str, Any]:
    return {
        "dataset_count": len(graphs),
        "node_count": sum(len(graph.nodes) for graph in graphs),
        "edge_count": sum(len(graph.edges) for graph in graphs),
        "nodes_by_dataset": {graph.dataset: len(graph.nodes) for graph in graphs},
        "edges_by_dataset": {graph.dataset: len(graph.edges) for graph in graphs},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-frontier", type=float, default=0.0)
    parser.add_argument("--max-link-distance-um", type=float, default=15.0)
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument("--validation-labels-dir", type=Path)
    parser.add_argument("--evaluate-gt-dir", type=Path)
    parser.add_argument("--holdout-data-dir", type=Path)
    parser.add_argument("--holdout-labels-dir", type=Path)
    parser.add_argument("--holdout-gt-dir", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--max-eval-distance", type=float, default=7.0)
    return parser


def _evaluate(
    csv_path: Path,
    run_dir: Path,
    gt_dir: Path,
    baseline_root: Path,
    max_distance: float,
) -> Dict[str, Any]:
    predicted = convert_csv_to_geff(csv_path, run_dir, baseline_root)
    return evaluate_geff_directory(predicted, gt_dir, baseline_root=baseline_root, max_distance=max_distance)


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    final_graphs = _load_labels_graphs(
        args.data_dir,
        args.labels_dir,
        args.run_dir,
        max_frames=args.max_frames,
        min_frontier=args.min_frontier,
        max_link_distance_um=args.max_link_distance_um,
    )
    artifact = write_submission_csv(final_graphs, args.run_dir / "submission.csv")
    metrics: Dict[str, Any] = {
        "artifact": str(artifact),
        "candidate_metrics": summarise_graphs(final_graphs),
        "evaluation": {},
    }
    if args.validation_data_dir is not None:
        if args.validation_labels_dir is None:
            raise ValueError("--validation-labels-dir is required with --validation-data-dir")
        validation_graphs = _load_labels_graphs(
            args.validation_data_dir,
            args.validation_labels_dir,
            args.run_dir,
            max_frames=args.max_frames,
            min_frontier=args.min_frontier,
            max_link_distance_um=args.max_link_distance_um,
        )
        validation_csv = write_submission_csv(validation_graphs, args.run_dir / "validation_submission.csv")
    else:
        validation_csv = artifact
    if args.evaluate_gt_dir is not None:
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required for official evaluation")
        local = _evaluate(validation_csv, args.run_dir / "pred_geffs", args.evaluate_gt_dir, args.baseline_root, args.max_eval_distance)
        metrics["local_score"] = local.get("score")
        metrics["in_domain_score"] = local.get("score")
        metrics["evaluation"]["in_domain"] = local
    if args.holdout_data_dir is not None:
        if args.holdout_labels_dir is None:
            raise ValueError("--holdout-labels-dir is required with --holdout-data-dir")
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required for official evaluation")
        holdout_graphs = _load_labels_graphs(
            args.holdout_data_dir,
            args.holdout_labels_dir,
            args.run_dir,
            max_frames=args.max_frames,
            min_frontier=args.min_frontier,
            max_link_distance_um=args.max_link_distance_um,
        )
        holdout_csv = write_submission_csv(holdout_graphs, args.run_dir / "holdout_submission.csv")
        holdout_gt = args.holdout_gt_dir or args.holdout_data_dir
        holdout = _evaluate(holdout_csv, args.run_dir / "holdout_geffs", holdout_gt, args.baseline_root, args.max_eval_distance)
        metrics["holdout_score"] = holdout.get("score")
        metrics["evaluation"]["holdout"] = holdout
    with (args.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True, default=str)
        handle.write("\n")
    report = validate_submission(artifact)
    if not report.valid:
        raise ValueError("Ultrack CSV failed local validation: {}".format("; ".join(report.errors[:3])))
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
