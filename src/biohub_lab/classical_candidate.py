"""Run the H1 classical peak/link candidate on OME-Zarr datasets."""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .classical import (
    TrackingGraph,
    infer_division_links,
    remove_isolated_nodes,
    track_frames,
    write_submission_csv,
)
from .evaluation import convert_csv_to_geff, evaluate_geff_directory
from .ome_zarr import OpenedOmeZarr, discover_zarr_datasets, open_ome_zarr


def quantile_linear(values: Iterable[float], quantile: float) -> float:
    """Compute a NumPy-compatible linear quantile using only the stdlib."""

    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("cannot compute a quantile from no finite values")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def threshold_for_dataset(
    explicit_threshold: Optional[float], samples: Sequence[float], quantile: float
) -> float:
    if explicit_threshold is not None:
        value = float(explicit_threshold)
        if not math.isfinite(value):
            raise ValueError("explicit threshold must be finite")
        return value
    return quantile_linear(samples, quantile)


def _flatten_sample(value: Any, limit: int = 100000) -> List[float]:
    """Flatten a frame while bounding the number of sampled voxels."""

    if hasattr(value, "ravel"):
        flat = value.ravel()
        stride = max(1, int(math.ceil(len(flat) / float(limit))))
        return [float(item) for item in flat[::stride]]
    result: List[float] = []

    def visit(item: Any) -> None:
        if len(result) >= limit:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
                if len(result) >= limit:
                    return
        else:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                result.append(number)

    visit(value)
    return result


def _sample_intensities(opened: OpenedOmeZarr, max_frames: Optional[int]) -> List[float]:
    frame_count = int(opened.array.shape[0])
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("dataset has no frames to process")
    sample_count = min(frame_count, 8)
    if sample_count == 1:
        indices = [0]
    else:
        indices = sorted({int(round(index)) for index in _linspace(0, frame_count - 1, sample_count)})
    samples: List[float] = []
    for index in indices:
        samples.extend(_flatten_sample(opened.array[index]))
    return samples


def _linspace(start: int, stop: int, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [start + index * step for index in range(count)]


def _process_dataset(
    dataset_path: Path,
    array_key: str,
    explicit_threshold: Optional[float],
    quantile: float,
    max_frames: Optional[int],
    neighborhood: Sequence[int],
    max_link_distance_um: float,
    max_detections_per_frame: Optional[int],
    infer_divisions: bool = False,
    sister_max_distance_um: float = 5.0,
    midpoint_tolerance_um: float = 3.0,
    max_parent_displacement_delta_um: float = 3.0,
    min_sibling_intensity_ratio: float = 0.9,
    filter_isolated_nodes: bool = False,
    ensure_timepoint_coverage: bool = False,
) -> TrackingGraph:
    opened = open_ome_zarr(dataset_path, array_key=array_key)
    threshold = threshold_for_dataset(
        explicit_threshold,
        _sample_intensities(opened, max_frames),
        quantile,
    )
    frame_count = int(opened.array.shape[0])
    if max_frames is not None:
        frame_count = min(frame_count, int(max_frames))
    graph = track_frames(
        (opened.array[index] for index in range(frame_count)),
        dataset=dataset_path.stem,
        threshold=threshold,
        neighborhood=neighborhood,
        scale_um=opened.scale,
        max_link_distance_um=max_link_distance_um,
        max_detections_per_frame=max_detections_per_frame,
        ensure_timepoint_coverage=ensure_timepoint_coverage,
    )
    if infer_divisions:
        graph = infer_division_links(
            graph,
            scale_um=opened.scale,
            max_link_distance_um=max_link_distance_um,
            sister_max_distance_um=sister_max_distance_um,
            midpoint_tolerance_um=midpoint_tolerance_um,
            max_parent_displacement_delta_um=max_parent_displacement_delta_um,
            min_sibling_intensity_ratio=min_sibling_intensity_ratio,
        )
    if filter_isolated_nodes:
        graph = remove_isolated_nodes(graph)
    graph.threshold = threshold
    return graph


def build_graphs(
    data_dir: Path,
    array_key: str = "0",
    explicit_threshold: Optional[float] = None,
    quantile: float = 0.995,
    max_frames: Optional[int] = None,
    neighborhood: Sequence[int] = (1, 1, 1),
    max_link_distance_um: float = 7.0,
    max_detections_per_frame: Optional[int] = 2000,
    infer_divisions: bool = False,
    sister_max_distance_um: float = 5.0,
    midpoint_tolerance_um: float = 3.0,
    max_parent_displacement_delta_um: float = 3.0,
    min_sibling_intensity_ratio: float = 0.9,
    filter_isolated_nodes: bool = False,
    ensure_timepoint_coverage: bool = False,
    workers: int = 1,
) -> List[TrackingGraph]:
    dataset_paths = discover_zarr_datasets(Path(data_dir), require_geff=False)
    worker_count = max(1, min(int(workers), len(dataset_paths))) if dataset_paths else 1

    def process(dataset_path: Path) -> TrackingGraph:
        started = time.monotonic()
        graph = _process_dataset(
            dataset_path,
            array_key,
            explicit_threshold,
            quantile,
            max_frames,
            neighborhood,
            max_link_distance_um,
            max_detections_per_frame,
            infer_divisions,
            sister_max_distance_um,
            midpoint_tolerance_um,
            max_parent_displacement_delta_um,
            min_sibling_intensity_ratio,
            filter_isolated_nodes,
            ensure_timepoint_coverage,
        )
        print(
            json.dumps(
                {
                    "event": "dataset_processed",
                    "data_dir": str(Path(data_dir)),
                    "dataset": dataset_path.stem,
                    "nodes": len(graph.nodes),
                    "edges": len(graph.edges),
                    "seconds": round(time.monotonic() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return graph

    if worker_count == 1:
        graphs = [process(dataset_path) for dataset_path in dataset_paths]
    else:
        # executor.map preserves the sorted dataset order, so concurrent
        # processing cannot change CSV row or node-id determinism.
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            graphs = list(executor.map(process, dataset_paths))
    if not graphs:
        raise FileNotFoundError("no .zarr datasets found in {}".format(data_dir))
    return graphs


def summarise_graphs(graphs: Sequence[TrackingGraph], thresholds: Mapping[str, float]) -> Dict[str, Any]:
    return {
        "dataset_count": len(graphs),
        "node_count": sum(len(graph.nodes) for graph in graphs),
        "edge_count": sum(len(graph.edges) for graph in graphs),
        "nodes_by_dataset": {graph.dataset: len(graph.nodes) for graph in graphs},
        "edges_by_dataset": {graph.dataset: len(graph.edges) for graph in graphs},
        "thresholds": dict(thresholds),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--neighborhood", type=int, nargs=3, default=(1, 1, 1), metavar=("Z", "Y", "X"))
    parser.add_argument("--max-link-distance-um", type=float, default=7.0)
    parser.add_argument("--max-detections-per-frame", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--infer-divisions", action="store_true")
    parser.add_argument("--sister-max-distance-um", type=float, default=5.0)
    parser.add_argument("--midpoint-tolerance-um", type=float, default=3.0)
    parser.add_argument("--max-parent-displacement-delta-um", type=float, default=3.0)
    parser.add_argument("--min-sibling-intensity-ratio", type=float, default=0.9)
    parser.add_argument("--filter-isolated-nodes", action="store_true")
    parser.add_argument(
        "--ensure-timepoint-coverage",
        action="store_true",
        help="Add a deterministic brightest-voxel fallback when a frame has no detections.",
    )
    parser.add_argument(
        "--validation-data-dir",
        type=Path,
        help="Optional in-domain image directory; final submission still comes from --data-dir.",
    )
    parser.add_argument("--evaluate-gt-dir", type=Path)
    parser.add_argument("--holdout-data-dir", type=Path)
    parser.add_argument("--holdout-gt-dir", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--max-eval-distance", type=float, default=7.0)
    return parser


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    graphs = build_graphs(
        args.data_dir,
        array_key=args.array_key,
        explicit_threshold=args.threshold,
        quantile=args.quantile,
        max_frames=args.max_frames,
        neighborhood=args.neighborhood,
        max_link_distance_um=args.max_link_distance_um,
        max_detections_per_frame=args.max_detections_per_frame,
        infer_divisions=getattr(args, "infer_divisions", False),
        sister_max_distance_um=getattr(args, "sister_max_distance_um", 5.0),
        midpoint_tolerance_um=getattr(args, "midpoint_tolerance_um", 3.0),
        max_parent_displacement_delta_um=getattr(args, "max_parent_displacement_delta_um", 3.0),
        min_sibling_intensity_ratio=getattr(args, "min_sibling_intensity_ratio", 0.9),
        filter_isolated_nodes=getattr(args, "filter_isolated_nodes", False),
        ensure_timepoint_coverage=getattr(args, "ensure_timepoint_coverage", False),
        workers=getattr(args, "workers", 1),
    )
    thresholds = {}
    for graph in graphs:
        if graph.threshold is not None:
            thresholds[graph.dataset] = graph.threshold
        elif graph.nodes:
            thresholds[graph.dataset] = min(node.intensity for node in graph.nodes)
        else:
            thresholds[graph.dataset] = float("nan")
    artifact = write_submission_csv(graphs, args.run_dir / "submission.csv")
    metrics: Dict[str, Any] = {
        "artifact": str(artifact),
        "candidate_metrics": summarise_graphs(graphs, thresholds),
        "division_inference": {
            "enabled": bool(getattr(args, "infer_divisions", False)),
            "sister_max_distance_um": float(getattr(args, "sister_max_distance_um", 5.0)),
            "midpoint_tolerance_um": float(getattr(args, "midpoint_tolerance_um", 3.0)),
            "max_parent_displacement_delta_um": float(getattr(args, "max_parent_displacement_delta_um", 3.0)),
            "min_sibling_intensity_ratio": float(getattr(args, "min_sibling_intensity_ratio", 0.9)),
            "filter_isolated_nodes": bool(getattr(args, "filter_isolated_nodes", False)),
        },
        "evaluation": {},
    }

    evaluation_csv = artifact
    validation_data_dir = getattr(args, "validation_data_dir", None)
    if validation_data_dir is not None:
        evaluation_graphs = build_graphs(
            validation_data_dir,
            array_key=args.array_key,
            explicit_threshold=args.threshold,
            quantile=args.quantile,
            max_frames=args.max_frames,
            neighborhood=args.neighborhood,
            max_link_distance_um=args.max_link_distance_um,
            max_detections_per_frame=args.max_detections_per_frame,
            infer_divisions=getattr(args, "infer_divisions", False),
            sister_max_distance_um=getattr(args, "sister_max_distance_um", 5.0),
            midpoint_tolerance_um=getattr(args, "midpoint_tolerance_um", 3.0),
            max_parent_displacement_delta_um=getattr(args, "max_parent_displacement_delta_um", 3.0),
            min_sibling_intensity_ratio=getattr(args, "min_sibling_intensity_ratio", 0.9),
            filter_isolated_nodes=getattr(args, "filter_isolated_nodes", False),
            ensure_timepoint_coverage=getattr(args, "ensure_timepoint_coverage", False),
            workers=getattr(args, "workers", 1),
        )
        evaluation_csv = write_submission_csv(
            evaluation_graphs,
            args.run_dir / "validation_submission.csv",
        )

    if args.evaluate_gt_dir is not None:
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required for official evaluation")
        pred_geffs = convert_csv_to_geff(
            evaluation_csv,
            args.run_dir / "pred_geffs",
            args.baseline_root,
        )
        local = evaluate_geff_directory(
            pred_geffs,
            args.evaluate_gt_dir,
            baseline_root=args.baseline_root,
            max_distance=args.max_eval_distance,
        )
        metrics["local_score"] = local.get("score")
        metrics["in_domain_score"] = local.get("score")
        metrics["evaluation"]["in_domain"] = local

    if args.holdout_data_dir is not None:
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required for official evaluation")
        holdout_graphs = build_graphs(
            args.holdout_data_dir,
            array_key=args.array_key,
            explicit_threshold=args.threshold,
            quantile=args.quantile,
            max_frames=args.max_frames,
            neighborhood=args.neighborhood,
            max_link_distance_um=args.max_link_distance_um,
            max_detections_per_frame=args.max_detections_per_frame,
            infer_divisions=getattr(args, "infer_divisions", False),
            sister_max_distance_um=getattr(args, "sister_max_distance_um", 5.0),
            midpoint_tolerance_um=getattr(args, "midpoint_tolerance_um", 3.0),
            max_parent_displacement_delta_um=getattr(args, "max_parent_displacement_delta_um", 3.0),
            min_sibling_intensity_ratio=getattr(args, "min_sibling_intensity_ratio", 0.9),
            filter_isolated_nodes=getattr(args, "filter_isolated_nodes", False),
            ensure_timepoint_coverage=getattr(args, "ensure_timepoint_coverage", False),
            workers=getattr(args, "workers", 1),
        )
        holdout_artifact = write_submission_csv(holdout_graphs, args.run_dir / "holdout_submission.csv")
        holdout_gt = args.holdout_gt_dir or args.holdout_data_dir
        holdout_geffs = convert_csv_to_geff(
            holdout_artifact,
            args.run_dir / "holdout_geffs",
            args.baseline_root,
        )
        holdout = evaluate_geff_directory(
            holdout_geffs,
            holdout_gt,
            baseline_root=args.baseline_root,
            max_distance=args.max_eval_distance,
        )
        metrics["holdout_score"] = holdout.get("score")
        metrics["evaluation"]["holdout"] = holdout

    with (args.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True, default=str)
        handle.write("\n")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
