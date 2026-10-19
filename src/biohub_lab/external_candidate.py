"""Core for detector-backed candidates (Cellpose/StarDist -> tracking CSV).

The detector-specific imports live in the CLI adapter below and are loaded
only when a real run requests them.  This module keeps mask-to-centroid and
adjacent-frame linking deterministic and testable without NumPy, TensorFlow,
or PyTorch.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .classical import Detection, TrackingGraph, link_detections, write_submission_csv
from .evaluation import convert_csv_to_geff, evaluate_geff_directory
from .ome_zarr import discover_zarr_datasets, open_ome_zarr
from .validation import validate_submission


def _shape(value: Any) -> Tuple[int, int, int]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is not None:
        shape = tuple(int(item) for item in raw_shape)
    else:
        try:
            shape = (len(value), len(value[0]), len(value[0][0]))
        except (IndexError, TypeError, AttributeError) as exc:
            raise ValueError("labels must be a 3D array-like object") from exc
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise ValueError("labels must have three non-empty dimensions (Z, Y, X)")
    return shape  # type: ignore[return-value]


def _at(value: Any, z: int, y: int, x: int) -> Any:
    try:
        return value[z, y, x]
    except (IndexError, TypeError):
        return value[z][y][x]


def _round_coordinate(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def centroids_from_labels(
    labels: Any,
    image: Optional[Any] = None,
    time_index: int = 0,
    max_detections: Optional[int] = None,
) -> List[Detection]:
    """Convert positive integer instance labels into deterministic centroids.

    Coordinates are rounded to integer voxels because the competition CSV
    contract used by this project stores integer node coordinates.  The mean
    image intensity inside each instance is retained as a confidence proxy;
    it is never used as a fabricated competition score.
    """

    if max_detections is not None and max_detections <= 0:
        return []
    if int(time_index) < 0:
        raise ValueError("time_index must be non-negative")
    label_shape = _shape(labels)
    if image is not None and _shape(image) != label_shape:
        raise ValueError("image and labels must have the same (Z, Y, X) shape")

    accum: Dict[int, List[float]] = {}
    for z in range(label_shape[0]):
        for y in range(label_shape[1]):
            for x in range(label_shape[2]):
                raw_label = _at(labels, z, y, x)
                try:
                    numeric_label = float(raw_label)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(numeric_label) or numeric_label <= 0.0:
                    continue
                label = int(numeric_label)
                if float(label) != numeric_label:
                    continue
                item = accum.setdefault(label, [0.0, 0.0, 0.0, 0.0, 0.0])
                item[0] += 1.0
                item[1] += float(z)
                item[2] += float(y)
                item[3] += float(x)
                if image is not None:
                    try:
                        intensity = float(_at(image, z, y, x))
                    except (TypeError, ValueError):
                        intensity = 0.0
                    item[4] += intensity if math.isfinite(intensity) else 0.0

    detections: List[Tuple[int, Detection]] = []
    for label, (count, sum_z, sum_y, sum_x, sum_intensity) in accum.items():
        mean_intensity = sum_intensity / count if image is not None else 1.0
        detections.append(
            (
                label,
                Detection(
                    t=int(time_index),
                    z=_round_coordinate(sum_z / count),
                    y=_round_coordinate(sum_y / count),
                    x=_round_coordinate(sum_x / count),
                    intensity=mean_intensity,
                ),
            )
        )
    detections.sort(key=lambda item: (-item[1].intensity, item[1].z, item[1].y, item[1].x, item[0]))
    result = [item[1] for item in detections]
    return result[:max_detections] if max_detections is not None else result


def track_label_frames(
    frames: Iterable[Tuple[Any, Any]],
    dataset: str,
    scale_um: Sequence[float] = (1.625, 0.40625, 0.40625),
    max_link_distance_um: float = 7.0,
    max_detections_per_frame: Optional[int] = None,
) -> TrackingGraph:
    """Assign global node IDs and greedy physical links to mask frames."""

    if not str(dataset):
        raise ValueError("dataset must be non-empty")
    nodes: List[Detection] = []
    edges: List[Tuple[int, int]] = []
    previous: List[Detection] = []
    for time_index, (image, labels) in enumerate(frames):
        detected = centroids_from_labels(
            labels,
            image=image,
            time_index=time_index,
            max_detections=max_detections_per_frame,
        )
        current = [
            Detection(
                t=item.t,
                z=item.z,
                y=item.y,
                x=item.x,
                intensity=item.intensity,
                node_id=len(nodes) + index,
            )
            for index, item in enumerate(detected)
        ]
        for previous_index, current_index in link_detections(
            previous,
            current,
            scale_um=scale_um,
            max_distance_um=max_link_distance_um,
        ):
            edges.append((previous[previous_index].node_id, current[current_index].node_id))
        nodes.extend(current)
        previous = current
    return TrackingGraph(dataset=str(dataset), nodes=nodes, edges=edges)


def normalise_detector_name(name: str) -> str:
    detector = str(name).strip().lower()
    if detector not in {"cellpose", "stardist"}:
        raise ValueError("detector must be one of: cellpose, stardist")
    return detector


def predict_labels(
    detector_name: str,
    model: Any,
    image: Any,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    anisotropy: Optional[float] = None,
    probability_threshold: float = 0.5,
    nms_threshold: float = 0.3,
) -> Any:
    """Run one detector using its documented 3D prediction contract.

    Cellpose's current API documents ``CellposeModel.eval(..., do_3D=True)``;
    StarDist's instance API returns ``(labels, details)``.  Imports are kept
    out of this function so the core can be tested with fakes.
    """

    detector = normalise_detector_name(detector_name)
    if detector == "cellpose":
        result = model.eval(
            image,
            do_3D=True,
            cellprob_threshold=float(cellprob_threshold),
            flow_threshold=float(flow_threshold),
            anisotropy=anisotropy,
        )
        if not isinstance(result, (tuple, list)) or not result:
            raise ValueError("Cellpose eval must return masks as its first result")
        return result[0]
    result = model.predict_instances(
        image,
        axes="ZYX",
        prob_thresh=float(probability_threshold),
        nms_thresh=float(nms_threshold),
    )
    if not isinstance(result, (tuple, list)) or not result:
        raise ValueError("StarDist predict_instances must return labels as its first result")
    return result[0]


def load_detector(
    detector_name: str,
    model_path: Optional[Any] = None,
    use_gpu: bool = False,
) -> Any:
    """Load a Cellpose model or an explicitly supplied 3D StarDist model."""

    detector = normalise_detector_name(detector_name)
    if detector == "cellpose":
        try:
            from cellpose import models  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Cellpose adapter requires the selected external Python environment") from exc
        model_class = getattr(models, "CellposeModel", None) or getattr(models, "Cellpose", None)
        if model_class is None:
            raise RuntimeError("installed Cellpose package exposes neither CellposeModel nor Cellpose")
        kwargs = {"gpu": bool(use_gpu)}
        if model_path is not None:
            kwargs["pretrained_model"] = str(model_path)
        return model_class(**kwargs)

    if model_path is None:
        raise ValueError("3D StarDist requires --model-path for a trained model directory")
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError("StarDist model path must be a model directory: {}".format(model_dir))
    try:
        from stardist.models import StarDist3D  # type: ignore
    except ImportError as exc:
        raise RuntimeError("StarDist adapter requires the selected external Python environment") from exc
    return StarDist3D(None, name=model_dir.name, basedir=str(model_dir.parent))


def _materialise_frame(frame: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return frame
    return np.asarray(frame)


def build_detector_graphs(
    data_dir: Path,
    detector_name: str,
    model: Any,
    array_key: str = "0",
    max_frames: Optional[int] = None,
    scale_um: Optional[Sequence[float]] = None,
    max_link_distance_um: float = 7.0,
    max_detections_per_frame: Optional[int] = 2000,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    probability_threshold: float = 0.5,
    nms_threshold: float = 0.3,
) -> List[TrackingGraph]:
    """Run one loaded detector over all OME-Zarr datasets in a directory."""

    detector = normalise_detector_name(detector_name)
    graphs: List[TrackingGraph] = []
    for dataset_path in discover_zarr_datasets(Path(data_dir), require_geff=False):
        opened = open_ome_zarr(dataset_path, array_key=array_key)
        frame_count = int(opened.array.shape[0])
        if max_frames is not None:
            frame_count = min(frame_count, int(max_frames))
        if frame_count <= 0:
            raise ValueError("dataset has no frames to process: {}".format(dataset_path))
        physical_scale = tuple(float(value) for value in (scale_um or opened.scale))
        anisotropy = physical_scale[0] / physical_scale[1]

        def stream():
            for time_index in range(frame_count):
                image = _materialise_frame(opened.array[time_index])
                labels = predict_labels(
                    detector,
                    model,
                    image,
                    cellprob_threshold=cellprob_threshold,
                    flow_threshold=flow_threshold,
                    anisotropy=anisotropy,
                    probability_threshold=probability_threshold,
                    nms_threshold=nms_threshold,
                )
                yield image, labels

        graphs.append(
            track_label_frames(
                stream(),
                dataset=dataset_path.stem,
                scale_um=physical_scale,
                max_link_distance_um=max_link_distance_um,
                max_detections_per_frame=max_detections_per_frame,
            )
        )
    if not graphs:
        raise FileNotFoundError("no .zarr datasets found in {}".format(data_dir))
    return graphs


def summarise_graphs(graphs: Sequence[TrackingGraph], detector: str) -> Dict[str, Any]:
    return {
        "detector": normalise_detector_name(detector),
        "dataset_count": len(graphs),
        "node_count": sum(len(graph.nodes) for graph in graphs),
        "edge_count": sum(len(graph.edges) for graph in graphs),
        "nodes_by_dataset": {graph.dataset: len(graph.nodes) for graph in graphs},
        "edges_by_dataset": {graph.dataset: len(graph.edges) for graph in graphs},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=("cellpose", "stardist"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--array-key", default="0")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-link-distance-um", type=float, default=7.0)
    parser.add_argument("--max-detections-per-frame", type=int, default=2000)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument("--evaluate-gt-dir", type=Path)
    parser.add_argument("--holdout-data-dir", type=Path)
    parser.add_argument("--holdout-gt-dir", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--max-eval-distance", type=float, default=7.0)
    return parser


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    model = load_detector(args.detector, args.model_path, args.gpu)
    final_graphs = build_detector_graphs(
        args.data_dir,
        args.detector,
        model,
        array_key=args.array_key,
        max_frames=args.max_frames,
        max_link_distance_um=args.max_link_distance_um,
        max_detections_per_frame=args.max_detections_per_frame,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        probability_threshold=args.probability_threshold,
        nms_threshold=args.nms_threshold,
    )
    artifact = Path(args.run_dir) / "submission.csv"
    write_submission_csv(final_graphs, artifact)
    metrics: Dict[str, Any] = {
        "artifact": str(artifact),
        "candidate_metrics": summarise_graphs(final_graphs, args.detector),
        "evaluation": {},
    }
    evaluation_graphs = final_graphs
    if args.validation_data_dir is not None:
        evaluation_graphs = build_detector_graphs(
            args.validation_data_dir,
            args.detector,
            model,
            array_key=args.array_key,
            max_frames=args.max_frames,
            max_link_distance_um=args.max_link_distance_um,
            max_detections_per_frame=args.max_detections_per_frame,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            probability_threshold=args.probability_threshold,
            nms_threshold=args.nms_threshold,
        )
        evaluation_csv = Path(args.run_dir) / "validation_submission.csv"
        write_submission_csv(evaluation_graphs, evaluation_csv)
    else:
        evaluation_csv = artifact

    if args.evaluate_gt_dir is not None:
        if args.baseline_root is None:
            raise ValueError("--baseline-root is required for official evaluation")
        predicted = convert_csv_to_geff(evaluation_csv, Path(args.run_dir) / "pred_geffs", args.baseline_root)
        local = evaluate_geff_directory(
            predicted,
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
        holdout_graphs = build_detector_graphs(
            args.holdout_data_dir,
            args.detector,
            model,
            array_key=args.array_key,
            max_frames=args.max_frames,
            max_link_distance_um=args.max_link_distance_um,
            max_detections_per_frame=args.max_detections_per_frame,
            cellprob_threshold=args.cellprob_threshold,
            flow_threshold=args.flow_threshold,
            probability_threshold=args.probability_threshold,
            nms_threshold=args.nms_threshold,
        )
        holdout_csv = Path(args.run_dir) / "holdout_submission.csv"
        write_submission_csv(holdout_graphs, holdout_csv)
        holdout_predicted = convert_csv_to_geff(
            holdout_csv,
            Path(args.run_dir) / "holdout_geffs",
            args.baseline_root,
        )
        holdout_result = evaluate_geff_directory(
            holdout_predicted,
            args.holdout_gt_dir or args.holdout_data_dir,
            baseline_root=args.baseline_root,
            max_distance=args.max_eval_distance,
        )
        metrics["holdout_score"] = holdout_result.get("score")
        metrics["evaluation"]["holdout"] = holdout_result

    with (Path(args.run_dir) / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True, default=str)
        handle.write("\n")
    report = validate_submission(artifact)
    if not report.valid:
        raise ValueError("detector CSV failed local validation: {}".format("; ".join(report.errors[:3])))
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
