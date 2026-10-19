"""Synthetic 3D+time benchmark for the classical candidate.

The benchmark is intentionally separate from the smoke candidate.  It creates
small nested-list volumes in the control environment, runs the real H1 peak
detector/linker, measures node/edge/division behaviour against known truth and
writes a valid submission artifact.  Its scores are diagnostic only and are
never eligible for Kaggle upload.
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .classical import (
    Detection,
    TrackingGraph,
    infer_division_links,
    remove_isolated_nodes,
    track_frames,
    write_submission_csv,
)
from .validation import validate_submission


Coordinate = Tuple[int, int, int]
Scale = Tuple[float, float, float]


@dataclass(frozen=True)
class SyntheticCase:
    """A deterministic microscopy-like sequence and its graph truth."""

    frames: List[List[List[List[float]]]]
    truth: TrackingGraph
    scale_um: Scale
    threshold: float
    seed: int


def _empty_frame(shape: Coordinate, rng: random.Random) -> List[List[List[float]]]:
    z_size, y_size, x_size = shape
    return [
        [
            [0.05 * rng.random() for _x in range(x_size)]
            for _y in range(y_size)
        ]
        for _z in range(z_size)
    ]


def _paint_peak(frame: List[List[List[float]]], coordinate: Coordinate, peak: float) -> None:
    """Paint a compact 3D peak whose center is a unique local maximum."""

    z0, y0, x0 = coordinate
    z_size, y_size, x_size = len(frame), len(frame[0]), len(frame[0][0])
    for z in range(max(0, z0 - 1), min(z_size, z0 + 2)):
        for y in range(max(0, y0 - 1), min(y_size, y0 + 2)):
            for x in range(max(0, x0 - 1), min(x_size, x0 + 2)):
                distance = abs(z - z0) + abs(y - y0) + abs(x - x0)
                value = max(0.0, float(peak) - 0.16 * distance)
                frame[z][y][x] = max(frame[z][y][x], value)


def _truth_coordinates() -> Mapping[int, Sequence[Coordinate]]:
    """Return a short sequence with one cell division and two controls."""

    return {
        0: ((4, 5, 5), (3, 12, 16), (7, 15, 7)),
        1: ((4, 5, 6), (3, 12, 17), (7, 15, 8)),
        2: ((4, 6, 7), (3, 13, 17), (7, 15, 9)),
        # The first cell divides into two children separated by two z voxels.
        3: ((4, 6, 8), (6, 6, 8), (3, 13, 18), (7, 14, 10)),
        4: ((4, 7, 9), (6, 7, 9), (3, 14, 18), (7, 14, 11)),
    }


def make_case(seed: int = 20260804, threshold: float = 0.75) -> SyntheticCase:
    """Build a deterministic 5-frame 3D case without NumPy or external data."""

    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    shape = (9, 18, 24)
    rng = random.Random(int(seed))
    coordinates = _truth_coordinates()
    frames = [_empty_frame(shape, rng) for _time in sorted(coordinates)]
    truth_nodes: List[Detection] = []
    node_ids: Dict[Tuple[int, Coordinate], int] = {}
    for time_index in sorted(coordinates):
        frame = frames[time_index]
        for coordinate in coordinates[time_index]:
            _paint_peak(frame, coordinate, peak=1.0)
            node_id = len(truth_nodes)
            node_ids[(time_index, coordinate)] = node_id
            truth_nodes.append(
                Detection(
                    t=time_index,
                    z=coordinate[0],
                    y=coordinate[1],
                    x=coordinate[2],
                    intensity=1.0,
                    node_id=node_id,
                )
            )

    # A bright non-cell peak tests precision and orphan handling.
    _paint_peak(frames[2], (2, 3, 20), peak=0.84)

    edges: List[Tuple[int, int]] = []
    for time_index in range(4):
        current = list(coordinates[time_index])
        following = list(coordinates[time_index + 1])
        if time_index < 2:
            for index in range(3):
                edges.append((node_ids[(time_index, current[index])], node_ids[(time_index + 1, following[index])]))
        elif time_index == 2:
            # First parent divides; the two control cells continue normally.
            edges.extend(
                [
                    (node_ids[(2, current[0])], node_ids[(3, following[0])]),
                    (node_ids[(2, current[0])], node_ids[(3, following[1])]),
                    (node_ids[(2, current[1])], node_ids[(3, following[2])]),
                    (node_ids[(2, current[2])], node_ids[(3, following[3])]),
                ]
            )
        else:
            for index in range(4):
                edges.append((node_ids[(time_index, current[index])], node_ids[(time_index + 1, following[index])]))
    truth = TrackingGraph(dataset="synthetic_zebrafish", nodes=truth_nodes, edges=edges)
    return SyntheticCase(
        frames=frames,
        truth=truth,
        scale_um=(1.625, 0.40625, 0.40625),
        threshold=float(threshold),
        seed=int(seed),
    )


def _distance_um(left: Detection, right: Detection, scale_um: Scale) -> float:
    dz = (left.z - right.z) * scale_um[0]
    dy = (left.y - right.y) * scale_um[1]
    dx = (left.x - right.x) * scale_um[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _match_nodes(
    predicted: Iterable[Detection],
    truth: Iterable[Detection],
    scale_um: Scale,
    tolerance_um: float,
) -> Dict[int, int]:
    predicted = list(predicted)
    truth = list(truth)
    candidates = [
        (_distance_um(left, right, scale_um), left.node_id, right.node_id)
        for left in predicted
        for right in truth
    ]
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_pred: Set[int] = set()
    matched_truth: Set[int] = set()
    matches: Dict[int, int] = {}
    for distance, predicted_id, truth_id in candidates:
        if distance > tolerance_um or predicted_id in matched_pred or truth_id in matched_truth:
            continue
        matched_pred.add(predicted_id)
        matched_truth.add(truth_id)
        matches[predicted_id] = truth_id
    return matches


def evaluate_graph(
    predicted: TrackingGraph,
    truth: TrackingGraph,
    scale_um: Scale = (1.625, 0.40625, 0.40625),
    tolerance_um: float = 0.75,
) -> Dict[str, float]:
    """Measure detection, adjacent-edge and division performance."""

    if tolerance_um < 0.0 or not math.isfinite(float(tolerance_um)):
        raise ValueError("tolerance_um must be finite and non-negative")
    matches = _match_nodes(predicted.nodes, truth.nodes, scale_um, float(tolerance_um))
    node_tp = len(matches)
    node_precision = node_tp / float(len(predicted.nodes)) if predicted.nodes else 0.0
    node_recall = node_tp / float(len(truth.nodes)) if truth.nodes else 1.0
    predicted_edges = set(predicted.edges)
    truth_edges = set(truth.edges)
    mapped_predicted_edges = {
        (matches[source_id], matches[target_id])
        for source_id, target_id in predicted_edges
        if source_id in matches and target_id in matches
    }
    edge_tp = len(mapped_predicted_edges & truth_edges)
    edge_precision = edge_tp / float(len(predicted_edges)) if predicted_edges else 0.0
    edge_recall = edge_tp / float(len(truth_edges)) if truth_edges else 1.0
    truth_outgoing: Dict[int, int] = {}
    for source_id, _target_id in truth_edges:
        truth_outgoing[source_id] = truth_outgoing.get(source_id, 0) + 1
    predicted_outgoing: Dict[int, int] = {}
    for source_id, target_id in mapped_predicted_edges:
        if source_id in truth_outgoing and target_id in {node.node_id for node in truth.nodes}:
            predicted_outgoing[source_id] = predicted_outgoing.get(source_id, 0) + 1
    division_parents = {node_id for node_id, count in truth_outgoing.items() if count == 2}
    division_tp = sum(1 for node_id in division_parents if predicted_outgoing.get(node_id, 0) >= 2)
    division_recall = division_tp / float(len(division_parents)) if division_parents else 1.0
    node_f1 = _f1(node_precision, node_recall)
    edge_f1 = _f1(edge_precision, edge_recall)
    # This is a diagnostic score only; it is deliberately not the Kaggle metric.
    diagnostic_score = 0.30 * node_f1 + 0.50 * edge_f1 + 0.20 * division_recall
    return {
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "edge_f1": edge_f1,
        "division_recall": division_recall,
        "diagnostic_score": diagnostic_score,
        "predicted_nodes": float(len(predicted.nodes)),
        "truth_nodes": float(len(truth.nodes)),
        "predicted_edges": float(len(predicted_edges)),
        "truth_edges": float(len(truth_edges)),
    }


def run_benchmark(
    run_dir: Path,
    seed: int = 20260804,
    threshold: float = 0.75,
    max_link_distance_um: float = 7.0,
    infer_divisions: bool = False,
    sister_max_distance_um: float = 5.0,
    midpoint_tolerance_um: float = 3.0,
    max_parent_displacement_delta_um: float = 3.0,
    min_sibling_intensity_ratio: float = 0.9,
    filter_isolated_nodes: bool = False,
) -> Dict[str, object]:
    """Run H1 on the synthetic case and persist a diagnostic artifact."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    case = make_case(seed=seed, threshold=threshold)
    predicted = track_frames(
        case.frames,
        dataset=case.truth.dataset,
        threshold=case.threshold,
        neighborhood=(1, 1, 1),
        scale_um=case.scale_um,
        max_link_distance_um=max_link_distance_um,
        fast=False,
    )
    if infer_divisions:
        predicted = infer_division_links(
            predicted,
            scale_um=case.scale_um,
            max_link_distance_um=max_link_distance_um,
            sister_max_distance_um=sister_max_distance_um,
            midpoint_tolerance_um=midpoint_tolerance_um,
            max_parent_displacement_delta_um=max_parent_displacement_delta_um,
            min_sibling_intensity_ratio=min_sibling_intensity_ratio,
        )
    if filter_isolated_nodes:
        predicted = remove_isolated_nodes(predicted)
    artifact = write_submission_csv([predicted], run_dir / "submission.csv")
    validation = validate_submission(artifact).as_dict()
    measured = evaluate_graph(predicted, case.truth, scale_um=case.scale_um)
    quality_gate_passed = bool(
        validation["valid"]
        and measured["node_recall"] >= 0.95
        and measured["node_precision"] >= (0.99 if filter_isolated_nodes else 0.0)
        and measured["edge_recall"] >= 0.85
        and measured["division_recall"] >= (1.0 if infer_divisions else 0.0)
    )
    metrics: Dict[str, object] = {
        "artifact": str(artifact),
        "evaluation_scope": "synthetic_diagnostic_only",
        "candidate": (
            "H7_division_aware+H8_clean_graph"
            if infer_divisions and filter_isolated_nodes
            else "H7_division_aware"
            if infer_divisions
            else "H8_clean_graph"
            if filter_isolated_nodes
            else "H1_peak_link"
        ),
        "seed": int(seed),
        "threshold": float(threshold),
        "max_link_distance_um": float(max_link_distance_um),
        "division_inference": {
            "enabled": bool(infer_divisions),
            "sister_max_distance_um": float(sister_max_distance_um),
            "midpoint_tolerance_um": float(midpoint_tolerance_um),
            "max_parent_displacement_delta_um": float(max_parent_displacement_delta_um),
            "min_sibling_intensity_ratio": float(min_sibling_intensity_ratio),
            "filter_isolated_nodes": bool(filter_isolated_nodes),
        },
        "contract_validation": validation,
        "quality_gate_passed": quality_gate_passed,
        "division_note": (
            "H7 adds one conservative second-daughter link using sister distance and parent-midpoint gates."
            if infer_divisions
            else "H1 is intentionally one-to-one; division_recall exposes the missing topology hypothesis."
        ),
        "postprocessing_note": (
            "H8 removes nodes with no temporal evidence and remaps IDs; this can remove legitimate one-frame cells."
            if filter_isolated_nodes
            else "No H8 isolated-node filtering applied."
        ),
        **measured,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--max-link-distance-um", type=float, default=7.0)
    parser.add_argument("--infer-divisions", action="store_true")
    parser.add_argument("--sister-max-distance-um", type=float, default=5.0)
    parser.add_argument("--midpoint-tolerance-um", type=float, default=3.0)
    parser.add_argument("--max-parent-displacement-delta-um", type=float, default=3.0)
    parser.add_argument("--min-sibling-intensity-ratio", type=float, default=0.9)
    parser.add_argument("--filter-isolated-nodes", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            run_benchmark(
                args.run_dir,
                args.seed,
                args.threshold,
                args.max_link_distance_um,
                infer_divisions=args.infer_divisions,
                sister_max_distance_um=args.sister_max_distance_um,
                midpoint_tolerance_um=args.midpoint_tolerance_um,
                max_parent_displacement_delta_um=args.max_parent_displacement_delta_um,
                min_sibling_intensity_ratio=args.min_sibling_intensity_ratio,
                filter_isolated_nodes=args.filter_isolated_nodes,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
