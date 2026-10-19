#!/usr/bin/env python3
"""Exercise DeepCenter rescue against a real primary CSV on a small time scope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from biohub_lab.deepcenter_candidate import (
    add_center_rescue_nodes,
    load_deepcenter_model,
    predict_center_peaks,
)
from biohub_lab.ome_zarr import open_ome_zarr
from biohub_lab.validation import validate_submission


COLUMNS = [
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_partial_rows(path: Path, dataset: str, frame_limit: int) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    nodes = [
        row
        for row in source_rows
        if row.get("dataset") == dataset
        and row.get("row_type") == "node"
        and 0 <= int(row["t"]) < frame_limit
    ]
    node_ids = {int(row["node_id"]) for row in nodes}
    edges = [
        row
        for row in source_rows
        if row.get("dataset") == dataset
        and row.get("row_type") == "edge"
        and int(row["source_id"]) in node_ids
        and int(row["target_id"]) in node_ids
    ]
    rows: list[dict[str, Any]] = []
    for row in nodes + edges:
        item = dict(row)
        for key in ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id"):
            item[key] = int(item[key])
        rows.append(item)
    for row_id, row in enumerate(rows):
        row["id"] = row_id
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_path = Path(args.dataset)
    input_csv = Path(args.input_csv)
    output_root = Path(args.output_root)
    dataset = dataset_path.stem
    frame_limit = int(args.frames)
    opened = open_ome_zarr(dataset_path)
    if frame_limit <= 1 or frame_limit > int(opened.array.shape[0]):
        raise ValueError("frames must be in [2, dataset_timepoints]")
    output_root.mkdir(parents=True, exist_ok=True)
    base_rows = _read_partial_rows(input_csv, dataset, frame_limit)
    model, config, _ = load_deepcenter_model(
        Path(args.checkpoint),
        Path(args.source_script),
        device=args.device,
    )
    raw_peaks: list[dict[str, Any]] = []
    for timepoint in range(frame_limit):
        raw_peaks.extend(
            predict_center_peaks(
                model,
                opened.array[timepoint],
                config=config,
                device=args.device,
                threshold=float(args.min_threshold),
                min_distance=1,
                time_index=timepoint,
            )
        )

    variants: dict[str, Any] = {}
    for threshold in (float(args.strict_threshold), float(args.balanced_threshold)):
        label = "deepcenter{:03d}".format(int(round(threshold * 100)))
        peaks = [item for item in raw_peaks if float(item["score"]) >= threshold]
        rescued, rescue_report = add_center_rescue_nodes(
            base_rows,
            peaks,
            frame_count=frame_limit,
            spatial_shape=tuple(int(value) for value in opened.array.shape[1:]),
            max_endpoint_distance_um=float(args.max_endpoint_distance_um),
            min_existing_distance_um=float(args.min_existing_distance_um),
            scale_um=opened.scale,
            max_out_degree=2,
            require_unique_parent=True,
        )
        output_path = output_root / ("h16-real-integration-smoke-" + label + ".csv")
        _write_rows(rescued, output_path)
        validation = validate_submission(
            output_path,
            expected_datasets={dataset},
            expected_timepoints=set(range(frame_limit)),
            include_topology=True,
        )
        variants[label] = {
            "threshold": threshold,
            "peaks_above_threshold": len(peaks),
            "rescue": rescue_report,
            "validation": validation.as_dict(),
            "submission": str(output_path),
            "submission_sha256": _sha256(output_path),
        }

    report = {
        "schema_version": 1,
        "status": "structural_integration_smoke_passed",
        "scope": "one real test dataset and first {} frames; not a complete holdout".format(frame_limit),
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "input_csv": str(input_csv),
        "input_csv_sha256": _sha256(input_csv),
        "frame_limit": frame_limit,
        "dataset_shape": [int(value) for value in opened.array.shape],
        "scale_um": [float(value) for value in opened.scale],
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "variants": variants,
        "holdout_score": None,
        "live_eligible": False,
        "generation_seconds": round(float(time.perf_counter() - started), 3),
    }
    report_path = output_root / "h16-real-integration-smoke.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frames", default=3, type=int)
    parser.add_argument("--min-threshold", default=0.35, type=float)
    parser.add_argument("--strict-threshold", default=0.40, type=float)
    parser.add_argument("--balanced-threshold", default=0.35, type=float)
    parser.add_argument("--max-endpoint-distance-um", default=7.0, type=float)
    parser.add_argument("--min-existing-distance-um", default=2.0, type=float)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset": report["dataset"],
                "frame_limit": report["frame_limit"],
                "generation_seconds": report["generation_seconds"],
                "variants": {
                    label: {
                        "peaks_above_threshold": item["peaks_above_threshold"],
                        "added_nodes": item["rescue"]["added_nodes"],
                        "valid": item["validation"]["valid"],
                    }
                    for label, item in report["variants"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
