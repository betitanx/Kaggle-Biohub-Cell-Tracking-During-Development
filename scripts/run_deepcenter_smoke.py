#!/usr/bin/env python3
"""Run a bounded one-frame smoke test for the public DeepCenter checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from biohub_lab.deepcenter_candidate import (
    load_deepcenter_model,
    predict_center_peaks,
)
from biohub_lab.ome_zarr import open_ome_zarr


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    checkpoint_path = Path(args.checkpoint)
    source_script = Path(args.source_script)
    output_path = Path(args.output)
    started = time.perf_counter()
    opened = open_ome_zarr(dataset_path)
    if int(args.frame) < 0 or int(args.frame) >= int(opened.array.shape[0]):
        raise ValueError("frame is outside the dataset time dimension")
    model, config, _ = load_deepcenter_model(
        checkpoint_path,
        source_script,
        device=args.device,
    )
    frame = opened.array[int(args.frame)]
    peaks = predict_center_peaks(
        model,
        frame,
        config=config,
        device=args.device,
        threshold=float(args.threshold),
        min_distance=int(args.min_distance),
        max_peaks=args.max_peaks,
        time_index=int(args.frame),
    )
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "status": "structural_smoke_passed",
        "candidate": "h16-deepcenter-gated-shape-prior",
        "dataset": dataset_path.name,
        "dataset_path": str(dataset_path),
        "frame": int(args.frame),
        "dataset_shape": [int(value) for value in opened.array.shape],
        "scale_um": [float(value) for value in opened.scale],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_script": str(source_script),
        "device": args.device,
        "inference_config": {
            "pool_factor": int(config.pool_factor),
            "base_channels": int(config.base_channels),
            "norm_lo_pct": float(config.norm_lo_pct),
            "norm_hi_pct": float(config.norm_hi_pct),
            "norm_clip_lo": float(config.norm_clip_lo),
            "norm_clip_hi": float(config.norm_clip_hi),
        },
        "peak_gate": {
            "threshold": float(args.threshold),
            "min_distance": int(args.min_distance),
            "max_peaks": args.max_peaks,
        },
        "peak_count": len(peaks),
        "top_score": max((float(item["score"]) for item in peaks), default=None),
        "top_peaks": peaks[:10],
        "coordinate_contract": "z,y,x in original voxels; xy pooled center maps with offset (pool_factor-1)/2",
        "graph_mutation": False,
        "holdout_score": None,
        "live_eligible": False,
        "generation_seconds": round(float(elapsed), 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame", default=0, type=int)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--min-distance", default=1, type=int)
    parser.add_argument("--max-peaks", default=None, type=int)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset": report["dataset"],
                "frame": report["frame"],
                "peak_count": report["peak_count"],
                "top_score": report["top_score"],
                "generation_seconds": report["generation_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
