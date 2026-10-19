"""Deterministic smoke candidate; it is not a computer-vision solution."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "weak_ensemble"), required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"id": 0, "dataset": "sample_1", "row_type": "node", "node_id": 10, "t": 0, "z": 2, "y": 3, "x": 4, "source_id": -1, "target_id": -1},
        {"id": 1, "dataset": "sample_1", "row_type": "node", "node_id": 11, "t": 1, "z": 2, "y": 3, "x": 5, "source_id": -1, "target_id": -1},
        {"id": 2, "dataset": "sample_1", "row_type": "edge", "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 10, "target_id": 11},
    ]
    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    with (args.run_dir / "submission.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if args.variant == "weak_ensemble":
        metrics = {"local_score": 0.72, "in_domain_score": 0.74, "holdout_score": 0.70, "division_jaccard": 0.25}
    else:
        metrics = {"local_score": 0.74, "in_domain_score": 0.78, "holdout_score": 0.65, "division_jaccard": 0.20}
    with (args.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
