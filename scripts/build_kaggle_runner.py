"""Generate an auditable self-contained Kaggle runner source file."""

from __future__ import annotations

import argparse
from pathlib import Path

from biohub_lab.kaggle_runner import write_classical_runner_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/kaggle/classical_q999_d10_coverage_runner.py"),
    )
    parser.add_argument("--quantile", type=float, default=0.999)
    parser.add_argument("--max-link-distance-um", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-detections-per-frame", type=int, default=2000)
    parser.add_argument("--expected-timepoints", type=int, default=100)
    parser.add_argument("--infer-divisions", action="store_true")
    parser.add_argument("--filter-isolated-nodes", action="store_true")
    args = parser.parse_args()
    output = write_classical_runner_source(
        args.output,
        args.project_root,
        quantile=args.quantile,
        max_link_distance_um=args.max_link_distance_um,
        workers=args.workers,
        max_detections_per_frame=args.max_detections_per_frame,
        expected_timepoints=args.expected_timepoints,
        infer_divisions=args.infer_divisions,
        filter_isolated_nodes=args.filter_isolated_nodes,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
