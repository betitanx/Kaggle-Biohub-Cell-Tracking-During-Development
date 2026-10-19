#!/usr/bin/env python
"""Run the private complete-train holdout evaluator on Kaggle.

This launcher pushes a private notebook and downloads only the evaluator
artifacts. It never calls the competition scorer and therefore consumes no
competition submission slot.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from biohub_lab.kaggle import NotebookSubmissionClient, _kaggle_api


def _configured_source(
    source: Path,
    output_root: Path,
    weights: str | None,
    division_weights: str | None = None,
    base_disappearance_weight: float | None = None,
    edge_floors: str | None = None,
    feature_tta: bool | None = None,
    feature_tta_blend: float | None = None,
    motion_consensus: bool | None = None,
    motion_consensus_max_distance_um: float | None = None,
    motion_consensus_max_prediction_error_um: float | None = None,
    motion_consensus_require_unique_parent: bool | None = None,
    motion_consensus_max_out_degree: int | None = None,
    deepcenter: bool | None = None,
    deepcenter_thresholds: str | None = None,
    deepcenter_max_endpoint_distance_um: float | None = None,
    deepcenter_min_existing_distance_um: float | None = None,
    deepcenter_require_unique_parent: bool | None = None,
    deepcenter_max_out_degree: int | None = None,
    edge_radius_variants: str | None = None,
    edge_max_distance_um: float | None = None,
    det_tta_mode: str | None = None,
) -> Path:
    if edge_radius_variants is not None and base_disappearance_weight is None:
        raise ValueError(
            "edge-radius variants require the parent's base disappearance weight"
        )
    if (
        weights is None
        and division_weights is None
        and base_disappearance_weight is None
        and edge_floors is None
        and feature_tta is None
        and feature_tta_blend is None
        and motion_consensus is None
        and motion_consensus_max_distance_um is None
        and motion_consensus_max_prediction_error_um is None
        and motion_consensus_require_unique_parent is None
        and motion_consensus_max_out_degree is None
        and deepcenter is None
        and deepcenter_thresholds is None
        and deepcenter_max_endpoint_distance_um is None
        and deepcenter_min_existing_distance_um is None
        and deepcenter_require_unique_parent is None
        and deepcenter_max_out_degree is None
        and edge_radius_variants is None
        and edge_max_distance_um is None
        and det_tta_mode is None
    ):
        return source
    output_root.mkdir(parents=True, exist_ok=True)
    configured = output_root / "kaggle_holdout_eval_configured.py"
    prefix = "import os\n"
    if weights is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DISAPPEARANCE_WEIGHTS'] = "
            + repr(weights)
            + "\n"
        )
    if division_weights is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DIVISION_WEIGHTS'] = "
            + repr(division_weights)
            + "\n"
        )
    if base_disappearance_weight is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT'] = "
            + repr(str(float(base_disappearance_weight)))
            + "\n"
        )
    if edge_floors is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_EDGE_FLOORS'] = "
            + repr(edge_floors)
            + "\n"
        )
    if feature_tta is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_FEATURE_TTA'] = "
            + repr("1" if feature_tta else "0")
            + "\n"
        )
    if feature_tta_blend is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_FEATURE_TTA_BLEND'] = "
            + repr(str(float(feature_tta_blend)))
            + "\n"
        )
    if motion_consensus is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_MOTION_CONSENSUS'] = "
            + repr("1" if motion_consensus else "0")
            + "\n"
        )
    if motion_consensus_max_distance_um is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_DISTANCE_UM'] = "
            + repr(str(float(motion_consensus_max_distance_um)))
            + "\n"
        )
    if motion_consensus_max_prediction_error_um is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_PREDICTION_ERROR_UM'] = "
            + repr(str(float(motion_consensus_max_prediction_error_um)))
            + "\n"
        )
    if motion_consensus_require_unique_parent is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_MOTION_CONSENSUS_REQUIRE_UNIQUE_PARENT'] = "
            + repr("1" if motion_consensus_require_unique_parent else "0")
            + "\n"
        )
    if motion_consensus_max_out_degree is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_OUT_DEGREE'] = "
            + repr(str(int(motion_consensus_max_out_degree)))
            + "\n"
        )
    if deepcenter is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER'] = "
            + repr("1" if deepcenter else "0")
            + "\n"
        )
    if deepcenter_thresholds is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER_THRESHOLDS'] = "
            + repr(deepcenter_thresholds)
            + "\n"
        )
    if deepcenter_max_endpoint_distance_um is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER_MAX_ENDPOINT_DISTANCE_UM'] = "
            + repr(str(float(deepcenter_max_endpoint_distance_um)))
            + "\n"
        )
    if deepcenter_min_existing_distance_um is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER_MIN_EXISTING_DISTANCE_UM'] = "
            + repr(str(float(deepcenter_min_existing_distance_um)))
            + "\n"
        )
    if deepcenter_require_unique_parent is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER_REQUIRE_UNIQUE_PARENT'] = "
            + repr("1" if deepcenter_require_unique_parent else "0")
            + "\n"
        )
    if deepcenter_max_out_degree is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DEEPCENTER_MAX_OUT_DEGREE'] = "
            + repr(str(int(deepcenter_max_out_degree)))
            + "\n"
        )
    if edge_radius_variants is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_EDGE_RADIUS_VARIANTS'] = "
            + repr(edge_radius_variants)
            + "\n"
        )
    if edge_max_distance_um is not None:
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_EDGE_MAX_DISTANCE_UM'] = "
            + repr(str(float(edge_max_distance_um)))
            + "\n"
        )
    if det_tta_mode is not None:
        if det_tta_mode not in {"flip4", "dihedral8"}:
            raise ValueError("det_tta_mode must be flip4 or dihedral8")
        prefix += (
            "os.environ['BIOHUB_HOLDOUT_DET_TTA_MODE'] = "
            + repr(det_tta_mode)
            + "\n"
        )
    motion_source = ""
    if any(
        value is not None
        for value in (
            motion_consensus,
            motion_consensus_max_distance_um,
            motion_consensus_max_prediction_error_um,
            motion_consensus_require_unique_parent,
            motion_consensus_max_out_degree,
        )
    ):
        module_path = Path(__file__).resolve().parents[1] / "src" / "biohub_lab" / "motion_consensus.py"
        motion_source = module_path.read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n\n", "", 1
        ) + "\n"
    deepcenter_source = ""
    if any(
        value is not None
        for value in (
            deepcenter,
            deepcenter_thresholds,
            deepcenter_max_endpoint_distance_um,
            deepcenter_min_existing_distance_um,
            deepcenter_require_unique_parent,
            deepcenter_max_out_degree,
        )
    ):
        module_path = Path(__file__).resolve().parents[1] / "src" / "biohub_lab" / "deepcenter_candidate.py"
        deepcenter_source = module_path.read_text(encoding="utf-8").replace(
            "from __future__ import annotations\n\n", "", 1
        ) + "\n"
    source_text = source.read_text(encoding="utf-8")
    future_lines = [
        line for line in source_text.splitlines(keepends=True)
        if line.startswith("from __future__ import ")
    ]
    if future_lines:
        marker = future_lines[-1]
        source_text = source_text.replace(
            marker,
            marker + prefix + motion_source + deepcenter_source,
            1,
        )
    else:
        source_text = prefix + motion_source + deepcenter_source + source_text
    configured.write_text(source_text, encoding="utf-8")
    return configured


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", default="cellmot_holdout_eval_h9_v10")
    parser.add_argument("--max-datasets", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    parser.add_argument(
        "--disappearance-weights",
        help="comma-separated ILP disappearance weights for a two-or-more-way ablation",
    )
    parser.add_argument(
        "--division-weights",
        help="comma-separated ILP division weights for a two-or-more-way ablation",
    )
    parser.add_argument(
        "--base-disappearance-weight",
        type=float,
        help="fixed ILP disappearance weight inherited by the division ablation",
    )
    parser.add_argument(
        "--edge-floors",
        help="comma-separated edge-candidate floors for a linking ablation",
    )
    parser.add_argument(
        "--feature-tta",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable feature-space XY TTA in the holdout evaluator",
    )
    parser.add_argument(
        "--feature-tta-blend",
        type=float,
        help="blend between original and feature-space TTA features in [0, 1]",
    )
    parser.add_argument(
        "--motion-consensus",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable conservative adjacent-frame motion consensus",
    )
    parser.add_argument(
        "--motion-consensus-max-distance-um",
        type=float,
        help="maximum source-to-target distance for motion consensus",
    )
    parser.add_argument(
        "--motion-consensus-max-prediction-error-um",
        type=float,
        help="maximum constant-velocity prediction error for motion consensus",
    )
    parser.add_argument(
        "--motion-consensus-require-unique-parent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="require one existing parent before using velocity consensus",
    )
    parser.add_argument(
        "--motion-consensus-max-out-degree",
        type=int,
        help="maximum source out-degree after motion consensus (1 or 2)",
    )
    parser.add_argument(
        "--deepcenter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable the gated DeepCenter auxiliary detector",
    )
    parser.add_argument(
        "--deepcenter-thresholds",
        help="comma-separated pair of DeepCenter heatmap thresholds",
    )
    parser.add_argument(
        "--deepcenter-max-endpoint-distance-um",
        type=float,
        help="maximum endpoint-to-peak distance for DeepCenter rescue",
    )
    parser.add_argument(
        "--deepcenter-min-existing-distance-um",
        type=float,
        help="minimum distance from an existing node for DeepCenter rescue",
    )
    parser.add_argument(
        "--deepcenter-require-unique-parent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="require a unique adjacent parent for DeepCenter endpoint extrapolation",
    )
    parser.add_argument(
        "--deepcenter-max-out-degree",
        type=int,
        help="maximum source out-degree after DeepCenter rescue (1 or 2)",
    )
    parser.add_argument(
        "--edge-radius-variants",
        help="comma-separated pair of physical edge radii in micrometres",
    )
    parser.add_argument(
        "--edge-max-distance-um",
        type=float,
        help="single physical edge-radius override in micrometres",
    )
    parser.add_argument(
        "--det-tta-mode",
        choices=("flip4", "dihedral8"),
        help="XY detection TTA family; flip4 is a faster local diagnostic",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.max_datasets <= 0:
        raise ValueError("--max-datasets must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ["BIOHUB_HOLDOUT_MAX_DATASETS"] = str(args.max_datasets)
    runner_source = _configured_source(
        args.source,
        args.output_root,
        args.disappearance_weights,
        args.division_weights,
        args.base_disappearance_weight,
        args.edge_floors,
        args.feature_tta,
        args.feature_tta_blend,
        args.motion_consensus,
        args.motion_consensus_max_distance_um,
        args.motion_consensus_max_prediction_error_um,
        args.motion_consensus_require_unique_parent,
        args.motion_consensus_max_out_degree,
        args.deepcenter,
        args.deepcenter_thresholds,
        args.deepcenter_max_endpoint_distance_um,
        args.deepcenter_min_existing_distance_um,
        args.deepcenter_require_unique_parent,
        args.deepcenter_max_out_degree,
        args.edge_radius_variants,
        args.edge_max_distance_um,
        args.det_tta_mode,
    )

    additional_sources = (
        "pilkwang/biohub-tracking-support-pack-50ep-v1",
        "yoikoarmor/biohub-last-call-postprocess-v1",
    )
    if any(
        value is not None
        for value in (
            args.deepcenter,
            args.deepcenter_thresholds,
            args.deepcenter_max_endpoint_distance_um,
            args.deepcenter_min_existing_distance_um,
            args.deepcenter_require_unique_parent,
            args.deepcenter_max_out_degree,
        )
    ):
        additional_sources += (
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
        )

    client = NotebookSubmissionClient(
        "biohub-cell-tracking-during-development",
        "your-kaggle-user/biohub-cell-candidates",
        api=_kaggle_api(),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        dataset_timeout_seconds=180.0,
        payload_mode="runner",
        enable_gpu=True,
        machine_shape="NvidiaTeslaT4",
        additional_dataset_sources=additional_sources,
    )
    kernel = client._push_notebook(
        args.experiment,
        "holdout_metrics.json",
        "20260901T-holdout",
        runner_source=runner_source,
    )
    (args.output_root / "kernel.json").write_text(
        json.dumps(kernel, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(kernel, sort_keys=True), flush=True)
    client._wait_for_kernel(kernel["kernel"])
    client.api.kernels_output(
        kernel["kernel"],
        str(args.output_root),
        file_pattern="holdout_metrics.json",
        force=True,
        quiet=True,
    )
    patterns = {
        "holdout_metrics.json",
        "holdout_submission_edge035.csv",
        "holdout_submission_edge045.csv",
        "holdout_submission_edge055.csv",
        "holdout_submission_edge030.csv",
        "holdout_submission_edge040.csv",
        "holdout_submission_deepcenter035.csv",
        "holdout_submission_deepcenter040.csv",
        "holdout_submission_radius084.csv",
        "holdout_submission_radius090.csv",
    }
    metrics_path = args.output_root / "holdout_metrics.json"
    if metrics_path.is_file():
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        patterns.update(
            Path(str(item["submission"])).name
            for item in (report.get("variants") or {}).values()
            if item.get("submission")
        )
    for pattern in sorted(patterns - {"holdout_metrics.json"}):
        client.api.kernels_output(
            kernel["kernel"],
            str(args.output_root),
            file_pattern=pattern,
            force=True,
            quiet=True,
        )
    print(
        json.dumps(
            {
                "kernel": kernel["kernel"],
                "output_root": str(args.output_root),
                "files": sorted(
                    path.name
                    for path in args.output_root.iterdir()
                    if path.is_file()
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
