"""Orchestrate timed runs, ranking and reproducible submission plans."""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .adaptive import propose_next_experiments
from .ranking import rank_records, select_submission_candidates
from .proposals import expand_experiment_specs
from .runner import run_experiment
from .validation import validate_submission


def collect_records(runs_dir: Path, cycle_id: Optional[str] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for result_path in sorted(Path(runs_dir).glob("*/result.json")):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if cycle_id is not None and record.get("cycle_id") != cycle_id:
                continue
            records.append(record)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return records


def refresh_artifact_validation(
    records: Iterable[Dict[str, Any]],
    expected_timepoints: Optional[int] = None,
) -> List[Dict[str, Any]]:
    required_timepoints = (
        range(int(expected_timepoints))
        if expected_timepoints is not None
        else None
    )
    refreshed: List[Dict[str, Any]] = []
    for original in records:
        record = dict(original)
        artifact = record.get("artifact")
        if artifact:
            is_diagnostic_only = (
                record.get("live_eligible", True) is False
                or str(record.get("family", "")) in {"smoke", "synthetic_quality"}
            )
            required_for_record = (
                record.get("notebook_expected_timepoints")
                if "notebook_expected_timepoints" in record
                else expected_timepoints
            )
            report = validate_submission(
                Path(str(artifact)),
                expected_timepoints=(
                    None
                    if is_diagnostic_only or required_for_record is None
                    else range(int(required_for_record))
                ),
            )
            record["artifact_valid"] = report.valid
            record["validation"] = report.as_dict()
        refreshed.append(record)
    return refreshed


def write_submission_plan(
    records: Iterable[Dict[str, Any]],
    output_dir: Path,
    limit: int = 5,
    competition: str = "biohub-cell-tracking-during-development",
    cycle_id: Optional[str] = None,
    window_started_at: Optional[str] = None,
    window_seconds: Optional[float] = None,
    submission_mode: str = "file",
    notebook_dataset: Optional[str] = None,
    notebook_payload_mode: Optional[str] = None,
    notebook_enable_gpu: Optional[bool] = None,
    notebook_machine_shape: Optional[str] = None,
    notebook_runner_source: Optional[str] = None,
    notebook_additional_dataset_sources: Optional[Iterable[str]] = None,
    expected_timepoints: Optional[int] = None,
    expected_datasets: Optional[Iterable[str] | str] = None,
    daily_submission_limit: Optional[int] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_records(
        refresh_artifact_validation(records, expected_timepoints=expected_timepoints)
    )
    selected = select_submission_candidates(ranked, limit=limit)
    ranking_path = output_dir / "ranking.csv"
    columns = ["experiment", "family", "live_eligible", "status", "decision", "artifact", "local_score", "holdout_score", "weakness_proxy", "specificity_penalty", "diversity_bonus", "selection_score"]
    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in ranked:
            writer.writerow({column: record.get(column, "") for column in columns})
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id,
        "competition": competition,
        "limit": int(daily_submission_limit if daily_submission_limit is not None else limit),
        "max_candidates": int(limit),
        "submission_mode": str(submission_mode),
        "ranking": str(ranking_path),
        "candidates": [
            {
                key: candidate.get(key)
                for key in (
                    "experiment",
                    "family",
                    "live_eligible",
                    "decision",
                    "artifact",
                    "local_score",
                    "holdout_score",
                    "weakness_proxy",
                    "diversity_bonus",
                    "selection_score",
                    "notebook_runner_kind",
                    "notebook_runner_source",
                    "notebook_runner_det_threshold",
                    "notebook_runner_options",
                    "notebook_expected_timepoints",
                    "notebook_additional_dataset_sources",
                )
            }
            for candidate in selected
        ],
    }
    if notebook_dataset:
        plan["notebook_dataset"] = str(notebook_dataset)
    if notebook_payload_mode:
        plan["notebook_payload_mode"] = str(notebook_payload_mode)
    if notebook_enable_gpu is not None:
        plan["notebook_enable_gpu"] = bool(notebook_enable_gpu)
    if notebook_machine_shape:
        plan["notebook_machine_shape"] = str(notebook_machine_shape)
    if notebook_runner_source:
        plan["notebook_runner_source"] = str(notebook_runner_source)
    if notebook_additional_dataset_sources:
        plan["notebook_additional_dataset_sources"] = [
            str(source) for source in notebook_additional_dataset_sources
        ]
    if expected_timepoints is not None:
        plan["expected_timepoints"] = int(expected_timepoints)
    if expected_datasets is not None:
        values = [expected_datasets] if isinstance(expected_datasets, str) else expected_datasets
        plan["expected_datasets"] = sorted({str(item).strip() for item in values if str(item).strip()})
    if window_started_at is not None:
        plan["window_started_at"] = str(window_started_at)
    if window_seconds is not None:
        plan["window_seconds"] = float(window_seconds)
    with (output_dir / "submission_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return plan


def write_proposal_queue(
    config: Dict[str, Any],
    records: Iterable[Dict[str, Any]],
    output_dir: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist the next bounded adaptive proposals for the following cycle."""

    adaptive = config.get("adaptive", {}) or {}
    if limit is None:
        limit = int(adaptive.get("max_proposals_per_cycle", 0))
    proposals = propose_next_experiments(
        config.get("experiments", []),
        records,
        max_proposals=limit if adaptive.get("enabled", False) else 0,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "proposals": [
            {
                key: proposal.get(key)
                for key in ("name", "family", "hypothesis", "proposal_reason", "tags", "variant", "command")
            }
            for proposal in proposals
        ],
    }
    with (output_dir / "proposals.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return payload


def run_enabled_experiments(
    config: Dict[str, Any],
    output_dir: Path,
    project_root: Path,
    duration_seconds: float,
    max_experiments: Optional[int] = None,
    cycle_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run enabled catalog entries sequentially until time or count is exhausted."""

    started = time.monotonic()
    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = Path(project_root) / resolved_output_dir
    results: List[Dict[str, Any]] = []
    def run_specs(specs: Iterable[Dict[str, Any]]) -> None:
        for spec in specs:
            if not spec.get("enabled", False):
                continue
            if max_experiments is not None and len(results) >= max_experiments:
                break
            remaining = duration_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            bounded = dict(spec)
            if cycle_id is not None:
                bounded["cycle_id"] = cycle_id
            bounded["timeout_seconds"] = min(int(spec.get("timeout_seconds", 3600)), max(1, int(remaining)))
            results.append(run_experiment(bounded, resolved_output_dir, project_root))

    run_specs(expand_experiment_specs(config.get("experiments", [])))
    adaptive = config.get("adaptive", {}) or {}
    if adaptive.get("enabled", False) and (max_experiments is None or len(results) < max_experiments):
        observed = collect_records(resolved_output_dir)
        proposals = propose_next_experiments(
            config.get("experiments", []),
            observed,
            max_proposals=int(adaptive.get("max_proposals_per_cycle", 0)),
        )
        run_specs(proposals)
    return results
