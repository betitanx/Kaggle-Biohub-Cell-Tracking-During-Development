"""Command-line entry point for the Biohub experiment lab."""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .controller import collect_records, run_enabled_experiments, write_proposal_queue, write_submission_plan
from .doctor import check_environment
from .kaggle import submit_plan
from .lablog import update_lab_proposals, update_lab_results, update_lab_submissions
from .openresearch import write_openresearch_proposals
from .runner import load_config
from .splits import create_group_holdout
from .validation import validate_submission


def wait_for_duration(
    started_at: float,
    duration_seconds: float,
    clock=time.monotonic,
    sleep=time.sleep,
) -> float:
    """Wait for the end of a wall-clock experiment window and return remaining time."""

    remaining = max(0.0, float(duration_seconds) - (float(clock()) - float(started_at)))
    if remaining > 0.0:
        sleep(remaining)
    return remaining


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biohub-lab")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--data-dir", type=Path)
    doctor.add_argument("--test-dir", type=Path)
    doctor.add_argument("--holdout-dir", type=Path)
    doctor.add_argument("--baseline-root", type=Path)
    doctor.add_argument("--ultrack-labels-dir", type=Path)
    doctor.add_argument("--ultrack-validation-labels-dir", type=Path)
    doctor.add_argument("--ultrack-holdout-labels-dir", type=Path)
    doctor.add_argument("--cell-tracking-python")
    split = sub.add_parser("split")
    split.add_argument("--source-dir", type=Path, required=True)
    split.add_argument("--train-dir", type=Path)
    split.add_argument("--holdout-dir", type=Path, required=True)
    split.add_argument("--fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("config/experiments.json"))
    run.add_argument("--duration-hours", type=float)
    run.add_argument("--duration-seconds", type=float)
    run.add_argument("--max-experiments", type=int)
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--live-submit", action="store_true")
    run.add_argument("--state", type=Path, default=Path("state/submissions.json"))
    run.add_argument("--wait-seconds", type=int, default=600)
    rank = sub.add_parser("rank")
    rank.add_argument("--runs-dir", type=Path, default=Path("runs"))
    rank.add_argument("--output-dir", type=Path, default=Path("outputs"))
    rank.add_argument("--limit", type=int, default=5)
    submit = sub.add_parser("submit")
    submit.add_argument("--plan", type=Path, default=Path("outputs/submission_plan.json"))
    submit.add_argument("--state", type=Path, default=Path("state/submissions.json"))
    submit.add_argument("--live-submit", action="store_true")
    submit.add_argument("--wait-seconds", type=int, default=600)
    submit.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only failures explicitly verified to have happened before Kaggle accepted an upload",
    )
    submit.add_argument("--competition")
    openresearch = sub.add_parser(
        "ingest-openresearch",
        help="convert an OpenResearch JSON snapshot into non-live Biohub proposals",
    )
    openresearch.add_argument("--input", type=Path, required=True)
    openresearch.add_argument("--output", type=Path, default=Path("outputs/openresearch-proposals.json"))
    openresearch.add_argument("--fixed-run-command", required=True)
    openresearch.add_argument("--existing-name", action="append", default=[])
    openresearch.add_argument("--competition", default="biohub-cell-tracking-during-development")
    openresearch.add_argument("--source-url", default="https://github.com/alphaXiv/OpenResearch")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        report = validate_submission(args.submission)
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        return 0 if report.valid else 2
    if args.command == "doctor":
        print(
            json.dumps(
                check_environment(
                    data_dir=args.data_dir,
                    extra_paths={
                        "test_dir": args.test_dir,
                        "holdout_dir": args.holdout_dir,
                        "baseline_root": args.baseline_root,
                        "ultrack_labels_dir": args.ultrack_labels_dir,
                        "ultrack_validation_labels_dir": args.ultrack_validation_labels_dir,
                        "ultrack_holdout_labels_dir": args.ultrack_holdout_labels_dir,
                    },
                    external_python=args.cell_tracking_python,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "split":
        manifest = create_group_holdout(
            args.source_dir,
            args.holdout_dir,
            fraction=args.fraction,
            seed=args.seed,
            train_dir=args.train_dir,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        if args.duration_hours is not None and args.duration_seconds is not None:
            raise SystemExit("use either --duration-hours or --duration-seconds")
        if args.duration_seconds is not None:
            duration = args.duration_seconds
        elif args.duration_hours is not None:
            duration = args.duration_hours * 3600.0
        else:
            duration = 4.0 * 3600.0
        config = load_config(args.config)
        started_at = time.monotonic()
        window_started_at = datetime.now(timezone.utc).isoformat()
        cycle_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8])
        run_enabled_experiments(
            config,
            args.output_dir,
            Path.cwd(),
            duration,
            args.max_experiments,
            cycle_id=cycle_id,
        )
        cycle_records = collect_records(args.output_dir, cycle_id=cycle_id)
        all_records = collect_records(args.output_dir)
        write_submission_plan(
            cycle_records,
            Path("outputs"),
            int(config.get("max_submissions_per_cycle", 5)),
            str(config["competition"]),
            cycle_id=cycle_id,
            window_started_at=window_started_at,
            window_seconds=duration,
            submission_mode=str(config.get("submission_mode", "file")),
            notebook_dataset=config.get("notebook_dataset"),
            notebook_payload_mode=config.get("notebook_payload_mode"),
            notebook_enable_gpu=config.get("notebook_enable_gpu"),
            notebook_machine_shape=config.get("notebook_machine_shape"),
            notebook_runner_source=config.get("notebook_runner_source"),
            notebook_additional_dataset_sources=config.get("notebook_additional_dataset_sources"),
            expected_timepoints=config.get("expected_timepoints"),
            expected_datasets=config.get("expected_datasets"),
            daily_submission_limit=int(config.get("daily_submission_limit", 5)),
        )
        proposal_queue = write_proposal_queue(config, all_records, Path("outputs"))
        lab_path = Path("lab.md")
        if lab_path.is_file():
            update_lab_results(lab_path, all_records)
            update_lab_proposals(lab_path, proposal_queue.get("proposals", []))
        if args.live_submit:
            with Path("outputs/submission_plan.json").open("r", encoding="utf-8") as handle:
                live_plan = json.load(handle)
            has_real_candidate = any(
                str(candidate.get("family", "")) != "smoke"
                and candidate.get("live_eligible", True) is not False
                for candidate in live_plan.get("candidates", [])
            )
            if has_real_candidate:
                wait_for_duration(started_at, duration)
            results = submit_plan(Path("outputs/submission_plan.json"), args.state, live=True, wait_seconds=args.wait_seconds)
            Path("outputs").mkdir(parents=True, exist_ok=True)
            Path("outputs/submission_results.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            if Path("lab.md").is_file():
                update_lab_submissions(Path("lab.md"), results)
            print(json.dumps(results, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "rank":
        records = collect_records(args.runs_dir)
        write_submission_plan(records, args.output_dir, args.limit)
        lab_path = Path("lab.md")
        if lab_path.is_file():
            update_lab_results(lab_path, records)
        return 0
    if args.command == "submit":
        if args.competition:
            with args.plan.open("r", encoding="utf-8") as handle:
                plan = json.load(handle)
            plan["competition"] = args.competition
            temporary_plan = args.plan.with_suffix(".override.json")
            temporary_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            try:
                results = submit_plan(
                    temporary_plan,
                    args.state,
                    args.live_submit,
                    args.wait_seconds,
                    retry_failed=args.retry_failed,
                )
            finally:
                temporary_plan.unlink(missing_ok=True)
        else:
            results = submit_plan(
                args.plan,
                args.state,
                args.live_submit,
                args.wait_seconds,
                retry_failed=args.retry_failed,
            )
        if Path("lab.md").is_file():
            update_lab_submissions(Path("lab.md"), results)
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "ingest-openresearch":
        result = write_openresearch_proposals(
            args.input,
            args.output,
            fixed_run_command=args.fixed_run_command,
            existing_names=args.existing_name,
            competition=args.competition,
            source_url=args.source_url,
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
